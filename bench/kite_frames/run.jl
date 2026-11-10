# The KITE in three frames, in one session: define it through circuit-quantization, move it through the published,
# normal-mode and optimum frames by the numeric loop route, solve each, time a frame-change loop, write a run record.
#
#   JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=bench/kite_frames bench/kite_frames/run.jl
#
# KITE_FRAMES_OUT     directory of the run record (default: results/ beside this script; not committed)
# KITE_FRAMES_ITERS   iterations of the timed loop (default 200)
# KITE_FRAMES_REFGRID 1 to also solve the published and optimum frames on the reference calculation's own grids and
#                     print the levels beside the paper's (default 0, off: it adds several minutes)
#
# Exits nonzero if two frames' energies differ by more than FRAME_TOL, if a coefficient of a loop spec is not a
# constant, if the RuntimeGeneratedFunctions cache of KronFreeCircuits grows during the loop (or does not grow for the
# probe coefficient), if Julia compiles anything between a frame change and the first eigensolve of its materialize in
# the timed loop, or if load_record does not return the frame and values written.
# Julia compile time from the first eigensolve to the end of a materialize only prints a warning: a rarely taken branch
# of the eigensolver may compile once in 200 solves, and no coefficient is compiled there (the cache count checks that).

using KronFreeCircuits, PythonCall, LinearAlgebra, Logging, Printf, Random
using KronFreeCircuits: _Constant
include(joinpath(@__DIR__, "..", "..", "test", "python", "kite.jl"))

# The leaves' high computed states have no weight on the phase mask and each one logs a warning from `fix_phases!`;
# those are counted and dropped, every other log message passes.
struct PhaseWarningFilter <: AbstractLogger
    inner::AbstractLogger
    dropped::Base.RefValue{Int}
end
Logging.min_enabled_level(l::PhaseWarningFilter) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::PhaseWarningFilter, args...) = Logging.shouldlog(l.inner, args...)
function Logging.handle_message(l::PhaseWarningFilter, level, message, args...; kw...)
    startswith(string(message), "fix_phases!") && return (l.dropped[] += 1; nothing)
    return Logging.handle_message(l.inner, level, message, args...; kw...)
end
const LOGGER = PhaseWarningFilter(ConsoleLogger(stderr), Ref(0))
global_logger(LOGGER)

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run single-threaded (julia -t 1), got $(Threads.nthreads()) threads")

const ITERS = parse(Int, get(ENV, "KITE_FRAMES_ITERS", "200"))
const REFGRID = get(ENV, "KITE_FRAMES_REFGRID", "0") == "1"
const OUT = get(ENV, "KITE_FRAMES_OUT", joinpath(@__DIR__, "results"))
const NEV = 4
# GHz, the largest allowed spread of any of the four energies across the frames.
# The refined and wider trees move the grid tree's levels by at most 7.9e-6 GHz (the normal frame's leaf truncation;
# 1.4e-7 in the other frames), so frames that agree to their discretization differ by under 1e-5; 2e-5 leaves a factor
# of two over that and fails on anything larger.
const FRAME_TOL = 2e-5

# The grids: ChargeBasis(ncut) on a compact mode; a PhaseGrid (order 8) of npts points over center ± halfwidth on an
# extended mode, centered at the minimum of the extended block's quadratic part, -E_L⁻¹ flux_linear.
# Each leaf (Dense) keeps its `keep` lowest states; the root keeps the lowest NEV of their product with Auto(), which
# resolves to LOBPCG there (the product, keep³, is beyond what a Dense root diagonalizes in reasonable time).
# GRID solves the three frames; REFINED (finer grids, same window, more kept states) and WIDER (same spacing, a wider
# window, more kept states) check its convergence; LOOP is the tree of the timed loop near the optimum.
const GRID = (ncut=10, npts=121, halfwidth=2.5π, keep=32)
const REFINED = (ncut=14, npts=161, halfwidth=2.5π, keep=40)
const WIDER = (ncut=10, npts=193, halfwidth=4π, keep=40)
const LOOP = (ncut=10, npts=121, halfwidth=2.5π, keep=24)

# The reference levels (GHz) of the circuit-quantization paper's KITE figure, as the repository owner supplied them
# (lists PUBLISHED and OPTIMIZED of the owner's untracked mean_field_ref_dont_commit/kite_reference/check_paper_numbers.py): the four lowest
# levels in the published coordinates by direct diagonalization on 1 254 400 states, and in the optimized frame on
# another grid.
const REF_PUBLISHED = [-9.523995, -9.510525, -8.453790, -8.337862]
const REF_OPTIMIZED = [-9.523992, -9.510522, -8.453784, -8.337853]

value(c::Coefficient) = real(c((;)))

# Each mode's leaf from the spec's own mode list (types can change between frames).
function leaf(spec::Spec, i::Int, center::Dict{Int,Float64}, g)
    m = spec.modes[i]
    basis = m.type === :c ? ChargeBasis(g.ncut) :
            m.type === :e ? PhaseGrid(center[i] - g.halfwidth, center[i] + g.halfwidth, g.npts) :
            error("mode $(m.name) has type :$(m.type); this bench builds :c and :e leaves only")
    return Node([m.name => basis]; solver=g.solver, compute=g.keep, keep=lowest(g.keep))
end

function kite_tree(spec::Spec, g; root_solver=Auto())
    e = findall(m -> m.type === :e, spec.modes)
    EL = [value(spec.E_L[i, j]) for i in e, j in e]
    c = -(EL \ [value(spec.flux_linear[i]) for i in e])
    center = Dict(zip(e, c))
    return Node([leaf(spec, i, center, g) for i in eachindex(spec.modes)]; solver=root_solver, compute=NEV,
                keep=lowest(NEV))
end

# Julia's cumulative compilation time, in ns (needs Base.cumulative_compile_timing(true) to count).
compile_ns() = first(Base.cumulative_compile_time_ns())

# A Dense solver that records, at its first call (the first eigensolve of a materialize), the time and Julia's
# cumulative compilation time.
mutable struct FirstStamp <: Solver
    t::UInt64
    c::UInt64
end
FirstStamp() = FirstStamp(0, 0)
function KronFreeCircuits.solve(s::FirstStamp, H::KronFreeOperator, m::Int)
    s.t == 0 && (s.t = time_ns(); s.c = compile_ns())
    return KronFreeCircuits.solve(Dense(), H, m)
end

solve_frame(spec, g) = energies(materialize(kite_tree(spec, merge(g, (solver=Dense(),))), spec, (;)))

fmt(E) = join((@sprintf("%.6f", x) for x in E), "  ")

# Entries of the RuntimeGeneratedFunctions body cache of KronFreeCircuits: one per distinct compiled coefficient
# expression ever built (entries are never removed), so it grows exactly when a new coefficient is compiled.
# This reads private internals of RuntimeGeneratedFunctions 0.5.27 (the module's cache constant and its `levels` field).
const _RGF_CACHE = getfield(KronFreeCircuits, Symbol("#_RuntimeGeneratedFunctions_cache"))
rgf_count() = sum(length, @atomic _RGF_CACHE.levels; init=0)

failures = String[]

# --- Define and quantize ----------------------------------------------------------------------------------------------
res, θ, frames = kite()
qc = QuantizedCircuit(res; values=θ)
println("KITE quantized; parameters ", θ)
describe(g) = @sprintf("ChargeBasis(%d), PhaseGrid of %d points over ±%.4f, %d kept per leaf",
                       g.ncut, g.npts, g.halfwidth, g.keep)
for (label, g) in (:grid => GRID, :refined => REFINED, :wider => WIDER, :loop => LOOP)
    @printf("%-8s %s\n", "$label:", describe(g))
end

# --- Three frames -----------------------------------------------------------------------------------------------------
println("\nfour lowest energies (GHz)")
E = Dict{Symbol,Vector{Float64}}()
for (name, S) in pairs(frames)
    spec = Spec(qc, S)
    E[name] = solve_frame(spec, GRID)
    @printf("  %-9s modes %-3s grid     %s\n", name, join(string.(m.type for m in spec.modes)), fmt(E[name]))
    for (label, g) in (:refined => REFINED, :wider => WIDER)
        Eg = solve_frame(spec, g)
        @printf("  %-9s           %-8s %s   (max change %.2e)\n", name, label, fmt(Eg), maximum(abs.(Eg - E[name])))
    end
end
spread = maximum(maximum(E[a][j] for a in keys(E)) - minimum(E[a][j] for a in keys(E)) for j in 1:NEV)
@printf("largest spread across frames: %.2e GHz (tolerance %.0e)\n", spread, FRAME_TOL)
spread <= FRAME_TOL || push!(failures, @sprintf("frames disagree by %.2e GHz > %.0e", spread, FRAME_TOL))

println("\nagainst the paper's reference levels (GHz; difference = this run - reference)")
@printf("  %-25s %s\n", "reference, published", fmt(REF_PUBLISHED))
@printf("  %-25s %s\n", "reference, optimized", fmt(REF_OPTIMIZED))
diffs(x) = join((@sprintf("%+.1e", d) for d in x), "   ")
for name in keys(frames)
    @printf("  %-25s %s\n", "$name - ref published", diffs(E[name] - REF_PUBLISHED))
    @printf("  %-25s %s\n", "$name - ref optimized", diffs(E[name] - REF_OPTIMIZED))
end
@printf("  ground doublet splitting: %.3f MHz (reference, published: %.3f MHz)\n",
        1e3 * (E[:published][2] - E[:published][1]), 1e3 * (REF_PUBLISHED[2] - REF_PUBLISHED[1]))

# --- Optional: the reference calculation's own grids ------------------------------------------------------------------
# The reference levels come from a direct solve on the bases of circuit-quantization's
# `default_bases(h, charge_cutoff=24, max_points=160, extent_scale=2.0)` in each frame (the owner's untracked
# mean_field_ref_dont_commit/kite_reference/kite_direct.py): ChargeBasis(24) on the compact mode and a grid of 160
# points centred at zero on each extended mode, with the same order-8 stencil as PhaseGrid's default.
# This block builds those bases with circuit-quantization, solves the same three-leaf tree on them (48 states kept per
# leaf, the root by Auto()), and prints the levels beside the paper's; it checks nothing.
const REFGRID_KEEP = 48
if REFGRID
    println("\nthe reference's own grids (KITE_FRAMES_REFGRID=1): $REFGRID_KEEP states kept per leaf")
    product_state = pyimport("circuit_quantization.product_state")
    for (name, ref) in ((:published, REF_PUBLISHED), (:optimum, REF_OPTIMIZED))
        S = frames[name]
        h = qc.handle.transformed(pyimport("numpy").array(S))
        spec = Spec(qc, S)
        bases = collect(product_state.default_bases(h; charge_cutoff=24, max_points=160, extent_scale=2.0))
        leaves = map(enumerate(spec.modes)) do (i, m)
            b = bases[i]
            basis = if pyconvert(Bool, b.compact)
                ChargeBasis(pyconvert(Int, b.cutoff))
            else
                extent, points = pyconvert(Float64, b.extent), pyconvert(Int, b.points)
                nodes = range(-extent, extent; length=points)
                @printf("  %-9s mode %d: PhaseGrid(-%.6f, %.6f, %d), step %.4f; largest node difference %.0e\n",
                        name, i, extent, extent, points, step(nodes),
                        maximum(abs.(collect(nodes) - pyconvert(Vector{Float64}, b.grid))))
                PhaseGrid(-extent, extent, points)
            end
            Node([m.name => basis]; solver=Dense(), compute=REFGRID_KEEP, keep=lowest(REFGRID_KEEP))
        end
        root = Node(leaves; solver=Auto(), compute=NEV, keep=lowest(NEV))
        t = @elapsed Er = energies(materialize(root, spec, (;)))
        @printf("  %-9s reference grid %s   (%.0f s)\n", name, fmt(Er), t)
        @printf("  %-9s paper          %s\n", "", fmt(ref))
        @printf("  %-9s difference     %s\n", "", diffs(Er - ref))
    end
end

# --- Timed frame-change loop ------------------------------------------------------------------------------------------
# Random admissible S near the optimum, for the frame change φ = S φ′ with mode types (c, e, e).
# Admissible entries are the compact block (kept the integer 1), the compact row's extended columns S[1, 2:3] and the
# extended block S[2:3, 2:3]; the extended rows' compact column S[2:3, 1] stays exactly zero.
const S_OPT = frames.optimum
function random_frame(rng)
    S = copy(S_OPT)
    S[1, 2:3] .+= 0.02 .* randn(rng, 2)
    S[2:3, 2:3] .+= 0.02 .* randn(rng, 2, 2)
    return S
end

all_constant(spec) = all(t -> t.coeff.f isa _Constant, circuit_terms(spec)) && constant_term(spec).f isa _Constant

# One iteration: the frame change alone, then the frame change through the first eigensolve, and the full materialize.
function iteration(rng)
    S = random_frame(rng)
    stamp = FirstStamp()
    c0 = compile_ns()
    t0 = time_ns()
    spec = Spec(qc, S)
    t1 = time_ns()
    tree = kite_tree(spec, merge(LOOP, (solver=stamp,)))
    sys = materialize(tree, spec, (;))
    t2 = time_ns()
    c2 = compile_ns()
    return (frame=(t1 - t0) / 1e6, setup=(stamp.t - t0) / 1e6, full=(t2 - t0) / 1e6,
            compile_setup=(stamp.c - c0) / 1e6, compile_rest=(c2 - stamp.c) / 1e6, spec=spec, sys=sys)
end

rng = Xoshiro(20261110)
for _ in 1:5                                     # warm-up
    iteration(rng)
end
GC.gc()
rgf_before = rgf_count()
Base.cumulative_compile_timing(true)
samples = NamedTuple[]
for k in 1:ITERS
    r = iteration(rng)
    all_constant(r.spec) || push!(failures, "iteration $k: a coefficient of the spec is not a _Constant")
    push!(samples, (frame=r.frame, setup=r.setup, full=r.full, compile_setup=r.compile_setup,
                    compile_rest=r.compile_rest, E0=energies(r.sys)[1]))
end
Base.cumulative_compile_timing(false)
rgf_after = rgf_count()
# The probe: compiling one coefficient grows the count by one, so the measure is live.
Coefficient("2 * E_J", [:E_J])
rgf_probe = rgf_count()

mean(x) = sum(x) / length(x)
frame_t, setup_t, full_t = (mean(getfield.(samples, f)) for f in (:frame, :setup, :full))
println("\ntimed loop: $ITERS iterations of random admissible S near the optimum (after 5 warm-up), 1 thread")
@printf("  frame change alone, Spec(qc, S):                 %8.3f ms mean\n", frame_t)
@printf("  frame change through materialize setup:          %8.3f ms mean\n", setup_t)
@printf("  frame change through the full materialize:       %8.3f ms mean\n", full_t)
n = min(20, ITERS)
@printf("  setup time, first %d / last %d iterations:       %8.3f / %.3f ms\n", n, n,
        mean(getfield.(samples[1:n], :setup)), mean(getfield.(samples[end-n+1:end], :setup)))
@printf("  ground energy over the loop: %.4f .. %.4f GHz\n", extrema(getfield.(samples, :E0))...)

println("\ncompilation check")
println("  every coefficient a _Constant in every iteration: ", !any(startswith("iteration"), failures))
@printf("  RuntimeGeneratedFunctions cache entries: %d before the loop, %d after\n", rgf_before, rgf_after)
@printf("  RuntimeGeneratedFunctions cache entries after compiling one probe coefficient: %d\n", rgf_probe)
rgf_after == rgf_before || push!(failures, "the RuntimeGeneratedFunctions cache grew by $(rgf_after - rgf_before)")
rgf_probe == rgf_after + 1 || push!(failures, "the probe coefficient did not grow the RuntimeGeneratedFunctions cache")
# Compilation in the setup window fails the run; compilation in the eigensolve window only warns (see the header).
for (label, f, fatal) in (("frame change through materialize setup", :compile_setup, true),
                          ("first eigensolve to end of materialize", :compile_rest, false))
    c = getfield.(samples, f)
    hit = findall(>(0), c)
    @printf("  Julia compile time, %s: %.3f ms total, in %d of %d iterations%s\n", label, sum(c), length(hit), ITERS,
            isempty(hit) ? "" : " (" * join((@sprintf("#%d: %.3f ms", k, c[k]) for k in hit[1:min(end, 8)]), ", ") *
                                ")")
    isempty(hit) && continue
    message = @sprintf("Julia compiled for %.3f ms (%s) in %d timed iterations", sum(c), label, length(hit))
    fatal ? push!(failures, message) : @warn message * "; no coefficient was compiled if the cache count is unchanged"
end

# --- Run record -------------------------------------------------------------------------------------------------------
mkpath(OUT)
path = joinpath(OUT, "kite_optimum.json")
write(path, run_record(res; frame=S_OPT, values=θ,
                       metadata=(bench="kite_frames", energies_GHz=E[:optimum], grid=GRID)))
rec = load_record(path)
round_trip = rec.frame == S_OPT && rec.values == NamedTuple{Tuple(Spec(res).parameters)}(θ)   # the circuit's order
round_trip || push!(failures, "load_record of $path does not return the frame and values written")
println("\nrun record: ", path, " (load_record returns its frame and values: ", round_trip, ")")

println("fix_phases! warnings dropped: ", LOGGER.dropped[])

if !isempty(failures)
    println(stderr, "\nFAILED:\n  ", join(failures, "\n  "))
    exit(1)
end
println("\nall checks passed")
