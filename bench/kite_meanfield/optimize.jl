# The frame optimization of the KITE on the product energy (design §5): minimize E_prod(S) + 1e-4 F_NM(S) over the
# admissible frames S (φ = S φ′) of circuit-quantization's parameterization, from the identity, in one Julia session that
# reaches circuit-quantization through this package's PythonCall extension.
#
#   JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=bench/kite_meanfield bench/kite_meanfield/optimize.jl
#
# KITE_OPTIMIZE_OUT       directory of the results (default: results/ beside this script; not committed)
# KITE_MEANFIELD_MAXITER  the optimizer's iteration limit (default 150)
#
# Every cost evaluation solves on suggest_bases(spec; charge_cutoff=16, max_points=400); a frame for which the formula of
# suggest_bases asks for more than 400 points on some mode scores Inf, so no score comes from a capped (under-resolved) grid.
# The optimizer stops at the gradient tolerance 1e-6, a cost change below 1e-9 GHz between iterations, a failed line
# search, or KITE_MEANFIELD_MAXITER (default 150) iterations.
#
# Exits nonzero unless the final frame, on the grids of levels.jl with its 33 starts, has exactly two distinct solutions
# within 0.5 GHz of the lowest, both within 2e-3 GHz of -9.4295 GHz, or if the lowest product energy at the final frame
# changes by more than 1e-6 GHz when its noncompact columns are scaled by (1.3, 0.8) or one of them changes sign.

using KronFreeCircuits, PythonCall, Optim, LinearAlgebra, Logging, Printf
using KronFreeCircuits: modes
include(joinpath(@__DIR__, "..", "..", "test", "python", "kite.jl"))
include(joinpath(@__DIR__, "starts.jl"))

# Two known messages are counted and dropped, every other log message passes: `fix_phases!` on the compact leaf when
# every cosine expectation is zero (its bare solve and the delocalized start), and the PhaseGrid edge warning of a kept
# state, which the score checks itself, on the lowest state, with `edge_weight`.
struct KnownWarnings <: AbstractLogger
    inner::AbstractLogger
    phases::Base.RefValue{Int}
    edges::Base.RefValue{Int}
end
Logging.min_enabled_level(l::KnownWarnings) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::KnownWarnings, args...) = Logging.shouldlog(l.inner, args...)
function Logging.handle_message(l::KnownWarnings, level, message, args...; kw...)
    s = string(message)
    startswith(s, "fix_phases!") && return (l.phases[] += 1; nothing)
    startswith(s, "PhaseGrid on mode") && return (l.edges[] += 1; nothing)
    return Logging.handle_message(l.inner, level, message, args...; kw...)
end
const LOGGER = KnownWarnings(ConsoleLogger(stderr), Ref(0), Ref(0))
global_logger(LOGGER)

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run single-threaded (julia -t 1), got $(Threads.nthreads()) threads")

const OUT = get(ENV, "KITE_OPTIMIZE_OUT", joinpath(@__DIR__, "results"))
const FIXTURE = joinpath(@__DIR__, "..", "..", "test", "fixtures", "kite", "meanfield_reference.json")
const KEEP = 4                  # states kept per leaf; the product energy uses the lowest
const REG = 1e-4                # weight of F_NM in the cost
const EDGE_TOL = 1e-8           # largest allowed edge_weight² of a leaf's lowest state before its grid grows
const GROWTHS = 3               # at most this many grid growths per evaluation
const STEP = 1e-6               # forward-difference step of the gradient, per parameter
const MAX_POINTS = 400          # the score's grid cap (circuit-quantization's default); a frame that needs more scores Inf
const UNCAPPED = 10^9           # max_points of the uncapped call that tells whether the cap bound
const RESOLUTION_WARN = 1e-5    # GHz, printed as a warning when the final frame moves by more on doubled grids
const MAXITER = parse(Int, get(ENV, "KITE_MEANFIELD_MAXITER", "150"))
# A capped run must not overwrite the recorded results of the full run in the default directory.
MAXITER != 150 && normpath(abspath(OUT)) == normpath(joinpath(@__DIR__, "results")) &&
    error("KITE_MEANFIELD_MAXITER = $MAXITER differs from the default 150: a capped run must set KITE_OPTIMIZE_OUT to " *
          "another directory than $OUT, so that it does not overwrite the recorded results")
# GHz, the product energy of both wells at the paper's optimum frame: the committed fixture
# test/fixtures/kite/meanfield_reference.json ("optimum", -9.429473599561 and -9.429473599557 on the figure's bases),
# rounded to four decimals.
const TARGET = -9.4295
const TARGET_TOL = 2e-3         # GHz
const WINDOW = 0.5              # GHz, the wells: distinct solutions within this of the lowest
const INVARIANCE_TOL = 1e-6     # GHz

loadavg() = join((@sprintf("%.2f", x) for x in Sys.loadavg()), " ")
const T0 = time()
failures = String[]
println("KITE frame optimization on the product energy")
println("Julia $(VERSION), $(Threads.nthreads()) thread, BLAS threads $(BLAS.get_num_threads()); load average at start: ", loadavg())

# --- The pieces -----------------------------------------------------------------------------------------------------------
# The KITE at its parameter values, evaluated once; circuit-quantization's admissible family of frames (the compact block
# fixed at the identity, the compact–noncompact shear free, the noncompact block exp(A) exp(B)) and its scale-free
# normal-mode cost, both from the base frame's numeric Hamiltonian `QC.handle`.
const KITE = kite()
const RES, θ = KITE[1], KITE[2]
const QC = QuantizedCircuit(RES; values=θ)
const NP = pyimport("numpy")
const NUMERICAL = pyimport("circuit_quantization.numerical")
const FAMILY = NUMERICAL.parameterization(QC.handle.types; relaxed=true)
const NM_COST = NUMERICAL.cost_normal_modes(NP.linalg.inv(QC.handle.E_C), QC.handle.E_L, QC.handle.types)
frame(x) = pyconvert(Matrix{Float64}, FAMILY.S(NP.asarray(x)))
F_NM(S) = pyconvert(Float64, NM_COST(NP.asarray(S)))

# One leaf per mode under a root.
leaf((m, b)) = Node([m => b]; solver=Dense(), compute=KEEP, keep=lowest(KEEP), arithmetic=:complex)
tree(bases) = Node(map(leaf, bases))

# The five starts of every evaluation: all modes at zero; the compact mode at π; the last extended mode at π; both; and
# the delocalized start.
const NAMES = QC.mode_names
const COMPACT, LAST_EXTENDED = NAMES[1], NAMES[end]
const STARTS = Any[Dict(m => 0.0 for m in NAMES), Dict(COMPACT => π), Dict(LAST_EXTENDED => π),
                   Dict(COMPACT => π, LAST_EXTENDED => π), nothing]

# The previous evaluation's lowest solution as a warm start on `root`.
# A warm start must be a solution of `root`, and the tables' keys move with the frame (the cosines' coefficients in the
# new coordinates), so each leaf's table is recomputed, by mode name, from the previous solution's state of that mode.
function carry(prev::MeanFieldSolution, root::Node, terms)
    old = Dict(only(modes(ch)) => c.solved for (ch, c) in prev.children)
    table(l) = expectations(old[only(modes(l))], child_keys(root, terms, l))
    children = IdDict{Node,MeanFieldChild}(l => MeanFieldChild(l, old[only(modes(l))], Term[], NaN, table(l))
                                           for l in root.children)
    return MeanFieldSolution(NaN, children, 0, false, :warm)
end

# The lowest solution over the starts and the warm start, on `bases`; while the lowest state of an extended leaf of the
# lowest solution has edge_weight² above EDGE_TOL, that grid grows by 1.5 in extent and points and the evaluation is
# repeated, at most GROWTHS times. Returns the solution, the number of growths, and whether a warm start was used.
function lowest_solution(spec, bases)
    terms = circuit_terms(spec)
    for growth in 0:GROWTHS
        root = tree(bases)
        starts = WARM[] === nothing ? STARTS : [STARTS; carry(WARM[], root, terms)]
        sol = first(mean_field_solutions(root, spec, (;); starts))
        COUNT.runs[] += length(starts)
        wide = [i for (i, l) in enumerate(root.children) if edge_weight(sol.children[l].solved)^2 > EDGE_TOL]
        (isempty(wide) || growth == GROWTHS) && return sol, growth, WARM[] !== nothing, bases
        for i in wide
            (m, g) = bases[i]
            bases[i] = m => PhaseGrid(1.5 * g.a, 1.5 * g.b, ceil(Int, 1.5 * g.N); order=g.order)
        end
    end
end

# Every evaluation, in order: (x, cost, E_prod, F_NM, growths, warm, bases).
const HISTORY = NamedTuple[]
const COUNT = (evaluations=Ref(0), runs=Ref(0), grew=Ref(0), no_warm=Ref(0), inadmissible=Ref(0), capped=Ref(0))
const WARM = Ref{Union{Nothing,MeanFieldSolution}}(nothing)

# The bookkeeping of one evaluation: the counters, the history, the next warm start; returns the cost.
function record!(x, sol, F; growths=0, warm=false, bases=nothing, capped=false)
    COUNT.evaluations[] += 1
    if sol === nothing
        (capped ? COUNT.capped : COUNT.inadmissible)[] += 1
    end
    sol !== nothing && !warm && (COUNT.no_warm[] += 1)
    growths > 0 && (COUNT.grew[] += 1)
    WARM[] = sol
    cost = sol === nothing ? Inf : sol.energy + REG * F
    push!(HISTORY, (x=copy(x), cost=cost, E=sol === nothing ? NaN : sol.energy, F=F, growths=growths, warm=warm,
                    bases=bases))
    return cost
end

# The score of the frame S(x): the lowest product energy plus REG times F_NM; Inf for a frame circuit-quantization
# rejects, and for one whose grids suggest_bases had to cap.
function frame_score(x)
    S = frame(x)
    spec = try
        Spec(QC, S)
    catch err
        err isa ArgumentError || rethrow()
        return record!(x, nothing, NaN)
    end
    bases = suggest_bases(spec; charge_cutoff=16, max_points=MAX_POINTS, when_capped=:allow)   # capping detected below
    points(bs) = [b isa PhaseGrid ? b.N : 0 for (_, b) in bs]
    points(bases) == points(suggest_bases(spec; charge_cutoff=16, max_points=UNCAPPED, when_capped=:allow)) ||
        return record!(x, nothing, NaN; capped=true)
    sol, growths, warm, bases = lowest_solution(spec, bases)
    return record!(x, sol, F_NM(S); growths, warm, bases)
end

# --- The optimization -----------------------------------------------------------------------------------------------------
# BFGS from x = 0 with this script's forward differences (step STEP in each parameter); the value at x is the one already
# computed when Optim asks for the gradient there.
# A non-finite cost at x gives a gradient of NaN, with no neighbour evaluated, and Optim's line search (HagerZhang) backs
# off. A component whose forward neighbour scores non-finite takes the backward difference instead; if that neighbour is
# non-finite too, the frame is boxed in and the run stops with an error.
const LAST = Ref{Any}(nothing)
function cost(x; score=frame_score)
    LAST[] !== nothing && LAST[][1] === score && LAST[][2] == x && return LAST[][3]
    f = score(x)
    LAST[] = (score, copy(x), f)
    return f
end
function difference(score, x, i, f0)
    xi = copy(x)
    xi[i] = x[i] + STEP
    f = score(xi)
    isfinite(f) && return (f - f0) / STEP
    xi[i] = x[i] - STEP
    b = score(xi)
    isfinite(b) && return (f0 - b) / STEP
    error("gradient component $i: the frame is boxed in by non-finite scores on both sides (x[$i] ± $STEP)")
end
function gradient!(G, x; score=frame_score)
    f0 = cost(x; score)
    isfinite(f0) || return fill!(G, NaN)
    for i in eachindex(x)
        G[i] = difference(score, x, i, f0)
    end
    return G
end

ITERATIONS = NamedTuple[]
mkpath(OUT)
# Emptied at start-up and appended at every iteration, so a run that ends badly can be examined.
const ITERLOG = joinpath(OUT, "iterations.csv")
function report(state)
    h = HISTORY[findlast(r -> r.x == state.x, HISTORY)]
    gnorm = norm(state.g_x, Inf)
    push!(ITERATIONS, (iteration=length(ITERATIONS), cost=state.f_x, E=h.E, F=h.F, gnorm=gnorm,
                       evaluations=COUNT.evaluations[]))
    @printf("iter %3d  cost %.12f  E_prod %.12f  F_NM %.6e  |g|_inf %.3e  evaluations %5d  [%.0f s]\n",
            length(ITERATIONS) - 1, state.f_x, h.E, h.F, gnorm, COUNT.evaluations[], time() - T0)
    flush(stdout)
    open(ITERLOG, "a") do io
        length(ITERATIONS) == 1 && println(io, "iteration,cost,E_prod,F_NM,gnorm_inf,evaluations,", join(("x$i" for i in eachindex(state.x)), ","))
        r = ITERATIONS[end]
        println(io, r.iteration, ",", r.cost, ",", r.E, ",", r.F, ",", r.gnorm, ",", r.evaluations, ",", join((repr(v) for v in state.x), ","))
    end
    return false
end

const NPAR = pyconvert(Int, FAMILY.size)
println("parameters: $NPAR; starts per evaluation: $(length(STARTS)) and the previous evaluation's solution")
rm(ITERLOG; force=true)
# successive_f_tol = 0: one iteration with a cost change of at most 1e-9 GHz ends the run.
result = optimize(cost, gradient!, zeros(NPAR), BFGS(),
                  Optim.Options(g_abstol=1e-6, f_abstol=1e-9, successive_f_tol=0, iterations=MAXITER, callback=report))
println()
println(result)
stop = result.stopped_by
println("the optimizer stopped by: ",
        stop.g_converged ? "the gradient tolerance 1e-6" :
        stop.f_converged ? "a cost change below 1e-9 GHz between iterations" :
        stop.ls_failed ? "a failed line search" :
        stop.iterations ? "the limit of $MAXITER iterations" : "another reason: $(Optim.termination_code(result))")
x_opt = Optim.minimizer(result)
T_OPT = time() - T0
RUNS_OPT = COUNT.runs[]
@printf("optimization: %d iterations, %d cost evaluations, %d self-consistent runs, %d evaluations grew a grid, %d without a warm start (the first included), %d inadmissible, %d rejected as needing more than %d points; %.1f s\n",
        Optim.iterations(result), COUNT.evaluations[], COUNT.runs[], COUNT.grew[], COUNT.no_warm[],
        COUNT.inadmissible[], COUNT.capped[], MAX_POINTS, T_OPT)

# The final score's resolution: the lowest product energy of the final frame on the score's own grids (as grown) with
# twice the points over the same extents, beside the score's value.
h_opt = HISTORY[findlast(r -> r.x == x_opt, HISTORY)]
doubled = [b isa PhaseGrid ? m => PhaseGrid(b.a, b.b, 2 * b.N; order=b.order) : m => b for (m, b) in h_opt.bases]
spec_opt = Spec(QC, frame(x_opt))
E_doubled = first(mean_field_solutions(tree(doubled), spec_opt, (;); starts=STARTS)).energy
COUNT.runs[] += length(STARTS)
@printf("\nfinal frame on the score's grids (%s): E_prod %.12f; on twice the points: %.12f; difference %.1e GHz\n",
        join((b isa PhaseGrid ? @sprintf("%s ±%.4f %d points", m, b.b, b.N) : "$m ChargeBasis($(b.ncut))" for (m, b) in h_opt.bases), ", "),
        h_opt.E, E_doubled, E_doubled - h_opt.E)
abs(E_doubled - h_opt.E) > RESOLUTION_WARN &&
    @printf("WARNING: the final score is not resolved to %.0e GHz\n", RESOLUTION_WARN)

# --- The final frame ------------------------------------------------------------------------------------------------------
# The noncompact columns rescaled so that (Sᵀ C S)_jj = 1, with C the inverse of E_C of the base frame.
S = frame(x_opt)
CMAT = inv(pyconvert(Matrix{Float64}, QC.handle.E_C))
noncompact = findall(!=("c"), pyconvert(Vector{String}, QC.handle.types))
for j in noncompact
    S[:, j] ./= sqrt((S' * CMAT * S)[j, j])
end
println("\nfinal frame S (φ = S φ′), noncompact columns scaled to (Sᵀ C S)_jj = 1:")
for i in axes(S, 1)
    println("  ", join((@sprintf("%10.6f", v) for v in S[i, :]), " "))
end

# The wells of a frame on the grids of levels.jl, from its 33 starts: the distinct solutions within WINDOW of the lowest.
function wells(S)
    spec = Spec(QC, S)
    root = tree(suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5, when_capped=:allow))
    sols = mean_field_solutions(root, spec, (;); starts=kite_starts(spec))
    COUNT.runs[] += length(kite_starts(spec))
    return filter(s -> s.energy - sols[1].energy <= WINDOW, sols)
end

final = wells(S)
reference = [pyconvert(Float64, s["energy"]) for s in pyimport("json").loads(read(FIXTURE, String))["optimum"]["solutions"]]
println("\nsolutions within $WINDOW GHz of the lowest, final frame, suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5), 33 starts:")
for (i, s) in enumerate(final)
    @printf("  well %d: E_prod = %.12f GHz, %d iterations, converged %s\n", i, s.energy, s.iterations, s.converged)
end
println("the paper's optimum frame on the same bases (test/fixtures/kite/meanfield_reference.json): ",
        join((@sprintf("%.12f", e) for e in reference), ", "), " GHz")
length(final) == 2 || push!(failures, "the final frame has $(length(final)) solutions within $WINDOW GHz of the lowest, expected 2")
for (i, s) in enumerate(final)
    abs(s.energy - TARGET) <= TARGET_TOL ||
        push!(failures, @sprintf("well %d: E_prod %.6f is not within %.0e GHz of %.4f", i, s.energy, TARGET_TOL, TARGET))
    s.converged || push!(failures, "well $i: the self-consistent loop did not converge")
end

# --- Invariances ----------------------------------------------------------------------------------------------------------
# The lowest product energy at the final frame, with its noncompact columns scaled by (1.3, 0.8), and with the sign of
# the first noncompact column flipped; grids from suggest_bases in each frame.
scaled = copy(S)
scaled[:, noncompact] .*= [1.3 0.8]
flipped = copy(S)
flipped[:, noncompact[1]] .*= -1
E_final = final[1].energy
E_scaled = first(wells(scaled)).energy
E_flipped = first(wells(flipped)).energy
println("\ninvariances, lowest product energy (GHz):")
@printf("  final frame              %.12f\n", E_final)
@printf("  columns scaled (1.3, 0.8) %.12f  difference %.1e\n", E_scaled, E_scaled - E_final)
@printf("  column %d sign flipped     %.12f  difference %.1e\n", noncompact[1], E_flipped, E_flipped - E_final)
for (what, e) in (("scaled", E_scaled), ("flipped", E_flipped))
    abs(e - E_final) <= INVARIANCE_TOL ||
        push!(failures, @sprintf("invariance (%s): the product energy changed by %.1e GHz", what, e - E_final))
end

# --- Results --------------------------------------------------------------------------------------------------------------
open(joinpath(OUT, "optimum_S.txt"), "w") do io
    for i in axes(S, 1)
        println(io, join((repr(v) for v in S[i, :]), " "))
    end
end
open(joinpath(OUT, "cost_history.csv"), "w") do io
    # points: each mode's basis size (2 ncut + 1 on a ChargeBasis) in the evaluation's final grids, empty for Inf
    size_of(b) = b isa PhaseGrid ? b.N : 2 * b.ncut + 1
    println(io, "evaluation,", join(("x$i" for i in 1:NPAR), ","), ",cost,E_prod,F_NM,growths,warm,points")
    for (k, h) in enumerate(HISTORY)
        pts = h.bases === nothing ? "" : join((size_of(b) for (_, b) in h.bases), "/")
        println(io, k, ",", join((repr(v) for v in h.x), ","), ",", h.cost, ",", h.E, ",", h.F, ",", h.growths, ",", h.warm, ",", pts)
    end
end
save_record(joinpath(OUT, "optimum_record.json"), Spec(QC, S); circuit=RES, frame=S, values=θ,
            metadata=(script="bench/kite_meanfield/optimize.jl", wells=[s.energy for s in final]))
println("\nwrote optimum_S.txt, cost_history.csv, iterations.csv (written at every iteration) and optimum_record.json to $OUT")

println("known warnings dropped: fix_phases! $(LOGGER.phases[]), PhaseGrid edge $(LOGGER.edges[])")
@printf("total: %d cost evaluations, %d self-consistent runs (%d of them after the optimization), wall time %.1f s; load average at end: %s\n",
        COUNT.evaluations[], COUNT.runs[], COUNT.runs[] - RUNS_OPT, time() - T0, loadavg())

if !isempty(failures)
    println(stderr, "\nFAILED:\n  ", join(failures, "\n  "))
    exit(1)
end
println("\nall checks passed")
