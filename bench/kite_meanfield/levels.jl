# Figure 8(b) of the circuit-quantization paper for the KITE: the largest error among the four lowest levels, against the
# paper's reference levels, as a function of the number `m` of states kept per leaf, in three frames, with each leaf's
# basis built from the mean-field states of the two wells of the self-consistent loop.
#
#   julia --project=bench/kite_meanfield bench/kite_meanfield/levels.jl
#
# KITE_MEANFIELD_REFSOLVE 1 to also solve the published frame on the reference calculation's own grids and print its four
#                         lowest levels beside the paper's (default 0, off).
#
# Exits nonzero if any of the twelve errors is outside a factor of two of the reference calculation's value, if the
# ordering of the three frames by error differs from the reference's at any `m`, if the residual of any of the four
# lowest levels of a root solve is above 1e-9 GHz, if a frame does not have exactly two wells, or if the replay of a
# union's merge (printed for the unions that dropped a vector) does not reproduce the union's report.

using KronFreeCircuits, LinearAlgebra, Logging, Printf
using KronFreeCircuits: operator, modes, _edge_support

# --- Known warnings -----------------------------------------------------------------------------------------------------
# Two known messages are counted and dropped, every other log message passes (after a line naming the step):
# - `fix_phases!` on the compact leaf, whenever it is solved with every cosine expectation zero (its bare solve, and the
#   first iteration of the delocalized start; see test/test_meanfield_kite.jl); recorded by step and column;
# - the PhaseGrid edge warning of a leaf that keeps 20 states, which names the kept states touching the grid's ends.
#   The lowest kept state each message names is recorded per mode, and the edge support of every state offered to a
#   union is measured directly (`augmented_children`), as are the edge supports of each well's 20 states.
struct KnownWarnings <: AbstractLogger
    inner::AbstractLogger
    phases::Dict{String,Vector{Int}}         # step => the column named by each fix_phases! message
    edges::Dict{String,Vector{Int}}          # mode => the lowest kept state named by each edge message
end
Logging.min_enabled_level(l::KnownWarnings) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::KnownWarnings, args...) = Logging.shouldlog(l.inner, args...)
function Logging.handle_message(l::KnownWarnings, level, message, args...; kw...)
    s = string(message)
    p = match(r"^fix_phases!: column (\d+) has no weight on the phase mask", s)
    p === nothing || return (push!(get!(l.phases, CONTEXT[], Int[]), parse(Int, p[1])); nothing)
    e = match(r"^PhaseGrid on mode :(\S+) is too small: kept states \[([0-9, ]+)\]", s)
    if e !== nothing
        push!(get!(l.edges, e[1], Int[]), minimum(parse.(Int, split(e[2], ","))))
        return nothing
    end
    # any other message passes, after a line on stdout naming the step that raised it
    println(stdout, "  [a log message on stderr was raised during: ", CONTEXT[], "]")
    flush(stdout)
    return Logging.handle_message(l.inner, level, message, args...; kw...)
end
const CONTEXT = Ref("start")
const LOGGER = KnownWarnings(ConsoleLogger(stderr), Dict{String,Vector{Int}}(), Dict{String,Vector{Int}}())
global_logger(LOGGER)

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run single-threaded (julia -t 1), got $(Threads.nthreads()) threads")

const REFSOLVE = get(ENV, "KITE_MEANFIELD_REFSOLVE", "0") == "1"
const FIXTURES = joinpath(@__DIR__, "..", "..", "test", "fixtures", "kite")
const FRAMES = (:published, :normal, :optimum)
const LABEL = Dict(:published => "published", :normal => "normal modes", :optimum => "optimum")
const MS = (8, 12, 16, 20)
const KEEP = 20                       # mean-field states kept per leaf: m goes to 20
const NLEV = 8                        # levels solved at the root
const RES_TOL = 1e-9                  # GHz, the largest allowed residual of the four lowest levels
const WINDOW = 0.5                    # GHz, the wells: distinct solutions within this of the lowest
const DISTINCT = 1e-3                 # mean_field_solutions' default distinctness of two solutions' tables

# --- The reference --------------------------------------------------------------------------------------------------------
# The paper's reference levels (GHz): the four lowest levels in the published coordinates by direct diagonalization on
# 1 254 400 states.
const REF = [-9.523995, -9.510525, -8.453790, -8.337862]

# The reference calculation's eight lowest levels (GHz) per frame and m, copied from the owner's untracked result files
# mean_field_ref_dont_commit/kite_reference/results/levels_{paper,normal_modes,optimized}.json (key "levels").
# Its twelve errors are computed from them below exactly as this run's: max(levels[1:4] - REF); rounded, they are the
# table of Figure 8(b): published 6.918e-1 2.213e-1 3.770e-2 6.776e-4, normal modes 6.218e-1 1.989e-1 3.600e-2 8.012e-4,
# optimum 3.668e-2 2.268e-3 2.609e-4 2.072e-5.
const REF_LEVELS = Dict(
    :published => Dict(
        8 => [-9.414960858464323, -9.41450668673451, -7.7619977562747104, -7.757393663606695,
              -5.10642951736468, -5.100974260989835, -0.06364953519266779, 0.021048845103172252],
        12 => [-9.497816097843195, -9.493723539672688, -8.232454380076422, -8.19852232268035,
               -6.571364096829432, -6.499640479561075, -4.3365757413386286, -4.069954549256892],
        16 => [-9.52041014822982, -9.509031393832162, -8.416088939403183, -8.319838892723942,
               -7.263077944942406, -6.9785375200942985, -5.979429018989293, -5.448632692920086],
        20 => [-9.523931886695816, -9.510498707591784, -8.453112425566, -8.337570947508341,
               -7.445836769839516, -7.072306310775001, -6.342993118610245, -5.808201268269957]),
    :normal => Dict(
        8 => [-9.427727310284647, -9.427283743218508, -7.8320339171108, -7.827762123468264,
              -5.297847468416586, -5.29320136904544, -0.9464627836864284, -0.8931742254404703],
        12 => [-9.50040413568868, -9.496378990315819, -8.254892862423942, -8.222611781348684,
               -6.642891369998425, -6.578399668685465, -4.486123686259544, -4.287402111734894],
        16 => [-9.520627381260748, -9.509399502574865, -8.41778518361917, -8.323776248493832,
               -7.2637268253057945, -6.991397807708723, -5.962798643068583, -5.463103045648059],
        20 => [-9.523919431810711, -9.51049907311002, -8.452988760498345, -8.337600803812862,
               -7.444930280032849, -7.072340075740534, -6.34085074133103, -5.807030540030452]),
    :optimum => Dict(
        8 => [-9.517404605919115, -9.503563775081371, -8.42868314632316, -8.301185492916394,
              -7.2996248136114215, -6.92018786028034, -6.120256588938493, -5.542058831824872],
        12 => [-9.52361009283952, -9.510097417387678, -8.452098489091341, -8.335594130551563,
               -7.444811093954891, -7.066716444987796, -6.334026134597757, -5.7991179445472625],
        16 => [-9.523938546006383, -9.510459140208946, -8.453636952859858, -8.33760107344629,
               -7.449518211441264, -7.073596696545526, -6.349066491045498, -5.814731253118485],
        20 => [-9.52399097452997, -9.510520546020862, -8.453776364530471, -8.337841282738882,
               -7.449944812387171, -7.074068102374226, -6.3514931818791585, -5.817025909945813]))

error_of(levels) = maximum(levels[1:4] - REF)
const REF_ERR = Dict(f => Dict(m => error_of(REF_LEVELS[f][m]) for m in MS) for f in FRAMES)

# The product-state gaps E_prod - REF[1] (GHz): the paper's quoted values, and the reference calculation's own (from
# "E_prod" of the same result files).
const GAP_PAPER = Dict(:published => 1.44, :optimum => 0.09)
const GAP_REFCALC = Dict(:published => 1.440807, :optimum => 0.094521)

# Appendix C.4 of the paper, optimum frame, from the owner's log mean_field_ref_dont_commit/kite_reference/results/leaves.log,
# on another grid (extent_scale = 1.0): for reference only, nothing is checked against them.
const C4_OTHER_GRID = (one_well = [2.3, 2.1e-1, 2.2e-2, 2.9e-3], bare = [4.1e-1, 3.3e-2, 4.4e-3, 4.8e-4],
                       two_wells = [3.7e-2, 2.3e-3, 2.7e-4, 2.6e-5])

loadavg() = join((@sprintf("%.2f", x) for x in Sys.loadavg()), " ")
failures = String[]

# --- The tree -------------------------------------------------------------------------------------------------------------
# One leaf per mode on the bases of suggest_bases (ChargeBasis(16) on the compact mode, two 120-point phase grids),
# each keeping its KEEP lowest mean-field states, and a root over them.
function kite_leaves(spec)
    return [Node([m => b]; solver=Dense(), compute=KEEP, keep=lowest(KEEP), arithmetic=:complex)
            for (m, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5, when_capped=:allow)]
end

# The root over `leaves`, solved for its NLEV lowest levels: densely up to 1000 product states, else by LOBPCG with a
# tolerance below RES_TOL (the residual is checked independently in `solve_root`).
root_over(leaves, N) = Node(leaves; solver=N <= 1000 ? Dense() : LOBPCG(tol=1e-10), compute=NLEV, keep=lowest(NLEV))

# The 33 starts of test/test_meanfield_kite.jl (`kite_starts`).
include(joinpath(@__DIR__, "starts.jl"))

# The largest change of any table entry between two solutions: mean_field_solutions' own distinctness measure.
table_change(a, b) = maximum(abs(a.children[ch].expectations[k] - b.children[ch].expectations[k])
                             for ch in keys(a.children) for k in keys(a.children[ch].expectations))

# The augmented children of `leaves` from `sources(leaf)`, each the union of its sources to m states, with its report
# printed under `label`.
# A union that cannot reach m is printed and rebuilt with allow_fewer = true; `short` records each such leaf.
# The merge is level by level, so a source's states 1 .. taken + dropped are the ones it offered; on a PhaseGrid leaf the
# largest edge support among them (the measure of the edge warning) is printed and kept in EDGE_USED.
const EDGE_USED = Ref(0.0)
function augmented_children(root, spec, leaves, bare, sources, m, short, label)
    terms = circuit_terms(spec)
    out = AugmentedChild[]
    for leaf in leaves
        srcs = sources(leaf)
        U, report = try
            union_basis(srcs; m)
        catch err
            err isa ArgumentError || rethrow()
            println("      union of ", only(modes(leaf)), " at m = $m cannot reach m:\n", err.msg)
            U, report = union_basis(srcs; m, allow_fewer=true)
            push!(short, (label=label, mode=only(modes(leaf)), m=m, states=size(U, 2)))
            U, report
        end
        g = only(leaf.children).second
        edge = ""
        if g isa PhaseGrid
            e = maximum(maximum(_edge_support(s.vectors[:, 1:report.taken[i] + report.dropped[i]], [g.N], 1, g.order ÷ 2);
                                init=0.0) for (i, s) in enumerate(srcs))
            EDGE_USED[] = max(EDGE_USED[], e)
            edge = @sprintf("; largest edge support of the states offered %.1e", e)
        end
        if any(>(0), report.dropped)
            r = remainders(srcs, m)
            report.dropped == [count(x -> x.source == i && x.norm <= 1e-6, r) for i in eachindex(srcs)] ||
                push!(failures, "the replayed merge of $(only(modes(leaf))) at m = $m does not reproduce the union's report")
            println("    $label m = $m, ", only(modes(leaf)), ": Gram-Schmidt remainders below 1e-3 (drop threshold 1e-6): ",
                    join((@sprintf("%s state %d %.1e", report.labels[x.source], x.level, x.norm) for x in r if x.norm < 1e-3), ", "))
        end
        println("    $label m = $m, ", only(modes(leaf)), ": ",
                join(("$(report.labels[i]) taken $(report.taken[i]) dropped $(report.dropped[i])"
                      for i in eachindex(report.labels)), ", "), edge,
                size(U, 2) < m ? "  (SHORT: $(size(U, 2)) states)" : "")
        push!(out, augment(root, terms, (;), leaf, bare[leaf], U, report))
    end
    return out
end

# The remainder norm of every candidate of a union, replaying union_basis's merge (level by level, classical Gram-Schmidt
# twice, drop threshold 1e-6, stop at m): printed for the unions that dropped a vector, to show how close to the
# threshold the dropped and the accepted candidates are.
function remainders(srcs, m)
    U = zeros(ComplexF64, size(srcs[1].vectors, 1), 0)
    out = NamedTuple{(:source, :level, :norm),Tuple{Int,Int,Float64}}[]
    level = 1
    while size(U, 2) < m && any(size(s.vectors, 2) >= level for s in srcs)
        for (i, s) in enumerate(srcs)
            level <= size(s.vectors, 2) || continue
            v = Vector{ComplexF64}(s.vectors[:, level])
            for _ in 1:2
                v -= U * (U' * v)
            end
            push!(out, (source=i, level=level, norm=norm(v)))
            if norm(v) > 1e-6
                U = hcat(U, v / norm(v))
                size(U, 2) == m && break
            end
        end
        level += 1
    end
    return out
end

# The root over the augmented children: its eight lowest levels, and the largest residual of the four lowest.
function solve_root(spec, leaves, aug)
    N = prod(size(a.solved.vectors, 2) for a in aug)
    root = root_over(leaves, N)
    sys = materialize(root, spec, (;); augmented=aug)
    E = energies(sys)
    H = operator(sys)
    V = sys.root.vectors
    res = maximum(norm(H * V[:, j] - E[j] * V[:, j]) for j in 1:4)
    return E, res, N
end

fmt(x) = @sprintf("%.4e", x)

# --- Three frames ---------------------------------------------------------------------------------------------------------
println("KITE, Figure 8(b): mean-field leaves, m = $(join(MS, ", ")), $KEEP mean-field states kept per leaf")
println("Julia $(VERSION), $(Threads.nthreads()) thread, BLAS threads $(BLAS.get_num_threads()); load average at start: ", loadavg())
ERR = Dict{Symbol,Dict{Int,Float64}}()
LEVELS = Dict{Symbol,Dict{Int,Vector{Float64}}}()
SHORT = NamedTuple[]
WELLS = Dict{Symbol,Any}()
VARIANTS = Dict{Symbol,Dict{Int,Float64}}(:one_well => Dict(), :bare => Dict())
WALL = Dict{Symbol,Float64}()
for f in FRAMES
    t0 = time()
    spec = load_spec(joinpath(FIXTURES, "$f.json"))
    leaves = kite_leaves(spec)
    terms = circuit_terms(spec)
    println("\n=== $(LABEL[f]) frame: modes ", join((string(m.name, " (", m.type, ")") for m in spec.modes), ", "))
    for leaf in leaves
        (m, b) = only(leaf.children)
        println("  ", m, ": ", b isa ChargeBasis ? "ChargeBasis($(b.ncut))" :
                               @sprintf("PhaseGrid(%.6f, %.6f, %d)", b.a, b.b, b.N))
    end

    # The self-consistent loop from the 33 starts, every run kept (distinct = -Inf), lowest energy first; the wells are
    # the distinct runs (mean_field_solutions' rule, at its default 1e-3) within WINDOW of the lowest.
    starts = kite_starts(spec)
    CONTEXT[] = "$(LABEL[f]) frame, the self-consistent loop"
    runs = mean_field_solutions(root_over(leaves, 0), spec, (;); starts, distinct=-Inf)
    sols = MeanFieldSolution[]
    for s in runs
        all(k -> table_change(s, k) > DISTINCT, sols) && push!(sols, s)
    end
    wells = filter(s -> s.energy - sols[1].energy <= WINDOW, sols)
    @printf("  %d starts, %d runs converged, %d distinct solutions, %d wells within %.1f GHz of the lowest\n",
            length(starts), count(s -> s.converged, runs), length(sols), length(wells), WINDOW)
    for (i, s) in enumerate(wells)
        st = s.start === nothing ? "delocalized" :
             join((@sprintf("%s=%+.4f", m.name, get(s.start, m.name, 0.0)) for m in spec.modes), " ")
        @printf("  well %d: E_prod = %.9f GHz, start %s, %d iterations, converged %s\n", i, s.energy, st,
                s.iterations, s.converged)
    end
    length(wells) == 2 || push!(failures, "$(LABEL[f]): $(length(wells)) wells, expected 2")
    gap = sols[1].energy - REF[1]
    if haskey(GAP_PAPER, f)
        @printf("  product-state gap E_prod - REF[1] = %.6f GHz (paper %.2f, reference calculation %.6f)\n", gap,
                GAP_PAPER[f], GAP_REFCALC[f])
    else
        @printf("  product-state gap E_prod - REF[1] = %.6f GHz\n", gap)
    end
    WELLS[f] = [(energy=s.energy, iterations=s.iterations, converged=s.converged) for s in wells]

    # The wells' states at the grid edges: the lowest of each well's KEEP states, on each grid, whose edge support (the
    # measure of the PhaseGrid edge warning) is above 1e-6; "none" when all KEEP are below.
    for leaf in leaves, (i, s) in enumerate(wells)
        g = only(leaf.children).second
        g isa PhaseGrid || continue
        sv = s.children[leaf].solved
        e = _edge_support(sv.vectors, sv.dims, 1, g.order ÷ 2)
        k = findfirst(>(1e-6), e)
        @printf("  edge support, %s, well %d: largest over states 1..%d %.1e; first state above 1e-6: %s\n",
                only(modes(leaf)), i, KEEP, maximum(e), k === nothing ? "none" : string(k))
    end

    # The figure: per m, the union of the wells' m lowest mean-field states per leaf, level by level, no bare ladder.
    CONTEXT[] = "$(LABEL[f]) frame, bare solves"
    bare = IdDict(leaf => bare_child(root_over(leaves, 0), terms, (;), leaf) for leaf in leaves)
    well_sources(m) = leaf -> [BasisSource("well $i", s.children[leaf].solved.vectors[:, 1:m],
                                           s.children[leaf].solved.energies[1:m]) for (i, s) in enumerate(wells)]
    ERR[f] = Dict{Int,Float64}()
    LEVELS[f] = Dict{Int,Vector{Float64}}()
    for m in MS
        t = time()
        CONTEXT[] = "$(LABEL[f]) frame, two wells, m = $m"
        aug = augmented_children(root_over(leaves, m^3), spec, leaves, bare, well_sources(m), m, SHORT, "two wells")
        E, res, N = solve_root(spec, leaves, aug)
        ERR[f][m] = error_of(E)
        LEVELS[f][m] = E
        res <= RES_TOL || push!(failures, @sprintf("%s m = %d: residual %.1e > %.0e", LABEL[f], m, res, RES_TOL))
        @printf("    m = %2d, %5d states: error %s (reference %s, ratio %.4f), largest level difference from the reference %.1e, residual %.1e  [%.1f s]\n",
                m, N, fmt(ERR[f][m]), fmt(REF_ERR[f][m]), ERR[f][m] / REF_ERR[f][m],
                maximum(abs.(E - REF_LEVELS[f][m])), res, time() - t)
        println("      levels ", join((@sprintf("%.9f", x) for x in E), " "))
        flush(stdout)                            # so that a warning on stderr lands beside its m in a merged log
    end

    # Appendix C.4, optimum frame only: one well (the lowest solution's states), and the bare ladder alone.
    if f === :optimum
        for (v, sources) in ((:one_well, m -> leaf -> [BasisSource("well 1", wells[1].children[leaf].solved.vectors[:, 1:m],
                                                                    wells[1].children[leaf].solved.energies[1:m])]),
                             (:bare, m -> leaf -> [BasisSource("bare", bare[leaf].vectors[:, 1:m], bare[leaf].energies[1:m])]))
            for m in MS
                t = time()
                CONTEXT[] = "$(LABEL[f]) frame, $v, m = $m"
                aug = augmented_children(root_over(leaves, m^3), spec, leaves, bare, sources(m), m, SHORT, string(v))
                E, res, N = solve_root(spec, leaves, aug)
                VARIANTS[v][m] = error_of(E)
                res <= RES_TOL || push!(failures, @sprintf("optimum %s m = %d: residual %.1e > %.0e", v, m, res, RES_TOL))
                @printf("    %-8s m = %2d, %5d states: error %s, residual %.1e  [%.1f s]\n", v, m, N, fmt(VARIANTS[v][m]),
                        res, time() - t)
                flush(stdout)
            end
        end
    end
    WALL[f] = time() - t0
    @printf("  wall time %.1f s; load average now %s\n", WALL[f], loadavg())
    flush(stdout)
end

# --- Comparison -----------------------------------------------------------------------------------------------------------
println("\n=== Figure 8(b): max(levels[1:4] - REF), GHz, this run / reference calculation (ratio)")
@printf("  %-13s %s\n", "frame", join((@sprintf("%-34s", "m = $m") for m in MS)))
for f in FRAMES
    @printf("  %-13s %s\n", LABEL[f],
            join((@sprintf("%-34s", "$(fmt(ERR[f][m])) / $(fmt(REF_ERR[f][m])) ($(@sprintf("%.4f", ERR[f][m] / REF_ERR[f][m])))")
                  for m in MS)))
end
for f in FRAMES, m in MS
    r = ERR[f][m] / REF_ERR[f][m]
    1 / 2 <= r <= 2 || push!(failures, @sprintf("%s m = %d: error %.4e is not within a factor of two of %.4e",
                                                LABEL[f], m, ERR[f][m], REF_ERR[f][m]))
end

println("\nordering of the frames by error, smallest first")
for m in MS
    mine = [LABEL[f] for f in FRAMES[sortperm([ERR[f][m] for f in FRAMES])]]
    ref = [LABEL[f] for f in FRAMES[sortperm([REF_ERR[f][m] for f in FRAMES])]]
    println("  m = $m: this run ", join(mine, " < "), "; reference ", join(ref, " < "), mine == ref ? "" : "  DIFFERENT")
    mine == ref || push!(failures, "m = $m: the ordering of the frames differs from the reference's")
end

println("\nlargest |level - reference level| over the eight levels, GHz")
for f in FRAMES
    @printf("  %-13s %s\n", LABEL[f], join(("m = $m: " * @sprintf("%.1e", maximum(abs.(LEVELS[f][m] - REF_LEVELS[f][m])))
                                            for m in MS), "   "))
end

println("\nAppendix C.4, optimum frame: max(levels[1:4] - REF), GHz")
println("  this grid (extent_scale 1.5) / the owner's log on another grid (extent_scale 1.0), for reference only")
for (k, (label, mine)) in enumerate((("two wells", ERR[:optimum]), ("one well", VARIANTS[:one_well]), ("bare", VARIANTS[:bare])))
    other = (C4_OTHER_GRID.two_wells, C4_OTHER_GRID.one_well, C4_OTHER_GRID.bare)[k]
    @printf("  %-10s %s\n", label, join((@sprintf("m = %2d: %s / %.1e   ", m, fmt(mine[m]), other[i]) for (i, m) in enumerate(MS))))
end

isempty(SHORT) || println("\nunions short of m (rebuilt with allow_fewer = true): ",
                          join(("$(s.label) $(s.mode) at m = $(s.m): $(s.states) states" for s in SHORT), "; "))

println("\nknown warnings dropped: fix_phases! ", sum(length, values(LOGGER.phases); init=0), " (by step: ",
        join(("$k: $(length(v)), columns $(join(sort!(unique(v)), " "))" for (k, v) in sort!(collect(LOGGER.phases))), "; "),
        ")")
println("PhaseGrid edge ",
        sum(length, values(LOGGER.edges); init=0),
        isempty(LOGGER.edges) ? "" :
        " (lowest kept state named, per mode: " *
        join(("$k $(minimum(v)) in $(length(v)) messages" for (k, v) in sort!(collect(LOGGER.edges))), ", ") * ")")
@printf("largest edge support of any state offered to a union: %.1e (the edge warning's threshold is 1e-6)\n", EDGE_USED[])
@printf("wall time per frame: %s s; load average at end: %s\n",
        join((@sprintf("%s %.1f", LABEL[f], WALL[f]) for f in FRAMES), ", "), loadavg())

# --- Optional: the reference solve ----------------------------------------------------------------------------------------
# The published frame on the reference calculation's own bases (ChargeBasis(24) and 160-point grids, the owner's
# kite_direct.py), on a plain tree: Dense leaves keeping K states each, the root by LOBPCG for the four lowest levels.
# Its convergence is shown by the change on raising K; it checks nothing.
function reference_solve()
    CONTEXT[] = "the reference solve"
    println("\n=== reference solve (KITE_MEANFIELD_REFSOLVE=1): published frame, suggest_bases(spec; charge_cutoff=24, max_points=160, extent_scale=2.0)")
    spec = load_spec(joinpath(FIXTURES, "published.json"))
    bases = suggest_bases(spec; charge_cutoff=24, max_points=160, extent_scale=2.0, when_capped=:allow)
    for (m, b) in bases
        println("  ", m, ": ", b isa ChargeBasis ? "ChargeBasis($(b.ncut))" : @sprintf("PhaseGrid(%.6f, %.6f, %d)", b.a, b.b, b.N))
    end
    prev = nothing
    for K in (40, 48, 56)
        t = time()
        # the compact leaf has 49 states and keeps at most all of them
        keeps = [b isa ChargeBasis ? min(K, 2b.ncut + 1) : K for (_, b) in bases]
        leaves = [Node([m => b]; solver=Dense(), compute=k, keep=lowest(k)) for ((m, b), k) in zip(bases, keeps)]
        sys = materialize(Node(leaves; solver=LOBPCG(tol=1e-9), compute=4, keep=lowest(4)), spec, (;))
        E = energies(sys)
        H = operator(sys)
        V = sys.root.vectors
        res = maximum(norm(H * V[:, j] - E[j] * V[:, j]) for j in 1:4)
        @printf("  K = %d (%d states): %s   residual %.1e  [%.0f s]\n", K, prod(keeps), join((@sprintf("%.7f", x) for x in E), " "),
                res, time() - t)
        @printf("           - REF:      %s", join((@sprintf("%+.1e", x) for x in E - REF), "   "))
        prev === nothing ? println() : @printf("   change from the previous K: %.1e\n", maximum(abs.(E - prev)))
        prev = E
    end
    @printf("  REF:                 %s\n", join((@sprintf("%.7f", x) for x in REF), " "))
end
REFSOLVE && reference_solve()

if !isempty(failures)
    println(stderr, "\nFAILED:\n  ", join(failures, "\n  "))
    exit(1)
end
println("\nall checks passed")
