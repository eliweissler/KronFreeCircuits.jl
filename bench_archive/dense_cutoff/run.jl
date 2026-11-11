# Benchmark for `Auto()`'s `dense_cutoff`: `Dense()` against the iterative solver `Auto()` would pick,
# per node, by product dimension.  CPU only.
#
#   julia --project=. bench/dense_cutoff/run.jl            # default BLAS threads
#   DC_BLAS_THREADS=1 julia --project=. bench/dense_cutoff/run.jl
#
# Knobs: DC_BLAS_THREADS (default: Julia's default), DC_SMOKE=1 (tiny sweep), DC_PARTS (comma list of
# leaf,branch; default both), DC_MAX_DENSE (largest product dimension at which Dense is timed;
# default 4000: above it Dense is already 15 times slower than the iterative solver and grows as n^3),
# DC_OUT (output stem; default bench/dense_cutoff/raw_<blas threads>t).
#
# Method.  `materialize` is timed per node (GC run first).  After a warm-up call at a small size for each
# solver, the first call of each configuration is a sample; faster configurations take four more, slower ones
# fewer, and the minimum is reported.  Compilation is excluded by the warm-up.
# Leaves are materialized alone, with `compute` = 6 and 20 for the iterative solver (Dense does not depend on it;
# it is timed with compute = 20).  Branches are two-leaf trees; the product dimension is k² for k kept child states.
# The leaves of a branch use Dense and their time is measured on its own and subtracted, leaving the parent solve.
# Every iterative result is compared with Dense's energies (`dE`, absolute, over the computed levels) where Dense ran.

using LinearAlgebra, Printf, Dates, Logging, Pkg
using JSON3
using KronFreeCircuits
using KronFreeCircuits: ShiftInvert, LOBPCG

const SMOKE = get(ENV, "DC_SMOKE", "0") == "1"
haskey(ENV, "DC_BLAS_THREADS") && BLAS.set_num_threads(parse(Int, ENV["DC_BLAS_THREADS"]))
const PARTS = split(get(ENV, "DC_PARTS", "leaf,branch"), ",")
const MAX_DENSE = parse(Int, get(ENV, "DC_MAX_DENSE", "4000"))
const FIX = joinpath(@__DIR__, "..", "..", "test", "fixtures")
const OUT = get(ENV, "DC_OUT", joinpath(@__DIR__, "raw_$(BLAS.get_num_threads())t" * (SMOKE ? "_smoke" : "")))

quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())   # small PhaseGrids warn about edge support

# ---- timing ----
const FAILURES = String[]
function sample(f)
    GC.gc()
    t0 = time_ns()
    v = f()
    return (time_ns() - t0) / 1e9, v
end
# A solver error (a failed residual check or no convergence) makes a failed sample: it is counted and its message kept,
# and the minimum is taken over the successful samples.  `bench` returns (minimum seconds or NaN, a successful result or
# `nothing`, samples taken, failed samples).
function bench(f)
    ts, v, nfail = Float64[], nothing, 0
    function one()
        try
            t, r = sample(f)
            push!(ts, t); v === nothing && (v = r)
            return t
        catch err
            err isa InterruptException && rethrow()
            msg = first(sprint(showerror, err), 200)
            println("  FAILED sample: ", msg)
            push!(FAILURES, msg); nfail += 1
            return 0.0
        end
    end
    t1 = one()
    extra = t1 < 1 ? 4 : t1 < 10 ? 2 : t1 < 40 ? 1 : 0
    for _ in 1:extra
        one()
    end
    return isempty(ts) ? NaN : minimum(ts), v, length(ts) + nfail, nfail
end

# ---- the problems ----
fluxonium = (load_spec(joinpath(FIX, "fluxonium.json")), (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π))
zeropi = (load_spec(joinpath(FIX, "zero_pi.json")), (E_J=10.0, E_L=0.04, E_CJ=20.0, E_CS=0.5, varphi_ext=2π * 0.23, n_g=0.1))
transmon = (load_spec(joinpath(FIX, "transmon.json")), (E_J=20.0, E_C=1.0, n_g=0.3))
two_transmons = (load_spec(joinpath(FIX, "two_transmons.json")),
                 (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.0, g=0.1))
two_zeropi_spec = load_spec(joinpath(FIX, "two_zero_pi.json"))
let p = (E_J=10.0, E_L=0.04, E_CJ=20.0, E_CS=0.5, varphi_ext=2π * 0.23, n_g=0.1), g = 0.05
    global two_zeropi = (two_zeropi_spec,
                         (E_J1=p.E_J, E_L1=p.E_L, E_CJ1=p.E_CJ, E_CS1=p.E_CS, varphi_ext1=p.varphi_ext, n_g1=p.n_g,
                          E_J2=p.E_J, E_L2=p.E_L, E_CJ2=p.E_CJ, E_CS2=p.E_CS, varphi_ext2=p.varphi_ext, n_g2=p.n_g, g=g))
end

# Leaf shapes: name => (children for a target product dimension, actual dimension).
leaf_children(::Val{:fluxonium}, D) = ([:varphi_1 => PhaseGrid(-6π, 6π, D; order=8)], D)
function leaf_children(::Val{:zeropi}, D)
    c = max(3, round(Int, sqrt(D / 10)))
    nc = (c - 1) ÷ 2
    N = round(Int, D / (2nc + 1))
    return ([:theta => ChargeBasis(nc), :phi => PhaseGrid(-6π, 6π, N; order=8)], (2nc + 1) * N)
end
function leaf_children(::Val{:transmon}, D)
    nc = (D - 1) ÷ 2
    return ([:varphi_1 => ChargeBasis(nc)], 2nc + 1)
end
const LEAF_SHAPES = [(:fluxonium, fluxonium, "1-mode extended PhaseGrid (fluxonium)"),
                     (:zeropi, zeropi, "2-mode ChargeBasis x PhaseGrid (zero-pi)"),
                     (:transmon, transmon, "1-mode ChargeBasis (transmon)")]
const LEAF_DIMS = SMOKE ? [200, 500] : [100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
const COMPUTES = [6, 20]

solvers_leaf = Dict("Dense" => () -> Dense(), "ShiftInvert" => () -> ShiftInvert())
branch_solver = Dict("Dense" => () -> Dense(), "LOBPCG" => () -> LOBPCG(preconditioner=:ritz, seed=:product))

env = Dict{String,Any}(
    "date" => string(now()), "julia" => string(VERSION), "blas_threads" => BLAS.get_num_threads(),
    "julia_threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model, "cpu_threads" => Sys.CPU_THREADS,
    "blas" => string(BLAS.get_config()),
    "git_commit" => strip(read(setenv(`git rev-parse HEAD`; dir=@__DIR__), String)),
    "packages" => Dict(string(d.name) => string(d.version) for (_, d) in Pkg.dependencies() if d.is_direct_dep && d.version !== nothing))
println("environment: ", JSON3.write(env))

record = Dict{String,Any}("environment" => env, "leaf" => Any[], "branch" => Any[])
t_start = time()

# ---- leaves ----
function run_leaves()
    for (name, (spec, θ), label) in LEAF_SHAPES
        # warm-up on a small size, every solver and compute
        ch, _ = leaf_children(Val(name), 200)
        for c in COMPUTES, (sn, mk) in solvers_leaf
            try; quiet(() -> materialize(Node(ch; solver=mk(), compute=c, keep=lowest(c)), spec, θ)); catch; end
        end
        for D in LEAF_DIMS
            ch, dim = leaf_children(Val(name), D)
            mat(s, c) = quiet(() -> materialize(Node(ch; solver=s, compute=c, keep=lowest(c)), spec, θ))
            rd = Dict{String,Any}("shape" => string(name), "label" => label, "target" => D, "dim" => dim)
            Ed = nothing
            if dim <= MAX_DENSE
                t, sys, n, nf = bench(() -> mat(Dense(), 20))
                rd["Dense"] = t; Ed = energies(sys)
            end
            for c in COMPUTES
                t, sys, n, nf = bench(() -> mat(ShiftInvert(), c))
                rd["ShiftInvert_$c"] = t; rd["samples_ShiftInvert_$c"] = n; rd["failed_ShiftInvert_$c"] = nf
                sys === nothing && continue
                Ed === nothing || (rd["dE_ShiftInvert_$c"] = maximum(abs.(energies(sys) .- Ed[1:c])))
            end
            push!(record["leaf"], rd)
            @printf("leaf %-9s dim %5d  Dense %s  SI6 %.3f  SI20 %.3f  [%.0f s elapsed]\n", name, dim,
                    haskey(rd, "Dense") ? @sprintf("%.3f", rd["Dense"]) : "  -  ", rd["ShiftInvert_6"], rd["ShiftInvert_20"], time() - t_start)
            flush(stdout)
        end
    end
end

# ---- branches ----
# two transmons: leaves ChargeBasis(60) (121 states), k kept each; two zero-pis: leaves ChargeBasis(4) x PhaseGrid(61) (549 states), k kept each.
function branch_tree(kind, k, solver, c)
    leafsolver = Dense()
    if kind === :transmons
        l1 = Node([:varphi_1 => ChargeBasis(60)]; solver=leafsolver, keep=lowest(k))
        l2 = Node([:varphi_2 => ChargeBasis(60)]; solver=leafsolver, keep=lowest(k))
    else
        mk(i) = Node([Symbol("theta$i") => ChargeBasis(4), Symbol("phi$i") => PhaseGrid(-6π, 6π, 61; order=6)];
                     solver=leafsolver, keep=lowest(k))
        l1, l2 = mk(1), mk(2)
    end
    return Node([l1, l2]; solver=solver, compute=c, keep=lowest(c)), l1, l2
end
const BRANCH_KS = SMOKE ? [16, 24] : [10, 14, 17, 23, 32, 39, 45, 55, 63, 77, 90]   # k² = 100 ... 8100
function run_branches()
    for (kind, (spec, θ), label) in [(:transmons, two_transmons, "two transmons"), (:zeropis, two_zeropi, "two zero-pis")]
        # warm-up
        for c in COMPUTES, (sn, mk) in branch_solver
            try; quiet(() -> materialize(first(branch_tree(kind, 16, mk(), c)), spec, θ)); catch; end
        end
        for k in BRANCH_KS
            kind === :zeropis && k > 90 && continue
            dim = k^2
            mat(s, c) = (tree = first(branch_tree(kind, k, s, c)); quiet(() -> materialize(tree, spec, θ)))
            # leaves alone (twice per tree), measured as the root being a leaf with no parent
            # (one leaf alone, with the single-leaf spec, times two: the same leaf problem without the coupling's routing)
            lspec, lθ, lch = kind === :transmons ? (transmon..., [:varphi_1 => ChargeBasis(60)]) :
                             (zeropi..., [:theta => ChargeBasis(4), :phi => PhaseGrid(-6π, 6π, 61; order=6)])
            leaf = Node(lch; solver=Dense(), keep=lowest(k))
            lt, _, _, _ = bench(() -> quiet(() -> materialize(leaf, lspec, lθ)))
            lt *= 2
            rd = Dict{String,Any}("shape" => string(kind), "label" => label, "k" => k, "dim" => dim, "leaves_seconds" => lt)
            Ed = nothing
            if dim <= MAX_DENSE
                t, sys, _, _ = bench(() -> mat(Dense(), 20))
                rd["Dense_total"] = t; Ed = energies(sys)
            end
            for c in COMPUTES
                t, sys, n, nf = bench(() -> mat(LOBPCG(preconditioner=:ritz, seed=:product), c))
                rd["LOBPCG_total_$c"] = t; rd["samples_LOBPCG_$c"] = n; rd["failed_LOBPCG_$c"] = nf
                sys === nothing && continue
                Ed === nothing || (rd["dE_LOBPCG_$c"] = maximum(abs.(energies(sys) .- Ed[1:c])))
            end
            push!(record["branch"], rd)
            @printf("branch %-9s k %3d dim %5d  leaves %.3f  Dense %s  LOBPCG6 %.3f  LOBPCG20 %.3f  [%.0f s elapsed]\n", kind, k, dim, lt,
                    haskey(rd, "Dense_total") ? @sprintf("%.3f", rd["Dense_total"]) : "  -  ", rd["LOBPCG_total_6"], rd["LOBPCG_total_20"], time() - t_start)
            flush(stdout)
        end
    end
end

"leaf" in PARTS && run_leaves()
"branch" in PARTS && run_branches()
record["wall_seconds"] = time() - t_start
record["failures"] = FAILURES
for rows in (record["leaf"], record["branch"]), rd in rows, (k, v) in rd   # JSON has no NaN: a configuration whose every sample failed is null
    v isa Float64 && isnan(v) && (rd[k] = nothing)
end
open(OUT * ".json", "w") do io
    JSON3.pretty(io, record)
end
@printf("done in %.0f s; wrote %s.json\n", record["wall_seconds"], OUT)
