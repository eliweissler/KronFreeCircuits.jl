# The full current-mirror circuit (arXiv:1908.04615, Eq. Ham, Tab. table:parameters) on the circuit-quantization spec
# specs/cm_N<N>.json (quantize.py): 2N - 1 periodic modes theta_i = phi_i - phi_2N in the charge basis (cutoff NC,
# 2 NC + 1 states each), zero flux and offset charges, E_J = 19 GHz. A rung j < N carries the modes theta_j and
# theta_(j+N); rung N only theta_N (node 2N is the reference).
#
#   CM_N=3 CM_TREE=rungs CM_K=40 julia --project=bench -t 8 --gcthreads=1 bench/current_mirror/cm.jl
#
# Trees (CM_TREE):
#   flat   one node over every mode (exact in the charge basis)
#   rungs  a leaf per rung (Dense, keeping CM_K levels) under one root
#   pairs  rung leaves (keeping CM_K) joined pairwise up to the root, each internal node keeping CM_K2 levels
# Knobs: CM_N, CM_NC (charge cutoff, default 10 as the paper), CM_K, CM_K2, CM_NLEV (root levels, default 12), CM_REPEAT (default 1: a second, timed materialize),
#        CM_TOL (root tolerance, default 1e-10), CM_ROOT (dense, lanczos or lobpcg; default lobpcg). Writes results/<tree>_N<N>_....json.

using KronFreeCircuits, LinearAlgebra, Printf
import JSON3

const N = parse(Int, get(ENV, "CM_N", "2"))
const NC = parse(Int, get(ENV, "CM_NC", "10"))
const TREE = get(ENV, "CM_TREE", "flat")
const K = parse(Int, get(ENV, "CM_K", "40"))
const K2 = parse(Int, get(ENV, "CM_K2", "100"))
const NLEV = parse(Int, get(ENV, "CM_NLEV", "12"))
const TOL = parse(Float64, get(ENV, "CM_TOL", "1e-10"))
const ROOT = get(ENV, "CM_ROOT", "lobpcg")   # dense, lanczos or lobpcg. Lanczos (one start vector) missed partners of
                                            # degenerate levels at N = 2 (11.2369, 11.5399 GHz); LOBPCG, a block method, did not
rootsolver() = ROOT == "dense" ? Dense() : ROOT == "lobpcg" ? LOBPCG(tol=TOL) : Lanczos(tol=TOL)
# the rungs' order around the ring for the pairwise tree (consecutive pairs from the front), e.g. CM_ORDER=2,3,1
const ORDER = haskey(ENV, "CM_ORDER") ? parse.(Int, split(ENV["CM_ORDER"], ",")) : collect(1:N)
# rung (a_j, b_j), theta (phi_i - phi_2N): leaves are rungs; junction (psi_k): leaves are plaquettes (psi_j, psi_(j+N));
# node (spec_nodes.py, all 2N node phases, the island's global rotor kept): every rung the same leaf (phi_j, phi_(j+N))
const COORDS = get(ENV, "CM_COORDS", "junction")
const SPEC = load_spec(joinpath(@__DIR__, "specs", COORDS == "theta" ? "cm_N$N.json" : "cm_N$(N)_$COORDS.json"))
# E_J = 19 GHz, zero flux and offsets: only the spec's own parameters (the node variant has no n_g parameters)
const θ = (; (p => (p === :E_J ? 19.0 : 0.0) for p in SPEC.parameters)...)


allmodes() = [Symbol(m.name) for m in SPEC.modes]
mode(name::Symbol) = name => ChargeBasis(NC)
mode(i::Int) = mode(allmodes()[i])
# the modes of leaf j: rung j (theta, rung) or the plaquette between rungs j and j+1 (junction); leaf N has one mode
rungmodes(j) = COORDS == "node" ? [mode(Symbol("phi$j")), mode(Symbol("phi$(j + N)"))] : COORDS == "rung" ? (j < N ? [mode(Symbol("a$j")), mode(Symbol("b$j"))] : [mode(Symbol("a$N"))]) :
               (j < N ? [mode(j), mode(j + N)] : [mode(N)])
root(children) = Node(children; solver=rootsolver(), compute=NLEV, keep=lowest(NLEV))
rung(j) = Node(rungmodes(j); solver=Dense(), compute=K, keep=lowest(K))
function tree()
    TREE == "flat" && return root([mode(i) for i in eachindex(SPEC.modes)])
    TREE == "rungs" && return root([rung(j) for j in 1:N])
    TREE == "pairs" || error("CM_TREE: flat, rungs or pairs")
    ns = [rung(j) for j in ORDER]
    # pair up to two children of the root, then the root over them
    while length(ns) > 2
        ns = [i + 1 <= length(ns) ? Node([ns[i], ns[i+1]]; solver=rootsolver(), compute=K2, keep=lowest(K2)) : ns[i]
              for i in 1:2:length(ns)]
    end
    return root(ns)
end

println("current mirror N = $N, tree $TREE, root $ROOT, NC = $NC", TREE == "flat" ? "" : ", K = $K", TREE == "pairs" ? ", K2 = $K2, order $(join(ORDER, ","))" : "",
        "; threads $(Threads.nthreads())")
t = tree()
t0 = time(); sys = materialize(t, SPEC, θ); t_first = time() - t0
const REPEAT = get(ENV, "CM_REPEAT", "1") == "1"    # a second, compiled materialize (CM_REPEAT=0 for long runs)
t_mat = REPEAT ? (t0 = time(); sys = materialize(t, SPEC, θ); time() - t0) : NaN
E = energies(sys)
@printf("materialize: first %.2f s (compile included), second %.2f s\n", t_first, t_mat)
@printf("levels from E0 (GHz): %s\n", join((@sprintf("%.6f", e - E[1]) for e in E), " "))
@printf("E0 = %.9f GHz\n", E[1])
mkpath(joinpath(@__DIR__, "results"))
name = ROOT * "_" * (COORDS == "theta" ? "" : COORDS * "_") * (TREE == "flat" ? "flat_N$(N)_nc$NC" : TREE == "rungs" ? "rungs_N$(N)_nc$(NC)_K$K" : "pairs_N$(N)_nc$(NC)_K$(K)_K2$(K2)_order$(join(ORDER))")
open(io -> JSON3.pretty(io, JSON3.write(Dict("N" => N, "nc" => NC, "tree" => TREE, "K" => K, "K2" => K2, "tol" => TOL, "root" => ROOT, "order" => ORDER,
                                             "energies" => E, "seconds_materialize" => isnan(t_mat) ? nothing : t_mat, "seconds_first" => t_first,
                                             "threads" => Threads.nthreads()))),
     joinpath(@__DIR__, "results", "$name.json"), "w")
println("wrote results/$name.json")
