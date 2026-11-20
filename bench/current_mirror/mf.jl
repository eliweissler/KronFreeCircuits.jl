# The current mirror with mean-field rung leaves (EXPERIMENT: needs the uncommitted src/meanfield.jl and src/augment.jl,
# which are never committed). Each rung leaf's basis is the union of its mean-field states in the circuit's wells (the
# self-consistent loop, every other rung replaced by its expectation values in the junction terms); the leaf is
# presented through its bare operator projected on that union (`augment`), so the root solves the exact Hamiltonian in
# a smaller basis. Compared with the exact (flat) levels and with bare rung leaves of the same size.
#
#   CM_N=3 julia --project=bench -t 8 --gcthreads=1 bench/current_mirror/mf.jl
#
# Starts of the loop: the classical minima phi_i = pi w i / N (theta_i = phi_i - phi_2N, mod 2 pi), winding w = 0 (the 0
# well), +-1 (vortex, antivortex: every rung's exciton phase at pi, the pi well), +-2, and the delocalized start.
# Knobs: CM_N, CM_NC (10), CM_KEEP (mean-field states per well and leaf, 40), CM_MS (m values, "10,20,30,40"),
#        CM_WINDOW (GHz above the lowest product energy for a solution to count as a well, 10), CM_NLEV (12),
#        CM_VARIANTS (comma list of mf, mf+bare, bare; default mf+bare,bare), CM_COORDS (junction, rung, theta).

using KronFreeCircuits, LinearAlgebra, Printf
import JSON3

const N = parse(Int, get(ENV, "CM_N", "3"))
const NC = parse(Int, get(ENV, "CM_NC", "10"))
const KEEP = parse(Int, get(ENV, "CM_KEEP", "40"))
const MS = parse.(Int, split(get(ENV, "CM_MS", "10,20,30,40"), ","))
const WINDOW = parse(Float64, get(ENV, "CM_WINDOW", "10"))
const NLEV = parse(Int, get(ENV, "CM_NLEV", "12"))
# rung (a_j, b_j), theta (phi_i - phi_2N): leaves are rungs; junction (psi_k): leaves are plaquettes (psi_j, psi_(j+N))
const COORDS = get(ENV, "CM_COORDS", "junction")
const SPEC = load_spec(joinpath(@__DIR__, "specs", COORDS == "theta" ? "cm_N$N.json" : "cm_N$(N)_$COORDS.json"))
const TERMS = circuit_terms(SPEC)
const θ = merge((E_J=19.0, varphi_ext=0.0), NamedTuple(Symbol("n_g$i") => 0.0 for i in 1:2N-1))

allmodes() = [Symbol(m.name) for m in SPEC.modes]
mode(name::Symbol) = name => ChargeBasis(NC)
mode(i::Int) = mode(allmodes()[i])
# the modes of leaf j: rung j (theta, rung) or the plaquette between rungs j and j+1 (junction); leaf N has one mode
rungmodes(j) = COORDS == "rung" ? (j < N ? [mode(Symbol("a$j")), mode(Symbol("b$j"))] : [mode(Symbol("a$N"))]) :
               (j < N ? [mode(j), mode(j + N)] : [mode(N)])
leafdim(j) = (2NC + 1)^length(rungmodes(j))
leaves = [Node(rungmodes(j); solver=Dense(), compute=min(KEEP, leafdim(j)), keep=lowest(min(KEEP, leafdim(j))),
               arithmetic=:complex) for j in 1:N]
root_over(n) = Node(leaves; solver=n <= 1500 ? Dense() : LOBPCG(tol=1e-10), compute=NLEV, keep=lowest(NLEV))

wrap(x) = mod(x + π, 2π) - π
# the classical minimum phi_i = pi w i / N in the spec's coordinates: theta_i = phi_i - phi_2N, or a_j = phi_j - phi_(j+N), b_j = phi_(j+N) - phi_2N
function start(w)
    φ = [π * w * i / N for i in 1:2N]
    COORDS == "junction" && return Dict(Symbol("psi$k") => wrap(φ[k+1] - φ[k]) for k in 1:2N-1)
    COORDS == "rung" || return Dict(Symbol("theta$i") => wrap(φ[i] - φ[2N]) for i in 1:2N-1)
    return merge(Dict(Symbol("a$j") => wrap(φ[j] - φ[j+N]) for j in 1:N), Dict(Symbol("b$j") => wrap(φ[j+N] - φ[2N]) for j in 1:N-1))
end
starts = Any[[start(w) for w in (0, 1, -1, 2, -2)];
             nothing]
println("current mirror N = $N, mean-field rung leaves, NC = $NC, $KEEP states per well and leaf; threads $(Threads.nthreads())")
t0 = time()
runs = mean_field_solutions(root_over(0), SPEC, θ; starts, distinct=-Inf)
change(a, b) = maximum(abs(a.children[ch].expectations[k] - b.children[ch].expectations[k])
                       for ch in keys(a.children) for k in keys(a.children[ch].expectations))
sols = MeanFieldSolution[]
for s in runs
    all(k -> change(s, k) > 1e-3, sols) && push!(sols, s)
end
wells = filter(s -> s.energy - sols[1].energy <= WINDOW, sols)
@printf("self-consistent loop: %d starts, %d converged, %d distinct, %d wells within %.1f GHz  [%.1f s]\n",
        length(starts), count(s -> s.converged, runs), length(sols), length(wells), WINDOW, time() - t0)
for (i, s) in enumerate(sols)
    st = s.start === nothing ? "delocalized" : join((@sprintf("%+.3f", s.start[m]) for m in allmodes()), " ")
    @printf("  solution %d: E_prod = %.6f GHz, start [%s], %d iterations, converged %s%s\n", i, s.energy, st,
            s.iterations, s.converged, s in wells ? "  (well)" : "")
end

exact = let f = joinpath(@__DIR__, "results", "lobpcg_flat_N$(N)_nc10.json")   # the exact levels (theta coordinates, NC 10; converged)
    isfile(f) ? Vector{Float64}(JSON3.read(read(f, String)).energies) : nothing
end
bare = IdDict(leaf => bare_child(root_over(0), TERMS, θ, leaf) for leaf in leaves)
function solve(sources, m, label)
    aug = AugmentedChild[]
    sizes = Int[]
    for leaf in leaves
        srcs = sources(leaf, m)
        U, report = try
            union_basis(srcs; m)
        catch err
            err isa ArgumentError || rethrow()
            union_basis(srcs; m, allow_fewer=true)
        end
        push!(sizes, size(U, 2))
        push!(aug, augment(root_over(0), TERMS, θ, leaf, bare[leaf], U, report))
    end
    n = prod(sizes)
    t = time()
    sys = materialize(root_over(n), SPEC, θ; augmented=aug)
    E = energies(sys)
    dt = time() - t
    err = exact === nothing || length(exact) < length(E) ? NaN : maximum(abs.((E .- E[1]) .- (exact[1:length(E)] .- exact[1])))
    e0 = exact === nothing ? NaN : E[1] - exact[1]
    @printf("  %-9s m = %3d: leaves %s, root %7d states: worst |Δ(E_k - E_0)| %.2e GHz, E0 - exact %.2e  [%.1f s]\n",
            label, m, join(sizes, "x"), n, err, e0, dt)
    println("      levels from E0: ", join((@sprintf("%.6f", e - E[1]) for e in E), " "))
    return (m=m, label=label, sizes=sizes, energies=E, worst=isnan(err) ? nothing : err, seconds=dt)
end
wellsrc(leaf, m) = [BasisSource("well $i", s.children[leaf].solved.vectors[:, 1:min(m, end)],
                                s.children[leaf].solved.energies[1:min(m, end)]) for (i, s) in enumerate(wells)]
baresrc(leaf, m) = [BasisSource("bare", bare[leaf].vectors[:, 1:min(m, end)], bare[leaf].energies[1:min(m, end)])]
exact === nothing || println("exact levels from E0: ", join((@sprintf("%.6f", e - exact[1]) for e in exact), " "))
out = []
# mean field plus the bare ladder: the wells' and the bare rung's states merged level by level (union_basis)
mixsrc(leaf, m) = [wellsrc(leaf, m); baresrc(leaf, m)]
const VARIANTS = split(get(ENV, "CM_VARIANTS", "mf+bare,bare"), ",")
for m in MS
    "mf" in VARIANTS && push!(out, solve(wellsrc, m, "mean field"))
    "mf+bare" in VARIANTS && push!(out, solve(mixsrc, m, "mf+bare"))
    "bare" in VARIANTS && push!(out, solve(baresrc, m, "bare"))
end
mkpath(joinpath(@__DIR__, "results"))
open(io -> JSON3.pretty(io, JSON3.write(Dict("N" => N, "nc" => NC, "keep" => KEEP, "wells" => [s.energy for s in wells],
                                             "solutions" => [s.energy for s in sols], "runs" => out))),
     joinpath(@__DIR__, "results", "mf_N$(N)_nc$(NC)_$COORDS.json"), "w")
println("wrote results/mf_N$(N)_nc$(NC)_$COORDS.json")
