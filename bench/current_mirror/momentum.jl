# Symmetry labels of the exact current-mirror levels (flat solve, theta coordinates, charge basis): the ring translation
# T (node i -> i+1, eigenvalue e^{i pi m / N}, momentum index m mod 2N), the half turn T^N (swaps the two nodes of every
# rung) and charge conjugation C (n -> -n). theta_i = phi_i - phi_2N carries the node charge n_i (i < 2N); node 2N's
# charge is -sum(n_i) in the zero total-charge sector. T acts on a charge state by rebuilding the node charges, shifting
# them cyclically and dropping node 2N; a state shifted outside the cutoff is lost (its weight is reported).
# Within each degenerate cluster (levels within CM_DEGTOL), T is diagonalized and C, T^N evaluated on its eigenvectors.
#
#   CM_N=2 julia --project=bench -t 8 --gcthreads=1 bench/current_mirror/momentum.jl

using KronFreeCircuits, LinearAlgebra, Printf
import JSON3

const N = parse(Int, get(ENV, "CM_N", "2"))
const NC = parse(Int, get(ENV, "CM_NC", "10"))
const NLEV = parse(Int, get(ENV, "CM_NLEV", "12"))
const DEGTOL = parse(Float64, get(ENV, "CM_DEGTOL", "1e-5"))
const SPEC = load_spec(joinpath(@__DIR__, "specs", "cm_N$N.json"))
const θ = merge((E_J=19.0, varphi_ext=0.0), NamedTuple(Symbol("n_g$i") => 0.0 for i in 1:2N-1))
const M = 2N - 1
names = [Symbol(m.name) for m in SPEC.modes]
@assert names == [Symbol("theta$i") for i in 1:M] names

root = Node([n => ChargeBasis(NC) for n in names]; solver=N <= 2 ? Dense() : LOBPCG(tol=1e-10), compute=NLEV, keep=lowest(NLEV))
t0 = time(); sys = materialize(root, SPEC, θ); tsolve = time() - t0
E = energies(sys); V = sys.root.vectors
D = 2NC + 1
@assert size(V, 1) == D^M
println("current mirror N = $N, flat solve ($(size(V, 1)) states, $(round(tsolve; digits=1)) s); levels from E0: ",
        join((@sprintf("%.6f", e - E[1]) for e in E), " "))

# the charge vector of product index (axis 1 fastest), charges -NC..NC in index order 1..D
charges(idx) = [((idx - 1) ÷ D^(k - 1)) % D - NC for k in 1:M]
index(n) = 1 + sum((n[k] + NC) * D^(k - 1) for k in 1:M)
# the image under a node map: node charges (n_1 .. n_2N), mapped, back to theta charges
function image(n, f)
    nodes = [n; -sum(n)]
    m = f(nodes)
    q = m[1:M]
    return all(abs.(q) .<= NC) ? index(q) : 0
end
shift(nodes) = circshift(nodes, 1)          # T: the charge on node i moves to node i+1
conj_(nodes) = -nodes                        # C
const IMG_T = [image(charges(i), shift) for i in 1:D^M]
const IMG_C = [image(charges(i), conj_) for i in 1:D^M]
apply(img, v) = (w = zeros(eltype(v), length(v)); for i in eachindex(v); img[i] > 0 && (w[img[i]] += v[i]); end; w)
lost(img, v) = sum(abs2(v[i]) for i in eachindex(v) if img[i] == 0; init=0.0)

clusters = Vector{Vector{Int}}()
for k in eachindex(E)
    if !isempty(clusters) && abs(E[k] - E[clusters[end][end]]) < DEGTOL
        push!(clusters[end], k)
    else
        push!(clusters, [k])
    end
end
println("  cluster  E - E0 (GHz)   m (T = e^{i π m/N})   T^N   C    |T|-1      lost weight")
rows = []
for c in clusters
    c[end] == length(E) && length(c) > 0 && E[end] - E[c[1]] < DEGTOL && c != clusters[1] && length(clusters) > 1 &&
        println("  (the last cluster may be cut by NLEV)")
    Vc = ComplexF64.(V[:, c])
    TV = hcat((apply(IMG_T, Vc[:, j]) for j in eachindex(c))...)
    Tc = Vc' * TV                                                  # T within the cluster
    F = eigen(Tc)
    W = Vc * F.vectors
    for j in eachindex(c)
        λ = F.values[j]
        m = mod(round(Int, angle(λ) / (π / N)), 2N)
        w = W[:, j] / norm(W[:, j])
        tN = real(dot(w, foldl((x, _) -> apply(IMG_T, x), 1:N; init=w)))
        cc = real(dot(w, apply(IMG_C, w)))
        lw = maximum(lost(IMG_T, Vc[:, i]) for i in eachindex(c))
        @printf("  %2d       %10.6f     m = %2d (angle %+.4f π)   %+.3f  %+.3f  %.1e   %.1e\n", c[1], E[c[1]] - E[1], m,
                angle(λ) / π, tN, cc, abs(λ) - 1, lw)
        push!(rows, Dict("level" => c[1], "energy" => E[c[1]] - E[1], "m" => m, "TN" => tN, "C" => cc, "absT" => abs(λ)))
    end
end
mkpath(joinpath(@__DIR__, "results"))
open(io -> JSON3.pretty(io, JSON3.write(Dict("N" => N, "nc" => NC, "energies" => E, "labels" => rows))),
     joinpath(@__DIR__, "results", "momentum_N$N.json"), "w")
