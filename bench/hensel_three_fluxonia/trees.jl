# The three-fluxonium benchmark's trees, built on a circuit-quantization spec (specs/three_fluxonia_<frame>.json, written
# by quantize.py): the spec's modes q1, q2, q3, c1, c2 (junction phases) and f1..f6 (frame modes), its E_C and E_L at the
# working point for the Fock zero-point spreads and the pairing, and the four-level tree of bench_archive (2026-11-07).
# Needs `ON` (Array or CuArray) from ../common.jl.

using LinearAlgebra, Printf, KronFreeCircuits

const JNAMES = [:q1, :q2, :q3, :c1, :c2]
# the working point (Sec. V): every qubit loop at Φ_0/2, both couplers unbiased; the spec's loop phases sit on the junctions
θ_work(_=nothing) = (varphi_q1=π, varphi_q2=π, varphi_q3=π, varphi_c1=0.0, varphi_c2=0.0)
const PAPER = [0.422, 0.705, 0.816, 1.038, 1.220, 1.477, 1.771]   # GHz, Table tab:q3levels (cap 128, working cutoff)
const PAPER_STIFF_SHIFT = [140, 100, 480, 96, 180, 440, 78] .* 1e-6   # GHz, the same table's "stiff shift" column

"""
The spec of `frame` and its numbers at the working point: E_C, E_L (GHz), the zero-point spreads σ = (8 E_C/E_L)^¼/√2
from the diagonals (the Fock bases' `phi_zpf`), and the mode names.
"""
function load_frame(frame; dir=joinpath(@__DIR__, "specs"))
    spec = load_spec(joinpath(dir, "three_fluxonia_$frame.json"))
    names = [m.name for m in spec.modes]
    names[1:5] == JNAMES || error("unexpected mode order $names")
    θ = θ_work()
    EC = [real(spec.E_C[i, j](θ)) for i in 1:11, j in 1:11]
    EL = [real(spec.E_L[i, j](θ)) for i in 1:11, j in 1:11]
    σ = (8 .* diag(EC) ./ diag(EL)) .^ (1 / 4) ./ sqrt(2)
    return spec, (; EC, EL, σ, names)
end

"Coupling magnitudes at the zero-point scale (GHz): |8 E_C,ij n_i n_j| + |E_L,ij σ_i σ_j|, n = 1/(2σ)."
function couplings(B)
    n = 1 ./ (2 .* B.σ)
    return [i == j ? 0.0 : abs(8 * B.EC[i, j] * n[i] * n[j]) + abs(B.EL[i, j] * B.σ[i] * B.σ[j]) for i in 1:11, j in 1:11]
end

"Each junction's frame partner, maximizing the total junction–frame coupling over all assignments (one frame mode left)."
function best_pairing(B)
    C = couplings(B)
    best, bestp = -Inf, Int[]
    perms = Vector{Vector{Int}}()
    rec(prefix, rest) = isempty(rest) ? push!(perms, prefix) : foreach(r -> rec([prefix; r], setdiff(rest, r)), rest)
    rec(Int[], collect(1:6))
    for p in perms
        v = sum(C[J, 5+p[J]] for J in 1:5)
        v > best && (best = v; bestp = p[1:5])
    end
    return Dict(JNAMES[J] => B.names[5+bestp[J]] for J in 1:5), B.names[5+setdiff(1:6, bestp)[1]]
end

# HENSEL3_QGRID = "a,N": the qubit grid ±aπ with N points (default 5,301: spacing π/30, the narrowest range on which
# no kept leaf state reaches the edge; ±3π and ±4π warn, 2026-11-12)
const QGRID = haskey(ENV, "HENSEL3_QGRID") ?
              (g = parse.(Float64, split(ENV["HENSEL3_QGRID"], ",")); PhaseGrid(-g[1] * π, g[1] * π, Int(g[2]); order=8)) :
              PhaseGrid(-5π, 5π, 301; order=8)
# HENSEL3_CGRID = "a,N": the coupler grid ±aπ with N points (default 4,241: spacing π/30, no edge warnings; ±2π and
# ±3π warn)
const CGRID = haskey(ENV, "HENSEL3_CGRID") ?
              (g = parse.(Float64, split(ENV["HENSEL3_CGRID"], ",")); PhaseGrid(-g[1] * π, g[1] * π, Int(g[2]); order=8)) :
              PhaseGrid(-4π, 4π, 241; order=8)
nfock(m) = m in (:f1, :f2) ? 30 : 20
const TOL = parse(Float64, get(ENV, "HENSEL_TOL", "1e-6"))
lobpcg(; guard=4) = LOBPCG(; tol=TOL, maxiter=3000, guard, on=ON)
const DENSE_MAX = parse(Int, get(ENV, "HENSEL_DENSE_MAX", "5000"))

"""
Four-level tree (bench_archive, 2026-11-07): single-mode leaves (qubits on QGRID, couplers on CGRID,
frame modes on Fock bases of 30 (f1, f2) or 20 levels) keeping k1j (junctions) or k1f (frame modes); pairs P(J, partner)
keeping k2; left wing L = [P(q1), P(c1)] keeping kM, joined with P(q2) into LC keeping kM2; right wing R = [P(q3), P(c2)]
keeping kM; root over [LC, R, the loose frame mode as a raw Fock axis of kf levels]. Nodes up to DENSE_MAX states are
Dense, larger ones LOBPCG.
"""
function fourlevel_root(B, partner, loose; k1j=40, k1f=20, k2=60, kM=240, kM2=240, kf=4, compute=8, smoke=false)
    idx = Dict(m => i for (i, m) in enumerate(B.names))
    qg = smoke ? PhaseGrid(-3π, 3π, 121; order=8) : QGRID
    cg = smoke ? PhaseGrid(-2π, 2π, 81; order=8) : CGRID
    basis(m) = m in (:q1, :q2, :q3) ? qg : m in (:c1, :c2) ? cg : FockBasis(smoke ? 10 : nfock(m); phi_zpf=B.σ[idx[m]])
    leaf(m, k) = (d = KronFreeCircuits.dimension(basis(m)); Node([m => basis(m)]; solver=Dense(), compute=min(k, d), keep=lowest(min(k, d))))
    function pair(J)
        kids = [leaf(J, k1j), leaf(partner[J], k1f)]
        d = prod(c.compute for c in kids)
        return Node(kids; solver=d <= 10_000 ? Dense() : lobpcg(), compute=min(k2, d), keep=lowest(min(k2, d)))
    end
    inter(kids, k) = (d = prod(c.compute for c in kids);
                      Node(kids; solver=d <= DENSE_MAX ? Dense() : lobpcg(), compute=min(k, d), keep=lowest(min(k, d))))
    L = inter([pair(:q1), pair(:c1)], kM)
    LCn = inter([L, pair(:q2)], kM2)
    R = inter([pair(:q3), pair(:c2)], kM)
    return Node([LCn, R, loose => FockBasis(kf; phi_zpf=B.σ[idx[loose]])]; solver=lobpcg(), compute, keep=lowest(compute))
end

"""
Flat reference (no intermediate truncation): one root over the five two-mode leaves (J, partner(J)), each solved exactly
(Dense) on its full product basis (fluxonium grid × Fock, coupler grid × Fock) and keeping `kq` (qubit pairs) or `kc`
(coupler pairs) states, and the loose frame mode as a raw Fock axis of `kf` levels; the root, of dimension
kq³ kc² kf, is solved by LOBPCG for `compute` levels with `guard` guard columns. The leaves' truncation is the only one,
so two sizes that agree make a converged reference in these coordinates.
"""
function flat_root(B, partner, loose; kq, kc, kf, compute=8, guard=4, smoke=false)
    idx = Dict(m => i for (i, m) in enumerate(B.names))
    qg = smoke ? PhaseGrid(-3π, 3π, 121; order=8) : QGRID
    cg = smoke ? PhaseGrid(-2π, 2π, 81; order=8) : CGRID
    basis(m) = m in (:q1, :q2, :q3) ? qg : m in (:c1, :c2) ? cg : FockBasis(smoke ? 10 : nfock(m); phi_zpf=B.σ[idx[m]])
    leaf2(J, k) = Node([J => basis(J), partner[J] => basis(partner[J])]; solver=Dense(), compute=k, keep=lowest(k))
    kids = Any[leaf2(:q1, kq), leaf2(:c1, kc), leaf2(:q2, kq), leaf2(:q3, kq), leaf2(:c2, kc)]
    push!(kids, loose => FockBasis(kf; phi_zpf=B.σ[idx[loose]]))
    return Node(kids; solver=lobpcg(; guard), compute, keep=lowest(compute))
end

"Root dimension of the flat reference, and its estimated eigensolver memory in bytes (real arithmetic)."
flat_dim(kq, kc, kf) = big(kq)^3 * big(kc)^2 * kf
# 14 vectors per block column, measured on the two-fluxonia flat solve (7.5 M states, 6 columns, 10.3 GB complex),
# plus the Kronecker apply's scratch; real arithmetic (the working point is real): 8 bytes per entry
flat_bytes(kq, kc, kf; compute=8, guard=4) = flat_dim(kq, kc, kf) * 8 * (14 * (compute + guard) + 4)
