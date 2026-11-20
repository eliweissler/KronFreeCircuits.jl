# Cross-check only (2026-11-12): the hand-built construction of the circuit, compared against the circuit-quantization
# specs (specs/) by spec_vs_julia.jl. The benchmark itself loads the specs (trees.jl, run.jl); this file is not used there.
# The three-fluxonium circuit of Hensel et al., arXiv:2611.05627v1 (Sec. V; App. A Table tab:params, App. B Table
# tab:modes), in pure Julia: netlist, normal modes (checked against the paper), local coordinates with exactly diagonal
# E_C, the shear of the frame modes by the rule of bench/hensel_two_fluxonia/frame_opt/shear_rule.md, the term list
# and the trees. The construction follows bench_archive/hensel_three_fluxonia/quantize_three_fluxonia.py (2026-11-07,
# circuit-quantization), whose partial shear was hand-picked; here the shear fractions come from the rule.
# Needs `ON` (Array or CuArray) from ../common.jl.

using LinearAlgebra, Printf, KronFreeCircuits

# ---- circuit (Table tab:params; nodes 1..12) ---------------------------------------------------------------------------
const E_CHARGE = 1.602176634e-19
const H_PLANCK = 6.62607015e-34
const EC_CONST = E_CHARGE^2 / (2H_PLANCK) * 1e6                   # e²/2h: E_C [GHz] = EC_CONST / C [fF]
const EL_CONST = (H_PLANCK / (2E_CHARGE) / 2π)^2 / H_PLANCK       # φ0²/h [GHz·nH]: E_L = EL_CONST / L [nH]
const NNODES = 12
# The table prints L_c = 8.17, L_end = 59.3, L_br = 36.8 nH, rounded from exact loop targets: the coupler loop 2L_c at
# E_L = 10 GHz and both qubit loops (L_end + L_c + L_4 and 2L_4 + 2L_c + L_br) at E_L = 2.00 GHz (the table's derived
# rows). With those exact values and L_4 = 14.3 nH the normal modes match Table tab:modes to 1.2 MHz (the stiff ones to
# the printed digit); with the printed, rounded values they are off by up to 3.4 MHz.
const L4 = 14.3
const LCLEG = EL_CONST / 20
const LEND = EL_CONST / 2 - LCLEG - L4
const LBR = EL_CONST / 2 - 2L4 - 2LCLEG
const INDUCTORS = [((2, 3), L4), ((5, 6), L4), ((7, 8), L4), ((10, 11), L4),
                   ((3, 5), LCLEG), ((4, 5), LCLEG), ((8, 10), LCLEG), ((9, 10), LCLEG),
                   ((1, 5), LEND), ((12, 9), LEND), ((4, 10), LBR)]
const JUNCTIONS = [(:q1, (1, 2), 3.70), (:q2, (6, 7), 3.70), (:q3, (11, 12), 3.70), (:c1, (4, 3), 10.0), (:c2, (8, 9), 10.0)]
const JNAMES = [j[1] for j in JUNCTIONS]
const C_SHUNT = 10.0                                               # fF across each qubit junction
const C_GROUND = Dict(1 => 70.0, 2 => 70.0, 6 => 70.0, 7 => 70.0, 11 => 70.0, 12 => 70.0,
                      3 => 25.0, 4 => 25.0, 8 => 25.0, 9 => 25.0, 5 => 25.0, 10 => 25.0)
# the working point: every qubit loop at Φ_0/2, both couplers unbiased (Sec. V)
const THETA = Dict(:q1 => :theta_q1, :q2 => :theta_q2, :q3 => :theta_q3, :c1 => :theta_c1, :c2 => :theta_c2)
const PARAMS = [:theta_q1, :theta_q2, :theta_q3, :theta_c1, :theta_c2]
θ_work(_=nothing) = (theta_q1=π, theta_q2=π, theta_q3=π, theta_c1=0.0, theta_c2=0.0)
const PAPER = [0.422, 0.705, 0.816, 1.038, 1.220, 1.477, 1.771]   # GHz, Table tab:q3levels (cap 128, working cutoff)
const PAPER_STIFF_SHIFT = [140, 100, 480, 96, 180, 440, 78] .* 1e-6   # GHz, the same table's "stiff shift" column

function add_branch!(K, (i, j), g)
    K[i, i] += g; K[j, j] += g; K[i, j] -= g; K[j, i] -= g
    return K
end
function capacitance()
    C = zeros(NNODES, NNODES)
    for (n, c) in C_GROUND
        C[n, n] += c
    end
    for (_, (i, j), _) in JUNCTIONS[1:3]
        add_branch!(C, (i, j), C_SHUNT)
    end
    return C
end
function inverse_inductance(; junctions=false)
    K = zeros(NNODES, NNODES)
    for (nodes, L) in INDUCTORS
        add_branch!(K, nodes, 1 / L)
    end
    junctions && for (_, nodes, EJ) in JUNCTIONS
        add_branch!(K, nodes, EJ / EL_CONST)                      # 1/L_J, junction linearized about zero phase
    end
    return K
end
const C_NODE = capacitance()
const LINV_GEO = inverse_inductance()
const PAPER_OMEGA = [2.267, 3.684, 4.270, 4.934, 5.235, 8.380, 9.151, 17.173, 17.350, 21.166, 21.558]
let
    F = eigen(Symmetric(inverse_inductance(; junctions=true)), Symmetric(C_NODE))
    ω = sort(sqrt.(max.(F.values, 0)) .* 1e12 ./ 2π ./ 1e9)[2:end]   # drop the free global mode
    println("normal modes ω_μ/2π (GHz): ", round.(ω; digits=3))
    maximum(abs, ω .- PAPER_OMEGA) < 1.5e-3 || error("normal modes differ from the paper's Table tab:modes: $ω")
end

# ---- local coordinates: x = A φ; five junction phases, six frame directions, the global phase --------------------------
function local_coordinates()
    e = Matrix{Float64}(I, NNODES, NNODES)
    D = reduce(vcat, [(e[i, :] .- e[j, :])' for (_, (i, j), _) in JUNCTIONS])        # 5 × 12
    Ci = inv(C_NODE)
    # frame rows: zero-sum and C⁻¹-orthogonal to every junction row, then C⁻¹-orthonormal (scaled to E_C = EC_CONST / 50)
    N = nullspace(vcat(ones(1, NNODES), D * Ci))                                      # 12 × 6
    G = Symmetric(N' * Ci * N)
    W = (N * inv(sqrt(G)))' ./ sqrt(50.0)                                             # 6 × 12, W C⁻¹ Wᵀ = I / 50
    g = fill(1 / NNODES, 1, NNODES)
    A0 = vcat(D, W, g)
    L0 = inv(A0)' * LINV_GEO * inv(A0)
    R = eigen(Symmetric(L0[6:11, 6:11])).vectors                                     # ascending: f1 the softest
    A = vcat(D, R' * W, g)
    Ai = inv(A)
    EC = EC_CONST * (A * Ci * A')[1:11, 1:11]
    EL = EL_CONST * (Ai' * LINV_GEO * Ai)
    maximum(abs, EC - Diagonal(EC)) < 1e-9 * maximum(abs, EC) || error("local E_C is not diagonal")
    maximum(abs, EL[12, :]) < 1e-9 || error("the global phase enters the inductive energy")
    for b in 1:5
        maximum(abs, D[b, :]' * Ai .- e[b, :]') < 1e-12 || error("junction $(JNAMES[b]) is not the single coordinate x_$b")
    end
    EL = EL[1:11, 1:11]
    # round-off zeros of the construction (E_L frame block, E_C) to exact zeros
    EL[abs.(EL) .< 1e-9 * maximum(abs, EL)] .= 0
    return (; A, EC=Matrix(Diagonal(diag(EC))), EL=(EL + EL') / 2, names=[JNAMES; [Symbol("f$i") for i in 1:6]])
end
const LC = local_coordinates()

# ---- the shear by the rule (shear_rule.md): x = S y, S = [I 0; A I], A_fJ = s*_fJ a*_fJ --------------------------------
"Ground-state ⟨δx²⟩ and ⟨n²⟩ of the junction leaf 4E_C n² + ½E_L x² − E_J cos(x + θ), on a fine grid."
function leaf_moments(EC, EL, EJ, θ; a=5π, N=801)
    x = range(-a, a; length=N); h = step(x)
    T = SymTridiagonal(fill(2 / h^2, N), fill(-1 / h^2, N - 1))                       # n² = −d²/dx² (second order, fine grid)
    H = 4EC .* T + Diagonal(0.5EL .* x .^ 2 .- EJ .* cos.(x .+ θ))
    ψ = eigen(H, 1:1).vectors[:, 1]
    ψ ./= norm(ψ)
    mx = sum(abs2.(ψ) .* x)
    X = sum(abs2.(ψ) .* (x .- mx) .^ 2)
    n2 = dot(ψ, T * ψ)
    return X, n2
end
const EJS = Dict(n => EJ for (n, _, EJ) in JUNCTIONS)
const THETA0 = Dict(:q1 => π, :q2 => π, :q3 => π, :c1 => 0.0, :c2 => 0.0)
sheared(A) = (S = [Matrix{Float64}(I, 5, 5) zeros(5, 6); A Matrix{Float64}(I, 6, 6)];
              Si = inv(S); (; S, EC=Si * LC.EC * Si', EL=S' * LC.EL * S))

"""
The rule's shear: for each frame mode f and junction J with E_L[f,J] ≠ 0, A_fJ = s*_fJ a*_fJ with a*_fJ = −E_L[f,J]/E_L[f,f]
and s* = (X + c)/(X + 2c + P), c = 4E_C[J]/Ω_f, P = (8E_C[J]/Ω_f)² ⟨n²⟩, from the ground state of junction J's leaf in
the current frame; iterated `iters` times (the leaves' inductances and Ω_f depend on the shear). `scale` multiplies
every fraction (1 = the rule, 0 = no shear); `full = true` gives the full shear.
"""
function rule_shear(; iters=3, scale=1.0, full=false, verbose=false)
    A = zeros(6, 5)
    astar = [abs(LC.EL[5+f, J]) > 1e-9 ? -LC.EL[5+f, J] / LC.EL[5+f, 5+f] : 0.0 for f in 1:6, J in 1:5]
    full && return (astar, ones(6, 5))
    s = zeros(6, 5)
    for it in 1:iters
        B = sheared(A)
        for J in 1:5
            X, n2 = leaf_moments(B.EC[J, J], B.EL[J, J], EJS[JNAMES[J]], THETA0[JNAMES[J]])
            for f in 1:6
                astar[f, J] == 0 && continue
                Ω = sqrt(8 * B.EC[5+f, 5+f] * B.EL[5+f, 5+f])
                c = 4 * B.EC[J, J] / Ω
                P = (8 * B.EC[J, J] / Ω)^2 * n2
                s[f, J] = (X + c) / (X + 2c + P)
            end
        end
        A = scale .* s .* astar
    end
    verbose && (println("shear fractions s* (rows f1..f6, columns q1 q2 q3 c1 c2):");
                for f in 1:6 println("   f$f: ", join([astar[f, J] == 0 ? "   -  " : @sprintf("%6.3f", s[f, J]) for J in 1:5], " ")) end)
    return astar .* s .* scale, s
end

"""
The frame with frame-onto-junction shear `A` (6 × 5): E_C, E_L, zero-point spreads σ (from the diagonals) and names.
"""
function frame_of(A)
    B = sheared(A)
    EC, EL = (B.EC + B.EC') / 2, (B.EL + B.EL') / 2
    σ = (8 .* diag(EC) ./ diag(EL)) .^ (1 / 4) ./ sqrt(2)
    return (; S=B.S, EC, EL, σ, names=LC.names, A)
end

"Coupling magnitudes at the zero-point scale (GHz): |8 E_C,ij n_i n_j| + |E_L,ij σ_i σ_j|, n = 1/(2σ)."
function couplings(B)
    n = 1 ./ (2 .* B.σ)
    return [i == j ? 0.0 : abs(8 * B.EC[i, j] * n[i] * n[j]) + abs(B.EL[i, j] * B.σ[i] * B.σ[j]) for i in 1:11, j in 1:11]
end

"Each junction's frame partner, maximizing the total junction–frame coupling (one frame mode left over)."
function best_pairing(B)
    C = couplings(B)
    best, bestp = -Inf, Int[]
    for p in permutations6()
        v = sum(C[J, 5+p[J]] for J in 1:5)
        v > best && (best = v; bestp = p[1:5])
    end
    loose = setdiff(1:6, bestp)[1]
    return Dict(JNAMES[J] => Symbol("f$(bestp[J])") for J in 1:5), Symbol("f$loose")
end
function permutations6()
    out = Vector{Vector{Int}}()
    function rec(prefix, rest)
        isempty(rest) && return push!(out, prefix)
        for r in rest
            rec([prefix; r], setdiff(rest, r))
        end
    end
    rec(Int[], collect(1:6))
    return out
end

# ---- terms: 4 E_C n² + ½ φᵀ E_L φ − Σ_b E_J,b cos(x_b + θ_b) (junction cosines single-mode in every sheared frame) -----
const ALPHA_TOL = 1e-10
coef(x::Real) = Coefficient(x)
function terms_of(B)
    N = 11; T = Term[]
    for i in 1:N
        push!(T, Term(coef(4B.EC[i, i]), [Factor(B.names[i], n2port())]))
        push!(T, Term(coef(0.5B.EL[i, i]), [Factor(B.names[i], phi2port())]))
    end
    for i in 1:N, j in i+1:N
        abs(B.EC[i, j]) > ALPHA_TOL && push!(T, Term(coef(8B.EC[i, j]), [Factor(B.names[i], nport()), Factor(B.names[j], nport())]))
        abs(B.EL[i, j]) > ALPHA_TOL && push!(T, Term(coef(B.EL[i, j]), [Factor(B.names[i], phiport()), Factor(B.names[j], phiport())]))
    end
    for (name, _, EJ) in JUNCTIONS, s in (1, -1)
        push!(T, Term(Coefficient("-0.5 * $EJ * exp($s * im * $(THETA[name]))", PARAMS), [Factor(name, expiphi(s))]))
    end
    return T
end

# ---- trees ----------------------------------------------------------------------------------------------------------------
const QGRID = PhaseGrid(-3π, 3π, 181; order=8)
const CGRID = PhaseGrid(-2π, 2π, 121; order=8)
nfock(m) = m in (:f1, :f2) ? 30 : 20
const TOL = parse(Float64, get(ENV, "HENSEL_TOL", "1e-6"))
lobpcg(; guard=4) = LOBPCG(; tol=TOL, maxiter=3000, guard, on=ON)
const DENSE_MAX = parse(Int, get(ENV, "HENSEL_DENSE_MAX", "5000"))

function tree_parts(B; smoke=false)
    idx = Dict(m => i for (i, m) in enumerate(B.names))
    qg = smoke ? PhaseGrid(-3π, 3π, 121; order=8) : QGRID
    cg = smoke ? PhaseGrid(-2π, 2π, 81; order=8) : CGRID
    basis(m) = m in (:q1, :q2, :q3) ? qg : m in (:c1, :c2) ? cg : FockBasis(smoke ? 10 : nfock(m); phi_zpf=B.σ[idx[m]])
    leaf(m, k) = (d = KronFreeCircuits.dimension(basis(m)); Node([m => basis(m)]; solver=Dense(), compute=min(k, d), keep=lowest(min(k, d))))
    pair(a, b, k1j, k1f, k2) = (kids = [leaf(a, k1j), leaf(b, k1f)]; d = prod(c.compute for c in kids);
                                Node(kids; solver=d <= 10_000 ? Dense() : lobpcg(), compute=min(k2, d), keep=lowest(min(k2, d))))
    inter(kids, k) = (d = prod(c.compute for c in kids);
                      Node(kids; solver=d <= DENSE_MAX ? Dense() : lobpcg(), compute=min(k, d), keep=lowest(min(k, d))))
    return (; basis, leaf, pair, inter, idx)
end

"""
Four-level tree (bench_archive, 2026-11-07): left wing L = [P(q1, ·), P(c1, ·)] keeping kM, joined with the central pair
P(q2, ·) into LC keeping kM2; right wing R = [P(q3, ·), P(c2, ·)] keeping kM; root over [LC, R, loose frame mode as a raw
Fock axis of kf levels]. `partner` maps each junction to its frame mode.
"""
function fourlevel_root(B, partner, loose; k1j=40, k1f=20, k2=60, kM=240, kM2=240, kf=4, compute=8, smoke=false)
    t = tree_parts(B; smoke)
    P(J) = t.pair(J, partner[J], k1j, k1f, k2)
    L = t.inter([P(:q1), P(:c1)], kM)
    LCn = t.inter([L, P(:q2)], kM2)
    R = t.inter([P(:q3), P(:c2)], kM)
    return Node([LCn, R, loose => FockBasis(kf; phi_zpf=B.σ[t.idx[loose]])]; solver=lobpcg(), compute, keep=lowest(compute))
end
