# The two-fluxonium circuit of Hensel, Cáliz, Torrente, Hita-Pérez and Riera, arXiv:2611.05627v1 (Sec. IV, App. A
# Table tab:params, App. B Table tab:modes), its normal modes (checked against Table tab:modes), the local and sheared
# coordinates, the term lists, and the trees. Moved unchanged from bench_archive/hensel_two_fluxonia/run.jl (2026-11-06)
# except for the solvers: plain LOBPCG, whose iterations and matvecs `node_timings` now reports.
# Needs `ON` (Array or CuArray) from ../common.jl.

# ---- circuit (Appendix A, Table tab:params; nodes 1..7) -------------------------------------------------------
const E_CHARGE = 1.602176634e-19                                  # C
const H_PLANCK = 6.62607015e-34                                   # J s
const EC_CONST = E_CHARGE^2 / (2H_PLANCK) * 1e6                   # e²/2h: Hz·F to GHz·fF; E_C = EC_CONST / C[fF]
const EL_CONST = (H_PLANCK / (2E_CHARGE) / 2π)^2 / H_PLANCK       # φ0²/h in GHz·nH: E_L = EL_CONST / L[nH]
const NNODES = 7
const C_GROUND, C_SHUNT = 50.0, 10.0                              # fF
const INDUCTORS = [((1, 4), 50.0), ((4, 5), 50.0), ((2, 3), 50.0), ((6, 7), 50.0), ((3, 4), 7.8), ((4, 7), 7.8)]  # nH
const JUNCTIONS = [(:q1, (1, 2), :E_Jq), (:q2, (5, 6), :E_Jq), (:c, (3, 7), :E_Jc)]
const EJ = (E_Jq=8.98, E_Jc=3.5)                                  # GHz
const PARAMS = [:E_Jq, :E_Jc, :theta_q1, :theta_q2, :theta_c]
θ_at(fc) = (E_Jq=EJ.E_Jq, E_Jc=EJ.E_Jc, theta_q1=π, theta_q2=π, theta_c=2π * fc)

function capacitance()
    C = C_GROUND * Matrix{Float64}(I, NNODES, NNODES)
    for (_, (i, j), _) in JUNCTIONS[1:2]
        C[i, i] += C_SHUNT; C[j, j] += C_SHUNT; C[i, j] -= C_SHUNT; C[j, i] -= C_SHUNT
    end
    return C
end
function add_branch!(K, (i, j), g)
    K[i, i] += g; K[j, j] += g; K[i, j] -= g; K[j, i] -= g
    return K
end
function inverse_inductance(; junctions=false)
    K = zeros(NNODES, NNODES)
    for (nodes, L) in INDUCTORS
        add_branch!(K, nodes, 1 / L)
    end
    junctions && for (_, nodes, key) in JUNCTIONS
        add_branch!(K, nodes, EJ[key] / EL_CONST)                 # 1/L_J in 1/nH
    end
    return K
end
const C_NODE = capacitance()
const LINV_GEO = inverse_inductance()
const LINV_TOT = inverse_inductance(; junctions=true)

# ---- normal modes (Sec. II.B): L_tot⁻¹ v = ω² C v, columns scaled to largest node entry one -----------------------
function normal_modes()
    F = eigen(Symmetric(LINV_TOT), Symmetric(C_NODE))
    order = sortperm(F.values)
    ω = sqrt.(max.(F.values[order], 0)) .* 1e12 ./ 2π ./ 1e9      # (1/nH)/fF = 1e24 s⁻², to GHz
    M = F.vectors[:, order[2:end]]                                # drop the global phase
    M = M ./ maximum(abs, M; dims=1)
    EC = EC_CONST * inv(M' * C_NODE * M)
    EL = EL_CONST * (M' * LINV_GEO * M)
    Ω = 2 .* sqrt.(2 .* diag(EC) .* diag(EL))
    σ = (8 .* diag(EC) ./ diag(EL)) .^ (1 / 4) ./ sqrt(2)
    α = Dict(name => M[i, :] .- M[j, :] for (name, (i, j), _) in JUNCTIONS)
    return (; ω=ω[2:end], M, EC, EL, Ω, σ, α)
end
const NM = normal_modes()
const PAPER_OMEGA = [3.088, 4.840, 6.804, 6.807, 9.903, 14.581]
const PAPER_LOCAL = [3.063, 4.839, 2.607, 2.598, 8.734, 14.576]
let
    println("normal modes ω_μ/2π (GHz): ", round.(NM.ω; digits=3), "   paper: ", PAPER_OMEGA)
    println("local oscillators Ω_μ (GHz): ", round.(NM.Ω; digits=3), "   paper: ", PAPER_LOCAL)
    maximum(abs, NM.ω .- PAPER_OMEGA) < 1e-3 || error("normal-mode frequencies differ from the paper's Table tab:modes")
    maximum(abs, NM.Ω .- PAPER_LOCAL) < 1e-3 || error("local-oscillator frequencies differ from the paper's Table tab:modes")
    maximum(abs, NM.EC - Diagonal(NM.EC)) < 1e-10 || error("E_C is not diagonal in the normal modes")
    for (name, _, _) in JUNCTIONS
        w = NM.α[name] .* NM.σ
        @printf("junction %s: N_eff = %.2f (paper: qubits 2.0, coupler 1.1)\n", name, sum(w .^ 2)^2 / sum(w .^ 4))
    end
end

# ---- local coordinates: x_b = junction phases, three frame modes; E_C exactly diagonal --------------------------
# Rows of A (x = A φ) are the junction differences d_b, three rows spanning their orthogonal complement (within
# zero-sum vectors), and the global phase.  C = C_g I + shunts on d_q1, d_q2, so rows orthogonal in node space
# give a diagonal charging matrix; the frame rows are rotated to diagonalize the frame block of E_L.
function local_coordinates()
    e = Matrix{Float64}(I, NNODES, NNODES)
    d = [e[i, :] .- e[j, :] for (_, (i, j), _) in JUNCTIONS]
    W = [1 1 0 0 -1 -1 0; 1 1 -2 0 1 1 -2; 1 1 1 -6 1 1 1] ./ 1.0
    W = W ./ sqrt.(sum(W .^ 2; dims=2))
    A2 = vcat(d[1]', d[2]', d[3]', W, fill(1 / NNODES, 1, NNODES))
    L2 = inv(A2)' * LINV_GEO * inv(A2)
    R = eigen(Symmetric(L2[4:6, 4:6])).vectors
    A = vcat(d[1]', d[2]', d[3]', R' * W, fill(1 / NNODES, 1, NNODES))
    Ainv = inv(A)
    K = A * inv(C_NODE) * A'
    L = Ainv' * LINV_GEO * Ainv
    EC = EC_CONST * K[1:6, 1:6]
    EL = EL_CONST * L[1:6, 1:6]
    maximum(abs, EC - Diagonal(EC)) < 1e-10 || error("local E_C is not diagonal")
    maximum(abs, L[7, :]) < 1e-12 || error("the global phase enters the inductive energy")
    for (b, (name, (i, j), _)) in enumerate(JUNCTIONS)
        row = (e[i, :] .- e[j, :])' * Ainv
        maximum(abs, row .- e[b, :]') < 1e-12 || error("junction $name is not the single coordinate x_$b")
    end
    σ = (8 .* diag(EC) ./ diag(EL)) .^ (1 / 4) ./ sqrt(2)
    return (; A, EC, EL, σ, names=[:q1, :q2, :c, :f1, :f2, :f3])
end
const LC = local_coordinates()

"""
The local basis with the frame modes in `frames` sheared onto the junction coordinates, φ_f = x_f + a x_J with
a = -E_L[f,f]⁻¹ E_L[f,J] (Eli, 2026-11-06): E_L loses its junction–frame block (the junction block becomes the Schur
complement, the effective loop inductances) and the coupling moves to charge–charge terms E_C' = S⁻¹ E_C S⁻ᵀ.
Every junction cosine stays a single mode.
"""
function sheared_coordinates(frames)
    names = LC.names
    J = 1:3
    F = [findfirst(==(f), names) for f in frames]
    S = Matrix{Float64}(I, 6, 6)
    S[F, J] = -LC.EL[F, F] \ LC.EL[F, J]
    Sinv = inv(S)
    EC = Sinv * LC.EC * Sinv'
    EL = S' * LC.EL * S
    σ = (8 .* diag(EC) ./ diag(EL)) .^ (1 / 4) ./ sqrt(2)
    return (; S, EC, EL, σ, names, frames)
end
println("local coordinates: E_C diag ", round.(diag(LC.EC); digits=4), ", E_L diag ", round.(diag(LC.EL); digits=3))

# ---- term lists ------------------------------------------------------------------------------------------------
coef(x::Real) = Coefficient(x)
cosine_coefficient(key::Symbol, θname::Symbol, s::Int) = Coefficient("-0.5 * $key * exp($s * im * $θname)", PARAMS)
const THETA = Dict(:q1 => :theta_q1, :q2 => :theta_q2, :c => :theta_c)
const ALPHA_TOL = 1e-10

"""
Terms of `4 E_C n² + ½ φᵀ E_L φ - Σ_b E_J,b cos(ωᵀ_b φ + θ_b)` on modes `names`.
`quadratic = false` leaves out the single-mode quadratic terms (a GivenSystem carries them as its energies).
`cosfactor(b, μ, s)` is the factor of junction `b` on mode index `μ` with sign `s` (`exp(s i ω_bμ φ_μ)`).
"""
function hamiltonian_terms(names, EC, EL, omegas; quadratic=true, cosfactor)
    N = length(names)
    T = Term[]
    if quadratic
        for i in 1:N
            push!(T, Term(coef(4EC[i, i]), [Factor(names[i], n2port())]))
            push!(T, Term(coef(0.5EL[i, i]), [Factor(names[i], phi2port())]))
        end
    end
    for i in 1:N, j in i+1:N
        abs(EC[i, j]) > ALPHA_TOL && push!(T, Term(coef(4(EC[i, j] + EC[j, i])), [Factor(names[i], nport()), Factor(names[j], nport())]))
        abs(EL[i, j]) > ALPHA_TOL && push!(T, Term(coef(0.5(EL[i, j] + EL[j, i])), [Factor(names[i], phiport()), Factor(names[j], phiport())]))
    end
    for (name, _, key) in JUNCTIONS, s in (1, -1)
        ω = omegas[name]
        fs = [cosfactor(name, μ, s) for μ in 1:N if abs(ω[μ]) > ALPHA_TOL]
        push!(T, Term(cosine_coefficient(key, THETA[name], s), fs))
    end
    return T
end

# ---- solvers and trees ------------------------------------------------------------------------------------------
# Root and branch nodes: LOBPCG at `tol` GHz on eigenvalue residuals (default 1e-6 = 1 kHz, ten times below the
# strictest target), four guard columns.
const TOL = parse(Float64, get(ENV, "HENSEL_TOL", "1e-6"))
root_solver(; tol=TOL, guard=4) = LOBPCG(; tol, maxiter=3000, guard, on=ON)

"""
Flat tree in local coordinates: fluxonium leaves q1, q2 on `grid` keeping `kq`; the coupler leaf (c, f1) on Fock bases
of `nc`, `nf1` levels keeping `kc`; f2, f3 as raw Fock axes of `kf2`, `kf3` levels; the root keeps `compute`.
"""
function local_root(; grid, kq, nc, nf1, kc, kf2, kf3, compute, B=LC)
    q(name) = Node([name => grid]; solver=Dense(), compute=kq, keep=lowest(kq))
    coupler = Node([:c => FockBasis(nc; phi_zpf=B.σ[3]), :f1 => FockBasis(nf1; phi_zpf=B.σ[4])];
                   solver=Dense(), compute=kc, keep=lowest(kc))
    return Node([q(:q1), q(:q2), coupler, :f2 => FockBasis(kf2; phi_zpf=B.σ[5]), :f3 => FockBasis(kf3; phi_zpf=B.σ[6])];
                solver=root_solver(), compute, keep=lowest(compute))
end

"""
Fully binary tree with single-mode leaves (Eli, 2026-11-06): 1D leaves keeping `k1q` (fluxonium grids) or `k1f` (Fock
modes of `nfock` levels), pair nodes P(a, b) keeping `k2`, M over P(c, f1) and P(q1, f3) keeping `kM`, root over
[M, P(q2, f2)] keeping `compute`. Pair nodes are Dense up to 10 000 product states, LOBPCG above.
"""
function binary_root(; grid, nfock, k1q, k1f, k2, kM, compute, B=LC)
    idx = Dict(name => i for (i, name) in enumerate(B.names))
    leaf(m) = m in (:q1, :q2) ? Node([m => grid]; solver=Dense(), compute=k1q, keep=lowest(k1q)) :
              Node([m => FockBasis(nfock[m]; phi_zpf=B.σ[idx[m]])]; solver=Dense(), compute=min(k1f, nfock[m]),
                   keep=lowest(min(k1f, nfock[m])))
    keptof(m) = m in (:q1, :q2) ? k1q : min(k1f, nfock[m])
    function pairnode(a, b)
        d = keptof(a) * keptof(b)
        k = min(k2, d)
        return Node([leaf(a), leaf(b)]; solver=d <= 10_000 ? Dense() : root_solver(), compute=k, keep=lowest(k))
    end
    M = Node([pairnode(:c, :f1), pairnode(:q1, :f3)]; solver=root_solver(), compute=kM, keep=lowest(kM))
    return Node([M, pairnode(:q2, :f2)]; solver=root_solver(), compute, keep=lowest(compute))
end

"""
The paper's own truncated Hamiltonian (Sec. II.B, App. C): six normal modes as GivenSystems with `m` Fock levels,
energies Ω_μ (k + ½), the truncated φ̂_μ = σ_μ (a + a†), and for each junction and sign the matrix exponential of
s i α_bμ φ̂_μ; one root over all six (m⁶ states). This is the space of their exact-HD reference.
"""
function paper_children(m)
    kids = GivenSystem[]
    for μ in 1:6
        a = diagm(1 => sqrt.(1.0:m-1))
        φ = NM.σ[μ] .* (a + a')
        ops = Dict{Symbol,Matrix{ComplexF64}}(:phi => φ)
        for (name, _, _) in JUNCTIONS, s in (1, -1)
            αb = NM.α[name][μ]
            abs(αb) > ALPHA_TOL || continue
            ops[Symbol("u_", name, s > 0 ? "p" : "m")] = exp(im * s * αb .* φ)
        end
        push!(kids, GivenSystem(Symbol("m", μ); energies=NM.Ω[μ] .* ((0:m-1) .+ 0.5), operators=ops))
    end
    return kids
end
paper_terms() = hamiltonian_terms([Symbol("m", μ) for μ in 1:6], NM.EC, NM.EL, NM.α; quadratic=false,
                                   cosfactor=(b, μ, s) -> Factor(Symbol("m", μ), Port(Symbol("u_", b, s > 0 ? "p" : "m"), 0)))
local_terms(B=LC) = hamiltonian_terms(B.names, B.EC, B.EL, Dict(:q1 => [1, 0, 0, 0, 0, 0], :q2 => [0, 1, 0, 0, 0, 0], :c => [0, 0, 1, 0, 0, 0]);
                                      cosfactor=(b, μ, s) -> Factor(B.names[μ], expiphi(s)))
