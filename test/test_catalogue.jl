# Pathological-case catalogue (docs/spikes/2026-10-16-pathological-test-catalogue/README.md):
# whole-pipeline regressions before Plan 2 acceptance, one testset per case identifier.
# Every reference is assembled independently of the apply routines (`primitive_hamiltonian`, closed
# forms, explicit Kronecker products); the larger versions are gated in test_zero_pi_reference.jl.

using Random
using KronFreeCircuits: Node, Dense, LOBPCG, lowest, materialize, energies, port, eigenvectors, materialized,
                        circuit_terms, constant_term, modes, localop, dense, branch_peak_phase, fix_phases!,
                        GivenSystem, Term, Factor, Coefficient, Port, positive_mask, SelectionRule, SelectionContext,
                        nport, n2port, phiport, phi2port, expiphi, ChargeBasis, PhaseGrid, FockBasis, load_spec

# A selection rule that inspects a port (so the port is formed on all computed vectors before
# slicing) and keeps the given permutation of the computed states.
struct _PermutedSelection <: SelectionRule
    perm::Vector{Int}
    port::Tuple{Symbol,Port}
end
KronFreeCircuits.select(r::_PermutedSelection, ::SelectionContext) = copy(r.perm)
KronFreeCircuits.ports_needed(r::_PermutedSelection) = Tuple{Symbol,Port}[r.port]

@testset "F2: connected with a parity-preserving port (public path)" begin
    # Completes F2 (operator level in test_solvers.jl, sector-confined warm start in test_materialize.jl):
    # selection through e^{2iφ}, which preserves charge parity, on eigenvectors that are parity-resolved.
    using KronFreeCircuits: connected
    src = """
    {"format": "kronfree-spec", "version": 1, "units": "GHz",
     "parameters": ["E_C", "E_2", "n_g"],
     "modes": [{"name": "varphi_1", "type": "c"}],
     "E_C": [["E_C"]], "E_L": [["0"]], "flux_linear": ["0"], "n_g": ["n_g"],
     "cosines": [{"energy": "E_2", "omega": [2], "offset": "0"}]}
    """
    terms = circuit_terms(KronFreeCircuits.parse_spec(src))
    ncut = 20; nv = collect(-ncut:ncut); ev = iseven.(nv); b = ChargeBasis(ncut)
    sel = connected([1], (:varphi_1, expiphi(2)), 0.05)
    Π = Diagonal(ifelse.(ev, 1.0, -1.0))                          # charge parity
    for ng in (0.2, 0.5)
        θ = (E_C=1.2, E_2=3.0, n_g=ng)
        A = Matrix(primitive_hamiltonian(terms, θ, [:varphi_1 => b]))
        @test norm(A[ev, .!ev]) == 0                             # the sectors are invariant
        Esec = sort([eigvals(Hermitian(A[ev, ev])); eigvals(Hermitian(A[.!ev, .!ev]))])
        sys = materialize(Node([:varphi_1 => b]; compute=12, keep=sel), terms, θ)
        V, E = eigenvectors(sys), energies(sys)
        @test length(E) > 1 && norm(V' * V - I) < 1e-10
        @test E[1] ≈ Esec[1] atol=1e-10
        @test maximum(_residuals(A, E, V)) < 1e-9
        P = port(sys, :varphi_1, expiphi(2))
        @test P ≈ V' * dense(localop(b, expiphi(2))) * V atol=1e-12
        w = [sum(abs2, V[ev, j]) for j in axes(V, 2)]
        if ng == 0.2
            # nondegenerate levels: the eigenvectors are parity-resolved, and the selection stays in the
            # ground state's sector
            wo = [sum(abs2, V[.!ev, j]) for j in axes(V, 2)]     # off-sector weights (1 - 1e-20 == 1.0, so test these, not w)
            @test all(j -> min(w[j], wo[j]) < 1e-20, axes(V, 2))
            @test all(==(w[1] > 0.5), w .> 0.5)
            Esame = eigvals(Hermitian(w[1] > 0.5 ? A[ev, ev] : A[.!ev, .!ev]))
            @test all(e -> minimum(abs.(Esame .- e)) < 1e-9, E)
        else
            # n_g = 1/2: the even and odd levels pair up to rounding (the window edge is unreached), so Dense
            # may return any rotation inside a pair, and the public path offers no way to request
            # parity-resolved vectors.  `connected` follows those rotations and can keep one member of a pair
            # without its partner: selection rules do not close over near-degenerate clusters (a contract
            # gap, decision pending; see outcomes.md).  Asserted without depending on the rotation: the
            # clusters are built from the independently diagonalized parity blocks (eigenvalues within 1e-8
            # relative of a kept energy); the kept span lies inside them, and it leaks parity only if it is
            # rank-deficient in at least one cluster (a split pair).
            tol(e) = 1e-8 * max(1, abs(e))
            Fe, Fo = eigen(Hermitian(A[ev, ev])), eigen(Hermitian(A[.!ev, .!ev]))
            cols, λ = Vector{ComplexF64}[], Float64[]
            for (F, mask) in ((Fe, ev), (Fo, .!ev)), j in eachindex(F.values)
                any(e -> abs(F.values[j] - e) <= tol(e), E) || continue
                q = zeros(ComplexF64, length(nv)); q[mask] = F.vectors[:, j]
                push!(cols, q); push!(λ, F.values[j])
            end
            Q = reduce(hcat, cols)                                   # parity-resolved, orthonormal
            @test norm(Q * (Q' * V) - V) < 1e-8                      # span(V) ⊂ span of the kept clusters
            clusters = unique([findall(l -> abs(l - e) <= tol(e), λ) for e in E])
            ranks = [count(>(1e-6), svdvals(Q[:, c]' * V)) for c in clusters]
            split = [λ[c[1]] for (c, r) in zip(clusters, ranks) if r < length(c)]
            leak = norm(Π * (V * V') - (V * V') * Π)
            @test leak < 1e-8 || !isempty(split)                     # leakage only through split clusters
            @info "F2 n_g = 1/2: " * (isempty(split) ? "no cluster split" : "split near-degenerate pairs at E = $(round.(split; digits=4))") *
                  ", ‖[Π, VV']‖ = $(round(leak; sigdigits=3))"
        end
    end
end

@testset "F4: exchange symmetry of identical, weakly coupled transmons" begin
    spec = load_spec(fixture("two_transmons.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θ = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=30.0, E_C2=1.2, n_g2=0.25, g=0.05)   # identical, symmetric coupling
    nc = 8; d = 2nc + 1; k = 6; m = 5
    b = ChargeBasis(nc)
    Hp = primitive_hamiltonian(terms, θ, [:varphi_1 => b, :varphi_2 => b]; constant=c(θ))
    S = spzeros(d^2, d^2); for i in 1:d, j in 1:d; S[i+d*(j-1), j+d*(i-1)] = 1; end   # exchange (n1, n2) -> (n2, n1)
    @test norm(S * Hp * S - Hp) < 1e-12                          # the fixture is exchange symmetric
    Fp = eigen(Hermitian(Matrix(Hp)))
    leaves() = [Node([:varphi_1 => b]; keep=lowest(k)), Node([:varphi_2 => b]; keep=lowest(k))]
    Random.seed!(71)
    top = Node(leaves(); solver=LOBPCG(tol=1e-10), compute=m, keep=lowest(m))
    sys = @test_logs min_level=Base.CoreLogging.Warn materialize(top, terms, θ; constant=c)
    q1, q2 = materialized(sys, top.children[1]), materialized(sys, top.children[2])
    @test q1.energies ≈ q2.energies atol=1e-12                  # identical leaves ...
    @test q1.vectors ≈ q2.vectors atol=1e-10                    # ... with the same phase-fixed states
    W = kron(q2.vectors, q1.vectors)
    Hproj = W' * Hp * W                                          # the projected Hamiltonian, built independently
    @test Hproj ≈ Hproj' atol=1e-12
    Fq = eigen(Hermitian(Hproj))
    E, C = energies(sys), eigenvectors(sys)
    @test E ≈ Fq.values[1:m] atol=1e-8
    @test maximum(_residuals(Hproj, E, C)) < 1e-8 && norm(C' * C - I) < 1e-10
    check_eigenpairs(KronFreeCircuits.operator(sys), E, C; tol=1e-9)
    @test E ≈ Fp.values[1:m] atol=1e-6                           # retained-subspace error of the low states is small
    Ψ = W * C
    parity = [real(dot(Ψ[:, j], S * Ψ[:, j])) for j in 1:m]
    @test all(p -> abs(abs(p) - 1) < 1e-8, parity)               # every state has definite exchange parity
    @test parity ≈ [real(dot(Fp.vectors[:, j], S * Fp.vectors[:, j])) for j in 1:m] atol=1e-6
    @test parity[2] * parity[3] < 0 && parity[4] * parity[5] < 0  # each excitation pair: one symmetric, one antisymmetric
    for j in 1:m                                                 # nondegenerate levels: one-dimensional projectors
        @test abs2(dot(Fq.vectors[:, j], C[:, j])) ≈ 1 atol=1e-8
    end
    # the single-excitation cluster against the primitive reference (retained-subspace error only)
    @test norm(Ψ[:, 2:3] * Ψ[:, 2:3]' - Fp.vectors[:, 2:3] * Fp.vectors[:, 2:3]') < 1e-4
    # Bell-like coefficients on |1,0⟩ and |0,1⟩ (product indices 2 and k+1): moduli 1/√2 (not 1/2), equal by symmetry
    for j in 2:3
        @test abs(C[2, j]) ≈ abs(C[k+1, j]) atol=1e-8
        @test abs(C[2, j]) ≈ 1 / sqrt(2) atol=1e-2
    end
    # the antisymmetric state vanishes on the exchange diagonal n1 = n2
    for j in findall(<(0), parity)
        @test maximum(abs.(diag(reshape(Ψ[:, j], d, d)))) < 1e-9
    end
end

@testset "F6: symmetric to asymmetric full zero-pi (small discretization)" begin
    spec = load_spec(fixture("full_zero_pi.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θa = full_zero_pi_params()                                    # generic n_g and flux; every asymmetry on
    θs = merge(θa, (dE_J=0.0, dC_J=0.0, dC=0.0, dE_L=0.0))
    cb, grid = ChargeBasis(6), PhaseGrid(-6π, 6π, 101; order=6)
    fock = FockBasis(6; phi_zpf=(2θa.E_Cz / θa.E_L)^(1 / 4))
    prim = [:theta => cb, :phi => grid, :zeta => fock]
    m = 4; kz = 8
    zpnode = Node([:theta => cb, :phi => grid]; solver=Dense(), compute=kz, keep=lowest(kz))
    top(pc) = Node([zpnode, :zeta => fock]; solver=LOBPCG(tol=1e-9, preconditioner=pc), compute=m,
                   keep=lowest(m), ports=[(:theta, nport()), (:phi, phiport()), (:zeta, nport())])
    mat(t, θ; kw...) = @test_logs (:warn, r"edge") match_mode=:any materialize(t, terms, θ; constant=c, kw...)
    configs = [("symmetric", θs), ("dE_J", merge(θs, (dE_J=θa.dE_J,))), ("dC_J", merge(θs, (dC_J=θa.dC_J,))),
               ("dC", merge(θs, (dC=θa.dC,))), ("dE_L", merge(θs, (dE_L=θa.dE_L,))), ("all", θa),
               ("all, small step", merge(θa, (n_g=θa.n_g + 1e-3,)))]
    Random.seed!(76)
    prev = nothing; sys_sym = nothing; sys_all = nothing
    for (name, θ) in configs
        EC = [real(spec.E_C[i, j](θ)) for i in 1:3, j in 1:3]
        @test isposdef(Symmetric(EC))                            # the charging matrix stays positive definite
        # each asymmetry is a step from the symmetric solution; the small step is from all asymmetries on
        prev = name == "symmetric" ? nothing : name == "all, small step" ? sys_all : sys_sym
        sys = mat(top(:ritz), θ; previous=prev)
        name == "symmetric" && (sys_sym = sys); name == "all" && (sys_all = sys)
        # the zero-pi leaf: its matrix and its eigenpairs against the independent primitive matrix
        zp = materialized(sys, zpnode)
        Hl = primitive_hamiltonian(filter(t -> all(in((:theta, :phi)), modes(t)), terms), θ, prim[1:2])
        name == "all" && @test Matrix(KronFreeCircuits.operator(sys, zpnode)) ≈ Hl atol=1e-10   # once: the leaf matrix
        @test maximum(_residuals(Hl, zp.energies, zp.vectors)) < 1e-9 && norm(zp.vectors' * zp.vectors - I) < 1e-10
        # the root: true residuals, Hermiticity and spectrum of the projected Hamiltonian, assembled independently
        Hfull = primitive_hamiltonian(terms, θ, prim; constant=c(θ))
        W = kron(Matrix{ComplexF64}(I, 6, 6), zp.vectors)
        Hproj = W' * (Hfull * W)
        @test norm(Hproj - Hproj') < 1e-10
        Fq = eigen(Hermitian(Hproj))
        E, C = energies(sys), eigenvectors(sys)
        @test E ≈ Fq.values[1:m] atol=1e-8
        @test maximum(_residuals(Hproj, E, C)) < 1e-8 && norm(C' * C - I) < 1e-10
        @test Fq.values[m+1] - Fq.values[m] > 1e-3 && norm(C * C' - Fq.vectors[:, 1:m] * Fq.vectors[:, 1:m]') < 1e-7
        # projected ports, against the primitive operators on the explicit primitive states
        Ψ = W * C
        for (mode, p) in ((:theta, nport()), (:phi, phiport()), (:zeta, nport()))
            O = primitive_hamiltonian([Term(Coefficient(1), [Factor(mode, p)])], θ, prim)
            P = port(sys, mode, p)
            @test P ≈ Ψ' * O * Ψ atol=1e-9
            @test P ≈ P' atol=1e-10
        end
        # the operator under test at the root: Hermitian and equal to the independent projection
        H = KronFreeCircuits.operator(sys)
        Hm = Matrix(H)
        @test norm(Hm - Hm') < 1e-10 && norm(Hm - Hproj) < 1e-9
        # fixed (:jacobi) and adaptive (:ritz) Jacobi on that operator, cold and warm: :ritz must converge,
        # :jacobi gives valid output or an explicit nonconvergence error; no required speed ordering
        for pc in (:ritz, :jacobi), X0 in (nothing, prev === nothing ? nothing : prev.root.vectors[:, 1:m])
            out = try KronFreeCircuits.lobpcg_solve(LOBPCG(tol=1e-9, preconditioner=pc), H, m; X0=X0) catch e; e end
            if out isa Exception
                @test pc === :jacobi && out isa ErrorException && occursin("did not converge", out.msg)
            else
                @test out[1] ≈ Fq.values[1:m] atol=1e-8
                @test maximum(_residuals(Hproj, out[1], out[2])) < 1e-8
                @test out[3].n_matvec > 0
            end
        end
    end
end

@testset "F8: hierarchical vs Dense on the same projection vs the primitive reference, cutoffs increased" begin
    # Extends the small F8 testset in test_materialize.jl: an independently assembled projection and
    # primitive reference, both offset charges nonzero (linear pieces routed to the leaves, constant at
    # the root), and child cutoffs up to the complete charge basis.
    spec = load_spec(fixture("two_transmons.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θ = (two_transmons_params(; g=4.0)..., n_g2=0.4)               # strong coupling
    b = ChargeBasis(8); d = 17; m = 4
    Hp = primitive_hamiltonian(terms, θ, [:varphi_1 => b, :varphi_2 => b]; constant=c(θ))
    Ep = eigvals(Hermitian(Matrix(Hp)))[1:m]
    leaves(k) = [Node([:varphi_1 => b]; keep=lowest(k)), Node([:varphi_2 => b]; keep=lowest(k))]
    Random.seed!(78)
    cutoffs = (3, 5, 8, 12, d)
    Eproj = Vector{Vector{Float64}}()
    for k in cutoffs
        sd = materialize(Node(leaves(k); solver=Dense(), keep=lowest(m)), terms, θ; constant=c)
        q1, q2 = sd.root.children
        W = kron(q2.vectors, q1.vectors)
        Fq = eigen(Hermitian(W' * Hp * W))                      # Dense on exactly the same projected Hamiltonian
        push!(Eproj, Fq.values[1:m])
        @test energies(sd) ≈ Fq.values[1:m] atol=1e-9
        @test maximum(_residuals(W' * Hp * W, energies(sd), eigenvectors(sd))) < 1e-9
        k^2 >= 3m + 5 || continue                                # LOBPCG needs N >= 3m + 5
        sl = materialize(Node(leaves(k); solver=LOBPCG(tol=1e-10), compute=m, keep=lowest(m)), terms, θ; constant=c)
        @test sl.root.children[1].vectors ≈ q1.vectors          # same leaves: the same projection
        @test energies(sl) ≈ Fq.values[1:m] atol=1e-8            # no solver error, whatever the truncation
        @test maximum(_residuals(W' * Hp * W, energies(sl), eigenvectors(sl))) < 1e-8
        @test eigenvectors(sl) * eigenvectors(sl)' ≈ eigenvectors(sd) * eigenvectors(sd)' atol=1e-7
    end
    errs = [maximum(Eproj[i] .- Ep) for i in eachindex(cutoffs)]
    @test all(i -> all(Eproj[i] .>= Ep .- 1e-10), eachindex(cutoffs))           # variational: never below the reference
    @test all(i -> all(Eproj[i+1] .<= Eproj[i] .+ 1e-10), 1:length(cutoffs)-1)  # nested retained spaces: monotone
    @test errs[1] > 0.1                                          # the inadequate cutoff: a retained-subspace error ...
    @test errs[end] < 1e-9                                       # ... which vanishes with the complete child bases
end

@testset "F9: undersized and changing phase grids (small)" begin
    # Fluxonium at half flux: the boundary diagnostic must track the edge support of the kept states,
    # and energy changes from the interval and the spacing are measured separately from solver residuals.
    # (A previous solution on a different interval with the same point count is rejected: T2 in
    # test_materialize.jl and test_selection.jl.)
    spec = load_spec(fixture("fluxonium.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θ = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π); k = 4; ref = reference("fluxonium").energies[1:k]
    function grid_run(grid)
        logs, sys = Test.collect_test_logs(() -> materialize(Node([:varphi_1 => grid]; compute=k, keep=lowest(k)), terms, θ; constant=c))
        V = eigenvectors(sys); h = grid.order ÷ 2
        s = [norm(V[[1:h; grid.N-h+1:grid.N], j]) for j in 1:k]           # edge support, recomputed
        @test any(l -> occursin("too small", string(l.message)), logs) == (maximum(s) > KronFreeCircuits._EDGE_TOL)
        @test all(l -> occursin("too small", string(l.message)), logs)       # no other diagnostics
        Hp = primitive_hamiltonian(terms, θ, [:varphi_1 => grid]; constant=c(θ))
        @test maximum(_residuals(Hp, energies(sys), V)) < 1e-9              # tiny solver residual, at every size
        return maximum(abs.(energies(sys) .- ref)), maximum(s)
    end
    Δ = 2π / 40                                                   # widen the interval at fixed spacing
    wide = [grid_run(PhaseGrid(-L, L, round(Int, 2L / Δ) + 1)) for L in (2π, 4π, 6π, 10π)]
    @test wide[1][2] > 1e-2 && wide[1][1] > 1e-1                 # undersized: large edge support and energy error
    @test wide[end][2] < KronFreeCircuits._EDGE_TOL               # wide enough: no diagnostic
    @test issorted([w[2] for w in wide]; rev=true)
    @test wide[end][1] < 1e-3                                     # what remains is the spacing error
    fine = [grid_run(PhaseGrid(-10π, 10π, N)) for N in (201, 401, 801)] # refine the spacing at a fixed adequate interval
    @test all(f -> f[2] < KronFreeCircuits._EDGE_TOL, fine)
    @test issorted([f[1] for f in fine]; rev=true) && fine[end][1] < 1e-6
end

@testset "P1: projected square versus square of the projection (Fock)" begin
    spec = load_spec(fixture("fluxonium.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θ = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π); z = (2θ.E_C / θ.E_L)^(1 / 4)
    allports = [(:varphi_1, p) for p in (nport(), n2port(), phiport(), phi2port())]
    # The declared Fock operators, from a = Σ √m |m-1⟩⟨m| on the window (not from localop): n² and φ² are
    # the closed-form band of the infinite-space squares, not squares of the truncated generators.
    function declared(N, z)
        a = diagm(1 => sqrt.(1.0:N-1)); nz = 1 / (2z); dg = Diagonal(2 .* (0:N-1) .+ 1.0)
        return (n=im * nz * (a' - a), φ=z * (a + a'), n2=nz^2 * (dg - a^2 - a'^2), φ2=z^2 * (dg + a^2 + a'^2))
    end
    for (N, k) in ((30, 4), (6, 3))
        fb = FockBasis(N; phi_zpf=z); O = declared(N, z)
        @test dense(localop(fb, n2port())) ≈ O.n2 && dense(localop(fb, phi2port())) ≈ O.φ2
        @test dense(localop(fb, nport())) ≈ O.n && dense(localop(fb, phiport())) ≈ O.φ
        @test abs(O.n2[N, N] - (O.n^2)[N, N]) ≈ N / (4z^2)       # the truncated square differs on the last level
        sys = materialize(Node([:varphi_1 => fb]; keep=lowest(k), ports=allports), terms, θ; constant=c)
        V = eigenvectors(sys)
        for (p, O2, O1) in ((n2port(), O.n2, O.n), (phi2port(), O.φ2, O.φ))
            P = port(sys, :varphi_1, p)
            @test P ≈ V' * O2 * V atol=1e-12                     # projection of the declared square
            @test P ≈ P'
            @test opnorm(P - port(sys, :varphi_1, p.kind === :n2 ? nport() : phiport())^2) > 1e-2   # ≠ (V'OV)²
            @test port(sys, :varphi_1, p.kind === :n2 ? nport() : phiport()) ≈ V' * O1 * V atol=1e-12
            # with appreciable weight on the last level, the square of the truncated generator is a second truncation
            N == 6 && @test opnorm(P - V' * O1^2 * V) > 1e-3
        end
    end
    # Through a branch: the root's n² port of an oscillator mode is the projection of the primitive n²
    # (via the child's projected square), not the square of the root's n port.
    src = """
    {"format": "kronfree-spec", "version": 1, "units": "GHz",
     "parameters": ["E_J", "E_C1", "n_g", "E_C2", "E_L", "g"],
     "modes": [{"name": "varphi_1", "type": "c"}, {"name": "x", "type": "h"}],
     "E_C": [["E_C1", "g/8"], ["g/8", "E_C2"]], "E_L": [["0", "0"], ["0", "E_L"]],
     "flux_linear": ["0", "0"], "n_g": ["n_g", "0"],
     "cosines": [{"energy": "E_J", "omega": [1, 0], "offset": "0"}]}
    """
    θb = (E_J=20.0, E_C1=1.0, n_g=0.2, E_C2=0.5, E_L=2.0, g=1.0)
    zx = 1.3 * (2θb.E_C2 / θb.E_L)^(1 / 4)                         # mismatched scale: the oscillator leaf mixes Fock states
    Nx = 10; fx = FockBasis(Nx; phi_zpf=zx); qb = ChargeBasis(8)
    top = Node([Node([:varphi_1 => qb]; keep=lowest(4)), Node([:x => fx]; keep=lowest(3))]; keep=lowest(4),
               ports=[(:x, p) for p in (nport(), n2port(), phiport(), phi2port())])
    sys = materialize(top, circuit_terms(KronFreeCircuits.parse_spec(src)), θb)
    Ψ = _primitive(sys.root); O = declared(Nx, zx)
    Id = Matrix{ComplexF64}(I, 17, 17)
    for (p, O2, p1) in ((n2port(), O.n2, nport()), (phi2port(), O.φ2, phiport()))
        P = port(sys, :x, p)
        @test P ≈ Ψ' * kron(O2, Id) * Ψ atol=1e-11
        @test opnorm(P - port(sys, :x, p1)^2) > 1e-2
    end
    @test port(sys, :x, nport()) ≈ Ψ' * kron(O.n, Id) * Ψ atol=1e-11
end

@testset "P2: port covariance under V -> V U (permutations, complex phases)" begin
    spec = load_spec(fixture("fluxonium.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θ = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=0.8π)
    grid = PhaseGrid(-10π, 10π, 201); k = 5
    sel = (:varphi_1, nport())                                    # formed on all computed vectors, before slicing
    req = [(:varphi_1, p) for p in (phiport(), n2port(), expiphi(0.5), expiphi(-0.5))]   # formed after slicing
    keys_ = vcat([sel], req)
    leaf(perm; kw...) = Node([:varphi_1 => grid]; compute=12, keep=_PermutedSelection(perm, sel), ports=req, kw...)
    function checkports(sys)
        V = eigenvectors(sys)
        for (mode, p) in keys_
            P = port(sys, mode, p)
            @test P ≈ V' * dense(localop(grid, p)) * V atol=1e-12  # the declared operator on the kept vectors
            p.kind === :expiphi || @test P ≈ P'                   # Hermitian observables stay Hermitian
        end
        @test port(sys, :varphi_1, expiphi(0.5)) ≈ port(sys, :varphi_1, expiphi(-0.5))' atol=1e-12   # conjugate ports
    end
    Random.seed!(82)
    perm = [3, 1, 5, 2, 4]; ph = cis.(2π .* rand(k))
    function phased(sys, U)                                       # the same previous solution with V -> V U
        r = sys.root
        mn = KronFreeCircuits.MaterializedNode(r.node, r.dims, r.energies, r.vectors * U, r.kept, r.ports, r.children,
                                               r.peak_component, r.peak_amplitudes)
        return KronFreeCircuits.System(mn, sys.spec, sys.terms, sys.θ, sys.constant)
    end
    # the fluxonium leaf is real by structure and its phases are signs: complex phases on the complex path, signs on the real one
    for (ar, phs) in ((:complex, ph), (:auto, Float64[1, -1, -1, 1, -1]))
        s0 = materialize(leaf(1:k; arithmetic=ar), terms, θ; constant=c); V0 = eigenvectors(s0)
        P0 = Dict(key => port(s0, key...) for key in keys_)
        checkports(s0)
        Uperm = Matrix{eltype(V0)}(I, k, k)[:, perm]; Uph = Matrix(Diagonal(phs))
        # V -> V U with U a permutation (the selection), a diagonal of phases (:follow from a phased previous),
        # and both; every port must transform as U' P U
        runs = [(materialize(leaf(perm; arithmetic=ar), terms, θ; constant=c), Uperm),
                (materialize(leaf(1:k; phase=:follow, arithmetic=ar), terms, θ; constant=c, previous=phased(s0, Uph)), Uph),
                (materialize(leaf(perm; phase=:follow, arithmetic=ar), terms, θ; constant=c, previous=phased(s0, Uperm * Uph)), Uperm * Uph)]
        for (sys, U) in runs
            @test arithmetic(sys) === (ar === :complex ? :complex : :real)
            @test eigenvectors(sys) ≈ V0 * U atol=1e-10
            for key in keys_
                @test port(sys, key...) ≈ U' * P0[key] * U atol=1e-10
            end
            checkports(sys)
        end
    end
    # at a branch: ports formed from the children's ports, a selection port at the root, ancestor ports
    # (expiphi(±1) on varphi_1) required from the leaf after its slicing
    terms2 = circuit_terms(load_spec(fixture("two_transmons.json"))); θ2 = two_transmons_params(; g=0.3)
    sel2 = (:varphi_1, nport()); req2 = [(:varphi_1, expiphi(1)), (:varphi_1, expiphi(-1)), (:varphi_2, nport())]
    top(perm; kw...) = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(5)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(5))];
                            keep=_PermutedSelection(perm, sel2), ports=req2, kw...)
    # the two-transmon branch is real by structure and its phases are signs: complex phases on a branch forced complex
    # (over the real leaves), signs on the real one
    for (ar, phs) in ((:complex, ph), (:auto, Float64[1, -1, -1, 1, -1]))
        t0 = materialize(top(1:k; arithmetic=ar), terms2, θ2)
        @test arithmetic(t0) === (ar === :complex ? :complex : :real)
        T0 = Dict(key => port(t0, key...) for key in vcat([sel2], req2))
        Ψ0 = _primitive(t0.root)
        Id = Matrix{ComplexF64}(I, 17, 17)
        prim2 = Dict(sel2 => kron(Id, dense(localop(ChargeBasis(8), nport()))),
                     req2[1] => kron(Id, dense(localop(ChargeBasis(8), expiphi(1)))),
                     req2[2] => kron(Id, dense(localop(ChargeBasis(8), expiphi(-1)))),
                     req2[3] => kron(dense(localop(ChargeBasis(8), nport())), Id))
        for (key, O) in prim2
            @test T0[key] ≈ Ψ0' * O * Ψ0 atol=1e-11              # the primitive operators on explicit primitive states
        end
        @test T0[req2[1]] ≈ T0[req2[2]]' atol=1e-12
        U = Matrix{eltype(eigenvectors(t0))}(I, k, k)[:, perm] * Diagonal(phs)
        t1 = materialize(top(perm; phase=:follow, arithmetic=ar), terms2, θ2; previous=phased(t0, U))
        @test arithmetic(t1) === arithmetic(t0)
        @test eigenvectors(t1) ≈ eigenvectors(t0) * U atol=1e-10
        for key in vcat([sel2], req2)
            @test port(t1, key...) ≈ U' * T0[key] * U atol=1e-10
        end
        @test port(t1, :varphi_1, expiphi(1)) ≈ port(t1, :varphi_1, expiphi(-1))' atol=1e-12
        @test port(t1, :varphi_2, nport()) ≈ port(t1, :varphi_2, nport())' atol=1e-12
    end
end

@testset "T4: phase-anchor cancellation" begin
    using KronFreeCircuits: _peak_denominator, _HYBRID_RATIO
    # A three-point child whose kept states s1, s2 both peak at point 1 (|0.8| and |0.6| > 0.4√2); s3 is
    # the antisymmetric (0, 1, -1)/√2.  Its peak-amplitude matrix: rows are the peak points (1, 1, 2).
    S = [0.8 0.6 0.0; sqrt(0.18) -0.4sqrt(2) 1/sqrt(2); sqrt(0.18) -0.4sqrt(2) -1/sqrt(2)]
    @test S' * S ≈ I atol=1e-15
    A = ComplexF64.(S[[1, 1, 2], :]); I2 = Matrix{ComplexF64}(I, 2, 2)
    full(C) = kron(I2, S) * vec(C)                                # the state on the (3 points) × (2 basis states) space
    anchor(comp) = (ci = CartesianIndices((3, 2))[comp]; LinearIndices((3, 2))[[1, 1, 2][ci[1]], ci[2]])
    # exact cancellation: the state (0, 1, 1)/√2 at basis state 1 has coefficients (0.6, -0.8, 0); the dominant
    # component is s2, whose peak point carries no amplitude
    C = zeros(ComplexF64, 3, 2); C[1, 1] = 0.6; C[2, 1] = -0.8
    comp, α = branch_peak_phase(C, [A, I2])
    @test comp == 2 && abs(α) < 1e-15 && abs(full(C)[anchor(comp)]) < 1e-15
    @test _peak_denominator(C, [A, I2], comp) ≈ 0.48 && abs(α) / 0.48 < _HYBRID_RATIO
    @test all(isfinite, C .* cis(-angle(α)))                     # finite: no unique phase is claimed at a zero anchor
    # near cancellation: a small amplitude still fixes the phase, at the explicit anchor
    Cn = copy(C); Cn[1, 1] += 1e-7; Cn ./= norm(Cn)
    comp, α = branch_peak_phase(Cn, [A, I2])
    @test comp == 2 && 0 < abs(α) < 1e-6 && α ≈ full(Cn)[anchor(comp)]
    @test real(α * cis(-angle(α))) ≈ abs(α) && abs(imag(α * cis(-angle(α)))) < 1e-20
    # tied dominant coefficients: s1 and s2 share their peak point, so either choice gives the same anchor amplitude
    Ct = zeros(ComplexF64, 3, 2); Ct[1, 1] = 1 / sqrt(2); Ct[2, 1] = -1 / sqrt(2)
    comp, α = branch_peak_phase(Ct, [A, I2])
    @test comp in (1, 2) && α ≈ full(Ct)[anchor(1)] ≈ full(Ct)[anchor(2)] ≈ (0.8 - 0.6) / sqrt(2)
    # empty and zero-weight masks fall back to the global argmax
    @test branch_peak_phase(C, [A, I2]; mask=falses(6)) == branch_peak_phase(C, [A, I2])
    zw = falses(3, 2); zw[3, :] .= true                          # only components where C vanishes
    @test branch_peak_phase(C, [A, I2]; mask=vec(zw)) == branch_peak_phase(C, [A, I2])
    # an arbitrary global phase: α rotates with it, the phase-fixed state does not (α ≠ 0)
    χ = cis(2.1)
    for Cx in (Cn, Ct)
        c1, α1 = branch_peak_phase(Cx, [A, I2]); c2, α2 = branch_peak_phase(χ .* Cx, [A, I2])
        @test c1 == c2 && α2 ≈ χ * α1
        @test χ .* Cx .* cis(-angle(α2)) ≈ Cx .* cis(-angle(α1))
    end
    @test all(isfinite, branch_peak_phase(χ .* C, [A, I2])[2])   # at the zero anchor: finite, nothing more
    # the leaf rule: global phase removed, and an empty mask falls back (with its warning) to the largest entry
    Random.seed!(84)
    V = randn(ComplexF64, 5, 2); V1 = fix_phases!(copy(V), :positive_sector, trues(5), nothing)
    @test fix_phases!(χ .* V, :positive_sector, trues(5), nothing) ≈ V1
    Ve = @test_logs (:warn, r"column 1 has no weight") (:warn, r"column 2 has no weight") fix_phases!(copy(V), :positive_sector, falses(5), nothing)
    @test all(isfinite, Ve) && all(j -> (i = argmax(abs.(Ve[:, j])); abs(imag(Ve[i, j])) < 1e-14 && real(Ve[i, j]) > 0), 1:2)

    # Through materialize.  The child is a node over a given three-point system with H_a = S diag(0, 1, 2) S';
    # the parent couples it to a given two-state system so that its ground state is exactly
    # ψ ⊗ |b1⟩, ψ = (ε, 1, 1)/‖·‖, whose dominant component (s2, b1) is anchored at point 1, where ψ = ε.
    Ha = S * Diagonal([0.0, 1.0, 2.0]) * S'
    function tree_terms(ε)
        ψ = [ε, 1.0, 1.0]; ψ ./= norm(ψ)
        ga = GivenSystem(:a; energies=zeros(3), operators=Dict(:h => Ha, :m => -ψ * ψ' - Ha))
        gb = GivenSystem(:b; energies=[0.0, 10.0], operators=Dict(:p1 => [1.0 0.0; 0.0 0.0]))
        child = Node([ga]; keep=lowest(3))
        parent = Node([child, gb]; keep=lowest(1))
        terms = [Term(Coefficient(1), [Factor(:a, Port(:h, 0))]),                       # routed to the child
                 Term(Coefficient(1), [Factor(:a, Port(:m, 0)), Factor(:b, Port(:p1, 0))])]   # in |b1⟩: H = -ψψ'
        return parent, terms, ψ
    end
    for ε in (0.0, 1e-3)
        parent, terms, ψ = tree_terms(ε)
        sys = @test_logs (:warn, r"heavy hybridization") materialize(parent, terms, (;))
        mn = sys.root; ch = mn.children[1]
        @test energies(sys) ≈ [-1.0] atol=1e-12
        @test all(isfinite, mn.vectors) && all(isfinite, mn.peak_amplitudes) && all(isfinite, ch.peak_amplitudes)
        Ψ = kron(I2, ch.vectors) * mn.vectors                      # explicit full-space state
        @test abs(dot(kron([1.0, 0.0], ψ), Ψ)) ≈ 1 atol=1e-12
        comp = mn.peak_component[1]
        @test comp == 2 && ch.peak_component[2] == 1             # anchored at s2's peak point
        @test Ψ[1] ≈ mn.peak_amplitudes[1, 1] atol=1e-14         # the anchor amplitude, physically
        if ε == 0
            @test abs(Ψ[1]) < 1e-14                              # exact cancellation (to rounding): phase not unique
        else
            @test abs(Ψ[1]) ≈ ε / norm([ε, 1, 1]) rtol=1e-8      # near cancellation: real, positive anchor amplitude
            @test real(Ψ[1]) > 0 && abs(imag(Ψ[1])) < 1e-14
        end
    end
    # a grandparent whose dominant component is that state: its own peak amplitude vanishes
    parent, terms, _ = tree_terms(0.0)
    sysg = @test_logs (:warn, r"heavy hybridization") (:warn, r"peak amplitude vanishes") materialize(Node([parent]; keep=lowest(1)), terms, (;))
    @test all(isfinite, sysg.root.vectors) && abs(abs(sysg.root.vectors[1]) - 1) < 1e-14
    # a branch whose kept state has no weight on its basis child's positive sector (charge -1 only)
    ga = GivenSystem(:a; energies=zeros(3), operators=Dict(:h => Ha))
    child = Node([ga]; keep=lowest(3)); br = Node([child, :q => ChargeBasis(1)]; keep=lowest(1))
    tq = [Term(Coefficient(1), [Factor(:a, Port(:h, 0))]), Term(Coefficient(1), [Factor(:q, n2port())]),
          Term(Coefficient(3), [Factor(:q, nport())])]                                  # q energies (-2, 0, 4)
    sb = @test_logs (:warn, r"kept state 1 has no weight on the phase mask") materialize(br, tq, (;))
    @test abs2(sb.root.vectors[1]) ≈ 1 && all(isfinite, sb.root.vectors) && sb.root.peak_component == [1]
    # a leaf with an empty positive sector (a grid entirely at negative phase)
    tg = [Term(Coefficient(1), [Factor(:x, n2port())]), Term(Coefficient(1600), [Factor(:x, phi2port())]),
          Term(Coefficient(6400), [Factor(:x, phiport())])]                             # n² + 1600 (φ + 2)² - 6400
    sl = @test_logs (:warn, r"column 1 has no weight") (:warn, r"column 2 has no weight") materialize(Node([:x => PhaseGrid(-3, -1, 81)]; keep=lowest(2)), tg, (;))
    Vl = eigenvectors(sl)
    @test all(isfinite, Vl) && all(j -> (i = argmax(abs.(Vl[:, j])); abs(imag(Vl[i, j])) < 1e-14 && real(Vl[i, j]) > 0), 1:2)
end

if get(ENV, "KRONFREE_SLOW_TESTS", "0") == "1"      # gated: about 6 s; the default suite checks a real branch against the complex path in test_materialize.jl
@testset "real branch: two coupled zero-pis equal the complex path" begin
    using KronFreeCircuits: arithmetic, follow, _initial_block
    tz = load_spec(fixture("two_zero_pi.json")); θ = two_zero_pi_params(g=0.05)
    mk(ar) = (q1 = Node([:theta1 => ChargeBasis(4), :phi1 => PhaseGrid(-6π, 6π, 61; order=6)]; solver=Dense(), keep=lowest(10), arithmetic=ar);
              q2 = Node([:theta2 => ChargeBasis(4), :phi2 => PhaseGrid(-6π, 6π, 61; order=6)]; solver=Dense(), keep=lowest(10), arithmetic=ar);
              Node([q1, q2]; solver=Dense(), keep=lowest(12), ports=[(:theta1, nport())], arithmetic=ar))
    tr, tc = mk(:auto), mk(:complex)
    sr = @test_logs (:warn, r"edge") match_mode=:any materialize(tr, tz, θ)
    sc = @test_logs (:warn, r"edge") match_mode=:any materialize(tc, tz, θ)
    @test arithmetic(sr) === :real && eltype(eigenvectors(sr)) === Float64
    @test energies(sr) ≈ energies(sc) atol=1e-11
    Pr, Pc = port(sr, :theta1, nport()), port(sc, :theta1, nport())
    @test eltype(Pr) === Float64
    # The exchange pairs of the two identical zero-pis are split by only 6e-8 to 3e-6, so the two paths' Dense solves rotate
    # within a pair by round-off over the gap (about 2e-8, as between any two equally valid solves; abs.(Pr) and abs.(Pc)
    # differ by 2e-9 in norm): compare ports through the physical overlap O of the two paths' kept states, Pr = O Pc O'.
    clusters(E) = (b = [0; findall(>(1e-4), diff(E)); length(E)]; [b[i]+1:b[i+1] for i in 1:length(b)-1])
    cs = clusters(energies(sc))
    @test 6 <= length(cs) < 12                                                            # several clusters, some pairs
    # LOBPCG at the branch (Auto's branch solver) in real arithmetic equals Dense
    tl = mk(:auto); tl.solver = LOBPCG(); tl.compute = 12
    sl = @test_logs (:warn, r"edge") match_mode=:any materialize(tl, tz, θ)
    @test arithmetic(sl) === :real
    @test energies(sl) ≈ energies(sr) atol=1e-8
    # the root's physical states agree with the complex path's up to per-state signs (the branch peak argmax may tie
    # between exchange-symmetric components, so signs are not compared) and the in-pair rotation, which enters the
    # diagonal overlaps only at second order
    Vr = eigenvectors(sr)
    # physical overlaps of two roots' kept states, through the leaves' kept-state overlaps (axis 1 fastest)
    phys(a, b) = (S = [a.root.children[i].vectors' * b.root.children[i].vectors for i in 1:2];
                  a.root.vectors' * kron(S[2], S[1]) * b.root.vectors)
    offcluster(O) = norm(O - cat((O[c, c] for c in cs)...; dims=(1, 2)))
    O = phys(sr, sc)
    @test Pr ≈ O * Pc * O' atol=1e-10
    @test abs.(diag(O)) ≈ ones(12) atol=1e-12
    @test offcluster(O) < 1e-10
    # the operator rebuilt for the real root is real, and the stored eigenpairs are its eigenpairs
    Hr = KronFreeCircuits.operator(sr)
    @test eltype(Hr) === Float64 && eltype(sr.root.peak_amplitudes) === Float64
    check_eigenpairs(Hr, energies(sr), Vr; tol=1e-9)
    # the branch :positive_sector phase on the real root: real vectors whose peak amplitude is positive
    amps = [c.peak_amplitudes for c in sr.root.children]
    for j in axes(Vr, 2)
        comp, α = branch_peak_phase(reshape(Vr[:, j], sr.root.dims...), amps)
        @test comp == sr.root.peak_component[j] && imag(α) == 0 && real(α) > 0
        @test sr.root.peak_amplitudes[j, j] ≈ real(α) atol=1e-14
    end
    # Review Focus 2: a parent over a child forced complex is complex (no term needs to touch it), with sc's energies
    q1c = Node([:theta1 => ChargeBasis(4), :phi1 => PhaseGrid(-6π, 6π, 61; order=6)]; solver=Dense(), keep=lowest(10), arithmetic=:complex)
    q2r = Node([:theta2 => ChargeBasis(4), :phi2 => PhaseGrid(-6π, 6π, 61; order=6)]; solver=Dense(), keep=lowest(10))
    tm = Node([q1c, q2r]; solver=Dense(), keep=lowest(12), ports=[(:theta1, nport())])
    sm = @test_logs (:warn, r"edge") match_mode=:any materialize(tm, tz, θ)
    @test arithmetic(sm, q1c) === :complex && arithmetic(sm, q2r) === :real && arithmetic(sm) === :complex
    @test energies(sm) ≈ energies(sc) atol=1e-11
    Om = phys(sm, sc)
    @test port(sm, :theta1, nport()) ≈ Om * Pc * Om' atol=1e-10
    # Review Focus 4: a previous solution from the other arithmetic, at g = 0.06 with follow() on the root
    θ′ = two_zero_pi_params(g=0.06)
    fol(ar) = (t = mk(ar); t.keep = follow(); t.phase = :follow; t)                       # as Node(...; keep=follow()) sets them
    matprev(t, prev) = @test_logs (:warn, r"edge") match_mode=:any materialize(t, tz, θ′; previous=prev)
    trf, tcf = fol(:auto), fol(:complex)
    rr, rc, cr, cc = matprev(trf, sr), matprev(trf, sc), matprev(tcf, sr), matprev(tcf, sc)
    @test arithmetic(rr) === arithmetic(rc) === :real && arithmetic(cr) === arithmetic(cc) === :complex
    @test eltype(eigenvectors(rc)) === Float64 && eltype(port(rc, :theta1, nport())) === Float64
    @test energies(rc) ≈ energies(rr) atol=1e-11
    @test energies(cr) ≈ energies(cc) atol=1e-11
    @test energies(rr) ≈ energies(cc) atol=1e-11
    # :follow fixes each state's sign (or phase) by the previous physical state, so states that followed the same previous
    # solution agree across arithmetics with their signs, whichever signs that solution's branch phase rule picked
    @test diag(phys(rc, cc)) ≈ ones(12) atol=1e-12
    @test diag(phys(rr, cr)) ≈ ones(12) atol=1e-12
    Orc = phys(rc, cc)
    @test port(rc, :theta1, nport()) ≈ Orc * port(cc, :theta1, nport()) * Orc' atol=1e-10
    # LOBPCG at the branch warm-started from the other arithmetic: complex into real dropped, real into complex converted
    @test _initial_block(sc.root, sc.root.dims, 12, Float64) === nothing
    @test _initial_block(sr.root, sr.root.dims, 12, ComplexF64) isa Matrix{ComplexF64}
    lob(ar) = (t = mk(ar); t.solver = LOBPCG(tol=1e-11); t.compute = 12; t)
    lr, lc = matprev(lob(:auto), sc), matprev(lob(:complex), sr)
    @test arithmetic(lr) === :real && arithmetic(lc) === :complex
    @test energies(lr) ≈ energies(cc) atol=1e-8
    @test energies(lc) ≈ energies(cc) atol=1e-8
end
end
