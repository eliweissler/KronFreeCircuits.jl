@testset "two zero-pi vs scqubits HilbertSpace" begin
    θ = two_zero_pi_params(g=0.05)
    mk(i) = Node([Symbol("theta$i") => ChargeBasis(15), Symbol("phi$i") => PhaseGrid(-6π, 6π, 301)]; solver=Lanczos(), compute=10, keep=lowest(10))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20))
    sys = @test_logs (:warn, r"PhaseGrid on mode :phi\d is too small") match_mode=:any materialize(top, load_spec(fixture("two_zero_pi.json")), θ)
    # `≈` with atol compares the 2-norm of the difference over all 20 levels.
    # Observed: max elementwise error 6.8e-5 (passes 1e-4), 2-norm about 2e-4 (misses 1e-4, within 1e-3).
    # Two causes: grid convergence on the shared ±6π range (scqubits' 7-point stencil on 1001 points vs
    # our order-8 stencil on 301 points), and about 1.5e-5 from the truncation order of the routed n_g
    # pieces (see the identical-discretization test below).
    @test energies(sys) ≈ reference("two_zero_pi").energies atol=1e-3
    @test maximum(abs.(energies(sys) .- reference("two_zero_pi").energies)) < 1e-4
    @test length(bare_labels(sys, top)) == 100
end
@testset "two zero-pi vs scqubits HilbertSpace, identical discretization" begin
    # scqubits truncates each *bare* zero-pi to 10 levels and adds g (n_θ1 - n_g1)(n_θ2 - n_g2) in that
    # truncated space.  The tree routes the linear pieces -g n_g2 n_θ1 and -g n_g1 n_θ2 to the leaves and
    # diagonalizes them before truncation, so its 10 kept states per leaf span a different subspace: the
    # two composites are different truncations of one Hamiltonian (observed max difference 1.5e-5, state 20),
    # as for the two transmons in test_materialize.jl.  scqubits' own construction, built from bare
    # materialized leaves on the identical discretization, reproduces the reference (observed 9.6e-13).
    θ = zero_pi_params(); g = 0.05                      # both copies have the same parameters
    leaf = Node([:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(10), ports=[(:theta, nport())])
    b = @test_logs (:warn, r"PhaseGrid on mode :phi is too small") match_mode=:any materialize(leaf, load_spec(fixture("zero_pi.json")), θ)
    D, n = Diagonal(energies(b)), port(b, :theta, nport()) - θ.n_g * I
    Hsc = kron(I(10), D) + kron(D, I(10)) + g * kron(n, n)
    @test maximum(abs.(eigvals(Hermitian(Matrix(Hsc)))[1:20] .- reference("two_zero_pi_small").energies)) < 1e-9
end
@testset "two zero-pi hierarchical tree vs scqubits HilbertSpace, identical discretization, n_g = 0" begin
    # With n_g1 = n_g2 = 0 the coupling g n_θ1 n_θ2 has no linear pieces to route to the leaves, so the
    # tree and scqubits truncate the same bare leaves and the two constructions coincide.
    θ = merge(two_zero_pi_params(g=0.05), (n_g1=0.0, n_g2=0.0))
    mk(i) = Node([Symbol("theta$i") => ChargeBasis(6), Symbol("phi$i") => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(10))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20))
    sys = @test_logs (:warn, r"PhaseGrid on mode :phi\d is too small") match_mode=:any materialize(top, load_spec(fixture("two_zero_pi.json")), θ)
    @test maximum(abs.(energies(sys) .- reference("two_zero_pi_small_ng0").energies)) < 1e-9
end
@testset "convergence (different stencils, shared grid)" begin
    @testset "zero-pi leaf vs scqubits (Lanczos on a 9k-dimensional leaf)" begin
        ref = reference("zero_pi"); θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
        leaf = Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-6π, 6π, 301; order=8)];
                    solver=Lanczos(), compute=8, keep=lowest(8), ports=[(:theta, nport())])
        # The φ range ±6π (scqubits' and this test's) is too small for these parameters: the states reach
        # the grid edge (edge support ~2e-3, hence the warning), so the result depends on the discretization
        # at the 3e-4 level (observed max error 3.0e-4, state 7).  scqubits 4.3.1 uses its default 7-point
        # stencil on 1001 points; on identical discretizations the agreement is ~1e-12 (the identical-discretization tests in test_materialize.jl).
        sys = @test_logs (:warn, r"PhaseGrid on mode :phi") materialize(leaf, spec_zp, θ)
        @test energies(sys) ≈ ref.energies atol=5e-4
        @test port(sys, :theta, nport()) ≈ port(sys, :theta, nport())'
        small = Node([:theta => ChargeBasis(8), :phi => PhaseGrid(-6π, 6π, 151)]; keep=lowest(8))
        smalll = Node([:theta => ChargeBasis(8), :phi => PhaseGrid(-6π, 6π, 151)]; solver=Lanczos(), compute=8, keep=lowest(8))
        Ed = energies(@test_logs (:warn, r"PhaseGrid") materialize(small, spec_zp, θ))
        El = energies(@test_logs (:warn, r"PhaseGrid") materialize(smalll, spec_zp, θ))
        @test El ≈ Ed atol=1e-8   # Lanczos = Dense
    end
    using KronFreeCircuits: modes
    @testset "full zero-pi: all three mode types in a two-level tree vs scqubits FullZeroPi" begin
        ref = reference("full_zero_pi"); θz = full_zero_pi_params()
        zp = Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-6π, 6π, 301; order=8)]; solver=Lanczos(), compute=30, keep=lowest(30))
        zeta = Node([:zeta => FockBasis(20; phi_zpf=(2θz.E_Cz/θz.E_L)^(1/4))]; keep=lowest(20))   # scqubits' oscillator basis for zeta
        full = Node([zp, zeta]; solver=Dense(), keep=lowest(10))
        # 30 kept zero-pi states on the ±6π grid reach its edge (cf. the zero-pi leaf test)
        sysz = @test_logs (:warn, r"PhaseGrid on mode :phi") materialize(full, load_spec(fixture("full_zero_pi.json")), θz)
        @test energies(sysz) ≈ ref.energies atol=1e-4      # same truncations as scqubits (zeropi_cutoff=30, zeta_cutoff=20)
        @test modes(full) == [:theta, :phi, :zeta]
    end
end
@testset "zero-pi leaf, converged on a wide grid" begin
    ref = reference("zero_pi_wide"); θ = zero_pi_params()
    # krylovdim=200: the default 2m + 10 = 26 does not converge in 1000 restarts on this stiff 65641-dimensional leaf.
    leaf = Node([:theta => ChargeBasis(20), :phi => PhaseGrid(-10π, 10π, 1601; order=8)]; solver=Lanczos(tol=1e-12, krylovdim=200), compute=8, keep=lowest(8))
    sys = @test_logs min_level=Base.CoreLogging.Warn materialize(leaf, load_spec(fixture("zero_pi.json")), θ)   # no edge warning expected
    @test maximum(abs.(energies(sys) .- ref.energies)) < 1e-6            # different stencils, both converged (observed 5.9e-11)
end

# ---- Pathological-case catalogue, larger versions (docs/spikes/2026-10-16-pathological-test-catalogue) ----
# Each references an independently assembled primitive operator (`primitive_hamiltonian` in helpers.jl)
# or an analytic oracle, never the apply routines.  Work measurements (H columns, times, failures) are
# collected in CATALOGUE_SLOW for the outcomes record; they are observations, not assertions.

using KronFreeCircuits: circuit_terms, constant_term, lobpcg_solve, solve
CATALOGUE_SLOW = Dict{String,Any}()

# The `k` lowest eigenpairs of a sparse Hermitian matrix by shift-invert Lanczos (KrylovKit on a sparse
# LU factorization), the shift below the Gershgorin lower bound; independent of the apply routines.
function lowest_sparse(H::SparseMatrixCSC, k::Int)
    n = size(H, 1)
    σ = minimum(real(H[i, i]) - (sum(abs, H[:, i]) - abs(H[i, i])) for i in 1:n) - 1
    F = lu(H - σ * I)
    vals, vecs, info = KronFreeCircuits.KrylovKit.eigsolve(x -> F \ x, randn(ComplexF64, n), k, :LR;
                                                          ishermitian=true, tol=1e-13, krylovdim=max(40, 4k))
    info.converged >= k || error("lowest_sparse: only $(info.converged) of $k converged")
    E = σ .+ 1 ./ real.(vals[1:k])
    p = sortperm(E)
    return E[p], reduce(hcat, vecs[1:k])[:, p]
end

@testset "F3 (slow): identical uncoupled zero-pis, larger leaves, several seeds" begin
    spec2 = load_spec(fixture("two_zero_pi.json")); terms, c = circuit_terms(spec2), constant_term(spec2)
    kl = 12; nc = 8; Ng = 151
    leafb(i) = [Symbol("theta$i") => ChargeBasis(nc), Symbol("phi$i") => PhaseGrid(-6π, 6π, Ng; order=8)]
    leaf(i) = Node(leafb(i); solver=Dense(), compute=kl, keep=lowest(kl))
    mat(tree, θ) = @test_logs (:warn, r"edge") match_mode=:any materialize(tree, terms, θ; constant=c)
    e(i, j) = (v = zeros(ComplexF64, kl^2); v[i+kl*(j-1)] = 1; v)               # product state (i, j)
    nθ = kron(Matrix{ComplexF64}(I, Ng, Ng), Diagonal(ComplexF64.(-nc:nc)))       # n_theta on a leaf's primitive space
    function check_leaves(sys, top, θ)
        for i in 1:2
            q = materialized(sys, top.children[i])
            lt = filter(t -> all(m -> endswith(string(m), string(i)), modes(t)), terms)
            Hl = primitive_hamiltonian(lt, θ, leafb(i))
            @test maximum(_residuals(Hl, q.energies, q.vectors)) < 1e-8
        end
    end
    # the projected parent Hamiltonian from the leaf energies and explicitly projected n_theta (not the ports)
    function projected(sys, top, θ)
        q1, q2 = materialized(sys, top.children[1]), materialized(sys, top.children[2])
        N1, N2 = q1.vectors' * nθ * q1.vectors, q2.vectors' * nθ * q2.vectors
        Ik = Matrix{ComplexF64}(I, kl, kl)
        return kron(Ik, Diagonal(q1.energies)) + kron(Diagonal(q2.energies), Ik) + θ.g * kron(N2, N1) + real(c(θ)) * I
    end
    seeds = (101, 102, 103)
    # g = 0: sorted pairwise sums; exchange partners exactly degenerate
    θ0 = two_zero_pi_params(; g=0.0)
    top0 = Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(40))
    s0 = mat(top0, θ0); check_leaves(s0, top0, θ0)
    E1 = materialized(s0, top0.children[1]).energies
    @test E1 ≈ materialized(s0, top0.children[2]).energies atol=1e-12
    pairs = sort([(E1[i] + E1[j] + real(c(θ0)), i, j) for i in 1:kl for j in 1:kl])
    sums = first.(pairs)
    H0 = KronFreeCircuits.operator(s0)
    complete = [m for m in 2:16 if sums[m+1] - sums[m] > 1e-4]                   # cutoffs ending on a complete cluster
    cut = [m for m in 2:16 if sums[m+1] - sums[m] < 1e-10]                        # cutoffs inside an exchange pair
    @test !isempty(complete) && !isempty(cut)
    work = Dict{String,Any}()
    for m in (complete[1], complete[end], cut[1]), sd in seeds
        Random.seed!(sd)
        E, V, res = lobpcg_solve(LOBPCG(tol=1e-10), H0, m)
        work["g=0 m=$m seed=$sd"] = res.n_matvec
        @test E ≈ sums[1:m] atol=1e-8
        check_eigenpairs(H0, E, V; tol=1e-9)
        if m in complete
            @test V * V' ≈ sum(e(p[2], p[3]) * e(p[2], p[3])' for p in pairs[1:m]) atol=1e-7
        else                                                   # the lower complete subspace, and the boundary pair
            @test V[:, 1:m-1] * V[:, 1:m-1]' ≈ sum(e(p[2], p[3]) * e(p[2], p[3])' for p in pairs[1:m-1]) atol=1e-7
            Pb = sum(e(p[2], p[3]) * e(p[2], p[3])' for p in pairs[m:m+1])
            @test norm(Pb * V[:, m] - V[:, m]) < 1e-7
        end
    end
    # product-state seeds in one exchange sector: (i, j) + (j, i) only; the guards supply the rest
    m = complete[end]
    up = unique([(min(p[2], p[3]), max(p[2], p[3])) for p in pairs])             # unordered pairs, in energy order
    X0 = reduce(hcat, [normalize(e(i, j) + e(j, i)) for (i, j) in up[1:m]])       # orthonormal, exchange-symmetric
    Random.seed!(104)
    E, V = solve(LOBPCG(tol=1e-10), H0, m; X0=X0)
    @test E ≈ sums[1:m] atol=1e-8
    check_eigenpairs(H0, E, V; tol=1e-9)
    # small detuning at g = 0, warm-started from the identical-leaf vectors
    θd = merge(θ0, (E_J2=θ0.E_J2 * 1.01,))
    sd_ = mat(Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(40)), θd)
    Ed1, Ed2 = sd_.root.children[1].energies, sd_.root.children[2].energies
    Hd = KronFreeCircuits.operator(sd_)
    Random.seed!(105)
    E, V = solve(LOBPCG(tol=1e-10), Hd, m; X0=eigenvectors(s0)[:, 1:m])
    @test E ≈ sort(vec(Ed1 .+ Ed2'))[1:m] .+ real(c(θd)) atol=1e-8
    check_eigenpairs(Hd, E, V; tol=1e-9)
    # small coupling, seeded from exchange-symmetric eigenstates only, several seeds
    θg = two_zero_pi_params(; g=0.05)
    topg = Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(40))
    sg = mat(topg, θg); check_leaves(sg, topg, θg)
    Hg = KronFreeCircuits.operator(sg); Hpg = projected(sg, topg, θg)
    @test Matrix(Hg) ≈ Hpg atol=1e-10
    Fg = eigen(Hermitian(Hpg))
    @test energies(sg) ≈ Fg.values[1:40] atol=1e-9
    Sx = zeros(kl^2, kl^2); for i in 1:kl, j in 1:kl; Sx[i+kl*(j-1), j+kl*(i-1)] = 1; end
    parity = [real(dot(Fg.vectors[:, j], Sx * Fg.vectors[:, j])) for j in 1:2m]
    @test all(p -> abs(abs(p) - 1) < 1e-8, parity[1:m]) && any(<(0), parity[1:m])
    sym = findall(>(0), parity)[1:m]
    for sd in seeds
        Random.seed!(sd)
        E, V, res = lobpcg_solve(LOBPCG(tol=1e-10), Hg, m; X0=Fg.vectors[:, sym])
        work["g=0.05 symmetric seed=$sd"] = res.n_matvec
        @test E ≈ Fg.values[1:m] atol=1e-8
        @test V * V' ≈ Fg.vectors[:, 1:m] * Fg.vectors[:, 1:m]' atol=1e-6
        @test maximum(_residuals(Hpg, E, V)) < 1e-8
    end
    CATALOGUE_SLOW["F3"] = (complete=complete, cut=cut, work=work)
end

@testset "F6 (slow): symmetric to asymmetric full zero-pi, cold and warm, :ritz and :jacobi" begin
    spec = load_spec(fixture("full_zero_pi.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θa = full_zero_pi_params(); θs = merge(θa, (dE_J=0.0, dC_J=0.0, dC=0.0, dE_L=0.0))
    cb, grid = ChargeBasis(10), PhaseGrid(-8π, 8π, 201; order=8)                 # leaf dimension 4221
    fock = FockBasis(10; phi_zpf=(2θa.E_Cz / θa.E_L)^(1 / 4))
    lb = [:theta => cb, :phi => grid]; prim = vcat(lb, [:zeta => fock])
    leafterms = filter(t -> all(in((:theta, :phi)), modes(t)), terms)
    kz = 10; m = 6
    mv = Int[]; rec = info -> push!(mv, info.n_matvec)
    quietly(f) = Test.collect_test_logs(f)
    # Allowed diagnostics: the grid edge warning and the documented guard-column nonconvergence warning
    # (kept pairs converged, a guard did not); returns the guard warnings for the record.
    function diagnostics(logs)
        @test all(l -> occursin("too small", string(l.message)) || occursin("guard columns did not converge", string(l.message)), logs)
        return join([string(l.message) for l in logs if occursin("guard", string(l.message))], "; ")
    end
    leafnode(pc) = Node(lb; solver=LOBPCG(preconditioner=pc, callback=rec), compute=kz, keep=lowest(kz))
    configs = [("symmetric", θs, nothing), ("dE_J", merge(θs, (dE_J=θa.dE_J,)), "symmetric"),
               ("dC_J", merge(θs, (dC_J=θa.dC_J,)), "symmetric"), ("dC", merge(θs, (dC=θa.dC,)), "symmetric"),
               ("dE_L", merge(θs, (dE_L=θa.dE_L,)), "symmetric"), ("all (large step)", θa, "symmetric"),
               ("all, small step", merge(θa, (n_g=θa.n_g + 1e-3, varphi_ext=θa.varphi_ext + 1e-3)), "all (large step)")]
    leafsys = Dict{String,Any}(); treesys = Dict{String,Any}(); results = Any[]
    for (name, θ, from) in configs
        EC = [real(spec.E_C[i, j](θ)) for i in 1:3, j in 1:3]
        @test isposdef(Symmetric(EC))
        Hl = primitive_hamiltonian(leafterms, θ, lb)
        Fl = name in ("symmetric", "all (large step)") ? eigen(Hermitian(Matrix(Hl)), 1:kz+1) : nothing
        # the leaf alone: cold and warm (previous = the leaf at `from`), :ritz always, :jacobi at two points
        for pc in (name in ("symmetric", "all (large step)") ? (:ritz, :jacobi) : (:ritz,)),
            warm in (from === nothing ? (false,) : (false, true))
            empty!(mv); Random.seed!(106)
            prev = warm ? leafsys[from] : nothing
            t = @elapsed out = try quietly(() -> materialize(leafnode(pc), leafterms, θ; previous=prev)) catch err; err end
            if out isa Exception
                @test out isa ErrorException && occursin("did not converge", out.msg)    # explicit nonconvergence
                push!(results, (name, pc, warm ? "warm" : "cold", "FAILED: " * out.msg, isempty(mv) ? 0 : last(mv), t, ""))
                continue
            end
            logs, s = out; guards = diagnostics(logs)
            E, V = energies(s), eigenvectors(s)
            @test maximum(_residuals(Hl, E, V)) < 1e-7 && norm(V' * V - I) < 1e-10      # tol = 1e-8 per unit vector
            Fl === nothing || (@test E ≈ Fl.values[1:kz] atol=1e-8)
            push!(results, (name, pc, warm ? "warm" : "cold", "converged", last(mv), t, guards))
            pc === :ritz && !warm && (leafsys[name] = s)
        end
        # the tree: the zero-pi leaf (LOBPCG :ritz) beside the raw zeta basis, warm-started from `from`
        zp = Node(lb; solver=LOBPCG(), compute=kz, keep=lowest(kz))
        top = Node([zp, :zeta => fock]; solver=LOBPCG(callback=rec), compute=m, keep=lowest(m),
                   ports=[(:theta, nport()), (:phi, KronFreeCircuits.phiport()), (:zeta, nport())])
        empty!(mv); Random.seed!(107)
        t = @elapsed (logs, s) = quietly(() -> materialize(top, terms, θ; constant=c, previous=from === nothing ? nothing : treesys[from]))
        guards = diagnostics(logs); treesys[name] = s
        z = materialized(s, zp)
        W = kron(Matrix{ComplexF64}(I, 10, 10), z.vectors)
        Hproj = W' * (primitive_hamiltonian(terms, θ, prim; constant=c(θ)) * W)
        @test norm(Hproj - Hproj') < 1e-9
        Hm = Matrix(KronFreeCircuits.operator(s))                     # the operator under test at the root
        @test norm(Hm - Hm') < 1e-9 && norm(Hm - Hproj) < 1e-8
        Fq = eigen(Hermitian(Hproj))
        @test energies(s) ≈ Fq.values[1:m] atol=1e-7
        @test maximum(_residuals(Hproj, energies(s), eigenvectors(s))) < 1e-7
        Ψ = W * eigenvectors(s)
        for (mode, p) in ((:theta, nport()), (:phi, KronFreeCircuits.phiport()), (:zeta, nport()))
            P = port(s, mode, p)
            @test P ≈ P' atol=1e-9
            @test P ≈ Ψ' * primitive_hamiltonian([Term(Coefficient(1), [Factor(mode, p)])], θ, prim) * Ψ atol=1e-8
        end
        push!(results, (name, :ritz, from === nothing ? "tree cold" : "tree warm", "converged", last(mv), t, guards))
    end
    CATALOGUE_SLOW["F6"] = results
end

@testset "F8 (slow): inadequate truncation of a strongly coupled full zero-pi, both cutoffs increased" begin
    spec = load_spec(fixture("full_zero_pi.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θ = merge(full_zero_pi_params(), (dC=0.8, dE_L=0.5))                     # strong theta-zeta and phi-zeta coupling
    EC = [real(spec.E_C[i, j](θ)) for i in 1:3, j in 1:3]; @test isposdef(Symmetric(EC))
    cb, grid = ChargeBasis(6), PhaseGrid(-6π, 6π, 101; order=6)
    fock = FockBasis(8; phi_zpf=(2θ.E_Cz / θ.E_L)^(1 / 4))
    prim = [:theta => cb, :phi => grid, :zeta => fock]; m = 6
    Hfull = primitive_hamiltonian(terms, θ, prim; constant=c(θ))                # 10504-dimensional primitive space
    Random.seed!(108)
    Ep, Vp = lowest_sparse(Hfull, m)
    @test maximum(_residuals(Hfull, Ep, Vp)) < 1e-8
    cutoffs = ((4, 2), (8, 3), (16, 4), (32, 6), (64, 8))
    Eproj = Vector{Vector{Float64}}()
    for (kz, kζ) in cutoffs
        zp = Node([:theta => cb, :phi => grid]; solver=Dense(), compute=kz, keep=lowest(kz))
        ze = Node([:zeta => fock]; keep=lowest(kζ))
        sd = @test_logs (:warn, r"edge") match_mode=:any materialize(Node([zp, ze]; solver=Dense(), keep=lowest(m)), terms, θ; constant=c)
        W = kron(materialized(sd, ze).vectors, materialized(sd, zp).vectors)
        Hproj = W' * (Hfull * W)
        Fq = eigen(Hermitian(Hproj))
        push!(Eproj, Fq.values[1:m])
        @test energies(sd) ≈ Fq.values[1:m] atol=1e-8                         # Dense on the same projection
        if kz * kζ >= 3m + 5
            sl = @test_logs (:warn, r"edge") match_mode=:any materialize(Node([zp, ze]; solver=LOBPCG(tol=1e-9), compute=m, keep=lowest(m)), terms, θ; constant=c)
            @test energies(sl) ≈ Fq.values[1:m] atol=1e-8                     # hierarchical = Dense on the projection
            @test maximum(_residuals(Hproj, energies(sl), eigenvectors(sl))) < 1e-8
        end
    end
    errs = [maximum(E .- Ep) for E in Eproj]
    @test all(E -> all(E .>= Ep .- 1e-8), Eproj)                                 # variational
    @test all(i -> all(Eproj[i+1] .<= Eproj[i] .+ 1e-8), 1:length(Eproj)-1)       # nested retained spaces
    @test errs[1] > 10 * errs[end]
    CATALOGUE_SLOW["F8"] = (cutoffs=cutoffs, errs=errs)
end

@testset "F9 (slow): zero-pi on undersized and changing phase grids" begin
    spec = load_spec(fixture("zero_pi.json")); terms, c = circuit_terms(spec), constant_term(spec)
    θ = zero_pi_params(); ref = reference("zero_pi_wide").energies; k = 8; cb = ChargeBasis(10)
    function grid_run(grid)
        lb = [:theta => cb, :phi => grid]
        logs, sys = Test.collect_test_logs(() -> materialize(Node(lb; solver=Lanczos(tol=1e-12, krylovdim=120), compute=k, keep=lowest(k)), terms, θ; constant=c))
        V = eigenvectors(sys); h = grid.order ÷ 2; X = reshape(V, 21, grid.N, k)
        s = [norm(X[:, [1:h; grid.N-h+1:grid.N], j]) for j in 1:k]                 # edge support, recomputed
        @test any(l -> occursin("too small", string(l.message)), logs) == (maximum(s) > KronFreeCircuits._EDGE_TOL)
        @test all(l -> occursin("too small", string(l.message)), logs)
        Hp = primitive_hamiltonian(terms, θ, lb; constant=c(θ))
        @test maximum(_residuals(Hp, energies(sys), V)) < 1e-8                      # tiny solver residuals at every size
        return (energies=energies(sys), edge=maximum(s), err=maximum(abs.(energies(sys) .- ref)))
    end
    Δ = 12π / 150                                                    # widen at fixed spacing
    wide = [grid_run(PhaseGrid(-L, L, round(Int, 2L / Δ) + 1; order=8)) for L in (6π, 8π, 10π, 12π)]
    @test wide[1].edge > 1e-4
    @test issorted([w.edge for w in wide]; rev=true)
    steps = [maximum(abs.(wide[i+1].energies .- wide[i].energies)) for i in 1:3]
    @test issorted(steps; rev=true)                                  # the interval error shrinks as the edges clear
    fine = [grid_run(PhaseGrid(-10π, 10π, N; order=8)) for N in (201, 401, 801)]   # refine at a fixed interval
    @test issorted([f.err for f in fine]; rev=true)
    CATALOGUE_SLOW["F9"] = (wide=[(w.edge, w.err) for w in wide], steps=steps, fine=[(f.edge, f.err) for f in fine])
end

@testset "ShiftInvert slicing on 4 threads (subprocess)" begin
    # The default suite runs on one thread, where slices = 3 runs the threaded loop serially; this runs
    # it on 4 threads in a fresh process. Gated because the subprocess pays its own compile time (~30–60 s).
    helpers = joinpath(@__DIR__, "helpers.jl")
    code = """
        using Test, LinearAlgebra, SparseArrays, Random, KronFreeCircuits
        using KronFreeCircuits: Dense, solve
        include($(repr(helpers)))
        @testset "threaded slicing" begin
            @test Threads.nthreads() == 4
            Random.seed!(83)
            H = three_oscillators(7)
            Ed, Vd = solve(Dense(), H, 0)
            E, V = solve(ShiftInvert(tol=1e-9, slices=3), H, 40)
            @test E ≈ Ed[1:40] atol=1e-8
            @test V * V' ≈ Vd[:, 1:40] * Vd[:, 1:40]' atol=1e-7
            check_eigenpairs(H, E, V; tol=1e-9)
        end
        println("THREADED-SLICING-OK")
        """
    out = read(`$(Base.julia_cmd()) -t 4 --project=$(Base.active_project()) -e $code`, String)
    @test occursin("THREADED-SLICING-OK", out)
end

# The slow-case measurements, printed once for the outcomes record (observations, not assertions).
println("Catalogue slow-case measurements:")
for k in sort!(collect(keys(CATALOGUE_SLOW)))
    v = CATALOGUE_SLOW[k]
    println("  ", k, ":")
    foreach(r -> println("    ", r), v isa Vector ? v : [v])
end
