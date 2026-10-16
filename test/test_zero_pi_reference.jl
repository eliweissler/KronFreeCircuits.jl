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
        # 30 kept zero-pi states on the ±6π grid reach its edge (cf. the zero-pi leaf test); the branch
        # also warns of small joint-peak amplitudes
        sysz = @test_logs (:warn, r"PhaseGrid on mode :phi") match_mode=:any materialize(full, load_spec(fixture("full_zero_pi.json")), θz)
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
