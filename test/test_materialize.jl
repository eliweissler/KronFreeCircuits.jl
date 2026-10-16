using KronFreeCircuits: Node, Dense, Lanczos, lowest, nport, PhaseGrid, ChargeBasis, load_spec,
                        materialize, energies, port, eigenvectors, materialized, MaterializedNode

@testset "fluxonium leaf vs scqubits" begin
    ref = reference("fluxonium"); θ = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π)
    leaf = Node([:varphi_1 => PhaseGrid(-10π, 10π, 1201; order=8)]; solver=Dense(), keep=lowest(6))
    sys = materialize(leaf, load_spec(fixture("fluxonium.json")), θ)
    @test energies(sys) ≈ ref.energies atol=1e-6
    @test_throws ArgumentError materialize(leaf, load_spec(fixture("fluxonium.json")), (E_J=8.9,))   # Review Focus 1
end
@testset "transmon leaf vs scqubits (same discretization)" begin
    ref = reference("transmon"); θ = (E_J=30.0, E_C=1.2, n_g=0.25)
    sys = materialize(Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(5)), load_spec(fixture("transmon.json")), θ)
    @test energies(sys) ≈ ref.energies atol=1e-10
end
@testset "zero-pi leaf vs scqubits (Lanczos on a 9k-dimensional leaf)" begin
    ref = reference("zero_pi"); θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    leaf = Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-6π, 6π, 301; order=8)];
                solver=Lanczos(), compute=8, keep=lowest(8), ports=[(:theta, nport())])
    # The φ range ±6π (scqubits' and this test's) is too small for these parameters: the states reach
    # the grid edge (edge support ~2e-3, hence the warning), so the result depends on the discretization
    # at the 3e-4 level (observed max error 3.0e-4, state 7).  scqubits 4.3.1 uses its default 7-point
    # stencil on 1001 points; on that exact discretization (PhaseGrid order 6, N = 1001) the agreement is 3e-12.
    sys = @test_logs (:warn, r"PhaseGrid on mode :phi") materialize(leaf, spec_zp, θ)
    @test energies(sys) ≈ ref.energies atol=5e-4
    @test port(sys, :theta, nport()) ≈ port(sys, :theta, nport())'
    small = Node([:theta => ChargeBasis(8), :phi => PhaseGrid(-6π, 6π, 151)]; keep=lowest(8))
    smalll = Node([:theta => ChargeBasis(8), :phi => PhaseGrid(-6π, 6π, 151)]; solver=Lanczos(), compute=8, keep=lowest(8))
    Ed = energies(@test_logs (:warn, r"PhaseGrid") materialize(small, spec_zp, θ))
    El = energies(@test_logs (:warn, r"PhaseGrid") materialize(smalll, spec_zp, θ))
    @test El ≈ Ed atol=1e-8   # Lanczos = Dense
end
@testset "requested port and edge warning" begin
    θ = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π); spec = load_spec(fixture("fluxonium.json"))
    sys = materialize(Node([:varphi_1 => PhaseGrid(-10π, 10π, 401)]; keep=lowest(6), ports=[(:varphi_1, nport())]), spec, θ)
    P = port(sys, :varphi_1, nport()); @test size(P) == (6, 6) && P ≈ P'
    @test_logs (:warn,) materialize(Node([:varphi_1 => PhaseGrid(-2, 2, 41)]; keep=lowest(6)), spec, θ)   # too-small grid warns
end
@testset "materialized accessors" begin
    θ = (E_J=30.0, E_C=1.2, n_g=0.25); leaf = Node([:varphi_1 => ChargeBasis(10)]; keep=lowest(3))
    sys = materialize(leaf, load_spec(fixture("transmon.json")), θ)
    mn = materialized(sys, leaf)
    @test mn isa MaterializedNode && mn === sys.root && mn.dims == [21] && mn.kept == 1:3
    V = eigenvectors(sys, leaf)
    @test size(V) == (21, 3) && V' * V ≈ I
    @test_throws ArgumentError port(sys, :varphi_1, nport())                  # not formed: lists available ports
    @test_throws ArgumentError materialized(sys, Node([:varphi_1 => ChargeBasis(10)]))   # not in this tree
end
@testset "phase :follow with a previous solution" begin
    spec = load_spec(fixture("transmon.json"))
    leaf = Node([:varphi_1 => ChargeBasis(10)]; keep=lowest(3), phase=:follow, ports=[(:varphi_1, nport())])
    sys0 = materialize(leaf, spec, (E_J=30.0, E_C=1.2, n_g=0.25))                  # no previous: positive sector
    sys1 = materialize(leaf, spec, (E_J=30.0, E_C=1.2, n_g=0.26); previous=sys0)
    O = diag(eigenvectors(sys0, leaf)' * eigenvectors(sys1, leaf))
    @test all(abs.(imag.(O)) .< 1e-12) && all(real.(O) .> 0.9)
    V = eigenvectors(sys1, leaf)
    @test port(sys1, :varphi_1, nport()) ≈ V' * Diagonal(-10.0:10.0) * V          # ports carry the fixed phases
end
