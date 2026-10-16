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
@testset "follow sweep over n_g" begin
    using Logging
    using KronFreeCircuits: follow
    spec = load_spec(fixture("transmon.json")); k = 4
    θ(ng) = (E_J=30.0, E_C=1.2, n_g=ng)
    leaf = Node([:varphi_1 => ChargeBasis(30)]; solver=Dense(), keep=follow(; minoverlap=0.9))
    @test leaf.phase === :follow
    # Seed: the lowest k states at n_g = 0 from a lowest(k) node in the same tree position; every
    # sweep point then follows the previous one through follow() with no explicit reference.
    prev = materialize(Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(k)), spec, θ(0.0))
    for ng in range(0.0, 0.4; length=5)
        sys = @test_logs min_level=Logging.Warn materialize(leaf, spec, θ(ng); previous=prev)
        Vp, V = eigenvectors(prev), eigenvectors(sys)
        @test size(V, 2) == k
        O = [dot(Vp[:, j], V[:, j]) for j in 1:k]
        @test all(real.(O) .> 0.9) && all(abs.(imag.(O)) .< 1e-12)          # phase continuity
        @test energies(sys) ≈ sort(energies(sys))
        prev = sys
    end
    # first point with an explicit reference and no previous: :positive_sector fallback, no error
    R = eigenvectors(materialize(Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(k)), spec, θ(0.0)))
    sys0 = materialize(Node([:varphi_1 => ChargeBasis(30)]; keep=follow(R)), spec, θ(0.0))
    @test abs.(diag(R' * eigenvectors(sys0))) ≈ ones(k)
end
@testset "branch materialization" begin
    using KronFreeCircuits: FockBasis, modes, constant_term
    θ = two_transmons_params(g=0.3); spec = load_spec(fixture("two_transmons.json"))
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20), ports=[(:varphi_1, nport())])
    sys = materialize(top, spec, θ)
    @testset "branch equals explicit product-basis construction" begin
        q1, q2 = materialized(sys, top.children[1]), materialized(sys, top.children[2])
        Hprod = kron(I(8), Diagonal(q1.energies)) + kron(Diagonal(q2.energies), I(8)) +
                0.3 * kron(q2.ports[(:varphi_2, nport())], q1.ports[(:varphi_1, nport())])
        # The linear pieces of g (n1 - n_g1)(n2 - n_g2) are single-mode terms routed to the leaves; the
        # constant 4 n_gᵀ E_C n_g = 4·1.2·0.25² is added at the root only, so it is not in the leaf energies.
        c = real(constant_term(spec)(θ)); @test c ≈ 0.3
        @test energies(sys) ≈ eigvals(Hermitian(Hprod))[1:20] .+ c atol=1e-10
        @test size(port(sys, :varphi_1, nport())) == (20, 20)
    end
    @testset "spec-form port and connected" begin
        using KronFreeCircuits: connected
        @test port(sys, :n, :varphi_1) == port(sys, :varphi_1, nport())
        @test port(sys, :n, :varphi_2, top.children[2]) == port(sys, :varphi_2, nport(), top.children[2])
        @test_throws ArgumentError port(sys, :n2, :varphi_1)                     # not formed
        @test_throws ArgumentError port(sys, :n, :varphi_1; k=1.0)
        kw = Node([mk(1), mk(2)]; keep=connected(; core=[1], port=(:n, :varphi_1), thresh=0.05))
        pos = Node([mk(1), mk(2)]; keep=connected([1], (:varphi_1, nport()), 0.05))
        skw, spos = materialize(kw, spec, θ), materialize(pos, spec, θ)
        @test materialized(skw, kw).kept == materialized(spos, pos).kept && length(materialized(skw, kw).kept) > 1
        @test energies(skw) ≈ energies(spos)
        @test port(skw, :n, :varphi_1) ≈ port(spos, :varphi_1, nport())
    end
    @testset "two transmons vs scqubits HilbertSpace" begin
        ref = reference("two_transmons").energies
        # scqubits truncates each *bare* transmon to 8 levels and adds g (n1 - n_g1)(n2 - n_g2) in that
        # truncated space.  Here the linear piece -g n_g1 n2 is routed to the varphi_2 leaf and diagonalized
        # before truncation, so the 8 kept transmon-2 states span a different subspace: the two composites
        # are different truncations of one Hamiltonian (observed max difference 2.1e-4, state 17; the lowest
        # three states agree to 2.5e-9).  scqubits' own construction, built from bare materialized leaves,
        # reproduces the reference exactly (observed 6e-13).
        @test energies(sys)[1:3] ≈ ref[1:3] atol=1e-8
        @test energies(sys) ≈ ref atol=5e-4
        spec_t = load_spec(fixture("transmon.json"))
        bare(E_J, E_C, n_g) = materialize(Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(8), ports=[(:varphi_1, nport())]),
                                          spec_t, (E_J=E_J, E_C=E_C, n_g=n_g))
        b1, b2 = bare(θ.E_J1, θ.E_C1, θ.n_g1), bare(θ.E_J2, θ.E_C2, θ.n_g2)
        n1, n2 = port(b1, :varphi_1, nport()), port(b2, :varphi_1, nport())
        Hsc = kron(I(8), Diagonal(energies(b1))) + kron(Diagonal(energies(b2)), I(8)) +
              θ.g * kron(n2 - θ.n_g2 * I, n1 - θ.n_g1 * I)
        @test eigvals(Hermitian(Hsc))[1:20] ≈ ref atol=1e-10
    end
    @testset "lanczos at the top matches dense" begin
        top_l = Node([mk(1), mk(2)]; solver=Lanczos(), compute=6, keep=lowest(6))
        @test energies(materialize(top_l, spec, θ)) ≈ energies(sys)[1:6] atol=1e-8
    end
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
    @testset "three-level tree" begin
        spec3 = load_spec(fixture("three_transmons.json")); θ3 = three_transmons_params(g=0.3)
        inner = Node([mk(1), mk(2)]; keep=lowest(12))
        third = Node([:varphi_3 => ChargeBasis(10)]; keep=lowest(4))
        outer = Node([inner, third]; keep=lowest(10))
        sys3 = materialize(outer, spec3, θ3)
        Ein = materialized(sys3, inner).energies; Eth = materialized(sys3, third).energies
        # The constant 4 n_gᵀ E_C n_g is added at the root only: 0.3 for the two-transmon top,
        # 0.3 + 4·0.8·0.1² for this outer node; the inner node carries none.
        c2, c3 = real(constant_term(spec)(θ)), real(constant_term(spec3)(θ3)); @test c3 ≈ 0.332
        @test energies(sys3) ≈ sort(vec(Ein .+ Eth'))[1:10] .+ c3 atol=1e-10   # uncoupled third mode: energies add
        @test Ein .+ c2 ≈ energies(sys)[1:12] atol=1e-10                       # inner node equals the two-transmon top
    end
end
@testset "term-list entry point" begin
    using KronFreeCircuits: circuit_terms, constant_term, modes, Coefficient, System
    θ = two_transmons_params(g=0.3); spec = load_spec(fixture("two_transmons.json"))
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    top = Node([mk(1), mk(2)]; keep=lowest(20))
    ts = filter(t -> !(:varphi_1 in modes(t) && :varphi_2 in modes(t)), circuit_terms(spec))
    @test length(ts) < length(circuit_terms(spec))
    c = constant_term(spec)
    sys = materialize(top, ts, θ; constant=c)
    @test sys isa System && sys.spec === nothing && sys.terms == ts
    E1, E2 = energies(sys, top.children[1]), energies(sys, top.children[2])
    @test energies(sys) ≈ sort(vec(E1 .+ E2'))[1:20] .+ real(c(θ)) atol=1e-10   # uncoupled: energies add
    @test energies(materialize(top, ts, θ)) ≈ energies(sys) .- real(c(θ)) atol=1e-10   # default constant 0
    s2 = materialize(top, spec, θ)
    @test s2.spec === spec && length(s2.terms) == length(circuit_terms(spec))
end
@testset "basis must match the mode type" begin
    spec = load_spec(fixture("transmon.json")); θ = (E_J=30.0, E_C=1.2, n_g=0.25)
    leaf = Node([:varphi_1 => PhaseGrid(-π, π, 41)])
    err = try materialize(leaf, spec, θ); nothing catch e e end
    @test err isa ArgumentError && occursin(":varphi_1", err.msg) && occursin(":c", err.msg) &&
          occursin("PhaseGrid", err.msg)
    using KronFreeCircuits: circuit_terms, FockBasis
    ts = circuit_terms(spec)
    @test_throws ArgumentError materialize(leaf, ts, θ; modetypes=Dict(:varphi_1 => :c))
    @test_throws ArgumentError materialize(Node([:varphi_1 => ChargeBasis(5)]), ts, θ; modetypes=Dict(:varphi_1 => :e))
    @test_throws ArgumentError materialize(Node([:varphi_1 => ChargeBasis(5)]), ts, θ; modetypes=Dict(:varphi_1 => :q))
    @test materialize(Node([:varphi_1 => ChargeBasis(5)]; keep=lowest(2)), ts, θ; modetypes=Dict(:varphi_1 => :c)) isa KronFreeCircuits.System
    flux = load_spec(fixture("fluxonium.json"))
    @test_throws ArgumentError materialize(Node([:varphi_1 => ChargeBasis(5)]), flux, (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π))
end
@testset "composite ports are rejected" begin
    json = """
    {"format": "kronfree-spec", "version": 1, "units": "GHz",
     "parameters": ["E_C", "E_J"],
     "modes": [{"name": "a", "type": "c"}, {"name": "b", "type": "c"}, {"name": "c", "type": "c"}],
     "E_C": [["E_C", "0", "0"], ["0", "E_C", "0"], ["0", "0", "E_C"]],
     "E_L": [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]],
     "flux_linear": ["0", "0", "0"], "n_g": ["0", "0", "0"],
     "cosines": [{"energy": "E_J", "omega": [1, 1, 1], "offset": "0"}]}
    """
    spec = KronFreeCircuits.parse_spec(json)
    tree = Node([Node([:a => ChargeBasis(3), :b => ChargeBasis(3)]), Node([:c => ChargeBasis(3)])])
    err = try materialize(tree, spec, (E_C=1.0, E_J=1.0)); nothing catch e e end
    @test err isa ArgumentError
    @test occursin("[:a, :b]", err.msg) && occursin("[:a, :b, :c]", err.msg) &&
          occursin("composite (multi-mode) ports are not yet supported", err.msg)
end
@testset "previous with a changed child kept count" begin
    θ = two_transmons_params(g=0.3); spec = load_spec(fixture("two_transmons.json"))
    l1 = Node([:varphi_1 => ChargeBasis(10)]; keep=lowest(4)); l2 = Node([:varphi_2 => ChargeBasis(10)]; keep=lowest(4))
    top = Node([l1, l2]; keep=lowest(6))
    sys0 = materialize(top, spec, θ)
    l1.keep = lowest(5)                                           # the child's kept count changes
    sys1 = materialize(top, spec, θ; previous=sys0)                # previous unused at the top: accepted
    @test materialized(sys1, top).dims == [5, 4]
    topf = Node([l1, l2]; keep=lowest(6), phase=:follow)
    l1.keep = lowest(4); sys0f = materialize(topf, spec, θ)
    l1.keep = lowest(5)
    @test_throws ArgumentError materialize(topf, spec, θ; previous=sys0f)   # :follow consumes previous
end
