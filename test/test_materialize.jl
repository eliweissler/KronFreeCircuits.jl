using Random
using KronFreeCircuits: Node, Dense, Lanczos, LOBPCG, lowest, nport, PhaseGrid, ChargeBasis, load_spec,
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
@testset "zero-pi leaf vs scqubits, identical discretization (7-point stencil, 101 points)" begin
    ref = reference("zero_pi_small"); θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    leaf = Node([:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(8), ports=[(:theta, nport())])
    sys = @test_logs (:warn, r"PhaseGrid on mode :phi is too small") match_mode=:any materialize(leaf, spec_zp, θ)
    @test maximum(abs.(energies(sys) .- ref.energies)) < 1e-9
    @test port(sys, :theta, nport()) ≈ port(sys, :theta, nport())'
    lan = Node([:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Lanczos(tol=1e-12), compute=8, keep=lowest(8))
    @test energies(@test_logs (:warn, r"PhaseGrid on mode :phi is too small") match_mode=:any materialize(lan, spec_zp, θ)) ≈ energies(sys) atol=1e-9   # Lanczos = Dense
end
@testset "full zero-pi vs scqubits, identical discretization" begin
    using KronFreeCircuits: modes
    ref = reference("full_zero_pi_small"); θ = full_zero_pi_params()
    zp = Node([:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(30))
    zeta = Node([:zeta => FockBasis(20; phi_zpf=(2θ.E_Cz/θ.E_L)^(1/4))]; keep=lowest(20))
    full = Node([zp, zeta]; solver=Dense(), keep=lowest(10))
    sys = @test_logs (:warn, r"PhaseGrid on mode :phi is too small") match_mode=:any materialize(full, load_spec(fixture("full_zero_pi.json")), θ)
    @test maximum(abs.(energies(sys) .- ref.energies)) < 1e-9
    @test modes(full) == [:theta, :phi, :zeta]
end
@testset "requested port and edge warning" begin
    θ = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π); spec = load_spec(fixture("fluxonium.json"))
    sys = materialize(Node([:varphi_1 => PhaseGrid(-10π, 10π, 401)]; keep=lowest(6), ports=[(:varphi_1, nport())]), spec, θ)
    P = port(sys, :varphi_1, nport()); @test size(P) == (6, 6) && P ≈ P'
    @test_logs (:warn,) materialize(Node([:varphi_1 => PhaseGrid(-2, 2, 41)]; keep=lowest(6)), spec, θ)   # too-small grid warns
end
@testset "selection ports are rotated to the fixed phases" begin
    using KronFreeCircuits: connected, localop, dense
    b = ChargeBasis(10); spec = load_spec(fixture("transmon.json")); θ = (E_J=30.0, E_C=1.2, n_g=0.25)
    leaf = Node([:varphi_1 => b]; keep=connected([1], (:varphi_1, nport()), 0.3))
    sys = materialize(leaf, spec, θ)
    V = eigenvectors(sys); P = port(sys, :varphi_1, nport())
    @test size(V, 2) > 1
    @test P ≈ V' * dense(localop(b, nport())) * V atol=1e-12
    @test P ≈ P'
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
    using KronFreeCircuits: follow
    spec = load_spec(fixture("transmon.json")); k = 4
    θ(ng) = (E_J=30.0, E_C=1.2, n_g=ng)
    leaf = Node([:varphi_1 => ChargeBasis(30)]; solver=Dense(), keep=follow(; minoverlap=0.9))
    @test leaf.phase === :follow
    # Seed: the lowest k states at n_g = 0 from a lowest(k) node in the same tree position; every
    # sweep point then follows the previous one through follow() with no explicit reference.
    prev = materialize(Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(k)), spec, θ(0.0))
    for ng in range(0.0, 0.4; length=5)
        sys = @test_logs min_level=Base.CoreLogging.Warn materialize(leaf, spec, θ(ng); previous=prev)
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
        # the 64 kept states include high, strongly interfering ones (diagnostic ratio down to 0.14)
        skw = @test_logs (:warn, r"Branch node") match_mode=:any materialize(kw, spec, θ)
        spos = @test_logs (:warn, r"Branch node") match_mode=:any materialize(pos, spec, θ)
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

using KronFreeCircuits: positive_mask

@testset "branch positive-sector phase at the dominant joint peak" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(g=0.3)
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    q1n, q2n = mk(1), mk(2); top = Node([q1n, q2n]; keep=lowest(20), ports=[(:varphi_1, nport())])
    sys = materialize(top, spec, θ); q1, q2, t = materialized(sys, q1n), materialized(sys, q2n), materialized(sys, top)
    V = t.vectors; mask1 = positive_mask(ChargeBasis(30))
    for j in 1:20
        ψ = kron(q2.vectors, q1.vectors) * V[:, j]                                # brute-force full-space oracle, axis 1 fastest
        a, b = Tuple(CartesianIndices((8, 8))[t.peak_component[j]])
        p = LinearIndices((61, 61))[q1.peak_component[a], q2.peak_component[b]]
        @test abs(imag(ψ[p])) < 1e-12 && real(ψ[p]) > 0                            # real positive at the joint peak
        @test ψ[p] ≈ t.peak_amplitudes[j, j]
        @test mask1[q1.peak_component[a]] && mask1[q2.peak_component[b]]           # the joint peak lies in the positive sector
        @test (a, b) == Tuple(argmax(abs.(reshape(V[:, j], 8, 8))))               # dominant component
    end
    # leaf peak amplitudes equal the vectors at the peaks
    @test q1.peak_amplitudes == q1.vectors[q1.peak_component, :]
    # ports and energies unchanged in modulus relative to a run with the old rule are not testable directly; check Hermiticity and spectrum
    @test port(sys, :varphi_1, nport()) ≈ port(sys, :varphi_1, nport())'
    # three-level tree: peak amplitudes of the outer node agree with a brute-force expansion through two levels
    spec3 = load_spec(fixture("three_transmons.json")); θ3 = three_transmons_params(g=0.3)
    inner = Node([mk(1), mk(2)]; keep=lowest(12)); third = Node([:varphi_3 => ChargeBasis(10)]; keep=lowest(4))
    outer = Node([inner, third]; keep=lowest(10)); sys3 = materialize(outer, spec3, θ3)
    o, inn, th = materialized(sys3, outer), materialized(sys3, inner), materialized(sys3, third)
    l1, l2 = materialized(sys3, inner.children[1]), materialized(sys3, inner.children[2])
    Vinner_full = kron(l2.vectors, l1.vectors) * inn.vectors                       # inner kept states in the leaf basis
    for j in 1:10
        ψ = kron(th.vectors, Vinner_full) * o.vectors[:, j]
        ai, c = Tuple(CartesianIndices((12, 4))[o.peak_component[j]])
        a, b = Tuple(CartesianIndices((8, 8))[inn.peak_component[ai]])
        p = LinearIndices((61, 61, 21))[l1.peak_component[a], l2.peak_component[b], th.peak_component[c]]
        @test abs(imag(ψ[p])) < 1e-12 && real(ψ[p]) > 0
        @test ψ[p] ≈ o.peak_amplitudes[j, j]
    end
end

@testset "mixed branch (node child and basis child): joint peak through the basis index" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(g=0.3)
    q1n = Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(8))
    top = Node([q1n, :varphi_2 => ChargeBasis(10)]; keep=lowest(10))
    sys = materialize(top, spec, θ); q1, t = materialized(sys, q1n), materialized(sys, top)
    W = kron(Matrix{ComplexF64}(I, 21, 21), q1.vectors)                           # top product basis -> full space
    function point(q)                                                              # a basis child's peak point is its index
        a, b = Tuple(CartesianIndices((8, 21))[t.peak_component[q]])
        return LinearIndices((61, 21))[q1.peak_component[a], b]
    end
    pts = [point(q) for q in 1:10]
    for j in 1:10
        ψ = W * t.vectors[:, j]
        @test abs(imag(ψ[pts[j]])) < 1e-12 && real(ψ[pts[j]]) > 0
        @test ψ[pts] ≈ t.peak_amplitudes[:, j]                                     # every row of the peak-amplitude matrix
    end
end

@testset "mixed branch with a PhaseGrid child: dominant component in the basis's positive sector" begin
    # a transmon coupled through g n_1 n_2 to a symmetric extended mode on a phase grid
    src = """
    {"format": "kronfree-spec", "version": 1, "units": "GHz",
     "parameters": ["E_J1", "E_C1", "n_g1", "E_J2", "E_C2", "E_L", "g"],
     "modes": [{"name": "varphi_1", "type": "c"}, {"name": "varphi_2", "type": "e"}],
     "E_C": [["E_C1", "g/8"], ["g/8", "E_C2"]], "E_L": [["0", "0"], ["0", "E_L"]],
     "flux_linear": ["0", "0"], "n_g": ["n_g1", "0"],
     "cosines": [{"energy": "E_J1", "omega": [1, 0], "offset": "0"}, {"energy": "E_J2", "omega": [0, 1], "offset": "0"}]}
    """
    θ = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=2.0, E_C2=1.0, E_L=4.0, g=0.3)
    grid = PhaseGrid(-10, 10, 101); q1n = Node([:varphi_1 => ChargeBasis(10)]; keep=lowest(4))
    top = Node([q1n, :varphi_2 => grid]; keep=lowest(8))
    sys = @test_logs min_level=Base.CoreLogging.Warn materialize(top, KronFreeCircuits.parse_spec(src), θ)   # no edge or hybridization warning
    q1, t = materialized(sys, q1n), materialized(sys, top)
    W = kron(Matrix{ComplexF64}(I, 101, 101), q1.vectors)                         # top product basis -> full space
    sector = findall(kron(positive_mask(grid), trues(4)))                          # axis 1 fastest
    for j in 1:8
        ψ = W * t.vectors[:, j]
        a, b = Tuple(CartesianIndices((4, 101))[t.peak_component[j]])
        @test positive_mask(grid)[b]                                               # the symmetric partner at -φ is excluded
        @test t.peak_component[j] == sector[argmax(abs.(t.vectors[sector, j]))]    # the argmax of |c| over the mask
        p = LinearIndices((21, 101))[q1.peak_component[a], b]
        @test abs(imag(ψ[p])) < 1e-12 && real(ψ[p]) > 0
        @test ψ[p] ≈ t.peak_amplitudes[j, j]
    end
end

@testset "LOBPCG at a leaf and at a branch equals Dense" begin
    θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    children = [:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]
    dense = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(children; solver=Dense(), keep=lowest(8)), spec_zp, θ)
    Random.seed!(21)
    lob = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(children; solver=LOBPCG(tol=1e-10), compute=8, keep=lowest(8)), spec_zp, θ)
    @test energies(lob) ≈ energies(dense) atol=1e-7
    # branch: the two-zero-pi small fixture tree used in test_zero_pi_reference.jl, with the top solver swapped
    θ2 = two_zero_pi_params(; g=0.05); spec2 = load_spec(fixture("two_zero_pi.json"))
    leaf(i) = Node([Symbol("theta$i") => ChargeBasis(6), Symbol("phi$i") => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(10))
    top_dense = Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(12))
    top_lob = Node([leaf(1), leaf(2)]; solver=LOBPCG(tol=1e-10), compute=12, keep=lowest(12))
    sd = @test_logs (:warn, r"edge") match_mode=:any materialize(top_dense, spec2, θ2)
    sl = @test_logs (:warn, r"edge") match_mode=:any materialize(top_lob, spec2, θ2)
    @test energies(sl) ≈ energies(sd) atol=1e-7
end

@testset "previous solution warm-starts a LOBPCG node" begin
    # Work is measured in H columns (n_matvec at the final callback), not iterations: the guard columns
    # converge from random on warm and cold starts alike, but once the seeded columns lock the active
    # block is small and each late iteration applies H to only a few columns.
    matvecs = Int[]
    cb = info -> (push!(matvecs, info.n_matvec); info)
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    tree() = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))];
                  solver=LOBPCG(tol=1e-10, callback=cb), compute=6, keep=lowest(6))
    Random.seed!(2)
    s1 = materialize(tree(), spec, θ); cold = last(matvecs); empty!(matvecs)
    s2 = materialize(tree(), spec, two_transmons_params(; g=0.101); previous=s1); warm = last(matvecs)
    @test warm < cold
    @test energies(s2) ≈ energies(materialize(Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))]; solver=Dense(), keep=lowest(6)), spec, two_transmons_params(; g=0.101))) atol=1e-7
end

@testset "warm start: unchanged Hamiltonian, partial block, exact seeds" begin
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    matvecs = Int[]; cb = info -> push!(matvecs, info.n_matvec)   # returns a Vector, not info: must be tolerated
    leaves() = [Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))]
    tree(; keep) = Node(leaves(); solver=LOBPCG(tol=1e-10, callback=cb), compute=8, keep=keep)
    Random.seed!(4)
    s1 = materialize(tree(; keep=lowest(8)), spec, θ); cold = last(matvecs); empty!(matvecs)
    s_same = materialize(tree(; keep=lowest(8)), spec, θ; previous=s1)                 # exact eigenvectors as the seed
    @test energies(s_same) ≈ energies(s1) atol=1e-9
    @test last(matvecs) < cold; empty!(matvecs)                                       # work in H columns
    s_part = materialize(tree(; keep=lowest(3)), spec, θ)                              # compute 8, keep 3: partial warm start
    s_part2 = materialize(tree(; keep=lowest(3)), spec, two_transmons_params(; g=0.101); previous=s_part)
    @test energies(s_part2) ≈ energies(materialize(Node(leaves(); solver=Dense(), keep=lowest(3)), spec, two_transmons_params(; g=0.101))) atol=1e-7
    V = eigenvectors(s1)                                                               # 36 × 8 host matrix
    check_eigenpairs(KronFreeCircuits.operator(s1), energies(s1), V; tol=1e-9)
end

@testset "F1: free rotor through n_g = 1/2, warm-started through materialize" begin
    spec = load_spec(fixture("transmon.json")); EC = 1.2; ncut = 15; N = 2ncut + 1
    nv = collect(-ncut:ncut)
    levels(ng) = sort(4EC .* (nv .- ng) .^ 2)                  # oracle: the sorted diagonal (the root constant included)
    charge(k) = (v = zeros(ComplexF64, N); v[k+ncut+1] = 1; v)
    P01 = charge(0) * charge(0)' + charge(1) * charge(1)'
    leaf(m; solver=LOBPCG(tol=1e-10)) = Node([:varphi_1 => ChargeBasis(ncut)]; solver=solver, compute=m, keep=lowest(m))
    sweep = (0.25, 0.5 - 1e-6, 0.5, 0.5 + 1e-6, 0.75)
    Random.seed!(61)
    # E_J = 0: each sweep point is warm-started from the previous one.  After the crossing the old
    # ground state (charge 0) is still an exact eigenstate; the seed must not trap the solver there.
    for m in (1, 2)                                           # the ground state only, and both crossing states
        prev = nothing
        for ng in sweep
            sys = materialize(leaf(m), spec, (E_J=0.0, E_C=EC, n_g=ng); previous=prev)
            E, V = energies(sys), eigenvectors(sys)
            @test E ≈ levels(ng)[1:m] atol=1e-9
            check_eigenpairs(KronFreeCircuits.operator(sys), E, V; tol=1e-9)
            @test norm(P01 * V - V) < 1e-8                    # inside the crossing pair {0, 1}
            ng == 0.5 || @test abs2(dot(charge(ng < 0.5 ? 0 : 1), V[:, 1])) ≈ 1 atol=1e-8   # the correct lowest state
            m == 2 && @test V * V' ≈ P01 atol=1e-8           # the complete pair (degenerate at n_g = 1/2)
            prev = sys
        end
    end
    # small E_J: an avoided crossing, against Dense on the same basis
    prev = nothing
    for ng in sweep
        θ = (E_J=0.1, E_C=EC, n_g=ng)
        sys = materialize(leaf(2), spec, θ; previous=prev)
        ref = materialize(leaf(2; solver=Dense()), spec, θ)
        @test energies(sys) ≈ energies(ref) atol=1e-8
        V, W = eigenvectors(sys), eigenvectors(ref)
        @test V * V' ≈ W * W' atol=1e-6
        check_eigenpairs(KronFreeCircuits.operator(sys), energies(sys), V; tol=1e-9)
        prev = sys
    end
end

@testset "F2: cos 2φ charge-parity sectors, sector-confined warm start through materialize" begin
    using KronFreeCircuits: indices
    src = """
    {"format": "kronfree-spec", "version": 1, "units": "GHz",
     "parameters": ["E_C", "E_2", "n_g"],
     "modes": [{"name": "varphi_1", "type": "c"}],
     "E_C": [["E_C"]], "E_L": [["0"]], "flux_linear": ["0"], "n_g": ["n_g"],
     "cosines": [{"energy": "E_2", "omega": [2], "offset": "0"}]}
    """
    spec = KronFreeCircuits.parse_spec(src); EC = 1.2; E2 = 3.0; ncut = 20; m = 4
    nv = collect(-ncut:ncut); ev = iseven.(nv)
    function oracle(ng)                                       # independent dense matrix, entry by entry
        A = Matrix{ComplexF64}(Diagonal(4EC .* (nv .- ng) .^ 2))
        for i in eachindex(nv), j in eachindex(nv)
            abs(nv[i] - nv[j]) == 2 && (A[i, j] -= E2 / 2)
        end
        return Hermitian(A)
    end
    θ(ng) = (E_C=EC, E_2=E2, n_g=ng)
    # The previous point keeps four even-parity states only (a Dense node at the same tree position):
    # the warm-start block is confined to the even sector, and the guards must supply the odd one.
    A = oracle(0.2); F = eigen(A)
    evidx = findall(j -> sum(abs2, F.vectors[ev, j]) > 0.5, 1:10)[1:m]
    prev = materialize(Node([:varphi_1 => ChargeBasis(ncut)]; solver=Dense(), compute=10, keep=indices(evidx)), spec, θ(0.2))
    @test all(j -> sum(abs2, eigenvectors(prev)[.!ev, j]) < 1e-20, 1:m)        # the seed is confined
    Random.seed!(62)
    for ng in (0.5, 0.2)
        A = oracle(ng); F = eigen(A)
        Esec = sort([eigvals(Hermitian(A[ev, ev])); eigvals(Hermitian(A[.!ev, .!ev]))])   # sectors separately, merged
        sys = materialize(Node([:varphi_1 => ChargeBasis(ncut)]; solver=LOBPCG(tol=1e-10), compute=m, keep=lowest(m)),
                          spec, θ(ng); previous=prev)
        E, V = energies(sys), eigenvectors(sys)
        @test E ≈ Esec[1:m] atol=1e-8
        @test V * V' ≈ F.vectors[:, 1:m] * F.vectors[:, 1:m]' atol=1e-6
        @test sum(abs2, V[.!ev, :]) > 0.5                     # odd-parity states recovered
        check_eigenpairs(KronFreeCircuits.operator(sys), E, V; tol=1e-9)
    end
end

@testset "F3: identical uncoupled zero-pis (small fixture)" begin
    using KronFreeCircuits: constant_term, indices
    spec2 = load_spec(fixture("two_zero_pi.json"))
    leaf(i) = Node([Symbol("theta$i") => ChargeBasis(6), Symbol("phi$i") => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), compute=10, keep=lowest(10))
    lobtop(m) = Node([leaf(1), leaf(2)]; solver=LOBPCG(tol=1e-10), compute=m, keep=lowest(m))
    mat(tree, θ; kw...) = @test_logs (:warn, r"edge") match_mode=:any materialize(tree, spec2, θ; kw...)
    e(i, j) = (v = zeros(ComplexF64, 100); v[i+10(j-1)] = 1; v)                 # product state (i, j), axis 1 fastest
    θ0 = two_zero_pi_params(; g=0.0)
    Random.seed!(63)
    # g = 0: the parent energies are the sorted pairwise sums; (1,2) and (2,1) are degenerate
    t3 = lobtop(3); s3 = mat(t3, θ0)
    E1 = energies(s3, t3.children[1]); c = real(constant_term(spec2)(θ0))
    @test E1 == energies(s3, t3.children[2])                  # identical leaves
    sums = sort(vec(E1 .+ E1')) .+ c
    E, V = energies(s3), eigenvectors(s3)
    @test E ≈ sums[1:3] atol=1e-8
    @test E[2] ≈ E[3] atol=1e-8                               # the exchange pair
    @test E[3] < sums[4] - 1e-3                               # m = 3 ends on a complete cluster
    @test V * V' ≈ sum(v * v' for v in (e(1, 1), e(1, 2), e(2, 1))) atol=1e-7    # complete clusters
    check_eigenpairs(KronFreeCircuits.operator(s3), E, V; tol=1e-9)
    # a cutoff inside the exchange pair (warm-started from s3: compute decreases from 3 to 2)
    t2 = lobtop(2); s2 = mat(t2, θ0; previous=s3)
    E, V = energies(s2), eigenvectors(s2)
    @test E ≈ sums[1:2] atol=1e-8
    @test abs2(dot(e(1, 1), V[:, 1])) ≈ 1 atol=1e-8           # the complete lower cluster
    P12 = e(1, 2) * e(1, 2)' + e(2, 1) * e(2, 1)'
    @test norm(P12 * V[:, 2] - V[:, 2]) < 1e-7                # boundary state inside the degenerate pair
    check_eigenpairs(KronFreeCircuits.operator(s2), E, V; tol=1e-9)
    # small detuning at g = 0, warm-started across the change: sums of the two leaves' energies
    θd = merge(θ0, (E_J2=θ0.E_J2 * 1.01,))
    td = lobtop(3); sdt = mat(td, θd; previous=s3)
    Ed1, Ed2 = energies(sdt, td.children[1]), energies(sdt, td.children[2])
    @test energies(sdt) ≈ sort(vec(Ed1 .+ Ed2'))[1:3] .+ real(constant_term(spec2)(θd)) atol=1e-8
    check_eigenpairs(KronFreeCircuits.operator(sdt), energies(sdt), eigenvectors(sdt); tol=1e-9)
    # small coupling, seeded from exchange-symmetric states only: the antisymmetric state must come from the guards
    θg = two_zero_pi_params(; g=0.05)
    sd = mat(Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(6)), θg)             # independent Dense parent
    S = zeros(100, 100); for i in 1:10, j in 1:10; S[i+10(j-1), j+10(i-1)] = 1; end   # exchange (i, j) -> (j, i)
    Vd = eigenvectors(sd)
    parity = [real(dot(Vd[:, k], S * Vd[:, k])) for k in 1:6]
    @test all(p -> abs(abs(p) - 1) < 1e-8, parity) && any(<(0), parity[1:3])
    sym = findall(>(0), parity)[1:3]
    psym = mat(Node([leaf(1), leaf(2)]; solver=Dense(), compute=6, keep=indices(sym)), θg)
    tg = lobtop(3); sg = mat(tg, θg; previous=psym)
    E, V = energies(sg), eigenvectors(sg)
    @test E ≈ energies(sd)[1:3] atol=1e-8
    @test V * V' ≈ Vd[:, 1:3] * Vd[:, 1:3]' atol=1e-6
    check_eigenpairs(KronFreeCircuits.operator(sg), E, V; tol=1e-9)
end

@testset "F8: deliberately inadequate hierarchical truncation" begin
    using KronFreeCircuits: constant_term
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(; g=4.0)   # strong coupling, n_g1 ≠ 0
    c = real(constant_term(spec)(θ))
    flat = materialize(Node([:varphi_1 => ChargeBasis(8), :varphi_2 => ChargeBasis(8)]; keep=lowest(4)), spec, θ)   # primitive space
    leaves(k) = [Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(k)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(k))]
    Random.seed!(64)
    errs = Float64[]
    for k in (5, 8)
        tl = Node(leaves(k); solver=LOBPCG(tol=1e-10), compute=4, keep=lowest(4))
        sl = materialize(tl, spec, θ)
        sd = materialize(Node(leaves(k); solver=Dense(), keep=lowest(4)), spec, θ)
        @test energies(sl) ≈ energies(sd) atol=1e-8          # same projected Hamiltonian: no solver error
        q1, q2 = materialized(sl, tl.children[1]), materialized(sl, tl.children[2])
        Hprod = kron(I(k), Diagonal(q1.energies)) + kron(Diagonal(q2.energies), I(k)) +
                θ.g * kron(q2.ports[(:varphi_2, nport())], q1.ports[(:varphi_1, nport())])   # explicit projection
        @test energies(sl) ≈ eigvals(Hermitian(Hprod))[1:4] .+ c atol=1e-8
        check_eigenpairs(KronFreeCircuits.operator(sl), energies(sl), eigenvectors(sl); tol=1e-9)
        push!(errs, maximum(abs.(energies(sl) .- energies(flat))))
    end
    @test errs[1] > 1e-2                                      # retained-subspace error at the inadequate cutoff
    @test errs[2] < errs[1] / 10                              # shrinking as the child cutoffs grow
end

@testset "S9: partial, resized and reordered warm starts" begin
    using KronFreeCircuits: indices, follow, _initial_block
    spec = load_spec(fixture("two_transmons.json"))
    θ1, θ2 = two_transmons_params(; g=0.1), two_transmons_params(; g=0.101)
    leaves() = [Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))]
    tree(compute, keep) = Node(leaves(); solver=LOBPCG(tol=1e-10), compute=compute, keep=keep)
    dense(θ, k) = energies(materialize(Node(leaves(); solver=Dense(), keep=lowest(k)), spec, θ))
    valid(sys) = check_eigenpairs(KronFreeCircuits.operator(sys), energies(sys), eigenvectors(sys); tol=1e-9)
    Random.seed!(65)
    order = [3, 1, 4, 2]
    s_r = materialize(tree(4, indices(order)), spec, θ1)
    @test energies(s_r) ≈ dense(θ1, 4)[order] atol=1e-8
    mn = s_r.root
    # the initial block: a copy of the kept vectors in kept order, padded with random columns or truncated
    X = _initial_block(mn, mn.dims, 4)
    @test X == mn.vectors && X !== mn.vectors
    X = _initial_block(mn, mn.dims, 7)
    @test size(X) == (36, 7) && X[:, 1:4] == mn.vectors && all(j -> norm(X[:, j]) > 1, 5:7) && rank(X) == 7
    @test _initial_block(mn, mn.dims, 2) == mn.vectors[:, 1:2]
    @test _initial_block(mn, [6, 5], 4) === nothing && _initial_block(nothing, mn.dims, 4) === nothing
    # reordered selection: the solver still returns the lowest pairs, the selection reorders them
    s_r2 = materialize(tree(4, indices(order)), spec, θ2; previous=s_r)
    @test energies(s_r2) ≈ dense(θ2, 4)[order] atol=1e-8
    valid(s_r2)
    # follow from the reordered previous: the same order, continuous phases
    s_f = @test_logs min_level=Base.CoreLogging.Warn materialize(tree(4, follow(; minoverlap=0.9)), spec, θ2; previous=s_r)
    @test energies(s_f) ≈ dense(θ2, 4)[order] atol=1e-8
    O = [dot(eigenvectors(s_r)[:, j], eigenvectors(s_f)[:, j]) for j in 1:4]
    @test all(real.(O) .> 0.9) && all(abs.(imag.(O)) .< 1e-12)
    valid(s_f)
    # increased compute (4 kept -> 7: three random columns), then decreased (7 -> 3: truncated)
    s_up = materialize(tree(7, lowest(7)), spec, θ2; previous=s_r)
    @test energies(s_up) ≈ dense(θ2, 7) atol=1e-8
    valid(s_up)
    s_down = materialize(tree(3, lowest(3)), spec, θ1; previous=s_up)
    @test energies(s_down) ≈ dense(θ1, 3) atol=1e-8
    valid(s_down)
    # a previous whose shape differs at this node: no initial block, a cold start, accepted
    small = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(5)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))];
                 solver=LOBPCG(tol=1e-10), compute=4, keep=lowest(4))
    s_c = materialize(small, spec, θ2; previous=s_r)
    @test materialized(s_c, small).dims == [5, 6]
    @test energies(s_c) ≈ energies(materialize(Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(5)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))];
                                                    solver=Dense(), keep=lowest(4)), spec, θ2)) atol=1e-8
    valid(s_c)
end
