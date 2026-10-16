using KronFreeCircuits: Node, Dense, Lanczos, lowest, modes, isleaf, axis_of, route, required_ports,
                        PhaseGrid, ChargeBasis, FockBasis, circuit_terms, load_spec, nport, phiport

@testset "route and required_ports" begin
    g = PhaseGrid(-6π, 6π, 50); c = ChargeBasis(4)
    q1 = Node([:theta1 => c, :phi1 => g]); q2 = Node([:theta2 => c, :phi2 => g])
    top = Node([q1, q2])
    @test modes(top) == [:theta1, :phi1, :theta2, :phi2] && axis_of(top, :phi2) == 2
    ts = circuit_terms(load_spec(fixture("two_zero_pi.json")))
    r = route(top, ts)
    @test length(r[top]) == 1 && Set(modes(r[top][1])) == Set([:theta1, :theta2])
    @test all(t -> issubset(modes(t), [:theta1, :phi1]), r[q1])
    rp = required_ports(top, r)
    @test rp[q1] == [(:theta1, nport())] && rp[q2] == [(:theta2, nport())] && isempty(rp[top])
    @test_throws ArgumentError route(Node([:theta1 => c]), ts)
end

@testset "Node validation" begin
    c = ChargeBasis(4)
    @test isleaf(Node([:a => c])) && !isleaf(Node([Node([:a => c])]))
    @test_throws ArgumentError Node([:a => c, :a => c])
    @test_throws ArgumentError Node([Node([:a => c]), Node([:a => c])])
    @test_throws ArgumentError Node([:a => c]; phase=:bogus)
    @test_throws ArgumentError Node([:a => c]; solver=Lanczos())
    @test_throws ArgumentError Node([:a => c, 3])
    @test_throws ArgumentError axis_of(Node([:a => c]), :zz)
    @test Node([:a => c]; solver=Lanczos(), compute=3).compute == 3
end

@testset "multi-level routing and ports" begin
    using KronFreeCircuits: Coefficient, Term, Factor, Port, phi2port, n2port
    c = ChargeBasis(2)
    la = Node([:a => c]; ports=[(:a, phiport())]); lb = Node([:b => c])
    inner = Node([la, lb]); root = Node([inner, :c => c])
    @test modes(root) == [:a, :b, :c] && axis_of(root, :c) == 2 && axis_of(root, :a) == 1
    mk(fs...) = Term(Coefficient(1.0), Factor[Factor(m, p) for (m, p) in fs])
    T1 = mk((:a, nport()), (:b, nport()))
    T2 = mk((:a, phiport()), (:c, nport()))
    T3 = mk((:a, n2port()))
    T4 = mk()
    T5 = mk((:a, phiport()), (:c, n2port()))      # same (a, phi) as T2
    r = route(root, [T1, T2, T3, T4, T5])
    @test r[inner] == [T1] && r[root] == [T2, T4, T5] && r[la] == [T3] && isempty(r[lb])
    rp = required_ports(root, r)
    @test rp[la] == [(:a, phiport()), (:a, nport())]   # T2/T5 (deduped), T1, then own port already present
    @test rp[lb] == [(:b, nport())]
    @test rp[inner] == [(:a, phiport())]
    @test isempty(rp[root])
    # own ports appended after inferred ones
    la2 = Node([:a => c]; ports=[(:a, n2port())])
    root2 = Node([Node([la2, lb]), :c => c])
    r2 = route(root2, [T1, T2])
    @test required_ports(root2, r2)[la2] == [(:a, phiport()), (:a, nport()), (:a, n2port())]
    # mode in tree but in no term
    bad = Node([inner, :c => c, :d => c])
    err = try route(bad, [T1, T2]); nothing catch e e end
    @test err isa ArgumentError && occursin(":d", err.msg)
end
