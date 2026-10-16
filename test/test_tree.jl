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
