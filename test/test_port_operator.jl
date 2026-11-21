using KronFreeCircuits: materialized, Node, Dense, lowest, nport, n2port, expiphi, exp2pin, ChargeBasis, PeriodicGrid,
                        load_spec, materialize, port, port_operator, arithmetic

@testset "port_operator" begin
    spec = load_spec(fixture("two_transmons.json"))
    # real: charge-basis leaves (n_g only shifts their diagonal); complex: periodic-grid leaves
    leaf(i, b) = Node([Symbol("varphi_$i") => b]; keep=lowest(6),
                      ports=[(Symbol("varphi_$i"), q) for q in (b isa ChargeBasis ? (nport(), expiphi(1), exp2pin(0.1)) :
                                                                                 (nport(), expiphi(1)))])
    top(b) = Node([leaf(1, b), leaf(2, b)]; solver=Dense(), keep=lowest(36), ports=[(:varphi_1, nport())])
    tr, tc = top(ChargeBasis(15)), top(PeriodicGrid(32))
    sys_r = materialize(tr, spec, two_transmons_params(g=0.3))
    sys_c = materialize(tc, spec, two_transmons_params(g=0.3))
    @test arithmetic(sys_r) === :real && arithmetic(sys_c) === :complex
    Id = Matrix{Float64}(I, 6, 6)
    dense_of(O) = O * Matrix{eltype(O)}(I, size(O)...)

    for (sys, t) in ((sys_r, tr), (sys_c, tc))
        P1 = port(sys, :varphi_1, nport(), t.children[1])
        P2 = port(sys, :varphi_2, nport(), t.children[2])
        @test dense_of(port_operator(sys, (:varphi_1, nport()))) ≈ kron(Id, P1)          # axis 1 fastest
        @test dense_of(port_operator(sys, (:varphi_2, nport()))) ≈ kron(P2, Id)
        @test dense_of(port_operator(sys, [(:varphi_1, nport()), (:varphi_2, nport())])) ≈ kron(P2, P1)
        @test dense_of(port_operator(sys, [(:varphi_2, nport()), (:varphi_1, nport())])) ≈ kron(P2, P1)  # any order
        for p in (sys === sys_r ? (expiphi(1), exp2pin(0.1)) : (expiphi(1),))
            @test dense_of(port_operator(sys, (:varphi_1, p))) ≈ kron(Id, port(sys, :varphi_1, p, t.children[1]))
        end
    end
    @test eltype(port_operator(sys_r, (:varphi_1, nport()))) == Float64
    @test eltype(port_operator(sys_r, (:varphi_1, exp2pin(0.1)))) == ComplexF64        # complex port on a real node
    @test eltype(port_operator(sys_c, (:varphi_1, nport()))) == ComplexF64

    err = try port_operator(sys_r, (:varphi_1, n2port())); nothing catch e; e end
    @test err isa ArgumentError && occursin("lacks port", err.msg)
    err = try port_operator(sys_r, (:nosuch, nport())); nothing catch e; e end
    @test err isa ArgumentError && occursin("nosuch", err.msg)
end

@testset "QuantumToolbox interface without the extension" begin
    sys = materialize(Node([:varphi_1 => ChargeBasis(10)]; keep=lowest(3)), load_spec(fixture("transmon.json")),
                      (E_J=30.0, E_C=1.2, n_g=0.0))
    err = try KronFreeCircuits.qt_hamiltonian(sys); nothing catch e; e end
    @test err isa MethodError
    @test occursin("load QuantumToolbox", sprint(showerror, err))
end

@testset "port_operator: ports a child did not form" begin
    # three transmons, :varphi_1 and :varphi_2 under one child node: the joint port n1 n2 is not formed there
    spec = load_spec(fixture("three_transmons.json"))
    leaf(i) = Node([Symbol("varphi_$i") => ChargeBasis(10)]; keep=lowest(4))
    mid = Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(8))
    root = Node([mid, leaf(3)]; solver=Dense(), keep=lowest(10))
    sys = materialize(root, spec, three_transmons_params(g=0.3))
    err = try port_operator(sys, [(:varphi_1, nport()), (:varphi_2, nport())]); nothing catch e; e end
    @test err isa ArgumentError
    @test err isa ArgumentError && occursin("available", err.msg) && occursin("varphi_1", err.msg)
    err = try port_operator(sys, (:varphi_1, n2port())); nothing catch e; e end
    @test err isa ArgumentError && occursin("lacks port", err.msg) && occursin("available", err.msg)
end
