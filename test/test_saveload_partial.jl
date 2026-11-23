using KronFreeCircuits: GivenSystem, Term, Factor, Coefficient

@testset "positions and one node" begin
    c = dc_chain3(); sys = materialize(c.root, c.spec, c.θ)
    p = tempname() * ".jld2"; save_system(p, sys)
    @test system_nodes(p) == [Int[], [1], [1, 1], [1, 2], [2]]
    mn = sys.root.children[1].children[2]
    n = load_node(p, [1, 2])
    @test n.energies == mn.energies && n.vectors == mn.vectors && n.kept == mn.kept && n.dims == mn.dims
    @test n.ports == mn.ports
    @test load_node(p).solver == "Dense" && load_node(p).on == "Array"   # Dense runs on the CPU
    @test_throws "valid positions" load_node(p, [3])
    @test_throws "valid positions" load_node(p, [1, 1, 1])
end

@testset "metadata" begin
    c = dc_chain3(); sys = materialize(c.root, c.spec, c.θ)
    p = tempname() * ".jld2"
    save_system(p, sys; metadata=(note="fig 3", grid=[1, 2, 3], flux=0.25))
    @test load_metadata(p) == Dict("note" => "fig 3", "grid" => [1, 2, 3], "flux" => 0.25)
    save_system(p, sys; metadata=Dict("r" => 1:3))
    @test load_metadata(p) == Dict("r" => [1, 2, 3])
    save_system(p, sys)
    @test isempty(load_metadata(p))
end

@testset "unstorable metadata" begin
    c = dc_chain3(); sys = materialize(c.root, c.spec, c.θ)
    p2 = tempname() * ".jld2"
    @test_throws "metadata" save_system(p2, sys; metadata=(f=sin,))
    @test !isfile(p2) && !isfile(p2 * ".partial")
end

@testset "given node and bad metadata keys" begin
    E1 = [0.0, 1.0, 2.5]; X1 = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
    g1 = GivenSystem(:a; energies=E1, operators=Dict(:x => X1))
    g2 = GivenSystem(:b; energies=[0.0, 0.8], operators=Dict(:x => ComplexF64[0 1; 1 0]))
    sys = materialize(Node([g1, g2]; solver=Dense(), keep=lowest(6), ports=[(:a, Port(:x, 0))]),
                      [Term(Coefficient(0.1), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])], (;))
    p = tempname() * ".jld2"; save_system(p, sys)
    n = load_node(p, [1])
    @test n.energies == sys.root.children[1].energies && n.solver === nothing && n.on === nothing
    c = dc_two_transmons(solver=Lanczos(), compute=10); lan = materialize(c.root, c.spec, c.θ)
    save_system(p, lan)
    @test load_node(p).solver == "Lanczos" && load_node(p).on == "Array"
    for k in ("", "a/b")
        @test_throws "metadata" save_system(p, sys; metadata=Dict(k => 1))
    end
end
