using KronFreeCircuits: JLD2

@testset "positions and one node" begin
    c = dc_chain3(); sys = materialize(c.root, c.spec, c.θ)
    p = tempname() * ".jld2"; save_system(p, sys)
    @test system_nodes(p) == [Int[], [1], [1, 1], [1, 2], [2]]
    mn = sys.root.children[1].children[2]
    n = load_node(p, [1, 2])
    @test n.energies == mn.energies && n.vectors == mn.vectors && n.kept == mn.kept && n.dims == mn.dims
    @test n.ports == mn.ports
    @test load_node(p).on == "Array"
    @test_throws ArgumentError load_node(p, [3])
    @test_throws ArgumentError load_node(p, [1, 1, 1])
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
