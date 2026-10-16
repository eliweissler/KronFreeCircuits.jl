using KronFreeCircuits: Node, Dense, lowest, nport, ChargeBasis, load_spec, materialize,
                        bare_labels, bare_overlaps, decompose, dressed_index

@testset "dressed/bare" begin
    spec = load_spec(fixture("two_transmons.json"))
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20), ports=[(:varphi_1, nport())])
    sys0 = materialize(top, spec, two_transmons_params(g=0.0))
    @test length(bare_labels(sys0, top)) == 64 && bare_labels(sys0, top)[2] == (2, 1)
    @test all(maximum(bare_overlaps(sys0, top); dims=2) .≈ 1)           # uncoupled: exact product states
    @test dressed_index(sys0, top, (1, 1)) == 1
    sys = materialize(top, spec, two_transmons_params(g=0.3))
    d = decompose(sys, top, 1)
    @test d[1].first == (1, 1)
    @test sum(abs2, last.(d)) ≈ 1 atol=1e-3
    @test dressed_index(sys, top, d[1].first) == 1
    @test dressed_index(sys0, top, (8, 8)) === nothing            # uncoupled: product state (8,8) ranks 64th, not among the 20 kept
    @test length(bare_labels(sys, top; depth=2)[1]) == 2 && bare_labels(sys, top; depth=2)[1][1] isa Tuple
    @test size(bare_overlaps(sys, top)) == (20, 64)
    @test_throws ArgumentError dressed_index(sys, top, (9, 1))
    @test_throws ArgumentError dressed_index(sys, top, (1,))
end
