using KronFreeCircuits: materialized, Node, Dense, lowest, nport, ChargeBasis, load_spec, materialize,
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
    @testset "depth=2 labels" begin
        q1, q2 = materialized(sys, top.children[1]), materialized(sys, top.children[2])
        dom(q, i) = (argmax(abs2.(view(q.vectors, :, i))),)
        L = bare_labels(sys, top; depth=2)
        for (i, j) in [(1, 1), (2, 1), (1, 3), (5, 7), (8, 8)]
            @test L[LinearIndices((8, 8))[i, j]] == (dom(q1, i), dom(q2, j))
        end
        @test length(unique(dom(q1, i) for i in 1:8)) >= 4   # charge-basis dominants are not all distinct (several states peak at the same n)
        @test issorted(abs2.(last.(d)); rev=true) && all(abs2.(last.(d)) .> 1e-3)
        @test_throws ArgumentError decompose(sys, top, 21)
    end
    @testset "dressed_index warning" begin
        @test_logs dressed_index(sys0, top, (1, 1))
        sysg = materialize(top, spec, two_transmons_params(g=1.5))
        B = bare_overlaps(sysg, top)
        ps = [p for p in 1:64 if 1e-12 < sum(B[:, p]) && maximum(B[:, p]) < 0.5]
        @test !isempty(ps)
        label = Tuple(CartesianIndices((8, 8))[first(ps)])
        @test_logs (:warn, r"overlap") dressed_index(sysg, top, label)
    end
end
