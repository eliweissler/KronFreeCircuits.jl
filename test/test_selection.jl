using KronFreeCircuits: SelectionContext, select, lowest, indices, connected, follow, ports_needed,
                        validate_selection, fix_phases!, nport

@testset "selection rules" begin
    P = zeros(ComplexF64, 6, 6); P[1,4] = P[4,1] = 0.5; P[4,6] = P[6,4] = 0.2; P[2,3] = P[3,2] = 0.9
    ctx = SelectionContext(collect(1.0:6), Matrix{ComplexF64}(I, 6, 6), Dict((:m, nport()) => P), nothing)
    @test select(connected([1, 2], (:m, nport()), 0.1), ctx) == [1, 2, 3, 4, 6]
    @test ports_needed(connected([1], (:m, nport()), 0.1)) == [(:m, nport())]
    @test select(lowest(3), ctx) == [1, 2, 3] && select(lowest(0), ctx) == 1:6
    @test_throws ArgumentError validate_selection([1, 1], 6, lowest(2))     # Review Focus 5
    @test_throws ArgumentError validate_selection(Int[], 6, lowest(0))
    @test_throws ArgumentError validate_selection([7], 6, indices([7]))
    V = Matrix{ComplexF64}(I, 6, 6)[:, [3, 1, 2, 4, 5, 6]]                   # permuted eigenvectors
    @test select(follow(Matrix{ComplexF64}(I, 6, 6)[:, 1:3]), SelectionContext(ctx.energies, V, ctx.ports, nothing)) == [2, 3, 1]
    @test_logs (:warn,) select(follow(Matrix{ComplexF64}(I, 6, 6)[:, 1:1]), SelectionContext(ctx.energies, V .* 0.1 .+ 0.05, ctx.ports, nothing))
end
@testset "fix_phases!" begin
    v = [0.3, -0.8im, 0.5]; mask = [false, true, true]
    W = fix_phases!(reshape(copy(v), :, 1), :positive_sector, mask, nothing)
    @test W[2, 1] ≈ 0.8 && abs(W[1, 1]) ≈ 0.3                                # largest in positive sector real positive
    r = reshape(exp(0.4im) .* v, :, 1)
    @test fix_phases!(reshape(copy(v), :, 1), :follow, mask, r) ≈ r
end
