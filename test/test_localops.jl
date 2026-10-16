using KronFreeCircuits: LocalOp, DenseOp, DiagonalOp, BandedOp, ShiftOp, dimension, diagonal, dense

@testset "LocalOp" begin
    n = 7
    ops = [DenseOp(randn(n, n)), DiagonalOp(randn(ComplexF64, n)),
           BandedOp(n, [-2, 0, 1], [randn(ComplexF64, n) for _ in 1:3]), ShiftOp(n, 2, 0.5im)]
    for op in ops
        @test dimension(op) == n
        @test dense(op') ≈ dense(op)'
        @test diagonal(op) ≈ diag(dense(op))
        @test eltype(dense(op)) == ComplexF64
    end
    @test eltype(DenseOp(randn(3, 3)).M) == ComplexF64            # real input stored complex
    S = dense(ShiftOp(4, 1, 1.0)); @test S[2, 1] == 1 && S[1, 4] == 0  # truncates at edge
end
