using KronFreeCircuits: LocalOp, DenseOp, DiagonalOp, BandedOp, ShiftOp, dimension, diagonal, dense

@testset "LocalOp" begin
    n = 7
    ops = [DenseOp(randn(n, n)), DiagonalOp(randn(ComplexF64, n)),
           BandedOp(n, [-2, 0, 1], [randn(ComplexF64, n) for _ in 1:3]), ShiftOp(n, 2, 0.5im)]
    for op in ops
        @test dimension(op) == n
        @test dense(op') == dense(op)'
        @test diagonal(op) ≈ diag(dense(op))
        @test eltype(dense(op)) == ComplexF64
    end
    @test eltype(DenseOp(randn(3, 3)).M) == ComplexF64            # real input stored complex
    S = dense(ShiftOp(4, 1, 1.0)); @test S[2, 1] == 1 && S[1, 4] == 0  # truncates at edge
    # s = 0 and s < 0, and a banded op without a zero offset
    for s in (0, -2)
        sop = ShiftOp(n, s, 0.5im)
        @test dense(sop') == dense(sop)' && diagonal(sop) == diag(dense(sop))
    end
    bop = BandedOp(n, [-1, 2], [randn(ComplexF64, n) for _ in 1:2])
    @test diagonal(bop) == zeros(ComplexF64, n) && dense(bop') == dense(bop)'
    # constructor validation
    @test_throws DimensionMismatch BandedOp(3, [0], [ones(2)])
    @test_throws DimensionMismatch BandedOp(3, [0], [ones(4)])
    @test_throws DimensionMismatch BandedOp(3, [0, 1], [ones(3)])
    @test_throws DimensionMismatch DenseOp(ones(2, 3))
    @test_throws ArgumentError BandedOp(-1, Int[], Vector{Float64}[])
    @test_throws ArgumentError ShiftOp(-1, 0, 1.0)
end
