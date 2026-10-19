using KronFreeCircuits: LocalOp, DenseOp, DiagonalOp, BandedOp, ShiftOp, dimension, diagonal, dense, realpart, imagpart

@testset "LocalOp" begin
    n = 7
    ops = [DenseOp(randn(n, n)), DiagonalOp(randn(ComplexF64, n)),
           BandedOp(n, [-2, 0, 1], [randn(ComplexF64, n) for _ in 1:3]), ShiftOp(n, 2, 0.5im)]
    for op in ops
        @test dimension(op) == n
        @test dense(op') == dense(op)'
        @test diagonal(op) ≈ diag(dense(op))
        @test eltype(dense(op)) == eltype(op)
    end
    @test eltype(DenseOp(randn(3, 3)).M) == Float64               # real input stays real
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

@testset "element-type-generic local operators" begin
    rng = MersenneTwister(301)
    A = randn(rng, 5, 5)
    @test DenseOp(A).M isa Matrix{Float64} && eltype(DenseOp(A)) === Float64
    @test eltype(DenseOp(complex.(A))) === ComplexF64
    for op in (DenseOp(randn(rng, ComplexF64, 5, 5)), DiagonalOp(randn(rng, ComplexF64, 5)),
               BandedOp(5, [-1, 2], [randn(rng, ComplexF64, 5), randn(rng, ComplexF64, 5)]), ShiftOp(5, 2, 0.3 - 0.7im))
        @test eltype(realpart(op)) === Float64 && eltype(imagpart(op)) === Float64
        @test dense(realpart(op)) + im * dense(imagpart(op)) == dense(op)
        @test typeof(realpart(op)).name === typeof(op).name          # same structure type
    end
    @test eltype(dense(BandedOp(4, [1], [ones(4)]))) === Float64 && eltype(diagonal(DiagonalOp(ones(3)))) === Float64
end
