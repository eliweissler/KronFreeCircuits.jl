using Random
using KronFreeCircuits: DenseOp, DiagonalOp, BandedOp, ShiftOp, dense, diagonal,
                        apply_axis!, AxisTerm, KronFreeOperator

@testset "apply_axis! vs kron" begin
    dims = [3, 4, 2]
    rng = MersenneTwister(1)
    mk(d) = [DenseOp(randn(rng, ComplexF64, d, d)), DiagonalOp(randn(rng, ComplexF64, d)),
             BandedOp(d, [-1, 1], [randn(rng, ComplexF64, d) for _ in 1:2]), ShiftOp(d, 1, 2.0)]
    for k in 1:3, op in mk(dims[k])
        ref = kron_reference(dims, [(1.0 + 0im, [(k, dense(op))])])
        X = randn(rng, ComplexF64, dims...); Y = similar(X)
        @test apply_axis!(Y, op, X, k) === Y
        @test vec(Y) ≈ ref * vec(X)
        Xb = randn(rng, ComplexF64, dims..., 5); Yb = similar(Xb)         # block axis
        apply_axis!(Yb, op, Xb, k)
        @test reshape(Yb, :, 5) ≈ ref * reshape(Xb, :, 5)
    end
    # negative shift and a banded op with a zero offset and a wide band, on the middle axis
    for op in (ShiftOp(4, -2, 0.5im), BandedOp(4, [-3, 0, 2], [randn(rng, ComplexF64, 4) for _ in 1:3]))
        ref = kron_reference(dims, [(1.0 + 0im, [(2, dense(op))])])
        Xb = randn(rng, ComplexF64, dims..., 3); Yb = similar(Xb)
        apply_axis!(Yb, op, Xb, 2)
        @test reshape(Yb, :, 3) ≈ ref * reshape(Xb, :, 3)
    end
end

@testset "KronFreeOperator" begin
    dims = [3, 4, 2]
    rng = MersenneTwister(2)
    ops = [DenseOp(randn(rng, ComplexF64, 3, 3)),
           BandedOp(4, [-1, 0, 1], [randn(rng, ComplexF64, 4) for _ in 1:3]),
           ShiftOp(2, 1, 2.0)]
    terms = [AxisTerm(0.5 + 0im, [(1, ops[1]), (3, ops[3])]), AxisTerm(-1.0im, [(2, ops[2])])]
    H = KronFreeOperator(dims, terms; constant=0.25)
    ref = kron_reference(dims, [(0.5 + 0im, [(1, dense(ops[1])), (3, dense(ops[3]))]),
                                (-1.0im, [(2, dense(ops[2]))])]) + 0.25I
    @test size(H) == (24, 24)
    @test eltype(H) == ComplexF64
    @test ishermitian(H)
    x = randn(rng, ComplexF64, 24)
    @test H * x ≈ ref * x
    @test Matrix(H) ≈ ref
    @test diagonal(H) ≈ diag(ref)
    X = randn(rng, ComplexF64, 24, 3)
    @test mul!(zeros(ComplexF64, 24, 3), H, X) ≈ ref * X                             # block
    @test H * x ≈ ref * x                                                            # width change back
    y = fill(ComplexF64(NaN), 24)
    @test mul!(y, H, x) ≈ ref * x                                                    # y overwritten
    # a term touching every axis, and a term with no factors
    H2 = KronFreeOperator(dims, [AxisTerm(2.0 + 0im, [(1, ops[1]), (2, ops[2]), (3, ops[3])]),
                                 AxisTerm(1.5 + 0im, Tuple{Int,KronFreeCircuits.LocalOp}[])])
    ref2 = kron_reference(dims, [(2.0 + 0im, [(1, dense(ops[1])), (2, dense(ops[2])), (3, dense(ops[3]))])]) + 1.5I
    @test Matrix(H2) ≈ ref2
    @test diagonal(H2) ≈ diag(ref2)
    @test_throws DimensionMismatch H * randn(ComplexF64, 23)
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, 24, 2), H, randn(ComplexF64, 24, 3))
end
