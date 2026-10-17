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

@testset "Reserved multiplication workspace" begin
    rng = MersenneTwister(23)
    dims = [3, 4, 2]
    terms = [
        AxisTerm(0.7, [(1, DenseOp(randn(rng, ComplexF64, 3, 3))),
                       (2, BandedOp(4, [-1, 1], [ones(4), ones(4)])),
                       (3, ShiftOp(2, 1, 0.4))]),
        AxisTerm(-0.2, [(2, DiagonalOp(ComplexF64[1, 2, 3, 4]))]),
    ]
    H = KronFreeOperator(dims, terms; constant=0.1, block_capacity=5)
    ref = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms]) + 0.1I
    b1, b2 = H.buf1, H.buf2
    @test size(b1) == (3, 4, 2, 5)
    # Locking shrinks the active block; later applications can reuse any reserved width.
    for width in (5, 2, 1, 4, 0)
        X = randn(rng, ComplexF64, 24, width)
        saved = copy(X)
        Y = fill(ComplexF64(NaN), size(X))
        @test mul!(Y, H, X) === Y
        @test Y ≈ ref * X
        @test X == saved
        @test H.buf1 === b1 && H.buf2 === b2
    end
    x = randn(rng, ComplexF64, 24)
    y = similar(x)
    @test mul!(y, H, x) ≈ ref * x
    @test H.buf1 === b1 && H.buf2 === b2
    @test_throws ArgumentError mul!(x, H, x)
    # A larger request may grow capacity once; shrinking must retain that allocation.
    X = randn(rng, ComplexF64, 24, 7)
    @test mul!(similar(X), H, X) ≈ ref * X
    grown1, grown2 = H.buf1, H.buf2
    @test size(grown1) == (3, 4, 2, 7)
    @test grown1 !== b1 && grown2 !== b2
    saved_y = copy(y)
    mul!(similar(x), H, x)
    @test H.buf1 === grown1 && H.buf2 === grown2
    @test y == saved_y
    @test_throws ArgumentError KronFreeOperator(dims, terms; block_capacity=0)
end

@testset "Active-column views avoid state-sized allocation" begin
    rng = MersenneTwister(24)
    H = KronFreeOperator([32, 32],
        [AxisTerm(0.7, [(1, DiagonalOp(ComplexF64.(1:32)))])];
        block_capacity=6)
    Xparent = randn(rng, ComplexF64, 1024, 5)
    Yparent = fill(ComplexF64(99), 1024, 5)
    X = view(Xparent, :, 2:4)
    Y = view(Yparent, :, 2:4)
    expected = H * copy(X)
    mul!(Y, H, X)
    mul!(Y, H, X)
    # Metadata allocations are allowed; a block-sized temporary or per-element
    # boxing in the output initialization is not.
    @test (@allocated mul!(Y, H, X)) < sizeof(X)
    @test Y ≈ expected
    @test all(==(99), Yparent[:, [1, 5]])
end
