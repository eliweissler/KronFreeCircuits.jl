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

@testset "fused diagonal part" begin
    rng = MersenneTwister(31)
    dims = [3, 4, 2]
    d1 = DiagonalOp(ComplexF64.(1:3)); d2 = DiagonalOp(ComplexF64.(randn(rng, 4))); d3 = DiagonalOp(ComplexF64.(randn(rng, 2)))
    M = DenseOp(randn(rng, ComplexF64, 3, 3)); B = BandedOp(4, [-1, 1], [ones(4), ones(4)])
    terms = [AxisTerm(0.5, [(1, d1)]), AxisTerm(-1.5, [(2, d2), (3, d3)]),    # pure diagonal: fused
             AxisTerm(0.7, [(1, M), (2, B)]), AxisTerm(0.2, [(1, M), (3, d3)])] # not fused
    H = KronFreeOperator(dims, terms; constant=0.3)
    ref = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms]) + 0.3I
    @test length(H.terms) == 2
    @test H.diag ≈ 0.5 .* kron(ones(2), ones(4), d1.d) .- 1.5 .* kron(d3.d, d2.d, ones(3)) .+ 0.3
    @test diagonal(H) ≈ diag(ref)
    x = randn(rng, ComplexF64, 24); @test H * x ≈ ref * x
    X = randn(rng, ComplexF64, 24, 3); @test mul!(similar(X), H, X) ≈ ref * X
    @test Matrix(H) ≈ ref
    Hd = KronFreeOperator(dims, terms[1:2]; constant=0.3)             # diagonal-only operator
    refd = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms[1:2]]) + 0.3I
    @test isempty(Hd.terms) && Hd * x ≈ refd * x && Hd.diag ≈ diag(refd)
    @test eltype(H) == ComplexF64
    @test_throws ArgumentError KronFreeOperator(dims, terms; block_capacity=0)
end

@testset "operator owns adapted factor data" begin
    H = KronFreeOperator([3, 2], [AxisTerm(1.0, [(1, DenseOp(ComplexF64[1 2 0; 0 1 0; 0 0 3]))])]; on=Array)
    @test H.on === Array && H.buf1 isa Array && H.diag isa Vector
    @test H.terms[1].factors[1][3] isa Matrix   # the dense factor's data lives in the operator
end

@testset "O3: extreme shifts and unusual bands" begin
    rng = MersenneTwister(41)
    n = 4
    # Explicit references from the documented index conventions (not from `dense`).
    shift_ref(n, s, f) = ComplexF64[i == j + s ? f : 0 for i in 1:n, j in 1:n]
    band_ref(n, offs, ws) = ComplexF64[sum((ws[b][i] for b in eachindex(offs) if j == i + offs[b]); init=0) for i in 1:n, j in 1:n]
    cases = Any[]
    for (s, f) in ((0, 1.5 - 0.5im), (n, 2.0), (n + 3, 0.3im), (-n, 1.0im), (-n - 2, 0.7), (n - 1, -1.2), (1 - n, 0.4 + 0.1im))
        push!(cases, (ShiftOp(n, s, f), shift_ref(n, s, f)))
    end
    for offs in ([2, -1], [1, -1, 0, 3], [n, -n - 2], [0, -3, 3])     # no main diagonal, unsorted, all out of range
        ws = [randn(rng, ComplexF64, n) for _ in offs]
        push!(cases, (BandedOp(n, offs, ws), band_ref(n, offs, ws)))
    end
    @test dense(ShiftOp(n, 0, 1.5 - 0.5im)) == (1.5 - 0.5im) * I(n)    # zero shift: scaled identity
    @test iszero(dense(ShiftOp(n, n + 3, 0.3im))) && iszero(dense(BandedOp(n, [n, -n - 2], [ones(n), ones(n)])))
    dims = [n, 3, n]
    M = DenseOp(randn(rng, ComplexF64, 3, 3))
    for (op, A) in cases
        @test dense(op) ≈ A
        @test diagonal(op) ≈ diag(A)
        @test dense(op') ≈ A'
        for k in (1, 3)
            X = randn(rng, ComplexF64, dims..., 2); Y = fill(ComplexF64(NaN), size(X))
            apply_axis!(Y, op, X, k)
            @test reshape(Y, :, 2) ≈ kron_reference(dims, [(1.0 + 0im, [(k, A)])]) * reshape(X, :, 2)
        end
        terms = [AxisTerm(0.3 - 0.2im, [(1, op), (2, M)]), AxisTerm(1.1, [(3, op)])]
        H = KronFreeOperator(dims, terms; constant=0.5)
        ref = kron_reference(dims, [(0.3 - 0.2im, [(1, A), (2, M.M)]), (1.1 + 0im, [(3, A)])]) + 0.5I
        X = randn(rng, ComplexF64, prod(dims), 3)
        @test H * X ≈ ref * X
        @test diagonal(H) ≈ diag(ref)
    end
end

@testset "O4: fused diagonal and cancellation" begin
    rng = MersenneTwister(42)
    dims = [3, 4, 2]
    N = prod(dims)
    x = randn(rng, ComplexF64, N); X = randn(rng, ComplexF64, N, 3)
    reference(terms, c) = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms]) + c * I
    # constant only
    Hc = KronFreeOperator(dims, AxisTerm[]; constant=2.5)
    @test isempty(Hc.terms) && Hc.diag == fill(2.5 + 0im, N)
    @test Hc * X ≈ 2.5 * X && diagonal(Hc) == fill(2.5 + 0im, N)
    # diagonal only, no constant
    dA = randn(rng, 3); dB = randn(rng, 2)
    tdiag = [AxisTerm(0.4, [(1, DiagonalOp(dA))]), AxisTerm(-0.9, [(1, DiagonalOp(dA)), (3, DiagonalOp(dB))])]
    Hd = KronFreeOperator(dims, tdiag)
    refd = reference(tdiag, 0)
    @test isempty(Hd.terms) && Hd.diag ≈ diag(refd) && Hd * X ≈ refd * X && diagonal(Hd) ≈ diag(refd)
    # no fused contribution: the remaining dense and banded terms have nonzero diagonals
    Dn = DenseOp(randn(rng, ComplexF64, 3, 3)); Bn = BandedOp(4, [1, 0, -1], [randn(rng, ComplexF64, 4) for _ in 1:3])
    tnone = [AxisTerm(0.7, [(1, Dn)]), AxisTerm(-0.3im, [(2, Bn), (3, ShiftOp(2, 0, 1.5))])]
    Hn = KronFreeOperator(dims, tnone)
    refn = reference(tnone, 0)
    @test length(Hn.terms) == 2 && iszero(Hn.diag)                     # nothing fused, so no double counting
    @test Hn * X ≈ refn * X && diagonal(Hn) ≈ diag(refn)
    # mixed: the fused vector holds only the pure-diagonal terms and the constant
    tmix = [tdiag; tnone]
    Hm = KronFreeOperator(dims, tmix; constant=-0.2)
    refm = reference(tmix, -0.2)
    @test length(Hm.terms) == 2 && Hm.diag ≈ diag(refd) .- 0.2
    @test Hm * x ≈ refm * x && Hm * X ≈ refm * X && diagonal(Hm) ≈ diag(refm)
    # exactly cancelling terms, fused and kept
    tcancel = [AxisTerm(1.3, [(2, DiagonalOp(randn(rng, 4)))]), AxisTerm(0.7, [(1, Dn), (2, Bn)])]
    tcancel = [tcancel; [AxisTerm(-t.coeff, t.factors) for t in tcancel]]
    Hz = KronFreeOperator(dims, tcancel)
    @test length(Hz.terms) == 2 && iszero(Hz.diag)
    @test norm(Hz * X) <= 1e-14 * norm(X) && norm(diagonal(Hz)) <= 1e-14
    # complex coefficients whose Hermitian sum is real on the diagonal
    c = 0.6 - 0.8im
    Dc = DiagonalOp(randn(rng, ComplexF64, 4)); A = DenseOp(randn(rng, ComplexF64, 3, 3))
    therm = [AxisTerm(c, [(2, Dc)]), AxisTerm(conj(c), [(2, Dc')]),
             AxisTerm(c, [(1, A), (3, DiagonalOp(dB))]), AxisTerm(conj(c), [(1, A'), (3, DiagonalOp(dB))])]
    Hh = KronFreeOperator(dims, therm; constant=0.1)
    refh = reference(therm, 0.1)
    @test norm(refh - refh') <= 1e-12 && length(Hh.terms) == 2
    @test norm(imag(Hh.diag)) <= 1e-14 && norm(imag(diagonal(Hh))) <= 1e-13
    @test Hh * X ≈ refh * X && diagonal(Hh) ≈ diag(refh) && Matrix(Hh) ≈ refh
end

@testset "O5: workspace ownership" begin
    rng = MersenneTwister(43)
    dims = [3, 4, 2]
    N = prod(dims)
    D = DenseOp(randn(rng, ComplexF64, 3, 3)); Dg = DiagonalOp(randn(rng, ComplexF64, 4))
    Bd = BandedOp(4, [-1, 1], [randn(rng, ComplexF64, 4) for _ in 1:2]); S = ShiftOp(2, 1, 0.5im)
    terms = [AxisTerm(0.7, [(1, D), (2, Bd), (3, S)]), AxisTerm(-0.4, [(1, D), (2, Dg)]), AxisTerm(0.2, [(2, Dg)])]
    snapshot() = (copy(D.M), copy(Dg.d), copy(Bd.offsets), deepcopy(Bd.weights), (S.s, S.factor))
    before = snapshot()
    ref = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms])
    H1 = KronFreeOperator(dims, terms; block_capacity=3)
    H2 = KronFreeOperator(dims, terms; block_capacity=3)
    # Separate instances own separate scratch and fused diagonals.
    for (a, b) in ((H1.buf1, H2.buf1), (H1.buf2, H2.buf2), (H1.buf1, H1.buf2), (H1.diag, H2.diag))
        @test a !== b && !Base.mightalias(a, b)
    end
    # Interleaved applications through independent instances, with stale scratch in one of them.
    X1 = randn(rng, ComplexF64, N, 3); X2 = randn(rng, ComplexF64, N, 2)
    saved1, saved2 = copy(X1), copy(X2)
    fill!(H1.buf1, NaN); fill!(H1.buf2, NaN)
    Y1 = mul!(similar(X1), H1, X1)
    Y2 = mul!(similar(X2), H2, X2)
    @test Y1 ≈ ref * X1 && Y2 ≈ ref * X2
    @test mul!(similar(X1), H2, X1) ≈ ref * X1 && mul!(similar(X2), H1, X2) ≈ ref * X2
    mul!(similar(X1, N, 5), H1, randn(rng, ComplexF64, N, 5))              # growth in one instance only
    @test size(H1.buf1, 4) == 5 && size(H2.buf1, 4) == 3
    @test X1 == saved1 && X2 == saved2                                        # inputs untouched
    @test snapshot() == before                                                # factor data untouched
    @test diagonal(H1) ≈ diag(ref) && H1.diag ≈ H2.diag                       # diagonal(H) does not write H.diag
end
