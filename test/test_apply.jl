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

@testset "ITensor dense factors match the gemm reference" begin
    rng = MersenneTwister(41)
    dims = [5, 6, 4]
    ops = [DenseOp(randn(rng, ComplexF64, d, d)) for d in dims]
    for k in 1:3, width in (1, 3)      # first, middle and last axis; vector and block
        H = KronFreeOperator(dims, [AxisTerm(1.0, [(k, ops[k])])]; block_capacity=3)
        X = randn(rng, ComplexF64, prod(dims), width)
        Y = similar(X); Yref = similar(X)
        apply_axis!(reshape(Yref, dims..., width), ops[k], reshape(X, dims..., width), k)
        @test mul!(Y, H, X) ≈ Yref
    end
    H2 = KronFreeOperator(dims, [AxisTerm(0.3 + 0.1im, [(1, ops[1]), (3, ops[3])]), AxisTerm(-0.8, [(2, ops[2]), (3, ops[3])])])
    ref = kron_reference(dims, [(0.3 + 0.1im, [(1, dense(ops[1])), (3, dense(ops[3]))]), (-0.8 + 0im, [(2, dense(ops[2])), (3, dense(ops[3]))])])
    X = randn(rng, ComplexF64, prod(dims), 2)
    @test mul!(similar(X), H2, X) ≈ ref * X
    @test Matrix(H2) ≈ ref
end

@testset "ITensor dense path: offset views, input preserved, no state-sized allocation" begin
    rng = MersenneTwister(42)
    dims = [60, 50]
    H = KronFreeOperator(dims, [AxisTerm(1.0, [(1, DenseOp(randn(rng, ComplexF64, 60, 60))), (2, DenseOp(randn(rng, ComplexF64, 50, 50)))])]; block_capacity=8)
    Xp = randn(rng, ComplexF64, 3000, 8); Yp = fill(ComplexF64(7), 3000, 8)
    X = view(Xp, :, 3:7); Y = view(Yp, :, 3:7)
    saved = copy(Xp)
    mul!(Y, H, X); mul!(Y, H, X)
    @test (@allocated mul!(Y, H, X)) < 64 * 1024          # 240 kB state block; metadata only
    @test Xp == saved
    @test all(==(7), Yp[:, [1, 2, 8]])
    @test Y ≈ H * copy(X)
end

@testset "mixed branch operator: fused diagonal + ITensor dense + banded kernel" begin
    rng = MersenneTwister(43)
    dims = [4, 7]
    E = DiagonalOp(ComplexF64.(sort(randn(rng, 4)))); P = DenseOp(randn(rng, ComplexF64, 4, 4))
    B = BandedOp(7, [-1, 0, 1], [ones(7), 2 .* ones(7), ones(7)])
    terms = [AxisTerm(1.0, [(1, E)]), AxisTerm(0.1, [(1, P), (2, B)])]
    H = KronFreeOperator(dims, terms)
    ref = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms])
    X = randn(rng, ComplexF64, 28, 3)
    @test mul!(similar(X), H, X) ≈ ref * X
    @test diagonal(H) ≈ diag(ref)
end

@testset "O1: axis identity, equal sizes, and complex factor order (ITensor dense path)" begin
    rng = MersenneTwister(44)
    for (dims, width) in (([3, 5, 7], 3), ([4, 4, 4], 4), ([1, 5, 3], 2))
        # A distinct complex nonsymmetric factor per axis, kept as plain matrices for the oracle.
        A = [randn(rng, ComplexF64, d, d) for d in dims]
        @test all(d == 1 || norm(M - transpose(M)) > 0.1 for (d, M) in zip(dims, A))
        groups = [[1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3]]     # single factors and chains
        terms = [AxisTerm(cis(0.4g) * (1 + g / 10), [(a, DenseOp(A[a])) for a in grp]) for (g, grp) in enumerate(groups)]
        ref = kron_reference(dims, [(ComplexF64(t.coeff), [(a, A[a]) for a in grp]) for (t, grp) in zip(terms, groups)])
        H = KronFreeOperator(dims, terms; block_capacity=width)
        # One distinct index per axis, even when dimensions coincide.
        @test length(H.inds) == 3 && allunique(H.inds)
        @test [KronFreeCircuits.ITensors.dim(i) for i in H.inds] == dims
        X = randn(rng, ComplexF64, prod(dims), width)
        @test mul!(similar(X), H, X) ≈ ref * X
        x = randn(rng, ComplexF64, prod(dims))
        @test H * x ≈ ref * x
        @test Matrix(H) ≈ ref
        # Each single-axis term against the gemm reference on the same equal-size state.
        for k in 1:3
            Hk = KronFreeOperator(dims, [AxisTerm(1.0, [(k, DenseOp(A[k]))])]; block_capacity=width)
            Yref = similar(X)
            apply_axis!(reshape(Yref, dims..., width), DenseOp(A[k]), reshape(X, dims..., width), k)
            @test mul!(similar(X), Hk, X) ≈ Yref
        end
    end
end

@testset "O2: shrinking/growing blocks, views, and stale scratch (ITensor dense path)" begin
    rng = MersenneTwister(45)
    dims = [4, 5, 3]
    A = [randn(rng, ComplexF64, d, d) for d in dims]
    terms = [AxisTerm(0.7 - 0.2im, [(1, DenseOp(A[1])), (2, DenseOp(A[2])), (3, DenseOp(A[3]))]),
             AxisTerm(0.4, [(2, DenseOp(A[2])), (3, BandedOp(3, [-1, 1], [ones(3), ones(3)]))]),
             AxisTerm(-1.1, [(2, DiagonalOp(randn(rng, 5)))])]
    ref = kron_reference(dims, [(ComplexF64(t.coeff), [(a, dense(op)) for (a, op) in t.factors]) for t in terms])
    N = prod(dims)
    H = KronFreeOperator(dims, terms; block_capacity=8)
    for (step, w) in enumerate((8, 3, 1, 12, 2))
        # Stale scratch: NaNs on even steps, garbage of the wrong scale on odd ones.
        isodd(step) ? (H.buf1 .= 1e300; H.buf2 .= -1e300) : (fill!(H.buf1, NaN); fill!(H.buf2, NaN))
        Xp = randn(rng, ComplexF64, N, w + 3); Yp = fill(ComplexF64(-5), N, w + 3)
        X = view(Xp, :, 2:w+1); Y = view(Yp, :, 3:w+2)                 # offset column views
        saved = copy(Xp)
        @test mul!(Y, H, X) === Y
        @test Y ≈ ref * X
        @test Xp == saved
        @test all(==(-5), Yp[:, [1, 2, w + 3]])
        @test haskey(H.blockinds, w)
        @test KronFreeCircuits.ITensors.dim(H.blockinds[w]) == w
    end
    @test size(H.buf1, 4) == 12                                         # grown once, retained
    # Vector versus one-column-matrix calls give the same result.
    x = randn(rng, ComplexF64, N)
    @test H * x ≈ ref * x
    @test vec(H * reshape(x, N, 1)) ≈ H * x
    # Overlapping input/output views are rejected.
    Z = randn(rng, ComplexF64, N, 4)
    @test_throws ArgumentError mul!(view(Z, :, 1:2), H, view(Z, :, 2:3))
    # After warm-up within capacity: buffers and block indices are reused, and the allocation is
    # per-contraction metadata (about 20 kB each here) that does not grow with the block width.
    b1, b2, i3, i8 = H.buf1, H.buf2, H.blockinds[3], H.blockinds[8]
    allocs = Dict{Int,Int}()
    for w in (3, 8)
        X = view(randn(rng, ComplexF64, N, w + 2), :, 2:w+1); Y = similar(X, N, w)
        mul!(Y, H, X); mul!(Y, H, X)
        allocs[w] = @allocated mul!(Y, H, X)
        @test Y ≈ ref * X
    end
    @test allocs[8] <= allocs[3] + 1024        # one block-sized temporary would add >= 5 * N * 16 bytes
    @test allocs[3] < 4 * 32 * 1024            # four dense contractions, metadata only
    @test H.buf1 === b1 && H.buf2 === b2 && H.blockinds[3] === i3 && H.blockinds[8] === i8
end

@testset "F7: complex three-mode cosine as an adjoint pair (operator level)" begin
    rng = MersenneTwister(46)
    EJ, δ = 1.3, 0.37π
    bc, be, bh = ChargeBasis(2), PhaseGrid(-2π, 2π, 11), FockBasis(6; phi_zpf=0.6)
    # Independent finite matrices of e^{±iφ} in each basis.
    Sc(k) = ComplexF64[i == j + k for i in 1:5, j in 1:5]                       # e^{ikφ}|n⟩ = |n+k⟩
    Ee(k) = Matrix(Diagonal(cis.(k .* collect(range(-2π, 2π; length=11)))))
    a = diagm(1 => sqrt.(1.0:59))
    Dh(k) = exp(im * k * 0.6 * (a + a'))[1:6, 1:6]                             # converged block of a large truncation
    # cos(φc + φe - φh + δ) = (e^{iδ} e^{iφc} e^{iφe} e^{-iφh} + h.c.) / 2
    for pos in ((1, 2, 3), (1, 3, 2), (2, 3, 1))                               # the Fock (dense) mode last, middle, first
        pc, pe, ph = pos
        dims = zeros(Int, 3); dims[pc] = 5; dims[pe] = 11; dims[ph] = 6
        term(s) = AxisTerm(-EJ / 2 * cis(s * δ),
                           sort([(pc, localop(bc, expiphi(s))), (pe, localop(be, expiphi(s))), (ph, localop(bh, expiphi(-s)))]; by=first))
        @test term(1).factors[findfirst(f -> f[1] == ph, term(1).factors)][2] isa DenseOp
        Tp = kron_reference(dims, [(-EJ / 2 * cis(δ), [(pc, Sc(1)), (pe, Ee(1)), (ph, Dh(-1))])])
        Tm = kron_reference(dims, [(-EJ / 2 * cis(-δ), [(pc, Sc(-1)), (pe, Ee(-1)), (ph, Dh(1))])])
        @test norm(Tm - Tp') <= 1e-12 * norm(Tp)                                # adjoint pair
        @test norm(Tp - Tp') > 0.1                                              # each term non-Hermitian
        Hp = KronFreeOperator(dims, [term(1)]; block_capacity=3)
        Hm = KronFreeOperator(dims, [term(-1)]; block_capacity=3)
        H = KronFreeOperator(dims, [term(1), term(-1)]; block_capacity=3)
        @test norm(Matrix(Hp) - Tp) <= 1e-10 * norm(Tp)
        @test norm(Matrix(Hm) - Matrix(Hp)') <= 1e-12 * norm(Tp)
        M = Matrix(H)
        @test norm(M - M') <= 1e-12 * norm(M)                                   # Hermitian sum
        @test norm(M - (Tp + Tm)) <= 1e-10 * norm(M)
        X = randn(rng, ComplexF64, prod(dims), 3)
        @test norm(mul!(similar(X), H, X) - (Tp + Tm) * X) <= 1e-10 * norm(X) * norm(M)
        x = randn(rng, ComplexF64, prod(dims))
        @test norm(H * x - (Tp + Tm) * x) <= 1e-10 * norm(x) * norm(M)
    end
    # Grouping two of the cosine's modes inside one child needs a composite port: rejected.
    json = """
    {"format": "kronfree-spec", "version": 1, "units": "GHz",
     "parameters": ["E_C", "E_J", "E_L"],
     "modes": [{"name": "c", "type": "c"}, {"name": "e", "type": "e"}, {"name": "h", "type": "h"}],
     "E_C": [["E_C", "0", "0"], ["0", "E_C", "0"], ["0", "0", "E_C"]],
     "E_L": [["0", "0", "0"], ["0", "E_L", "0"], ["0", "0", "E_L"]],
     "flux_linear": ["0", "0", "0"], "n_g": ["0", "0", "0"],
     "cosines": [{"energy": "E_J", "omega": [1, 1, -1], "offset": "0.37*pi"}]}
    """
    spec = KronFreeCircuits.parse_spec(json)
    tree = Node([Node([:c => bc, :e => be]), Node([:h => bh])])
    err = try materialize(tree, spec, (E_C=1.0, E_J=EJ, E_L=0.5)); nothing catch e e end
    @test err isa ArgumentError
    @test occursin("[:c, :e]", err.msg) && occursin("composite (multi-mode) ports are not yet supported", err.msg)
end

@testset "sparse(H), gershgorin_bound, separable_eigenbases, product_seed" begin
    rng = MersenneTwister(71)
    H = three_oscillators(5)                                   # 125 states, structured factors only
    S = sparse(H)
    @test S ≈ Matrix(H) && S === sparse(H)                    # exact and memoized
    @test gershgorin_bound(H) >= maximum(eigvals(Hermitian(Matrix(H))))
    @test gershgorin_bound(H) <= 1.5 * maximum(eigvals(Hermitian(Matrix(H))))   # not absurdly loose on this fixture
    sep = separable_eigenbases(H)
    @test length(sep.U) == 3 && all(U -> U' * U ≈ I, sep.U)
    X = product_seed(H, 6)
    @test size(X) == (125, 6) && X' * X ≈ I
    # the seed's Rayleigh quotients sit below the median of the spectrum
    E = eigvals(Hermitian(Matrix(H)))
    @test all(real(diag(X' * (Matrix(H) * X))) .< E[end ÷ 2])
    # a branch-like operator (dense factors, no single-axis structure) seeds with unit vectors by the diagonal
    termsb = [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(1:4)))]), AxisTerm(1.0, [(2, DiagonalOp(ComplexF64.([0.0, 5.0, 10.0])))]),
              AxisTerm(0.1, [(1, DenseOp(randn(rng, ComplexF64, 4, 4))), (2, DenseOp(randn(rng, ComplexF64, 3, 3)))])]
    Hb = KronFreeOperator([4, 3], termsb)
    Xb = product_seed(Hb, 3)
    @test all(count(!iszero, Xb[:, j]) == 1 for j in 1:3) && sort(findall(!iszero, Xb[:, 1]))[1] == argmin(real(Hb.diag))
    # ... ranked by the full diagonal: a kept zero-shift ShiftOp ⊗ DiagonalOp term (not fused into H.diag) reorders the seed
    Hk = KronFreeOperator([4, 3], [termsb; AxisTerm(1.0, [(1, ShiftOp(4, 0, 1.0)), (2, DiagonalOp(ComplexF64.([0.0, -20.0, 0.0])))])])
    Dk = KronFreeCircuits._full_diagonal(Hk)
    @test argmin(Dk) != argmin(real(Hk.diag))
    Xk = product_seed(Hk, 3)
    @test findall(!iszero, Xk[:, 1]) == [argmin(Dk)] && [only(findall(!iszero, Xk[:, j])) for j in 1:3] == sortperm(Dk)[1:3]
    # the nonzero cap refuses before assembling
    Hbig = KronFreeOperator([300, 300], [AxisTerm(1.0, [(1, DenseOp(randn(rng, ComplexF64, 300, 300))), (2, DenseOp(randn(rng, ComplexF64, 300, 300)))])])
    err = try sparse(Hbig; maxnnz=10^6); nothing catch e; e end
    @test err isa ArgumentError && occursin("maxnnz", err.msg)
    @test infnorm(ShiftOp(5, 1, 0.3)) == 0.3 && infnorm(DiagonalOp(ComplexF64[1, -2, 0])) == 2
end

@testset "derived quantities: every factor kind, the constant, non-identity bases, shared memos" begin
    rng = MersenneTwister(72)
    herm(d) = (A = randn(rng, ComplexF64, d, d); (A + A') / 2)
    # sparse(op) and infnorm(op) against the dense matrix, for every LocalOp
    for op in (DenseOp(randn(rng, ComplexF64, 4, 4)), DiagonalOp(randn(rng, 5)), DiagonalOp(ComplexF64[1, -2, 0]),
               BandedOp(5, [-1, 0, 2], [randn(rng, ComplexF64, 5) for _ in 1:3]), BandedOp(4, [-3, 1], [randn(rng, 4) for _ in 1:2]),
               ShiftOp(5, -2, 0.5im), ShiftOp(3, 1, 0.3), ShiftOp(3, 4, 1.0))
        @test sparse(op) isa SparseMatrixCSC{eltype(op)} && sparse(op) == dense(op)
        @test infnorm(op) ≈ opnorm(dense(op), Inf) atol=1e-14
    end
    # A Hermitian operator on dims [4, 5, 3]: non-diagonal single-axis terms on every axis, a fused
    # two-axis diagonal coupling, a banded ⊗ shift coupling with its adjoint, and a constant.
    dims = [4, 5, 3]
    A1 = herm(4)
    w = randn(rng, ComplexF64, 5)
    B2 = BandedOp(5, [-1, 0, 1], [[0; conj.(w[1:4])], randn(rng, 5), [w[1:4]; 0]])     # Hermitian tridiagonal
    Bc = BandedOp(4, [0, 2], [randn(rng, ComplexF64, 4) for _ in 1:2])
    c = 0.3 + 0.2im
    terms = [AxisTerm(1.0, [(1, DenseOp(A1))]), AxisTerm(0.5, [(2, B2)]),
             AxisTerm(0.4, [(3, ShiftOp(3, 1, 1.0))]), AxisTerm(0.4, [(3, ShiftOp(3, -1, 1.0))]),
             AxisTerm(0.2, [(1, DiagonalOp(randn(rng, 4))), (2, DiagonalOp(randn(rng, 5)))]),
             AxisTerm(c, [(1, Bc), (3, ShiftOp(3, 1, 1.0))]), AxisTerm(conj(c), [(1, Bc'), (3, ShiftOp(3, -1, 1.0))])]
    H = KronFreeOperator(dims, terms; constant=0.7)
    ref = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms]) + 0.7I
    @test norm(ref - ref') <= 1e-14 * norm(ref)
    @test length(H.diagterms) == 2                                               # the fused coupling and the constant
    S = sparse(H)
    @test S isa SparseMatrixCSC{ComplexF64} && norm(Matrix(S) - ref) <= 1e-13 * norm(ref)
    @test H.sparse[] === S
    @test gershgorin_bound(H) >= maximum(eigvals(Hermitian(ref)))
    @test gershgorin_bound(H) === gershgorin_bound(H)
    # per-axis bases diagonalize the single-axis sums; the constant sits in axis 1's energies
    sep = separable_eigenbases(H)
    @test sep === separable_eigenbases(H)
    single = [A1 + 0.7I, 0.5 * dense(B2), 0.4 * (dense(ShiftOp(3, 1, 1.0)) + dense(ShiftOp(3, -1, 1.0)))]
    for k in 1:3
        @test sep.U[k]' * sep.U[k] ≈ I
        @test sep.U[k] * Diagonal(sep.λ[k]) * sep.U[k]' ≈ single[k]
        @test !(sep.U[k] ≈ I)
    end
    # the seed is the lowest product states of the separable part, against brute force
    Hsep = kron_reference(dims, [(1.0 + 0im, [(k, single[k])]) for k in 1:3])
    sums = vec([a + b + d for a in sep.λ[1], b in sep.λ[2], d in sep.λ[3]])
    X = product_seed(H, 5)
    @test size(X) == (60, 5) && X' * X ≈ I
    @test real(diag(X' * Hsep * X)) ≈ sort(sums)[1:5]
    @test norm(X' * Hsep * X - Diagonal(sort(sums)[1:5])) <= 1e-12
    @test_throws ArgumentError product_seed(H, 0)
    @test_throws ArgumentError product_seed(H, 61)
    # memos are shared with a re-capacitied operator
    Hc = KronFreeCircuits.with_block_capacity(H, 3)
    @test sparse(Hc) === S && separable_eigenbases(Hc) === sep && Hc.diagterms === H.diagterms
    # a non-Hermitian single-axis sum is refused
    Hn = KronFreeOperator([3], [AxisTerm(1.0, [(1, ShiftOp(3, 1, 1.0))])])
    err = try separable_eigenbases(Hn); nothing catch e; e end
    @test err isa ArgumentError && occursin("Hermitian", err.msg)
end

@testset "KronFreeOperator element type follows its data" begin
    rng = MersenneTwister(302)
    terms = [AxisTerm(1.0, [(1, DiagonalOp(collect(0.0:3)))]), AxisTerm(0.5, [(1, BandedOp(4, [1, -1], [ones(4), ones(4)])), (2, DenseOp(randn(rng, 3, 3) |> x -> x + x'))])]
    H = KronFreeOperator([4, 3], terms)
    @test eltype(H) === Float64 && eltype(Matrix(H)) === Float64
    D, B, C = dense(terms[1].factors[1][2]), dense(terms[2].factors[1][2]), dense(terms[2].factors[2][2])
    @test Matrix(H) ≈ kron(Matrix(1.0I, 3, 3), D) + 0.5 * kron(C, B) atol=1e-14          # axis 1 fastest
    Hc = KronFreeOperator([4, 3], [AxisTerm(ComplexF64(t.coeff), t.factors) for t in terms])
    @test eltype(Hc) === ComplexF64
    @test Matrix(Hc) ≈ Matrix(H) atol=1e-14
    @test all(t -> all(f -> !(f[2] isa DenseOp) || eltype(f[3]) === ComplexF64, t.factors), Hc.terms)   # dense data converted
    X = randn(rng, ComplexF64, 12, 3)
    @test H * X ≈ Matrix(H) * X atol=1e-13                                                              # mixed convenience path
    Y = similar(X); mul!(Y, H, X); @test Y ≈ Matrix(H) * X atol=1e-13
    @test eltype(three_oscillators(4)) === ComplexF64                                                   # complex data stays complex
end

@testset "real operator on complex vectors: checks and values" begin
    rng = MersenneTwister(7)
    H = KronFreeOperator([3, 4], [AxisTerm(0.7, [(1, DenseOp(Matrix(Symmetric(randn(rng, 3, 3))))), (2, ShiftOp(4, 1, 1.0))]),
                                   AxisTerm(0.3, [(2, DiagonalOp(randn(rng, 4)))])]; constant=0.1)
    @test eltype(H) === Float64
    x = randn(rng, ComplexF64, 12)
    @test H * x ≈ Matrix(H) * x
    X = randn(rng, ComplexF64, 12, 5)
    y = zeros(ComplexF64, 12, 2)
    mul!(y, H, view(X, :, 2:3)); @test y ≈ Matrix(H) * X[:, 2:3]                          # a block view
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, 12, 3), H, reshape(x, 12, 1))             # n×1 into n×3
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, 12), H, randn(rng, ComplexF64, 11))
    @test_throws ArgumentError mul!(X, H, X)
    # integer data gives a floating operator
    Hi = KronFreeOperator([2], [AxisTerm(2, [(1, DenseOp([1 2; 2 1]))])]; constant=1)
    v = [1.0, -2.0]
    @test eltype(Hi) === Float64 && Hi * v ≈ Matrix(Hi) * v
end
