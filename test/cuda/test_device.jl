using Test, LinearAlgebra, Random, CUDA, cuTENSOR, KronFreeCircuits
using KronFreeCircuits: LOBPCG, lobpcg_solve, RitzJacobi, _diagonal_expectation!
using KronFreeCircuits: AxisTerm, solve, dense, solver_callback, with_block_capacity, constant_term, indices
import LOBPCGEigensolver

# check_eigenpairs (../helpers.jl) on a device operator: each H * v runs on the device and is
# compared on the host.
struct HostApplied{O}
    H::O
end
Base.:*(W::HostApplied, v::AbstractVector) = Array(W.H * CuArray(v))
check_eigenpairs(H::KronFreeOperator{<:CuArray}, E, V; tol) = check_eigenpairs(HostApplied(H), E, V; tol=tol)

# The operator `H` with its data and scratch on the device.
to_device(H::KronFreeOperator) = KronFreeCircuits._on_storage(H, CuArray)

@testset "operator on the device equals host; no device allocation after warm-up" begin
    rng = MersenneTwister(61)
    dims = [40, 30, 3]
    terms = [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(1:40)))]),
             AxisTerm(0.2, [(1, DenseOp(randn(rng, ComplexF64, 40, 40))), (2, DenseOp(randn(rng, ComplexF64, 30, 30)))]),
             AxisTerm(0.1, [(2, ShiftOp(30, 2, 0.5)), (3, BandedOp(3, [-1, 1], [ones(3), ones(3)]))])]
    Hh = KronFreeOperator(dims, terms; block_capacity=6)
    Hd = KronFreeOperator(dims, terms; block_capacity=6, on=CuArray)
    Xh = randn(rng, ComplexF64, 3600, 6); Xd = CuArray(Xh)
    Yd = similar(Xd); Yv = view(Yd, :, 2:5); Xv = view(Xd, :, 2:5)
    @test Array(mul!(Yd, Hd, Xd)) ≈ mul!(similar(Xh), Hh, Xh)
    mul!(Yv, Hd, Xv); mul!(Yv, Hd, Xv)
    CUDA.synchronize()
    @test (CUDA.@allocated mul!(Yv, Hd, Xv)) < 4096
    @test diagonal(Hd) ≈ diagonal(Hh)
    # Matrix is host-only by contract: a device operator is rejected with an informative error.
    @test_throws ArgumentError Matrix(Hd)
end

@testset "fused diagonal expectation" begin
    rng = MersenneTwister(62)
    D = CuArray(randn(rng, 5000)); X = CuArray(randn(rng, ComplexF64, 5000, 7)); theta = CUDA.zeros(Float64, 7)
    _diagonal_expectation!(theta, D, X)
    @test Array(theta) ≈ vec(sum(Array(D) .* abs2.(Array(X)); dims=1) ./ sum(abs2.(Array(X)); dims=1))
    CUDA.synchronize()
    @test (CUDA.@allocated _diagonal_expectation!(theta, D, X)) == 0
    # The Ritz initialization's arguments (a prefix view of the cache, the solver's active block)
    # reach the fused method, not the generic broadcast-and-reduce fallback.
    m = which(_diagonal_expectation!, Tuple{typeof(view(theta, 1:3)), typeof(D), typeof(X[:, 1:3])})
    @test m.module === Base.get_extension(KronFreeCircuits, :KronFreeCircuitsCUDAExt)
end

@testset "LOBPCG on the device equals Dense on the host" begin
    Random.seed!(63)
    H = three_oscillators(6)
    Ed, _ = solve(Dense(), H, 0)
    for pc in (:jacobi, :ritz)
        E, V = solve(LOBPCG(tol=1e-9, preconditioner=pc, on=CuArray), H, 8)
        @test E ≈ Ed[1:8] atol=1e-7
        @test E isa Vector{Float64} && V isa Matrix{ComplexF64}
        check_eigenpairs(H, E, V; tol=1e-8)
    end
    θ = two_zero_pi_params(; g=0.05); spec = load_spec(fixture("two_zero_pi.json"))
    leaf(i) = Node([Symbol("theta$i") => ChargeBasis(6), Symbol("phi$i") => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(10))
    sd = materialize(Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(12)), spec, θ)
    sg = materialize(Node([leaf(1), leaf(2)]; solver=LOBPCG(tol=1e-10, on=CuArray), compute=12, keep=lowest(12)), spec, θ)
    @test energies(sg) ≈ energies(sd) atol=1e-7
end

# ---- Pathological-test catalogue, Task 7 device repeats (ports of the CPU testsets) ----

@testset "O1: axis identity, equal sizes, and complex factor order (ITensor dense path)" begin
    rng = MersenneTwister(44)
    for (dims, width) in (([3, 5, 7], 3), ([4, 4, 4], 4), ([1, 5, 3], 2))
        # A distinct complex nonsymmetric factor per axis, kept as plain matrices for the oracle.
        A = [randn(rng, ComplexF64, d, d) for d in dims]
        @test all(d == 1 || norm(M - transpose(M)) > 0.1 for (d, M) in zip(dims, A))
        groups = [[1], [2], [3], [1, 2], [2, 3], [1, 3], [1, 2, 3]]     # single factors and chains
        terms = [AxisTerm(cis(0.4g) * (1 + g / 10), [(a, DenseOp(A[a])) for a in grp]) for (g, grp) in enumerate(groups)]
        ref = kron_reference(dims, [(ComplexF64(t.coeff), [(a, A[a]) for a in grp]) for (t, grp) in zip(terms, groups)])
        H = KronFreeOperator(dims, terms; block_capacity=width, on=CuArray)
        # One distinct index per axis, even when dimensions coincide.
        @test length(H.inds) == 3 && allunique(H.inds)
        @test [KronFreeCircuits.ITensors.dim(i) for i in H.inds] == dims
        X = randn(rng, ComplexF64, prod(dims), width); Xd = CuArray(X)
        @test Array(mul!(similar(Xd), H, Xd)) ≈ ref * X
        x = randn(rng, ComplexF64, prod(dims))
        @test Array(H * CuArray(x)) ≈ ref * x
        @test_throws ArgumentError Matrix(H)
        # Each single-axis term against the gemm reference on the same equal-size state.
        for k in 1:3
            Hk = KronFreeOperator(dims, [AxisTerm(1.0, [(k, DenseOp(A[k]))])]; block_capacity=width, on=CuArray)
            Yref = similar(X)
            apply_axis!(reshape(Yref, dims..., width), DenseOp(A[k]), reshape(X, dims..., width), k)
            @test Array(mul!(similar(Xd), Hk, Xd)) ≈ Yref
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
    H = KronFreeOperator(dims, terms; block_capacity=8, on=CuArray)
    for (step, w) in enumerate((8, 3, 1, 12, 2))
        # Stale scratch: NaNs on even steps, garbage of the wrong scale on odd ones.
        isodd(step) ? (H.buf1 .= 1e300; H.buf2 .= -1e300) : (fill!(H.buf1, NaN); fill!(H.buf2, NaN))
        Xp = CuArray(randn(rng, ComplexF64, N, w + 3)); Yp = CUDA.fill(ComplexF64(-5), N, w + 3)
        X = view(Xp, :, 2:w+1); Y = view(Yp, :, 3:w+2)                 # offset column views
        saved = Array(Xp)
        @test mul!(Y, H, X) === Y
        @test Array(Y) ≈ ref * Array(X)
        @test Array(Xp) == saved
        @test all(==(-5), Array(Yp)[:, [1, 2, w + 3]])
        @test haskey(H.blockinds, w)
        @test KronFreeCircuits.ITensors.dim(H.blockinds[w]) == w
    end
    @test size(H.buf1, 4) == 12                                         # grown once, retained
    # Vector versus one-column-matrix calls give the same result.
    x = CuArray(randn(rng, ComplexF64, N))
    @test Array(H * x) ≈ ref * Array(x)
    @test Array(vec(H * reshape(x, N, 1))) ≈ Array(H * x)
    # Overlapping input/output views are rejected.
    Z = CuArray(randn(rng, ComplexF64, N, 4))
    @test_throws ArgumentError mul!(view(Z, :, 1:2), H, view(Z, :, 2:3))
    # After warm-up within capacity: buffers and block indices are reused, and no device memory
    # is allocated, whatever the block width.
    b1, b2, i3, i8 = H.buf1, H.buf2, H.blockinds[3], H.blockinds[8]
    allocs = Dict{Int,Int}()
    for w in (3, 8)
        X = view(CuArray(randn(rng, ComplexF64, N, w + 2)), :, 2:w+1); Y = similar(X, N, w)
        mul!(Y, H, X); mul!(Y, H, X)
        CUDA.synchronize()
        allocs[w] = CUDA.@allocated mul!(Y, H, X)
        @test Array(Y) ≈ ref * Array(X)
    end
    @test allocs[8] <= allocs[3] + 1024        # one block-sized temporary would add >= 5 * N * 16 bytes
    @test allocs[3] < 4096
    @test H.buf1 === b1 && H.buf2 === b2 && H.blockinds[3] === i3 && H.blockinds[8] === i8
end

@testset "O4: fused diagonal and cancellation" begin
    rng = MersenneTwister(42)
    dims = [3, 4, 2]
    N = prod(dims)
    x = randn(rng, ComplexF64, N); X = randn(rng, ComplexF64, N, 3)
    xd, Xd = CuArray(x), CuArray(X)
    reference(terms, c) = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms]) + c * I
    # constant only
    Hc = KronFreeOperator(dims, AxisTerm[]; constant=2.5, on=CuArray)
    @test isempty(Hc.terms) && Array(Hc.diag) == fill(2.5 + 0im, N)
    @test Array(Hc * Xd) ≈ 2.5 * X && diagonal(Hc) == fill(2.5 + 0im, N)
    # diagonal only, no constant
    dA = randn(rng, 3); dB = randn(rng, 2)
    tdiag = [AxisTerm(0.4, [(1, DiagonalOp(dA))]), AxisTerm(-0.9, [(1, DiagonalOp(dA)), (3, DiagonalOp(dB))])]
    Hd = KronFreeOperator(dims, tdiag; on=CuArray)
    refd = reference(tdiag, 0)
    @test isempty(Hd.terms) && Array(Hd.diag) ≈ diag(refd) && Array(Hd * Xd) ≈ refd * X && diagonal(Hd) ≈ diag(refd)
    # no fused contribution: the remaining dense and banded terms have nonzero diagonals
    Dn = DenseOp(randn(rng, ComplexF64, 3, 3)); Bn = BandedOp(4, [1, 0, -1], [randn(rng, ComplexF64, 4) for _ in 1:3])
    tnone = [AxisTerm(0.7, [(1, Dn)]), AxisTerm(-0.3im, [(2, Bn), (3, ShiftOp(2, 0, 1.5))])]
    Hn = KronFreeOperator(dims, tnone; on=CuArray)
    refn = reference(tnone, 0)
    @test length(Hn.terms) == 2 && iszero(Array(Hn.diag))              # nothing fused, so no double counting
    @test Array(Hn * Xd) ≈ refn * X && diagonal(Hn) ≈ diag(refn)
    # mixed: the fused vector holds only the pure-diagonal terms and the constant
    tmix = [tdiag; tnone]
    Hm = KronFreeOperator(dims, tmix; constant=-0.2, on=CuArray)
    refm = reference(tmix, -0.2)
    @test length(Hm.terms) == 2 && Array(Hm.diag) ≈ diag(refd) .- 0.2
    @test Array(Hm * xd) ≈ refm * x && Array(Hm * Xd) ≈ refm * X && diagonal(Hm) ≈ diag(refm)
    # exactly cancelling terms, fused and kept
    tcancel = [AxisTerm(1.3, [(2, DiagonalOp(randn(rng, 4)))]), AxisTerm(0.7, [(1, Dn), (2, Bn)])]
    tcancel = [tcancel; [AxisTerm(-t.coeff, t.factors) for t in tcancel]]
    Hz = KronFreeOperator(dims, tcancel; on=CuArray)
    @test length(Hz.terms) == 2 && iszero(Array(Hz.diag))
    @test norm(Array(Hz * Xd)) <= 1e-14 * norm(X) && norm(diagonal(Hz)) <= 1e-14
    # complex coefficients whose Hermitian sum is real on the diagonal
    c = 0.6 - 0.8im
    Dc = DiagonalOp(randn(rng, ComplexF64, 4)); A = DenseOp(randn(rng, ComplexF64, 3, 3))
    therm = [AxisTerm(c, [(2, Dc)]), AxisTerm(conj(c), [(2, Dc')]),
             AxisTerm(c, [(1, A), (3, DiagonalOp(dB))]), AxisTerm(conj(c), [(1, A'), (3, DiagonalOp(dB))])]
    Hh = KronFreeOperator(dims, therm; constant=0.1, on=CuArray)
    refh = reference(therm, 0.1)
    @test norm(refh - refh') <= 1e-12 && length(Hh.terms) == 2
    @test norm(imag(Array(Hh.diag))) <= 1e-14 && norm(imag(diagonal(Hh))) <= 1e-13
    @test Array(Hh * Xd) ≈ refh * X && diagonal(Hh) ≈ diag(refh)
    @test_throws ArgumentError Matrix(Hh)
end

# Single-axis operators and helpers of test/test_solvers.jl (not part of ../helpers.jl).
dense_axis_operator(A::AbstractMatrix) = KronFreeOperator([size(A, 1)], [AxisTerm(1.0, [(1, DenseOp(A))])])
diagonal_axis_operator(d::AbstractVector) = KronFreeOperator([length(d)], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(d)))])])
random_unitary(N) = Matrix(qr(randn(ComplexF64, N, N)).Q)
function hermitian_with_spectrum(Q, d)
    A = Q * Diagonal(d) * Q'
    return (A + A') / 2
end
struct WidthRecorder{O}
    H::O
    widths::Vector{Int}
end
LinearAlgebra.mul!(Y::AbstractVecOrMat, W::WidthRecorder, X::AbstractVecOrMat) = (push!(W.widths, size(X, 2)); mul!(Y, W.H, X))

@testset "S1: exact and wrong-subspace seeds; bounded recovery" begin
    Random.seed!(41)
    d = collect(range(0.0, 20.0; length=40))
    H = diagonal_axis_operator(d)
    Id = Matrix{ComplexF64}(I, 40, 40)
    for (X0, label) in ((Id[:, 1:3], "lowest"), (Id[:, 6:8], "excited"))
        E, V = @test_logs solve(LOBPCG(tol=1e-10, on=CuArray), H, 3; X0=X0)
        @test E ≈ d[1:3] atol=1e-8
        @test V * V' ≈ Id[:, 1:3] * Id[:, 1:3]' atol=1e-8
        check_eigenpairs(H, E, V; tol=1e-9)
    end
    E, V = solve(LOBPCG(tol=1e-10, on=CuArray), H, 3)
    E2, V2 = @test_logs solve(LOBPCG(tol=1e-10, on=CuArray), H, 3; X0=V)
    @test E2 ≈ d[1:3] atol=1e-8
    check_eigenpairs(H, E2, V2; tol=1e-9)
    # Device adjustment: in device arithmetic the eigensolver can converge on H = c I (every block is
    # exact) where the host one fails; the outcome must be an informative rejection or valid pairs.
    Hc = KronFreeOperator([30], AxisTerm[]; constant=2.5)
    out = try solve(LOBPCG(on=CuArray), Hc, 3; X0=randn(ComplexF64, 30, 3)) catch e; e end
    if out isa Exception
        @test out isa ErrorException && occursin("Dense", out.msg)
    else
        @test out[1] ≈ fill(2.5, 3) atol=1e-12
        check_eigenpairs(Hc, out...; tol=1e-12)
    end
end

@testset "S3: H = 0, H = cI, tight clusters and a cutoff inside a cluster" begin
    Random.seed!(43)
    # Device adjustment (as in S1): an informative rejection or valid pairs.
    for H in (KronFreeOperator([30], AxisTerm[]), KronFreeOperator([30], AxisTerm[]; constant=2.5))
        out = try solve(LOBPCG(on=CuArray), H, 3) catch e; e end
        if out isa Exception
            @test out isa ErrorException && occursin("Dense", out.msg)
        else
            @test out[1] ≈ fill(real(H.diag[1]), 3) atol=1e-12
            check_eigenpairs(H, out...; tol=1e-12)
        end
        @test solve(Dense(), H, 3)[1] ≈ fill(H.diag[1], 3)
    end
    N = 40
    Q = random_unitary(N)
    rest = collect(range(1.0, 10.0; length=N - 3))
    for δ in (1e-3, 1e-6, 1e-9, 0.0)
        H = dense_axis_operator(hermitian_with_spectrum(Q, [0, δ, 2δ, rest...]))
        E, V = solve(LOBPCG(tol=1e-9, on=CuArray), H, 3)
        @test E ≈ [0, δ, 2δ] atol=1e-8
        @test V * V' ≈ Q[:, 1:3] * Q[:, 1:3]' atol=1e-7
        check_eigenpairs(H, E, V; tol=1e-8)
    end
    H = dense_axis_operator(hermitian_with_spectrum(Q, [0, 1, 1, 1, collect(range(2.0, 10.0; length=N - 4))...]))
    E, V = solve(LOBPCG(tol=1e-9, on=CuArray), H, 2)
    @test E ≈ [0, 1] atol=1e-8
    @test norm(V * (V' * Q[:, 1]) - Q[:, 1]) < 1e-7
    Pb = Q[:, 2:4] * Q[:, 2:4]'
    @test norm(Pb * V[:, 2] - V[:, 2]) < 1e-7
    check_eigenpairs(H, E, V; tol=1e-8)
end

@testset "S5: no locking, several locks at once, one active column, complete locking" begin
    H = three_oscillators(6); m = 10
    Hd = to_device(H)
    D = CuArray(real.(diagonal(H)))
    Ed = solve(Dense(), H, m)[1]
    function aligned(P, seen)
        all(P.trace[t][2] == seen[t-1][2][end-P.trace[t][1]+1:end] for t in 2:min(length(P.trace), length(seen)))
    end
    Random.seed!(45)
    P = RitzJacobi(D, m; eta=0.05, trace=true)
    seen = Any[]
    LOBPCGEigensolver.lobpcg(with_block_capacity(Hd, m), CuArray(randn(ComplexF64, 216, m)), I, P, 1e-14, 2;
                             callback=solver_callback(P, info -> push!(seen, (info.n_locked, Array(info.λs)))))
    @test all(==(0), first.(seen)) && all(w == m for (w, _) in P.trace) && length(P.trace) == 3
    @test aligned(P, seen)
    multi = false; single = false
    for seed in 1:3
        Random.seed!(seed)
        P = RitzJacobi(D, m; eta=0.05, trace=true)
        W = WidthRecorder(with_block_capacity(Hd, m), Int[])
        seen = Any[]
        res = LOBPCGEigensolver.lobpcg(W, CuArray(randn(ComplexF64, 216, m)), I, P, 1e-9, 300;
                                       callback=solver_callback(P, info -> push!(seen, (info.n_locked, Array(info.λs)))))
        @test Array(res.λ) ≈ Ed atol=1e-7
        check_eigenpairs(H, Array(res.λ), Array(res.X); tol=1e-8)
        @test seen[end][1] == m
        @test aligned(P, seen)
        @test minimum(W.widths) >= 1 && minimum(first.(P.trace)) >= 1
        multi |= any(>=(2), diff(first.(seen)))
        single |= any(w == 1 for (w, _) in P.trace)
    end
    @test multi
    @test single
end

@testset "F3: identical uncoupled zero-pis (small fixture)" begin
    spec2 = load_spec(fixture("two_zero_pi.json"))
    leaf(i) = Node([Symbol("theta$i") => ChargeBasis(6), Symbol("phi$i") => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), compute=10, keep=lowest(10))
    lobtop(m) = Node([leaf(1), leaf(2)]; solver=LOBPCG(tol=1e-10, on=CuArray), compute=m, keep=lowest(m))
    mat(tree, θ; kw...) = @test_logs (:warn, r"edge") match_mode=:any materialize(tree, spec2, θ; kw...)
    e(i, j) = (v = zeros(ComplexF64, 100); v[i+10(j-1)] = 1; v)
    θ0 = two_zero_pi_params(; g=0.0)
    Random.seed!(63)
    t3 = lobtop(3); s3 = mat(t3, θ0)
    E1 = energies(s3, t3.children[1]); c = real(constant_term(spec2)(θ0))
    @test E1 ≈ energies(s3, t3.children[2]) atol=1e-12
    @test eigenvectors(s3, t3.children[1]) ≈ eigenvectors(s3, t3.children[2]) atol=1e-10
    sums = sort(vec(E1 .+ E1')) .+ c
    E, V = energies(s3), eigenvectors(s3)
    @test E ≈ sums[1:3] atol=1e-8
    @test E[2] ≈ E[3] atol=1e-8
    @test E[3] < sums[4] - 1e-3
    @test V * V' ≈ sum(v * v' for v in (e(1, 1), e(1, 2), e(2, 1))) atol=1e-7
    check_eigenpairs(KronFreeCircuits.operator(s3), E, V; tol=1e-9)
    t2 = lobtop(2); s2 = mat(t2, θ0; previous=s3)
    E, V = energies(s2), eigenvectors(s2)
    @test E ≈ sums[1:2] atol=1e-8
    @test abs2(dot(e(1, 1), V[:, 1])) ≈ 1 atol=1e-8
    P12 = e(1, 2) * e(1, 2)' + e(2, 1) * e(2, 1)'
    @test norm(P12 * V[:, 2] - V[:, 2]) < 1e-7
    check_eigenpairs(KronFreeCircuits.operator(s2), E, V; tol=1e-9)
    θd = merge(θ0, (E_J2=θ0.E_J2 * 1.01,))
    td = lobtop(3); sdt = mat(td, θd; previous=s3)
    Ed1, Ed2 = energies(sdt, td.children[1]), energies(sdt, td.children[2])
    @test energies(sdt) ≈ sort(vec(Ed1 .+ Ed2'))[1:3] .+ real(constant_term(spec2)(θd)) atol=1e-8
    check_eigenpairs(KronFreeCircuits.operator(sdt), energies(sdt), eigenvectors(sdt); tol=1e-9)
    θg = two_zero_pi_params(; g=0.05)
    sd = mat(Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(6)), θg)
    S = zeros(100, 100); for i in 1:10, j in 1:10; S[i+10(j-1), j+10(i-1)] = 1; end
    Vd = eigenvectors(sd)
    parity = [real(dot(Vd[:, k], S * Vd[:, k])) for k in 1:6]
    @test all(p -> abs(abs(p) - 1) < 1e-8, parity) && any(<(0), parity[1:3])
    sym = findall(>(0), parity)[1:3]
    psym = mat(Node([leaf(1), leaf(2)]; solver=Dense(), compute=6, keep=indices(sym)), θg)
    tg = lobtop(3); sg = mat(tg, θg; previous=psym)
    E, V = energies(sg), eigenvectors(sg)
    @test E ≈ energies(sd)[1:3] atol=1e-8
    @test V * V' ≈ Vd[:, 1:3] * Vd[:, 1:3]' atol=1e-6
    check_eigenpairs(KronFreeCircuits.operator(sg), E, V; tol=1e-9)
end

@testset "F7: complex three-mode cosine as an adjoint pair (device operator and LOBPCG on the flat tree)" begin
    rng = MersenneTwister(46)
    EJ, δ = 1.3, 0.37π
    bc, be, bh = ChargeBasis(2), PhaseGrid(-2π, 2π, 11), FockBasis(6; phi_zpf=0.6)
    Sc(k) = ComplexF64[i == j + k for i in 1:5, j in 1:5]
    Ee(k) = Matrix(Diagonal(cis.(k .* collect(range(-2π, 2π; length=11)))))
    a = diagm(1 => sqrt.(1.0:59))
    Dh(k) = exp(im * k * 0.6 * (a + a'))[1:6, 1:6]
    for pos in ((1, 2, 3), (1, 3, 2), (2, 3, 1))
        pc, pe, ph = pos
        dims = zeros(Int, 3); dims[pc] = 5; dims[pe] = 11; dims[ph] = 6
        term(s) = AxisTerm(-EJ / 2 * cis(s * δ),
                           sort([(pc, localop(bc, expiphi(s))), (pe, localop(be, expiphi(s))), (ph, localop(bh, expiphi(-s)))]; by=first))
        Tp = kron_reference(dims, [(-EJ / 2 * cis(δ), [(pc, Sc(1)), (pe, Ee(1)), (ph, Dh(-1))])])
        Tm = kron_reference(dims, [(-EJ / 2 * cis(-δ), [(pc, Sc(-1)), (pe, Ee(-1)), (ph, Dh(1))])])
        @test norm(Tm - Tp') <= 1e-12 * norm(Tp)
        M = Tp + Tm
        @test norm(M - M') <= 1e-12 * norm(M)                                   # Hermitian sum
        Hp = KronFreeOperator(dims, [term(1)]; block_capacity=3, on=CuArray)
        H = KronFreeOperator(dims, [term(1), term(-1)]; block_capacity=3, on=CuArray)
        X = randn(rng, ComplexF64, prod(dims), 3); Xd = CuArray(X)
        @test norm(Array(mul!(similar(Xd), Hp, Xd)) - Tp * X) <= 1e-10 * norm(X) * norm(Tp)
        @test norm(Array(mul!(similar(Xd), H, Xd)) - M * X) <= 1e-10 * norm(X) * norm(M)
        x = randn(rng, ComplexF64, prod(dims))
        @test norm(Array(H * CuArray(x)) - M * x) <= 1e-10 * norm(x) * norm(M)
    end
    # The supported flat tree through the public path: LOBPCG on the device equals Dense.
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
    θ = (E_C=1.0, E_J=EJ, E_L=0.5)
    children = [:c => bc, :e => be, :h => bh]
    Random.seed!(47)
    sd = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(children; solver=Dense(), keep=lowest(6)), spec, θ)
    sg = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(children; solver=LOBPCG(tol=1e-9, on=CuArray), compute=6, keep=lowest(6)), spec, θ)
    @test energies(sg) ≈ energies(sd) atol=1e-7
    @test eigenvectors(sg) * eigenvectors(sg)' ≈ eigenvectors(sd) * eigenvectors(sd)' atol=1e-6
    check_eigenpairs(KronFreeCircuits.operator(sg), energies(sg), eigenvectors(sg); tol=1e-8)
end
