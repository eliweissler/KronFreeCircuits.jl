using Random
using KronFreeCircuits: Dense, Lanczos, solve, AxisTerm, DiagonalOp, DenseOp, ShiftOp

@testset "solvers agree" begin
    Random.seed!(1234)
    H = three_oscillators(6)
    Ed, Vd = solve(Dense(), H, 0)
    El, Vl = solve(Lanczos(tol=1e-10), H, 5)
    @test El ≈ Ed[1:5] atol=1e-8
    @test all(abs.(diag(Vl' * Vd[:, 1:5])) .≈ 1)
    @test Ed ≈ eigvals(Hermitian(kron_three_oscillators(6)))
    @test typeof(solve(Dense(), H, 0)) == typeof(solve(Lanczos(), H, 5)) == Tuple{Vector{Float64},Matrix{ComplexF64}}
    @test issorted(El)
end

@testset "Dense: blocked matrix, sliced eigenpairs, size warning" begin
    using KronFreeCircuits: _dense_too_large, Node, ChargeBasis, lowest, materialize, load_spec
    H = three_oscillators(7)                                        # n = 343: two column blocks
    n = size(H, 1)
    @test Matrix(H) == mul!(Matrix{ComplexF64}(undef, n, n), H, Matrix{ComplexF64}(I, n, n))
    @test Matrix(H) ≈ kron_three_oscillators(7)
    E, V = solve(Dense(), H, 3)
    Ea, Va = solve(Dense(), H, 0)
    @test length(E) == 3 && size(V) == (n, 3) && length(Ea) == n
    @test E ≈ Ea[1:3] && abs.(diag(V' * Va[:, 1:3])) ≈ ones(3)
    @test length(solve(Dense(), H, n + 5)[1]) == n                  # m beyond n: all pairs
    @test_throws ArgumentError solve(Dense(), H, -1)
    @test _dense_too_large(Dense(), 10_001) && !_dense_too_large(Dense(), 10_000)
    @test !_dense_too_large(Lanczos(), 10^6)
    # Dense with compute > 0 hands the selection rule only the lowest `compute` pairs
    spec = load_spec(fixture("transmon.json")); θ = (E_J=30.0, E_C=1.2, n_g=0.25)
    sys = materialize(Node([:varphi_1 => ChargeBasis(10)]; compute=4, keep=lowest(0)), spec, θ)
    @test length(KronFreeCircuits.energies(sys)) == 4
end

@testset "Lanczos failure modes" begin
    Random.seed!(1)
    H = three_oscillators(6)
    n = size(H, 1)
    # genuine stall: one restart cycle cannot reach tol = 1e-14
    err = try solve(Lanczos(tol=1e-14, maxiter=1), H, 5); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("converged", err.msg) && occursin("normres", err.msg)
    @test_throws ArgumentError solve(Lanczos(), H, n + 1)
    @test_throws ArgumentError solve(Lanczos(), H, 0)
end

@testset "ShiftInvert equals Dense; real path; residual check; certificate" begin
    Random.seed!(81)
    H = three_oscillators(6)                                   # real coefficients: the assembled matrix is exactly real, the real path runs
    Ed, Vd = solve(Dense(), H, 0)
    E, V = solve(ShiftInvert(tol=1e-9), H, 12)
    @test E ≈ Ed[1:12] atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
    @test V * V' ≈ Vd[:, 1:12] * Vd[:, 1:12]' atol=1e-7
    E2, V2 = solve(ShiftInvert(tol=1e-9, block=2), H, 12); @test E2 ≈ Ed[1:12] atol=1e-8; check_eigenpairs(H, E2, V2; tol=1e-9)
    E3, _ = solve(ShiftInvert(tol=1e-9, certify=true), H, 12); @test E3 ≈ Ed[1:12] atol=1e-8
    # the zero-pi leaf on the small grid (1313 states) against Dense, real arithmetic expected
    θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    leaf = Node([:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(8))
    sd = @test_logs (:warn, r"edge") match_mode=:any materialize(leaf, spec_zp, θ)
    si = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(leaf.children; solver=ShiftInvert(tol=1e-10), compute=8, keep=lowest(8)), spec_zp, θ)
    @test energies(si) ≈ energies(sd) atol=1e-8
    @test eltype(KronFreeCircuits.operator(si)) === Float64                       # the symmetric zero-pi leaf is real by structure
    check_eigenpairs(KronFreeCircuits.operator(si), energies(si), eigenvectors(si); tol=1e-9)
    # a loose inner tolerance must raise, not return
    err = try solve(ShiftInvert(tol=1e-10, inner_tol=1e-4), H, 12); nothing catch e; e end
    @test err isa ErrorException && occursin("residual", err.msg)
    @test_throws ArgumentError ShiftInvert(block=4)
    @test_throws ArgumentError ShiftInvert(slices=0)
    @test_throws ArgumentError ShiftInvert(slices=:many)
    @test_throws ArgumentError ShiftInvert(slices=2, shift=1.0)            # slicing targets the lowest m
end

@testset "ShiftInvert: stiff leaves pass the true-residual check (polish step, roundoff floor)" begin
    # ‖H‖ grows like 1/h² on a grid and like ncut² in a charge basis. Lanczos stops the last pairs just under inner_tol on
    # the inverted operator, and H v − E v = −(E − σ)(H − σ) r turns that Krylov residual r, which lies in stiff
    # directions, into a true residual up to |E − σ|·‖H‖·inner_tol. Without the polish step these runs raised on 4 of the
    # 10 seeds (fluxonium, 2000 states, ‖H‖ ≈ 1.8e5) and on 2 of 10 (transmon, 2001 states, ‖H‖ ≈ 4.0e6).
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())    # the fluxonium grid warns on edge support
    flux = (load_spec(fixture("fluxonium.json")), (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π),
            [:varphi_1 => PhaseGrid(-6π, 6π, 2000; order=8)], 6)
    trans = (load_spec(fixture("transmon.json")), (E_J=20.0, E_C=1.0, n_g=0.3), [:varphi_1 => ChargeBasis(1000)], 20)
    for (spec, θ, ch, m) in (flux, trans)
        sd = quiet(() -> materialize(Node(ch; solver=Dense(), keep=lowest(m)), spec, θ))
        H = KronFreeCircuits.operator(sd)
        for seed in 1:10
            Random.seed!(seed)
            E, V = solve(ShiftInvert(), H, m)
            @test E ≈ energies(sd) atol=1e-9
            check_eigenpairs(H, E, V; tol=1e-8)
        end
    end
    # tol below double precision's reach: on the fluxonium grid eps·‖H‖ ≈ 4e-11, so true residuals of 3e-11 to 9e-11
    # exceed tol = 1e-11; they are accepted at the roundoff floor 10 eps·‖H‖ (debug message), not raised
    spec, θ, ch, m = flux
    sd = quiet(() -> materialize(Node(ch; solver=Dense(), keep=lowest(m)), spec, θ))
    H = KronFreeCircuits.operator(sd)
    A = sparse(H); A = (A + A') / 2
    floor = KronFreeCircuits._residual_floor(A)
    @test floor ≈ 10 * eps() * opnorm(A, Inf) && 1e-10 < floor < 1e-9
    for seed in 1:3
        Random.seed!(seed)
        logs, (E, V) = Test.collect_test_logs(min_level=Base.CoreLogging.Debug) do
            solve(ShiftInvert(tol=1e-11), H, m)
        end
        @test E ≈ energies(sd) atol=1e-9
        check_eigenpairs(H, E, V; tol=floor)
        @test count(r -> r.level == Base.CoreLogging.Debug && occursin("roundoff floor", string(r.message)), logs) == 1
    end
    # the floor does not hide a bad pair: a loose inner tolerance still raises, and the error names the floor
    Random.seed!(1)
    err = try solve(ShiftInvert(tol=1e-11, inner_tol=1e-5), H, m); nothing catch e; e end
    @test err isa ErrorException && occursin("roundoff floor", err.msg) && occursin("tighten inner_tol", err.msg)
end

@testset "ShiftInvert: complex path, explicit shifts, LU fallback" begin
    using KronFreeCircuits: _factor_shifted
    Random.seed!(82)
    # (a) a purely imaginary Hermitian coupling makes the assembled matrix complex
    d = 6
    K = zeros(ComplexF64, d, d); for i in 1:d-1; K[i, i+1] = im; K[i+1, i] = -im; end
    X = Matrix{ComplexF64}(I, d, d); for i in 1:d-1; X[i, i+1] = X[i+1, i] = 0.1; end
    terms = [AxisTerm(1.0 + i, [(i, DiagonalOp(collect(0.0:d-1)))]) for i in 1:3]
    push!(terms, AxisTerm(0.3, [(1, DenseOp(K)), (2, DenseOp(X))]))
    Hc = KronFreeCircuits.KronFreeOperator([d, d, d], terms)
    @test Matrix(Hc) ≈ Matrix(Hc)'
    @test eltype(Hc) === ComplexF64
    Ed, Vd = solve(Dense(), Hc, 0)
    for blk in (1, 2)
        E, V = solve(ShiftInvert(tol=1e-9, block=blk, certify=true), Hc, 10)
        @test E ≈ Ed[1:10] atol=1e-8
        check_eigenpairs(Hc, E, V; tol=1e-9)
        @test V * V' ≈ Vd[:, 1:10] * Vd[:, 1:10]' atol=1e-7
    end
    E, V = solve(ShiftInvert(tol=1e-9, slices=3), Hc, 10)                  # slicing in complex arithmetic
    @test E ≈ Ed[1:10] atol=1e-8
    check_eigenpairs(Hc, E, V; tol=1e-9)
    @test V * V' ≈ Vd[:, 1:10] * Vd[:, 1:10]' atol=1e-7

    H = real_three_oscillators(6)
    @test eltype(H) === Float64
    Ed, Vd = solve(Dense(), H, 0)
    # (b) an explicit interior shift returns the eigenvalues nearest it, which the certificate refuses
    σ = Ed[20] + 1e-3
    E, V = solve(ShiftInvert(tol=1e-9, shift=σ), H, 4)
    @test E ≈ Ed[17:20] atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
    err = try solve(ShiftInvert(tol=1e-9, shift=σ, certify=true), H, 4); nothing catch e; e end
    @test err isa ErrorException && occursin("certificate", err.msg)
    # (c) an explicit shift far below the spectrum
    E, V = solve(ShiftInvert(tol=1e-9, shift=Ed[1] - 20, certify=true), H, 5)
    @test E ≈ Ed[1:5] atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
    @test V * V' ≈ Vd[:, 1:5] * Vd[:, 1:5]' atol=1e-7
    # (d) H − σI indefinite: Cholesky raises PosDefException and LU takes over. An explicit shift just
    # above E_min is at or above the E_min estimate, so solve factors it by LU directly; the nearest
    # eigenvalues to it are then the lowest, and match Dense.
    A = real(sparse(H)); A = (A + A') / 2
    F, ischol = _factor_shifted(A, Ed[1] + 1e-3, false)
    @test !ischol && F isa SparseArrays.UMFPACK.UmfpackLU
    @test _factor_shifted(A, Ed[1] - 1, false)[2]
    E, V = solve(ShiftInvert(tol=1e-9, shift=Ed[1] + 1e-3, certify=true), H, 5)
    @test E ≈ Ed[1:5] atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
end

@testset "ShiftInvert: the automatic shift retries Cholesky at lower shifts" begin
    using KronFreeCircuits: _auto_factor, _lowest_estimate, _gershgorin_lower
    Warn = Base.CoreLogging.Warn
    # (a) the helper, on a tridiagonal matrix of known lowest eigenvalue, from a deliberately wrong estimate
    n = 20
    A = spdiagm(-1 => -ones(n - 1), 0 => fill(2.0, n), 1 => -ones(n - 1))
    λ1 = 2 - 2cos(π / (n + 1))
    @test _gershgorin_lower(A) ≤ λ1
    @test _gershgorin_lower(A) == 0.0
    # σ₀ = E_min − 1 lies above λ1, and the first retry at E_min − 4 lies below it
    F, σ, ischol = @test_logs min_level=Warn _auto_factor(A, λ1 + 1.5, 1.0)
    @test ischol && F isa SparseArrays.CHOLMOD.Factor
    @test σ ≈ λ1 + 1.5 - 4 && σ < λ1
    # E_min so far above λ1 that every 4^r retry fails: only the Gershgorin shift g − margin factors
    F, σ, ischol = @test_logs min_level=Warn _auto_factor(A, λ1 + 1000, 1.0)
    @test ischol && F isa SparseArrays.CHOLMOD.Factor
    @test σ == _gershgorin_lower(A) - 1.0 && σ < λ1
    # the estimate is accurate: no retry
    @test _auto_factor(A, λ1, 1.0)[2] == λ1 - 1.0
    # the retry reports the shift that worked at debug level
    logs, _ = Test.collect_test_logs(min_level=Base.CoreLogging.Debug) do
        _auto_factor(A, λ1 + 1.5, 1.0)
    end
    @test count(r -> r.level == Base.CoreLogging.Debug && occursin("retried", string(r.message)), logs) == 1
    # a complex Hermitian matrix: the Gershgorin bound takes the real diagonal and the complex off-diagonal moduli
    Ac = sparse(ComplexF64[2 im 0; -im 3 0.5; 0 0.5 1])
    @test _gershgorin_lower(Ac) == min(2 - 1, 3 - 1.5, 1 - 0.5)
    # absent diagonal entries count as zero: complex (no stored diagonal at all) and real (partly absent)
    Az = sparse(ComplexF64[0 0.5im; -0.5im 0])
    @test nnz(Az) == 2 && _gershgorin_lower(Az) == -0.5
    Ap = sparse([0.0 1 0; 1 2 0.5; 0 0.5 0])
    @test nnz(Ap) == 5 && _gershgorin_lower(Ap) == min(0 - 1, 2 - 1.5, 0 - 0.5)

    # (b) end to end: a short estimate whose error exceeds shift_margin, so the first Cholesky fails
    H = real_three_oscillators(8)
    Ed, _ = solve(Dense(), H, 0)
    As = sparse(H); As = (As + As') / 2
    # the seed with the largest overshoot of seeds 1:20, so the test does not hang on one RNG stream
    gaps = map(1:20) do seed
        Random.seed!(seed)
        _lowest_estimate(As, 1) - Ed[1]
    end
    seed = argmax(gaps)
    gaps[seed] > 1e-8 || error("no seed in 1:20 makes the short estimate overshoot λ1 (largest gap $(gaps[seed])); " *
                               "the end-to-end retry test needs a harder estimate")
    s = ShiftInvert(tol=1e-9, shift_margin=gaps[seed] / 2, maxiter=1, krylovdim=60)    # σ₀ above λ1: Cholesky fails
    Random.seed!(seed)
    logs, (E, V) = Test.collect_test_logs(min_level=Base.CoreLogging.Debug) do
        solve(s, H, 3)
    end
    @test !any(r -> r.level >= Warn, logs)
    @test count(r -> r.level == Base.CoreLogging.Debug && occursin("retried", string(r.message)), logs) == 1
    @test E ≈ Ed[1:3] atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
end

@testset "ShiftInvert: the default certificate recovers degenerate levels by deflation" begin
    Random.seed!(83)
    @test ShiftInvert().certify === :auto                              # decided in solve: certified unless an explicit interior shift
    @test ShiftInvert(shift=1.0).certify === :auto
    @test_throws ArgumentError ShiftInvert(certify=:always)
    # a 4-fold ground level: single-vector Lanczos resolves one copy per Krylov space, the inertia count sees all four
    H4 = KronFreeCircuits.KronFreeOperator([45], [AxisTerm(1.0, [(1, DiagonalOp(Float64[0, 0, 0, 0, 1:41...]))])])
    Ed, Vd = solve(Dense(), H4, 0)
    @test Ed[1:6] ≈ [0, 0, 0, 0, 1, 2] atol=1e-12
    for m in (5, 2, 4)                                                 # above, inside and at the top of the level
        E, V = solve(ShiftInvert(), H4, m)
        @test E ≈ Ed[1:m] atol=1e-8
        check_eigenpairs(H4, E, V; tol=1e-8)
        m != 2 && @test V * V' ≈ Vd[:, 1:m] * Vd[:, 1:m]' atol=1e-8     # m = 2 cuts the level: no unique projector
    end
    E, V = solve(ShiftInvert(block=2), H4, 5)
    @test E ≈ Ed[1:5] atol=1e-8
    check_eigenpairs(H4, E, V; tol=1e-8)
    # an explicit shift below the spectrum (Cholesky) is certified and recovered by default too
    for seed in 1:5
        Random.seed!(seed)
        E, V = solve(ShiftInvert(shift=-1.0), H4, 5)
        @test E ≈ Ed[1:5] atol=1e-8
        check_eigenpairs(H4, E, V; tol=1e-8)
    end
    # an explicit interior shift with certify = true: a short count raises, naming both possible causes
    errs = Exception[]
    for seed in 1:6
        Random.seed!(seed)
        try
            E, V = solve(ShiftInvert(shift=0.4, certify=true), H4, 5)
            @test E ≈ Ed[1:5] atol=1e-8                                # certified: then it is right
        catch e
            push!(errs, e)
        end
    end
    @test !isempty(errs)
    @test all(e -> e isa ErrorException && occursin("certificate", e.msg) && occursin("nearest σ", e.msg) &&
                   occursin("degenerate level", e.msg) && occursin("try block = 2", e.msg), errs)
    errs2 = Exception[]
    for seed in 1:6
        Random.seed!(seed)
        try solve(ShiftInvert(shift=0.4, certify=true, block=2), H4, 5) catch e; push!(errs2, e) end
    end
    @test all(e -> e isa ErrorException && !occursin("try block = 2", e.msg), errs2)      # no hint to use what is in use
    # an explicit shift exactly on an eigenvalue: H − σI is singular, refused naming the shift
    err = try solve(ShiftInvert(shift=0.0), H4, 5); nothing catch e; e end
    @test err isa ArgumentError && occursin("singular", err.msg) && occursin("σ = 0.0", err.msg)
    # nearly on an eigenvalue: the factorization succeeds, the residual check fails, and the message names the shift
    Random.seed!(1)
    err = try solve(ShiftInvert(shift=1 + 1e-15), H4, 5); nothing catch e; e end
    @test err isa ErrorException && occursin("nearly singular", err.msg) && occursin("σ = 1.000000000000001", err.msg)
    # The opt-out has no completeness check: single-vector Lanczos may undercount the 4-fold level and
    # return a wrong spectrum silently, so only the absence of an error is asserted.
    E0, V0 = solve(ShiftInvert(certify=false), H4, 5)
    @test length(E0) == 5
    # the 2-fold fixture of the LOBPCG "repeated eigenvalue" testset, with the default ShiftInvert
    H2 = KronFreeCircuits.KronFreeOperator([12], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))])])
    Ed2, Vd2 = solve(Dense(), H2, 3)
    for m in (2, 3)
        E2, V2 = solve(ShiftInvert(), H2, m)
        @test E2 ≈ Ed2[1:m] atol=1e-8
        @test V2 * V2' ≈ Vd2[:, 1:m] * Vd2[:, 1:m]' atol=1e-8
        check_eigenpairs(H2, E2, V2; tol=1e-9)
    end
    # every level 3-fold, in complex arithmetic: an imaginary hopping coupling two modes, and a third axis
    # of three degenerate states
    d = 6
    K = zeros(ComplexF64, d, d); for i in 1:d-1; K[i, i+1] = im; K[i+1, i] = -im; end
    X = Matrix{ComplexF64}(I, d, d); for i in 1:d-1; X[i, i+1] = X[i+1, i] = 0.1; end
    Hc = KronFreeCircuits.KronFreeOperator([d, d, 3], [AxisTerm(1.0, [(1, DiagonalOp(collect(0.0:d-1)))]),
                                                      AxisTerm(1.3, [(2, DiagonalOp(collect(0.0:d-1)))]),
                                                      AxisTerm(0.3, [(1, DenseOp(K)), (2, DenseOp(X))]),
                                                      AxisTerm(1.0, [(3, DiagonalOp(zeros(3)))])])
    @test eltype(Hc) === ComplexF64
    Edc, Vdc = solve(Dense(), Hc, 0)
    for m in (6, 7)
        E, V = solve(ShiftInvert(), Hc, m)
        @test E ≈ Edc[1:m] atol=1e-8
        check_eigenpairs(Hc, E, V; tol=1e-8)
        m == 6 && @test V * V' ≈ Vdc[:, 1:6] * Vdc[:, 1:6]' atol=1e-8
    end
end

@testset "ShiftInvert block = 2: an exhausted block Krylov space is not a breakdown" begin
    # 64 states in 15 distinct levels, every level above the ground degenerate: two start vectors span an invariant
    # block Krylov space of dimension at most 30, below the Krylov dimension. KrylovKit's block Lanczos stops there
    # only when the residual falls below tol; one between tol and its block-QR tolerance emptied the next block
    # (ArgumentError: blocklength must be >(0)) on about one seed in four.
    H = two_identical_real_oscillators(8)
    for Hx in (H, complexified(H))
        Ed, Vd = solve(Dense(), Hx, 0)
        # m = 10 tops levels 0–3 (1 + 2 + 3 + 4 states); m = 8 cuts the 4-fold level 3, leaving 6 in complete levels
        for (s, m) in ((ShiftInvert(block=2), 10), (ShiftInvert(block=2), 8), (ShiftInvert(block=2, slices=2), 10)),
            seed in 1:40
            Random.seed!(seed)
            r = try solve(s, Hx, m) catch e; e end
            @test !(r isa Exception)
            r isa Exception && continue
            E, V = r
            @test E ≈ Ed[1:m] atol=1e-8
            check_eigenpairs(Hx, E, V; tol=1e-8)
            c = m == 10 ? 10 : 6
            @test V[:, 1:c] * V[:, 1:c]' ≈ Vd[:, 1:c] * Vd[:, 1:c]' atol=1e-8
        end
    end
end

@testset "ShiftInvert: highly degenerate operators, few distinct levels" begin
    # identical uncoupled oscillators: every distinct level is reachable by `block` copies per Krylov space, so a
    # level holding more copies is recovered only by the certificate's deflated rounds
    osc(d, k) = KronFreeCircuits.KronFreeOperator(fill(d, k), [AxisTerm(1.0, [(i, DiagonalOp(collect(0.0:d-1)))]) for i in 1:k])
    # the number of states in complete levels among the lowest m: the projector on them is unique
    complete(Ed, m) = something(findlast(c -> Ed[c+1] - Ed[c] > 1e-6, 1:m), 0)
    four = KronFreeCircuits.KronFreeOperator([36], [AxisTerm(1.0, [(1, DiagonalOp(repeat(0.0:3; inner=9)))])])
    cases = (
        # KrylovKit's block Lanczos took sqrt of a slightly negative u'Ru in its residual norms (a DomainError) on
        # about half the seeds, when the residual block became rank-deficient
        ("osc3 d = 4, block = 2", osc(4, 3), ShiftInvert(block=2), 16),
        ("osc2 d = 6, interior shift, certify = true, block = 2", osc(6, 2), ShiftInvert(block=2, shift=0.5, certify=true), 5),
        # (distinct levels) × block < m + 1: the first solve converged fewer than m + 1 pairs (an error before the
        # certificate could deflate); with that error lifted, a deflated round also converged the deflated operator's
        # zero eigenvalue, a vector in the span of the computed ones (seed 12 of the 36-state block = 1 case)
        ("osc3 d = 4, block = 1", osc(4, 3), ShiftInvert(), 16),
        ("osc2 d = 4, block = 1", osc(4, 2), ShiftInvert(), 12),
        ("osc2 d = 4, block = 2", osc(4, 2), ShiftInvert(block=2), 12),
        ("36 states in 4 levels, block = 1", four, ShiftInvert(), 12),
        ("36 states in 4 levels, block = 2", four, ShiftInvert(block=2), 12),
    )
    for (name, H, s, m) in cases, Hx in (H, complexified(H))
        Ed, Vd = solve(Dense(), Hx, 0)
        c = complete(Ed, m)
        @testset "$name, $(eltype(Hx))" begin
            for seed in 1:30
                Random.seed!(seed)
                r = try solve(s, Hx, m) catch e; e end
                @test !(r isa Exception)
                r isa Exception && continue
                E, V = r
                @test E ≈ Ed[1:m] atol=1e-8
                check_eigenpairs(Hx, E, V; tol=1e-8)
                @test V[:, 1:c] * V[:, 1:c]' ≈ Vd[:, 1:c] * Vd[:, 1:c]' atol=1e-8
            end
        end
    end
end

@testset "ShiftInvert slicing equals the single-shift solve" begin
    Random.seed!(82)
    H = three_oscillators(7)                                   # 343 states
    E1, V1 = solve(ShiftInvert(tol=1e-9), H, 40)
    E3, V3 = solve(ShiftInvert(tol=1e-9, slices=3), H, 40)
    @test E3 ≈ E1 atol=1e-8; @test V3 * V3' ≈ V1 * V1' atol=1e-7
    check_eigenpairs(H, E3, V3; tol=1e-9)
    # :auto is Threads.nthreads() slices (1 on a single thread); the result does not depend on it
    Ea, Va = solve(ShiftInvert(tol=1e-9, slices=:auto), H, 40)
    @test Ea ≈ E1 atol=1e-8; @test Va * Va' ≈ V1 * V1' atol=1e-7
    check_eigenpairs(H, Ea, Va; tol=1e-9)
    # more slices than states: the empty slices collapse
    E5, V5 = solve(ShiftInvert(tol=1e-9, slices=5), H, 3)
    @test E5 ≈ E1[1:3] atol=1e-8; @test V5 * V5' ≈ V1[:, 1:3] * V1[:, 1:3]' atol=1e-7
    check_eigenpairs(H, E5, V5; tol=1e-9)
end

@testset "ChebyshevFilter equals Dense; seeds; bound safety" begin
    Random.seed!(91)
    H = three_oscillators(6)
    Ed, Vd = solve(Dense(), H, 0)
    for (seed, deg) in ((:product, 50), (:random, 50), (:product, 200))
        locked = Float64[]
        s = ChebyshevFilter(tol=1e-9, degree=deg, extra=12, seed=seed,
                            callback=info -> append!(locked, info.residuals[1:info.n_locked]))
        E, V = solve(s, H, 8)
        @test E ≈ Ed[1:8] atol=1e-7
        check_eigenpairs(H, E, V; tol=1e-8)
        @test V * V' ≈ Vd[:, 1:8] * Vd[:, 1:8]' atol=1e-6
        # locked at tol/2, accepted at tol: the fresh true residual keeps the margin (measured max 1.3e-10, :random seed)
        @test maximum(j -> norm(H * V[:, j] - E[j] * V[:, j]), eachindex(E)) < s.tol
        @test !isempty(locked) && maximum(locked) <= 0.5 * s.tol
    end
    θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    leaf = Node([:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(8))
    sd = @test_logs (:warn, r"edge") match_mode=:any materialize(leaf, spec_zp, θ)
    sc = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(leaf.children; solver=ChebyshevFilter(tol=1e-9), compute=8, keep=lowest(8)), spec_zp, θ)
    @test energies(sc) ≈ energies(sd) atol=1e-7
    # a user bound below the true λ_max is detected, not silently wrong
    λtrue = maximum(Ed)
    err = try solve(ChebyshevFilter(tol=1e-9, degree=100, bound=0.5 * λtrue), H, 8); nothing catch e; e end
    @test err isa ErrorException && occursin("bound", err.msg)
    @test_throws ArgumentError ChebyshevFilter(degree=0)
    @test_throws ArgumentError ChebyshevFilter(extra=1)       # no column could serve as a guard
    @test_throws ArgumentError ChebyshevFilter(extra=0)
    @test ChebyshevFilter(extra=2).extra == 2
end

# An exactly invariant product seed that excludes the ground state: product states of energy i + 10j on a
# 10 × 10 grid (unit vectors, since the single-axis terms are diagonal) and a two-axis coupling 82 between
# the product states of energy 50 and 51, whose block [50 82; 82 51] has eigenvalues ≈ −31.5 and 132.5.
# The seed of nb = 43 columns is the unit vectors of energies 0..42, an exactly invariant span (zero
# residuals) that contains none of the ground state.
# `pairs` lists couplings `(i1, i2, j, c)`: strength c between the product states (i1, j) and (i2, j)
# (axis-1 indices i1, i2 from 1, axis-2 level j from 0); `shifts` lists `(i, j, δ)`: δ added to the energy
# of the product state (i, j) by a diagonal two-axis term.
function _invariant_seed_trap(pairs=[(1, 2, 5, 82.0)]; shifts=[], T=ComplexF64)
    terms = AxisTerm[AxisTerm(1.0, [(1, DiagonalOp(collect(0.0:9)))]), AxisTerm(10.0, [(2, DiagonalOp(collect(0.0:9)))])]
    for (i1, i2, j, c) in pairs
        X = zeros(T, 10, 10); X[i1, i2] = X[i2, i1] = 1
        P = zeros(T, 10, 10); P[j+1, j+1] = 1
        push!(terms, AxisTerm(c, [(1, DenseOp(X)), (2, DenseOp(P))]))
    end
    for (i, j, δ) in shifts
        Pi = zeros(T, 10, 10); Pi[i, i] = 1
        Pj = zeros(T, 10, 10); Pj[j+1, j+1] = 1
        push!(terms, AxisTerm(δ, [(1, DenseOp(Pi)), (2, DenseOp(Pj))]))
    end
    return KronFreeCircuits.KronFreeOperator([10, 10], terms)
end

# The coupling c between product energies 50 and 51 that puts the lower eigenvalue of [50 c; c 51] at e.
_trap_coupling(e) = sqrt(((101 - 2e)^2 - 1) / 4)

@testset "ChebyshevFilter: a lower state outside an invariant seed span, near or among the seeded energies" begin
    variants = [("pushed to -0.001, just below the lowest seeded energy", [(1, 2, 5, _trap_coupling(-0.001))], (3,)),
                ("pushed exactly to 0, degenerate with a seeded state", [(1, 2, 5, _trap_coupling(0.0))], (2, 3)),
                ("pushed to 0.5, between seeded energies", [(1, 2, 5, _trap_coupling(0.5))], (3,)),
                ("coupling just outside the seed (energies 43, 44)", [(4, 5, 4, 60.0)], (3,)),
                ("3-fold coupling among energies 50, 51, 52", [(1, 2, 5, 60.0), (2, 3, 5, 60.0), (1, 3, 5, 60.0)], (3,))]
    for (name, pairs, ms) in variants
        H = _invariant_seed_trap(pairs)
        Ed, Vd = solve(Dense(), H, 0)
        @test Ed[1] < 0.75                                             # a state below the seeded energy 1, outside the seed span
        P = KronFreeCircuits.product_seed(H, 43)
        @test norm(H * P - P * (P' * (H * P))) == 0
        for m in ms, tol in (1e-8, 1e-4), extra in (40, 2)
            E, V = solve(ChebyshevFilter(tol=tol, extra=extra), H, m)
            @testset "$name, m = $m, tol = $tol, extra = $extra" begin
                @test E ≈ Ed[1:m] atol=10tol * max(1, maximum(abs, Ed[1:m]))
                check_eigenpairs(H, E, V; tol=2tol)
            end
        end
    end
    # a block one short of the space (nb = N − 1) with a degenerate top level: the guard columns stay below
    # the top column, which sets λ_cut, so the iteration converges (no LAPACK failure, no maxiter error)
    Hd = KronFreeCircuits.KronFreeOperator([10], [AxisTerm(1.0, [(1, DiagonalOp([0.0:7..., 9, 9]))])])
    for extra in (4, 3)
        E, V = solve(ChebyshevFilter(extra=extra), Hd, 5)
        @test E ≈ 0:4 atol=1e-8
        check_eigenpairs(Hd, E, V; tol=1e-8)
    end
end

@testset "ChebyshevFilter: guard columns degenerate with the top column (it sets λ_cut) do not block the stop" begin
    diagop(v) = KronFreeCircuits.KronFreeOperator([length(v)], [AxisTerm(1.0, [(1, DiagonalOp(Float64.(v)))])])
    # the two-transmon charge-basis leaf with identical transmons: every excited level is 2-fold
    spec = KronFreeCircuits.load_spec(fixture("two_transmons.json"))
    θt = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=30.0, E_C2=1.2, n_g2=0.25, g=0.0)
    tt = KronFreeCircuits.materialize(KronFreeCircuits.Node([:varphi_1 => KronFreeCircuits.ChargeBasis(8), :varphi_2 => KronFreeCircuits.ChargeBasis(8)];
                                                            solver=Dense(), keep=KronFreeCircuits.lowest(4)), spec, θt)
    cases = [("two transmons, ChargeBasis(8), m = 1", KronFreeCircuits.operator(tt), 1, 2),
             ("two transmons, ChargeBasis(8), m = 3", KronFreeCircuits.operator(tt), 3, 2),
             ("diag(0..4, 5, 5 + 1e-7, ...), m = 3", diagop([0, 1, 2, 3, 4, 5, 5 + 1e-7, 6:40...]), 3, 4),
             ("diag(0, 1, 1 + 1e-7, ...), m = 1", diagop([0, 1, 1 + 1e-7, 2:40...]), 1, 2),
             # a seeded pair at 2 and 2 + 10 tol, and a state at 0.5 missing from the seed span
             ("trap: pair at 2, 2 + 1e-7 and a missing state at 0.5, m = 3",
              _invariant_seed_trap([(1, 2, 5, _trap_coupling(0.5))]; shifts=[(4, 0, -1 + 1e-7)]), 3, 2)]
    for (name, H, m, extra) in cases
        Ed, _ = solve(Dense(), H, 0)
        @testset "$name, extra = $extra" begin
            E, V = solve(ChebyshevFilter(extra=extra), H, m)
            @test E ≈ Ed[1:m] atol=1e-7
            check_eigenpairs(H, E, V; tol=2e-8)
        end
    end
end

@testset "ChebyshevFilter: a uniformly weak filter (a large spectral bound) keeps its guards" begin
    # The invariant-seed trap with the missing state at `miss` and the product state (10, 10) lifted by `lift`:
    # the Gershgorin bound ≈ lift makes the filter's gain small at every column, the bottom one included.
    # Each case must match Dense to tol, or raise the maxiter error; never a wrong spectrum.
    tol = 1e-4
    converged = 0
    for lift in (1e5, 1e6), miss in (-1e-3, -1e-2)
        H = _invariant_seed_trap([(1, 2, 5, _trap_coupling(miss))]; shifts=[(10, 9, lift)])
        Ed, _ = solve(Dense(), H, 0)
        @test Ed[1] ≈ miss atol=1e-12
        for m in (1, 3), extra in (40, 10, 5)
            r = try solve(ChebyshevFilter(tol=tol, extra=extra, maxiter=400), H, m) catch e; e end
            @testset "lift = $lift, miss = $miss, m = $m, extra = $extra" begin
                if r isa Exception
                    @test r isa ErrorException && occursin("maxiter = 400", r.msg)
                else
                    E, V = r
                    @test E ≈ Ed[1:m] atol=tol
                    check_eigenpairs(H, E, V; tol=2tol)
                    converged += 1
                end
            end
        end
    end
    @test converged == 24                                              # all converge within maxiter = 400 (6–296 iterations)
    # the review's reproduction: lift 1e6, miss −0.01, defaults but tol = 1e-4, m = 1
    H = _invariant_seed_trap([(1, 2, 5, _trap_coupling(-0.01))]; shifts=[(10, 9, 1e6)])
    E, V = solve(ChebyshevFilter(tol=1e-4), H, 1)
    @test E ≈ [-0.01] atol=1e-4
    check_eigenpairs(H, E, V; tol=2e-4)
end

@testset "ChebyshevFilter: a deep ground state does not waive the guards protecting the states above it" begin
    # Axis 1 has energies [−D, 0, 1, …, 8], axis 2 [0, D + 2000, …]: the product ground state lies at −D, far
    # below the rest, and every seeded state above it at 0..7. A two-axis coupling between the product states
    # (2, 2) and (3, 2) (energies D + 2000 and D + 2001, outside the seed span) pushes their lower eigenvalue
    # to −0.5, below the seeded 0. Measured against the bottom Ritz value −D every guard's log-gain ratio was
    # ≈ 0.03, so every guard was waived from the first iteration and, with tol = 1e-3 and extra = 2 or 5, the
    # block [−D, 0] was accepted with passing residuals (the state at −0.5 missed). Each case must match Dense
    # or raise the maxiter error; never a wrong spectrum.
    D, miss = 1e4, -0.5
    ax1 = [-D; 0.0:8...]
    ax2 = [0.0; (D + 2000.0) .+ (0:8)]
    e1, e2 = ax1[2] + ax2[2], ax1[3] + ax2[2]
    c = sqrt(((e1 + e2) / 2 - miss)^2 - ((e2 - e1) / 2)^2)
    X = zeros(ComplexF64, 10, 10); X[2, 3] = X[3, 2] = 1
    Pj = zeros(ComplexF64, 10, 10); Pj[2, 2] = 1
    H = KronFreeCircuits.KronFreeOperator([10, 10], AxisTerm[AxisTerm(1.0, [(1, DiagonalOp(ax1))]),
                                                             AxisTerm(1.0, [(2, DiagonalOp(ax2))]),
                                                             AxisTerm(c, [(1, DenseOp(X)), (2, DenseOp(Pj))])])
    Ed, _ = solve(Dense(), H, 0)
    @test Ed[1:3] ≈ [-D, miss, 0] atol=1e-8
    tol = 1e-3
    converged = 0
    for extra in (2, 5)
        r = try solve(ChebyshevFilter(tol=tol, extra=extra, maxiter=400), H, 2) catch e; e end
        @testset "D = $D, miss = $miss, m = 2, tol = $tol, extra = $extra" begin
            if r isa Exception
                @test r isa ErrorException && occursin("maxiter = 400", r.msg) && occursin("extra", r.msg)
            else
                E, V = r
                @test E ≈ Ed[1:2] atol=10tol
                check_eigenpairs(H, E, V; tol=2tol)
                converged += 1
            end
        end
    end
    @test converged == 2                                               # both converge within maxiter = 400 (337 and 245 iterations)
    # at the default maxiter = 50 the guards have not converged: a loud error advising a larger extra, not [−D, 0]
    for extra in (2, 5)
        err = try solve(ChebyshevFilter(tol=tol, extra=extra), H, 2); nothing catch e; e end
        @test err isa ErrorException && occursin("maxiter = 50", err.msg) && occursin("raise extra", err.msg)
    end
end

@testset "ChebyshevFilter: an exactly invariant product seed that excludes the ground state" begin
    H = _invariant_seed_trap()
    Ed, Vd = solve(Dense(), H, 0)
    @test Ed[1:3] ≈ [(101 - sqrt(1 + 4 * 82^2)) / 2, 0, 1] atol=1e-10
    P = KronFreeCircuits.product_seed(H, 43)
    @test norm(H * P - P * (P' * (H * P))) == 0                        # the seed span is exactly invariant
    @test norm(P' * Vd[:, 1]) == 0                                     # and excludes the ground state
    for seed in (:product, :random)
        Random.seed!(93)
        its = Int[]
        E, V = solve(ChebyshevFilter(seed=seed, callback=info -> push!(its, info.iteration)), H, 3)
        @test E ≈ Ed[1:3] atol=1e-8
        check_eigenpairs(H, E, V; tol=1e-8)
        @test V * V' ≈ Vd[:, 1:3] * Vd[:, 1:3]' atol=1e-6
        @test last(its) >= 1                                           # at least one filter application
    end
    # No return at iteration 0, even from an exact eigenblock seed (a diagonal operator) with nb < N.
    Hdiag = KronFreeCircuits.KronFreeOperator([100], [AxisTerm(1.0, [(1, DiagonalOp(collect(0.0:99)))])])
    its = Int[]
    E, V = solve(ChebyshevFilter(extra=10, callback=info -> push!(its, info.iteration)), Hdiag, 3)
    @test E ≈ [0, 1, 2] atol=1e-8
    check_eigenpairs(Hdiag, E, V; tol=1e-8)
    @test last(its) >= 1
end

using KronFreeCircuits: _svqb!, _not_done_reason

@testset "ChebyshevFilter: SVQB drop and refill against locked columns; Cholesky-QR; applied degree" begin
    rng = MersenneTwister(92)
    N = 30
    L = Matrix(qr(randn(rng, ComplexF64, N, 4)).Q)                     # 4 locked orthonormal columns
    Z = randn(rng, ComplexF64, N, 10)
    Z[:, 4] = Z[:, 2]; Z[:, 7] = 2 .* Z[:, 2]; Z[:, 9] = Z[:, 1] .+ Z[:, 3]   # rank 7: three directions dropped and refilled
    P = Z - L * (L' * Z)
    W = _svqb!(similar(Z), copy(Z), L, similar(Z), rng)
    @test size(W, 2) == 10
    @test norm(W' * W - I) < 1e-12
    @test norm(L' * W) < 1e-12
    @test norm(P - W * (W' * P)) < 1e-10 * norm(P)                      # the kept directions span Z ⟂ L
    # A block wider than the space beside the locked columns (27 > N − 4): the refill finds no room
    # (each random column is dropped, the empty-pass path) and raises. solve cannot build this case,
    # since its active and locked columns together number nb ≤ N.
    Zw = randn(rng, ComplexF64, N, N - 3)
    err = try _svqb!(similar(Zw), Zw, L, similar(Zw), rng); nothing catch e; e end
    @test err isa ErrorException && occursin("refill", err.msg)
    # Cholesky-QR end to end; the narrow three-oscillator spectrum lowers the applied degree (the cap).
    H = three_oscillators(6)
    Ed, Vd = solve(Dense(), H, 0)
    degs = Int[]
    E, V = solve(ChebyshevFilter(tol=1e-9, ortho=:cholqr, callback=info -> push!(degs, info.degree)), H, 8)
    @test E ≈ Ed[1:8] atol=1e-7
    check_eigenpairs(H, E, V; tol=1e-8)
    @test V * V' ≈ Vd[:, 1:8] * Vd[:, 1:8]' atol=1e-6
    @test degs[1] == 0 && all(d -> 1 <= d < 100, degs[2:end])
    err = try solve(ChebyshevFilter(tol=1e-9, maxiter=1), H, 8); nothing catch e; e end
    @test err isa ErrorException && occursin("maxiter = 1", err.msg) && occursin("amplification cap", err.msg)
    # the maxiter reason when the lowest m are locked but a lower unlocked Ritz value is pending
    @test occursin("below the 3-th locked energy", _not_done_reason([1.0, 2.0, 3.0, 0.5], [0, 0, 0, 1e-3], 3, 3, 1e-9))
    # the reason when the wanted columns are locked but a counted guard is not: raise extra or maxiter
    r = _not_done_reason([1.0, 2.0, 3.0, 4.0, 5.0], [0, 0, 0, 1e-3, 1e-3], 3, 3, 1e-9; need=4)
    @test occursin("guard", r) && occursin("raise extra", r) && occursin("maxiter", r)
    @test !occursin("raise extra", _not_done_reason([1.0, 2.0, 3.0, 4.0], [0, 0, 1e-3, 1e-3], 2, 3, 1e-9; need=4))
end

using KronFreeCircuits: LOBPCG, lobpcg_solve, FixedJacobi, RitzJacobi, solver_callback, with_block_capacity, _full_diagonal,
                         _perturbed_seed
import LOBPCGEigensolver

@testset "LOBPCG agrees with Dense; all preconditioners; independent validation" begin
    Random.seed!(7)
    H = three_oscillators(6)                       # N = 216
    Ed, Vd = solve(Dense(), H, 0)
    for pc in (:none, :jacobi, :ritz)
        E, V = solve(LOBPCG(tol=1e-9, preconditioner=pc), H, 8)
        @test E ≈ Ed[1:8] atol=1e-7
        check_eigenpairs(H, E, V; tol=1e-8)
        @test V * V' ≈ Vd[:, 1:8] * Vd[:, 1:8]' atol=1e-7       # same subspace
        @test issorted(E) && typeof((E, V)) == Tuple{Vector{Float64},Matrix{ComplexF64}}
    end
    @test Array(_full_diagonal(H)) ≈ real.(diag(kron_three_oscillators(6)))   # independent of _device_diagonal
end

@testset "LOBPCG argument errors and nonconvergence" begin
    H = three_oscillators(3)                       # N = 27
    @test_throws ArgumentError solve(LOBPCG(), H, 8)          # 3*8+5 > 27
    err = try solve(LOBPCG(), H, 8); nothing catch e; e end
    @test occursin("Dense", err.msg)
    @test_throws ArgumentError solve(LOBPCG(), H, 0)
    @test_throws ArgumentError LOBPCG(preconditioner=:magic)
    # A device storage type without the CUDA extension loaded is rejected before any work.
    struct NotAnArray end
    if Base.get_extension(KronFreeCircuits, :KronFreeCircuitsCUDAExt) === nothing
        err = try solve(LOBPCG(on=NotAnArray), three_oscillators(6), 3); nothing catch e; e end
        @test err isa ArgumentError && occursin("CUDA", err.msg)
    end
    @test_throws ArgumentError LOBPCG(guard=-1)
    H6 = three_oscillators(6)
    err = try solve(LOBPCG(tol=1e-14, maxiter=1), H6, 5); nothing catch e; e end
    @test err isa ErrorException && occursin("converged", err.msg) && occursin("residual", err.msg)
    # the guard shrinks to fit: N = 27, m = 7 leaves room for g = 0
    E, V = solve(LOBPCG(tol=1e-10, guard=4), H, 7)
    @test E ≈ solve(Dense(), H, 7)[1] atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
end

@testset "repeated eigenvalue is recovered twice (projector, not just energies)" begin
    H = KronFreeOperator([12], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))])])
    Random.seed!(3)
    E, V = solve(LOBPCG(tol=1e-10, guard=0, seed=:random), H, 2)
    Ed, Vd = solve(Dense(), H, 2)
    @test E ≈ [0.0, 0.0] atol=1e-8
    @test Ed ≈ [0.0, 0.0] atol=1e-12
    @test V * V' ≈ Vd * Vd' atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
    # the perturbed product seed of a diagonal operator takes the direct path: no retry is logged, even at debug level
    (logs, (E2, V2)) = Test.collect_test_logs(min_level=Base.CoreLogging.Debug) do
        solve(LOBPCG(tol=1e-10, guard=0), H, 2)
    end
    @test !any(r -> occursin("retrying", string(r.message)), logs)
    @test E2 ≈ [0.0, 0.0] atol=1e-8
    @test V2 * V2' ≈ Vd * Vd' atol=1e-8
    check_eigenpairs(H, E2, V2; tol=1e-9)
end

@testset "product-seeded failure: retry is logged once at debug level; warm start composition" begin
    # H = cI has exactly zero residuals for every block, so the product-seeded attempt and the random retry both fail:
    # one debug-level retry record, then the informative error.
    Random.seed!(43)                                 # the random retry's block (and so its failure) is then deterministic, as in S3
    Hc = KronFreeOperator([30], AxisTerm[]; constant=2.5 + 0im)   # complex, as before: the real draw of the retry block converges
    ret = Ref{Any}(nothing)
    logs, _ = Test.collect_test_logs(min_level=Base.CoreLogging.Debug) do
        ret[] = try solve(LOBPCG(), Hc, 3); nothing catch e; e end
    end
    @test count(r -> r.level == Base.CoreLogging.Debug && occursin("retrying", string(r.message)), logs) == 1
    @test ret[] isa ErrorException && occursin("Dense", ret[].msg)
    # the initial block of a warm start: X0 first, then the next product states by energy as guards
    Random.seed!(77)
    H = three_oscillators(4); N = size(H, 1); m, g = 3, 4; T = eltype(H)
    P = product_seed(H, m + g)
    X0 = Matrix(qr(randn(T, N, m)).Q)
    first, guards = KronFreeCircuits._seed_columns(T, N, m, X0, P, Random.default_rng())
    @test isapprox(first, X0; atol=1e-6) && first != X0
    @test isapprox(guards, P[:, m+1:end]; atol=1e-6) && guards != P[:, m+1:end]
    first, guards = KronFreeCircuits._seed_columns(T, N, m, nothing, P, Random.default_rng())      # cold product block
    @test isapprox(first, P[:, 1:m]; atol=1e-6) && isapprox(guards, P[:, m+1:end]; atol=1e-6)
    first, guards = KronFreeCircuits._seed_columns(T, N, m, X0, nothing, Random.default_rng())     # seed=:random: random guards
    @test isapprox(first, X0; atol=1e-6) && guards === nothing
    first, guards = KronFreeCircuits._seed_columns(T, N, m, nothing, nothing, Random.default_rng())
    @test size(first) == (N, m) && guards === nothing
    # end to end: the warm start with product guards still returns the lowest m
    E, V = solve(LOBPCG(tol=1e-9), H, m; X0=X0)
    @test E ≈ solve(Dense(), H, m)[1] atol=1e-7
    check_eigenpairs(H, E, V; tol=1e-8)
end

@testset "user callbacks of any return type run with the original info" begin
    Random.seed!(9)
    H = three_oscillators(6)
    seen = Int[]
    for cb in (info -> push!(seen, info.n_iter), info -> nothing, info -> info)
        E, V = solve(LOBPCG(tol=1e-9, preconditioner=:ritz, callback=cb), H, 6)
        @test E ≈ solve(Dense(), H, 6)[1] atol=1e-7
        check_eigenpairs(H, E, V; tol=1e-8)
    end
    @test length(seen) >= 2                          # the first callback recorded iterations
    # an exception from the user's callback propagates unchanged, with no recovery retry
    for X0 in (nothing, randn(ComplexF64, size(H, 1), 6))
        err = @test_logs (try solve(LOBPCG(callback=info -> error("user abort")), H, 6; X0=X0); nothing catch e; e end)
        @test err isa ErrorException && err.msg == "user abort"
    end
end

@testset "warm starts: exact lowest, exact excited, rank-deficient and sector-confined seeds" begin
    Random.seed!(11)
    H = three_oscillators(6)
    Ed, Vd = solve(Dense(), H, 0)
    E, V, cold = lobpcg_solve(LOBPCG(tol=1e-9), H, 6)
    Ew, Vw, warm = lobpcg_solve(LOBPCG(tol=1e-9), H, 6; X0=V)
    @test warm.n_matvec < cold.n_matvec
    check_eigenpairs(H, E, V; tol=1e-8)
    check_eigenpairs(H, Ew, Vw; tol=1e-8)
    @test Ew ≈ Ed[1:6] atol=1e-7
    # an exact warm start from Dense eigenvectors: no recovery warning, same energies, less work than cold
    Ex, Vx, exact = @test_logs lobpcg_solve(LOBPCG(tol=1e-9), H, 6; X0=Vd[:, 1:6])
    @test Ex ≈ Ed[1:6] atol=1e-8
    @test exact.n_matvec < cold.n_matvec
    check_eigenpairs(H, Ex, Vx; tol=1e-8)
    for X0 in (Vd[:, 1:6],                          # exact lowest eigenvectors
               Vd[:, 3:8],                          # exact excited subspace: guards must bring the lowest two back
               repeat(Vd[:, 1:1], 1, 6))            # rank-deficient block
        E2, V2 = solve(LOBPCG(tol=1e-10, guard=4), H, 6; X0=X0)
        @test E2 ≈ Ed[1:6] atol=1e-8
        @test V2 * V2' ≈ Vd[:, 1:6] * Vd[:, 1:6]' atol=1e-7
        check_eigenpairs(H, E2, V2; tol=1e-9)
    end
    # sector-confined seed: a diagonal operator whose even sites hold the ground state; N = 40 leaves
    # room for the full guard (g = min(4, (40 - 5) ÷ 3 - 3) = 4)
    d = zeros(40); d[1:2:40] .= 2:2:40; d[2:2:40] .= 0.5:2:38.5
    Hs = KronFreeOperator([40], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(d)))])])
    X0 = Matrix{ComplexF64}(I, 40, 40)[:, 1:2:5]     # exact eigenvectors of odd sites (energies 2, 4, 6): zero residual
    # the exact seed has a zero residual; the seed perturbation keeps the eigensolver from failing on it
    Es, Vs = @test_logs solve(LOBPCG(tol=1e-10, guard=4), Hs, 3; X0=X0)
    @test Es ≈ [0.5, 2.0, 2.5] atol=1e-8               # the guards must find site 2 (0.5) and site 4 (2.5)
    check_eigenpairs(Hs, Es, Vs; tol=1e-9)
end

@testset "Ritz cache stays aligned with the solver's active columns, through locking" begin
    Random.seed!(5)
    H = three_oscillators(6); m = 10
    D = real.(diagonal(H))
    P = RitzJacobi(D, m; eta=0.05, trace=true)
    seen = Any[]
    cb = info -> push!(seen, (info.n_iter, info.n_locked, copy(info.λs)))
    X0 = randn(ComplexF64, size(H, 1), m)
    Hb = with_block_capacity(H, m)
    res = LOBPCGEigensolver.lobpcg(Hb, X0, I, P, 1e-9, 300; callback=solver_callback(P, cb))
    @test res.λ ≈ solve(Dense(), H, m)[1] atol=1e-7
    check_eigenpairs(H, res.λ, res.X; tol=1e-8)
    @test length(seen) >= 5                                   # enough iterations to mean something
    @test any(w < m for (w, _) in P.trace)                    # locking actually shrank the active block
    @test seen[end][2] == m
    # From the second prepare onward, the cached values for the active columns are exactly the
    # previous iteration's Ritz values for those same columns (locked prefix removed).
    for t in 2:min(length(P.trace), length(seen))
        width, theta = P.trace[t]
        _, _, λs = seen[t - 1]
        @test theta == λs[end - width + 1:end]
    end
end

@testset "with_block_capacity shares operator data, never scratch" begin
    Random.seed!(40)
    H = three_oscillators(4)
    Hb = with_block_capacity(H, 7)
    @test Hb.diag === H.diag && Hb.terms === H.terms && Hb.inds === H.inds && Hb.dims === H.dims
    @test size(Hb.buf1) == size(Hb.buf2) == (4, 4, 4, 7)
    for (a, b) in ((Hb.buf1, H.buf1), (Hb.buf1, H.buf2), (Hb.buf2, H.buf1), (Hb.buf2, H.buf2), (Hb.buf1, Hb.buf2))
        @test !Base.mightalias(a, b)
    end
    X = randn(ComplexF64, 64, 7)
    @test Hb * X ≈ kron_three_oscillators(4) * X
    @test_throws ArgumentError with_block_capacity(H, 0)
end

@testset "warm-start perturbation size, guard nonconvergence warning, eta in the diagonal's type" begin
    Random.seed!(39)
    for N in (50, 2000), T in (ComplexF64, Float64)
        X0 = randn(T, N, 3) .* [1e-3 1 1e3]
        Xp = _perturbed_seed(T, X0)
        for j in 1:3
            @test norm(Xp[:, j] - X0[:, j]) ≈ sqrt(eps(real(T))) * norm(X0[:, j]) rtol=1e-6
        end
        @test _perturbed_seed(T, X0) == Xp                     # deterministic, independent of the global RNG
    end
    # the lowest m converge but maxiter runs out before the guards do: a warning, not silence
    H = three_oscillators(6)
    Ed, Vd = solve(Dense(), H, 0)
    E, V = @test_logs (:warn, r"guard columns did not converge") solve(LOBPCG(tol=1e-9, maxiter=2), H, 2; X0=Vd[:, 1:2])
    @test E ≈ Ed[1:2] atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-8)
    # RitzJacobi stores eta in the diagonal's real type
    P = RitzJacobi(Float32[1, 2, 3], 2; eta=0.01)
    @test P.eta isa Float32
end

# ---- Pathological-test catalogue (docs/spikes/2026-10-16-pathological-test-catalogue), Task 4 cases ----

# A single-axis operator from a dense Hermitian matrix (independent of the operator under test).
dense_axis_operator(A::AbstractMatrix) = KronFreeOperator([size(A, 1)], [AxisTerm(1.0, [(1, DenseOp(A))])])
diagonal_axis_operator(d::AbstractVector) = KronFreeOperator([length(d)], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(d)))])])
random_unitary(N) = Matrix(qr(randn(ComplexF64, N, N)).Q)
function hermitian_with_spectrum(Q, d)
    A = Q * Diagonal(d) * Q'
    return (A + A') / 2
end

# Records the block width of every operator application lobpcg makes.
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
    # exact lowest and exact excited eigenvectors have exactly zero residuals; the seed perturbation
    # keeps them usable, so no recovery retry (and no warning) is needed
    for (X0, label) in ((Id[:, 1:3], "lowest"), (Id[:, 6:8], "excited"))
        E, V = @test_logs solve(LOBPCG(tol=1e-10), H, 3; X0=X0)
        @test E ≈ d[1:3] atol=1e-8
        @test V * V' ≈ Id[:, 1:3] * Id[:, 1:3]' atol=1e-8
        check_eigenpairs(H, E, V; tol=1e-9)
    end
    # unchanged-H reuse from the solver's own (numerically, not exactly, converged) output: no retry
    E, V = solve(LOBPCG(tol=1e-10), H, 3)
    E2, V2 = @test_logs solve(LOBPCG(tol=1e-10), H, 3; X0=V)
    @test E2 ≈ d[1:3] atol=1e-8
    check_eigenpairs(H, E2, V2; tol=1e-9)
    # bounded: on H = c I every block has a zero residual, so the retry fails too and an informative error propagates
    Hc = KronFreeOperator([30], AxisTerm[]; constant=2.5 + 0im)   # complex, as before: the real draw of the retry block converges
    err = @test_logs (:warn, r"warm-start") (try solve(LOBPCG(), Hc, 3; X0=randn(ComplexF64, 30, 3)); nothing catch e; e end)
    @test err isa ErrorException && occursin("Dense", err.msg)
end

@testset "S2: zero, duplicate, nearly dependent and badly scaled seeds" begin
    Random.seed!(42)
    H = three_oscillators(6)
    Ed, Vd = solve(Dense(), H, 0)
    # an all-zero block (or column) is rejected up front: the eigensolver's orthogonalization never ends on it
    @test_throws ArgumentError solve(LOBPCG(), H, 4; X0=zeros(ComplexF64, 216, 4))
    X0 = randn(ComplexF64, 216, 4); X0[:, 3] .= 0
    @test_throws ArgumentError solve(LOBPCG(), H, 4; X0=X0)
    for X0 in (repeat(Vd[:, 1:1], 1, 4),                                      # duplicate columns
               repeat(Vd[:, 1:1], 1, 4) .+ 1e-10 .* randn(ComplexF64, 216, 4),  # nearly dependent
               randn(ComplexF64, 216, 4) .* [1e-8 1 1e8 1e3])                 # arbitrary column scales
        E, V = solve(LOBPCG(tol=1e-9), H, 4; X0=X0)
        @test E ≈ Ed[1:4] atol=1e-7
        @test V * V' ≈ Vd[:, 1:4] * Vd[:, 1:4]' atol=1e-7
        check_eigenpairs(H, E, V; tol=1e-8)
    end
end

@testset "S3: H = 0, H = cI, tight clusters and a cutoff inside a cluster" begin
    Random.seed!(43)
    # every block has an exactly zero residual: informative rejection pointing to Dense (which handles them)
    for H in (KronFreeOperator([30], AxisTerm[]), KronFreeOperator([30], AxisTerm[]; constant=2.5))
        err = try solve(LOBPCG(), H, 3); nothing catch e; e end
        @test err isa ErrorException && occursin("Dense", err.msg)
        @test solve(Dense(), H, 3)[1] ≈ fill(H.diag[1], 3)
    end
    N = 40
    Q = random_unitary(N)
    rest = collect(range(1.0, 10.0; length=N - 3))
    for δ in (1e-3, 1e-6, 1e-9, 0.0)                     # complete cluster {0, δ, 2δ}, down to an exact triple
        H = dense_axis_operator(hermitian_with_spectrum(Q, [0, δ, 2δ, rest...]))
        E, V = solve(LOBPCG(tol=1e-9), H, 3)
        @test E ≈ [0, δ, 2δ] atol=1e-8
        @test V * V' ≈ Q[:, 1:3] * Q[:, 1:3]' atol=1e-7   # the cluster subspace, not individual vectors
        check_eigenpairs(H, E, V; tol=1e-8)
    end
    # cutoff m = 2 inside the exact triple {1, 1, 1}: no unique rank-2 target projector
    H = dense_axis_operator(hermitian_with_spectrum(Q, [0, 1, 1, 1, collect(range(2.0, 10.0; length=N - 4))...]))
    E, V = solve(LOBPCG(tol=1e-9), H, 2)
    @test E ≈ [0, 1] atol=1e-8
    @test norm(V * (V' * Q[:, 1]) - Q[:, 1]) < 1e-7       # the lower complete subspace is contained
    Pb = Q[:, 2:4] * Q[:, 2:4]'
    @test norm(Pb * V[:, 2] - V[:, 2]) < 1e-7             # the boundary vector lies in the boundary eigenspace
    check_eigenpairs(H, E, V; tol=1e-8)
end

@testset "S4: exact resonance and invalid regularization" begin
    D = [1.0, 2.0, 3.0, 4.0]
    P = RitzJacobi(D, 2; eta=0.1)
    X = zeros(ComplexF64, 4, 2); X[2, 1] = 1; X[3, 2] = 1   # diagonal expectations exactly D[2] and D[3]
    LOBPCGEigensolver.precondprep!(P, X)
    @test P.theta == [2.0, 3.0]
    R = ones(ComplexF64, 4, 2)
    ldiv!(P, R)
    @test all(isfinite, R)
    @test R[2, 1] == 1 / 0.1 && R[3, 2] == 1 / 0.1        # D[i] == θ_j gives R / η
    @test R[1, 1] ≈ 1 / hypot(1.0, 0.1)
    for bad in (0.0, -1.0, NaN, Inf)                      # rejected at construction, before any iteration
        @test_throws ArgumentError LOBPCG(eta=bad)
        @test_throws ArgumentError RitzJacobi(D, 2; eta=bad)
    end
    # a very small positive eta is a stress case: valid output or explicit nonconvergence
    Random.seed!(44)
    H = three_oscillators(6)
    out = try solve(LOBPCG(tol=1e-9, eta=1e-8), H, 4) catch e; e end
    if out isa Exception
        @test out isa ErrorException && occursin("converged", out.msg)
    else
        @test out[1] ≈ solve(Dense(), H, 4)[1] atol=1e-7
        check_eigenpairs(H, out...; tol=1e-8)
    end
end

@testset "S5: no locking, several locks at once, one active column, complete locking" begin
    H = three_oscillators(6); m = 10
    D = real.(diagonal(H))
    Ed = solve(Dense(), H, m)[1]
    # aligned cache check: from the second prepare on, the cache holds the previous iteration's Ritz values
    function aligned(P, seen)
        all(P.trace[t][2] == seen[t-1][2][end-P.trace[t][1]+1:end] for t in 2:min(length(P.trace), length(seen)))
    end
    # no locking: two iterations at a tolerance nothing reaches
    Random.seed!(45)
    P = RitzJacobi(D, m; eta=0.05, trace=true)
    seen = Any[]
    LOBPCGEigensolver.lobpcg(with_block_capacity(H, m), randn(ComplexF64, 216, m), I, P, 1e-14, 2;
                             callback=solver_callback(P, info -> push!(seen, (info.n_locked, copy(info.λs)))))
    @test all(==(0), first.(seen)) && all(w == m for (w, _) in P.trace) && length(P.trace) == 3
    @test aligned(P, seen)
    # full runs, several seeds: record every lock step and every block width the operator sees
    multi = false; single = false
    for seed in 1:3
        Random.seed!(seed)
        P = RitzJacobi(D, m; eta=0.05, trace=true)
        W = WidthRecorder(with_block_capacity(H, m), Int[])
        seen = Any[]
        res = LOBPCGEigensolver.lobpcg(W, randn(ComplexF64, 216, m), I, P, 1e-9, 300;
                                       callback=solver_callback(P, info -> push!(seen, (info.n_locked, copy(info.λs)))))
        @test res.λ ≈ Ed atol=1e-7
        check_eigenpairs(H, res.λ, res.X; tol=1e-8)
        @test seen[end][1] == m                                   # complete locking
        @test aligned(P, seen)
        @test minimum(W.widths) >= 1 && minimum(first.(P.trace)) >= 1   # no zero-width application or preconditioning
        multi |= any(>=(2), diff(first.(seen)))
        single |= any(w == 1 for (w, _) in P.trace)
    end
    @test multi                                                   # several columns locked in one iteration
    @test single                                                  # a single remaining active column was iterated
end

@testset "S6: aH + bI, scaled tolerance and eta; finite shift below the diagonal" begin
    Random.seed!(46)
    cos_op, nvec, ω = _three_osc_pieces(6)
    function scaled(a, b)
        terms = [[AxisTerm(a * ω[i], [(i, DiagonalOp(nvec))]) for i in 1:3];
                 [AxisTerm(a * 0.2, [(i, DenseOp(cos_op)), (j, DenseOp(cos_op))]) for (i, j) in ((1, 2), (2, 3), (1, 3))]]
        return KronFreeOperator([6, 6, 6], terms; constant=b)
    end
    a, b = 1000.0, 7.0e4
    H1, H2 = scaled(1.0, 0.0), scaled(a, b)
    E1, V1 = solve(LOBPCG(tol=1e-9), H1, 5)
    for pc in (:ritz, :jacobi)
        E2, V2 = solve(LOBPCG(tol=1e-9 * a, eta=0.01 * a, preconditioner=pc), H2, 5)
        @test E2 ≈ a .* E1 .+ b atol=1e-6 * a
        @test V2 * V2' ≈ V1 * V1' atol=1e-7
        check_eigenpairs(H2, E2, V2; tol=1e-8 * a)
    end
    P = FixedJacobi(_full_diagonal(H2))
    @test all(isfinite, P.inv) && all(x -> 0 < x <= 1, P.inv)   # σ = min(D) − 1 strictly below every D[i]
    @test_throws ArgumentError FixedJacobi(1e17 .+ collect(0.0:9.0))   # min(D) − 1 rounds to min(D)
    @test_throws ArgumentError FixedJacobi([1.0, NaN])
end

@testset "S7: constant diagonal, strong off-diagonal coupling" begin
    Random.seed!(47)
    A = randn(ComplexF64, 60, 60); A = (A + A') / 2; A[diagind(A)] .= 3.0
    H = dense_axis_operator(A)
    F = eigen(Hermitian(A))
    for pc in (:none, :jacobi, :ritz)
        out = try lobpcg_solve(LOBPCG(tol=1e-9, preconditioner=pc), H, 4) catch e; e end
        if out isa Exception
            @test out isa ErrorException && occursin("converged", out.msg)
        else
            E, V, res = out
            @test E ≈ F.values[1:4] atol=1e-8
            @test V * V' ≈ F.vectors[:, 1:4] * F.vectors[:, 1:4]' atol=1e-7
            check_eigenpairs(H, E, V; tol=1e-8)
            @test res.n_matvec > 0
        end
    end
end

@testset "S8: dimensions, arguments and explicit nonconvergence" begin
    Random.seed!(48)
    d = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    H8 = diagonal_axis_operator(d)                        # N = 8 = 3m + 5 for m = 1
    E, V = solve(LOBPCG(tol=1e-10), H8, 1)
    @test E ≈ [1.0] atol=1e-8
    check_eigenpairs(H8, E, V; tol=1e-9)
    @test_throws ArgumentError solve(LOBPCG(), diagonal_axis_operator(d[1:7]), 1)   # N = 3m + 4
    for m in (0, -1, 9)
        @test_throws ArgumentError solve(LOBPCG(), H8, m)
    end
    for kw in ((tol=0.0,), (tol=-1e-8,), (tol=NaN,), (tol=Inf,), (maxiter=0,))
        @test_throws ArgumentError LOBPCG(; kw...)
    end
    H = three_oscillators(6)
    @test_throws DimensionMismatch solve(LOBPCG(), H, 4; X0=randn(ComplexF64, 215, 4))
    @test_throws DimensionMismatch solve(LOBPCG(), H, 4; X0=randn(ComplexF64, 216, 3))
    for bad in (NaN, Inf)
        X0 = randn(ComplexF64, 216, 4); X0[5, 2] = bad
        @test_throws ArgumentError solve(LOBPCG(), H, 4; X0=X0)
    end
    err = try solve(LOBPCG(tol=1e-12, maxiter=2), H, 4); nothing catch e; e end
    @test err isa ErrorException && occursin("4 of 4 eigenpairs did not converge", err.msg) && occursin("residual", err.msg)
end

@testset "F1: free rotor through n_g = 1/2 (operator level)" begin
    EC = 1.2; ncut = 10; N = 2ncut + 1
    n = collect(-ncut:ncut)
    levels(ng) = sort(4EC .* (n .- ng) .^ 2)               # oracle: the sorted diagonal
    rotor(ng; EJ=0.0) = KronFreeOperator([N], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(4EC .* (n .- ng) .^ 2)))]),
                                               AxisTerm(-EJ / 2, [(1, ShiftOp(N, 1, 1.0 + 0im))]),
                                               AxisTerm(-EJ / 2, [(1, ShiftOp(N, -1, 1.0 + 0im))])])
    charge(k) = (v = zeros(ComplexF64, N); v[k+ncut+1] = 1; v)
    proj(ks) = (B = reduce(hcat, charge.(ks)); B * B')
    Random.seed!(51)
    # n_g = 0: exact ±n pairs; m = 5 is {0}, {±1}, {±2}
    E, V = solve(LOBPCG(tol=1e-10), rotor(0.0), 5)
    @test E ≈ levels(0.0)[1:5] atol=1e-8
    @test V * V' ≈ proj([0, -1, 1, -2, 2]) atol=1e-8
    check_eigenpairs(rotor(0.0), E, V; tol=1e-9)
    # n_g = 1/2: n = 0 and n = 1 degenerate; m = 4 is {0, 1}, {−1, 2}
    E, V = solve(LOBPCG(tol=1e-10), rotor(0.5), 4)
    @test E ≈ levels(0.5)[1:4] atol=1e-8
    @test V * V' ≈ proj([0, 1, -1, 2]) atol=1e-8
    check_eigenpairs(rotor(0.5), E, V; tol=1e-9)
    # n_g = 1/2 ∓ 1e-6: the lowest charge state switches from 0 to 1
    for (ng, k) in ((0.5 - 1e-6, 0), (0.5 + 1e-6, 1))
        E, V = solve(LOBPCG(tol=1e-10), rotor(ng), 4)
        @test E ≈ levels(ng)[1:4] atol=1e-9
        @test abs2(dot(charge(k), V[:, 1])) ≈ 1 atol=1e-8
        check_eigenpairs(rotor(ng), E, V; tol=1e-9)
    end
    # sweep through the crossing with the previous eigenvectors of both crossing states
    _, Vb = solve(LOBPCG(tol=1e-10), rotor(0.5 - 1e-6), 2)
    E, V = solve(LOBPCG(tol=1e-10), rotor(0.5 + 1e-6), 2; X0=Vb)
    @test E ≈ levels(0.5 + 1e-6)[1:2] atol=1e-9
    @test abs2(dot(charge(1), V[:, 1])) ≈ 1 atol=1e-8
    check_eigenpairs(rotor(0.5 + 1e-6), E, V; tol=1e-9)
    # ground state only: the old ground state stays an exact eigenstate after the crossing (wrong invariant subspace)
    _, Vg = solve(LOBPCG(tol=1e-10), rotor(0.5 - 1e-6), 1)
    E, V = solve(LOBPCG(tol=1e-10), rotor(0.5 + 1e-6), 1; X0=Vg)
    check_eigenpairs(rotor(0.5 + 1e-6), E, V; tol=1e-9)          # a valid eigenpair ...
    # ... and the lowest: the seed column locks at once, but the guards are converged too and the
    # lower state they find sorts first
    @test E ≈ levels(0.5 + 1e-6)[1:1] atol=1e-9
    @test abs2(dot(charge(1), V[:, 1])) ≈ 1 atol=1e-8
    # the exactly converged old ground state (zero residual) needs no recovery either
    E, V = @test_logs solve(LOBPCG(tol=1e-10), rotor(0.5 + 1e-6), 1; X0=reshape(charge(0), N, 1))
    @test E ≈ levels(0.5 + 1e-6)[1:1] atol=1e-9
    _, Vd1 = solve(Dense(), rotor(0.5 + 1e-6), 1)
    @test abs2(dot(Vd1[:, 1], V[:, 1])) ≈ 1 atol=1e-8           # the same state as Dense (charge 1)
    @test abs2(dot(charge(1), V[:, 1])) ≈ 1 atol=1e-8
    check_eigenpairs(rotor(0.5 + 1e-6), E, V; tol=1e-9)
    # small E_J: an avoided crossing, against a dense oracle
    for ng in (0.5 - 1e-6, 0.5, 0.5 + 1e-6)
        H = rotor(ng; EJ=0.1)
        F = eigen(Hermitian(Matrix(Diagonal(ComplexF64.(4EC .* (n .- ng) .^ 2))) - 0.05 * (diagm(1 => ones(N - 1)) + diagm(-1 => ones(N - 1)))))
        E, V = solve(LOBPCG(tol=1e-10), H, 4)
        @test E ≈ F.values[1:4] atol=1e-8
        @test V * V' ≈ F.vectors[:, 1:4] * F.vectors[:, 1:4]' atol=1e-6
        check_eigenpairs(H, E, V; tol=1e-9)
    end
end

@testset "F2: cos 2φ charge-parity sectors (operator level)" begin
    EC = 1.2; E2 = 3.0
    function cos2(nv, ng; E1=0.0)
        N = length(nv)
        terms = [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(4EC .* (nv .- ng) .^ 2)))]),
                 AxisTerm(-E2 / 2, [(1, ShiftOp(N, 2, 1.0 + 0im))]), AxisTerm(-E2 / 2, [(1, ShiftOp(N, -2, 1.0 + 0im))])]
        E1 == 0 || append!(terms, [AxisTerm(-E1 / 2, [(1, ShiftOp(N, 1, 1.0 + 0im))]),
                                   AxisTerm(-E1 / 2, [(1, ShiftOp(N, -1, 1.0 + 0im))])])
        return KronFreeOperator([N], terms)
    end
    function cos2_dense(nv, ng; E1=0.0)                    # independent oracle, entry by entry
        A = Matrix{ComplexF64}(Diagonal(4EC .* (nv .- ng) .^ 2))
        for i in eachindex(nv), j in eachindex(nv)
            abs(nv[i] - nv[j]) == 2 && (A[i, j] -= E2 / 2)
            abs(nv[i] - nv[j]) == 1 && (A[i, j] -= E1 / 2)
        end
        return Hermitian(A)
    end
    Random.seed!(52)
    nv = collect(-20:20); N = length(nv); m = 4           # N = 41 leaves room for the full guard
    ev = iseven.(nv)
    for ng in (0.2, 0.5)
        H = cos2(nv, ng); A = cos2_dense(nv, ng)
        Esec = sort([eigvals(Hermitian(A[ev, ev])); eigvals(Hermitian(A[.!ev, .!ev]))])   # sectors separately, merged
        F = eigen(A)
        @test F.values[1:m] ≈ Esec[1:m]
        X0 = zeros(ComplexF64, N, m); X0[ev, :] .= randn(ComplexF64, count(ev), m)       # seed confined to even parity
        for seed in (nothing, X0)
            E, V = solve(LOBPCG(tol=1e-10), H, m; X0=seed)
            @test E ≈ Esec[1:m] atol=1e-8
            @test V * V' ≈ F.vectors[:, 1:m] * F.vectors[:, 1:m]' atol=1e-6
            check_eigenpairs(H, E, V; tol=1e-9)
        end
    end
    # exact pairing at n_g = 1/2 needs a window invariant under n -> 1 - n, e.g. -M:M+1
    nr = collect(-10:11); evr = iseven.(nr)
    A = cos2_dense(nr, 0.5)
    @test eigvals(Hermitian(A[evr, evr])) ≈ eigvals(Hermitian(A[.!evr, .!evr]))
    H = cos2(nr, 0.5)
    E, V = solve(LOBPCG(tol=1e-10), H, 4)
    @test E[1] ≈ E[2] atol=1e-8
    @test E[3] ≈ E[4] atol=1e-8
    F = eigen(A)
    @test V * V' ≈ F.vectors[:, 1:4] * F.vectors[:, 1:4]' atol=1e-7   # complete pairs
    check_eigenpairs(H, E, V; tol=1e-9)
    # a cos φ term couples the sectors: the ground state carries weight on both parities
    H = cos2(nv, 0.2; E1=0.5); A = cos2_dense(nv, 0.2; E1=0.5)
    E, V = solve(LOBPCG(tol=1e-10), H, m)
    @test E ≈ eigvals(A)[1:m] atol=1e-8
    @test sum(abs2, V[ev, 1]) > 1e-3 && sum(abs2, V[.!ev, 1]) > 1e-3
    check_eigenpairs(H, E, V; tol=1e-9)
end

@testset "LOBPCG product-state seed cuts the cold start" begin
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    work = Dict{Symbol,Int}()
    tree(seed) = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))];
                      solver=LOBPCG(tol=1e-10, seed=seed, callback=info -> (work[seed] = info.n_matvec; nothing)), compute=6, keep=lowest(6))
    Random.seed!(5)
    sp = materialize(tree(:product), spec, θ); sr = materialize(tree(:random), spec, θ)
    @test energies(sp) ≈ energies(sr) atol=1e-9
    @test work[:product] < work[:random]
    check_eigenpairs(KronFreeCircuits.operator(sp), energies(sp), eigenvectors(sp); tol=1e-9)
    @test_throws ArgumentError LOBPCG(seed=:bogus)
end

if get(ENV, "KRONFREE_SLOW_TESTS", "0") == "1"      # gated (and the file is slow-tier): about 3 s; the degenerate and invariant-seed probes below run every solver in real arithmetic
@testset "solvers run in the operator's element type; real equals complex" begin
    for (name, H) in (("three real oscillators", real_three_oscillators(6)),
                      ("degenerate pair", two_identical_real_oscillators(12)))
        Hc = complexified(H)
        @test eltype(H) === Float64 && eltype(Hc) === ComplexF64
        Ed, Vd = solve(Dense(), H, 0); Edc, Vdc = solve(Dense(), Hc, 0)
        @test eltype(Vd) === Float64 && eltype(Vdc) === ComplexF64
        @test Ed ≈ Edc atol=1e-12
        for s in (Lanczos(), LOBPCG(), ShiftInvert(), ShiftInvert(block=2),
                  ChebyshevFilter(extra=name == "degenerate pair" ? 10 : 40))
            name == "degenerate pair" && s isa Lanczos && continue      # documented: single-vector Krylov misses copies
            E, V = solve(s, H, 10)
            @test eltype(V) === Float64
            @test E ≈ Ed[1:10] atol=1e-8
            check_eigenpairs(H, E, V; tol=1e-8)
            Ec, Vc = solve(s, Hc, 10)
            @test eltype(Vc) === ComplexF64
            @test Ec ≈ E atol=1e-8
            cut = findlast(<(Ed[10] + 1e-6), Ed)
            name == "degenerate pair" && @test cut == 10                # the projector check below runs
            cut == 10 && @test V * V' ≈ Vd[:, 1:10] * Vd[:, 1:10]' atol=1e-6
        end
    end
end
end

@testset "real solvers: a 4-fold degenerate ground level" begin
    w = [0.0 1 0 0 0; 1 0.5 1 0 0; 0 1 1.5 1 0; 0 0 1 1.0 1; 0 0 0 1 2.0]
    h = zeros(10, 10); h[1:5, 1:5] = w; h[6:10, 6:10] = w                  # two identical decoupled wells
    H = KronFreeCircuits.KronFreeOperator([10, 10], [AxisTerm(1.0, [(1, DenseOp(h))]), AxisTerm(1.0, [(2, DenseOp(h))])])
    @test eltype(H) === Float64
    Ed, Vd = solve(Dense(), H, 0)
    @test Ed[4] - Ed[1] < 1e-12 && Ed[5] - Ed[4] > 0.1                      # exactly 4-fold, then a gap
    for s in (ShiftInvert(), ShiftInvert(block=2), LOBPCG(), ChebyshevFilter(extra=10))
        E, V = solve(s, H, 4)
        @test eltype(V) === Float64
        @test E ≈ Ed[1:4] atol=1e-8
        check_eigenpairs(H, E, V; tol=1e-8)
        @test V * V' ≈ Vd[:, 1:4] * Vd[:, 1:4]' atol=1e-6                  # all four copies
    end
end

@testset "real ChebyshevFilter: the invariant product seed excludes the ground state" begin
    H = _invariant_seed_trap(; T=Float64)
    @test eltype(H) === Float64
    Ed, Vd = solve(Dense(), H, 0)
    P = KronFreeCircuits.product_seed(H, 43)
    @test eltype(P) === Float64
    @test norm(H * P - P * (P' * (H * P))) == 0
    @test norm(P' * Vd[:, 1]) == 0
    for m in (3, 6), seed in (:product, :random)
        Random.seed!(94)
        E, V = solve(ChebyshevFilter(seed=seed), H, m)
        @test eltype(V) === Float64
        @test E ≈ Ed[1:m] atol=1e-8
        check_eigenpairs(H, E, V; tol=1e-8)
        m == 3 && @test V * V' ≈ Vd[:, 1:3] * Vd[:, 1:3]' atol=1e-6
    end
end

@testset "real solvers: a near crossing, two levels 1e-7 apart at m and m + 1" begin
    Random.seed!(95)
    Q = Matrix(qr(randn(10, 10)).Q)
    a = Q * Diagonal(collect(0.0:9)) * Q'; a = (a + a') / 2                  # dense real, levels 0, 1, ..., 9
    b = [0.0, 2 + 1e-7, 3.3, 4.9, 5.7, 6.1, 7.7, 8.3, 9.1, 10.4]
    H = KronFreeCircuits.KronFreeOperator([10, 10], [AxisTerm(1.0, [(1, DenseOp(a))]), AxisTerm(1.0, [(2, DiagonalOp(b))])])
    @test eltype(H) === Float64
    Ed, Vd = solve(Dense(), H, 0)
    @test Ed[4] - Ed[3] ≈ 1e-7 atol=1e-9
    for s in (Lanczos(), ShiftInvert(), ShiftInvert(block=2), LOBPCG(), ChebyshevFilter())
        E4, V4 = solve(s, H, 4)                                             # the whole cluster
        @test eltype(V4) === Float64
        @test E4 ≈ Ed[1:4] atol=1e-8
        check_eigenpairs(H, E4, V4; tol=1e-8)
        @test V4 * V4' ≈ Vd[:, 1:4] * Vd[:, 1:4]' atol=1e-6
        E3, V3 = solve(s, H, 3)                                             # the cut inside the cluster
        @test E3 ≈ Ed[1:3] atol=1e-6
        check_eigenpairs(H, E3, V3; tol=1e-6)
        @test norm(Vd[:, 1:2]' * V3) ≈ sqrt(2) atol=1e-6                      # the two levels below the cluster are in the span
        @test norm(Vd[:, 5:end]' * V3) < 1e-6                               # nothing above the cluster
    end
end

@testset "LOBPCG warm start: complex block on a real operator" begin
    H = real_three_oscillators(6); m = 4
    Ed, _ = solve(Dense(), H, 0)
    _, V0 = solve(LOBPCG(tol=1e-9), H, m)
    # An exactly real complex block is converted and used (no retry warning), and the result stays real.
    E, V = @test_logs solve(LOBPCG(tol=1e-9), H, m; X0=ComplexF64.(V0))
    @test eltype(V) === Float64
    @test E ≈ Ed[1:m] atol=1e-7
    check_eigenpairs(H, E, V; tol=1e-8)
    # A genuinely complex block is rejected, naming both types.
    X0 = ComplexF64.(V0); X0[1, 1] += 0.1im
    err = try solve(LOBPCG(tol=1e-9), H, m; X0=X0); nothing catch e; e end
    @test err isa ArgumentError && occursin("ComplexF64", err.msg) && occursin("Float64", err.msg) && occursin("real", err.msg)
    # A real block on a complex operator still converts.
    Hc = complexified(H)
    Ec, Vc = solve(LOBPCG(tol=1e-9), Hc, m; X0=V0)
    @test eltype(Vc) === ComplexF64
    @test Ec ≈ Ed[1:m] atol=1e-7
end
