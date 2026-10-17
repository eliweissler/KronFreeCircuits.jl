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
    E, V = solve(LOBPCG(tol=1e-10, guard=0), H, 2)
    Ed, Vd = solve(Dense(), H, 2)
    @test E ≈ [0.0, 0.0] atol=1e-8
    @test Ed ≈ [0.0, 0.0] atol=1e-12
    @test V * V' ≈ Vd * Vd' atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
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
    Hc = KronFreeOperator([30], AxisTerm[]; constant=2.5)
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
