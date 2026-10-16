using Random
using KronFreeCircuits: Dense, Lanczos, solve

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
