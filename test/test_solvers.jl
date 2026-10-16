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
