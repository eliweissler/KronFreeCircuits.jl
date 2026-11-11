using Test, LinearAlgebra, Random, KronFreeCircuits
using KronFreeCircuits: _complement_solve, _full_diagonal, with_block_capacity, operator

@testset "the projected solve against the sum over the remaining states, real and complex" for periodic in (false, true)
    c = dc_two_transmons(; periodic, leafkeep=8); sys = dc_solve(c)                  # root dimension 64, real or complex
    H = with_block_capacity(operator(sys), 4); M = Matrix(operator(sys)); F = eigen(Hermitian(M))
    Φ = F.vectors[:, 1:6]; s = F.values[1:4]                                         # the six lowest states projected out
    B = randn(MersenneTwister(1), eltype(M), 64, 4)
    X, iterations, residuals = _complement_solve(H, _full_diagonal(H), Φ, B, s; tol=1e-12, maxiter=5000)
    for j in 1:4
        ref = sum(F.vectors[:, l] * (F.vectors[:, l]' * B[:, j]) / (F.values[l] - s[j]) for l in 7:64)
        @test norm(X[:, j] - ref) <= 1e-8 * norm(ref)
    end
    @test norm(Φ' * X) < 1e-10 && maximum(residuals) <= 1e-11 && iterations < 5000           # 10 · tol
    @test all(iszero, _complement_solve(H, _full_diagonal(H), Φ, Φ[:, 1:4], s; tol=1e-10, maxiter=10)[1])   # Q B = 0: x = 0
    @test_throws "did not converge" _complement_solve(H, _full_diagonal(H), Φ, B, s; tol=1e-10, maxiter=2)
end

@testset "a node that keeps every state of its space needs no solve and no re-diagonalization" begin
    # retain = :never: `retained.complete` is false, but the kept vectors span the space (dimension 4)
    ref = derivatives(dc_solve(dc_two_transmons(; leafkeep=2, keep=lowest(4))); wrt=:g)
    for solver in (Lanczos(tol=1e-12), Dense())
        sys = dc_solve(dc_two_transmons(; leafkeep=2, solver, compute=4, keep=lowest(4), retain=:never))
        @test !sys.root.retained.complete
        d = @test_logs derivatives(sys; wrt=:g)                                     # no @info of a re-diagonalization
        @test isempty(d.linear_solves)
        @test energies(d, :g) ≈ energies(ref, :g) atol=1e-10
    end
end

@testset "selection-boundary warning" begin
    # the doublet of the cos 2φ element with only its lower state kept: detuning 0.0593, coupling 0.7456 to E_1
    c = dc_cos2phi(keep=lowest(1)); sys = dc_solve(c)
    d = @test_logs (:warn, r"^selection boundary:") match_mode=:any derivatives(sys; wrt=:E_1, scale=(E_1=0.05,))
    b = only(d.boundary)
    @test b.node === c.root && b.parameter === :E_1 && b.kept == 1 && b.unkept == 2
    @test b.detuning ≈ 0.0593 atol=1e-3
    @test b.coupling ≈ 0.7456 atol=1e-3
    @test isempty(derivatives(sys; wrt=:E_1, scale=(E_1=0.001,)).boundary)           # 0.7456·0.001 < 0.1·0.0593
    @test isempty(derivatives(sys; wrt=:E_1).boundary)                               # no scale: the plain relative-gap test
    # without a scale, an exactly degenerate kept/unkept pair is reported whatever its coupling
    c0 = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(2), E_2=0.0); s0 = dc_solve(c0)   # energies 0, 4 | 4
    d0 = @test_logs (:warn, r"^selection boundary:") match_mode=:any derivatives(s0; wrt=:n_g)
    @test only(d0.boundary).kept == 2 && only(d0.boundary).unkept == 3 && isnan(only(d0.boundary).coupling)
    @test all(isfinite, energies(d0, :n_g))                                          # -8 E_C n does not couple the pair
    # a kept state degenerate with an unkept one and coupled to it: NaN, and the boundary record is its report
    dE = quiet(() -> derivatives(s0; wrt=:E_2))                                      # -cos 2φ couples |−1⟩ and |+1⟩
    @test isnan(energies(dE, :E_2)[2]) && isfinite(energies(dE, :E_2)[1])
    @test only(dE.boundary).kept == 2 && only(dE.boundary).unkept == 3
    @test only(dE.boundary).coupling ≈ 0.5 atol=1e-12
end
