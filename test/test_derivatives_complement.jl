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
    # a known set that skips a lower state: H − E_3 is not positive on the complement, which holds state 2
    @test_throws "a state below or degenerate with the kept state was not computed (or was not retained)" _complement_solve(
        H, _full_diagonal(H), F.vectors[:, [1, 3, 4]], B[:, 1:3], F.values[[1, 3, 4]]; tol=1e-10, maxiter=5000)
    # a right-hand side almost inside the known states (‖Q B_j‖ ≈ 1e-9 ‖B_j‖): `tol ‖Q B_j‖` is below round-off, so every
    # column stops at the round-off floor (measured 7e-15 ‖B_j‖ on the real tree), and is still the derivative to 2.4e-6
    Bf = Φ * randn(MersenneTwister(2), 6, 4) .+ 1e-9 .* (F.vectors[:, 7:64] * randn(MersenneTwister(3), eltype(M), 58, 4))
    Xf, _, resf, fsf, atf, frf = _complement_solve(H, _full_diagonal(H), Φ, Bf, s; tol=1e-10, maxiter=5000)
    @test isempty(fsf) && atf == 1:4 && all(iszero, resf) && maximum(frf) <= 1000 * eps()
    for j in 1:4
        ref = sum(F.vectors[:, l] * (F.vectors[:, l]' * Bf[:, j]) / (F.values[l] - s[j]) for l in 7:64)
        @test norm(Xf[:, j] - ref) <= 1e-4 * norm(ref)
    end
end

@testset "weak coupling: a child-internal parameter's right-hand side lies almost inside the known states" begin
    # (∂H/∂E_J2) ψ_j ≈ (∂E_child) ψ_j: ‖Q R_j‖/‖R_j‖ ≈ 5e-5; each column converges to lintol on its own scale ‖Q R_j‖
    tw(solver) = dc_two_transmons(; leafkeep=12, solver, compute=8, keep=lowest(6), g=0.03)
    di = derivatives(dc_solve(tw(LOBPCG(tol=1e-11))); wrt=:E_J2)
    dd = derivatives(dc_solve(tw(Dense())); wrt=:E_J2)
    @test all(s -> s.residual <= 1e-9 && isempty(s.at_floor), di.linear_solves)      # 10 · lintol, relative to ‖Q R_j‖
    @test energies(di, :E_J2) ≈ energies(dd, :E_J2) rtol=1e-8 atol=1e-9
    @test norm(eigenvectors(di, :E_J2) - eigenvectors(dd, :E_J2), Inf) <= 1e-6 * norm(eigenvectors(dd, :E_J2), Inf)
    for p in (nport(), expiphi(1.0))
        @test norm(port(di, :E_J2, :varphi_1, p) - port(dd, :E_J2, :varphi_1, p), Inf) <= 1e-6 * norm(port(dd, :E_J2, :varphi_1, p), Inf)
    end
    # each state's derivative on its own scale (the matrix-wide comparison above cannot see a small column); columns
    # whose dense derivative is below 1e-12 in norm are skipped (`percol_relerr`)
    @test maximum(percol_relerr(eigenvectors(di, :E_J2), eigenvectors(dd, :E_J2))) <= 1e-7
    # at the ordinary coupling the column of state 1 (‖ψ′‖ = 4.9e-5) is small against ‖R_1‖ = 0.86: a convergence scale of
    # ‖R_j‖ stopped it at 9.9e-9 of its own norm; on the complement's own scale it is 4.6e-12
    t3(solver) = dc_two_transmons(; leafkeep=12, solver, compute=8, keep=lowest(6), g=0.3)
    d3 = derivatives(dc_solve(t3(LOBPCG(tol=1e-11))); wrt=:E_J2)
    @test maximum(percol_relerr(eigenvectors(d3, :E_J2), eigenvectors(derivatives(dc_solve(t3(Dense())); wrt=:E_J2), :E_J2))) <= 1e-9
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
    dE = @test_logs (:warn, r"^Node on modes \[:phi\]: kept states \[2\] are degenerate") (:warn, r"^selection boundary:") derivatives(s0; wrt=:E_2)      # -cos 2φ couples |−1⟩ and |+1⟩
    @test isnan(energies(dE, :E_2)[2]) && isfinite(energies(dE, :E_2)[1])
    @test only(dE.boundary).kept == 2 && only(dE.boundary).unkept == 3
    @test only(dE.boundary).coupling ≈ 0.5 atol=1e-12
end

@testset "a re-diagonalized node whose selection boundary cuts an exactly degenerate pair" begin
    # the cos 2φ element at E_1 = E_2 = n_g = 0 on charges -3:3: energies 0, 4, 4, 16 | 16, 36, 36; keeping four states
    # cuts the pair at 16. The re-diagonalization's partner of kept state 4 is made orthogonal to it, so the column of
    # state 1 (coupled to |±2⟩ by cos 2φ) counts the pair once, as in the retaining twin, and no spurious clash appears.
    mk(retain) = (root = Node([:phi => ChargeBasis(3)]; solver=Dense(), compute=4, keep=lowest(4), retain);
                  (; root, spec=load_spec(fixture("cos2phi.json")), θ=(E_C=1.0, n_g=0.0, E_1=0.0, E_2=0.0)))
    never, always = dc_solve(mk(:never)), dc_solve(mk(:always))
    @test energies(never) ≈ [0, 4, 4, 16] atol=1e-12
    pair = r"^Node on modes \[:phi\]: kept states \[2, 3\] are degenerate"                 # cos 2φ couples |±1⟩ directly
    dn = @test_logs (:info, r"re-diagonalized") (:warn, pair) (:warn, r"^selection boundary:") derivatives(never; wrt=:E_2)
    da = @test_logs (:warn, pair) (:warn, r"^selection boundary:") derivatives(always; wrt=:E_2)
    @test energies(dn, :E_2)[1] ≈ energies(da, :E_2)[1] atol=1e-12
    @test eigenvectors(dn, :E_2)[:, 1] ≈ eigenvectors(da, :E_2)[:, 1] atol=1e-12
    dg = @test_logs (:info, r"re-diagonalized") (:warn, r"^selection boundary:") derivatives(never; wrt=:n_g)
    # the @info is emitted once per node, however many calls re-diagonalize it
    io = IOBuffer()
    Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(io, Base.CoreLogging.Info)) do
        derivatives(never; wrt=:n_g); derivatives(never; wrt=:n_g)
    end
    @test count("re-diagonalized", String(take!(io))) == 1
    @test all(isfinite, energies(dg, :n_g))
end
