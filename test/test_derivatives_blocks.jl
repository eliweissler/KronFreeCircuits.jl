using Test, LinearAlgebra, KronFreeCircuits

@testset "the doublet of the cos 2φ element: report, true point derivatives, effective Hamiltonian" begin
    # detuning Δ = 0.05932237 GHz, coupling c = |⟨1|∂H/∂E_1|2⟩| = 0.74562858, so c·δx/Δ = 0.63 at δx = 0.05
    c = dc_cos2phi(keep=lowest(2)); sys = dc_solve(c); E0 = energies(sys)
    d = derivatives(sys; wrt=:E_1, order=2, scale=(E_1=0.05,))
    b = only(blocks(d))
    @test b isa NearDegenerateBlock && b.node === c.root && b.parameters == (:E_1,) && b.states == [1, 2] && !b.exact && b.order == 2
    @test b.detuning ≈ 0.05932237 atol=1e-6
    @test b.coupling ≈ 0.74562858 atol=1e-6
    @test b.energies ≈ E0
    # the point derivatives are the true ones: E_1 = 0 is a sweet spot with curvature 4c²/Δ of the transition frequency
    E1, E2 = energies(d, :E_1), energies(d, :E_1, :E_1)
    @test maximum(abs, E1) < 1e-9
    # measured when the plan was written: the finite-difference curvature is 37.48731 and 4c²/Δ is 37.48751, 5e-6 apart
    # (the doublet's couplings to the levels 9 GHz away)
    @test E2[2] - E2[1] ≈ 4 * b.coupling^2 / b.detuning rtol=1e-4
    check_fd(E2, fd_second(θ -> energies(dc_solve(c, θ)), c.θ, :E_1; h=2e-3); rtol=1e-5)          # a step small against Δ/c = 0.08
    # at δ = 0.05 the second-order Taylor expansion is off by more than 1e-3 GHz, and the effective Hamiltonian is not
    for (δ, tol) in ((0.05, 1e-6), (0.02, 5e-8))
        exact = energies(dc_solve(c, shifted(c.θ, :E_1, δ)))
        @test block_levels(b, (E_1=δ,)).energies ≈ exact atol=tol
    end
    @test maximum(abs, E0 .+ 0.05 .* E1 .+ 0.5 * 0.05^2 .* E2 .- energies(dc_solve(c, shifted(c.θ, :E_1, 0.05)))) > 1e-3
    @test_throws ArgumentError block_levels(b, (n_g=0.01,))
    # first order only
    b1 = only(blocks(derivatives(sys; wrt=:E_1, scale=(E_1=0.05,))))
    @test b1.order == 1 && isempty(b1.H2)
    @test block_levels(b1, (E_1=0.01,)).energies ≈ energies(dc_solve(c, shifted(c.θ, :E_1, 0.01))) atol=1e-5
    # the threshold: |c| δx > r Δ
    @test isempty(blocks(derivatives(sys; wrt=:E_1, scale=(E_1=0.001,))))                 # 7.5e-4 < 0.1 · 0.0593
    @test length(blocks(derivatives(sys; wrt=:E_1, scale=(E_1=0.001,), r=0.01))) == 1     # 7.5e-4 > 0.01 · 0.0593
    @test isempty(blocks(derivatives(sys; wrt=:E_1)))                                     # no scale: only the exact guard
    # a coupling that vanishes by symmetry removes the pair: ∂H/∂n_g conserves the Cooper-pair parity of the doublet
    @test isempty(blocks(derivatives(sys; wrt=:n_g, scale=(n_g=0.01,))))
end

@testset "a mixed pair: the union of the two parameters' blocks" begin
    c = dc_cos2phi(keep=lowest(2)); sys = dc_solve(c)
    d = derivatives(sys; wrt=(:E_1, :n_g), order=2, mixed=((:E_1, :n_g),), scale=(E_1=0.05, n_g=0.01))
    @test sort([b.parameters for b in blocks(d)]) == [(:E_1,), (:E_1, :n_g)]
    bm = only(b for b in blocks(d) if b.parameters == (:E_1, :n_g))
    @test bm.states == [1, 2] && haskey(bm.H2, (:E_1, :n_g)) && haskey(bm.H2, (:n_g, :n_g)) && haskey(bm.H1, :n_g)
    θd = merge(c.θ, (E_1=0.03, n_g=0.492))
    @test block_levels(bm, (E_1=0.03, n_g=0.002)).energies ≈ energies(dc_solve(c, θd)) atol=1e-6
end

@testset "exact degeneracy with a coupling: undefined per state, defined for the block" begin
    c = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(0), E_2=0.0); sys = dc_solve(c)     # energies 0, 4, 4
    d = quiet(() -> derivatives(sys; wrt=:E_2))
    b = only(blocks(d))
    @test b.exact && b.states == [2, 3] && b.detuning <= 1e-9
    @test b.coupling ≈ 0.5 atol=1e-12
    @test isnan.(energies(d, :E_2)) == [false, true, true]
    @test abs(tr(b.H1[:E_2])) < 1e-12                                                    # the block-invariant first derivative
    @test block_levels(b, (E_2=0.1,)).energies ≈ energies(dc_solve(c, shifted(c.θ, :E_2, 0.1)))[2:3] atol=1e-12
    @test isempty(blocks(derivatives(sys; wrt=:n_g)))                                    # degenerate but uncoupled: not a block
    # −cos φ couples |−1⟩ and |+1⟩ only through |0⟩: no block at first order, an exact block at second order, whose
    # effective Hamiltonian is 4 + (δ²/16) [1 1; 1 1], with levels 4 and 4 + δ²/8
    @test isempty(blocks(derivatives(sys; wrt=:E_1)))
    d2 = quiet(() -> derivatives(sys; wrt=:E_1, order=2))
    b2 = only(blocks(d2))
    @test b2.exact && b2.states == [2, 3] && b2.parameters == (:E_1,) && b2.order == 2
    @test isnan.(energies(d2, :E_1, :E_1)) == [false, true, true]
    @test energies(d2, :E_1, :E_1)[1] ≈ -0.25 atol=1e-12
    @test all(isfinite, energies(d2, :E_1))
    @test block_levels(b2, (E_1=0.1,)).energies ≈ [4.0, 4.0 + 0.1^2 / 8] atol=1e-12
    @test block_levels(b2, (E_1=0.1,)).energies ≈ energies(dc_solve(c, shifted(c.θ, :E_1, 0.1)))[2:3] atol=1e-6
end

@testset "a block created by a second-order clash gets the pieces of the passes already run" begin
    # the pass (n_g, n_g) runs before (E_1, E_1), whose clash creates the mixed block: its piece (n_g, n_g) (8 I from
    # ∂²H/∂n_g² = 8 E_C) is formed again from rebuilt operators; without it the levels below would be off by 4e-4
    c = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(0), E_2=0.0); sys = dc_solve(c)
    d = quiet(() -> derivatives(sys; wrt=(:n_g, :E_1), order=2, mixed=((:E_1, :n_g),)))
    @test sort([b.parameters for b in blocks(d)]) == [(:E_1,), (:E_1, :n_g)]
    bm = only(b for b in blocks(d) if b.parameters == (:E_1, :n_g))
    @test bm.exact && bm.states == [2, 3] && bm.order == 2
    @test bm.H2[(:n_g, :n_g)] ≈ 8I atol=1e-12
    @test bm.H2[(:E_1, :E_1)] ≈ fill(0.125, 2, 2) atol=1e-12
    @test norm(bm.H2[(:E_1, :n_g)]) < 1e-12
    @test sort(real(diag(bm.H1[:n_g]))) ≈ [-8.0, 8.0] atol=1e-12
    # the error of the effective Hamiltonian is cubic in the displacement (measured: 1.3e-5 and 1.0e-7)
    for (δ1, δg, tol) in ((0.1, 0.01, 2e-5), (0.02, 0.002, 2e-7))
        exact = energies(dc_solve(c, merge(c.θ, (E_1=δ1, n_g=δg))))[2:3]
        @test block_levels(bm, (E_1=δ1, n_g=δg)).energies ≈ exact atol=tol
    end
end
