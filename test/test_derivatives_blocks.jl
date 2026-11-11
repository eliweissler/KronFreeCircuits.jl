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
    # the block report does not alter the point derivatives (design §4.4): the same numbers with and without `scale`
    plain = derivatives(sys; wrt=(:E_1, :n_g), order=2, mixed=((:E_1, :n_g),))
    for x in (:E_1, :n_g)
        @test energies(d, x) == energies(plain, x) && eigenvectors(d, x) == eigenvectors(plain, x)
    end
    for (x, λ) in ((:E_1, :E_1), (:n_g, :n_g), (:E_1, :n_g))
        @test energies(d, x, λ) == energies(plain, x, λ) && eigenvectors(d, x, λ) == eigenvectors(plain, x, λ)
    end
end

@testset "exact degeneracy with a coupling: undefined per state, defined for the block" begin
    c = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(0), E_2=0.0); sys = dc_solve(c)     # energies 0, 4, 4
    d = @test_logs (:warn, r"^Node on modes \[:phi\]: kept states \[2, 3\] are degenerate with states they couple to") derivatives(sys; wrt=:E_2)
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
    d2 = @test_logs (:warn, r"^Node on modes \[:phi\]: kept states \[2, 3\] have no second derivatives \(NaN\): states \[2, 3\] are degenerate with states they couple to at second order") derivatives(sys; wrt=:E_1, order=2)
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
    d = only_warnings(r"^Node on modes \[:phi\]: kept states \[2, 3\] have no second derivatives \(NaN\): states \[2, 3\] are degenerate with states they couple to at second order",
                      () -> derivatives(sys; wrt=(:n_g, :E_1), order=2, mixed=((:E_1, :n_g),)))
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

# A child with an exactly degenerate, directly coupled pair of kept states (|−1⟩, |+1⟩ of leaf a at E_1 = E_2 = 0, coupled
# by ∂H/∂E_2 = cos 2φ directly and by ∂H/∂E_1 = cos φ at second order only) below a parent that is not degenerate.
# The n_g of mode a cancels the linear term the coupling routes to leaf a.
const nan_child_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_1", "E_2", "E_Jb", "g"],
 "modes": [{"name": "a", "type": "c"}, {"name": "b", "type": "c"}],
 "E_C": [["1.0", "0.0375"], ["0.0375", "0.9"]],
 "E_L": [["0", "0"], ["0", "0"]],
 "flux_linear": ["0", "0"], "n_g": ["-0.004875", "0.13"],
 "cosines": [{"energy": "E_1", "omega": [1, 0], "offset": "0"}, {"energy": "E_2", "omega": [2, 0], "offset": "0"},
             {"energy": "E_Jb", "omega": [0, 1], "offset": "0"}, {"energy": "g", "omega": [0, 2], "offset": "0"}]}
"""

@testset "a child's undefined derivative makes its parent's NaN, Dense or iterative parent, without an error (design P5)" begin
    function nan_tree(rootsolver; compute=0, keep=lowest(6))
        a = Node([:a => ChargeBasis(1)]; solver=Dense(), keep=lowest(3))
        b = Node([:b => ChargeBasis(20)]; solver=Dense(), keep=lowest(12))
        root = Node([a, b]; solver=rootsolver, compute, keep, ports=[(:b, nport())])
        return (; root, a, spec=KronFreeCircuits.parse_spec(nan_child_json), θ=(E_1=0.0, E_2=0.0, E_Jb=15.0, g=0.3))
    end
    child_first = r"^Node on modes \[:a\]: kept states \[2, 3\] are degenerate with states they couple to"
    child_second_first = r"^Node on modes \[:a\]: kept states \[2, 3\] have no second derivatives \(NaN\): states \[2, 3\] are degenerate at first order$"
    child_second_only = r"^Node on modes \[:a\]: kept states \[2, 3\] have no second derivatives \(NaN\): states \[2, 3\] are degenerate with states they couple to at second order"
    parent = r"^Node on modes \[:a, :b\]: the derivatives of kept states \[1, 2, 3, 4(, 5, 6)?\] are undefined \(NaN\) because a child's derivative is undefined \(children on modes \[:a\]\)$"
    allnan(d) = all(isnan, d.energies) && all(isnan, d.vectors) && all(isnan, d.gauge) && all(P -> all(isnan, P), values(d.ports))
    for (c, iterative) in ((nan_tree(Dense()), false), (nan_tree(LOBPCG(tol=1e-10); compute=6, keep=lowest(4)), true))
        sys = dc_solve(c)
        # first order: the child's pair is NaN, so is every parent state; the child's block is reported
        d = @test_logs (:warn, child_first) (:warn, parent) derivatives(sys; wrt=:E_2)
        @test allnan(d.first[:E_2]) && isnan(energies(d, :E_2, c.a)[2]) && isfinite(energies(d, :E_2, c.a)[1])
        @test any(b -> b.node === c.a && b.states == [2, 3] && b.exact, blocks(d))
        @test !any(b -> b.node === c.root, blocks(d))
        @test iterative == !isempty(d.linear_solves)
        # second order in E_2: the child's NaN at first order reaches the parent at both orders
        d2 = @test_logs (:warn, child_first) (:warn, parent) (:warn, child_second_first) (:warn, parent) derivatives(sys; wrt=:E_2, order=2)
        @test allnan(d2.second[(:E_2, :E_2)])
        @test any(b -> b.node === c.a && b.states == [2, 3], blocks(d2))
        # the child clashes at second order only: the parent's first derivatives are finite, its second NaN
        d3 = @test_logs (:warn, child_second_only) (:warn, parent) derivatives(sys; wrt=:E_1, order=2)
        @test all(isfinite, energies(d3, :E_1)) && allnan(d3.second[(:E_1, :E_1)])
        @test energies(d3, :E_1, :E_1, c.a)[1] ≈ -0.25 atol=1e-9                      # the child's ground state stays defined
        @test any(b -> b.node === c.a && b.states == [2, 3] && b.exact, blocks(d3))
    end
end

@testset "block_levels of a block whose effective Hamiltonian is not finite" begin
    # energies 0, 4 | 4: the kept state 2 is exactly degenerate with the unkept state 3, which ∂H/∂E_2 = cos 2φ couples
    # to it, so the second-order pieces of E_2 of the block (1, 2) (flagged along E_1) are NaN
    c0 = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(2), E_2=0.0); s0 = dc_solve(c0)
    d = only_warnings(r"^(selection boundary:|Node on modes \[:phi\]: kept states \[2\])",
                      () -> derivatives(s0; wrt=(:E_1, :E_2), order=2, mixed=((:E_1, :E_2),), scale=(E_1=1.0,)))
    b = only(filter(b -> b.parameters == (:E_1, :E_2), blocks(d)))
    @test !all(isfinite, b.H2[(:E_2, :E_2)])
    @test_throws "a block state is exactly degenerate with a state outside the kept set" block_levels(b, (E_2=0.1,))
    @test_throws "selection-boundary records" block_levels(b, (E_1=0.1, E_2=0.1))
    @test length(block_levels(only(filter(b -> b.parameters == (:E_1,), blocks(d))), (E_1=0.1,)).energies) == 2
end
