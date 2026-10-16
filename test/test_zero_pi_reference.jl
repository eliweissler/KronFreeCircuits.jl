@testset "two zero-pi vs scqubits HilbertSpace" begin
    θ = two_zero_pi_params(g=0.05)
    mk(i) = Node([Symbol("theta$i") => ChargeBasis(15), Symbol("phi$i") => PhaseGrid(-6π, 6π, 301)]; solver=Lanczos(), compute=10, keep=lowest(10))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20))
    sys = materialize(top, load_spec(fixture("two_zero_pi.json")), θ)
    # `≈` with atol compares the 2-norm of the difference over all 20 levels.
    # Observed: max elementwise error 6.8e-5 (passes 1e-4), 2-norm about 2e-4 (misses 1e-4, within 1e-3).
    # Cause is leaf-level (Task 11): grid convergence on the shared ±6π range
    # (scqubits' 7-point stencil on 1001 points vs our order-8 stencil on 301 points).
    @test energies(sys) ≈ reference("two_zero_pi").energies atol=1e-3
    @test maximum(abs.(energies(sys) .- reference("two_zero_pi").energies)) < 1e-4
    @test length(bare_labels(sys, top)) == 100
end
