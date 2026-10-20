# Gated (KRONFREE_SLOW_TESTS=1): the zero-pi Born–Oppenheimer reproduction of Di Paolo et al., arXiv:1811.07921v1, Appendix B
# (design §7 of docs/superpowers/specs/2026-10-19-born-oppenheimer-design.md; bench/born_oppenheimer/results.md).
# The settings and inputs come from the reference fixture written by bench/born_oppenheimer/zero_pi.jl
# (producer `independent BO numerics`); the published coefficients are compared against, never stored as results.

@testset "zero-pi BO: Di Paolo E_alpha and E_gamma (gated)" begin
    ref = KronFreeCircuits.JSON3.read(read(fixture(joinpath("bo_reference", "zero_pi_bo_reference.json")), String))
    @test ref.producer == "independent BO numerics"
    # Inputs (decided 2026-10-20): E_Cphi/ħω_p = 1/(16·0.165), derived from ω_p = sqrt(8 E_J E_CJ)/ħ and C_φ = 2 C_J; the rest as printed.
    @test ref.inputs_hbar_omega_p.E_Cphi == 1 / (16 * 0.165) && ref.inputs_hbar_omega_p.E_J == 0.165 && ref.inputs_hbar_omega_p.E_L == 1e-3
    inputs = (E_Cphi=Float64(ref.inputs_GHz.E_Cphi), E_Ctheta=Float64(ref.inputs_GHz.E_Ctheta), E_J=Float64(ref.inputs_GHz.E_J),
              E_L=Float64(ref.inputs_GHz.E_L))
    @test inputs.E_Cphi ≈ 40 / (16 * 0.165) && inputs.E_J ≈ 6.6 && inputs.E_L ≈ 0.04

    fl = ref.fast_leaf
    leaf = Node([:phi => PhaseGrid(fl.a, fl.b, fl.N; order=fl.order)]; solver=Dense(), compute=2, keep=lowest(2))
    grid = periodic_grid(ref.theta_grid.M; start=ref.theta_grid.start)
    p = quiet_promote(load_spec(fixture("zero_pi_bo.json")), :theta)
    # converged settings: no PhaseGrid edge warning at any point
    S = @test_logs min_level=Base.CoreLogging.Warn [bo_surface(leaf, p, merge(inputs, (varphi_ext=2π * f,)), grid) for f in ref.fluxes]
    model = [:constant => ["1", "cos(varphi_ext)", "cos(2*varphi_ext)", "cos(3*varphi_ext)", "cos(4*varphi_ext)"],
             (:cos, 2) => ["1", "cos(varphi_ext)"], (:cos, 1) => ["cos(varphi_ext/2)"]]
    @test [String.(m.expressions) for m in ref.model] == last.(model)              # the fixture's model
    F = fit_surfaces(S, model)
    i(key, e) = findfirst(==((key, e)), F.terms)
    E_alpha = F.constants[i((:cos, 2), "1")] / 40
    E_beta = -F.constants[i((:cos, 2), "cos(varphi_ext)")] / 40
    E_gamma = F.constants[i((:cos, 1), "cos(varphi_ext/2)")] / 40

    # acceptance: 0.2% relative on E_alpha and E_gamma against the published fit (E_beta is reported only)
    @test abs(E_alpha / 1.8608e-2 - 1) < 2e-3
    @test abs(E_gamma / 2.6625e-5 - 1) < 2e-3
    @test E_beta > 0                                                      # the published sign

    # π-periodicity of the fast ground energy at half flux, below the convergence level 1e-5·E_gamma
    half = S[findfirst(==(0.5), ref.fluxes)]
    M = length(grid)
    @test maximum(abs(half.energies[1, mod1(j + M ÷ 2, M)] - half.energies[1, j]) for j in 1:M) < ref.pi_periodicity_bound_GHz
    @test ref.pi_periodicity_bound_GHz ≈ 1e-5 * E_gamma * 40 rtol=1e-3

    # agreement with the reference fixture to its recorded numerical error
    for (k, v) in ((:E_alpha, E_alpha), (:E_gamma, E_gamma), (:E_beta, E_beta))
        @test abs(v / ref.constants[k] - 1) <= ref.numerical_error_relative[k]
    end

    # the committed reduced spec (test/fixtures/bo/zero_pi_reduced.json, validated by validate_reduced.py) is this fit
    path = fixture(joinpath("bo", "zero_pi_reduced.json"))
    saved = load_spec(path)
    @test saved.parameters == F.spec.parameters
    for φ in (0.0, 0.7, π)
        θ = (E_Ctheta=inputs.E_Ctheta, varphi_ext=φ, n_g=0.0)
        @test saved.constant(θ) ≈ F.spec.constant(θ) rtol=1e-9
        @test [c.energy(θ) for c in saved.cosines] ≈ [c.energy(θ) for c in F.spec.cosines] rtol=1e-9
    end
    md = KronFreeCircuits.JSON3.read(read(path, String))
    @test md._bo.fast_leaf.N == fl.N && md._bo.fast_leaf.order == fl.order && md._bo.fast_leaf.solver == "Dense"
    @test md._bo.theta_samples.M == ref.theta_grid.M
    leaf_s = Node([:theta => ChargeBasis(20)]; solver=Dense(), keep=lowest(6))
    @test energies(materialize(leaf_s, saved, (E_Ctheta=inputs.E_Ctheta, varphi_ext=0.0, n_g=0.0))) ≈
          energies(materialize(leaf_s, F.spec, (E_Ctheta=inputs.E_Ctheta, varphi_ext=0.0, n_g=0.0))) rtol=1e-9
end
