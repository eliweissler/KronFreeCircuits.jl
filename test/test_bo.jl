@testset "BO surfaces" begin
    # L6 crossing: fast compact mode f, slow compact s, H = 4E_C (n_f - n_g)^2 - E_J cos(s) cos(φ_f)
    # (two cosines, omega [1,1] and [1,-1], energy E_J/2). At n_g = 0.5 the two lowest bands cross at s = π/2.
    p = quiet_promote(crossing_spec(), :s)
    leaf = Node([:f => ChargeBasis(4)]; solver=Dense(), keep=lowest(4))
    g = periodic_grid(16)         # each materialize costs about 30 ms, so the grid is kept small; g[5] == π/2
    @test g == 2π .* (0:15) ./ 16 && periodic_grid(4; start=0.5) == 0.5 .+ 2π .* (0:3) ./ 4
    # At s = π/2 the coupling vanishes, the kept states are bare charge states, and the third one has no weight on the phase mask.
    S = @test_logs (:warn, r"column 3 has no weight on the phase mask") match_mode=:any bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0), g; bands=2)
    @test S.energies ≈ dense_crossing_reference(g; E_C=1.0, E_J=0.2, n_g=0.5, ncut=4)[1:2, :] atol=1e-12
    @test all(S.energies[1, :] .<= S.energies[2, :])                          # envelope, not a followed branch
    @test S.energies[1, 5] ≈ S.energies[2, 5] atol=1e-12                       # g[5] == π/2: exact crossing
    @test S.grid == g && S.θ == (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0) && S.promoted === p
    Sa = @test_logs (:warn, r"column 3 has no weight on the phase mask") match_mode=:any bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.45, n_gs=0.0), g)   # avoided crossing
    @test minimum(Sa.energies[2, :] .- Sa.energies[1, :]) > 1e-3
    Sr = bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0), [0.3, 0.3])
    @test Sr.energies[:, 1] == Sr.energies[:, 2]                               # repeated point
    @test_throws ArgumentError bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0, s=0.1), g)
    @test_throws ArgumentError bo_surface(Node([:f => ChargeBasis(4)]; solver=Dense(), keep=lowest(2)), p,
                                          (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0), g; bands=3)
    @test_throws ArgumentError bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0), g; bands=0)
end

@testset "compact slow stage" begin
    # known harmonics: a surface built by hand round-trips through slow_spec to a dense periodic-grid reference
    S = synthetic_surface(periodic_grid(32), θ -> 0.3 - 0.7cos(2θ) + 0.05sin(θ) + 0.01cos(5θ))
    a0, a, b = fourier_coefficients(S)
    @test a0 ≈ 0.3 && a[2] ≈ -0.7 && b[1] ≈ 0.05 && a[5] ≈ 0.01 && maximum(abs, a[[1,3,4]]) < 1e-14
    @test length(a) == 16 && length(b) == 16 && b[16] == 0
    for M in (8, 9)                                                             # arbitrary data is reproduced exactly at the grid points
        Sr = synthetic_surface(periodic_grid(M), θ -> sin(7θ + 0.3) + θ^2 / 10)
        c0, c, d = fourier_coefficients(Sr)
        @test c0 .+ [sum(c[k] * cos(k * x) + d[k] * sin(k * x) for k in eachindex(c)) for x in Sr.grid] ≈ Sr.energies[1, :]
    end
    spec, dropped = slow_spec(S; tol=1e-3)
    @test dropped < 1e-14
    spec2, dropped2 = slow_spec(S; K=2)
    @test dropped2 ≈ 0.01
    @test spec.parameters == [:E_Cs, :n_gs]
    leaf = Node([:s => ChargeBasis(20)]; solver=Dense(), keep=lowest(6))
    @test energies(materialize(leaf, spec, (E_Cs=0.05, n_gs=0.1))) ≈
          dense_periodic_reference(S, E_C=0.05, n_g=0.1)[1:6] rtol=1e-8
    @test_throws ArgumentError slow_spec(S; K=2, tol=1e-3)
    @test_throws ArgumentError slow_spec(S; K=-1)
    @test_throws ArgumentError slow_spec(S; band=2)
    @test_throws ArgumentError fourier_coefficients(synthetic_surface(range(0, 2π; length=33), cos))   # endpoint repeated
    @test_throws ArgumentError fourier_coefficients(synthetic_surface(sort(rand(MersenneTwister(1), 32)) .* 2π, cos))
    # a surface on an extended slow mode has no Fourier stage
    e = quiet_promote(KronFreeCircuits.parse_spec("""{"format": "kronfree-spec", "version": 1, "parameters": [],
        "modes": [{"name": "s", "type": "e"}, {"name": "f", "type": "c"}],
        "E_C": [["1", "0"], ["0", "1"]], "E_L": [["0", "0"], ["0", "0"]], "flux_linear": ["0", "0"], "n_g": ["0", "0"],
        "cosines": [], "constant": "0"}"""), :s)
    @test_throws ArgumentError fourier_coefficients(BOSurface(e, periodic_grid(8), zeros(1, 8), NamedTuple()))
end

@testset "zero-pi BO against an independent dense BO" begin
    zp = quiet_promote(load_spec(fixture("zero_pi_bo.json")), :theta)
    θ = (E_Cphi=15.12, E_Ctheta=0.007, E_J=6.6, E_L=0.04, varphi_ext=0.5, n_g=0.0)
    leaf = Node([:phi => PhaseGrid(-8π, 8π, 241)]; solver=Dense(), keep=lowest(2))
    S = @test_logs (:warn, r"PhaseGrid on mode :phi is too small") match_mode=:any bo_surface(leaf, zp, Base.structdiff(θ, (n_g=0,)), periodic_grid(16))
    @test S.energies ≈ dense_zero_pi_bo_reference(θ, periodic_grid(16), PhaseGrid(-8π, 8π, 241)) rtol=1e-10
    spec, _ = slow_spec(S)
    @test Set(spec.parameters) == Set([:E_Ctheta, :n_g])               # fast-dependent parameters substituted
end
