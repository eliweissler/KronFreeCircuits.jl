@testset "BO surfaces" begin
    # L6 crossing: fast compact mode f, slow compact s, H = 4E_C (n_f - n_g)^2 - E_J cos(s) cos(φ_f)
    # (two cosines, omega [1,1] and [1,-1], energy E_J/2). At n_g = 0.5 the two lowest bands cross at s = π/2.
    p = quiet_promote(crossing_spec(), :s)
    leaf = Node([:f => ChargeBasis(4)]; solver=Dense(), keep=lowest(4))
    g = periodic_grid(64)
    @test g == 2π .* (0:63) ./ 64 && periodic_grid(4; start=0.5) == 0.5 .+ 2π .* (0:3) ./ 4
    # At s = π/2 the coupling vanishes, the kept states are bare charge states, and the third one has no weight on the phase mask.
    S = @test_logs (:warn, r"column 3 has no weight on the phase mask") match_mode=:any bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0), g; bands=2)
    @test S.energies ≈ dense_crossing_reference(g; E_C=1.0, E_J=0.2, n_g=0.5, ncut=4)[1:2, :] atol=1e-12
    @test all(S.energies[1, :] .<= S.energies[2, :])                          # envelope, not a followed branch
    @test S.energies[1, 17] ≈ S.energies[2, 17] atol=1e-12                     # g[17] == π/2: exact crossing
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
    @test length(a) == 16 && length(b) == 16 && abs(b[16]) < 1e-14
    for M in (8, 9)                                                             # arbitrary data is reproduced exactly at the grid points
        Sr = synthetic_surface(periodic_grid(M), θ -> sin(7θ + 0.3) + θ^2 / 10)
        c0, c, d = fourier_coefficients(Sr)
        @test c0 .+ [sum(c[k] * cos(k * x) + d[k] * sin(k * x) for k in eachindex(c)) for x in Sr.grid] ≈ Sr.energies[1, :]
    end
    Sn = synthetic_surface(periodic_grid(8; start=0.3), θ -> cos(4θ) + 0.5sin(4θ) + 0.2)     # Nyquist harmonic on a shifted grid
    n0, na, nb = fourier_coefficients(Sn)
    @test maximum(abs, n0 .+ [na[4] * cos(4x) + nb[4] * sin(4x) for x in Sn.grid] .- Sn.energies[1, :]) < 1e-12
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

@testset "fits" begin
    # synthetic recovery to round-off: Di Paolo's form on a compact slow mode
    U(φ) = θ -> 1.1 - (0.74 - 4e-4cos(φ))cos(2θ) - 0.0011cos(φ / 2)cos(θ)
    mk(φ; fast_value=0.7) = synthetic_surface(periodic_grid(32), U(φ); θ=(varphi_ext=φ,), fast_value)
    mk_other_fixed_param() = mk(0.0; fast_value=1.3)
    surfs = [mk(φ) for φ in 2π .* (0:0.125:0.5)]
    model = [:constant => ["1"], (:cos, 2) => ["1", "cos(varphi_ext)"], (:cos, 1) => ["cos(varphi_ext/2)"]]
    F = fit_surfaces(surfs, model)
    @test F.constants ≈ [1.1, 0.74, -4e-4, 0.0011] atol=1e-13
    @test F.terms == [(:constant, "1"), ((:cos, 2), "1"), ((:cos, 2), "cos(varphi_ext)"), ((:cos, 1), "cos(varphi_ext/2)")]
    @test F.rms_residual < 1e-13 && F.max_residual < 1e-13 && F.rms_residual <= F.max_residual
    @test length(F.stderr) == 4 && all(<(1e-12), F.stderr)
    @test :varphi_ext in F.spec.parameters
    @test :dummy ∉ F.spec.parameters                                          # fast parameter held fixed: substituted, never free
    @test Set(F.spec.parameters) == Set([:varphi_ext, :E_Cs, :n_gs])           # fitted over, plus the slow stage's own
    @test [m.name for m in F.spec.modes] == [:s] && F.spec.modes[1].type === :c
    # the reduced spec is the model: constants multiply the entries directly
    @test length(F.spec.cosines) == 2 && F.spec.cosines[1].omega == [2.0] && F.spec.cosines[2].omega == [1.0]
    for φ in (0.3, 1.7)
        θ = (varphi_ext=φ, E_Cs=0.05, n_gs=0.0)
        @test F.spec.constant(θ) ≈ 1.1
        @test F.spec.cosines[1].energy(θ) ≈ 0.74 - 4e-4cos(φ)
        @test F.spec.cosines[2].energy(θ) ≈ 0.0011cos(φ / 2)
        @test all(c -> c.offset(θ) == 0, F.spec.cosines)
    end
    @test spec_equal(parse_spec(spec_json(F.spec; metadata=F.metadata)), F.spec)
    md = KronFreeCircuits.JSON3.read(spec_json(F.spec; metadata=F.metadata))
    @test md._generator.package == "KronFreeCircuits" && md._generator.version == string(pkgversion(KronFreeCircuits))
    @test md._bo.slow_mode.name == "s" && md._bo.fixed_parameters.dummy == 0.7 && length(md._bo.surfaces) == 5
    @test [m.name for m in md._bo.source_spec.modes] == ["s", "f"] && md._bo.source_spec.format == "kronfree-spec"
    @test parse_spec(spec_json(F.spec; metadata=F.metadata)) isa KronFreeCircuits.Spec
    @test length(md._bo.surfaces[2].grid) == 32 && collect(md._bo.surfaces[2].grid) ≈ surfs[2].grid
    @test md._bo.surfaces[2].grid_size == 32 && md._bo.surfaces[2].fitted_values.varphi_ext ≈ 2π * 0.125
    @test collect(md._bo.constants) ≈ F.constants && length(md._bo.model) == 4 && md._bo.band == 1
    @test md._bo.rms_residual == F.rms_residual

    # sine harmonics become cosines with offset -π/2
    Fs = fit_surfaces([synthetic_surface(periodic_grid(16), θ -> 0.2 + 0.3sin(θ) - 0.4cos(3θ); θ=(varphi_ext=0.0,))],
                      [:constant => ["1"], (:sin, 1) => ["1"], (:cos, 3) => ["1"]])
    @test Fs.constants ≈ [0.2, -0.3, 0.4] atol=1e-13
    @test Fs.spec.cosines[1].offset(NamedTuple()) ≈ -π / 2 && Fs.spec.cosines[1].omega == [1.0]
    # a fitted slow stage agrees with the Fourier stage
    Ss = synthetic_surface(periodic_grid(16), θ -> 0.2 + 0.3sin(θ) - 0.4cos(3θ))
    leaf = Node([:s => ChargeBasis(12)]; solver=Dense(), keep=lowest(4))
    @test energies(materialize(leaf, Fs.spec, (E_Cs=0.05, n_gs=0.1))) ≈
          energies(materialize(leaf, slow_spec(Ss)[1], (E_Cs=0.05, n_gs=0.1))) rtol=1e-10

    # standard errors, with noise: σ² = RSS/(n - p), cov = σ² (AᵀA)⁻¹
    rng = MersenneTwister(7)
    noise = 1e-3 .* randn(rng, 32)
    Sn = synthetic_surface(periodic_grid(32), θ -> 0.5 + 0.2cos(θ); θ=(varphi_ext=0.0,))
    Sn.energies .+= noise'
    Fn = fit_surfaces([Sn], [:constant => ["1"], (:cos, 1) => ["1"]])
    A = hcat(ones(32), -cos.(Sn.grid))
    c = (A'A) \ (A'Sn.energies[1, :])
    rss = sum(abs2, Sn.energies[1, :] - A * c)
    @test Fn.constants ≈ c
    @test Fn.stderr ≈ sqrt.(rss / 30 .* diag(inv(A'A)))
    @test Fn.rms_residual ≈ sqrt(rss / 32)
    Fe = fit_surfaces([synthetic_surface(periodic_grid(3), θ -> θ; θ=(varphi_ext=0.0,))],        # as many points as constants
                      [:constant => ["1"], (:cos, 1) => ["1"], (:sin, 1) => ["1"]])
    @test all(isnan, Fe.stderr) && all(isfinite, Fe.constants) && Fe.rms_residual < 1e-14
    @test KronFreeCircuits.JSON3.read(spec_json(Fe.spec; metadata=Fe.metadata))._bo.stderr == [nothing, nothing, nothing]

    # quadratic model in two parameters around the working point (0.5, 0.1): recover all six constants
    hidden = [0.8, -0.3, 0.12, 0.5, -0.07, 0.9]
    q(φ, n) = hidden ⋅ [1, φ - 0.5, n - 0.1, (φ - 0.5)^2, (φ - 0.5) * (n - 0.1), (n - 0.1)^2]
    Q = [synthetic_surface(periodic_grid(8), θ -> q(φ, n); θ=(varphi_ext=φ, n_g1=n)) for φ in (0.3, 0.5, 0.7) for n in (0.0, 0.1, 0.25)]
    Fq = fit_surfaces(Q, [:constant => ["1", "(varphi_ext-0.5)", "(n_g1-0.1)", "(varphi_ext-0.5)^2", "(varphi_ext-0.5)*(n_g1-0.1)", "(n_g1-0.1)^2"]])
    @test Fq.constants ≈ hidden atol=1e-12
    @test Set(Fq.spec.parameters) == Set([:varphi_ext, :n_g1, :E_Cs, :n_gs])
    @test Fq.spec.constant((varphi_ext=0.6, n_g1=0.2, E_Cs=1.0, n_gs=0.0)) ≈ q(0.6, 0.2)

    # L6 oscillators, extended slow mode through :x2: U(s) = ω/2 + ½(a − c²/b) s², ω = sqrt(8 E_Cf b)
    p = quiet_promote(coupled_oscillators_spec(), :s)
    leaf = Node([:f => PhaseGrid(-12, 12, 301)]; solver=Dense(), keep=lowest(1))
    θ = (a=1.0, b=2.0, c=0.6, E_Cf=0.5, E_Cs=0.1)
    S = bo_surface(leaf, p, θ, range(-1, 1; length=21); bands=1)
    Fo = fit_surfaces([S], [:constant => ["1"], :x2 => ["1"]])
    @test Fo.constants ≈ [sqrt(8 * 0.5 * 2.0) / 2, (1.0 - 0.6^2 / 2.0) / 2] rtol=1e-7
    @test Fo.spec.parameters == [:E_Cs]                                  # a, b, c, E_Cf substituted
    @test Fo.spec.E_L[1, 1](NamedTuple()) ≈ 2 * Fo.constants[2] && Fo.spec.E_C[1, 1]((E_Cs=0.1,)) ≈ 0.1
    @test Fo.spec.modes[1].type === :e && isempty(Fo.spec.cosines)
    @test spec_equal(parse_spec(spec_json(Fo.spec; metadata=Fo.metadata)), Fo.spec)
    Fx = fit_surfaces([S], [:constant => ["1"], :x => ["1"], :x2 => ["1"]])     # a linear term is fitted as zero
    @test abs(Fx.constants[2]) < 1e-12
    @test Fx.constants[[1, 3]] ≈ Fo.constants rtol=1e-9
    @test Fx.spec.flux_linear[1](NamedTuple()) ≈ Fx.constants[2]

    # parameters: fitted-over stay free, fast-dependent ones are substituted into E_C[s,s] and n_g[s], the rest stay free
    spec = parse_spec("""{"format": "kronfree-spec", "version": 1, "parameters": ["E_Cs", "d", "n_gs"],
        "modes": [{"name": "s", "type": "c"}, {"name": "f", "type": "c"}],
        "E_C": [["E_Cs*d", "0"], ["0", "d"]], "E_L": [["0", "0"], ["0", "0"]], "flux_linear": ["0", "0"], "n_g": ["n_gs", "0"],
        "cosines": [], "constant": "0"}""")
    pd = quiet_promote(spec, :s)
    D(d) = BOSurface(pd, periodic_grid(4), fill(1.0 + 2d, 1, 4), (d=d,))
    Fd = fit_surfaces([D(0.5), D(1.5)], [:constant => ["1", "d"]])
    @test Fd.constants ≈ [1.0, 2.0] && Set(Fd.spec.parameters) == Set([:d, :E_Cs, :n_gs])        # d fitted over: stays free
    Fe2 = fit_surfaces([D(0.5), D(0.5)], [:constant => ["1"]])
    @test Set(Fe2.spec.parameters) == Set([:E_Cs, :n_gs]) && Fe2.spec.E_C[1, 1]((E_Cs=2.0,)) ≈ 2.0 * 0.5      # d fixed: substituted

    # surfaces promoted separately from equal source specs fit together
    Sf = BOSurface(_synthetic_promoted(fresh=true), surfs[2].grid, surfs[2].energies, surfs[2].θ)
    @test Sf.promoted !== surfs[1].promoted
    @test fit_surfaces([surfs[1], Sf, surfs[3]], model).constants ≈ F.constants atol=1e-13

    # errors
    @test_throws ArgumentError fit_surfaces([surfs[1], mk_other_fixed_param()], model)    # differing fixed parameter, Review Focus 2
    @test_throws r"dummy" fit_surfaces([surfs[1], mk_other_fixed_param()], model)
    @test_throws ArgumentError fit_surfaces(surfs, [:x2 => ["1"]])                         # Review Focus 5
    @test_throws ArgumentError fit_surfaces(surfs, [:x => ["1"]])
    @test_throws ArgumentError fit_surfaces(surfs, [(:cos, 1.5) => ["1"]])                 # compact modes have integer harmonics
    @test_throws ArgumentError fit_surfaces(surfs, [(:cos, 0) => ["1"]])
    @test_throws ArgumentError fit_surfaces(surfs, [:quadratic => ["1"]])
    @test_throws ArgumentError fit_surfaces(surfs, [(:tan, 1) => ["1"]])
    @test_throws ArgumentError fit_surfaces([S], [:constant => ["1"], (:cos, 1) => ["1"]]; band=2)
    @test_throws ArgumentError fit_surfaces(surfs, [:constant => ["1"], (:cos, 2) => ["no_such_parameter"]])
    @test_throws ArgumentError fit_surfaces(surfs, [:constant => ["E_Cs"]])               # fitted over a parameter the surfaces lack
    @test_throws ArgumentError fit_surfaces(surfs, [:constant => ["1", "1"]])             # rank-deficient design matrix
    @test_throws ArgumentError fit_surfaces(surfs[1:1], [:constant => ["1"], (:sin, 16) => ["1"]])   # sin vanishes on every grid point
    @test_throws ArgumentError fit_surfaces([synthetic_surface(periodic_grid(2), cos; θ=(varphi_ext=0.0,))],
                                            [:constant => ["1"], (:cos, 1) => ["1"], (:sin, 1) => ["1"]])   # fewer points than constants
    @test_throws ArgumentError fit_surfaces(BOSurface[], model)
    @test_throws ArgumentError fit_surfaces(surfs, Pair[])
    @test_throws ArgumentError fit_surfaces(surfs, [:constant => String[]])
    @test_throws ArgumentError fit_surfaces([surfs[1], D(0.5)], model)                     # different source specs
    @test_throws ArgumentError fit_surfaces([surfs[1], synthetic_surface(periodic_grid(32), cos; θ=(varphi_ext=0.0, n_g1=0.0))], model)   # θ keys differ
end
