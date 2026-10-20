using KronFreeCircuits: Spec, ModeSpec, CosineSpec, ChargeCosineSpec, parse_spec, load_spec
using KronFreeCircuits: Node, Dense, lowest, PhaseGrid, materialize, energies

@testset "Spec" begin
    zp_src = read(fixture("zero_pi.json"), String)
    s = load_spec(fixture("zero_pi.json"))
    @test s isa Spec
    @test s.parameters == [:E_J, :E_L, :E_CJ, :E_CS, :varphi_ext, :n_g]
    @test [m.name for m in s.modes] == [:theta, :phi] && s.modes[1].type == :c && s.modes[2].type == :e
    @test size(s.E_C) == (2, 2) && length(s.cosines) == 2
    @test s.cosines[1].omega == [1.0, 1.0] && s.cosines[2].omega == [1.0, -1.0]
    @test s.E_C[1,1]((E_CS=1.0,)) ≈ 0.5
    @test s.E_L[2,2]((E_L=0.04,)) ≈ 0.08
    @test s.cosines[1].offset((varphi_ext=1.0,)) ≈ -0.5
    @test s.n_g[1]((n_g=0.1,)) ≈ 0.1
    @test isempty(s.charge_cosines)

    bad = replace(zp_src, "[1, 1]" => "[0.5, 1]")
    @test_throws ArgumentError parse_spec(bad)                 # Review Focus 2
    err = try parse_spec(bad) catch e e end
    @test occursin("theta", sprint(showerror, err))
    @test_throws ArgumentError parse_spec(replace(zp_src, "\"E_CS/2\"" => "\"E_XX/2\""))
    @test_throws ArgumentError parse_spec(replace(zp_src, "kronfree-spec" => "other-spec"))
    @test_throws ArgumentError parse_spec(replace(zp_src, "\"version\": 1" => "\"version\": 2"))
    @test_throws ArgumentError parse_spec(replace(zp_src, "\"n_g\": [\"n_g\", \"0\"]" => "\"n_g\": [\"n_g\"]"))
    @test_throws ArgumentError parse_spec(replace(zp_src, "[\"0\", \"2*E_L\"]" => "[\"0\", \"2*E_L\", \"0\"]"))
    @test_throws ArgumentError parse_spec(replace(zp_src, "\"units\"" => "\"unit\""))     # unknown key
    @test parse_spec(replace(zp_src, "\"units\"" => "\"_units\"")).parameters == s.parameters   # _keys ignored
    # a non-integer omega is allowed on an extended mode
    @test parse_spec(replace(zp_src, "[1, -1]" => "[1, -0.5]")).cosines[2].omega == [1.0, -0.5]

    # charge cosines: kappa must vanish on compact modes; defaults when absent
    with_cc(kappa) = replace(zp_src, "\"cosines\":" =>
        "\"charge_cosines\": [{\"energy\": \"E_J\", \"kappa\": $kappa, \"offset\": \"0\"}],\n  \"cosines\":")
    cc = parse_spec(with_cc("[0, 1]"))
    @test length(cc.charge_cosines) == 1 && cc.charge_cosines[1].kappa == [0.0, 1.0]
    @test_throws ArgumentError parse_spec(with_cc("[1, 0]"))
    @test s.constant((E_J=10.0,)) ≈ 20.0

    # every fixture loads; spot checks
    for f in ["fluxonium.json", "transmon.json", "two_transmons.json", "three_transmons.json",
              "zero_pi.json", "full_zero_pi.json", "two_zero_pi.json"]
        sf = load_spec(fixture(f))
        N = length(sf.modes)
        @test size(sf.E_C) == (N, N) && size(sf.E_L) == (N, N)
        @test length(sf.n_g) == N && length(sf.flux_linear) == N
        @test all(length(c.omega) == N for c in sf.cosines)
    end
    fl = load_spec(fixture("fluxonium.json"))
    @test fl.constant((E_L=0.5, varphi_ext=2.0)) ≈ 1.0 && fl.flux_linear[1]((E_L=0.5, varphi_ext=2.0)) ≈ 1.0
    tt = load_spec(fixture("two_transmons.json"))
    @test tt.E_C[1,2]((g=0.8,)) ≈ 0.1 && tt.E_C[2,1]((g=0.8,)) ≈ 0.1
    t3 = load_spec(fixture("three_transmons.json"))
    @test [m.name for m in t3.modes] == [:varphi_1, :varphi_2, :varphi_3] && t3.cosines[3].omega == [0.0, 0.0, 1.0]
    fz = load_spec(fixture("full_zero_pi.json"))
    @test [(m.name, m.type) for m in fz.modes] == [(:theta, :c), (:phi, :e), (:zeta, :h)]
    @test fz.parameters == [:E_J, :E_L, :E_CJ, :E_CS, :E_Cz, :dE_J, :dC_J, :dC, :dE_L, :varphi_ext, :n_g]
    @test length(fz.cosines) == 4
    z2 = load_spec(fixture("two_zero_pi.json"))
    @test [m.name for m in z2.modes] == [:theta1, :phi1, :theta2, :phi2]
    @test z2.E_C[1,3]((g=0.8,)) ≈ 0.1 && z2.E_C[3,1]((g=0.8,)) ≈ 0.1
end

@testset "Spec expressions are real-valued" begin
    zp_src = read(fixture("zero_pi.json"), String)
    err = try parse_spec(replace(zp_src, "\"E_CS/2\"" => "\"E_CS/2*im\"")); nothing catch e; e end
    @test err isa ArgumentError && occursin("E_C[1,1]", err.msg) && occursin("E_CS/2*im", err.msg) && occursin("`im`", err.msg)
    with_offset(o) = replace(zp_src, r"\"offset\": \"[^\"]*\"" => "\"offset\": \"$o\""; count=1)
    @test_throws ArgumentError parse_spec(with_offset("cis(varphi_ext)"))
    @test_throws ArgumentError parse_spec(with_offset("real(cispi(varphi_ext))"))
    @test_throws ArgumentError parse_spec(replace(zp_src, "\"n_g\": [\"n_g\", \"0\"]" => "\"n_g\": [\"n_g + 0*im\", \"0\"]"))
    @test parse_spec(with_offset("-varphi_ext/2 + sqrt(abs(varphi_ext))")) isa Spec      # real-valued calls stay allowed
    # every spec fixture still loads
    specs = filter(f -> endswith(f, ".json") && f != "scqubits_reference.json", readdir(joinpath(@__DIR__, "fixtures")))
    @test length(specs) == 9 && all(f -> load_spec(fixture(f)) isa Spec, specs)
end

@testset "Spec: the format document's fluxonium, dotted operators" begin
    src = """{"format": "kronfree-spec", "version": 1, "units": "GHz",
      "parameters": ["E_C", "E_L", "varphi_ext", "E_J"],
      "modes": [{"name": "varphi_1", "type": "e"}],
      "E_C": [["E_C"]], "E_L": [["E_L"]], "flux_linear": ["E_L * varphi_ext"], "n_g": ["0"],
      "cosines": [{"energy": "E_J", "omega": [1], "offset": "0"}], "constant": "0"}"""
    dotted = replace(src, "\"E_L * varphi_ext\"" => "\"E_L .* varphi_ext .^ 1\"", "[[\"E_C\"]]" => "[[\"E_C .+ 0\"]]",
                     "\"E_J\"}" => "\"E_J ./ 1 .- 0\"}")
    @test dotted != src
    θ = (E_C=1.0, E_L=0.5, varphi_ext=0.3, E_J=4.0)
    leaf = Node([:varphi_1 => PhaseGrid(-6π, 6π, 121; order=6)]; solver=Dense(), keep=lowest(4))
    e1 = energies(materialize(leaf, parse_spec(src), θ))
    e2 = energies(materialize(leaf, parse_spec(dotted), θ))
    @test e1 ≈ e2
end

@testset "Spec: SI-valued coefficients with integer literals beyond Int64" begin
    # A transmon in SI units, as circuit-quantization exports it with element values in farads and joules:
    # E_C = e²/(2C h) in GHz, with integer literals beyond typemax(Int64).
    src = """{"format": "kronfree-spec", "version": 1, "units": "GHz",
      "parameters": ["C_J", "E_J"],
      "modes": [{"name": "varphi_1", "type": "c"}],
      "E_C": [["(345107820312500000000 // 213914163877964163) ./ (100000000000000000000 * C_J)"]],
      "E_L": [["0"]], "flux_linear": ["0"], "n_g": ["0"],
      "cosines": [{"energy": "E_J", "omega": [1], "offset": "0"}], "constant": "0"}"""
    s = parse_spec(src)
    C_J = 1.2e-14
    exact = (big(345107820312500000000) // 213914163877964163) / (big(10)^20 * big(C_J))
    @test abs(real(s.E_C[1, 1]((C_J=C_J, E_J=10.0))) - Float64(exact)) <= 4eps(Float64(exact))
end

@testset "Spec: parameter names are normalized like Julia identifiers" begin
    # µ (U+00B5, micro sign) and μ (U+03BC, Greek mu) are the same Julia identifier; a declared name must match its use.
    src = """{"format": "kronfree-spec", "version": 1,
      "parameters": ["µ", "E_J"],
      "modes": [{"name": "varphi_1", "type": "c"}],
      "E_C": [["μ"]], "E_L": [["0"]], "flux_linear": ["0"], "n_g": ["0"],
      "cosines": [{"energy": "E_J", "omega": [1], "offset": "0"}], "constant": "0"}"""
    s = parse_spec(src)
    @test s.parameters == [Symbol("μ"), :E_J]
    @test s.E_C[1, 1]((; Symbol("μ") => 0.25)) == 0.25
    @test_throws ArgumentError parse_spec(replace(src, "\"E_J\"]" => "\"E_J\", \"μ\"]"))   # now duplicates
end

@testset "Spec: E_C and E_L must be symmetric" begin
    # E_C and E_L are symmetric by construction (scaled inverse capacitance and inverse inductance matrices), so an
    # asymmetric pair of entries is non-physical and is rejected rather than symmetrized.
    spec(EC, EL) = """{"format": "kronfree-spec", "version": 1,
      "parameters": ["a", "b", "c", "g", "x"],
      "modes": [{"name": "p", "type": "e"}, {"name": "q", "type": "e"}],
      "E_C": $EC, "E_L": $EL, "flux_linear": ["0", "0"], "n_g": ["0", "0"],
      "cosines": [], "constant": "0"}"""
    sym = """[["a", "0"], ["0", "a"]]"""
    msg(src) = try parse_spec(src); nothing catch e; e isa ArgumentError ? e.msg : e end
    m = msg(spec("""[["a", "b"], ["c", "a"]]""", sym))
    @test m isa String && occursin("E_C", m) && occursin("E_C[1,2]", m) && occursin("E_C[2,1]", m) &&
          occursin("\"b\"", m) && occursin("\"c\"", m) && occursin("symmetric", m)
    m = msg(spec(sym, """[["a", "2*x"], ["x", "a"]]"""))
    @test m isa String && occursin("E_L[1,2]", m) && occursin("E_L[2,1]", m) && occursin("\"2*x\"", m) && occursin("\"x\"", m)
    # a comparison that cannot be evaluated (sqrt of a negative real at every test point) is rejected too
    m = msg(spec("""[["a", "sqrt(-a)"], ["sqrt(-b)", "a"]]""", sym))
    @test m isa String && occursin("E_C[1,2]", m) && occursin("cannot be evaluated", m)
    # a non-analytic pair cannot be proven equal at three points (abs(x) = x on all of them); identical text passes
    m = msg(spec("""[["a", "abs(x)"], ["x", "a"]]""", sym))
    @test m isa String && occursin("E_C[1,2]", m) && occursin("\"abs(x)\"", m) && occursin("not real-analytic", m)
    @test parse_spec(spec("""[["a", "abs(x)"], ["abs(x)", "a"]]""", sym)) isa Spec
    # equal entries spelled differently pass
    s = parse_spec(spec("""[["a", "g/8"], ["0.125*g", "a"]]""", """[["a", "2*x"], ["x+x", "a"]]"""))
    @test s isa Spec && s.E_C[1, 2]((g=0.8,)) ≈ s.E_C[2, 1]((g=0.8,)) ≈ 0.1
    @test parse_spec(spec("""[["a", 0.5], ["1/2", "a"]]""", sym)) isa Spec      # a number against an expression
    # every hand-written and generated fixture is symmetric and loads
    for dir in ("", "generated")
        for f in filter(f -> endswith(f, ".json") && f != "scqubits_reference.json", readdir(joinpath(@__DIR__, "fixtures", dir)))
            @test load_spec(fixture(joinpath(dir, f))) isa Spec
        end
    end
end
