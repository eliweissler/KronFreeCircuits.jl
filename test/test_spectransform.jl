@testset "substitute" begin
    full = load_spec(fixture("full_zero_pi.json"))       # modes theta (c), phi (e), zeta (h)
    s = substitute(full, (dC_J=0.0,))
    @test :dC_J ∉ s.parameters
    @test KronFreeCircuits._iszero(s.E_C[1, 2])          # E_CS*dC_J/4 -> 0
    @test !KronFreeCircuits._iszero(s.E_C[1, 3])         # -sqrt(2)*E_CS*dC/4 untouched
    @test coefficients_agree(s, full, (dC_J=0.0,))
    c = substitute(Coefficient("E_J*cos(varphi_ext) + 2", [:E_J, :varphi_ext]), (varphi_ext=0.0,))
    @test c.deps == [:E_J] && c((E_J=3.0,)) ≈ 5.0
    @test substitute(Coefficient("(1-1)*E_J", [:E_J]), NamedTuple()) |> KronFreeCircuits._iszero
    @test_throws ArgumentError substitute(full, (nonexistent=1.0,))
    @test_throws ArgumentError substitute(full, (dCJ=1.0im,))

    # every symbol-free subtree folds to one Float64 literal; free symbols stay symbolic
    sub(src, vals; params=[:a, :b]) = substitute(Coefficient(src, params), vals)
    @test sub("a*(2 + 3)", NamedTuple()).ex == :(a * 5.0)
    @test sub("a*sqrt(2)*cos(pi/3)", NamedTuple()).ex == :(a * $(sqrt(2.0)) * $(cos(pi / 3)))
    @test sub("a*b", (b=2,)).ex == :(a * 2.0)
    @test sub("a*b + b^2", (b=2,)).ex == :(a * 2.0 + 4.0)
    @test sub("sin(pi*b)^2 + a", (b=0.5,)).ex == :(1.0 + a)
    @test sub("1 // 3 + a", NamedTuple()).ex == :($(1 / 3) + a)
    @test sub("a*(1 // 3)", NamedTuple()).ex.args[3] === 1 / 3
    @test sub("a*(pi + ℯ)", NamedTuple()).ex.args[3] === Float64(pi + ℯ)
    @test sub("b", (b=2,)).ex === 2.0 && sub("b", (b=2,)).deps == Symbol[]
    @test sub("a", (b=2.0,)).deps == [:a]
    @test sub("a + b", (a=1.0, b=2.0)).ex === 3.0
    # the substituted function agrees with the original at the substituted point
    c0 = Coefficient("a*exp(-b^2/2) + sqrt(b) - a^2*cos(b)", [:a, :b])
    c1 = substitute(c0, (b=0.7,))
    @test c1.deps == [:a]
    @test isapprox(c1((a=1.3,)), c0((a=1.3, b=0.7)); rtol=1e-14)
    # a symbol-free subtree that is not real is an error
    @test_throws ArgumentError sub("a*exp(im*b)", (b=0.3,))
    @test_throws ArgumentError sub("a*sqrt(b)", (b=-1.0,))
    @test_throws ArgumentError sub("a + 1/b", (b=0.0,))
    # values must be finite reals
    @test_throws ArgumentError substitute(full, (dC_J=NaN,))
    @test_throws ArgumentError substitute(Coefficient("a", [:a]), (a=1 + 0im,))

    # cosines and charge cosines whose energy vanishes are removed; the rest are kept in order
    two = parse_spec("""{"format": "kronfree-spec", "version": 1, "parameters": ["E_J1", "E_J2", "E_X", "n_g"],
        "modes": [{"name": "t", "type": "c"}, {"name": "x", "type": "e"}],
        "E_C": [["1", "0"], ["0", "1"]], "E_L": [["0", "0"], ["0", "1"]], "flux_linear": ["0", "0"], "n_g": ["n_g", "0"],
        "cosines": [{"energy": "E_J1", "omega": [1, 0], "offset": "0"}, {"energy": "E_J2*E_J1", "omega": [1, 1], "offset": "0"},
                    {"energy": "E_J2", "omega": [2, 0], "offset": "n_g"}],
        "charge_cosines": [{"energy": "E_X", "kappa": [0, 1], "offset": "0"}, {"energy": "E_X*E_J1", "kappa": [0, 2], "offset": "0"}],
        "constant": "E_J1 + E_J2"}""")
    t = substitute(two, (E_J1=0.0, n_g=0.25))
    @test t.parameters == [:E_J2, :E_X]
    @test [c.omega for c in t.cosines] == [[2.0, 0.0]]
    @test [c.kappa for c in t.charge_cosines] == [[0.0, 1.0]]
    @test coefficients_agree(t, two, (E_J1=0.0, n_g=0.25))
    @test !coefficients_agree(t, two, (E_J1=1.0, n_g=0.25))                  # the helper does detect a wrong substitution
    u = substitute(two, (E_J2=3.0,))
    @test u.parameters == [:E_J1, :E_X, :n_g] && length(u.cosines) == 3 && length(u.charge_cosines) == 2
    @test coefficients_agree(u, two, (E_J2=3.0,))
    @test KronFreeCircuits._iszero(substitute(two, (E_J1=0.0, E_J2=0.0, E_X=0.0)).constant)
    @test isempty(substitute(two, (E_J1=0.0, E_J2=0.0, E_X=0.0)).cosines)
    # the result writes and reads back as the same functions
    @test spec_equal(parse_spec(spec_json(t)), t)
    @test spec_equal(parse_spec(spec_json(s)), s)
end

# Modes s (e), f (e) and c (c) with every part of design Section 4.1's table that carries over: E_C[s,s] and n_g[s] (dropped),
# E_L[s,s], E_L[s,f] (nonzero), flux_linear[s], a cosine over s, f and c, a cosine on s alone, a cosine off s, and a charge cosine with kappa[s] = 0.
const _TABLE_HEAD = """{"format": "kronfree-spec", "version": 1,
    "parameters": ["Es", "Ef", "Ec", "Lss", "Lsf", "Lff", "hs", "hf", "ngs", "ngc", "EJ1", "EJ2", "EJ3", "EN", "ph", "ph2", "K0"],
    "modes": [{"name": "s", "type": "e"}, {"name": "f", "type": "e"}, {"name": "c", "type": "c"}],"""
table_spec() = parse_spec(_TABLE_HEAD * """
    "E_C": [["Es", "0", "0"], ["0", "Ef", "0"], ["0", "0", "Ec"]],
    "E_L": [["Lss", "Lsf", "0"], ["Lsf", "Lff", "0"], ["0", "0", "0"]],
    "flux_linear": ["hs", "hf", "0"], "n_g": ["ngs", "0", "ngc"],
    "cosines": [{"energy": "EJ1", "omega": [1, -1, 2], "offset": "ph"},
                {"energy": "EJ2", "omega": [2, 0, 0], "offset": "ph2"},
                {"energy": "EJ3", "omega": [0, 1, 1], "offset": "0"}],
    "charge_cosines": [{"energy": "EN", "kappa": [0, 1, 0], "offset": "ph"}],
    "constant": "K0"}""")
# The hand-built fast spec of `table_spec()` promoted over `s` with the new parameter `x`.
table_spec_expected_fast() = parse_spec("""{"format": "kronfree-spec", "version": 1,
    "parameters": ["Es", "Ef", "Ec", "Lss", "Lsf", "Lff", "hs", "hf", "ngs", "ngc", "EJ1", "EJ2", "EJ3", "EN", "ph", "ph2", "K0", "x"],
    "modes": [{"name": "f", "type": "e"}, {"name": "c", "type": "c"}],
    "E_C": [["Ef", "0"], ["0", "Ec"]], "E_L": [["Lff", "0"], ["0", "0"]],
    "flux_linear": ["hf + Lsf*x", "0"], "n_g": ["0", "ngc"],
    "cosines": [{"energy": "EJ1", "omega": [-1, 2], "offset": "ph + x"},
                {"energy": "EJ3", "omega": [1, 1], "offset": "0"}],
    "charge_cosines": [{"energy": "EN", "kappa": [1, 0], "offset": "ph"}],
    "constant": "K0 + Lss*x^2/2 + hs*x - EJ2*cos(2*x + ph2)"}""")
# A charge cosine that depends on the slow mode's charge.
charge_cosine_spec() = parse_spec("""{"format": "kronfree-spec", "version": 1, "parameters": ["EN"],
    "modes": [{"name": "s", "type": "e"}, {"name": "f", "type": "e"}],
    "E_C": [["1", "0"], ["0", "1"]], "E_L": [["0", "0"], ["0", "1"]], "flux_linear": ["0", "0"], "n_g": ["0", "0"],
    "cosines": [], "charge_cosines": [{"energy": "EN", "kappa": [1, 0], "offset": "0"}], "constant": "0"}""")

# The classical potential `½ φᵀ E_L φ + φ·flux_linear - Σ energy cos(omega·φ + offset) + constant` of a spec at `θ`.
function _spec_potential(spec, θ, φ)
    V = spec.constant(θ) + sum(φ .* [c(θ) for c in spec.flux_linear])
    V += sum(φ[i] * spec.E_L[i, j](θ) * φ[j] for i in eachindex(φ), j in eachindex(φ)) / 2
    return real(V - sum(c.energy(θ) * cos(sum(c.omega .* φ) + c.offset(θ)) for c in spec.cosines; init=0))
end

@testset "promote_mode" begin
    using KronFreeCircuits: ModeSpec
    note = r"promote_mode.*dropped"
    promote(args...; kw...) = @test_logs (:info, note) promote_mode(args...; kw...)
    zp_spec = load_spec(fixture("zero_pi.json"))
    θz = zero_pi_params()

    # one spec exercising every table row
    full = table_spec()
    p = promote(full, :s; parameter=:x)
    @test spec_equal(p.fast, table_spec_expected_fast())
    @test p.fast.parameters[end] == :x && p.fast.parameters[1:end-1] == full.parameters
    @test p.mode == ModeSpec(:s, :e) && p.parameter == :x && p.source === full
    @test p.E_C === full.E_C[1, 1] && p.n_g === full.n_g[1]
    @test spec_equal(parse_spec(spec_json(p.fast)), p.fast)
    # the default parameter name is the mode's name
    @test promote(full, :s).parameter == :s && :s in promote(full, :s).fast.parameters
    # the classical potential is carried over exactly: V_full(φ_s, φ_f, φ_c) == V_fast(φ_f, φ_c; x = φ_s)
    θ = (; (n => 0.3 + 0.17i for (i, n) in enumerate(full.parameters))...)
    for φ in ([0.4, -1.1, 0.7], [-2.3, 0.2, 1.9], [1.5, 0.9, -0.4])
        @test _spec_potential(full, θ, φ) ≈ _spec_potential(p.fast, merge(θ, (x=φ[1],)), φ[2:3]) rtol=1e-12
    end
    # offset charge: the (s,s) part of 4 n_gᵀ E_C n_g leaves the fast spec, the carried-over constants do not
    x = 0.8; θx = merge(θ, (x=x,))
    carried = θ.Lss * x^2 / 2 + θ.hs * x - θ.EJ2 * cos(2x + θ.ph2)
    @test real(constant_term(full)(θ)) ≈ real(constant_term(p.fast)(θx)) - carried + 4 * θ.Es * θ.ngs^2 rtol=1e-12
    @test p.E_C(θ) == θ.Es && p.n_g(θ) == θ.ngs

    # zero-pi: the fast spec at θ = 0.7 equals the full Hamiltonian at that θ, against a dense φ-only build on the same grid
    zp = promote(zp_spec, :theta)
    @test zp.fast.modes == [ModeSpec(:phi, :e)] && zp.parameter == :theta
    grid = PhaseGrid(-6π, 6π, 201)
    n2 = Matrix(dense(localop(grid, n2port())))
    φ = KronFreeCircuits.coordinates(grid)
    Hdense = 4 * (θz.E_CJ / 2) * n2 + Diagonal(θz.E_L .* φ .^ 2 .- 2 * θz.E_J * cos(0.7) .* cos.(φ .- θz.varphi_ext / 2) .+ 2 * θz.E_J)
    sys = @test_logs (:warn, r"PhaseGrid on mode :phi is too small") match_mode=:any materialize(Node([:phi => grid]; solver=Dense(), keep=lowest(4)), zp.fast, merge(θz, (theta=0.7,)))
    @test maximum(abs.(energies(sys) .- eigvals(Hermitian(real(Hdense)))[1:4])) < 1e-10

    # errors: a charge coupling, a charge cosine on the slow mode, and bad parameter names
    fullzp = load_spec(fixture("full_zero_pi.json"))
    err = try; promote_mode(fullzp, :theta); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("E_C", err.msg) && occursin("theta", err.msg)     # E_C[theta,phi] and E_C[theta,zeta]
    @test_throws ArgumentError promote_mode(fullzp, :phi)                                      # E_C[phi,theta] is the same coupling
    @test promote(substitute(fullzp, (dC_J=0.0, dC=0.0)), :theta).fast.modes == [ModeSpec(:phi, :e), ModeSpec(:zeta, :h)]
    err = try; promote_mode(charge_cosine_spec(), :s); nothing; catch e; e; end
    @test err isa ArgumentError && occursin("kappa", err.msg) && occursin("s", err.msg)
    @test promote(charge_cosine_spec(), :f).fast.modes == [ModeSpec(:s, :e)]                   # kappa[f] = 0 only matters for slow f
    @test_throws ArgumentError promote_mode(zp_spec, :theta; parameter=:E_J)                   # already a parameter
    @test_throws ArgumentError promote_mode(zp_spec, :theta; parameter=:phi)                   # a remaining mode
    @test_throws ArgumentError promote_mode(zp_spec, :theta; parameter=:pi)                    # reserved name
    @test_throws ArgumentError promote_mode(zp_spec, :theta; parameter=:cos)
    @test_throws ArgumentError promote_mode(load_spec(fixture("transmon.json")), :varphi_1)    # nothing left to be fast
    @test_throws ArgumentError promote_mode(zp_spec, :nonexistent)
end
