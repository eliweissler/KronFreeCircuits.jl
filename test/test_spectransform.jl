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
