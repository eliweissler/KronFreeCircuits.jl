using KronFreeCircuits: Port, Factor, Term, modes, circuit_terms, constant_term, load_spec, parse_spec,
                        nport, n2port, phiport, phi2port, expiphi, exp2pin

@testset "circuit_terms" begin
    θ = (E_J=8.0, E_C=1.0, E_L=0.5, varphi_ext=π)
    ts = circuit_terms(load_spec(fixture("fluxonium.json")))
    kinds = Dict((t.factors[1].port.kind, t.factors[1].port.k) => t.coeff(θ) for t in ts)
    @test kinds[(:n2, 0.0)] ≈ 4.0 && kinds[(:phi2, 0.0)] ≈ 0.25 && kinds[(:phi, 0.0)] ≈ 0.5π
    @test kinds[(:expiphi, 1.0)] ≈ -4.0 && kinds[(:expiphi, -1.0)] ≈ -4.0
    @test length(ts) == 5
    zp = circuit_terms(load_spec(fixture("two_zero_pi.json")))
    cross = only(filter(t -> Set(modes(t)) == Set([:theta1, :theta2]), zp))
    @test cross.coeff((g=0.1,)) ≈ 0.1
    @test constant_term(load_spec(fixture("transmon.json")))((E_C=0.3, n_g=0.25)) ≈ 4*0.3*0.25^2
end

@testset "Port" begin
    @test expiphi(1) == expiphi(1.0) && hash(expiphi(1)) == hash(expiphi(1.0))
    @test expiphi(1) != exp2pin(1) && expiphi(1) != expiphi(-1)
    @test Dict(nport() => 1)[Port(:n, 0)] == 1
    @test hash(expiphi(-0.0)) == hash(expiphi(0.0)) && expiphi(-0.0) == expiphi(0.0)
    using KronFreeCircuits: port_of_kind
    @test port_of_kind(:n) == nport() && port_of_kind(:phi2) == phi2port() && port_of_kind(:expiphi, 0.5) == expiphi(0.5)
    @test port_of_kind(:exp2pin, -1) == exp2pin(-1)
    @test_throws ArgumentError port_of_kind(:n, 1.0)
    @test_throws ArgumentError port_of_kind(:cos)
    @test port_of_kind(:d2) == Port(:d2, 0) && port_of_kind(:d1wrap, -1) == Port(:d1wrap, -1)
    @test_throws ArgumentError port_of_kind(:d2wrap, 0)
    @test_throws ArgumentError port_of_kind(:d1, 1)
end

@testset "circuit_terms branches" begin
    src = """{
      "format": "kronfree-spec", "version": 1,
      "parameters": ["E", "x", "ng"],
      "modes": [{"name": "a", "type": "e"}, {"name": "b", "type": "e"}],
      "E_C": [["1", "2.5"], ["2.5", "5"]],
      "E_L": [["2", "2.5"], ["2.5", "0"]],
      "flux_linear": ["0", "0"],
      "n_g": ["ng", "0.5"],
      "cosines": [{"energy": "E", "omega": [1, -2], "offset": "x"},
                  {"energy": "E", "omega": [0, 1], "offset": "0"}],
      "charge_cosines": [{"energy": "E", "kappa": [1, -0.5], "offset": "x"}]
    }"""
    ts = circuit_terms(parse_spec(src))
    sig(t) = [(f.mode, f.port.kind, f.port.k) for f in t.factors]
    get(s) = only(filter(t -> sig(t) == s, ts))
    θ = (E=1.7, x=0.4, ng=0.3)

    # diagonal and symmetrized off-diagonal charge/flux terms
    @test get([(:a, :n2, 0.0)]).coeff(θ) ≈ 4
    @test get([(:b, :n2, 0.0)]).coeff(θ) ≈ 20
    @test get([(:a, :n, 0.0), (:b, :n, 0.0)]).coeff(θ) ≈ 4 * (2.5 + 2.5)
    @test get([(:a, :phi2, 0.0)]).coeff(θ) ≈ 1
    @test get([(:a, :phi, 0.0), (:b, :phi, 0.0)]).coeff(θ) ≈ (2.5 + 2.5) / 2
    # linear charge terms from n_g
    @test get([(:a, :n, 0.0)]).coeff(θ) ≈ -4 * ((1 + 1) * 0.3 + (2.5 + 2.5) * 0.5)
    @test get([(:b, :n, 0.0)]).coeff(θ) ≈ -4 * ((2.5 + 2.5) * 0.3 + (5 + 5) * 0.5)
    # identically zero terms are dropped (E_L[b,b] = 0, flux_linear = 0)
    @test !any(t -> sig(t) in ([(:b, :phi2, 0.0)], [(:a, :phi, 0.0)], [(:b, :phi, 0.0)]), ts)

    # cosine: factor with +omega carries exp(+i offset), the -omega term exp(-i offset)
    plus = get([(:a, :expiphi, 1.0), (:b, :expiphi, -2.0)])
    minus = get([(:a, :expiphi, -1.0), (:b, :expiphi, 2.0)])
    @test plus.coeff(θ) ≈ -0.5 * 1.7 * cis(0.4)
    @test minus.coeff(θ) ≈ -0.5 * 1.7 * cis(-0.4)
    @test modes(plus) == [:a, :b]
    # omega[j] == 0 is skipped
    @test get([(:b, :expiphi, 1.0)]).coeff(θ) ≈ -0.5 * 1.7
    @test get([(:b, :expiphi, -1.0)]).coeff(θ) ≈ -0.5 * 1.7

    # charge cosine: exp2pin factors and offset shift -2π κ·n_g
    shift = 2π * (1.0 * 0.3 + (-0.5) * 0.5)
    cp = get([(:a, :exp2pin, 1.0), (:b, :exp2pin, -0.5)])
    cm = get([(:a, :exp2pin, -1.0), (:b, :exp2pin, 0.5)])
    @test cp.coeff(θ) ≈ -0.5 * 1.7 * cis(0.4 - shift)
    @test cm.coeff(θ) ≈ -0.5 * 1.7 * cis(-(0.4 - shift))
end
