@testset "port characters match the operator data exactly" begin
    bases = [ChargeBasis(4), PhaseGrid(-4π, 4π, 41; order=6), PeriodicGrid(12; order=4), FockBasis(9; phi_zpf=0.7)]
    ports = [nport(), n2port(), phiport(), phi2port(), expiphi(1), expiphi(-2), exp2pin(1), exp2pin(-1),
             Port(:d1, 0), Port(:d2, 0), Port(:d1wrap, 1), Port(:d1wrap, -1), Port(:d2wrap, 1), Port(:d2wrap, -1)]
    nchecked = 0
    for b in bases, p in ports
        op = try localop(b, p) catch e; e isa ArgumentError || rethrow(); continue end
        c = character(b, p)
        M = dense(op)
        if c.kind === :real
            @test all(iszero, imag(M))
        elseif c.kind === :imaginary
            @test all(iszero, real(M)) && !all(iszero, M)
        else
            @test c.kind === :paired && c.partner != p && conj(M) == dense(localop(b, c.partner))
        end
        nchecked += 1
    end
    @test nchecked == 30          # served pairs: ChargeBasis 6, PhaseGrid 8 (2π/Δ = 10, so exp2pin(±1) is served), PeriodicGrid 8, FockBasis 8
    @test character(PhaseGrid(-4π, 4π, 41; order=6), expiphi(2)).partner == expiphi(-2)
    @test character(ChargeBasis(3), expiphi(1)).kind === :real && character(ChargeBasis(3), exp2pin(1)).partner == exp2pin(-1)
    @test character(FockBasis(9; phi_zpf=0.7), exp2pin(1)).kind === :real
    @test_throws ArgumentError character(PeriodicGrid(12; order=4), nport())
    g = GivenSystem(:r; energies=[0.0, 1.0, 2.0], operators=Dict(:a => [0 1 0; 0 0 1.4; 0 0 0], :y => im * [0 1 0; -1 0 0; 0 0 0], :z => [1 1im 0; -1im 1 0; 0 0 1]))
    @test character(g, Port(:a, 0)).kind === :real && character(g, Port(:y, 0)).kind === :imaginary && character(g, Port(:z, 0)).kind === :general
end

classified(root, spec, θ) = (b = KronFreeCircuits._tree_bases(root); terms = circuit_terms(spec, b);
                             r = KronFreeCircuits.route(root, terms); KronFreeCircuits.classify(root, r, constant_term(spec, b), θ))

@testset "structural classification of the fixtures" begin
    zp = load_spec(fixture("zero_pi.json")); θzp = zero_pi_params()
    leaf = Node([:theta => ChargeBasis(4), :phi => PhaseGrid(-6π, 6π, 41; order=6)]; solver=Dense(), keep=lowest(4))
    @test classified(leaf, zp, θzp)[leaf]                                                   # real at flux 0.23, n_g 0.1
    @test classified(leaf, zp, (θzp..., varphi_ext=0.0, n_g=0.0))[leaf]                      # same answer at every θ
    tz = load_spec(fixture("two_zero_pi.json")); θtz = two_zero_pi_params(g=0.05)
    q1 = Node([:theta1 => ChargeBasis(3), :phi1 => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4))
    q2 = Node([:theta2 => ChargeBasis(3), :phi2 => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4))
    top = Node([q1, q2]; keep=lowest(4))
    c = classified(top, tz, θtz); @test c[top] && c[q1] && c[q2]
    fz = load_spec(fixture("full_zero_pi.json"))     # dE_J, dC_J, dC are parameters: complex by structure at every θ
    zpn = Node([:theta => ChargeBasis(3), :phi => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4))
    ftop = Node([zpn, :zeta => FockBasis(4; phi_zpf=0.5)]; keep=lowest(4))
    fθ = merge(full_zero_pi_params(), (dE_J=0.0, dC_J=0.0, dC=0.0, dE_L=0.0))                # symmetric values, same answer
    cf = classified(ftop, fz, fθ); @test !cf[zpn] && !cf[ftop]
    tr = load_spec(fixture("transmon.json")); θtr = (E_J=20.0, E_C=1.0, n_g=0.3)
    @test classified(Node([:varphi_1 => ChargeBasis(10)]), tr, θtr) |> d -> all(values(d))
    pg = Node([:varphi_1 => PeriodicGrid(16; order=4)])
    @test !classified(pg, tr, θtr)[pg]
    @test !classified(pg, tr, (θtr..., n_g=0.5))[pg]                                         # Review Focus 1: exact value, still complex
    lit = parse_spec(replace(read(fixture("transmon.json"), String), "\"n_g\": [\"n_g\"]" => "\"n_g\": [\"0.5\"]"))
    @test classified(pg, lit, (E_J=20.0, E_C=1.0))[pg]                                       # a spec literal is structure
    @test classified(Node([:varphi_1 => FockBasis(12; phi_zpf=0.6)]), load_spec(fixture("fluxonium.json")), (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π)) |> d -> all(values(d))
end

@testset "classification: forced complex, descendants, complex θ, :real assertion" begin
    tz = load_spec(fixture("two_zero_pi.json")); θtz = two_zero_pi_params(g=0.05)
    q1 = Node([:theta1 => ChargeBasis(3), :phi1 => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4), arithmetic=:complex)
    q2 = Node([:theta2 => ChargeBasis(3), :phi2 => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4))
    top = Node([q1, q2]; keep=lowest(4))
    c = classified(top, tz, θtz); @test !c[q1] && c[q2] && !c[top]                         # Review Focus 2
    @test_throws ArgumentError classified(top, tz, (θtz..., g=0.05 + 0im))                  # all parameters are real (spec 20f, Real inputs)
    @test_throws ArgumentError Node([:x => ChargeBasis(2)]; arithmetic=:fast)
    pg = Node([:varphi_1 => PeriodicGrid(16; order=4)]; arithmetic=:real)
    err = try classified(pg, load_spec(fixture("transmon.json")), (E_J=20.0, E_C=1.0, n_g=0.3)); nothing catch e; e end
    @test err isa ArgumentError && occursin("d2wrap", err.msg)
    # a SQUID-like cosine with a flux offset on a charge mode: complex even at zero flux (Review Focus 1)
    sq = parse_spec("""{"format":"kronfree-spec","version":1,"units":"GHz","parameters":["E_J","E_C","x"],
        "modes":[{"name":"q","type":"c"}],"E_C":[["E_C"]],"E_L":[["0"]],"flux_linear":["0"],"n_g":["0"],
        "cosines":[{"energy":"E_J","omega":[1],"offset":"x"}],"constant":"0"}""")
    nq = Node([:q => ChargeBasis(5)])
    @test !classified(nq, sq, (E_J=1.0, E_C=0.2, x=0.0))[nq]
    # a drive W·a + conj(W)·adag on a given system whose operators `a`, `adag` are real (each its own conjugate
    # partner): the test needs each coefficient real by itself, which holds because every parameter is real.
    # A complex-typed W is rejected, naming the parameter.
    g = GivenSystem(:r; energies=[0.0, 1.0, 2.0], operators=Dict(:a => diagm(1 => [1.0, sqrt(2)]), :adag => diagm(-1 => [1.0, sqrt(2)])))
    drive = [Term(Coefficient("W", [:W]), [Factor(:r, Port(:a, 0))]), Term(Coefficient("conj(W)", [:W]), [Factor(:r, Port(:adag, 0))])]
    root = Node([Node([:q => ChargeBasis(3)]; keep=lowest(2)), g]; keep=lowest(3))
    terms = vcat(circuit_terms(sq), drive)
    r = KronFreeCircuits.route(root, terms)
    errw = try KronFreeCircuits.classify(root, r, Coefficient(0), (E_J=1.0, E_C=0.2, x=0.0, W=0.3 + 0.2im)); nothing catch e; e end
    @test errw isa ArgumentError && occursin(":W", errw.msg)
    @test_throws ArgumentError materialize(root, terms, (E_J=1.0, E_C=0.2, x=0.0, W=0.3 + 0.2im))
    # the same drive under an offset-free charge child (the `sq` child is complex by structure, through its offset x)
    s0 = parse_spec("""{"format":"kronfree-spec","version":1,"units":"GHz","parameters":["E_J","E_C"],
        "modes":[{"name":"q","type":"c"}],"E_C":[["E_C"]],"E_L":[["0"]],"flux_linear":["0"],"n_g":["0"],
        "cosines":[{"energy":"E_J","omega":[1],"offset":"0"}],"constant":"0"}""")
    root0 = Node([Node([:q => ChargeBasis(3)]; keep=lowest(2)), g]; keep=lowest(3))
    r0 = KronFreeCircuits.route(root0, vcat(circuit_terms(s0), drive))
    @test KronFreeCircuits.classify(root0, r0, Coefficient(0), (E_J=1.0, E_C=0.2, W=0.3)) |> d -> all(values(d))
    @test_throws ArgumentError KronFreeCircuits.classify(root0, r0, Coefficient(0), (E_J=1.0, E_C=0.2, W=0.3 + 0.2im))
    # a lone real-valued W on an imaginary given operator: its conjugate image would need the coefficient -W, so complex
    gp = GivenSystem(:r; energies=[0.0, 1.0], operators=Dict(:y => [0 1im; -1im 0]))
    rootp = Node([Node([:q => ChargeBasis(3)]; keep=lowest(2)), gp]; keep=lowest(2), arithmetic=:real)
    rp = KronFreeCircuits.route(rootp, vcat(circuit_terms(s0), [Term(Coefficient("W", [:W]), [Factor(:r, Port(:y, 0))])]))
    errp = try KronFreeCircuits.classify(rootp, rp, Coefficient(0), (E_J=1.0, E_C=0.2, W=0.3)); nothing catch e; e end
    @test errp isa ArgumentError && occursin("y", errp.msg) && occursin("W", errp.msg) && occursin(":r", errp.msg)
end

@testset "materialize classifies; arithmetic accessor" begin
    tr = load_spec(fixture("transmon.json")); θtr = (E_J=20.0, E_C=1.0, n_g=0.3)
    @test_throws ArgumentError materialize(Node([:varphi_1 => PeriodicGrid(16; order=4)]; solver=Dense(), arithmetic=:real), tr, θtr)
    n = Node([:varphi_1 => ChargeBasis(6)]; solver=Dense(), keep=lowest(3), arithmetic=:complex)
    sys = materialize(n, tr, θtr)
    @test arithmetic(sys) === :complex && arithmetic(sys, n) === :complex
    @test n.arithmetic === :complex && Node([:x => ChargeBasis(2)]).arithmetic === :auto
    # a composite term on a :real node raises the composite error, not a conjugate-image one (checks run first)
    comp = [Term(Coefficient("E*im", [:E]), [Factor(:a, nport()), Factor(:a, n2port())])]
    errc = try materialize(Node([:a => ChargeBasis(3)]; solver=Dense(), arithmetic=:real), comp, (E=1.0,)); nothing catch e; e end
    @test errc isa ArgumentError && occursin("several factors", errc.msg)
end

@testset "classification: analyticity gate, throwing coefficients, tolerance" begin
    C(s, ps...) = Coefficient(s, collect(Symbol, ps))
    T(c, p) = Term(c, [Factor(:a, p)])
    function leafreal(ts; b=ChargeBasis(4), arithmetic=:auto)
        L = Node([:a => b]; solver=Dense(), arithmetic=arithmetic)
        return KronFreeCircuits.classify(L, KronFreeCircuits.route(L, ts), Coefficient(0), (E=1.0, x=-0.7))[L]
    end
    pg = PhaseGrid(-4π, 4π, 41; order=6)
    n2 = T(C("E", :E), n2port())
    # Hermitian, and real on the test points' range [0.5, 2], but complex at x < 0: non-analytic calls are gated
    @test !leafreal([n2, T(C("E*cis(x - abs(x))", :E, :x), expiphi(1)), T(C("E*cis(abs(x) - x)", :E, :x), expiphi(-1))])
    @test !leafreal([n2, T(C("E*sqrt(x - 0.4 + 0*im)", :E, :x), expiphi(1)), T(C("E*conj(sqrt(x - 0.4 + 0*im))", :E, :x), expiphi(-1))])
    @test !leafreal([T(C("abs(E)", :E), n2port())]) && !leafreal([T(C("E*sign(x)", :E, :x), n2port())])
    # gated conservatively, although conj(z^0.5) = conj(z)^0.5 holds off the negative real axis
    @test !leafreal([T(C("(E + im*x)^0.5", :E, :x), expiphi(1)), T(C("conj((E + im*x)^0.5)", :E, :x), expiphi(-1))]; b=pg)
    err = try leafreal([T(C("abs(E)", :E), n2port())]; arithmetic=:real); nothing catch e; e end
    @test err isa ArgumentError && occursin("abs", err.msg)
    # kept: real-valued sqrt, log and powers, integer-literal powers of complex values, exp, cis, conj
    @test leafreal([T(C("sqrt(E*x) + log(E) + (E*x)^1.5", :E, :x), n2port())])
    @test leafreal([n2, T(C("E*cis(x)", :E, :x), expiphi(1)), T(C("E*cis(-x)", :E, :x), expiphi(-1))]; b=pg)
    @test leafreal([n2, T(C("(E + im*x)^2*exp(im*x)", :E, :x), expiphi(1)), T(C("conj((E + im*x)^2)*exp(-im*x)", :E, :x), expiphi(-1))]; b=pg)
    # a coefficient that throws at a test point (x - 3 < 0 on [0.5, 2]) fails the test
    @test !leafreal([T(C("sqrt(x - 3)", :x), n2port())])
    # agreement to 1e-12 relative to the larger modulus
    pair(δ) = [n2, T(C("E", :E), expiphi(1)), T(C("E*(1 + $δ)", :E), expiphi(-1))]
    @test !leafreal(pair(1e-10); b=pg) && leafreal(pair(1e-14); b=pg)
end
