using Test, KronFreeCircuits

@testset "mean-field terms" begin
    K = KronFreeCircuits
    quad(m) = [Term(Coefficient(4.0), [Factor(m, n2port())]), Term(Coefficient(0.5), [Factor(m, phi2port())])]
    t1 = Term(Coefficient(2.0), [Factor(:a, phiport()), Factor(:b, phiport())])
    t2 = Term(Coefficient(0.5 + 0.25im), [Factor(:a, expiphi(1.0)), Factor(:b, expiphi(-2.0)), Factor(:c, expiphi(1.0))])
    t3 = Term(Coefficient(0.7), [Factor(:b, phiport()), Factor(:c, phiport())])           # inside B: not straddling
    terms = vcat(quad(:a), quad(:b), quad(:c), [t1, t2, t3])
    grid = PhaseGrid(-6.0, 6.0, 61)
    A = Node([:a => grid]; solver=Dense(), compute=4, keep=lowest(4))
    B = Node([:b => grid, :c => grid]; solver=Dense(), compute=4, keep=lowest(4))
    root = Node([A, B]; solver=Dense(), compute=2, keep=lowest(2))
    @test Set(straddling_terms(root, terms)) == Set([t1, t2])
    kb, kbc = [(:b, phiport())], K._sortkey!([(:b, expiphi(-2.0)), (:c, expiphi(1.0))])
    @test Set(child_keys(root, terms, B)) == Set([kb, kbc])
    @test Set(child_keys(root, terms, A)) == Set([[(:a, phiport())], [(:a, expiphi(1.0))]])
    others = IdDict{Node,Expectations}(B => Expectations(kb => 0.3, kbc => 0.2 - 0.1im))
    mf = mean_field_terms(root, terms, (;), A, others)
    val(key) = only(t.coeff((;)) for t in mf if K._factor_key(t) == key)
    @test length(mf) == 2
    @test val([(:a, phiport())]) ≈ 2.0 * 0.3
    @test val([(:a, expiphi(1.0))]) ≈ (0.5 + 0.25im) * (0.2 - 0.1im)
    # the joint expectation of B is not the product of its single ones (Review Focus 1)
    # (the grid is small on purpose, so the solve warns once per mode about its edge support)
    b = @test_logs (:warn, r"too small") (:warn, r"too small") K._solve_subtree(B, terms, (;);
                       keys=[kb, kbc, [(:b, expiphi(-2.0))], [(:c, expiphi(1.0))]], complex=true)
    e = expectations(b, [kbc, [(:b, expiphi(-2.0))], [(:c, expiphi(1.0))]])
    @test abs(e[kbc] - e[[(:b, expiphi(-2.0))]] * e[[(:c, expiphi(1.0))]]) > 1e-4
    @test expectations(b, [kb]; weights=[1, 0, 0, 0])[kb] ≈ expectations(b, [kb])[kb]
    @test_throws ArgumentError expectations(b, [kb]; weights=[0.5, 0.2])
    # starting tables
    p = pinned_expectations([kb, kbc], Dict(:b => 0.5, :c => -1.0))
    @test keys(p) == keys(Expectations(kb => 0.5, kbc => 0.0))
    @test p[kb] ≈ 0.5
    @test p[kbc] ≈ cis(-2.0 * 0.5) * cis(1.0 * -1.0)
    @test all(iszero, values(delocalized_expectations([kb, kbc])))
    # a term reaching outside the parent is absent unless its outside expectation is supplied
    t4 = Term(Coefficient(3.0), [Factor(:a, phiport()), Factor(:z, phiport())])
    with_z = vcat(terms, quad(:z), [t4])
    @test length(mean_field_terms(root, with_z, (;), A, others)) == 2
    out = Expectations([(:z, phiport())] => 0.1)
    mfz = mean_field_terms(root, with_z, (;), A, others; outside=out)
    @test only(t.coeff((;)) for t in mfz if K._factor_key(t) == [(:a, phiport())]) ≈ 2.0 * 0.3 + 3.0 * 0.1
    @test_throws ArgumentError mean_field_terms(root, with_z, (;), A, others; outside=Expectations())
    # a parent with a bare mode child is refused (Review Focus 2)
    bad = Node([A, :b => grid, :c => grid]; solver=Dense(), compute=2, keep=lowest(2))
    @test_throws "Node" straddling_terms(bad, terms)
end

@testset "mean-field child and product energy" begin
    spec = coupled_oscillators_spec()
    grid = PhaseGrid(-10.0, 10.0, 161)
    S = Node([:s => grid]; solver=Dense(), compute=6, keep=lowest(6))
    F = Node([:f => grid]; solver=Dense(), compute=6, keep=lowest(6))
    root = Node([S, F]; solver=Dense(), compute=4, keep=lowest(4))
    # uncoupled: the product state is exact, so one mean-field solve gives the exact ground energy
    θ0 = (a=1.0, b=4.0, c=0.0, E_Cf=1.0, E_Cs=0.5); terms = circuit_terms(spec)
    bs, bf = bare_child(root, terms, θ0, S), bare_child(root, terms, θ0, F)
    es = expectations(bs, child_keys(root, terms, S)); ef = expectations(bf, child_keys(root, terms, F))
    E = product_energy(root, terms, θ0, IdDict(S => bs.energies[1], F => bf.energies[1]), IdDict(S => es, F => ef))
    @test E ≈ energies(materialize(root, spec, θ0))[1] rtol=1e-10
    # coupled: a mean-field child is the bare child plus a linear term, and the product energy is an upper bound
    θ = (a=1.0, b=4.0, c=0.8, E_Cf=1.0, E_Cs=0.5)
    shifted = IdDict{Node,Expectations}(F => pinned_expectations(child_keys(root, terms, F), Dict(:f => 0.4)))
    ms = mean_field_child(root, terms, θ, S, bare_child(root, terms, θ, S), shifted)
    @test ms.solved.energies[1] ≈ 0.5 * sqrt(8 * 0.5 * 1.0) - (0.8 * 0.4)^2 / (2 * 1.0) atol=1e-6   # ½ω − (c φ_f)²/2a
    @test ms.expectations[[(:s, phiport())]] ≈ -0.8 * 0.4 / 1.0 atol=1e-6
    @test ms.bare_energy ≈ ms.solved.energies[1] - 0.8 * 0.4 * real(ms.expectations[[(:s, phiport())]]) atol=1e-9
    exact = energies(materialize(Node([:s => grid, :f => grid]; solver=LOBPCG(), compute=2, keep=lowest(2)), spec, θ))[1]
    mf = mean_field_child(root, terms, θ, F, bare_child(root, terms, θ, F), IdDict{Node,Expectations}(S => ms.expectations))
    Ep = product_energy(root, terms, θ, IdDict(S => ms.bare_energy, F => mf.bare_energy),
                        IdDict(S => ms.expectations, F => mf.expectations))
    @test Ep >= exact - 1e-9
end
