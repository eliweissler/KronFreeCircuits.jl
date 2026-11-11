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
