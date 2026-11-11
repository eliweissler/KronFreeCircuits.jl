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

@testset "self-consistent loop" begin
    spec = load_spec(fixture("zero_pi.json")); θ = zero_pi_params(); terms = circuit_terms(spec)
    grid = PhaseGrid(-12π, 12π, 201; order=6)                             # the zero-pi grid of test_materialize.jl
    A = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=4, keep=lowest(4))
    B = Node([:phi => grid]; solver=Dense(), compute=4, keep=lowest(4))
    root = Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))
    # the solves of the charge leaf note a phase-convention fallback for one of its kept states
    sols = @test_logs (:warn, r"no weight on the phase mask") (:warn, r"no weight on the phase mask") mean_field_solutions(root, spec, θ)
    @test issorted([s.energy for s in sols]) && all(s.converged for s in sols)
    exact = energies(materialize(Node([:theta => ChargeBasis(8), :phi => grid];
                                      solver=Dense(), compute=1, keep=lowest(1)), spec, θ))[1]
    @test sols[1].energy >= exact - 1e-9                                  # an upper bound
    # a fixed point: one more solve of each child from the converged tables changes nothing
    s = sols[1]; tables = IdDict{Node,Expectations}(n => c.expectations for (n, c) in s.children)
    bare = @test_logs (:warn, r"no weight on the phase mask") IdDict(n => bare_child(root, terms, θ, n) for n in (A, B))
    again = mean_field_child(root, terms, θ, A, bare[A], IdDict{Node,Expectations}(B => tables[B]))
    @test maximum(abs(again.expectations[k] - tables[A][k]) for k in keys(tables[A])) < 1e-5
    # a warm start from a solution converges at once to the same energy
    w = self_consistent(root, terms, θ, bare; start=s, constant=real(KronFreeCircuits.constant_term(spec)(θ)))
    @test w.energy ≈ s.energy atol=1e-9
    @test w.iterations <= 3
    @test w.start === :warm                                               # no reference to the solution it came from
    # a displacement on a mode that is not under the parent is refused, not silently ignored
    @test_throws "mode :ph, which is not under" self_consistent(root, terms, θ, bare; start=Dict(:ph => π))
    # the iteration limit is reported, not hidden (Review Focus 3)
    short = self_consistent(root, terms, θ, bare; start=Dict(:phi => π), maxiter=1)
    @test !short.converged && short.iterations == 1
    @test length(default_starts(root)) == 3^2 + 1
    @test_throws "max_starts" default_starts(root; max_starts=4)
end

# With this coupling sign, a start with both modes in the same well flips between (+,+) and (−,−) under the together-update.
# Whether mixing then switches on is decided by round-off in the rise count, so such a run may end at the symmetric saddle
# or unconverged; only the opposite-well pair is asserted.
@testset "two coupled fluxonia at half flux: one solution per pair of wells" begin
    pair = numeric_spec(; modes=[:a => :e, :b => :e], E_C=[1.0 0.0; 0.0 1.0], E_L=[1.0 0.1; 0.1 1.0],
                        flux_linear=[0.0, 0.0], n_g=[0.0, 0.0], cosines=[(4.0, [1.0, 0.0], π), (4.0, [0.0, 1.0], π)])
    grid = PhaseGrid(-12.0, 12.0, 161)
    A = Node([:a => grid]; solver=Dense(), compute=4, keep=lowest(4))
    B = Node([:b => grid]; solver=Dense(), compute=4, keep=lowest(4))
    root = Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))
    exact = energies(materialize(Node([:a => grid, :b => grid]; solver=LOBPCG(), compute=2, keep=lowest(2)), pair, (;)))[1]
    # (a) the opposite-well starts: a degenerate pair of distinct converged solutions, above the exact ground energy
    opposite = mean_field_solutions(root, pair, (;); starts=[Dict(:a => 2.0, :b => -2.0), Dict(:a => -2.0, :b => 2.0)])
    @test length(opposite) == 2 && all(s.converged for s in opposite)
    @test opposite[1].energy ≈ opposite[2].energy atol=1e-8
    @test all(s.energy >= exact - 1e-9 for s in opposite)
    # (b) all four starts: the opposite-well pair is lowest, and every run reports its iterations honestly
    starts = [Dict(:a => x, :b => y) for x in (2.0, -2.0), y in (2.0, -2.0)]
    sols = mean_field_solutions(root, pair, (;); starts=vec(starts))
    @test sols[1].converged && sols[2].converged
    @test sols[1].energy ≈ opposite[1].energy atol=1e-8
    @test sols[2].energy ≈ opposite[1].energy atol=1e-8
    @test all(1 <= s.iterations <= 300 for s in sols)
    @test all(s.iterations == 300 for s in sols if !s.converged)
    # detail 8 under mixing: the reported children reproduce the reported energy (a same-well start switches mixing on)
    bare = IdDict(n => bare_child(root, circuit_terms(pair), (;), n) for n in (A, B))
    same = self_consistent(root, circuit_terms(pair), (;), bare; start=Dict(:a => 2.0, :b => 2.0))
    E = product_energy(root, circuit_terms(pair), (;), IdDict{Node,Float64}(n => c.bare_energy for (n, c) in same.children),
                       IdDict{Node,Expectations}(n => c.expectations for (n, c) in same.children))
    @test E ≈ same.energy rtol=1e-12
end
