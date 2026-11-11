using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: Retained, GivenSystem, Term, Factor, Coefficient, Port

@testset "retention of computed pairs" begin
    spec = load_spec(fixture("transmon.json")); θ = (E_J=30.0, E_C=1.2, n_g=0.25)
    b = :varphi_1 => ChargeBasis(30)                                               # 61 states
    # all 61 states kept: the high charge states, on negative charges, have no weight on the phase mask
    full = (@test_logs (:warn, r"no weight on the phase mask") match_mode=:any materialize(Node([b]; solver=Dense()), spec, θ)).root
    @test full.retained.complete && isempty(full.retained.unkept) && size(full.retained.vectors) == (61, 0)
    # Dense with a compute limit: the whole spectrum is retained, and the selection still sees `compute` pairs
    mn = materialize(Node([b]; solver=Dense(), compute=10, keep=lowest(4)), spec, θ).root
    r = mn.retained
    @test r isa Retained{Float64} && r.complete && !r.followed
    @test length(r.energies) == 61 && isapprox(r.energies, full.energies; atol=1e-10)
    @test mn.kept == 1:4 && r.unkept == 5:61 && size(r.vectors) == (61, 57)
    @test norm(r.vectors' * mn.vectors) < 1e-12 && r.vectors' * r.vectors ≈ I
    @test length(materialize(Node([b]; solver=Dense(), compute=10, keep=lowest(0)), spec, θ).root.kept) == 10
    mi = materialize(Node([b]; solver=Dense(), compute=6, keep=indices([1, 3, 4])), spec, θ).root
    @test mi.kept == [1, 3, 4] && mi.retained.unkept == vcat(2, 5:61)
    # :never keeps the energies and nothing else
    nv = materialize(Node([b]; solver=Dense(), compute=10, keep=lowest(4), retain=:never), spec, θ).root.retained
    @test nv.vectors === nothing && !nv.complete && nv.unkept == 5:10 && isapprox(nv.energies, full.energies[1:10]; atol=1e-10)
    @test_throws ArgumentError Node([b]; retain=:sometimes)
    # retention changes what is stored beside the kept states, not the kept states: the full decomposition of a retaining
    # compute-limited Dense node gives the same kept energies, vectors (with their fixed phases) and ports
    # (measured when the final review was written: at most 4e-13)
    twin(retain) = materialize(Node([b]; solver=Dense(), compute=10, keep=lowest(4), retain,
                                    ports=[(:varphi_1, nport()), (:varphi_1, expiphi(1.0))]), spec, θ)
    sa, sn = twin(:auto), twin(:never)
    @test sa.root.retained.complete && !sn.root.retained.complete
    @test energies(sa) ≈ energies(sn) atol=1e-12
    @test eigenvectors(sa) ≈ eigenvectors(sn) atol=1e-12
    for p in (nport(), expiphi(1.0))
        @test port(sa, :varphi_1, p) ≈ port(sn, :varphi_1, p) atol=1e-12
    end
    # a Dense node above 1000 states retains only when asked
    fl = load_spec(fixture("fluxonium.json")); θf = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=2.0)
    grid = :varphi_1 => PhaseGrid(-10π, 10π, 1201; order=8)
    big = materialize(Node([grid]; solver=Dense(), compute=6, keep=lowest(4)), fl, θf).root.retained
    @test big.vectors === nothing && !big.complete && length(big.energies) == 6
    always = materialize(Node([grid]; solver=Dense(), compute=6, keep=lowest(4), retain=:always), fl, θf).root.retained
    @test always.complete && size(always.vectors) == (1201, 1197)
    # an iterative node retains its computed-but-unkept pairs; a given system is complete
    sp2 = load_spec(fixture("two_transmons.json")); θ2 = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.15, g=0.3)
    leaf(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(12))
    it = materialize(Node([leaf(1), leaf(2)]; solver=LOBPCG(tol=1e-10), compute=8, keep=lowest(6)), sp2, θ2).root
    @test !it.retained.complete && it.retained.unkept == [7, 8] && size(it.retained.vectors) == (144, 2)
    @test length(it.retained.energies) == 8 && norm(it.retained.vectors' * it.vectors) < 1e-8
    g = GivenSystem(:a; energies=[0.0, 1.0], operators=Dict(:x => ComplexF64[0 1; 1 0]))
    gs = materialize(Node([g]; solver=Dense()), [Term(Coefficient(0.1), [Factor(:a, Port(:x, 0))])], (;))
    @test gs.root.children[1].retained.complete
    # `followed` records that a previous solution fixed the phases
    tr(ng) = (E_J=30.0, E_C=1.2, n_g=ng)
    fnode = Node([b]; solver=Dense(), keep=lowest(3), phase=:follow)
    s0 = materialize(fnode, spec, tr(0.20))                                       # no previous: the positive-sector rule
    s1 = materialize(fnode, spec, tr(0.21); previous=s0)
    @test !s0.root.retained.followed && s1.root.retained.followed
end
