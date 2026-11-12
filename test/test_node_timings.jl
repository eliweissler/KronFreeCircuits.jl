@testset "node_timings" begin
    θ = two_transmons_params(g=0.3); spec = load_spec(fixture("two_transmons.json"))
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(8))
    top = Node([mk(1), mk(2)]; solver=LOBPCG(tol=1e-9), compute=6, keep=lowest(6))
    plain = materialize(top, spec, θ)
    sys, recs = node_timings(() -> materialize(top, spec, θ))

    @test energies(sys) == energies(plain)                # the hook changes nothing it times
    @test length(recs) == 3                               # two leaves and the root, children first
    @test [r.modes for r in recs] == [[:varphi_1], [:varphi_2], [:varphi_1, :varphi_2]]
    leaf, root = recs[1], recs[3]
    @test leaf.dims == [61] && leaf.dim == 61 && leaf.kept == 8 && leaf.solver === :Dense
    @test leaf.iterations === nothing && leaf.n_matvec === nothing && leaf.warm === nothing
    @test root.dims == [8, 8] && root.dim == 64 && root.compute == 6 && root.kept == 6 && root.solver === :LOBPCG
    @test root.iterations isa Int && root.iterations >= 1 && root.n_matvec isa Int && root.n_matvec >= 6
    @test root.warm === false
    @test all(r -> r.operator_seconds >= 0 && r.solve_seconds >= 0 && r.assemble_seconds >= 0, recs)

    # a warm start from `previous` is reported, and outside `node_timings` nothing is recorded
    _, recs2 = node_timings(() -> materialize(top, spec, θ; previous=plain))
    @test recs2[end].warm === true
    @test KronFreeCircuits._NODE_LOG[] === nothing
end
