@testset "derivatives at an LOBPCG node on the device equal the host's full-spectrum route" begin
    tree(solver) = dc_two_transmons(; leafkeep=12, solver, compute=8, keep=lowest(6))            # root dimension 144
    cg, ch = tree(LOBPCG(tol=1e-11, on=CuArray)), tree(Dense())
    sg, sh = materialize(cg.root, cg.spec, cg.θ), materialize(ch.root, ch.spec, ch.θ)
    # the root really runs on the device: its resolved solver and the forward operator of its linear solve are on CuArray
    @test KronFreeCircuits._resolve_node_solver(cg.root, sg.root.children).on === CuArray
    @test KronFreeCircuits.operator(sg).on === CuArray
    for x in (:g, :n_g1)
        dg = derivatives(sg; wrt=x, order=2); dh = derivatives(sh; wrt=x, order=2)
        @test energies(dg, x) ≈ energies(dh, x) rtol=1e-8 atol=1e-9
        @test energies(dg, x, x) ≈ energies(dh, x, x) rtol=1e-6 atol=1e-8
        @test norm(eigenvectors(dg, x) - eigenvectors(dh, x), Inf) <= 1e-6 * norm(eigenvectors(dh, x), Inf)
        @test norm(port(dg, x, :varphi_1, nport()) - port(dh, x, :varphi_1, nport()), Inf) <= 1e-6 * norm(port(dh, x, :varphi_1, nport()), Inf)
        @test !isempty(dg.linear_solves) && eigenvectors(dg, x) isa Matrix                      # results are host arrays
    end
end
