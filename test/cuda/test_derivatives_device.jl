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

# ---- The derivative pass at a device node: complex nodes, mixed pairs, weak coupling, a near-degenerate block ----
# Each testset below solves the iterative node on the device and compares with the same tree's `Dense()` twin on the
# host, at the tolerances test/test_derivatives_iterative.jl uses for the same comparison on the CPU (lines cited).
# The storage assertions fail if the node's solve silently ran on the host.
dv_rel(a, b) = norm(a - b, Inf) / norm(b, Inf)
dv_offdiag(A) = A - Diagonal(diag(A))
dv_tree(solver; kw...) = dc_two_transmons(; leafkeep=12, solver, compute=8, keep=lowest(6), kw...)     # root dimension 144
function dv_pair(; kw...)
    cg, ch = dv_tree(LOBPCG(tol=1e-11, on=CuArray); kw...), dv_tree(Dense(); kw...)
    sg, sh = materialize(cg.root, cg.spec, cg.θ), materialize(ch.root, ch.spec, ch.θ)
    @test KronFreeCircuits._resolve_node_solver(cg.root, sg.root.children).on === CuArray
    @test KronFreeCircuits.operator(sg).on === CuArray
    return sg, sh, ch.root
end
# every linear solve of the pass converged to 10 · lintol, none at the round-off floor (test_derivatives_iterative.jl:23, :83)
function dv_solves_ok(d)
    @test !isempty(d.linear_solves)
    @test all(s -> isempty(s.at_floor) && s.residual <= 1e-9, d.linear_solves)
    return maximum(s.iterations for s in d.linear_solves), maximum(s.residual for s in d.linear_solves)
end
dv_max!(w, k, v) = (w[k] = max(get(w, k, 0.0), v); v)

@testset "derivatives at a complex LOBPCG node on the device equal the host's full-spectrum route" begin
    sg, sh, rh = dv_pair(periodic=true)                  # periodic grids: complex leaves and a complex root
    @test eltype(eigenvectors(sg)) <: Complex
    keys_ = collect(keys(KronFreeCircuits.materialized(sh, rh).ports))
    @test (:varphi_1, expiphi(1.0)) in keys_             # a non-Hermitian port among the root's ports
    w = Dict{Symbol,Float64}(); its, res = 0, 0.0
    for x in (:E_J2, :g, :n_g1)
        dg = derivatives(sg; wrt=x, order=2); dh = derivatives(sh; wrt=x, order=2)
        (i, r) = dv_solves_ok(dg); its, res = max(its, i), max(res, r)
        # first order: test_derivatives_iterative.jl:7, :8, :10; per column 1e-9 at g = 0.3 (:88, :89)
        @test energies(dg, x) ≈ energies(dh, x) rtol=1e-8 atol=1e-9
        dv_max!(w, :E1, dv_rel(energies(dg, x), energies(dh, x)))
        @test dv_max!(w, :V1, dv_rel(eigenvectors(dg, x), eigenvectors(dh, x))) <= 1e-6
        @test dv_max!(w, :V1col, maximum(percol_relerr(eigenvectors(dg, x), eigenvectors(dh, x)))) <= 1e-9
        for (mode, p) in keys_
            @test dv_max!(w, :P1, dv_rel(port(dg, x, mode, p), port(dh, x, mode, p))) <= 1e-6
        end
        # the off-diagonal elements of the non-Hermitian port alone (the phase-convention term)
        Pg, Ph = port(dg, x, :varphi_1, expiphi(1.0)), port(dh, x, :varphi_1, expiphi(1.0))
        @test dv_max!(w, :P1off, dv_rel(dv_offdiag(Pg), dv_offdiag(Ph))) <= 1e-6
        # second order (x, x): test_derivatives_iterative.jl:65, :66, :68; per column 1e-9 at g = 0.3 (:90)
        @test energies(dg, x, x) ≈ energies(dh, x, x) rtol=1e-6 atol=1e-8
        dv_max!(w, :E2, dv_rel(energies(dg, x, x), energies(dh, x, x)))
        @test dv_max!(w, :V2, dv_rel(eigenvectors(dg, x, x), eigenvectors(dh, x, x))) <= 1e-5
        @test dv_max!(w, :V2col, maximum(percol_relerr(eigenvectors(dg, x, x), eigenvectors(dh, x, x)))) <= 1e-9
        for (mode, p) in keys_
            @test dv_max!(w, :P2, dv_rel(port(dg, x, x, mode, p), port(dh, x, x, mode, p))) <= 1e-5
        end
        Pg, Ph = port(dg, x, x, :varphi_1, expiphi(1.0)), port(dh, x, x, :varphi_1, expiphi(1.0))
        @test dv_max!(w, :P2off, dv_rel(dv_offdiag(Pg), dv_offdiag(Ph))) <= 1e-5
        @test eigenvectors(dg, x) isa Matrix
    end
    @info "complex device node: worst relative disagreement with Dense; largest linear-solve iteration count and residual" (; w...) its res
end

@testset "mixed second derivatives at an LOBPCG node on the device equal the host's full-spectrum route" begin
    pairs = ((:g, :n_g1), (:E_J2, :g))            # one parameter inside a leaf, one in the root's coupling
    w = Dict{Symbol,Float64}(); its, res = 0, 0.0
    for periodic in (false, true)
        sg, sh, rh = dv_pair(; periodic)
        keys_ = collect(keys(KronFreeCircuits.materialized(sh, rh).ports))
        dg = derivatives(sg; wrt=(:g, :n_g1, :E_J2), order=2, mixed=pairs)
        dh = derivatives(sh; wrt=(:g, :n_g1, :E_J2), order=2, mixed=pairs)
        (i, r) = dv_solves_ok(dg); its, res = max(its, i), max(res, r)
        for (x, λ) in pairs        # test_derivatives_iterative.jl:65, :66, :68
            @test energies(dg, x, λ) ≈ energies(dh, x, λ) rtol=1e-6 atol=1e-8
            dv_max!(w, :E, dv_rel(energies(dg, x, λ), energies(dh, x, λ)))
            @test dv_max!(w, :V, dv_rel(eigenvectors(dg, x, λ), eigenvectors(dh, x, λ))) <= 1e-5
            dv_max!(w, :Vcol, maximum(percol_relerr(eigenvectors(dg, x, λ), eigenvectors(dh, x, λ))))
            for (mode, p) in keys_
                @test dv_max!(w, :P, dv_rel(port(dg, x, λ, mode, p), port(dh, x, λ, mode, p))) <= 1e-5
            end
        end
    end
    @info "mixed pairs on the device: worst relative disagreement with Dense; largest linear-solve iteration count and residual" (; w...) its res
end

@testset "weak coupling at a complex LOBPCG node on the device: per column against the host's full-spectrum route" begin
    w = Dict{Float64,Float64}(); its, res = 0, 0.0
    for g in (0.03, 3e-4)
        sg, sh, rh = dv_pair(; periodic=true, g)
        dg = derivatives(sg; wrt=:E_J2); dh = derivatives(sh; wrt=:E_J2)
        (i, r) = dv_solves_ok(dg); its, res = max(its, i), max(res, r)
        # test_derivatives_iterative.jl:84, :85, :88, :89, :92
        @test energies(dg, :E_J2) ≈ energies(dh, :E_J2) rtol=1e-8 atol=1e-9
        @test dv_rel(eigenvectors(dg, :E_J2), eigenvectors(dh, :E_J2)) <= 1e-6
        @test (w[g] = maximum(percol_relerr(eigenvectors(dg, :E_J2), eigenvectors(dh, :E_J2)))) <= 1e-7
        for p in (nport(), expiphi(1.0))
            @test dv_rel(port(dg, :E_J2, :varphi_1, p), port(dh, :E_J2, :varphi_1, p)) <= 1e-6
        end
    end
    @info "weak coupling on the device: worst per-column relative disagreement with Dense, by g; largest iteration count and residual" sort(collect(w)) its res
end

# The KITE tree of test/test_derivatives_kite.jl (`kite_deriv_tree`, `KITE_SYM`; a copy, see that file for the fixture,
# the bases and the settings): three Dense leaves of twelve states and a complex root of dimension 1728.
dv_kite_tree(solver; retain=:auto) =
    Node([Node([m => b]; solver=Dense(), keep=lowest(12)) for (m, b) in
          (:varphi_1 => ChargeBasis(16), :varphi_2 => PhaseGrid(-18.915907192616967, 18.915907192616967, 120; order=8),
           :varphi_3 => PhaseGrid(-17.714011073078225, 17.714011073078225, 120; order=8))];
         solver, compute=6, keep=lowest(4), retain)

@testset "the KITE doublet at an LOBPCG node on the device: block and point derivatives equal the host's" begin
    spec = load_spec(fixture("kite/optimum_symbolic.json")); θ = (E_J=16.83, delta=0.0, n_g1=0.1)
    rootg, rooth = dv_kite_tree(LOBPCG(tol=1e-11, on=CuArray)), dv_kite_tree(Dense(); retain=:always)
    tg = @elapsed (sg = quiet(() -> materialize(rootg, spec, θ)))
    sh = quiet(() -> materialize(rooth, spec, θ))
    @test KronFreeCircuits._resolve_node_solver(rootg, sg.root.children).on === CuArray
    @test KronFreeCircuits.operator(sg).on === CuArray
    @test eltype(eigenvectors(sg)) <: Complex
    tdg = @elapsed (dg = derivatives(sg; wrt=:delta, order=2, scale=(delta=0.01,)))
    dh = derivatives(sh; wrt=:delta, order=2, scale=(delta=0.01,))
    its, res = dv_solves_ok(dg)
    # the block on the doublet, the same at both routes (relative 1e-9)
    bg = only(b for b in blocks(dg) if b.node === rootg && b.parameters == (:delta,) && 1 in b.states)
    bh = only(b for b in blocks(dh) if b.node === rooth && b.parameters == (:delta,) && 1 in b.states)
    @test bg.states == bh.states == [1, 2]
    @test bg.detuning ≈ bh.detuning rtol=1e-9
    @test bg.coupling ≈ bh.coupling rtol=1e-9
    # the effective Hamiltonian's levels, folded through the complement solve on the device (test_derivatives_kite.jl:79)
    lv = maximum(maximum(abs, block_levels(bg, (delta=δ,)).energies - block_levels(bh, (delta=δ,)).energies)
                 for δ in (0.01, -0.01, 0.005))
    @test lv <= 1e-9
    # point derivatives: energies at test_derivatives_kite.jl:74 and :75; vectors per column at the matrix-wide
    # tolerances of test_derivatives_iterative.jl:8 and :66, applied to each column
    @test energies(dg, :delta) ≈ energies(dh, :delta) rtol=1e-6
    @test energies(dg, :delta, :delta) ≈ energies(dh, :delta, :delta) rtol=1e-5
    v1 = maximum(percol_relerr(eigenvectors(dg, :delta), eigenvectors(dh, :delta)))
    v2 = maximum(percol_relerr(eigenvectors(dg, :delta, :delta), eigenvectors(dh, :delta, :delta)))
    @test v1 <= 1e-6
    @test v2 <= 1e-5
    @info "KITE on the device" materialize_s=tg derivatives_s=tdg its res detuning=(bg.detuning, bh.detuning) coupling=(bg.coupling, bh.coupling) levels=lv E1=dv_rel(energies(dg, :delta), energies(dh, :delta)) E2=dv_rel(energies(dg, :delta, :delta), energies(dh, :delta, :delta)) v1 v2
end
