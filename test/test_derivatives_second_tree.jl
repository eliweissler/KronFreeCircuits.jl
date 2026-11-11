using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: materialized, _all_nodes

# every node's second derivatives (energies, vectors, ports, joint ports) against finite differences of `materialize`
function check_tree2(c, x, λ; hx, hλ=hx, rtol=1e-5, atol=1e-7)
    at = dc_memo(c); sys = at(c.θ)
    d = derivatives(sys; wrt=unique([x, λ]), order=2, mixed=x === λ ? () : ((x, λ),))
    fd(f) = fd_pair(f, c.θ, x, λ; hx, hλ)
    for node in _all_nodes(c.root)
        mn = materialized(sys, node)
        check_fd(energies(d, x, λ, node), fd(θ -> energies(at(θ), node)); rtol, atol)
        check_fd(eigenvectors(d, x, λ, node), fd(θ -> eigenvectors(at(θ), node)); rtol, atol)
        for (mode, p) in keys(mn.ports)
            check_fd(port(d, x, λ, mode, p, node), fd(θ -> port(at(θ), mode, p, node)); rtol, atol)
        end
        for K in keys(mn.joint_ports)
            check_fd(port(d, x, λ, K, node), fd(θ -> port(at(θ), K, node)); rtol, atol)
        end
    end
    return d
end

@testset "second order through trees against finite differences" begin
    check_tree2(dc_two_transmons(), :g, :g; hx=1e-2)
    check_tree2(dc_two_transmons(), :E_J2, :E_J2; hx=2e-2)
    check_tree2(dc_two_transmons(), :g, :E_J2; hx=1e-2, hλ=2e-2)
    # complex leaves and root (Review Focus 3).  On the periodic grids the leaves' energies are of order 1e3 and their
    # second derivative in g of order 1e-3, so its finite difference carries round-off of 1e-7: hence atol = 1e-6.
    check_tree2(dc_two_transmons(periodic=true), :g, :g; hx=1e-2, atol=1e-6)
    check_tree2(dc_two_transmons(periodic=true), :g, :n_g1; hx=1e-2, hλ=5e-3)
    check_tree2(dc_chain3(), :g12, :g12; hx=1e-2)
    check_tree2(dc_chain3(), :g23, :g23; hx=1e-2)
    check_tree2(dc_chain3(), :E_J1, :g23; hx=2e-2, hλ=1e-2)
end

@testset "second order through trees: paired ports with complex coefficients, and joint ports" begin
    check_tree2(dc_zero_pi(), :varphi_ext, :varphi_ext; hx=1e-2)
    check_tree2(dc_zero_pi(), :E_J, :varphi_ext; hx=2e-2, hλ=1e-2)
    check_tree2(dc_abc(), :varphi_ext, :varphi_ext; hx=1e-2)
    check_tree2(dc_abc(), :E_J3, :varphi_ext; hx=1e-2)
end

@testset "Review Focus 5: a mixed pair whose parameters enter at different levels" begin
    # n_g1 and n_g2 enter inside leaves, g and g23 in a coupling above them; E_J of dc_abc inside the leaves, varphi_ext at the root
    check_tree2(dc_two_transmons(), :g, :n_g1; hx=1e-2, hλ=5e-3)
    check_tree2(dc_abc(), :E_J, :varphi_ext; hx=2e-2, hλ=1e-2)
    d = check_tree2(dc_chain3(), :n_g2, :g23; hx=5e-3, hλ=1e-2)
    @test norm(energies(d, :n_g2, :g23), Inf) > 1e-3
    # the result does not depend on the order in which the pair is given
    sys = dc_solve(dc_chain3())
    d1 = derivatives(sys; wrt=(:n_g2, :g23), order=2, mixed=((:n_g2, :g23),))
    d2 = derivatives(sys; wrt=(:g23, :n_g2), order=2, mixed=((:g23, :n_g2),))
    @test energies(d1, :n_g2, :g23) ≈ energies(d2, :n_g2, :g23) rtol=1e-10
    @test eigenvectors(d1, :g23, :n_g2) ≈ eigenvectors(d2, :n_g2, :g23) rtol=1e-9
    # the two calls above compute the same pass: the pass itself with the roles of x and λ swapped agrees to round-off
    K = KronFreeCircuits
    routed = K._derivative_terms(sys); res = IdDict{Any,K.Resolvent}(); parts = (:coefficients, :ports, :energies)
    lin = (; lintol=1e-10, linmaxiter=5000, log=nothing, truncate=false)
    pass1(x) = K._first_order_pass(sys, routed, x, res; degenerate_tol=1e-9, parts, lin,
                                   boundary=(; records=Any[], x, δx=nothing, r=0.1))
    fx, fλ = pass1(:n_g2), pass1(:g23)
    a = K._second_order_pass(sys, routed, :n_g2, :g23, fx, fλ, res; parts, records=Any[])
    b = K._second_order_pass(sys, routed, :g23, :n_g2, fλ, fx, res; parts, records=Any[])
    @test a.energies ≈ b.energies rtol=1e-12 atol=1e-12
    @test a.vectors ≈ b.vectors rtol=1e-12 atol=1e-12
    @test a.ports[(:varphi_1, nport())] ≈ b.ports[(:varphi_1, nport())] rtol=1e-12 atol=1e-12
end

@testset "a mixed derivative for dephasing; the two-parameter API" begin
    # ∂²f_01/∂E_J∂varphi_ext of the fluxonium off its sweet spots
    c = dc_fluxonium(); at = dc_memo(c); sys = at(c.θ)
    d = derivatives(sys; wrt=(:E_J, :varphi_ext), order=2, mixed=((:E_J, :varphi_ext),))
    f01(E) = E[2] - E[1]
    @test f01(energies(d, :E_J, :varphi_ext)) ≈ -0.053526672 rtol=1e-6          # measured by finite differences when the plan was written
    check_fd(f01(energies(d, :E_J, :varphi_ext)), fd_mixed(θ -> f01(energies(at(θ))), c.θ, :E_J, :varphi_ext; hx=1e-2, hλ=1e-2); rtol=1e-6)
    @test haskey(d.second, (:E_J, :E_J)) && haskey(d.second, (:E_J, :varphi_ext)) && haskey(d.second, (:varphi_ext, :varphi_ext))
    @test_throws ArgumentError energies(derivatives(sys; wrt=(:E_J, :varphi_ext), order=2), :E_J, :varphi_ext)   # pair not requested
    @test_throws ArgumentError derivatives(sys; wrt=:E_J, order=2, mixed=((:E_J, :varphi_ext),))                 # not in wrt
    @test_throws ArgumentError derivatives(sys; wrt=(:E_J, :varphi_ext), mixed=((:E_J, :varphi_ext),))            # needs order = 2
    @test_throws ArgumentError derivatives(sys; wrt=:E_J, order=3)
    # a subtree that does not depend on one of the two parameters has a zero mixed derivative
    c3 = dc_chain3(); d3 = derivatives(dc_solve(c3); wrt=(:g12, :E_J3), order=2, mixed=((:g12, :E_J3),))
    @test all(iszero, energies(d3, :g12, :E_J3, c3.inner)) && all(iszero, energies(d3, :g12, :E_J3, c3.third))
    @test norm(energies(d3, :g12, :E_J3), Inf) > 0
end

@testset "second order: joint ports of a child along both parameters; clashes and first-order NaN columns" begin
    # E_J enters both children of dc_abc: the leaf `ab` has second derivatives of its joint ports, the root reads them,
    # and the root's three-mode cosine has child data on two axes, so its ordered pairs (the factor 2 at x == λ) enter
    check_tree2(dc_abc(), :E_J, :E_J; hx=2e-2)
    # charges -1, 0, 1 at n_g = 0: |−1⟩ and |+1⟩ are degenerate, the second kept and the third unkept
    c0 = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(2), E_2=0.0); s0 = dc_solve(c0)
    # -cos φ couples the pair only through |0⟩, at second order: NaN, and a boundary record under the pair key
    d = @test_logs (:warn, r"^selection boundary: .*parameter :E_1") (:warn, r"degenerate") (:warn, r"^selection boundary: .*second order in \(:E_1, :E_1\)") derivatives(s0; wrt=:E_1, order=2)
    @test all(isfinite, energies(d, :E_1)) && isnan(energies(d, :E_1, :E_1)[2])
    @test energies(d, :E_1, :E_1)[1] ≈ -0.25 atol=1e-12
    b = only(filter(r -> r.parameter == (:E_1, :E_1), d.boundary))
    @test b.node === c0.root && b.kept == 2 && b.unkept == 3 && b.detuning == 0
    @test b.coupling ≈ 0.125 atol=1e-12
    # -cos 2φ couples them directly: NaN at first order stays NaN at second order, alone and mixed, with no new record
    dE = quiet(() -> derivatives(s0; wrt=(:E_1, :E_2), order=2, mixed=((:E_1, :E_2),)))
    @test isnan(energies(dE, :E_2)[2]) && isnan(energies(dE, :E_2, :E_2)[2]) && isnan(energies(dE, :E_1, :E_2)[2])
    @test all(isnan, eigenvectors(dE, :E_2, :E_2)[:, 2]) && all(isnan, eigenvectors(dE, :E_1, :E_2)[:, 2])
    @test isfinite(energies(dE, :E_2, :E_2)[1]) && isfinite(energies(dE, :E_1, :E_2)[1])
    @test !any(r -> r.parameter in ((:E_2, :E_2), (:E_1, :E_2)), dE.boundary)
end
