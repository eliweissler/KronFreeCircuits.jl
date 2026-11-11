using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: materialized, _all_nodes

# every node's energies, vectors, ports and joint ports against finite differences of `materialize`
function check_tree(c, x; h=1e-3, rtol=1e-6, atol=1e-9, kwargs...)
    at = dc_memo(c); sys = at(c.θ); d = derivatives(sys; wrt=x, kwargs...)
    for node in _all_nodes(c.root)
        mn = materialized(sys, node)
        check_fd(energies(d, x, node), fd_derivative(θ -> energies(at(θ), node), c.θ, x; h); rtol, atol)
        check_fd(eigenvectors(d, x, node), fd_derivative(θ -> eigenvectors(at(θ), node), c.θ, x; h); rtol, atol)
        for (mode, p) in keys(mn.ports)
            @test eltype(port(d, x, mode, p, node)) === eltype(mn.ports[(mode, p)])
            check_fd(port(d, x, mode, p, node), fd_derivative(θ -> port(at(θ), mode, p, node), c.θ, x; h); rtol, atol)
        end
        for K in keys(mn.joint_ports)
            check_fd(port(d, x, K, node), fd_derivative(θ -> port(at(θ), K, node), c.θ, x; h); rtol, atol)
        end
    end
    return d
end

@testset "two-level trees against finite differences" begin
    for x in (:g, :n_g1, :E_J2, :E_C1); check_tree(dc_two_transmons(), x); end                 # real
    for x in (:g, :n_g1); check_tree(dc_two_transmons(periodic=true), x); end                  # complex leaves and root
    for x in (:varphi_ext, :n_g, :E_J); check_tree(dc_zero_pi(), x); end                       # real, complex coefficients on paired ports
    check_tree(dc_zero_pi(), :E_L; h=2.5e-4)                                                   # E_L = 0.04: a step of 1e-3 is too coarse (measured)
    for x in (:varphi_ext, :E_J3, :n_ga, :E_J); check_tree(dc_abc(), x); end                   # joint ports; complex root on real children
    @test maximum(abs, derivatives(dc_solve(dc_abc()); wrt=:varphi_ext).first[:varphi_ext].gauge) > 1e-3
end

@testset "a three-level tree against finite differences" begin
    for x in (:g12, :g23, :E_J1, :n_g2); check_tree(dc_chain3(), x); end
end

@testset "Review Focus 1: a coupling switched off at θ still has a derivative" begin
    c = dc_two_transmons(g=0.0)
    d = check_tree(c, :g)
    @test maximum(abs, eigenvectors(d, :g)) > 0.5                                # the root states do move with g
    @test maximum(abs, energies(d, :g, c.leaves[1])) > 0.01                      # and so do the leaves, through -g n_g2 n_1
    # (the root energies' own derivative is of order 1e-6 here; `check_tree`'s atol = 1e-9 covers its finite difference)
end

@testset "the three contributions" begin
    c = dc_chain3(); sys = dc_solve(c)
    fd = first(fd_derivative(θ -> energies(dc_solve(c, θ)), c.θ, :E_J1; h=1e-3))
    @test norm(fd, Inf) > 0.5
    # E_J1 enters only inside a leaf: with coefficients alone, every node above the leaf sees nothing
    @test all(iszero, energies(derivatives(sys; wrt=:E_J1, _parts=(:coefficients,)), :E_J1))
    # without the children's energy derivatives
    @test norm(energies(derivatives(sys; wrt=:E_J1, _parts=(:coefficients, :ports)), :E_J1) - fd, Inf) > 0.1
    # without the children's port derivatives (n_g1 moves the leaf's charge operator)
    c2 = dc_two_transmons(); s2 = dc_solve(c2)
    fd2 = first(fd_derivative(θ -> energies(dc_solve(c2, θ)), c2.θ, :n_g1; h=1e-3))
    @test norm(energies(derivatives(s2; wrt=:n_g1, _parts=(:coefficients, :energies)), :n_g1) - fd2, Inf) > 1e-2
    @test norm(energies(derivatives(s2; wrt=:n_g1), :n_g1) - fd2, Inf) < 1e-7
end

@testset "the API" begin
    c = dc_chain3(); sys = dc_solve(c)
    d = derivatives(sys; wrt=(:g12, :E_J1))
    @test d isa Derivatives && d.wrt == [:g12, :E_J1] && d.order == 1
    @test derivatives(sys; wrt=[:g12, :E_J1, :g12]).wrt == [:g12, :E_J1]
    @test energies(derivatives(sys; wrt=:g12), :g12) == energies(d, :g12)
    # Review Focus 2: the constant 4 n_gᵀ E_C n_g moves with n_g2 by 8 (1.0·0.09 + (0.3/8)·0.21 + (0.2/8)·0.1) = 0.803 here.
    # `check_tree(dc_chain3(), :n_g2)` above compares absolute energies at every node; here the pass's root derivative is
    # shown to be the derivative of the constant-free energies plus exactly that shift, and the children's to carry none.
    at = dc_memo(c); dn = derivatives(sys; wrt=:n_g2)
    bare(θ) = (t = at(θ); energies(t) .- real(t.constant(t.θ)))
    check_fd(energies(dn, :n_g2) .- 0.803, fd_derivative(bare, c.θ, :n_g2; h=1e-3); rtol=1e-6)
    check_fd(energies(dn, :n_g2, c.inner), fd_derivative(θ -> energies(at(θ), c.inner), c.θ, :n_g2; h=1e-3); rtol=1e-6)
    # a subtree with no dependence on the parameter reads as zeros of the right shape
    m3 = materialized(sys, c.third)
    @test all(iszero, energies(d, :g12, c.third)) && size(eigenvectors(d, :g12, c.third)) == size(m3.vectors)
    @test all(iszero, port(d, :g12, :varphi_3, nport(), c.third)) && d.first[:g12].children[2] === nothing
    @test_throws ArgumentError energies(d, :g23)                                 # not in `wrt`
    @test_throws ArgumentError port(d, :g12, :varphi_2, n2port())                # not formed
    @test_throws ArgumentError derivatives(sys; wrt=:nope)
    @test_throws ArgumentError derivatives(sys; wrt=:g12, scale=(E_J1=0.1,))
    @test_throws ArgumentError derivatives(sys; wrt=:g12, scale=(g12=-1.0,))
    @test_throws ArgumentError derivatives(sys; wrt=:g12, r=0.0)
    @test_throws "`degenerate_tol` must be nonnegative" derivatives(sys; wrt=:g12, degenerate_tol=-1e-9)
    @test_throws "`lintol` must be positive" derivatives(sys; wrt=:g12, lintol=0.0)
    @test_throws "`linmaxiter` must be positive" derivatives(sys; wrt=:g12, linmaxiter=0)
    @test_throws "node on modes [:varphi_9]" energies(d, :g12, Node([:varphi_9 => ChargeBasis(2)]; solver=Dense()))
    # the spec-form spelling of a port, at first and second order, as `port(sys, kind, mode, node; k)`
    @test port(d, :g12, :n, :varphi_1) == port(d, :g12, :varphi_1, nport())
    @test port(d, :g12, :n, :varphi_1, c.inner) == port(d, :g12, :varphi_1, nport(), c.inner)
    d2 = derivatives(sys; wrt=(:g12, :E_J1), order=2, mixed=((:g12, :E_J1),))
    @test port(d2, :g12, :E_J1, :n, :varphi_1) == port(d2, :g12, :E_J1, :varphi_1, nport())
    @test port(d2, :E_J1, :E_J1, :n, :varphi_1, c.inner) == port(d2, :E_J1, :E_J1, :varphi_1, nport(), c.inner)
    @test isempty(Test.detect_ambiguities(KronFreeCircuits))
    # a parameter that nothing depends on: an error when the spec does not declare it (a θ key the spec never named, or a
    # parameter `substitute` replaced by a number), else zeros with one warning
    extra = dc_solve(c, merge(c.θ, (unused=1.0,)))
    @test_throws "nothing in this system depends on and which its spec does not declare" derivatives(extra; wrt=:unused)
    @test_throws ":E_J1, :E_J2, :E_J3, :g12, :g23, :n_g2" derivatives(extra; wrt=:unused)
    sub = KronFreeCircuits.substitute(c.spec, (n_g2=0.09,))
    ssub = quiet(() -> materialize(c.root, sub, c.θ))
    @test_throws ":n_g2" derivatives(ssub; wrt=:n_g2)
    declared = KronFreeCircuits.parse_spec(replace(chain3_json, "\"n_g2\"]" => "\"n_g2\", \"spare\"]"))
    sdecl = quiet(() -> materialize(c.root, declared, merge(c.θ, (spare=1.0,))))
    dz = @test_logs (:warn, "nothing in this system depends on :spare; its derivatives are zero") derivatives(sdecl; wrt=(:spare, :g12))
    @test all(iszero, energies(dz, :spare)) && energies(dz, :g12) ≈ energies(d, :g12)
    sterms = quiet(() -> materialize(c.root, circuit_terms(c.spec), merge(c.θ, (spare=1.0,)); constant=constant_term(c.spec)))
    @test_logs (:warn, r"depends on :spare") derivatives(sterms; wrt=:spare)
    # a node whose phases followed a previous solution cannot be differentiated; the same node solved fresh can
    spec = load_spec(fixture("transmon.json")); θ(ng) = (E_J=30.0, E_C=1.2, n_g=ng)
    leaf = Node([:varphi_1 => ChargeBasis(30)]; solver=Dense(), keep=lowest(3), phase=:follow)
    s0 = materialize(leaf, spec, θ(0.20))
    @test_throws "previous" derivatives(materialize(leaf, spec, θ(0.21); previous=s0); wrt=:n_g)
    @test all(isfinite, energies(derivatives(s0; wrt=:n_g), :n_g))
end

@testset "augmented and given-system children, and a node without its complete spectrum, are differentiated (an undeclared given system is refused)" begin
    # an augmented child is differentiated at its fixed stored basis (Task 10; test_derivatives_stored.jl checks values)
    c = dc_two_transmons(); terms = circuit_terms(c.spec); B = c.leaves[2]
    bareB = bare_child(c.root, terms, c.θ, B)
    augB = augment(c.root, terms, c.θ, B, bareB, Matrix{ComplexF64}(bareB.vectors), UnionReport(["bare"], [4], [0]))
    sa = quiet(() -> materialize(c.root, c.spec, c.θ; augmented=[augB]))
    @test all(isfinite, energies(derivatives(sa; wrt=:g), :g))
    # a `GivenSystem` child (its energies and operators are numbers of unknown dependence) is refused unless declared
    # parameter-independent (Task 10)
    X = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
    ga = GivenSystem(:a; energies=[0.0, 1.0, 2.5], operators=Dict(:x => X))
    gb = GivenSystem(:b; energies=[0.0, 0.8], operators=Dict(:x => X[1:2, 1:2]))
    top = Node([ga, gb]; solver=Dense(), keep=lowest(6))
    sg = materialize(top, [Term(Coefficient("g", [:g]), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])], (g=0.1,))
    @test_throws "GivenSystem :a" derivatives(sg; wrt=:g)
    @test all(isfinite, energies(derivatives(sg; wrt=:g, given_independent=true), :g))
    # a node without its complete spectrum is no longer refused: the sum over states is completed by a linear solve on
    # the complement of its computed states (Task 6), recorded with its true residual
    si = dc_solve(dc_two_transmons(solver=Lanczos(tol=1e-12), compute=8))
    di = derivatives(si; wrt=:g)
    @test !isempty(di.linear_solves) && all(s -> s.residual <= 1e-9, di.linear_solves)
    @test energies(di, :g) ≈ energies(derivatives(dc_solve(dc_two_transmons()); wrt=:g), :g) rtol=1e-8 atol=1e-9
end
