using Test, LinearAlgebra, Random, KronFreeCircuits
using KronFreeCircuits: GivenSystem, Term, Factor, Coefficient, Port, UnionReport

@testset "an augmented leaf child: derivatives at fixed basis" begin
    c = dc_zero_pi(); terms = circuit_terms(c.spec); rep = UnionReport(["bare"], [10], [0])
    bare(θ) = quiet(() -> bare_child(c.root, terms, θ, c.B))
    U = bare(c.θ).vectors * Matrix(qr(randn(MersenneTwister(5), ComplexF64, 10, 10)).Q)       # fixed for every θ
    cache = Dict{Any,Any}()                                                                   # one set of solves for every comparison
    at(θ) = get!(() -> quiet(() -> materialize(c.root, c.spec, θ; augmented=[augment(c.root, terms, θ, c.B, bare(θ), U, rep)])), cache, θ)
    sys = at(c.θ)
    for (x, h) in ((:E_L, 2.5e-4), (:E_CJ, 1e-2), (:E_J, 1e-3), (:varphi_ext, 1e-3))          # E_L = 0.04 and E_CJ = 20 set their steps
        d = derivatives(sys; wrt=x)
        check_fd(energies(d, x), fd_derivative(θ -> energies(at(θ)), c.θ, x; h); rtol=1e-6, atol=1e-9)
        check_fd(energies(d, x, c.B), fd_derivative(θ -> energies(at(θ), c.B), c.θ, x; h); rtol=1e-6, atol=1e-9)
        check_fd(eigenvectors(d, x, c.B), fd_derivative(θ -> eigenvectors(at(θ), c.B), c.θ, x; h); rtol=1e-6, atol=1e-9)
        check_fd(port(d, x, :phi, phiport()), fd_derivative(θ -> port(at(θ), :phi, phiport()), c.θ, x; h); rtol=1e-6, atol=1e-9)
    end
    d2 = derivatives(sys; wrt=(:E_L, :E_J), order=2, mixed=((:E_J, :E_L),))
    check_fd(energies(d2, :E_J, :E_L), fd_mixed(θ -> energies(at(θ)), c.θ, :E_J, :E_L; hx=2e-2, hλ=5e-4); rtol=1e-5, atol=1e-6)
end

@testset "an augmented branch child, whose own children move" begin
    c = dc_chain3(); terms = circuit_terms(c.spec); rep = UnionReport(["bare"], [10], [0])
    bare(θ) = quiet(() -> bare_child(c.root, terms, θ, c.inner))
    U = Matrix{ComplexF64}(bare(c.θ).vectors)                                                 # 16 × 10 coefficients, fixed
    aug(θ) = augment(c.root, terms, θ, c.inner, bare(θ), U, rep)
    cache = Dict{Any,Any}()
    at(θ) = get!(() -> quiet(() -> materialize(c.root, c.spec, θ; augmented=[aug(θ)])), cache, θ)
    sys = at(c.θ)
    for x in (:g12, :g23, :E_J1)
        d = derivatives(sys; wrt=x)
        check_fd(energies(d, x), fd_derivative(θ -> energies(at(θ)), c.θ, x; h=1e-3); rtol=1e-6, atol=1e-9)
        check_fd(port(d, x, :varphi_1, nport()), fd_derivative(θ -> port(at(θ), :varphi_1, nport()), c.θ, x; h=1e-3); rtol=1e-6, atol=1e-9)
    end
    # the augmented child was built at another value of a parameter that only a node below it uses: materialize does not
    # see that (it checks the child's own bare terms); the derivative pass must refuse
    θ1 = shifted(c.θ, :E_J1, 0.5)
    stale = quiet(() -> materialize(c.root, c.spec, θ1; augmented=[aug(c.θ)]))           # materialize accepts it
    @test_throws "the augmented child on modes" derivatives(stale; wrt=:g23)
    @test_throws "E_J1" derivatives(stale; wrt=:g23)
end

@testset "second order through an augmented branch child beside a real sibling" begin
    # the augmented child runs in complex arithmetic and its sibling stays real: the root's second-order operator mixes
    # their derivative data (`_second_replacements`)
    c = dc_chain3(); terms = circuit_terms(c.spec); rep = UnionReport(["bare"], [10], [0])
    bare(θ) = quiet(() -> bare_child(c.root, terms, θ, c.inner))
    U = Matrix{ComplexF64}(bare(c.θ).vectors)
    cache = Dict{Any,Any}()
    at(θ) = get!(() -> quiet(() -> materialize(c.root, c.spec, θ; augmented=[augment(c.root, terms, θ, c.inner, bare(θ), U, rep)])), cache, θ)
    sys = at(c.θ)
    @test materialized(sys, c.inner) isa KronFreeCircuits.MaterializedNode{ComplexF64}
    @test materialized(sys, c.third) isa KronFreeCircuits.MaterializedNode{Float64}
    d = derivatives(sys; wrt=(:g12, :g23, :E_J1), order=2, mixed=((:g12, :g23), (:E_J1, :g23)))
    # the augmented node holds the whole spectrum of its projected problem: no linear solve, no selection boundary
    @test isempty(d.linear_solves) && isempty(d.boundary)
    for (x, λ) in ((:g12, :g12), (:g12, :g23), (:E_J1, :g23))
        check_fd(energies(d, x, λ), fd_pair(θ -> energies(at(θ)), c.θ, x, λ; hx=1e-2); rtol=1e-5, atol=1e-6)
        check_fd(port(d, x, λ, :varphi_1, nport()), fd_pair(θ -> port(at(θ), :varphi_1, nport()), c.θ, x, λ; hx=1e-2);
                 rtol=1e-5, atol=1e-6)
    end
end

@testset "a given system is refused unless declared parameter-independent" begin
    g1 = GivenSystem(:a; energies=[0.0, 1.0, 2.5], operators=Dict(:x => ComplexF64[0 1 0; 1 0 √2; 0 √2 0]))
    g2 = GivenSystem(:b; energies=[0.0, 0.8], operators=Dict(:x => ComplexF64[0 1; 1 0]))
    top = Node([g1, g2]; solver=Dense(), keep=lowest(6), ports=[(:a, Port(:x, 0))])
    terms = [Term(Coefficient("g", [:g]), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])]
    gsolve(θ) = materialize(top, terms, θ)
    θ = (g=0.1,); sys = gsolve(θ)
    @test_throws "given_independent" derivatives(sys; wrt=:g)
    d = derivatives(sys; wrt=:g, order=2, given_independent=true)
    check_fd(energies(d, :g), fd_derivative(t -> energies(gsolve(t)), θ, :g; h=1e-3); rtol=1e-6)
    check_fd(energies(d, :g, :g), fd_second(t -> energies(gsolve(t)), θ, :g; h=1e-2); rtol=1e-5)
    check_fd(port(d, :g, :a, Port(:x, 0)), fd_derivative(t -> port(gsolve(t), :a, Port(:x, 0)), θ, :g; h=1e-3); rtol=1e-6)
end
