module PathologicalDerivativeReview
using Test, LinearAlgebra, Random, KronFreeCircuits
using KronFreeCircuits: Port, Factor, Term, Coefficient

@testset "analytic complex two-level derivatives: small gaps, energy shifts, reordered selection" begin
    Z = ComplexF64[1 0; 0 -1]
    Y = ComplexF64[0 -im; im 0]
    g = GivenSystem(:spin; energies=[0.0, 0.0], operators=Dict(:z => Z, :y => Y))
    ps = [:x, :y, :shift]
    terms = [Term(Coefficient("x", ps), [Factor(:spin, Port(:z, 0))]),
             Term(Coefficient("y", ps), [Factor(:spin, Port(:y, 0))])]
    for θ in ((x=0.3, y=0.4, shift=0.0), (x=3e-5, y=4e-5, shift=0.0),
              (x=-0.3, y=0.4, shift=1000.0)), perm in ([1, 2], [2, 1])
        root = Node([g]; solver=Dense(), keep=indices(perm), ports=[(:spin, Port(:z, 0))])
        sys = materialize(root, terms, θ; constant=Coefficient("shift + x*y", ps))
        d = derivatives(sys; wrt=(:x, :y, :shift), order=2, mixed=((:x, :y),), given_independent=true)
        x, y, R = θ.x, θ.y, hypot(θ.x, θ.y)
        signs = [-1.0, 1.0][perm]
        @test energies(sys) ≈ θ.shift .+ x*y .+ signs .* R atol=1e-12
        @test energies(d, :x) ≈ y .+ signs .* (x/R) atol=1e-9
        @test energies(d, :y) ≈ x .+ signs .* (y/R) atol=1e-9
        @test energies(d, :shift) ≈ ones(2) atol=1e-12
        @test energies(d, :x, :x) ≈ signs .* (y^2/R^3) rtol=1e-8 atol=1e-9
        @test energies(d, :y, :y) ≈ signs .* (x^2/R^3) rtol=1e-8 atol=1e-9
        @test energies(d, :x, :y) ≈ 1 .- signs .* (x*y/R^3) rtol=1e-8 atol=1e-9
        @test energies(d, :shift, :shift) ≈ zeros(2) atol=1e-10
        @test real(diag(port(d, :x, :spin, Port(:z, 0)))) ≈ signs .* (y^2/R^3) rtol=1e-8 atol=1e-9
        V, Vx, Vxx = eigenvectors(sys), eigenvectors(d, :x), eigenvectors(d, :x, :x)
        @test norm(V' * Vx + Vx' * V) <= 1e-10 * max(1, norm(Vx))
        @test norm(V' * Vxx + Vxx' * V + 2Vx' * Vx) <= 1e-10 * max(1, norm(Vxx))
    end
end

@testset "union basis: complex duplicate subspaces exhaust rank" begin
    rng = MersenneTwister(11011)
    Q = Matrix(qr(randn(rng, ComplexF64, 9, 3)).Q)
    W = Matrix(qr(randn(rng, ComplexF64, 3, 3)).Q)
    sources = [BasisSource("original", Q, [0., 1., 2.]),
               BasisSource("rotated", Q*W, [0., 1., 2.]),
               BasisSource("rephased", Q[:, [3, 1, 2]] * Diagonal(cis.([0.3, 1.2, -0.7])), [0., 1., 2.])]
    saved = [copy(s.vectors) for s in sources]
    U, report = union_basis(sources; m=5, allow_fewer=true)
    @test size(U) == (9, 3)
    @test U' * U ≈ I atol=1e-12
    @test U * U' ≈ Q * Q' atol=1e-12
    @test sum(report.taken) == 3 && sum(report.dropped) == 6
    @test all(s.vectors == v for (s, v) in zip(sources, saved))
    @test_throws ArgumentError union_basis(sources; m=5)
end

@testset "charge-cosine coefficient derivatives through n_g and offset" begin
    spec = parse_spec("""
    {"format":"kronfree-spec","version":1,"units":"GHz",
     "parameters":["E_C","E_L","A","n_g","beta","x"],
     "modes":[{"name":"q","type":"h"}],
     "E_C":[["E_C"]], "E_L":[["E_L"]],
     "flux_linear":["0"], "n_g":["n_g"], "cosines":[],
     "charge_cosines":[{"energy":"A","kappa":[0.37],"offset":"beta*x"}]}
    """)
    θ = (E_C=0.7, E_L=1.4, A=0.8, n_g=0.17, beta=0.31, x=-0.4)
    root = Node([:q => FockBasis(18; phi_zpf=(2θ.E_C/θ.E_L)^0.25)];
                solver=Dense(), compute=10, keep=lowest(5),
                ports=[(:q, exp2pin(0.37))])
    at(t) = materialize(root, spec, t)
    sys = at(θ)
    d = derivatives(sys; wrt=(:A, :n_g, :x), order=2,
                    mixed=((:A, :n_g), (:A, :x), (:n_g, :x)))
    shifted(t, p, h) = merge(t, NamedTuple{(p,)}((t[p] + h,)))
    first(f, p, h) = (f(shifted(θ, p, h)) - f(shifted(θ, p, -h))) / (2h)
    second(f, p, h) = (f(shifted(θ, p, h)) - 2f(θ) + f(shifted(θ, p, -h))) / h^2
    mixed(f, p, q, h) = (f(shifted(shifted(θ, p, h), q, h)) -
                              f(shifted(shifted(θ, p, h), q, -h)) -
                              f(shifted(shifted(θ, p, -h), q, h)) +
                              f(shifted(shifted(θ, p, -h), q, -h))) / (4h^2)
    energy(t) = energies(at(t))[1]
    for p in (:A, :n_g, :x)
        @test energies(d, p)[1] ≈ first(energy, p, 2e-4) rtol=2e-6 atol=2e-8
        @test energies(d, p, p)[1] ≈ second(energy, p, 2e-3) rtol=2e-5 atol=2e-6
    end
    for (p, q) in ((:A, :n_g), (:A, :x), (:n_g, :x))
        @test energies(d, p, q)[1] ≈ mixed(energy, p, q, 2e-3) rtol=2e-5 atol=2e-6
    end
    # The formed non-Hermitian exp(2π i κ n) port also carries the state derivative.
    port11(t) = port(at(t), :q, exp2pin(0.37))[1, 1]
    @test port(d, :n_g, :q, exp2pin(0.37))[1, 1] ≈ first(port11, :n_g, 2e-4) rtol=2e-6 atol=2e-8
end
end
