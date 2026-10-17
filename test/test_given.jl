using KronFreeCircuits: GivenSystem, Term, Factor, Coefficient
@testset "two given systems: explicit Kronecker reference" begin
    E1 = [0.0, 1.0, 2.5]; X1 = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
    E2 = [0.0, 0.8];     X2 = ComplexF64[0 1; 1 0]
    g1 = GivenSystem(:a; energies=E1, operators=Dict(:x => X1))
    g2 = GivenSystem(:b; energies=E2, operators=Dict(:x => X2))
    top = Node([g1, g2]; solver=Dense(), keep=lowest(6), ports=[(:a, Port(:x, 0))])
    terms = [Term(Coefficient(0.1), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])]
    sys = materialize(top, terms, (;))
    Href = kron(Matrix{ComplexF64}(I, 2, 2), Diagonal(E1)) + kron(Diagonal(E2), Matrix{ComplexF64}(I, 3, 3)) + 0.1 * kron(X2, X1)
    @test energies(sys) ≈ eigvals(Hermitian(Href))
    @test bare_labels(sys) == [Tuple(Int(i) for i in Tuple(c)) for c in CartesianIndices((3, 2))]
    @test port(sys, :a, Port(:x, 0)) ≈ port(sys, :a, Port(:x, 0))'
    # a single-mode term on a given mode is applied at the parent through the given port
    sys2 = materialize(top, vcat(terms, [Term(Coefficient(0.3), [Factor(:b, Port(:x, 0))])]), (;))
    @test energies(sys2) ≈ eigvals(Hermitian(Href + 0.3 * kron(X2, Matrix{ComplexF64}(I, 3, 3))))
end

@testset "a materialized transmon re-entered as a given system matches the node tree" begin
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    leaf1 = Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(5), ports=[(:varphi_1, nport())])
    leaf2 = Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(5))
    full = materialize(Node([leaf1, leaf2]; solver=Dense(), keep=lowest(8)), spec, θ)
    one = materialize(leaf1, load_spec(fixture("transmon.json")), (E_J=θ.E_J1, E_C=θ.E_C1, n_g=θ.n_g1))
    # `energies(one)` include transmon.json's constant 4 E_C n_g^2 (added at its root); the leaf's own
    # eigenvalues exclude it, as in the full tree, where constant_term(spec) carries it instead.
    g = GivenSystem(:q; energies=energies(one) .- real(one.constant(one.θ)),
                    operators=Dict(:n => port(one, :varphi_1, nport())))
    # The spec's own terms: those all on varphi_1 are the given system's Hamiltonian (the n_g2 cross term
    # among them vanishes, n_g2 = 0); in the rest, varphi_1 is renamed to the given mode.
    rename(t) = Term(t.coeff, [Factor(f.mode == :varphi_1 ? :q : f.mode, f.port) for f in t.factors])
    terms = [rename(t) for t in circuit_terms(spec) if !all(f.mode == :varphi_1 for f in t.factors)]
    sys = materialize(Node([g, leaf2]; solver=Dense(), keep=lowest(8)), terms, θ;
                      constant=constant_term(spec), modetypes=Dict(:varphi_2 => :c))
    @test energies(sys) ≈ energies(full) atol=1e-10
    # :phi is not provided (the spec's terms keep leaf2's Hamiltonian, so it materializes cleanly first)
    bad = [Term(Coefficient(1.0), [Factor(:q, Port(:phi, 0)), Factor(:varphi_2, nport())])]
    @test_throws ArgumentError materialize(Node([g, leaf2]; solver=Dense(), keep=lowest(8)), vcat(terms, bad), θ)
end

@testset "GivenSystem validation, extra_terms, follow with a given child" begin
    @test_throws ArgumentError GivenSystem(:a; energies=[1.0, 0.0], operators=Dict(:x => zeros(ComplexF64, 2, 2)))
    @test_throws ArgumentError GivenSystem(:a; energies=[0.0, 1.0], operators=Dict(:x => zeros(ComplexF64, 3, 3)))
    @test_throws ArgumentError GivenSystem(:a; energies=Float64[], operators=Dict{Symbol,Matrix{ComplexF64}}())
    θ = (E_J=30.0, E_C=1.2, n_g=0.25); spec = load_spec(fixture("transmon.json"))
    r = GivenSystem(:r; energies=[0.0, 5.0, 10.0], operators=Dict(:x => ComplexF64[0 1 0; 1 0 √2; 0 √2 0]))
    tree() = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(4)), r]; solver=Dense(), compute=6, keep=follow())
    extra = [Term(Coefficient(0.2), [Factor(:varphi_1, nport()), Factor(:r, Port(:x, 0))])]
    s1 = materialize(Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(4)), r]; solver=Dense(), keep=lowest(6)), spec, θ; extra_terms=extra)
    s2 = materialize(tree(), spec, θ; previous=s1, extra_terms=extra)
    @test energies(s2) ≈ energies(s1)
    @test materialized(s2, r).vectors == I
end
