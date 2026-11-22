using KronFreeCircuits: Term, Factor, Coefficient, System, circuit_terms, constant_term, JLD2,
                        _write_terms, _read_terms, _write_theta, _read_theta, _write_system_terms, _read_system_terms,
                        _terms_identical, _identical, _make
@testset "spec-built terms are rebuilt" begin
    c = dc_two_transmons(); sys = materialize(c.root, c.spec, c.θ)
    path = tempname() * ".jld2"
    JLD2.jldopen(f -> _write_system_terms(f, sys), path, "w")
    JLD2.jldopen(path, "r") do f
        @test f["terms/from_spec"] == true
        @test length(f["terms/coefficients"]) == 0
        spec, terms, constant = _read_system_terms(f)
        @test spec_equal(spec, c.spec)
        @test _terms_identical(terms, sys.terms)
        @test _identical(constant, sys.constant)
    end
end
@testset "extra terms are written" begin
    θ = (E_J=30.0, E_C=1.2, n_g=0.25); spec = load_spec(fixture("transmon.json"))
    r = GivenSystem(:r; energies=[0.0, 5.0, 10.0], operators=Dict(:x => ComplexF64[0 1 0; 1 0 √2; 0 √2 0]))
    root = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(4)), r]; solver=Dense(), keep=lowest(6))
    t = Term(Coefficient(0.2), [Factor(:varphi_1, nport()), Factor(:r, Port(:x, 0))])
    sys = materialize(root, spec, θ; extra_terms=[t])
    path = tempname() * ".jld2"
    JLD2.jldopen(f -> _write_system_terms(f, sys), path, "w")
    JLD2.jldopen(path, "r") do f
        @test length(f["terms/coefficients"]) == 1
        _, terms, constant = _read_system_terms(f)
        @test _terms_identical(terms, sys.terms)
        @test _identical(constant, sys.constant)
    end
end
@testset "edited term list falls back" begin
    c = dc_two_transmons(); spec = c.spec; ts = circuit_terms(spec)
    s1 = materialize(c.root, vcat(ts[2:end], ts[1:1]), c.θ; constant=constant_term(spec))
    full = materialize(c.root, spec, c.θ)
    s2 = System(full.root, full.spec, reverse(full.terms), full.θ, full.constant)
    for sys in (s1, s2)
        path = tempname() * ".jld2"
        JLD2.jldopen(f -> _write_system_terms(f, sys), path, "w")
        JLD2.jldopen(path, "r") do f
            @test f["terms/from_spec"] == false
            @test length(f["terms/coefficients"]) == length(sys.terms)
            sp, terms, constant = _read_system_terms(f)
            @test (sp === nothing) == (sys.spec === nothing)
            @test _terms_identical(terms, sys.terms)
            @test _identical(constant, sys.constant)
        end
    end
end
@testset "refuses a coefficient that would not round-trip" begin
    t = Term(_make(:(E_J * $(Float32(0.5))), [:E_J]), [Factor(:a, nport())])
    @test_throws "does not round-trip" JLD2.jldopen(f -> _write_terms(f, "terms", [t]), tempname() * ".jld2", "w")
end
@testset "unicode names" begin
    θ = (φ₁=0.3, E_J′=2)
    t = Term(_make(:(E_J′ * cos(φ₁)), [:E_J′, :φ₁]), [Factor(:φ₁, nport())])
    path = tempname() * ".jld2"
    JLD2.jldopen(path, "w") do f
        _write_theta(f, "theta", θ); _write_terms(f, "terms", [t])
    end
    JLD2.jldopen(path, "r") do f
        θ′ = _read_theta(f, "theta")
        @test θ′ == θ && typeof(θ′.E_J′) == Int
        @test _terms_identical(_read_terms(f, "terms"), [t])
    end
    @test_throws ArgumentError JLD2.jldopen(f -> _write_theta(f, "theta", (z=1.0im,)), tempname() * ".jld2", "w")
end
