include("circuits.jl")

@testset "in-memory Spec equals the committed JSON: $name" for name in
        ("transmon", "fluxonium", "two_transmons", "three_transmons", "zero_pi", "full_zero_pi", "two_zero_pi")
    res, kw = fixture_result(name)
    a = Spec(res; kw...)
    b = load_spec(joinpath(@__DIR__, "..", "fixtures", "generated", name * ".json"))
    @test a.parameters == b.parameters
    @test [(m.name, m.type) for m in a.modes] == [(m.name, m.type) for m in b.modes]
    θ = NamedTuple{Tuple(a.parameters)}(Tuple(0.5 .+ rand(length(a.parameters))))
    same(x, y) = isapprox(x(θ), y(θ); rtol=1e-12, atol=1e-14)
    @test all(same.(a.E_C, b.E_C)) && all(same.(a.E_L, b.E_L))
    @test all(same.(a.flux_linear, b.flux_linear)) && all(same.(a.n_g, b.n_g)) && same(a.constant, b.constant)
    @test length(a.cosines) == length(b.cosines)
    @test all(c.omega == d.omega && same(c.energy, d.energy) && same(c.offset, d.offset) for (c, d) in zip(a.cosines, b.cosines))
end

@testset "Spec(res) keywords" begin
    res, kw = fixture_result("transmon")
    @test kw.subs == Dict("n_g1" => "n_g")
    θ = (E_C=0.3, n_g=0.2, E_J=7.0)

    s = Spec(res)                                                   # defaults: no subs, varphi_k mode names
    @test s.parameters == [:E_C, :n_g1, :E_J] && only(s.modes).name == :varphi_1

    s = Spec(res; subs=(n_g1="n_g",), mode_names=[:phase])          # a NamedTuple, and Symbol mode names
    @test s.parameters == [:E_C, :n_g, :E_J] && only(s.modes).name == :phase
    @test only(s.E_C)(θ) == 0.3 && only(s.n_g)(θ) == 0.2

    s = Spec(res; subs=Dict(:E_J => 2.5, "n_g1" => 1 // 4))          # numbers; Symbol and String keys
    @test s.parameters == [:E_C]
    @test only(s.cosines).energy((E_C=0.3,)) == 2.5 && only(s.n_g)((E_C=0.3,)) == 0.25

    s = Spec(res; subs=(E_J="E_C",))                                # a name that is another symbol of the circuit
    @test s.parameters == [:E_C, :n_g1] && only(s.cosines).energy((E_C=0.3, n_g1=0.0)) == 0.3

    err = try
        Spec(res; subs=(E_K=1.0,))
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("E_K", err.msg) && occursin("E_C, E_J, n_g1", err.msg)

    res, kw = fixture_result("two_transmons")                       # simplify=false keeps the same functions
    a, b = Spec(res; kw..., simplify=false), Spec(res; kw...)
    @test sort(a.parameters) == sort(b.parameters)                  # the order follows the printed expressions
    θ = NamedTuple{Tuple(a.parameters)}(Tuple(0.5 .+ rand(length(a.parameters))))
    @test all(isapprox(x(θ), y(θ); rtol=1e-12) for (x, y) in zip(a.E_C, b.E_C))
end

@testset "Spec(res) round-off zeroing reaches to_spec" begin
    include("kite.jl")
    iz = KronFreeCircuits.iszero_expr
    res, θ, frames = kite()
    # The KITE at θ in the normal frame turned by 1e-10 rad: numeric Float matrices, with E_L[2,3] ≈ 6e-12 of max|E_L|.
    ε = 1e-10
    S = frames.normal * [1 0 0; 0 cos(ε) -sin(ε); 0 sin(ε) cos(ε)]
    a = Spec(res; frame=S, subs=θ)
    @test isempty(a.parameters) && !iz(a.E_L[2, 3]) && !iz(a.E_L[3, 2])
    b = Spec(res; frame=S, subs=θ, zero_rtol=1e-9)
    @test iz(b.E_L[2, 3]) && iz(b.E_L[3, 2]) && b.E_L[2, 2]((;)) == a.E_L[2, 2]((;))
    @test iz(Spec(res; frame=S, subs=θ, zero_rtol=0, zero_atol=1e-10).E_L[2, 3])
    # the normal frame's round-off: zeroed by default, kept with zeroing off
    @test iz(Spec(res; frame=frames.normal, subs=θ).E_L[2, 3])
    @test !iz(Spec(res; frame=frames.normal, subs=θ, zero_rtol=0, zero_atol=0).E_L[2, 3])
    # a bad tolerance names the keyword, before any Python call
    @test_throws "zero_rtol" Spec(PythonCall.pybuiltins.None; zero_rtol=-1e-12)
    @test_throws "zero_atol" Spec(PythonCall.pybuiltins.None; zero_atol=NaN)
end
