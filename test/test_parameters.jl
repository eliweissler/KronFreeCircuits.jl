using KronFreeCircuits: Coefficient, phase_factor, iszero_expr, free_symbols

@testset "Coefficient" begin
    c = Coefficient("E_J*cos(varphi_ext/2) + 1/2", [:E_J, :varphi_ext])
    @test c.deps == [:E_J, :varphi_ext]
    @test c((E_J=2.0, varphi_ext=0.0)) ≈ 2.5
    @test_throws ArgumentError Coefficient("E_J*E_X", [:E_J])
    @test_throws ArgumentError c((E_J=2.0,))
    err = try c((E_J=2.0,)) catch e e end
    @test occursin("varphi_ext", sprint(showerror, err))
    @test (2 * c)((E_J=2.0, varphi_ext=0.0)) ≈ 5.0
    @test (c * c)((E_J=2.0, varphi_ext=0.0)) ≈ 6.25
    @test (c + c)((E_J=2.0, varphi_ext=0.0)) ≈ 5.0
    @test (-c)((E_J=2.0, varphi_ext=0.0)) ≈ -2.5
    @test phase_factor(Coefficient("varphi_ext", [:varphi_ext]), -1)((varphi_ext=π/2,)) ≈ exp(-im*π/2)
    @test iszero_expr(Coefficient("0", Symbol[])) && !iszero_expr(c)
    @test Coefficient(3).deps == Symbol[]
    @test Coefficient(3)(NamedTuple()) === ComplexF64(3)
    @test free_symbols(Meta.parse("E_J*cos(pi*x) + im")) == [:E_J, :x]
end
@testset "Coefficient expressions are restricted" begin
    @test Coefficient("sqrt(E_J/E_C)", [:E_J, :E_C])((E_J=8.0, E_C=2.0)) ≈ 2
    @test Coefficient("θ^2", [:θ])((θ=3.0,)) ≈ 9                   # the generated argument has a reserved name
    @test Coefficient("pi*x", [:x])((x=2.0,)) ≈ 2π
    @test Coefficient("cis(x) * conj(im) + 1//2 - abs(-x)", [:x])((x=0.0,)) ≈ 1/2 - im
    for src in ("E_J + (println(\"x\"); 0)", "E_J +", "", "  ", "E_J )", "x; y", "begin E_J end",
                "@show E_J", "E_J = 1", "x -> x", "E_J.f", "E_J::Float64", "\"E_J\"", "eval(E_J)",
                "E_J .// 1", "E_J[1]", "sqrt.(E_J)", "E_J .< 1", "cos(E_J; k=1)", "e*E_J")
        @test_throws ArgumentError Coefficient(src, [:E_J])
    end
    err = try Coefficient("E_J + (println(\"x\"); 0)", [:E_J]); nothing catch e e end
    @test occursin(":block", err.msg)
    err = try Coefficient("run(E_J)", [:E_J]); nothing catch e e end
    @test occursin(":run", err.msg)
    @test_throws ArgumentError Coefficient("x", [:pi])                # reserved constant
    @test_throws ArgumentError Coefficient("x", [:x, :cos])           # reserved function name
    @test_throws ArgumentError Coefficient("cos", [:x])               # a function name is not a parameter
    bad = replace(read(fixture("transmon.json"), String), "\"E_J\"" => "\"e\"")
    @test_throws ArgumentError KronFreeCircuits.parse_spec(bad)
end

@testset "Coefficient accepts SymPy's dotted operators and grammar items" begin
    pts = [(x=0.7, y=1.3), (x=2.0, y=-0.4), (x=1.1, y=3.0)]
    pairs = [("x .^ 2 .* y / 3 + x / 2 + pi * sqrt(x)", "x ^ 2 * y / 3 + x / 2 + pi * sqrt(x)"),
             ("x .+ y", "x + y"), ("x ./ y", "x / y"), ("x .- y", "x - y"), ("x .* y", "x * y"),
             ("-x .^ 2 .- .-y", "-x ^ 2 - -y"), ("sqrt(x .+ 1) .^ 2", "sqrt(x + 1) ^ 2")]
    for (d, s) in pairs
        cd, cs = Coefficient(d, [:x, :y]), Coefficient(s, [:x, :y])
        @test cd.deps == cs.deps
        @test all(θ -> cd(θ) == cs(θ), pts)
    end
    # the stored expression contains no dotted heads at any depth
    function heads(ex)
        ex isa Expr || return Symbol[]
        return vcat(ex.args[1] isa Symbol ? [ex.args[1]] : Symbol[], reduce(vcat, heads.(ex.args[2:end]); init=Symbol[]))
    end
    @test !any(h -> startswith(string(h), "."), heads(Coefficient("x .^ 2 .* y .+ x ./ 2", [:x, :y]).ex))
    @test KronFreeCircuits.free_symbols(Coefficient("x .+ y", [:x, :y])) == [:x, :y]
    @test_throws ArgumentError Coefficient("x .+ foo(y)", [:x, :y])
    @test_throws ArgumentError Coefficient("foo.(x)", [:x])
    @test_throws ArgumentError Coefficient("x .+ cos.(y)", [:x, :y])
    @test_throws ArgumentError Coefficient("x .< y", [:x, :y])
    # log with a base, scientific literals, Euler's number
    @test Coefficient("log(2, x)", [:x])((x=8.0,)) ≈ 3
    @test Coefficient("1.5e-3", Symbol[])(NamedTuple()) ≈ 1.5e-3
    @test Coefficient("2E9", Symbol[])(NamedTuple()) ≈ 2e9
    @test Coefficient("2e9", Symbol[]).deps == Symbol[]
    @test Coefficient("2e9 * x", [:x])((x=1.0,)) ≈ 2e9
    @test Coefficient("ℯ", Symbol[])(NamedTuple()) ≈ exp(1)
    @test Coefficient("ℯ^x", [:x])((x=2.0,)) ≈ exp(2)
end

@testset "Coefficient folds integer literals beyond Int64" begin
    # Meta.parse turns them into @int128_str / @big_str macrocalls (SI-valued exports, loop fluxes in webers).
    ulps(v, t) = abs(real(v) - Float64(t)) / eps(Float64(t))
    @test ulps(Coefficient("(9223372036854775808 // 3) * x", [:x])((x=1.7,)), (big(9223372036854775808) // 3) * big(1.7)) <= 4
    @test ulps(Coefficient("100000000000000000000 * x", [:x])((x=1.7,)), big(10)^20 * big(1.7)) <= 4
    y = 2.3e-9
    @test ulps(Coefficient("(345107820312500000000 // 213914163877964163) ./ (pi ^ 2 * y)", [:y])((y=y,)),
               (big(345107820312500000000) // 213914163877964163) / (big(pi)^2 * big(y))) <= 4
    big40 = "1" * "0"^39 * "1"   # beyond Int128: a @big_str literal, Rational{BigInt} arithmetic
    @test ulps(Coefficient("($big40 // 3) * x", [:x])((x=1.7,)), (parse(BigInt, big40) // 3) * big(1.7)) <= 4
    @test ulps(Coefficient("(-42724710240000000000000 * pi * x - 1) ./ (44173801 * x)", [:x])((x=0.3,)),
               (-big(42724710240000000000000) * big(pi) * big(0.3) - 1) / (44173801 * big(0.3))) <= 4   # a negative literal
    @test Coefficient("100000000000000000000 * x", [:x]).ex == :($(Int128(10)^20) * x)
    @test imag(Coefficient("100000000000000000000 * x", [:x])((x=1.7,))) == 0
    # any other macrocall, or a string macro that is not a decimal integer, is still rejected
    for src in ("@show x", "@eval 1", "x * @show(x)", "big\"1.5\"", "int128\"x\"", "big\"0x10\"", "r\"1\"",
                "@int128_str(\"1\", \"2\")")
        @test_throws ArgumentError Coefficient(src, [:x])
    end
end

@testset "identical coefficient expressions share one compiled function" begin
    a, b = Coefficient("E_J * cos(x) + 1/2", [:E_J, :x]), Coefficient("E_J * cos(x) + 1/2", [:E_J, :x])
    @test typeof(a.f) == typeof(b.f)
    @test a((E_J=2.0, x=0.3)) == b((E_J=2.0, x=0.3))
    @test typeof(Coefficient("E_J * cos(x)", [:E_J, :x]).f) != typeof(a.f)
    @test Coefficient("θ^2 + E_J", [:θ, :E_J])((θ=3.0, E_J=1.0)) ≈ 10        # a parameter named θ is not shadowed
    reserved = KronFreeCircuits._COEFF_ARG
    @test_throws ArgumentError KronFreeCircuits.check_parameter_names([:x, reserved])
    @test_throws ArgumentError Coefficient("x", [reserved])

    # Rebuilding the coefficients per point (the spec path) costs about as much as the term list built once.
    leaf = Node([:varphi_1 => ChargeBasis(6)]; keep=lowest(2))
    spec, θ = load_spec(fixture("transmon.json")), (E_J=20.0, E_C=1.0, n_g=0.3)
    bases = KronFreeCircuits._tree_bases(leaf)
    terms, c = KronFreeCircuits.circuit_terms(spec, bases), KronFreeCircuits.constant_term(spec, bases)
    modetypes = Dict{Symbol,Symbol}(m.name => m.type for m in spec.modes)
    materialize(leaf, spec, θ); materialize(leaf, terms, θ; constant=c, modetypes=modetypes)     # warm
    t_spec = minimum(@elapsed(materialize(leaf, spec, θ)) for _ in 1:10)
    t_terms = minimum(@elapsed(materialize(leaf, terms, θ; constant=c, modetypes=modetypes)) for _ in 1:10)
    @test t_spec < 10 * t_terms + 5e-3
end
