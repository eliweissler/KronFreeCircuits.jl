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
    @test Coefficient("θ^2", [:θ])((θ=3.0,)) ≈ 9                   # the generated argument is gensym'd
    @test Coefficient("pi*x", [:x])((x=2.0,)) ≈ 2π
    @test Coefficient("cis(x) * conj(im) + 1//2 - abs(-x)", [:x])((x=0.0,)) ≈ 1/2 - im
    for src in ("E_J + (println(\"x\"); 0)", "E_J +", "", "  ", "E_J )", "x; y", "begin E_J end",
                "@show E_J", "E_J = 1", "x -> x", "E_J.f", "E_J::Float64", "\"E_J\"", "eval(E_J)",
                "E_J .+ 1", "E_J[1]", "cos(E_J; k=1)", "e*E_J")
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
