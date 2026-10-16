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
