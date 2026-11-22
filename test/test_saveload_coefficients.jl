using KronFreeCircuits: _coefficient_string, _saved_coefficient, _identical, _make, phase_factor
rt(c) = _saved_coefficient(_coefficient_string(c), copy(c.deps))
@testset "coefficient text round trip" begin
    E_J = Coefficient("E_J", [:E_J]); phi = Coefficient("varphi_ext", [:varphi_ext])
    cases = [Coefficient(2), Coefficient(1.5), Coefficient(-0.25), Coefficient(1.0 + 0.0im), Coefficient(0.3 - 2.0im),
             Coefficient(1 // 3), Coefficient(Int128(2)^70), Coefficient(big(2)^130),
             -E_J, 4 * E_J, (0.5 + 1.0im) * E_J, (1 // 2) * E_J, E_J * phase_factor(Coefficient(0.7), 1),
             phase_factor(phi, -1), Coefficient("sqrt(E_J) * cos(pi * varphi_ext)", [:E_J, :varphi_ext])]
    for c in cases
        @test _identical(rt(c), c)
        @test rt(c)((E_J=3.0, varphi_ext=0.4)) == c((E_J=3.0, varphi_ext=0.4))
    end
    @test !_identical(Coefficient(1), Coefficient(1.0))                     # typeof matters
    @test !_identical(Coefficient(1.0 + 0.0im), Coefficient(1.0))
    @test_throws ArgumentError _saved_coefficient("run(`ls`)", Symbol[])     # allowlist still applies
    @test_throws ArgumentError _saved_coefficient("Complex(E_J, 1.0)", [:E_J])  # only literal arguments
    @test_throws ArgumentError _saved_coefficient("E_J", Symbol[])          # undeclared symbol
end
