using KronFreeCircuits: Port, Factor, Term, modes, circuit_terms, constant_term, load_spec,
                        nport, n2port, phiport, phi2port, expiphi, exp2pin

@testset "circuit_terms" begin
    θ = (E_J=8.0, E_C=1.0, E_L=0.5, varphi_ext=π)
    ts = circuit_terms(load_spec(fixture("fluxonium.json")))
    kinds = Dict((t.factors[1].port.kind, t.factors[1].port.k) => t.coeff(θ) for t in ts)
    @test kinds[(:n2, 0.0)] ≈ 4.0 && kinds[(:phi2, 0.0)] ≈ 0.25 && kinds[(:phi, 0.0)] ≈ 0.5π
    @test kinds[(:expiphi, 1.0)] ≈ -4.0 && kinds[(:expiphi, -1.0)] ≈ -4.0
    @test length(ts) == 5
    zp = circuit_terms(load_spec(fixture("two_zero_pi.json")))
    cross = only(filter(t -> Set(modes(t)) == Set([:theta1, :theta2]), zp))
    @test cross.coeff((g=0.1,)) ≈ 0.1
    @test constant_term(load_spec(fixture("transmon.json")))((E_C=0.3, n_g=0.25)) ≈ 4*0.3*0.25^2
end

@testset "Port" begin
    @test expiphi(1) == expiphi(1.0) && hash(expiphi(1)) == hash(expiphi(1.0))
    @test expiphi(1) != exp2pin(1) && expiphi(1) != expiphi(-1)
    @test Dict(nport() => 1)[Port(:n, 0)] == 1
end
