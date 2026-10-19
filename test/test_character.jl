@testset "port characters match the operator data exactly" begin
    bases = [ChargeBasis(4), PhaseGrid(-4π, 4π, 41; order=6), PeriodicGrid(12; order=4), FockBasis(9; phi_zpf=0.7)]
    ports = [nport(), n2port(), phiport(), phi2port(), expiphi(1), expiphi(-2), exp2pin(1), exp2pin(-1),
             Port(:d1, 0), Port(:d2, 0), Port(:d1wrap, 1), Port(:d1wrap, -1), Port(:d2wrap, 1), Port(:d2wrap, -1)]
    nchecked = 0
    for b in bases, p in ports
        op = try localop(b, p) catch e; e isa ArgumentError || rethrow(); continue end
        c = character(b, p)
        M = dense(op)
        if c.kind === :real
            @test all(iszero, imag(M))
        elseif c.kind === :imaginary
            @test all(iszero, real(M)) && !all(iszero, M)
        else
            @test c.kind === :paired && c.partner != p && conj(M) == dense(localop(b, c.partner))
        end
        nchecked += 1
    end
    @test nchecked == 30          # served pairs: ChargeBasis 6, PhaseGrid 8 (2π/Δ = 10, so exp2pin(±1) is served), PeriodicGrid 8, FockBasis 8
    @test character(PhaseGrid(-4π, 4π, 41; order=6), expiphi(2)).partner == expiphi(-2)
    @test character(ChargeBasis(3), expiphi(1)).kind === :real && character(ChargeBasis(3), exp2pin(1)).partner == exp2pin(-1)
    @test character(FockBasis(9; phi_zpf=0.7), exp2pin(1)).kind === :real
    @test_throws ArgumentError character(PeriodicGrid(12; order=4), nport())
    g = GivenSystem(:r; energies=[0.0, 1.0, 2.0], operators=Dict(:a => [0 1 0; 0 0 1.4; 0 0 0], :y => im * [0 1 0; -1 0 0; 0 0 0], :z => [1 1im 0; -1im 1 0; 0 0 1]))
    @test character(g, Port(:a, 0)).kind === :real && character(g, Port(:y, 0)).kind === :imaginary && character(g, Port(:z, 0)).kind === :general
end
