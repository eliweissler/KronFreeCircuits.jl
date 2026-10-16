using KronFreeCircuits: Basis, ChargeBasis, PhaseGrid, FockBasis, localop, fd_weights, coordinates,
                        positive_mask, edge_support, displacement, modetype, dimension, dense,
                        nport, n2port, phiport, phi2port, expiphi, exp2pin

@testset "Bases" begin
    @test fd_weights(8, 2) ≈ [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
    g = PhaseGrid(-10, 10, 801; order=8); φ = coordinates(g); Δ = φ[2]-φ[1]
    ψ = exp.(-φ.^2 ./ 2)
    @test dense(localop(g, n2port())) * ψ ≈ -(φ.^2 .- 1) .* ψ atol=1e-7      # -∂²ψ, interior accuracy
    @test dense(localop(g, nport())) * ψ ≈ im .* φ .* ψ atol=1e-7           # -i∂ψ
    @test dense(localop(g, nport()))' ≈ dense(localop(g, nport()))          # Hermitian
    @test_throws ArgumentError localop(PhaseGrid(-1, 1, 10), exp2pin(1))     # Review Focus 3
    c = ChargeBasis(3)
    @test dense(localop(c, expiphi(1))) ≈ diagm(-1 => ones(6))               # |n+1><n|
    @test_throws ArgumentError localop(c, expiphi(0.5))
    @test_throws ArgumentError localop(c, phiport())
    f = FockBasis(40; phi_zpf=0.7); a = diagm(1 => sqrt.(1:39)); nz = 1/(2*0.7)
    @test dense(localop(f, phiport())) ≈ 0.7 * (a + a')
    @test dense(localop(f, nport())) ≈ im * nz * (a' - a)                       # charge, not number
    @test dense(localop(f, n2port()))[1:38, 1:38] ≈ ((im * nz * (a' - a))^2)[1:38, 1:38]
    @test dense(localop(f, phi2port()))[1:38, 1:38] ≈ ((0.7 * (a + a'))^2)[1:38, 1:38]
    @test dense(localop(f, n2port()))[40, 40] ≈ nz^2 * (2*39 + 1)              # closed form at the cutoff
    D = displacement(40, 0.3im); @test (D * D')[1:20, 1:20] ≈ I(20) atol=1e-10
    @test D[1:10, 1:10] ≈ exp(0.3im*(a + a'))[1:10, 1:10] atol=1e-8
    @test positive_mask(g) == (φ .>= 0) && all(positive_mask(f))
    @test edge_support(g, reshape(ψ, :, 1), 4)[1] < 1e-15
end

@testset "Bases: grid exp2pin direction" begin
    # e^{2πik n̂} with n̂ = -i∂_φ maps ψ(φ) to ψ(φ + 2πk); a plane wave e^{iqφ} has n = q.
    g = PhaseGrid(-4π, 4π, 81) ; φ = coordinates(g)            # Δ = 2π/20
    @test φ[2] - φ[1] ≈ 2π/20
    q = 0.3; ψ = exp.(im * q .* φ); s = 20
    for k in (1, -1)
        Uψ = dense(localop(g, exp2pin(k))) * ψ
        int = (s+1):(length(φ)-s)
        @test Uψ[int] ≈ exp(2π * im * k * q) .* ψ[int]
    end
end

@testset "Bases: metadata and remaining ports" begin
    c = ChargeBasis(3); g = PhaseGrid(-1.0, 1.0, 11; order=4); f = FockBasis(6; phi_zpf=0.5)
    @test (dimension(c), dimension(g), dimension(f)) == (7, 11, 6)
    @test (modetype(c), modetype(g), modetype(f)) == (:c, :e, :h)
    @test coordinates(c) == collect(-3.0:3.0) && coordinates(f) == collect(0.0:5.0)
    @test positive_mask(c) == [false, false, false, true, true, true, true]
    @test fd_weights(8, 1) ≈ [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
    for o in (2, 4, 6, 8)
        x = (-o÷2):(o÷2)
        w1 = fd_weights(o, 1); w2 = fd_weights(o, 2)
        @test length(w1) == o + 1 && length(w2) == o + 1
        @test sum(w1 .* x) ≈ 1 && abs(sum(w2)) < 1e-12 && w1 ≈ -reverse(w1) && w2 ≈ reverse(w2)
        @test sum(w2 .* x.^2) ≈ 2
    end
    @test_throws ArgumentError PhaseGrid(-1, 1, 10; order=3)
    @test_throws ArgumentError localop(c, phi2port())
    @test dense(localop(c, nport())) ≈ diagm(-3.0:3.0)
    @test dense(localop(c, n2port())) ≈ diagm((-3.0:3.0) .^ 2)
    @test dense(localop(c, expiphi(-2))) ≈ diagm(2 => ones(5))
    @test dense(localop(c, exp2pin(0.25))) ≈ diagm(exp.(2π * im * 0.25 .* (-3:3)))
    φ = coordinates(g)
    @test dense(localop(g, phiport())) ≈ diagm(φ)
    @test dense(localop(g, phi2port())) ≈ diagm(φ .^ 2)
    @test dense(localop(g, expiphi(1.5))) ≈ diagm(exp.(1.5im .* φ))
    # Fock exponentials: displacement with the stated arguments, low block vs dense exp
    F = FockBasis(60; phi_zpf=0.4); a = diagm(1 => sqrt.(1:59)); nz = 1/(2*0.4)
    @test dense(localop(F, expiphi(1)))[1:15, 1:15] ≈ exp(im * 0.4 * (a + a'))[1:15, 1:15] atol=1e-8
    @test dense(localop(F, exp2pin(0.1)))[1:15, 1:15] ≈ exp(2π * im * 0.1 * im * nz * (a' - a))[1:15, 1:15] atol=1e-8
    # displacement: unitary low block and D(α)D(-α) = 1 for complex α
    α = 0.4 - 0.25im; Dp = displacement(60, α); Dm = displacement(60, -α)
    @test (Dp * Dm)[1:20, 1:20] ≈ I(20) atol=1e-10
    @test Dp[1:15, 1:15] ≈ exp(α * a' - conj(α) * a)[1:15, 1:15] atol=1e-8
    # edge_support: per-column norm over the first and last w points
    V = zeros(11, 2); V[1, 1] = 3; V[11, 1] = 4; V[6, 2] = 1
    @test edge_support(g, V, 2) ≈ [5.0, 0.0]
end
