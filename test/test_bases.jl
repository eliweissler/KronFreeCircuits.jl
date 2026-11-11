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

@testset "PeriodicGrid: cyclic stencils of -i∂ and -∂²; plane-wave symbols; sectors" begin
    using KronFreeCircuits: check_sectors
    for order in (2, 4, 6, 8), N in (16, 24)
        b = PeriodicGrid(N; order=order)
        Δ = 2π / N; h = order ÷ 2
        w1 = KronFreeCircuits.fd_weights(order, 1); w2 = KronFreeCircuits.fd_weights(order, 2)
        # the cyclic matrices built directly with mod1
        D1 = zeros(ComplexF64, N, N); D2 = zeros(N, N)
        for i in 1:N, j in -h:h
            D1[i, mod1(i + j, N)] += -im * w1[j+h+1] / Δ
            D2[i, mod1(i + j, N)] += -w2[j+h+1] / Δ^2
        end
        n = dense(localop(b, nport())); n2 = dense(localop(b, n2port()))
        @test n ≈ D1 && n2 ≈ D2
        @test n ≈ n' && maximum(abs, real(n)) == 0          # Hermitian, purely imaginary
        @test n2 ≈ n2' && eltype(n2) <: Real                 # real symmetric
        # plane waves diagonalize both, with the symbols σ₁(k) = Σ_j w¹_j sin(jkΔ)/Δ and σ₂(k) = -Σ_j w²_j cos(jkΔ)/Δ²
        φ = coordinates(b)
        for k in (0, 1, -2, 3)
            ψ = cis.(k .* φ)
            σ1 = sum(w1[j+h+1] * sin(j * k * Δ) for j in -h:h) / Δ
            σ2 = -sum(w2[j+h+1] * cos(j * k * Δ) for j in -h:h) / Δ^2
            @test n * ψ ≈ σ1 .* ψ atol=1e-12
            @test n2 * ψ ≈ σ2 .* ψ atol=1e-12
        end
        @test dense(localop(b, expiphi(2))) ≈ Diagonal(cis.(2 .* coordinates(b)))
        @test sparse(localop(b, nport())) ≈ n && sparse(localop(b, n2port())) ≈ n2
        for p in (Port(:d1, 0), Port(:d2, 0), Port(:d1wrap, 1), Port(:d2wrap, -1), Port(:n, 1), Port(:n2, 1), Port(:exp2pin, 1))
            @test_throws ArgumentError localop(b, p)
        end
    end
    @test KronFreeCircuits._check_bases(Node([:phi => PeriodicGrid(16)]), Dict(:phi => :c)) === nothing
    @test_throws ArgumentError KronFreeCircuits._check_bases(Node([:phi => PeriodicGrid(16)]), Dict(:phi => :e))
    @test_throws ArgumentError localop(PeriodicGrid(16), phiport())
    @test_throws ArgumentError localop(PeriodicGrid(16), phi2port())
    @test_throws ArgumentError localop(PeriodicGrid(16), expiphi(0.5))
    @test_throws ArgumentError check_sectors(PeriodicGrid(15), [2], :phi)
    @test check_sectors(PeriodicGrid(16; sectors=4), [2, 4], :phi) === nothing
    @test_throws ArgumentError PeriodicGrid(15; sectors=2)
    @test KronFreeCircuits.modetype(PeriodicGrid(16)) === :c && KronFreeCircuits.spacing(PeriodicGrid(16)) ≈ 2π / 16
end

@testset "suggest_bases" begin
    s = numeric_spec(; modes=[:a => :c, :b => :e], E_C=[0.072 0.072; 0.072 2.484170953858], E_L=[0.0 0.0; 0.0 2.54],
                     flux_linear=[0.0, 0.0], n_g=[0.0, 0.0], cosines=[(17.3349, [1, -1.0], 0.0), (16.3251, [1, -1.0], 0.0)])
    b = Dict(suggest_bases(s; charge_cutoff=16, max_points=120, extent_scale=1.5, when_capped=:allow))
    @test b[:a] == ChargeBasis(16)
    zpf = (2 * 2.484170953858 / 2.54)^0.25; ω = sqrt(8 * 2.484170953858 * 2.54)
    ext = 1.5 * max(6zpf, sqrt(2 * (2 * (17.3349 + 16.3251) + 12ω + 4 * 2.484170953858) / 2.54))
    @test b[:b] == PhaseGrid(-ext, ext, 120)                       # capped: the formula asks for more than 120 points
    @test Dict(suggest_bases(s; max_points=10_000))[:b].N == ceil(Int, 2 * (ext / 1.5) / min(zpf / 6, π / 6)) + 1
    # a binding cap (the KITE-like settings above): one warning naming the mode, the points the formula asks for and the
    # cap by default, an error or silence on request, and the same bases in every case; no warning when it does not bind
    kw = (charge_cutoff=16, max_points=120, extent_scale=1.5)
    asks = ceil(Int, 2 * ext / min(zpf / 6, π / 6)) + 1
    capped = @test_logs (:warn, Regex("mode :b \\(the formula asks for $asks points\\) at max_points = 120")) suggest_bases(s; kw...)
    @test capped == suggest_bases(s; kw..., when_capped=:allow) == (@test_logs suggest_bases(s; kw..., when_capped=:allow))
    @test Dict(capped)[:b].N == 120 < asks
    @test_throws Regex("mode :b .*$asks points.*max_points = 120") suggest_bases(s; kw..., when_capped=:error)
    @test_logs suggest_bases(s; kw..., max_points=10_000)
    @test_logs suggest_bases(s; kw..., max_points=asks, when_capped=:error)      # a cap equal to the formula's count does not bind
    @test_throws "when_capped" suggest_bases(s; kw..., when_capped=:ignore)
    @test_throws "when_capped" suggest_bases(s; max_points=10_000, when_capped=:ignore)   # checked even when nothing is capped
    # the modes in the order of the spec, and the keywords passed on
    @test first.(suggest_bases(s)) == [:a, :b]
    @test Dict(suggest_bases(s; order=4))[:b].order == 4
    # a mode no cosine touches takes the step zpf / 6
    h = numeric_spec(; modes=[:x => :h], E_C=fill(0.5, 1, 1), E_L=fill(2.0, 1, 1), flux_linear=[0.0], n_g=[0.0])
    zh = (2 * 0.5 / 2.0)^0.25; eh = max(6zh, sqrt(2 * (12 * sqrt(8 * 0.5 * 2.0) + 4 * 0.5) / 2.0))
    @test only(suggest_bases(h; max_points=10_000)) == (:x => PhaseGrid(-eh, eh, ceil(Int, 2eh / (zh / 6)) + 1))
    # refusals
    @test_throws ArgumentError suggest_bases(numeric_spec(; modes=[:x => :e], E_C=fill(0.5, 1, 1), E_L=fill(0.0, 1, 1),
                                                          flux_linear=[0.0], n_g=[0.0]))
    @test_throws ArgumentError suggest_bases(s; charge_cutoff=0)
    @test_throws ArgumentError suggest_bases(s; max_points=1)
    @test_throws ArgumentError suggest_bases(s; extent_scale=0)
end
