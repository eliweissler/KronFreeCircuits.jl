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

# Test-only references for ⟨n|D(α)|m⟩ = √(m!/n!) e^{-|α|²/2} α^(n-m) L_m^(n-m)(|α|²), n ≥ m, with
# ⟨m|D(α)|n⟩ = conj(⟨n|D(-α)|m⟩), in BigFloat arithmetic, which has the exponent range for the unscaled factors.
# `_displacement_ref` runs the three-term recurrence of the Laguerre polynomial in its degree; `_displacement_ref_sum`
# is the explicit alternating sum, an independent formula that needs (and is given) many more digits.
function _displacement_ref(n::Int, m::Int, α::Number; precision::Int=256)
    n < m && return conj(_displacement_ref(m, n, -α; precision))
    setprecision(BigFloat, precision) do
        αb = Complex{BigFloat}(α); x = abs2(αb); a = n - m
        p, q = big(0.0), big(1.0)                               # L_{j-1}, L_j of L^(a) at j = 0
        for j in 1:m
            p, q = q, ((2j - 1 + a - x) * q - (j - 1 + a) * p) / j
        end
        sqrt(factorial(big(m)) / factorial(big(n))) * exp(-x / 2) * αb^a * q
    end
end
function _displacement_ref_sum(n::Int, m::Int, α::Number; precision::Int=4096)
    n < m && return conj(_displacement_ref_sum(m, n, -α; precision))
    setprecision(BigFloat, precision) do
        αb = Complex{BigFloat}(α); x = abs2(αb); a = n - m
        L = sum((-1)^k * binomial(big(n), big(m - k)) * x^k / factorial(big(k)) for k in 0:m)
        sqrt(factorial(big(m)) / factorial(big(n))) * exp(-x / 2) * αb^a * L
    end
end

@testset "Bases: displacement where the Laguerre polynomial and the Gaussian leave the Float64 range" begin
    # e^{-|α|²/2} underflows and L_m^(a)(|α|²) overflows for |α|² ≳ 1420 (and the polynomial alone for N ≳ 1030) while
    # their product is an ordinary number: the matrix elements are evaluated on a carried scale and stay finite.
    D = displacement(400, 40.0im)
    @test all(isfinite, D)
    reference = setprecision(256) do                             # the diagonal element of Fock level 399, an independent recurrence
        x = BigFloat(1600)
        p, q = BigFloat(1), 1 - x
        for m in 2:399
            p, q = q, ((2m - 1 - x) * q - (m - 1) * p) / m
        end
        Float64(exp(-x / 2) * q)
    end
    @test reference ≈ -0.03524982424275595 rtol=1e-15
    @test D[end, end] ≈ reference atol=1e-12
    @test abs(D[end, end]) > 0.03                                # an appreciable element, not a zero in place of a NaN
    # every element of the first 200 levels of D(100i) is below the smallest Float64: an exact zero matrix, not NaN
    D100 = displacement(200, 100.0im)
    @test all(isfinite, D100) && maximum(abs, D100) <= 1e-300
    @test abs(_displacement_ref(199, 199, 100.0im)) < big"1e-1000"
    # the polynomial alone overflows at a small argument once N is large
    Dbig = displacement(1100, 0.3im)
    @test all(isfinite, Dbig)
    for (n, m) in ((1099, 1099), (1099, 1080), (1080, 1099), (1000, 990), (500, 1099))   # measured error of the whole matrix: 1.5e-12
        @test Dbig[n+1, m+1] ≈ ComplexF64(_displacement_ref(n, m, 0.3im)) atol=1e-10
    end

    # selected elements against the references, with an absolute tolerance. `tiny` lists elements far below that
    # tolerance but inside the Float64 range, whose scale must also be right (a relative tolerance).
    samples = (
        # |α|²/2 = 800: the Gaussian is below the smallest Float64. The old evaluation returned NaN for n + m ≥ 564.
        (N=400, α=40.0im, entries=((399, 399), (399, 398), (398, 399), (399, 380), (380, 399), (380, 380), (350, 350), (399, 300),
                                   (300, 399), (340, 340), (17, 5), (0, 0)),
         tiny=((399, 0), (0, 399), (200, 133), (133, 399), (300, 300), (200, 200), (100, 100))),
        # both components nonzero, |α|²/2 = 762.5
        (N=400, α=25.0 - 30.0im, entries=((399, 399), (399, 390), (390, 399), (384, 399), (399, 329), (329, 399), (350, 395), (10, 3)),
         tiny=((250, 60), (60, 250), (399, 0), (150, 150))),
        (N=400, α=-31.25 + 23.5im, entries=((399, 399), (395, 380), (380, 395), (399, 371), (360, 360)), tiny=((120, 40), (0, 399))),
        # |α|²/2 between log(floatmax) = 709.8 and -log of the smallest subnormal = 744.4: the gradual-underflow range
        (N=400, α=-37.7, entries=((399, 399), (399, 396), (396, 399), (373, 346), (346, 373), (395, 352), (300, 300)),
         tiny=((399, 0), (200, 200))),
        (N=400, α=38.5im, entries=((399, 399), (398, 399), (399, 360), (360, 390)), tiny=((250, 250), (0, 399))),
        # below the range: elements the old evaluation had right
        (N=400, α=37.0im, entries=((399, 399), (399, 296), (296, 399), (372, 372), (359, 336), (12, 7)), tiny=((399, 0), (150, 150))),
        (N=400, α=26.0im, entries=((399, 399), (399, 41), (41, 399), (226, 210), (334, 64), (100, 100)), tiny=((3, 1), (0, 0))),
        (N=300, α=12.5 - 9.25im, entries=((299, 299), (299, 0), (0, 243), (150, 100), (100, 299), (141, 232), (239, 46)), tiny=((17, 5), (0, 0))),
        (N=200, α=10.0 + 5.0im, entries=((199, 199), (199, 0), (2, 148), (146, 116), (10, 198), (60, 61)), tiny=((1, 0), (0, 0))),
        (N=60, α=0.4 - 0.25im, entries=((59, 59), (58, 53), (53, 58), (55, 54), (0, 0), (1, 0), (0, 1), (30, 12)), tiny=((59, 0), (0, 40))),
        (N=40, α=0.3im, entries=((39, 39), (39, 37), (37, 39), (39, 38), (0, 0), (5, 2)), tiny=((39, 0), (20, 0))),
        (N=6, α=1.0, entries=((0, 0), (5, 5), (4, 2), (2, 4), (5, 4), (5, 0), (0, 5)), tiny=()),
    )
    for s in samples
        Ds = displacement(s.N, s.α)
        @test all(isfinite, Ds)
        for (list, relative) in ((s.entries, false), (s.tiny, true)), (n, m) in list
            ref = _displacement_ref(n, m, s.α)
            @test abs(ref - _displacement_ref(n, m, s.α; precision=512)) <= 1e-40 * abs(ref)   # the reference is converged
            if relative
                @test floatmin(Float64) * 1e20 < abs(ref) < 1e-16
                @test Ds[n+1, m+1] ≈ ComplexF64(ref) rtol=1e-9
            else
                @test Ds[n+1, m+1] ≈ ComplexF64(ref) atol=1e-12
            end
        end
    end
    # the recurrence reference against the explicit sum, at the difficult samples
    for (n, m, α) in ((399, 399, 40.0im), (399, 380, 40.0im), (390, 399, 25.0 - 30.0im), (396, 399, -37.7), (133, 399, 40.0im))
        ref = _displacement_ref_sum(n, m, α)
        @test abs(ref - _displacement_ref(n, m, α)) <= 1e-40 * abs(ref)
    end

    # zero displacement and small dimensions
    @test displacement(5, 0.0) == I(5) && displacement(5, 0.0im) == I(5) && displacement(1, 0) == ones(1, 1)
    @test displacement(1, 2.0 - 1.0im) ≈ fill(exp(-2.5), 1, 1)
    for α in (0.7, -1.3im, 0.6 + 0.8im, 40.0im)
        @test displacement(2, α) ≈ exp(-abs2(α) / 2) * [1 -conj(α); α 1-abs2(α)] atol=1e-300 rtol=1e-13
    end
    Dtiny = displacement(3, 1e-200)                              # |α|² underflows; α does not
    @test Dtiny[2, 1] ≈ 1e-200 && Dtiny[1, 2] ≈ -1e-200 && Dtiny[3, 2] ≈ sqrt(2) * 1e-200 && Dtiny[3, 1] == 0 && Dtiny[3, 3] == 1
    # opposite arguments: D(-α) = D(α)', exactly for a real or an imaginary α
    for α in (40.0im, -40.0, 37.7, -38.5im, 0.3im, 2.0)
        @test displacement(400, -α) == displacement(400, α)'
    end
    for α in (25.0 - 30.0im, 12.5 - 9.25im, 0.4 - 0.25im)
        @test displacement(400, -α) ≈ displacement(400, α)' atol=1e-12
    end
    # phases: a real α gives a real matrix; an imaginary α gives real elements for n - m even and imaginary ones for n - m odd
    Dr = displacement(400, -40.0)
    @test all(iszero ∘ imag, Dr) && real(Dr) == transpose(real(displacement(400, 40.0)))
    @test all(iszero(isodd(i - j) ? real(D[i, j]) : imag(D[i, j])) for i in 1:400, j in 1:400)
    @test D[400, 399] ≈ ComplexF64(_displacement_ref(399, 398, 40.0im)) atol=1e-12
    @test real(Dr[400, 399]) ≈ -imag(D[400, 399]) && real(Dr[399, 400]) ≈ imag(D[399, 400])   # (±1)^a against (±i)^a at a = 1

    # the ports of a Fock basis reach the same evaluation with their conventions: e^{ikφ̂} = D(i k φ_zpf), e^{2πik n̂} = D(-2π k n_zpf)
    F = FockBasis(400; phi_zpf=0.5)                               # n_zpf = 1
    E = dense(localop(F, expiphi(80)))                           # D(40i)
    @test all(isfinite, E) && E == D
    @test E[400, 400] ≈ reference atol=1e-12
    @test dense(localop(F, expiphi(-80))) == E'
    T = dense(localop(F, exp2pin(6)))                            # D(-12π), |α|²/2 = 710.6
    @test all(isfinite, T) && all(iszero ∘ imag, T) && T == displacement(400, -2π * 6)
    for (n, m) in ((399, 399), (399, 396), (396, 399), (360, 381))
        @test T[n+1, m+1] ≈ ComplexF64(_displacement_ref(n, m, -2π * 6)) atol=1e-12
    end
    @test dense(localop(F, exp2pin(-6))) == T'
end

@testset "Bases: displacement of an argument that is not a Float64 or a ComplexF64" begin
    # The argument is converted once, a real one to Float64 and any other to ComplexF64, so that every numeric type gives
    # exactly (`isequal`, elementwise) the matrix of its conversion. Integers, rationals, `Bool` and irrational constants
    # used to throw a `MethodError` (no `frexp`), and single precision used to carry its own round-off into `|α|²`.
    reals = (1, -2, 7, 1//2, -7//3, true, big(3), 1.1f0, -2.5f0, Float16(1.1), big"1.1", big(π), π, ℯ, -0.0f0)
    for α in reals
        Dα = displacement(6, α)
        @test Dα isa Matrix{ComplexF64}
        @test isequal(Dα, displacement(6, Float64(α)))
        @test all(iszero ∘ imag, Dα)                             # a real α gives a real matrix
        @test displacement(6, -α) == Dα'                         # D(-α) = D(α)', exactly
    end
    @test isequal(displacement(6, 0x02), displacement(6, 2.0))   # unsigned (its negation wraps around: no D(-α) here)
    complexes = (2 + 3im, 3im, -3im, 3 + 0im, 1//2 + 3//4 * im, -3//4 * im, 1.5f0 - 0.25f0im, 1.1f0im, -1.1f0 + 0.0f0im,
                 Complex{BigFloat}(big"1.1", big"-0.3"), Complex{Bool}(false, true), Complex{Bool}(true, true))
    for α in complexes
        Dα = displacement(6, α)
        @test Dα isa Matrix{ComplexF64}
        @test isequal(Dα, displacement(6, ComplexF64(α)))
        if iszero(real(α)) || iszero(imag(α))                    # the exact phases of a real or an imaginary α
            @test displacement(6, -α) == Dα'
            @test all(iszero(iszero(real(α)) && isodd(i - j) ? real(Dα[i, j]) : imag(Dα[i, j])) for i in 1:6, j in 1:6)
        else
            @test displacement(6, -α) ≈ Dα' atol=1e-14
        end
    end
    # the values themselves, on two levels: e^{-|α|²/2} [1 -α*; α 1-|α|²]
    for α in (1, -2, 1//2, true, π, 2 - 1im, 3//4 * im, 1.1f0)
        z = ComplexF64(α)
        @test displacement(2, α) ≈ exp(-abs2(z) / 2) * [1 -conj(z); z 1-abs2(z)] rtol=1e-13
    end
    # where the factors leave the Float64 range, with an integer and a rational argument
    @test isequal(displacement(400, 40), displacement(400, 40.0)) && isequal(displacement(400, -40im), displacement(400, -40.0im))
    @test displacement(400, -77//2 * im) == displacement(400, 77//2 * im)'
    # a zero of any type is the identity, and so is a nonzero argument below the Float64 range
    for α in (0, false, 0//1, 0.0f0, big(0), big"0.0", 0 + 0im, 0//1 + 0//1 * im, big"1e-400", -big"1e-400" * im)
        @test displacement(4, α) == I(4)
    end
    # a finite |α| ≥ 2^20 gives the zero matrix, also beyond the Float64 range; a non-finite α gives NaN
    for α in (2^20, -2^20, typemax(Int), typemin(Int), 2^40 * im, 3 * 2^20 // 2, 1.0f20, big"1e400", -big"1e400", big(10)^400,
              big(10)^400 // 3, Complex{BigFloat}(1, big"-1e400"))
        Dα = displacement(3, α)
        @test Dα isa Matrix{ComplexF64} && all(iszero, Dα)
    end
    for α in (1//0, -1//0, Inf32, -Inf32, NaN32, Float16(NaN), big(Inf), -big(Inf), big(NaN), complex(NaN32, 1.0f0),
              complex(0.0f0, -Inf32), Complex{BigFloat}(1, Inf), (1//0) * im)
        Dα = displacement(3, α)
        @test Dα isa Matrix{ComplexF64} && all(isnan, Dα)
    end
    @test displacement(0, 5) isa Matrix{ComplexF64} && size(displacement(0, 1//2)) == (0, 0)
end
