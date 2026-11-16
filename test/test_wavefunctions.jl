# Charge and phase wavefunctions of a single mode, in the convention ⟨φ|n⟩ = e^{inφ}/√2π
# (docs/superpowers/specs/2026-11-15-wavefunction-transforms-design.md).

const KFC = KronFreeCircuits

# ψ(φ) = (1/√2π) Σ_n c_n e^{inφ}, and the quadrature of (1/√2π) ∫ ψ(φ) e^{-inφ} dφ with weight `w` per sample.
wf_direct(c, n, φ) = [sum(c[q] * cis(n[q] * x) for q in eachindex(n)) / sqrt(2π) for x in φ]
wf_inverse(ψ, φ, n, w) = [w * sum(ψ[j] * cis(-m * φ[j]) for j in eachindex(φ)) / sqrt(2π) for m in n]
wf_coefficients(N) = [cis(0.7q^2) / q for q in 1:N]      # deterministic, complex, no symmetry

@testset "wavefunctions: grids and charge basis" begin
    @testset "ChargeBasis to phase is the Fourier series on the grid of PeriodicGrid(N)" begin
        b, N = ChargeBasis(3), 7
        c = wf_coefficients(N)
        φ, ψ = phase_wavefunction(b, c)
        @test φ == -π .+ 2π .* (0:6) ./ 7
        @test ψ ≈ wf_direct(c, -3:3, φ) rtol = 1e-13
        @test (2π / 7) * sum(abs2, ψ) ≈ sum(abs2, c)
    end

    @testset "ChargeBasis charge is the coefficients, copied" begin
        c = wf_coefficients(7)
        n, cn = charge_wavefunction(ChargeBasis(3), c)
        @test n == -3:3
        @test cn == c
        @test cn !== c
    end

    @testset "the notes' N-point grid φ_n = 2π(n - n0)/N, every n0" begin
        c = wf_coefficients(7)
        for n0 in 0:6
            δ = π - 2π * n0 / 7                                   # the notes' grid is this package's, translated by δ
            shifted = phase_wavefunction(ChargeBasis(3), c .* cis.((-3:3) .* δ))[2]
            @test shifted ≈ wf_direct(c, -3:3, 2π .* ((0:6) .- n0) ./ 7) rtol = 1e-13
        end
    end

    @testset "PeriodicGrid(N = $N)" for N in (7, 8)
        g = PeriodicGrid(N; order=2)
        v = real.(wf_coefficients(N))
        Δ = 2π / N
        φ, ψ = phase_wavefunction(g, v)
        @test φ == KFC.coordinates(g)
        @test ψ == v ./ sqrt(Δ)
        @test eltype(ψ) == Float64
        ℓ, c = charge_wavefunction(g, v)
        @test ℓ == (N == 7 ? (-3:3) : (-4:3))
        @test c ≈ wf_inverse(ψ, φ, ℓ, Δ) rtol = 1e-13
        @test sum(abs2, c) ≈ sum(abs2, v)
    end

    @testset "charge -> phase -> charge returns the coefficients" begin
        c = wf_coefficients(7)
        _, ψ = phase_wavefunction(ChargeBasis(3), c)
        @test charge_wavefunction(PeriodicGrid(7; order=2), ψ .* sqrt(2π / 7))[2] ≈ c rtol = 1e-13
    end

    @testset "PhaseGrid(N = $N) charge is the quadrature of the Fourier integral" for N in (40, 41)
        g = PhaseGrid(-3.0, 5.0, N; order=2)
        Δ = 8 / (N - 1)
        v = wf_coefficients(N)
        φ, ψ = phase_wavefunction(g, v)
        @test φ == KFC.coordinates(g)
        @test ψ == v ./ sqrt(Δ)
        n, ψn = charge_wavefunction(g, v)
        @test n ≈ 2π .* (N == 40 ? (-20:19) : (-20:20)) ./ (N * Δ) rtol = 1e-14
        @test ψn ≈ wf_inverse(ψ, φ, n, Δ) rtol = 1e-12
        @test (2π / (N * Δ)) * sum(abs2, ψn) ≈ sum(abs2, v)
    end

    @testset "a Gaussian transforms to the conjugate Gaussian, real" begin
        g = PhaseGrid(-12.0, 12.0, 241; order=2)
        φ, Δ = KFC.coordinates(g), KFC.spacing(g)
        s = 0.8
        σ = 1 / (2s)
        v = sqrt(Δ) .* (2π * s^2)^(-1 / 4) .* exp.(-φ .^ 2 ./ (4s^2))
        n, ψn = charge_wavefunction(g, v)
        @test ψn ≈ (2π * σ^2)^(-1 / 4) .* exp.(-n .^ 2 ./ (4σ^2)) atol = 1e-10
        @test maximum(abs.(imag.(ψn))) < 1e-10
        # e^{i 2φ} ψ(φ) carries charge +2: the sign of the kernel.
        n, shifted = charge_wavefunction(g, v .* cis.(2.0 .* φ))
        @test abs(n[argmax(abs.(shifted))] - 2.0) <= n[2] - n[1]
    end

    @testset "one state per column" begin
        c = wf_coefficients(7)
        V = hcat(c, reverse(c))
        for f in (phase_wavefunction, charge_wavefunction), b in (ChargeBasis(3), PeriodicGrid(7; order=2), PhaseGrid(-1.0, 1.0, 7; order=2))
            x, W = f(b, V)
            @test size(W) == (7, 2)
            @test W[:, 1] == f(b, c)[2]
            @test W[:, 2] == f(b, reverse(c))[2]
            @test x == f(b, c)[1]
        end
    end

    @testset "a vector with the wrong number of rows" begin
        for f in (phase_wavefunction, charge_wavefunction), b in (ChargeBasis(3), PeriodicGrid(7; order=2), PhaseGrid(-1.0, 1.0, 7; order=2))
            @test_throws DimensionMismatch f(b, ones(6))
        end
    end

    @testset "an integer unit vector" begin
        φ, ψ = phase_wavefunction(ChargeBasis(1), [0, 1, 0])
        @test ψ isa Vector
        @test ψ ≈ fill(1 / sqrt(2π), 3)
        n, c = charge_wavefunction(PeriodicGrid(3; order=2), [1, 1, 1])
        @test c ≈ [0, sqrt(3), 0] atol = 1e-14
    end
end

@testset "wavefunctions: Fock basis" begin
    b = FockBasis(12; phi_zpf=0.7)
    Id = Matrix(1.0I, 12, 12)

    @testset "phase functions are orthonormal and reproduce the phi port" begin
        x = range(-12, 12; length=2401)
        φ, H = phase_wavefunction(b, Id, x)
        @test φ == collect(x)
        @test eltype(H) == Float64
        @test step(x) * H'H ≈ I atol = 1e-9
        @test step(x) * H' * Diagonal(x) * H ≈ dense(localop(b, phiport())) atol = 1e-9
    end

    @testset "charge functions are orthonormal and reproduce the n port" begin
        q = range(-12, 12; length=2401)
        n, C = charge_wavefunction(b, Id, q)
        @test n == collect(q)
        @test step(q) * C'C ≈ I atol = 1e-9
        @test step(q) * C' * Diagonal(q) * C ≈ dense(localop(b, nport())) atol = 1e-9
    end

    @testset "the charge functions are the Fourier transform of the phase functions" begin
        g = PhaseGrid(-14.0, 14.0, 561; order=2)
        _, Hg = phase_wavefunction(b, Id, KFC.coordinates(g))
        n, F = charge_wavefunction(g, Hg .* sqrt(KFC.spacing(g)))
        @test F ≈ charge_wavefunction(b, Id, n)[2] atol = 1e-9
    end

    @testset "a level whose Gaussian factor underflows where the function does not" begin
        B = FockBasis(1200; phi_zpf=1.0)                 # level 1199 turns at |x| ≈ 69.3; e^{-x²/4} underflows at |x| ≈ 54.6
        e = zeros(1200); e[1200] = 1
        x = range(-80, 80; length=8001)
        _, ψ = phase_wavefunction(B, e, x)
        @test all(isfinite, ψ)
        @test step(x) * sum(abs2, ψ) ≈ 1 rtol = 1e-8
        @test ψ[1] == 0 || abs(ψ[1]) < 1e-50             # far outside the turning point
    end

    @testset "a Fock basis has no grid: the points are required" begin
        @test_throws ArgumentError phase_wavefunction(b, Id)
        @test_throws ArgumentError charge_wavefunction(b, Id)
    end

    @testset "a vector with the wrong number of rows" begin
        @test_throws DimensionMismatch phase_wavefunction(b, ones(11), [0.0])
        @test_throws DimensionMismatch charge_wavefunction(b, ones(11), [0.0])
    end

    @testset "no points, and a single state" begin
        @test size(phase_wavefunction(b, Id, Float64[])[2]) == (0, 12)
        @test size(charge_wavefunction(b, Id, Float64[])[2]) == (0, 12)
        @test phase_wavefunction(b, Id[:, 3], [0.3, 0.4])[2] == phase_wavefunction(b, Id, [0.3, 0.4])[2][:, 3]
    end
end
