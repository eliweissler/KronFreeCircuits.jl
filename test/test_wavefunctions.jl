# Charge and phase wavefunctions of a single mode, in the convention ⟨φ|n⟩ = e^{inφ}/√2π.

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

@testset "wavefunctions: solved nodes" begin
    p = dc_fluxonium()
    sys = Base.CoreLogging.with_logger(() -> materialize(p.root, p.spec, p.θ), Base.CoreLogging.NullLogger())
    g = PhaseGrid(-10π, 10π, 601; order=8)

    @testset "a node with one basis child uses that basis and the node's eigenvectors" begin
        @test phase_wavefunction(sys) == phase_wavefunction(g, eigenvectors(sys))
        @test phase_wavefunction(sys, p.root) == phase_wavefunction(g, eigenvectors(sys))
        @test charge_wavefunction(sys) == charge_wavefunction(g, eigenvectors(sys))
        @test charge_wavefunction(sys, p.root) == charge_wavefunction(g, eigenvectors(sys))
        n, ψn = charge_wavefunction(sys)
        @test size(ψn) == (601, 5)
        @test all(j -> (n[2] - n[1]) * sum(abs2, view(ψn, :, j)) ≈ 1, 1:5)
    end

    @testset "fluxonium on a Fock basis agrees with the grid in both representations" begin
        fock = Node([:varphi_1 => FockBasis(200; phi_zpf=(2 * 2.5 / 0.5)^0.25)]; solver=Dense(), keep=lowest(5))
        sysF = materialize(fock, p.spec, p.θ)
        # Measured 2026-11-15: energies agree to 7e-10 and the wavefunctions to 4e-10 in both representations.
        @test energies(sysF) ≈ energies(sys) atol = 1e-8
        φ, ψ = phase_wavefunction(sys)
        n, ψn = charge_wavefunction(sys)
        φF, ψF = phase_wavefunction(sysF, φ)
        nF, ψnF = charge_wavefunction(sysF, fock, n)
        @test φF == φ && nF == n
        @test phase_wavefunction(sysF, fock, φ) == (φF, ψF)
        for j in 1:5
            o = (φ[2] - φ[1]) * dot(ψF[:, j], ψ[:, j])           # ψ ≈ ψF · o
            @test abs(o) ≈ 1 atol = 1e-8
            @test ψ[:, j] ≈ ψF[:, j] .* o atol = 1e-8
            @test ψn[:, j] ≈ ψnF[:, j] .* o atol = 1e-8         # the same phase aligns both representations
        end
        @test_throws ArgumentError phase_wavefunction(sysF)      # a Fock basis has no grid
    end

    @testset "a leaf of a larger tree" begin
        q = dc_two_transmons()
        s2 = materialize(q.root, q.spec, q.θ)
        φ, ψ = phase_wavefunction(s2, q.leaves[1])
        @test (φ, ψ) == phase_wavefunction(ChargeBasis(30), eigenvectors(s2, q.leaves[1]))
        @test size(ψ) == (61, 4)
    end

    @testset "multi-mode nodes are not implemented yet" begin
        q = dc_two_transmons()
        s2 = materialize(q.root, q.spec, q.θ)
        for f in (phase_wavefunction, charge_wavefunction)
            err = try f(s2); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("multi-mode", err.msg) && occursin("2 children", err.msg)
        end
        leaf = Node([:varphi_1 => PhaseGrid(-10π, 10π, 601; order=8)]; solver=Dense(), compute=8, keep=lowest(5))
        parent = Node([leaf]; solver=Dense(), keep=lowest(3))
        s1 = Base.CoreLogging.with_logger(() -> materialize(parent, p.spec, p.θ), Base.CoreLogging.NullLogger())
        for f in (phase_wavefunction, charge_wavefunction)
            err = try f(s1); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("multi-mode", err.msg) && occursin("not a basis", err.msg)
        end
        @test size(phase_wavefunction(s1, leaf)[2]) == (601, 5)
    end
end

@testset "wavefunctions: inputs beyond dense eigenvectors" begin
    bases = (ChargeBasis(3), PeriodicGrid(7; order=2), PhaseGrid(-1.0, 1.0, 7; order=2))

    @testset "the identity, structured or sparse, gives the transform matrix" begin
        dense_id = Matrix(1.0I, 7, 7)
        for f in (phase_wavefunction, charge_wavefunction), b in bases, V in (I(7), sparse(1.0I, 7, 7))
            @test f(b, V)[2] == f(b, dense_id)[2]
        end
        F = FockBasis(7; phi_zpf=0.7)
        @test phase_wavefunction(F, I(7), [0.1, 0.2])[2] == phase_wavefunction(F, dense_id, [0.1, 0.2])[2]
        @test charge_wavefunction(F, I(7), [0.1, 0.2])[2] == charge_wavefunction(F, dense_id, [0.1, 0.2])[2]
    end

    @testset "results are dense arrays in floating point, whatever holds V" begin
        for f in (phase_wavefunction, charge_wavefunction), b in bases
            @test f(b, 1:7)[2] isa Vector{<:Union{Float64,ComplexF64}}
            @test f(b, I(7))[2] isa Matrix{<:Union{Float64,ComplexF64}}
            @test f(b, view(ones(7, 2), :, 1))[2] isa Vector
        end
        @test charge_wavefunction(ChargeBasis(1), [0, 1, 0])[2] isa Vector{Float64}      # the stored representation, floated
        @test charge_wavefunction(ChargeBasis(1), [0, 1im, 0])[2] isa Vector{ComplexF64}
        @test phase_wavefunction(PeriodicGrid(3; order=2), [0, 1, 0])[2] isa Vector{Float64}
    end

    @testset "points passed to a basis that has its own grid" begin
        for f in (phase_wavefunction, charge_wavefunction), b in bases
            err = try f(b, ones(7), [0.0, 0.1]); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("its own grid", err.msg) && occursin("FockBasis", err.msg)
        end
        p = dc_fluxonium()
        sys = Base.CoreLogging.with_logger(() -> materialize(p.root, p.spec, p.θ), Base.CoreLogging.NullLogger())
        @test_throws ArgumentError phase_wavefunction(sys, [0.0, 0.1])
        @test_throws ArgumentError charge_wavefunction(sys, p.root, [0.0, 0.1])
    end

    @testset "Fock values below the Float64 range, and points at ±Inf and NaN" begin
        B = FockBasis(1200; phi_zpf=1.0)
        e = zeros(1200); e[1200] = 1
        _, ψ = phase_wavefunction(B, e, [100.0, 110.0, 400.0, Inf, -Inf, NaN, 69.0])
        @test 0 < abs(ψ[1]) < 1e-300                       # subnormal: 5.8e-311
        @test ψ[2] == 0 && ψ[3] == 0                       # below the range
        @test ψ[4] == 0 && ψ[5] == 0
        @test isnan(ψ[6])
        @test 1e-3 < abs(ψ[7]) < 1                         # near the turning point, unaffected by the other rows
    end
end
