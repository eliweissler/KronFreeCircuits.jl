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

    @testset "branch nodes are not supported" begin
        q = dc_two_transmons()
        s2 = materialize(q.root, q.spec, q.θ)
        for f in (phase_wavefunction, charge_wavefunction)
            err = try f(s2); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("leaf", err.msg)
        end
        leaf = Node([:varphi_1 => PhaseGrid(-10π, 10π, 601; order=8)]; solver=Dense(), compute=8, keep=lowest(5))
        parent = Node([leaf]; solver=Dense(), keep=lowest(3))
        s1 = Base.CoreLogging.with_logger(() -> materialize(parent, p.spec, p.θ), Base.CoreLogging.NullLogger())
        for f in (phase_wavefunction, charge_wavefunction)
            err = try f(s1); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("leaf", err.msg)
        end
        @test size(phase_wavefunction(s1, leaf)[2]) == (601, 5)
        zspec = load_spec(fixture("zero_pi.json"))
        mixed = Node([:theta => ChargeBasis(6), Node([:phi => PhaseGrid(-10π, 10π, 151; order=6)]; solver=Dense(), keep=lowest(4))];
                     solver=Dense(), keep=lowest(3))
        sm = Base.CoreLogging.with_logger(() -> materialize(mixed, zspec, zero_pi_params()), Base.CoreLogging.NullLogger())
        for f in (phase_wavefunction, charge_wavefunction)
            err = try f(sm); nothing catch e; e end
            @test err isa ArgumentError && occursin("leaf", err.msg)
        end
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

@testset "wavefunctions: multi-mode leaves" begin
    oned(b, rep, V, pts) = pts === nothing ? (rep === :phase ? phase_wavefunction(b, V) : charge_wavefunction(b, V)) :
                                             (rep === :phase ? phase_wavefunction(b, V, pts) : charge_wavefunction(b, V, pts))
    function product_check(bases, names, reps, points)
        vs = [wf_coefficients(KFC.dimension(b)) .* (k == 2 ? 1 : cis(0.3k)) for (k, b) in enumerate(bases)]
        v = reduce((a, b) -> kron(b, a), vs)                                   # axis 1 fastest
        axes, ψ = KFC._wavefunction(Tuple(bases), Tuple(names), Tuple(reps), reshape(v, :, 1), points)
        ones_ = [oned(b, r, vk, get(points, nm, nothing)) for (b, nm, r, vk) in zip(bases, names, reps, vs)]
        @test keys(axes) == Tuple(names)
        @test all(axes[k] == ones_[k][1] for k in eachindex(names))
        expect = reduce((a, b) -> reshape(a, size(a)..., 1) .* reshape(b, ntuple(_ -> 1, ndims(a))..., :), [o[2] for o in ones_])
        @test size(ψ) == (size(expect)..., 1)
        @test ψ[ntuple(_ -> Colon(), length(bases))..., 1] ≈ expect
    end

    @testset "product state, $reps" for reps in ((:phase, :phase), (:phase, :charge), (:charge, :phase), (:charge, :charge))
        product_check([ChargeBasis(3), PhaseGrid(-4.0, 4.0, 11)], [:a, :b], reps, (;))
        product_check([PeriodicGrid(10), FockBasis(6; phi_zpf=0.7)], [:a, :b], reps, (b = range(-3, 3; length=9),))
    end

    @testset "three modes, mixed" begin
        product_check([ChargeBasis(2), PhaseGrid(-3.0, 3.0, 9), PeriodicGrid(11)], [:a, :b, :c], [:charge, :phase, :charge], (;))
    end

    @testset "a real result stays real, a mixed one is complex" begin
        b = (PhaseGrid(-3.0, 3.0, 9), PeriodicGrid(10))
        V = reshape(collect(1.0:90.0), :, 1)
        @test eltype(KFC._wavefunction(b, (:a, :b), (:phase, :phase), V, (;))[2]) == Float64
        @test eltype(KFC._wavefunction((ChargeBasis(2), b[1]), (:a, :b), (:phase, :phase), reshape(collect(1.0:45.0), :, 1), (;))[2]) == ComplexF64
    end

    spec = load_spec(fixture("zero_pi.json")); θz = zero_pi_params()
    gφ = PhaseGrid(-10π, 10π, 151; order=6)
    leaf = Node([:theta => ChargeBasis(6), :phi => gφ]; solver=Dense(), keep=lowest(4))
    sys = @test_logs min_level=Base.CoreLogging.Warn materialize(leaf, spec, θz)       # no grid-edge warning
    Δφ = KFC.spacing(gφ); Δn = 2π / (151Δφ)

    @testset "zero-pi leaf: shapes and normalization" begin
        axes, ψ = phase_wavefunction(sys)
        @test keys(axes) == (:theta, :phi) && size(ψ) == (13, 151, 4)
        @test axes.phi == KFC.coordinates(gφ)
        @test all(s -> (2π / 13) * Δφ * sum(abs2, view(ψ, :, :, s)) ≈ 1, 1:4)
        axes, ψ = charge_wavefunction(sys)
        @test axes.theta == collect(-6.0:6.0)
        @test all(s -> Δn * sum(abs2, view(ψ, :, :, s)) ≈ 1, 1:4)
        axes, ψ = phase_wavefunction(sys; charge=(:theta,))
        @test axes.theta == collect(-6.0:6.0) && axes.phi == KFC.coordinates(gφ)
        @test all(s -> Δφ * sum(abs2, view(ψ, :, :, s)) ≈ 1, 1:4)
        @test (axes, ψ) == phase_wavefunction(sys, leaf; charge=:theta)
        @test charge_wavefunction(sys; phase=(:phi,)) == (axes, ψ)
    end

    @testset "states" begin
        _, ψ = phase_wavefunction(sys)
        _, ψ2 = phase_wavefunction(sys; states=2)
        @test size(ψ2) == (13, 151, 1) && ψ2 ≈ ψ[:, :, 2:2]
        @test phase_wavefunction(sys; states=[1, 3])[2] ≈ ψ[:, :, [1, 3]]
    end

    @testset "the keywords on a single-mode node" begin
        p = dc_fluxonium()
        s1 = Base.CoreLogging.with_logger(() -> materialize(p.root, p.spec, p.θ), Base.CoreLogging.NullLogger())
        @test phase_wavefunction(s1; charge=(:varphi_1,)) == charge_wavefunction(s1)
        @test charge_wavefunction(s1; phase=:varphi_1) == phase_wavefunction(s1)
        φ, ψ = phase_wavefunction(s1)
        @test phase_wavefunction(s1; states=2:3) == (φ, ψ[:, 2:3])
    end

    @testset "errors name the mode" begin
        for (f, kw) in ((phase_wavefunction, (charge=(:nope,),)), (charge_wavefunction, (phase=(:nope,),)),
                        (phase_wavefunction, (points=(nope=[0.0],),)))
            err = try f(sys; kw...); nothing catch e; e end
            @test err isa ArgumentError && occursin("nope", err.msg)
        end
        err = try phase_wavefunction(sys; points=(phi=[0.0],)); nothing catch e; e end
        @test err isa ArgumentError && occursin("phi", err.msg)
        fock = Node([:theta => ChargeBasis(6), :phi => FockBasis(40; phi_zpf=2.0)]; solver=Dense(), keep=lowest(2))
        sf = Base.CoreLogging.with_logger(() -> materialize(fock, spec, θz), Base.CoreLogging.NullLogger())
        err = try phase_wavefunction(sf); nothing catch e; e end
        @test err isa ArgumentError && occursin("phi", err.msg)
        err = try phase_wavefunction(sf, [0.0]); nothing catch e; e end                    # positional points: name the Fock mode
        @test err isa ArgumentError && occursin("(phi = ", err.msg) && !occursin("(theta = ", err.msg)
        err = try phase_wavefunction(sys, [0.0]); nothing catch e; e end                   # no Fock mode at all
        @test err isa ArgumentError && occursin("no FockBasis", err.msg)
        axes, ψ = phase_wavefunction(sf; points=(phi=range(-5, 5; length=21),))
        @test size(ψ) == (13, 21, 2) && axes.phi == collect(range(-5, 5; length=21))
    end
end
