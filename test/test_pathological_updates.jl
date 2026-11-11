# Small adversarial cases for the recent window/composite/periodic changes.
# Keep this file self-contained so it can also run through the named-file runner.
using KronFreeCircuits: Term, Factor, Coefficient, dense, localop, operator, within,
    materialized, nport, n2port, expiphi, _initial_block

@testset "window sweep: shrinking and growing children with composite ports" begin
    b = ChargeBasis(2)
    bases = [:z => b, :a => b, :c => b]
    terms = [Term(Coefficient("scale", [:scale]), [Factor(:z, n2port())]),
             Term(Coefficient(1.7), [Factor(:a, n2port())]),
             Term(Coefficient(2.3), [Factor(:c, n2port())])]
    for s in (-1, 1)
        push!(terms, Term(Coefficient(-0.2), [Factor(:z, expiphi(s)), Factor(:a, expiphi(-s)), Factor(:c, expiphi(s))]))
    end
    # The composite port survives even when its entire child collapses to one state.
    za = Node(bases[1:2]; solver=Dense(), keep=within(1.1))
    c = Node(bases[3:3]; solver=Dense(), keep=lowest(3))
    root = Node([za, c]; compute=40, keep=lowest(40))
    previous = nothing
    for (scale, nkept) in ((0.4, 3), (2.0, 1), (0.4, 3))
        θ = (; scale)
        sys = materialize(root, terms, θ; previous)
        @test length(energies(sys, za)) == nkept
        @test size(operator(sys)) == (3nkept, 3nkept)
        P = kron(_primitive(materialized(sys, c)), _primitive(materialized(sys, za)))
        Hflat = primitive_hamiltonian(terms, θ, bases)
        M = Matrix(operator(sys))
        @test M ≈ P' * Hflat * P atol=1e-12
        @test M ≈ M' atol=1e-12
        @test energies(sys) ≈ eigvals(Hermitian(P' * Hflat * P)) atol=1e-12
        @test eigenvectors(sys)' * eigenvectors(sys) ≈ I
        if previous !== nothing
            @test _initial_block(previous.root, sys.root.dims, length(energies(sys))) === nothing
        end
        previous = sys
    end
end

@testset "minimal periodic grids: every Fourier mode and the Nyquist mode" begin
    for order in (2, 4, 6, 8), N in (order + 1, order + 2)
        b = PeriodicGrid(N; order)
        φ = KronFreeCircuits.coordinates(b)
        Δ = 2π / N
        h = order ÷ 2
        # Rational stencil weights are converted independently into Fourier eigenvalues.
        w1 = Float64.(KronFreeCircuits._FD1[order])
        w2 = Float64.(KronFreeCircuits._FD2[order])
        A = dense(localop(b, nport())); B = dense(localop(b, n2port()))
        @test A ≈ A' && B ≈ B'
        for k in 0:N-1
            v = cis.(k .* φ) / sqrt(N)
            λ1 = sum(w1[j+h+1] * sin(j*k*Δ) for j in -h:h) / Δ
            λ2 = -sum(w2[j+h+1] * cos(j*k*Δ) for j in -h:h) / Δ^2
            @test A * v ≈ λ1 * v atol=2e-13
            @test B * v ≈ λ2 * v atol=2e-13
        end
        if iseven(N)
            v = cis.((N ÷ 2) .* φ) / sqrt(N)
            @test norm(A * v) < 1e-13
            @test real(dot(v, B * v)) > 1 # n² must not be assembled by squaring n
        end
    end
end

@testset "Fock displacement: large arguments must remain finite" begin
    # Exact matrix elements of a unitary have modulus <= 1. The unscaled Laguerre
    # recurrence currently overflows before multiplying by an underflowed Gaussian.
    D = KronFreeCircuits.displacement(400, 40.0im)
    @test_broken all(isfinite, D)
    # This is not harmless underflow of negligible entries: the last diagonal
    # element is about -0.03525. Evaluate exp(-x/2) L_399(x) at 256-bit precision.
    reference = setprecision(256) do
        x = BigFloat(1600)
        p, q = BigFloat(1), 1 - x
        for m in 2:399
            p, q = q, ((2m - 1 - x) * q - (m - 1) * p) / m
        end
        Float64(exp(-x / 2) * q)
    end
    @test abs(reference) > 0.03
    @test_broken isapprox(D[end, end], reference; atol=1e-12)
    # A smaller displacement exercises the same recurrence without overflow.
    Dsmall = KronFreeCircuits.displacement(40, 3.0im)
    @test all(isfinite, Dsmall)
    @test opnorm(Dsmall) <= 1 + 1e-12
    @test KronFreeCircuits.displacement(40, -3.0im) ≈ Dsmall'
end
