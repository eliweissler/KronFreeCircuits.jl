# Adversarial and pathological cases from feature updates review (2026-11-11).
# Tests covering holes in test coverage and isolating edge-case behavior.
module PathologicalReview
using Test, LinearAlgebra, KronFreeCircuits

@testset "Fock displacement: non-float numeric arguments (integers, rationals, extreme values)" begin
    # Ordinary integers and rationals: `displacement(N::Int, α::Number)` is typed for all `Number`,
    # but internally `r, x, θ = abs(α), abs2(α), angle(α)` leaves `r` as `Int` or `Rational`.
    # Julia Base only defines `frexp` for `Union{Float16, Float32, Float64, BigFloat}`,
    # so calling `displacement` on nonzero integers or rationals throws `MethodError: no method matching frexp(...)`.
    # (Note: `iszero(α)` short-circuits at line 309, so `displacement(N, 0)` passes).
    
    # Passing float cases:
    @test all(isfinite, KronFreeCircuits.displacement(3, 1.0))
    @test all(isfinite, KronFreeCircuits.displacement(3, 1.0f0))
    @test all(isfinite, KronFreeCircuits.displacement(3, 1 + 2im))   # abs(1+2im) is Float64
    @test KronFreeCircuits.displacement(3, 0) == I(3)               # iszero short-circuit
    
    # Failing integer and rational cases (currently broken due to frexp MethodError):
    @test_broken KronFreeCircuits.displacement(3, 1) isa Matrix{ComplexF64}
    @test_broken KronFreeCircuits.displacement(3, -2) isa Matrix{ComplexF64}
    @test_broken KronFreeCircuits.displacement(3, 1//2) isa Matrix{ComplexF64}

    # Boundary and extreme arguments:
    # Dimension 0:
    D0 = KronFreeCircuits.displacement(0, 5.0)
    @test size(D0) == (0, 0)
    @test size(KronFreeCircuits.displacement(0, 0)) == (0, 0)

    # Non-finite inputs:
    @test all(isnan, KronFreeCircuits.displacement(3, NaN))
    @test all(isnan, KronFreeCircuits.displacement(3, complex(NaN, 1.0)))
    @test all(isnan, KronFreeCircuits.displacement(3, complex(1.0, NaN)))
    @test all(isnan, KronFreeCircuits.displacement(3, Inf))
    @test all(isnan, KronFreeCircuits.displacement(3, -Inf))
    @test all(isnan, KronFreeCircuits.displacement(3, complex(Inf, 0.0)))
    @test all(isnan, KronFreeCircuits.displacement(3, complex(0.0, -Inf)))

    # Large argument cutoff (|α| >= 2^20 returns zero matrix):
    D_cutoff = KronFreeCircuits.displacement(3, 0x1p20)
    @test all(iszero, D_cutoff)
    @test all(iszero, KronFreeCircuits.displacement(3, -0x1p20))

    # Subnormal argument: gradual underflow correctly preserves digits of subnormal elements
    D_subnormal = KronFreeCircuits.displacement(3, 1e-320)
    @test D_subnormal ≈ I(3)
    @test D_subnormal[1, 1] == 1.0 && D_subnormal[2, 2] == 1.0 && D_subnormal[3, 3] == 1.0
    @test D_subnormal[2, 1] ≈ 1e-320 && D_subnormal[1, 2] ≈ -1e-320 && iszero(D_subnormal[3, 1])

    # Adjoint exactness for negative imaginary argument:
    @test KronFreeCircuits.displacement(10, -3.0im) == adjoint(KronFreeCircuits.displacement(10, 3.0im))
    @test KronFreeCircuits.displacement(10, -40.0im) == adjoint(KronFreeCircuits.displacement(10, 40.0im))
end

@testset "ChainRules spectrum: forward tangent type promotion and complex tangents" begin
    Ext = Base.get_extension(KronFreeCircuits, :KronFreeCircuitsChainRulesCoreExt)
    if Ext !== nothing
        CRC = Ext.ChainRulesCore
        b = ChargeBasis(1)
        root = Node([:phi => b]; solver=Dense())
        spec = load_spec(joinpath(@__DIR__, "fixtures", "cos2phi.json"))
        θ = (E_C=1.0, E_1=0.2, E_2=0.1, n_g=0.0)
        no = CRC.NoTangent()

        # Complex forward tangent: in `frule`, `Ė` is initialized as `zero(y.energies)` which is `Vector{Float64}`.
        # Adding `δ .* dy.energies` in-place throws InexactError when `δ` has an imaginary part.
        # (Compare with `Ṗ`, which uses `promote_type(eltype(P), ...)` to accommodate promoted tangent types).
        @test_broken begin
            _, ydot = CRC.frule((no, no, no, (E_C=1.0 + 0.5im,)), spectrum, root, spec, θ)
            ydot.energies isa Vector{ComplexF64}
        end

        # Pullback and pushforward with unused parameters in θ:
        θ_unused = merge(θ, (unused=42.0,))
        y, back = CRC.rrule(spectrum, root, spec, θ_unused)
        grad = back((energies=ones(3),))[4]
        @test grad[:unused] == 0.0
        @test grad[:E_C] != 0.0

        # Pushforward along unused parameter:
        _, ydot_unused = CRC.frule((no, no, no, (unused=5.0,)), spectrum, root, spec, θ_unused)
        @test all(iszero, ydot_unused.energies)
        @test all(all(iszero, P) for P in ydot_unused.ports)

        # Pullback with multiple mixed ports (real-character and complex-character):
        p_real = (:phi, nport())
        p_cplx = (:phi, expiphi(1.0))
        root_ports = Node([:phi => ChargeBasis(2)]; solver=Dense(), ports=[p_real, p_cplx])
        y_mp, back_mp = CRC.rrule(spectrum, root_ports, spec, θ; ports=[p_real, p_cplx])
        @test length(y_mp.ports) == 2

        # Verify that passing ZeroTangent() vs an explicit zero matrix gives identical pullback:
        P̄_real = zeros(Float64, size(y_mp.ports[1]))
        P̄_cplx = zeros(ComplexF64, size(y_mp.ports[2]))
        P̄_cplx[1, 2] = 1.0 - 2.0im

        n_kept = length(y_mp.energies)
        g_explicit = back_mp((energies=zeros(n_kept), ports=[P̄_real, P̄_cplx]))[4]
        g_zero_tangent = back_mp((energies=zeros(n_kept), ports=[CRC.ZeroTangent(), P̄_cplx]))[4]
        g_nothing = back_mp((energies=zeros(n_kept), ports=[nothing, P̄_cplx]))[4]
        @test g_explicit[:E_C] == g_zero_tangent[:E_C] == g_nothing[:E_C]
        @test g_explicit[:E_1] == g_zero_tangent[:E_1] == g_nothing[:E_1]
    end
end

@testset "ChainRules _dot_nonzero: numerical, subnormal, and structural zeros" begin
    Ext = Base.get_extension(KronFreeCircuits, :KronFreeCircuitsChainRulesCoreExt)
    if Ext !== nothing
        dotnz = Ext._dot_nonzero
        Z = Ext.ChainRulesCore.ZeroTangent()

        # Subnormal nonzero weights on NaN must propagate NaN:
        @test isnan(dotnz([5e-324], [NaN]))
        @test isnan(dotnz([complex(0.0, 5e-324)], [complex(0.0, NaN)]))

        # Signed zeros and complex zeros:
        @test dotnz([0.0, -0.0, complex(0.0, 0.0), complex(-0.0, -0.0)], [NaN, NaN, NaN, NaN]) == 0.0

        # Structural zeros mixed with NaN:
        @test dotnz(Any[Z, nothing, Z], [NaN, NaN, NaN]) == 0.0
        @test dotnz(Z, fill(NaN, 5)) == 0.0
        @test dotnz(Ext.ChainRulesCore.NoTangent(), fill(NaN, 5)) == 0.0

        # Dimension mismatch on length:
        @test_throws DimensionMismatch dotnz([1.0, 2.0], [1.0])
    end
end

@testset "perturbative derivatives: spec constant-only parameters" begin
    # Circuit with a parameter that appears only in `spec.constant`:
    spec = parse_spec("""{
        "format": "kronfree-spec",
        "version": 1,
        "units": "GHz",
        "parameters": ["E_C", "shift"],
        "modes": [{"name": "phi", "type": "c"}],
        "E_C": [["E_C"]],
        "E_L": [["0"]],
        "flux_linear": ["0"],
        "n_g": ["0"],
        "cosines": [],
        "charge_cosines": [],
        "constant": "shift + 0.5 * shift^2"
    }""")
    b = ChargeBasis(1)
    root = Node([:phi => b]; solver=Dense(), ports=[(:phi, nport())])
    θ = (E_C=1.0, shift=2.0)
    sys = materialize(root, spec, θ)

    d = derivatives(sys; wrt=(:E_C, :shift), order=2, mixed=((:E_C, :shift),))
    # d(constant)/d(shift) = 1 + shift = 1 + 2 = 3.0:
    @test energies(d, :shift) ≈ fill(3.0, length(energies(sys))) atol=1e-12
    # d^2(constant)/d(shift)^2 = 1.0:
    @test energies(d, :shift, :shift) ≈ fill(1.0, length(energies(sys))) atol=1e-12
    # mixed d^2/d(shift)d(E_C) = 0.0:
    @test energies(d, :E_C, :shift) ≈ zeros(length(energies(sys))) atol=1e-12
    # Port derivatives along shift are identically zero:
    @test all(iszero, port(d, :shift, :phi, nport()))
end

end
