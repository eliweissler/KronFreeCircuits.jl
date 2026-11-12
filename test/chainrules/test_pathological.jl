# Included by the optional ChainRules tier. Also runnable from the main project:
# julia --compiled-modules=existing --project -e 'using KronFreeCircuits; include("test/chainrules/test_pathological.jl")'
module PathologicalChainRulesReview
using Test, LinearAlgebra, KronFreeCircuits
const Ext = Base.get_extension(KronFreeCircuits, :KronFreeCircuitsChainRulesCoreExt)
const CRC = Ext.ChainRulesCore

@testset "AD zero weights must not propagate unrelated degenerate derivatives" begin
    key = (:phi, nport())
    root = Node([:phi => ChargeBasis(1)]; solver=Dense(), ports=[key])
    spec = load_spec(joinpath(@__DIR__, "..", "fixtures", "cos2phi.json"))
    θ = (E_C=1.0, E_1=0.0, E_2=0.0, n_g=0.0)
    # H = diag(4,0,4); E_2 couples only the excited pair. Ground energy is
    # identically zero for small E_2, although individual excited derivatives
    # are deliberately undefined by the derivative API at the crossing.
    sys = materialize(root, spec, θ)
    d = derivatives(sys; wrt=:E_2)
    @test energies(d, :E_2)[1] == 0
    @test all(isnan, energies(d, :E_2)[2:3])
    for e2 in (-0.01, 0.01)
        @test energies(materialize(root, spec, merge(θ, (E_2=e2,))))[1] == 0
    end
    y, back = CRC.rrule(spectrum, root, spec, θ)
    grad = back((energies=[1.0, 0.0, 0.0],))[4]
    @test grad[:E_2] == 0.0
    @test back((energies=[1.0, -0.0, 0.0],))[4][:E_2] == 0.0
    # A nonzero cotangent on the undefined pair must still propagate NaN.
    @test isnan(back((energies=[0.0, 1.0, 0.0],))[4][:E_2])
    # A NaN cotangent is not zero: it propagates, even onto a defined (and zero) derivative.
    @test isnan(back((energies=[NaN, 0.0, 0.0],))[4][:E_2])
    no = CRC.NoTangent()
    _, sparse = CRC.frule((no, no, no, (E_C=1.0,)), spectrum, root, spec, θ)
    _, explicit = CRC.frule((no, no, no, (E_C=1.0, E_1=0.0, E_2=0.0, n_g=0.0)), spectrum, root, spec, θ)
    @test sparse.energies ≈ [0.0, 4.0, 4.0] atol=1e-12
    @test explicit.energies ≈ sparse.energies
    _, negzero = CRC.frule((no, no, no, (E_C=1.0, E_1=-0.0, E_2=-0.0, n_g=0.0 + 0.0im)), spectrum, root, spec, θ)
    @test negzero.energies ≈ sparse.energies
    # A nonzero tangent along the parameter with the undefined derivatives must still give NaN there, and
    # only there; a NaN tangent is not zero and propagates everywhere.
    _, along = CRC.frule((no, no, no, (E_C=0.0, E_2=1.0)), spectrum, root, spec, θ)
    @test along.energies[1] == 0 && all(isnan, along.energies[2:3])
    _, nantangent = CRC.frule((no, no, no, (E_C=NaN, E_2=0.0)), spectrum, root, spec, θ)
    @test all(isnan, nantangent.energies)
    # The two rules stay consistent on the defined part: ⟨ȳ, J Δ⟩ = ⟨Jᵀ ȳ, Δ⟩ for Δ along E_C.
    ȳ = [1.0, 2.0, 3.0]
    @test dot(ȳ, sparse.energies) ≈ back((energies=ȳ,))[4][:E_C]

    # Elementwise-zero cotangents need the same treatment for ports. Only the
    # ground-state diagonal is used here; undefined excited-state rows and
    # columns must not contaminate that scalar contraction.
    yp, backp = CRC.rrule(spectrum, root, spec, θ; ports=[key])
    Pbar = zeros(ComplexF64, size(only(yp.ports)))
    Pbar[1, 1] = 1
    portgrad = backp((energies=zeros(3), ports=[Pbar]))[4]
    @test portgrad[:E_2] == 0.0
    @test backp((ports=[Pbar],))[4][:E_2] == 0.0                      # the energies' cotangent absent
    # A nonzero port cotangent, real or imaginary, on an undefined entry must still propagate NaN.
    dP = port(derivatives(sys; wrt=:E_2), :E_2, key...)
    @test !isnan(dP[1, 1]) && isnan(dP[2, 3])
    for w in (1.0, 1.0im)
        Pnan = copy(Pbar)
        Pnan[2, 3] = w
        @test isnan(backp((energies=zeros(3), ports=[Pnan]))[4][:E_2])
    end
end

@testset "the zero-aware contraction of the rules" begin
    dotnz = Ext._dot_nonzero
    Z = CRC.ZeroTangent()
    # without zero weights it is `real(dot(w, d))`: conjugation on the weight
    w, d = [1.5, -2.0, 0.25], [2.0, 3.0, -4.0]
    @test dotnz(w, d) == real(dot(w, d)) == -4.0
    wc = [1.0 + 2.0im 0.5im; -1.0 3.0 - 1.0im]
    dc = [2.0 - 1.0im 4.0; 1.0im -2.0 + 2.0im]
    @test dotnz(wc, dc) ≈ real(dot(wc, dc))
    @test dotnz(wc, real(dc)) ≈ real(dot(wc, real(dc)))               # a complex weight on a real derivative
    @test dotnz([1.0im], [1.0im]) == 1.0 && dotnz([1.0im], [1.0 + 0.0im]) == 0.0
    # structural zeros: for the whole weight, or as entries
    @test dotnz(Z, [NaN, 1.0]) === 0.0
    @test dotnz(CRC.NoTangent(), fill(NaN, 2, 2)) === 0.0
    @test dotnz(Any[Z, 2.0, nothing], [NaN, 3.0, NaN]) == 6.0
    # numeric zeros, `-0.0` and complex zeros included, leave out undefined entries
    @test dotnz([0.0, 2.0, -0.0], [NaN, 3.0, NaN]) == 6.0
    @test dotnz([0, 1], [NaN, 2.0]) == 2.0
    @test dotnz([0.0 + 0.0im, 1.0 - 1.0im, complex(-0.0, 0.0), complex(0.0, -0.0)],
                [complex(NaN, NaN), 2.0 + 1.0im, complex(NaN, 0.0), complex(0.0, NaN)]) == 1.0
    @test dotnz(zeros(3), fill(NaN, 3)) === 0.0
    @test dotnz(zeros(ComplexF64, 2, 2), fill(complex(NaN, NaN), 2, 2)) === 0.0
    @test dotnz(Float64[], Float64[]) === 0.0
    # a nonzero weight, however small, on an undefined entry gives NaN
    @test isnan(dotnz([1.0, 1e-300], [2.0, NaN]))
    @test isnan(dotnz([0.0, 1.0im], [1.0, complex(NaN, 0.0)]))
    @test isnan(dotnz([0.0, complex(0.0, 5e-324)], [NaN, complex(0.0, NaN)]))
    # a NaN weight is not zero: it propagates, even onto a zero derivative
    @test isnan(dotnz([NaN, 0.0], [0.0, NaN]))
    @test isnan(dotnz([complex(0.0, NaN)], [1.0 + 0.0im]))
    # the lengths must agree, as for `dot`
    @test_throws DimensionMismatch dotnz([1.0, 2.0], [1.0])
end

@testset "frule of spectrum: complex and non-Float64 tangents" begin
    key = [(:phi, nport()), (:phi, expiphi(1.0))]
    root = Node([:phi => ChargeBasis(2)]; solver=Dense(), ports=key)
    spec = load_spec(joinpath(@__DIR__, "..", "fixtures", "cos2phi.json"))
    θ = (E_C=1.0, E_1=0.2, E_2=0.1, n_g=0.0)
    no = CRC.NoTangent()
    fwd(Δ; ports=key, kw...) = CRC.frule((no, no, no, Δ), spectrum, root, spec, θ; ports, kw...)[2]
    y = spectrum(root, spec, θ; ports=key)
    @test all(P -> P isa Matrix{Float64}, y.ports)                    # a real tree: real ports, with real derivatives

    # a real tangent: real tangents of the energies and of the real ports, as before
    a, b = (E_C=0.3, E_1=-1.2, E_2=0.7, n_g=0.25), (E_C=-0.5, E_1=0.4, E_2=0.0, n_g=1.5)
    ẏa, ẏb = fwd(a), fwd(b)
    @test ẏa.energies isa Vector{Float64} && all(P -> P isa Matrix{Float64}, ẏa.ports)
    @test ẏb.energies isa Vector{Float64} && all(P -> P isa Matrix{Float64}, ẏb.ports)
    @test norm(ẏa.energies) > 1e-2 && norm(ẏb.energies) > 1e-2 && all(P -> norm(P) > 1e-2, ẏa.ports) && all(P -> norm(P) > 1e-2, ẏb.ports)
    # a complex tangent: complex tangents of both (it used to throw an `InexactError`, the tangent of the energies being
    # a `Vector{Float64}` and that of a real port a `Matrix{Float64}`), linear in the tangent: ẏ(a + ib) = ẏ(a) + i ẏ(b)
    ẏc = fwd(map(complex, a, b))
    @test ẏc.energies isa Vector{ComplexF64} && all(P -> P isa Matrix{ComplexF64}, ẏc.ports)
    @test ẏc.energies ≈ ẏa.energies + im * ẏb.energies rtol=1e-13
    for p in eachindex(key)
        @test ẏc.ports[p] ≈ ẏa.ports[p] + im * ẏb.ports[p] rtol=1e-13
    end
    # without ports (the energies alone), along one parameter, and through a `Tangent`
    ẏ1 = fwd((E_C=1.0 + 0.5im,); ports=Tuple{Symbol,KronFreeCircuits.Port}[])
    @test ẏ1.energies isa Vector{ComplexF64} && isempty(ẏ1.ports)
    @test ẏ1.energies ≈ (1.0 + 0.5im) * fwd((E_C=1.0,)).energies rtol=1e-13
    ẏt = fwd(CRC.Tangent{typeof(θ)}(; E_1=2.0im))
    @test ẏt.energies ≈ 2.0im * fwd((E_1=1.0,)).energies rtol=1e-13
    # the port side on its own: an imaginary tangent times the real derivative of a real port
    ẏi = fwd((E_1=1.0im,))
    for p in eachindex(key)
        @test ẏi.ports[p] isa Matrix{ComplexF64}
        @test ẏi.ports[p] ≈ im * fwd((E_1=1.0,)).ports[p] rtol=1e-13
    end

    # no widening by a real tangent of another type, nor by a component that is zero (it is skipped, whatever its type)
    ẏ0 = fwd((E_C=1.0,))
    for Δ in ((E_C=1,), (E_C=1.0f0,), (E_C=1//1,), (E_C=true,), (E_C=1.0, n_g=0.0 + 0.0im), (E_C=1.0, E_1=0.0im, E_2=big"0.0"))
        ẏ = fwd(Δ)
        @test ẏ.energies isa Vector{Float64} && all(P -> P isa Matrix{Float64}, ẏ.ports)
        @test ẏ.energies == ẏ0.energies && ẏ.ports == ẏ0.ports
    end
    # a tangent of a wider real type carries its type through
    ẏbig = fwd((E_C=big"1.0", E_1=0.5))
    @test ẏbig.energies isa Vector{BigFloat} && all(P -> P isa Matrix{BigFloat}, ẏbig.ports)
    @test ẏbig.energies ≈ fwd((E_C=1.0, E_1=0.5)).energies rtol=1e-13
    # a complex tangent that is zero: zeros without the derivative pass (whose options here would be refused), and a
    # nonzero one along a parameter the spectrum does not depend on gives zeros as well
    ẏz = fwd((E_C=0.0 + 0.0im, E_1=complex(-0.0, 0.0)); derivative_options=(; lintol=-1.0))
    @test ẏz.energies isa Vector{Float64} && all(iszero, ẏz.energies) && all(P -> all(iszero, P), ẏz.ports)
    _, ẏu = CRC.frule((no, no, no, (unused=1.0 + 2.0im,)), spectrum, root, spec, merge(θ, (unused=42.0,)); ports=key)
    @test all(iszero, ẏu.energies) && all(P -> all(iszero, P), ẏu.ports)

    # undefined derivatives with a complex tangent: NaN only where a nonzero component selects it
    θd = (E_C=1.0, E_1=0.0, E_2=0.0, n_g=0.0)                         # the degenerate, coupled pair of the first testset
    rootd = Node([:phi => ChargeBasis(1)]; solver=Dense())
    pushd(Δ) = CRC.frule((no, no, no, Δ), spectrum, rootd, spec, θd)[2].energies
    @test pushd((E_C=1.0im, E_2=0.0)) ≈ [0.0, 4.0im, 4.0im] atol=1e-12
    @test pushd((E_C=1.0im, E_2=0.0 + 0.0im)) ≈ [0.0, 4.0im, 4.0im] atol=1e-12
    along = pushd((E_C=0.0, E_2=1.0im))
    @test along isa Vector{ComplexF64} && along[1] == 0 && all(isnan, along[2:3])
    @test all(isnan, pushd((E_C=complex(0.0, NaN),)))
end
end
