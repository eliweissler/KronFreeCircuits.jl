# The ChainRulesCore tier: the rrule and frule of `spectrum` (the ChainRulesCore extension), directly and through Zygote.
# Run with
#     julia --project=test/chainrules test/chainrules/runtests.jl
# The default tier (test/runtests.jl, `Pkg.test()`) never includes this directory.
using Test, LinearAlgebra, KronFreeCircuits, ChainRulesCore, Zygote
include("../helpers.jl")                     # it loads JSON3 and Random at top level, hence the two entries in the Project

@test Base.get_extension(KronFreeCircuits, :KronFreeCircuitsChainRulesCoreExt) !== nothing
key = [(:varphi_1, nport()), (:varphi_1, expiphi(1.0))]
loss(y) = (y.energies[2] - y.energies[1] - 5.0)^2 + 0.3 * y.energies[4] + abs2(y.ports[1][1, 2]) + real(y.ports[2][2, 3]) - 2 * imag(y.ports[2][1, 3])

@testset "rrule and frule of spectrum against finite differences" for periodic in (false, true)
    c = dc_two_transmons(; periodic)
    F(θ) = loss(spectrum(c.root, c.spec, θ; ports=key))
    fd = [first(fd_derivative(F, c.θ, x; h=1e-3)) for x in keys(c.θ)]
    # the pullback, fed the loss's cotangent by hand
    y, back = ChainRulesCore.rrule(spectrum, c.root, c.spec, c.θ; ports=key)
    Ē = zeros(length(y.energies)); Ē[2] = 2 * (y.energies[2] - y.energies[1] - 5.0); Ē[1] = -Ē[2]; Ē[4] = 0.3
    P̄1 = zeros(ComplexF64, size(y.ports[1])); P̄1[1, 2] = 2 * y.ports[1][1, 2]
    P̄2 = zeros(ComplexF64, size(y.ports[2])); P̄2[2, 3] = 1; P̄2[1, 3] = -2im
    θ̄ = back(Tangent{typeof(y)}(; energies=Ē, ports=[P̄1, P̄2]))[4]
    @test [θ̄[x] for x in keys(c.θ)] ≈ fd rtol=1e-6
    # the pushforward along each parameter
    for (i, x) in enumerate(keys(c.θ))
        Δθ = Tangent{typeof(c.θ)}(; NamedTuple{(x,)}((1.0,))...)
        _, ẏ = ChainRulesCore.frule((NoTangent(), NoTangent(), NoTangent(), Δθ), spectrum, c.root, c.spec, c.θ; ports=key)
        @test ẏ.energies ≈ first(fd_derivative(θ -> spectrum(c.root, c.spec, θ; ports=key).energies, c.θ, x; h=1e-3)) rtol=1e-6 atol=1e-9
    end
    # an engine, on a loss written in plain Julia
    g = Zygote.gradient(F, c.θ)[1]
    @test [g[x] for x in keys(c.θ)] ≈ fd rtol=1e-6
end

@testset "the rules' tangents: missing cotangents, unused parameters, the pushforward of ports" begin
    c = dc_two_transmons()
    θ = merge(c.θ, (unused=0.5,))          # a parameter that no term or the constant depends on
    y, back = ChainRulesCore.rrule(spectrum, c.root, c.spec, θ; ports=key)
    @test y.energies == spectrum(c.root, c.spec, θ; ports=key).energies
    t = back(Tangent{typeof(y)}(; energies=ones(length(y.energies))))
    @test t[1:3] == (NoTangent(), NoTangent(), NoTangent())
    @test t[4][:unused] == 0
    @test all(iszero, (back(ZeroTangent())[4][x] for x in keys(θ)))
    # a zero cotangent of one port contributes nothing
    P̄2 = zeros(ComplexF64, size(y.ports[2])); P̄2[1, 3] = 1 - 2im
    @test back(Tangent{typeof(y)}(; ports=[ZeroTangent(), P̄2]))[4][:g] ≈
          back(Tangent{typeof(y)}(; energies=zeros(length(y.energies)), ports=[zero(y.ports[1]), P̄2]))[4][:g]
    # the pushforward of a complex off-diagonal port element along a combination of two parameters
    Δθ = Tangent{typeof(θ)}(; g=1.0, n_g1=-0.5, unused=3.0)
    _, ẏ = ChainRulesCore.frule((NoTangent(), NoTangent(), NoTangent(), Δθ), spectrum, c.root, c.spec, θ; ports=key)
    P13(t) = spectrum(c.root, c.spec, t; ports=key).ports[2][1, 3]
    fd = first(fd_derivative(P13, c.θ, :g; h=1e-3)) - 0.5 * first(fd_derivative(P13, c.θ, :n_g1; h=1e-3))
    @test ẏ.ports[2][1, 3] ≈ fd rtol=1e-6
    @test all(eltype(ẏ.ports[p]) === eltype(y.ports[p]) for p in eachindex(y.ports))     # real ports of a real root
    # a zero cotangent or tangent gives zeros without running the derivative pass (whose options here would be refused)
    bad = (; lintol=-1.0)
    yb, backb = ChainRulesCore.rrule(spectrum, c.root, c.spec, θ; ports=key, derivative_options=bad)
    @test all(iszero, (backb(ZeroTangent())[4][x] for x in keys(θ)))
    @test all(iszero, (backb(Tangent{typeof(yb)}(; energies=zeros(length(yb.energies))))[4][x] for x in keys(θ)))
    @test_throws "`lintol` must be positive" backb(Tangent{typeof(yb)}(; energies=ones(length(yb.energies))))
    Δ0 = Tangent{typeof(θ)}(; g=0.0)
    _, ẏ0 = ChainRulesCore.frule((NoTangent(), NoTangent(), NoTangent(), Δ0), spectrum, c.root, c.spec, θ; ports=key,
                                 derivative_options=bad)
    @test all(iszero, ẏ0.energies) && all(P -> all(iszero, P), ẏ0.ports)
    # options the rules set themselves are refused
    @test_throws ArgumentError ChainRulesCore.rrule(spectrum, c.root, c.spec, θ; ports=key, derivative_options=(; order=2))
end
