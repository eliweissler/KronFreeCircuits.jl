module KronFreeCircuitsChainRulesCoreExt

# The differentiation rules of `spectrum` (design §4.6): an `rrule` and an `frule`, each running the first-order
# upward passes of `derivatives` once and contracting them with the cotangent or the tangent.
# The convention for complex outputs is that of `spectrum`'s docstring.

using KronFreeCircuits: KronFreeCircuits, Node, Port, Spec, spectrum, materialize
using ChainRulesCore: ChainRulesCore, AbstractZero, NoTangent, Tangent, ZeroTangent, unthunk
using LinearAlgebra: dot

# The component `name` of a cotangent or tangent of a `NamedTuple`, unthunked; a zero (or `nothing`) when absent.
_component(::Union{Nothing,AbstractZero}, ::Symbol) = ZeroTangent()
_component(t::Tangent, name::Symbol) = _unzero(unthunk(getproperty(t, name)))
_component(t::NamedTuple, name::Symbol) = _unzero(unthunk(get(t, name, ZeroTangent())))
_unzero(x) = x === nothing ? ZeroTangent() : x
_iszero(x) = x isa AbstractZero
# A cotangent or tangent component that is zero: a structural zero, or numbers that are all zero.
_allzero(x) = _iszero(x) || (x isa Number && iszero(x)) || (x isa AbstractArray && all(_allzero, x))

# `Σ_i Ē_i ∂E_i/∂x + Σ_p Re Σ_{ij} conj(P̄_p[i, j]) ∂P_p[i, j]/∂x` for one parameter's derivatives `dy`.
function _contract(Ē, P̄, dy)
    s = 0.0
    _iszero(Ē) || (s += real(dot(Ē, dy.energies)))
    _iszero(P̄) && return s
    length(P̄) == length(dy.ports) ||
        throw(DimensionMismatch("the cotangent of `ports` has $(length(P̄)) entries, the output $(length(dy.ports))"))
    for (p, Pp) in enumerate(P̄)
        Pp = _unzero(unthunk(Pp))
        _iszero(Pp) || (s += real(dot(Pp, dy.ports[p])))
    end
    return s
end

function ChainRulesCore.rrule(::typeof(spectrum), root::Node, spec::Spec, θ::NamedTuple;
                              ports::AbstractVector=Tuple{Symbol,Port}[], derivative_options::NamedTuple=(;))
    KronFreeCircuits._check_spectrum_options(ports, derivative_options)
    sys = materialize(root, spec, θ)
    y = KronFreeCircuits._spectrum_output(sys, ports)
    jacobian = Ref{Any}(nothing)            # the derivatives, formed on the pullback's first call
    function spectrum_pullback(ȳ)
        ȳ = unthunk(ȳ)
        Ē, P̄ = _component(ȳ, :energies), _component(ȳ, :ports)
        if _allzero(Ē) && (_iszero(P̄) || all(P -> _allzero(_unzero(unthunk(P))), P̄))
            # a zero cotangent: zeros, without the derivative pass
            return NoTangent(), NoTangent(), NoTangent(), Tangent{typeof(θ)}(; map(_ -> 0.0, θ)...)
        end
        jacobian[] === nothing && (jacobian[] = KronFreeCircuits._spectrum_derivatives(sys, ports, derivative_options))
        J = jacobian[]
        θ̄ = map(x -> haskey(J, x) ? _contract(Ē, P̄, J[x]) : 0.0, NamedTuple{keys(θ)}(keys(θ)))
        return NoTangent(), NoTangent(), NoTangent(), Tangent{typeof(θ)}(; θ̄...)
    end
    return y, spectrum_pullback
end

function ChainRulesCore.frule((_, _, _, Δθ), ::typeof(spectrum), root::Node, spec::Spec, θ::NamedTuple;
                              ports::AbstractVector=Tuple{Symbol,Port}[], derivative_options::NamedTuple=(;))
    KronFreeCircuits._check_spectrum_options(ports, derivative_options)
    sys = materialize(root, spec, θ)
    y = KronFreeCircuits._spectrum_output(sys, ports)
    Δθ = unthunk(Δθ)
    Ė = zero(y.energies)
    if all(x -> _allzero(_component(Δθ, x)), keys(θ))
        # a zero tangent: zeros, without the derivative pass
        return y, Tangent{typeof(y)}(; energies=Ė, ports=[zero(P) for P in y.ports])
    end
    J = KronFreeCircuits._spectrum_derivatives(sys, ports, derivative_options)
    # the tangent of a port has the promoted element type of the port and of its derivative
    Ṗ = [zeros(promote_type(eltype(P), (eltype(dy.ports[p]) for dy in values(J))...), size(P)) for (p, P) in enumerate(y.ports)]
    for (x, dy) in J
        δ = _component(Δθ, x)
        _iszero(δ) && continue
        Ė .+= δ .* dy.energies
        for p in eachindex(Ṗ)
            Ṗ[p] .+= δ .* dy.ports[p]
        end
    end
    return y, Tangent{typeof(y)}(; energies=Ė, ports=Ṗ)
end

end # module
