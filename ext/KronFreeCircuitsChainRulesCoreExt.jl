module KronFreeCircuitsChainRulesCoreExt

# The differentiation rules of `spectrum` (design §4.6): an `rrule` and an `frule`, each running the first-order
# upward passes of `derivatives` once and contracting them with the cotangent or the tangent.
# The convention for complex outputs is that of `spectrum`'s docstring.

using KronFreeCircuits: KronFreeCircuits, Node, Port, Spec, spectrum, materialize
using ChainRulesCore: ChainRulesCore, AbstractZero, NoTangent, Tangent, ZeroTangent, unthunk

# The component `name` of a cotangent or tangent of a `NamedTuple`, unthunked; a zero (or `nothing`) when absent.
_component(::Union{Nothing,AbstractZero}, ::Symbol) = ZeroTangent()
_component(t::Tangent, name::Symbol) = _unzero(unthunk(getproperty(t, name)))
_component(t::NamedTuple, name::Symbol) = _unzero(unthunk(get(t, name, ZeroTangent())))
_unzero(x) = x === nothing ? ZeroTangent() : x
_iszero(x) = x isa AbstractZero
# A cotangent or tangent component that is zero: a structural zero, or numbers that are all zero.
_allzero(x) = _iszero(x) || (x isa Number && iszero(x)) || (x isa AbstractArray && all(_allzero, x))

# `Re Σ_i conj(w_i) d_i` over the entries whose weight `w_i` is not zero: `real(dot(w, d))` with the zero-weight
# entries left out, so that an undefined (`NaN`) entry of `d` contributes only where its weight selects it.
# A weight is zero when it is a structural zero (or `nothing`) or a number with `iszero` (`-0.0` and a complex zero
# included); a `NaN` weight is not zero and gives `NaN`, as does a nonzero weight on a `NaN` entry.
# A structural zero for the whole of `w` gives `0.0`; otherwise `w` and `d` must have the same length, as for `dot`.
function _dot_nonzero(w, d)
    _iszero(w) && return 0.0
    length(w) == length(d) ||
        throw(DimensionMismatch("a cotangent has $(length(w)) entries, the output it belongs to $(length(d))"))
    s = 0.0
    for (wi, di) in zip(w, d)
        wi = _unzero(wi)
        (_iszero(wi) || iszero(wi)) && continue
        s += real(conj(wi) * di)
    end
    return s
end

# `Σ_i Ē_i ∂E_i/∂x + Σ_p Re Σ_{ij} conj(P̄_p[i, j]) ∂P_p[i, j]/∂x` for one parameter's derivatives `dy`, each sum
# over the entries whose cotangent is not zero (`_dot_nonzero`).
function _contract(Ē, P̄, dy)
    s = _dot_nonzero(Ē, dy.energies)
    _iszero(P̄) && return s
    length(P̄) == length(dy.ports) ||
        throw(DimensionMismatch("the cotangent of `ports` has $(length(P̄)) entries, the output $(length(dy.ports))"))
    for (p, Pp) in enumerate(P̄)
        s += _dot_nonzero(_unzero(unthunk(Pp)), dy.ports[p])
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
    if all(x -> _allzero(_component(Δθ, x)), keys(θ))
        # a zero tangent: zeros, without the derivative pass
        return y, Tangent{typeof(y)}(; energies=zero(y.energies), ports=[zero(P) for P in y.ports])
    end
    J = KronFreeCircuits._spectrum_derivatives(sys, ports, derivative_options)
    # The parameters whose derivatives are used, each with its tangent component: a structural or a numeric zero
    # leaves its parameter out, and with it both the parameter's derivatives and the component's type.
    active = filter(((δ, _),) -> !_allzero(δ), [(_component(Δθ, x), dy) for (x, dy) in J])
    # The tangent of an output has the promoted element type of the output, of the numbers that multiply its
    # derivatives (a complex or a wider tangent of θ gives a complex or a wider tangent of a real output) and, for a
    # port, of its derivative; a real `Float64` tangent leaves the energies' `Float64`.
    Tδ = promote_type(Union{}, (typeof(δ) for (δ, _) in active if δ isa Number)...)
    Ė = zeros(promote_type(eltype(y.energies), Tδ), length(y.energies))
    Ṗ = [zeros(promote_type(eltype(P), Tδ, (eltype(dy.ports[p]) for dy in values(J))...), size(P)) for (p, P) in enumerate(y.ports)]
    for (δ, dy) in active
        Ė .+= δ .* dy.energies
        for p in eachindex(Ṗ)
            Ṗ[p] .+= δ .* dy.ports[p]
        end
    end
    return y, Tangent{typeof(y)}(; energies=Ė, ports=Ṗ)
end

end # module
