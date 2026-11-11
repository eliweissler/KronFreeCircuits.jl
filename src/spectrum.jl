"""
    spectrum(root::Node, spec::Spec, θ::NamedTuple; ports=Tuple{Symbol,Port}[], derivative_options=(;))
        -> (; energies::Vector{Float64}, ports::Vector{<:Matrix})

The kept energies of the root of the tree `root` materialized at `θ`, and its requested ports, as copies.
`ports` holds `(mode, port)` pairs that the root forms (list them in the root's `ports`); `ports[i]` of the result is
`port(sys, ports[i]...)` of `sys = materialize(root, spec, θ)`, in the element type of the stored port, which may be
complex and is not Hermitian in general.
A pair the root did not form is an `ArgumentError`, as for [`port`](@ref).

This is the one function that the differentiation rules of the ChainRulesCore extension cover, so that a loss written
in plain Julia over the energies and port elements can be differentiated by an autodiff engine that uses those rules.
The core itself has no autodiff code; the rules (an `rrule` and an `frule`) load with ChainRulesCore, which
`KronFreeCircuits` already loads through its dependencies.
Both rules call [`derivatives`](@ref) once, with `wrt` the keys of `θ` that some term or the constant of the spec
depends on and with `derivative_options` (keywords of `derivatives` such as `lintol` or `given_independent`; not
`wrt`, `order` or `mixed`), which `spectrum` itself ignores.
The derivatives are those of the calculation as performed (see [`derivatives`](@ref)); a key of `θ` that nothing
depends on has a zero derivative, and a tree that `derivatives` refuses makes the rules throw its `ArgumentError`.
The `rrule` follows ChainRules' convention for a real loss `L` of complex intermediates: the cotangent of a complex
port element `P` is `∂L/∂Re P + i ∂L/∂Im P`, so the pullback returns, for each key `x` of `θ`,
`Σ_i Ē_i ∂E_i/∂x + Σ_p Re Σ_{ij} conj(P̄_p[i, j]) ∂P_p[i, j]/∂x`.
Second derivatives are not differentiated through: a loss that needs them calls [`derivatives`](@ref) itself.
"""
function spectrum(root::Node, spec::Spec, θ::NamedTuple; ports::AbstractVector=Tuple{Symbol,Port}[],
                  derivative_options::NamedTuple=(;))
    _check_spectrum_options(ports, derivative_options)
    return _spectrum_output(materialize(root, spec, θ), ports)
end

function _check_spectrum_options(ports::AbstractVector, derivative_options::NamedTuple)
    for p in ports
        p isa Tuple{Symbol,Port} ||
            throw(ArgumentError("`ports` must hold (mode, port) pairs such as (:varphi_1, nport()), got $(repr(p))"))
    end
    for k in (:wrt, :order, :mixed)
        haskey(derivative_options, k) &&
            throw(ArgumentError("`derivative_options` may not set `$k`: the rules of `spectrum` differentiate along " *
                                "the parameters of θ to first order"))
    end
    return nothing
end

# The output of `spectrum` for the materialized `sys`: copies of the root's kept energies and of its ports `ports`.
function _spectrum_output(sys::System, ports::AbstractVector)
    P = isempty(ports) ? Matrix{Float64}[] : [copy(port(sys, p...)) for p in ports]
    return (; energies=copy(energies(sys)), ports=P)
end

# The keys of `sys.θ` that some term or the constant depends on, in the order of `sys.θ`.
function _spectrum_parameters(sys::System)
    deps = Set{Symbol}(sys.constant.deps)
    for t in sys.terms
        union!(deps, t.coeff.deps)
    end
    return Symbol[x for x in keys(sys.θ) if x in deps]
end

# The first derivatives of `spectrum`'s output, `x => (; energies, ports)` for each key `x` of `sys.θ` that some term
# or the constant depends on (keys absent from the result have zero derivatives), from one call of `derivatives`.
function _spectrum_derivatives(sys::System, ports::AbstractVector, derivative_options::NamedTuple)
    _check_spectrum_options(ports, derivative_options)
    wrt = _spectrum_parameters(sys)
    out = Dict{Symbol,Any}()
    if isempty(wrt)          # nothing to differentiate; a tree the pass refuses is refused all the same
        _check_differentiable(sys; given_independent=get(derivative_options, :given_independent, false))
        return out
    end
    d = derivatives(sys; wrt, derivative_options...)
    for x in wrt
        out[x] = (; energies=energies(d, x), ports=[port(d, x, p...) for p in ports])
    end
    return out
end
