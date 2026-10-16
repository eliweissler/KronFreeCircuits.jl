"""
    Port(kind, k)

A named single-mode operator: `:n` (charge), `:n2` (charge squared), `:phi`, `:phi2`,
`:expiphi` (`exp(i k φ)`) or `:exp2pin` (`exp(2π i k n)`).
`k` is the exponent coefficient (zero for the polynomial ports).
"""
struct Port
    kind::Symbol
    k::Float64
    Port(kind::Symbol, k::Real) = new(kind, Float64(k) + 0.0)     # + 0.0 maps -0.0 to 0.0 (hash)
end

Base.:(==)(a::Port, b::Port) = a.kind === b.kind && a.k == b.k
Base.hash(p::Port, h::UInt) = hash(p.k, hash(p.kind, hash(:Port, h)))

nport() = Port(:n, 0)
n2port() = Port(:n2, 0)
phiport() = Port(:phi, 0)
phi2port() = Port(:phi2, 0)
expiphi(k::Real) = Port(:expiphi, k)
exp2pin(k::Real) = Port(:exp2pin, k)

const _POLY_PORTS = (n=nport, n2=n2port, phi=phiport, phi2=phi2port)

"""
    port_of_kind(kind::Symbol, k=0.0) -> Port

The `Port` named by `kind` (`:n, :n2, :phi, :phi2, :expiphi, :exp2pin`); `k` is the exponent
coefficient of the exponential ports and must be zero for the polynomial ones.
"""
function port_of_kind(kind::Symbol, k::Real=0.0)
    if haskey(_POLY_PORTS, kind)
        iszero(k) || throw(ArgumentError("port kind :$kind takes no exponent, got k = $k"))
        return _POLY_PORTS[kind]()
    end
    kind === :expiphi && return expiphi(k)
    kind === :exp2pin && return exp2pin(k)
    throw(ArgumentError("unknown port kind :$kind; expected :n, :n2, :phi, :phi2, :expiphi or :exp2pin"))
end

"""
    Factor(mode, port)

The operator `port` acting on the mode named `mode`.
"""
struct Factor
    mode::Symbol
    port::Port
end

"""
    Term(coeff, factors)

A product term: a parameter-dependent coefficient times single-mode port operators.
"""
struct Term
    coeff::Coefficient
    factors::Vector{Factor}
end

"""
    modes(t::Term) -> Vector{Symbol}

The modes the term acts on, in factor order.
"""
modes(t::Term) = [f.mode for f in t.factors]

# Structural test for an expression that is identically zero, seeing through
# sums and products of zero literals (which the Coefficient algebra does not simplify).
_iszero_ex(x::Number) = iszero(x)
_iszero_ex(::Any) = false
function _iszero_ex(ex::Expr)
    ex.head === :call || return false
    op, args = ex.args[1], ex.args[2:end]
    op === :+ && return all(_iszero_ex, args)
    op === :* && return any(_iszero_ex, args)
    op === :- && return all(_iszero_ex, args)
    op === :/ && return _iszero_ex(args[1])
    return false
end
_iszero(c::Coefficient) = iszero_expr(c) || _iszero_ex(c.ex)

# Sum of coefficients, skipping those that are identically zero.
function _sum(cs)
    acc = nothing
    for c in cs
        _iszero(c) && continue
        acc = acc === nothing ? c : acc + c
    end
    return acc === nothing ? Coefficient(0) : acc
end

"""
    circuit_terms(spec::Spec) -> Vector{Term}

Expand the spec into product terms (the constant is excluded; see [`constant_term`](@ref)).
Terms whose coefficient is identically zero are dropped.
"""
function circuit_terms(spec::Spec)
    N = length(spec.modes)
    names = [m.name for m in spec.modes]
    terms = Term[]
    add!(c, fs) = _iszero(c) || push!(terms, Term(c, fs))

    for i in 1:N
        add!(4 * spec.E_C[i, i], [Factor(names[i], n2port())])
    end
    for i in 1:N, j in i+1:N
        add!(4 * (spec.E_C[i, j] + spec.E_C[j, i]),
             [Factor(names[i], nport()), Factor(names[j], nport())])
    end
    for i in 1:N
        lin = _sum((spec.E_C[i, j] + spec.E_C[j, i]) * spec.n_g[j] for j in 1:N)
        add!(-4 * lin, [Factor(names[i], nport())])
    end
    for i in 1:N
        add!(0.5 * spec.E_L[i, i], [Factor(names[i], phi2port())])
    end
    for i in 1:N, j in i+1:N
        add!(0.5 * (spec.E_L[i, j] + spec.E_L[j, i]),
             [Factor(names[i], phiport()), Factor(names[j], phiport())])
    end
    for i in 1:N
        add!(spec.flux_linear[i], [Factor(names[i], phiport())])
    end

    for c in spec.cosines, s in (1, -1)
        fs = [Factor(names[j], expiphi(s * c.omega[j])) for j in 1:N if c.omega[j] != 0]
        add!(-0.5 * (c.energy * phase_factor(c.offset, s)), fs)
    end
    for c in spec.charge_cosines, s in (1, -1)
        fs = [Factor(names[j], exp2pin(s * c.kappa[j])) for j in 1:N if c.kappa[j] != 0]
        shift = _sum(c.kappa[j] * spec.n_g[j] for j in 1:N)
        offset = c.offset + (-2π) * shift
        add!(-0.5 * (c.energy * phase_factor(offset, s)), fs)
    end
    return terms
end

"""
    constant_term(spec::Spec) -> Coefficient

The scalar part `constant + 4 n_gᵀ E_C n_g` produced by expanding `(n - n_g)`.
"""
function constant_term(spec::Spec)
    N = length(spec.modes)
    quad = _sum(spec.n_g[i] * spec.E_C[i, j] * spec.n_g[j] for i in 1:N, j in 1:N)
    return spec.constant + 4 * quad
end
