"""
    Port(kind, k)

A named single-mode operator: `:n` (charge), `:n2` (charge squared), `:phi`, `:phi2`,
`:expiphi` (`exp(i k φ)`) or `:exp2pin` (`exp(2π i k n)`), or a derivative piece of a
`PeriodicGrid`: `:d1`, `:d2` (interior bands of `-i ∂_φ`, `-∂_φ²`) and `:d1wrap`, `:d2wrap`
(their wrap corners).
`k` is the exponent coefficient (zero for the polynomial ports and `:d1`, `:d2`), or the corner
`±1` of the wrap ports.
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

The `Port` named by `kind` (`:n, :n2, :phi, :phi2, :expiphi, :exp2pin, :d1, :d1wrap, :d2, :d2wrap`);
`k` is the exponent coefficient of the exponential ports, the corner `±1` of the wrap ports
`:d1wrap, :d2wrap`, and must be zero for the others.
"""
function port_of_kind(kind::Symbol, k::Real=0.0)
    if haskey(_POLY_PORTS, kind)
        iszero(k) || throw(ArgumentError("port kind :$kind takes no exponent, got k = $k"))
        return _POLY_PORTS[kind]()
    end
    kind === :expiphi && return expiphi(k)
    kind === :exp2pin && return exp2pin(k)
    if kind in (:d1, :d2)
        iszero(k) || throw(ArgumentError("port kind :$kind takes no exponent, got k = $k"))
        return Port(kind, 0)
    end
    if kind in (:d1wrap, :d2wrap)
        k in (1, -1) || throw(ArgumentError("port kind :$kind needs the corner k = +1 or -1, got k = $k"))
        return Port(kind, k)
    end
    throw(ArgumentError("unknown port kind :$kind; expected :n, :n2, :phi, :phi2, :expiphi, :exp2pin, " *
                        ":d1, :d1wrap, :d2 or :d2wrap"))
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

# Whether a mode on basis `b` is twisted (its offset charge enters through the boundary condition);
# the `PeriodicGrid` method is in bases.jl.
_is_twisted(::Any) = false

# Per mode, whether it is twisted under `bases` (modes absent from `bases` are not).
_twisted(spec::Spec, bases::AbstractDict) = [haskey(bases, m.name) && _is_twisted(bases[m.name]) for m in spec.modes]

# The twisted first (`kind = :d1`) or second (`:d2`) derivative of a mode with offset charge `ng`, as
# `(twist, port)` pieces: the interior band (twist `nothing`, coefficient 1) and the two wrap corners
# with their twist phases `e^{±2πi n_g}`.
_twisted_pieces(ng::Coefficient, kind::Symbol) =
    [(nothing, Port(kind, 0)), (twist_factor(ng, 1), Port(Symbol(kind, :wrap), 1)),
     (twist_factor(ng, -1), Port(Symbol(kind, :wrap), -1))]

_times(c::Coefficient, ::Nothing) = c
_times(c::Coefficient, t::Coefficient) = c * t

"""
    circuit_terms(spec::Spec) -> Vector{Term}
    circuit_terms(spec::Spec, bases) -> Vector{Term}

Expand the spec into product terms (the constant is excluded; see [`constant_term`](@ref)).
Terms whose coefficient is identically zero are dropped.

`bases` maps mode names to their `Basis` (modes absent from it are expanded in the charge basis, and
the first form means every compact mode in the charge basis).
A compact mode `m` on a [`PeriodicGrid`](@ref) is *twisted*: in the quadratic form
`4 Σ_ij (n_i - n_g,i) E_C[i,j] (n_j - n_g,j)` its `(n_m - n_g,m)` is the twisted derivative
`D₁ = d1 + e^{2πi n_g,m} d1wrap₊ + e^{-2πi n_g,m} d1wrap₋`, so its diagonal kinetic term is
`4 E_C[m,m]` times `D₂ = d2 + e^{2πi n_g,m} d2wrap₊ + e^{-2πi n_g,m} d2wrap₋` (three terms), a cross
term with mode `j` is `4 (E_C[m,j] + E_C[j,m]) D₁ ⊗ n_j` (three terms, nine when `j` is twisted too),
and its linear term is `D₁` times `-4 Σ_j (E_C[m,j] + E_C[j,m]) n_g,j` over the non-twisted `j`.
The canonical linear and constant pieces are formed only from non-twisted modes (see
[`constant_term`](@ref)).
The twist phases are coefficients ([`twist_factor`](@ref)), so the operators stay parameter-free.
Cosines on a twisted mode keep `expiphi(ω)` with integer `ω`.
"""
circuit_terms(spec::Spec) = circuit_terms(spec, Dict{Symbol,Any}())

function circuit_terms(spec::Spec, bases::AbstractDict)
    N = length(spec.modes)
    names = [m.name for m in spec.modes]
    tw = _twisted(spec, bases)
    terms = Term[]
    add!(c, fs) = _iszero(c) || push!(terms, Term(c, fs))
    # (n_i - n_g,i) without its offset: the charge port, or the twisted derivative's pieces
    charge(i) = tw[i] ? _twisted_pieces(spec.n_g[i], :d1) : [(nothing, nport())]

    for i in 1:N
        if tw[i]
            for (t, p) in _twisted_pieces(spec.n_g[i], :d2)
                add!(_times(4 * spec.E_C[i, i], t), [Factor(names[i], p)])
            end
        else
            add!(4 * spec.E_C[i, i], [Factor(names[i], n2port())])
        end
    end
    for i in 1:N, j in i+1:N
        c = 4 * (spec.E_C[i, j] + spec.E_C[j, i])
        for (ti, p_i) in charge(i), (tj, p_j) in charge(j)
            add!(_times(_times(c, ti), tj), [Factor(names[i], p_i), Factor(names[j], p_j)])
        end
    end
    for i in 1:N
        lin = _sum((spec.E_C[i, j] + spec.E_C[j, i]) * spec.n_g[j] for j in 1:N if !tw[j])
        for (t, p) in charge(i)
            add!(_times(-4 * lin, t), [Factor(names[i], p)])
        end
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
    constant_term(spec::Spec, bases) -> Coefficient

The scalar part `constant + 4 n_gᵀ E_C n_g` produced by expanding `(n - n_g)`.
With `bases` (see [`circuit_terms`](@ref)), the quadratic form keeps only the entries `(i, j)` where
neither mode is twisted, since a twisted mode's `(n_m - n_g,m)` is its twisted derivative as a whole.
"""
constant_term(spec::Spec) = constant_term(spec, Dict{Symbol,Any}())

function constant_term(spec::Spec, bases::AbstractDict)
    N = length(spec.modes)
    tw = _twisted(spec, bases)
    quad = _sum(spec.n_g[i] * spec.E_C[i, j] * spec.n_g[j] for i in 1:N, j in 1:N if !tw[i] && !tw[j])
    return spec.constant + 4 * quad
end
