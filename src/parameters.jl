using RuntimeGeneratedFunctions

const _CONSTANT_SYMBOLS = (:pi, :π, :im, :ℯ, :e)

"""
    free_symbols(ex) -> Vector{Symbol}

Symbols in value position of `ex`, in first-appearance order without duplicates.
Call heads (function names) and the constants `pi, π, im, ℯ, e` are skipped.
"""
function free_symbols(ex)
    out = Symbol[]
    _collect_symbols!(out, ex)
    return out
end

function _collect_symbols!(out::Vector{Symbol}, ex)
    if ex isa Symbol
        (ex in _CONSTANT_SYMBOLS || ex in out) || push!(out, ex)
    elseif ex isa Expr
        args = ex.head === :call ? ex.args[2:end] : ex.args
        for a in args
            _collect_symbols!(out, a)
        end
    end
    return out
end

"""
    Coefficient

A compiled scalar expression of named circuit parameters.
`ex` is the expression, `deps` the parameters it uses, and `f(θ::NamedTuple)` its `ComplexF64` evaluation.
"""
struct Coefficient
    ex::Union{Expr,Symbol,Number}
    deps::Vector{Symbol}
    f::Function
end

function _compile(ex, deps::Vector{Symbol})
    assigns = [:($d = θ.$d) for d in deps]
    fex = :(θ -> begin
        $(assigns...)
        ComplexF64($ex)
    end)
    return @RuntimeGeneratedFunction(fex)
end

_make(ex, deps::Vector{Symbol}) = Coefficient(ex, deps, _compile(ex, deps))

function Coefficient(src::AbstractString, params::AbstractVector{Symbol})
    ex = Meta.parse(src)
    deps = free_symbols(ex)
    for d in deps
        d in params || throw(ArgumentError("symbol :$d in \"$src\" is not a declared parameter"))
    end
    return _make(ex, deps)
end

Coefficient(x::Number) = _make(x, Symbol[])

function (c::Coefficient)(θ::NamedTuple)
    for d in c.deps
        haskey(θ, d) || throw(ArgumentError("parameter :$d missing from θ"))
    end
    return c.f(θ)::ComplexF64
end

_union_deps(a::Coefficient, b::Coefficient) = union(a.deps, b.deps)

Base.:*(x::Number, c::Coefficient) = _make(:($x * $(c.ex)), copy(c.deps))
Base.:*(a::Coefficient, b::Coefficient) = _make(:($(a.ex) * $(b.ex)), _union_deps(a, b))
Base.:+(a::Coefficient, b::Coefficient) = _make(:($(a.ex) + $(b.ex)), _union_deps(a, b))
Base.:-(c::Coefficient) = _make(:(-($(c.ex))), copy(c.deps))

"""
    phase_factor(offset::Coefficient, sign::Int) -> Coefficient

The coefficient `exp(sign * im * offset)`.
"""
phase_factor(offset::Coefficient, sign::Int) =
    _make(:(exp($(sign) * im * $(offset.ex))), copy(offset.deps))

iszero_expr(c::Coefficient) = c.ex isa Number && iszero(c.ex)
