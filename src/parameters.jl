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

# The argument is gensym'd so that any parameter name (e.g. `θ`) is safe, and the result type is
# interpolated as a value so no parameter can shadow it.
function _compile(ex, deps::Vector{Symbol})
    arg = gensym(:θ)
    assigns = [:($d = $arg.$d) for d in deps]
    fex = :($arg -> begin
        $(assigns...)
        $(ComplexF64)($ex)
    end)
    return @RuntimeGeneratedFunction(fex)
end

_make(ex, deps::Vector{Symbol}) = Coefficient(ex, deps, _compile(ex, deps))

# Functions a coefficient expression may call, and the named constants it may use.
const _ALLOWED_CALLS = (:+, :-, :*, :/, :^, ://, :sqrt, :exp, :cos, :sin, :tan, :cosh, :sinh, :tanh,
                       :log, :abs, :cis, :cispi, :sign, :conj, :real, :imag)
const _ALLOWED_CONSTANTS = (:pi, :π, :im, :ℯ)

"""
    check_parameter_names(params)

Throw an `ArgumentError` if a parameter name is reserved: one of the constants `pi, π, im, ℯ, e`
or a function name allowed in coefficient expressions.
"""
function check_parameter_names(params)
    for p in params
        p in _CONSTANT_SYMBOLS && throw(ArgumentError("parameter name :$p is reserved for a constant"))
        p in _ALLOWED_CALLS && throw(ArgumentError("parameter name :$p is reserved for a function"))
    end
    return nothing
end

# Accept only numeric literals, symbols, and calls of allowlisted functions on such expressions.
function _check_expr(ex, src::AbstractString)
    bad(what) = throw(ArgumentError("coefficient \"$src\": $what is not allowed"))
    if ex isa Number
        return nothing
    elseif ex isa Symbol
        ex in _CONSTANT_SYMBOLS && !(ex in _ALLOWED_CONSTANTS) && bad("the symbol :$ex")
        return nothing
    elseif ex isa Expr
        ex.head === :call || bad("expression head :$(ex.head)")
        f = ex.args[1]
        (f isa Symbol && f in _ALLOWED_CALLS) || bad("calling $(repr(f))")
        for a in @view ex.args[2:end]
            _check_expr(a, src)
        end
        return nothing
    else
        bad("the literal $(repr(ex))")
    end
end

"""
    Coefficient(src::AbstractString, params::Vector{Symbol})
    Coefficient(x::Number)

Compile the expression `src` of the declared `params`.
`src` may use numeric literals, the parameters, the constants `pi, π, im, ℯ`, and calls of
`+ - * / ^ // sqrt exp cos sin tan cosh sinh tanh log abs cis cispi sign conj real imag`;
anything else (blocks, assignments, macros, string literals, other functions) throws an `ArgumentError`.
"""
function Coefficient(src::AbstractString, params::AbstractVector{Symbol})
    isempty(strip(src)) && throw(ArgumentError("coefficient expression is empty"))
    check_parameter_names(params)
    ex = Meta.parse(src; raise=false)
    _check_expr(ex, src)
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

"""
    twist_factor(ng::Coefficient, sign::Int) -> Coefficient

The twist phase `e^{sign · 2πi n_g}` of a `PeriodicGrid` wrap corner, built with `cispi` so that it
is exactly real (±1, zero imaginary part) at `n_g ∈ {0, ½}`.
"""
twist_factor(ng::Coefficient, sign::Int) =
    _make(:(cispi($(2 * sign) * $(ng.ex))), copy(ng.deps))

iszero_expr(c::Coefficient) = c.ex isa Number && iszero(c.ex)

free_symbols(c::Coefficient) = free_symbols(c.ex)
