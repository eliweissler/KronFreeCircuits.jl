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

# SymPy's Julia printer emits the dotted operators; for scalars they mean the plain operators.
# Normalize them at parse time so everything downstream sees only scalar call heads (and integer literals, below).
const _DOTTED_OPS = Dict(:.+ => :+, :.- => :-, :.* => :*, :./ => :/, :.^ => :^)

# Meta.parse turns an integer literal beyond typemax(Int64) into a macrocall of @int128_str or @big_str (e.g. the
# SI-valued constants of an exported circuit); fold exactly those, with a decimal-integer string, into the integer.
const _BIG_INT_MACROS = Dict(Symbol("@int128_str") => Int128, Symbol("@big_str") => BigInt)

function _big_int_literal(ex::Expr)
    (ex.head === :macrocall && length(ex.args) == 3) || return nothing
    m, s = ex.args[1], ex.args[3]
    name = m isa GlobalRef ? m.name : m
    (name isa Symbol && haskey(_BIG_INT_MACROS, name)) || return nothing
    (s isa String && occursin(r"^[+-]?[0-9]+$", s)) || return nothing   # the parser folds a leading sign in
    return parse(_BIG_INT_MACROS[name], s)
end

_normalize_dots(ex) = ex
function _normalize_dots(ex::Expr)
    n = _big_int_literal(ex)
    n === nothing || return n
    ex.head === :call || return ex
    f = ex.args[1]
    f = get(_DOTTED_OPS, f, f)
    return Expr(:call, f, map(_normalize_dots, ex.args[2:end])...)
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
the dotted operators `.+ .- .* ./ .^` mean the scalar operators;
integer literals beyond `typemax(Int64)` are `Int128` or `BigInt` values; anything else (blocks, assignments, macros, string literals, other functions) throws an `ArgumentError`.
"""
function Coefficient(src::AbstractString, params::AbstractVector{Symbol})
    isempty(strip(src)) && throw(ArgumentError("coefficient expression is empty"))
    check_parameter_names(params)
    ex = _normalize_dots(Meta.parse(src; raise=false))
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
Exported for hand-written terms on a twisted mode's charge (the README's periodic-grid section).
"""
twist_factor(ng::Coefficient, sign::Int) =
    _make(:(cispi($(2 * sign) * $(ng.ex))), copy(ng.deps))

iszero_expr(c::Coefficient) = c.ex isa Number && iszero(c.ex)

free_symbols(c::Coefficient) = free_symbols(c.ex)
