# Writing a `Spec` back out as kronfree-spec version 1 JSON.
# Coefficient expressions are printed in the format's scalar grammar: explicit `*`, every binary operation parenthesized
# (so no precedence or associativity is relied upon), negative literals parenthesized, `π`/`pi` as `pi`.

const _BINARY_OPS = (:+, :-, :*, :/, :^, ://)

_number_string(x::Integer) = x < 0 ? "($x)" : string(x)
function _number_string(x::Real)
    isfinite(x) || throw(ArgumentError("cannot write the non-finite number $x in a spec expression"))
    s = repr(Float64(x))                    # shortest string that round-trips
    return startswith(s, "-") ? "($s)" : s
end

"""
    expr_string(ex) -> String

The text of the coefficient expression `ex` (a `Coefficient`'s `ex`) in the `kronfree-spec` scalar grammar.
`parse_spec` of the result gives back the same function.
"""
expr_string(x::Number) = x isa Complex ? throw(ArgumentError("cannot write the complex number $x")) : _number_string(x)
function expr_string(s::Symbol)
    s in (:pi, :π) && return "pi"
    s === :ℯ && return "ℯ"
    s in (:e, :im) && throw(ArgumentError("cannot write the symbol :$s in a spec expression"))
    return String(s)
end
function expr_string(ex::Expr)
    ex.head === :call || throw(ArgumentError("cannot write an expression with head :$(ex.head)"))
    f, args = ex.args[1], ex.args[2:end]
    f isa Symbol || throw(ArgumentError("cannot write a call of $(repr(f))"))
    strs = map(expr_string, args)
    if f in _BINARY_OPS && length(args) >= 2        # n-ary `+` and `*` print as a flat parenthesized chain
        return "(" * join(strs, " $f ") * ")"
    elseif f in (:+, :-) && length(args) == 1       # unary sign
        return "($f" * strs[1] * ")"
    end
    return "$f(" * join(strs, ", ") * ")"
end

# An omega or kappa entry: an integer where it is integral (compact modes require that), otherwise a float.
_json_number(x::Float64) = isinteger(x) && abs(x) < 2.0^53 ? Int(x) : x

"""
    spec_json(spec::Spec; metadata=Dict{String,Any}()) -> String

The `kronfree-spec` version 1 JSON text of `spec`, indented; `parse_spec` of it gives a spec with the same functions.
`metadata` entries are appended after the spec's own keys; every key must start with `_` (the format ignores those),
otherwise an `ArgumentError` is thrown.
"""
function spec_json(spec::Spec; metadata::AbstractDict{<:AbstractString}=Dict{String,Any}())
    for k in keys(metadata)
        startswith(k, "_") || throw(ArgumentError("metadata key \"$k\" must start with \"_\""))
    end
    matrix(M) = [[expr_string(M[i, j].ex) for j in axes(M, 2)] for i in axes(M, 1)]
    vector(v) = [expr_string(c.ex) for c in v]
    pairs_ = Pair{Symbol,Any}[
        :format => "kronfree-spec",
        :version => 1,
        :units => "GHz",
        :parameters => String.(spec.parameters),
        :modes => [(name=String(m.name), type=String(m.type)) for m in spec.modes],
        :E_C => matrix(spec.E_C),
        :E_L => matrix(spec.E_L),
        :flux_linear => vector(spec.flux_linear),
        :n_g => vector(spec.n_g),
        :cosines => [(energy=expr_string(c.energy.ex), omega=_json_number.(c.omega), offset=expr_string(c.offset.ex))
                     for c in spec.cosines],
    ]
    isempty(spec.charge_cosines) ||
        push!(pairs_, :charge_cosines => [(energy=expr_string(c.energy.ex), kappa=_json_number.(c.kappa),
                                           offset=expr_string(c.offset.ex)) for c in spec.charge_cosines])
    push!(pairs_, :constant => expr_string(spec.constant.ex))
    for k in sort!(collect(String, keys(metadata)))
        push!(pairs_, Symbol(k) => metadata[k])
    end
    obj = NamedTuple{Tuple(first.(pairs_))}(Tuple(last.(pairs_)))
    io = IOBuffer()
    JSON3.pretty(io, JSON3.write(obj))              # `pretty` prints; it returns nothing
    return String(take!(io))
end

"""
    save_spec(path::AbstractString, spec::Spec; metadata=Dict{String,Any}()) -> path

Write `spec` to `path` as `kronfree-spec` version 1 JSON (see [`spec_json`](@ref)); `load_spec(path)` reads it back.
"""
function save_spec(path::AbstractString, spec::Spec; metadata::AbstractDict{<:AbstractString}=Dict{String,Any}())
    write(path, spec_json(spec; metadata), "\n")
    return path
end
