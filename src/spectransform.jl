# Transforming a `Spec`: replacing parameters by values.

# `ex` with every symbol of `values` replaced by its Float64 value, and every subtree that has no free symbols left folded
# to one Float64 literal.
# Bare numbers and constants (`pi`, `ℯ`, `im`) are left as written; only a call is folded.
# Folding evaluates the subtree as a ComplexF64, so it follows the arithmetic of `Coefficient` itself.
_substitute_ex(x::Number, values) = x
_substitute_ex(s::Symbol, values) = haskey(values, s) ? Float64(values[s]) : s
function _substitute_ex(ex::Expr, values)
    ex.head === :call || throw(ArgumentError("cannot substitute into an expression with head :$(ex.head)"))
    new = Expr(:call, ex.args[1], (_substitute_ex(a, values) for a in ex.args[2:end])...)
    isempty(free_symbols(new)) || return new
    z = try
        _make(new, Symbol[])(NamedTuple())
    catch e
        e isa DomainError && throw(ArgumentError("cannot fold the constant subexpression $(expr_string(new)): $(e.msg)"))
        rethrow()
    end
    iszero(imag(z)) ||
        throw(ArgumentError("the constant subexpression $(expr_string(new)) is not real (it evaluates to $z)"))
    isfinite(real(z)) ||
        throw(ArgumentError("the constant subexpression $(expr_string(new)) is not finite (it evaluates to $(real(z)))"))
    return real(z)
end

function _check_values(values::NamedTuple)
    for (k, v) in pairs(values)
        v isa Real && isfinite(v) || throw(ArgumentError("the value of $k must be a finite real number, got $(repr(v))"))
    end
    return nothing
end

"""
    substitute(c::Coefficient, values::NamedTuple) -> Coefficient

`c` with every symbol in `keys(values)` replaced by its `Float64` value, and every subexpression left without free symbols folded to one `Float64` literal.
`deps` is recomputed.
Values must be finite `Real`s, else an `ArgumentError` is thrown.
Throws `ArgumentError` if a folded subexpression is not real (nonzero imaginary part) or not finite.
"""
function substitute(c::Coefficient, values::NamedTuple)
    _check_values(values)
    ex = _substitute_ex(c.ex, values)
    return _make(ex, free_symbols(ex))
end

"""
    substitute(spec::Spec, values::NamedTuple) -> Spec

`spec` with each parameter in `keys(values)` replaced by its value in every coefficient (see the `Coefficient` method).
The substituted names are removed from `parameters`, and cosines and charge cosines whose `energy` vanishes
(structurally, by `_iszero`: a zero literal, a product with a zero factor, a sum of zeros) are removed.
Values must be finite `Real`s and every name must be a parameter of `spec`, otherwise an `ArgumentError` is thrown.
"""
function substitute(spec::Spec, values::NamedTuple)
    _check_values(values)
    for k in keys(values)
        k in spec.parameters || throw(ArgumentError("substitute: :$k is not a parameter of the spec"))
    end
    sub(c::Coefficient) = substitute(c, values)
    keep(cs) = filter(c -> !_iszero(c.energy), cs)
    cosines = [CosineSpec(sub(c.energy), c.omega, sub(c.offset)) for c in spec.cosines]
    charge_cosines = [ChargeCosineSpec(sub(c.energy), c.kappa, sub(c.offset)) for c in spec.charge_cosines]
    return Spec(filter(p -> p ∉ keys(values), spec.parameters), spec.modes, sub.(spec.E_C), sub.(spec.E_L),
                sub.(spec.flux_linear), sub.(spec.n_g), keep(cosines), keep(charge_cosines), sub(spec.constant))
end

# Promoting a mode of a `Spec` to a parameter (design Section 4.1).

"""
    PromotedSpec

The result of [`promote_mode`](@ref).
`fast` is the spec of the remaining modes, whose parameters are those of `source` followed by `parameter`, the name that holds the slow coordinate.
`mode` is the slow mode (name and type), and `E_C` and `n_g` are the source's `E_C[s,s]` and `n_g[s]` for the slow stage.
"""
struct PromotedSpec
    fast::Spec
    source::Spec
    mode::ModeSpec
    parameter::Symbol
    E_C::Coefficient
    n_g::Coefficient
end

# `a + b`, with a structurally zero summand left out.
_add(a::Coefficient, b::Coefficient) = _iszero(b) ? a : _iszero(a) ? b : a + b

"""
    promote_mode(spec::Spec, mode::Symbol; parameter::Symbol=mode) -> PromotedSpec

Fix the coordinate of `mode` (the slow mode `s`) at the value of a new parameter `parameter`, giving a spec over the remaining (fast) modes.
Each part of `spec` carries over exactly:
`E_C[s,s]` and `n_g[s]` are dropped from the fast spec (an `@info` says so; they are in the result for the slow stage);
`E_L[s,s]` becomes the constant `½ E_L[s,s] p²`;
`E_L[s,f]` adds `E_L[s,f] p` to `flux_linear[f]`;
`flux_linear[s]` becomes the constant `flux_linear[s] p`;
a cosine with `omega[s] ≠ 0` keeps its other multipliers and its `offset` gains `omega[s] p`, or, if it has no other multiplier, becomes the constant `-energy cos(omega[s] p + offset)`.
Constants accumulate onto `spec.constant`, and `parameter` is appended to the parameters.
The offset-charge constant `4 n_gᵀ E_C n_g` loses its `(s,s)` part with `E_C[s,s]` and `n_g[s]`.

Throws `ArgumentError` if `mode` is not a mode, `spec` has only one mode, `E_C[s,f]` is not (structurally) zero for some fast mode `f`, a charge cosine has `kappa[s] ≠ 0`, or `parameter` is already a parameter, the name of a fast mode, or a reserved name.
The checks look at the coefficient expressions, never at parameter values: remove a coupling that vanishes only at particular values first with [`substitute`](@ref).
"""
function promote_mode(spec::Spec, mode::Symbol; parameter::Symbol=mode)
    names = [m.name for m in spec.modes]
    s = findfirst(==(mode), names)
    s === nothing && throw(ArgumentError("promote_mode: :$mode is not a mode of the spec (modes: $(join(":" .* string.(names), ", ")))"))
    length(names) > 1 ||
        throw(ArgumentError("promote_mode: mode :$mode is the only mode, so nothing is left to be fast"))
    fast = [i for i in eachindex(names) if i != s]
    parameter in spec.parameters &&
        throw(ArgumentError("promote_mode: the new parameter :$parameter is already a parameter of the spec"))
    parameter in names[fast] &&
        throw(ArgumentError("promote_mode: the new parameter :$parameter is the name of a remaining mode"))
    try
        check_parameter_names([parameter])
    catch e
        e isa ArgumentError && throw(ArgumentError("promote_mode: $(e.msg)"))
        rethrow()
    end
    for f in fast, (i, j) in ((s, f), (f, s))
        _iszero(spec.E_C[i, j]) ||
            throw(ArgumentError("promote_mode: E_C[$(names[i]),$(names[j])] = $(expr_string(spec.E_C[i, j].ex)) couples the slow mode " *
                                ":$mode to the fast mode :$(names[f]) through the charge; a charge coupling has no exact carry-over " *
                                "(substitute the parameters that make it vanish first)"))
    end
    for (k, c) in enumerate(spec.charge_cosines)
        c.kappa[s] == 0 ||
            throw(ArgumentError("promote_mode: charge_cosines[$k] has kappa[$(names[s])] = $(c.kappa[s]) ≠ 0, so it is not diagonal in " *
                                "the coordinate of the slow mode :$mode; it has no exact carry-over"))
    end

    p = _make(parameter, [parameter])
    drop_s(v) = v[fast]
    constant = spec.constant
    constant = _add(constant, 0.5 * spec.E_L[s, s] * p * p)
    constant = _add(constant, spec.flux_linear[s] * p)
    flux_linear = copy(spec.flux_linear)
    for f in fast
        flux_linear[f] = _add(flux_linear[f], 0.5 * (spec.E_L[s, f] + spec.E_L[f, s]) * p)
    end
    cosines = CosineSpec[]
    for c in spec.cosines
        ω = c.omega[s]
        rest = drop_s(c.omega)
        if ω == 0
            push!(cosines, CosineSpec(c.energy, rest, c.offset))
            continue
        end
        offset = _make(:($(c.offset.ex) + $ω * $parameter), union(c.offset.deps, [parameter]))
        if all(iszero, rest)
            constant = _add(constant, _make(:(-($(c.energy.ex)) * cos($(offset.ex))), union(c.energy.deps, offset.deps)))
        else
            push!(cosines, CosineSpec(c.energy, rest, offset))
        end
    end
    charge_cosines = [ChargeCosineSpec(c.energy, drop_s(c.kappa), c.offset) for c in spec.charge_cosines]

    @info "promote_mode: E_C[$mode,$mode] and n_g[$mode] are dropped from the fast spec; they are kept in the PromotedSpec for the slow stage"
    fastspec = Spec([spec.parameters; parameter], spec.modes[fast], spec.E_C[fast, fast], spec.E_L[fast, fast], flux_linear[fast],
                    spec.n_g[fast], cosines, charge_cosines, constant)
    return PromotedSpec(fastspec, spec, spec.modes[s], parameter, spec.E_C[s, s], spec.n_g[s])
end
