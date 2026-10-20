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
