"""
    differentiate(c::Coefficient, x::Symbol) -> Coefficient

The derivative `∂c/∂x` for real parameters, by symbolic differentiation of `c.ex`.
The result's expression is lightly simplified as it is built: zero terms, unit factors, `f^1`, `f^0` and `1^g` are removed, a zero factor makes a product zero, and an operation on numbers alone is evaluated where that does not change its value.
It is a constant coefficient (a number expression, zero when `c` does not depend on `x`) when it simplifies to a number.
`sign(f)` differentiates to zero, except that an `ArgumentError` is thrown when `f` depends on `x` and may be complex.
Second derivatives are two calls.
"""
function differentiate(c::Coefficient, x::Symbol)
    ex = _diff(c.ex, x)
    return _make(ex, free_symbols(ex))
end

_dzero(e) = e isa Number && iszero(e)
_done(e) = e isa Number && isone(e)

# `f(args...)` on numbers alone, or `nothing` when that evaluation would change the value or throws.
# Integer `+ - * ^` are evaluated exactly: a result that overflows `Int64` is an `Int128` or a `BigInt`, never wrapped.
# A non-finite result from finite operands, or a zero product, quotient or power of nonzero operands (an underflow), is not folded.
function _fold(f::Symbol, args)
    op = getfield(Base, f)
    v = try
        op(args...)
    catch
        return nothing
    end
    v isa Number || return nothing
    if all(a -> a isa Integer, args) && f in (:+, :-, :*, :^)
        exact = try
            op(map(big, args)...)
        catch
            return nothing
        end
        exact == v && return v
        return typemin(Int128) <= exact <= typemax(Int128) ? Int128(exact) : exact
    end
    all(isfinite, args) && !isfinite(v) && return nothing
    f in (:*, :/, ://, :^) && iszero(v) && !any(iszero, args) && return nothing
    return v
end

# The call `f(args...)`, simplified: the only place where simplification happens.
# The zero rules come first, so a zero factor or numerator is never lost to a fold.
function _build(f::Symbol, args...)
    f === :* && any(_dzero, args) && return 0
    (f === :/ || f === ://) && _dzero(args[1]) && return 0
    if all(a -> a isa Number, args)
        v = _fold(f, args)
        v === nothing || return v
        return Expr(:call, f, args...)
    end
    if f === :+
        ts = filter(!_dzero, collect(Any, args))
        isempty(ts) && return 0
        length(ts) == 1 && return ts[1]
        return Expr(:call, :+, ts...)
    elseif f === :- && length(args) == 2
        _dzero(args[2]) && return args[1]
        _dzero(args[1]) && return _build(:-, args[2])
    elseif f === :*
        fs = filter(!_done, collect(Any, args))
        isempty(fs) && return 1
        length(fs) == 1 && return fs[1]
        return Expr(:call, :*, fs...)
    elseif f === :/ || f === ://
        _done(args[2]) && return args[1]
    elseif f === :^
        _done(args[2]) && return args[1]
        (_dzero(args[2]) || _done(args[1])) && return 1
    end
    return Expr(:call, f, args...)
end

# `∂ex/∂x` as a simplified expression.
_diff(ex::Number, ::Symbol) = 0
_diff(ex::Symbol, x::Symbol) = ex === x ? 1 : 0
_diff(ex, ::Symbol) = throw(ArgumentError("cannot differentiate the node $(repr(ex)) of type $(typeof(ex))"))
function _diff(ex::Expr, x::Symbol)
    (ex.head === :call && ex.args[1] isa Symbol) || throw(ArgumentError("cannot differentiate the expression $ex"))
    f, args = ex.args[1], ex.args[2:end]
    ds = [_diff(a, x) for a in args]
    if f === :+
        return _build(:+, ds...)
    elseif f === :-
        return _build(:-, ds...)
    elseif f === :*
        terms = [_build(:*, (j == i ? ds[i] : args[j] for j in eachindex(args))...) for i in eachindex(args)]
        return _build(:+, terms...)
    elseif f === :/ || f === ://
        a, b = args
        da, db = ds
        _dzero(db) && return _build(:/, da, b)
        return _build(:-, _build(:/, da, b), _build(:/, _build(:*, a, db), _build(:^, b, 2)))
    elseif f === :^
        b, g = args
        db, dg = ds
        if _dzero(dg)
            return _build(:*, g, _build(:^, b, _build(:-, g, 1)), db)
        elseif _dzero(db)
            return _build(:*, ex, _build(:log, b), dg)
        else
            return _build(:*, ex, _build(:+, _build(:*, dg, _build(:log, b)), _build(:/, _build(:*, g, db), b)))
        end
    end
    length(args) == 1 || throw(ArgumentError("cannot differentiate the call $ex"))
    a, da = args[1], ds[1]
    if f === :sqrt
        return _build(:/, da, _build(:*, 2, ex))
    elseif f === :exp
        return _build(:*, ex, da)
    elseif f === :log
        return _build(:/, da, a)
    elseif f === :cos
        return _build(:*, _build(:-, _build(:sin, a)), da)
    elseif f === :sin
        return _build(:*, _build(:cos, a), da)
    elseif f === :tan
        return _build(:/, da, _build(:^, _build(:cos, a), 2))
    elseif f === :cosh
        return _build(:*, _build(:sinh, a), da)
    elseif f === :sinh
        return _build(:*, _build(:cosh, a), da)
    elseif f === :tanh
        return _build(:/, da, _build(:^, _build(:cosh, a), 2))
    elseif f === :cis
        return _build(:*, :im, ex, da)
    elseif f === :cispi
        return _build(:*, :im, :pi, ex, da)
    elseif f === :conj || f === :real || f === :imag
        return _build(f, da)
    elseif f === :abs
        return _build(:/, _build(:real, _build(:*, _build(:conj, a), da)), ex)
    elseif f === :sign
        !_dzero(da) && _maybe_complex(a) &&
            throw(ArgumentError("cannot differentiate $ex: the sign of a possibly complex value is not piecewise constant"))
        return 0
    end
    throw(ArgumentError("cannot differentiate the call $ex"))
end
