# The derivative of a node's operator along one parameter, and two-sided ports `W† (⊗ₐ Oₐ) V`.

"""
    NodeDerivative{T}

The derivative data of one node along one parameter (first order) or one pair of parameters (second order); it mirrors
[`MaterializedNode`](@ref), with `T` the node's element type.
`energies` are the derivatives of the kept energies, `vectors` those of the kept vectors (`prod(dims) × nkept`).
`gauge[j]` is `Im⟨ψ_j|∂ψ_j⟩` of kept state `j` (zero on a real node).
`ports` and `joint_ports` have the keys and element types of the node's own, each the derivative of that port.
`peak_amplitudes` is the derivative of the node's `peak_amplitudes`.
`children` holds, per axis, `nothing` (a basis, or a child that does not depend on the parameter) or the child's
`NodeDerivative`.
"""
struct NodeDerivative{T<:Union{Float64,ComplexF64}}
    energies::Vector{Float64}
    vectors::Matrix{T}
    gauge::Vector{Float64}
    ports::Dict{Tuple{Symbol,Port},NodeMatrix}
    joint_ports::Dict{_FactorKey,NodeMatrix}
    peak_amplitudes::Matrix{T}
    children::Vector{Any}
end

# The child's derivative of its port (a one-factor sub-key `Ka`) or joint port (several factors) as an operator.
function _derivative_op(d::NodeDerivative, Ka::_FactorKey)
    if length(Ka) == 1
        haskey(d.ports, only(Ka)) ||
            error("internal error: the child's derivative data lacks the port $(_key_string(Ka))")
        return DenseOp(d.ports[only(Ka)])
    end
    haskey(d.joint_ports, Ka) || error("internal error: the child's derivative data lacks the joint port $(_key_string(Ka))")
    return DenseOp(d.joint_ports[Ka])
end

"""
    _derivative_operator(mn, terms, θ, x, dx; constant=Coefficient(0), on=Array, parts=(:coefficients, :ports, :energies))

`∂H/∂x` of the node `mn` on its product basis at `θ`, in the node's arithmetic (`Float64` when `mn` is real).
`terms` are the terms acting at this node, `dx[a]` the derivative data of the child on axis `a` (a `NodeDerivative`) or
`nothing`, and `constant` the root's scalar (zero below the root).
Three contributions, each selected by `parts`: the coefficients' derivatives on the routed terms (`:coefficients`); for
each term and each of its axes whose child has derivative data, the term with that child's port or joint port replaced
by its derivative (`:ports`); the children's energy derivatives on the diagonal (`:energies`).
On a real node each variant is the real expansion of `Re(value ⊗ ops)`, with nothing merged across terms.
The operator is built from the terms themselves, never from the forward operator, whose real form drops the terms
whose coefficient vanishes at `θ`.
"""
function _derivative_operator(mn::MaterializedNode, terms::Vector{Term}, θ::NamedTuple, x::Symbol, dx::Vector{Any};
                              constant::Coefficient=Coefficient(0), on=Array,
                              parts=(:coefficients, :ports, :energies))
    _check_child_data(mn, dx)
    axisterms = AxisTerm[]
    push_variant! = _variant_pusher(mn, axisterms)
    for t in terms
        ops = _axis_ops(mn.node, mn.children, _factor_key(t))
        if :coefficients in parts
            dc = differentiate(t.coeff, x)
            iszero_expr(dc) || push_variant!(dc(θ), ops)
        end
        if :ports in parts
            c = t.coeff(θ)
            iszero(c) && continue
            for set in _first_replacements(_op_axes(ops), dx)
                push_variant!(c, _replace_ops(ops, set))
            end
        end
    end
    :energies in parts && _push_energies!(axisterms, dx)
    return _operator_with_constant(mn, axisterms, differentiate(constant, x)(θ), on)
end

"""
    _derivative_operator(mn, terms, θ, x, λ, dx, dλ, dxλ; constant=Coefficient(0), on=Array, parts=(:coefficients, :ports, :energies))

`∂²H/∂x∂λ` of the node `mn` on its product basis at `θ` (`x == λ` allowed), in the node's arithmetic.
`dx[a]`, `dλ[a]` are the first-order derivative data of the child on axis `a` along `x` and `λ`, `dxλ[a]` its
second-order data along the pair (each a `NodeDerivative` or `nothing`), and `constant` the root's scalar.
By the product rule, for a term with coefficient `c` and child operators `ops`: `c_xλ` on `ops` (`:coefficients`);
`c_x` on `ops` with one entry replaced by the child's `λ`-derivative and `c_λ` with one replaced by its `x`-derivative,
`c` with one entry replaced by the child's second derivative, and `c` with two distinct entries replaced by the
`x`- and the `λ`-derivative (every ordered pair) (`:ports`); the children's second energy derivatives on the
diagonal (`:energies`).
With `x == λ` (and `dx`, `dλ` the same data) the cross terms appear twice, which is the factor 2 of the product rule.
A variant whose coefficient value is zero is left out; on a real node each variant is real-expanded as at first order.
"""
function _derivative_operator(mn::MaterializedNode, terms::Vector{Term}, θ::NamedTuple, x::Symbol, λ::Symbol,
                              dx::Vector{Any}, dλ::Vector{Any}, dxλ::Vector{Any};
                              constant::Coefficient=Coefficient(0), on=Array,
                              parts=(:coefficients, :ports, :energies))
    foreach(d -> _check_child_data(mn, d), (dx, dλ, dxλ))
    axisterms = AxisTerm[]
    push_variant! = _variant_pusher(mn, axisterms)
    for t in terms
        ops = _axis_ops(mn.node, mn.children, _factor_key(t))
        cx, cλ = differentiate(t.coeff, x), differentiate(t.coeff, λ)
        if :coefficients in parts && !iszero_expr(cx)
            cxλ = differentiate(cx, λ)
            iszero_expr(cxλ) || push_variant!(cxλ(θ), ops)
        end
        :ports in parts || continue
        axes = _op_axes(ops)
        if !iszero_expr(cx)
            v = cx(θ)
            for set in _first_replacements(axes, dλ)
                push_variant!(v, _replace_ops(ops, set))
            end
        end
        if !iszero_expr(cλ)
            v = cλ(θ)
            for set in _first_replacements(axes, dx)
                push_variant!(v, _replace_ops(ops, set))
            end
        end
        c = t.coeff(θ)
        for set in _second_replacements(axes, dx, dλ, dxλ)
            push_variant!(c, _replace_ops(ops, set))
        end
    end
    :energies in parts && _push_energies!(axisterms, dxλ)
    return _operator_with_constant(mn, axisterms, differentiate(differentiate(constant, x), λ)(θ), on)
end

function _check_child_data(mn::MaterializedNode, d::Vector{Any})
    length(d) == length(mn.children) ||
        throw(ArgumentError("$(length(d)) entries of derivative data for a node with $(length(mn.children)) children"))
    return nothing
end

# A function `push_variant!(value, ops)` that appends `value` on the operators `ops` (`(axis, op, sub-key)`) to
# `axisterms`, complex or, on a real node, real-expanded; a zero `value` is left out.
function _variant_pusher(mn::MaterializedNode, axisterms::Vector{AxisTerm})
    real = mn isa MaterializedNode{Float64}
    leaves = real ? _mode_leaves(mn.node) : nothing
    return function (value::Number, ops)
        iszero(value) && return nothing
        if real
            fs = Tuple{Int,LocalOp,Symbol,Int}[(a, op, _key_character(leaves, Ka).kind, 1) for (a, op, Ka) in ops]
            for r in _real_expansion(value, fs)
                push!(axisterms, AxisTerm(r.coeff, r.factors))
            end
        else
            push!(axisterms, AxisTerm(ComplexF64(value), Tuple{Int,LocalOp}[(a, op) for (a, op, _) in ops]))
        end
        return nothing
    end
end

# The children's energy derivatives in `d` (per axis a `NodeDerivative` or `nothing`) on the diagonal.
function _push_energies!(axisterms::Vector{AxisTerm}, d::Vector{Any})
    for (a, da) in enumerate(d)
        da === nothing && continue
        push!(axisterms, AxisTerm(1.0, Tuple{Int,LocalOp}[(a, DiagonalOp(da.energies))]))
    end
    return axisterms
end

function _operator_with_constant(mn::MaterializedNode, axisterms::Vector{AxisTerm}, c0::Number, on)
    c0 = mn isa MaterializedNode{Float64} ? Base.real(c0) : ComplexF64(c0)
    return KronFreeOperator(mn.dims, axisterms; constant=c0, on)
end

# The axis of each entry of `ops` (as `_axis_ops` returns them).
_op_axes(ops) = Int[a for (a, _, _) in ops]

# The replacement sets of the product rule, each a vector of `(entry, data)`: `entry` indexes `axes` (the axis of each
# entry), `data` is the `NodeDerivative` whose quantity replaces that entry's.
# First order: one entry replaced by its axis's data in `d`, for every entry whose axis has data.
_first_replacements(axes::Vector{Int}, d::Vector{Any}) =
    [[(g, d[a])] for (g, a) in enumerate(axes) if d[a] !== nothing]
# Second order: one entry replaced by its axis's data in `dxλ`, and two distinct entries `g ≠ h` (every ordered pair)
# replaced by the data of `dx` and of `dλ`.
function _second_replacements(axes::Vector{Int}, dx::Vector{Any}, dλ::Vector{Any}, dxλ::Vector{Any})
    sets = [[(g, dxλ[a])] for (g, a) in enumerate(axes) if dxλ[a] !== nothing]
    for (g, a) in enumerate(axes), (h, b) in enumerate(axes)
        g != h && dx[a] !== nothing && dλ[b] !== nothing && push!(sets, [(g, dx[a]), (h, dλ[b])])
    end
    return sets
end

# `ops` with the entries of the replacement `set` replaced by the child's derivative of its port or joint port.
function _replace_ops(ops, set)
    replaced = copy(ops)
    for (g, d) in set
        a, _, Ka = ops[g]
        replaced[g] = (a, _derivative_op(d, Ka), Ka)
    end
    return replaced
end

"""
    _sandwich(mn, W, V, K, ops) -> NodeMatrix

`W† (⊗ₐ Oₐ) V` on the product basis of the node `mn`, for the factor key `K` at `mn` (one factor or several), in the
key's natural type there: `Matrix{Float64}` for a real-character key of a real node, else `Matrix{ComplexF64}`.
`ops` has the form `_axis_ops(mn.node, mn.children, K)` returns, one `(axis, op, sub-key)` per child the key touches; a
caller replaces an entry's `op` by `DenseOp` of a child's derivative to differentiate the operator.
With `W = V = mn.vectors` it is the stored port or joint port of `K`.
On a real node `W` and `V` are real, and the real and imaginary components are formed separately (`_real_sandwich`, which also forms a real node's joint ports).
"""
function _sandwich(mn::MaterializedNode, W::AbstractMatrix, V::AbstractMatrix, K::_FactorKey,
                   ops::Vector{Tuple{Int,LocalOp,_FactorKey}})
    if !(mn isa MaterializedNode{Float64})
        cops = Tuple{Int,LocalOp}[(a, op isa DenseOp ? DenseOp(convert(Matrix{ComplexF64}, op.M)) : op) for (a, op, _) in ops]
        return convert(Matrix{ComplexF64}, W)' * _apply_ops(convert(Matrix{ComplexF64}, V), mn.dims, cops)
    end
    return _real_sandwich(_mode_leaves(mn.node), mn.dims, convert(Matrix{Float64}, W), convert(Matrix{Float64}, V), K, ops)
end
