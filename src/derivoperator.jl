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
    node = mn.node
    children = mn.children
    length(dx) == length(children) ||
        throw(ArgumentError("$(length(dx)) entries of derivative data for a node with $(length(children)) children"))
    real = mn isa MaterializedNode{Float64}
    leaves = real ? _mode_leaves(node) : nothing
    axisterms = AxisTerm[]
    # One variant: `value` on the operators `ops` (`(axis, op, sub-key)`), complex or real-expanded.
    function push_variant!(value::Number, ops)
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
    for t in terms
        ops = _axis_ops(node, children, _factor_key(t))
        if :coefficients in parts
            dc = differentiate(t.coeff, x)
            if !iszero_expr(dc)
                v = dc(θ)
                iszero(v) || push_variant!(v, ops)
            end
        end
        if :ports in parts
            c = t.coeff(θ)
            iszero(c) && continue
            for (i, (a, _, Ka)) in enumerate(ops)
                dx[a] === nothing && continue
                replaced = copy(ops)
                replaced[i] = (a, _derivative_op(dx[a], Ka), Ka)
                push_variant!(c, replaced)
            end
        end
    end
    if :energies in parts
        for (a, d) in enumerate(dx)
            d === nothing && continue
            push!(axisterms, AxisTerm(1.0, Tuple{Int,LocalOp}[(a, DiagonalOp(d.energies))]))
        end
    end
    c0 = differentiate(constant, x)(θ)
    c0 = real ? Base.real(c0) : ComplexF64(c0)
    return KronFreeOperator(mn.dims, axisterms; constant=c0, on)
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
