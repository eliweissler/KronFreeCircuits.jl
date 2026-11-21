# A port as an operator on a node's product basis, applied matrix-free: the drive operators of time evolution in the
# product basis (the QuantumToolbox extension), where `port` gives the same operator in the node's kept eigenbasis.

"""
    port_operator(sys, key, node=root; on=Array) -> KronFreeOperator

The port `key = (mode, p)`, or the joint port `key = [(mode, p), ...]` of factors on distinct modes, as an operator on
the product basis of `node` (its children's kept states, axis 1 fastest), identity on the axes the key does not touch.
Each factor is its child's operator: the child node's stored port, or the basis operator of a leaf mode.
It is real exactly when every factor's operator is real (a real child's real-character port), else complex, and
Hermitian (`ishermitian`) exactly when every factor is, to rounding (`n`, `φ` and their squares; not `e^{iφ}` or
`e^{2πin·k}`, k ≠ 0).
`on` places it as a solver's `on` does (`CuArray` needs `using CUDA, cuTENSOR`).

A factor whose port the child node did not form throws an `ArgumentError` (list the port in that child's `ports`); so do
a mode outside `node` and an augmented node (its stored states are not on its product basis).
"""
function port_operator(sys::System, key::Tuple{Symbol,Port}, node::Node=sys.root.node; on=Array)
    return port_operator(sys, Tuple{Symbol,Port}[key], node; on)
end

function port_operator(sys::System, key::AbstractVector{<:Tuple{Symbol,Port}}, node::Node=sys.root.node; on=Array)
    isempty(key) && throw(ArgumentError("port_operator: the key is empty"))
    any(a -> a.node === node, sys.augmented) &&
        throw(ArgumentError("the node on modes $(modes(node)) is augmented: its stored states are not on its product " *
                            "basis, so it has no port operator there"))
    inside = Set(modes(node))
    for (m, _) in key
        m in inside || throw(ArgumentError("mode :$m is not a mode of the node on modes $(modes(node))"))
    end
    K = _sortkey!(Tuple{Symbol,Port}[k for k in key])
    length(unique(first.(K))) == length(K) ||
        throw(ArgumentError("port_operator: the factors of a joint key must be on distinct modes, got $(_key_string(K))"))
    mn = materialized(sys, node)
    _check_child_ports(node, mn, K)
    ops = Tuple{Int,LocalOp}[(a, op) for (a, op, _) in _axis_ops(node, mn.children, K)]
    hermitian = all(((_, op),) -> (M = sparse(op); isapprox(M, M'; rtol=1e-10, atol=1e-12 * max(1, norm(M, Inf)))), ops)
    return KronFreeOperator(mn.dims, [AxisTerm(1.0, ops)]; on, hermitian)
end

# An ArgumentError, listing what is available, for a factor group of `K` whose port (one factor) or joint port (several)
# its child node did not form; leaf bases and given systems check their own operators.
function _check_child_ports(node::Node, mn::MaterializedNode, K::_FactorKey)
    for (a, Ka) in _axis_groups(node, K)
        ch = mn.children[a]
        ch isa MaterializedNode && !(ch.node isa GivenSystem) || continue
        formed = length(Ka) == 1 ? haskey(ch.ports, Ka[1]) : haskey(ch.joint_ports, Ka)
        formed && continue
        avail = vcat(sort!([_key_string(Tuple{Symbol,Port}[k]) for k in keys(ch.ports)]),
                     sort!([_key_string(k) for k in keys(ch.joint_ports)]))
        throw(ArgumentError("child node on modes $(_child_modes(ch.node)) lacks port $(_key_string(Ka)) " *
                            "(list it in that node's `ports`); available: " * (isempty(avail) ? "none" : join(avail, ", "))))
    end
    return nothing
end
