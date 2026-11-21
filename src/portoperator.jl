# A port as an operator on a node's product basis, applied matrix-free: the drive operators of time evolution in the
# product basis (the QuantumToolbox extension), where `port` gives the same operator in the node's kept eigenbasis.

"""
    port_operator(sys, key, node=root; on=Array) -> KronFreeOperator

The port `key = (mode, p)`, or the joint port `key = [(mode, p), ...]` of factors on distinct modes, as an operator on
the product basis of `node` (its children's kept states, axis 1 fastest), identity on the axes the key does not touch.
Each factor is its child's operator: the child node's stored port, or the basis operator of a leaf mode.
It is real exactly when every factor's operator is real (a real child's real-character port), else complex.
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
    ops = Tuple{Int,LocalOp}[(a, op) for (a, op, _) in _axis_ops(node, mn.children, K)]
    return KronFreeOperator(mn.dims, [AxisTerm(1.0, ops)]; on)
end
