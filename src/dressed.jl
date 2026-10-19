# Dressed and bare states: scqubits-style lookup between a node's kept eigenstates and the
# product states of its children's kept states (the node's own eigenvector basis).

"""
    bare_labels(sys, node=root; depth=1) -> Vector{Tuple}

The bare label of every product state of `node`, in product-basis order (axis 1 fastest): the
tuple of child indices.
With `depth > 1`, a node child's index is replaced by that child's dominant bare label,
computed with `depth - 1`; basis children keep their basis index.
"""
function bare_labels(sys::System, node::Node=sys.root.node; depth::Int=1)
    depth >= 1 || throw(ArgumentError("depth must be at least 1"))
    return _bare_labels(sys, materialized(sys, node), depth)
end

function _bare_labels(sys::System, mn::MaterializedNode, depth::Int)
    base = [Tuple(Int(i) for i in Tuple(c)) for c in CartesianIndices(Tuple(mn.dims))]
    depth == 1 && return Tuple[b for b in base]
    # dominant label of each kept state of each node child, at depth - 1
    dominant = Dict{Int,Vector{Tuple}}()
    for (a, ch) in enumerate(mn.children)
        ch isa MaterializedNode || continue
        labels = _bare_labels(sys, ch, depth - 1)
        dominant[a] = Tuple[labels[argmax(abs2.(view(ch.vectors, :, i)))] for i in 1:length(ch.kept)]
    end
    return Tuple[Tuple(haskey(dominant, a) ? dominant[a][i] : i for (a, i) in enumerate(b)) for b in base]
end

"""
    bare_overlaps(sys, node=root) -> Matrix{Float64}

`length(kept) × prod(dims)` matrix of `|⟨product state p | dressed state i⟩|²`.
"""
bare_overlaps(sys::System, node::Node=sys.root.node) = abs2.(permutedims(materialized(sys, node).vectors))

"""
    decompose(sys, node, i; thresh=1e-3) -> Vector{Pair{Tuple,T}}

Bare label => amplitude of kept state `i` on every product state with weight above `thresh`,
sorted by descending weight.
The amplitudes have the element type `T` of the node's stored eigenvectors (`Float64` for a node in real arithmetic).
"""
function decompose(sys::System, node::Node, i::Int; thresh::Real=1e-3)
    mn = materialized(sys, node)
    1 <= i <= length(mn.kept) || throw(ArgumentError("state index $i out of range 1:$(length(mn.kept))"))
    return _decompose(mn, i, thresh)
end
decompose(sys::System, i::Int; kwargs...) = decompose(sys, sys.root.node, i; kwargs...)

function _decompose(mn::MaterializedNode, i::Int, thresh::Real)
    v = view(mn.vectors, :, i)
    cart = CartesianIndices(Tuple(mn.dims))
    ps = [p for p in eachindex(v) if abs2(v[p]) > thresh]
    sort!(ps; by=p -> -abs2(v[p]))
    return Pair{Tuple,eltype(v)}[Tuple(Int(k) for k in Tuple(cart[p])) => v[p] for p in ps]
end

"""
    dressed_index(sys, node, label) -> Union{Int,Nothing}

Index of the kept state with the largest overlap with the product state `label`; warns when
that overlap is below 0.5, and returns `nothing` when the product state has (numerically) no
weight on the kept states.
"""
function dressed_index(sys::System, node::Node, label::Tuple)
    mn = materialized(sys, node)
    length(label) == length(mn.dims) && all(l -> l isa Integer, label) &&
        all(1 .<= collect(label) .<= mn.dims) ||
        throw(ArgumentError("label $label is not a valid product state for dims $(Tuple(mn.dims))"))
    p = LinearIndices(Tuple(mn.dims))[label...]
    w = abs2.(view(mn.vectors, p, :))
    sum(w) < 1e-12 && return nothing
    w_max, i = findmax(w)
    w_max < 0.5 && @warn "dressed state $i has overlap only $w_max with the product state $label"
    return i
end
dressed_index(sys::System, label::Tuple) = dressed_index(sys, sys.root.node, label)
