# Materialization: evaluate each node's Hamiltonian at θ, diagonalize, fix phases, select
# the kept states and form the ports its ancestors need, children first.

"""
    MaterializedNode

A diagonalized node.
`dims` are the axis sizes of the node's product basis (a basis child's dimension, or a node
child's number of kept states); `vectors` is `prod(dims) × length(kept)` in that basis, with
`energies` the kept eigenvalues (absolute, GHz) and `kept` their indices among the computed
eigenpairs.  `ports[(mode, port)]` is the port operator in the kept eigenbasis.
`children` holds, in axis order, the child's `Basis` or `MaterializedNode`.
"""
struct MaterializedNode
    node::Node
    dims::Vector{Int}
    energies::Vector{Float64}
    vectors::Matrix{ComplexF64}
    kept::Vector{Int}
    ports::Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}
    children::Vector{Any}
end

"""
    System

A materialized tree: the root `MaterializedNode`, the `spec` it was built from (`nothing` when
built from a term list), the `terms` it was built from, and the parameters `θ` it was evaluated at.
"""
struct System
    root::MaterializedNode
    spec::Union{Nothing,Spec}
    terms::Vector{Term}
    θ::NamedTuple
end

const _EDGE_TOL = 1e-6

"""
    materialize(root, spec, θ; previous=nothing) -> System
    materialize(root, terms, θ; constant=Coefficient(0), previous=nothing, modetypes=Dict()) -> System

Diagonalize every node of the tree under `root` at parameters `θ`, children first.
The first form expands `spec` with [`circuit_terms`](@ref) and [`constant_term`](@ref) and
delegates to the second, which takes a user-built (filtered, transformed) term list and a
scalar `constant`.

The constant is added at the root only: the `energies` of every child node exclude it.
`previous`, a `System` of the same tree, supplies each node's previous solution (same tree
position) to the `:follow` phase convention and to selection rules.
A parameter missing from `θ` raises an `ArgumentError` naming it.

Each `mode => basis` leaf child is checked against the mode's type: a compact mode (`:c`) needs
a `ChargeBasis`, an extended (`:e`) or harmonic (`:h`) mode a `PhaseGrid` or `FockBasis`.
The spec form takes the types from the spec; the term-list form from `modetypes` (mode => type),
skipping modes absent from it (an empty dictionary skips the check).

A term routed to a branch node with two or more factors inside the same child would need a
composite (multi-mode) port of that child; these are not supported and throw an `ArgumentError`.
"""
function materialize(root::Node, terms::Vector{Term}, θ::NamedTuple; constant::Coefficient=Coefficient(0),
                     previous::Union{Nothing,System}=nothing,
                     modetypes::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    _check_bases(root, modetypes)
    routed = route(root, terms)
    _check_composite(routed)
    needed = required_ports(root, routed)
    c = constant(θ)
    prev = previous === nothing ? nothing : previous.root
    mroot = _materialize(root, θ, routed, needed, c, prev)
    return System(mroot, nothing, terms, θ)
end

function materialize(root::Node, spec::Spec, θ::NamedTuple; previous::Union{Nothing,System}=nothing)
    modetypes = Dict{Symbol,Symbol}(m.name => m.type for m in spec.modes)
    sys = materialize(root, circuit_terms(spec), θ; constant=constant_term(spec), previous=previous,
                      modetypes=modetypes)
    return System(sys.root, spec, sys.terms, sys.θ)
end

const _MODE_TYPE_NAMES = Dict(:c => "compact", :e => "extended", :h => "harmonic")

# Every `mode => basis` child against the mode's type (modes absent from `modetypes` are skipped).
function _check_bases(root::Node, modetypes::Dict{Symbol,Symbol})
    isempty(modetypes) && return nothing
    for node in _all_nodes(root), ch in node.children
        ch isa Pair || continue
        m, b = ch.first, ch.second
        t = get(modetypes, m, nothing)
        t === nothing && continue
        ok = t === :c ? b isa ChargeBasis : t in (:e, :h) && (b isa PhaseGrid || b isa FockBasis)
        ok && continue
        want = t === :c ? "a ChargeBasis" : t in (:e, :h) ? "a PhaseGrid or FockBasis" : "a known mode type (c, e or h)"
        throw(ArgumentError("mode :$m has type :$t ($(get(_MODE_TYPE_NAMES, t, "unknown"))), which needs " *
                            "$want, but is given $(repr(b))"))
    end
    return nothing
end

# A term routed to a branch with two or more factors inside one child node would need a port on
# several of that child's modes at once (a composite port); plan 1 does not form them.
function _check_composite(routed)
    for (node, ts) in routed, t in ts
        axes_ = [axis_of(node, f.mode) for f in t.factors]
        for a in unique(axes_)
            count(==(a), axes_) > 1 || continue
            ch = node.children[a]
            ch isa Node ||
                throw(ArgumentError("a term on modes $(modes(t)) has several factors on mode :$(ch.first)"))
            throw(ArgumentError("a term on modes $(modes(t)) has several factors inside the child node " *
                                "with modes $(modes(ch)); composite (multi-mode) ports are not yet supported"))
        end
    end
end

_childdim(b::Basis) = dimension(b)
_childdim(mn::MaterializedNode) = length(mn.kept)

_childmask(b::Basis) = Vector{Bool}(positive_mask(b))
_childmask(mn::MaterializedNode) = fill(true, length(mn.kept))

_childop(b::Basis, ::Symbol, p::Port) = localop(b, p)
function _childop(mn::MaterializedNode, mode::Symbol, p::Port)
    haskey(mn.ports, (mode, p)) ||
        throw(ArgumentError("child node lacks port $(_portname(p)) on mode :$mode"))
    return DenseOp(mn.ports[(mode, p)])
end

# The previous solution of child `i`, or `nothing`.
function _prev_child(prev, i::Int)
    prev === nothing && return nothing
    ch = prev.children[i]
    return ch isa MaterializedNode ? ch : nothing
end

# Kronecker product of per-axis vectors, axis 1 fastest.
_kronaxes(vs) = foldl((acc, v) -> kron(v, acc), vs; init=[true])

# ⟨v_i| O_axis |v_j⟩ for the columns of V, with O acting on axis `axis` of the product basis.
function _port_matrix(V::Matrix{ComplexF64}, dims::Vector{Int}, axis::Int, op::LocalOp)
    X = reshape(V, dims..., size(V, 2))
    Y = similar(X)
    apply_axis!(Y, op, X, axis)
    return V' * reshape(Y, size(V))
end

# Per column of V, the 2-norm of its entries on the first and last `w` points of axis `axis`.
function _edge_support(V::AbstractMatrix, dims::Vector{Int}, axis::Int, w::Int)
    N = dims[axis]
    X = reshape(V, prod(dims[1:axis-1]), N, prod(dims[axis+1:end]), size(V, 2))
    edge = [i for i in 1:N if i <= w || i > N - w]
    return [sqrt(sum(abs2, view(X, :, edge, :, j))) for j in axes(V, 2)]
end

function _materialize(node::Node, θ::NamedTuple, routed, needed, constant, prev)
    children = Any[]
    for (i, ch) in enumerate(node.children)
        push!(children, ch isa Node ?
              _materialize(ch, θ, routed, needed, 0.0, _prev_child(prev, i)) : ch.second)
    end
    dims = Int[_childdim(c) for c in children]
    # Only the :follow phase convention and the follow rule (whose previous vectors take precedence
    # over an explicit reference) consume `prev` here; elsewhere a changed child kept count must not
    # make `previous` unusable for the whole tree.
    uses_prev = node.phase === :follow || node.keep isa Follow
    prev === nothing || !uses_prev || prev.dims == dims ||
        throw(ArgumentError("previous solution at this tree position has dims $(prev.dims), expected $dims"))

    # (2)-(3) The node's operator.
    terms = AxisTerm[]
    for t in routed[node]
        fs = Tuple{Int,LocalOp}[]
        for f in t.factors
            a = axis_of(node, f.mode)
            push!(fs, (a, _childop(children[a], f.mode, f.port)))
        end
        sort!(fs; by=first)          # one factor per axis: checked by _check_composite
        push!(terms, AxisTerm(t.coeff(θ), fs))
    end
    for (a, c) in enumerate(children)
        c isa MaterializedNode && push!(terms, AxisTerm(1.0, Tuple{Int,LocalOp}[(a, DiagonalOp(c.energies))]))
    end
    H = KronFreeOperator(dims, terms; constant=constant)

    # (4) Diagonalize.
    m = node.compute == 0 ? prod(dims) : node.compute
    E, V = solve(node.solver, H, m)

    # (5)-(7) Ports the selection rule inspects, on all computed vectors; select; slice.
    # Selection rules see only moduli (port elements, overlaps), so phases are fixed on the kept
    # vectors only (fixing them on every computed vector would cost a pass over all of them and
    # warn for high states with no weight on the mask), and the selection ports are rotated to match.
    ports = Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}()
    for (mode, p) in ports_needed(node.keep)
        a = axis_of(node, mode)
        ports[(mode, p)] = _port_matrix(V, dims, a, _childop(children[a], mode, p))
    end
    idx = validate_selection(select(node.keep, SelectionContext(E, V, ports, prev)), size(V, 2), node.keep)
    E, V = E[idx], V[:, idx]

    # :follow makes each overlap with the previous kept vector real and positive; with no previous
    # solution, :positive_sector is used instead.
    mask = _kronaxes(_childmask(c) for c in children)
    V0 = copy(V)
    if node.phase === :follow && prev !== nothing
        size(prev.vectors) == size(V) ||
            throw(ArgumentError("phase :follow: the previous solution keeps $(size(prev.vectors, 2)) states, this one $(size(V, 2))"))
        fix_phases!(V, :follow, mask, prev.vectors)
    else
        fix_phases!(V, :positive_sector, mask, nothing)
    end
    ph = [dot(view(V0, :, j), view(V, :, j)) for j in axes(V, 2)]    # V[:, j] = ph[j] V0[:, j]
    for k in keys(ports)
        ports[k] = conj.(ph) .* ports[k][idx, idx] .* transpose(ph)
    end

    # (8) The remaining required ports, on the kept vectors.
    for (mode, p) in needed[node]
        haskey(ports, (mode, p)) && continue
        a = axis_of(node, mode)
        ports[(mode, p)] = _port_matrix(V, dims, a, _childop(children[a], mode, p))
    end

    # (9) Grid edge support of the kept states.
    for (a, c) in enumerate(children)
        c isa PhaseGrid || continue
        s = _edge_support(V, dims, a, c.order ÷ 2)
        bad = findall(>(_EDGE_TOL), s)
        isempty(bad) || @warn "PhaseGrid on mode :$(node.children[a].first) is too small: kept states $bad have edge support up to $(maximum(s)) > $_EDGE_TOL; enlarge the grid"
    end

    return MaterializedNode(node, dims, E, V, collect(Int, idx), ports, children)
end

"""
    materialized(sys, node) -> MaterializedNode

The materialization of `node` (a node of the tree `sys` was built from).
"""
function materialized(sys::System, node::Node)
    stack = MaterializedNode[sys.root]
    while !isempty(stack)
        mn = pop!(stack)
        mn.node === node && return mn
        append!(stack, (c for c in mn.children if c isa MaterializedNode))
    end
    throw(ArgumentError("node is not part of this system's tree"))
end

"Kept energies of `node` (default: the root), absolute, GHz."
energies(sys::System, node::Node=sys.root.node) = materialized(sys, node).energies

"Kept eigenvectors of `node` in its product basis, one column per kept state."
eigenvectors(sys::System, node::Node=sys.root.node) = materialized(sys, node).vectors

"""
    port(sys, mode, p, node=root) -> Matrix{ComplexF64}

The port `p` of `mode` in the kept eigenbasis of `node`.
"""
function port(sys::System, mode::Symbol, p::Port, node::Node=sys.root.node)
    mn = materialized(sys, node)
    haskey(mn.ports, (mode, p)) && return mn.ports[(mode, p)]
    avail = join(("(:$m, $(_portname(q)))" for (m, q) in sort!(collect(keys(mn.ports)); by=string)), ", ")
    throw(ArgumentError("port $(_portname(p)) on mode :$mode was not formed for this node; available: " *
                        (isempty(avail) ? "none" : avail)))
end
