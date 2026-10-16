abstract type Solver end

"Full dense diagonalization."
struct Dense <: Solver end

"""
    Lanczos(; tol=1e-10, krylovdim=0, maxiter=100, verbosity=0)

Iterative solver; `krylovdim == 0` lets the solver choose.
`maxiter` bounds the restarts and `verbosity` is passed to KrylovKit (0 is silent).
"""
struct Lanczos <: Solver
    tol::Float64
    krylovdim::Int
    maxiter::Int
    verbosity::Int
end
Lanczos(; tol=1e-10, krylovdim=0, maxiter=100, verbosity=0) = Lanczos(tol, krylovdim, maxiter, verbosity)

abstract type SelectionRule end

"Keep the `k` lowest eigenstates (`k == 0`: keep all computed)."
struct Lowest <: SelectionRule
    k::Int
end
lowest(k::Int) = Lowest(k)

ports_needed(::SelectionRule) = Tuple{Symbol,Port}[]

"""
    Node(children; solver=Dense(), compute=0, keep=lowest(0), phase=:positive_sector, ports=[])

A hierarchical unit.
Each child is a `mode => basis` pair or another `Node`.
`compute == 0` means all eigenpairs (only valid with `Dense`).
"""
mutable struct Node
    children::Vector{Any}
    solver::Solver
    compute::Int
    keep::SelectionRule
    phase::Symbol
    ports::Vector{Tuple{Symbol,Port}}

    function Node(children::AbstractVector; solver::Solver=Dense(), compute::Int=0,
                  keep::SelectionRule=lowest(0), phase::Symbol=:positive_sector,
                  ports=Tuple{Symbol,Port}[])
        isempty(children) && throw(ArgumentError("a Node needs at least one child"))
        for ch in children
            ok = ch isa Node || (ch isa Pair && ch.first isa Symbol && ch.second isa Basis)
            ok || throw(ArgumentError("each child must be a `mode => basis` pair or a Node, got $(repr(ch))"))
        end
        phase in (:positive_sector, :follow) ||
            throw(ArgumentError("phase must be :positive_sector or :follow, got :$phase"))
        compute >= 0 || throw(ArgumentError("compute must be nonnegative"))
        compute == 0 && !(solver isa Dense) &&
            throw(ArgumentError("compute == 0 (all eigenpairs) requires the Dense solver"))
        node = new(Any[ch for ch in children], solver, compute, keep, phase,
                   Tuple{Symbol,Port}[(m, p) for (m, p) in ports])
        ms = modes(node)
        for (i, m) in enumerate(ms)
            m in view(ms, 1:i-1) && throw(ArgumentError("mode :$m appears more than once in the tree"))
        end
        return node
    end
end

isleaf(node::Node) = all(ch -> ch isa Pair, node.children)

_child_modes(ch::Pair) = Symbol[ch.first]
_child_modes(ch::Node) = modes(ch)

"""
    modes(node::Node) -> Vector{Symbol}

All modes beneath `node`, in axis order.
"""
modes(node::Node) = reduce(vcat, (_child_modes(ch) for ch in node.children); init=Symbol[])

"""
    axis_of(node, mode) -> Int

Index of the child of `node` whose subtree contains `mode`.
"""
function axis_of(node::Node, mode::Symbol)
    for (i, ch) in enumerate(node.children)
        mode in _child_modes(ch) && return i
    end
    throw(ArgumentError("mode :$mode is not in this node"))
end

_subnodes(node::Node) = (ch for ch in node.children if ch isa Node)

function _all_nodes(root::Node)
    out = Node[]
    stack = Node[root]
    while !isempty(stack)
        n = pop!(stack)
        push!(out, n)
        append!(stack, reverse!(collect(_subnodes(n))))
    end
    return out
end

"""
    route(root, terms) -> IdDict{Node,Vector{Term}}

Assign each term to the lowest node containing all its modes (factor-free terms go to the root).
"""
function route(root::Node, terms::Vector{Term})
    treemodes = Set(modes(root))
    seen = Set{Symbol}()
    routed = IdDict{Node,Vector{Term}}(n => Term[] for n in _all_nodes(root))
    for t in terms
        for m in modes(t)
            m in treemodes ||
                throw(ArgumentError("mode :$m appears in a term but is not in the tree"))
            push!(seen, m)
        end
        node = root
        while true
            ms = modes(t)
            next = nothing
            for ch in _subnodes(node)
                if all(in(modes(ch)), ms)
                    next = ch
                    break
                end
            end
            (next === nothing || isempty(ms)) && break
            node = next
        end
        push!(routed[node], t)
    end
    for m in modes(root)
        m in seen || throw(ArgumentError("mode :$m is in the tree but in no term of the spec"))
    end
    return routed
end

"""
    required_ports(root, routed) -> IdDict{Node,Vector{Tuple{Symbol,Port}}}

The `(mode, port)` operators each node must expose to its ancestors.
"""
function required_ports(root::Node, routed::IdDict)
    out = IdDict{Node,Vector{Tuple{Symbol,Port}}}()
    function visit(node::Node, ancestors::Vector{Node})
        req = Tuple{Symbol,Port}[]
        if !isempty(ancestors)
            inside = Set(modes(node))
            for a in ancestors, t in get(routed, a, Term[]), f in t.factors
                f.mode in inside && push!(req, (f.mode, f.port))
            end
        end
        append!(req, ports_needed(node.keep))
        append!(req, node.ports)
        out[node] = unique!(req)
        for ch in _subnodes(node)
            visit(ch, vcat(ancestors, node))
        end
    end
    visit(root, Node[])
    return out
end
