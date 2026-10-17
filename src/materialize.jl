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
A [`GivenSystem`](@ref) child is materialized as a `MaterializedNode` whose `node` is the given
system, with identity `vectors` and `peak_amplitudes`, all states kept, its operators as the ports
`(mode, Port(kind, 0))`, and `children = Any[g]`.

`peak_component[j]` is the linear index, in the node's product basis, of kept state `j`'s peak
point: at a leaf (all children bases) the positive-sector argmax used by `:positive_sector`, at a
branch the dominant coefficient `argmax |c|` (restricted to the positive sector on basis-child
axes), whose physical point is resolved recursively through the children's `peak_component` (a
basis child's index is its own point).
`peak_amplitudes[p, j]` is the exact full-space amplitude of kept state `j` at the peak point of
kept state `p`; a leaf's is `vectors[peak_component, :]`.
"""
struct MaterializedNode
    node::Union{Node,GivenSystem}
    dims::Vector{Int}
    energies::Vector{Float64}
    vectors::Matrix{ComplexF64}
    kept::Vector{Int}
    ports::Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}
    children::Vector{Any}
    peak_component::Vector{Int}
    peak_amplitudes::Matrix{ComplexF64}
end

"""
    System

A materialized tree: the root `MaterializedNode`, the `spec` it was built from (`nothing` when
built from a term list), the `terms` it was built from, the parameters `θ` it was evaluated at,
and the scalar `constant` added at the root.
"""
struct System
    root::MaterializedNode
    spec::Union{Nothing,Spec}
    terms::Vector{Term}
    θ::NamedTuple
    constant::Coefficient
end

const _EDGE_TOL = 1e-6
const _HYBRID_RATIO = 0.5

"""
    materialize(root, spec, θ; previous=nothing, extra_terms=Term[]) -> System
    materialize(root, terms, θ; constant=Coefficient(0), previous=nothing, modetypes=Dict()) -> System

Diagonalize every node of the tree under `root` at parameters `θ`, children first.
The first form expands `spec` with [`circuit_terms`](@ref) and [`constant_term`](@ref),
appends `extra_terms` (terms not in the circuit spec, such as couplings to a
[`GivenSystem`](@ref)), and delegates to the second, which takes a user-built (filtered,
transformed) term list and a scalar `constant`.

The constant is added at the root only: the `energies` of every child node exclude it.
`previous`, a `System` of the same tree, supplies each node's previous solution (same tree
position) to the `:follow` phase convention, to selection rules, and to `LOBPCG` nodes as their
initial block.
That block is the previous kept vectors (their raw coefficients in the node's product basis),
padded on the right with random columns to `compute` or truncated to it; a node whose `dims`
changed starts cold.
Only the kept vectors survive in a `MaterializedNode`, so when `compute > keep` the remaining
columns start random and the block is only partially warm (retaining the whole computed block is
future work).
The `follow` rule and the `:follow` phase compare physical overlaps: the previous kept vectors are
first transported into the current product basis through the overlaps of the previous and current
child kept states, computed recursively (the identity on basis axes).
This needs, at every such node and below it, the same product dims and equal bases on basis axes;
otherwise an `ArgumentError` is thrown (overlaps across different primitive bases are not
implemented, and equal dimensions alone do not establish the same basis).
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
    return System(mroot, nothing, terms, θ, constant)
end

function materialize(root::Node, spec::Spec, θ::NamedTuple; previous::Union{Nothing,System}=nothing,
                     extra_terms::Vector{Term}=Term[])
    modetypes = Dict{Symbol,Symbol}(m.name => m.type for m in spec.modes)
    sys = materialize(root, vcat(circuit_terms(spec), extra_terms), θ; constant=constant_term(spec),
                      previous=previous, modetypes=modetypes)
    return System(sys.root, spec, sys.terms, sys.θ, sys.constant)
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
                throw(ArgumentError("a term on modes $(modes(t)) has several factors on mode :$(only(_child_modes(ch)))"))
            throw(ArgumentError("a term on modes $(modes(t)) has several factors inside the child node " *
                                "with modes $(modes(ch)); composite (multi-mode) ports are not yet supported"))
        end
    end
end

_childdim(b::Basis) = dimension(b)
_childdim(mn::MaterializedNode) = length(mn.kept)

_childmask(b::Basis) = Vector{Bool}(positive_mask(b))
_childmask(mn::MaterializedNode) = fill(true, length(mn.kept))

# The peak-amplitude matrix a branch contracts with on a child's axis: a node child's stored
# `peak_amplitudes`, or for a basis child the identity (its peak point is the basis index itself).
_childamps(b::Basis) = Diagonal(ones(ComplexF64, dimension(b)))
_childamps(mn::MaterializedNode) = mn.peak_amplitudes

_childop(b::Basis, ::Symbol, p::Port) = localop(b, p)
function _childop(mn::MaterializedNode, mode::Symbol, p::Port)
    mn.node isa GivenSystem && return _givenop(mn, mode, p)
    haskey(mn.ports, (mode, p)) ||
        throw(ArgumentError("child node lacks port $(_portname(p)) on mode :$mode"))
    return DenseOp(mn.ports[(mode, p)])
end

# A given child's operator of kind `p.kind` (its operators are exponent-free: `p.k == 0`).
function _givenop(mn::MaterializedNode, mode::Symbol, p::Port)
    p.k == 0 || throw(ArgumentError("given system :$mode: port $(_portname(p)) has a nonzero exponent k = $(p.k); " *
                                    "given operators are looked up by kind alone and need k == 0"))
    haskey(mn.ports, (mode, p)) ||
        throw(ArgumentError("given system :$mode provides no operator of kind :$(p.kind); available: " *
                            join((":$(q.kind)" for (_, q) in sort!(collect(keys(mn.ports)); by=string)), ", ")))
    return DenseOp(mn.ports[(mode, p)])
end

# A given child as a node: identity eigenvectors and peak amplitudes, all states kept, its energies
# on the parent's axis (through `_node_operator`) and its operators as exponent-free ports.
function _materialize_given(g::GivenSystem)
    k = dimension(g)
    ports = Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}((g.mode, Port(kind, 0)) => M for (kind, M) in g.operators)
    return MaterializedNode(g, [k], g.energies, Matrix{ComplexF64}(I, k, k), collect(1:k), ports, Any[g],
                            collect(1:k), Matrix{ComplexF64}(I, k, k))
end

# The previous solution of child `i`, or `nothing`.
function _prev_child(prev, i::Int)
    prev === nothing && return nothing
    ch = prev.children[i]
    return ch isa MaterializedNode ? ch : nothing
end

# Whether a previous and a current basis-axis child span the same basis: `==` on the immutable basis
# structs (range, points, order, cutoff, `phi_zpf`); for the axis of a given system's own
# materialization (its `children = Any[g]`), the same mode and dimension, since its labels are the user's.
_same_basis(a::Basis, b::Basis) = a == b
_same_basis(a::GivenSystem, b::GivenSystem) = a.mode == b.mode && length(a.energies) == length(b.energies)

# Check that `prev` and the current `children` (with product dims `dims`) can be compared through
# physical overlaps: the same dims, a node on the same axes, and the same basis (`_same_basis`) on the
# basis axes.
function _check_transportable(prev::MaterializedNode, children, dims)
    prev.dims == dims ||
        throw(ArgumentError("physical overlap with the previous solution: its product dims $(prev.dims) differ " *
                            "from the current $dims; transport across changed retained dimensions is not supported"))
    for (a, (p, c)) in enumerate(zip(prev.children, children))
        if (p isa Basis && c isa Basis) || (p isa GivenSystem && c isa GivenSystem)
            _same_basis(p, c) || throw(ArgumentError("physical overlap with the previous solution: axis $a has basis $(repr(p)) " *
                                          "previously and $(repr(c)) now; physical overlaps across different primitive " *
                                          "bases are not implemented"))
        elseif !(p isa MaterializedNode && c isa MaterializedNode)
            throw(ArgumentError("physical overlap with the previous solution: axis $a holds a $(nameof(typeof(p))) " *
                                "previously and a $(nameof(typeof(c))) now"))
        end
    end
    return nothing
end

"""
    _transport_previous(prev::MaterializedNode, children, dims) -> Matrix

The previous kept vectors expressed in the current product basis, `R̃ = (⊗_a S_a') prev.vectors`
with `S_a = _child_overlap(prev.children[a], children[a])` on node axes and the identity on basis
axes, so that `R̃' * V` are physical overlaps `⟨previous kept i | current j⟩` (exact: the children's
discarded states do not enter an overlap with a current state).
Requires the checks of `_check_transportable`; `prev.vectors` is never written.
"""
function _transport_previous(prev::MaterializedNode, children, dims)
    _check_transportable(prev, children, dims)
    X = copy(prev.vectors)
    nodeaxes = [a for a in eachindex(children) if children[a] isa MaterializedNode]
    isempty(nodeaxes) && return X
    # apply_axis! forbids aliasing: alternate between two distinct work arrays, one application per axis.
    W = similar(X)
    shape = (dims..., size(X, 2))
    for a in nodeaxes
        S = _child_overlap(prev.children[a], children[a])
        apply_axis!(reshape(W, shape), DenseOp(Matrix(S')), reshape(X, shape), a)
        X, W = W, X
    end
    return X
end

"""
    _child_overlap(prev::MaterializedNode, cur::MaterializedNode) -> Matrix

The physical overlaps `⟨prev kept i | cur kept j⟩`: at a leaf `prev.vectors' * cur.vectors`, at a
branch `_transport_previous(prev, cur.children, cur.dims)' * cur.vectors`.
Throws an `ArgumentError` when the dims or a basis child differ anywhere below.
"""
_child_overlap(prev::MaterializedNode, cur::MaterializedNode) =
    _transport_previous(prev, cur.children, cur.dims)' * cur.vectors

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

# The array storage a solver's block lives on.
storage(::Solver) = Array
storage(s::LOBPCG) = s.on

# How many of the lowest eigenpairs the node's solver returns.
_compute_count(node::Node, dims) = node.compute == 0 ? prod(dims) : node.compute

# (2)-(3) The node's operator at θ: its routed `terms` on the children's local operators (a basis
# child's, or a node child's ports), each node child's kept energies on its axis, and `constant`.
# Scratch is reserved for the solver's full block (LOBPCG: the computed pairs and its guard
# columns), on the solver's storage.
function _node_operator(node::Node, children, terms::Vector{Term}, θ::NamedTuple, constant)
    dims = Int[_childdim(c) for c in children]
    axisterms = AxisTerm[]
    for t in terms
        fs = Tuple{Int,LocalOp}[]
        for f in t.factors
            a = axis_of(node, f.mode)
            push!(fs, (a, _childop(children[a], f.mode, f.port)))
        end
        sort!(fs; by=first)          # one factor per axis: checked by _check_composite
        push!(axisterms, AxisTerm(t.coeff(θ), fs))
    end
    for (a, c) in enumerate(children)
        c isa MaterializedNode && push!(axisterms, AxisTerm(1.0, Tuple{Int,LocalOp}[(a, DiagonalOp(c.energies))]))
    end
    m = _compute_count(node, dims)
    capacity = node.solver isa LOBPCG ? m + node.solver.guard : 1
    return KronFreeOperator(dims, axisterms; constant=constant, block_capacity=capacity, on=storage(node.solver))
end

# The initial block of an LOBPCG node: `nothing` without a previous solution of the same `dims`,
# else a copy of its kept vectors (raw coefficients, in kept order) truncated to `m` columns or
# padded on the right with `randn` columns (never zeros, which the solver rejects) of element type `T`.
# Truncation follows the kept order, not energy order (a previous kept with a reordered `indices`
# rule contributes its first columns); the solver's guard columns recover any lower states.
_initial_block(prev, dims, m::Int) =
    _initial_block(prev, dims, m, prev === nothing ? Nothing : eltype(prev.vectors))
function _initial_block(prev, dims, m::Int, ::Type{T}) where {T}
    (prev === nothing || prev.dims != dims) && return nothing
    k = min(size(prev.vectors, 2), m)
    X = Matrix{T}(undef, size(prev.vectors, 1), m)
    X[:, 1:k] .= view(prev.vectors, :, 1:k)
    randn!(view(X, :, k+1:m))
    return X
end

# Diagonalize: only LOBPCG takes an initial block, built from the previous solution.
_solve(s::Solver, H, m, prev, dims) = solve(s, H, m)
_solve(s::LOBPCG, H, m, prev, dims) = solve(s, H, m; X0=_initial_block(prev, dims, m, eltype(H)))

function _materialize(node::Node, θ::NamedTuple, routed, needed, constant, prev)
    children = Any[]
    for (i, ch) in enumerate(node.children)
        push!(children, ch isa Node ? _materialize(ch, θ, routed, needed, 0.0, _prev_child(prev, i)) :
                        ch isa GivenSystem ? _materialize_given(ch) : ch.second)
    end
    dims = Int[_childdim(c) for c in children]
    # Only the :follow phase convention and the follow rule (whose previous vectors take precedence
    # over an explicit reference) consume `prev` here; elsewhere a changed child kept count must not
    # make `previous` unusable for the whole tree.
    uses_prev = node.phase === :follow || node.keep isa Follow
    prev === nothing || !uses_prev || prev.dims == dims ||
        throw(ArgumentError("previous solution at this tree position has dims $(prev.dims), expected $dims"))
    # The previous kept vectors in the current product basis, so that the follow rule and the :follow
    # phase compare physical overlaps when the children were re-diagonalized (this also checks that the
    # basis children, here and below, are unchanged).  The LOBPCG initial block stays raw.
    reference = prev === nothing || !uses_prev ? nothing : _transport_previous(prev, children, dims)

    # (2)-(3) The node's operator.
    H = _node_operator(node, children, routed[node], θ, constant)

    # (4) Diagonalize.
    _dense_too_large(node.solver, prod(dims)) &&
        @warn("Dense node on modes $(modes(node)) has product dimension $(prod(dims)) > $_DENSE_WARN_DIM; " *
              (all(c -> c isa Basis, children) ?
                   "this is a leaf: Lanczos for a few states, or LOBPCG only with a warm start; see README" :
                   "this is a branch: LOBPCG is advised"), _id=(:kronfree_dense_large, objectid(node)), maxlog=1)
    m = _compute_count(node, dims)
    E, V = _solve(node.solver, H, m, prev, dims)

    # (5)-(7) Ports the selection rule inspects, on all computed vectors; select; slice.
    # Selection rules see only moduli (port elements, overlaps), so phases are fixed on the kept
    # vectors only (fixing them on every computed vector would cost a pass over all of them and
    # warn for high states with no weight on the mask), and the selection ports are rotated to match.
    ports = Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}()
    for (mode, p) in ports_needed(node.keep)
        a = axis_of(node, mode)
        ports[(mode, p)] = _port_matrix(V, dims, a, _childop(children[a], mode, p))
    end
    idx = validate_selection(select(node.keep, SelectionContext(E, V, ports, prev, reference)), size(V, 2), node.keep)
    E, V = E[idx], V[:, idx]

    # :follow makes each physical overlap with the previous kept vector real and positive; with no previous
    # solution, :positive_sector is used instead.  :positive_sector makes, at a leaf, the largest
    # entry on the positive sector real and positive; at a branch, the exact full-space amplitude at
    # the joint peak of the dominant product component (`branch_peak_phase`), the argmax restricted
    # to the positive sector on basis-child axes only (node-child masks are all true).
    # Peaks are taken on the unphased vectors, as fix_phases! and branch_peak_phase do.
    mask = _kronaxes(_childmask(c) for c in children)
    isleaf = all(c -> c isa Basis, children)
    amps = isleaf ? AbstractMatrix{ComplexF64}[] : AbstractMatrix{ComplexF64}[_childamps(c) for c in children]
    V0 = copy(V)
    sector = findall(mask)
    peaks = [_sector_peak(view(V0, :, j), sector) for j in axes(V0, 2)]
    peak = Int[first(pk) for pk in peaks]
    if node.phase === :follow && prev !== nothing
        size(prev.vectors) == size(V) ||
            throw(ArgumentError("phase :follow: the previous solution keeps $(size(prev.vectors, 2)) states, this one $(size(V, 2))"))
        fix_phases!(V, :follow, mask, reference)
    elseif isleaf
        fix_phases!(V, :positive_sector, mask, nothing)
    else
        for j in axes(V, 2)
            C = reshape(view(V0, :, j), dims...)
            last(peaks[j]) && @warn "Branch node on modes $(modes(node)): kept state $j has no weight on the " *
                                    "phase mask; using its largest entry" maxlog=10
            α = _joint_peak_amplitude(C, amps, peak[j])          # = branch_peak_phase(C, amps; mask=mask)
            d = _peak_denominator(C, amps, peak[j])
            if !isfinite(d) || d < 1e-12 * norm(C)
                @warn "Branch node on modes $(modes(node)): kept state $j: its dominant product component's own " *
                      "peak amplitude vanishes ($d), so its phase is fixed by an amplitude of $(abs(α))" maxlog=10
            elseif abs(α) / d < _HYBRID_RATIO
                @warn "Branch node on modes $(modes(node)): kept state $j has joint-peak amplitude $(abs(α)), " *
                      "ratio $(abs(α) / d) < $_HYBRID_RATIO to its dominant product component's (heavy " *
                      "hybridization); its phase is fixed by a small amplitude" maxlog=10
            end
            view(V, :, j) .*= exp(-im * angle(α))
        end
    end
    ph = [dot(view(V0, :, j), view(V, :, j)) for j in axes(V, 2)]    # V[:, j] = ph[j] V0[:, j]
    for k in keys(ports)
        ports[k] = conj.(ph) .* ports[k][idx, idx] .* transpose(ph)
    end

    # Exact full-space amplitudes of the kept states at each kept state's peak point.
    if isleaf
        peak_amps = V[peak, :]
    else
        peak_amps = Matrix{ComplexF64}(undef, length(peak), length(peak))
        cis = CartesianIndices(Tuple(dims))
        for (p, comp) in enumerate(peak)
            ci = cis[comp]
            peak_amps[p, :] = _contract_rows(V, [view(amps[a], ci[a], :) for a in eachindex(dims)])
        end
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
        isempty(bad) || @warn "PhaseGrid on mode :$(node.children[a].first) is too small: kept states $bad have edge support up to $(maximum(s)) > $_EDGE_TOL; enlarge the grid" maxlog=10
    end

    return MaterializedNode(node, dims, E, V, collect(Int, idx), ports, children, peak, peak_amps)
end

"""
    materialized(sys, node) -> MaterializedNode

The materialization of `node` (a node or a [`GivenSystem`](@ref) of the tree `sys` was built from).
"""
function materialized(sys::System, node::Union{Node,GivenSystem})
    stack = MaterializedNode[sys.root]
    while !isempty(stack)
        mn = pop!(stack)
        mn.node === node && return mn
        append!(stack, (c for c in mn.children if c isa MaterializedNode))
    end
    throw(ArgumentError("node is not part of this system's tree"))
end

"""
    operator(sys, node=root) -> KronFreeOperator

The Hamiltonian of `node` in its product basis at `sys.θ`, rebuilt from the materialized children
exactly as `materialize` built it (the root's includes the constant; its scratch and storage follow
the node's solver).
"""
function operator(sys::System, node::Node=sys.root.node)::KronFreeOperator
    mn = materialized(sys, node)
    routed = route(sys.root.node, sys.terms)
    c = node === sys.root.node ? sys.constant(sys.θ) : 0.0
    return _node_operator(node, mn.children, routed[node], sys.θ, c)
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

"""
    port(sys, kind::Symbol, mode::Symbol, node=root; k=0.0) -> Matrix{ComplexF64}

The spec-form spelling `port(sys, :n, :varphi_2)`: `kind` is one of `:n, :n2, :phi, :phi2,
:expiphi, :exp2pin`, and `k` the exponent coefficient of the exponential ports.
"""
port(sys::System, kind::Symbol, mode::Symbol, node::Node=sys.root.node; k::Real=0.0) =
    port(sys, mode, port_of_kind(kind, k), node)
