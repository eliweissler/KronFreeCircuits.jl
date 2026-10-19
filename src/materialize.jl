# Materialization: evaluate each node's Hamiltonian at θ, diagonalize, fix phases, select
# the kept states and form the ports its ancestors need, children first.

"""
    MaterializedNode{T}

A diagonalized node, stored in its arithmetic: `T = Float64` for a node classified real (see [`classify`](@ref)),
else `ComplexF64`.
`dims` are the axis sizes of the node's product basis (a basis child's dimension, or a node
child's number of kept states); `vectors` is `prod(dims) × length(kept)` in that basis, with
`energies` the kept eigenvalues (absolute, GHz) and `kept` their indices among the computed
eigenpairs.  `ports[(mode, port)]` is the port operator in the kept eigenbasis.
`children` holds, in axis order, the child's `Basis` or `MaterializedNode`.
Ports have their natural type (spec Section 20f, "Outputs"): on a real node a real-character port is a
`Matrix{Float64}` and an imaginary, paired or general one a `Matrix{ComplexF64}`; on a complex node every port is a
`Matrix{ComplexF64}`.
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
struct MaterializedNode{T<:Union{Float64,ComplexF64}}
    node::Union{Node,GivenSystem}
    dims::Vector{Int}
    energies::Vector{Float64}
    vectors::Matrix{T}
    kept::Vector{Int}
    ports::Dict{Tuple{Symbol,Port},NodeMatrix}
    children::Vector{Any}
    peak_component::Vector{Int}
    peak_amplitudes::Matrix{T}
end

# From any matrices: `vectors` and `peak_amplitudes` in their promoted element type, the ports as stored.
function MaterializedNode(node, dims, energies, vectors::AbstractMatrix, kept, ports::AbstractDict, children,
                          peak_component, peak_amplitudes::AbstractMatrix)
    T = promote_type(eltype(vectors), eltype(peak_amplitudes))
    return MaterializedNode{T}(node, dims, energies, vectors, kept, ports, children, peak_component, peak_amplitudes)
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
The first form expands `spec` with [`circuit_terms`](@ref) and [`constant_term`](@ref) given the
tree's `mode => basis` children (so a compact mode on a [`PeriodicGrid`](@ref) gets its twisted
derivative pieces), appends `extra_terms` (terms not in the circuit spec, such as couplings to a
[`GivenSystem`](@ref)), and delegates to the second, which takes a user-built (filtered,
transformed) term list and a scalar `constant`.
The second checks every `PeriodicGrid` against the harmonics `k` of every `expiphi(k)` factor on its
mode in the term list (the spec's cosines and `extra_terms` alike): `k` must be an integer and
divide the grid size ([`check_sectors`](@ref)).

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
A parameter missing from `θ`, or one whose value is not of a `Real` type (all system parameters are real; pass a
complex amplitude as two real parameters), raises an `ArgumentError` naming it.

Each `mode => basis` leaf child is checked against the mode's type: a compact mode (`:c`) needs
a `ChargeBasis` or `PeriodicGrid`, an extended (`:e`) or harmonic (`:h`) mode a `PhaseGrid` or `FockBasis`.
The spec form takes the types from the spec; the term-list form from `modetypes` (mode => type),
skipping modes absent from it (an empty dictionary skips the check).

Every node is classified once, from the term list and the tree, as real or complex arithmetic (its `arithmetic`
option; see [`Node`](@ref)); a node with `arithmetic = :real` whose structure does not guarantee a real Hamiltonian
throws an `ArgumentError`.

A term routed to a branch node with two or more factors inside the same child would need a
composite (multi-mode) port of that child; these are not supported and throw an `ArgumentError`.
"""
function materialize(root::Node, terms::Vector{Term}, θ::NamedTuple; constant::Coefficient=Coefficient(0),
                     previous::Union{Nothing,System}=nothing,
                     modetypes::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    _check_real_parameters(θ)
    _check_bases(root, modetypes)
    _check_periodic_harmonics(root, terms)
    routed = route(root, terms)
    _check_composite(routed)
    realnodes = classify(root, routed, constant, θ)
    needed = required_ports(root, routed)
    c = constant(θ)
    prev = previous === nothing ? nothing : previous.root
    mroot = _materialize(root, θ, routed, needed, c, prev, realnodes)
    return System(mroot, nothing, terms, θ, constant)
end

function materialize(root::Node, spec::Spec, θ::NamedTuple; previous::Union{Nothing,System}=nothing,
                     extra_terms::Vector{Term}=Term[])
    modetypes = Dict{Symbol,Symbol}(m.name => m.type for m in spec.modes)
    _check_bases(root, modetypes)
    bases = _tree_bases(root)
    sys = materialize(root, vcat(circuit_terms(spec, bases), extra_terms), θ; constant=constant_term(spec, bases),
                      previous=previous, modetypes=modetypes)
    return System(sys.root, spec, sys.terms, sys.θ, sys.constant)
end

# Every `PeriodicGrid` child against the harmonics `k` of every `expiphi(k)` factor on its mode, over the
# whole term list (the spec's cosines and any extra terms alike): each `k` must be an integer and divide
# the grid size ([`check_sectors`](@ref)).
function _check_periodic_harmonics(root::Node, terms::Vector{Term})
    for (mode, b) in _tree_bases(root)
        b isa PeriodicGrid || continue
        harmonics = Int[]
        for t in terms, f in t.factors
            f.mode === mode && f.port.kind === :expiphi || continue
            k = f.port.k
            abs(k - round(k)) <= 1e-12 ||
                throw(ArgumentError("mode :$mode: a term has the factor expiphi($k), but a PeriodicGrid needs an " *
                                    "integer harmonic k (the grid is 2π-periodic)"))
            push!(harmonics, round(Int, k))
        end
        check_sectors(b, unique!(harmonics), mode)
    end
    return nothing
end

# Every `mode => basis` child in the tree.
function _tree_bases(root::Node)
    bases = Dict{Symbol,Basis}()
    for node in _all_nodes(root), ch in node.children
        ch isa Pair && (bases[ch.first] = ch.second)
    end
    return bases
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
        ok = t === :c ? (b isa ChargeBasis || b isa PeriodicGrid) : t in (:e, :h) && (b isa PhaseGrid || b isa FockBasis)
        ok && continue
        want = t === :c ? "a ChargeBasis or PeriodicGrid" : t in (:e, :h) ? "a PhaseGrid or FockBasis" : "a known mode type (c, e or h)"
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
_childamps(b::Basis) = Diagonal(ones(dimension(b)))
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
    ports = Dict{Tuple{Symbol,Port},NodeMatrix}((g.mode, Port(kind, 0)) => M for (kind, M) in g.operators)
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
`R̃` has the promoted element type of `prev.vectors` and the overlaps (real only when all of them are).
Requires the checks of `_check_transportable`; `prev.vectors` is never written.
"""
function _transport_previous(prev::MaterializedNode, children, dims)
    _check_transportable(prev, children, dims)
    nodeaxes = [a for a in eachindex(children) if children[a] isa MaterializedNode]
    isempty(nodeaxes) && return copy(prev.vectors)
    S = Dict(a => _child_overlap(prev.children[a], children[a]) for a in nodeaxes)
    T = promote_type(eltype(prev.vectors), (eltype(M) for M in values(S))...)
    X = Matrix{T}(prev.vectors)                    # a copy
    # apply_axis! forbids aliasing: alternate between two distinct work arrays, one application per axis.
    W = similar(X)
    shape = (dims..., size(X, 2))
    for a in nodeaxes
        apply_axis!(reshape(W, shape), DenseOp(Matrix{T}(S[a]')), reshape(X, shape), a)
        X, W = W, X
    end
    return X
end

"""
    _child_overlap(prev::MaterializedNode, cur::MaterializedNode) -> Matrix

The physical overlaps `⟨prev kept i | cur kept j⟩`: at a leaf `prev.vectors' * cur.vectors`, at a
branch `_transport_previous(prev, cur.children, cur.dims)' * cur.vectors`, in the promoted element type.
Throws an `ArgumentError` when the dims or a basis child differ anywhere below.
"""
_child_overlap(prev::MaterializedNode, cur::MaterializedNode) =
    _overlap(_transport_previous(prev, cur.children, cur.dims), cur.vectors)

# Kronecker product of per-axis vectors, axis 1 fastest.
_kronaxes(vs) = foldl((acc, v) -> kron(v, acc), vs; init=[true])

# ⟨v_i| O_axis |v_j⟩ for the columns of V, with O acting on axis `axis` of the product basis.
# On real `V` it takes only real operators (the components the method below passes).
function _port_matrix(V::NodeMatrix, dims::Vector{Int}, axis::Int, op::LocalOp)
    X = reshape(V, dims..., size(V, 2))
    Y = similar(X)
    apply_axis!(Y, op, X, axis)
    return V' * reshape(Y, size(V))
end

# The port with operator `op` and conjugation character `ch` (spec Section 20f).
# On a complex node the character is not needed, and a real dense operator (a real child's port) is converted first,
# since BLAS has no mixed real–complex gemm.
# On a real node the port is formed from the components `R = realpart(op)`, `I = imagpart(op)` in real arithmetic, in
# its natural type: a real port is `Vᵀ R V` (`Matrix{Float64}`), an imaginary one `i Vᵀ I V`, and a paired or general one
# `Vᵀ R V + i Vᵀ I V` (`Matrix{ComplexF64}`, whose real and imaginary parts are the separately computed components).
_port_matrix(V::Matrix{ComplexF64}, dims::Vector{Int}, axis::Int, op::LocalOp, ::PortCharacter) =
    _port_matrix(V, dims, axis, op isa DenseOp ? DenseOp(convert(Matrix{ComplexF64}, op.M)) : op)
function _port_matrix(V::Matrix{Float64}, dims::Vector{Int}, axis::Int, op::LocalOp, ch::PortCharacter)
    ch.kind === :real && return _port_matrix(V, dims, axis, realpart(op))
    Im = _port_matrix(V, dims, axis, imagpart(op))
    ch.kind === :imaginary && return complex.(zero(Im), Im)
    return complex.(_port_matrix(V, dims, axis, realpart(op)), Im)
end

# The port `p` of `mode` on the columns of `V`, a node's vectors over its materialized `children`; its character is
# the port's in the leaf of its mode, which a real node child's ports carry (spec Section 20f).
function _node_port(node::Node, children, dims::Vector{Int}, V::NodeMatrix, mode::Symbol, p::Port)
    a = axis_of(node, mode)
    return _port_matrix(V, dims, a, _childop(children[a], mode, p), _factor_character(_mode_leaves(node), Factor(mode, p)))
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
storage(::ShiftInvert) = Array
storage(s::ChebyshevFilter) = s.on
storage(a::Auto) = a.on

# How many of the lowest eigenpairs the node's solver returns.
_compute_count(node::Node, dims) = node.compute == 0 ? prod(dims) : node.compute

"""
    resolve(solver, info) -> Solver

The solver that diagonalizes a node, given its `solver` field and `info = (; isleaf, dims, m)`:
whether all its children are bases, its product dimensions, and the number of eigenpairs requested.
Every solver resolves to itself; [`Auto`](@ref) resolves by the node's shape (see its docstring).
`materialize` resolves each node once, before building its operator, and builds the operator's
storage and scratch from the resolved solver.
"""
resolve(s::Solver, info) = s
function resolve(a::Auto, info)
    prod(info.dims) <= a.dense_cutoff && return Dense()
    info.isleaf && return a.on === Array ? ShiftInvert() : ChebyshevFilter(on=a.on)
    return LOBPCG(preconditioner=:ritz, seed=:product, on=a.on)
end

# The resolved solver of `node` over its materialized `children` (bases or `MaterializedNode`s);
# `materialize` and `operator` both go through here, so they build the same operator.
function _resolve_node_solver(node::Node, children)
    dims = Int[_childdim(c) for c in children]
    s = resolve(node.solver, (; isleaf=all(c -> c isa Basis, children), dims, m=_compute_count(node, dims)))
    if node.compute == 0 && !(s isa Dense)
        how = s === node.solver ? "got $(nameof(typeof(s)))" :
              "$(repr(node.solver)) resolved to $(nameof(typeof(s))) for this node of dimension $(prod(dims))"
        throw(ArgumentError("compute == 0 (all eigenpairs) requires the Dense solver; $how (modes $(modes(node))); " *
                            "set compute to the number of eigenpairs needed, or pass solver=Dense() to keep all of them"))
    end
    return s
end

# (2)-(3) The node's operator at θ: its routed `terms` on the children's local operators (a basis
# child's, or a node child's ports), each node child's kept energies on its axis, and `constant`.
# With `real` (a node classified real) it is built from real data only: `Float64` terms from
# `_real_node_terms` and `real(constant)`; otherwise it is `ComplexF64`, whatever the types of its data (a real child's
# ports and energies are converted).
# Scratch is reserved for the resolved `solver`'s full block (LOBPCG: the computed pairs and its guard
# columns; ChebyshevFilter: the computed pairs and its extra columns), on the solver's storage.
function _node_operator(node::Node, solver::Solver, children, terms::Vector{Term}, θ::NamedTuple, constant; real::Bool)
    dims = Int[_childdim(c) for c in children]
    if real
        axisterms = AxisTerm[t for t in _real_node_terms(node, children, terms, θ)]
        constant = Base.real(constant)
    else
        constant = ComplexF64(constant)
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
    end
    for (a, c) in enumerate(children)
        c isa MaterializedNode && push!(axisterms, AxisTerm(1.0, Tuple{Int,LocalOp}[(a, DiagonalOp(c.energies))]))
    end
    m = _compute_count(node, dims)
    capacity = solver isa LOBPCG ? m + solver.guard :
               solver isa ChebyshevFilter ? min(m + solver.extra, prod(dims)) : 1
    return KronFreeOperator(dims, axisterms; constant=constant, block_capacity=capacity, on=storage(solver))
end

# The real terms of a real node: each routed term's expansion of `Re(c ⊗ O)` (`_real_expansion`), with the real terms
# on equal keys summed, so a conjugate pair contributes its real terms once, doubled.
# A key lists, per factor, `(axis, mode, canonical port, component)`: the mode is needed because a node child's axis
# carries several modes, whose ports on that axis are different operators.
# A summed term keeps the components of the first term on its key, its coefficient mapped onto them by that term's sign.
# A factor's character is its port's in the leaf of its mode, which a real node child's ports carry (spec Section 20f).
function _real_node_terms(node::Node, children, terms::Vector{Term}, θ::NamedTuple)
    leaves = _mode_leaves(node)
    index = Dict{Vector{Tuple{Int,Symbol,Port,Bool}},Int}()
    sums = Float64[]                       # summed coefficient on the canonical components
    firsts = Tuple{Int,Vector{Tuple{Int,LocalOp}}}[]   # the first term's sign and components
    for t in terms
        fs = Tuple{Int,Symbol,LocalOp,PortCharacter}[]
        for f in t.factors
            a = axis_of(node, f.mode)
            push!(fs, (a, f.mode, _childop(children[a], f.mode, f.port), _factor_character(leaves, f)))
        end
        sort!(fs; by=first)          # one factor per axis: checked by _check_composite
        for r in _real_expansion(t.coeff(θ), Tuple{Int,LocalOp,PortCharacter}[(a, op, ch) for (a, _, op, ch) in fs])
            key = Tuple{Int,Symbol,Port,Bool}[(a, m, first(_canonical_port(ch)), inS) for ((a, m, _, ch), inS) in zip(fs, r.S)]
            i = get!(index, key, length(sums) + 1)
            if i > length(sums)
                push!(sums, r.coeff * r.sign)
                push!(firsts, (r.sign, r.factors))
            else
                sums[i] += r.coeff * r.sign
            end
        end
    end
    return AxisTerm{Float64}[AxisTerm(s * sign, ops) for (s, (sign, ops)) in zip(sums, firsts) if !iszero(s)]
end

# The initial block of an LOBPCG node: `nothing` without a previous solution of the same `dims`,
# else a copy of its kept vectors (raw coefficients, in kept order) truncated to `m` columns or
# padded on the right with `randn` columns (never zeros, which the solver rejects) of element type `T`.
# Real previous vectors are converted for a complex operator; complex ones are dropped (a cold start) for a real
# operator (spec Section 20f; only when `arithmetic` or the term list changed between the calls).
# Truncation follows the kept order, not energy order (a previous kept with a reordered `indices`
# rule contributes its first columns); the solver's guard columns recover any lower states.
_initial_block(prev, dims, m::Int) =
    _initial_block(prev, dims, m, prev === nothing ? Nothing : eltype(prev.vectors))
function _initial_block(prev, dims, m::Int, ::Type{T}) where {T}
    (prev === nothing || prev.dims != dims) && return nothing
    T <: Real && !(eltype(prev.vectors) <: Real) && return nothing
    k = min(size(prev.vectors, 2), m)
    X = Matrix{T}(undef, size(prev.vectors, 1), m)
    X[:, 1:k] .= view(prev.vectors, :, 1:k)
    randn!(view(X, :, k+1:m))
    return X
end

# Diagonalize: only LOBPCG takes an initial block, built from the previous solution.
_solve(s::Solver, H, m, prev, dims) = solve(s, H, m)
_solve(s::LOBPCG, H, m, prev, dims) = solve(s, H, m; X0=_initial_block(prev, dims, m, eltype(H)))

function _materialize(node::Node, θ::NamedTuple, routed, needed, constant, prev, realnodes)
    children = Any[]
    for (i, ch) in enumerate(node.children)
        push!(children, ch isa Node ? _materialize(ch, θ, routed, needed, 0.0, _prev_child(prev, i), realnodes) :
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

    # (2)-(3) The node's operator, built for the resolved solver (Auto() picks one by the node's shape).
    # A node classified real runs in real arithmetic; branches stay complex for now, over real leaves too.
    solver = _resolve_node_solver(node, children)
    isleaf = all(c -> c isa Basis, children)
    H = _node_operator(node, solver, children, routed[node], θ, constant; real=isleaf && realnodes[node])

    # (4) Diagonalize.
    _dense_too_large(solver, prod(dims)) &&
        @warn("Dense node on modes $(modes(node)) has product dimension $(prod(dims)) > $_DENSE_WARN_DIM; " *
              (all(c -> c isa Basis, children) ?
                   "this is a leaf: ShiftInvert() on the CPU or ChebyshevFilter(on=CuArray) on a GPU (or Auto()) is advised; see README" :
                   "this is a branch: LOBPCG (or Auto()) is advised"), _id=(:kronfree_dense_large, objectid(node)), maxlog=1)
    m = _compute_count(node, dims)
    E, V = _solve(solver, H, m, prev, dims)

    # (5)-(7) Ports the selection rule inspects, on all computed vectors; select; slice.
    # Selection rules see only moduli (port elements, overlaps), so phases are fixed on the kept
    # vectors only (fixing them on every computed vector would cost a pass over all of them and
    # warn for high states with no weight on the mask), and the selection ports are rotated to match.
    ports = Dict{Tuple{Symbol,Port},NodeMatrix}()
    for (mode, p) in ports_needed(node.keep)
        ports[(mode, p)] = _node_port(node, children, dims, V, mode, p)
    end
    idx = validate_selection(select(node.keep, SelectionContext(E, V, ports, prev, reference)), size(V, 2), node.keep)
    E, V = E[idx], V[:, idx]

    # :follow makes each physical overlap with the previous kept vector real and positive; with no previous
    # solution, :positive_sector is used instead.  :positive_sector makes, at a leaf, the largest
    # entry on the positive sector real and positive; at a branch, the exact full-space amplitude at
    # the joint peak of the dominant product component (`branch_peak_phase`), the argmax restricted
    # to the positive sector on basis-child axes only (node-child masks are all true).
    # Peaks are taken on the unphased vectors, as fix_phases! and branch_peak_phase do.
    # On real vectors every phase is a sign (`_unphase`), so a real node never acquires complex vectors.
    mask = _kronaxes(_childmask(c) for c in children)
    amps = isleaf ? AbstractMatrix[] : AbstractMatrix[_childamps(c) for c in children]
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
            view(V, :, j) .*= _unphase(eltype(V), α)
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
        peak_amps = Matrix{eltype(V)}(undef, length(peak), length(peak))
        cis = CartesianIndices(Tuple(dims))
        for (p, comp) in enumerate(peak)
            ci = cis[comp]
            peak_amps[p, :] = _contract_rows(V, [view(amps[a], ci[a], :) for a in eachindex(dims)])
        end
    end

    # (8) The remaining required ports, on the kept vectors.
    for (mode, p) in needed[node]
        haskey(ports, (mode, p)) && continue
        ports[(mode, p)] = _node_port(node, children, dims, V, mode, p)
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
the node's solver, as resolved by `materialize`); it is real exactly when the node was materialized in real arithmetic.
"""
function operator(sys::System, node::Node=sys.root.node)::KronFreeOperator
    mn = materialized(sys, node)
    routed = route(sys.root.node, sys.terms)
    c = node === sys.root.node ? sys.constant(sys.θ) : 0.0
    return _node_operator(node, _resolve_node_solver(node, mn.children), mn.children, routed[node], sys.θ, c;
                          real=mn isa MaterializedNode{Float64})
end

"""
    arithmetic(sys, node=root) -> Symbol

The arithmetic `node` was materialized in: `:real` when it is stored as a `MaterializedNode{Float64}` (real
eigenvectors), else `:complex`.
"""
arithmetic(sys::System, node::Node=sys.root.node) = materialized(sys, node) isa MaterializedNode{Float64} ? :real : :complex

"Kept energies of `node` (default: the root), absolute, GHz."
energies(sys::System, node::Node=sys.root.node) = materialized(sys, node).energies

"Kept eigenvectors of `node` in its product basis, one column per kept state, as stored (`Matrix{Float64}` for a node in real arithmetic)."
eigenvectors(sys::System, node::Node=sys.root.node) = materialized(sys, node).vectors

"""
    port(sys, mode, p, node=root) -> Matrix{Float64} or Matrix{ComplexF64}

The port `p` of `mode` in the kept eigenbasis of `node`, as stored: a `Matrix{Float64}` for a real-character port of a
node in real arithmetic, else a `Matrix{ComplexF64}` (see [`MaterializedNode`](@ref)).
"""
function port(sys::System, mode::Symbol, p::Port, node::Node=sys.root.node)
    mn = materialized(sys, node)
    haskey(mn.ports, (mode, p)) && return mn.ports[(mode, p)]
    avail = join(("(:$m, $(_portname(q)))" for (m, q) in sort!(collect(keys(mn.ports)); by=string)), ", ")
    throw(ArgumentError("port $(_portname(p)) on mode :$mode was not formed for this node; available: " *
                        (isempty(avail) ? "none" : avail)))
end

"""
    port(sys, kind::Symbol, mode::Symbol, node=root; k=0.0) -> Matrix

The spec-form spelling `port(sys, :n, :varphi_2)`: `kind` is one of `:n, :n2, :phi, :phi2,
:expiphi, :exp2pin`, and `k` the exponent coefficient of the exponential ports.
"""
port(sys::System, kind::Symbol, mode::Symbol, node::Node=sys.root.node; k::Real=0.0) =
    port(sys, mode, port_of_kind(kind, k), node)

"""
    charge_port(sys, mode) -> Matrix{Float64} or Matrix{ComplexF64}

The physical charge `n` of `mode` in the kept eigenbasis of the leaf that holds it, computed on
demand from that leaf's kept vectors, in its natural type (as [`port`](@ref)): real for a `ChargeBasis` mode of a leaf
in real arithmetic, complex otherwise (always on a `PeriodicGrid` mode).
For a [`PeriodicGrid`](@ref) mode it is `V'(D₁)V + n_g I`, with the twisted derivative
`D₁ = d1 + e^{2πi n_g} d1wrap₊ + e^{-2πi n_g} d1wrap₋` (which is `n - n_g` in the twisted gauge) and
`n_g` the mode's offset charge from `sys.spec`, both evaluated at `sys.θ`; it needs a system built
from a spec.
For a `ChargeBasis` mode it is `V' n V`, equal to `port(sys, mode, nport(), leaf)` when that port was
formed.
A `PeriodicGrid` does not serve `nport()` itself, so this is the accessor for its charge.
"""
function charge_port(sys::System, mode::Symbol)
    leaf = _leaf_holding(sys.root, mode)
    leaf === nothing && throw(ArgumentError("mode :$mode is not a basis child of any node of this system"))
    a = axis_of(leaf.node, mode)
    b = leaf.children[a]
    V, dims = leaf.vectors, leaf.dims
    b isa PeriodicGrid || return _port_matrix(V, dims, a, localop(b, nport()), character(b, nport()))
    sys.spec === nothing &&
        throw(ArgumentError("charge_port on the PeriodicGrid mode :$mode needs its offset charge, but the system " *
                            "was built from a term list; materialize it from a spec"))
    idx = findfirst(m -> m.name === mode, sys.spec.modes)
    idx === nothing && throw(ArgumentError("mode :$mode is not a mode of the system's spec"))
    ng = sys.spec.n_g[idx]
    Q = ng(sys.θ) * Matrix{ComplexF64}(I, size(V, 2), size(V, 2))
    for (t, p) in _twisted_pieces(ng, :d1)
        Q .+= (t === nothing ? 1.0 : t(sys.θ)) .* _port_matrix(V, dims, a, localop(b, p), character(b, p))
    end
    return Q
end

# The materialized node with a `mode => basis` child, or `nothing`.
function _leaf_holding(root::MaterializedNode, mode::Symbol)
    stack = MaterializedNode[root]
    while !isempty(stack)
        mn = pop!(stack)
        mn.node isa Node && any(ch -> ch isa Pair && ch.first === mode, mn.node.children) && return mn
        append!(stack, (c for c in mn.children if c isa MaterializedNode))
    end
    return nothing
end
