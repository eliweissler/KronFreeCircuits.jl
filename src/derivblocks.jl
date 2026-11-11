# Near-degenerate blocks (design §4.4): where the second-order Taylor expansion of the kept levels cannot be trusted over
# the caller's scale of a parameter, and the effective Hamiltonian that replaces it there.

"""
    NearDegenerateBlock

A group of kept states of one node whose second-order Taylor expansion along `parameters` is not trusted over the
caller's `scale`, with the block's effective Hamiltonian; returned by [`blocks`](@ref).
`parameters` is `(x,)` for a parameter, or the mixed pair `(x, λ)` (ordered by name) for the union of the blocks of
`x`, of `λ` and of the mixed second-order pass.
`states` are the kept-state indices at `node` (columns of its vectors), increasing, and `energies` their energies.
A pair `(i, l)` of kept states is flagged along `x` when `|⟨ψ_l|∂H/∂x|ψ_i⟩| δx > r |E_i − E_l|` with a nonzero
coupling (`δx` the `scale` of `x`), or when it is an exact degeneracy (a detuning within `degenerate_tol`) with a
nonzero coupling, at first order or, through other states, at second order; a block is a connected component of the
flagged pairs.
`detuning` is the smallest `|E_i − E_l|` and `coupling` the largest coupling among the flagged pairs: the first-order
matrix element `|⟨ψ_l|∂H/∂x|ψ_i⟩|`, or for a pair flagged by a second-order clash the second-order one, the overlap of
`ψ_l` with the second-order right-hand side of `ψ_i`.
For a block flagged in more than one way `coupling` is the largest of quantities that may have different units
(first-order couplings to different parameters, a second-order matrix element for a pair flagged by a second-order
clash): it is for reporting, not for arithmetic.
`exact` is true when a flagged pair is exactly degenerate: the per-state derivatives of its states are `NaN`, and the
block's effective Hamiltonian is what is defined.
`order` is the order of the effective Hamiltonian in the displacement (the `order` of [`derivatives`](@ref)).
`H1[x]` is `⟨ψ_b|∂H/∂x|ψ_b′⟩` over the block's states, for each parameter `x` of `parameters`.
`H2[(x, y)]` (each pair of `parameters`, by `_pairkey`; empty when `order == 1`) is
`⟨ψ_b|∂²H/∂x∂y|ψ_b′⟩ + F^{xy} + F^{yx}`, with `F^{xy} = ½ (M^{xy} + (M^{yx})†)`,
`M^{xy}[b, b′] = ⟨(∂H/∂x) ψ_b|Y^y_{b′}⟩` and `Y^y_{b′} = Σ_{l ∉ B} ψ_l ⟨ψ_l|∂H/∂y|ψ_{b′}⟩ / (E_{b′} − E_l)` over every
state of the node outside the block, kept or not: the outside states folded in to second order in the displacement.
A block state exactly degenerate with an unkept state it couples to (a selection-boundary record) makes these pieces `NaN`.
`tr(H1[x])` and `tr(H2[(x, x)])` are the block-invariant sums of the first and second derivatives of the block's
energies, which exist even when the per-state derivatives are `NaN`.
See [`block_levels`](@ref).
"""
struct NearDegenerateBlock
    node::Node
    parameters::Tuple{Vararg{Symbol}}
    states::Vector{Int}
    energies::Vector{Float64}
    detuning::Float64
    coupling::Float64
    exact::Bool
    order::Int
    H1::Dict{Symbol,Matrix{ComplexF64}}
    H2::Dict{Tuple{Symbol,Symbol},Matrix{ComplexF64}}
end

"""
    blocks(d::Derivatives) -> Vector{NearDegenerateBlock}

The near-degenerate blocks found by [`derivatives`](@ref): at every visited node, the blocks of each parameter of
`wrt` and of each mixed pair (see [`NearDegenerateBlock`](@ref)).
A block is a validity report: the point derivatives are the true ones whatever the blocks, except those of the states
of an exact block, which are `NaN`.
"""
blocks(d::Derivatives) = NearDegenerateBlock[b for b in d.blocks]

"""
    block_levels(b::NearDegenerateBlock, displacement::NamedTuple) -> (; energies, states)

The energies (ascending) and states of the block `b` at a finite displacement of its parameters, such as
`block_levels(b, (phi_ext = 2e-3,))`, from the dense diagonalization of its effective Hamiltonian
`H_eff(δ) = diag(E_B) + Σ_x δ_x H1[x] + ½ Σ_x δ_x² H2[(x, x)] + δ_x δ_λ H2[(x, λ)]` (the last term for a mixed block,
the second-order terms when `b.order == 2`).
`states` holds one column per level, its coefficients in the block's states at `θ`.
A parameter not named is not displaced; a name that is not one of `b.parameters` is an `ArgumentError`, and so is an
effective Hamiltonian that is not finite at the displacement (a block state exactly degenerate with a state outside the
kept set, see the selection-boundary records).
The energies differ from those of a re-solve at the displaced point (at fixed selection) by `O(|δ|³ v³ / g²)`, with
`v` the block's largest coupling to the states outside it and `g` its gap to them; with `b.order == 1` by
`O(|δ|² v² / g)`.
"""
function block_levels(b::NearDegenerateBlock, displacement::NamedTuple)
    for k in keys(displacement)
        k in b.parameters ||
            throw(ArgumentError("the block's parameters are $(join(":" .* string.(b.parameters), ", ")); the " *
                                "displacement names :$k"))
    end
    δ(x) = Float64(get(displacement, x, 0.0))
    H = Matrix{ComplexF64}(Diagonal(b.energies))
    for (x, M) in b.H1
        H .+= δ(x) .* M
    end
    for ((x, y), M) in b.H2
        H .+= (x === y ? δ(x)^2 / 2 : δ(x) * δ(y)) .* M
    end
    all(isfinite, H) ||
        throw(ArgumentError("the effective Hamiltonian of the block of states $(b.states) of the node on modes " *
                            "$(modes(b.node)) is not finite at this displacement: a block state is exactly degenerate " *
                            "with a state outside the kept set that it couples to (see the selection-boundary records, " *
                            "`d.boundary`), or a child's derivative is undefined"))
    F = eigen(Hermitian((H + H') / 2))
    return (; energies=F.values, states=F.vectors)
end

# The search while the passes run.

# A flagged pair `(i, l)`, `i < l`, of kept states.
const _Flag = @NamedTuple{detuning::Float64, coupling::Float64, exact::Bool}

# A block of one node and one parameter set, while the passes run.
mutable struct _Group
    states::Vector{Int}
    H1::Dict{Symbol,Matrix{ComplexF64}}
    H2::Dict{Tuple{Symbol,Symbol},Matrix{ComplexF64}}
end

# What the search holds per visited node: `C[x] = V† (∂H/∂x) V` from the first-order pass of `x`; the flagged pairs per
# source, `(x,)` (the first-order pass of `x` and the second-order pass `(x, x)`) or a mixed pair (its second-order
# pass); the blocks per parameter set; and the parameter sets whose blocks must be formed again (`dirty`).
struct _NodeBlocks
    mn::MaterializedNode
    C::Dict{Symbol,Matrix}
    flags::Dict{Tuple{Vararg{Symbol}},Dict{Tuple{Int,Int},_Flag}}
    groups::Dict{Tuple{Vararg{Symbol}},Vector{_Group}}
    dirty::Set{Tuple{Vararg{Symbol}}}
end
_NodeBlocks(mn::MaterializedNode) = _NodeBlocks(mn, Dict{Symbol,Matrix}(), Dict{Tuple{Vararg{Symbol}},Dict{Tuple{Int,Int},_Flag}}(),
                                                Dict{Tuple{Vararg{Symbol}},Vector{_Group}}(), Set{Tuple{Vararg{Symbol}}}())

# The block search of one call of `derivatives`: its tree, the dictionaries of the passes' data (filled as they run),
# the parameters, mixed pairs, order, `scale` and `r`, the per-node state (`nodes`, in the order first visited), and the
# second-order passes completed so far (`done`).
struct _BlockSearch
    sys::System
    routed::Any
    parts::Any
    resolvents::IdDict{Any,Resolvent}
    first::Dict{Symbol,Any}
    second::Dict{Tuple{Symbol,Symbol},Any}
    params::Vector{Symbol}
    mixed::Vector{Tuple{Symbol,Symbol}}
    order::Int
    scale::NamedTuple
    r::Float64
    nodes::IdDict{Any,_NodeBlocks}
    visited::Vector{Any}
    done::Vector{Tuple{Symbol,Symbol}}
end
_BlockSearch(sys, routed, parts, resolvents, first, second, params, mixed, order, scale, r) =
    _BlockSearch(sys, routed, parts, resolvents, first, second, params, mixed, order, scale, r,
                 IdDict{Any,_NodeBlocks}(), Any[], Tuple{Symbol,Symbol}[])

# The parameter sets that have blocks: each parameter, then each mixed pair.
_block_sets(s::_BlockSearch) = Tuple{Vararg{Symbol}}[[(x,) for x in s.params]; s.mixed]
# The flag sources whose union makes the blocks of the set `P`.
_sources(P::Tuple{Vararg{Symbol}}) = length(P) == 1 ? [P] : [(P[1],), (P[2],), P]
# The parameter sets whose blocks a new flag of `source` changes.
_affected(s::_BlockSearch, source) = length(source) == 1 ? [source; [m for m in s.mixed if source[1] in m]] : [source]
# The second-order pieces of a block of the set `P`.
_pair_keys(P::Tuple{Vararg{Symbol}}) = length(P) == 1 ? [(P[1], P[1])] : [(P[1], P[1]), (P[2], P[2]), _pairkey(P...)]

function _flag!(pairs::Dict{Tuple{Int,Int},_Flag}, i::Int, l::Int, detuning::Real, coupling::Real, exact::Bool)
    key = minmax(i, l)
    old = get(pairs, key, nothing)
    pairs[key] = old === nothing ? _Flag((detuning, coupling, exact)) :
                 _Flag((old.detuning, max(old.coupling, coupling), old.exact || exact))
    return pairs
end

function _node_blocks!(s::_BlockSearch, mn::MaterializedNode)
    return get!(s.nodes, mn.node) do
        push!(s.visited, mn.node)
        _NodeBlocks(mn)
    end
end

"""
    _first_order_flags!(search, mn, x, R, clashes)

Records at the node `mn`, for the parameter `x`, its couplings `C = V† R` (with `R = (∂H/∂x) V`) and its flagged pairs
of kept states: each kept–kept clash of `clashes` (from `_apply_resolvent`), and, when `search.scale` has `δx` for `x`,
each pair with `|C[l, i]| δx > r |E_i − E_l|` whose coupling is not zero by the per-column rule in either direction.
"""
function _first_order_flags!(s::_BlockSearch, mn::MaterializedNode, x::Symbol, R::AbstractMatrix,
                             clashes::Vector{Vector{Int}})
    nb = _node_blocks!(s, mn)
    E = mn.energies
    C = mn.vectors' * R
    nb.C[x] = C
    pairs = get!(() -> Dict{Tuple{Int,Int},_Flag}(), nb.flags, (x,))
    for i in eachindex(clashes), l in clashes[i]
        l == 0 || _flag!(pairs, i, l, abs(E[i] - E[l]), abs(C[l, i]), true)
    end
    δx = get(s.scale, x, nothing)
    δx === nothing && return nothing
    zero_overlap = [1e-10 * norm(view(R, :, i)) for i in axes(R, 2)]
    for i in eachindex(E), l in eachindex(E)
        l == i && continue
        coupled = abs(C[l, i]) > zero_overlap[i] || abs(C[i, l]) > zero_overlap[l]
        coupled && abs(C[l, i]) * δx > s.r * abs(E[i] - E[l]) &&
            _flag!(pairs, i, l, abs(E[i] - E[l]), abs(C[l, i]), false)
    end
    return nothing
end

# After the first-order passes: the blocks of every parameter and mixed pair at every visited node, with their `H1`.
function _first_order_blocks!(s::_BlockSearch)
    for node in s.visited, P in _block_sets(s)
        _regroup!(s, s.nodes[node], P)
    end
    return nothing
end

"""
    _second_order_flags!(search, mn, key, RHS, clashes)

Records at the node `mn` the kept–kept clashes of the second-order pass `key` (from `_apply_resolvent` on its
right-hand side `RHS`) as exact flagged pairs, under the source `(x,)` for the pass `(x, x)` and `key` for a mixed pass,
with the coupling `|⟨ψ_l|RHS[:, i]⟩|`; the blocks they change are formed again by `_second_order_blocks!`.
"""
function _second_order_flags!(s::_BlockSearch, mn::MaterializedNode, key::Tuple{Symbol,Symbol}, RHS::AbstractMatrix,
                              clashes::Vector{Vector{Int}})
    any(c -> any(!=(0), c), clashes) || return nothing
    nb = _node_blocks!(s, mn)
    E = mn.energies
    C = mn.vectors' * RHS
    source = key[1] === key[2] ? (key[1],) : key
    pairs = get!(() -> Dict{Tuple{Int,Int},_Flag}(), nb.flags, source)
    for i in eachindex(clashes), l in clashes[i]
        l == 0 || _flag!(pairs, i, l, abs(E[i] - E[l]), abs(C[l, i]), true)
    end
    union!(nb.dirty, _affected(s, source))
    return nothing
end

"""
    _second_order_blocks!(search, mn, res, x, λ, Hx, Hλ, Hxλ)

At the node `mn`, in the second-order pass `(x, λ)`: the blocks changed by its clashes are formed again (with the
pieces of the passes already run), then every block whose parameters contain `x` and `λ` gets
`H2[_pairkey(x, λ)]` from the operators the pass built and the resolvent `res` (`_block_H2`).
"""
function _second_order_blocks!(s::_BlockSearch, mn::MaterializedNode, res::Resolvent, x::Symbol, λ::Symbol,
                               Hx::KronFreeOperator, Hλ::KronFreeOperator, Hxλ::KronFreeOperator)
    nb = get(s.nodes, mn.node, nothing)
    nb === nothing && return nothing
    for P in nb.dirty
        _regroup!(s, nb, P)
    end
    empty!(nb.dirty)
    key = _pairkey(x, λ)
    for (P, groups) in nb.groups
        x in P && λ in P || continue
        for g in groups
            g.H2[key] = _block_H2(mn, res, g.states, Hx, Hλ, Hxλ)
        end
    end
    return nothing
end

# The blocks of the set `P` at a node: the connected components of the union of its sources' flagged pairs.
# A component already a block keeps its pieces; a new one gets `H1` from the stored couplings, zero second-order pieces
# (when `order == 2`), and the pieces of the second-order passes already run, formed from rebuilt operators.
function _regroup!(s::_BlockSearch, nb::_NodeBlocks, P::Tuple{Vararg{Symbol}})
    pairs = _merged_flags(nb, P)
    old = get(nb.groups, P, _Group[])
    new = _Group[]
    for S in _components(collect(keys(pairs)))
        k = findfirst(g -> g.states == S, old)
        if k !== nothing
            push!(new, old[k])
            continue
        end
        H1 = Dict{Symbol,Matrix{ComplexF64}}(x => haskey(nb.C, x) ? Matrix{ComplexF64}(nb.C[x][S, S]) :
                                                  zeros(ComplexF64, length(S), length(S)) for x in P)
        g = _Group(S, H1, Dict{Tuple{Symbol,Symbol},Matrix{ComplexF64}}())
        if s.order == 2
            for key in _pair_keys(P)
                g.H2[key] = zeros(ComplexF64, length(S), length(S))
                key in s.done && _reform!(s, nb, g, key)
            end
        end
        push!(new, g)
    end
    isempty(new) ? delete!(nb.groups, P) : (nb.groups[P] = new)
    return nothing
end

function _merged_flags(nb::_NodeBlocks, P::Tuple{Vararg{Symbol}})
    pairs = Dict{Tuple{Int,Int},_Flag}()
    for src in _sources(P), ((i, l), f) in get(nb.flags, src, Dict{Tuple{Int,Int},_Flag}())
        _flag!(pairs, i, l, f.detuning, f.coupling, f.exact)
    end
    return pairs
end

# The connected components of the graph whose edges are `pairs`, each as an increasing vector, ordered by first state.
function _components(pairs::Vector{Tuple{Int,Int}})
    parent = Dict{Int,Int}()
    root(i) = (p = get!(parent, i, i); p == i ? i : (parent[i] = root(p)))
    for (i, l) in pairs
        a, b = root(i), root(l)
        a == b || (parent[max(a, b)] = min(a, b))
    end
    comps = Dict{Int,Vector{Int}}()
    for i in keys(parent)
        push!(get!(() -> Int[], comps, root(i)), i)
    end
    return sort!([sort!(c) for c in values(comps)]; by=first)
end

# The piece `H2[key]` of the group `g`, for a second-order pass already run: the node's operators rebuilt from the
# stored first- and second-order data; left zero when that pass did not visit the node.
function _reform!(s::_BlockSearch, nb::_NodeBlocks, g::_Group, key::Tuple{Symbol,Symbol})
    p, q = key
    node = nb.mn.node
    np, nq = _node_derivative(s.first[p], s.sys, node), _node_derivative(s.first[q], s.sys, node)
    npq = _node_derivative(s.second[key], s.sys, node)
    (np === nothing || nq === nothing || npq === nothing) && return nothing
    res = s.resolvents[node]
    Hp, Hq, Hpq = _pair_operators(s.sys, s.routed, nb.mn, res, p, q, np.children, nq.children, npq.children;
                                  parts=s.parts)
    g.H2[key] = _block_H2(nb.mn, res, g.states, Hp, Hq, Hpq)
    return nothing
end

"""
    _block_H2(mn, res, B, Hy, Hz, Hyz) -> Matrix{ComplexF64}

The second-order piece `⟨ψ_b|Hyz|ψ_b′⟩ + F^{yz} + F^{zy}` of the effective Hamiltonian of the block `B` (kept-state
indices of `mn`), with `F^{yz} = ½ (M^{yz} + (M^{zy})†)`, `M^{yz} = ((Hy) V_B)† Y^z` and `Y^z` the resolvent `res`
applied to `(Hz) V_B` at the block's energies, every block state excluded (see [`NearDegenerateBlock`](@ref)).
"""
function _block_H2(mn::MaterializedNode{T}, res::Resolvent, B::Vector{Int}, Hy::KronFreeOperator,
                   Hz::KronFreeOperator, Hyz::KronFreeOperator) where {T}
    VB = mn.vectors[:, B]
    apply(H, X) = Array(H * adapt(H.on, Matrix{T}(X)))
    shifts = mn.energies[B]
    exclude = [copy(B) for _ in B]
    Ry = apply(Hy, VB)
    Yy = Base.first(_apply_resolvent(res, Ry, shifts, exclude))
    Rz, Yz = Hz === Hy ? (Ry, Yy) : (R = apply(Hz, VB); (R, Base.first(_apply_resolvent(res, R, shifts, exclude))))
    Myz, Mzy = Ry' * Yz, Rz' * Yy
    return Matrix{ComplexF64}(VB' * apply(Hyz, VB) + (Myz + Mzy') / 2 + (Mzy + Myz') / 2)
end

# The blocks found, as `NearDegenerateBlock`s: per node in the order first visited, per parameter set, by first state.
function _collect_blocks(s::_BlockSearch)
    out = Any[]
    for node in s.visited
        nb = s.nodes[node]
        E = nb.mn.energies
        for P in _block_sets(s), g in get(nb.groups, P, _Group[])
            flags = [f for ((i, l), f) in _merged_flags(nb, P) if i in g.states]
            push!(out, NearDegenerateBlock(node, P, g.states, E[g.states], minimum(f.detuning for f in flags),
                                           maximum(f.coupling for f in flags), any(f.exact for f in flags), s.order,
                                           g.H1, g.H2))
        end
    end
    return out
end
