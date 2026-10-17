"""
    SelectionContext(energies, vectors, ports, previous[, reference])

What a selection rule may inspect after a node is diagonalized: the computed `energies`,
the eigenvector columns `vectors`, the port matrices in the eigenbasis keyed by `(mode, port)`,
`previous`, the node's previous `MaterializedNode` (or `nothing`), and `reference`, the previous
kept vectors expressed in the current product basis (or `nothing`).
`materialize` passes as `reference` the previous kept vectors transported through the children's
physical overlaps, so that `reference' * vectors` are physical overlaps even when the children were
re-diagonalized.
The four-argument form takes `previous.vectors` (raw coefficients) as the reference, which is the
physical overlap only when the children's bases are unchanged.
"""
struct SelectionContext
    energies::Vector{Float64}
    vectors::Matrix{ComplexF64}
    ports::Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}
    previous::Any
    reference::Union{Nothing,AbstractMatrix}
end
SelectionContext(energies, vectors, ports, previous) =
    SelectionContext(energies, vectors, ports, previous, previous === nothing ? nothing : previous.vectors)

"Keep the eigenstates at the given indices, in the given order."
struct Indices <: SelectionRule
    list::Vector{Int}
end
indices(v::AbstractVector{<:Integer}) = Indices(collect(Int, v))

"""
    connected(core, (mode, port), thresh)
    connected(; core, port, thresh)

Keep `core` plus every state reachable from it through matrix elements of `port` on `mode`
with modulus above `thresh`, iterated to closure, sorted (Eli's `trunc_by_thresh`).
In the keyword form `port` is either `(mode::Symbol, p::Port)` or `(kind::Symbol, mode::Symbol)`
with `kind` one of `:n, :n2, :phi, :phi2, :expiphi, :exp2pin`.
"""
struct Connected <: SelectionRule
    core::Vector{Int}
    port::Tuple{Symbol,Port}
    thresh::Float64
end
connected(core::AbstractVector{<:Integer}, port::Tuple{Symbol,Port}, thresh::Real) =
    Connected(unique(Int.(core)), port, Float64(thresh))
connected(; core::AbstractVector{<:Integer}, port::Tuple{Symbol,Union{Symbol,Port}}, thresh::Real) =
    connected(core, _mode_port(port), thresh)

_mode_port(p::Tuple{Symbol,Port}) = p
_mode_port((kind, mode)::Tuple{Symbol,Symbol}) = (mode, port_of_kind(kind))

"""
    follow(reference=nothing; minoverlap=0.5)

Adiabatic following: keep one current eigenstate per reference column, in reference order.
The reference is the node's previous kept eigenvectors when a previous materialization exists,
transported into the current product basis through the children's physical overlaps (see
[`SelectionContext`](@ref)); the explicit `reference` matrix (columns in the node's product basis)
is used only when there is none.
Assignment is global greedy by overlap: all (reference, current) pairs are taken in order of
decreasing `|⟨ref|cur⟩|`, a pair being assigned when both states are still free.
A warning lists the reference states whose assigned overlap is below `minoverlap`.
A node with this rule defaults to the `:follow` phase convention.
"""
struct Follow <: SelectionRule
    reference::Union{Nothing,Matrix{ComplexF64}}
    minoverlap::Float64
end
follow(reference::Union{Nothing,AbstractMatrix}=nothing; minoverlap::Real=0.5) =
    Follow(reference === nothing ? nothing : Matrix{ComplexF64}(reference), Float64(minoverlap))

ports_needed(rule::Connected) = Tuple{Symbol,Port}[rule.port]

select(rule::Lowest, ctx::SelectionContext) =
    collect(1:(rule.k == 0 ? length(ctx.energies) : rule.k))

select(rule::Indices, ::SelectionContext) = copy(rule.list)

function select(rule::Connected, ctx::SelectionContext)
    haskey(ctx.ports, rule.port) ||
        throw(ArgumentError("connected selection needs port $(rule.port), which the context lacks"))
    P = ctx.ports[rule.port]
    n = size(P, 1)
    bad = filter(i -> !(1 <= i <= n), rule.core)
    isempty(bad) || throw(ArgumentError("connected selection with core $(rule.core): core states $bad lie outside 1:$n"))
    inlist = fill(false, n)
    list = unique(rule.core)
    inlist[list] .= true
    i = 1
    while i <= length(list)
        s = list[i]
        for j in 1:size(P, 2)
            if !inlist[j] && abs(P[s, j]) > rule.thresh
                inlist[j] = true
                push!(list, j)
            end
        end
        i += 1
    end
    return sort(list)
end

function select(rule::Follow, ctx::SelectionContext)
    R = if ctx.reference !== nothing
        ctx.reference
    elseif rule.reference !== nothing
        rule.reference
    else
        throw(ArgumentError("follow selection needs a reference or a previous solution"))
    end
    size(R, 1) == size(ctx.vectors, 1) ||
        throw(ArgumentError("follow selection: reference vectors have $(size(R, 1)) rows, current vectors $(size(ctx.vectors, 1))"))
    nref, ncur = size(R, 2), size(ctx.vectors, 2)
    nref <= ncur ||
        throw(ArgumentError("follow selection has $nref reference columns but only $ncur computed states"))
    O = abs.(R' * ctx.vectors)
    # Global greedy assignment: take (reference, current) pairs in order of decreasing overlap.
    order = sortperm(vec(O); rev=true)
    idx = zeros(Int, nref)
    taken = falses(ncur)
    nassigned = 0
    for lin in order
        r, c = Tuple(CartesianIndices(O)[lin])
        (idx[r] != 0 || taken[c]) && continue
        idx[r] = c
        taken[c] = true
        nassigned += 1
        nassigned == nref && break
    end
    weak = [r for r in 1:nref if O[r, idx[r]] < rule.minoverlap]
    isempty(weak) ||
        @warn "follow: overlap below minoverlap = $(rule.minoverlap) for reference states $weak"
    return idx
end

"""
    validate_selection(idx, compute, rule)

Check that `idx` is nonempty, has no duplicates, and lies in `1:compute`; return `idx`.
"""
function validate_selection(idx, compute::Int, rule)
    name = string(typeof(rule))
    isempty(idx) && throw(ArgumentError("$name selected no states"))
    allunique(idx) || throw(ArgumentError("$name selected duplicate states: $idx"))
    bad = filter(i -> !(1 <= i <= compute), idx)
    isempty(bad) || throw(ArgumentError("$name selected states $bad outside 1:$compute"))
    return idx
end
