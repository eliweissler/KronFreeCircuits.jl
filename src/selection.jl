"""
    SelectionContext(energies, vectors, ports, previous)

What a selection rule may inspect after a node is diagonalized: the computed `energies`,
the eigenvector columns `vectors`, the port matrices in the eigenbasis keyed by `(mode, port)`,
and `previous`, the node's previous `MaterializedNode` (or `nothing`).
"""
struct SelectionContext
    energies::Vector{Float64}
    vectors::Matrix{ComplexF64}
    ports::Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}
    previous::Any
end

"Keep the eigenstates at the given indices, in the given order."
struct Indices <: SelectionRule
    list::Vector{Int}
end
indices(v::AbstractVector{<:Integer}) = Indices(collect(Int, v))

"""
Keep `core` plus every state reachable from it through port matrix elements above `thresh`
(Eli's `trunc_by_thresh`).
"""
struct Connected <: SelectionRule
    core::Vector{Int}
    port::Tuple{Symbol,Port}
    thresh::Float64
end
connected(core::AbstractVector{<:Integer}, port::Tuple{Symbol,Port}, thresh::Real) =
    Connected(collect(Int, core), port, Float64(thresh))

"""
    follow(reference=nothing; minoverlap=0.5)

Adiabatic following: keep one current eigenstate per reference column, in reference order.
The reference is the node's previous kept eigenvectors when a previous materialization exists;
the explicit `reference` matrix (columns in the node's product basis) is used only when there is none.
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
    list = copy(rule.core)
    i = 1
    while i <= length(list)
        s = list[i]
        for j in 1:size(P, 2)
            if abs(P[s, j]) > rule.thresh && j ∉ list
                push!(list, j)
            end
        end
        i += 1
    end
    return sort(list)
end

function select(rule::Follow, ctx::SelectionContext)
    R = if ctx.previous !== nothing
        ctx.previous.vectors
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
