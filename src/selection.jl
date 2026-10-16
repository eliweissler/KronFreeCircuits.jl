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
Keep the states with largest overlap with the `reference` columns
(`nothing`: the node's previous eigenvectors), in reference order.
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
    R = if rule.reference !== nothing
        rule.reference
    elseif ctx.previous !== nothing
        ctx.previous.vectors
    else
        throw(ArgumentError("follow selection needs a reference or a previous solution"))
    end
    O = abs.(R' * ctx.vectors)
    assigned = falses(size(O, 2))
    idx = Int[]
    weak = Int[]
    for r in 1:size(O, 1)
        best, bestval = 0, -Inf
        for c in 1:size(O, 2)
            if !assigned[c] && O[r, c] > bestval
                best, bestval = c, O[r, c]
            end
        end
        best == 0 && throw(ArgumentError("follow selection has more reference columns than computed states"))
        assigned[best] = true
        push!(idx, best)
        bestval < rule.minoverlap && push!(weak, r)
    end
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
