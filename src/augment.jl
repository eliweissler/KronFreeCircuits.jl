# Basis augmentation: several sources of vectors on a child's own space are merged into one orthonormal basis.

"""
    BasisSource(label, vectors, energies; cap = nothing, window = nothing)

One source of vectors on a child's own product space, for example the lowest mean-field states of the child in one self-consistent solution.
The columns of `vectors` are unit vectors, lowest state first, and `energies` holds one energy per column.
At most `cap` columns are offered, and only the columns whose energy is within `window` of the first energy are offered.
The arrays are copied and converted to `ComplexF64` and `Float64`.
"""
struct BasisSource
    label::String
    vectors::Matrix{ComplexF64}
    energies::Vector{Float64}
    cap::Union{Nothing,Int}
    window::Union{Nothing,Float64}
end

function BasisSource(label::AbstractString, vectors::AbstractMatrix, energies::AbstractVector{<:Real};
                     cap::Union{Nothing,Integer} = nothing, window::Union{Nothing,Real} = nothing)
    size(vectors, 2) == length(energies) ||
        throw(ArgumentError("source \"$label\" has $(size(vectors, 2)) columns but $(length(energies)) energies"))
    cap === nothing || cap >= 0 || throw(ArgumentError("source \"$label\": `cap` must be at least 0, got $cap"))
    window === nothing || window >= 0 || throw(ArgumentError("source \"$label\": `window` must be at least 0, got $window"))
    return BasisSource(String(label), Matrix{ComplexF64}(vectors), Vector{Float64}(energies),
                       cap === nothing ? nothing : Int(cap), window === nothing ? nothing : Float64(window))
end

"""
    UnionReport(labels, taken, dropped)

What `union_basis` did with each source, in the order the sources were given.
`taken[i]` vectors of source `i` were accepted, and `dropped[i]` were offered and rejected as dependent on the vectors already accepted.
A column that was never reached because the merge stopped at `m` is neither taken nor dropped.
"""
struct UnionReport
    labels::Vector{String}
    taken::Vector{Int}
    dropped::Vector{Int}
end

function Base.show(io::IO, ::MIME"text/plain", r::UnionReport)
    print(io, "UnionReport")
    for i in eachindex(r.labels)
        print(io, "\n  ", r.labels[i], ": taken ", r.taken[i], ", dropped ", r.dropped[i])
    end
end
Base.show(io::IO, r::UnionReport) = show(io, MIME"text/plain"(), r)

# The number of columns of `s` that are offered: limited by `cap`, and by `window` (the scan stops at the first column outside).
function _offered(s::BasisSource)
    n = size(s.vectors, 2)
    s.cap === nothing || (n = min(n, s.cap))
    if s.window !== nothing && n > 0
        k = 1
        while k < n && s.energies[k + 1] - s.energies[1] <= s.window
            k += 1
        end
        n = k
    end
    return n
end

"""
    union_basis(sources; m, drop = 1e-6, allow_fewer = false) -> (U, report)

Merge `sources`, a vector of `BasisSource`, into one orthonormal basis `U` of `m` columns.
The merge is level by level: the first offered column of each source in the order given, then the second of each, and so on, until `m` vectors are accepted.
A candidate is orthogonalized against the accepted vectors by classical Gram–Schmidt applied twice, and accepted when the norm of the remainder is above `drop`; it is then normalized.
A source that is capped or windowed stops offering at its limit, so an uncapped source fills the remainder.
`report` is a `UnionReport` counting, for each source, the vectors accepted and the vectors offered and rejected.
Ending with fewer than `m` vectors is an `ArgumentError` that prints the report, unless `allow_fewer = true`, which returns the vectors found.
"""
function union_basis(sources::Vector{BasisSource}; m::Integer, drop::Real = 1e-6, allow_fewer::Bool = false)
    isempty(sources) && throw(ArgumentError("`union_basis` needs at least one source"))
    m >= 1 || throw(ArgumentError("`m` must be at least 1, got $m"))
    drop >= 0 || throw(ArgumentError("`drop` must be at least 0, got $drop"))
    labels = [s.label for s in sources]
    nrow = size(sources[1].vectors, 1)
    for s in sources
        size(s.vectors, 1) == nrow ||
            throw(ArgumentError("source \"$(s.label)\" has vectors of length $(size(s.vectors, 1)), " *
                                "but source \"$(sources[1].label)\" has vectors of length $nrow"))
        size(s.vectors, 2) == length(s.energies) ||
            throw(ArgumentError("source \"$(s.label)\" has $(size(s.vectors, 2)) columns but $(length(s.energies)) energies"))
        for j in axes(s.vectors, 2)
            abs(norm(view(s.vectors, :, j)) - 1) <= 1e-8 ||
                throw(ArgumentError("column $j of source \"$(s.label)\" is not a unit vector (norm $(norm(view(s.vectors, :, j))))"))
        end
    end
    offered = [_offered(s) for s in sources]
    taken = zeros(Int, length(sources))
    dropped = zeros(Int, length(sources))
    U = Matrix{ComplexF64}(undef, nrow, 0)
    count = 0
    level = 1
    while count < m && any(offered .>= level)
        for (i, s) in enumerate(sources)
            level <= offered[i] || continue
            v = Vector{ComplexF64}(s.vectors[:, level])
            for _ in 1:2
                v -= U * (U' * v)
            end
            nv = norm(v)
            if nv > drop
                U = hcat(U, v / nv)
                taken[i] += 1
                count += 1
                count == m && break
            else
                dropped[i] += 1
            end
        end
        level += 1
    end
    report = UnionReport(labels, taken, dropped)
    if count < m && !allow_fewer
        throw(ArgumentError("`union_basis` found only $count of the $m requested independent vectors; the rest of the offered vectors were dropped as dependent or not offered:\n" *
                            sprint(show, report) * "\npass `allow_fewer = true` to return the $count vectors found"))
    end
    return U, report
end

"""
    AugmentedChild

A child presented to its parent through a basis of its own choosing, the union basis `U`, instead of its ordinary solve.
`node` is the child, and `basis` is `U` on the child's own product space, one orthonormal column per vector.
`terms` are the child's bare terms, those `route` places at the child itself, with no mean-field terms.
`solved` holds the child's bare operator projected on `basis` and diagonalized: its vectors are `basis * W`, lowest energy first, all `size(basis, 2)` of them.
`solved` forms the ports and joint ports that its parent's terms need inside the child, as `child_keys` lists them.
`report` is the `UnionReport` of the merge that built `basis`.
Pass it to `materialize` with `augmented` to use it in place of the child's ordinary solve.
"""
struct AugmentedChild
    node::Node
    solved::MaterializedNode
    basis::Matrix{ComplexF64}
    terms::Vector{Term}
    report::UnionReport
end

"""
    augment(parent, terms, θ, child, bare, U, report) -> AugmentedChild

The child `child` of `parent` presented through the orthonormal basis `U` on its own product space.
`bare` is the child's bare solve (`bare_child`), whose children span the product space and are not solved again.
The child's bare operator `H`, on its own terms only and in complex arithmetic, is projected to `U' * H * U` and diagonalized densely.
Every column is kept, so the child presents `size(U, 2)` states, and its `compute` and `keep` settings do not apply.
The kept vectors are `U * W` with `W` the eigenvectors of the projection, with the child's phase convention and the ports and joint ports of `child_keys(parent, terms, child)`.
`report` is stored with the result, as is a copy of `U`.
The result is used as built, at these `terms` and `θ`: `materialize` does not check them against its own, and it is the caller's job to rebuild it when they change.
A union basis that is not made of eigenvectors of the child can trigger the parent's "heavy hybridization" phase warning, which reports on the phase convention and is not an error.
An `ArgumentError` is thrown when `bare` is not a solve of `child`, when `U` does not have one row per state of the product space or has no column, and when `U' * U` differs from the identity by more than `1e-8`.
"""
function augment(parent::Node, terms::Vector{Term}, θ::NamedTuple, child::Node, bare::MaterializedNode,
                 U::AbstractMatrix, report::UnionReport)
    bare.node === child || throw(ArgumentError("`bare` is not a solution of `child` (on modes $(modes(child)))"))
    keys = child_keys(parent, terms, child)
    n = prod(bare.dims)
    size(U, 1) == n ||
        throw(ArgumentError("the union basis of the child on modes $(modes(child)) has $(size(U, 1)) rows, " *
                            "but the child's product space has dimension $n"))
    size(U, 2) >= 1 || throw(ArgumentError("the union basis of the child on modes $(modes(child)) has no column"))
    basis = Matrix{ComplexF64}(U)                                      # a new array, also when `U` is one already
    dev = maximum(abs, basis' * basis - I)
    dev <= 1e-8 ||
        throw(ArgumentError("the union basis of the child on modes $(modes(child)) is not orthonormal: " *
                            "U' * U differs from the identity by up to $dev (largest deviation), above 1e-8"))
    bare_terms = _terms_at(child, terms)
    H = _node_operator(child, Dense(), bare.children, bare_terms, θ, 0.0; real=false)
    Hp = basis' * (H * basis)
    E, W = eigen(Hermitian((Hp + Hp') / 2))
    ports, joint_ports = _split_keys(keys)
    solved = _assemble_node(child, bare.children, Vector{Float64}(E), Matrix{ComplexF64}(basis * W);
                            ports, joint_ports, real=false)
    return AugmentedChild(child, solved, basis, bare_terms, report)
end

# The parent of every node under `root` except the root, after checking `augmented` against the tree: each entry an
# `AugmentedChild` whose node is in the tree and is not the root, no node given twice, no augmented node inside another,
# no `previous`, and neither an augmented node nor an ancestor of one with `arithmetic = :real`.
function _check_augmented(root::Node, augmented::AbstractVector, previous)
    previous === nothing ||
        throw(ArgumentError("`augmented` cannot be combined with `previous`: an augmented child has no previous solution to follow"))
    parents = IdDict{Node,Node}()
    for node in _all_nodes(root), ch in _subnodes(node)
        parents[ch] = node
    end
    seen = Base.IdSet{Node}()
    for a in augmented
        a isa AugmentedChild ||
            throw(ArgumentError("`augmented` must hold `AugmentedChild` entries (from `augment`), got a $(typeof(a))"))
        n = a.node
        n === root && throw(ArgumentError("the root (on modes $(modes(root))) cannot be augmented: an augmented node needs a parent"))
        haskey(parents, n) || throw(ArgumentError("the augmented node on modes $(modes(n)) is not a node of this tree"))
        n in seen && throw(ArgumentError("the node on modes $(modes(n)) is augmented twice"))
        push!(seen, n)
        n.arithmetic === :real &&
            throw(ArgumentError("the augmented node on modes $(modes(n)) has arithmetic = :real, but an augmented node " *
                                "runs in complex arithmetic"))
        p = n
        while haskey(parents, p)
            p = parents[p]
            p.arithmetic === :real &&
                throw(ArgumentError("the node on modes $(modes(p)) has arithmetic = :real, but it is an ancestor of the " *
                                    "augmented node on modes $(modes(n)), which runs in complex arithmetic"))
        end
    end
    for a in augmented, b in augmented
        p = a.node
        while haskey(parents, p)
            p = parents[p]
            p === b.node &&
                throw(ArgumentError("the augmented node on modes $(modes(a.node)) is inside the augmented node on modes " *
                                    "$(modes(b.node)), which is not solved, so the inner one would not be used"))
        end
    end
    return parents
end

"""
    augmented(sys) -> Vector{AugmentedChild}

The augmented children that `sys` was built with (see `materialize`), in the order given, as a new vector.
It is empty for a system built without them.
"""
augmented(sys::System) = AugmentedChild[a for a in sys.augmented]

# The augmented children of a tree by node, each with the ports and joint ports the tree needs of it
# (`_augmented_node`); every augmented node and every ancestor of one is marked complex in `realnodes`.
function _augmented_nodes!(realnodes::IdDict, parents::IdDict{Node,Node}, augmented::AbstractVector, needed, needed_joint)
    out = IdDict{Node,MaterializedNode}()
    for a in augmented
        out[a.node] = _augmented_node(a, needed[a.node], needed_joint[a.node])
        n = a.node
        realnodes[n] = false
        while haskey(parents, n)
            n = parents[n]
            realnodes[n] = false
        end
    end
    return out
end

# The solved node of `a` with the ports and joint ports `ports` and `joint_ports` that the tree needs of it: `a.solved`
# itself when it has them all, else a copy with the missing ones formed from its kept vectors (`a` is not modified).
function _augmented_node(a::AugmentedChild, ports::Vector{Tuple{Symbol,Port}}, joint_ports::Vector{_FactorKey})
    s = a.solved
    missing_ports = [k for k in ports if !haskey(s.ports, k)]
    missing_joint = [K for K in joint_ports if !haskey(s.joint_ports, K)]
    isempty(missing_ports) && isempty(missing_joint) && return s
    leaves = _mode_leaves(s.node)
    P = copy(s.ports)
    J = copy(s.joint_ports)
    for (mode, p) in missing_ports
        P[(mode, p)] = _node_port(s.node, leaves, s.children, s.dims, s.vectors, mode, p)
    end
    for K in missing_joint
        J[K] = _joint_port(s.node, leaves, s.children, s.dims, s.vectors, K)
    end
    return typeof(s)(s.node, s.dims, s.energies, s.vectors, s.kept, P, J, s.children, s.peak_component, s.peak_amplitudes)
end

"""
    mean_field_augment(parent, spec, θ; m, window = 0.5, max_solutions = 4, bare = false, cap = nothing,
                       starts = default_starts(parent), allow_fewer = false, kwargs...)
        -> (; children, solutions, used, energy, unconverged)

The augmented children of every child of `parent`, built from its mean-field solutions: a convenience over the pieces that adds no behaviour of its own.
It runs `mean_field_solutions(parent, spec, θ; starts, kwargs...)`, passing on every keyword not named here.
It uses the solutions whose energy is within `window` of the lowest (inclusive), in order, at most `max_solutions` of them.
For each child of `parent`, in order, it forms the child's `bare_child` and one `BasisSource` per used solution, labelled `"solution <index>"`.
That source holds the child's kept mean-field states in the solution, at most its first `m`, with their mean-field energies, and is limited by `cap`.
With `bare = true` a last source, labelled `"bare"`, holds the bare child's kept states with no limit.
It merges the sources with `union_basis(sources; m, allow_fewer)` and presents the child through the result with `augment`.
A used solution in which a child kept fewer than `min(m, cap)` mean-field states (`m` when `cap` is `nothing`) is an `ArgumentError`, raised before any union.
`children` holds the `AugmentedChild` of each child of `parent`, in order, and `solutions` every solution found.
`used` holds the indices in `solutions` of the solutions used.
`energy` is the lowest product energy, `solutions[1].energy`.
`unconverged` counts the used solutions that did not converge; they are used like the others, with no warning.
An `m < 1`, a `window < 0`, a `max_solutions < 1` and a `cap < 1` are `ArgumentError`s.
"""
function mean_field_augment(parent::Node, spec::Spec, θ::NamedTuple; m::Int, window::Real = 0.5, max_solutions::Int = 4,
                            bare::Bool = false, cap::Union{Nothing,Int} = nothing, starts = default_starts(parent),
                            allow_fewer::Bool = false, kwargs...)
    m >= 1 || throw(ArgumentError("`m` must be at least 1, got $m"))
    window >= 0 || throw(ArgumentError("`window` must be at least 0, got $window"))
    max_solutions >= 1 || throw(ArgumentError("`max_solutions` must be at least 1, got $max_solutions"))
    cap === nothing || cap >= 1 || throw(ArgumentError("`cap` must be at least 1 or `nothing`, got $cap"))
    terms = circuit_terms(spec)
    solutions = mean_field_solutions(parent, spec, θ; starts, kwargs...)
    used = first(Int[i for i in eachindex(solutions) if solutions[i].energy - solutions[1].energy <= window], max_solutions)
    unconverged = count(i -> !solutions[i].converged, used)
    need = cap === nothing ? m : min(m, cap)
    for i in used, ch in parent.children
        kept = size(solutions[i].children[ch].solved.vectors, 2)
        kept >= need || throw(ArgumentError("the child on modes $(modes(ch)) kept $kept mean-field states in solution $i, " *
                                            "but the union needs $need of them; raise the node's `compute`/`keep`"))
    end
    children = AugmentedChild[]
    for ch in parent.children
        b = bare_child(parent, terms, θ, ch)
        sources = BasisSource[]
        for i in used
            s = solutions[i].children[ch].solved
            k = min(m, size(s.vectors, 2))
            push!(sources, BasisSource("solution $i", s.vectors[:, 1:k], s.energies[1:k]; cap))
        end
        bare && push!(sources, BasisSource("bare", b.vectors, b.energies))
        U, report = union_basis(sources; m, allow_fewer)
        push!(children, augment(parent, terms, θ, ch, b, U, report))
    end
    return (; children, solutions, used, energy = solutions[1].energy, unconverged)
end
