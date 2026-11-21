# Basis augmentation from mean-field solutions (experimental; see experimental/meanfield.jl and augment.jl).

"""
    mean_field_augment(parent, spec, θ; m, window = 0.5, max_solutions = 4, bare = false, cap = nothing,
                       starts = default_starts(parent), allow_fewer = false, kwargs...)
        -> (; children, solutions, used, energy, unconverged)

The augmented children of every child of `parent`, built from its mean-field solutions: a convenience over the pieces that adds no behaviour of its own.
It runs `mean_field_solutions(parent, spec, θ; starts, bare, kwargs...)`, with `bare` the bare solves below, passing on every keyword not named here.
It uses the solutions whose energy is within `window` of the lowest (inclusive), in order, at most `max_solutions` of them.
`window` is an energy in the spec's unit, and its default `0.5` assumes energies of order one (GHz).
It forms the `bare_child` of every child of `parent` once, before the loop, and passes them to `mean_field_solutions` as `bare`, so that the solutions' states, the union basis and `augment` share one product basis per child.
For each child of `parent`, in order, it forms one `BasisSource` per used solution, labelled `"solution <index>"`.
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
    _check_node_children(parent)
    # the bare solves are formed once: the solutions' states are coordinates on their children, and `augment` and the
    # "bare" source take the same objects
    bares = IdDict{Node,MaterializedNode}(ch => bare_child(parent, terms, θ, ch) for ch in parent.children)
    solutions = mean_field_solutions(parent, spec, θ; starts, bare=bares, kwargs...)
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
        b = bares[ch]
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
