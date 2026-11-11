# Grids per frame: a basis for each mode of a spec, sized from the mode's own scales, and a check of a solved grid leaf.

"""
    suggest_bases(spec, θ = (;); charge_cutoff = 16, max_points = 120, extent_scale = 1.0, order = 8) -> Vector{Pair{Symbol,Basis}}

A basis for each mode of `spec` at `θ`, as `mode => basis` pairs in the order of `spec.modes`.
A compact mode gets `ChargeBasis(charge_cutoff)`.
Any other mode `i` gets a symmetric `PhaseGrid`, sized from the real parts of the coefficients at `θ`.
With `E_C = E_C[i,i]` and `E_L = E_L[i,i]`, the zero-point width is `zpf = (2 E_C / E_L)^(1/4)` and the oscillator frequency `ω = sqrt(8 E_C E_L)`.
`ΣE_J` is the sum of `|energy|` over the cosines whose `omega[i]` is nonzero, and `k_max` the largest `|omega[i]|` among them.
The grid spans `±extent` with `extent = extent_scale * max(6 zpf, sqrt(2 (2 ΣE_J + 12 ω + 4 E_C) / E_L))`.
Its step is `min(zpf / 6, π / (6 k_max))`, or `zpf / 6` when no cosine touches the mode.
It has `min(ceil(2 extent / step) + 1, max_points)` points and finite differences of the given `order`.
A mode that is not compact and has `E_L[i,i] <= 0` or `E_C[i,i] <= 0` is an `ArgumentError`, as are `charge_cutoff < 1`, `max_points < 2` and `extent_scale <= 0`.
"""
function suggest_bases(spec::Spec, θ::NamedTuple = (;); charge_cutoff::Int = 16, max_points::Int = 120,
                       extent_scale::Real = 1.0, order::Int = 8)
    charge_cutoff >= 1 || throw(ArgumentError("`charge_cutoff` must be at least 1, got $charge_cutoff"))
    max_points >= 2 || throw(ArgumentError("`max_points` must be at least 2, got $max_points"))
    extent_scale > 0 || throw(ArgumentError("`extent_scale` must be positive, got $extent_scale"))
    out = Pair{Symbol,Basis}[]
    for (i, mode) in enumerate(spec.modes)
        if mode.type === :c
            push!(out, mode.name => ChargeBasis(charge_cutoff))
            continue
        end
        E_C = real(spec.E_C[i, i](θ))
        E_L = real(spec.E_L[i, i](θ))
        E_L > 0 || throw(ArgumentError("mode :$(mode.name) is not compact but has E_L[$i,$i] = $E_L at θ; " *
                                       "a phase grid needs E_L > 0"))
        E_C > 0 || throw(ArgumentError("mode :$(mode.name) has E_C[$i,$i] = $E_C at θ; a phase grid needs E_C > 0"))
        zpf = (2 * E_C / E_L)^(1 / 4)
        ω = sqrt(8 * E_C * E_L)
        touching = [c for c in spec.cosines if c.omega[i] != 0]
        ΣE_J = sum((abs(real(c.energy(θ))) for c in touching); init = 0.0)
        extent = extent_scale * max(6 * zpf, sqrt(2 * (2 * ΣE_J + 12 * ω + 4 * E_C) / E_L))
        step = isempty(touching) ? zpf / 6 : min(zpf / 6, π / (6 * maximum(abs(c.omega[i]) for c in touching)))
        points = min(ceil(Int, 2 * extent / step) + 1, max_points)
        push!(out, mode.name => PhaseGrid(-extent, extent, points; order))
    end
    return out
end

"""
    edge_weight(solved::MaterializedNode) -> Float64

The weight of the lowest kept state of a solved single-mode leaf on the ends of its `PhaseGrid`.
It is measured as the PhaseGrid edge warning of `materialize` measures it: the 2-norm of the state's entries on the first `order ÷ 2` and the last `order ÷ 2` grid points together.
The warning fires for a kept state whose measure exceeds `1e-6`.
A leaf on any other basis has nothing at an edge and gives `0.0`.
A node that is not a single-mode leaf is an `ArgumentError`.
"""
function edge_weight(solved::MaterializedNode)
    solved.node isa Node ||
        throw(ArgumentError("`edge_weight` needs a solved single-mode leaf, but the node is a GivenSystem"))
    length(solved.children) == 1 && only(solved.children) isa Basis ||
        throw(ArgumentError("`edge_weight` needs a solved single-mode leaf, but the node on modes " *
                            "$(modes(solved.node)) is not one"))
    g = only(solved.children)
    g isa PhaseGrid || return 0.0
    j = argmin(solved.energies)
    return _edge_support(solved.vectors[:, j:j], solved.dims, 1, g.order ÷ 2)[1]
end
