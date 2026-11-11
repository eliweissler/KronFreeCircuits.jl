# Grids per frame: a basis for each mode of a spec, sized from the mode's own scales, and a check of a solved grid leaf.

"""
    suggest_bases(spec, θ = (;); charge_cutoff = 16, max_points = 120, extent_scale = 1.0, order = 8,
                  when_capped = :warn) -> Vector{Pair{Symbol,Basis}}

A basis for each mode of `spec` at `θ`, as `mode => basis` pairs in the order of `spec.modes`.
A compact mode gets `ChargeBasis(charge_cutoff)`.
Any other mode `i` gets a symmetric `PhaseGrid`, sized from the real parts of the coefficients at `θ`.
With `E_C = E_C[i,i]` and `E_L = E_L[i,i]`, the zero-point width is `zpf = (2 E_C / E_L)^(1/4)` and the oscillator frequency `ω = sqrt(8 E_C E_L)`.
`ΣE_J` is the sum of `|energy|` over the cosines whose `omega[i]` is nonzero, and `k_max` the largest `|omega[i]|` among them.
The grid spans `±extent` with `extent = extent_scale * max(6 zpf, sqrt(2 (2 ΣE_J + 12 ω + 4 E_C) / E_L))`.
Its step is `min(zpf / 6, π / (6 k_max))`, or `zpf / 6` when no cosine touches the mode.
It has `min(ceil(2 extent / step) + 1, max_points)` points and finite differences of the given `order`.

The cap binds when the formula asks for more than `max_points` points on a mode.
The grid then keeps its extent with fewer points, so its step is larger than the formula's: the mode is under-resolved, the finite-difference kinetic energy is too low, and an energy computed on such a grid (a product energy, a level) is no longer an upper bound of the resolved problem's, and can go below its ground level.
`when_capped` says what a binding cap does: `:warn` (the default) logs one warning naming every capped mode with the points the formula asks for and the cap, `:error` throws an `ArgumentError` with the same content, and `:allow` caps silently, for a caller that caps on purpose.
The bases returned are the same in all three cases.
A mode that is not compact and has `E_L[i,i] <= 0` or `E_C[i,i] <= 0` is an `ArgumentError`, as are `charge_cutoff < 1`, `max_points < 2`, `extent_scale <= 0` and a `when_capped` other than `:warn`, `:error` and `:allow`.
"""
function suggest_bases(spec::Spec, θ::NamedTuple = (;); charge_cutoff::Int = 16, max_points::Int = 120,
                       extent_scale::Real = 1.0, order::Int = 8, when_capped::Symbol = :warn)
    charge_cutoff >= 1 || throw(ArgumentError("`charge_cutoff` must be at least 1, got $charge_cutoff"))
    max_points >= 2 || throw(ArgumentError("`max_points` must be at least 2, got $max_points"))
    extent_scale > 0 || throw(ArgumentError("`extent_scale` must be positive, got $extent_scale"))
    when_capped in (:warn, :error, :allow) ||
        throw(ArgumentError("`when_capped` must be :warn, :error or :allow, got :$when_capped"))
    out = Pair{Symbol,Basis}[]
    capped = Pair{Symbol,Int}[]                              # mode => the points the formula asks for
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
        asked = ceil(Int, 2 * extent / step) + 1
        asked > max_points && push!(capped, mode.name => asked)
        push!(out, mode.name => PhaseGrid(-extent, extent, min(asked, max_points); order))
    end
    if !isempty(capped) && when_capped !== :allow
        msg = "suggest_bases capped " * join(("the grid of mode :$m (the formula asks for $n points)" for (m, n) in capped), ", ") *
              " at max_points = $max_points: a capped grid's step is larger than the formula's, so its mode is " *
              "under-resolved and an energy on these bases is not an upper bound of the resolved problem's; raise " *
              "`max_points`, or pass `when_capped = :allow` to cap on purpose"
        when_capped === :error && throw(ArgumentError(msg))
        @warn msg
    end
    return out
end

"""
    edge_weight(solved::MaterializedNode) -> Float64

The amplitude of the lowest kept state of a solved single-mode leaf on the ends of its `PhaseGrid`; its square is the state's weight (probability) there.
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
