# The first-order rule at one node: energies (Hellmann–Feynman), kept states (resolvent plus the gauge term of the
# phase convention), ports, joint ports and peak amplitudes.

# Per axis of `mn`: `nothing` for a basis child (its peak point is the basis index itself), else the child's
# `peak_amplitudes`.
_peak_amps(mn::MaterializedNode) = Any[c isa MaterializedNode ? c.peak_amplitudes : nothing for c in mn.children]

"""
    _peak_map(mn, X, amps) -> Matrix

The node's peak functionals applied to the columns of `X` (on the node's product basis):
`out[p, j] = Σ X[(i₁,…,iₙ), j] · amps[1][c₁(p), i₁] ⋯ amps[n][cₙ(p), iₙ]`, with `(c₁(p), …)` the product-basis index
`mn.peak_component[p]` of kept state `p`'s peak; `amps[a] === nothing` is the identity on axis `a`.
With `X = mn.vectors` and `amps = _peak_amps(mn)` it is the stored `mn.peak_amplitudes`.
"""
function _peak_map(mn::MaterializedNode, X::AbstractMatrix, amps::Vector{Any})
    length(amps) == length(mn.dims) ||
        throw(ArgumentError("$(length(amps)) amplitude matrices for a node with $(length(mn.dims)) axes"))
    all(isnothing, amps) && return X[mn.peak_component, :]
    S = promote_type(eltype(X), (eltype(A) for A in amps if A !== nothing)...)
    out = Matrix{S}(undef, length(mn.peak_component), size(X, 2))
    cis = CartesianIndices(Tuple(mn.dims))
    for (p, comp) in enumerate(mn.peak_component)
        ci = cis[comp]
        rows = [A === nothing ? _unit(mn.dims[a], ci[a]) : view(A, ci[a], :) for (a, A) in enumerate(amps)]
        out[p, :] = _contract_rows(X, rows)
    end
    return out
end
_unit(n::Int, i::Int) = (e = zeros(n); e[i] = 1.0; e)

"""
    _port_derivatives(mn, V1, dx) -> (ports, joint_ports)

The first derivative of every port and joint port `P = V† O V` of `mn`, given the derivative `V1` of its kept vectors
and its children's derivative data `dx` (per axis a `NodeDerivative` or `nothing`).
Each is `V1† O V + V† O V1` (formed separately, as `O` need not be Hermitian) plus, for each child operator of the key
whose child has derivative data, `V† O' V` with that operator replaced by the child's derivative.
"""
function _port_derivatives(mn::MaterializedNode, V1::AbstractMatrix, dx::Vector{Any})
    V = mn.vectors
    function derivative(K::_FactorKey)
        ops = _axis_ops(mn.node, mn.children, K)
        P = _sandwich(mn, V1, V, K, ops) + _sandwich(mn, V, V1, K, ops)
        for set in _first_replacements(_op_axes(ops), dx)
            P += _sandwich(mn, V, V, K, _replace_ops(ops, set))
        end
        return P
    end
    ports = Dict{Tuple{Symbol,Port},NodeMatrix}(key => derivative([key]) for key in keys(mn.ports))
    joint = Dict{_FactorKey,NodeMatrix}(K => derivative(K) for K in keys(mn.joint_ports))
    return ports, joint
end

"""
    _node_first_order(mn, H1, res, dx; boundary=nothing) -> NodeDerivative

The first derivatives of the node `mn` along one parameter, from `H1 = ∂H/∂x` (a `KronFreeOperator` on any storage),
the node's resolvent `res` and its children's derivative data `dx` (as given to `_derivative_operator`).
Energies by Hellmann–Feynman, `∂E_j = Re⟨ψ_j|H1|ψ_j⟩`.
States by `∂ψ_j = w_j + i a_j ψ_j`, with `w_j = Σ_{l≠j} ψ_l ⟨ψ_l|H1|ψ_j⟩ / (E_j − E_l)` and the gauge `a_j` that keeps
the peak amplitude `ℓ_j · ψ_j` real (zero on a real node).
A kept state degenerate with a state it couples to has `NaN` derivatives, with one warning per call.
The result's `children` is `dx` itself.
With `boundary = (; records, x, δx, r)` the selection-boundary test of `_selection_boundary!` is run for the parameter `x` and its records pushed to `records`.
"""
function _node_first_order(mn::MaterializedNode{T}, H1::KronFreeOperator, res::Resolvent, dx::Vector{Any};
                           boundary=nothing) where {T}
    V = mn.vectors
    E = mn.energies
    n = length(E)
    R = Array(H1 * adapt(H1.on, V))
    dE = Float64[real(dot(view(V, :, j), view(R, :, j))) for j in 1:n]
    W, clashes = _apply_resolvent(res, R, E, [[j] for j in 1:n])
    bad = findall(!isempty, clashes)
    if !isempty(bad)
        dE[bad] .= NaN
        @warn "Node on modes $(modes(mn.node)): kept states $bad are degenerate with states they couple to " *
              "($(join(("$j with $(clashes[j])" for j in bad), ", ")); 0 is an unkept state); their derivatives are NaN"
    end
    boundary === nothing ||
        _selection_boundary!(boundary.records, mn, res, R, clashes, boundary.x; δx=boundary.δx, r=boundary.r)
    amps = _peak_amps(mn)
    L1 = _peak_map_sum(mn, V, amps, _first_replacements(collect(eachindex(amps)), dx))
    if T === Float64
        gauge = zeros(n)
        V1 = Matrix{T}(W)
    else
        L0 = _peak_map(mn, W, amps)
        A = mn.peak_amplitudes
        gauge = Float64[-imag(L0[j, j] + (L1 === nothing ? 0 : L1[j, j])) / real(A[j, j]) for j in 1:n]
        V1 = Matrix{T}(W .+ V .* transpose(im .* gauge))
    end
    A1 = _peak_map(mn, V1, amps)
    L1 === nothing || (A1 = A1 + L1)
    ports, joint = _port_derivatives(mn, V1, dx)
    return NodeDerivative{T}(dE, V1, gauge, ports, joint, Matrix{T}(A1), dx)
end

# `Σ` over the replacement sets `sets` (as `_first_replacements` or `_second_replacements` give them, over the node's
# axes) of `_peak_map(mn, X, amps)` with the amplitudes of each set's axes replaced by its data's `peak_amplitudes`:
# the derivative of the peak functionals carried by the children; `nothing` when there is no set.
function _peak_map_sum(mn::MaterializedNode, X::AbstractMatrix, amps::Vector{Any}, sets)
    L = nothing
    for set in sets
        replaced = copy(amps)
        for (k, d) in set
            replaced[k] = d.peak_amplitudes
        end
        M = _peak_map(mn, X, replaced)
        L = L === nothing ? M : L + M
    end
    return L
end

"""
    _port_second_derivatives(mn, V2, Vx, Vλ, dx, dλ, dxλ) -> (ports, joint_ports)

The second derivative along `(x, λ)` of every port and joint port `P = V† O V` of `mn`, given the derivatives `Vx`,
`Vλ` and `V2` of its kept vectors and its children's first-order data `dx`, `dλ` and second-order data `dxλ` (per axis
a `NodeDerivative` or `nothing`).
With `S(W, U; O) = W† O U` (`_sandwich`), `O_x` the operator with one child operator replaced by the child's
`x`-derivative (summed over the entries), and `O_xλ` with one replaced by the child's second derivative or two distinct
ones by the `x`- and the `λ`-derivative (summed over the entries and the ordered pairs):
`S(V2, V; O) + S(V, V2; O) + S(Vx, Vλ; O) + S(Vλ, Vx; O) + S(Vx, V; O_λ) + S(V, Vx; O_λ) + S(Vλ, V; O_x) + S(V, Vλ; O_x) + S(V, V; O_xλ)`.
"""
function _port_second_derivatives(mn::MaterializedNode, V2::AbstractMatrix, Vx::AbstractMatrix, Vλ::AbstractMatrix,
                                  dx::Vector{Any}, dλ::Vector{Any}, dxλ::Vector{Any})
    V = mn.vectors
    function derivative(K::_FactorKey)
        ops = _axis_ops(mn.node, mn.children, K)
        axes = _op_axes(ops)
        S(W, U, o) = _sandwich(mn, W, U, K, o)
        P = S(V2, V, ops) + S(V, V2, ops) + S(Vx, Vλ, ops) + S(Vλ, Vx, ops)
        for set in _first_replacements(axes, dλ)
            o = _replace_ops(ops, set)
            P += S(Vx, V, o) + S(V, Vx, o)
        end
        for set in _first_replacements(axes, dx)
            o = _replace_ops(ops, set)
            P += S(Vλ, V, o) + S(V, Vλ, o)
        end
        for set in _second_replacements(axes, dx, dλ, dxλ)
            P += S(V, V, _replace_ops(ops, set))
        end
        return P
    end
    ports = Dict{Tuple{Symbol,Port},NodeMatrix}(key => derivative([key]) for key in keys(mn.ports))
    joint = Dict{_FactorKey,NodeMatrix}(K => derivative(K) for K in keys(mn.joint_ports))
    return ports, joint
end

"""
    _node_second_order(mn, Hx, Hλ, Hxλ, res, dX, dΛ, dxλ) -> NodeDerivative

The second derivatives of the node `mn` along the pair `(x, λ)` (`x == λ` allowed), from `Hx = ∂H/∂x`, `Hλ = ∂H/∂λ`,
`Hxλ = ∂²H/∂x∂λ` (`KronFreeOperator`s on any storage), the node's resolvent `res`, its own first-order data `dX`, `dΛ`
(whose `children` hold the children's) and its children's second-order data `dxλ`.
With `W^x = ∂_xV − i V diag(a^x)` the part of each first derivative orthogonal to its state (likewise `W^λ`), per kept
state `j`, `RHS = (Hx W^λ − W^λ diag(∂_xE)) + (Hλ W^x − W^x diag(∂_λE)) + Hxλ V`:
the energies are `∂²E_j = Re⟨ψ_j|RHS_j⟩`;
the part of `∂²ψ_j` orthogonal to `ψ_j` is `u_j`, the resolvent applied to `RHS_j` plus `i W^x_j a^λ_j + i W^λ_j a^x_j`;
and `∂²ψ_j = u_j + (n_j + i b_j) ψ_j`, with `n_j = −Re⟨W^x_j|W^λ_j⟩ − a^x_j a^λ_j` from the normalization and `b_j`
from the second derivative of the phase convention (`ℓ_j · ψ_j` real; zero on a real node).
The result's `gauge` is `b`, its `children` is `dxλ` itself.
A kept state degenerate with a state its right-hand side couples to, or already `NaN` at first order, has `NaN`
derivatives, with one warning per call.
"""
function _node_second_order(mn::MaterializedNode{T}, Hx::KronFreeOperator, Hλ::KronFreeOperator,
                            Hxλ::KronFreeOperator, res::Resolvent, dX::NodeDerivative, dΛ::NodeDerivative,
                            dxλ::Vector{Any}) where {T}
    V = mn.vectors
    E = mn.energies
    n = length(E)
    dx, dλ = dX.children, dΛ.children
    Vx, Vλ = dX.vectors, dΛ.vectors
    Wx = T === Float64 ? Vx : Vx .- V .* transpose(im .* dX.gauge)
    Wλ = T === Float64 ? Vλ : Vλ .- V .* transpose(im .* dΛ.gauge)
    apply(H, X) = Array(H * adapt(H.on, Matrix{T}(X)))
    RHS = apply(Hx, Wλ) .- Wλ .* transpose(dX.energies) .+ apply(Hλ, Wx) .- Wx .* transpose(dΛ.energies) .+ apply(Hxλ, V)
    undefined(d::NodeDerivative, j) = isnan(d.energies[j]) || any(isnan, view(d.vectors, :, j))
    first_bad = [j for j in 1:n if undefined(dX, j) || undefined(dΛ, j)]
    RHS[:, first_bad] .= 0                      # kept out of the resolvent (and of a linear solve)
    d2E = Float64[real(dot(view(V, :, j), view(RHS, :, j))) for j in 1:n]
    G, clashes = _apply_resolvent(res, RHS, E, [[j] for j in 1:n])
    clashing = findall(!isempty, clashes)
    bad = sort!(union(first_bad, clashing))
    if !isempty(bad)
        @warn "Node on modes $(modes(mn.node)): kept states $bad have no second derivatives (NaN): " *
              (isempty(clashing) ? "" : "states $clashing are degenerate with states they couple to at second order " *
               "($(join(("$j with $(clashes[j])" for j in clashing), ", ")); 0 is an unkept state)") *
              (isempty(clashing) || isempty(first_bad) ? "" : "; ") *
              (isempty(first_bad) ? "" : "states $first_bad are degenerate at first order")
    end
    normalization = Float64[-real(dot(view(Wx, :, j), view(Wλ, :, j))) - dX.gauge[j] * dΛ.gauge[j] for j in 1:n]
    amps = _peak_amps(mn)
    axes = collect(eachindex(amps))
    Lx = _peak_map_sum(mn, Vλ, amps, _first_replacements(axes, dx))                  # ℓ^x · ∂_λV
    Lλ = _peak_map_sum(mn, Vx, amps, _first_replacements(axes, dλ))                  # ℓ^λ · ∂_xV
    Lxλ = _peak_map_sum(mn, V, amps, _second_replacements(axes, dx, dλ, dxλ))       # ℓ^{xλ} · V
    carried = filter(!isnothing, Any[Lx, Lλ, Lxλ])
    if T === Float64
        gauge = zeros(n)
        V2 = Matrix{T}(G .+ V .* transpose(normalization))
    else
        U = G .+ im .* (Wx .* transpose(dΛ.gauge) .+ Wλ .* transpose(dX.gauge))
        L0 = _peak_map(mn, U, amps)
        A = mn.peak_amplitudes
        gauge = Float64[-imag(L0[j, j] + sum((L[j, j] for L in carried); init=0)) / real(A[j, j]) for j in 1:n]
        V2 = Matrix{T}(U .+ V .* transpose(normalization .+ im .* gauge))
        gauge[bad] .= NaN
    end
    d2E[bad] .= NaN
    V2[:, bad] .= NaN
    A2 = _peak_map(mn, V2, amps)
    for L in carried
        A2 = A2 + L
    end
    ports, joint = _port_second_derivatives(mn, V2, Vx, Vλ, dx, dλ, dxλ)
    return NodeDerivative{T}(d2E, V2, gauge, ports, joint, Matrix{T}(A2), dxλ)
end

"""
    _selection_boundary!(records, mn, res, R, clashes, x; δx, r) -> records

The selection-boundary test (design §3) of the node `mn` along the parameter `x`, over each kept state `i` and each other known state `l` of its resolvent `res`, with `R = (∂H/∂x) V` and `clashes` from `_apply_resolvent`.
The coupling `|⟨ψ_l|R[:, i]⟩|` of a pair is zero by the per-column rule of `_apply_resolvent`, `≤ 1e-10 ‖R[:, i]‖`.
With a scale `δx` the pair is recorded when its coupling is not zero and `coupling · δx > r |E_i − E_l|`; without one (`δx === nothing`) when `|E_i − E_l| ≤ 1e-6 max(|E_i − E_1|, |E_l − E_1|)`, with `E_1` the lowest known energy, and its `coupling` is `NaN`.
In either case a clash of column `i` with an unkept state (a `0` in `clashes[i]`) is recorded with its coupling, the partner recomputed by the same rule; it replaces a plain record of the same pair.
Each record is `(; node, parameter, kept, unkept, detuning, coupling)`, with `kept` a column of the node's vectors and `unkept` the state's index among the node's computed pairs; one `@warn` reports them all.
"""
function _selection_boundary!(records::Vector, mn::MaterializedNode, res::Resolvent, R::AbstractMatrix,
                              clashes::Vector{Vector{Int}}, x::Symbol; δx, r)
    E, Eo = res.energies, res.other_energies
    isempty(Eo) && return records
    Co = res.other_vectors' * R                 # as in `_apply_resolvent`
    E1 = min(minimum(E), minimum(Eo))
    found = NamedTuple[]
    for i in axes(R, 2)
        zero_overlap = 1e-10 * norm(view(R, :, i))
        unkept_clash = 0 in clashes[i]
        for l in eachindex(Eo)
            detuning = abs(E[i] - Eo[l])
            coupling = abs(Co[l, i])
            coupled = coupling > zero_overlap
            clash = unkept_clash && detuning <= res.degenerate_tol && coupled
            plain = δx === nothing ? detuning <= 1e-6 * max(abs(E[i] - E1), abs(Eo[l] - E1)) :
                                     coupled && coupling * δx > r * detuning
            (clash || plain) || continue
            push!(found, (; node=mn.node, parameter=x, kept=i, unkept=res.other_indices[l], detuning,
                          coupling=clash || δx !== nothing ? Float64(coupling) : NaN))
        end
    end
    isempty(found) && return records
    finite = [f.coupling for f in found if !isnan(f.coupling)]
    @warn "selection boundary: node on modes $(modes(mn.node)), parameter :$x: kept states close to unkept computed " *
          "states, (kept, unkept) = $(join(("($(f.kept), $(f.unkept))" for f in found), ", ")); smallest detuning " *
          "$(minimum(f.detuning for f in found)) GHz, largest coupling $(isempty(finite) ? NaN : maximum(finite)); " *
          "the fixed-selection derivative is fragile there"
    append!(records, found)
    return records
end

# The pass through the whole tree, and the API.

_pairkey(x::Symbol, λ::Symbol) = x <= λ ? (x, λ) : (λ, x)

"""
    Derivatives

The derivatives of a materialized tree, returned by [`derivatives`](@ref) and read through [`energies`](@ref),
[`eigenvectors`](@ref) and [`port`](@ref) with a parameter name.
`sys` is the differentiated [`System`](@ref), `wrt` the parameters (in the order processed), `order` the order, and
`mixed` the mixed pairs of second order.
`first[x]` is the root's first-order derivative data along `x` (a `NodeDerivative`, the children's nested in it), or
`nothing` when nothing in the tree depends on `x`.
`second` and `blocks` hold second-order data and near-degenerate blocks; this version of the pass leaves them empty.
`boundary` holds the selection-boundary records `(; node, parameter, kept, unkept, detuning, coupling)`: a kept state
(`kept`, a column of the node's vectors) close to an unkept computed state (`unkept`, its index among the node's
computed pairs), with `coupling` `NaN` when it was not computed.
`linear_solves` holds one record `(; modes, columns, iterations, residual)` per linear solve on the complement of a
node's computed states, with `residual` the largest relative true residual of its columns.
"""
struct Derivatives
    sys::System
    wrt::Vector{Symbol}
    order::Int
    mixed::Vector{Tuple{Symbol,Symbol}}
    first::Dict{Symbol,Any}
    second::Dict{Tuple{Symbol,Symbol},Any}
    blocks::Vector{Any}
    boundary::Vector{Any}
    linear_solves::Vector{Any}
end

"""
    derivatives(sys::System; wrt, order=1, mixed=(), scale=(;), r=0.1, degenerate_tol=1e-9, lintol=1e-10,
                linmaxiter=5000, given_independent=false) -> Derivatives

The first derivatives, along each parameter of `wrt`, of every kept energy, kept vector, port and joint port of every
node of the materialized tree `sys`, by perturbation theory carried upward through the tree.
They are the derivatives of the calculation as performed: the selection of kept states, the bases, the solvers, the
real or complex arithmetic of every node and its discrete phase choice (its peak indices, `peak_component`) are held
fixed, and nothing is re-solved.
Each node's `∂H/∂x` has three contributions: the derivatives of the coefficients of the terms routed to it (and, at the
root, of the constant), its children's port and joint-port derivatives, and its children's energy derivatives.
The element types are those of the stored quantities: `Float64` energies, and vectors and ports of the types
[`materialized`](@ref) stores them in.
A node whose subtree (and, at the root, the constant) does not depend on a parameter is not visited and reads as zeros.

Keywords:
- `wrt`: a parameter name or a collection of names, each a key of `sys.θ`; duplicates are removed.
- `order`: the order of the derivatives; only `1` is available.
- `mixed`: the pairs of parameters whose mixed second derivatives are wanted (second order); must be empty.
- `scale`: a `NamedTuple` of typical displacements `δx` of parameters in `wrt`, positive and finite, for the
  near-degeneracy test `|⟨ψ_l|∂H/∂x|ψ_i⟩| δx > r |E_i − E_l|`.
- `r`: the ratio of that test, positive.
  This version reports no near-degenerate blocks between kept states yet.
  It does run the selection-boundary test at every visited node: a kept state close to an unkept computed state (by
  the test above when `scale` has an entry for the parameter, else when their detuning is at most `1e-6` times their
  distance from the node's lowest computed energy), or degenerate with one it couples to, is recorded in `boundary`
  with one warning per node and parameter, since the derivative at a fixed selection is fragile there.
  States beyond a node's `compute` were never computed and cannot be tested.
- `degenerate_tol`: the detuning (GHz) at or below which two states are exactly degenerate; a kept state degenerate
  with a state it couples to has `NaN` derivatives.
- `lintol`, `linmaxiter`: the relative tolerance and iteration limit of the linear solve at a node without its
  complete spectrum (an iterative node): the part of each state derivative orthogonal to the node's computed states
  solves `(E_j − H) w_j = (∂H/∂x) ψ_j` on their orthogonal complement by a preconditioned conjugate gradient, whose true residual
  is checked and recorded in `linear_solves`; a solve that misses the tolerance is an error.
  A node with `retain = :never` projects out its kept states alone, which must then be its lowest computed ones.
- `given_independent`: declares the `GivenSystem` children independent of the parameters; a tree with a
  `GivenSystem` child is an `ArgumentError` in this version whatever its value, as is a tree with an augmented child.

A node whose phases followed a previous solution (`phase = :follow` with `previous`) is an `ArgumentError`.
"""
function derivatives(sys::System; wrt, order::Int=1, mixed=(), scale::NamedTuple=(;), r::Real=0.1,
                     degenerate_tol::Real=1e-9, lintol::Real=1e-10, linmaxiter::Int=5000,
                     given_independent::Bool=false, _parts=(:coefficients, :ports, :energies), _truncate::Bool=false)
    params = _check_wrt(wrt, sys.θ)
    order == 1 || throw(ArgumentError("order = $order is not available; only first derivatives (order = 1) are"))
    isempty(mixed) || throw(ArgumentError("`mixed` pairs need order = 2, which is not available"))
    for (k, v) in pairs(scale)
        k in params || throw(ArgumentError("`scale` has an entry for :$k, which is not in `wrt` ($(join(":" .* string.(params), ", ")))"))
        v isa Real && isfinite(v) && v > 0 ||
            throw(ArgumentError("`scale` for :$k must be a positive finite number, got $v"))
    end
    r > 0 || throw(ArgumentError("`r` must be positive, got $r"))
    _check_differentiable(sys)
    routed = route(sys.root.node, sys.terms)
    resolvents = IdDict{Any,Resolvent}()
    first = Dict{Symbol,Any}()
    boundary, linear_solves = Any[], Any[]
    lin = (; lintol, linmaxiter, log=linear_solves, truncate=_truncate)
    for x in params
        bd = (; records=boundary, x, δx=get(scale, x, nothing), r)
        first[x] = _first_order_pass(sys, routed, x, resolvents; degenerate_tol, parts=_parts, lin, boundary=bd)
    end
    return Derivatives(sys, params, order, Tuple{Symbol,Symbol}[], first, Dict{Tuple{Symbol,Symbol},Any}(), Any[],
                       boundary, linear_solves)
end

# `wrt` as a vector of distinct parameter names of `θ`, in order.
function _check_wrt(wrt, θ::NamedTuple)
    names = wrt isa Symbol ? Any[wrt] : (wrt isa Union{AbstractVector,Tuple,AbstractSet} ? collect(Any, wrt) :
            throw(ArgumentError("`wrt` must be a parameter name or a collection of names, got $(repr(wrt))")))
    isempty(names) && throw(ArgumentError("`wrt` names no parameter"))
    for x in names
        x isa Symbol || throw(ArgumentError("`wrt` must hold parameter names (Symbols), got $(repr(x))"))
        x in keys(θ) || throw(ArgumentError("`wrt` names :$x, which is not a parameter of the system; its parameters " *
                                            "are $(join(":" .* string.(keys(θ)), ", "))"))
    end
    return unique(Symbol[x for x in names])
end

# The tree may hold no node the pass cannot differentiate: a node whose phases followed a previous solution, a
# `GivenSystem` child or an augmented child.
function _check_differentiable(sys::System)
    isempty(sys.augmented) ||
        throw(ArgumentError("the node on modes $(modes(first(sys.augmented).node)) is an augmented child; derivatives " *
                            "of a tree with an augmented child are not available yet"))
    stack = MaterializedNode[sys.root]
    while !isempty(stack)
        mn = pop!(stack)
        mn.node isa GivenSystem &&
            throw(ArgumentError("the tree has the child GivenSystem :$(mn.node.mode); derivatives of a tree with a " *
                                "GivenSystem child are not available yet"))
        mn.retained.followed &&
            throw(ArgumentError("the phases of the node on modes $(modes(mn.node)) followed a previous solution, " *
                                "which cannot be differentiated; re-materialize at this point without `previous`"))
        append!(stack, reverse!([c for c in mn.children if c isa MaterializedNode]))       # axis order
    end
    return nothing
end

# The first-order pass along `x`: every node whose subtree depends on `x`, children first; the root's `NodeDerivative`,
# or `nothing`.  `resolvents` caches one `Resolvent` per node across parameters; `lin` holds the keywords of the linear
# solve given to `_resolvent`, and `boundary` those of the selection-boundary test (`_node_first_order`).
# At a node with a linear solve, `∂H/∂x` is built on the storage of the node's forward operator.
function _first_order_pass(sys::System, routed, x::Symbol, resolvents::IdDict{Any,Resolvent}; degenerate_tol, parts,
                           lin, boundary)
    root = sys.root.node
    depends = IdDict{Any,Bool}()
    function dependent(mn::MaterializedNode)
        haskey(depends, mn.node) && return depends[mn.node]
        here = any(t -> x in t.coeff.deps, routed[mn.node]) || (mn.node === root && x in sys.constant.deps)
        return depends[mn.node] = here || any(dependent(c) for c in mn.children if c isa MaterializedNode)
    end
    function visit(mn::MaterializedNode)
        dependent(mn) || return nothing
        dx = Any[c isa MaterializedNode ? visit(c) : nothing for c in mn.children]
        constant = mn.node === root ? sys.constant : Coefficient(0)
        res = get!(() -> _node_resolvent(sys, mn; degenerate_tol, lin...), resolvents, mn.node)
        on = res.complement === nothing ? Array : res.complement.H.on
        H1 = _derivative_operator(mn, routed[mn.node], sys.θ, x, dx; constant, parts, on)
        return _node_first_order(mn, H1, res, dx; boundary)
    end
    return visit(sys.root)
end

# The resolvent of `mn`, with its forward operator rebuilt only when `_resolvent` needs it: a node without its complete
# spectrum that does not keep every state of its space (a Dense one to re-diagonalize, any other for its linear solve,
# unless `truncate` leaves the solve out).
function _node_resolvent(sys::System, mn::MaterializedNode; degenerate_tol, lintol, linmaxiter, log, truncate)
    dense = _resolve_node_solver(mn.node, mn.children) isa Dense
    needs_H = !mn.retained.complete && size(mn.vectors, 2) < prod(mn.dims) && (dense || !truncate)
    return _resolvent(mn, needs_H ? operator(sys, mn.node) : nothing; degenerate_tol, lintol, linmaxiter, log, truncate)
end

"""
    _node_derivative(root_data, sys, node) -> NodeDerivative or nothing

The derivative data of `node` within the root's `root_data`, found by walking the materialized tree and the derivative
tree together; `nothing` when `node` (or an ancestor) was not visited.
"""
function _node_derivative(root_data, sys::System, node::Node)
    stack = Tuple{MaterializedNode,Any}[(sys.root, root_data)]
    while !isempty(stack)
        mn, d = pop!(stack)
        mn.node === node && return d
        for (a, c) in enumerate(mn.children)
            c isa MaterializedNode && push!(stack, (c, d === nothing ? nothing : d.children[a]))
        end
    end
    throw(ArgumentError("node is not part of this system's tree"))
end

function _first_data(d::Derivatives, x::Symbol)
    haskey(d.first, x) ||
        throw(ArgumentError("no first derivative along :$x was computed; available: $(join(":" .* string.(d.wrt), ", "))"))
    return d.first[x]
end

"""
    energies(d::Derivatives, x, node=root) -> Vector{Float64}

The first derivatives along the parameter `x` of the kept energies of `node`, absolute (GHz per unit of `x`): the
root's include the derivative of the constant, a child's do not.
They are derivatives of the calculation as performed, with the selection, bases, solvers, arithmetic and peak indices
held fixed (see [`derivatives`](@ref)).
"""
function energies(d::Derivatives, x::Symbol, node::Node=d.sys.root.node)
    nd = _node_derivative(_first_data(d, x), d.sys, node)
    return nd === nothing ? zeros(length(materialized(d.sys, node).energies)) : nd.energies
end

"""
    eigenvectors(d::Derivatives, x, node=root) -> Matrix

The first derivatives along `x` of the kept eigenvectors of `node` in its product basis, one column per kept state, in
the element type of the stored vectors, with the selection, bases, solvers, arithmetic and peak indices held fixed
(see [`derivatives`](@ref)); on a complex node they include the term that keeps the phase convention.
"""
function eigenvectors(d::Derivatives, x::Symbol, node::Node=d.sys.root.node)
    nd = _node_derivative(_first_data(d, x), d.sys, node)
    return nd === nothing ? zero(materialized(d.sys, node).vectors) : nd.vectors
end

"""
    port(d::Derivatives, x, mode, p, node=root) -> Matrix
    port(d::Derivatives, x, key::AbstractVector{Tuple{Symbol,Port}}, node=root) -> Matrix

The first derivative along `x` of the port `p` of `mode`, or of the joint port of the factors `key`, in the kept
eigenbasis of `node`, in the element type of the stored port, with the selection, bases, solvers, arithmetic and peak
indices held fixed (see [`derivatives`](@ref)).
A port or joint port the node did not form is an `ArgumentError`, as for [`port`](@ref) of a `System`.
"""
function port(d::Derivatives, x::Symbol, mode::Symbol, p::Port, node::Node=d.sys.root.node)
    data = _first_data(d, x)
    P = port(d.sys, mode, p, node)
    nd = _node_derivative(data, d.sys, node)
    return nd === nothing ? zero(P) : nd.ports[(mode, p)]
end

function port(d::Derivatives, x::Symbol, key::AbstractVector{Tuple{Symbol,Port}}, node::Node=d.sys.root.node)
    length(key) == 1 && return port(d, x, only(key)..., node)
    data = _first_data(d, x)
    P = port(d.sys, key, node)
    nd = _node_derivative(data, d.sys, node)
    return nd === nothing ? zero(P) : nd.joint_ports[_sortkey!(collect(Tuple{Symbol,Port}, key))]
end
