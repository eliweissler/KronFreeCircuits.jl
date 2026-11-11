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
        for (g, (a, _, Ka)) in enumerate(ops)
            dx[a] === nothing && continue
            replaced = copy(ops)
            replaced[g] = (a, _derivative_op(dx[a], Ka), Ka)
            P += _sandwich(mn, V, V, K, replaced)
        end
        return P
    end
    ports = Dict{Tuple{Symbol,Port},NodeMatrix}(key => derivative([key]) for key in keys(mn.ports))
    joint = Dict{_FactorKey,NodeMatrix}(K => derivative(K) for K in keys(mn.joint_ports))
    return ports, joint
end

"""
    _node_first_order(mn, H1, res, dx) -> NodeDerivative

The first derivatives of the node `mn` along one parameter, from `H1 = ∂H/∂x` (a `KronFreeOperator` on any storage),
the node's resolvent `res` and its children's derivative data `dx` (as given to `_derivative_operator`).
Energies by Hellmann–Feynman, `∂E_j = Re⟨ψ_j|H1|ψ_j⟩`.
States by `∂ψ_j = w_j + i a_j ψ_j`, with `w_j = Σ_{l≠j} ψ_l ⟨ψ_l|H1|ψ_j⟩ / (E_j − E_l)` and the gauge `a_j` that keeps
the peak amplitude `ℓ_j · ψ_j` real (zero on a real node).
A kept state degenerate with a state it couples to has `NaN` derivatives, with one warning per call.
The result's `children` is `dx` itself.
"""
function _node_first_order(mn::MaterializedNode{T}, H1::KronFreeOperator, res::Resolvent, dx::Vector{Any}) where {T}
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
    amps = _peak_amps(mn)
    L1 = nothing
    for (k, d) in enumerate(dx)
        d === nothing && continue
        a1 = copy(amps)
        a1[k] = d.peak_amplitudes
        L = _peak_map(mn, V, a1)
        L1 = L1 === nothing ? L : L1 + L
    end
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
