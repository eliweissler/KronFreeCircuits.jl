# The resolvent `(s − H)⁻¹` of one node, applied away from chosen kept states: the sum over the node's other known states.

"""
    Resolvent{T}

How `(s − H)⁻¹` is applied at one node, away from chosen kept states.
`energies` and `vectors` are the node's kept energies and vectors, as stored (phase-fixed).
`other_energies` and `other_vectors` are the other known eigenpairs of the node's eigenproblem: the retained unkept ones, or those of a re-diagonalization.
`complement` is `nothing` when the kept and other states together span the whole space of the node's eigenproblem; otherwise it is the linear solver on the orthogonal complement of the known states.
`degenerate_tol` (GHz) is the detuning at or below which two states are exactly degenerate.
"""
struct Resolvent{T<:Union{Float64,ComplexF64}}
    energies::Vector{Float64}
    vectors::Matrix{T}
    other_energies::Vector{Float64}
    other_vectors::Matrix{T}
    complement::Any
    degenerate_tol::Float64
end

"""
    _resolvent(mn, H=nothing; degenerate_tol=1e-9, projected=false) -> Resolvent

The resolvent of the node `mn`.
With `projected` (the kept states are declared the whole space, as at an augmented node) or when `mn` holds its complete spectrum, the other states are the retained unkept ones and nothing else is needed.
Otherwise a `Dense` node is re-diagonalized from its forward operator `H` (with an `@info`), and the other states are the eigenpairs at the positions not in `mn.kept`.
Any other node needs a linear solve on the complement of its computed states, which is not available yet (an `ArgumentError`).
"""
function _resolvent(mn::MaterializedNode{T}, H::Union{Nothing,KronFreeOperator}=nothing; degenerate_tol::Real=1e-9,
                    projected::Bool=false) where {T}
    r = mn.retained
    if projected || r.complete
        Vo = r.vectors === nothing ? Matrix{T}(undef, size(mn.vectors, 1), 0) : r.vectors
        return Resolvent{T}(copy(mn.energies), mn.vectors, r.energies[r.unkept], Vo, nothing, Float64(degenerate_tol))
    end
    N = prod(mn.dims)
    if _resolve_node_solver(mn.node, mn.children) isa Dense
        H === nothing &&
            throw(ArgumentError("the node on modes $(modes(mn.node)) did not retain its computed pairs; " *
                                "its forward operator is needed to re-diagonalize it"))
        @info "Node on modes $(modes(mn.node)) (dimension $N) did not retain its computed pairs; it is re-diagonalized " *
              "for its derivatives; set retain = :always on it to avoid this"
        F = eigen(Hermitian(Matrix(H)))
        other = setdiff(1:N, mn.kept)
        return Resolvent{T}(copy(mn.energies), mn.vectors, F.values[other], convert(Matrix{T}, F.vectors[:, other]),
                            nothing, Float64(degenerate_tol))
    end
    throw(ArgumentError("the node on modes $(modes(mn.node)) does not hold its complete spectrum; its derivatives need a " *
                        "linear solve on the complement of its computed states, which is not implemented yet"))
end

"""
    _apply_resolvent(res, R, shifts, exclude) -> (W, clashes)

`W[:, j] = Σ_l ψ_l ⟨ψ_l|R[:, j]⟩ / (shifts[j] − E_l)`, over every state `l` of the node's space except the kept states (column indices of `res.vectors`) listed in `exclude[j]`.
A state `l` with `|shifts[j] − E_l| ≤ res.degenerate_tol` is left out of the sum when its overlap is zero, `|⟨ψ_l|R[:, j]⟩| ≤ 1e-10 ‖R[:, j]‖`, and is otherwise a clash of column `j`; every other term is included whatever its detuning.
`clashes[j]` lists the clashing states of column `j`: a kept state by its column index, the other states (if any clash) by a single `0`.
Column `j` is defined exactly when `clashes[j]` is empty; an undefined column of `W` is `NaN`.
"""
function _apply_resolvent(res::Resolvent, R::AbstractMatrix, shifts::Vector{Float64}, exclude::Vector{Vector{Int}})
    res.complement === nothing ||
        throw(ArgumentError("the linear solve on the complement of the computed states is not implemented yet"))
    n = size(R, 2)
    length(shifts) == length(exclude) == n ||
        throw(DimensionMismatch("$n columns, $(length(shifts)) shifts and $(length(exclude)) exclusion lists"))
    Ck = res.vectors' * R                       # ⟨ψ_l|R[:, j]⟩, kept states
    Co = res.other_vectors' * R                 # the other states
    tol = res.degenerate_tol
    clashes = [Int[] for _ in 1:n]
    for j in 1:n
        zero_overlap = 1e-10 * norm(view(R, :, j))
        for l in axes(Ck, 1)
            if l in exclude[j]
                Ck[l, j] = 0
                continue
            end
            δ = shifts[j] - res.energies[l]
            if abs(δ) <= tol
                abs(Ck[l, j]) <= zero_overlap || push!(clashes[j], l)
                Ck[l, j] = 0
            else
                Ck[l, j] /= δ
            end
        end
        other_clash = false
        for l in axes(Co, 1)
            δ = shifts[j] - res.other_energies[l]
            if abs(δ) <= tol
                other_clash |= abs(Co[l, j]) > zero_overlap
                Co[l, j] = 0
            else
                Co[l, j] /= δ
            end
        end
        other_clash && push!(clashes[j], 0)
    end
    W = res.vectors * Ck + res.other_vectors * Co
    for j in 1:n
        isempty(clashes[j]) || (W[:, j] .= NaN)
    end
    return W, clashes
end
