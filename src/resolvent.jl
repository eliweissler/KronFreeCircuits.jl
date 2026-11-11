# The resolvent `(s − H)⁻¹` of one node, applied away from chosen kept states: the sum over the node's other known states.

"""
    Resolvent{T}

How `(s − H)⁻¹` is applied at one node, away from chosen kept states.
`energies` and `vectors` are the node's kept energies and vectors, as stored (phase-fixed).
`other_energies` and `other_vectors` are the other known eigenpairs of the node's eigenproblem: the retained unkept ones, or those of a re-diagonalization.
`other_indices` are their positions in the node's ascending spectrum (indices among its computed pairs).
`complement` is `nothing` when the kept and other states together span the whole space of the node's eigenproblem; otherwise it is the `_Complement`, the linear solve on the orthogonal complement of the known states.
`degenerate_tol` (GHz) is the detuning at or below which two states are exactly degenerate.
"""
struct Resolvent{T<:Union{Float64,ComplexF64}}
    energies::Vector{Float64}
    vectors::Matrix{T}
    other_energies::Vector{Float64}
    other_vectors::Matrix{T}
    other_indices::Vector{Int}
    complement::Any
    degenerate_tol::Float64
end

"""
    _Complement

The linear solve of an iterative node on the orthogonal complement of its known states (see [`_complement_solve`](@ref)).
`H` is the node's forward operator on its solver's storage, `D` its real diagonal there, `basis` the known states (kept, then retained unkept) on that storage.
`tol` and `maxiter` are the solver's limits, `log` is `nothing` or a vector that receives one record `(; modes, columns, iterations, residual, from_start)` per solve (see [`_apply_resolvent`](@ref)), and `modes` are the node's modes.
"""
struct _Complement
    H::KronFreeOperator
    D::AbstractVector
    basis::AbstractMatrix
    tol::Float64
    maxiter::Int
    log::Any
    modes::Vector{Symbol}
end

"""
    _resolvent(mn, H=nothing; degenerate_tol=1e-9, projected=false, lintol=1e-10, linmaxiter=5000, log=nothing,
               truncate=false) -> Resolvent

The resolvent of the node `mn`.
With `projected` (the kept states are declared the whole space, as at an augmented node), when `mn` holds its complete spectrum, or when it keeps every state of its space, the other states are the retained unkept ones (if any) and nothing else is needed.
Otherwise a `Dense` node is re-diagonalized from its forward operator `H` (with one `@info` per node), and the other states are the eigenpairs at the positions not in `mn.kept`, projected off the stored kept vectors (so that a selection boundary that cuts an exactly degenerate pair does not count the pair twice).
Any other node (an iterative one) gets a linear solve on the orthogonal complement of its known states, on its forward operator `H` (on its solver's storage), with the tolerance `lintol` and the iteration limit `linmaxiter`; each solve is recorded in `log`.
Its known states are the kept and the retained unkept ones; a node that did not retain its unkept vectors (`retain = :never`) has its kept states alone, which must then be its lowest computed ones (else an `ArgumentError`).
`truncate` leaves the complement out (the sum over the known states alone); it is wrong, and exists for one test.
"""
function _resolvent(mn::MaterializedNode{T}, H::Union{Nothing,KronFreeOperator}=nothing; degenerate_tol::Real=1e-9,
                    projected::Bool=false, lintol::Real=1e-10, linmaxiter::Int=5000, log=nothing,
                    truncate::Bool=false) where {T}
    r = mn.retained
    N = prod(mn.dims)
    k = size(mn.vectors, 2)
    # the retained unkept states, if any: their energies, vectors and positions in the spectrum
    idx = r.vectors === nothing ? Int[] : r.unkept
    Vo = r.vectors === nothing ? Matrix{T}(undef, size(mn.vectors, 1), 0) : r.vectors
    Eo = r.energies[idx]
    if projected || r.complete || k == N
        return Resolvent{T}(copy(mn.energies), mn.vectors, Eo, Vo, collect(idx), nothing, Float64(degenerate_tol))
    end
    if _resolve_node_solver(mn.node, mn.children) isa Dense
        H === nothing &&
            throw(ArgumentError("the node on modes $(modes(mn.node)) did not retain its computed pairs; " *
                                "its forward operator is needed to re-diagonalize it"))
        @info "Node on modes $(modes(mn.node)) (dimension $N) did not retain its computed pairs; it is re-diagonalized " *
              "for its derivatives; set retain = :always on it to avoid this" _id=(:kronfree_rediagonalized, objectid(mn.node)) maxlog=1
        F = eigen(Hermitian(Matrix(H)))
        other = setdiff(1:N, mn.kept)
        Vo = _other_vectors(convert(Matrix{T}, F.vectors), F.values, mn.vectors, mn.kept, other, degenerate_tol)
        return Resolvent{T}(copy(mn.energies), mn.vectors, F.values[other], Vo, other, nothing, Float64(degenerate_tol))
    end
    r.vectors === nothing && mn.kept != 1:k &&
        throw(ArgumentError("the node on modes $(modes(mn.node)) did not retain its unkept computed vectors " *
                            "(retain = :never) and keeps states $(mn.kept), not its $k lowest: the complement of its " *
                            "kept states is not an invariant subspace above them; set retain = :auto or :always on it"))
    truncate && return Resolvent{T}(copy(mn.energies), mn.vectors, Eo, Vo, collect(idx), nothing, Float64(degenerate_tol))
    H === nothing &&
        throw(ArgumentError("the node on modes $(modes(mn.node)) does not hold its complete spectrum; its forward " *
                            "operator is needed for the linear solve on the complement of its computed states"))
    Hk = with_block_capacity(H, k)
    basis = adapt(H.on, hcat(mn.vectors, Vo))
    c = _Complement(Hk, _full_diagonal(Hk), basis, Float64(lintol), linmaxiter, log, collect(Symbol, modes(mn.node)))
    return Resolvent{T}(copy(mn.energies), mn.vectors, Eo, Vo, collect(idx), c, Float64(degenerate_tol))
end

# The re-diagonalization's vectors at the positions `other`, made orthogonal to the stored kept vectors `K` (at the
# positions `kept`): where the selection boundary cuts an exactly degenerate cluster (energies within `tol` of each
# other), the re-diagonalization's partners of a kept state need not be orthogonal to it and would count part of it
# twice; within each such cluster the partners are replaced by an orthonormal basis of the cluster's span with the
# stored kept vectors projected out. Elsewhere the vectors are the re-diagonalization's.
function _other_vectors(U::Matrix{T}, values::Vector{Float64}, K::AbstractMatrix, kept::Vector{Int}, other::Vector{Int},
                        tol::Real) where {T}
    Vo = U[:, other]
    col = Dict(l => c for (c, l) in enumerate(other))
    for k in kept
        cluster = findall(e -> abs(e - values[k]) <= tol, values)
        partners = [l for l in cluster if haskey(col, l)]
        isempty(partners) && continue
        P = U[:, cluster]
        P .-= K * (K' * P)
        Uc = svd(P).U[:, 1:length(partners)]           # the cluster's span less the kept states, orthonormal
        Vo[:, [col[l] for l in partners]] .= Uc
    end
    return Vo
end

# Re⟨A_j, B_j⟩ per column, on the host.
_colredot(A::AbstractMatrix, B::AbstractMatrix) = vec(Array(real.(sum(conj.(A) .* B; dims=1))))

"""
    _complement_solve(H, D, basis, B, shifts; tol, maxiter, eta=0.01, scale=norms of B) -> (X, iterations, residuals, from_start)

`X[:, j]` solves `Q (H − shifts[j]) Q x = Q B[:, j]` with `x ⟂` the columns of `basis`, `Q = 1 − basis basis'`, by a preconditioned conjugate gradient on all columns in lockstep, each with its own scalars, preconditioned by the Jacobi diagonal `1 / hypot(D − shifts[j], eta)`.
`basis` (orthonormal columns) and `B` live on `H`'s storage, `D` is `H`'s real diagonal there, and `X` is returned on that storage.
`B` is best given already on the complement (its known-state components subtracted by the caller, as `_apply_resolvent` does); it is projected again here.
`scale[j]` is the norm of the whole right-hand side of column `j` before its known-state components were subtracted (`‖R_j‖`, the scale of the whole derivative); it defaults to the column norms of `B`.
Convergence is measured against it: a column is converged when its recurrence residual is at most `tol scale[j]`, because the round-off of the projection (about `eps ‖R_j‖`) cannot go below a tolerance relative to `‖Q B_j‖` when `Q B_j` is a small part of `R_j`.
A column with `‖Q B[:, j]‖ ≤ tol scale[j]` is converged from the start with `x_j = 0` exactly; `from_start` lists those columns.
`residuals[j]` is the relative true residual `‖Q (H − shifts[j]) X[:, j] − Q B[:, j]‖ / scale[j]`, recomputed with a fresh application of `H` after the loop (for a column converged from the start, `‖Q B[:, j]‖ / scale[j]`, and `0` when `scale[j] = 0`), and must be at most `10 tol`.
Throws an error containing "did not converge" when a column misses the tolerance (in the loop or in the final check), and one containing "not positive on the complement" when `H − shifts[j]` is not positive there on a column that has not converged.
"""
function _complement_solve(H::KronFreeOperator, D::AbstractVector, basis::AbstractMatrix, B::AbstractMatrix,
                           shifts::Vector{Float64}; tol::Real, maxiter::Int, eta::Real=0.01,
                           scale::Union{Nothing,AbstractVector{<:Real}}=nothing)
    n = size(B, 2)
    length(shifts) == n || throw(DimensionMismatch("$n columns and $(length(shifts)) shifts"))
    scale = scale === nothing ? vec(Array(_colnorms(B))) : collect(Float64, scale)
    length(scale) == n || throw(DimensionMismatch("$n columns and $(length(scale)) scales"))
    C = similar(B, size(basis, 2), n)
    project!(Y) = (mul!(C, basis', Y); mul!(Y, basis, C, -1, 1); Y)          # Y − Φ (Φ' Y), in place
    S = _to_storage(B, reshape(shifts, 1, :))
    Minv = 1 ./ hypot.(D .- S, eta)
    row(v) = _to_storage(B, reshape(v, 1, :))
    goal = tol .* scale                         # the absolute residual each column must reach
    QB = project!(copy(B))
    b = vec(Array(_colnorms(QB)))
    converged = b .<= goal
    from_start = findall(converged)
    QB .*= row(Float64.(.!converged))           # x_j = 0 exactly for those columns
    X = fill!(similar(B), 0)
    Rr = copy(QB)
    Z = project!(Minv .* Rr)
    P = copy(Z)
    ρ = _colredot(Rr, Z)
    AP = similar(B)
    rn = copy(b)
    iterations = 0
    while !all(converged) && iterations < maxiter
        iterations += 1
        mul!(AP, H, P)
        AP .-= P .* S
        project!(AP)
        κ = _colredot(P, AP)
        for j in 1:n
            converged[j] || κ[j] > 0 ||
                error("the linear solve on the complement is not positive on the complement at column $j " *
                      "(⟨p, (H − s) p⟩ = $(κ[j]) after $iterations iterations, residual $(rn[j] / scale[j]) relative to " *
                      "the right-hand side): a state below or degenerate with the kept state was not computed (or was " *
                      "not retained); raise `compute` (or retain the node's computed states)")
        end
        α = [converged[j] ? 0.0 : ρ[j] / κ[j] for j in 1:n]
        X .+= P .* row(α)
        Rr .-= AP .* row(α)
        rn = vec(Array(_colnorms(Rr)))
        converged .|= rn .<= goal
        all(converged) && break
        Z .= Minv .* Rr
        project!(Z)
        ρn = _colredot(Rr, Z)
        β = [converged[j] ? 0.0 : ρn[j] / ρ[j] for j in 1:n]
        P .= Z .+ P .* row(β)
        ρ = ρn
    end
    if !all(converged)
        j = findfirst(!, converged)
        error("the linear solve on the complement did not converge at column $j: residual $(rn[j] / scale[j]) " *
              "relative to the right-hand side after $iterations iterations (tolerance $tol, at most $maxiter iterations)")
    end
    # the true residual, with a fresh application of H
    mul!(AP, H, X)
    AP .-= X .* S
    project!(AP)
    AP .-= QB
    true_res = vec(Array(_colnorms(AP)))
    residuals = [j in from_start ? (scale[j] == 0 ? 0.0 : b[j] / scale[j]) : true_res[j] / scale[j] for j in 1:n]
    for j in 1:n
        residuals[j] <= 10 * tol ||
            error("the linear solve on the complement did not converge at column $j: its true residual is " *
                  "$(residuals[j]) relative to the right-hand side after $iterations iterations, above 10 × $tol")
    end
    return X, iterations, residuals, from_start
end

"""
    _apply_resolvent(res, R, shifts, exclude) -> (W, clashes)

`W[:, j] = Σ_l ψ_l ⟨ψ_l|R[:, j]⟩ / (shifts[j] − E_l)`, over every state `l` of the node's space except the kept states (column indices of `res.vectors`) listed in `exclude[j]`.
At a node with a `complement`, the part on the orthogonal complement of the known states is the solution of `Q (H − shifts[j]) Q x = −Q R[:, j]` (see [`_complement_solve`](@ref)).
Its right-hand side is formed on the host as `R[:, j]` minus its components on every known state (kept and other), from the overlaps the sum above already uses: one cancellation, not a projection of a nearly parallel vector, since for a parameter internal to a child `R[:, j]` lies almost inside the span of the known states.
Its convergence is measured against `‖R[:, j]‖`; it is moved to the operator's storage for the solve and the result back to the host, and the solve is recorded in the complement's `log` as `(; modes, columns, iterations, residual, from_start)`: `residual` the largest true residual relative to `‖R[:, j]‖`, `from_start` the columns converged from the start (their complement part already below the tolerance, with no solve).
A state `l` with `|shifts[j] − E_l| ≤ res.degenerate_tol` is left out of the sum when its overlap is zero, `|⟨ψ_l|R[:, j]⟩| ≤ 1e-10 ‖R[:, j]‖`, and is otherwise a clash of column `j`; every other term is included whatever its detuning.
`clashes[j]` lists the clashing states of column `j`: a kept state by its column index, the other states (if any clash) by a single `0`.
Column `j` is defined exactly when `clashes[j]` is empty and `R[:, j]` is finite; an undefined column of `W` is `NaN`.
A column of `R` that is not finite (a child's derivative is undefined) is left out of the sum and of the solve, has no clash, and is `NaN` in `W`.
"""
function _apply_resolvent(res::Resolvent, R::AbstractMatrix, shifts::Vector{Float64}, exclude::Vector{Vector{Int}})
    n = size(R, 2)
    length(shifts) == length(exclude) == n ||
        throw(DimensionMismatch("$n columns, $(length(shifts)) shifts and $(length(exclude)) exclusion lists"))
    nonfinite = [j for j in 1:n if !all(isfinite, view(R, :, j))]
    if !isempty(nonfinite)
        R = copy(R)
        R[:, nonfinite] .= 0
    end
    Ck = res.vectors' * R                       # ⟨ψ_l|R[:, j]⟩, kept states
    Co = res.other_vectors' * R                 # the other states
    c = res.complement
    # the part of R on the complement of the known states, with one cancellation
    Rc = c === nothing ? nothing : R - res.vectors * Ck - res.other_vectors * Co
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
    if c !== nothing
        scale = [norm(view(R, :, j)) for j in 1:n]
        B = adapt(c.H.on, convert(Matrix{eltype(c.basis)}, -Rc))
        X, iterations, residuals, from_start = try
            _complement_solve(c.H, c.D, c.basis, B, shifts; tol=c.tol, maxiter=c.maxiter, scale)
        catch e
            e isa ErrorException ? error("node on modes $(c.modes): " * e.msg) : rethrow()
        end
        W .+= Array(X)
        c.log === nothing ||
            push!(c.log, (; modes=c.modes, columns=n, iterations, residual=maximum(residuals; init=0.0), from_start))
    end
    for j in 1:n
        (isempty(clashes[j]) && !(j in nonfinite)) || (W[:, j] .= NaN)
    end
    return W, clashes
end
