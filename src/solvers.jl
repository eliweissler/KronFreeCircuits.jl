using KrylovKit: KrylovKit

"""
    solve(solver, H, m) -> (vals::Vector{Float64}, vecs::Matrix{ComplexF64})

Eigenpairs of the Hermitian operator `H`, ascending.
`Dense` returns the lowest `m` pairs, or all when `m == 0` or `m >= size(H, 1)`;
`Lanczos`, `LOBPCG` and `ShiftInvert` return the lowest `m` (`ShiftInvert` with an interior `shift`: the `m` nearest it).
"""
function solve end

function solve(::Dense, H::KronFreeOperator, m::Int)
    m >= 0 || throw(ArgumentError("Dense requires m >= 0, got $m"))
    n = size(H, 1)
    A = Hermitian(Matrix(H))
    F = 0 < m < n ? eigen!(A, 1:m) : eigen!(A)
    return Vector{Float64}(F.values), Matrix{ComplexF64}(F.vectors)
end

# The column norms of `A`, as a real 1 × size(A, 2) array on `A`'s storage.
_colnorms(A::AbstractMatrix) = sqrt.(sum(abs2, A; dims=1))

# A small host array `M` (any wrapper: view, reshape, triangular) copied densely onto the storage of `like`.
_to_storage(like::AbstractArray, M::AbstractArray) = copyto!(similar(like, eltype(M), size(M)), Array(M))

"""
    _worst_true_residual(H, E, V) -> (worst, j)

The largest true residual norm `‖H v_j − E_j v_j‖` over the columns of `V` and its column, with a
fresh application of the package operator `H`; `V` lives on `H`'s storage, `E` is a host vector.
Shared acceptance check of the iterative solvers (each raises its own error).
"""
function _worst_true_residual(H::KronFreeOperator, E::AbstractVector, V::AbstractMatrix)
    R = H * V
    R .-= V .* _to_storage(V, reshape(E, 1, :))
    return findmax(vec(Array(_colnorms(R))))
end

# Above this product dimension a Dense node is slow and memory-hungry (16 n² bytes); LOBPCG is advised at a branch, Lanczos or a warm-started LOBPCG at a stiff leaf.
const _DENSE_WARN_DIM = 10_000
_dense_too_large(solver::Solver, n::Integer) = solver isa Dense && n > _DENSE_WARN_DIM

function solve(s::Lanczos, H::KronFreeOperator, m::Int)
    n = size(H, 1)
    m >= 1 || throw(ArgumentError("Lanczos requires m >= 1, got $m"))
    m <= n || throw(ArgumentError("Lanczos: m = $m exceeds the operator dimension $n"))
    krylovdim = max(2m + 10, s.krylovdim)
    vals, vecs, info = KrylovKit.eigsolve(v -> H * v, randn(ComplexF64, n), m, :SR;
                                          ishermitian=true, tol=s.tol, krylovdim=krylovdim,
                                          maxiter=s.maxiter, verbosity=s.verbosity)
    length(vals) >= m && info.converged >= m ||
        error("Lanczos: only $(info.converged) of $m eigenpairs converged (normres = $(info.normres))")
    E = real.(vals[1:m])
    V = reduce(hcat, vecs[1:m])
    p = sortperm(E)
    return Vector{Float64}(E[p]), Matrix{ComplexF64}(V[:, p])
end
