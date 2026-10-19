using KrylovKit: KrylovKit

"""
    Solver

The abstract type of node eigensolvers: [`Auto`](@ref) (the default), `Dense`, `Lanczos`, `LOBPCG`,
`ShiftInvert` and `ChebyshevFilter`. The contract below (spec §20e decision 6) is what a user-defined
solver implements; a user solver needs only `solve`.

- `KronFreeCircuits.solve(s, H, m; X0=nothing) -> (E, V)`: the lowest `m` eigenpairs of the Hermitian
  operator `H::KronFreeOperator`, with `E::Vector{Float64}` ascending and `V::Matrix{eltype(H)}` on the host
  (`size(H, 1) × m`, orthonormal columns, `V[:, j]` the eigenvector of `E[j]`). Every true residual
  `‖H V[:, j] − E[j] V[:, j]‖` must be below the solver's tolerance, checked with `H` itself before
  returning (an inner Krylov tolerance is not an acceptance criterion). `materialize` calls
  `solve(s, H, m)` for every solver except `LOBPCG`, which receives the previous block (`nothing` or
  `m` columns) as `X0`; a user solver may accept `X0=nothing` but need not.
- `KronFreeCircuits.storage(s)` (optional, default `Array`): the array type `materialize` builds `H`
  on; a device solver returns its device array type, and still returns `E` and `V` on the host.
- `KronFreeCircuits.resolve(s, info)` (optional, default `s`): the solver that actually runs, given
  `info = (; isleaf, dims, m)`; this is how [`Auto`](@ref) picks a solver by node shape.

`H` is real (`eltype(H) <: Real`) at a node that is real by structure (spec §20f), so a user solver must accept real operators and return real `V` for them.

What `H` provides: `mul!(Y, H, X)` and `H * X` on vectors and blocks of any width on `storage(s)`;
`size(H)` and `eltype(H)`; and, on the host, computed when asked for and memoized on the operator,
`diagonal(H)`, `sparse(H; maxnnz)` (from SparseArrays), `gershgorin_bound(H)`,
`separable_eigenbases(H)` and `product_seed(H, k)`. On `Array` storage `Matrix(H)` forms the dense matrix.
A node with `compute == 0` (all eigenpairs) accepts only a solver that is, or resolves to, `Dense`.
The `Node` constructor exempts only `Auto` from this check, so a user solver with a custom `resolve`
that resolves to `Dense` must set `compute` explicitly.
"""
Solver

"""
    solve(solver, H, m) -> (vals::Vector{Float64}, vecs::Matrix{eltype(H)})

Eigenpairs of the Hermitian operator `H`, ascending.
`Dense` returns the lowest `m` pairs, or all when `m == 0` or `m >= size(H, 1)`;
`Lanczos`, `LOBPCG` and `ShiftInvert` return the lowest `m` (`ShiftInvert` with an interior `shift`: the `m` nearest it).
See [`Solver`](@ref) for the contract a user-defined solver implements.
"""
function solve end

function solve(::Dense, H::KronFreeOperator, m::Int)
    m >= 0 || throw(ArgumentError("Dense requires m >= 0, got $m"))
    n = size(H, 1)
    A = Hermitian(Matrix(H))
    F = 0 < m < n ? eigen!(A, 1:m) : eigen!(A)
    return Vector{Float64}(F.values), Matrix{eltype(H)}(F.vectors)
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
    vals, vecs, info = KrylovKit.eigsolve(v -> H * v, randn(eltype(H), n), m, :SR;
                                          ishermitian=true, tol=s.tol, krylovdim=krylovdim,
                                          maxiter=s.maxiter, verbosity=s.verbosity)
    length(vals) >= m && info.converged >= m ||
        error("Lanczos: only $(info.converged) of $m eigenpairs converged (normres = $(info.normres))")
    E = real.(vals[1:m])
    V = reduce(hcat, vecs[1:m])
    p = sortperm(E)
    return Vector{Float64}(E[p]), Matrix{eltype(H)}(V[:, p])
end
