using KrylovKit: KrylovKit

"""
    solve(solver, H, m) -> (vals::Vector{Float64}, vecs::Matrix{ComplexF64})

Eigenpairs of the Hermitian operator `H`, ascending.
`Dense` returns the lowest `m` pairs, or all when `m == 0` or `m >= size(H, 1)`;
`Lanczos` returns the lowest `m`.
"""
function solve end

function solve(::Dense, H::KronFreeOperator, m::Int)
    m >= 0 || throw(ArgumentError("Dense requires m >= 0, got $m"))
    n = size(H, 1)
    A = Hermitian(Matrix(H))
    F = 0 < m < n ? eigen!(A, 1:m) : eigen!(A)
    return Vector{Float64}(F.values), Matrix{ComplexF64}(F.vectors)
end

# Above this product dimension a Dense node is slow and memory-hungry (16 n² bytes); Lanczos is advised.
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
