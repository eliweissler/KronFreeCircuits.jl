# LOBPCG solver: LOBPCGEigensolver.lobpcg used through its public interface only
# (`lobpcg`, `precondprep!`, `ldiv!` and the `callback` keyword), with a fixed shifted
# Jacobi or an adaptive per-Ritz Jacobi preconditioner.
#
# Callback contract relied on (LOBPCGEigensolver 0.1.4, `lobpcg_impl.jl`): each iteration calls
# `precondprep!(P, X_active)` then `ldiv!(P, R_active)`, then counts newly locked columns (a
# prefix of the active ones, in order), then `callback(info)` with `info.λs` the Ritz values of
# the columns that were active during the iteration, before the active views shrink.

"""
    FixedJacobi(D)

Shifted Jacobi preconditioner: `ldiv!(P, R)` scales row `i` of `R` by `P.inv[i] = 1 / (D[i] − σ)`,
`σ = minimum(D) − 1` (a margin of one energy unit). `D` is the real diagonal of the Hamiltonian.
Throws `ArgumentError` when `D` is not finite or the shift is not representable strictly below
the diagonal (an enormous constant offset).
"""
struct FixedJacobi{V}
    inv::V
    function FixedJacobi(D::AbstractVector{<:Real})
        all(isfinite, D) || throw(ArgumentError("FixedJacobi: the diagonal has non-finite entries"))
        Dmin = minimum(D)
        σ = Dmin - 1
        σ < Dmin || throw(ArgumentError("FixedJacobi: the shift minimum(D) - 1 is not representable below " *
                                        "minimum(D) = $Dmin; remove the constant offset or use another preconditioner"))
        inv = 1 ./ (D .- σ)
        return new{typeof(inv)}(inv)
    end
end

function LinearAlgebra.ldiv!(P::FixedJacobi, R::AbstractMatrix)
    R .*= P.inv
    return R
end

_check_eta(eta) = eta > 0 && isfinite(eta) ||
    throw(ArgumentError("eta must be positive and finite, got $eta"))

"""
    RitzJacobi(D, capacity; eta, trace=false)

Adaptive per-Ritz Jacobi preconditioner: `ldiv!(P, R)` divides column `j` of the active residual
block by `hypot.(D .- θ_j, eta)`, with `D` the real diagonal of the Hamiltonian and `θ_j` the
cached energy of active column `j`. At the first `precondprep!` the cache is the diagonal
expectation `Σ_i D_i |X_ij|² / Σ_i |X_ij|²` of each column; afterwards it holds the previous
iteration's Ritz values, copied in by `_ritz_update!` from the solver callback (hand
`solver_callback(P, user)` to `lobpcg`), with the newly locked prefix dropped.
`capacity` is the full block width; `eta` is stored in the diagonal's real type. With `trace = true`, each `precondprep!` pushes
`(width, copy of the active cache)` to `P.trace`.
"""
mutable struct RitzJacobi{V,W,R<:Real}
    D::V
    theta::W
    eta::R
    locked::Int
    width::Int
    ready::Bool
    trace::Union{Nothing,Vector{Any}}
end

function RitzJacobi(D::AbstractVector{<:Real}, capacity::Integer; eta::Real, trace::Bool=false)
    _check_eta(eta)
    capacity >= 1 || throw(ArgumentError("RitzJacobi: capacity must be positive, got $capacity"))
    theta = fill!(similar(D, capacity), zero(eltype(D)))
    return RitzJacobi(D, theta, convert(real(eltype(D)), eta), 0, 0, false, trace ? Any[] : nothing)
end

"""
    _diagonal_expectation!(theta, D, X) -> theta

`theta[j] = Σ_i D[i] |X[i,j]|² / Σ_i |X[i,j]|²` for every column of `X`.
"""
function _diagonal_expectation!(theta::AbstractVector, D::AbstractVector, X::AbstractMatrix)
    # Generic (device) fallback: broadcast and reduce.
    A = abs2.(X)
    copyto!(theta, vec(sum(D .* A; dims=1) ./ sum(A; dims=1)))
    return theta
end

# Host arrays: a fused loop per column, with no N×k temporary.
function _diagonal_expectation!(theta::AbstractVector, D::Array, X::AbstractMatrix)
    size(X, 1) == length(D) || throw(DimensionMismatch("_diagonal_expectation!: X has $(size(X, 1)) rows, D has $(length(D))"))
    length(theta) == size(X, 2) || throw(DimensionMismatch("_diagonal_expectation!: theta has $(length(theta)) entries for $(size(X, 2)) columns"))
    R = promote_type(eltype(D), real(eltype(X)))
    for j in axes(X, 2)
        weighted = zero(R)
        weight = zero(R)
        @inbounds @simd for i in axes(X, 1)
            a = abs2(X[i, j])
            weighted += D[i] * a
            weight += a
        end
        theta[j] = weighted / weight
    end
    return theta
end

function LOBPCGEigensolver.precondprep!(P::RitzJacobi, X)
    b = size(X, 2)
    if !P.ready
        b <= length(P.theta) ||
            throw(DimensionMismatch("RitzJacobi: a block of $b columns exceeds the cache capacity $(length(P.theta))"))
        _diagonal_expectation!(view(P.theta, 1:b), P.D, X)
        P.ready = true
    else
        b == P.width || error("RitzJacobi: the solver prepares $b active columns but the cache holds $(P.width); " *
                              "pass solver_callback(P, …) to lobpcg so the cache follows locking")
    end
    P.width = b
    P.trace === nothing || push!(P.trace, (b, Array(view(P.theta, 1:b))))
    return P
end

function LinearAlgebra.ldiv!(P::RitzJacobi, R::AbstractMatrix)
    size(R, 2) == P.width ||
        throw(DimensionMismatch("RitzJacobi: prepared for $(P.width) columns, applied to $(size(R, 2))"))
    R ./= hypot.(P.D .- transpose(view(P.theta, 1:P.width)), P.eta)
    return R
end

# Copy the active Ritz values of this iteration into the cache, dropping the newly locked
# prefix: `info.λs` covers the columns active during the iteration, and the solver has already
# counted the columns it locks at its end. Values are copied, never aliased.
function _ritz_update!(P::RitzJacobi, info)
    b = P.width
    length(info.λs) == b ||
        error("RitzJacobi: the callback reports $(length(info.λs)) Ritz values for $b prepared columns")
    newly = info.n_locked - P.locked
    0 <= newly <= b || error("RitzJacobi: locking went from $(P.locked) to $(info.n_locked) columns with $b active")
    remaining = b - newly
    remaining > 0 && copyto!(view(P.theta, 1:remaining), view(info.λs, newly+1:b))
    P.locked = info.n_locked
    P.width = remaining
    return nothing
end

"""
    solver_callback(P, user)

The single callback handed to `lobpcg`: runs the user's callback (unless `nothing`), then the
Ritz cache update when `P` is a `RitzJacobi`. Both receive the original `info`; return values
are ignored (never compose with `∘`, which would feed the user's return value to the update).
"""
solver_callback(P, user) = _SolverCallback(P, user)

# A closure written as a callable struct: holding `user` untyped keeps lobpcg from compiling a
# new specialization for every user callback.
struct _SolverCallback{P}
    precon::P
    user::Any
end

function (cb::_SolverCallback)(info)
    cb.user === nothing || cb.user(info)
    cb.precon isa RitzJacobi && _ritz_update!(cb.precon, info)
    return nothing
end

_lobpcg_preconditioner(s::LOBPCG, D, width::Int) =
    s.preconditioner === :ritz ? RitzJacobi(D, width; eta=s.eta) :
    s.preconditioner === :jacobi ? FixedJacobi(D) : I

# A warm-start block with each column perturbed by a vector of norm sqrt(eps) times the
# column's norm, along a fixed pseudo-random unit direction (its own MersenneTwister, never the
# global RNG). An exactly zero residual column would otherwise become NaN in the eigensolver's
# orthogonalization.
function _perturbed_seed(T, X0)
    X = Matrix{T}(Array(X0))
    rng = Random.MersenneTwister(0x4b46)
    δ = sqrt(eps(real(T)))
    for j in axes(X, 2)
        c = view(X, :, j)
        r = randn(rng, T, length(c))
        c .+= (δ * norm(c) / norm(r)) .* r
    end
    return X
end

# Thrown inside the `catch` of the eigensolver's own error, which stays attached ("caused by").
_lobpcg_failed(err) =
    error("LOBPCG: the eigensolver failed ($(sprint(showerror, err))). An exactly zero residual block " *
          "(an operator proportional to the identity on the block, or exact eigenvectors) can break its " *
          "orthogonalization; use Dense() for such operators")

"""
    lobpcg_solve(s::LOBPCG, H::KronFreeOperator, m; X0=nothing) -> (E, V, result)

The lowest `m` eigenpairs of `H` (`E` ascending, host arrays) and `lobpcg`'s result for the
full block (`λ, X, residual_norms, residual_history, n_matvec, …`).
An operator not stored as `s.on` is copied there (data and scratch) for the solve.
`X0` is an `N × m` initial block (finite, no zero column) or `nothing` (random). `s.guard`
random columns are appended (shrunk to fit `N ≥ 3(m + g) + 5`). The eigensolver is asked to
converge every column of the block, guards included (`n_conv_check = m + g`), and the lowest `m`
of the sorted block are returned. If `maxiter` runs out with the lowest `m` converged but some
guard columns not, the pairs are returned with a warning that a lower state may have been missed.
Warm starts follow a bounded policy. First, each column of `X0` is perturbed by a vector of norm
`sqrt(eps)` times the column's norm, along a fixed pseudo-random unit direction (a seeded
`MersenneTwister`, not the global RNG), so an exact eigenvector seed no longer has the exactly zero residual that breaks the
eigensolver's orthogonalization. Second, if the eigensolver still throws on the warm-start
block, it is retried once from a random block with a warning. A failure on a random block raises
an error. Unconverged pairs raise an error and are never retried.
"""
function lobpcg_solve(s::LOBPCG, H::KronFreeOperator, m::Int; X0=nothing)
    if s.on !== Array && Base.get_extension(KronFreeCircuits, :KronFreeCircuitsCUDAExt) === nothing
        throw(ArgumentError("LOBPCG(on=$(s.on)) needs `using CUDA, cuTENSOR` so the KronFreeCircuits CUDA extension loads; " *
                            "without it the device path falls back to allocating methods"))
    end
    N = size(H, 1)
    m >= 1 || throw(ArgumentError("LOBPCG requires m >= 1, got $m"))
    N >= 3m + 5 || throw(ArgumentError("LOBPCG needs N ≥ 3m + 5, got N = $N for m = $m; " *
                                       "the block eigensolver is unreliable on so small a problem, use Dense()"))
    if X0 !== nothing
        size(X0) == (N, m) || throw(DimensionMismatch("LOBPCG: X0 has size $(size(X0)), expected ($N, $m)"))
        all(isfinite, X0) || throw(ArgumentError("LOBPCG: X0 has non-finite entries"))
        # The eigensolver's initial orthogonalization never terminates on an exactly zero column.
        for j in axes(X0, 2)
            iszero(view(X0, :, j)) && throw(ArgumentError("LOBPCG: column $j of X0 is zero"))
        end
    end
    T = eltype(H)
    rng = Random.default_rng()
    g = min(s.guard, (N - 5) ÷ 3 - m)
    w = m + g
    H = _on_storage(H, s.on; capacity=max(w, _block_capacity(H)))   # the operator lives with the block
    _block_capacity(H) < w && (H = with_block_capacity(H, w))
    D = adapt(s.on, _full_diagonal(H))
    # An exception from the user's callback propagates unchanged, without the recovery retry.
    user_threw = Ref(false)
    user = s.callback === nothing ? nothing : info -> try
        s.callback(info)
    catch
        user_threw[] = true
        rethrow()
    end

    function attempt(seed)
        X = Matrix{T}(undef, N, w)
        X[:, 1:m] .= seed
        randn!(rng, view(X, :, m+1:w))
        P = _lobpcg_preconditioner(s, D, w)
        # Every column, guards included, must converge: a lower state found by a guard then sorts
        # into the lowest m of the library's final λ instead of being cut off by early locking.
        return LOBPCGEigensolver.lobpcg(H, adapt(s.on, X), I, P, s.tol, s.maxiter;
                                        n_conv_check=w, callback=solver_callback(P, user))
    end

    res = try
        attempt(X0 === nothing ? randn(rng, T, N, m) : _perturbed_seed(T, X0))
    catch err
        (err isa InterruptException || user_threw[]) && rethrow()
        X0 === nothing && _lobpcg_failed(err)
        @warn "LOBPCG: the eigensolver failed on the warm-start block; retrying once from a random block" exception = err
        try
            attempt(randn(rng, T, N, m))
        catch err2
            (err2 isa InterruptException || user_threw[]) && rethrow()
            _lobpcg_failed(err2)
        end
    end

    resid = Array(res.residual_norms)
    bad = count(r -> !(r <= s.tol), view(resid, 1:m))
    bad == 0 || error("LOBPCG: $bad of $m eigenpairs did not converge within $(s.maxiter) iterations " *
                      "(only $(m - bad) converged; maximum residual $(maximum(view(resid, 1:m))) > tol = $(s.tol))")
    gbad = count(r -> !(r <= s.tol), view(resid, m+1:w))
    gbad == 0 || @warn "LOBPCG: $gbad of $g guard columns did not converge within $(s.maxiter) iterations; " *
                       "a lower state may have been missed"
    E = Vector{real(T)}(Array(res.λ)[1:m])
    V = Matrix{T}(Array(view(res.X, :, 1:m)))
    return E, V, res
end

solve(s::LOBPCG, H::KronFreeOperator, m::Int; X0=nothing) = lobpcg_solve(s, H, m; X0=X0)[1:2]
