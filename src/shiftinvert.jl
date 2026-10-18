# ShiftInvert leaf solver (spec §20e decision 3): KrylovKit Lanczos on (H − σI)⁻¹ with H − σI
# factored directly (CHOLMOD Cholesky, or UMFPACK LU) from the assembled sparse matrix, in real
# arithmetic when the assembled matrix is exactly real. The Krylov tolerance applies to the inverted
# operator; acceptance is the true residual against the package operator.

"""
    isreal_assembled(H::KronFreeOperator; maxnnz=10^8) -> Bool

Whether the assembled matrix `sparse(H; maxnnz)` and the fused diagonal `H.diag` have exactly zero
imaginary part: an exact check on assembled data, never a threshold. [`ShiftInvert`](@ref) runs in
real arithmetic exactly when this holds.
"""
isreal_assembled(H::KronFreeOperator; maxnnz::Integer=10^8) = _isreal_assembled(sparse(H; maxnnz=maxnnz), H.diag)

_isreal_assembled(A::SparseMatrixCSC, diag::AbstractVector) =
    all(x -> iszero(imag(x)), diag) && all(x -> iszero(imag(x)), nonzeros(A))

# The tolerance of the short Lanczos estimate of the lowest eigenvalue that places the automatic shift.
const _EMIN_TOL = 1e-6

# A Ritz value of the lowest eigenvalue of `A` (an upper bound on it, converged or not), from a short
# Lanczos run in `A`'s arithmetic.
function _lowest_estimate(A::AbstractMatrix, maxiter::Int)
    vals, _, _ = KrylovKit.eigsolve(x -> A * x, randn(eltype(A), size(A, 1)), 1, :SR;
                                    ishermitian=true, tol=_EMIN_TOL, maxiter=maxiter, verbosity=0)
    return real(first(vals))
end

# Cholesky of A − σI when σ lies below the estimate, falling back to LU when it is not positive
# definite; LU always for an interior σ. Returns the factorization and whether it is a Cholesky.
function _factor_shifted(A::SparseMatrixCSC, σ::Real, interior::Bool)
    if !interior
        try
            return cholesky(Hermitian(A); shift=-σ), true
        catch err
            err isa PosDefException || rethrow()
        end
    end
    return lu(A - σ * I), false
end

# The number of eigenvalues of A below a cut: the negative entries of D in the LDLᵀ factorization
# of A − τI (Sylvester's law of inertia), at the first cut of `τs` whose factorization succeeds.
# CHOLMOD's LDLᵀ is simplicial with no pivoting: D is diagonal (1×1 pivots only, no 2×2 blocks to
# resolve), and a tiny pivot can break it down even on a well-conditioned indefinite matrix, so
# callers pass a second cut elsewhere in the same gap to retry once. Returns `(count, τ)`.
function _count_below(A::SparseMatrixCSC, τs)
    for τ in τs
        F = ldlt(Hermitian(A); shift=-τ, check=false)
        issuccess(F) && return count(d -> real(d) < 0, diag(sparse(F.LD))), τ
    end
    error("ShiftInvert: the inertia certificate could not be computed: the LDLᵀ factorization (no pivoting) " *
          "of H − τI broke down at every cut τ ∈ $(collect(τs))")
end

function solve(s::ShiftInvert, H::KronFreeOperator, m::Int)
    H.on === Array || throw(ArgumentError("ShiftInvert is a CPU solver, but the operator is stored on $(H.on); " *
                                          "use LOBPCG for device storage"))
    s.slices == 1 || throw(ArgumentError("slicing arrives in Task 3"))
    n = size(H, 1)
    m >= 1 || throw(ArgumentError("ShiftInvert requires m >= 1, got $m"))
    m <= n || throw(ArgumentError("ShiftInvert: m = $m exceeds the operator dimension $n"))

    A = sparse(H; maxnnz=s.maxnnz)
    realpath = _isreal_assembled(A, H.diag)
    realpath && (A = real(A))
    # One exactly Hermitian matrix for the estimate and every factorization: the assembled matrix is
    # Hermitian only up to the summation order of duplicate entries.
    A = (A + A') / 2
    @debug(realpath ? "ShiftInvert: real arithmetic (the assembled matrix is exactly real)" :
                      "ShiftInvert: complex arithmetic (the assembled matrix has a nonzero imaginary part)")

    Emin = _lowest_estimate(A, s.maxiter)
    σ = s.shift === :auto ? Emin - s.shift_margin : s.shift
    F, ischol = _factor_shifted(A, σ, σ >= Emin)
    if s.shift === :auto && !ischol
        @warn "ShiftInvert: H − σI is not positive definite at the automatic shift σ = $σ (the estimate E_min = $Emin " *
              "is off by more than shift_margin = $(s.shift_margin)); factored by LU, so states below σ may be missed. " *
              "Increase shift_margin or pass certify=true"
    end

    # A certificate needs the (m+1)-th energy to place its cut between E_m and E_{m+1}.
    howmany = s.certify ? min(m + 1, n) : m
    krylovdim = max(2howmany + 10, s.krylovdim)
    T = eltype(A)
    x0 = s.block == 1 ? randn(T, n) : KrylovKit.Block([randn(T, n) for _ in 1:s.block])
    vals, vecs, info = KrylovKit.eigsolve(x -> F \ x, x0, howmany, :LM; ishermitian=true, tol=s.inner_tol,
                                          krylovdim=krylovdim, maxiter=s.maxiter, verbosity=0)
    length(vals) >= howmany && info.converged >= howmany ||
        error("ShiftInvert: only $(info.converged) of $howmany eigenpairs of the inverted operator converged " *
              "(inner_tol = $(s.inner_tol), maxiter = $(s.maxiter))")
    Eall = σ .+ inv.(real.(vals[1:howmany]))
    p = sortperm(Eall)
    Eall = Eall[p]
    E = Vector{real(eltype(H))}(Eall[1:m])
    V = Matrix{eltype(H)}(reduce(hcat, vecs[p[1:m]]))

    # Acceptance: the true residual of every returned pair, with the package operator.
    R = H * V .- V .* transpose(E)
    res = [norm(view(R, :, j)) for j in axes(R, 2)]
    worst, jw = findmax(res)
    worst <= s.tol ||
        error("ShiftInvert: the true residual ‖H v − E v‖ = $worst of pair $jw exceeds tol = $(s.tol) " *
              "(inner_tol = $(s.inner_tol) on the inverted operator); tighten inner_tol")

    if s.certify
        # The cut halfway into the gap above E_m, and a retry a quarter of the gap further up.
        gap = howmany > m ? Eall[m+1] - E[m] : 2s.tol
        below, τ = _count_below(A, (E[m] + gap / 2, E[m] + 3gap / 4))
        below == m || error("ShiftInvert: completeness certificate failed: the inertia of H − τI counts $below " *
                            "eigenvalues below τ = $τ, but $m were computed")
    end
    return E, V
end
