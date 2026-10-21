# ShiftInvert leaf solver (spec §20e decision 3): KrylovKit Lanczos on (H − σI)⁻¹ with H − σI
# factored directly (CHOLMOD Cholesky, or UMFPACK LU) from the assembled sparse matrix, in real
# arithmetic when the operator is real (`eltype(H) <: Real`). The Krylov tolerance applies to the inverted
# operator; one step of block inverse iteration polishes the pairs, and acceptance is the true residual against the
# package operator.

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
# definite (`nothing` instead with `fallback=false`); LU always for an interior σ. Returns the factorization
# and whether it is a Cholesky.
function _factor_shifted(A::SparseMatrixCSC, σ::Real, interior::Bool; fallback::Bool=true)
    if !interior
        try
            return cholesky(Hermitian(A); shift=-σ), true
        catch err
            err isa PosDefException || rethrow()
        end
        fallback || return nothing, false
    end
    try
        return lu(A - σ * I), false
    catch err
        err isa SingularException || rethrow()
        throw(ArgumentError("ShiftInvert: H − σI is singular at the shift σ = $σ (σ is an eigenvalue of H, to working " *
                            "precision); move the shift off the eigenvalue, or use shift = :auto"))
    end
end

# The Gershgorin lower bound min_i (Re A_ii − Σ_{j≠i} |A_ij|) of a Hermitian sparse matrix: at most its
# lowest eigenvalue, so A − σI is positive definite for every σ below it.
function _gershgorin_lower(A::SparseMatrixCSC)
    n = size(A, 1)
    g = Vector(real.(diag(A)))                     # dense, with zeros for absent diagonal entries
    rows, vals = rowvals(A), nonzeros(A)
    for j in 1:n, k in nzrange(A, j)
        rows[k] == j || (g[j] -= abs(vals[k]))     # column j of a Hermitian A holds the moduli of row j
    end
    return minimum(g)
end

# The automatic shift's factorization: Cholesky of A − σI at σ = Emin − margin, retried at Emin − margin·4^r
# (r = 1, 2, 3) and then at the Gershgorin shift g − margin when the estimate Emin overshoots the lowest eigenvalue
# by more than the margin; LU at σ = Emin − margin only if every Cholesky fails. Returns `(F, σ, ischol)`.
function _auto_factor(A::SparseMatrixCSC, Emin::Real, margin::Real)
    σ0 = Emin - margin
    F, ischol = _factor_shifted(A, σ0, false; fallback=false)
    ischol && return F, σ0, true
    # A retry can land just below λ₁, where H − σI is ill-conditioned yet still Cholesky-factorable; the
    # true-residual check catches a Lanczos failure there.
    for σ in [Emin .- margin .* 4.0 .^ (1:3); _gershgorin_lower(A) - margin]
        F, ischol = _factor_shifted(A, σ, false; fallback=false)
        if ischol
            @debug "ShiftInvert: H − σI was not positive definite at the automatic shift σ = $σ0 (E_min = $Emin); " *
                   "retried Cholesky at σ = $σ"
            return F, σ, true
        end
    end
    F, _ = _factor_shifted(A, σ0, false)
    return F, σ0, false
end

# The number of eigenvalues of A below τ: the negative entries of D in the LDLᵀ factorization of
# A − τI (Sylvester's law of inertia), or `nothing` when the factorization breaks down.
# CHOLMOD's LDLᵀ is simplicial with no pivoting: D is diagonal (1×1 pivots only, no 2×2 blocks to
# resolve), and a tiny pivot can break it down even on a well-conditioned indefinite matrix, so
# callers retry at another cut.
function _inertia(A::SparseMatrixCSC, τ::Real)
    F = ldlt(Hermitian(A); shift=-τ, check=false)
    return issuccess(F) ? count(d -> real(d) < 0, diag(sparse(F.LD))) : nothing
end

# The inertia count at the first cut of `τs` whose factorization succeeds (callers pass a second cut
# elsewhere in the same gap to retry once). Returns `(count, τ)`.
function _count_below(A::SparseMatrixCSC, τs)
    for τ in τs
        c = _inertia(A, τ)
        c === nothing || return c, τ
    end
    error("ShiftInvert: the inertia certificate could not be computed: the LDLᵀ factorization (no pivoting) " *
          "of H − τI broke down at every cut τ ∈ $(collect(τs))")
end

# ---- Spectrum slicing (slices > 1) ----

_nslices(s::ShiftInvert) =
    s.slices === :auto ? (s.shift === :auto && Threads.nthreads() > 1 ? Threads.nthreads() : 1) : s.slices

# The eigenvalue count below τ when no eigenvalue lies within ε of τ (the counts at τ − ε and τ + ε
# agree), else `nothing`, also when either factorization breaks down. A guarded cut puts every
# computed energy, accurate to well within ε, on the correct side.
function _guarded_count(A::SparseMatrixCSC, τ::Real, ε::Real)
    below = _inertia(A, τ - ε)
    below === nothing && return nothing
    return _inertia(A, τ + ε) == below ? below : nothing
end

# A guarded count at τ, or at a cut nudged from it by growing fractions of `width` on alternating
# sides, inside (lo, hi). Returns `(τ, count)`, or `nothing` when every nudged cut fails.
function _nudged_count(A::SparseMatrixCSC, τ::Real, ε::Real, width::Real; lo::Real=-Inf, hi::Real=Inf)
    for j in 0:16
        t = τ + (isodd(j) ? 1 : -1) * cld(j, 2) * width / 64
        lo < t < hi || continue
        c = _guarded_count(A, t, ε)
        c === nothing || return t, c
    end
    return nothing
end

function _edge_count(A::SparseMatrixCSC, τ::Real, ε::Real, width::Real)
    r = _nudged_count(A, τ, ε, width)
    r === nothing && error("ShiftInvert: slicing could not place an interval edge near τ = $τ: the LDLᵀ " *
                           "factorization of H − τI broke down, or an eigenvalue lay within $ε, at every nudged cut")
    return r
end

# A guarded cut τ in (lo, hi) with tlo ≤ N(τ) ≤ thi (N the count below), by bisection on inertia
# counts, given guarded ends with N(lo) < tlo ≤ N(hi) = Nhi. Counts are exact integers, so it ends in
# about log₂ steps; when a cluster of eigenvalues spans the target band, or no guarded cut is found
# near a midpoint, it returns the current upper end. Returns `(τ, N(τ))`.
function _bisect_cut(A::SparseMatrixCSC, lo::Real, hi::Real, Nhi::Int, tlo::Int, thi::Int, ε::Real; maxsteps::Int=64)
    for _ in 1:maxsteps
        hi - lo > 4ε || break
        r = _nudged_count(A, (lo + hi) / 2, ε, hi - lo; lo=lo, hi=hi)
        r === nothing && break
        τ, c = r
        if c < tlo
            lo = τ
        else
            hi, Nhi = τ, c
            c <= thi && break
        end
    end
    return hi, Nhi
end

# The intervals [cuts[i], cuts[i+1]) and the count below each cut: the left edge has no eigenvalue
# below it, the right edge at least m (and at most about m / 10 more, unless a cluster spans E_m),
# and the interior cuts split the total M into k parts of about M / k. A target an earlier cut has
# already passed is dropped, so fewer than k slices can result (when M < k, or across a cluster).
function _slice_intervals(s::ShiftInvert, A::SparseMatrixCSC, Emin::Real, m::Int, k::Int, ε::Real)
    w = s.shift_margin
    τL, cL = _edge_count(A, Emin - w, ε, w)
    j = 0
    while cL > 0                                 # the E_min estimate is off by more than shift_margin
        (j += 1) <= 60 || error("ShiftInvert: slicing found no cut with no eigenvalue below it")
        τL, cL = _edge_count(A, Emin - w * 2.0^j, ε, w)
    end
    # Raise the left edge to just below the estimate (an upper bound on E_1, usually accurate far
    # beyond these offsets), so the first slice's shift sits among its eigenvalues.
    for δ in (1e-3, 1e-2, 1e-1) .* w
        Emin - δ > τL || break
        if _guarded_count(A, Emin - δ, ε) == 0
            τL = Emin - δ
            break
        end
    end
    lo, j = τL, 0
    hi, Nhi = _edge_count(A, Emin + w, ε, w)
    while Nhi < m                                # double the guess above E_min until it holds m states
        (j += 1) <= 60 || error("ShiftInvert: slicing found no cut with $m eigenvalues below it")
        lo = hi
        hi, Nhi = _edge_count(A, Emin + w * 2.0^j, ε, w)
    end
    τR, M = _bisect_cut(A, lo, hi, Nhi, m, m + max(1, m ÷ 10), ε)
    cuts, counts = [τL], [0]
    for i in 1:k-1
        t = round(Int, i * M / k)
        slack = max(1, round(Int, M / 10k))
        last(counts) < t - slack || continue
        τ, c = _bisect_cut(A, last(cuts), τR, M, t - slack, t + slack, ε)
        last(counts) < c < M && (push!(cuts, τ); push!(counts, c))
    end
    push!(cuts, τR); push!(counts, M)
    return cuts, counts
end

# One slice [a, b) holding `c` eigenvalues: shift-invert Lanczos with σ at its center (interior, so
# LU) for the `c` eigenvalues nearest σ, which are exactly the slice's; converged energies outside
# [a, b) (neighbors the inverse map also pulls in) are dropped.
function _solve_slice(s::ShiftInvert, A::SparseMatrixCSC, a::Real, b::Real, c::Int, x0)
    σ = (a + b) / 2
    F, _ = _factor_shifted(A, σ, true)
    vals, vecs, info = _eigsolve(s, x -> F \ x, x0, c, max(2c + 10, s.krylovdim))
    nconv = min(info.converged, length(vals))
    nconv == 0 && return Float64[], typeof(vecs)()
    E, W = _polish(A, F, reduce(hcat, vecs[1:nconv]))
    keep = findall(e -> a <= e < b, E)
    return E[keep], [W[:, j] for j in keep]
end

# Duplicates removed from a union of slice results sorted by energy: a pair within 10·tol of a kept
# energy whose vector is parallel to the kept one (|overlap| > 0.99) is dropped; one that is not
# parallel (a degenerate partner) is orthogonalized against it.
function _dedup_union(E::AbstractVector, vecs::AbstractVector, tol::Real)
    keepE, keepV = similar(E, 0), similar(vecs, 0)
    for (e, v) in zip(E, vecs)
        dup, noted = false, false
        for i in reverse(eachindex(keepE))
            keepE[i] < e - 10tol && break
            ov = dot(keepV[i], v)
            if abs(ov) > 0.99
                dup = true
                break
            end
            if abs(ov) > 1e-8
                v = v - keepV[i] * ov
                noted = true
            end
        end
        if dup
            @debug "ShiftInvert: dropped a state at E = $e returned twice"
            continue
        end
        noted && @debug "ShiftInvert: re-orthogonalized a state at E = $e against a degenerate partner"
        push!(keepE, e)
        push!(keepV, v / norm(v))
    end
    return keepE, keepV
end

# The lowest m eigenpairs of the exactly Hermitian assembled matrix `A` by spectrum slicing over
# `k` threads, every slice certified complete by its inertia count. A final Rayleigh–Ritz on the kept
# block makes the vectors from different slices orthonormal to working precision.
function _solve_sliced(s::ShiftInvert, A::SparseMatrixCSC, Emin::Real, m::Int, k::Int)
    n = size(A, 1)
    cuts, counts = _slice_intervals(s, A, Emin, m, k, 10s.tol)
    nsl = length(cuts) - 1
    want = diff(counts)
    @debug "ShiftInvert: $nsl slices, cuts $cuts, holding $want eigenvalues"
    T = eltype(A)
    # Starting vectors drawn serially, so the result does not depend on the thread schedule.
    x0s = [s.block == 1 ? randn(T, n) : KrylovKit.Block([randn(T, n) for _ in 1:s.block]) for _ in 1:nsl]
    results = Vector{Any}(undef, nsl)
    Threads.@threads for i in 1:nsl
        results[i] = try
            _solve_slice(s, A, cuts[i], cuts[i+1], want[i], x0s[i])
        catch err
            err
        end
    end
    for r in results
        r isa Exception && throw(r)
    end
    Eall = reduce(vcat, first.(results))
    Vall = reduce(vcat, last.(results))
    p = sortperm(Eall)
    Eall, Vall = _dedup_union(Eall[p], Vall[p], s.tol)
    for i in 1:nsl
        got = count(e -> cuts[i] <= e < cuts[i+1], Eall)
        got == want[i] || error("ShiftInvert: slice $i of $nsl, [$(cuts[i]), $(cuts[i+1])), returned $got eigenpairs, " *
                                "but its inertia count is $(want[i]) (inner_tol = $(s.inner_tol), maxiter = $(s.maxiter)) " *
                                "($(_degeneracy_hint(s)); or a single shift, whose certificate recovers missed copies)")
    end
    Q = Matrix(qr(reduce(hcat, Vall[1:m])).Q)
    G = eigen(Hermitian(Q' * (A * Q)))
    return G.values, Q * G.vectors
end

# The degeneracy hint of the count-mismatch errors; block = 2 is suggested only when it is not already in use.
_degeneracy_hint(s::ShiftInvert) =
    s.block == 1 ? "a degenerate level may hold more copies than single-vector Lanczos (block = 1) can resolve; try block = 2" :
                   "a degenerate level may hold more copies than block Lanczos with block = 2 can resolve; raise maxiter or use Dense()"

# One step of block inverse iteration with the factorization `F` of H − σI, then Rayleigh–Ritz with `A`: the Ritz
# pairs of span(F \ V). Lanczos stops a pair of the inverse with its Krylov residual r = F⁻¹v − θv just under
# inner_tol, and H v − E v = −(E − σ)(H − σ) r: the components of r on states far above the wanted ones (E_k up to
# ‖H‖, which grows like 1/h² on a grid and like ncut² in a charge basis) reach the true residual magnified by
# |E − σ|·|E_k − σ|, up to 1e-5 on 8000-state leaves. The step damps each by (E − σ)/(E_k − σ), down to the roundoff
# floor, and leaves the span of the wanted states unchanged.
function _polish(A::SparseMatrixCSC, F, V::AbstractMatrix)
    Q = Matrix(qr(F \ V).Q)
    G = eigen(Hermitian(Q' * (A * Q)))
    return G.values, Q * G.vectors
end

# Acceptance: the true residual of every returned pair, with the package operator, at most max(tol, `floor`), the
# rule shared with ChebyshevFilter (`_residual_floor`).
function _check_residuals(s::ShiftInvert, H::KronFreeOperator, E::AbstractVector, V::AbstractMatrix, floor::Real)
    worst, jw = _worst_true_residual(H, E, V)
    bound = floor > s.tol ? "the roundoff floor 10 eps‖H‖ = $floor (above tol = $(s.tol))" : "tol = $(s.tol)"
    worst <= max(s.tol, floor) ||
        error("ShiftInvert: the true residual ‖H v − E v‖ = $worst of pair $jw exceeds $bound " *
              "(inner_tol = $(s.inner_tol) on the inverted operator); tighten inner_tol")
    worst > s.tol && @debug "ShiftInvert: the true residual $worst of pair $jw exceeds tol = $(s.tol) and is accepted " *
                            "at the roundoff floor 10 eps‖H‖ = $floor"
    return nothing
end

function solve(s::ShiftInvert, H::KronFreeOperator, m::Int)
    H.on === Array || throw(ArgumentError("ShiftInvert is a CPU solver, but the operator is stored on $(H.on); " *
                                          "use LOBPCG for device storage"))
    n = size(H, 1)
    m >= 1 || throw(ArgumentError("ShiftInvert requires m >= 1, got $m"))
    m <= n || throw(ArgumentError("ShiftInvert: m = $m exceeds the operator dimension $n"))

    A = sparse(H; maxnnz=s.maxnnz)
    realpath = eltype(H) <: Real
    # One exactly Hermitian matrix for the estimate and every factorization: the assembled matrix is
    # Hermitian only up to the summation order of duplicate entries.
    A = (A + A') / 2
    @debug(realpath ? "ShiftInvert: real arithmetic (the operator is real)" :
                      "ShiftInvert: complex arithmetic (the operator is complex)")

    Emin = _lowest_estimate(A, s.maxiter)
    k = _nslices(s)
    if k > 1
        Es, Vs = _solve_sliced(s, A, Emin, m, k)
        E, V = Vector{real(eltype(H))}(Es), Matrix{eltype(H)}(Vs)
        _check_residuals(s, H, E, V, _residual_floor(gershgorin_bound(H), E[1]))
        return E, V
    end
    if s.shift === :auto
        F, σ, ischol = _auto_factor(A, Emin, s.shift_margin)
    else
        σ = s.shift
        F, ischol = _factor_shifted(A, σ, σ >= Emin)
    end
    if s.shift === :auto && !ischol
        @warn "ShiftInvert: H − σI is not positive definite at the automatic shift σ = $σ (the estimate E_min = $Emin " *
              "is off by more than shift_margin = $(s.shift_margin)); factored by LU, so states below σ may be missed. " *
              "Increase shift_margin (the certificate, on by default, raises on a miss)"
    end

    # `certify = :auto` certifies whenever the lowest m are sought: σ below the spectrum (Cholesky), or the
    # automatic shift (also on its LU fallback, where a miss must raise). An explicit interior shift returns
    # the states nearest it and is not certified unless certify = true.
    cert = s.certify === :auto ? (ischol || s.shift === :auto) : s.certify
    # A certificate needs the (m+1)-th energy to place its cut between E_m and E_{m+1}.
    howmany = cert ? min(m + 1, n) : m
    vals, vecs = _si_eigsolve(s, F, howmany, nothing, n, eltype(A); partial=cert)
    Eall = σ .+ inv.(real.(vals))
    p = sortperm(Eall)
    Eall, Vall = Eall[p], reduce(hcat, vecs[p])
    cert && ((Eall, Vall) = _certify(s, A, F, ischol, σ, Eall, Vall, m))
    Eall, Vall = _polish(A, F, Vall)

    E = Vector{real(eltype(H))}(Eall[1:m])
    V = Matrix{eltype(H)}(Vall[:, 1:m])
    try
        _check_residuals(s, H, E, V, _residual_floor(gershgorin_bound(H), E[1]))
    catch err
        # An explicit shift within √eps of an eigenvalue factors (LU succeeds) but the inverse's huge
        # eigenvalue there swamps the others: name the shift rather than inner_tol.
        (err isa ErrorException && s.shift !== :auto) || rethrow()
        d = minimum(e -> abs(e - σ), Eall)
        d <= sqrt(eps(Float64)) * max(1, abs(σ)) || rethrow()
        error(err.msg * ". The shift σ = $σ lies within $d of a computed eigenvalue, so H − σI is nearly singular " *
              "and the inverse resolves only that pair: move the shift away from the eigenvalue, or use shift = :auto")
    end
    orth = norm(V' * V - I)
    orth <= sqrt(eps(real(eltype(H)))) ||
        error("ShiftInvert: the returned vectors are not orthonormal (‖V'V − I‖ = $orth)")
    return E, V
end

# The tolerance below which KrylovKit's block QR drops a vector of the block Lanczos residual (its default).
const _BLOCK_QR_TOL = 1e-12

# KrylovKit Lanczos for the `howmany` eigenpairs of largest magnitude of the Hermitian `op`, from `x0` (a vector,
# or a `KrylovKit.Block`). Block Lanczos (KrylovKit 0.10.4) stops at an invariant subspace only when the residual
# norm falls to tol, but its block QR drops every residual vector below `_BLOCK_QR_TOL`: a residual between the
# two (a few distinct levels exhaust the block Krylov space below krylovdim) leaves an empty block, an
# ArgumentError. The run then repeats from `x0` with tol raised to √b·`_BLOCK_QR_TOL` for a block of b vectors:
# every residual vector below `_BLOCK_QR_TOL` puts the residual norm below that tol, so it stops there.
# Block Lanczos also takes the residual norm of a Ritz vector u as √(u'Ru), R the Gram matrix of the residual
# block; when the block is nearly rank-deficient (a level with one reachable copy exhausts one of its
# directions), u'Ru of a converged Ritz vector is rounding noise, and a negative one raises a DomainError
# (before any stop test, so no tol avoids it). The run then repeats as single-vector Lanczos from the first
# vector of `x0`, which has no such Gram matrix; the certificate recovers the degenerate copies it misses.
function _eigsolve(s::ShiftInvert, op, x0, howmany::Int, krylovdim::Int)
    lanczos(x, tol) = KrylovKit.eigsolve(op, deepcopy(x), howmany, :LM; ishermitian=true, tol=tol, qr_tol=_BLOCK_QR_TOL,
                                         krylovdim=krylovdim, maxiter=s.maxiter, verbosity=0)
    x0 isa KrylovKit.Block || return lanczos(x0, s.inner_tol)
    try
        try
            return lanczos(x0, s.inner_tol)
        catch err
            (err isa ArgumentError && occursin("blocklength", err.msg)) || rethrow()
            tol = max(s.inner_tol, sqrt(length(x0)) * _BLOCK_QR_TOL)
            @debug "ShiftInvert: the block Krylov space became invariant with a residual between inner_tol and the " *
                   "block QR tolerance; repeating the block Lanczos run with tol = $tol"
            return lanczos(x0, tol)
        end
    catch err
        (err isa DomainError && occursin("sqrt", err.msg)) || rethrow()
        @debug "ShiftInvert: block Lanczos took the square root of a negative residual norm (a nearly rank-deficient " *
               "residual block); repeating the run as single-vector Lanczos"
        return lanczos(x0[1], s.inner_tol)
    end
end

# `howmany` eigenpairs of largest magnitude of (H − σI)⁻¹ = F⁻¹ by KrylovKit, restricted to the orthogonal
# complement of the orthonormal columns of `D` (`nothing`: the whole space): there the deflated operator
# P F⁻¹ P, P = I − D D', has the eigenpairs of F⁻¹ off span(D), and zero on it. With `partial`, fewer than
# `howmany` converged pairs (at least one) are returned rather than raising: a Krylov space exhausted by a few
# distinct levels converges only (distinct levels) × block of them, and the certificate recovers the rest.
function _si_eigsolve(s::ShiftInvert, F, howmany::Int, D, n::Int, T; partial::Bool=false)
    proj(x) = D === nothing ? x : x - D * (D' * x)
    krylovdim = max(2howmany + 10, s.krylovdim)
    x0 = s.block == 1 ? proj(randn(T, n)) : KrylovKit.Block([proj(randn(T, n)) for _ in 1:s.block])
    op = D === nothing ? (x -> F \ x) : (x -> proj(F \ proj(x)))
    vals, vecs, info = _eigsolve(s, op, x0, howmany, krylovdim)
    # The converged pairs, without those of the deflated operator's zero eigenvalue: their vectors lie in span(D)
    # (a converged Ritz vector lies in span(D) or orthogonal to it), and the inverse holds no such state.
    conv = filter(i -> D === nothing || norm(D' * vecs[i]) < 1 / 2, 1:min(info.converged, length(vals)))
    nconv = length(conv)
    if partial && 1 <= nconv < howmany
        @debug "ShiftInvert: $nconv of $howmany eigenpairs of the inverted operator converged; the certificate recovers the rest"
        return vals[conv], vecs[conv]
    end
    nconv >= howmany ||
        error("ShiftInvert: only $nconv of $howmany eigenpairs of the inverted operator converged " *
              "(inner_tol = $(s.inner_tol), maxiter = $(s.maxiter))" *
              (D === nothing ? "" : " in a deflated round of the certificate ($(_degeneracy_hint(s)))"))
    return vals[conv[1:howmany]], vecs[conv[1:howmany]]
end

# The completeness certificate on the sorted computed pairs `(Eall, Vall)`. The cut τ lies in the first gap
# above E_m wider than 2·tol (computed energies closer than that may be one level, each lying within tol of an
# eigenvalue), after the j-th computed energy; the inertia of H − τI must count j eigenvalues below τ. A larger
# count means missed states (a degenerate level holding more copies than Lanczos with the block resolves). With
# σ below the spectrum (Cholesky), the missing ones plus one are found by Lanczos on the inverse deflated
# against every computed vector, merged by Rayleigh–Ritz on the joint orthonormal block, and the certificate
# repeats. With σ not below the spectrum (LU) a larger count raises: Lanczos on the inverse targets the states
# nearest σ, so deflation need not reach the missing ones. When no such gap lies among the computed energies
# (the level at E_m runs past them, or fewer than m + 1 were computed), max(2, m + 2 − k) more states are found
# by deflation on either path; that only extends the computed set, and a return still requires a matching count
# at a gap. A deflated round returns the pairs that converge, at least one (else it raises), so every round adds
# states and at most n − k rounds run. Returns the pairs, sorted.
function _certify(s::ShiftInvert, A::SparseMatrixCSC, F, ischol::Bool, σ::Real, Eall, Vall, m::Int)
    n = size(A, 1)
    rounds = 0
    while true
        k = length(Eall)
        k == n && return Eall, Vall                          # every eigenpair computed: complete
        i = findfirst(i -> Eall[i+1] - Eall[i] > 2s.tol, m:k-1)
        if i === nothing                                     # no gap above E_m among the computed states
            below, j, τ = nothing, k, Eall[k]
            want = max(2, m + 2 - k)
        else
            j = m + i - 1
            # The cut halfway into the gap, and a retry a quarter of the gap further up.
            gap = Eall[j+1] - Eall[j]
            below, τ = _count_below(A, (Eall[j] + gap / 2, Eall[j] + 3gap / 4))
            below == j && return Eall, Vall
            msg = "ShiftInvert: completeness certificate failed: the inertia of H − τI counts $below " *
                  "eigenvalues below τ = $τ, but $j were computed"
            below > j || error(msg)
            ischol || error(msg * ". The shift σ = $σ is not below the spectrum (factored by LU), so either Lanczos on " *
                            "the inverse returned the states nearest σ rather than the lowest, or $(_degeneracy_hint(s)); " *
                            "deflated recovery runs only with σ below the spectrum (use shift = :auto or a shift below E_min)")
            want = below - j + 1
        end
        rounds += 1
        want = min(want, n - k)
        @debug "ShiftInvert: certificate round $rounds: " *
               (below === nothing ? "no gap above E_m among the $k computed states" : "$below below τ = $τ, $j computed") *
               "; finding $want more on the deflated inverse"
        _, vecs = _si_eigsolve(s, F, want, Vall, n, eltype(A); partial=true)
        W = reduce(hcat, vecs)
        for _ in 1:2
            W .-= Vall * (Vall' * W)
        end
        Q = hcat(Vall, Matrix(qr(W).Q))
        G = eigen(Hermitian(Q' * (A * Q)))
        Eall, Vall = G.values, Q * G.vectors
    end
end
