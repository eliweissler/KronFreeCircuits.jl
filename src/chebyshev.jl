# ChebyshevFilter: Chebyshev-filtered block subspace iteration (Zhou–Saad ChFSI) on the package
# operator, host or device (spec §20e decision 3). The block, its image under H and the recurrence
# scratch stay on the solver's storage; only nb × nb matrices (Gram, Rayleigh–Ritz, rotations) and
# 1 × nb rows (column norms, Ritz values) cross to the host, plus the seed going in and the result
# coming out. Acceptance is the true residual against the package operator.

# Relative eigenvalue cutoff of the unit-diagonal Gram matrix in SVQB.
const _SVQB_CUTOFF = 1e-12

# Columns lock below this fraction of the acceptance threshold max(tol, roundoff floor), leaving the fresh acceptance
# residual a margin over the Rayleigh–Ritz one.
const _LOCK_FRACTION = 0.5

# The Hermitian part of a small matrix, on the host.
_host_hermitian(G::AbstractMatrix) = (Gh = Array(G); Hermitian((Gh + Gh') / 2))

# Divide every column of `Z` by its norm (a zero column stays zero).
function _normalize_columns!(Z::AbstractMatrix)
    Z ./= max.(_colnorms(Z), floatmin(real(eltype(Z))))
    return Z
end

# One step of the three-term recurrence T_{k+1} = 2 x T_k − T_{k−1} with x = (H − c) / e:
# `C ← (2/e)(H B − c B) − A`. Allocates nothing beyond the operator's reserved scratch.
function _chebyshev_step!(C::AbstractMatrix, H::KronFreeOperator, B::AbstractMatrix, A::AbstractMatrix, c::Real, e::Real)
    mul!(C, H, B)
    C .= (2 / e) .* (C .- c .* B) .- A
    return C
end

"""
    _chebyshev_filter!(A, B, C, H, X, λcut, λmax, degree, rescale_every) -> Z

`Z = T_degree((H − c) / e) X` with `c = (λmax + λcut) / 2`, `e = (λmax − λcut) / 2`, up to a positive
scale per column: every `rescale_every` steps the live pair is divided column-wise by the norm of the
newer member. `A`, `B`, `C` are scratch of `X`'s size; the result is one of them.
"""
function _chebyshev_filter!(A, B, C, H::KronFreeOperator, X, λcut::Real, λmax::Real, degree::Int, rescale_every::Int)
    c = (λmax + λcut) / 2
    e = (λmax - λcut) / 2
    copyto!(A, X)                                                # T_0
    mul!(B, H, A)
    B .= (B .- c .* A) ./ e                                      # T_1
    for k in 2:degree
        _chebyshev_step!(C, H, B, A, c, e)
        A, B, C = B, C, A
        if k % rescale_every == 0
            sc = max.(_colnorms(B), floatmin(real(eltype(B))))
            A ./= sc
            B ./= sc
        end
    end
    return B
end

# Z ← Z − L (L' Z), twice (classical Gram–Schmidt against the orthonormal columns of L, repeated).
function _project_out!(Z::AbstractMatrix, L::AbstractMatrix)
    size(L, 2) == 0 && return Z
    for _ in 1:2
        mul!(Z, L, L' * Z, -one(eltype(Z)), one(eltype(Z)))
    end
    return Z
end

# One SVQB pass: the Gram matrix of Z with unit diagonal, its eigendecomposition on the host, and the
# directions with eigenvalue above `_SVQB_CUTOFF` times the largest, written orthonormal into the first
# k columns of W. Returns k (0 for an all-zero block).
function _svqb_pass!(W::AbstractMatrix, Z::AbstractMatrix)
    size(Z, 2) == 0 && return 0
    R = real(eltype(Z))
    G = _host_hermitian(Z' * Z)
    d = 1 ./ sqrt.(max.(real(diag(G)), floatmin(R)))
    F = eigen(Hermitian(d .* Matrix(G) .* transpose(d)))
    top = maximum(F.values)
    keep = top > 0 ? findall(>(_SVQB_CUTOFF * top), F.values) : Int[]
    k = length(keep)
    k == 0 && return 0
    M = (d .* view(F.vectors, :, keep)) ./ transpose(sqrt.(F.values[keep]))
    mul!(view(W, :, 1:k), Z, _to_storage(Z, M))
    return k
end

# SVQB of Z against the locked columns L into W (same width as Z): two passes; dropped directions are
# refilled with random vectors (host `randn` copied to the storage) orthogonalized the same way. A random
# column that loses all but a √eps fraction of its norm to the projections against L and the kept columns
# has no room left and is zeroed (dropped); after 10 refill rounds short of the width, an error. A final
# projection against L removes round-off the 1/√λ scaling amplified along it. Z and S are consumed as scratch.
function _svqb!(W::AbstractMatrix, Z::AbstractMatrix, L::AbstractMatrix, S::AbstractMatrix, rng)
    N, na = size(Z)
    _project_out!(Z, L)
    k1 = _svqb_pass!(S, Z)
    k = _svqb_pass!(W, view(S, :, 1:k1))
    refills = 0
    while k < na
        refills += 1
        refills > 10 && error("ChebyshevFilter: SVQB could not refill $(na - k) dropped directions with random vectors; " *
                              "the block is wider than the space left beside the locked columns")
        r = na - k
        Rr = view(Z, :, 1:r)
        copyto!(Rr, randn(rng, eltype(Z), N, r))
        before = vec(Array(_colnorms(Rr)))
        _project_out!(Rr, L)
        _project_out!(Rr, view(W, :, 1:k))
        after = vec(Array(_colnorms(Rr)))
        room = after .> sqrt(eps(real(eltype(Z)))) .* before
        all(room) || (Rr .*= _to_storage(Rr, reshape(eltype(Z).(room), 1, :)))
        _normalize_columns!(Rr)
        j1 = _svqb_pass!(S, Rr)
        j1 == 0 && continue
        Sv = view(S, :, 1:j1)
        _project_out!(Sv, view(W, :, 1:k))
        k += _svqb_pass!(view(W, :, k+1:na), Sv)
    end
    _project_out!(W, L)
    return W
end

# Cholesky-QR of Z against the locked columns L into W, two passes (S is scratch).
function _cholqr!(W::AbstractMatrix, Z::AbstractMatrix, L::AbstractMatrix, S::AbstractMatrix)
    _project_out!(Z, L)
    src = Z
    for dst in (S, W)
        F = cholesky(_host_hermitian(src' * src); check=false)
        issuccess(F) || error("ChebyshevFilter: Cholesky-QR failed, the filtered block is numerically rank deficient; " *
                              "use ortho=:svqb (it drops and refills dependent directions)")
        mul!(dst, src, _to_storage(src, inv(F.U)))
        src = dst
    end
    _project_out!(W, L)                     # round-off the U⁻¹ scaling amplified along L
    return W
end

_orthonormalize!(s::ChebyshevFilter, W, Z, L, S, rng) =
    s.ortho === :svqb ? _svqb!(W, Z, L, S, rng) : _cholqr!(W, Z, L, S)

# Rayleigh–Ritz on the orthonormal block W with HW = H W: the Ritz vectors and their images written
# into Xa and HXa (which must not alias W, HW); returns the ascending Ritz values (host).
function _rayleigh_ritz!(Xa, HXa, W, HW)
    F = eigen(_host_hermitian(W' * HW))
    Q = _to_storage(W, F.vectors)
    mul!(Xa, W, Q)
    mul!(HXa, HW, Q)
    return F.values
end

# True residual norms ‖HX_j − θ_j X_j‖ (host vector), with S scratch of X's size.
function _residual_norms!(S, X, HX, θ::AbstractVector)
    S .= HX .- X .* _to_storage(X, reshape(θ, 1, :))
    return vec(Array(_colnorms(S)))
end

# The lower end of the damped interval: the largest Ritz value θ_nb of the current block (Zhou–Saad's
# choice, as in the validated spike). Every block direction is then amplified, the lowest m by the
# most, and the asymptotic rate of the m-th is T_p(x(λ_{nb+1})) / T_p(x(λ_m)), set by the whole block.
# A cut at θ_{m+1} (with or without a margin) took 2–3 times the iterations on the zero-pi leaf, since
# it leaves the extra columns in the damped interval, where they carry no convergence.
_chebyshev_cut(θ::AbstractVector, m::Int, λmax::Real) = θ[end]

# The degree applied in one iteration: `degree`, lowered so that the filter amplifies the lowest Ritz
# value θ_1 (minimum over the block, locked columns included) at most 1/√eps over the cut, |T_d(x(θ_1))| = cosh(d acosh|x(θ_1)|) ≤ 1/√eps.
# Past that, a column whose own Ritz value sits near the cut is swamped by its (round-off or
# not-yet-converged) components along the lowest states: the filtered block's condition number exceeds
# 1/√eps, Cholesky-QR breaks down and SVQB's truncation mixes the dropped differences into the leading
# directions, which stalls the lowest residuals near √eps ‖H‖. On a narrow spectrum (the three-oscillator
# test: λ_max ≈ 17 against a ground state at 0.46) the unlowered degree-100 filter amplifies θ_1 by
# about 1e91 and the cap binds hard (degree 14 of 100); on a grid leaf with a large Gershgorin bound it
# binds moderately (the small zero-pi leaf runs at degree 65–81 of 100, lower in a random seed's first
# iterations, so a requested degree above about 80 buys nothing there).
function _chebyshev_degree(degree::Int, θ1::Real, λcut::Real, λmax::Real)
    x = abs((θ1 - (λmax + λcut) / 2) / ((λmax - λcut) / 2))
    x > 1 || return degree
    cap = acosh(inv(sqrt(eps(typeof(x))))) / acosh(x)
    return cap >= degree ? degree : max(floor(Int, cap), 1)
end

_bound_name(s::ChebyshevFilter) = s.bound === :gershgorin ? "the Gershgorin bound" : "the user bound bound = $(s.bound)"

# Any Rayleigh quotient above λmax proves the bound wrong (θ ≤ λ_max(H) ≤ bound for a valid one).
function _check_bound(s::ChebyshevFilter, θ::AbstractVector, λmax::Real)
    slack = sqrt(eps(typeof(λmax))) * max(abs(λmax), one(λmax))
    top = maximum(θ)
    top <= λmax + slack ||
        error("ChebyshevFilter: a Ritz value $top exceeds the upper end λ_max = $λmax of the filter interval " *
              "($(_bound_name(s))), so the bound is below the true largest eigenvalue; use bound=:gershgorin")
    return nothing
end

# The relative size of the random block added to the product seed. A filter step's gain is capped at
# about 1/√eps ≈ 7e7 over λ_cut (`_chebyshev_degree`), so a component of relative size δ along a low
# state the product states miss (their span can be exactly invariant under H) reaches O(1) only after
# about log(1/δ) / log(7e7) filter steps beyond the first. At δ = √eps (LOBPCG's seed perturbation) the
# component is still ≈ 1e-8 after one step, below the lowest columns, which are then exact and lock at
# once; at δ = 1e-2 it dominates a column within the first iteration or two. The cost is a seed 1e-2
# off the product states, which the first filter step removes.
const _SEED_ADMIXTURE = 1e-2

# The guard columns beyond m that must also lock before the iteration stops, as LOBPCG's
# `n_conv_check = m + g`: up to `_CHEBYSHEV_GUARDS`, and always one extra column fewer than `extra`. The
# top column of the block sets λ_cut = θ_nb, so its own Ritz value sits at the cut, where the filter
# separates nothing: a guard there would never converge (measured: extra = 1 with one guard ran out of
# maxiter on every probe).
const _CHEBYSHEV_GUARDS = 4
_chebyshev_guards(extra::Int) = min(extra - 1, _CHEBYSHEV_GUARDS)

# A guard column counts only while the last filter's log-gain at its Ritz value, deg · acosh|x(θ_j)| with x
# the affine map of the filter interval [λ_cut, λ_max] onto [−1, 1], is at least this fraction of the
# log-gain at the m-th lowest Ritz value θ_m (the highest wanted state). A column's unwanted components lie
# at or above the cut, where |T| ≤ 1, so its residual falls by about exp(−log-gain) per iteration relative
# to them. A guard (near-)degenerate with the top column, which sets the cut, has log-gain ≈ 0 and would
# never lock (a 2-fold level at the top of the block, or a pair 1e-7 apart): its ratio is ≈ 0 and it is
# waived. When
# the filter is weak everywhere (a spectral bound far above the block, so every column gains little per
# iteration) the ratios stay near 1 and the guards are kept: a test on the absolute gain (round 2 counted
# a guard only above a gain of 1e3) waived them all there and accepted a block that had not yet surfaced a
# missing low state. The reference is θ_m, not the bottom Ritz value θ_1 (round 3's choice): a ground state
# far below the rest (θ_1 = −1e4, the other states near 0) gains so much more than every guard that their
# ratios were ≈ 0.03, every guard was waived from the first iteration, and with extra = 2–5 and tol = 1e-3
# a state at −0.5 outside the seed span was missed with passing residuals. Against θ_m a deep ground state
# cannot waive the guards protecting the states above it. The ratio compares a guard with the m-th wanted
# column's own rate, so a counted guard needs at most about 1 / 0.1 = 10 times its iterations, usually far
# fewer; near the cut that is the price of the guards at small `extra` (three oscillators, m = 8,
# extra = 5: 7 iterations, against 2 with no guards). A
# larger fraction is not safe: at 0.5 a weak-filter solve with extra = 2 waived its guard while the column
# carrying the missing state descended through the guard's energy (ratio briefly ≈ 0) and returned a
# wrong ground state; at 0.1 that window is narrow and the solve raises the maxiter error instead.
const _GUARD_GAIN_FRACTION = 0.1

# The filter's log-gain at Ritz value t per unit degree: acosh|x(t)|, zero at or inside the damped interval.
function _log_gain(t::Real, λcut::Real, λmax::Real)
    x = abs((t - (λmax + λcut) / 2) / ((λmax - λcut) / 2))
    return x > 1 ? acosh(x) : zero(x)
end

# The number of leading guard columns m+1 : m+g whose log-gain under the last filter (cut λcut, degree deg)
# is at least `_GUARD_GAIN_FRACTION` times that of the m-th lowest Ritz value; θ ascends over them, so the
# counted ones are a prefix. All g before the first filter (no cut yet; the loop cannot stop then anyway),
# and when the m-th lowest Ritz value itself gains nothing (no reference rate to compare with).
function _counted_guards(θ::AbstractVector, m::Int, g::Int, λcut::Real, λmax::Real, deg::Int)
    (deg > 0 && isfinite(λcut)) || return g
    ref = _log_gain(partialsort(θ, m), λcut, λmax)
    ref > 0 || return g
    k = 0
    for j in m+1:m+g
        _log_gain(θ[j], λcut, λmax) >= _GUARD_GAIN_FRACTION * ref || break
        k += 1
    end
    return k
end

# The product seed `P` plus, column by column, a random vector of norm `_SEED_ADMIXTURE` times the
# column's (a fixed generator, so the seed is reproducible).
function _admixed_seed(T, P)
    X = Matrix{T}(Array(P))
    rng = Random.MersenneTwister(0x4346)
    for j in axes(X, 2)
        c = view(X, :, j)
        r = randn(rng, T, length(c))
        c .+= (_SEED_ADMIXTURE * norm(c) / norm(r)) .* r
    end
    return X
end

# Why the iteration has not stopped, for the maxiter error; `need` is m plus the guard columns.
function _not_done_reason(θ::AbstractVector, res::AbstractVector, nlock::Int, m::Int, tol::Real; need::Int=m)
    nlock < need && return "only $nlock of $need eigenpairs converged ($m wanted and $(need - m) guard columns; largest " *
                           "unconverged residual $(maximum(view(res, nlock+1:length(θ)))) > tol = $tol)" *
                           (nlock >= m ? "; the unconverged columns are guards: raise extra (so the guards lie further " *
                                         "below λ_cut) or maxiter" : "")
    return "the lowest $need columns are locked, but the unlocked Ritz value $(θ[nlock+1]) (residual $(res[nlock+1]) > " *
           "tol = $tol) lies below the $m-th locked energy $(partialsort(θ[1:nlock], m)), a lower state not yet converged"
end

# Stop once the lowest `need` (m plus guards) columns are locked and no unlocked Ritz value lies below the
# m-th locked energy.
function _chebyshev_done(θ::AbstractVector, nlock::Int, m::Int, need::Int=m)
    nlock >= need || return false
    nlock == length(θ) && return true
    return θ[nlock+1] >= partialsort(θ[1:nlock], m)
end

"""
    solve(s::ChebyshevFilter, H::KronFreeOperator, m) -> (E, V)

The lowest `m` eigenpairs of `H` (`E` ascending, host arrays), by Chebyshev-filtered subspace
iteration on a block of `nb = min(m + s.extra, N)` columns stored as `s.on` (an operator stored
elsewhere is copied there, data and scratch, for the solve). See [`ChebyshevFilter`](@ref).
"""
function solve(s::ChebyshevFilter, H::KronFreeOperator, m::Int)
    if s.on !== Array && Base.get_extension(KronFreeCircuits, :KronFreeCircuitsCUDAExt) === nothing
        throw(ArgumentError("ChebyshevFilter(on=$(s.on)) needs `using CUDA, cuTENSOR` so the KronFreeCircuits CUDA extension " *
                            "loads; without it the device path falls back to allocating methods"))
    end
    N = size(H, 1)
    m >= 1 || throw(ArgumentError("ChebyshevFilter requires m >= 1, got $m"))
    m <= N || throw(ArgumentError("ChebyshevFilter: m = $m exceeds the operator dimension $N"))
    nb = min(m + s.extra, N)
    T = eltype(H)
    R = real(T)
    rng = Random.default_rng()
    λmax = R(s.bound === :gershgorin ? gershgorin_bound(H) : s.bound)
    hbound = gershgorin_bound(H)                                               # for the roundoff floor, whatever the bound
    # Host data either way. The product seed gets a random admixture (`_SEED_ADMIXTURE`): its span can be
    # exactly invariant under H and miss a low state entirely, which no filter could then amplify.
    X0 = s.seed === :product ? _admixed_seed(T, product_seed(H, nb)) : randn(rng, T, N, nb)
    g = max(min(_chebyshev_guards(s.extra), nb - 1 - m), 0)                      # guards stay below the top column
    H = _on_storage(H, s.on; capacity=max(nb, _block_capacity(H)))             # the operator lives with the block
    _block_capacity(H) < nb && (H = with_block_capacity(H, nb))

    # Block, image and three scratch blocks of nb columns, all on s.on.
    bufs = [adapt(s.on, Matrix{T}(X0))]
    append!(bufs, [similar(bufs[1]) for _ in 1:2])
    X = similar(bufs[1])
    HX = similar(bufs[1])
    none = view(X, :, 1:0)

    # The seed's Rayleigh–Ritz.
    W, S = bufs[2], bufs[3]
    _normalize_columns!(bufs[1])
    _orthonormalize!(s, W, bufs[1], none, S, rng)
    mul!(bufs[1], H, W)
    θ = Vector{R}(_rayleigh_ritz!(X, HX, W, bufs[1]))
    _check_bound(s, θ, λmax)

    res = fill(R(Inf), nb)
    nlock = 0
    it = 0
    λcut = R(NaN)
    deg = 0
    while true
        act = nlock+1:nb
        res[act] .= _residual_norms!(view(bufs[1], :, 1:length(act)), view(X, :, act), view(HX, :, act), view(θ, act))
        # The seed's Rayleigh–Ritz is never trusted: every column is filtered at least once before it can
        # lock, so a small component along a state outside the seed's span is amplified first. A block
        # spanning the whole space (nb = N) is exact as it stands.
        filtered = it >= 1 || nb == N
        floor = _residual_floor(hbound, minimum(θ))
        while filtered && nlock < nb && res[nlock+1] <= _LOCK_FRACTION * max(s.tol, floor)
            nlock += 1
        end
        s.callback === nothing ||
            s.callback((; iteration=it, λ=copy(θ), residuals=copy(res), n_locked=nlock, λ_cut=λcut, λ_max=λmax, degree=deg))
        need = m + _counted_guards(θ, m, g, λcut, λmax, deg)                     # m plus the counted guards
        _chebyshev_done(θ, nlock, m, need) && break
        it >= s.maxiter &&
            error("ChebyshevFilter: within maxiter = $(s.maxiter) iterations " * _not_done_reason(θ, res, nlock, m, s.tol; need=need) *
                  (floor > s.tol ? "; columns lock below half the roundoff floor 10 eps·‖H‖ = $floor, which exceeds tol" : "") *
                  (0 < deg < s.degree ? "; the last filter ran at degree $deg of $(s.degree), lowered by the amplification " *
                                        "cap 1/√eps (a narrow spectrum), so more iterations may be needed" : ""))
        it += 1
        act = nlock+1:nb
        na = length(act)
        λcut = _chebyshev_cut(θ, m, λmax)
        λcut < λmax || error("ChebyshevFilter: the block's largest Ritz value $(θ[nb]) reaches the upper end λ_max = $λmax " *
                             "of the filter interval ($(_bound_name(s))); the block spans the top of the spectrum " *
                             "(nb = $nb of N = $N), use Dense()")
        A, B, C = (view(b, :, 1:na) for b in bufs)
        deg = _chebyshev_degree(s.degree, minimum(θ), λcut, λmax)
        Z = _chebyshev_filter!(A, B, C, H, view(X, :, act), λcut, λmax, deg, s.rescale_every)
        all(isfinite, Z) ||
            error("ChebyshevFilter: the filtered block is not finite: the upper end λ_max = $λmax of the filter interval " *
                  "($(_bound_name(s))) is below the true largest eigenvalue, so the filter amplifies the top of the " *
                  "spectrum; use bound=:gershgorin")
        Wv, Sv = Z === A ? (B, C) : Z === B ? (C, A) : (A, B)    # the two scratch blocks the result is not in
        _normalize_columns!(Z)
        _orthonormalize!(s, Wv, Z, view(X, :, 1:nlock), Sv, rng)
        HZ = mul!(Z, H, Wv)                                       # Z is free after orthonormalization
        θ[act] .= _rayleigh_ritz!(view(X, :, act), view(HX, :, act), Wv, HZ)
        _check_bound(s, view(θ, act), λmax)
    end

    # The lowest m locked pairs, checked again with a fresh application of the package operator.
    idx = sortperm(view(θ, 1:nlock))[1:m]
    E = θ[idx]
    Vs = X[:, idx]
    worst, jw = _worst_true_residual(H, E, Vs)
    floor = _residual_floor(hbound, E[1])
    bound = floor > s.tol ? "the roundoff floor 10 eps‖H‖ = $floor (above tol = $(s.tol))" : "tol = $(s.tol)"
    worst <= max(s.tol, floor) ||
        error("ChebyshevFilter: the true residual ‖H v − E v‖ = $worst of pair $jw exceeds $bound on return")
    worst > s.tol && @debug "ChebyshevFilter: the true residual $worst of pair $jw exceeds tol = $(s.tol) and is accepted " *
                            "at the roundoff floor 10 eps‖H‖ = $floor"
    return Vector{R}(E), Matrix{T}(Array(Vs))
end
