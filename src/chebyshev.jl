# ChebyshevFilter: Chebyshev-filtered block subspace iteration (Zhou–Saad ChFSI) on the package
# operator, host or device (spec §20e decision 3). The block, its image under H and the recurrence
# scratch stay on the solver's storage; only nb × nb matrices (Gram, Rayleigh–Ritz, rotations) and
# 1 × nb rows (column norms, Ritz values) cross to the host, plus the seed going in and the result
# coming out. Acceptance is the true residual against the package operator.

# Relative eigenvalue cutoff of the unit-diagonal Gram matrix in SVQB.
const _SVQB_CUTOFF = 1e-12

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

# Why the iteration has not stopped, for the maxiter error.
function _not_done_reason(θ::AbstractVector, res::AbstractVector, nlock::Int, m::Int, tol::Real)
    nlock < m && return "only $nlock of $m eigenpairs converged (largest unconverged residual " *
                        "$(maximum(view(res, nlock+1:length(θ)))) > tol = $tol)"
    return "the lowest $m columns are locked, but the unlocked Ritz value $(θ[nlock+1]) (residual $(res[nlock+1]) > " *
           "tol = $tol) lies below the $m-th locked energy $(partialsort(θ[1:nlock], m)), a lower state not yet converged"
end

# Stop once the lowest m columns are locked and no unlocked Ritz value lies below the m-th locked energy.
function _chebyshev_done(θ::AbstractVector, nlock::Int, m::Int)
    nlock >= m || return false
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
    X0 = s.seed === :product ? product_seed(H, nb) : randn(rng, T, N, nb)      # host data either way
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
        while nlock < nb && res[nlock+1] <= s.tol
            nlock += 1
        end
        s.callback === nothing ||
            s.callback((; iteration=it, λ=copy(θ), residuals=copy(res), n_locked=nlock, λ_cut=λcut, λ_max=λmax, degree=deg))
        _chebyshev_done(θ, nlock, m) && break
        it >= s.maxiter &&
            error("ChebyshevFilter: within maxiter = $(s.maxiter) iterations " * _not_done_reason(θ, res, nlock, m, s.tol) *
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
    worst <= s.tol ||
        error("ChebyshevFilter: the true residual ‖H v − E v‖ = $worst of pair $jw exceeds tol = $(s.tol) on return")
    return Vector{R}(E), Matrix{T}(Array(Vs))
end
