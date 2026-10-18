# Single-mode bases; each hands out the named port operators as structured local operators.

abstract type Basis end

"""
    ChargeBasis(ncut)

Compact-mode charge basis with states `|n⟩`, `n = -ncut..ncut`, and `e^{ikφ}|n⟩ = |n+k⟩`.
"""
struct ChargeBasis <: Basis
    ncut::Int
    function ChargeBasis(ncut::Integer)
        ncut >= 0 || throw(ArgumentError("ChargeBasis requires ncut >= 0, got $ncut"))
        return new(ncut)
    end
end

"""
    PhaseGrid(a, b, N; order=8)

Extended-mode phase grid `range(a, b; length=N)` with `n = -i ∂_φ` discretized by central
finite differences of the given `order` (2, 4, 6 or 8).
"""
struct PhaseGrid <: Basis
    a::Float64
    b::Float64
    N::Int
    order::Int
    function PhaseGrid(a::Real, b::Real, N::Integer, order::Integer)
        order in (2, 4, 6, 8) || throw(ArgumentError("PhaseGrid order must be 2, 4, 6 or 8, got $order"))
        b > a || throw(ArgumentError("PhaseGrid requires b > a, got a = $a, b = $b"))
        N > order || throw(ArgumentError("PhaseGrid requires N > order = $order, got N = $N"))
        return new(Float64(a), Float64(b), N, order)
    end
end
PhaseGrid(a::Real, b::Real, N::Integer; order::Integer=8) = PhaseGrid(a, b, N, order)

"""
    PeriodicGrid(N; order=8, sectors=1)

Compact-mode phase grid on `[-π, π)` without the endpoint, `φ_i = -π + 2π i/N` for
`i = 0..N-1`, spacing `Δ = 2π/N`, with central finite-difference stencils of the given `order`
(2, 4, 6 or 8) that wrap around the circle.
The basis carries no offset charge, so its operators are parameter-free.
A wrapped stencil is split into an interior band and two wrap corners; the twisted boundary
`ψ(φ + 2π) = e^{-2πi n_g} ψ(φ)` enters through the coefficients `e^{±2πi n_g}` that term
construction puts on the corners.
`sectors` adds the requirement that `N` be divisible by it, beyond the harmonics declared on the
mode (see `check_sectors`).

Ports:

| port | operator |
|:-----|:---------|
| `Port(:d1, 0)` | interior band of `-i ∂_φ` (stencil entries that stay inside `1:N`) |
| `Port(:d1wrap, ±1)` | wrap corners of `-i ∂_φ` (convention below) |
| `Port(:d2, 0)`, `Port(:d2wrap, ±1)` | the same split of `-∂_φ²` |
| `expiphi(k)`, integer `k` | diagonal `e^{ikφ}` |

Corner convention, for each stencil reach `j = 1..order/2`:
`Port(:d·wrap, +1)` is the upper-right corner, rows `i ≤ j`, column `i - j + N` (`BandedOp`
offset `N - j`), and carries `e^{+2πi n_g}`, since `ψ(φ - 2π) = e^{+2πi n_g} ψ(φ)`;
`Port(:d·wrap, -1)` is the lower-left corner, rows `i > N - j`, column `i + j - N` (offset
`j - N`), the exact adjoint of `+1`, and carries `e^{-2πi n_g}`.

The twisted operators `d + e^{2πi n_g} d₊ + e^{-2πi n_g} d₋` are Hermitian for real `n_g` since
`d₋ = d₊'`.
The twisted second derivative `d2 + e^{2πi n_g} d2wrap₊ + e^{-2πi n_g} d2wrap₋` is real when the
twist phase is ±1, i.e. `n_g ∈ {0, ½}`, so a single uncoupled mode's Hamiltonian is real there.
The `d1` pieces are imaginary-antisymmetric like `PhaseGrid`'s `:n`.
The ports `:n` and `:n2` are not served: term construction assembles the charge from the
derivative pieces and the offset charge.
`:phi` and `:phi2` are not served because `φ` is multivalued on a circle, and `:exp2pin` is a
later item.
"""
struct PeriodicGrid <: Basis
    N::Int
    order::Int
    sectors::Int
    function PeriodicGrid(N::Integer, order::Integer, sectors::Integer)
        order in (2, 4, 6, 8) || throw(ArgumentError("PeriodicGrid order must be 2, 4, 6 or 8, got $order"))
        N >= order + 1 || throw(ArgumentError("PeriodicGrid requires N >= order + 1 = $(order + 1), got N = $N"))
        sectors >= 1 || throw(ArgumentError("PeriodicGrid requires sectors >= 1, got $sectors"))
        N % sectors == 0 ||
            throw(ArgumentError("PeriodicGrid requires N divisible by sectors = $sectors, got N = $N"))
        return new(N, order, sectors)
    end
end
PeriodicGrid(N::Integer; order::Integer=8, sectors::Integer=1) = PeriodicGrid(N, order, sectors)

_is_twisted(::PeriodicGrid) = true          # term construction uses its derivative pieces (terms.jl)

"""
    FockBasis(N; phi_zpf)

Harmonic-oscillator basis truncated to `N` levels with `φ̂ = φ_zpf (a + a†)` and
`n̂ = i n_zpf (a† - a)`, `n_zpf = 1/(2 φ_zpf)`.
"""
struct FockBasis <: Basis
    N::Int
    phi_zpf::Float64
    function FockBasis(N::Integer, phi_zpf::Real)
        N >= 1 || throw(ArgumentError("FockBasis requires N >= 1, got $N"))
        phi_zpf > 0 || throw(ArgumentError("FockBasis requires phi_zpf > 0, got $phi_zpf"))
        return new(N, Float64(phi_zpf))
    end
end
FockBasis(N::Integer; phi_zpf::Real) = FockBasis(N, phi_zpf)

dimension(b::ChargeBasis) = 2b.ncut + 1
dimension(b::PhaseGrid) = b.N
dimension(b::PeriodicGrid) = b.N
dimension(b::FockBasis) = b.N

modetype(::ChargeBasis) = :c
modetype(::PhaseGrid) = :e
modetype(::PeriodicGrid) = :c
modetype(::FockBasis) = :h

coordinates(b::ChargeBasis) = collect(Float64, -b.ncut:b.ncut)
coordinates(b::PhaseGrid) = collect(range(b.a, b.b; length=b.N))
coordinates(b::PeriodicGrid) = collect(-π .+ 2π .* (0:b.N-1) ./ b.N)
coordinates(b::FockBasis) = collect(0.0:b.N-1)

positive_mask(b::Union{ChargeBasis,PhaseGrid,PeriodicGrid}) = coordinates(b) .>= 0
positive_mask(b::FockBasis) = fill(true, b.N)

spacing(b::PhaseGrid) = (b.b - b.a) / (b.N - 1)
spacing(b::PeriodicGrid) = 2π / b.N

const _FD1 = Dict(
    2 => [-1 // 2, 0, 1 // 2],
    4 => [1 // 12, -2 // 3, 0, 2 // 3, -1 // 12],
    6 => [-1 // 60, 3 // 20, -3 // 4, 0, 3 // 4, -3 // 20, 1 // 60],
    8 => [1 // 280, -4 // 105, 1 // 5, -4 // 5, 0, 4 // 5, -1 // 5, 4 // 105, -1 // 280],
)
const _FD2 = Dict(
    2 => [1 // 1, -2, 1],
    4 => [-1 // 12, 4 // 3, -5 // 2, 4 // 3, -1 // 12],
    6 => [1 // 90, -3 // 20, 3 // 2, -49 // 18, 3 // 2, -3 // 20, 1 // 90],
    8 => [-1 // 560, 8 // 315, -1 // 5, 8 // 5, -205 // 72, 8 // 5, -1 // 5, 8 // 315, -1 // 560],
)

"""
    fd_weights(order, derivative) -> Vector{Float64}

Central finite-difference weights `w_j`, `j = -order/2..order/2`, with
`f^{(derivative)}(x_i) ≈ Σ_j w_j f(x_{i+j}) / Δ^derivative`.
"""
function fd_weights(order::Int, derivative::Int)
    order in (2, 4, 6, 8) || throw(ArgumentError("fd_weights: order must be 2, 4, 6 or 8, got $order"))
    derivative == 1 && return Float64.(_FD1[order])
    derivative == 2 && return Float64.(_FD2[order])
    throw(ArgumentError("fd_weights: derivative must be 1 or 2, got $derivative"))
end

function _portname(p::Port)
    p.kind in (:expiphi, :exp2pin) && return "$(p.kind)($(p.k))"
    p.kind in (:d1wrap, :d2wrap) && return "$(p.kind)($(isinteger(p.k) ? Int(p.k) : p.k))"
    return string(p.kind)
end
_unavailable(b::Basis, p::Port) =
    ArgumentError("$(nameof(typeof(b))): port $(_portname(p)) is not available")

"""
    localop(b::Basis, p::Port) -> LocalOp

The operator of port `p` in basis `b`.
"""
function localop(b::ChargeBasis, p::Port)
    d = dimension(b)
    n = collect(-b.ncut:b.ncut)
    if p.kind === :n
        return DiagonalOp(Float64.(n))
    elseif p.kind === :n2
        return DiagonalOp(Float64.(n .^ 2))
    elseif p.kind === :expiphi
        abs(p.k - round(p.k)) <= 1e-12 ||
            throw(ArgumentError("ChargeBasis: port expiphi($(p.k)) needs an integer k"))
        return ShiftOp(d, round(Int, p.k), 1.0)          # e^{ikφ}|n⟩ = |n+k⟩
    elseif p.kind === :exp2pin
        return DiagonalOp(exp.(2π * im * p.k .* n))
    end
    throw(_unavailable(b, p))
end

function localop(b::PhaseGrid, p::Port)
    N, Δ, h = b.N, spacing(b), b.order ÷ 2
    φ = coordinates(b)
    if p.kind === :n                                       # -i ∂_φ, Hermitian since w¹ is antisymmetric
        w = fd_weights(b.order, 1)
        offs = [j for j in -h:h if j != 0]
        return BandedOp(N, offs, [fill(ComplexF64(-im * w[j+h+1] / Δ), N) for j in offs])
    elseif p.kind === :n2                                  # -∂_φ²
        w = fd_weights(b.order, 2)
        return BandedOp(N, collect(-h:h), [fill(-w[j+h+1] / Δ^2, N) for j in -h:h])
    elseif p.kind === :phi
        return DiagonalOp(φ)
    elseif p.kind === :phi2
        return DiagonalOp(φ .^ 2)
    elseif p.kind === :expiphi
        return DiagonalOp(exp.(im * p.k .* φ))
    elseif p.kind === :exp2pin
        # e^{2πik n̂} with n̂ = -i∂_φ: ψ(φ) -> ψ(φ + 2πk), i.e. Y[i] = X[i+s] = ShiftOp(N, -s, 1)
        s = 2π * p.k / Δ
        abs(s - round(s)) < 1e-8 ||
            throw(ArgumentError("PhaseGrid: port $(_portname(p)) needs 2πk/Δ to be an integer, got $s"))
        return ShiftOp(N, -round(Int, s), 1.0)
    end
    throw(_unavailable(b, p))
end

# How to obtain the charge of a `PeriodicGrid` mode, which the basis does not serve as a port.
const _PERIODIC_CHARGE_ROUTES =
    "use KronFreeCircuits.charge_port(sys, mode) for the leaf's physical charge matrix; for a port or coupling at an " *
    "ancestor, request the three pieces Port(:d1, 0), Port(:d1wrap, 1) and Port(:d1wrap, -1) and assemble " *
    "n = d1 + cispi(2 n_g)·d1wrap₊ + cispi(−2 n_g)·d1wrap₋ + n_g·I (for a Term: three terms whose coefficients carry " *
    "the twist factors twist_factor(n_g, ±1), plus the n_g·I share folded into the remaining factors); the automatic " *
    "expansion of ports= and extra_terms is a planned follow-up"

const _PERIODIC_UNAVAILABLE = Dict(
    :n => "the charge is assembled by term construction from the derivative ports :d1, :d1wrap and the offset charge; " *
          _PERIODIC_CHARGE_ROUTES,
    :n2 => "the charge squared is assembled by term construction from the derivative ports :d2, :d2wrap and the offset " *
           "charge; for the charge n itself, " * _PERIODIC_CHARGE_ROUTES,
    :phi => "φ is multivalued on a circle; use expiphi(k) with integer k",
    :phi2 => "φ is multivalued on a circle; use expiphi(k) with integer k",
    :exp2pin => "exp2pin on a periodic grid is not implemented yet",
)

# Interior band (corner = 0) or one wrap corner of the wrapped central stencil
# `Y[i] = Σ_j c[j+h+1] X[mod1(i + j, N)]`, `j = -h..h`. The upper corner (corner = +1) holds the
# entries with `i + j < 1` (rows `i ≤ -j`, column `i + j + N`), the lower corner (corner = -1)
# those with `i + j > N` (rows `i > N - j`, column `i + j - N`).
function _periodic_stencil(N::Int, h::Int, c::Vector{T}, corner::Int) where {T}
    js = [j for j in -h:h if !iszero(c[j+h+1])]
    corner == 1 && filter!(<(0), js)
    corner == -1 && filter!(>(0), js)
    # a wrapped neighbour i + j outside 1:N is column mod1(i + j, N): offset j + N below 1, j - N above N
    offs = [j + (corner == 1 ? N : corner == -1 ? -N : 0) for j in js]
    ws = [T[1 <= i + o <= N ? c[j+h+1] : zero(T) for i in 1:N] for (j, o) in zip(js, offs)]
    return BandedOp(N, offs, ws)
end

function localop(b::PeriodicGrid, p::Port)
    N, Δ, h = b.N, spacing(b), b.order ÷ 2
    if p.kind in (:d1, :d2, :d1wrap, :d2wrap)
        wrap = p.kind in (:d1wrap, :d2wrap)
        if wrap
            p.k in (1, -1) || throw(ArgumentError("PeriodicGrid: port $(_portname(p)) needs k = +1 or -1 (the corner)"))
        else
            iszero(p.k) || throw(ArgumentError("PeriodicGrid: port $(p.kind) takes k = 0, got k = $(p.k)"))
        end
        c = p.kind in (:d1, :d1wrap) ? ComplexF64.(-im .* fd_weights(b.order, 1) ./ Δ) :   # -i ∂_φ
                                       -fd_weights(b.order, 2) ./ Δ^2                       # -∂_φ²
        return _periodic_stencil(N, h, c, wrap ? Int(p.k) : 0)
    elseif p.kind === :expiphi
        abs(p.k - round(p.k)) <= 1e-12 ||
            throw(ArgumentError("PeriodicGrid: port expiphi($(p.k)) needs an integer k on a 2π-periodic grid"))
        return DiagonalOp(exp.(im * round(Int, p.k) .* coordinates(b)))
    elseif haskey(_PERIODIC_UNAVAILABLE, p.kind)
        throw(ArgumentError("PeriodicGrid: port $(_portname(p)) is not available: $(_PERIODIC_UNAVAILABLE[p.kind])"))
    end
    throw(_unavailable(b, p))
end

"""
    check_sectors(b::PeriodicGrid, harmonics, mode) -> nothing

Require `N` divisible by every integer harmonic `ω` of a cosine on `mode` and by `b.sectors`,
so the translation by `2π/ω` maps the grid to itself and the charge-parity sectors are exact.
"""
function check_sectors(b::PeriodicGrid, harmonics::AbstractVector{<:Integer}, mode::Symbol)
    for ω in harmonics
        iszero(ω) && continue
        b.N % ω == 0 || throw(ArgumentError("mode :$mode: PeriodicGrid N = $(b.N) is not divisible by the harmonic " *
                                            "ω = $ω of a cosine (an expiphi(ω) factor) on this mode; choose N a multiple of $(abs(ω))"))
    end
    b.N % b.sectors == 0 ||
        throw(ArgumentError("mode :$mode: PeriodicGrid N = $(b.N) is not divisible by sectors = $(b.sectors)"))
    return nothing
end

function localop(b::FockBasis, p::Port)
    N, φz = b.N, b.phi_zpf
    nz = 1 / (2φz)
    up = [i < N ? sqrt(i) : 0.0 for i in 1:N]               # A[i, i+1] of a  (level m = i-1)
    dn = [sqrt(i - 1) for i in 1:N]                         # A[i, i-1] of a†
    diag2 = [2(i - 1) + 1.0 for i in 1:N]                   # ⟨m|(a+a†)²|m⟩ = 2m+1
    up2 = [i <= N - 2 ? sqrt(i * (i + 1)) : 0.0 for i in 1:N]   # A[i, i+2] = √((m+1)(m+2)), m = i-1
    dn2 = [i >= 3 ? sqrt((i - 2) * (i - 1)) : 0.0 for i in 1:N] # A[i, i-2] = √((m+1)(m+2)), m = i-3
    if p.kind === :phi
        return BandedOp(N, [1, -1], [φz .* up, φz .* dn])
    elseif p.kind === :n                                     # i n_zpf (a† - a)
        return BandedOp(N, [1, -1], [ComplexF64.(-im * nz .* up), ComplexF64.(im * nz .* dn)])
    elseif p.kind === :phi2
        return BandedOp(N, [0, 2, -2], [φz^2 .* diag2, φz^2 .* up2, φz^2 .* dn2])
    elseif p.kind === :n2                                    # n_zpf² (2a†a + 1) - n_zpf² (a² + a†²)
        return BandedOp(N, [0, 2, -2], [nz^2 .* diag2, -nz^2 .* up2, -nz^2 .* dn2])
    elseif p.kind === :expiphi                               # e^{ikφ̂} = D(i k φ_zpf)
        return DenseOp(displacement(N, im * p.k * φz))
    elseif p.kind === :exp2pin                               # e^{2πik n̂} = D(-2π k n_zpf)
        return DenseOp(displacement(N, -2π * p.k * nz))
    end
    throw(_unavailable(b, p))
end

"""
    displacement(N, α) -> Matrix{ComplexF64}

Closed-form matrix elements of `D(α) = exp(α a† - α* a)` on the first `N` Fock levels
(exact elements of the infinite operator, not the exponential of a truncated generator).
"""
function displacement(N::Int, α::Number)
    D = zeros(ComplexF64, N, N)
    x = abs2(α)
    if iszero(α)
        for i in 1:N; D[i, i] = 1; end
        return D
    end
    lf = cumsum([0.0; log.(1:N-1)])                         # lf[j+1] = log(j!)
    logr, θ = log(abs(α)), angle(α)
    for a in 0:N-1                                           # a = n - m >= 0
        Lprev, L = 0.0, 1.0                                  # L_{-1}, L_0 of L^{(a)}
        for m in 0:N-1-a
            if m == 1
                Lprev, L = L, 1 + a - x
            elseif m > 1
                Lprev, L = L, ((2(m - 1) + 1 + a - x) * L - (m - 1 + a) * Lprev) / m
            end
            n = m + a
            mag = exp(0.5 * (lf[m+1] - lf[n+1]) + a * logr - x / 2) * L
            D[n+1, m+1] = mag * cis(a * θ)                   # ⟨n|D(α)|m⟩
            a > 0 && (D[m+1, n+1] = mag * cis(-a * (θ + π)))  # conj(⟨n|D(-α)|m⟩)
        end
    end
    return D
end

"""
    edge_support(b::PhaseGrid, V, w) -> Vector{Float64}

Per column of `V`, the 2-norm of its entries on the first `w` and last `w` grid points.
"""
function edge_support(b::PhaseGrid, V::AbstractMatrix, w::Int)
    size(V, 1) == b.N || throw(DimensionMismatch("edge_support: V has $(size(V, 1)) rows, grid has $(b.N) points"))
    w >= 0 || throw(ArgumentError("edge_support: w must be >= 0, got $w"))
    idx = [i for i in 1:b.N if i <= w || i > b.N - w]
    return [norm(view(V, idx, j)) for j in axes(V, 2)]
end
