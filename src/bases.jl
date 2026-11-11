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
(2, 4, 6 or 8) that wrap around the circle (periodic boundary condition, the *periodic gauge*).
The basis carries no offset charge and its operators are parameter-free: the offset charge enters
only through the coefficients of the canonical quadratic form `4 (n - n_g)ᵀ E_C (n - n_g)` (see
[`circuit_terms`](@ref)), exactly as on every other basis.
`sectors` adds the requirement that `N` be divisible by it, beyond the harmonics declared on the
mode (see `check_sectors`).

Ports:

| port | operator |
|:-----|:---------|
| `nport()` | the cyclic central stencil of `-i ∂_φ` (imaginary, Hermitian) |
| `n2port()` | the cyclic central stencil of `-∂_φ²` (real, symmetric) |
| `expiphi(k)`, integer `k` | diagonal `e^{ikφ}` |

Plane waves `e^{ikφ}` with integer `k` are exact eigenvectors of both stencils, with symbols
`σ₁(k) = Σ_j w¹_j sin(jkΔ)/Δ ≈ k` and `σ₂(k) = -Σ_j w²_j cos(jkΔ)/Δ² ≈ k²`, so the free rotor's
discrete spectrum is `4 E_C (σ₂(k) - 2 n_g σ₁(k) + n_g²)`; at `n_g = ½` the pairs `(k, 1 - k)` are
degenerate only to `O(Δ^order)`.
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

const _PERIODIC_UNAVAILABLE = Dict(
    :phi => "φ is multivalued on a circle; use expiphi(k) with integer k",
    :phi2 => "φ is multivalued on a circle; use expiphi(k) with integer k",
    :exp2pin => "exp2pin on a periodic grid is not implemented yet",
)

# The cyclic central stencil `Y[i] = Σ_{j=-h..h} c[j+h+1] X[mod1(i + j, N)]` as one `BandedOp`: every nonzero
# `j` is an interior offset, and a wrapped neighbour `i + j` outside `1:N` is column `mod1(i + j, N)`, i.e. the
# extra offsets `j + N` (for `j < 0`) and `j - N` (for `j > 0`); each weight vector is zero on the rows where its
# offset leaves `1:N`.
function _cyclic_stencil(N::Int, h::Int, c::Vector{T}) where {T}
    js = [j for j in -h:h if !iszero(c[j+h+1])]
    offs = Int[]; ws = Vector{T}[]
    for j in js, o in (j < 0 ? (j, j + N) : j > 0 ? (j, j - N) : (j,))
        push!(offs, o)
        push!(ws, T[1 <= i + o <= N ? c[j+h+1] : zero(T) for i in 1:N])
    end
    return BandedOp(N, offs, ws)
end

function localop(b::PeriodicGrid, p::Port)
    N, Δ, h = b.N, spacing(b), b.order ÷ 2
    if p.kind in (:n, :n2)
        iszero(p.k) || throw(ArgumentError("PeriodicGrid: port $(p.kind) takes k = 0, got k = $(p.k)"))
        return p.kind === :n ? _cyclic_stencil(N, h, ComplexF64.(-im .* fd_weights(b.order, 1) ./ Δ)) :   # -i ∂_φ
                               _cyclic_stencil(N, h, -fd_weights(b.order, 2) ./ Δ^2)                       # -∂_φ²
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

# `(α/|α|)^a` exactly for purely real or purely imaginary `α` (a power of ±1 or ±i), so that the
# phases of `displacement` carry no round-off and conjugation relations between ports hold exactly.
function _unit_power(α::Number, a::Int)
    iszero(imag(α)) && return ComplexF64(real(α) > 0 || iseven(a) ? 1 : -1)
    s = imag(α) > 0 ? 1 : -1                                 # α/|α| = s i
    return (ComplexF64(1), ComplexF64(0, s), ComplexF64(-1), ComplexF64(0, -s))[mod(a, 4)+1]
end

# log(2) - Float64(log(2)): with it, `k log(2)` is subtracted from an exponent to well below one ulp.
const _LOG2_LO = 2.3190468138462996e-17

"""
    displacement(N, α) -> Matrix{ComplexF64}

Closed-form matrix elements of `D(α) = exp(α a† - α* a)` on the first `N` Fock levels
(exact elements of the infinite operator, not the exponential of a truncated generator, so the block is not unitary).

For `n = m + a ≥ m` the element is `⟨n|D(α)|m⟩ = √(m!/n!) e^{-|α|²/2} α^a L_m^{(a)}(|α|²)`, and `⟨m|D(α)|n⟩ = conj(⟨n|D(-α)|m⟩)`,
so `D(-α) = D(α)'`. The factors leave the `Float64` range long before the element does (for `|α| ≳ 37.7` the Laguerre
polynomial overflows and `e^{-|α|²/2}` underflows, and the polynomial also overflows for `N ≳ 1030` at any `α`), so the magnitudes
`f_m = √(m!/(m+a)!) |α|^a e^{-|α|²/2} L_m^{(a)}(|α|²)` are computed as a whole, as `f · 2^e` with the binary exponent `e`
carried apart from the `Float64` `f`:

- the seed `f_0 = |α|^a e^{-|α|²/2} / √(a!)` follows from that of `a - 1` by the factor `|α| / √a`, renormalized at every step;
- `f_m = ((2m - 1 + a - |α|²) f_{m-1} - √((m-1)(m-1+a)) f_{m-2}) / √(m(m+a))`, the recurrence of the Laguerre polynomial in its
  degree with the normalization absorbed, in which `f_{m-1}` and `f_{m-2}` share one exponent and are rescaled together.

The recurrence runs from the region where the elements are exponentially small into the one where they oscillate, the
direction in which it is stable, and an element below the `Float64` range comes out as zero. A finite `|α| ≥ 2^20` gives
the zero matrix (every element is below the range for any `N` that fits in memory) and a non-finite `α` gives NaN.
"""
function displacement(N::Int, α::Number)
    D = zeros(ComplexF64, N, N)
    if iszero(α)
        for i in 1:N; D[i, i] = 1; end
        return D
    end
    r, x, θ = abs(α), abs2(α), angle(α)
    isfinite(r) || return fill!(D, complex(NaN, NaN))
    r < 0x1p20 || return D
    exact = iszero(imag(α)) || iszero(real(α))
    k = round(-x / 2 / log(2.0))                             # e^{-x/2} = g · 2^e with g = e^{-x/2 - k log 2} in [1/√2, √2]
    g, e = exp(fma(-k, log(2.0), -x / 2) - k * _LOG2_LO), Int(k)
    rm, re = frexp(r)                                        # r = rm · 2^re, so that a subnormal r keeps its digits
    for a in 0:N-1                                           # a = n - m >= 0
        if a > 0                                             # g · 2^e = r^a e^{-x/2} / √(a!)
            g, de = frexp(g * (rm / sqrt(a)))
            e += re + de
        end
        fprev, f, s, ef = 0.0, g, 0.0, e                     # f_{m-1}, f_m on the scale 2^ef; s = √(m(m+a))
        for m in 0:N-1-a
            if m > 0
                snew = sqrt(m * (m + a))
                fprev, f = f, ((2m - 1 + a - x) * f - s * fprev) / snew
                s = snew
                if abs(f) > 0x1p256                          # |f_m 2^ef| <= 1: f grows only while ef is far below zero
                    fprev *= 0x1p-256; f *= 0x1p-256; ef += 256
                end
            end
            n = m + a
            mag = ldexp(f, ef)
            D[n+1, m+1] = mag * (exact ? _unit_power(α, a) : cis(a * θ))                 # ⟨n|D(α)|m⟩
            a > 0 && (D[m+1, n+1] = mag * (exact ? conj(_unit_power(-α, a)) : cis(-a * (θ + π))))  # conj(⟨n|D(-α)|m⟩)
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
