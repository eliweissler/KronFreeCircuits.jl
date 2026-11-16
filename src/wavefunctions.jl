# Phase and charge wavefunctions of a single mode, in the convention ⟨φ|n⟩ = e^{inφ}/√2π: the one the port operators
# already use (`e^{ikφ}|n⟩ = |n+k⟩`, `n = -i∂_φ`).  Function values, on the basis's own grid.

# The indices conjugate to an `N`-point grid, centred: `-(N-1)/2..(N-1)/2` for odd `N`, `-N/2..N/2-1` for even `N`
# (the order `fftshift` gives).
_conjugate_index(N::Int) = -(N ÷ 2):((N - 1) ÷ 2)

function _check_rows(b::Basis, V::AbstractVecOrMat, name::AbstractString)
    size(V, 1) == dimension(b) ||
        throw(DimensionMismatch("$name: V has $(size(V, 1)) rows, the basis has $(dimension(b)) states"))
    return nothing
end

# X[k] = Σ_j V[j] e^{-2πi jk/N} along the first axis, `j = 0..N-1`, in the order of `_conjugate_index(N)`.
_centred_dft(V::AbstractVecOrMat) = FFTW.fftshift(FFTW.fft(ComplexF64.(V), 1), 1)

_sign(k::Integer) = isodd(k) ? -1 : 1                       # e^{±iπk}, exactly

"""
    phase_wavefunction(basis, V) -> (φ, ψ)
    phase_wavefunction(basis::FockBasis, V, φ) -> (φ, ψ)
    phase_wavefunction(sys, [node], [φ]) -> (φ, ψ)

The phase wavefunction `ψ(φ) = ⟨φ|ψ⟩` of the states in the columns of `V` (a vector, or a matrix with one state per
column, in `basis`), as function values at the points `φ`, with `⟨φ|n⟩ = e^{inφ}/√2π`.
`ψ` has the shape of `V` with one row per point, and `Δ Σ_j |ψ(φ_j)|² = ‖v‖²` for the spacing `Δ` of `φ`.
The points are the basis's own grid: nothing is interpolated.

| basis | `φ` | `ψ` |
|:------|:----|:----|
| `ChargeBasis(ncut)`, `N = 2ncut + 1` | `-π + 2πj/N`, the grid of `PeriodicGrid(N)` | `(1/√2π) Σ_n c_n e^{inφ_j}` |
| `PeriodicGrid`, `PhaseGrid` | the grid | `v_j/√Δ` |

The values of a real grid vector stay real; those of a charge basis are `ComplexF64`.
See [`charge_wavefunction`](@ref) for the conjugate representation.
"""
function phase_wavefunction end

"""
    charge_wavefunction(basis, V) -> (n, ψ̃)
    charge_wavefunction(basis::FockBasis, V, n) -> (n, ψ̃)
    charge_wavefunction(sys, [node], [n]) -> (n, ψ̃)

The charge wavefunction `⟨n|ψ⟩` of the states in the columns of `V` (a vector, or a matrix with one state per column, in
`basis`), with `⟨φ|n⟩ = e^{inφ}/√2π`; the result has the shape of `V` with one row per charge.

| basis | `n` | values |
|:------|:----|:-------|
| `ChargeBasis(ncut)` | `-ncut..ncut` | `c_n`, a copy of `V` |
| `PeriodicGrid(N)`, `Δ = 2π/N` | integers `ℓ` | `c_ℓ = (√2π/N) Σ_j ψ(φ_j) e^{-iℓφ_j}`, with `Σ_ℓ |c_ℓ|² = ‖v‖²` |
| `PhaseGrid(a, b, N)`, spacing `Δ` | `n_k = 2πk/(NΔ)` | `ψ̃(n_k) = (Δ/√2π) Σ_j ψ(φ_j) e^{-i n_k φ_j}`, with `Δn Σ_k |ψ̃(n_k)|² = ‖v‖²`, `Δn = 2π/(NΔ)` |

Here `ψ(φ_j) = v_j/√Δ` are the values [`phase_wavefunction`](@ref) returns, and `ℓ` and `k` run over `-(N-1)/2..(N-1)/2`
for odd `N` and `-N/2..N/2-1` for even `N`.
A compact mode has a discrete charge and an extended one a continuous charge, sampled at the points conjugate to its grid.
The sums are one fast Fourier transform, and the grid's origin enters as a phase factor, so a real even `ψ(φ)` gives a
real `ψ̃(n)`.

Limits:

- The charge resolution of a `PhaseGrid` is fixed by its window: `Δn = 2π/(NΔ)` over `|n| ≤ π/Δ`
  (`Δn ≈ 0.1` for `PhaseGrid(-10π, 10π, 601)`). A finer charge grid needs a wider phase window.
- A `PhaseGrid` holds both endpoints, so the transform's period is `NΔ`, one step more than `b - a`; this is harmless for a
  state that vanishes at the edges of the grid, and wraps the state around otherwise.
- The charges of an even-`N` `PeriodicGrid` are not those of a `ChargeBasis`, which always has an odd dimension.
- A grid basis discretizes `n` by finite differences, so the result is the state's representation in charge, not an
  eigenvector of the Hamiltonian written in a charge basis.
"""
function charge_wavefunction end

function phase_wavefunction(b::ChargeBasis, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "phase_wavefunction")
    N = dimension(b)
    # ψ(φ_j) = (1/√2π) Σ_n (c_n e^{-iπn}) e^{2πi jn/N} at φ_j = -π + 2πj/N
    C = ComplexF64.(V) .* _sign.(-b.ncut:b.ncut)
    return collect(-π .+ 2π .* (0:N-1) ./ N), FFTW.bfft(FFTW.ifftshift(C, 1), 1) ./ sqrt(2π)
end

function phase_wavefunction(b::Union{PeriodicGrid,PhaseGrid}, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "phase_wavefunction")
    return coordinates(b), V ./ sqrt(spacing(b))
end

function charge_wavefunction(b::ChargeBasis, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "charge_wavefunction")
    return coordinates(b), copy(V)
end

function charge_wavefunction(b::PeriodicGrid, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "charge_wavefunction")
    ℓ = _conjugate_index(b.N)
    # c_ℓ = (1/√N) Σ_j v_j e^{-iℓφ_j} with φ_j = -π + 2πj/N
    return collect(Float64, ℓ), _centred_dft(V) .* (_sign.(ℓ) ./ sqrt(b.N))
end

function charge_wavefunction(b::PhaseGrid, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "charge_wavefunction")
    Δ = spacing(b)
    n = 2π .* _conjugate_index(b.N) ./ (b.N * Δ)
    # ψ̃(n_k) = √(Δ/2π) e^{-i n_k a} Σ_j v_j e^{-2πi jk/N} with φ_j = a + jΔ
    return collect(n), _centred_dft(V) .* (sqrt(Δ / 2π) .* cis.(-b.a .* n))
end
