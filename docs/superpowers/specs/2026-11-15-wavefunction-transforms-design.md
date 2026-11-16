# Charge and phase wavefunctions of a single mode

Date: 2026-11-15. Status: approved by Eli 2026-11-15; the element-type sentence was amended after the branch review (same day).

## Intent

Plot the phase and charge wavefunctions of a fluxonium now, and multi-dimensional wavefunctions later.
A state stored in one single-mode basis is returned as function values in the phase or the charge representation, on the basis's own grid (no oversampling).

## Goals, non-goals

- Goal: `phase_wavefunction` and `charge_wavefunction` for `ChargeBasis`, `PeriodicGrid`, `PhaseGrid` and `FockBasis`, and for a solved node with one basis child.
- Goal: one convention throughout, `⟨φ|n⟩ = e^{inφ}/√2π`, the one in `rotor_codes/0 notes/FFT.tex` and the one the package's operators already use (`e^{ikφ}|n⟩ = |n+k⟩`, `n = -i∂_φ`).
- Non-goal (this change): multi-mode states. A node with more than one child throws an `ArgumentError` that says so.
- Non-goal: interpolation onto a finer grid, plotting itself, changing how eigenvectors are stored or phased.
- Anti-goal: following the grid that `rotor_codes/rotors.py` actually samples (its `get_phi_shift` has the opposite sign to the notes' `n0`, and its labels are half a step from its samples). The notes' formulas are followed at the true grid points.

## Approaches

1. **Evaluate at chosen points.** Each basis gives a matrix of its states at any points. Flexible resolution, but it oversamples. Rejected by Eli for that reason.
2. **FFT basis change on the native grid (chosen by Eli).** A unitary discrete Fourier transform between the stored vector and its conjugate grid, scaled to function values. Critically sampled, exact round trip.
3. Re-express a whole `System` in another basis. Far more than plotting needs. Dropped.

`FockBasis` has no grid, so it uses Hermite functions at points the caller passes (agreed by Eli).

## Design

```julia
φ, ψ = phase_wavefunction(basis, V)        # ChargeBasis, PeriodicGrid, PhaseGrid
n, ψ̃ = charge_wavefunction(basis, V)
φ, ψ = phase_wavefunction(fock, V, φ)      # FockBasis: the points are required
n, ψ̃ = charge_wavefunction(fock, V, n)
φ, ψ = phase_wavefunction(sys, node)       # eigenvectors of a node with one basis child (plus points for Fock)
n, ψ̃ = charge_wavefunction(sys, node)
```

`V` is a vector, or a matrix with one state per column, with `dimension(basis)` rows.
The result has the shape of `V` with one row per returned coordinate.

Values are function values in the notes' normalization (my call, unchallenged in the conversation): `Δ Σ|ψ(φ_j)|² = ‖v‖²` in phase, `Δn Σ|ψ̃(n_k)|² = ‖v‖²` in a continuous charge, `Σ|c_ℓ|² = ‖v‖²` in a discrete charge.

| Basis | Phase | Charge |
|:--|:--|:--|
| `ChargeBasis(ncut)`, `N = 2ncut+1` | `ψ(φ_j) = (1/√2π) Σ_n c_n e^{inφ_j}` on `φ_j = -π + 2πj/N`, the grid of `PeriodicGrid(N)` | `c_n` itself, `n = -ncut..ncut` |
| `PeriodicGrid(N)`, `Δ = 2π/N` | `v_j/√Δ` on its grid | `c_ℓ = (√2π/N) Σ_j ψ(φ_j) e^{-iℓφ_j}`, integer `ℓ` |
| `PhaseGrid(a, b, N)`, spacing `Δ` | `v_j/√Δ` on its grid | `ψ̃(n_k) = (Δ/√2π) Σ_j ψ(φ_j) e^{-i n_k φ_j}`, `n_k = 2πk/(NΔ)` |
| `FockBasis(N; phi_zpf)` | `Σ_m c_m h_m(φ; φ_zpf)` | `Σ_m c_m (-i)^m h_m(n; n_zpf)`, `n_zpf = 1/(2φ_zpf)` |

`h_m(x; s) = (2π s²)^{-1/4} (2^m m!)^{-1/2} H_m(x/(√2 s)) e^{-x²/(4s²)}`, evaluated by GSL.jl (`gsl_sf_hermite_func_array`).
Eli chose GSL over the package's own recurrence on 2026-11-15, after both matched a 4096-bit reference to `5e-14` up to level 1199; SpecialPolynomials, ClassicalOrthogonalPolynomials, HypergeometricFunctions and FastGaussQuadrature were tried and do not do this job (overflow at high level, or nodes and weights only).

Index range of `ℓ` and `k`: `-(N-1)/2..(N-1)/2` for odd `N` (the notes' case), `-N/2..N/2-1` for even `N` (the `fftshift` order; the notes do not cover even `N`).

The grid offset is carried by the phase factor `e^{-iℓφ_0}` (the notes' `n0`, here `φ_0 = -π` or `a`), so a real even `ψ(φ)` gives a real `ψ̃(n)`.

Element types: a representation the basis already stores keeps the (floated) element type of `V`, so the phase values of a real grid vector, the charge coefficients of a real charge-basis vector and the phase values of a real Fock vector stay real; every transformed result is `ComplexF64`.
Results are dense host arrays, new ones, whatever holds `V`.

## Known limits (documented in the docstrings)

- Charge resolution of a `PhaseGrid` is fixed by its window: `Δn = 2π/(NΔ)`, range `|n| ≤ π/Δ`. For `PhaseGrid(-10π, 10π, 601)` that is `Δn ≈ 0.1`. A finer charge plot needs a wider phase window.
- A `PhaseGrid` holds both endpoints, so the transform's period is `NΔ`, one step more than `b - a`. This is harmless when the state vanishes at the edges, which the package already warns about.
- The charge of an even-`N` `PeriodicGrid` has no matching `ChargeBasis` (which is always odd).
- A grid basis discretizes `n` by finite differences, so its transformed vector is a change of representation, not an eigenvector of a charge-basis Hamiltonian.

## My calls, left to correct

- **GSL.jl becomes a dependency** (Eli's choice, above; GPL, compatible).
- **FFTW becomes a dependency** (compatible: the package is GPL-3). It handles any `N`, including the prime 601, and transforms along one axis of a multi-dimensional array, which the later multi-mode work needs. The alternative is a dependency-free direct sum, same output, `O(N²)`.
- New file `src/wavefunctions.jl`, tests in `test/test_wavefunctions.jl` (fast tier), a short README section, two new exports.

## Tests

- Each formula against a direct evaluation of the sum in the table, odd and even `N`.
- The notes' `N`-point transform reproduced at `φ_n = 2π(n - n0)/N`.
- Round trip: `ChargeBasis(ncut)` to phase, rescaled by `√Δ`, through `PeriodicGrid(2ncut+1)` back to the same coefficients.
- Norms as stated above.
- Fock: `∫ h_m φ h_m'` and `∫ h_m (-i∂_φ) h_m'` reproduce the `:phi` and `:n` ports; the charge functions are the Fourier transform of the phase functions.
- Fluxonium ground state on a `PhaseGrid` and on a `FockBasis`: the two phase wavefunctions agree, and so do the two charge wavefunctions, up to the global phase.
- A multi-child node throws.

## Left to the builder

Internal names and the layout of the README section.
