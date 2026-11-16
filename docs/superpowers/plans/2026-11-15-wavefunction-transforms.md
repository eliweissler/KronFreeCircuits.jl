# Charge and Phase Wavefunctions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `phase_wavefunction` and `charge_wavefunction` return a single-mode state as function values in the phase or charge representation, on the basis's own grid.

**Architecture:** One new file, `src/wavefunctions.jl`, with a method per basis. Grid and charge bases use an FFT along the first axis with the grid offset applied as a phase factor; `FockBasis` evaluates Hermite functions at points the caller passes. A `System` method reads the eigenvectors of a node with one basis child and dispatches on that basis.

**Tech Stack:** Julia 1.13, FFTW.jl (new dependency), Test.

**Spec:** `docs/superpowers/specs/2026-11-15-wavefunction-transforms-design.md`

## Global Constraints

- Convention everywhere: `⟨φ|n⟩ = e^{inφ}/√2π`.
- Function-value normalization: `Δ Σ|ψ(φ_j)|² = ‖v‖²`, `Δn Σ|ψ̃(n_k)|² = ‖v‖²`, `Σ|c_ℓ|² = ‖v‖²`.
- Conjugate index range: `-(N ÷ 2):((N - 1) ÷ 2)`, i.e. `-(N-1)/2..(N-1)/2` for odd `N` and `-N/2..N/2-1` for even `N`.
- Every function returns `(coordinates::Vector{Float64}, values)`; `values` has the shape of `V` with one row per coordinate.
- `V` is an `AbstractVecOrMat{<:Number}` with `dimension(basis)` rows, one state per column. It is never mutated or aliased by the result.
- Results are host arrays. Phase values of a real grid vector keep a real element type; everything else is `ComplexF64`.
- No interpolation, no plotting, no change to stored eigenvectors or their phases.
- Work on a new branch `wavefunction-transforms`, not on `main`. Do not add the untracked files already in the working tree.
- Test command for this feature: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_wavefunctions.jl"])'`.

## Review Focus

1. `FockBasis` called without points: an `ArgumentError` saying a Fock basis has no grid and the points must be passed, not a `MethodError`. (Task 2)
2. `V` with the wrong number of rows: a `DimensionMismatch` naming both sizes. (Task 1, Task 2)
3. A node whose single child is another node or a `GivenSystem`, or a node with several children: an `ArgumentError` saying multi-mode wavefunctions are not implemented yet. (Task 3)
4. Integer or vector input, e.g. the unit vector `[0, 1, 0]`: works and returns a vector. (Task 1)
5. Large `N` and far points on a Fock basis (`e^{-x²/(4s²)}` underflows while `h_m` does not): finite values, correct norm, zero where the function is below the `Float64` range. (Task 2)

---

### Task 1: Grid and charge bases

**Files:**
- Create: `src/wavefunctions.jl`
- Create: `test/test_wavefunctions.jl`
- Modify: `Project.toml` (FFTW in `[deps]`, `FFTW = "1"` in `[compat]`)
- Modify: `src/KronFreeCircuits.jl` (`using FFTW: FFTW`, `include("wavefunctions.jl")` after `include("dressed.jl")`, export `phase_wavefunction, charge_wavefunction`)

**Interfaces:**
- Consumes: `dimension`, `coordinates`, `spacing` from `src/bases.jl`.
- Produces:
  - `phase_wavefunction(b::Union{ChargeBasis,PeriodicGrid,PhaseGrid}, V::AbstractVecOrMat{<:Number}) -> (φ::Vector{Float64}, ψ)`
  - `charge_wavefunction(b::Union{ChargeBasis,PeriodicGrid,PhaseGrid}, V::AbstractVecOrMat{<:Number}) -> (n::Vector{Float64}, ψ̃)`
  - `_conjugate_index(N::Int) = -(N ÷ 2):((N - 1) ÷ 2)`
  - `_check_rows(b::Basis, V)`: throws `DimensionMismatch("…: V has $(size(V, 1)) rows, the basis has $(dimension(b)) states")`.

- [ ] **Step 1: Add the dependency**

Run: `julia --project=. -e 'using Pkg; Pkg.add("FFTW")'`, then add `FFTW = "1"` under `[compat]`.
Expected: `Project.toml` lists FFTW; `julia --project=. -e 'using KronFreeCircuits'` still loads.

- [ ] **Step 2: Write the failing tests**

In `test/test_wavefunctions.jl`, testset `"wavefunctions: grids and charge basis"`. Deterministic data, no random numbers: `c = [cis(0.7q^2) / q for q in 1:N]`.

```julia
direct(c, n, φ) = [sum(c[q] * cis(n[q] * x) for q in eachindex(n)) / sqrt(2π) for x in φ]
inverse(ψ, φ, n, w) = [w * sum(ψ[j] * cis(-m * φ[j]) for j in eachindex(φ)) / sqrt(2π) for m in n]
```

- `ChargeBasis(3)`, `N = 7`: `φ == -π .+ 2π .* (0:6) ./ 7`; `ψ ≈ direct(c, -3:3, φ)` (`rtol = 1e-13`); `(2π/7) * sum(abs2, ψ) ≈ sum(abs2, c)`.
- `charge_wavefunction(ChargeBasis(3), c)`: `n == -3:3`, values `== c` and `!== c`.
- The notes' `N`-point grid `φ_n = 2π(n - n0)/N`, for each `n0 in 0:6`: with `δ = π - 2π n0/7`, `phase_wavefunction(ChargeBasis(3), c .* cis.((-3:3) .* δ))[2] ≈ direct(c, -3:3, 2π .* ((0:6) .- n0) ./ 7)`.
- `PeriodicGrid(N; order=2)` for `N in (7, 8)`, `v = real.(c)`: phase values `== v ./ sqrt(2π/N)` with `eltype Float64` on `coordinates`; charge `ℓ == _conjugate_index(N)` (`-3:3`, `-4:3`); `c_ℓ ≈ inverse(ψ, φ, ℓ, 2π/N)`; `sum(abs2, c_ℓ) ≈ sum(abs2, v)`.
- Round trip: `φ, ψ = phase_wavefunction(ChargeBasis(3), c)`; `charge_wavefunction(PeriodicGrid(7; order=2), ψ .* sqrt(2π/7))[2] ≈ c`.
- `PhaseGrid(-3.0, 5.0, N; order=2)` for `N in (40, 41)`, `Δ = 8/(N-1)`: `n == 2π .* _conjugate_index(N) ./ (N*Δ)`; `ψ̃ ≈ inverse(v ./ sqrt(Δ), φ, n, Δ)`; `(2π/(N*Δ)) * sum(abs2, ψ̃) ≈ sum(abs2, v)`.
- Gaussian on `g = PhaseGrid(-12.0, 12.0, 241; order=2)`, `s = 0.8`, `σ = 1/(2s)`, `v = sqrt(Δ) .* (2π*s^2)^(-1/4) .* exp.(-φ.^2 ./ (4s^2))`: `ψ̃ ≈ (2π*σ^2)^(-1/4) .* exp.(-n.^2 ./ (4σ^2))` (`atol = 1e-10`), so the imaginary part vanishes.
- Sign of the kernel: `v .* cis.(2.0 .* φ)` on the same grid has `n[argmax(abs.(ψ̃))]` within one charge step of `+2.0`.
- A two-column matrix gives a two-column result whose columns equal the single-vector results.
- Review Focus 2: `@test_throws DimensionMismatch` for a 6-row `V` on `ChargeBasis(3)`, both functions.
- Review Focus 4: `phase_wavefunction(ChargeBasis(1), [0, 1, 0])[2] ≈ fill(1/sqrt(2π), 3)` and is a `Vector`.

- [ ] **Step 3: Run the tests to verify they fail**

Expected: FAIL with `UndefVarError: phase_wavefunction`.

- [ ] **Step 4: Implement the six methods in `src/wavefunctions.jl`**

With `X = FFTW.fftshift(FFTW.fft(V, 1), 1)` and `k = _conjugate_index(N)`:

- `PeriodicGrid` charge: `c_ℓ = cis(ℓ π) X_ℓ / √N` (grid origin `φ_0 = -π`).
- `PhaseGrid` charge: `n_k = 2πk/(NΔ)`, `ψ̃_k = √(Δ/2π) cis(-n_k a) X_k`.
- `ChargeBasis` phase: `ψ = FFTW.bfft(FFTW.ifftshift(c .* cis.(-π .* n), 1), 1) ./ √2π` on `coordinates(PeriodicGrid)`'s formula `-π .+ 2π .* (0:N-1) ./ N` (do not construct a `PeriodicGrid`, whose `N` is bounded below by its stencil order).
- Grid phase: `V ./ sqrt(spacing(b))` on `coordinates(b)`.
- `ChargeBasis` charge: `copy(V)` on `coordinates(b)`.

Each docstring states its formula, its normalization, and the limits in the spec's "Known limits" section that apply to it.

- [ ] **Step 5: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Project.toml src/KronFreeCircuits.jl src/wavefunctions.jl test/test_wavefunctions.jl
git commit -m "wavefunctions: charge and phase representations of grid and charge bases"
```

---

### Task 2: Fock basis

**Files:**
- Modify: `src/wavefunctions.jl`
- Modify: `test/test_wavefunctions.jl`

**Interfaces:**
- Consumes: `_check_rows` from Task 1; `FockBasis` fields `N`, `phi_zpf`.
- Produces:
  - `phase_wavefunction(b::FockBasis, V::AbstractVecOrMat{<:Number}, φ::AbstractVector{<:Real}) -> (Vector{Float64}, values)`
  - `charge_wavefunction(b::FockBasis, V::AbstractVecOrMat{<:Number}, n::AbstractVector{<:Real}) -> (Vector{Float64}, values)`
  - `_hermite_functions(x::AbstractVector{<:Real}, N::Int, s::Float64) -> Matrix{Float64}`, `length(x) × N`, column `m + 1` holding `h_m(x; s) = (2π s²)^{-1/4} (2^m m!)^{-1/2} H_m(x/(√2 s)) e^{-x²/(4s²)}`.

- [ ] **Step 1: Write the failing tests**

Testset `"wavefunctions: Fock basis"`, `b = FockBasis(12; phi_zpf=0.7)`, `Id = Matrix(1.0I, 12, 12)`.

- Phase: `x = range(-12, 12; length=2401)`, `H = phase_wavefunction(b, Id, x)[2]`: `step(x) * H'H ≈ I` and `step(x) * H' * Diagonal(x) * H ≈ dense(localop(b, phiport()))`, both `atol = 1e-9`; `eltype(H) == Float64`.
- Charge: `q = range(-12, 12; length=2401)`, `C = charge_wavefunction(b, Id, q)[2]`: `step(q) * C'C ≈ I` and `step(q) * C' * Diagonal(q) * C ≈ dense(localop(b, nport()))`, `atol = 1e-9`.
- Charge is the Fourier transform of phase: `g = PhaseGrid(-14.0, 14.0, 561; order=2)`, `φ, Hg = phase_wavefunction(b, Id, coordinates(g))`, `n, F = charge_wavefunction(g, Hg .* sqrt(spacing(g)))`; `F ≈ charge_wavefunction(b, Id, n)[2]` (`atol = 1e-9`). (`coordinates` and `spacing` are not exported; qualify them.)
- Review Focus 5: `B = FockBasis(1200; phi_zpf=1.0)`, `e` the unit vector on level 1199, `x = range(-110, 110; length=22001)`: all values finite, `step(x) * sum(abs2, ψ) ≈ 1` (`rtol = 1e-8`).
- Review Focus 1: `@test_throws ArgumentError phase_wavefunction(b, Id)` and the same for `charge_wavefunction`.
- Review Focus 2: `@test_throws DimensionMismatch` for an 11-row `V`.
- Empty points: `size(phase_wavefunction(b, Id, Float64[])[2]) == (0, 12)`.

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL with a `MethodError` on the three-argument call.

- [ ] **Step 3: Implement `_hermite_functions` and the Fock methods**

Recurrence in `ξ = x/(√2 s)`: `h_0 = (2π s²)^{-1/4} e^{-ξ²/2}`, `h_m = √(2/m) ξ h_{m-1} - √((m-1)/m) h_{m-2}`.
Carry the binary exponent apart from the mantissa, as `displacement` does in `src/bases.jl`, so the seed's underflow at `|ξ| ≳ 38` does not zero levels that are still in range.
Phase values: `_hermite_functions(φ, N, phi_zpf) * V`.
Charge values: `_hermite_functions(n, N, 1/(2 phi_zpf)) * ((-im) .^ (0:N-1) .* V)`, with the four exact powers of `-i` (no `cis`).
The two-argument Fock methods throw `ArgumentError("FockBasis has no grid: pass the points, e.g. phase_wavefunction(b, V, φ)")`.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wavefunctions.jl test/test_wavefunctions.jl
git commit -m "wavefunctions: Hermite functions of the Fock basis in phase and charge"
```

---

### Task 3: Solved nodes and the README

**Files:**
- Modify: `src/wavefunctions.jl`
- Modify: `test/test_wavefunctions.jl`
- Modify: `README.md` (new section `## Wavefunctions` before `## Tests`)

**Interfaces:**
- Consumes: the basis methods of Tasks 1 and 2; `materialized(sys, node)`, whose result has `children` (in axis order, each a `Basis` or `MaterializedNode`) and `vectors`.
- Produces, for `f in (phase_wavefunction, charge_wavefunction)`:
  - `f(sys::System, node::Node, points...)`: the basis method on the node's one basis child and `eigenvectors(sys, node)`.
  - `f(sys::System, points::AbstractVector{<:Real}...)`: the same at the root.

- [ ] **Step 1: Write the failing tests**

Testset `"wavefunctions: solved nodes"`.

- `p = dc_fluxonium()`, `sys = materialize(p.root, p.spec, p.θ)`, `g = PhaseGrid(-10π, 10π, 601; order=8)`: `phase_wavefunction(sys) == phase_wavefunction(g, eigenvectors(sys))`; `charge_wavefunction(sys, p.root) == charge_wavefunction(g, eigenvectors(sys))`; each column of `ψ̃` has `step * sum(abs2, ·) ≈ 1`.
- Fluxonium on a Fock basis agrees with the grid, in both representations, under one global phase per state:
  ```julia
  fock = Node([:varphi_1 => FockBasis(200; phi_zpf=(2 * 2.5 / 0.5)^0.25)]; solver=Dense(), keep=lowest(5))
  sysF = materialize(fock, p.spec, p.θ)
  φ, ψ = phase_wavefunction(sys);  n, ψ̃ = charge_wavefunction(sys)
  _, ψF = phase_wavefunction(sysF, φ);  _, ψ̃F = charge_wavefunction(sysF, n)
  for j in 1:5
      o = step(φ) * dot(ψF[:, j], ψ[:, j])          # ψ ≈ ψF · o
      @test abs(o) ≈ 1 atol = 1e-6
      @test ψ[:, j] ≈ ψF[:, j] .* o atol = 1e-4
      @test ψ̃[:, j] ≈ ψ̃F[:, j] .* o atol = 1e-4   # the same phase aligns both representations
  end
  ```
  (`φ` is a `Vector`; use its spacing `φ[2] - φ[1]`.) If the tolerances fail, measure the actual disagreement, report it, and tighten or loosen only with the reason stated in a comment; a disagreement above `1e-3` is a bug, not a tolerance.
- Review Focus 3: `q = dc_two_transmons()`, `s2 = materialize(q.root, q.spec, q.θ)`: `@test_throws ArgumentError phase_wavefunction(s2)` (several children), and the message contains `"multi-mode"`. A node with one node child throws the same way (build `Node([p.root])`-style parent from the fluxonium leaf, or use the nearest existing helper).

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL with a `MethodError` on `phase_wavefunction(::System)`.

- [ ] **Step 3: Implement the `System` methods**

`ArgumentError("phase_wavefunction: the node has $(k) children; multi-mode wavefunctions are not implemented yet")`, and for a single non-basis child `"… the node's child is not a basis; multi-mode wavefunctions are not implemented yet"`.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Write the README section**

A fluxonium example on a `PhaseGrid` and on a `FockBasis`, the convention `⟨φ|n⟩ = e^{inφ}/√2π`, the normalization, the table of formulas from the spec, and the four known limits.

- [ ] **Step 6: Run the fast tier**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: every file passes, including `test_wavefunctions.jl` (it is not listed in `SLOW_FILES`).

- [ ] **Step 7: Commit**

```bash
git add src/wavefunctions.jl test/test_wavefunctions.jl README.md
git commit -m "wavefunctions: methods on a solved single-mode node; README section"
```
