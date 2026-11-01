# Design: the `PeriodicGrid` in the periodic gauge (2026-10-30)

Status: design for Eli's approval; amends architecture spec §20d (periodic phase grid) and §20f (real arithmetic).
Decided with Eli on 2026-10-30; the sweet-spot cost accepted 2026-10-31 after the builder check.

## 1. Intent

The `PeriodicGrid` basis for compact modes currently works in the *twisted gauge*: the offset charge `n_g` enters only through the boundary condition `ψ(φ + 2π) = e^{-2πi n_g} ψ(φ)`, realized as the phases `e^{±2πi n_g}` on the two wrap corners of every derivative stencil, and the kinetic term is the plain second derivative.
This change moves the basis to the *periodic gauge*: wavefunctions are strictly `2π`-periodic, `n̂ = -i ∂_φ` and `n̂² = -∂_φ²` are fixed, parameter-free cyclic operators, and the offset charge enters only where the paper's quantization puts it, in the canonical form `4 (n̂ - n_g)ᵀ E_C (n̂ - n_g)`.

The two gauges are unitarily equivalent (the paper's §"Quantization with Multiple Compact Modes": a twist `q` and an offset `n_g` combine into `n_g - q`, so the twisted gauge is that argument run in reverse with `q = -n_g`).
The spectra agree to discretization error; the change is in how the package is organized, not in what it computes.

### Why

1. **Fidelity to the paper and to standard practice.**
   After the change the package's compact modes quantize exactly as the paper's text does: `n̂` is an operator with integer spectrum (in the continuum), `n_g` is a c-number offset, and the quadratic form expands into `n̂ n̂`, linear and constant pieces for every mode alike.
   A reader can match term construction to the paper line by line, and the `PeriodicGrid` looks like every other basis (`ChargeBasis`, `PhaseGrid`, `FockBasis`): it serves `nport()` and `n2port()`, and nothing in term construction knows which basis a mode is on.
2. **Charge driving.**
   A gate drive is `n_g(t) = n_g + n_g^{AC}(t)`, so `H(t) = H₀ + f(t) n̂ + c(t)` with `f(t) = -8 E_C n_g^{AC}(t)` and `c(t)` a c-number.
   In the periodic gauge `n̂` is a fixed operator, and the `nport()` matrix in the kept eigenbasis is exactly what a dynamics solver needs beside the static `H₀`.
   In the twisted gauge a time-dependent `n_g` moves the wrap-corner phases, so the operator and its eigenbasis change with the drive; expressing a drive as "modulate the `n` operator as a whole" is not available.
   More fundamentally (Eli, 2026-10-31), the twisted gauge is the unitary `U(t) = e^{-i n_g(t) φ}` applied to the periodic-gauge wavefunction.
   For an AC gate voltage `U` is time dependent, and the transformed Hamiltonian acquires the generator `i U̇ U† = ṅ_g φ`, a term in the multivalued `φ` that is not an operator on the circle; the paper's remark "because `q_j` is static, this adds no further terms" holds only for a static quasimomentum, not for a driven offset charge.
   The periodic gauge keeps the drive where it belongs, as `f(t) n̂` in the Hamiltonian.
3. **One fewer mechanism.**
   The derivative-piece ports, the twist factors, the twisted charge assembly, the per-node routing of charge pieces to ancestors, and the special cases in `circuit_terms`, `constant_term`, `materialize` and `charge_port` all go away.

### What it costs (accepted)

In the periodic gauge the linear term `-8 E_C n_g n̂` has imaginary character on a phase grid (`n̂` is `-i` times a real antisymmetric stencil), so a `PeriodicGrid` node is real by structure only when that term's coefficient is identically zero: `n_g` a literal `0` in the spec (or structurally zero through `E_C`).
The twisted gauge was also real at the literal `n_g = "0.5"` (twist phase `-1`); that case now runs complex.
The charge sweet spot therefore loses the ~2× of real arithmetic on `PeriodicGrid` leaves.
Eli accepted this on 2026-10-30; §20f's later item "real arithmetic at special parameter values" is unchanged and is the place to recover it if it ever matters.

**The sweet-spot degeneracy is exact only to discretization error.**
In the twisted gauge the discrete free-rotor spectrum is `4 E_C σ₂(k - n_g)` (§4.5), which at `n_g = ½` is exactly even under `k ↦ 1 - k`, so the sweet-spot pairs of the free rotor and of a pure cos 2φ qubit are exactly degenerate on the grid, as in the charge basis.
In the periodic gauge the discrete spectrum `4 E_C (σ₂(k) - 2 n_g σ₁(k) + n_g²)` has that symmetry only in the continuum, because the finite-difference `σ₁(k)` is not exactly `k`; the pairs split by `O(Δ^order)`.
Measured with the order-8 stencil at `E_C = 1` (builder check, 2026-10-30): the lowest pair splits by 1.1e-8 at `N = 32`, 4.4e-11 at 64 and 9e-14 at 128 (formula; the dense eigensolve returns 1.8e-12 there, its roundoff on eigenvalues of order 1.6e4); the next pair by 4e-6, 1.7e-8 and 6.6e-11.
For the pure cos 2φ qubit at `E_2 = 15 E_C` (implementation, 2026-11-01) the lowest two pairs split by 4.5e-9 and 1.1e-7 at `N = 64`, 1.8e-11 and 4.3e-10 at 128, and 1.3e-12 and 2.1e-12 at 256, against 6e-14 in the charge basis.
For a protected qubit the sweet-spot splitting is a figure of merit, so this is recorded in the README and pinned by a test (§5), and tolerances at `n_g = ½` are derived from it.
Nothing in the periodic gauge restores exactness short of spectral (dense Fourier) differentiation, which is not this design.
Eli accepted this on 2026-10-31.

## 2. Goals, non-goals, anti-goals

Goals:

- `PeriodicGrid` serves `nport()` (`:n`), `n2port()` (`:n2`) and `expiphi(k)` for integer `k` as parameter-free local operators, with the same characters as `PhaseGrid`'s (`:n` imaginary, `:n2` real, `expiphi` paired).
- `circuit_terms` and `constant_term` expand the quadratic form identically for every mode; no basis dictionary, no twisted branch.
- The spec form and the term-list form of `materialize` treat `nport()` on a `PeriodicGrid` mode like on any other basis (as a factor, a requested port, or in a selection rule), with no offset-charge lookup.
- `charge_port(sys, mode)` is `V′ n V` on every basis.
- All twisted-gauge code and its public surface are removed: the `twist_factor` export, the port kinds `:d1, :d1wrap, :d2, :d2wrap`, `_is_twisted`, `_twisted`, `_twisted_pieces`, `_expand_twisted_charge`, `_assembled_charge`, the `_PERIODIC_*` messages, and the ancestor routing of charge pieces in `tree.jl`.
- Validation against the charge basis on identical physics at `n_g ∈ {0, ½, generic}`, as before, plus the cross-check of the charge port and of the periodic-gauge discretization formula for the free rotor (§4).
- Spec §20d and §20f, `README.md` ("Periodic phase grid", "Real arithmetic") and `STATUS.md` say what the package does now; the retired convention is recorded in one dated paragraph of §20d, not deleted from history.

Non-goals:

- Emitting a static-plus-drive pair, or a drive section in the spec format.
  The package's deliverable for driving is a clean, time-independent `nport()`; composing `H₀ + f(t) n̂` is the user's (Eli, 2026-10-30: "that scope is right for now").
- Serving `exp2pin(k)` on a `PeriodicGrid`.
  In the periodic gauge it is a plain grid translation by `2πk` when `2πk/Δ` is an integer (as on `PhaseGrid`), so it is now trivial, but it stays a later item (my call: out of scope, to keep this change a pure gauge change).
- Half-integer `expiphi(k)` (quasiparticle ports, §20b).
  Still refused: `e^{iφ/2}` is not single-valued on the periodic grid in either gauge.
- Recovering real arithmetic at `n_g = ½`.
- Any edit to the paper (`circuit_quantization/main.tex`).

Anti-goals:

- A change in physics.
  Every spectrum the slow reference tests check (zero-pi convergence against the charge basis, two transmons with both offset charges nonzero, cos 2φ sectors) must still pass at its present tolerance or a tolerance the builder can justify from the discretization formula of §4.
- Inferring realness from parameter *values*.
  The §20f rule stands: a node is real only by structure, decided from the term list.
  Nothing new is special-cased; the literal `"0.5"` simply stops qualifying because the linear term no longer vanishes.

## 3. Constraints

- Operator data stays parameter-free (spec §11.1); `n_g` lives only in coefficients.
- Every spec expression stays real-valued (§20f "Real inputs"); the only complex structure entering through this basis is the `-i` in the `:n` stencil and the `expiphi` diagonals, exactly as on `PhaseGrid`.
- Hermitian by construction: the `:n` stencil is `-i` times an antisymmetric cyclic matrix; `:n2` is a symmetric cyclic matrix.
- `check_sectors`, the `sectors` field, the integer-harmonic check and the mode-type check (compact → `ChargeBasis` or `PeriodicGrid`) are unchanged.
- The test tiers (`runtests.jl`'s `SLOW_FILES`, `REFERENCE_FILES`) are unchanged in structure; `test_periodic.jl` stays in the fast tier.

## 4. The design (decided)

### 4.1 Operators

`localop(b::PeriodicGrid, p)`:

- `:n` (`k = 0`): the cyclic central stencil of `-i ∂_φ` of order `b.order`, `Y[i] = Σ_{j≠0} (-i w¹_j/Δ) X[mod1(i + j, N)]`, as one `BandedOp` whose offsets are the interior offsets `j ∈ -h..h, j ≠ 0` and the wrap offsets `j ± N`, each with zero weight on the rows where that offset leaves `1:N` (the existing `_periodic_stencil` already builds each piece this way; the builder merges the three pieces into one operator, or keeps `_periodic_stencil` as the helper and concatenates its offsets and weights).
- `:n2` (`k = 0`): the same for `-∂_φ²` with the order-`b.order` second-derivative weights, including the diagonal.
- `expiphi(k)`, integer `k`: unchanged.
- `:phi`, `:phi2`: refused with the existing message (multivalued on a circle).
- `:exp2pin`: refused as "not implemented yet" (non-goal above).
- `:d1, :d1wrap, :d2, :d2wrap`: no longer port kinds anywhere (`port_of_kind` rejects them like any unknown kind).

`character(b::PeriodicGrid, p)`: `:n` imaginary, `:n2` real, `expiphi(k)` paired (real at `k = 0`), after validating the port as `localop` does; this is `PhaseGrid`'s table minus `:phi, :phi2, :exp2pin`.

### 4.2 Term construction

`circuit_terms(spec)` and `constant_term(spec)` are the present charge-basis expansion, applied to every mode:
diagonal `4 E_C[i,i] n2port()`, cross `4 (E_C[i,j] + E_C[j,i]) nport() ⊗ nport()`, linear `-4 Σ_j (E_C[i,j] + E_C[j,i]) n_g[j]` on `nport()`, and the constant `constant + 4 n_gᵀ E_C n_g`.
The two-argument methods `circuit_terms(spec, bases)` and `constant_term(spec, bases)` are removed, and their callers (the spec form of `materialize`, tests) use the one-argument forms (my call; nothing else distinguished bases at the term level).
Terms with identically zero coefficient are dropped as now, which is what makes a literal `n_g = "0"` real by structure.

### 4.3 Materialization

- The spec form of `materialize` no longer expands `nport()` on `PeriodicGrid` modes and no longer passes offset charges down; the term-list form no longer refuses `nport()` on them.
- `tree.jl`: ancestor port requests for `nport()` are routed unchanged on every basis; the twisted-piece routing is removed.
- `charge_port(sys, mode)` is `_port_matrix(V, dims, a, localop(b, nport()), character(b, nport()))` for every basis, needs no spec, and its docstring loses the `PeriodicGrid` paragraph.
- `_check_bases`, `check_sectors` and the harmonic check are unchanged.

### 4.4 Real arithmetic

No code change beyond removing the twisted special cases: `classify` already decides from the term list and port characters.
Consequences to record (§20f, README):

- a `PeriodicGrid` node is real when every term on it has real character with a real coefficient, which for the canonical form means the linear `nport()` term is absent (literal `n_g = "0"`, or `E_C` structurally decoupling it) and no `nport() ⊗ nport()` cross term reaches it with a single `nport()` factor on this node (two imaginary factors inside one node multiply to a real joint port, as §20i already handles);
- the literal `"0.5"` is no longer a real case;
- `n_g` a free parameter is complex, as before.

### 4.5 Discretization (for the tests)

Plane waves `e^{ikφ}`, integer `k`, diagonalize both cyclic stencils on the grid with symbols `σ₁(k) = Σ_j w¹_j sin(jkΔ)/Δ ≈ k` and `σ₂(k) = -Σ_j w²_j cos(jkΔ)/Δ² ≈ k²`.
The periodic-gauge free rotor therefore has the exact discrete spectrum `4 E_C (σ₂(k) - 2 n_g σ₁(k) + n_g²)`, which differs from the twisted gauge's `4 E_C σ₂(k - n_g)` and from `4 E_C (σ₁(k) - n_g)²` by `O(Δ^order)`.
The builder uses this formula (not the twisted one now in `test_periodic.jl`) as the "exact for the discretization" reference, and keeps the continuum comparison at the present tolerances or ones justified from it.
Convergence in `N` is still of order `b.order`.
At `n_g = ½` the pair `(k, 1 - k)` splits by `4 E_C [σ₂(k) - σ₂(1-k) - σ₁(k) + σ₁(1-k)]`, which is `O(Δ^order)` and is the formula the sweet-spot test of §5 checks against.

### 4.6 Symmetry sectors

The half-period translation `T: φ → φ + π` on an even-`N` grid is now a plain `circshift` by `N/2` with `T² = I`, and it commutes with `n̂`, `n̂²` and `cos 2φ` at every `n_g`; the twist phase on the wrapped half disappears from the cos 2φ sector test.

## 5. Tests

- `test/test_bases.jl`: the "twisted stencil equals the thesis construction" testset becomes "cyclic stencils": `dense(localop(b, nport()))` equals the dense cyclic `-i ∂_φ` built directly with `mod1`, is skew-Hermitian times `-i` (Hermitian, purely imaginary), `n2port()` is real symmetric, both are exactly diagonalized by plane waves with the symbols of §4.5, `sparse(...) ≈ dense(...)`, and the refused ports throw.
- `test/test_character.jl`: the served-pair count and the `PeriodicGrid` rows of the character sweep follow the new table; `character(PeriodicGrid, nport())` is imaginary rather than throwing; the `arithmetic=:real` error names `n` (the imaginary linear term), not `d2wrap`.
- `test/test_periodic.jl`: rewritten for the periodic gauge — free rotor against the continuum and against the §4.5 formula at `n_g ∈ {0, 0.3, 0.5}`; the matrix has no imaginary part only at `n_g = 0`; the sweet-spot splitting of the lowest two free-rotor pairs at `n_g = ½` equals the §4.5 formula to roundoff and decays by about `2^order` per doubling of `N` over `N ∈ {32, 64, 128}`, and the pure cos 2φ sector test at `N = 128` sees its lowest two pairs degenerate to the measured 1.8e-11 and 4.3e-10 (bounds 4e-11 and 1e-9), with the 2⁸ decay from `N = 64` and the charge basis's exact degeneracy as controls; transmon against the charge basis; charge port equals the charge-basis `n` in the eigenbasis up to the phase convention; cos 2φ sectors with the plain translation; two transmons coupled through their charges, with `n_g1` a free symbol only of the linear coefficient and no operator data depending on it; `nport()` at the leaf, above the leaf, in a selection rule and in `extra_terms`, with no spec needed in the term-list form; both offset charges nonzero (sign of the offset); `constant_term(spec)` is `constant + 4 n_gᵀ E_C n_g` for every basis.
- `test/test_terms.jl`: `port_of_kind` rejects the removed kinds.
- `test/test_composite.jl`, `test_composite_slow.jl`, `test_zero_pi_reference.jl`: unchanged in intent; rerun, tolerances adjusted only with a reason recorded in the test.
- Acceptance: the fast tier passes; `KRONFREE_SLOW_TESTS=1` on `test_character.jl`, `test_composite_slow.jl` and `test_zero_pi_reference.jl` passes; `grep -rn twist src test README.md | grep -v MersenneTwister` finds nothing.

## 6. Documentation

- Spec §20d: a dated paragraph "Decided 2026-10-30: periodic gauge" stating the change, the two reasons (including the time-dependent generator argument), both costs, and that the twisted-gauge description above it is historical.
- Spec §20f: the "spec literal is structure" example changes from `n_g = "0.5"` to `n_g = "0"`; the `PeriodicGrid` sentence under "Complex structure ... enters only through" loses `twist_factor`.
- `README.md`: the "Periodic phase grid" section describes the periodic gauge, the served ports and the `nport()` example without the assembly paragraphs, and states both costs (complex at `n_g ≠ 0`; sweet-spot pairs degenerate only to `O(Δ^order)`, with the measured numbers); the "Real arithmetic" section's `"0.5"` mention becomes `"0"`.
- `docs/superpowers/STATUS.md`: item 6 (real arithmetic at special values) notes the new boundary; a line records the gauge change and its date.
- Docstrings of `PeriodicGrid`, `circuit_terms`, `constant_term`, `materialize`, `charge_port` and the `terms.jl` header follow.
- Plans and spike notes already written are history and are not edited.

## 7. Left to the builder

- Whether `_periodic_stencil` survives as a helper or the cyclic stencil is built in one pass.
- Exact tolerances in the rewritten tests, derived from §4.5 where they change.
- Order of edits and commit granularity; one branch off `main`, merging is Eli's.
