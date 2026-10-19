# Status, 2026-10-19

## Where things stand

Plans 1, 1b, 2, 2b, 3 and 4 are done.
Plan 3 runs nodes whose structure guarantees a real Hamiltonian in `Float64` (spec §20f); on the two-zero-pi tree the full materialization takes 9.9 s on CPU and 5.0 s on the A4500, against 15.4 s and 12.6 s in plan 2b (`docs/spikes/2026-10-18-plan3-benchmark/results.md`).
Plan 4 makes spec files come from `circuit-quantization` (`to_spec`, `save_spec`, `load_spec`; format authority `~/src/circuit-quantization/docs/spec-format.md`), and seven of the eight hand-written fixtures are cross-checked against generated ones (spec §20g).
Two ShiftInvert fixes from side sessions are merged: a block-Lanczos breakdown at invariant subspaces, and recovery on highly degenerate operators.
All of this is on the branch `plan4-julia-integration`, which fast-forwards from `main` (77b2da1); merging and pushing are Eli's.

Verification on the branch: default suite passes at 932ddf9; gated suite (`KRONFREE_SLOW_TESTS=1`) and device suite on ewdesktop (457/457) pass at e2a6c62, and the changes after it touch only spec loading (symmetry check), tests, docs and generated-file metadata.
The SDD ledgers of plans 3 and 4 (every ruling and deferred minor) are in the side-session notes folder of the session that ran them.

## Next plan

Plan 5, automatic differentiation (spec §11, with §20c decision 9 and §20f "What stays complex": derivative operators and pullbacks promote to complex where a cotangent or derivative operator is complex).
Then plan 6 (export, caching, NPZ), the sweep layer, and the noise plan (which needs the format's `operators`/`noise_channels` sections, §20b, on both sides).

## Open items, in priority order

1. Default-suite time has grown to about 180–270 s on the loaded Mac (base before plan 3: about 140–160 s); plan 3's share is about +40 s of first-call compilation of the `Float64` paths, not run time. Accept it, or gate whole test files (decision pending, see below).
2. `ShiftInvert(slices=8)` measured 4.35 s against 2.9 s for a single shift on the benchmark leaf; `Auto()` keeps the single shift.
3. `dense_cutoff = 2500` in `Auto()` is untuned; no benchmark node is near it.
4. A `PeriodicGrid` mode's charge above its leaf: `ports=` and `extra_terms` with `nport()` throw with the three-piece recipe; automatic expansion is a proposed follow-up.
5. ShiftInvert, remaining loud failures (pre-existing, rare): a marginal true-residual failure on block = 2 over random-multiplicity spectra; a `LAPACKException(22)` from `stegr!` inside KrylovKit's scalar Lanczos (about 5 in 36 000 Cholesky runs); an interior shift with `certify` raises on a short count (no deflated recovery on the LU path); slicing has no deflated recovery.
6. Real arithmetic at special parameter values (a parameter `n_g` at 0 or ½ on a `PeriodicGrid`, zero flux in a charge-basis SQUID) is a recorded later item (§20f); such circuits run complex.
7. Pre-existing: `_primitive(Any)` is defined in both test/helpers.jl and test/test_selection.jl (method-overwrite warning); a Cholesky failure at the automatic shift falls back to LU with a warning rather than retrying a lower shift; `ChebyshevFilter` at degree ≤ 3 with tol ≥ 1e-2 can still race (documented).
8. The cos 2φ fixture has no generated counterpart (one compact mode with junction harmonics 2 and 1 is not a lumped circuit).

9. Born–Oppenheimer surfaces (§13.1), checked on 2026-10-19 with a throwaway two-stage zero-pi spike (Di Paolo et al., arXiv:1811.07921v1, App. B): a user-side "promote mode to parameter" term-list transformation (about 20 lines) gives a fast φ node that classifies real and matches an independent dense λ₀(θ̃) to 2e-12 over 64 points; the low-level `KronFreeOperator` takes the resulting surface U(θ) as a `DiagonalOp` with the twisted `PeriodicGrid` kinetic term and matches a dense build to 3e-15.
   The gap is the one recorded in §20a: a data-valued operator (the surface fed back as a potential) has no entry into a `Term`, so stage 2 cannot go through `materialize`; it should be designed with plan 5, since autodiff wraps `materialize`.
   The promotion transformation itself is not in the package yet (§13.1 requires only that nothing makes it hard).

## Decisions still Eli's

- Default-suite time budget (item 1 above).
- Selection cluster closure over near-degenerate states and the §11.3 warning (needed before autodiff).
- Guard stopping rule for LOBPCG on stiff leaves.
- Branch phase at the joint peak: antisymmetric exchange states with vanishing joint-peak amplitude (fallback vs the tabled Cauchy–Schwarz search), and the related observation from plan 3 that the peak argmax can tie on exchange-symmetric trees, so global signs of some root states differ between arithmetics (energies and ports agree).
- Whether to commit the spike directories (`docs/spikes/2026-10-16-itensor-audit`, `2026-10-16-pathological-test-catalogue`, `2026-10-17-leaf-solvers`), untracked in the main checkout.
- Resuming 3D leaves via the (θ, φ) + ζ hierarchy.

## Decided recently (for reference)

- Real arithmetic only when structurally guaranteed; all parameters real; spec expressions never use `im`, `cis`, `cispi` (§20f).
- Native output types for real nodes; `ShiftInvert` follows the operator's element type (`isreal_assembled` retired).
- The exporter moved ahead of autodiff for fixture provenance; it is a general `circuit-quantization` save/load module (`io.py`).
- `full_zero_pi.json` keeps scqubits 4.3.1's `dC_J` sign, opposite to Groszkowski et al. (arXiv:1708.02886v1, eq. `eq:Hzeropi1`); the mapping is documented in the fixture, §20g and circuit-quantization's format document.
- Nonsymmetric `E_C`/`E_L` are rejected on load in both packages (non-physical).
