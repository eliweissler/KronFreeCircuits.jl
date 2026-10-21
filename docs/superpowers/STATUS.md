# Status, 2026-10-20

## Where things stand

Plans 1, 1b, 2, 2b, 3 and 4 are done.
Plan 3 runs nodes whose structure guarantees a real Hamiltonian in `Float64` (spec §20f); on the two-zero-pi tree the full materialization takes 9.9 s on CPU and 5.0 s on the A4500, against 15.4 s and 12.6 s in plan 2b (`docs/spikes/2026-10-18-plan3-benchmark/results.md`).
Plan 4 makes spec files come from `circuit-quantization` (`to_spec`, `save_spec`, `load_spec`; format authority `~/src/circuit-quantization/docs/spec-format.md`), and seven of the eight hand-written fixtures are cross-checked against generated ones (spec §20g).
The Born–Oppenheimer reduction is done and merged into this branch (design `docs/superpowers/specs/2026-10-19-born-oppenheimer-design.md`; architecture §20h): `promote_mode`, `bo_surface`, `slow_spec`, `fit_surfaces` and reduced specs saved with `save_spec`; the gated `test/test_born_oppenheimer_reference.jl` reproduces Di Paolo et al.'s E_α and E_γ to 0.2% (−0.11% and −0.12%) with E_Cφ derived from the plasma-frequency identity (decided 2026-10-20; `bench/born_oppenheimer/results.md`).
Two ShiftInvert fixes from side sessions are merged: a block-Lanczos breakdown at invariant subspaces, and recovery on highly degenerate operators.
All of this is on the branch `plan4-julia-integration`, which fast-forwards from `main`; merging and pushing are Eli's.
Coefficients now share one compiled function per distinct expression (a fixed argument name in `_compile`), so a sweep through `materialize(spec)` no longer recompiles every coefficient at every point (about 35 ms per point before, 2.5 ms after on the zero-pi BO leaf).

Verification on the branch (2026-10-20): the fast tier and the full suite (`KRONFREE_SLOW_TESTS=1`, every file, 600 s) pass on the Mac at dc4afc6; the device suite on ewdesktop (457/457) last passed at e2a6c62 and has not been rerun since the BO work.
The SDD ledgers of plans 3 and 4 (every ruling and deferred minor) are in the side-session notes folder of the session that ran them.

## Next plan

Pre-autodiff quick wins done (2026-10-20): test tiers, reduced specs cross-checked numerically against circuit-quantization, BO hardening, and the duplicate `_primitive` test helper removed; merging this branch into `main` is Eli's.
Born–Oppenheimer follow-ups, next: the KITE of Smith et al. (BO design §9: varphi slow, (φ, θ) fast, after a shear removing the charge–charge coupling), and the data-valued `DiagonalOp` entry into a `Term` (§20a; BO design §2 and §4.3, deferred to plan 5).
Plan 5, automatic differentiation (spec §11, with §20c decision 9 and §20f "What stays complex": derivative operators and pullbacks promote to complex where a cotangent or derivative operator is complex).
Then plan 6 (export, caching, NPZ), the sweep layer, and the noise plan (which needs the format's `operators`/`noise_channels` sections, §20b, on both sides).
Proposed by Eli (2026-10-20), not yet designed:
- Local matrix-element queries in the full space after materialization (a single-mode operator, or a product across subsystems, between the root's kept states, lifted on demand through the stored eigenvectors); this is the core the noise plan's element operators would use. A product of factors on modes inside one child needs the composite port that is still design debt (§20a).
- A more robust circuit ingest than the JSON interchange in the long term, such as a direct SymPy interface to circuit-quantization; today the interchange is guarded by load checks and, for the BO reduced specs, a numeric cross-check of spectra and potentials (`test/fixtures/bo/validate_reduced.py` → `test/fixtures/bo_reference/python_reduced_levels.json`, compared in `test/test_bo.jl`).

## Open items, in priority order

1. Test time is now tiered (Eli, 2026-10-20): the default `Pkg.test()` runs the 14 unit-level files in about 50 s; `KRONFREE_SLOW_TESTS=1` runs every file, about 10 minutes on the loaded Mac (`SLOW_FILES` and `REFERENCE_FILES` in test/runtests.jl); `Pkg.test(test_args=[...])` runs named files. Most of the full suite's time is first-call compilation of the materialize and solver paths; a PrecompileTools workload would cut it for tests and users alike (not done).
2. `ShiftInvert(slices=8)` measured 4.35 s against 2.9 s for a single shift on the benchmark leaf; `Auto()` keeps the single shift.
3. Done (2026-10-20, `bench/dense_cutoff/results.md`): `Auto()`'s `dense_cutoff` is lowered from 2500 to 1000.
   `Dense` loses to `ShiftInvert` (leaves) and `LOBPCG` (branches) from below 100 to about 350 states, or about 700 to 960 for a 1-mode transmon leaf, on an M2 at 1 and 4 BLAS threads, so about 1000 is the speed-optimal cutoff for all shapes together; between 1000 and 2000 states `Dense` takes 0.04 to 0.9 s per node against 0.02 to 0.09 s.
   It had stayed at 2500 while `ShiftInvert` failed its own true-residual check intermittently from about 1000 states up; with the fix of item 5 the rerun had no failed sample, and a 30-seed scan of the three leaf shapes at 1000 to 8000 states went from 112 raised solves of 900 to none.
   Separate leaf and branch cutoffs are not worth it.
4. Done (2026-10-20): `nport()` on a `PeriodicGrid` mode expands automatically in the spec form of `materialize` (`ports=` on any node, selection rules, `extra_terms`) into `d1 + e^{2πi n_g} d1wrap₊ + e^{-2πi n_g} d1wrap₋ + n_g I`; the term-list form, which has no offset charge, still refuses it, naming the mode.
5. ShiftInvert, remaining loud failures (pre-existing, rare): a `LAPACKException(22)` from `stegr!` inside KrylovKit's scalar Lanczos (about 5 in 36 000 Cholesky runs); an interior shift with `certify` raises on a short count (no deflated recovery on the LU path); slicing has no deflated recovery (a slice count mismatch on degenerate spectra: 52 of 240 `slices = 2` runs on random-multiplicity and decoupled-sector spectra in the 2026-10-20 scan).
   Fixed (2026-10-20, Eli's choice): the true-residual check at `tol = 1e-8` raised on 2 to 8 of 10 seeds on 2000- to 8000-state fluxonium and transmon leaves, and on 12 to 100 % of near-crossing runs with a stiff tail, although the vectors matched `Dense` to 4e-11.
   Lanczos stops the last pairs of `(H − σI)⁻¹` with a Krylov residual `r` just under `inner_tol`, and `H v − E v = −(E − σ)(H − σ) r` magnifies its stiff components by up to `|E − σ|·‖H‖` (`‖H‖` grows like `1/h²` on a grid, `ncut²` in a charge basis).
   `ShiftInvert` now polishes its pairs by one block inverse-iteration step with its factorization and a Rayleigh–Ritz step (1 to 8 ms), and accepts residuals up to the roundoff floor `10 eps·‖H‖` (∞-norm of the assembled matrix) when that exceeds `tol`, with a debug message.
   An adversarial scan against `Dense` (1440 runs: random multiplicities, near crossings 1e-7 and 1e-9 apart at `m`/`m + 1` with a stiff tail, decoupled sectors; real and complex; block 1, block 2, two slices) gave no wrong energy, projector or orthonormality before or after.
   The "marginal true-residual failure on block = 2 over random-multiplicity spectra" recorded here before was likely the same mechanism; block 2 had none in the scan.
   Shared rule (2026-10-21, Eli: the two leaf solvers are alternatives for the same node): `ChebyshevFilter` accepts by the same `max(tol, 10 eps·‖H‖)`, `‖H‖` from `gershgorin_bound(H)` in both (`_residual_floor`), and locks below half of it.
   Its reach stays shorter: its residuals stall at about 2.6, 10 and 25 eps·‖H‖ on 1000-, 4000- and 8000-point fluxonium grids (polished `ShiftInvert`: 1 to 3), so it raises at `maxiter` on the 8000-point grid at `tol = 1e-8`, energies converged to 1e-12; a Jacobi-corrected Rayleigh–Ritz step would fix charge bases (184 → 2 eps·‖H‖) but gains only about 4× per step on an 8th-order grid, and was not added.
   Separately, `ChebyshevFilter` at m = 6 on the 7999-state transmon leaf did not converge in 40 to 50 iterations (residual 2e-5; m = 20 converges).
  Investigated (2026-10-21): no stall and no solver defect, but a weak filter (inherent to the method at a fixed degree).
  With `λ_cut = θ_46 ≈ 2080` against `λ_max ≈ 6.4e7` the residuals fall steadily by 1.72 per iteration, as `cosh(100 · acosh|x(θ_6)|) ≈ 1.71` predicts, so reaching the lock threshold takes 52 iterations.
  At m = 20 the cut is `θ_60 ≈ 3550`, with 2.19 per iteration (predicted 2.17) and 37 iterations.
  The amplification cap does not bind (it allows degree ≈ 1600), the guards lock within two iterations of the wanted columns, and the energies agree with `ShiftInvert` to 5e-13.
  `degree = 200` converges in 18 iterations (3600 block applies against 5200); documented in the `ChebyshevFilter` docstring.
  The `maxiter` error now names a weak filter: when the last filter's log-gain `degree · acosh|x(θ_m)|` is below 2 (1.13 here), it reports the per-iteration factor `cosh` of it and advises raising `degree` or `maxiter`, or `ShiftInvert` on the CPU (`_WEAK_FILTER_GAIN`).
   `LOBPCG` keeps its own acceptance (Eli, 2026-10-21: it serves branches, a different situation): it returns pairs validated by LOBPCGEigensolver's residual estimate at `tol`, without a fresh true residual (one fresh residual measured 1.2e-8 at `tol = 1e-8` on an explicit 8000-point leaf), and at an explicit 8000-state zero-pi leaf it needs `maxiter` above the default 1000 (4000 converges).
6. Real arithmetic at special parameter values (a parameter `n_g` at 0 or ½ on a `PeriodicGrid`, zero flux in a charge-basis SQUID) is a recorded later item (§20f); such circuits run complex.
7. Pre-existing: `ChebyshevFilter` at degree ≤ 3 with tol ≥ 1e-2 can still race (documented).
8. The cos 2φ fixture has no generated counterpart (one compact mode with junction harmonics 2 and 1 is not a lumped circuit).

9. Born–Oppenheimer surfaces (§13.1), checked on 2026-10-19 with a throwaway two-stage zero-pi spike (Di Paolo et al., arXiv:1811.07921v1, App. B): a user-side "promote mode to parameter" term-list transformation (about 20 lines) gives a fast φ node that classifies real and matches an independent dense λ₀(θ̃) to 2e-12 over 64 points; the low-level `KronFreeOperator` takes the resulting surface U(θ) as a `DiagonalOp` with the twisted `PeriodicGrid` kinetic term and matches a dense build to 3e-15.
   The gap is the one recorded in §20a: a data-valued operator (the surface fed back as a potential) has no entry into a `Term`, so stage 2 cannot go through `materialize`; it should be designed with plan 5, since autodiff wraps `materialize`.
   The promotion transformation is now in the package as a spec-level `promote_mode` (BO plan, §20h); the data-valued entry is still open.

## Decisions still Eli's

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
