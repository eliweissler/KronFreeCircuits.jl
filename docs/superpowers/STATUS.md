# Status, 2026-11-11

**2026-11-10:** three projects are scoped before a feature freeze (circuit-quantization interface, mean-field basis augmentation, perturbative derivatives); see `docs/superpowers/HANDOFF-2026-11-10.md` for the designs, the plan in progress and what is blocked.
The mean-field proposal and plan of 2026-10-28 (documents that are not in this repository) are superseded by the 2026-11-10 design `docs/superpowers/specs/2026-11-10-mean-field-augmentation-design.md` and were never implemented.

## The circuit-quantization interface (project 1, 2026-11-10)

One Julia session can now define a circuit with circuit-quantization through PythonCall, build its `Spec` in memory in any admissible coordinate frame, solve it, and write a run record (design `docs/superpowers/specs/2026-11-10-circuit-quantization-interface-design.md`, plan `docs/superpowers/plans/2026-11-10-circuit-quantization-interface.md`, architecture §20j; §2, §15 and §20b are amended).
The work is on `main` (local; not pushed as of 2026-11-11, and the remote branch `cq-interface` holds only an early commit of it); pushing is Eli's.

Done:
- In the core, without Python: constant coefficients that fold and never compile (`_Constant`), and `numeric_spec`, a `Spec` of constants from plain arrays.
- The package extension `KronFreeCircuitsPythonCallExt` (weak dependency `PythonCall`): `circuit_quantization()`, the symbolic route `Spec(res; frame=S)`, the loop route `QuantizedCircuit(res; values)` and `Spec(qc, S)`, and the run records `record_json`, `save_record` and `load_record`.
- After the final review, by the owner's decision, `run_record` was replaced by `save_record(path, spec; circuit, frame, values, ...)` and `record_json(spec; ...)`, which write the spec that was solved (by `spec_json`) with the circuit, frame and values as provenance, and by default check that the provenance rebuilds that spec (architecture §20j, amendment of 2026-11-10).
- The guard that the core stays free of Python: a default-tier test that no file under `src/` loads or calls PythonCall, and that the extension is not loaded there.
- `bench/kite_frames`: the KITE defined, moved through its published, normal-mode and optimum frames, solved and recorded in one session (results in `bench/kite_frames/README.md`).
- Round-off zeroing as an explicit option, `zero_rtol = 1e-12` and `zero_atol = 0.0` on `numeric_spec`, `Spec(qc[, S])`, `Spec(res; ...)`, `record_json` and `save_record` (Eli's decision: zeroing at a relative tolerance is acceptable but must be explicit; a bench may pass `1e-9`), committed here at `106c0b8` and in circuit-quantization at `c8c8714`.

Measured on ewdesktop (i7-11700K, Julia 1.13.2, one thread), 2026-11-10: the loop route takes 0.134 ms per frame change for the three-mode KITE (target 5 ms), and the symbolic route 0.37 s for the KITE and 0.80 s for the six-mode coupled zero-pis.
The bench's loop figures are in `bench/kite_frames/README.md` and in architecture §20j.
In the bench the three frames' four lowest levels agree to 7.2e-6 GHz and lie about 1.4e-5 GHz above the circuit-quantization paper's reference levels; solved on the reference calculation's own grids (charge cutoff 24, 160-point phase grids, steps about 0.28 to 0.32), the KITE reproduces the paper's levels to within 5e-7 GHz, so the offset is the reference grid's discretization error (`bench/kite_frames/README.md`; `run.jl` reproduces it with `KITE_FRAMES_REFGRID=1`, run on 2026-11-10 on commit `d3f338c`).

Regression check: a two-fluxonium bench smoke run before and after this branch gave bitwise identical levels in all four stages.
Test status, 2026-11-10, after the record functions were renamed (`save_record` and `record_json` in place of `run_record`, commit `ebe350c`) and their review fixes (commit `af14c5c`, which changed only the extension, the Python tier's tests and documents): the Python tier passed on the code of `af14c5c` with 336 tests against circuit-quantization at `c8c8714` (branch `to-spec-zero-tolerance`).
The default tier passed on the code of `af14c5c`.
The KITE bench passed all its checks with default settings on the code of `af14c5c`.
The slow tier (`KRONFREE_SLOW_TESTS=1`) passed on the code of `ebe350c`, whose `src/` and default- and slow-tier tests `af14c5c` does not change; its only warnings were the 20 `fix_phases!` warnings that the commit before this branch also prints.
The bench's run with `KITE_FRAMES_REFGRID=1` was last made on the code of `d3f338c`.

Acceptance item 5, the three-fluxonium bench from a spec produced by `to_spec` with no hand assembly (plan Task 9):
- The bench's spec now comes from circuit-quantization's `to_spec(..., zero_rtol=1e-9)`, with no hand assembly.
- Regenerated on 2026-11-10, it has the same parameters, modes, cosine vectors and zero pattern as the hand-assembled spec, and coefficients within 2.4e-9 of the matrix scale (the old script rounded its numbers to 8 decimals).
- The bench directory `bench/hensel_three_fluxonia/` is outside version control, and its `run.jl` was not run, by the owner's instruction.
- The script still makes the junction columns of the coordinate matrix `Z` exact, because one cosine otherwise keeps an `omega` entry of 1.0000000000000002.
- On this machine the frame modes f1 and f2 come out with the opposite sign from the files made on the Mac.

Dependency on circuit-quantization (`~/src/circuit-quantization`): the interface needs that repository at commit `c8c8714` or later.
Its commits of 2026-11-10 on its `main` (at `9c26fd2`) bring an integer-valued float compact block accepted, the export round-off repair, the numeric frame change with its checks, and run records.
The zero-tolerance keywords of its `to_spec` are only in `c8c8714`, on its local, unpushed branch `to-spec-zero-tolerance`, which is not yet merged into that repository's `main`.
`Spec(res)` always passes those keywords and `record_json` and `save_record` forward them, so every call of any of them needs that commit, and so does the Python tier; against an older circuit-quantization they throw an `ArgumentError` naming the required commit.
On ewdesktop that repository is checked out on its local branch `to-spec-zero-tolerance`; returning it to `main` breaks `Spec(res)`, `record_json`, `save_record` and the Python tier until that branch is merged.

Python tier, on ewdesktop (the environment is the conda environment `cq-z4`; the plan's `circuit_quantization` environment exists only on the Mac):

```bash
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$HOME/anaconda3/envs/cq-z4/bin/python $HOME/.juliaup/bin/julia --project=test/python -e 'using Pkg; Pkg.instantiate()'
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$HOME/anaconda3/envs/cq-z4/bin/python $HOME/.juliaup/bin/julia --project=test/python test/python/runtests.jl
```

## Mean-field basis augmentation (project 2, 2026-11-11)

A child of a node can be solved in the mean field of its siblings, and the mean-field states of several self-consistent solutions become the basis through which it is presented to its parent, whose operator is unchanged (design `docs/superpowers/specs/2026-11-10-mean-field-augmentation-design.md`, plan `docs/superpowers/plans/2026-11-10-mean-field-augmentation.md`, architecture §20k, README "Mean-field bases").
The work is on `main` (local; not pushed as of 2026-11-11); pushing is Eli's.
The ledger with every ruling is `.superpowers/sdd/2026-11-10-mean-field-augmentation/progress.md`, a local file that is not tracked (`.superpowers/` is git-ignored); the content of each ruling cited below is stated where it is cited.

Done:
- Mean-field terms from expectation tables, the mean-field child and the product energy, the self-consistent loop with several starts and warm starts, and the distinct solutions (`src/meanfield.jl`).
- The union basis, the augmented child (which keeps its basis and bare terms for the perturbative-derivatives project), `materialize(...; augmented)`, `augmented(sys)`, and the wrapper `mean_field_augment` (`src/augment.jl`).
- `suggest_bases` and `edge_weight` (`src/suggest.jl`).
- The KITE fixtures from the prototype, circuit-quantization's `product_state.py` (`test/python/generate_kite_fixtures.jl`, `test/fixtures/kite/`), and their comparison in all three frames (`test/test_meanfield_kite.jl`, slow tier).
- `bench/kite_meanfield`: Figure 8(b) (`levels.jl`) and the frame optimization (`optimize.jl`), both described in its README; the optimization's recorded results are tracked in `bench/kite_meanfield/recorded/`.
- After the final review: each child's bare solve is formed once and shared by the loop and `augment` (`mean_field_solutions(...; bare)`), an augmented child records its `θ` and `materialize` refuses it at a `θ` that differs in a parameter its bare terms depend on, and `suggest_bases` reports a binding grid cap (`when_capped = :warn`, `:error` or `:allow`).

Measured (details in architecture §20k and `bench/kite_meanfield/README.md`):
- Against the prototype's wells, product energies agree to 1.1e-11 GHz or better in every frame, and mean-field levels to 5.9e-13 GHz in the normal-mode and optimum frames and 2.50e-6 GHz in the published frame.
- The paper's reference ground energy is reproduced: on the grid of the reference calculation (the owner's script that produced the figure) the published frame's four lowest levels lie within 3.2e-7 GHz of the paper's, with the tree converged to 6.5e-9 GHz.
- Figure 8(b): all twelve errors are within a factor of two of the reference calculation's, with the same ordering of the frames at every `m`; the ratios are 1.0000 except in the published frame at `m = 16` and `20` (0.9986 and 0.9726).
- The frame optimization from the identity ends at a frame whose two wells are at −9.429473654505 GHz against −9.429473599561 for the paper's optimum on the same bases, and whose scaled `S` is within 1.03e-4 of the paper's `S_OPT` in every entry (0.696303 against 0.6962 is the largest difference), not equal to it to the four printed decimals.

Missed, and why:
- The published frame's mean-field levels miss the design's 1e-6 GHz target on one mode (2.50e-6 against the prototype; compared to 1e-5 in the test): this loop's convergence test uses its tables, the prototype's also ⟨φ⟩ and ⟨n⟩ of every mode, so the two stop one iteration apart (traced; the operators agree to 5e-13 at equal inputs; Ruling 10: the convergence test stays on the tables and that one comparison is made to 1e-5).
- The published frame's Figure 8(b) ratios 0.9986 and 0.9726: inferred, not proven, to come from Gram–Schmidt remainders on `varphi_2` that straddle the `1e-6` drop threshold, so the union depends on round-off; the reference calculation's own union is not recorded.
  Those remainders are the loop's convergence residue, not differences between the two wells, as measured in the final review (a separate script, not in the repository): tightening the loop's `tol` from 1e-11 to 1e-13 lowers the distance of well 2's states 1 to 8 from the span of well 1's twelve on `varphi_2` tenfold (1.2e-8…1.8e-6 to 1.1e-9…1.7e-7) and leaves the other modes unchanged; the tables converge only to about `sqrt(tol)` ≈ 3e-6.
- The frame optimization did not converge: it ended on its 150-iteration cap.
  The warm start makes the score history-dependent at about 3e-10 GHz, a floor of about 3e-4 on the forward-difference gradient, so the gradient tolerance of 1e-6 cannot be met (Ruling 14: the warm start stays in the score).
  The plan's 76-point grid cap made the score non-variational in stretched frames (it went below the true ground level), so the score uses 400 points and scores `Inf` above that (Ruling 12).
  The recorded run predates a gradient fix for a case that never occurred in it (Ruling 15).

Departures from the design (each with its reason in architecture §20k): the figure's grids and starts are those of the script that produced it (33 charge states and 120 points per extended mode from `suggest_bases(charge_cutoff=16, max_points=120, extent_scale=1.5)`, and 33 starts), not the notebook's; a solution's stored states are solved at its final tables, one extra solve per child (Ruling 9); the two-fluxonium test asserts only what is physics, not four fixed points (Ruling 4); a warm start is recorded as `:warm` (Ruling 5); `operator(sys, node)` throws for an augmented node, and nested and `:real` augmented nodes are errors (Ruling 7); the optimizer's stopping rules: a gradient norm below 1e-6, a cost change below 1e-9 GHz, a failed line search, or 150 iterations (Ruling 13).
Not done: the Harmonium benchmark (design §6.2) is deferred to its own plan, the next one for this feature; derivatives through the mean-field bases are not provided (the perturbative-derivatives project holds them fixed).

Known limitations a user can trip on: `operator` on an augmented node throws; a warm start whose tables lack a key fails with a `KeyError` once mixing is on (keys move with the frame, so `optimize.jl` rebuilds tables by mode name); `constant` passed to the spec form of `mean_field_solutions` is a duplicate keyword; an augmented child is checked against the call's `θ` only in the parameters its own bare terms depend on (not against the terms, nor the parameters of the terms inside a branch child); grid growth in `optimize.jl` can exceed its 400-point cap.

Open design points from the final review, for the owner (not fixed):
- The damping trigger ("after the energy rose twice") does not detect a period-2 cycle, and round-off decides whether mixing switches on, so the set of converged runs is not reproducible across machines.
- `drop = 1e-6` sits below the accuracy of the vectors at `tol = 1e-11` (tables converge to about `sqrt(tol)` ≈ 3e-6) and spends basis slots on convergence residue: tie it to `sqrt(tol)`, or expose it through the wrapper, before the Harmonium work.
- `converged` is a step test, so the energy is good to about 1e-10, not to `tol`.
- Every ancestor of an augmented child is forced complex, which doubles the root solve's cost.
- `default_starts` enumerates both π and −π on compact modes, duplicates that count toward `max_starts`.
- The duplicate filter of `mean_field_solutions` can keep an unconverged run over a converged duplicate.
- The wrapper uses unconverged runs without a warning (it returns their count).
- `mean_field_solutions` keeps every run's vectors until the distinct filter (design §7's memory point is open).
- In `optimize.jl`, a tighter loop `tol`, or freezing the warm start across a gradient stencil, would lower the gradient floor more cheaply than cold starts.

Test status, 2026-11-11, on the final code of this feature, after the final review's fixes (ewdesktop, Julia 1.13.2, under heavy load from another job, load average 8 to 12):
- Default tier: passed, 1221 tests, no warnings, 4m15s.
- Slow tier (`KRONFREE_SLOW_TESTS=1`): passed, 17283 tests, 21m58s; its only warnings are the 20 `fix_phases!` warnings that `main` also prints.
- Python tier: passed, 336 of 336 tests, no warnings, 2m36s, against circuit-quantization at `c8c8714`.
- `bench/kite_meanfield/levels.jl`: all its checks passed, with the twelve Figure 8(b) errors and ratios unchanged, 4m32s.
- `bench/kite_meanfield/optimize.jl` with `KITE_MEANFIELD_MAXITER=1` (and `KITE_OPTIMIZE_OUT` elsewhere): ran end to end and exited with status 1, as a one-iteration run must (its wells are not at the optimum); its iterations 0 and 1 equal the recorded run's bitwise.

## Perturbative derivatives (project 3, 2026-11-11)

First and second derivatives of every kept energy, kept state, port and joint port of a materialized tree, with respect to named spec parameters, at the root and at every node, by perturbation theory carried upward through the tree (design `docs/superpowers/specs/2026-11-10-perturbative-derivatives-design.md`, plan `docs/superpowers/plans/2026-11-11-perturbative-derivatives.md`, architecture §20l, README "Derivatives"; §2, §11, §13, §13.1, §15, §16, §18, §20a, §20b and §20c are amended).
This replaces the reverse-mode plan of architecture §11, which this file called "plan 5".
The work is, as of 2026-11-11, on the local branch `derivatives`, off the local `main`; merging it into `main` and pushing are Eli's.
The run's ledger is a local file that is not tracked (`.superpowers/` is git-ignored); the content of each ruling cited below is stated where it is cited.
It was built overnight without waiting for plan approval, on Eli's instruction of 2026-11-10; the decisions he has not seen are listed under "Decisions still Eli's".

Done:
- Symbolic derivatives of coefficient expressions over the whole grammar, `differentiate` (`src/diffcoeff.jl`).
- Retention of the computed but unkept pairs, `Node(...; retain = :auto | :always | :never)` and `Retained` (on by default except at a `Dense` node above 1000 states, and except in the solves that are never differentiated: the mean-field loop's and `bo_surface`'s).
- The derivative operator with its three contributions, the first- and second-order rules at one node, the passes through the tree, and the API `derivatives(sys; wrt, order, mixed, scale, ...)` read through `energies`, `eigenvectors` and `port` (`src/derivoperator.jl`, `src/resolvent.jl`, `src/derivatives.jl`).
- Iterative nodes: a projected, Jacobi-preconditioned conjugate-gradient solve on the complement of the computed states, in which each column converges to `lintol` relative to its own part outside the computed states, or stops at the round-off floor (since the final review and its re-review, which found the solve failing, then inaccurate, for weakly coupled children; tested per column at couplings 0.3 to 3e-4); the selection-boundary warning (architecture §11.3).
- Near-degenerate blocks with an effective Hamiltonian, `blocks` and `block_levels` (`src/derivblocks.jl`).
- Augmented children (basis held fixed) and `GivenSystem` children declared independent with `given_independent = true`.
- `spectrum(root, spec, θ; ports)` and its `rrule` and `frule` in the extension `KronFreeCircuitsChainRulesCoreExt` (weak dependency `ChainRulesCore`), tested directly and through Zygote in `test/chainrules`.
- A symbolic KITE fixture (`test/fixtures/kite/optimum_symbolic.json`, generator `test/python/generate_kite_symbolic.jl`) and its acceptance tests, slow tier.
- `bench/derivative_design`: a three-parameter design problem solved with BFGS on the Zygote gradient; `bench/retention_cost.jl` and `bench/derivative_sums.jl`.
- Device tests of the derivative pass at `LOBPCG(on = CuArray)` nodes (`test/cuda/test_derivatives_device.jl`): a real and a complex two-transmon root, mixed pairs on both, weak coupling at the complex root, and the KITE doublet's block at the KITE root (the last four added on 2026-11-11).

Measured (details in architecture §20l):
- Finite differences: the reviewers' independent checks at untested parameter values agree to 2.4e-10 to 1e-7 at first order and to 2.8e-9 to 4e-7 at second order for energies and ports (3.5e-6 for the second derivatives of the vectors of a complex two-level tree), on leaves, two- and three-level trees, joint ports and augmented children.
- The KITE doublet (Δ = 0.0135992 GHz) at an `LOBPCG` root: the block is found; first and second derivatives in the flux offset agree with finite differences to 4.8e-9 and 6.0e-7 relative; the curvature matches `4c²/Δ` to 1.3e-6 relative; the effective Hamiltonian reproduces a full re-solve within 9.9e-9 GHz at δ = 0.01, where the Taylor expansion misses by 1.17e-2; the iterative and dense routes agree to 3.2e-12.
- The effective Hamiltonian's error on the `cos 2φ` doublet falls as δ⁴ (1.155e-7, 4.24e-9, 2.87e-10, 1.85e-11 GHz at δ = 0.05, 0.02, 0.01, 0.005).
- The linear solve: at most 190 iterations in the tests, 49 to 60 at the KITE root; at the default `LOBPCG` tolerance the derivatives still match the dense route to 4.3e-12 at `g = 0.3`.
  At couplings 0.03 to 3e-4 it threw at the default settings until the final review's fix, and after that fix it was inaccurate for small columns (2.8e-6 of a column's own norm at `g = 3e-4`); with the stopping rule of the re-review's fix, all 32 cases of the reviewer's comparison finish in 3 to 7 iterations, and every kept state's derivative is within 4.4e-10 of its own norm from the dense route at couplings 0.03 to 3e-4 and within 4.4e-11 at 0.3, at first and second order (measured per column); the default solvers stay within 5.1e-11 of it at `g = 0.3`.
- On the device (2026-11-11, RTX A4500, `test/cuda/test_derivatives_device.jl`), every comparison with the host's `Dense` twin is measured.
  At a complex `LOBPCG` root of two transmons (periodic grids), along `E_J2`, `g` and `n_g1`: first-order energies within 2.0e-13, vectors 7.2e-13 (4.6e-12 per column), ports 2.0e-12 (the off-diagonal elements of `expiphi` 6.5e-13); second order (x, x) energies 3.0e-14, vectors 3.0e-12 (8.0e-12 per column), ports 2.2e-12 (`expiphi` off-diagonal 1.0e-12); all relative.
  Mixed pairs `(g, n_g1)` and `(E_J2, g)` on the real and the complex tree: energies 5.1e-13, vectors 3.0e-12 (5.8e-12 per column), ports 2.4e-12.
  Weak coupling at the complex root, along `E_J2`: 4.3e-11 per column at `g = 0.03` and 3.7e-10 at `g = 3e-4`.
  At the KITE's `LOBPCG` root (1728 states, complex): the doublet's block on states [1, 2] has the same detuning (0.0135992 GHz, 6.3e-12 relative) and coupling (1.73193, 6.8e-13) as the `Dense` route; `block_levels` at δ = ±0.01 and 0.005 agree to 9.9e-14 GHz; point derivatives in `delta` agree to 3.6e-12 (first) and 7.6e-12 (second) for energies and 7.4e-12 and 1.7e-11 per column for vectors.
  Every linear solve converged to a residual of at most 9.8e-11, none at the floor, in at most 7 iterations on the two-transmon trees and 59 at the KITE root.
- The final review's cross-feature experiments against finite differences (three-level complex trees with an iterative interior node, augmented children under an iterative root, `indices` and `within` selections, flux offsets on periodic grids, `retain = :never` leaves, Zygote through an iterative root) agree with them to within about 1.5 times the finite differences' own error; its comparison of the default solvers with the dense route (not with finite differences) agrees to 5.1e-11 (architecture §20l has the table).
- The optimization demonstration: BFGS converged on its gradient tolerance in 13 iterations (1.6 s), reaching the planted design to about 1e-11; the gradient agreed with finite differences to 1.9e-9 relative at the start.
- The cost of retention at a `Dense` leaf with `compute = 10`: forward time ×4.8 and memory 73 KB → 5.2 MB at 801 states (transmon), ×1.9 and 83 KB → 6.5 MB at 901 (fluxonium); 0.0064 → 0.0111 s and 30 KB → 0.51 MB on the KITE tree.
- How fast the sum over states converges (first order, three cases on two `Dense` leaves): 1e-8 in the state derivatives needs about the 40 to 80 nearest of 596 other states on a fluxonium and 10 of 57 on a transmon; port derivatives converge faster (architecture §20l has the table).

Missed, and why:
- The device tier did not pass: 13 errors in older complex-`LOBPCG` testsets of `test/cuda/test_device.jl`, none in the new derivative test (10 of 10).
  Diagnosed as a fault on this machine, not this project and not the dependency versions: cuBLAS's complex-double triangular solve (`cublasZtrsm_v2_64`) fails on every call with `CUBLAS_STATUS_EXECUTION_FAILED`, even on a 2×2 matrix, and LOBPCGEigensolver reports it as a failure to orthonormalize; it fails the same way on `origin/main`, with the older environment, and in a copy of the environment that passed on 2026-10-18.
  Why it passed on 2026-10-18 is not established, nor whether the CUDA runtime artifact is damaged (its tree hash does not match its name).
  Ruling 11: recorded as a machine fault, not worked around.
  Since 2026-11-11 a complex iterative node's derivative pass, mixed pairs and a near-degenerate block are tested on the device as well (measured above, test status below).
- Iterative-node tests tighten the eigensolver tolerance (Decision 11), and second-order finite-difference tests use `rtol = 1e-5` (the design's target is `1e-6`) because second differences are themselves good only to `1e-8` to `8e-6` on the test circuits; the reviewers' checks above give the actual agreement.
- The second-order state, normalization and gauge formulas were derived for this project, because Eli's notes (`tight_binding.tex`) are not on this machine; they were checked on random Hermitian families (1e-11 and 1e-8), re-derived by a reviewer and confirmed by finite differences, but not compared with the notes (Ruling 4).
- Found by the final review and fixed in its fix wave: the linear solve failed at default settings for a parameter internal to a weakly coupled child under an iterative parent (its convergence was measured against the small part of the right-hand side outside the computed states, below round-off); a `NaN` child under an iterative parent threw instead of propagating; a `wrt` parameter nothing depends on gave silent zeros; retention reached the mean-field loop and `bo_surface`, where it is never used.
- Untested paths a user could reach: a `NaN` child column entering a parent through more than one level (one level is tested); a near-degenerate block at an augmented node; a pair that clashes only in a mixed second-order pass; the terms routed below an augmented node are not checked against those of its bare solve.

Departures from the design and decisions Eli has not seen (each with its reason in architecture §20l):
- `scale` (the `δx` of the block test) is a keyword `NamedTuple` per parameter (Ruling 3; design §4.4 marks this as unconfirmed).
- The `:follow` restriction is narrowed (Ruling 2): only a node whose phases were actually fixed by following a previous solution is an `ArgumentError` (workaround: re-materialize at the point without `previous`); a `follow` selection is fine with `phase = :positive_sector` given explicitly, or without `previous` (note that `keep = follow(...)` defaults to `phase = :follow`).
- Retention is on by default (Ruling 7, despite the measured cost above); the default was not changed after the convergence measurement, nor after the final review measured a stored sweep (the `System`s of a 21-point sweep of one 901-point leaf take 130 MB with retention against 0.81 MB without).
- The linear solve is a hand-written conjugate gradient that runs on any device (P3), and the `ShiftInvert` factorization is not reused, which is that choice's cost; each column converges on the scale of its own complement part, `‖Q R_j‖`, or stops at the round-off floor.
- The selection-boundary threshold without `scale` is the plan's (P8): a detuning at most 1e-6 times the distance from the lowest computed energy.
- A `wrt` parameter nothing depends on is an `ArgumentError` when the spec does not declare it, else zeros with a warning.
- Exactly degenerate coupled states give `NaN` derivatives with a reported block, and a `NaN` child makes every ancestor `NaN` (plan Decision P5), at a `Dense` or an iterative parent, with one warning per node naming the child.
- The effective Hamiltonian of a block is second order in the displacement (plan Decision P6).
- Design §4.8 is partly wrong on Born–Oppenheimer reduced specs: a `fit_surfaces` spec keeps the fitted-over parameters, so derivatives with respect to them are derivatives of the fit, not an error (Ruling 2).
- The plan assumed the symbolic KITE fixture differs from `test/fixtures/kite/optimum.json` by a flux gauge; a reviewer showed they are the same circuit (coefficients coincide at `delta = 0`, levels agree to 1e-13).
- `ChainRulesCore` is a weak dependency whose extension is in practice always active (it is loaded through `ITensors`); Zygote 0.7.16 works on Julia 1.13.2; differentiating twice through the rule fails with an obscure "Flatten" error, and second derivatives come from `derivatives(...; order = 2)`.

Not done: a reverse sweep, third order, and derivatives with respect to the frame or through basis construction (design non-goals); a truncated, convergence-checked sum over the nearest states (measured, not built).

Side effects on this machine during the run:
- Implementers twice ran `Pkg.develop` in a test environment by mistake: once in a scratch worktree's `test/` (the worktree was deleted), once in this checkout's `test/` (reverted with `git checkout -- test/Project.toml` before any test ran); no repository environment was left changed.
- `test/chainrules` was resolved and instantiated (Zygote 0.7.16, ChainRules 1.74.13 and their dependencies downloaded), `bench/derivative_design` likewise (Optim 2.3.2, Zygote 0.7.16), and `test/cuda`, which had no Manifest, offline (nothing downloaded; it resolved CUDA 6.5.1 and GPUArrays 12.2.0); their Manifests are git-ignored.
- The root project was resolved once for the new extension: only its `project_hash` and the extension lines of the git-ignored root Manifest changed, no package version.
- The device diagnosis used scratch worktrees (removed) and instantiated one offline with the older Manifest (nothing downloaded).

Baseline test status, 2026-11-11, on the code before the final review's fixes, with the first version of these documents (ewdesktop, Julia 1.13.2, under load from another job, load average 8 to 19):
- Default tier: passed, 28 files, no warnings, 26m33s.
- Slow tier (`KRONFREE_SLOW_TESTS=1`): passed, 40m47s; its only warnings are the 20 `fix_phases!` warnings that `main` also prints; its notices are the `F2 n_g = 1/2` note of `test/test_catalogue.jl` (also on `main`) and the two this project's tests print on purpose, the largest linear-solve iteration count of `test/test_derivatives_iterative.jl` (190) and of `test/test_derivatives_kite.jl` (60).
- Python tier: passed, 336 of 336 tests, no warnings, 3m51s, against circuit-quantization at `c8c8714`.
- Rules tier (`test/chainrules`): passed, 26 tests (25 in its three testsets, 9 + 9 + 7, and the extension check), no warnings, 1m40s.
- Device tier (`test/cuda`, the GPU otherwise idle): 311 passed and 13 errored, as expected from the cuBLAS fault above: all 13 in the complex-`LOBPCG` testsets of `test/cuda/test_device.jl`, and the derivative test 10 of 10.
- `bench/derivative_design/design.jl`: all four checks passed, with the recorded run's numbers to every printed digit except timings, 1m13s.
- `bench/retention_cost.jl`: ran; the sizes equal the recorded ones, and the times were 0.035 → 0.087 s (transmon), 0.076 → 0.152 s (fluxonium) and 0.0055 → 0.0102 s (KITE) for `retain = :never` → `:auto`.

Test status, 2026-11-11, on the code after the final review's fixes and before the convergence-floor change below (ewdesktop, Julia 1.13.2, load average 1 to 6.6 from another job):
- Default tier: passed, 28 files, no warnings, 5m50s.
- Slow tier (`KRONFREE_SLOW_TESTS=1`): passed, 15m03s; its only warnings are the 20 `fix_phases!` warnings that `main` also prints; its notices are the `F2 n_g = 1/2` note of `test/test_catalogue.jl` and the largest linear-solve iteration counts of `test/test_derivatives_iterative.jl` (188) and `test/test_derivatives_kite.jl` (60).
- Python tier: passed, 336 of 336 tests, no warnings, 1m42s, against circuit-quantization at `c8c8714`.
- Rules tier (`test/chainrules`): passed, 30 tests in its three testsets (9 + 9 + 12) and the extension check, no warnings, 52s.
- Device tier (`test/cuda`, the GPU otherwise idle before and after): 313 passed and 13 errored, the 13 expected from the cuBLAS fault above, at the same lines of `test/cuda/test_device.jl`; the derivative test 12 of 12, including its new check that the root runs on `CuArray`.
- `bench/derivative_design/design.jl`: all four checks passed (gradient at the start 1.936e-9 relative, final loss 3.8e-23), 34s.
- `bench/retention_cost.jl`: ran; the sizes equal the recorded ones, and the times were 0.0185 → 0.0508 s (transmon), 0.0515 → 0.0986 s (fluxonium) and 0.0032 → 0.0060 s (KITE) for `retain = :never` → `:auto`.
- The final review's scripts, rerun: the weak-coupling comparison (all 32 cases finish, state derivatives within 2.1e-8 of the dense route), the conjugate-gradient floor check (no column stagnates, at `lintol` down to 1e-11, in four fresh solves; before, one stagnated in each) and the `NaN` check (a `Dense` and an `LOBPCG` parent report `NaN` at both orders, with the child's block, and nothing throws).

Test status of the complement solve's stopping-rule fix (2026-11-11, after the re-review; branch `derivatives-cg-floor`, ewdesktop, one Julia process at a time while the owner benchmarked): only these were run after the change, all passing: `test_derivatives_complement.jl`, `test_retention.jl`, `test_meanfield.jl`, `test_derivatives_iterative.jl`, `test_derivatives_kite.jl`, `test_derivatives_node.jl`, `test_derivatives_tree.jl`, `test_derivatives_second.jl`, `test_derivatives_second_tree.jl`, `test_derivatives_blocks.jl`, `test_derivatives_stored.jl` and `test_spectrum.jl` by name, and the final review's scripts for weak coupling, the conjugate-gradient floor, the default solvers and the per-column accuracy.
After the follow-up commit (the floor's accuracy bound documented, the solve checks of the tests extended to floor columns, a fast per-column case at `g = 0.3`, the floor error message), `test_derivatives_complement.jl`, `test_derivatives_tree.jl` and `test_derivatives_iterative.jl` were run by name, all passing.
Every tier was then rerun once on the code with this change, 2026-11-11, on a quiet machine (load average below 1), one at a time.
The default tier passed with no warnings in 3m35s.
The slow tier passed in 13m11s; its only warnings are the 20 `fix_phases!` ones, and its two notices are the iteration counts that `test_derivatives_iterative.jl` and `test_derivatives_kite.jl` print on purpose.
The Python tier passed 336 of 336 in 1m30s.
The rules tier passed (9, 9 and 12 tests in its three testsets) in 52 s.
The device tier gave 313 passed and 13 errored: the 13 errors are the known complex-`LOBPCG` testsets of `test/cuda/test_device.jl` (lines 61, 252, 258, 280, 313, 394, 435 to 437, 522 and 524 to 526), and `test/cuda/test_derivatives_device.jl` passed 12 of 12.
After a reboot of the machine on 2026-11-11 the cuBLAS fault was gone, and the device tier, rerun on the same code with the GPU otherwise idle, passed 469 of 469 in 2m24s, with no errors: the complex-`LOBPCG` testsets pass and `test/cuda/test_derivatives_device.jl` passes 12 of 12.
After the device tests of the derivative pass at a complex node, mixed pairs, weak coupling and the KITE block were added (2026-11-11, branch `derivatives-cg-floor`, the GPU otherwise idle before and after), the device tier passed 573 of 573 in 2m43s; `test/cuda/test_derivatives_device.jl` passes 116 of 116.

## Where things stand

Plans 1, 1b, 2, 2b, 3 and 4 are done.
Plan 3 runs nodes whose structure guarantees a real Hamiltonian in `Float64` (spec §20f); on the two-zero-pi tree the full materialization takes 9.9 s on CPU and 5.0 s on the A4500, against 15.4 s and 12.6 s in plan 2b (`docs/spikes/2026-10-18-plan3-benchmark/results.md`).
Plan 4 makes spec files come from `circuit-quantization` (`to_spec`, `save_spec`, `load_spec`; format authority `~/src/circuit-quantization/docs/spec-format.md`), and seven of the eight hand-written fixtures are cross-checked against generated ones (spec §20g).
The Born–Oppenheimer reduction is done and merged into this branch (design `docs/superpowers/specs/2026-10-19-born-oppenheimer-design.md`; architecture §20h): `promote_mode`, `bo_surface`, `slow_spec`, `fit_surfaces` and reduced specs saved with `save_spec`; the gated `test/test_born_oppenheimer_reference.jl` reproduces Di Paolo et al.'s E_α and E_γ to 0.2% (−0.11% and −0.12%) with E_Cφ derived from the plasma-frequency identity (decided 2026-10-20; `bench/born_oppenheimer/results.md`).
Two ShiftInvert fixes from side sessions are merged: a block-Lanczos breakdown at invariant subspaces, and recovery on highly degenerate operators.
All of this is merged into `main` and is on `origin/main`.
Coefficients now share one compiled function per distinct expression (a fixed argument name in `_compile`), so a sweep through `materialize(spec)` no longer recompiles every coefficient at every point (about 35 ms per point before, 2.5 ms after on the zero-pi BO leaf).

Verification on the branch (2026-10-20): the fast tier and the full suite (`KRONFREE_SLOW_TESTS=1`, every file, 600 s) pass on the Mac at dc4afc6; the device suite on ewdesktop (457/457) last passed at e2a6c62 and has not been rerun since the BO work.
The full suite (`KRONFREE_SLOW_TESTS=1`, every file) passed on `composite-ports` at a199fe6 in 728 s on the Mac on 2026-10-26.
The SDD ledgers of plans 3 and 4 (every ruling and deferred minor) are in the side-session notes folder of the session that ran them.

Composite ports (joint ports) are done (2026-10-26) on the branch `composite-ports`, off `main`; merging and pushing are Eli's.
A term with several factors inside one child is applied through that child's joint port, formed before the projection (architecture §20i; requirements `docs/superpowers/specs/2026-10-26-composite-ports-requirements.md`).
The Z4 hierarchy `[[1,4,5],[2,6,7],[3]]` is now accepted; the pilot rerun and the convergence sweep are pending on ewdesktop.

The periodic-gauge change (2026-10-30/11-01) is merged into `main` (fast-forward, 2026-11-01); pushing is Eli's.

## Next plan

Pre-autodiff quick wins done (2026-10-20): test tiers, reduced specs cross-checked numerically against circuit-quantization, BO hardening, and the duplicate `_primitive` test helper removed; merging this branch into `main` is Eli's.
Born–Oppenheimer follow-ups, next: the KITE of Smith et al. (BO design §9: varphi slow, (φ, θ) fast, after a shear removing the charge–charge coupling), and the data-valued `DiagonalOp` entry into a `Term` (§20a; BO design §2 and §4.3; it was deferred to plan 5 only because the old approach wrapped `materialize`, and the perturbative derivatives no longer block it).
Z4 three-way pilot rerun on ewdesktop with the `composite-ports` branch (Eli): the command is in the requirements document; check that the hierarchy materializes, the root operator is Hermitian to roundoff, and record the lowest eight energies and the run time in the pilot's `run.json`; then the convergence sweep.
Plan 5, automatic differentiation, was replaced by the perturbative derivatives above (architecture §20l).
Then plan 6 (export, caching, NPZ), the sweep layer, and the noise plan (which needs the format's `operators`/`noise_channels` sections, §20b, on both sides).
Proposed by Eli (2026-10-20), not yet designed:
- Local matrix-element queries in the full space after materialization (a single-mode operator, or a product across subsystems, between the root's kept states, lifted on demand through the stored eigenvectors); this is the core the noise plan's element operators would use. A product of factors on modes inside one child needs a joint port; the joint port now exists (§20i) for terms of the Hamiltonian, and only the user-requested form (`Node(ports=...)`, selection rules) is missing.
- A more robust circuit ingest than the JSON interchange in the long term, such as a direct SymPy interface to circuit-quantization (since 2026-11-10 the circuit-quantization interface above builds specs in memory; the JSON route stays); the file route is guarded by load checks and, for the BO reduced specs, a numeric cross-check of spectra and potentials (`test/fixtures/bo/validate_reduced.py` → `test/fixtures/bo_reference/python_reduced_levels.json`, compared in `test/test_bo.jl`).

## Open items, in priority order

1. Test time is now tiered (Eli, 2026-10-20): the default `Pkg.test()` runs the unit-level files (14 in about 50 s then; 28 since the perturbative derivatives, 5m50s on ewdesktop at load average 5 to 6.6, 26 minutes under heavier load); `KRONFREE_SLOW_TESTS=1` runs every file, about 10 minutes on the loaded Mac (`SLOW_FILES` and `REFERENCE_FILES` in test/runtests.jl); `Pkg.test(test_args=[...])` runs named files. Most of the full suite's time is first-call compilation of the materialize and solver paths; a PrecompileTools workload would cut it for tests and users alike (not done).
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
6. Real arithmetic at special parameter values (a parameter `n_g` at 0 on a `PeriodicGrid`, zero flux in a charge-basis SQUID) is a recorded later item (§20f); such circuits run complex. Since the periodic gauge (2026-10-30) a grid node is real only at the literal `n_g = "0"`.
7. Pre-existing: `ChebyshevFilter` at degree ≤ 3 with tol ≥ 1e-2 can still race (documented).
8. The cos 2φ fixture has no generated counterpart (one compact mode with junction harmonics 2 and 1 is not a lumped circuit).

9. Born–Oppenheimer surfaces (§13.1), checked on 2026-10-19 with a throwaway two-stage zero-pi spike (Di Paolo et al., arXiv:1811.07921v1, App. B): a user-side "promote mode to parameter" term-list transformation (about 20 lines) gives a fast φ node that classifies real and matches an independent dense λ₀(θ̃) to 2e-12 over 64 points; the low-level `KronFreeOperator` takes the resulting surface U(θ) as a `DiagonalOp` with the `PeriodicGrid` kinetic term and matches a dense build to 3e-15.
   The gap is the one recorded in §20a: a data-valued operator (the surface fed back as a potential) has no entry into a `Term`, so stage 2 cannot go through `materialize`; it was to be designed with plan 5, since autodiff was to wrap `materialize`, which the perturbative derivatives (§20l) do not, so it is now an item of its own.
   The promotion transformation is now in the package as a spec-level `promote_mode` (BO plan, §20h); the data-valued entry is still open.

## Decisions still Eli's

- Selection cluster closure over near-degenerate states (not part of the perturbative-derivatives project; the §11.3 warning now exists as its selection-boundary test).
- How the `scale` (`δx`) of the near-degeneracy test is supplied (design §4.4); built as a keyword `NamedTuple` per parameter.
- The retention default, in view of its measured cost and of how fast the sums converge: whether a convergence-checked sum over the nearest states should replace full retention (design §4.5; both tables in architecture §20l).
  Stored results pay it too: the `System`s of a 21-point sweep of one 901-point leaf take 130 MB with retention against 0.81 MB without (`retain = :never` per node restores the old size); the solves inside the mean-field loop and `bo_surface` no longer retain.
- Plan Decision P5 at symmetric operating points: a non-degenerate root whose derivative exists reports `NaN` because a child keeps both members of an exactly degenerate, coupled pair; a parallel-transport gauge inside the child would give correct parent energies (not built).
- Test tolerances looser than the agreement measured (not changed): iterative and device vectors and ports 1e-6 / 1e-5 against 1e-11 to 1e-12; KITE `block_levels` 1e-4 against 1e-8, its curvature 1e-2 against 1.3e-6; trees 1e-6 against at most 4.3e-8; stored-data children 1e-6 / 1e-5 against 2e-9 / 4e-7; the rules 1e-6 against about 1e-8.
- Not built, for a later decision: a pairs-only second-order request; reuse of the `ShiftInvert` factorization; `show` methods for `Derivatives` and `NearDegenerateBlock`; `extra_terms` in `spectrum`; a clear error for differentiating twice through the rule.
- The plan's choices listed under "Departures from the design" above (and in architecture §20l): `scale`, the narrowed `:follow` rule, the hand-written solver, the selection-boundary threshold without `scale`, the handling of unused parameters.
- Whether an augmented child's union basis should ever move with the parameters (today it is held fixed, so the derivative is that of the projected problem).
- Whether followed sweeps (`phase = :follow` with `previous`) need differentiating; today they are refused.
- The device fault: reboot or reset the GPU and rerun a 2×2 `CUBLAS.trsm!` with `ComplexF64`; if it persists, re-download the CUDA runtime artifact, then report it to JuliaGPU/CUDA.jl (no compat bound: the older stack fails the same way).
  Resolved on 2026-11-11 by a reboot of the machine: the 2×2 `CUBLAS.trsm!` with `ComplexF64` then succeeded and the device tier passed 469 of 469.
  What the reboot changed is not established; it also brought kernel 7.0.0-42 in place of 7.0.0-38, with the same driver version (595.99.02).
- Comparing the second-order state and normalization formulas derived for the derivatives (architecture §20l) with `tight_binding.tex`.
- Merging `derivatives` into `main`, and pushing.
- Guard stopping rule for LOBPCG on stiff leaves.
- Branch phase at the joint peak: antisymmetric exchange states with vanishing joint-peak amplitude (fallback vs the tabled Cauchy–Schwarz search), and the related observation from plan 3 that the peak argmax can tie on exchange-symmetric trees, so global signs of some root states differ between arithmetics (energies and ports agree).
- Whether to commit the spike directories (`docs/spikes/2026-10-16-itensor-audit`, `2026-10-16-pathological-test-catalogue`, `2026-10-17-leaf-solvers`), untracked in the main checkout.
- Resuming 3D leaves via the (θ, φ) + ζ hierarchy.

## Decided recently (for reference)

- `PeriodicGrid` in the periodic gauge (2026-10-30, design `2026-10-30-periodic-gauge-design.md`): fixed cyclic `n`, `n2`; offset charge only in the quadratic form; `twist_factor` and the derivative-piece ports removed; the sweet-spot pairs degenerate to `O(Δ^order)` on the grid.
- Real arithmetic only when structurally guaranteed; all parameters real; spec expressions never use `im`, `cis`, `cispi` (§20f).
- Native output types for real nodes; `ShiftInvert` follows the operator's element type (`isreal_assembled` retired).
- The exporter moved ahead of autodiff for fixture provenance; it is a general `circuit-quantization` save/load module (`io.py`).
- `full_zero_pi.json` keeps scqubits 4.3.1's `dC_J` sign, opposite to Groszkowski et al. (arXiv:1708.02886v1, eq. `eq:Hzeropi1`); the mapping is documented in the fixture, §20g and circuit-quantization's format document.
- Nonsymmetric `E_C`/`E_L` are rejected on load in both packages (non-physical).
