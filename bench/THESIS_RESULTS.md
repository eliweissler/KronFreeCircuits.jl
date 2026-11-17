# Benchmark results for the thesis: where to find them

A map for writing up four benchmark cases. Each section gives the question the case answers, the headline numbers, the
files that hold them, and what not to claim. **The case READMEs are the authoritative record**: quote numbers from
them (or the raw files they name), not from this summary, and re-check against them if they disagree. Paths are
relative to `bench/` on branch `bench-harness`. Status as of 2026-11-17.

## Common ground

- **Machines.** Desktop "ewdesktop": Intel i7-11700K (8 cores / 16 threads), NVIDIA RTX A4500 (20 GB), Julia 1.13.2;
  KronFree runs use 8 threads unless stated. Alpine (CU Boulder): one H200 NVL (141 GB) on an AMD EPYC 9555 node and
  one A100 80 GB PCIe on an AMD EPYC 7502 node, 8 cores per job (`gpu_cluster/README.md`). Published references ran on
  their own hardware (stated per case).
- **Timing conventions.** Timed repeats come after one untimed first call (compilation excluded; first-call times are
  reported separately where they matter). "Idle machine" means one job at a time. Mean ± std of the repeats unless
  stated.
- **Accuracy rule** (`README.md`, "The accuracy rule"): where a case has a ladder of truncations, the headline is the
  time to reach 1 MHz on every transition among the lowest 8 levels, with the next rung as the check.
- **Grid edges.** Every run reports PhaseGrid edge warnings; a headline should come from a warning-free grid or say
  why the warnings do not matter (see the two-fluxonia caveat).
- **Paper sources.** The arXiv LaTeX sources are in `~/arxiv_sources_kronfree_bench/<id>/`. Verify any equation,
  convention or quoted number of a paper against that source, not memory. Notes: `~/arxiv_sources_kronfree_bench/notes/`.
- **GPU summary across cases**: `gpu_cluster/README.md` (one table, H200 / A100 / A4500 / desktop CPU).

---

## 1. Two coupled zero-pi qubits (`coupled_zero_pi/`)

**Question.** How fast does KronFree solve the hierarchy of the two-qubit zero-pi gate study (arXiv:2609.16630, the
author's own Python pipeline on scqubits 4.3.1), at the same discretization and with the same answer?

**Setup.** Each qubit's (θ, φ) leaf: `ChargeBasis(90)` × `PhaseGrid(±6π, 300, order 6)` = 54 300 states, lowest 300
levels, kept by the pipeline's connectivity rule (154 and 155 states). Root: 23 870 states, lowest 1000 levels, LOBPCG
(tol 1e-8, 8 guard columns). The pipeline computes 3000 levels; the benchmark uses 1000 on both sides.

**Headline numbers** (`coupled_zero_pi/README.md`, "Timing"; 5 repeats):

| | 1000 levels (s) |
|---|---|
| Python pipeline (scqubits), desktop | 3303 ± 117 |
| KronFree, desktop CPU (8 threads) | 80.9 ± 0.6 (41×) |
| KronFree, all on the A4500 | 42.9 ± 0.4 (77×) |
| KronFree, all on the H200 / A100 | 8.4 ± 0.2 / 19.5 ± 0.4 |

- **Agreement** ("Agreement with the pipeline"): each qubit's 300 levels ≤ 2.5e-11 GHz, kept sets identical, the 1000
  coupled levels ≤ 1.3e-11 GHz (median 6e-12).
- **Breakdown**: leaves (`ShiftInvert` 22 s each on the CPU, `ChebyshevFilter` 4.7 s each on the A4500), root
  assembly + solve, peak memory: same table.
- **Pipeline stages** (build 1081 s; qubit 2 + coupled `eigsh` 2222 s): bullets under the table.
- **Solver studies (2026-11-16)**: CPU leaves with the matrix-free `ChebyshevFilter` are 2.5× slower than
  `ShiftInvert` (`leaf_solvers.jl`); a perfect root seed would save at most about half the root solve, so a
  perturbative seed is not worth it (`warm_start.jl`). Both in the README bullets.

**Files.** `results/overnight_{cpu,gpu_root,gpu_all}/` and `.log` (KronFree), `results/pipeline_timing_1000.json`
(pipeline, 5 runs), `reference/two_qubit_data_default.json` (the pipeline's output), `results/leaf_solvers_cpu_20261116.log`,
`results/warm_start_gpu_20261116.log`; Alpine: `gpu_cluster/results/zeropi-{h200,a100}-*/`.

**Do not claim.** That the pipeline was tuned for speed (it is research code, and its 3303 s includes the 1081 s
scqubits `Circuit` build); compare against "the study's pipeline", same machine.

## 2. The enumeration's top circuit, c66 (`autodiff_c66/`, `scqubits_solve/`)

**Question.** For the top circuit of the enumeration (n3_g1_c66: J, L, CL; two modes), how fast are a solve, the
Jacobian and the full Hessian of ω01, α and four transition matrix elements with respect to E_J, E_L13, E_C23, E_L23
and the flux, against SQcircuit's PyTorch autodiff (its dev-ew branch) and against scqubits?

**Setup.** KronFree tree: varphi1 on `FockBasis(60)` keeping 40, varphi2 on `PhaseGrid(±24π, 1153, order 8)` keeping
60, a 2400-state root by Lanczos keeping 6. SQcircuit at its enumeration truncation [150, 80] in its own coordinates.
scqubits in KronFree's exact coordinates, bases and tree.

**Headline numbers** (desktop, idle, 10 trials):

| | KronFree | SQcircuit | scqubits |
|---|---|---|---|
| one solve | 0.244 ± 0.059 s | 62.2 ± 1.9 s (~255×) | 0.982 ± 0.044 s (4×) |
| converged Jacobian (solve + 5 parameters) | ~0.37 s | 2063 s at n_eig = 32 (~5500×) | — |
| Jacobian + full Hessian | ~0.77 s (0.530 ± 0.082 s for the derivative call) | not re-timed (see gaps) | — |

Sources: `autodiff_c66/README.md` ("Timing": the n_eig table, the root-solver table, the 0.37 s / 2063 s statement)
and `scqubits_solve/README.md` (the solve table).

- **Convergence of our tree** (README "Convergence of the KronFree tree", `convergence.jl`, `results/convergence.json`):
  ω01 and α to 4e-9 and 6e-9 GHz, Jacobians to 4e-8 of a row; the matrix-element derivatives to 4e-7, except the
  |n2_01|² row (1.3e-5 Jacobian, 1.4e-4 Hessian). All far below the differences to SQcircuit.
- **Agreement.** Values: SQcircuit and KronFree agree to 8e-5 relative (SQcircuit's truncation); scqubits to 9e-13 GHz.
  KronFree's derivatives match central differences. SQcircuit's autodiff gets the energy rows right at any n_eig but
  its matrix-element gradients sum over computed eigenpairs only and need n_eig ≈ 32 to converge (the n_eig table is
  the thesis figure-in-waiting: cost vs accuracy of truncated eigenvector derivatives).
- **Root solver choice**: the root-solver table (Dense 1.56 s vs Lanczos 0.24 s) motivates the `Auto` follow-up in
  `README.md`.
- **GPU**: slower on this small problem everywhere (H200 1.15 s); `gpu_cluster/README.md`.

**Files.** `autodiff_c66/results/kronfree.json`, `kronfree_quiet.log`, `root_solvers_quiet.log`,
`sqcircuit_neig_quiet.json` (the idle-machine n_eig sweep), `sqcircuit_solve_trials.json`, `sqcircuit_quiet.json`;
`scqubits_solve/results/scqubits_c66.json`. Ignore `*_smoke.json` and the non-`quiet` loaded-machine files except as
noted in the README.

**Do not claim.** That SQcircuit and KronFree solve the same discretized problem (they don't; the scqubits comparison
is the like-for-like one). The |n2_01|² row is 9 % off in SQcircuit's autodiff at every n_eig, unexplained; leave that
row out of speed-accuracy claims or state it.

## 3. Two fluxonia with a tunable coupler (`hensel_two_fluxonia/`)

**Question.** Against Hensel et al. (arXiv:2611.05627v1 Sec. IV: exact HD with Lanczos and a DMRG/TN method, published
times), how fast does a KronFree tree reach their 1 MHz accuracy class on the lowest 8 levels over their six-point
coupler-flux grid? And against scqubits on the identical tree?

**Headline numbers** (`hensel_two_fluxonia/README.md`, "The comparison", per flux point):

| method | per point |
|---|---|
| paper: exact HD, Lanczos (16-core Xeon 6517P, published) | 13.6 min |
| paper: TN, bond cap 8 / 64 (published) | 2.4 / 17.4 min |
| their space solved flat by this package, A4500 | 2.4 min |
| **KronFree binary tree, 1 MHz rung (kM 120), 8 CPU cores** | **0.86 s** (~950× exact HD, ~170× TN) |
| KronFree, 10 kHz rung (kM 240) | 1.83 s (A4500: 1.25 s and 1.98 s for the two rungs) |
| scqubits, same tree and solver (cold points) | 19.4 s vs our 0.71 s (27×); kM 240: 58 vs 1.22 s (48×) |

- **Convergence**: the ladder table ("The ladder") — rung-to-rung changes and distance to an independent flat
  reference (1.47 M states); the rungs converge onto it (0.04 kHz at the top).
- **Where the time goes**, **memory** (0.69 GB net vs their 16 GB), **scqubits breakdown** (time in solves vs in
  building operators): README sections of those names.
- **GPU**: the tree does not need it; Alpine numbers in the README's last section.

**Files.** Headline (grid ±7π/421, no edge warnings, commit `c94938d`): `results/cpu-ladder-grid7-20261117-merged/`,
`cpu-memory-grid7-rung{1,2}-merged/`, `gpu-ladder-grid7-20261116/`. Earlier ±3π/181 runs: `cpu-ladder-20261111-1916/`,
`gpu-ladder-*`, `cpu-memory-rung{1,2}/`; `gpu-paper-*` (their space), `cpu-grid-*/` (grid check); scqubits: `scqubits_ref/matched-*.json`; Alpine: `gpu_cluster/results/hensel2*-*/`.
The published numbers: `numbers.tex` and sec04 of `~/arxiv_sources_kronfree_bench/2611.05627/` (lines cited in the README).

**Do not claim.** A like-for-like hardware comparison with the paper (8-core desktop vs 16-core Xeon; the same-machine
flat solve of their space is the hardware-neutral check). Our sweep points are warm-started; theirs are cold (the
first point is cold for us too and costs about the same, 0.73 s).

## 4. DMRG of the fluxonium junction array, reproducing Di Paolo et al. (`array_fluxonium/`)

**Question.** Does a DMRG built on KronFree's circuit-quantization specs and site eigenbases reproduce Di Paolo et al.'s
(arXiv:1912.01018) Fig. 3: the charge dispersion Δω01 = ω01(n_g = 0) − ω01(½) of an N = 40 array fluxonium, z = 0.045
to 0.15?

**Headline numbers** (`array_fluxonium/README.md`):

- Figure: `results/dmrg_vs_dipaolo.{png,pdf}` (`plot_vs_dipaolo.py`, conda env `qutip4`): our DMRG at all 22 z of their
  figure, their DMRG and their single-mode Eq. 6, all read from their figure's vector PDF (`digitize_fig3.py`).
- Every one of the 22 points is at one setting: site basis D = 14, charge cutoff 12 (16 at z = 0.045, 0.05, identical).
  Ratios ours/theirs: integer z (table "Against Di Paolo's Fig. 3", second ratio row): 1.099 at 0.05, 1.001-0.999 for
  0.06-0.15. Half-step z (table "The half-step points"): 1.01-1.00 for 0.055-0.095, 0.999 for 0.105-0.145. From
  z = 0.07 up ours is a consistent 0.05-0.07 % below theirs. z = 0.045: 1.9, at the precision floor (17 Hz splitting);
  say "order of magnitude".
- Their top panel's ω01 at Φ0/2 matches ours to the fourth digit at z = 0.10-0.14.
- **What it took** (good thesis material, README sections "Small z, tightened" and "The charge cutoff"): the site
  eigenbasis D = 10 → 14 fixed z = 0.06-0.07; the site charge cutoff 8 → 12 fixed z = 0.05 (which had the wrong sign);
  the energy shift made the variance and tolerance meaningful; a random initial MPS reproduces every digit. Di Paolo
  note the same eigenbasis-over-charge-basis point (superfluxo.tex l. 136; they use 15 site states at N = 120).
- **Single-mode model**: the table "The N = 40 sweep against the single-mode model" and the paragraph "Their
  single-mode model is renormalized"; Eq. 6 with the exact ε0 vs the printed Eq. 5 (a factor √π; probably a typo).
- **Cost**: 0.5-8 h per (z, n_g) on one CPU core. GPU vs CPU at z = 0.095 (13 sweeps, identical results): H200 202 s,
  A100 320 s, A4500 423 s, desktop CPU 1648 s (8 BLAS threads) / 2184 s (1); README section "DMRG on the GPU at
  production settings" and `gpu_cluster/README.md`.

**Files.** Logs: `results/sweep-20261112-1518-z*.log` (D = 10 sweep), `results/rerun3-z*.log` (third level at large z),
`results/tight-20261114/` (D and cutoff studies), `results/tight-20261115/` (ncut 12, CPU: z ≤ 0.07 and every
half-step point), `results/tight-20261116-gpu/` (ncut 12, A4500: the integer points 0.08-0.15), `results/gpu-compare-2026111{4,6}/`; specs `specs/array_N40_z*.json`; digitized reference
`results/dipaolo_fig3_digitized.json`.

**Do not claim.** A speed comparison with Di Paolo (they report no compute times). That the DMRG is part of the
package (it is a bench script on ITensors; the package supplies the spec, site eigenbases and operators).

---

## Gaps

Closed (2026-11-16/17): the two-fluxonia headline on a warning-free grid (CPU and A4500 rerun on ±7π/421, the numbers
above); a recorded convergence check of the c66 tree (`autodiff_c66/convergence.jl`); the DMRG integer points at the
converged settings (all 22 points now D = 14, ncut 12).

Decided not to run: SQcircuit's Hessian at a converged n_eig. Quote it as an estimate, about 10 × 2063 s ≈ 6 h (ten
Jacobians at n_eig = 32), from a measured 4154 s at n_eig = 4 on a loaded machine (`autodiff_c66/README.md`).

Deferred (DMRG is secondary): a timed N = 40 comparison of the raw charge site basis against the eigen site basis.

Optional, not run:
1. **Zero-pi at the pipeline's own 3000 levels** (the benchmark uses 1000 on both sides).
2. **Zero-pi flux sweep with warm starts.** The study's real use is a sweep; warm-starting the root from the previous
   point should roughly halve the root solve (bound measured in `warm_start.jl`).
3. **Memory for the reference codes** (pipeline, SQcircuit, scqubits); recorded for KronFree only, and for the paper in
   the two-fluxonia case.
