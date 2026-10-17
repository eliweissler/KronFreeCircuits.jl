# Plan 2 benchmark: stored-parent regression and end-to-end materialization

Date: 2026-10-17.
Machine: `ewdesktop` (i7-11700K, 8 physical cores; NVIDIA RTX A4500, driver 595.99.02).
Package source: commit `c28d67c` plus the benchmark scripts of the commit that adds this file (`bench/common.jl`, `bench/parent_solve.jl`, `bench/end_to_end.jl`).

## Summary

- **Part A (stored parent) meets both expectations.**
  With the audit's boundary (no guard columns), `:ritz` on the A4500 takes 3.64 s against the audit's 3.20 s (1.14×), with the audit's work count exactly (514 H columns).
  With the default `guard=4` it takes 4.35 s (1.36×).
  `:ritz` is 2.75–3.79× faster than `:jacobi` on both devices.
  Every solve validates, and CPU and GPU energies agree to 1.5e-12.
- **Part B as specified fails at its first stage.**
  A cold 300-state leaf with `LOBPCG(:ritz)` does not converge within the default `maxiter = 1000`, on either device.
  On CPU, 291 of 300 pairs converge after 266 s; on the A4500, 294 of 300 after 192 s.
  With `maxiter = 3000` (a diagnostic, not the configuration) the 300 requested pairs converge, but the guard columns do not, so the solve runs all 3000 iterations: 294 s on CPU, 217 s on the A4500.
  `Lanczos()` solves the same leaf in 84 s, which matches the audit's 90.4 s.
- **The rest of the pipeline was measured with `Lanczos()` leaves as a labelled substitute.**
  A full-tree materialization takes 237 s on CPU: two 84 s leaves, a 68 s cold parent solve, and about 0.5 s for everything else.
  On the A4500 the cold parent solve takes 41 s.
  Port formation after the device solve takes 1.24 s on the host with one BLAS thread, about a third of a warm parent solve.
- **The next stage to optimize is the leaf solver, then device-side port formation.**

## Environment and versions

| | |
|---|---|
| Julia | 1.13.2, run as `julia -t 1` |
| BLAS | OpenBLAS (`LBTConfig([ILP64] libopenblas64_.so)`); 8 threads for CPU runs, 1 for GPU runs |
| KronFreeCircuits | 0.1.0 (`c28d67c` + bench scripts) |
| ITensors / NDTensors | 0.9.32 / 0.4.32 |
| LOBPCGEigensolver | 0.1.4 |
| KernelAbstractions | 0.9.45 |
| KrylovKit | 0.10.4 |
| Adapt / JSON3 | 4.7.4 / 1.14.3 |
| CUDA.jl / cuTENSOR.jl | 6.4.2 / 6.4.2 (CUDA runtime 13.4.0, driver API 13.3.0 as CUDA.jl reports them) |

Each JSON file records its own environment, including the package versions it loaded.
All runs used the `test/cuda` environment in the disposable directory `ewdesktop:~/kronfree-plan2-ci/`.
That environment holds CUDA and cuTENSOR, but CPU runs never load them.

## Part A: the audit's stored retained parent

The operator is the audit's `actual two zero-pi 300x300` parent (`larger-real300-operators.bin`, SHA-256 `c66d7253…4ab4ae`, the same file the audit used).
It has dims 300 × 300, two diagonal energy terms, and one dense charge–charge coupling with coefficient 0.05.
The script rebuilds it as a `KronFreeOperator` and checks it against the stored probe: ‖H x − y‖ / ‖y‖ = 9.6e-17.
It first solves the stored point cold for 120 states (`:ritz`, `guard=4`).
It then scales the coupling by 1.01 (the audit's 1 % step) and warm-starts from those vectors, with `tol = 1e-8` throughout.

How the numbers were obtained:

- Both operators are built on the solve's device (`on=CuArray` or `Array`) with `block_capacity = 120 + guard` before any timing.
- Each warm configuration runs one untimed `lobpcg_solve` and then three timed ones.
  A timed sample is one `lobpcg_solve(LOBPCG(; preconditioner, guard, on), H, 120; X0)` call, wrapped in `CUDA.synchronize()` and run after `GC.gc()`.
  The table gives the median, with the minimum–maximum range below it.
- A sample therefore includes `lobpcg_solve`'s own entry work, which the audit's resident timing did not.
  This entry work is the host-side seed perturbation (0.25 s), the block upload (0.027 s) and the result download (0.099 s), 0.38 s in total on the GPU.
  On CPU it is 0.23 s + 0.09 s.
  Each part was measured once per run, separately.
- Every timed sample is validated afterwards, outside the timing: each true residual ‖H v − λ v‖ is recomputed with a host copy of the operator and must be below 1e-8, and ‖V'V − I‖ must be below 1e-10.
- The default guard (4) and the audit's boundary (`guard=0`) ran as separate processes, `BENCH_GUARDS=4` and `BENCH_GUARDS=0`.
  Each process repeats the cold solve, so there are two cold samples per device.

| Configuration | Audit A4500 | A4500, guard 0 | A4500, guard 4 | Audit CPU, 8 threads | CPU 8 threads, guard 0 | CPU 8 threads, guard 4 |
|---|---:|---:|---:|---:|---:|---:|
| fixed Jacobi, seconds | 10.178 | **10.985** (10.940–10.988) | **16.450** (16.428–20.002) | 15.671 | **16.923** (16.857–17.057) | **25.588** (25.028–36.475) |
| fixed Jacobi, H columns / iterations | 1409 | 1415 / 59 | 1883–2143 / 317–585 | 1409 | 1415 / 59 | 1929–2571 / 347–1000 |
| `:ritz`, seconds | 3.197 | **3.642** (3.627–3.834) | **4.346** (4.326–4.366) | 5.721 | **6.153** (6.136–6.188) | **7.676** (7.281–8.069) |
| `:ritz`, H columns / iterations | 514 | 514 / 7 | 571–574 / 23–26 | 514 | 514 / 7 | 574 / 26 |
| fixed / `:ritz` | 3.18× | 3.02× | 3.79× | 2.74× | 2.75× | 3.33× |
| ours / audit, `:ritz` | | 1.14× | 1.36× | | 1.08× | 1.34× |

Every sample has true residuals below 9.85e-9 and orthonormality error below 1.4e-13.
Within a process, `:jacobi` and `:ritz` energies agree to 1.1e-12.
CPU and GPU energies agree (`BENCH_COMPARE`) to 1.5e-12 for every configuration, cold and warm; the requirement is 1e-9.
One CPU `:jacobi, guard=4` sample reached `maxiter = 1000` with one guard column unconverged, and warned.
Its 120 requested pairs had converged and validated; it is the 36.5 s maximum.

Other part-A measurements:

| Measurement | CPU, 8 threads | A4500 |
|---|---:|---:|
| Cold solve of the stored point (`:ritz`, guard 4), one sample per process | 67.2 s, 70.3 s (4718, 4908 H columns) | 41.7 s, 40.6 s (4864, 4725 H columns) |
| Operator construction on the device | 0.022 s | 0.150 s (first construction), 0.002 s (second) |
| Operator upload alone (`_on_storage(H_host, CuArray)`, median of 3) | — | 0.001 s |
| Port formation after the warm `:ritz` solve: the coupling's two charge ports on 120 states, host, median of 3 | 0.25 s | 1.24 s (one BLAS thread) |

Port formation calls `KronFreeCircuits._port_matrix` with the stored dense factors on the solved vectors.
These are the ports an ancestor of this parent would need.

## Part B: two coupled symmetric zero-pis, stage by stage

The tree is `two_zero_pi.json` at `two_zero_pi_params(; g = 0.05)`.
Each leaf is `[theta_i => ChargeBasis(15), phi_i => PhaseGrid(-6π, 6π, 301; order=8)]` (dimension 9331), with `LOBPCG(:ritz)` computing and keeping 300 states.
The parent is 300 × 300 = 90 000 states and keeps the lowest 120 with `LOBPCG(:ritz)`.
The parent also forms the `n` ports of `theta1` and `theta2` in its kept basis (`ports=[...]`), so that port formation after the parent solve is a measured stage.
This parent is the stored part-A operator.
The root energies of the materialized tree agree with part A's cold energies to 4.3e-10.

### The specified leaf solve fails

| Leaf solve (leaf 1 materialized alone, one sample, no full-size warm-up) | CPU, 8 threads | A4500 |
|---|---:|---:|
| `LOBPCG(:ritz)`, `maxiter = 1000` (the configuration) | **FAILED** after 266.3 s: 291 / 300 converged, maximum residual 5.4e-6, 101 120 H columns | **FAILED** after 191.5 s: 294 / 300 converged, maximum residual 9.7e-7, 97 637 H columns |
| `LOBPCG(:ritz)`, `maxiter = 3000` (diagnostic) | 294.1 s, 3000 iterations, 111 671 H columns; 300 pairs validated; 2 of 4 guard columns unconverged (warning) | 217.0 s, 3000 iterations, 103 779 H columns; 300 pairs validated; 2 of 4 guard columns unconverged (warning) |
| `Lanczos()` (substitute; host), median of 2 per leaf | 84.2 s (84.0–84.4) and 84.1 s (84.0–84.1) | — |

The leaf energies agree to 2.1e-11 across all three converged solves: CPU `LOBPCG` (3000), GPU `LOBPCG` (3000) and `Lanczos`.
A progress trace of the GPU solve (an exploratory run, not saved) showed the pattern.
No column locks in the first 100 iterations, which take 71 s and 30 704 H columns.
296 columns are locked at iteration 1000 and all 300 by iteration 1400 (about 195 s).
The remaining guard columns stall.
Smoke-sized leaves (13 × 101 points, 30 states) show the same stall: they need 850–1000 iterations, and several leaf solves end at `maxiter` with guard warnings (`logs/*-smoke-b.log`).
Those smoke solves validated, because their 30 requested pairs had converged.
The likely cause is that the diagonal preconditioners carry little information for this leaf.
In the charge × phase-grid basis the cosine potential is off-diagonal, since it shifts the charge.
The phase-grid kinetic stencil has the same diagonal at every point.
So `D` resolves neither the low-lying structure nor the stiff high-frequency part of the spectrum.
This diagnosis was not tested further.

### Stage breakdown, with `Lanczos()` leaves

The parent stays `LOBPCG(:ritz)` on the run's device.
Each line is a separate measurement.
The leaf-dominated stages are single samples or medians of 2, because one full materialization takes about 4 minutes and each remote command was bounded at 10 minutes.
Every validated solve is below its tolerance: residuals of 4.3e-11 at the leaves (Lanczos) and below 9.5e-9 at the parent, with orthonormality error below 4e-12.

| Stage | CPU, 8 threads | A4500 (host work on one BLAS thread) | How obtained |
|---|---:|---:|---|
| Leaf materialize, each leaf | 84.2 s, 84.1 s | — | `leaf` stage, `materialize` on the leaf's own terms, median of 2 |
| Full-tree materialize, cold | 237.4 s | 208.5 s | CPU: `tree` stage, one timed sample. GPU: the untimed setup `materialize` of the `parts` run, measured with `@elapsed` (its leaves run `Lanczos` on the host with one BLAS thread) |
| Leaf operator construction | < 1 ms | < 1 ms | `operator(sys, leaf1)`, median of 2 after a warm-up |
| Parent operator construction (on the solver's storage) | < 1 ms | 1 ms | `operator(sys)` |
| Parent operator upload alone | — | 1 ms | `_on_storage(host copy, CuArray; capacity = 124)` |
| Parent solve alone, cold, 120 states | 68.3 s (68.1–68.5), 4794 H columns, 53 iterations | 41.0 s (40.9–41.1), 4750 H columns, 47 iterations | `lobpcg_solve(root solver, operator(sys), 120)`, warm-up then 2 samples |
| Parent eigenvector download (90 000 × 120) | — | 0.052 s | `Array(CuArray(eigenvectors(sys)))`, the download alone |
| Port formation, each leaf (1 port, 300 states) | 0.021 s | 0.113 s | `_port_matrix` on the leaf's kept vectors, for every port `required_ports` assigns |
| Port formation, root (2 ports, 120 states) | 0.254 s | 1.237 s | the same, on the root's kept vectors |
| Remainder (selection, phase fixing, peak amplitudes, edge checks) | ≈ 0.5 s | — | CPU tree total minus the medians above; the medians come from different processes, so this is approximate |

The parent's 1.24 s port formation after a device solve is a third of the 3.64 s warm parent solve (part A).
It is 3 % of the cold one.

### Robustness cases from the catalogue

All of these are observations, not speed gates.

- **F3 (identical uncoupled zero-pis, g = 0).**
  Run at smoke size only (13 × 101 leaves, 30 kept states, 12 parent states), on both devices.
  The tree is materialized warm from the g = 0.05 system.
  Parent energies match the sorted pairwise sums of its leaf energies to 5.7e-14 (warm) and 8.5e-14 (cold).
  All 5 exchange-degenerate pairs are found.
  On the complete 12-state prefix, the weight lost from the oracle's product-state subspace is below 1e-14.
  The warm parent takes 65 H columns; a cold parent solve takes 234–236.
  Not run at full size: the specified LOBPCG leaves fail, and with Lanczos leaves the run needs two materializations (about 8 minutes on CPU) plus solves, which does not fit one remote command.
- **F8 (strong coupling g = 0.5 and a leaf-cutoff study).**
  Run at smoke size only, on both devices.
  The tree is materialized warm across the large step 0.05 → 0.5.
  For the lowest 6 parent states, truncating the leaves to k = 20, 10 and 5 states changes the energies by 8.9e-4, 8.8e-3 and 2.2e-2 relative to k = 30.
  LOBPCG and Dense agree to 2.6e-13 on every truncated projected operator.
  At the large step, the warm parent solve takes 159–161 H columns and the cold one 303–307.
  Not run at full size, for the same reason as F3.
- **F9 (phase-grid convergence).**
  Run at full size on CPU with `Lanczos()` leaves, one sample each.
  Widening the leaf grid from 301 to 451 points at the same spacing (12π → 18π) moves the ground state from −1.819215 to −1.819242.
  The largest change is 0.12 over the lowest 20 states, 6.7 over the lowest 100 and 29 over all 300 (GHz).
  The maximum edge support falls from 0.11 to 0.069, and 300 → 295 kept states have edge support above 1e-6.
  The lowest ten states move by between 2.2e-5 and 1.3e-2 GHz.
  The 301-point grid is too small for most of the 300 kept states, as the expected grid-edge warnings say.
  The smoke-sized version (101 → 151 points) shows the same pattern on both devices.
- **F6 (asymmetric `full_zero_pi.json`).** Not run: it needs its own three-mode tree, and there was no time left in this task for it.

### Warnings

The run logs (`logs/`) and the `warning_counts` field of each JSON file record the warnings.
A warning carrying `maxlog` is counted only while the console still shows it, and the counts include the small-tree compilation warm-up.
They are:

- grid-edge warnings on every leaf, expected on this grid;
- guard-column warnings on every 300-state LOBPCG leaf solve that returned, and on many smoke-sized ones;
- five branch-phase hybridization warnings at the full-size parent (kept states 15, 46, 50, 67 and 83);
- one guard warning in part A (CPU, `:jacobi`, guard 4).

## Expectations

**Part A, `:ritz` on the A4500 within 1.5× of 3.20 s: met.**
It takes 3.64 s at the audit's boundary (no guards) and 4.35 s with the default four guards.
At `guard = 0` the work matches the audit exactly (514 H columns, 7 iterations).
Of the 0.44 s difference, 0.38 s is `lobpcg_solve`'s entry work (seed perturbation, block upload, result download), which the audit's resident timing excluded.
The resident solve is therefore within about 2 % of the audit.
The CPU rows are 8 % slower than the audit's at the same work (514 H columns for `:ritz`; 1415 against 1409 for `:jacobi`).
The default guards cost 19 % on the A4500 and 25 % on CPU at `:ritz`, and 50 % on both devices at `:jacobi`, where the four random guard columns converge slowly.

**Part A, `:ritz` at least 2× faster than `:jacobi` on both devices: met**, at 3.02× and 3.79× on the A4500 and 2.75× and 3.33× on CPU.

**Part B, the leaf solve well under the audit's 90 s: missed.**
The configured `LOBPCG(:ritz)` leaf does not converge 300 states within `maxiter = 1000` on either device.
Raising `maxiter` to 3000 converges them in 217 s (GPU) and 294 s (CPU), 2.6–3.5× slower than `Lanczos()` at 84 s, and the guard columns still do not converge.
**The leaf solver is the stage to optimize next.**
LOBPCG needs a preconditioner that sees the off-diagonal phase-grid structure, for example an exact solve with each one-dimensional factor, or a shift-and-invert of the dominant axis.
Until then, `Lanczos()` is the practical choice for cold primitive leaves.

**Part B, port formation after the device solve visible as its own line: met.**
It takes 1.24 s at the parent on the host with one BLAS thread, against 0.25 s on eight CPU threads.
That is a third of a warm parent solve on the A4500.
**Device-side port formation** (spec §20c, decision 4) is the next stage after the leaf solver, and the more important one in warm sweeps, where the parent solve is short.

**Residuals below tolerance and host/device agreement to 1e-9: met for every solve that completed.**
Residuals stay below the 1e-8 tolerance throughout, and orthonormality error stays below 4e-12.
Parent energies agree between devices to 1.5e-12, and leaf energies to 2.1e-11.

## Commands

On the Mac (smoke checks, CPU, not saved here):

```bash
BENCH_SMOKE=1 BENCH_DEVICE=cpu julia --project=. -t 1 bench/end_to_end.jl
BENCH_SMOKE=1 BENCH_DEVICE=cpu BENCH_FIXTURE=docs/spikes/2026-10-16-itensor-audit/profiling/larger-real300-operators.bin \
    julia --project=. -t 1 bench/parent_solve.jl
```

Preparing `ewdesktop`:

```bash
rsync -a --delete --exclude .git --exclude Manifest.toml --exclude 'test/cuda/Manifest.toml' --exclude docs/spikes \
    --exclude bench/fixtures --exclude out ./ ewdesktop:~/kronfree-plan2-ci/
ssh ewdesktop 'mkdir -p ~/kronfree-plan2-ci/bench/fixtures'
scp docs/spikes/2026-10-16-itensor-audit/profiling/larger-real300-operators.bin ewdesktop:kronfree-plan2-ci/bench/fixtures/
ssh ewdesktop 'cd ~/kronfree-plan2-ci && ~/.juliaup/bin/julia --project=test/cuda -e "using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()"'
```

Every run below was one sequential, foreground `ssh ewdesktop` invocation of this form:

```bash
cd ~/kronfree-plan2-ci && env <variables> BENCH_COMMIT=c28d67c timeout 590 ~/.juliaup/bin/julia -t 1 --project=test/cuda <script>
```

| Output (this directory) | Script | Variables |
|---|---|---|
| `parent_solve-cpu-guard4.json` | `bench/parent_solve.jl` | `BENCH_DEVICE=cpu BENCH_GUARDS=4 BENCH_FIXTURE=bench/fixtures/larger-real300-operators.bin` |
| `parent_solve-cpu-guard0.json` | `bench/parent_solve.jl` | `BENCH_DEVICE=cpu BENCH_GUARDS=0 BENCH_FIXTURE=…` |
| `parent_solve-gpu-guard4.json` | `bench/parent_solve.jl` | `BENCH_DEVICE=gpu BENCH_GUARDS=4 BENCH_FIXTURE=… BENCH_COMPARE=out/parent_solve-cpu-guard4.json` |
| `parent_solve-gpu-guard0.json` | `bench/parent_solve.jl` | `BENCH_DEVICE=gpu BENCH_GUARDS=0 BENCH_FIXTURE=… BENCH_COMPARE=out/parent_solve-cpu-guard0.json` |
| `end_to_end-{cpu,gpu}-leaf.json` | `bench/end_to_end.jl` | `BENCH_DEVICE=… BENCH_STAGES=leaf BENCH_LEAVES=1 BENCH_WARMUP=0 BENCH_LEAF_REPEATS=1` |
| `end_to_end-{cpu,gpu}-leaf-maxiter3000.json` | `bench/end_to_end.jl` | the same plus `BENCH_LEAF_MAXITER=3000` |
| `end_to_end-cpu-lanczos-leaf.json` | `bench/end_to_end.jl` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=lanczos BENCH_STAGES=leaf BENCH_LEAVES=1,2 BENCH_WARMUP=0 BENCH_LEAF_REPEATS=2` |
| `end_to_end-cpu-lanczos-tree.json` | `bench/end_to_end.jl` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=lanczos BENCH_STAGES=tree BENCH_WARMUP=0 BENCH_LEAF_REPEATS=1` |
| `end_to_end-{cpu,gpu}-lanczos-parts.json` | `bench/end_to_end.jl` | `BENCH_DEVICE=… BENCH_LEAF_SOLVER=lanczos BENCH_STAGES=parts BENCH_WARMUP=0 BENCH_LEAF_REPEATS=0 BENCH_REPEATS=2` |
| `end_to_end-cpu-lanczos-f9.json` | `bench/end_to_end.jl` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=lanczos BENCH_STAGES=f9` |
| `end_to_end-{cpu,gpu}-smoke.json` | `bench/end_to_end.jl` | `BENCH_DEVICE=… BENCH_SMOKE=1 BENCH_REPEATS=2` (all stages) |

Each run's JSON went to `out/` through `BENCH_OUT=out/<name>` and was copied here with `scp`, with the run's console output in `logs/`.
The failed-leaf runs (`end_to_end-{cpu,gpu}-leaf.json`) and the F9 run came from an earlier revision of `end_to_end.jl`.
That revision timed each stage's validation inside its `repeated` samples.
The failed runs never reached validation, so their times stand.
F9 times only `materialize` and never used `repeated`, and its code is identical in the final script.
Every other part-B file and all part-A files come from the committed scripts.
The stored fixture (5.5 MB) is not committed: it lives in the untracked audit directory `docs/spikes/2026-10-16-itensor-audit/profiling/`.
