# Plan 2b benchmark: the end-to-end two-zero-pi tree with the new leaf solvers

Date: 2026-10-17.
Machine: `ewdesktop` (i7-11700K, 8 physical cores / 16 logical; NVIDIA RTX A4500).
Package source: `src/` as of commit `0b70c8b` (Task 8 of plan 2b; unchanged by this task).
Benchmark script: `bench/end_to_end.jl` with the Task 9 edits (the leaf solver options `shiftinvert`, `chebyshev` and `auto`, and `BENCH_SLICES`), committed with this file.
The script revision of each run is the `commit` field of its JSON file: `0b70c8b+task9` for the first two runs and `0b70c8b+task9b` for the rest.
The two revisions differ only in the `parts` stage's leaf solve alone, which in `task9b` resets the leaf counter inside the timed call and records `ChebyshevFilter`'s iteration count; for `ShiftInvert` the timed path is the same.

## Summary

- **The full two-zero-pi tree now materializes in 15.4 s on CPU (8 BLAS threads) and 12.6 s on the A4500, against plan 2's 237 s on CPU** (15× and 19× faster).
  With the default `Auto()` leaves, the full tree takes 15.7 s on CPU and 12.4 s on the A4500.
- **Leaf, 9331 states, lowest 300.**
  `ShiftInvert()` materializes a leaf in 2.9 s on CPU (solve alone 2.6 s), against 84 s for plan 2's `Lanczos()` leaf and 69 s for `Dense()`.
  `ChebyshevFilter(on=CuArray)` materializes a leaf in 2.3 s on the A4500 (solve alone 2.2 s, 4 filter iterations at degree 100).
  Spectrum slicing, `ShiftInvert(slices=8)` on 8 Julia threads, takes 4.4 s, slower than the single shift.
- **Parent, 90 000 states, lowest 120, seeded from product states.**
  The cold parent solve takes 8.9 s on CPU and 4.9 s on the A4500, against 68.3 s and 41.0 s cold-random in plan 2, with 671 H columns and 16 iterations against 4794 and 4750.
- **The parent is now the largest stage**: 58 % of the CPU tree and 39 % of the A4500 tree.
  On the A4500, host port formation after the device solve (1.4 s) and the rest of the host work (about 1.6 s) together are a quarter of the tree.
- **Every timed solve validates**, and the energies agree across solvers and devices to 4.7e-12.

## Environment and versions

| | |
|---|---|
| Julia | 1.13.2, run as `julia -t 1` (the two slicing runs: `julia -t 8`) |
| BLAS | OpenBLAS (`LBTConfig([ILP64] libopenblas64_.so)`); 8 threads for CPU runs (1 in the first slicing run), 1 for GPU runs |
| KronFreeCircuits | 0.1.0 (`src/` at `0b70c8b`) |
| ITensors / NDTensors | 0.9.32 / 0.4.32 |
| LOBPCGEigensolver | 0.1.4 |
| KernelAbstractions | 0.9.45 |
| KrylovKit | 0.10.4 |
| Adapt / JSON3 | 4.7.4 / 1.14.3 |
| CUDA.jl / cuTENSOR.jl | 6.4.2 / 6.4.2 (CUDA runtime 13.4.0, driver API 13.3.0 as CUDA.jl reports them) |

Each JSON file records its own environment, including the package versions it loaded.
All runs used the `test/cuda` environment in the disposable directory `ewdesktop:~/kronfree-plan2-ci/`; CPU runs never load CUDA or cuTENSOR.

## The tree

The tree is plan 2's part B: `two_zero_pi.json` at `two_zero_pi_params(; g = 0.05)`.
Each leaf is `[theta_i => ChargeBasis(15), phi_i => PhaseGrid(-6π, 6π, 301; order=8)]` (dimension 9331), computing and keeping 300 states with the run's leaf solver.
The parent is 300 × 300 = 90 000 states and keeps the lowest 120.
It forms the `n` ports of `theta1` and `theta2` in its kept basis, so port formation after the parent solve is measured.
The parent is the script's counted `LOBPCG(:ritz)` on the run's device, with the default `seed=:product`: the solver `Auto()` resolves to at a branch, with a callback that records iterations and H columns.
So `BENCH_LEAF_SOLVER=auto` runs exactly the default path at the leaves, and the parent's settings equal what `Auto()` would choose.
Every JSON sample records the solver each node resolved to (`resolved_solver`, `resolved_solvers`): `ShiftInvert` on CPU and `ChebyshevFilter` on the A4500 for both leaves under `auto`, and `LOBPCG` at the root.
The leaf's assembled matrix is exactly real (`leaf_assembled_real: true` in every CPU file), so `ShiftInvert` ran in real arithmetic; `ChebyshevFilter` applies the package operator, whose coefficients are complex (spec §20e).

## How the numbers were obtained

- Each stage is a timed operation with one untimed full-size warm-up and then three samples (`BENCH_WARMUP=1`, `BENCH_REPEATS=3`).
  A compilation warm-up on a small tree with the same solver types runs first in every process and is not timed.
- A sample is wall time after `GC.gc()`, with `CUDA.synchronize()` before and after on the GPU.
  The tables give the median, with the minimum–maximum range in parentheses.
- Every solve is validated after its sample, outside the timing: each true residual ‖H v − E v‖ is recomputed with a host copy of the operator and must be below 1e-8, and ‖V'V − I‖ must be below 1e-10.
- `leaf` materializes each leaf alone on its own terms; `tree` is the full cold `materialize`; `parts` builds each operator on the solver's storage and times the leaf solve alone (`KronFreeCircuits.solve(resolved leaf solver, H, 300)`), the parent solve alone (`lobpcg_solve(root solver, H, 120)`), and port formation alone, on the materialized system of the last `tree` sample.

## Leaf (9331 states, lowest 300)

| Leaf | CPU, 8 BLAS threads | A4500 | Source |
|---|---:|---:|---|
| `ShiftInvert()`, leaf materialize, leaf 1 / leaf 2 | **2.908 s** (2.892–2.955) / 2.899 s (2.874–2.915) | — | `end_to_end-cpu-shiftinvert-leaf+tree+parts.json`, `leaf` |
| `ShiftInvert()`, solve alone, leaf 1 | 2.620 s (2.619–2.635) | — | the same file, `parts` |
| `ShiftInvert(slices=8)`, 8 Julia threads, 1 BLAS thread, leaf materialize, leaf 1 / leaf 2 | 4.354 s (4.322–4.570) / 4.356 s (4.350–4.447) | — | `end_to_end-cpu-shiftinvert-slices8-leaf.json` |
| `ShiftInvert(slices=8)`, 8 Julia threads, 8 BLAS threads (diagnostic), leaf 1 | 4.457 s (4.235–4.465) | — | `end_to_end-cpu-shiftinvert-slices8-blas8-leaf.json` |
| `ChebyshevFilter(on=CuArray)`, leaf materialize, leaf 1 / leaf 2 | — | **2.298 s** (2.282–2.455) / 2.251 s (2.244–2.261), 4 iterations, degree 100 | `end_to_end-gpu-chebyshev-leaf+tree+parts.json`, `leaf` |
| `ChebyshevFilter(on=CuArray)`, solve alone, leaf 1 | — | 2.187 s (2.187–2.198), 4 iterations | the same file, `parts` |
| plan 2: `Lanczos()` / `Dense()` leaf materialize (for comparison) | 84.2 s / 69.0 s | — | `../2026-10-17-plan2-benchmark/results.md` |

Leaf materialize includes operator construction, the solve, phase fixing and the edge check; for `ShiftInvert` it also includes `sparse(H)` assembly and the factorization, about 0.3 s over the solve alone.
`ChebyshevFilter` does not count H columns (the field is `-1`); its iteration count and last filter degree come from its callback.

## Full tree, cold

| Leaf solver (parent: seeded `LOBPCG(:ritz)`) | CPU, 8 BLAS threads | A4500 | Source |
|---|---:|---:|---|
| `ShiftInvert()` (CPU) / `ChebyshevFilter(on=CuArray)` (A4500), explicit | **15.376 s** (15.221–15.390) | **12.554 s** (12.519–12.583) | `end_to_end-cpu-shiftinvert-leaf+tree+parts.json`, `end_to_end-gpu-chebyshev-leaf+tree+parts.json`, `tree` |
| `Auto(on=…)`, the default path | **15.746 s** (15.709–15.822) | **12.394 s** (12.377–12.469) | `end_to_end-cpu-auto-tree.json`, `end_to_end-gpu-auto-tree.json` |
| plan 2: `Lanczos()` leaves, cold-random `LOBPCG(:ritz)` parent | 237.4 s (one sample) | 208.5 s (unvalidated) | `../2026-10-17-plan2-benchmark/results.md` |

The explicit and `Auto` rows run the same solvers with the same settings; they differ by 2 % on CPU and 1 % on the A4500, between processes.
The parent took 671 H columns and 16 iterations in every full-size sample on both devices.

## Stage breakdown

| Stage | CPU, 8 BLAS threads | A4500 (host work on one BLAS thread) | How obtained |
|---|---:|---:|---|
| Leaf materialize, each leaf | 2.908 s, 2.899 s | 2.298 s, 2.251 s | `leaf` stage |
| Parent solve alone, cold, 120 states | **8.882 s** (8.410–9.016), 671 H columns, 16 iterations | **4.914 s** (4.901–4.922), 671 H columns, 16 iterations | `parts`: `lobpcg_solve(root solver, operator(sys), 120)` |
| plan 2: parent solve alone, cold-random | 68.3 s, 4794 H columns, 53 iterations | 41.0 s, 4750 H columns, 47 iterations | `../2026-10-17-plan2-benchmark/results.md` |
| Leaf operator construction | < 1 ms | < 1 ms | `operator(sys, leaf1)` on the solver's storage |
| Parent operator construction | < 1 ms | 1 ms | `operator(sys)` |
| Parent operator upload alone | — | 1 ms | `_on_storage(host copy, CuArray; capacity = 124)` |
| Parent eigenvector download (90 000 × 120) | — | 0.052 s | `Array(CuArray(eigenvectors(sys)))` |
| Port formation, each leaf (1 port, 300 states) | 0.021 s, 0.021 s | 0.112 s, 0.111 s | `_port_matrix` on the leaf's kept vectors |
| Port formation, root (2 ports, 120 states) | 0.240 s | 1.219 s | the same, on the root's kept vectors |
| Full tree | 15.376 s | 12.554 s | `tree` stage |
| Remainder (selection, phase fixing, peak amplitudes, edge checks) | ≈ 0.41 s | ≈ 1.65 s | full tree − both leaves − parent solve − all port formation; the medians come from the same process but different stages, so this is approximate |

CPU remainder: 15.376 − 2.908 − 2.899 − 8.882 − (0.021 + 0.021 + 0.240) = 0.405 s.
A4500 remainder: 12.554 − 2.298 − 2.251 − 4.914 − (0.112 + 0.111 + 1.219) = 1.649 s.
Leaf-alone materializations form no ports, so the leaf port formation is subtracted from the tree once.
On the A4500 the parent's host work (port formation and the remainder) runs on one BLAS thread, so it is 4–5× slower than on CPU.

## Validation

Every timed sample in every file validates (`summary: no failed stages or samples` in every log; each file's `ok` fields are all true).

| Solve | Largest true residual | Largest ‖V'V − I‖ |
|---|---:|---:|
| `ShiftInvert()` leaves (leaf, tree and parts samples) | 1.8e-11 | 2.0e-12 |
| `ShiftInvert(slices=8)` leaves | 7.0e-9 | 1.9e-12 |
| `ChebyshevFilter(on=CuArray)` leaves | 9.997e-9 | 1.1e-12 |
| Seeded `LOBPCG(:ritz)` parent, CPU / A4500 | 9.309e-9 / 9.309e-9 | 4.6e-14 / 6.3e-14 |
| Smoke run (`Dense` leaves, as `Auto()` resolves 1313-state leaves) | 1.5e-12 leaves, 1.5e-9 parent | 6.6e-15 |

Energy agreement (maximum absolute difference, GHz):

- leaf energies, `ShiftInvert` against `ChebyshevFilter`: 2.1e-12 (leaf 1), 1.9e-12 (leaf 2);
- leaf energies, `ShiftInvert` against `ShiftInvert(slices=8)`: 4.5e-13 and 3.0e-13;
- leaf energies, `ShiftInvert` against plan 2's `Lanczos()` leaves: 2.8e-12 and 4.7e-12;
- root energies, CPU against A4500: 2.2e-12 (explicit leaf solvers) and 2.1e-12 (`Auto`);
- root energies against plan 2's CPU full tree: 3.4e-12 (CPU) and 2.7e-12 (A4500).

The `ChebyshevFilter` residual, 9.997e-9, sits just below the 1e-8 tolerance: the solver locks a column once its residual is below `tol`, so the host recomputation lands at the threshold.
`ShiftInvert` returns residuals far below it because the Lanczos tolerance on the inverted operator is `inner_tol = 1e-13`.

### Warnings

The run logs (`logs/`) and each JSON file's `warning_counts` record the warnings; a warning with `maxlog` is counted only while the console still shows it.
They are the same as plan 2's at the full size: grid-edge warnings on every leaf (expected on this grid; plan 2 finding (d)), and branch-phase hybridization warnings at the parent for kept states 15, 46, 50, 67 and 83.
There were no guard warnings, no `ShiftInvert` LU-fallback warnings and no `Dense` size warnings.

## Expectations

- **`ShiftInvert` leaf ≈ 2.5–3 s all-in on CPU: met** (2.91 s materialize, 2.62 s solve alone).
- **Sliced `ShiftInvert` ≈ 1.2–1.5 s on 8 threads: missed.**
  `ShiftInvert(slices=8)` on `julia -t 8` takes 4.35 s with one BLAS thread and 4.46 s with eight, 1.5× slower than the single shift.
  The spike's 1.2 s was not reproduced by the package's slicing path; this run did not investigate why (the package's slicing does one `LU` per slice and two `LDLᵀ` factorizations per inertia count, and BLAS/UMFPACK threads are not pinned inside the threaded loop).
- **`ChebyshevFilter` leaf ≈ 2.5 s on the A4500 in complex arithmetic: met** (2.19 s solve alone, 2.30 s materialize).
- **Seeded parent ≈ 6 s on the A4500 against 41 s cold-random: met** (4.91 s; 8.88 s on CPU against 68.3 s).
- **Full tree against plan 2's 237 s: 15.4 s on CPU and 12.6 s on the A4500** (15.7 s and 12.4 s with `Auto()`).

## What to tune

- **`Auto()`'s `dense_cutoff = 2500` is a placeholder and this benchmark does not constrain it**: no node of this tree is near the cutoff (the leaves have 9331 states and the parent 90 000).
  The only bracketing data are at 9331 states, where `Dense()` (69 s, plan 2) is 24× slower than `ShiftInvert()` (2.9 s), so the crossover is well below 9331.
  Tuning it needs a sweep of leaf sizes from about 500 to 9000 states, timing `Dense()` against `ShiftInvert()` (CPU) and `ChebyshevFilter` (GPU) for the same `compute`.
- **`Auto()` should not prefer slicing when threads are available**, on this evidence: the package's 8-slice solve is slower than the single shift with either BLAS setting.
  Whether slicing can reach the spike's 1.2 s (fewer slices, Cholesky instead of `LU` per slice, cheaper inertia counts) is an open item.
- **The seeded parent is now the dominant stage** (8.9 s of 15.4 s on CPU, 4.9 s of 12.6 s on the A4500).
  The real-arithmetic plan (spec §20e decision 1: about 2× everywhere) is the next lever for both the parent and the GPU leaf, whose spike measured 1.2 s in real arithmetic.
- **On the A4500, host-side work after the device solves is a quarter of the tree**: root port formation 1.22 s and about 1.65 s of selection, phase fixing and peak amplitudes, all on one BLAS thread.
  Device-side port formation (plan 2's finding (c)) and moving the post-solve bookkeeping to the device or to more host threads would take most of it.
- **The `ShiftInvert` times above were measured with the then-default `certify=false`.**
  `ShiftInvert()` now certifies by default (after this benchmark, to catch degenerate levels single-vector Lanczos undercounts): one more Lanczos pair (`m + 1`) and one `LDLᵀ` factorization of the 9331-state leaf per solve (two if the first breaks down), plus a deflated Lanczos run and one or two factorizations per recovery round when the count is short.
  Not re-measured (the certificate's `LDLᵀ` is simplicial, unlike the supernodal Cholesky of the shift, so its cost is not bounded by the 0.3 s of assembly and factorization above).
  Likewise the `ChebyshevFilter` rows predate the product seed's `1e-2` random admixture, the guard columns of the stop rule and the rule that every column is filtered once before it locks (on the small test leaves the guards add about one iteration).

## Commands

On the Mac (smoke checks of the script, CPU, not saved here):

```bash
BENCH_SMOKE=1 BENCH_REPEATS=1 BENCH_LEAF_SOLVER=auto|shiftinvert|chebyshev BENCH_STAGES=leaf,tree,parts \
    julia --project=. -t 1 bench/end_to_end.jl
```

Preparing `ewdesktop`:

```bash
rsync -a --delete --exclude .git --exclude Manifest.toml --exclude 'test/cuda/Manifest.toml' --exclude docs/spikes \
    --exclude bench/fixtures --exclude out --exclude .superpowers ./ ewdesktop:~/kronfree-plan2-ci/
ssh ewdesktop 'cd ~/kronfree-plan2-ci && ~/.juliaup/bin/julia --project=test/cuda -e "using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()"'
```

The second sync (before the `task9b` runs) also excluded `logs`.
Every run below was one `ssh ewdesktop` invocation of this form, in the order listed:

```bash
cd ~/kronfree-plan2-ci && env <variables> BENCH_OUT=out/<output> BENCH_COMMIT=<revision> timeout 590 ~/.juliaup/bin/julia -t <threads> --project=test/cuda bench/end_to_end.jl > logs/<log> 2>&1
```

The two `leaf,tree,parts` runs were launched with `nohup` and `timeout 1800` instead, with the JSON file and the log as the completion signal.

| Output (this directory) | Threads | Revision | Variables | Log (`logs/`) |
|---|---|---|---|---|
| `end_to_end-cpu-smoke-auto-leaf+tree.json` | 1 | `task9` | `BENCH_DEVICE=cpu BENCH_SMOKE=1 BENCH_LEAF_SOLVER=auto BENCH_STAGES=leaf,tree` | `cpu-smoke-auto.log` |
| `end_to_end-cpu-shiftinvert-leaf+tree+parts.json` | 1 | `task9` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=shiftinvert BENCH_STAGES=leaf,tree,parts` | `cpu-shiftinvert.log` |
| `end_to_end-cpu-shiftinvert-slices8-leaf.json` | 8 | `task9b` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=shiftinvert BENCH_SLICES=8 BENCH_BLAS_THREADS=1 BENCH_STAGES=leaf` | `cpu-shiftinvert-slices8.log` |
| `end_to_end-gpu-chebyshev-leaf+tree+parts.json` | 1 | `task9b` | `BENCH_DEVICE=gpu BENCH_LEAF_SOLVER=chebyshev BENCH_STAGES=leaf,tree,parts` | `gpu-chebyshev.log` |
| `end_to_end-gpu-auto-tree.json` | 1 | `task9b` | `BENCH_DEVICE=gpu BENCH_LEAF_SOLVER=auto BENCH_STAGES=tree` | `gpu-auto-tree.log` |
| `end_to_end-cpu-auto-tree.json` | 1 | `task9b` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=auto BENCH_STAGES=tree` | `cpu-auto-tree.log` |
| `end_to_end-cpu-shiftinvert-slices8-blas8-leaf.json` (diagnostic) | 8 | `task9b` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=shiftinvert BENCH_SLICES=8 BENCH_BLAS_THREADS=8 BENCH_STAGES=leaf BENCH_LEAVES=1` | `cpu-shiftinvert-slices8-blas8.log` |

Revisions are `0b70c8b+task9` and `0b70c8b+task9b` (see the top of this file).
Each JSON went to `out/` and was copied here with `scp`, with its console log in `logs/`.
`cuda-tests.log` in this directory is the device test suite's log at `1798c29` (the post-merge fixes: perturbed `ChebyshevFilter` product seed, certified `ShiftInvert`, periodic harmonic check; 332 of 332 pass on the A4500, as at `acc0a6c`).
