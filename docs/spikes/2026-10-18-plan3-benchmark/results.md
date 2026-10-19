# Plan 3 benchmark: the end-to-end two-zero-pi tree in real and complex arithmetic

Date: 2026-10-19 (runs from 23:39 on 2026-10-18 to about 00:15 on 2026-10-19, `ewdesktop` local time).
Machine: `ewdesktop` (i7-11700K, 8 physical cores / 16 logical; NVIDIA RTX A4500).
Package source and benchmark script as measured: commit `e7317ed` (plan 3 through Task 8, plus the `BENCH_ARITHMETIC` knob in `bench/end_to_end.jl`); every JSON file records it as `commit`.
Later commits (the results commit `b581216` and the fix commit after it) do not touch `src/`; the fix commit changes only the unmeasured `f8` stage and the formatting of `bench/end_to_end.jl`, so the timed code paths are those of `e7317ed`.
The comparison column "plan 2b" is `../2026-10-17-plan2b-benchmark/results.md` (package source `0b70c8b`).

## Summary

- **In real arithmetic the full tree takes 9.9 s on CPU and 5.0 s on the A4500, against 15.4 s and 12.6 s in plan 2b** (1.55× and 2.5×).
  These plan 2b to plan 3 deltas also include post-2b solver changes: `ChebyshevFilter` takes 5 iterations instead of 4, and `ShiftInvert` now certifies (about +4 %); see Notes.
  With the default `Auto()` leaves the full tree takes 9.9 s on CPU and 5.1 s on the A4500, against 15.7 s and 12.4 s.
- **Real against complex arithmetic in plan 3, same tree and solvers: 2.2–2.3× on CPU and 2.6× on the A4500 for the full tree.**
  The parent solve gains 2.5× on CPU (8.59 s to 3.47 s) and 3.3× on the A4500 (5.08 s to 1.53 s).
  The `ChebyshevFilter` leaf on the A4500 gains 2.5× (2.50 s to 1.02 s solve alone).
  Port formation gains 3.7–4.0× on both devices.
- **The `ShiftInvert` leaf on CPU gains 2.2× against the complex arithmetic of this plan, but nothing against plan 2b** (2.90 s in both).
  Plan 2b's `ShiftInvert` already ran the leaf in real arithmetic by testing whether the assembled matrix is exactly real (`leaf_assembled_real: true`); plan 3 replaces that test by the structural rule, and `BENCH_ARITHMETIC=complex` is the only way to get complex arithmetic back.
  The plan 3 complex leaf (6.2 s) is therefore slower than plan 2b's.
  So the real-vs-complex column for the full CPU tree (2.3×) overstates the gain against plan 2b (1.55×).
- **A CPU `ChebyshevFilter` leaf gains 1.27× from real arithmetic** (plan 3 complex 43.7 s, plan 3 real 34.3 s); plan 2b had no CPU `ChebyshevFilter` run.
  `ShiftInvert` (2.9 s real) remains 12× faster there.
- **Every timed solve validates** (largest true residual 9.31e-9, largest orthonormality error 2.4e-12 over all runs).
  Energies agree across arithmetics, solvers and devices: the 120 root energies of all eight tree runs, and of plan 2b's CPU tree, agree to 1.9e-12 (1.7e-12 against plan 2b).

## Environment and versions

| | |
|---|---|
| Julia | 1.13.2, run as `julia -t 1` |
| BLAS | OpenBLAS (`LBTConfig([ILP64] libopenblas64_.so)`); 8 threads for CPU runs, 1 for GPU runs (the script's defaults, as in plan 2b) |
| KronFreeCircuits | 0.1.0 (`src/` and `bench/` at `e7317ed`) |
| ITensors / NDTensors | 0.9.32 / 0.4.32 |
| LOBPCGEigensolver | 0.1.4 |
| KernelAbstractions | 0.9.45 |
| KrylovKit | 0.10.4 |
| Adapt / JSON3 | 4.7.4 / 1.14.3 |
| CUDA.jl / cuTENSOR.jl | 6.4.2 / 6.4.2 (CUDA runtime 13.4.0, driver API 13.3.0) |

These are the plan-2b versions.
All runs used the `test/cuda` environment in the disposable directory `ewdesktop:~/kronfree-plan2-ci/`; CPU runs never load CUDA or cuTENSOR.

## The tree and the method

The tree, the leaf and parent settings, the repeat counts (one untimed full-size warm-up, then three samples; `BENCH_WARMUP=1`, `BENCH_REPEATS=3`) and the validation are those of plan 2b: two leaves `[theta_i => ChargeBasis(15), phi_i => PhaseGrid(-6π, 6π, 301; order=8)]` (9331 states) computing and keeping 300 states, a 90 000-state parent keeping 120 with the seeded `LOBPCG(:ritz)`, and the theta ports of both leaves formed in the parent's kept basis.
`BENCH_ARITHMETIC=auto|complex` sets `arithmetic=` on every node of the tree (leaves and parent).
Each JSON records the knob (`arithmetic`) and the arithmetic each materialized node actually ran in (`node_arithmetic`, and per sample in the `leaf` and `tree` stages), read from `arithmetic(sys, node)`.
In every `auto` file all three nodes are `real`; in every `complex` file all three are `complex`.
Every sample records the solver each node resolved to: `ShiftInvert` on CPU and `ChebyshevFilter` on the A4500 at the leaves under `auto`, and `LOBPCG` at the parent.
Times are medians of three samples, with the minimum–maximum range in parentheses in the JSON files.
The "real vs complex" columns are the ratio of the two plan 3 medians.

## Stage tables

### CPU, `ShiftInvert()` leaves, 8 BLAS threads

| Stage | plan 2b | plan 3 complex | plan 3 real | real vs complex |
|---|---:|---:|---:|---:|
| Leaf solve alone (leaf 1) | 2.620 s | 5.899 s | **2.727 s** | 2.16× |
| Leaf materialize, leaf 1 / leaf 2 | 2.908 s / 2.899 s | 6.225 s / 6.195 s | **2.895 s / 2.866 s** | 2.15× / 2.16× |
| Parent solve alone, 120 states (671 H columns, 16 iterations in every run) | 8.882 s | 8.586 s | **3.469 s** | 2.48× |
| Port formation, each leaf (1 port) | 0.021 s | 0.028 s | 0.007 s | 4.0× |
| Port formation, root (2 ports) | 0.240 s | 0.240 s | 0.064 s | 3.75× |
| Full tree, cold | 15.376 s | 22.314 s | **9.890 s** | 2.26× |
| Full tree, `Auto()` leaves | 15.746 s | 21.793 s | **9.924 s** | 2.20× |

### A4500, `ChebyshevFilter(on=CuArray)` leaves, 1 BLAS thread

| Stage | plan 2b | plan 3 complex | plan 3 real | real vs complex |
|---|---:|---:|---:|---:|
| Leaf solve alone (leaf 1) | 2.187 s | 2.504 s | **1.022 s** | 2.45× |
| Leaf materialize, leaf 1 / leaf 2 | 2.298 s / 2.251 s | 2.752 s / 2.544 s | **1.089 s / 1.069 s** | 2.53× / 2.38× |
| Parent solve alone, 120 states | 4.914 s | 5.078 s | **1.531 s** | 3.32× |
| Port formation, each leaf (1 port) | 0.112 s | 0.111 s | 0.028 s | 3.96× |
| Port formation, root (2 ports) | 1.219 s | 1.212 s | 0.332 s | 3.65× |
| Parent eigenvector download | 0.052 s | 0.052 s | 0.026 s | 2.0× |
| Full tree, cold | 12.554 s | 13.176 s | **5.042 s** | 2.61× |
| Full tree, `Auto()` leaves | 12.394 s | 13.304 s | **5.081 s** | 2.62× |

The parent operator upload (about 1 ms) and the operator constructions (under 1 ms) are negligible in all runs.

### CPU, `ChebyshevFilter()` leaf alone (no plan 2b run)

| Stage | plan 3 complex | plan 3 real | real vs complex |
|---|---:|---:|---:|
| Leaf materialize, leaf 1 / leaf 2 | 43.655 s / 43.621 s | 34.326 s / 34.337 s | 1.27× / 1.27× |

Five filter iterations at degree 100 in both arithmetics.

### Where the full tree spends its time (real arithmetic)

| Part | CPU | A4500 |
|---|---:|---:|
| Both leaves | 5.76 s | 2.16 s |
| Parent solve | 3.47 s | 1.53 s |
| All port formation | 0.08 s | 0.39 s |
| Remainder (selection, phase fixing, peak amplitudes, edge checks) | 0.58 s | 0.97 s |
| Full tree | 9.89 s | 5.04 s |

The remainder is the full tree minus the other rows; the medians come from different stages of one process, so it is approximate.
In plan 2b it was 0.41 s on CPU and 1.65 s on the A4500, and in plan 3 complex it is 1.0 s and 1.4 s.
On the A4500 the host-side remainder is now a fifth of the tree.

## Validation

Every timed sample in every file validates (`summary: no failed stages or samples` in every log).

| Run | Largest true residual | Largest ‖V'V − I‖ |
|---|---:|---:|
| CPU `ShiftInvert` leaves + `LOBPCG` parent, complex / real | 9.309e-9 / 9.309e-9 | 1.8e-12 / 1.9e-12 |
| A4500 `ChebyshevFilter` leaves + `LOBPCG` parent, complex / real | 9.309e-9 / 9.309e-9 | 1.9e-12 / 1.3e-12 |
| CPU `ChebyshevFilter` leaf alone, complex / real | 5.0e-9 / 5.0e-9 | 1.5e-12 / 1.6e-12 |

The largest residual in each tree file belongs to the seeded parent solve, as in plan 2b.

Energy agreement across arithmetics (maximum absolute difference of the energies stored in the JSON, GHz):

- leaf energies, `ShiftInvert`, real against complex: 3.7e-13 (leaf 1) and 4.4e-13 (leaf 2);
- leaf energies, `ChebyshevFilter` on the A4500, real against complex: 3.5e-12 (both leaves);
- leaf energies, `ChebyshevFilter` on CPU, real against complex: 3.1e-12;
- root energies of the full tree, real against complex: 6.2e-13 (CPU `ShiftInvert`), 7.0e-13 (A4500), 8.8e-13 (CPU `Auto`), 7.0e-13 (A4500 `Auto`);
- root energies across all eight tree runs (both devices, both arithmetics, explicit and `Auto` leaves): 1.9e-12; against plan 2b's CPU tree: 1.7e-12.

The parent solve alone reproduces the tree's energies exactly in every run (`max_energy_difference_to_tree` is 0), and port formation alone reproduces the stored ports exactly (`max_difference_to_stored` is 0), in both arithmetics.

### Warnings

The warnings are the same in both arithmetics and as in plan 2b: grid-edge warnings on each leaf and 10 branch-phase hybridization warnings at the parent (see each JSON's `warning_counts`).
No run failed or retried.

## Notes

- **`ShiftInvert`'s leaf gain against plan 2b is about 1×, as expected**, because plan 2b already ran it in real arithmetic (2.62 s solve alone then, 2.73 s now).
  The 2.2× in the real-vs-complex column measures the cost of forcing complex arithmetic, which plan 2b could not do.
- **The parent and the GPU leaf are where plan 3 gains over plan 2b**: the parent solve is 2.6× faster on CPU and 3.2× on the A4500, the A4500 leaf 2.1× faster, and the A4500 full tree 2.5× faster.
- **The CPU parent gains 2.5× from real arithmetic with 8 BLAS threads**, a little more than the factor 2 of flops alone; this run did not investigate why.
- **Port formation gains about 4×**, more than the factor 2 of the arithmetic alone; this run did not investigate why.
- **The `ChebyshevFilter` leaf took 5 iterations here against 4 in plan 2b**, in both arithmetics; the solver changed after plan 2b (the product seed's random admixture, the guard columns and the filter-every-column-once rule, listed in plan 2b's "What to tune").
  The A4500 complex leaf (2.50 s solve alone) is therefore 14 % slower than plan 2b's (2.19 s) for that reason, not because of arithmetic.
- **The `ShiftInvert` times include certification** (`certify=true`, the default since plan 2b's last section), which plan 2b's numbers did not; the real leaf solve alone (2.73 s) is 4 % slower than plan 2b's (2.62 s), consistent with plan 2b's measured certificate cost of about 5 %.
- **The CPU `ChebyshevFilter` leaf is not competitive** (34 s real, 44 s complex) against `ShiftInvert` (2.9 s real, 6.2 s complex), so `Auto()` keeps `ShiftInvert` on CPU.
- The compilation warm-up of a process (not a timed stage) took 30–34 s on CPU and 67–69 s on the A4500.

## Commands

On the Mac (smoke checks of the knob, CPU, not saved here):

```bash
BENCH_SMOKE=1 BENCH_ARITHMETIC=complex julia --project=. -t 1 bench/end_to_end.jl
BENCH_SMOKE=1 BENCH_ARITHMETIC=auto    julia --project=. -t 1 bench/end_to_end.jl
```

Preparing `ewdesktop` (from the worktree root):

```bash
rsync -a --delete --exclude .git --exclude Manifest.toml --exclude 'test/cuda/Manifest.toml' --exclude docs/spikes \
    --exclude bench/fixtures --exclude out --exclude .superpowers ./ ewdesktop:~/kronfree-plan2-ci/
```

Every run below was one foreground `ssh ewdesktop` invocation of this form, sequential, one at a time, with `BENCH_ARITHMETIC` set to `complex` and then `auto` for each row (the complex runs first):

```bash
cd ~/kronfree-plan2-ci && env <variables> BENCH_ARITHMETIC=<arith> BENCH_OUT=out/plan3/<output>-<arith>.json BENCH_COMMIT=e7317ed \
    timeout 1800 ~/.juliaup/bin/julia -t 1 --project=test/cuda bench/end_to_end.jl > out/plan3/<output>-<arith>.log 2>&1
```

| Output (this directory; `-complex` / `-auto`) | Variables |
|---|---|
| `cpu-shiftinvert-leaf+tree+parts-<arith>.json` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=shiftinvert BENCH_STAGES=leaf,tree,parts` |
| `gpu-chebyshev-leaf+tree+parts-<arith>.json` | `BENCH_DEVICE=gpu BENCH_LEAF_SOLVER=chebyshev BENCH_STAGES=leaf,tree,parts` |
| `cpu-auto-tree-<arith>.json` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=auto BENCH_STAGES=tree` |
| `gpu-auto-tree-<arith>.json` | `BENCH_DEVICE=gpu BENCH_LEAF_SOLVER=auto BENCH_STAGES=tree` |
| `cpu-chebyshev-leaf-<arith>.json` | `BENCH_DEVICE=cpu BENCH_LEAF_SOLVER=chebyshev BENCH_STAGES=leaf` |

The console logs are in `logs/` under the same names.
`cuda-tests.log` in this directory is the device test suite's log from Task 8.
