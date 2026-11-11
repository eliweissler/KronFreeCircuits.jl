# `Auto()`'s `dense_cutoff`: Dense against ShiftInvert and LOBPCG

Benchmark of `bench/dense_cutoff/run.jl` (CPU only), rerun on 2026-10-20 at commit `9d90f5a` (branch `small-features` plus the `ShiftInvert` polish step, working tree otherwise clean).
Raw data: `raw_4t.json` (default BLAS threads) and `raw_1t.json` (`BLAS.set_num_threads(1)`); the tables below are generated from them.

## Recommendation

**`dense_cutoff = 1000`** (lowered from 2500).

- Speed says about 1000.
  Every iterative solver beats `Dense` well below 2500:
  the crossover is about 120 to 350 states for the fluxonium and zero-pi leaves, below 100 to about 240 for both branches, and about 700 to 960 for the 1-mode transmon leaf.
  The most conservative crossover, the transmon leaf at compute 6 or 20, is just under 1000, so 1000 is the speed-optimal cutoff for all shapes together.
- Between 1000 and 2000 states (the largest sampled size below 2500) Dense takes 0.04 to 0.9 s per node against 0.02 to 0.09 s for the iterative solvers; below 1000 the difference is at most about 0.1 s.
- Reliability no longer says keep 2500.
  The first run of this benchmark (commit `9fab776`) found `ShiftInvert()` raising its own true-residual check (`tol = 1e-8`, "tighten inner_tol") on a fraction of ordinary runs from about 1000 states up: on the fluxonium leaf at compute 6 and the transmon leaf at compute 20, up to 5 of 5 samples at 6000 states, with residuals from 1e-8 to about 1e-5 while the energies agreed with Dense to 2e-9.
  The cause: Lanczos stops the last wanted pairs of `(H − σI)⁻¹` with a Krylov residual `r` just under `inner_tol`, and `H v − E v = −(E − σ)(H − σ) r` magnifies the components of `r` on high-lying states by up to `|E − σ|·‖H‖`, where `‖H‖` grows like `1/h²` on a phase grid and like `ncut²` in a charge basis (2.9e6 on the 8000-point fluxonium grid, 6.4e7 on the 8001-state transmon).
  The vectors were accurate (within 4e-11 of Dense's); the residual measured stiff remnants of size about 1e-12.
  `ShiftInvert` now polishes its pairs by one step of block inverse iteration with its factorization and a Rayleigh–Ritz step, and accepts a residual up to the roundoff floor `10 eps·‖H‖` when that exceeds `tol`.
  In this rerun no sample failed, in either thread setting, on any shape or size, and the `ShiftInvert` times are unchanged within noise (the polish costs 1 to 8 ms).
  The seed scan below confirms it on more samples than the benchmark's five.
- Leaves and branches do not cross over at clearly different dimensions (branches below 100 to about 240, leaves 120 to 960), and below 1000 the absolute gain is tens of milliseconds, so separate leaf and branch cutoffs are not worth implementing.

Caveats.
Crossovers depend on the machine and on the BLAS thread count: `Dense` is dominated by LAPACK and scales with threads (at 4000 states 2.3 to 2.9 s with 4 BLAS threads, 4.7 to 6.3 s with 1), while the sparse solves do not benefit.
The sparse solvers' cost also depends on the operator's sparsity and the spectrum, which the three leaf shapes and two branch shapes sample only coarsely.
A machine with a slower LAPACK or fewer cores would move every crossover down, and one with a fast many-core BLAS would move them up.
The machine was a normally loaded laptop (desktop apps, other Julia sessions), not an idle one; the 1-thread run was the more loaded of the two (Dense on the 3999-state transmon 6.3 s against 4.3 s in the first run).

## Seed scan of the residual check

`ShiftInvert()` through `materialize`, 30 seeds (`Random.seed!(1:30)`) per configuration, on the benchmark's leaf shapes and parameters; `raised` counts solves that raised the true-residual error.
Before: commit `988b71e` (no polish, absolute `tol`); after: commit `9d90f5a`.
`worst residual` is the largest true residual over the returned pairs of the successful solves after the fix; `‖H‖∞` is the ∞-norm of the assembled matrix.

| shape | dim | compute | ‖H‖∞ | raised before | raised after | worst residual after |
|---|---:|---:|---:|---:|---:|---:|
| fluxonium | 1000 | 6 | 4.6e+04 | 1/30 | 0/30 | 3.0e-11 |
| fluxonium | 1000 | 20 | 4.6e+04 | 0/30 | 0/30 | 3.3e-11 |
| fluxonium | 1500 | 6 | 1.0e+05 | 8/30 | 0/30 | 7.4e-11 |
| fluxonium | 1500 | 20 | 1.0e+05 | 0/30 | 0/30 | 7.9e-11 |
| fluxonium | 2500 | 6 | 2.9e+05 | 15/30 | 0/30 | 1.7e-10 |
| fluxonium | 2500 | 20 | 2.9e+05 | 0/30 | 0/30 | 1.7e-10 |
| fluxonium | 4000 | 6 | 7.3e+05 | 11/30 | 0/30 | 3.3e-10 |
| fluxonium | 4000 | 20 | 7.3e+05 | 0/30 | 0/30 | 4.9e-10 |
| fluxonium | 8000 | 6 | 2.9e+06 | 19/30 | 0/30 | 1.0e-09 |
| fluxonium | 8000 | 20 | 2.9e+06 | 1/30 | 0/30 | 1.8e-09 |
| zeropi | 999 | 6 | 2.3e+03 | 0/30 | 0/30 | 1.2e-12 |
| zeropi | 999 | 20 | 2.3e+03 | 0/30 | 0/30 | 1.9e-11 |
| zeropi | 1496 | 6 | 3.4e+03 | 0/30 | 0/30 | 1.6e-12 |
| zeropi | 1496 | 20 | 3.4e+03 | 0/30 | 0/30 | 1.7e-11 |
| zeropi | 2505 | 6 | 5.1e+03 | 0/30 | 0/30 | 2.3e-12 |
| zeropi | 2505 | 20 | 5.1e+03 | 0/30 | 0/30 | 7.1e-12 |
| zeropi | 4009 | 6 | 8.2e+03 | 0/30 | 0/30 | 2.5e-12 |
| zeropi | 4009 | 20 | 8.2e+03 | 0/30 | 0/30 | 3.0e-12 |
| zeropi | 7992 | 6 | 1.6e+04 | 0/30 | 0/30 | 3.9e-12 |
| zeropi | 7992 | 20 | 1.6e+04 | 0/30 | 0/30 | 1.3e-11 |
| transmon | 999 | 6 | 1.0e+06 | 0/30 | 0/30 | 1.7e-10 |
| transmon | 999 | 20 | 1.0e+06 | 1/30 | 0/30 | 1.5e-10 |
| transmon | 1499 | 6 | 2.2e+06 | 0/30 | 0/30 | 3.4e-10 |
| transmon | 1499 | 20 | 2.2e+06 | 2/30 | 0/30 | 3.3e-10 |
| transmon | 2499 | 6 | 6.2e+06 | 0/30 | 0/30 | 8.7e-10 |
| transmon | 2499 | 20 | 6.2e+06 | 7/30 | 0/30 | 1.0e-09 |
| transmon | 3999 | 6 | 1.6e+07 | 1/30 | 0/30 | 2.6e-09 |
| transmon | 3999 | 20 | 1.6e+07 | 18/30 | 0/30 | 2.3e-09 |
| transmon | 7999 | 6 | 6.4e+07 | 3/30 | 0/30 | 1.1e-08 |
| transmon | 7999 | 20 | 6.4e+07 | 25/30 | 0/30 | 9.9e-09 |

Before the fix 112 of 900 solves raised, all with the true-residual error; after it, none.
The failures tracked `‖H‖`: none on the zero-pi leaf (`‖H‖∞` at most 1.6e4), up to 19 of 30 (fluxonium, compute 6) and 25 of 30 (transmon, compute 20) where `‖H‖∞` reaches 3e6 and 6e7.
After the fix the worst residual is 0.7 to 3 eps·‖H‖∞ on the fluxonium and transmon leaves, the roundoff scale (`Dense`'s own residuals on these leaves measured 0.4 to 5.4 eps·‖H‖).
On the 7999-state transmon it reaches 1.1e-8, above `tol = 1e-8` and below the roundoff floor 10 eps·‖H‖∞ = 1.4e-7, which accepts it.

## Environment

| | |
|---|---|
| machine | Apple M2, `Sys.CPU_THREADS` = 8 |
| Julia | 1.13.2, `Threads.nthreads()` = 1 |
| BLAS | OpenBLAS 0.3.30 (ILP64); `BLAS.get_num_threads()` = 4 (Julia's default here) for `raw_4t.json`, 1 for `raw_1t.json` |
| packages | KrylovKit 0.10.4, LOBPCGEigensolver 0.1.4, ITensors 0.9.32, KernelAbstractions 0.10.1, Adapt 4.7.4, JSON3 1.14.3, RuntimeGeneratedFunctions 0.5.27 |
| git commit | `9d90f5ace58fdeb17659fc56f22fb294db1f3a97` |
| wall time | 341 s (default threads), 469 s (1 thread) |

## Method

- `materialize` of the node alone, with the spec from `test/fixtures` (`fluxonium.json`, `zero_pi.json`, `transmon.json`, `two_transmons.json`, `two_zero_pi.json`) and the parameters in `run.jl`; the minimum of 5 samples (fewer when a configuration takes more than a second), after a warm-up call at a small size for every solver.
  A failed sample (a solver error) is counted and excluded from the minimum.
- Leaves: `PhaseGrid(-6π, 6π, N; order=8)` (fluxonium); `ChargeBasis` × `PhaseGrid`, sizes grown together (zero-pi, for example 3 × 33 at 100 states, 17 × 133 at 2000, 31 × 258 at 8000); `ChargeBasis(n)` with 2n+1 states (transmon).
  `Dense()` is timed with `compute = 20` (it does not depend on compute) up to 4000 states; `ShiftInvert()` at `compute` = 6 and 20 (`keep = lowest(compute)`).
- Branches: a parent of two leaves, each materialized by `Dense()` and keeping k states, so the parent has k² states.
  Two transmons (leaves `ChargeBasis(60)`, 121 states) and two zero-pis (leaves `ChargeBasis(4)` × `PhaseGrid(-6π, 6π, 61; order=6)`, 549 states).
  `Dense()` against `LOBPCG(preconditioner=:ritz, seed=:product)` at the parent, cold start, compute 6 and 20.
  The table gives total times; "parent" subtracts twice one leaf solve, measured separately (for the zero-pi children the leaves are 0.1 s and swamp the parent, so only totals are given and crossovers use totals, which understates the iterative gain).
- Every iterative result is compared with Dense's energies where Dense ran (`max dE`, over the computed levels).
- Pruned: `Dense()` is not timed above 4000 states (15 to 20 times slower than the iterative solver there, growing as n³; the 8000-state point would add several minutes);
  the sweep starts at 100 states rather than 500 so that the crossovers of the fluxonium, zero-pi and branch shapes (all below 500) are bracketed.
  The leaf sweep has 11 sizes, the branch sweep 11 kept counts k = 10 to 90.

## Tables

### BLAS threads = 4

**Leaves.** `materialize` time of one node in seconds (minimum of the successful samples); `fail` is failed samples out of samples taken, for ShiftInvert only. Dense is timed with compute 20 up to dimension 4000.

`fluxonium` (1-mode extended PhaseGrid (fluxonium))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.002 | 0.002 | 0/5 | 0.002 | 0/5 | 1e-13 |
| 200 | 0.003 | 0.003 | 0/5 | 0.004 | 0/5 | 3e-13 |
| 300 | 0.005 | 0.005 | 0/5 | 0.005 | 0/5 | 1e-12 |
| 500 | 0.012 | 0.010 | 0/5 | 0.011 | 0/5 | 2e-12 |
| 1000 | 0.043 | 0.033 | 0/5 | 0.035 | 0/5 | 6e-12 |
| 1500 | 0.101 | 0.055 | 0/5 | 0.058 | 0/5 | 2e-11 |
| 2000 | 0.211 | 0.069 | 0/5 | 0.078 | 0/5 | 2e-11 |
| 3000 | 0.910 | 0.104 | 0/5 | 0.113 | 0/5 | 3e-11 |
| 4000 | 2.893 | 0.135 | 0/5 | 0.146 | 0/5 | 9e-11 |
| 6000 | - | 0.220 | 0/5 | 0.240 | 0/5 | - |
| 8000 | - | 0.284 | 0/5 | 0.308 | 0/5 | - |

`zeropi` (2-mode ChargeBasis x PhaseGrid (zero-pi))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 0.002 | 0.002 | 0/5 | 0.003 | 0/5 | 7e-14 |
| 201 | 0.004 | 0.004 | 0/5 | 0.004 | 0/5 | 1e-13 |
| 300 | 0.007 | 0.005 | 0/5 | 0.006 | 0/5 | 1e-13 |
| 497 | 0.015 | 0.008 | 0/5 | 0.010 | 0/5 | 1e-13 |
| 999 | 0.054 | 0.018 | 0/5 | 0.021 | 0/5 | 4e-13 |
| 1496 | 0.120 | 0.033 | 0/5 | 0.039 | 0/5 | 7e-13 |
| 2002 | 0.277 | 0.047 | 0/5 | 0.059 | 0/5 | 1e-12 |
| 2992 | 0.998 | 0.084 | 0/5 | 0.100 | 0/5 | 2e-12 |
| 4009 | - | 0.138 | 0/5 | 0.162 | 0/5 | - |
| 6003 | - | 0.275 | 0/5 | 0.297 | 0/5 | - |
| 7992 | - | 0.372 | 0/5 | 0.422 | 0/5 | - |

`transmon` (1-mode ChargeBasis (transmon))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 0.001 | 0.002 | 0/5 | 0.003 | 0/5 | 1e-12 |
| 199 | 0.003 | 0.005 | 0/5 | 0.005 | 0/5 | 3e-12 |
| 299 | 0.005 | 0.009 | 0/5 | 0.009 | 0/5 | 6e-12 |
| 499 | 0.009 | 0.021 | 0/5 | 0.023 | 0/5 | 2e-11 |
| 999 | 0.037 | 0.034 | 0/5 | 0.036 | 0/5 | 9e-11 |
| 1499 | 0.088 | 0.048 | 0/5 | 0.051 | 0/5 | 2e-10 |
| 1999 | 0.193 | 0.060 | 0/5 | 0.064 | 0/5 | 4e-10 |
| 2999 | 0.849 | 0.093 | 0/5 | 0.095 | 0/5 | 7e-10 |
| 3999 | 2.301 | 0.120 | 0/5 | 0.128 | 0/5 | 2e-09 |
| 5999 | - | 0.183 | 0/5 | 0.196 | 0/5 | - |
| 7999 | - | 0.257 | 0/5 | 0.279 | 0/5 | - |

**Branches.** Two-leaf trees; `dim` = k² for k kept states per child. Parent time = total minus two leaf solves (a leaf alone, measured separately, times two; shown); for the zero-pi children the leaves (0.1 s) swamp the parent, so only totals are given. `fail` as above (LOBPCG: none occurred).

`two transmons`

| k | dim | leaves | Dense total | Dense parent | LOBPCG c=6 total | parent | LOBPCG c=20 total | parent | max dE (c=20) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.003 | 0.005 | 0.002 | 0.005 | 0.001 | 0.006 | 0.002 | 1e-13 |
| 14 | 196 | 0.003 | 0.006 | 0.003 | 0.005 | 0.001 | 0.006 | 0.003 | 1e-13 |
| 17 | 289 | 0.003 | 0.008 | 0.005 | 0.005 | 0.002 | 0.007 | 0.004 | 8e-14 |
| 23 | 529 | 0.003 | 0.015 | 0.012 | 0.005 | 0.002 | 0.008 | 0.005 | 2e-13 |
| 32 | 1024 | 0.003 | 0.043 | 0.040 | 0.006 | 0.003 | 0.011 | 0.008 | 5e-13 |
| 39 | 1521 | 0.003 | 0.107 | 0.103 | 0.008 | 0.005 | 0.016 | 0.012 | 8e-13 |
| 45 | 2025 | 0.003 | 0.227 | 0.223 | 0.010 | 0.006 | 0.020 | 0.017 | 1e-12 |
| 55 | 3025 | 0.004 | 0.909 | 0.906 | 0.011 | 0.008 | 0.026 | 0.022 | 2e-12 |
| 63 | 3969 | 0.004 | 2.321 | 2.317 | 0.013 | 0.010 | 0.031 | 0.027 | 2e-12 |
| 77 | 5929 | 0.004 | - | - | 0.019 | 0.016 | 0.040 | 0.036 | - |
| 90 | 8100 | 0.004 | - | - | 0.023 | 0.020 | 0.051 | 0.047 | - |

`two zeropis`

| k | dim | leaves | Dense total | LOBPCG c=6 total | LOBPCG c=20 total | max dE (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.084 | 0.086 | 0.087 | 0.090 | 1e-13 |
| 14 | 196 | 0.084 | 0.089 | 0.087 | 0.090 | 1e-13 |
| 17 | 289 | 0.084 | 0.090 | 0.087 | 0.090 | 2e-13 |
| 23 | 529 | 0.084 | 0.099 | 0.092 | 0.101 | 1e-13 |
| 32 | 1024 | 0.099 | 0.151 | 0.120 | 0.151 | 2e-13 |
| 39 | 1521 | 0.148 | 0.318 | 0.215 | 0.481 | 2e-13 |
| 45 | 2025 | 0.231 | 4.078 | 0.430 | 0.538 | 1e-13 |
| 55 | 3025 | 0.194 | 6.621 | 0.164 | 0.183 | 2e-13 |
| 63 | 3969 | 0.142 | 4.025 | 0.155 | 0.178 | 2e-13 |
| 77 | 5929 | 0.120 | - | 0.141 | 0.166 | - |
| 90 | 8100 | 0.114 | - | 0.140 | 0.181 | - |

Crossovers (product dimension where the iterative solver becomes faster than Dense; log-log interpolation between the bracketing sizes; branches by total time, leaf solves included, which dilutes the difference; the two-transmon leaves cost 4 ms, the two-zero-pi leaves about 0.1 s):

| shape | solver, compute | crossover |
|---|---|---|
| leaf fluxonium | ShiftInvert, 6 | about 168 (between 100 and 200) |
| leaf fluxonium | ShiftInvert, 20 | about 350 (between 300 and 500) |
| leaf zeropi | ShiftInvert, 6 | about 116 (between 99 and 201) |
| leaf zeropi | ShiftInvert, 20 | about 237 (between 201 and 300) |
| leaf transmon | ShiftInvert, 6 | about 934 (between 499 and 999) |
| leaf transmon | ShiftInvert, 20 | about 962 (between 499 and 999) |
| branch two transmons | LOBPCG, 6 | < 100 (iterative already ahead by 1.0x) |
| branch two transmons | LOBPCG, 20 | about 187 (between 100 and 196) |
| branch two zeropis | LOBPCG, 6 | about 117 (between 100 and 196) |
| branch two zeropis | LOBPCG, 20 | about 239 (between 196 and 289) |

### BLAS threads = 1

**Leaves.** `materialize` time of one node in seconds (minimum of the successful samples); `fail` is failed samples out of samples taken, for ShiftInvert only. Dense is timed with compute 20 up to dimension 4000.

`fluxonium` (1-mode extended PhaseGrid (fluxonium))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.001 | 0.002 | 0/5 | 0.002 | 0/5 | 2e-13 |
| 200 | 0.003 | 0.003 | 0/5 | 0.004 | 0/5 | 3e-13 |
| 300 | 0.005 | 0.005 | 0/5 | 0.005 | 0/5 | 8e-13 |
| 500 | 0.014 | 0.009 | 0/5 | 0.011 | 0/5 | 3e-12 |
| 1000 | 0.066 | 0.033 | 0/5 | 0.034 | 0/5 | 9e-12 |
| 1500 | 0.189 | 0.058 | 0/5 | 0.060 | 0/5 | 2e-11 |
| 2000 | 0.479 | 0.073 | 0/5 | 0.080 | 0/5 | 3e-11 |
| 3000 | 1.635 | 0.130 | 0/5 | 0.120 | 0/5 | 4e-11 |
| 4000 | 4.745 | 0.190 | 0/5 | 0.224 | 0/5 | 1e-10 |
| 6000 | - | 0.299 | 0/5 | 0.311 | 0/5 | - |
| 8000 | - | 0.407 | 0/5 | 0.429 | 0/5 | - |

`zeropi` (2-mode ChargeBasis x PhaseGrid (zero-pi))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 0.002 | 0.003 | 0/5 | 0.004 | 0/5 | 8e-14 |
| 201 | 0.007 | 0.007 | 0/5 | 0.008 | 0/5 | 1e-13 |
| 300 | 0.010 | 0.007 | 0/5 | 0.008 | 0/5 | 1e-13 |
| 497 | 0.022 | 0.011 | 0/5 | 0.017 | 0/5 | 2e-13 |
| 999 | 0.128 | 0.026 | 0/5 | 0.030 | 0/5 | 4e-13 |
| 1496 | 0.298 | 0.062 | 0/5 | 0.064 | 0/5 | 5e-13 |
| 2002 | 0.723 | 0.065 | 0/5 | 0.073 | 0/5 | 9e-13 |
| 2992 | 3.276 | 0.118 | 0/5 | 0.139 | 0/5 | 2e-12 |
| 4009 | - | 0.196 | 0/5 | 0.246 | 0/5 | - |
| 6003 | - | 0.385 | 0/5 | 0.406 | 0/5 | - |
| 7992 | - | 0.567 | 0/5 | 0.621 | 0/5 | - |

`transmon` (1-mode ChargeBasis (transmon))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 0.002 | 0.003 | 0/5 | 0.004 | 0/5 | 7e-13 |
| 199 | 0.005 | 0.009 | 0/5 | 0.009 | 0/5 | 3e-12 |
| 299 | 0.013 | 0.018 | 0/5 | 0.014 | 0/5 | 6e-12 |
| 499 | 0.018 | 0.031 | 0/5 | 0.030 | 0/5 | 2e-11 |
| 999 | 0.089 | 0.049 | 0/5 | 0.068 | 0/5 | 9e-11 |
| 1499 | 0.275 | 0.068 | 0/5 | 0.071 | 0/5 | 2e-10 |
| 1999 | 0.616 | 0.094 | 0/5 | 0.092 | 0/5 | 4e-10 |
| 2999 | 3.253 | 0.198 | 0/5 | 0.164 | 0/5 | 7e-10 |
| 3999 | 6.288 | 0.199 | 0/5 | 0.190 | 0/5 | 2e-09 |
| 5999 | - | 0.259 | 0/5 | 0.275 | 0/5 | - |
| 7999 | - | 0.356 | 0/5 | 0.428 | 0/5 | - |

**Branches.** Two-leaf trees; `dim` = k² for k kept states per child. Parent time = total minus two leaf solves (a leaf alone, measured separately, times two; shown); for the zero-pi children the leaves (0.1 s) swamp the parent, so only totals are given. `fail` as above (LOBPCG: none occurred).

`two transmons`

| k | dim | leaves | Dense total | Dense parent | LOBPCG c=6 total | parent | LOBPCG c=20 total | parent | max dE (c=20) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.004 | 0.006 | 0.002 | 0.005 | 0.001 | 0.007 | 0.003 | 1e-13 |
| 14 | 196 | 0.004 | 0.010 | 0.006 | 0.006 | 0.002 | 0.008 | 0.004 | 1e-13 |
| 17 | 289 | 0.004 | 0.014 | 0.010 | 0.009 | 0.005 | 0.010 | 0.006 | 2e-13 |
| 23 | 529 | 0.005 | 0.030 | 0.024 | 0.007 | 0.002 | 0.012 | 0.006 | 3e-13 |
| 32 | 1024 | 0.004 | 0.102 | 0.098 | 0.008 | 0.004 | 0.025 | 0.021 | 4e-13 |
| 39 | 1521 | 0.005 | 0.294 | 0.289 | 0.009 | 0.004 | 0.029 | 0.024 | 7e-13 |
| 45 | 2025 | 0.005 | 0.911 | 0.907 | 0.012 | 0.007 | 0.024 | 0.020 | 1e-12 |
| 55 | 3025 | 0.004 | 2.743 | 2.739 | 0.019 | 0.015 | 0.033 | 0.029 | 2e-12 |
| 63 | 3969 | 0.004 | 7.209 | 7.205 | 0.023 | 0.019 | 0.055 | 0.051 | 2e-12 |
| 77 | 5929 | 0.005 | - | - | 0.023 | 0.018 | 0.060 | 0.055 | - |
| 90 | 8100 | 0.005 | - | - | 0.041 | 0.036 | 0.106 | 0.101 | - |

`two zeropis`

| k | dim | leaves | Dense total | LOBPCG c=6 total | LOBPCG c=20 total | max dE (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.138 | 0.238 | 0.174 | 0.169 | 2e-13 |
| 14 | 196 | 0.135 | 0.147 | 0.187 | 0.158 | 2e-13 |
| 17 | 289 | 0.135 | 0.144 | 0.147 | 0.142 | 1e-13 |
| 23 | 529 | 0.131 | 0.159 | 0.163 | 0.146 | 2e-13 |
| 32 | 1024 | 0.139 | 0.245 | 0.145 | 0.150 | 1e-13 |
| 39 | 1521 | 0.131 | 0.419 | 0.135 | 0.141 | 2e-13 |
| 45 | 2025 | 0.125 | 0.802 | 0.133 | 0.146 | 1e-13 |
| 55 | 3025 | 0.129 | 2.480 | 0.151 | 0.179 | 2e-13 |
| 63 | 3969 | 0.139 | 7.184 | 0.192 | 0.149 | 1e-13 |
| 77 | 5929 | 0.105 | - | 0.115 | 0.147 | - |
| 90 | 8100 | 0.103 | - | 0.125 | 0.170 | - |

Crossovers (product dimension where the iterative solver becomes faster than Dense; log-log interpolation between the bracketing sizes; branches by total time, leaf solves included, which dilutes the difference; the two-transmon leaves cost 4 ms, the two-zero-pi leaves about 0.1 s):

| shape | solver, compute | crossover |
|---|---|---|
| leaf fluxonium | ShiftInvert, 6 | about 165 (between 100 and 200) |
| leaf fluxonium | ShiftInvert, 20 | about 307 (between 300 and 500) |
| leaf zeropi | ShiftInvert, 6 | about 184 (between 99 and 201) |
| leaf zeropi | ShiftInvert, 20 | about 206 (between 201 and 300) |
| leaf transmon | ShiftInvert, 6 | about 695 (between 499 and 999) |
| leaf transmon | ShiftInvert, 20 | about 791 (between 499 and 999) |
| branch two transmons | LOBPCG, 6 | < 100 (iterative already ahead by 1.1x) |
| branch two transmons | LOBPCG, 20 | about 129 (between 100 and 196) |
| branch two zeropis | LOBPCG, 6 | < 100 (iterative already ahead by 1.4x) |
| branch two zeropis | LOBPCG, 20 | < 100 (iterative already ahead by 1.4x) |

## Notes on the tables

- The fluxonium and zero-pi leaves cross at about 120 to 350 states; the difference there is a few milliseconds and within timing noise.
- The transmon leaf is the slowest `ShiftInvert` case at small size (0.021 s at 499 states against 0.009 s for the other leaves) while `Dense` costs the same as on the other leaves, hence the later crossover; the cause was not investigated.
- `ShiftInvert` seconds are the minimum over successful samples (none failed in this rerun).
- LOBPCG times are cold starts with no warm `previous`; a sweep reusing the previous point would be faster still.
