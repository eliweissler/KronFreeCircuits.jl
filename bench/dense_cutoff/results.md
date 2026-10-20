# `Auto()`'s `dense_cutoff`: Dense against ShiftInvert and LOBPCG

Benchmark of `bench/dense_cutoff/run.jl` (CPU only), run on 2026-10-20 at commit `9fab776` (branch `small-features`, working tree otherwise clean).
Raw data: `raw_4t.json` (default BLAS threads) and `raw_1t.json` (`BLAS.set_num_threads(1)`); the tables below are generated from them.

## Recommendation

**Keep `dense_cutoff = 2500` for now.**
Lower it to about 1000 once `ShiftInvert`'s residual check stops failing spuriously (below).

- Speed alone says lower.
  Every iterative solver beats `Dense` well below 2500:
  the crossover is about 150 to 300 states for the fluxonium and zero-pi leaves and for both branches, and about 700 to 1000 for the 1-mode transmon leaf.
  The most conservative crossover, the transmon leaf at compute 6 or 20, is about 1000, so 1000 is the speed-optimal cutoff for all shapes together.
- Between 1000 and 2500 the saving is small, up to about 0.4 s per node (Dense 0.04 to 0.5 s against 0.02 to 0.08 s), and below 1000 it is under 0.05 s.
- Reliability says keep.
  `ShiftInvert()` (what `Auto()` resolves a CPU leaf to) raises its own true-residual check (`tol = 1e-8`, "tighten inner_tol") on a fraction of ordinary runs.
  Failures are random from run to run and appear from about 1000 to 1500 states up, on the fluxonium leaf at compute 6 and the transmon leaf at compute 20 (up to 5 of 5 samples at 6000 states); none occurred on the zero-pi leaf.
  Dense and the branch solver (LOBPCG) had none.
  The residuals that exceed 1e-8 range from 1e-8 to about 1e-5 (seen in the first benchmark runs); the energies of the successful samples agree with Dense to 2e-9 or better wherever Dense ran.
  Lowering the cutoff to 1000 would route the 1000 to 2500 band, which `Dense` serves reliably today, to `ShiftInvert`, trading at most 0.3 s for a nonzero chance of a raised error.
  This failure is more frequent than STATUS open item 5 ("rare") records and is worth its own investigation (a scaled or relative residual tolerance, or a retry).
- Leaves and branches do not cross over at clearly different dimensions (branches 100 to 300, leaves 150 to 1000), and below 1000 the absolute gain is tens of milliseconds, so separate leaf and branch cutoffs are not worth implementing.
  If a branch-only cutoff were wanted, about 500 would be safe (LOBPCG had no failed sample in any of the 22 branch configurations, in either run).

Caveats.
Crossovers depend on the machine and on the BLAS thread count: `Dense` is dominated by LAPACK and scales with threads (at 4000 states 2.3 s with 4 BLAS threads, 4.1 s with 1), while the sparse solves do not benefit (equal, or faster with one thread, at 8000 states).
The sparse solvers' cost also depends on the operator's sparsity and the spectrum, which the three leaf shapes and two branch shapes sample only coarsely.
On this M2 with 1 thread the transmon crossover moves from about 1000 to about 700; a machine with a slower LAPACK or fewer cores would move every crossover down, and one with a fast many-core BLAS would move them up.
The machine was a normally loaded laptop (desktop apps, an idle editor Julia process), not an idle one.

## Environment

| | |
|---|---|
| machine | Apple M2, `Sys.CPU_THREADS` = 8 |
| Julia | 1.13.2, `Threads.nthreads()` = 1 |
| BLAS | OpenBLAS 0.3.30 (ILP64); `BLAS.get_num_threads()` = 4 (Julia's default here) for `raw_4t.json`, 1 for `raw_1t.json` |
| packages | KrylovKit 0.10.4, LOBPCGEigensolver 0.1.4, ITensors 0.9.32, KernelAbstractions 0.9.45, Adapt 4.7.4, JSON3 1.14.3, RuntimeGeneratedFunctions 0.5.27 |
| git commit | `9fab776ee96de2a8b0b2164d4a9e1b688cb38486` |
| wall time | 243 s (default threads), 264 s (1 thread) |

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
| 100 | 0.001 | 0.001 | 0/5 | 0.002 | 0/5 | 2e-13 |
| 200 | 0.003 | 0.003 | 0/5 | 0.003 | 0/5 | 3e-13 |
| 300 | 0.005 | 0.004 | 0/5 | 0.005 | 0/5 | 9e-13 |
| 500 | 0.011 | 0.009 | 0/5 | 0.011 | 0/5 | 2e-12 |
| 1000 | 0.041 | 0.032 | 1/5 | 0.035 | 0/5 | 8e-12 |
| 1500 | 0.097 | 0.057 | 0/5 | 0.056 | 0/5 | 2e-11 |
| 2000 | 0.223 | 0.068 | 1/5 | 0.071 | 0/5 | 3e-11 |
| 3000 | 0.903 | 0.103 | 1/5 | 0.107 | 0/5 | 4e-11 |
| 4000 | 2.309 | 0.134 | 1/5 | 0.140 | 0/5 | 1e-10 |
| 6000 | - | 0.199 | 1/5 | 0.207 | 0/5 | - |
| 8000 | - | 0.277 | 2/5 | 0.289 | 0/5 | - |

`zeropi` (2-mode ChargeBasis x PhaseGrid (zero-pi))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 0.002 | 0.002 | 0/5 | 0.002 | 0/5 | 8e-14 |
| 201 | 0.004 | 0.003 | 0/5 | 0.004 | 0/5 | 1e-13 |
| 300 | 0.006 | 0.005 | 0/5 | 0.005 | 0/5 | 1e-13 |
| 497 | 0.014 | 0.008 | 0/5 | 0.009 | 0/5 | 2e-13 |
| 999 | 0.049 | 0.017 | 0/5 | 0.020 | 0/5 | 4e-13 |
| 1496 | 0.114 | 0.032 | 0/5 | 0.037 | 0/5 | 7e-13 |
| 2002 | 0.254 | 0.046 | 0/5 | 0.053 | 0/5 | 1e-12 |
| 2992 | 0.998 | 0.086 | 0/5 | 0.096 | 0/5 | 2e-12 |
| 4009 | - | 0.140 | 0/5 | 0.152 | 0/5 | - |
| 6003 | - | 0.253 | 0/5 | 0.281 | 0/5 | - |
| 7992 | - | 0.367 | 0/5 | 0.409 | 0/5 | - |

`transmon` (1-mode ChargeBasis (transmon))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 0.001 | 0.002 | 0/5 | 0.002 | 0/5 | 9e-13 |
| 199 | 0.002 | 0.005 | 0/5 | 0.005 | 0/5 | 3e-12 |
| 299 | 0.004 | 0.009 | 0/5 | 0.008 | 0/5 | 6e-12 |
| 499 | 0.009 | 0.021 | 0/5 | 0.021 | 0/5 | 2e-11 |
| 999 | 0.035 | 0.034 | 0/5 | 0.034 | 0/5 | 9e-11 |
| 1499 | 0.082 | 0.047 | 0/5 | 0.048 | 1/5 | 2e-10 |
| 1999 | 0.182 | 0.060 | 0/5 | 0.061 | 2/5 | 4e-10 |
| 2999 | 0.843 | 0.093 | 0/5 | 0.094 | 3/5 | 7e-10 |
| 3999 | 2.247 | 0.122 | 0/5 | 0.124 | 2/5 | 2e-09 |
| 5999 | - | 0.176 | 0/5 | - | 5/5 | - |
| 7999 | - | 0.251 | 1/5 | 0.264 | 4/5 | - |

**Branches.** Two-leaf trees; `dim` = k² for k kept states per child. Parent time = total minus two leaf solves (a leaf alone, measured separately, times two; shown); for the zero-pi children the leaves (0.1 s) swamp the parent, so only totals are given. `fail` as above (LOBPCG: none occurred).


`two transmons`

| k | dim | leaves | Dense total | Dense parent | LOBPCG c=6 total | parent | LOBPCG c=20 total | parent | max dE (c=20) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.003 | 0.004 | 0.002 | 0.004 | 0.001 | 0.005 | 0.002 | 1e-13 |
| 14 | 196 | 0.003 | 0.006 | 0.003 | 0.004 | 0.001 | 0.006 | 0.003 | 1e-13 |
| 17 | 289 | 0.003 | 0.007 | 0.005 | 0.005 | 0.002 | 0.007 | 0.004 | 8e-14 |
| 23 | 529 | 0.003 | 0.016 | 0.013 | 0.005 | 0.002 | 0.009 | 0.006 | 2e-13 |
| 32 | 1024 | 0.003 | 0.048 | 0.045 | 0.007 | 0.004 | 0.013 | 0.010 | 5e-13 |
| 39 | 1521 | 0.003 | 0.127 | 0.124 | 0.009 | 0.006 | 0.018 | 0.014 | 8e-13 |
| 45 | 2025 | 0.004 | 0.297 | 0.293 | 0.012 | 0.008 | 0.026 | 0.022 | 1e-12 |
| 55 | 3025 | 0.004 | 1.157 | 1.153 | 0.018 | 0.014 | 0.037 | 0.033 | 2e-12 |
| 63 | 3969 | 0.004 | 2.915 | 2.911 | 0.021 | 0.017 | 0.050 | 0.046 | 2e-12 |
| 77 | 5929 | 0.005 | - | - | 0.032 | 0.026 | 0.065 | 0.060 | - |
| 90 | 8100 | 0.005 | - | - | 0.039 | 0.034 | 0.084 | 0.078 | - |

`two zeropis`

| k | dim | leaves | Dense total | LOBPCG c=6 total | LOBPCG c=20 total | max dE (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.122 | 0.133 | 0.134 | 0.136 | 1e-13 |
| 14 | 196 | 0.128 | 0.140 | 0.136 | 0.142 | 1e-13 |
| 17 | 289 | 0.134 | 0.142 | 0.137 | 0.142 | 2e-13 |
| 23 | 529 | 0.130 | 0.157 | 0.147 | 0.162 | 1e-13 |
| 32 | 1024 | 0.140 | 0.212 | 0.130 | 0.147 | 2e-13 |
| 39 | 1521 | 0.129 | 0.296 | 0.137 | 0.150 | 2e-13 |
| 45 | 2025 | 0.123 | 0.451 | 0.132 | 0.149 | 1e-13 |
| 55 | 3025 | 0.134 | 1.342 | 0.139 | 0.134 | 2e-13 |
| 63 | 3969 | 0.104 | 3.000 | 0.150 | 0.177 | 2e-13 |
| 77 | 5929 | 0.118 | - | 0.148 | 0.178 | - |
| 90 | 8100 | 0.117 | - | 0.145 | 0.210 | - |

Crossovers (product dimension where the iterative solver becomes faster than Dense; log-log interpolation between the bracketing sizes; branches by total time, leaf solves included, which dilutes the difference; the two-transmon leaves cost 4 ms, the two-zero-pi leaves about 0.1 s):

| shape | solver, compute | crossover |
|---|---|---|
| leaf fluxonium | ShiftInvert, 6 | about 155 (between 100 and 200) |
| leaf fluxonium | ShiftInvert, 20 | about 324 (between 300 and 500) |
| leaf zeropi | ShiftInvert, 6 | about 156 (between 99 and 201) |
| leaf zeropi | ShiftInvert, 20 | about 218 (between 201 and 300) |
| leaf transmon | ShiftInvert, 6 | about 968 (between 499 and 999) |
| leaf transmon | ShiftInvert, 20 | about 989 (between 499 and 999) |
| branch two transmons | LOBPCG, 6 | < 100 (iterative already ahead by 1.1x) |
| branch two transmons | LOBPCG, 20 | about 196 (between 196 and 289) |
| branch two zeropis | LOBPCG, 6 | about 113 (between 100 and 196) |
| branch two zeropis | LOBPCG, 20 | about 263 (between 196 and 289) |

### BLAS threads = 1

**Leaves.** `materialize` time of one node in seconds (minimum of the successful samples); `fail` is failed samples out of samples taken, for ShiftInvert only. Dense is timed with compute 20 up to dimension 4000.


`fluxonium` (1-mode extended PhaseGrid (fluxonium))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.001 | 0.002 | 0/5 | 0.002 | 0/5 | 1e-13 |
| 200 | 0.003 | 0.003 | 0/5 | 0.003 | 0/5 | 4e-13 |
| 300 | 0.006 | 0.004 | 0/5 | 0.005 | 0/5 | 6e-13 |
| 500 | 0.015 | 0.009 | 0/5 | 0.011 | 0/5 | 2e-12 |
| 1000 | 0.070 | 0.035 | 0/5 | 0.037 | 0/5 | 9e-12 |
| 1500 | 0.205 | 0.061 | 0/5 | 0.062 | 0/5 | 2e-11 |
| 2000 | 0.470 | 0.074 | 2/5 | 0.075 | 0/5 | 3e-11 |
| 3000 | 1.591 | 0.113 | 2/5 | 0.115 | 0/5 | 5e-11 |
| 4000 | 4.119 | 0.146 | 2/5 | 0.148 | 0/5 | 1e-10 |
| 6000 | - | 0.223 | 1/5 | 0.239 | 0/5 | - |
| 8000 | - | 0.305 | 1/5 | 0.306 | 2/5 | - |

`zeropi` (2-mode ChargeBasis x PhaseGrid (zero-pi))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 0.002 | 0.002 | 0/5 | 0.002 | 0/5 | 8e-14 |
| 201 | 0.003 | 0.003 | 0/5 | 0.004 | 0/5 | 2e-13 |
| 300 | 0.006 | 0.005 | 0/5 | 0.005 | 0/5 | 1e-13 |
| 497 | 0.016 | 0.007 | 0/5 | 0.009 | 0/5 | 2e-13 |
| 999 | 0.075 | 0.017 | 0/5 | 0.020 | 0/5 | 3e-13 |
| 1496 | 0.202 | 0.030 | 0/5 | 0.035 | 0/5 | 5e-13 |
| 2002 | 0.442 | 0.043 | 0/5 | 0.050 | 0/5 | 8e-13 |
| 2992 | 1.517 | 0.076 | 0/5 | 0.085 | 0/5 | 2e-12 |
| 4009 | - | 0.124 | 0/5 | 0.137 | 0/5 | - |
| 6003 | - | 0.244 | 0/5 | 0.262 | 0/5 | - |
| 7992 | - | 0.339 | 0/5 | 0.369 | 0/5 | - |

`transmon` (1-mode ChargeBasis (transmon))

| dim | Dense | ShiftInvert c=6 | fail | ShiftInvert c=20 | fail | max dE vs Dense (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 99 | 0.001 | 0.002 | 0/5 | 0.002 | 0/5 | 8e-13 |
| 199 | 0.002 | 0.005 | 0/5 | 0.005 | 0/5 | 3e-12 |
| 299 | 0.005 | 0.008 | 0/5 | 0.009 | 0/5 | 6e-12 |
| 499 | 0.012 | 0.021 | 0/5 | 0.022 | 0/5 | 2e-11 |
| 999 | 0.059 | 0.034 | 0/5 | 0.035 | 0/5 | 9e-11 |
| 1499 | 0.172 | 0.047 | 0/5 | 0.048 | 1/5 | 2e-10 |
| 1999 | 0.387 | 0.059 | 0/5 | 0.061 | 1/5 | 4e-10 |
| 2999 | 1.440 | 0.091 | 0/5 | 0.096 | 3/5 | 7e-10 |
| 3999 | 4.298 | 0.124 | 0/5 | 0.128 | 1/5 | 2e-09 |
| 5999 | - | 0.187 | 0/5 | 0.206 | 4/5 | - |
| 7999 | - | 0.268 | 1/5 | - | 5/5 | - |

**Branches.** Two-leaf trees; `dim` = k² for k kept states per child. Parent time = total minus two leaf solves (a leaf alone, measured separately, times two; shown); for the zero-pi children the leaves (0.1 s) swamp the parent, so only totals are given. `fail` as above (LOBPCG: none occurred).


`two transmons`

| k | dim | leaves | Dense total | Dense parent | LOBPCG c=6 total | parent | LOBPCG c=20 total | parent | max dE (c=20) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.003 | 0.004 | 0.002 | 0.004 | 0.001 | 0.005 | 0.002 | 1e-13 |
| 14 | 196 | 0.003 | 0.005 | 0.003 | 0.004 | 0.002 | 0.005 | 0.003 | 1e-13 |
| 17 | 289 | 0.003 | 0.008 | 0.005 | 0.004 | 0.002 | 0.007 | 0.004 | 2e-13 |
| 23 | 529 | 0.003 | 0.019 | 0.016 | 0.005 | 0.002 | 0.007 | 0.004 | 3e-13 |
| 32 | 1024 | 0.003 | 0.069 | 0.066 | 0.005 | 0.003 | 0.010 | 0.007 | 4e-13 |
| 39 | 1521 | 0.003 | 0.193 | 0.190 | 0.007 | 0.004 | 0.013 | 0.010 | 7e-13 |
| 45 | 2025 | 0.003 | 0.445 | 0.442 | 0.008 | 0.005 | 0.016 | 0.013 | 1e-12 |
| 55 | 3025 | 0.003 | 1.720 | 1.718 | 0.010 | 0.007 | 0.021 | 0.018 | 2e-12 |
| 63 | 3969 | 0.003 | 4.382 | 4.379 | 0.012 | 0.009 | 0.028 | 0.025 | 2e-12 |
| 77 | 5929 | 0.003 | - | - | 0.015 | 0.012 | 0.039 | 0.036 | - |
| 90 | 8100 | 0.003 | - | - | 0.023 | 0.020 | 0.067 | 0.064 | - |

`two zeropis`

| k | dim | leaves | Dense total | LOBPCG c=6 total | LOBPCG c=20 total | max dE (c=20) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.101 | 0.096 | 0.092 | 0.094 | 2e-13 |
| 14 | 196 | 0.091 | 0.094 | 0.094 | 0.095 | 2e-13 |
| 17 | 289 | 0.091 | 0.097 | 0.097 | 0.105 | 1e-13 |
| 23 | 529 | 0.095 | 0.127 | 0.099 | 0.097 | 2e-13 |
| 32 | 1024 | 0.091 | 0.162 | 0.097 | 0.107 | 1e-13 |
| 39 | 1521 | 0.114 | 0.321 | 0.095 | 0.103 | 2e-13 |
| 45 | 2025 | 0.091 | 0.576 | 0.099 | 0.110 | 1e-13 |
| 55 | 3025 | 0.095 | 2.069 | 0.110 | 0.121 | 2e-13 |
| 63 | 3969 | 0.091 | 4.861 | 0.114 | 0.123 | 1e-13 |
| 77 | 5929 | 0.089 | - | 0.102 | 0.134 | - |
| 90 | 8100 | 0.088 | - | 0.104 | 0.150 | - |

Crossovers (product dimension where the iterative solver becomes faster than Dense; log-log interpolation between the bracketing sizes; branches by total time, leaf solves included, which dilutes the difference; the two-transmon leaves cost 4 ms, the two-zero-pi leaves about 0.1 s):

| shape | solver, compute | crossover |
|---|---|---|
| leaf fluxonium | ShiftInvert, 6 | about 144 (between 100 and 200) |
| leaf fluxonium | ShiftInvert, 20 | about 278 (between 200 and 300) |
| leaf zeropi | ShiftInvert, 6 | about 178 (between 99 and 201) |
| leaf zeropi | ShiftInvert, 20 | about 233 (between 201 and 300) |
| leaf transmon | ShiftInvert, 6 | about 709 (between 499 and 999) |
| leaf transmon | ShiftInvert, 20 | about 723 (between 499 and 999) |
| branch two transmons | LOBPCG, 6 | < 100 (iterative already ahead by 1.1x) |
| branch two transmons | LOBPCG, 20 | about 182 (between 100 and 196) |
| branch two zeropis | LOBPCG, 6 | < 100 (iterative already ahead by 1.0x) |
| branch two zeropis | LOBPCG, 20 | < 100 (iterative already ahead by 1.0x) |

## Notes on the tables

- The fluxonium and zero-pi leaves cross at about 150 to 300 states; the difference there is a few milliseconds and within timing noise.
- The transmon leaf is the slowest `ShiftInvert` case at small size (0.021 s at 499 states against 0.009 s for the other leaves) while `Dense` costs the same as on the other leaves, hence the later crossover; the cause was not investigated.
- `ShiftInvert` seconds are the minimum over successful samples; a configuration with failures would cost a retry in real use.
- LOBPCG times are cold starts with no warm `previous`; a sweep reusing the previous point would be faster still.
