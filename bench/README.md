# Benchmark suite

Timing and accuracy benchmarks of KronFreeCircuits.jl against published or same-machine references, for the thesis
and paper. The earlier exploratory benches are in `../bench_archive/`.

Cases (one directory each):

| case | reference times | status |
|---|---|---|
| `hensel_two_fluxonia/` | Hensel et al., arXiv:2611.05627 Sec. IV (published), and our own flat solve of their space | first case, harness testbed |
| KITE | scqubits `Circuit`, same machine | planned |
| coupled soft 0-π | Eli's Python pipeline (arXiv:2609.16630), same machine | planned |
| Hensel three fluxonia | arXiv:2611.05627 Sec. V DMRG (published) | planned |
| `autodiff_c66/` (J, L, CL) | SQcircuit dev-ew (PyTorch autodiff), conda env `sqcircuit-bench`, same machine | compared; to re-time on a quiet machine |
| current mirror, full circuit | none (exact diagonalization stops at N = 3) | on hold |

## Environment

```bash
julia --project=bench -e 'using Pkg; Pkg.instantiate()'
julia --project=bench -t 8 bench/hensel_two_fluxonia/run.jl
```

`bench/Project.toml` takes the package from this checkout (`[sources]`), so a run measures the working tree.
Knobs shared by every case (`common.jl`):

| variable | meaning | default |
|---|---|---|
| `BENCH_DEVICE` | `cpu` or `gpu` (loads CUDA and cuTENSOR) | `cpu` |
| `BENCH_REPEATS` | timed repeats per point after the first (cold) call | `3` |
| `BENCH_BLAS_THREADS` | BLAS threads on the CPU | `8` |
| `BENCH_SMOKE` | `1` for tiny sizes (checks the script, not the physics) | `0` |
| `BENCH_OUT` | results directory | `<case>/results/<host>-<date>/` |

## What every run records

One JSON file per case run (`results.json`), with one record per (setting, parameter point):

1. **Identity**: case, label, commit (and whether the tree is dirty), host, CPU, device, Julia/BLAS threads, Julia
   and package versions, date; the tree's settings (bases and cutoffs, kept counts, solver per node), the solver
   tolerance and the number of levels.
2. **Sizes**: the flat product dimension (what one product space of every leaf basis would hold), the root dimension
   and their ratio.
3. **Accuracy**: the lowest `k` levels (`k = 8`) and transitions ω₀ⱼ; the error against the case's reference; the
   change under one refinement step of the ladder (below).
4. **Cost**: wall time of the first call (with compilation) and the median of the warm repeats; per node (from
   `node_timings`): operator build, eigensolve and assembly (selection, phases, ports) times, dimension, kept count,
   solver, and for LOBPCG nodes iterations, matvecs and whether the start was warm. Time per root matvec =
   root solve time / root matvecs. In a sweep each point is warm-started from the previous one, as a user would;
   the first point is the cold one.
5. **Memory**: the process's peak RSS after the run (a high-water mark: it only grows within a process, so the
   headline setting is also run alone in a fresh process for its memory figure; the record also holds the RSS
   after loading the packages, `baseline_maxrss_bytes`, so a figure net of the baseline, as Hensel et al. report, can
   be formed; it still includes the compiled code), bytes allocated during the warm
   call, and the stored system's size (`Base.summarysize`). On the GPU, the device memory in use after the run.
6. **Reference side**: the same fields where we run the reference ourselves; for a published reference, only what
   the paper reports, with its hardware.

## The accuracy rule: time to a fixed accuracy

The headline is the time to reach a fixed accuracy on the transitions among the lowest `k` levels: **1 MHz**.
That is also the accuracy class of Hensel et al. (an absolute tolerance on each transition, arXiv:2611.05627v1
`numbers.tex` `\vTolMHz`, Sec. II); their exact-HD Lanczos ran at a 10 kHz residual tolerance (`\vKrylovTolTwoQ`,
App. D), far below the class, so the Hensel cases also report the stricter 10 kHz rung.

1. Each case defines a **ladder**: a sequence of tree settings, each refining the previous one (more kept states,
   larger cutoffs).
2. The **accurate setting** is the first rung whose transitions (all of them among the lowest `k`, at every parameter
   point) change by less than the target on the next rung. The next rung is the check and is run too.
3. The **headline time** is the accurate setting's warm time per point. The check's time is reported beside it,
   since a user also has to run it to know the setting is good enough.
4. Where the case has an independent converged or exact reference, the accurate setting must also lie within the
   target of it; this catches a ladder that has stalled rather than converged.
5. Every solver tolerance is at least ten times below the target (LOBPCG `tol` is in GHz on eigenvalue residuals).

## Follow-ups

- **`Auto` for small roots** (2026-11-12): study whether `Auto()` should pick `Lanczos` on the CPU for small roots that keep only a
  few states. On the c66 autodiff tree (a 2 400-state root keeping 6 levels) Lanczos took 0.32 s against 2.21 s for
  `Dense`'s full decomposition and 0.64 s for `LOBPCG`, with the Jacobian and Hessian also fastest (0.21 / 0.73 s against
  0.25 / 1.23 and 0.27 / 1.27 s) and every result the same to 1e-13. The study should map the crossover in root size and
  in the number of kept states against `Dense` and `LOBPCG`, with and without `retain`, and with the derivative pass.
