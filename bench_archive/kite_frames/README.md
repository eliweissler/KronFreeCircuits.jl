# The KITE in three frames, in one session

`run.jl` defines the KITE (the cos 2φ transmon) through circuit-quantization, moves it through three coordinate frames by the numeric loop route, solves each frame, times a frame-change loop, and writes a run record, all in one Julia session.

## Command

From the repository root, with `$CQ_PYTHON` the Python of an environment that has circuit-quantization installed:

```
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=bench/kite_frames bench/kite_frames/run.jl
```

Run it single-threaded (the default `julia` without `-t` or `JULIA_NUM_THREADS`); the script sets one BLAS thread and stops if Julia has more than one thread.
`KITE_FRAMES_OUT` sets the directory of the run record (default `bench/kite_frames/results/`), and `KITE_FRAMES_ITERS` the number of timed iterations (default 200).
`KITE_FRAMES_REFGRID=1` adds the check on the reference calculation's own grids described under "Against the paper's reference levels" (off by default; it added 27 s to the run below).
The `results/` directory is ignored by git: the run record is output, not committed.
The run takes five to eight minutes on the machine below, most of it in the 200 full `materialize` calls of the loop.

## What the script does

1. It builds the KITE with `kite()` of `test/python/kite.jl` (included, not copied), at the parameter values of circuit-quantization's `examples/hierarchical_kite.py`, at half flux and zero offset charge, and makes one `QuantizedCircuit(res; values=θ)`.
2. For each of the three frames of `kite()` (published, the identity; normal, circuit-quantization's `normal_mode_transformation`; optimum, the paper's `S*`) it builds the spec by the loop route, `Spec(qc, S)`, builds a three-leaf tree from that spec's own mode list, and prints the four lowest energies.
   The tree has one leaf per mode: `ChargeBasis` for a compact mode, `PhaseGrid` (order 8) for an extended mode, centered at the minimum of the extended block's quadratic part, `-E_L⁻¹ flux_linear`; it stops on any other mode type (none occurs here: every frame is `c e e`).
   Each leaf is diagonalized with `Dense()` and keeps its lowest states; the root keeps the lowest 4 of their product.
3. It solves each frame twice more, on a refined tree and on a wider one, and prints the change: that is the convergence evidence below.
   It then prints each frame's four lowest levels against the reference levels of the circuit-quantization paper's KITE figure, with the differences.
4. It times 200 iterations, after 5 untimed warm-up iterations, of "random admissible `S` near the optimum → `Spec(qc, S)` → `materialize`", and checks in every iteration that every coefficient of the spec is a constant.
5. It writes the run record of the optimum frame with `save_record(path, spec; circuit=res, frame=S*, values=θ, metadata)`, where `spec` is the spec it solved in that frame in step 2, and checks that `load_record` of the written file gives back that spec, that frame and those values (in the circuit's parameter order).

It exits nonzero if any of the four energies differs between two frames by more than `2e-5` GHz, if a coefficient of a loop spec is not a `KronFreeCircuits._Constant`, if the compiled-coefficient count grows during the loop (or does not grow for the probe, below), if Julia compiles anything between a frame change and the first eigensolve of its `materialize` in any timed iteration, or if `load_record` does not return what was written.
Julia compile time from the first eigensolve to the end of `materialize` prints a warning but does not fail the run: a rarely taken branch of the eigensolver may compile once in 200 full solves without any coefficient being compiled, which the cache count checks separately.
The frame tolerance comes from the convergence check below: the refined and wider trees move the frames' levels by at most 7.9e-6 GHz, so frames that agree to their discretization differ by under 1e-5 GHz, and `2e-5` leaves a factor of two.

### The root solver

The brief asked for a `Dense` root.
The root here uses `Auto()`, which resolves to LOBPCG on these sizes, because a `Dense` root cannot reach the cutoffs the frames need in reasonable time: with `k` states kept per leaf the root has `k³` states, and in an exploration before this script (multithreaded BLAS, not part of the run below) a `Dense` root took 23 s at `k = 18` (5832 states) and 121 s at `k = 22` (10 648 states), while the ground energy of the published and normal frames still moved by 4e-3 and 8e-3 GHz from `k = 16` to `k = 24`.
The leaves are `Dense`.

### The random frames

The frame is `φ = S φ′` with mode types `(c, e, e)`.
circuit-quantization allows a nonzero entry `S[i, j]` only for the type pairs (row, column) (c, c), (c, e), (e, e) (and pairs with harmonic modes, absent here), and requires the compact block to be an integer unimodular matrix.
So each loop frame is `S*` plus `0.02 randn` on `S[1, 2:3]` and `S[2:3, 2:3]`; `S[1, 1]` stays the integer 1 and `S[2:3, 1]` stays exactly zero.
The generator is `Xoshiro(20261110)`, so the run is reproducible.

## Results

Measured on 2026-11-10 on an Intel i7-11700K with Julia 1.13.2, one Julia thread and one BLAS thread.

Trees (each leaf keeps the given number of states; `ChargeBasis(n)` has `2n + 1` states):

| tree | compact leaf | extended leaves | kept per leaf |
|:-----|:-------------|:----------------|--------------:|
| grid (the frames) | `ChargeBasis(10)` | 121 points over center ± 2.5π | 32 |
| refined | `ChargeBasis(14)` | 161 points over center ± 2.5π | 40 |
| wider | `ChargeBasis(10)` | 193 points over center ± 4π | 40 |
| loop | `ChargeBasis(10)` | 121 points over center ± 2.5π | 24 |

The four lowest energies (GHz), as printed:

```
  published modes cee grid     -9.523981  -9.510510  -8.453776  -8.337848
  published           refined  -9.523981  -9.510510  -8.453776  -8.337848   (max change 2.94e-08)
  published           wider    -9.523981  -9.510510  -8.453776  -8.337848   (max change 2.18e-08)
  normal    modes cee grid     -9.523974  -9.510503  -8.453770  -8.337841
  normal              refined  -9.523981  -9.510510  -8.453776  -8.337848   (max change 7.26e-06)
  normal              wider    -9.523982  -9.510511  -8.453777  -8.337848   (max change 7.91e-06)
  optimum   modes cee grid     -9.523981  -9.510510  -8.453776  -8.337848
  optimum             refined  -9.523981  -9.510510  -8.453776  -8.337848   (max change 2.80e-08)
  optimum             wider    -9.523981  -9.510510  -8.453776  -8.337848   (max change 1.37e-07)
largest spread across frames: 7.23e-06 GHz (tolerance 2e-05)
```

The ground energy is −9.523981 GHz, the paper's −9.524 GHz.
The three frames agree to 7.2e-6 GHz, within the tolerance of `2e-5` GHz that the script enforces.

### Against the paper's reference levels

The reference levels of the circuit-quantization paper's KITE figure, as the repository owner supplied them, are hard-coded in the script: the four lowest levels in the published coordinates by direct diagonalization on 1 254 400 states, and in the optimized frame on a different grid (the paper states that the two agree within 1e-5 GHz).
As printed (GHz; difference = this run − reference):

```
  reference, published      -9.523995  -9.510525  -8.453790  -8.337862
  reference, optimized      -9.523992  -9.510522  -8.453784  -8.337853
  published - ref published +1.4e-05   +1.5e-05   +1.4e-05   +1.4e-05
  published - ref optimized +1.1e-05   +1.2e-05   +7.7e-06   +5.2e-06
  normal - ref published    +2.1e-05   +2.2e-05   +2.0e-05   +2.1e-05
  normal - ref optimized    +1.8e-05   +1.9e-05   +1.4e-05   +1.2e-05
  optimum - ref published   +1.4e-05   +1.5e-05   +1.4e-05   +1.4e-05
  optimum - ref optimized   +1.1e-05   +1.2e-05   +7.8e-06   +5.4e-06
  ground doublet splitting: 13.470 MHz (reference, published: 13.470 MHz)
```

The ground doublet splitting agrees with the reference to the printed 1 kHz.
Every level of this run lies above the published-coordinates reference by 1.4e-5 to 1.5e-5 GHz in the published and optimum frames (2.0e-5 to 2.2e-5 GHz in the normal frame, whose own truncation error is about 8e-6 GHz, below).
That nearly uniform offset is not explained by the discretization of this run: the refined and wider trees move the published and optimum levels by at most 1.4e-7 GHz, a hundred times less.
For scale, the two reference lists differ from each other by up to 9e-6 GHz, and this run's levels lie between 5e-6 and 1.2e-5 GHz above the optimized-frame reference.

The offset is the reference calculation's own discretization error.
`run.jl` reproduces it when `KITE_FRAMES_REFGRID=1` is set: it builds the reference's bases with circuit-quantization, solves the published and optimum frames on them, and prints the levels beside the paper's (it checks nothing).
The reference is a direct solve on the grids of circuit-quantization's `default_bases(h, charge_cutoff=24, max_points=160, extent_scale=2.0)`: `ChargeBasis(24)` and two phase grids of 160 points centred at zero, with steps of 0.2846 and 0.2832 in the published frame (extents ±22.624245 and ±22.517312) and 0.3172 and 0.2971 in the optimized frame (±25.221210 and ±23.618681), against 0.131 here, with the same order-8 stencils.
On exactly those grids (`PhaseGrid(-extent, extent, 160)` reproduces the reference's nodes to 7e-15), with the same three-leaf tree keeping 48 states per leaf (110 592 product states at the root), the run of 2026-11-10 with `KITE_FRAMES_REFGRID=1` on the code of commit `d3f338c` printed (9 s and 18 s for the two frames):

| frame | levels on the reference grid (GHz) | minus the paper's reference (GHz) |
|:------|:-----------------------------------|:----------------------------------|
| published | −9.523995, −9.510525, −8.453790, −8.337862 | −2.5e-7, +1.4e-7, −3.1e-7, −1.3e-7 |
| optimized | −9.523992, −9.510522, −8.453784, −8.337853 | −3.3e-7, +5.6e-8, +1.3e-7, −4.5e-7 |

In an earlier exploration (not printed by `run.jl`), keeping 56 states per leaf changed these by at most 1e-8 GHz (40 per leaf, by about 1.5e-7), so the tree's truncation error on these grids is well below 1e-6 GHz.
The remaining differences, at most 4.5e-7 GHz, are within the rounding of the reference values to 1e-6 GHz.
So the reference's coarse grid lowers its levels by about 1.4e-5 GHz (the order-8 stencil underestimates the kinetic energy at that step), and this run's levels, converged on a grid more than twice as fine, are the more accurate ones.
The limits: the comparison reproduces the reference's discretization, not its code (the Hamiltonian comes from this package's `Spec(qc, S)` and the operators from its `PhaseGrid` and `ChargeBasis`), and it agrees with the reference values only to their printed six decimals.

### Convergence

The convergence evidence is the change on refining, printed above.
In the published and optimum frames the grid tree's energies move by at most 3e-8 GHz on the refined tree (finer spacing, larger charge cutoff, 40 kept states) and at most 1.4e-7 GHz on the wider one (a window of ±4π instead of ±2.5π at the same spacing, 40 kept states).
In the normal-mode frame they move by up to 7.3e-6 GHz (refined) and 7.9e-6 GHz (wider), both towards the other frames' values: the normal frame converges more slowly in the number of kept leaf states, and its truncation error at 32 kept states is the whole of the 7.2e-6 GHz spread between frames.
On the refined and wider trees all three frames agree to about 1e-6 GHz.
The remaining discretization error of the grid tree is therefore about 1e-7 GHz in the published and optimum frames and about 8e-6 GHz in the normal frame.

`materialize` warns during the frame solves that the `PhaseGrid` windows are too small: kept leaf states have more than `1e-6` of their norm on the four edge points.
The leaves are harmonic oscillators (the cosines couple all three modes, so they act only at the root), and in the published frame even the leaf ground state is wide (about `1.6e-5` amplitude at ±2.5π, with `E_L = 2.54` and `E_C ≈ 2.48` GHz; an exploration result, not printed by `run.jl`); the higher kept leaf states reach the edge in every frame.
The root's low states are confined by the junction cosines, and the wider tree (±4π) gives the same energies to 1.4e-7 GHz in the published and optimum frames, so the warning does not affect the four energies reported.
The script also drops the leaves' `fix_phases!` warnings (high leaf states with no weight on the phase mask) and prints how many it dropped.

### Timings

Means over the 200 iterations:

| quantity | mean |
|:---------|-----:|
| (a) frame change alone, `Spec(qc, S)` | 0.355 ms |
| (b) frame change through `materialize`'s setup | 0.569 ms |
| frame change through the full `materialize` (context) | 1194 ms |

Timing (b), first 20 and last 20 iterations: 0.571 ms and 0.566 ms.
These are from the run of 2026-11-10 on the code of commit `d3f338c` (the final fix pass of the review), which also printed every block in this README except the reference-grid table, printed by a second run on the same code with `KITE_FRAMES_REFGRID=1`; its energies are identical, to the printed digits, to those of the run before it.
Other Julia jobs may have been running on the machine; the timings vary from run to run by tens of percent (two earlier runs on the same day gave 0.366 ms, 0.594 ms and 1312 ms, and, under heavier load, 0.530 ms, 0.942 ms and 1831 ms).

Both (a) and (b) are under the 5 ms target of design §4.3.
The full `materialize` is dominated by the root's LOBPCG on 24³ = 13 824 product states, single-threaded; it is not a frame-change cost.

Timing (a) is `Spec(qc, S)` alone: the size and finiteness check in Julia, the NumPy `transformed(S)` in Python (admissibility checks, the matrix products), the conversion of the returned arrays to Julia, and `numeric_spec`.

Timing (b) runs from just before `Spec(qc, S)` to the first eigensolve of `materialize`.
The leaves' solver is a `Dense()` wrapper that records `time_ns()` at its first call, and `materialize` calls it first at the first leaf, after all its setup and before any other diagonalization (`src/materialize.jl`: `_materialize` recurses into the children first, and the first leaf's `_solve` is the first eigensolve).
So (b) includes: `Spec(qc, S)`; building the tree from the spec (the grid centers solve a 2×2 system); in `materialize`, the mode-type check, `circuit_terms` and `constant_term` of the spec, the parameter check, the periodic-harmonic check, routing the terms to nodes, the duplicate-mode check, the real/complex classification, the required ports and joint ports, the constant's value; and, at the first leaf, resolving its solver and building its `KronFreeOperator` (evaluating every routed coefficient).
It excludes the first leaf's eigensolve and everything after it, including the operator builds of the second and third leaves and of the root, which happen after the first eigensolve.
No file under `src/` was changed to measure it.

### Compilation check

As printed:

```
compilation check
  every coefficient a _Constant in every iteration: true
  RuntimeGeneratedFunctions cache entries: 0 before the loop, 0 after
  RuntimeGeneratedFunctions cache entries after compiling one probe coefficient: 1
  Julia compile time, frame change through materialize setup: 0.000 ms total, in 0 of 200 iterations
  Julia compile time, first eigensolve to end of materialize: 0.000 ms total, in 0 of 200 iterations
```

Three measures, all over the 200 timed iterations:

- Every `t.coeff.f` of `circuit_terms(spec)`, and the function of `constant_term(spec)`, is a `KronFreeCircuits._Constant` in every iteration (asserted in the loop).
- The RuntimeGeneratedFunctions body cache of `KronFreeCircuits` holds one entry per distinct compiled coefficient expression and never drops one, so it grows whenever any path (including `_make` on an `Expr` in `src/bo.jl` or `src/spectransform.jl`) compiles a new coefficient; it holds 0 entries before and after the loop, so nothing in this session ever compiled a coefficient.
  To show the measure is live, the script then compiles one coefficient, `Coefficient("2 * E_J", [:E_J])`, and the count becomes 1.
- Julia's cumulative compilation time (`Base.cumulative_compile_time_ns`) does not increase in any iteration, neither from the frame change to the first eigensolve nor from there to the end of `materialize`.
  A nonzero value in the first window fails the run; in the second it prints a warning (it has been zero in both windows in every run of the script so far).

Timing (b) does not grow over the loop (0.571 ms in the first 20 iterations, 0.566 ms in the last 20), so no per-iteration cost builds up.
