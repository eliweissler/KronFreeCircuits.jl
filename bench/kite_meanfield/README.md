# Figure 8(b): the KITE from mean-field leaves

`levels.jl` reproduces Figure 8(b) of the circuit-quantization paper with this package: for the KITE in three coordinate frames, the largest error among the four lowest levels, against the paper's reference levels, as a function of the number `m` of states kept per leaf, with each leaf's basis built from the mean-field states of the two wells of the self-consistent loop.
It compares the twelve errors with those of the repository owner's reference calculation and fails if they do not agree.

## Command

From the repository root (Julia 1.13; instantiate the project once with `julia --project=bench/kite_meanfield -e 'using Pkg; Pkg.instantiate()'`):

```
julia --project=bench/kite_meanfield bench/kite_meanfield/levels.jl
```

`KITE_MEANFIELD_REFSOLVE=1` adds the reference solve described below (off by default; its three solves took 9, 10 and 15 s in the run below).
The script uses only the committed fixtures `test/fixtures/kite/{published,normal,optimum}.json` and this package; it needs no Python.
Run it single-threaded (no `-t`, no `JULIA_NUM_THREADS`); the script sets one BLAS thread and stops if Julia has more than one thread.
It does not use the GPU.
No `Manifest.toml` is committed (`Manifest*.toml` is ignored by git).

## What the script does

For each frame (published coordinates, normal modes, the product-state optimum), with `θ = (;)`:

1. It loads the frame's spec and builds one leaf per mode on the bases of `suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5)`: `ChargeBasis(16)` (33 states) on the compact mode and a 120-point `PhaseGrid` on each extended mode.
   Each leaf is a `Node` with `solver = Dense()`, `compute = 20`, `keep = lowest(20)` and `arithmetic = :complex`, so it keeps 20 mean-field states (`m` goes to 20).
2. It runs `mean_field_solutions` once from the 33 starts of `test/test_meanfield_kite.jl`: the compact mode at `0` or `π`, each extended mode at `0`, `π/2`, `π` or `-π/2`, and the delocalized start.
   It passes `distinct = -Inf`, which keeps every run (lowest energy first, ties in the order of the starts), so that it can count the converged runs; it then keeps the distinct ones by `mean_field_solutions`' own rule at its default `1e-3`.
   The wells are the distinct solutions within `0.5` GHz of the lowest, in that order.
3. For each `m` in 8, 12, 16, 20 and each leaf it forms one `BasisSource` per well, the well's `m` lowest mean-field states of that leaf, and merges them with `union_basis(sources; m)` (level by level, Gram–Schmidt twice, drop threshold `1e-6`), with no bare ladder.
   It presents each leaf through its union with `augment` (the leaf's bare operator projected on the union) and solves the root over the three augmented leaves, `m³` states, for its eight lowest levels with `materialize(root, spec, (;); augmented)`.
   The root is `Dense()` at `m = 8` (512 states) and `LOBPCG(tol = 1e-10)` above; the script recomputes the residual `‖H v − E v‖` of the four lowest levels with `KronFreeCircuits.operator(sys)` and fails if one is above `1e-9` GHz.
4. The error is `max(levels[1:4] - REF)` with `REF = [-9.523995, -9.510525, -8.453790, -8.337862]` GHz, the paper's reference levels.
5. In the optimum frame it also solves the two variants of the paper's Appendix C.4 at the same four `m`: one well only (the lowest solution's `m` states), and the bare ladder alone (the leaf's own `m` lowest eigenstates from `bare_child`).

A union that cannot reach `m` would be printed and rebuilt with `allow_fewer = true` and marked; none occurred.
For each union that dropped a vector the script replays the merge and prints every candidate's Gram–Schmidt remainder below `1e-3`, and checks that the replay reproduces the union's report.

The reference values are hard-coded in the script with a comment naming their source, the owner's untracked `mean_field_ref_dont_commit/kite_reference/results/levels_*.json`: the reference calculation's eight lowest levels per frame and `m`, from which the script computes its twelve errors with the same formula.
The script exits nonzero if any of the twelve errors is outside a factor of two of the reference's, if the ordering of the three frames by error differs from the reference's at any `m`, if a residual is above `1e-9` GHz, if a frame does not have exactly two wells, or if a replayed merge does not reproduce its report.

## Result

All numbers below are from one run, `KITE_MEANFIELD_REFSOLVE=1 julia --project=bench/kite_meanfield bench/kite_meanfield/levels.jl`, on 2026-11-10, which ended with `all checks passed` and exit status 0.

### The twelve errors

`max(levels[1:4] - REF)` in GHz, this run / the reference calculation (ratio):

| frame | m = 8 | m = 12 | m = 16 | m = 20 |
|---|---|---|---|---|
| published | 6.9179e-01 / 6.9179e-01 (1.0000) | 2.2134e-01 / 2.2134e-01 (1.0000) | 3.7649e-02 / 3.7701e-02 (0.9986) | 6.5900e-04 / 6.7757e-04 (0.9726) |
| normal modes | 6.2176e-01 / 6.2176e-01 (1.0000) | 1.9890e-01 / 1.9890e-01 (1.0000) | 3.6005e-02 / 3.6005e-02 (1.0000) | 8.0124e-04 / 8.0124e-04 (1.0000) |
| optimum | 3.6677e-02 / 3.6677e-02 (1.0000) | 2.2679e-03 / 2.2679e-03 (1.0000) | 2.6093e-04 / 2.6093e-04 (1.0000) | 2.0717e-05 / 2.0717e-05 (1.0000) |

Every error is within a factor of two of the reference's; the largest departure is the published frame at `m = 20`, ratio 0.9726.
The ordering of the frames by error is the reference's at every `m`: optimum < normal modes < published at `m = 8, 12, 16`, and optimum < published < normal modes at `m = 20`.

### The levels

The largest difference between this run's eight levels and the reference calculation's, in GHz:

| frame | m = 8 | m = 12 | m = 16 | m = 20 |
|---|---|---|---|---|
| published | 2.3e-06 | 5.6e-07 | 2.9e-04 | 6.5e-05 |
| normal modes | 1.5e-13 | 1.2e-13 | 1.3e-13 | 1.1e-13 |
| optimum | 9.9e-14 | 7.2e-14 | 1.5e-13 | 1.2e-13 |

In the normal-mode and optimum frames the reproduction is exact to round-off.
In the published frame it is close at `m = 8` and `12` and departs by up to 2.9e-4 GHz at `m = 16` and `20`; the identified cause is under "The published frame" below.
The largest residual of the four lowest levels of any root solve was 9.7e-11 GHz.

### The wells

Every frame has two wells, degenerate in product energy to the nine printed digits, and every well converged.

| frame | runs converged of 33 | distinct solutions | well 1: E_prod (GHz), start, iterations | well 2: start, iterations |
|---|---|---|---|---|
| published | 18 | 3 | -8.083188250, (0, 0, π/2), 220 | (0, π, -π/2), 220 |
| normal modes | 13 | 3 | -8.466359372, (π, -π/2, π/2), 214 | (π, π, π/2), 220 |
| optimum | 33 | 2 | -9.429473600, (0, 0, 0), 59 | (π, -π/2, π), 100 |

A start is written as the displacements of (φ₁, φ₂, φ₃).
The product-state gaps `E_prod - REF[1]` are 1.440807 GHz in the published frame (the paper's 1.44; the reference calculation's 1.440807), 1.057636 GHz in the normal-mode frame, and 0.094521 GHz in the optimum frame (the paper's 0.09; the reference calculation's 0.094521).

### The union reports

Taken and dropped per well; the compact mode `varphi_1` and, outside the published frame, every leaf took `m/2` from each well and dropped none.

| frame | leaf | m = 8 | m = 12 | m = 16 | m = 20 |
|---|---|---|---|---|---|
| published | varphi_1 | 4+4, 0 dropped | 6+6, 0 dropped | 8+8, 0 dropped | 10+10, 0 dropped |
| published | varphi_2 | 4+4, 0 dropped | 6+6, 0 dropped | well 1 9, well 2 7 (1 dropped) | well 1 13, well 2 7 (5 dropped) |
| published | varphi_3 | 4+4, 0 dropped | 6+6, 0 dropped | 8+8, 0 dropped | 10+10, 0 dropped |
| normal modes | each leaf | 4+4, 0 dropped | 6+6, 0 dropped | 8+8, 0 dropped | 10+10, 0 dropped |
| optimum | each leaf | 4+4, 0 dropped | 6+6, 0 dropped | 8+8, 0 dropped | 10+10, 0 dropped |

The one-well and bare-ladder unions of the optimum frame took all `m` of their single source and dropped none.
No union fell short of `m`.

### Appendix C.4, optimum frame

`max(levels[1:4] - REF)` in GHz, this run on this grid (`extent_scale = 1.5`) / the owner's log `leaves.log` on another grid (`extent_scale = 1.0`), for reference only, with no pass or fail attached:

| basis | m = 8 | m = 12 | m = 16 | m = 20 |
|---|---|---|---|---|
| two wells | 3.6677e-02 / 3.7e-02 | 2.2679e-03 / 2.3e-03 | 2.6093e-04 / 2.7e-04 | 2.0717e-05 / 2.6e-05 |
| one well | 2.2535e+00 / 2.3e+00 | 2.1421e-01 / 2.1e-01 | 2.2673e-02 / 2.2e-02 | 3.5793e-03 / 2.9e-03 |
| bare ladder | 4.0510e-01 / 4.1e-01 | 3.2522e-02 / 3.3e-02 | 4.3791e-03 / 4.4e-03 | 6.0693e-04 / 4.8e-04 |

The appendix's conclusion holds on this grid: at every `m` two wells give the smallest error, the bare ladder the next, and one well the largest.

### The reference solve

With `KITE_MEANFIELD_REFSOLVE=1` the script solves the published frame on the bases of `suggest_bases(spec; charge_cutoff=24, max_points=160, extent_scale=2.0)`, `ChargeBasis(24)` and 160-point grids over ±22.624245 and ±22.517312, the reference calculation's own grid, on a plain tree: `Dense` leaves keeping `K` states each (the compact leaf all of its 49 when `K > 49`), the root by `LOBPCG(tol = 1e-9)` for its four lowest levels.

| K | states | E₀ | E₁ | E₂ | E₃ | change from the previous K |
|---|---|---|---|---|---|---|
| 40 | 64000 | -9.5239951 | -9.5105247 | -8.4537902 | -8.3378620 | |
| 48 | 110592 | -9.5239952 | -9.5105249 | -8.4537903 | -8.3378621 | 1.4e-07 |
| 56 | 153664 | -9.5239953 | -9.5105249 | -8.4537903 | -8.3378621 | 6.5e-09 |
| REF | | -9.5239950 | -9.5105250 | -8.4537900 | -8.3378620 | |

At `K = 56` the levels differ from `REF` by -2.5e-07, +1.3e-07, -3.2e-07 and -1.3e-07 GHz, inside the six-decimal rounding of `REF`.
Raising `K` from 48 to 56 moved them by at most 6.5e-09 GHz, so the tree is converged far below `1e-6` GHz; the residuals were at most 7.9e-10 GHz.

### Warnings

Two known messages are counted and dropped by the script's logger, which records where each came from; every other message passes, after a line naming the step that raised it.

- `fix_phases!` (108 messages): only in bare solves of the compact leaf (9 per frame, columns 2 to 18), in the self-consistent loop (18 per frame: the loop's own bare solve and the first iteration of the delocalized start, whose cosine expectations are all zero), and in the bare-ladder variant, whose basis is the compact leaf's bare states.
  In each case the columns named are the even ones, the second of each charge pair `|±n⟩`, degenerate at zero offset charge when no cosine acts; no well's states are among them.
- The `PhaseGrid` edge warning (19678 messages, from the solves of the extended leaves, which keep 20 states): the lowest kept state any of them named is state 17, on either grid.
  The script measures the edge support of every state offered to any union directly: the largest is 6.4e-07, below the warning's `1e-6`.
  Of the wells' own 20 states, those above `1e-6` start at state 17 (published, `varphi_3`, well 2, 1.1e-05), 19 or 20, above every state a two-well union reached (at most state 13); the one-well variant uses all 20 states of the optimum frame's well 1, none of which is above `1e-6`; no well's lowest states touch an edge.

One other warning passed, raised during the optimum frame's one-well solve at `m = 12`: the root's "heavy hybridization" phase note for its kept state 2 (joint-peak ratio 0.27 < 0.5).
`augment`'s docstring describes it: a union basis that is not made of eigenvectors of the child can trigger it, and it reports on the phase convention, not an error.

### Wall time and load

Wall times per frame, including the loop from 33 starts, the four root solves and, in the optimum frame, the eight variant solves: published 77.0 s (it includes Julia's compilation), normal modes 39.9 s, optimum 36.9 s; the reference solve took 9, 10 and 15 s for `K = 40, 48, 56`, and the whole run 3 min 12 s (from `time`).
The run was single-threaded on the CPU, with one BLAS thread.
Another job was running on this machine from the same checkout; the load average (1, 5, 15 min) was 0.69 0.81 1.26 at the start and 0.98 0.89 1.21 at the end of the figure, so the times were taken under that light load.

## The published frame

The published frame reproduces the reference's errors with ratios 0.9986 at `m = 16` and 0.9726 at `m = 20`, and its levels to 2.9e-04 GHz, against round-off in the other two frames; the target (a factor of two) is met, and this is the cause of the looser agreement.
In this frame the two wells' mean-field states coincide on the mode `varphi_2` up to a few parts in a million: the replayed merge prints Gram–Schmidt remainders of the second well's states 1 to 12 of 3.9e-06, 4.9e-06, 2.2e-06, 3.5e-04, 2.3e-06, 2.2e-06, 1.7e-06, 8.3e-07, 7.5e-08, 3.6e-07, 1.1e-07 and 4.0e-07, against the drop threshold `1e-6`.
So the union takes the second well's states 1 to 7 as normalized remainders of order 1e-6, which are dominated by the differences between the two wells' states at that level, and drops state 8 and beyond (one drop at `m = 16`, five at `m = 20`, the first well filling in).
The test `test/test_meanfield_kite.jl` records that in this frame the reference calculation stops its loop one iteration later than this package, so the two codes' wells differ slightly, at the level of these remainders; the directions admitted, and which candidates fall on which side of `1e-6` (state 7 at 1.7e-06 and state 8 at 8.3e-07 are close to it), therefore depend on those differences and on round-off, and the two codes' unions on `varphi_2` need not be the same.
At `m = 8` and `12` the second well offers its states 1 to 4 and 1 to 6, whose remainders (2.2e-06 to 3.5e-04) are all at least twice the threshold, and the levels agree with the reference to 2.3e-06 and 5.6e-07 GHz; the departure appears at `m = 16` and beyond, where states 7 and 8, the ones nearest the threshold, are offered.
In the normal-mode and optimum frames no union dropped a vector, and the levels agree with the reference to 1.5e-13 GHz.
Which candidates the reference calculation admitted on `varphi_2` is not recorded in its result files, so this reading is inferred from the remainders printed here, not checked against the reference's own union.
