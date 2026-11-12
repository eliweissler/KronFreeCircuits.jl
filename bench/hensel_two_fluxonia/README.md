# Two fluxonia with a tunable coupler (Hensel et al., arXiv:2611.05627v1 Sec. IV)

The suite's first case. The circuit, its normal modes (checked against the paper's Table tab:modes on every run), the
local and sheared coordinates and the trees come from the exploratory study in
`../../bench_archive/hensel_two_fluxonia/` (2026-11-06), which also chose the trees.
Run: `julia --project=bench -t 8 bench/hensel_two_fluxonia/run.jl [ladder] [reference] [paper]` (see the header of
`run.jl` for the stages and knobs). Results below: 2026-11-11, commit `4f35e62`, ewdesktop (i7-11700K, 8 cores, RTX
A4500 20 GB, Julia 1.13.2), raw records in `results/`.

## The comparison

The paper's accuracy class is 1 MHz on each transition (`\vTolMHz`); every cost below is per point of the six-point
coupler-flux grid, lowest eight levels (six for the GPU paper-space run), warm points 2 to 6 as the paper reports.

| method | space | machine | per point | memory | error vs converged |
|---|---|---|---|---|---|
| paper: exact HD, Lanczos (published) | 14⁶ = 7.5 M normal-mode Fock | 16-core Xeon 6517P | 13.6 min | 16 GB | 1.04 MHz (their space) |
| paper: TN, bond cap 8 (published) | same, MPO | same | 2.4 min | 0.66 GB | in the class |
| paper: TN, bond cap 64 (published) | same | same | 17.4 min | 2.1 GB | in the class |
| their space, flat LOBPCG (this package) | 7.5 M | A4500 | 2.4 min (1.5–4.0) | 10.3 GB GPU, 9.1 GB host peak | 1.04 MHz |
| **binary tree, kM 120: accurate at 1 MHz** | 7 200 root | 8 CPU cores | **0.84 s** | 0.59 GB net (1.31 GB peak) | 0.17 MHz |
| **binary tree, kM 240: accurate at 10 kHz** | 14 400 root | 8 CPU cores | **1.83 s** | 0.67 GB net (1.39 GB peak) | 3.9 kHz |
| same two trees | | A4500 | 1.17 s, 1.92 s | 0.25, 0.65 GB GPU | same |
| reference: flat local tree, kc 80 | 1.47 M root | A4500 / 8 CPU cores | 8.0 s / 36.7 s | 4.4 GB GPU | (is the reference) |

Published rows: Table tab:cost (sec04 ll. 118-135, numbers.tex ll. 79-89), wall time summed over grid points 2 to 6,
divided by 5 here; peak memory net of the process baseline. "Net" for our trees is the peak RSS of a fresh process
running that rung alone minus its RSS after loading the packages (0.72 GB), so it still includes compiled code.

At the paper's own 1 MHz class the tree is accurate after **0.84 s per point on 8 CPU cores**, against 13.6 min for the
paper's exact HD (about 970×) and 2.4 min for its cheapest DMRG (about 170×), on a CPU with half the cores. A user
also runs the check rung to know the setting is enough: 0.84 + 1.83 s. Even at 10 kHz, a hundred times stricter than
the class, it is 1.83 s per point (about 450× and 80×). Against the same machine solving the paper's space flat on the
GPU (2.4 min per point, itself 1.04 MHz from the converged answer, just outside the class), the 1 MHz tree is about
170× faster and the GPU is not needed.

## Same tree in scqubits (`scqubits_ref/`)

The like-for-like reference: scqubits 4.3.1 (conda env `scqubits-bench`; it needs `qutip<5.2` and `scipy<1.15`, see
`scqubits_ref/requirements-pinned.txt`) on the same circuit (a scqubits YAML of `circuit.jl`'s elements), the same
sheared frame (the full transformation in one `configure` call, fluxes on the junction branches), the same binary
tree (`system_hierarchy`), the same truncations, the same bases (`STENCIL = 9` is our order-8 grid; f2 and f3 get
22 Fock levels because scqubits keeps at most dim − 2, whose lowest 20 are our 20 of 20) and the same solvers: dense
at the pair nodes, scqubits' own at the leaves, and at M and the root scipy's LOBPCG with our settings (m + 4 guard
columns, product-state start, shifted Jacobi 1/(D − min D + 1), tol 1e-6) or scqubits' default ARPACK. Every point is
cold on both sides (no warm start), 8 threads each, 2026-11-12 on ewdesktop, one job at a time on an otherwise idle
machine. The levels agree to 3×10⁻⁸ GHz at every flux.

| solver at M and the root (both codes: dense below) | kM 120 (accurate at 1 MHz) | kM 240 (accurate at 10 kHz) |
|---|---|---|
| **ours**, LOBPCG, shifted Jacobi | **0.71 s** (0.67–0.77) | **1.22 s** (1.13–1.25) |
| ours, LOBPCG, default per-Ritz preconditioner | 0.83 s (0.74–0.87) | 1.69 s (1.53–1.85) |
| **scqubits**, scipy LOBPCG, our settings | **19.4 s** (14–118) | **58 s** (42–158) |
| scqubits as shipped (no solver set anywhere) | 25.1 s (14.5–27.1) | 86 s (54–91) |
| scqubits, ARPACK, pair nodes dense | | 85 s (56–89) |
| scqubits, torch.lobpcg on the GPU (four real points) | 23.1 s (22.3–23.9) | 181 s (89–447) |
| scqubits, torch.lobpcg on the CPU (four real points) | 238 s (173–296) | stopped (over an hour) |
| scqubits, dense at M, ARPACK at the root (Φ_c = 0.3 only) | 21.2 s | |

Median per point over the six fluxes (range); the torch rows over the four fluxes torch solved (Φ_c = 0.1 to 0.4: at
0 and 0.5 scqubits' nodes are complex, and torch.lobpcg returns wrong values on complex Hermitian input, so those
points fell back to scipy's LOBPCG and are left out). scqubits' one-off setup (`Circuit` and `configure`): 7.4 s at
kM 120, 22 s at kM 240. At Φ_c = 0.3, kM 120, the other scqubits rows took 17.3 s (scipy LOBPCG) and 26.0 s (as
shipped); our tree 0.70 s. Every row's levels agree with ours to 3×10⁻⁸ GHz.
Same tree, same solver: about **27×** at kM 120 and **48×** at kM 240; against scqubits as shipped, 35× and 70×.
No solver choice in scqubits closes the gap: dense at M (it needs 120 of 3600 states) is as fast there as LOBPCG
(3.6 against 3.8 s at Φ_c = 0.3), torch.lobpcg on the GPU is no faster than scipy's on these node sizes (its M solves
also stalled at two kM 240 points), and on the CPU it is 10× slower; to meet our absolute 1e-6 GHz residual its
relative test needs a tolerance near 3×10⁻⁸.
Where scqubits' time goes, median per point: at kM 240, 22 s in the eigensolves and 36 s in everything else (building
the subsystem Hamiltonians and the interaction operators in the kept bases as sparse matrices); at kM 120, 9.6 and
9.7 s. scqubits' non-solve work alone is 14× (kM 120) to 30× (kM 240) our whole time per point.
Its solves are also slower than ours on the same nodes with the same preconditioner (the M node: 4 to 12 s against
0.4 to 0.9 s), for two visible reasons: scqubits does not fix the phases of a subsystem's eigenvectors, so its nodes
are complex Hermitian at Φ_c = 0 and 0.5 (ours stay real), and scipy's LOBPCG converged slowly on single points (the
104 s and 110 s M solves behind the 118 s and 158 s outliers); the rest is the implementation.
On these trees our shifted-Jacobi preconditioner is 15 to 30 % faster than our default per-Ritz one.

## The ladder

Binary tree in local coordinates with f2, f3 sheared onto the junctions; k1 = kept states of the fluxonium / Fock
leaves, k2 of the pair nodes, kM of the node over P(c, f1) and P(q1, f3). Worst change of any transition among the
lowest eight levels over the six fluxes:

| rung | change to next | vs reference | CPU s/pt | GPU s/pt |
|---|---|---|---|---|
| k1 40/20, k2 60, kM 120 | 170 kHz | 174 kHz | 0.84 | 1.17 |
| k1 40/20, k2 60, kM 240 | 0.47 kHz | 3.9 kHz | 1.83 | 1.92 |
| k1 60/30, k2 100, kM 240 | 3.4 kHz | 3.5 kHz | 4.37 | 3.76 |
| k1 60/30, k2 100, kM 480 | — | 0.04 kHz | 11.66 | 9.57 |

The rungs converge onto the reference (0.04 kHz at the top), so the ladder has not stalled.
The 3.4 to 3.9 kHz of rungs 2 and 3 against the reference is the reference's own truncation (its kc 80 coupler leaf
was 5 kHz from its neighbours in the 2026-11-06 study), not the tree's.

## Where the time goes

At the 1 MHz rung (CPU, Φ_c = 0.3, 0.77 s): the M node over (c, f1, q1, f3), a 3 600-state LOBPCG solve for 120 states,
takes 0.49 s (2 350 matvecs, because it computes many states of a small space); the root, 7 200 states for 8 levels,
0.08 s; the dense pair nodes 0.17 s; everything else (operator builds, ports) under 0.03 s. At 10 kHz the M node
(240 states) is 1.19 s of 1.55. The GPU does not help these trees (1.17 vs 0.84 s): every solve is small. It does help
the flat million-state reference (8.0 vs 36.7 s) and the paper's 7.5 M space.

## Caveats

- Different hardware: the published times are on a 16-core Xeon 6517P, ours on an 8-core i7-11700K; the same-machine
  row (their space on our GPU) is the like-for-like check on the speed-up's size.
- Each sweep point is warm-started from the previous one; the paper's published times are cold solves (its seeded
  chains mostly failed its checks). The first point of each sweep is cold and costs about the same here (0.73 s for the
  1 MHz rung once compiled), so this does not change the comparison.
- The ladder keeps one fluxonium phase grid (±3π, 181 points), on which kept leaf states above about the tenth reach
  the edge (warnings in the logs). The grid check (`HENSEL_GRID`, rungs 1 and 2, `results/cpu-grid-*`) moves every
  transition among the lowest eight levels by at most 1.6×10⁻¹⁰ GHz on ±3π with 241 points, 3×10⁻¹¹ on ±4π / 241 and
  8×10⁻¹² on ±5π / 301 (where no kept state reaches the edge): the edge states are high in the leaf spectrum and do not
  reach the low levels.
- The paper-space GPU run keeps six columns and no guard (memory), as in the 2026-11-06 study; the host CPU solve of
  that space (24 to 43 min per point on 8 cores, 2026-11-06) was not repeated.
