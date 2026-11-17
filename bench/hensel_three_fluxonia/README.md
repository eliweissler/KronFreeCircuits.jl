# Three fluxonia with two tunable couplers (Hensel et al., arXiv:2611.05627v1 Sec. V)

The eleven-mode circuit of the paper at its working point (every qubit loop at Φ_0/2, both couplers unbiased), lowest
eight levels. Exact diagonalization is out of reach (the paper's estimate: 2.2 TB, 66 h), so accuracy is judged by the
tree's own convergence, against the paper's Table tab:q3levels, and (pending) against a flat reference on a large-memory
server. Earlier exploratory study: `bench_archive/hensel_three_fluxonia/` (2026-11-07).

## Files

- `quantize.py` (circuit-quantization, conda env `cq-z4`): the circuit of Table tab:params, local coordinates (junction
  phases q1, q2, q3, c1, c2; frame modes f1..f6, E_C exactly diagonal, frame block of E_L diagonal), the shear of the
  frame modes onto the junctions, and the kronfree-spec export: `specs/three_fluxonia_<frame>.json`, with the shear
  fractions in `specs/three_fluxonia_<frame>_shear.json`. Frames: `rule` (the shear of
  `../hensel_two_fluxonia/frame_opt/shear_rule.md`, per frame mode and junction), `none`, `full`, `archive` (the
  2026-11-07 study's hand-picked fractions f1..f6 = 0.35, 0.40, 0.60, 0.80, 0.85, 0.95 of the full shear).
- `run.jl`, `trees.jl`: load a spec and run the four-level tree ladder (`HENSEL3_STAGE=ladder`, the default) or the flat
  reference (`HENSEL3_STAGE=flat`, one root over the five junction–partner leaves; `HENSEL3_DRYRUN=1` prints sizes and
  memory).
- `crosscheck/`: an independent hand-built construction of the same circuit in Julia (`julia_construction.jl`) and its
  comparison with the specs (`spec_vs_julia.jl`, `.log`): E_L to 1e-10 GHz, E_C to 7e-11 GHz in every frame, the levels
  of a small tree to 9e-13 GHz; and the GPU ladder of the rule frame run on that construction, identical to the spec's.

**Inductances.** The table prints L_c = 8.17, L_end = 59.3, L_br = 36.8 nH, rounded from exact loop targets (coupler loop
2L_c at E_L = 10 GHz, both qubit loops at E_L = 2.00 GHz, the table's derived rows). With the exact values the normal
modes match Table tab:modes to 1.2 MHz (the four stiff modes to the printed digit); with the printed values, as in the
2026-11-07 study, to 3.4 MHz, which moves the converged transitions by up to 1.5 MHz (that study found them 1.6 to 5.0
MHz below the paper; here 1.2 to 3.5).

**Shear by the rule.** s*_fJ = (X + c)/(X + 2c + P), c = 4E_C[J]/Ω_f, P = (8E_C[J]/Ω_f)²⟨n_J²⟩, X = ⟨δx_J²⟩ of junction J's
leaf ground state, iterated three times:

| frame mode (Ω, GHz) | q1 | q2 | q3 | c1 | c2 |
|---|---|---|---|---|---|
| f1 (2.9) | 0.78 | 0.75 | 0.78 | 0.18 | 0.18 |
| f2 (4.3) | 0.82 | 0.80 | 0.82 | 0.22 | 0.22 |
| f3 (8.5) | 0.90 | 0.88 | 0.90 | 0.35 | 0.35 |
| f4 (8.7) | 0.90 | 0.89 | 0.90 | 0.36 | 0.36 |
| f5 (21.1) | 0.96 | 0.95 | 0.96 | 0.57 | 0.57 |
| f6 (21.5) | 0.96 | 0.95 | 0.96 | 0.57 | 0.57 |

The fluxonia move slowly (inter-well), so every frame mode is stiff against them; the couplers are fast single wells
(about 17 GHz), so every frame mode is soft against them. Pairing (each junction with the frame mode it couples to most,
the same in every frame): (q1, f3), (q2, f4), (q3, f6), (c1, f1), (c2, f2), f5 loose at the root.

## Results (2026-11-12, ewdesktop, A4500 GPU, commit 4d2f693)

Four-level tree: leaves k1 40 (junction grids) / 20 (Fock), pairs k2 60, wings kM, the left wing with q2's pair kM2,
loose f5 at 4 levels; LOBPCG tol 1e-6 GHz. Raw records in `results/gpu-<frame>-20261112-1142/`.

Transitions from the ground state, deviation from the paper (MHz), and the warm wall time:

| frame | (kM, kM2) | deviation from the paper (MHz) | worst vs converged | s per point | root LOBPCG iterations |
|---|---|---|---|---|---|
| **rule** | (240, 240) | −1.22 −1.14 −1.22 −2.33 −1.82 −2.52 −0.65 | 2.9 MHz | 32 | 19 |
| **rule** | **(480, 480)** | **−1.26 −1.23 −1.30 −2.80 −2.56 −2.96 −3.52** | **17 kHz** | **153** | 20 |
| rule | (480, 960) | −1.27 −1.23 −1.31 −2.81 −2.57 −2.98 −3.53 | (converged) | 462 | 20 |
| archive | (240, 240) | +0.21 +0.39 −0.13 +7.33 +9.08 +2.19 +19.04 | 22.6 MHz | 34 | 35 |
| archive | (480, 480) | −1.21 −0.94 −1.25 −2.59 −2.14 −2.57 −2.77 | 0.76 MHz | 172 | 35 |
| full | (240, 240) | +4.76 +4.67 +4.33 +15.76 +14.44 +12.01 +46.10 | 49.6 MHz | 42 | 88 |
| full | (480, 480) | +0.52 +0.81 −0.05 +2.12 +2.47 +1.14 +7.50 | 11.0 MHz | 231 | 80 |

"Worst vs converged" is the largest difference of a transition from the rule frame at (480, 960).

- **The rule's frame converges fastest.** At the same tree size it is about 8× more accurate than the hand-picked shear
  and about 17× (at 240) to 650× (at 480) more accurate than the full shear; at (480, 480) it is converged to 17 kHz,
  where the hand-picked shear is 0.76 MHz and the full shear 11 MHz away. Its nodes are also cheaper to solve (20 root
  iterations against 35 and 80), so it is the fastest per point as well.
- **All three frames approach the same answer** from above (the hand-picked within 0.8 MHz at (480, 480), the full shear
  within 11 MHz and falling), so the converged transitions do not depend on the frame.
- **Against the paper:** at the 1 MHz class the tree is accurate at (480, 480), 153 s per point on the A4500. The converged
  transitions lie 1.2 to 3.5 MHz below the paper's, all on the low side. The paper's values are printed to 1 MHz and come
  from a truncated normal-mode space (12 Fock levels on the five low modes, 4 on the six stiff ones) whose stiff-mode
  shift for one more level alone is up to 0.48 MHz (its own table); in the two-fluxonium case the paper's truncated space
  sat 1.04 MHz above the converged answer. The flat reference (below) is to settle this independently.

## Grid range (2026-11-12)

The runs above used the 2026-11-07 study's grids, qubits on ±3π (181 points) and couplers on ±2π (121), on which kept
leaf states reach the grid edge (warnings in the logs: qubits from state 15, couplers from state 5, edge support up to
0.34). At the same spacing π/30, a scan on a small tree found the narrowest warning-free ranges: qubits ±5π (301
points) and couplers ±4π (241); ±4π qubits warn from state 36 and ±3π couplers from state 21. These are now the
defaults (`HENSEL3_QGRID`, `HENSEL3_CGRID`). The rule frame at (480, 480) on them (`results/gpu-rule-cleangrids/`) has
no range warning and moves every transition by at most 1.9e-13 GHz from the table above (couplers alone on ±3π:
2.3e-13, `results/gpu-rule-cgrid3pi/`), so the grid range plays no part in the offset from the paper.

## Cost

| what | wall time |
|---|---|
| paper, DMRG bond cap 32, one solve (16-core Xeon 6517P) | 81 min |
| paper, DMRG bond cap 128 (the table's levels) | 13 h |
| paper, the whole study (bond ladder) | 19 h |
| paper, exact HD | not run (estimate 66 h, 2.2 TB) |
| this, four-level tree, rule frame, (480, 480): accurate at 1 MHz | 153 s per point, A4500 |
| this, the check rung (480, 960) | 462 s per point, A4500 |
| this, four-level tree, (480, 480), CPU (2× EPYC 7713, 32 threads) | 225 s per point |
| this, four-level tree, (480, 480), desktop CPU (i7-11700K, 8 threads) | 217 s per point |
| this, flat reference converged to 10 kHz, (24, 48, 4), CPU | 2.8 h, 60 GB |

Against the paper's solve that its table comes from (cap 128, 13 h), the tree's accurate rung is about 300× faster;
against its cap-32 solve (81 min), about 32×. Different hardware (one GPU against a 16-core CPU node); on a CPU node the tree's accurate rung takes 225 s.

## Flat reference (2026-11-10 to 14, hal: 2× AMD EPYC 7713, 2 TB; rule frame)

One root over the five junction–partner leaves and f5 (`HENSEL3_STAGE=flat`), KrylovKit Lanczos (tol 1e-6 GHz), 32
threads, qubit grids ±5π and coupler grids ±4π (no range warnings). Raw records in `results/hal-flat-*/`; run with
`HENSEL3_SOLVER=lanczos` on the threaded CPU apply (ITensors permutes in place of Base's serial `permutedims!`).

| (kq, kc, kf) | dimension | wall | root matvecs × s | peak RSS | largest change from the previous rung |
|---|---|---|---|---|---|
| (16, 32, 4) | 1.7e7 | 32 min | 338 × 5.5 | 12 GB | – |
| (24, 48, 4) | 1.3e8 | 2.8 h | 427 × 23 | 60 GB | 244 kHz |
| (32, 64, 4) | 5.4e8 | 14.2 h | 433 × 118 | 260 GB | 2.8 kHz |
| (32, 64, 6) | 8.1e8 | 20.6 h | 476 × 155 | 363 GB | < 1 Hz (kf only) |

Converged transitions (GHz) at (32, 64): 0.420734, 0.703767, 0.814692, 1.035187, 1.217428, 1.474016, 1.767461; against
the paper −1.27, −1.23, −1.31, −2.81, −2.57, −2.98, −3.54 MHz.

- **Converged to 10 kHz** at (24, 48, 4), checked by (32, 64, 4), which moves no transition by more than 2.8 kHz. The
  loose f5 at 4 levels is enough: 6 levels at (32, 64) change nothing to 1 Hz (and 4 to 10 levels at (12, 24) nothing to
  1e-3 Hz). Wider grids (couplers ±2π → ±3π → ±4π, qubits ±3π → ±5π) moved nothing by 1e-3 kHz. (40, 80, 4), projected
  at 44 h and 790 GB, was stopped before its first solve.
- **The tree agrees:** the rule frame at (480, 960) is within 5.2 kHz of it, at (480, 480) within 21 kHz. So the 1.2 to
  3.5 MHz offset from the paper is the paper's, most likely its four stiff modes kept at 4 Fock levels.
- **Threads** (16, 32, 4, before the grids were widened): 16 / 32 / 64 / 128 threads take 6.5 / 5.3 / 5.0 / 5.5 s per
  matvec; the apply is memory-bandwidth bound, so 32 was used. LOBPCG on the same rung: 627 matvecs, 47 min, 35 GB
  (Lanczos 33 min, 8 GB).

## This machine on the clean grids (2026-11-17, ewdesktop, commit `c94938d`)

The rule-frame ladder rerun on the current code and the default warning-free grids (qubits ±5π/301, couplers ±4π/241),
one timed repeat after the first call, idle machine (`results/gpu-rule-20261117/`, `results/cpu-rule-20261117/`). No
range warnings; every transition and every deviation from the paper as in the table above to the printed digit; the
(480, 480) rung within 21 kHz and (480, 960) within 6 kHz of the flat reference at (32, 64, 4).

| rung (kM, kM2) | A4500 (s per point) | CPU, i7-11700K, 8 threads (s per point) |
|---|---|---|
| (240, 240) | 30.5 | 42.5 |
| **(480, 480)**, accurate at 1 MHz | **153** | **217** |
| (480, 960), the check | 461 | 603 |

Peak RSS 6.3 GB (GPU run; 6.1 GB on the device) and 7.5 GB (CPU run). The A4500 times equal the 2026-11-12 ones on the
old grids (32 / 153 / 462 s), so the table above and the cost table stand. The desktop's 8 cores match hal's 32 threads
(225 s): the apply is memory-bandwidth bound (see the thread scan above).

## CPU timing of the tree (2026-11-14, hal, 32 threads)

Rule frame ladder, one repeat (`results/hal-ladder-t32/`), same levels as the GPU runs: (240, 240) 59 s, (480, 480)
225 s, (480, 960) 574 s per point; 31 min wall, 7.5 GB peak.

## Alpine GPUs (2026-11-14; `../gpu_cluster/`)

Per point, ladder (240, 240) / accurate (480, 480) / check (480, 960): H200 8.4 / 15.4 / 33.5 s, A100 17.1 / 30.9 /
71.4 s (A4500 32 / 153 / 462 s); same levels, no edge warnings.
