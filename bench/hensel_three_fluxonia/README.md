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

Against the paper's solve that its table comes from (cap 128, 13 h), the tree's accurate rung is about 300× faster;
against its cap-32 solve (81 min), about 32×. Different hardware (one GPU against a 16-core CPU node); the CPU timing of
the tree is pending.

## Pending

- **Flat reference** (large-memory server, 2 TB, 128 cores): `HENSEL3_STAGE=flat` at two consecutive rungs (likely
  kq, kc, kf = 32, 64, 4 and 40, 80, 4: 5.4e8 and 1.6e9 states, real arithmetic), after smoke runs that measure memory
  and time per state on that machine.
- **CPU timing** of the tree ladder.
