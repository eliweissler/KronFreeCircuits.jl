# Derivatives of the enumeration's top circuit against SQcircuit's autodiff (n3_g1_c66: J, L, CL)

The Jacobian and Hessian of the qubit's quantities with respect to its parameters, at the enumeration optimum, from
KronFreeCircuits' perturbative derivatives and from SQcircuit's PyTorch autodiff (dev-ew branch, conda env
`sqcircuit-bench`). Parameters: E_J, E_L13, E_C23, E_L23 (GHz) and the loop flux f (Φ0). Quantities: ω01, α, and
|⟨0|O|1⟩|² for O = n1, n2, φ1, φ2 of the appendix's modes varphi1 = (φ3 − φ2)/2 and varphi2 = φ2 − φ1.

## Files

- `quantize.py` (conda env `cq-z4`): the circuit in circuit-quantization; `specs/c66_sqcircuit.json` is the model
  sircuitenum hands SQcircuit (a 20 GHz ground capacitor on every node), `specs/c66_appendix.json` the paper's appendix.
- `kronfree.jl`: values, Jacobian and Hessian (`derivatives(sys; order=2)`, all 10 mixed pairs), each checked against
  central differences; writes `results/kronfree.json`. Tree: varphi1 on 60 Fock levels keeping 40, varphi2 on
  PhaseGrid(±24π, 1153, order 8) keeping 60, the 2400-state root by Lanczos keeping 6. No range warning.
- `sq_circuit.py`, `sq_grad.py`: SQcircuit at its enumeration truncation [150, 80]; the autodiff Jacobian, its central
  differences, and a Hessian by central differences of the autodiff Jacobian; writes `results/sqcircuit.json`.
- `sq_neig_sweep.py`: SQcircuit's autodiff Jacobian against the number of computed eigenpairs n_eig;
  writes `results/sqcircuit_neig.json`.
- `root_solvers.jl`: the KronFree tree with each root solver, CPU and GPU.
- `convergence.jl`: the tree's truncations refined one at a time and together (`results/convergence.json`).
- `sq_retime.py`: SQcircuit's one solve and central-difference Jacobian, re-timed (`results/sqcircuit_quiet.json`).
- `sq_solve_trials.py`: SQcircuit's one solve, 10 trials (`results/sqcircuit_solve_trials.json`).
- The single-mode companion is `../autodiff_fluxonium/`.

## Agreement (2026-11-12/13)

- Values: SQcircuit and KronFree agree to 8e-5 relative (SQcircuit's truncation).
- KronFree's Jacobian against its own central differences, and its Hessian against central differences of the
  Jacobian, agree to the differences' precision. SQcircuit's central-difference Jacobian agrees with KronFree's to
  ≤ 4e-4 of each row's largest entry in every row.
- SQcircuit's autodiff gets the energy rows (Hellmann–Feynman) right at any n_eig: ω01 1e-5, α 3e-5. The matrix
  elements' gradients need eigenvector derivatives, which SQcircuit (`Circuit.get_partial_vec`) sums over the computed
  eigenpairs only, so they converge with n_eig (table below). The ω01 and α Hessians agree to 7e-5 and 2e-4.
- |n2_01|² stays 9 % off in SQcircuit's autodiff at every n_eig (its E_J, E_L13 and E_C23 columns), against both
  KronFree and SQcircuit's own central differences; not pursued.

## Convergence of the KronFree tree (2026-11-16, `convergence.jl`)

Each truncation refined in turn against the production tree (Fock 60 keeping 40; grid ±24π, 1153 points, keeping
60); largest change of the values (absolute), and of each Jacobian row and Hessian over its largest entry:

| refinement | values | Jacobian | Hessian |
|---|---|---|---|
| Fock 80, keep 60 | ≤ 9e-14 | ≤ 8e-13 | ≤ 1e-12 |
| grid 1537 points on ±24π (finer) | ≤ 1.4e-9 | ≤ 5e-10 | ≤ 1.4e-9 |
| grid ±32π, 1537 points (wider) | ≤ 2.6e-12 | ≤ 6e-12 | ≤ 5e-12 |
| grid leaf keeps 80 | ≤ 5.1e-8 (ω01 4e-9 GHz) | ≤ 3.7e-7; \|n2_01\|² 1.3e-5 | ≤ 2.6e-6; \|n2_01\|² 1.4e-4 |
| all at once (Fock 80/60, ±32π, 2049 points, keep 80) | same as keep 80 | same | same |

No grid-edge warnings in any variant. The binding truncation is the grid leaf's 60 kept states, and then only for the
|n2_01|² derivatives (the junction charge, whose matrix element couples to high leaf states); ω01 and α are converged
to 4e-9 and 6e-9 GHz with their Jacobians to 4e-8. Every number compared with SQcircuit (8e-5 relative in values,
4e-4 in its central-difference Jacobian) is converged well below those differences.

## Timing (ewdesktop, idle machine, 2026-11-13)

SQcircuit, the autodiff Jacobian (one solve plus the gradients of the six quantities) against n_eig, one run each
(`sq_neig_sweep.py`, `OUT_TAG=_quiet`); deviation from KronFree's Jacobian as the largest entry error over each row's
largest entry, for the rows other than |n2_01|²:

| n_eig | solve + autodiff Jacobian (s) | ω01, α | \|n1\|², \|φ1\|², \|φ2\|² |
|---|---|---|---|
| 4 | 296 | 1e-5, 3e-5 | 0.10, 0.38, 0.41 |
| 8 | 549 | 1e-5, 3e-5 | 1.3e-2, 3.4e-3, 2.3e-2 |
| 16 | 1054 | 1e-5, 3e-5 | 3.1e-4, 8.5e-5, 1.1e-3 |
| 32 | 2063 | 1e-5, 3e-5 | 1.7e-4, 8.6e-5, 5.0e-4 |

n_eig = 32 is converged (64 and 128 changed nothing in the earlier loaded run, at 4032 and 8100 s). SQcircuit's one
solve (n_eig = 4) takes 62.2 ± 1.9 s (mean ± std of 10 trials, `sq_solve_trials.py`), the central-difference
Jacobian 692 s (`sq_retime.py`, one run). Its
Hessian, by central differences of the autodiff Jacobian, was not re-timed: 4154 s on the loaded machine at n_eig = 4
(≈ 10 Jacobians; at a converged n_eig it would be ≈ 10 × 2063 s).

KronFree, mean ± std of 10 trials after a compiling run, 8 threads. `kronfree.jl` (Lanczos root; also after 3
untimed warm-up runs): materialize 0.244 ± 0.059 s, Jacobian (first derivatives, 5 parameters) 0.129 ± 0.044 s,
Jacobian and full Hessian (first and second, 10 mixed pairs) 0.530 ± 0.082 s; the spread is garbage collection
(materialize 204-213 ms in 7 trials, 325-333 ms in 3; the order-2 call alternates 450 and 605 ms). By root solver (`root_solvers.jl`; the Hessian column is the order-2 call,
first derivatives included):

| root solver | materialize (s) | Jacobian (s) | Jacobian and full Hessian (s) | max rel. diff vs Dense |
|---|---|---|---|---|
| Dense (CPU) | 1.563 ± 0.057 | 0.103 ± 0.003 | 0.456 ± 0.075 | — |
| Lanczos (CPU) | 0.235 ± 0.066 | 0.132 ± 0.051 | 0.531 ± 0.093 | 1e-11 |
| Lanczos (GPU) | 0.671 ± 0.073 | 0.301 ± 0.057 | 1.184 ± 0.099 | 9e-12 |
| LOBPCG (CPU) | 0.280 ± 0.078 | 0.149 ± 0.077 | 0.564 ± 0.147 | 5e-10 |
| LOBPCG (GPU) | 0.916 ± 0.166 | 0.316 ± 0.112 | 1.181 ± 0.131 | 5e-10 |

The converged Jacobian takes about 0.37 s with KronFree (solve and Jacobian, Lanczos on the CPU) against 2063 s with
SQcircuit, about 5500 times faster; the exact Hessian with it about 0.77 s. The GPU is slower on this small root
(launch and transfer overhead). The earlier loaded-machine runs (`results/sqcircuit_neig.json`, 427-2131 s for
n_eig 4-32) were 4-30 % slower.

Solve times against scqubits in the same coordinates and truncation: `../scqubits_solve/`.

## GPUs (Alpine 2026-11-14, `../gpu_cluster/`; desktop A4500 2026-11-17)

KronFree with `KF_DEVICE=gpu` (grid leaf LOBPCG and root Lanczos on the GPU, Fock leaf Dense): materialize 1.15 s
(H200), 1.89 s (A100); first and second derivatives 1.41 s and 2.49 s, against 0.24 s and 0.53 s on the desktop CPU.
scqubits torch-cuda (complex root through the real embedding) build + diagonalize 3.5 s (H200), 7.1 s (A100).
A4500 (desktop, 2026-11-17, `results/kronfree_gpu_desktop.json`): materialize 2.28 ± 0.11 s, first and second derivatives
2.13 ± 0.12 s (first alone 0.61 s); Jacobian and Hessian against central differences 1.1e-8 and 5.8e-6, no range warning.
