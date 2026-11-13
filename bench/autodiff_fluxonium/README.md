# Derivatives of the enumeration's optimized fluxonium against SQcircuit's autodiff (single mode)

The single-mode companion of `../autodiff_c66/` (two modes): the Jacobian and Hessian of the qubit's quantities at the
enumeration paper's optimized fluxonium, from KronFreeCircuits' perturbative derivatives and from SQcircuit's PyTorch
autodiff (dev-ew branch, conda env `sqcircuit-bench`). The circuit is `enumeration_figures/make_figs.py`
`integer_fluxonium`'s `param_set_us`: one edge with C, J, L, E_C = 2.63, E_J = 2.85, E_L = 0.051 GHz, loop flux 0, no
junction capacitance (cj = 0), sircuitenum's explicit ground (a 20 GHz capacitor on each node), SQcircuit truncation
120. Parameters: E_C, E_J, E_L (GHz) and the loop flux f (Φ0). Quantities: ω01, α, and |⟨0|O|1⟩|² for O = n, φ of the
mode varphi = φ2 − φ1.

## Files

- `quantize.py` (conda env `cq-z4`): the circuit in circuit-quantization, `specs/fluxonium_sqcircuit.json`. The two
  ground capacitors in series give E_C,eff = 40 E_C/(E_C + 40). circuit-quantization's junction branch needs a
  capacitance, so the shunt C sits on the junction branch (the same edge).
- `kronfree.jl`: values, Jacobian and Hessian, each checked against central differences; one node, varphi on
  PhaseGrid(±16π, 513, order 8), Dense, 6 levels (±24π with 769 points gives the same digits; no range warning).
  Writes `results/kronfree.json`.
- `sq_circuit.py`, `sq_grad.py` (`N_EIG=` env, default 4): SQcircuit's autodiff Jacobian, its central differences,
  and a Hessian by central differences of the autodiff Jacobian; writes `results/sqcircuit_neig<N>.json`.
- `sq_neig_sweep.py`: the autodiff Jacobian against n_eig (up to 64: SQcircuit's sparse solver needs n_eig < 119);
  `sq_retime.py`: one solve and the central-difference Jacobian.

## Agreement (2026-11-13)

- Values: SQcircuit and KronFree agree to 4e-6 relative (ω01 1.0681593 against 1.0681599 GHz).
- KronFree's Jacobian matches its central differences to 6e-9 of each row's largest entry, its Hessian the central
  differences of the Jacobian to 1e-6 (the differences' error).
- SQcircuit's autodiff: ω01 and α rows agree with KronFree to 3e-7 at any n_eig, and their Hessians to 2e-5. The
  matrix-element rows converge with n_eig, as in the two-mode case, and here all of them do: at n_eig = 64 every row
  is within 3e-7 of SQcircuit's central differences and 2e-6 of KronFree.

## Timing (ewdesktop, idle machine, 2026-11-13; mean ± std of 10 trials after a compiling run; KronFree also after 3 untimed warm-up runs, which removed a slow first trial)

| | time (s) |
|---|---|
| KronFree materialize | 0.0297 ± 0.0037 |
| KronFree Jacobian (first derivatives, 4 parameters) | 0.0068 ± 0.0038 (8 trials at 5.0-5.1 ms, 2 at 14 ms) |
| KronFree Jacobian and full Hessian (first and second, 6 mixed pairs) | 0.0266 ± 0.0037 |
| SQcircuit one solve (n_eig 4) | 0.0229 ± 0.0010 |
| SQcircuit central-difference Jacobian (8 solves) | 0.193 ± 0.007 |

SQcircuit's solve plus autodiff Jacobian against n_eig, with each row's deviation from KronFree's Jacobian:

| n_eig | time (s) | ω01, α | \|n\|², \|φ\|² |
|---|---|---|---|
| 4 | 0.108 ± 0.035 | 3e-7, 3e-7 | 0.20, 1.0e-2 |
| 8 | 0.166 ± 0.004 | 3e-7, 3e-7 | 3.5e-2, 2.6e-3 |
| 16 | 0.297 ± 0.001 | 3e-7, 3e-7 | 2.1e-3, 1.6e-5 |
| 32 | 0.564 ± 0.003 | 3e-7, 3e-7 | 1.3e-5, 2.4e-6 |
| 64 | 1.069 ± 0.002 | 3e-7, 3e-7 | 3.3e-7, 2.3e-6 |

On one mode a single solve costs SQcircuit about as much as KronFree; the derivatives are where they part: KronFree's
Jacobian 0.007 s against SQcircuit's converged autodiff Jacobian 1.07 s (n_eig 64) or 0.19 s by central differences,
and KronFree's exact Hessian 0.027 s.
