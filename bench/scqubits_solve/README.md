# Solve times against scqubits, same coordinates and truncation (the two autodiff circuits)

The solve-time half of the autodiff comparisons (`../autodiff_fluxonium/`, one mode; `../autodiff_c66/`, two modes)
against scqubits 4.3.1 (conda env `scqubits-bench`; numpy on OpenBLAS 0.3.29, all 16 threads). `scq_solve.py` reads
the same circuit-quantization spec as KronFree and hands scqubits' `Circuit` its Hamiltonian,
4 nᵀ E_C n + ½ φᵀ E_L φ + φᵀ flux_linear − Σ E cos(ωᵀφ + offset), as a symbolic Hamiltonian (every coefficient a
symbolic parameter), in KronFree's coordinates and bases:

- fluxonium: varphi on a 513-point grid over ±16π with the 9-point stencil (`scq.settings.STENCIL = 9`, the
  coefficients of KronFree's PhaseGrid order 8), 6 levels;
- c66: hierarchical diagonalization, varphi1 in the harmonic basis (cutoff 60, oscillator length (8 E_C/E_L)^¼,
  KronFree's FockBasis(60)) truncated to 40, varphi2 on a 1153-point grid over ±24π truncated to 60, 6 levels of the
  root (KronFree's tree; its leaves keep 40 and 60).

Agreement: ω01 and α agree with KronFree to 3e-13 GHz (fluxonium) and 9e-13 GHz (c66): the same discretization.

## Timing (ewdesktop, idle machine, 2026-11-13; mean ± std of 10 trials after 3 warm-ups)

scqubits is timed two ways: building the `Circuit` from the symbolic Hamiltonian and diagonalizing, and setting new
parameter values on a built `Circuit` and diagonalizing (alternating between two parameter points, so nothing cached
is reused). The second matches KronFree's `materialize` (a built tree at new parameters, 8 threads).

| | fluxonium (s) | c66 (s) |
|---|---|---|
| KronFree materialize | 0.0297 ± 0.0037 | 0.244 ± 0.059 (Lanczos root) |
| scqubits, set parameters + diagonalize | 0.151 ± 0.019 | 0.982 ± 0.044 |
| scqubits, build Circuit + diagonalize | 0.173 ± 0.013 | 0.990 ± 0.028 |
| SQcircuit one solve (truncation 120; [150, 80]) | 0.0229 ± 0.0010 | 62.2 ± 1.9 |

KronFree solves 5 times faster than scqubits on one mode and 4 times on two. SQcircuit's solves are in its own
coordinates and truncation (harmonic-oscillator bases, no hierarchy), not the ones above.
