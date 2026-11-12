# Two capacitively coupled zero-pi qubits (Eli's two-qubit gate study, arXiv:2609.16630)

The benchmark's reference implementation is the study's Python pipeline (`two_qubit_gate_for_zero_pi_qubit/ham_data.py`,
scqubits 4.3.1); this case first reproduces its output, then times both.

## Files

- `quantize.py` (circuit-quantization, conda env `cq-z4`): reads the pipeline's parameter file (default
  `data/Two_qubit_data_Sorted_Truc/params_truc_default.yaml`: zero flux, n_g = 0, no disorder), builds its `zp_yml`
  netlist with the package's zero-pi helpers (`test/fixtures/generate_specs.py`), applies its `Ztransform_2zeropi`, and
  writes `specs/coupled_zero_pi.json` (modes theta1, phi1, zeta1, theta2, phi2, zeta2; loop fluxes and theta offset
  charges symbolic). Its E_C and E_L match the pipeline's scqubits Hamiltonian (its summary file) to every printed digit:
  4E_C(θ) 0.352261, 4E_C(φ) 2.130838, 4E_C(ζ) 0.342739, g = 8E_C(θ1, θ2) 0.030962, E_L/2 0.377 GHz.
- `run.jl`: the pipeline's hierarchy in KronFreeCircuits. Each qubit's (θ, φ) leaf: `ChargeBasis(90)` × `PhaseGrid(-6π,
  6π, 300; order=6)` (scqubits' default 7-point stencil), its lowest 300 levels, keeping `connected([1, 3], (θ, n), 1e-5)`
  (the package's port of the pipeline's `trunc_by_thresh`); ζ1, ζ2 one Fock level each (exactly decoupled; the pipeline
  leaves them out); the root's lowest `ZP_NTOTAL` levels (default 1000; the pipeline computes 3000).
- `reference.py`: reduces the pipeline's npz to `reference/<name>.json` (eval1, eval2, the kept index sets, g, evals_tot).

## Agreement with the pipeline (2026-11-12, `two_qubit_data_default.npz`)

| quantity | agreement |
|---|---|
| each qubit's 300 levels (E_k − E_0) | ≤ 2.5×10⁻¹¹ GHz; absolute ground energies equal to 9 digits (−7.146053252, −6.195902844) |
| kept sets (connected to 0 and 2 through n_θ > 10⁻⁵) | identical: 154 and 155 indices |
| lowest 1000 coupled levels | ≤ 1.3×10⁻¹¹ GHz (median 6×10⁻¹²) |

## Timing so far (ewdesktop, 8 CPU cores; not yet the benchmark)

658 s per point: each 54 300-state qubit leaf 23 s (`ShiftInvert`, 300 levels), the 23 870-state root 552 s (`Dense`, a
full decomposition for 1000 levels). The root solver is the thing to tune before timing against the pipeline.
