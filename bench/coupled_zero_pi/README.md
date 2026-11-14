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

- `pipeline_timing.py` (conda env `qutip4`): times the pipeline's own stages (`_build_circuit`, `_compute_subsystem1`,
  `_compute_sweep_point`) at `truc_total` = 1000, without its file writes, and checks the levels against the reference.
  (`ham_data.py`'s command line returns after a flux-derivative sweep, before the coupled solve, so it is not used.)

## Timing (2026-11-14, ewdesktop: i7-11700K, 8 cores / 16 threads, RTX A4500, otherwise idle; 1000 coupled levels)

Five timed repeats each. KronFree: `BENCH_REPEATS=5 ZP_ROOT=lobpcg julia --project=bench -t 8 --gcthreads=1 run.jl`
(LOBPCG root, tol 1e-8, 8 guard columns), after one untimed first call; `BENCH_DEVICE=gpu` for the GPU, `ZP_LEAF=device`
for GPU leaves (`Auto(on=CuArray)`, i.e. `ChebyshevFilter`). Pipeline: `pipeline_timing.py 1000 5`, numpy on all 16 hardware threads.

| | mean ± std (s) | first call (s) | leaves (s each) | root assemble + solve (s) | peak memory (GB) |
|---|---|---|---|---|---|
| Python pipeline (scqubits 4.3.1) | **3303 ± 117** | – | – | – | – |
| KronFree, CPU (8 threads) | **80.9 ± 0.6** | 123.6 | `ShiftInvert` 21.8, 22.3 | 5.2 + 31.0 | 8.6 |
| KronFree, GPU root, CPU leaves | **102.2 ± 0.9** | 158.3 | `ShiftInvert` 34.1, 33.8 | 10.8 + 21.7 | 7.5 host, 6.6 GPU |
| KronFree, all on the GPU | **42.9 ± 0.4** | 92.2 | `ChebyshevFilter` 4.8, 4.6 | 10.4 + 22.2 | 6.7 host, 6.9 GPU |

- Pipeline stages: build (scqubits `Circuit`, hierarchy, g) 1081 ± 2 s; qubit 1 0.7 s (its eigensystem apparently
  already computed during the build, so only the total is quoted); qubit 2 + coupled `eigsh` 2222 ± 117 s (runs
  alternate between about 2350 and 2130 s).
- KronFree is 41× (CPU) and 77× (GPU) faster. Every run reproduces the reference: KronFree ≤ 1.3×10⁻¹¹ GHz over 1000
  levels with identical kept sets, the pipeline ≤ 4.5×10⁻¹² GHz. No grid-edge warnings; each KronFree run prints 10
  "heavy hybridization" phase-fixing warnings at the root (sign conventions only).
- In GPU mode the harness sets BLAS to 1 thread, so CPU leaves slow from 22 s to 34 s; that costs more than the GPU
  saves at the root. The root's LOBPCG: 23 870 states, 134–140 iterations, about 6 930 matvecs.
- The earlier `Dense` root (2026-11-12, loaded machine) took 552 s of a 658 s point.
- CUDA test tier (`test/cuda`) the same night: 595/595.

Results: `results/overnight_{cpu,gpu_root,gpu_all}/`, `results/pipeline_timing_1000.json` (5 runs),
`results/pipeline_timing_1000_single.json` (one run on 2026-11-13), logs `results/overnight_*.log`.
