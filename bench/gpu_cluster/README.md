# KronFree benchmark suite on Alpine GPUs (2026-11-14)

GPU timings of the first five suite cases (bench/README.md, branch bench-harness) on CU Boulder Alpine, run from
/projects/elwe6267/src/KronFreeCircuits.jl (needs commit f604839 or later: the KF_DEVICE / SCQ_SOLVER options).

- `setup_python.sh`: the scqubits + torch env (scqubits-bench pins), once, on an acompile node.
- `case.sh <case> <smoke|h200|a100>`: one case; results in `results/<case>-<mode>-<jobid>/` (run.log, one log per step,
  slurm-job.txt, gpu-monitor.csv). Cases: setup (instantiate + precompile, check torch on the GPU), autodiff
  (KronFree derivatives on the GPU for the fluxonium and c66; scqubits torch.lobpcg on the GPU for both), zeropi
  (1000 levels, LOBPCG root; leaves on the CPU, then on the GPU; 5 repeats), hensel2 (ladder, reference, paper's flat
  space), hensel2_scq (scqubits on the same tree, torch.lobpcg on the GPU, kM 120 and 240), hensel3 (the four-level
  ladder, rule frame).
- DMRG cases (2026-11-16; need bench-harness 54755a4 or later for `specs/array_N40_z0.095.json` and `ESHIFT`):
  `dmrg_setup` instantiates and precompiles `bench/array_fluxonium/dmrg` and `dmrg_gpu` (ITensors, CUDA); `dmrg` times
  the N = 40 array fluxonium at z = 0.095, n_g = 0 at the production settings (D 14, ncut 12, cutoff 1e-12, shift), 13
  sweeps fixed (1 at smoke size), on the GPU and then on the node's CPU with 8 BLAS threads. Desktop: A4500 423 s, i7
  8 threads 1648 s, 1 thread 2184 s (`bench/array_fluxonium/README.md`). `sbatch smoke.slurm dmrg_setup dmrg` first.
- `smoke.slurm [cases]`: smoke sizes on a 20 GB A100 slice (gpu-testing, 1 h).
- `h200.slurm <case>`: full size on one H200 (gpu-normal).
- `a100.slurm <case>`: full size on one 80 GB A100 (gpu-normal).

SQcircuit has no GPU path (its torch mode wraps scipy's CPU eigs and qutip), so it is not run here.

## Results (`results/<case>-<gpu>-<jobid>/`)

Nodes: H200 NVL (141 GB) with an AMD EPYC 9555; A100 80 GB PCIe with an AMD EPYC 7502; 8 CPU cores per job, Julia
1.13.2 (`-t 8`, BLAS 1 thread in GPU mode), torch 2.15.0+cu132. All jobs ran commit f604839; the Alpine checkout had one
uncommitted file (not identified). Smoke runs (20 GB A100 slice) are not kept. The desktop columns are the local
results of each case (i7-11700K, 8 cores; RTX A4500), from the case READMEs.

| Case | Quantity | H200 | A100 | Desktop A4500 | Desktop CPU |
|---|---|---|---|---|---|
| coupled_zero_pi, 1000 levels (5 repeats) | leaves and root on the GPU | **8.4 ± 0.2 s** | 19.5 ± 0.4 s | 42.9 ± 0.4 s | — |
| | root on the GPU, leaves on the CPU | 41.6 ± 0.4 s | 90.2 ± 0.5 s | 102.2 ± 0.9 s | — |
| | KronFree CPU / the study's Python pipeline | — | — | — | 80.9 ± 0.6 s / 3303 ± 117 s |
| hensel_three_fluxonia, per point | ladder (240, 240) | **8.4 s** | 17.1 s | 32 s | — |
| | accurate (480, 480) | **15.4 s** | 30.9 s | 153 s | — |
| | check (480, 960) | **33.5 s** | 71.4 s | 462 s | — |
| hensel_two_fluxonia, per point | ladder kM 120 (1 MHz rung) | **0.53 s** | 0.83 s | 2.56 s | — |
| | ladder kM 240 (10 kHz rung) | **0.64 s** | 1.00 s | 3.38 s | — |
| | k1 60/30, kM 480 (largest rung) | **1.94 s** | 4.09 s | 17.5 s | — |
| | flat reference (1.47 M states) | 1.0-1.4 s | 1.8-2.5 s | — | — |
| | paper's flat space, m = 14 (7.5 M states) | 6.6-14.1 s | 13-26 s | — | — |
| hensel_two_fluxonia scqubits (torch-cuda), 6 points | kM 120 | **139 s** | 786 s | — | — |
| | kM 240 | **284 s** | 985 s | — | — |
| autodiff_fluxonium (10 trials) | materialize | 0.20 ± 0.04 s | 0.37 ± 0.12 s | 0.43 s | **0.030 ± 0.004 s** |
| | first and second derivatives | 0.48 ± 0.14 s | 0.94 ± 0.25 s | — | **0.027 ± 0.004 s** |
| autodiff_c66 (10 trials) | materialize | 1.15 ± 0.08 s | 1.89 ± 0.13 s | 2.3 s | **0.24 ± 0.06 s** |
| | first and second derivatives (5 params) | 1.41 ± 0.22 s | 2.49 ± 0.22 s | — | **0.53 ± 0.08 s** |
| scqubits_solve, build + diagonalize (10 trials) | fluxonium, torch-cuda / default CPU | 0.64 ± 0.00 s | 1.42 ± 0.00 s | — | **0.17 ± 0.01 s** |
| | c66, torch-cuda / default CPU | 3.5 ± 0.8 s | 7.1 ± 1.9 s | — | **0.99 ± 0.03 s** |
| array_fluxonium DMRG, N = 40, z = 0.095 (13 sweeps) | GPU / node CPU, 8 BLAS threads | **202 s** / 1147 s | 320 s / 2842 s | 423 s | 1648 s (8 threads), 2184 s (1) |

- Agreement: zero-pi's 1000 levels match the pipeline's reference to 1.26e-11 GHz on both GPUs; the Hensel ladders give
  the same rung-to-rung changes (1.70e-4, 4.65e-7, 3.44e-6 GHz) and E1 - E0 on every machine; three fluxonia
  E1 - E0 = 0.420738 GHz on the accurate rung, 1.3-3.5 MHz from the paper as locally. The autodiff Jacobians and
  Hessians match central differences to about 1e-8 and 1e-6 (as on the CPU); scqubits torch-cuda matches KronFree to
  about 1e-12 in omega01 and alpha. The two scqubits Hensel runs agree to 1e-12 GHz except at Phi_c = 0.5, kM 120
  (2.6e-8 GHz), where the scipy LOBPCG fallback for the complex point warned it missed its 1e-6 tolerance.
- The GPU wins once the root is large (zero-pi, the Hensel cases); the autodiff and scqubits_solve problems are too
  small, and the desktop CPU beats every GPU there (launch and transfer overhead).
- hensel2_scq: the torch-cuda solves take 2-7 s on both GPUs; the per-point time is the dense CPU pair nodes and the
  scipy fallback for the complex points (Phi_c = 0, 0.5). The A100 node's CPU was slow (a dense n = 400 solve 22 s
  there, 0.1 s on the H200 node), so its numbers here are pessimistic.
- Edge warnings: none, except the default ±3π/181 grid in hensel2 (states 11-24 on q1, q2; both GPUs) and the ±5π/301
  grid on the A100 (the k1 60 rungs). The H200 job also ran the ladder on ±7π/421 (`hensel2_grid7.log`): no warnings,
  same rung changes. Ladder times in the table: H200 and A4500 on ±7π/421, A100 on ±5π/301 (`hensel2_grid5.log`).
- array_fluxonium DMRG (2026-11-16, jobs 35781815 (H200) and 35781817 (A100); `results/dmrg-*`): ITensors with the MPS and MPO on
  the GPU, production settings (D 14, ncut 12, cutoff 1e-12, shift), sweeps fixed. All five runs (two GPUs, two node
  CPUs, the desktop) give the same E0, E1, variances and bonds to every printed digit. The GPU gain is modest (bond 35,
  site dimension 14: small contractions); the H200 is 2.1× the A4500. Node CPUs: EPYC 9555 (H200) 1.4× the desktop i7,
  EPYC 7502 (A100) 1.7× slower. FP64 emulation on the A4500 (cuBLAS eager) was over 15× slower for 3 sweeps (stopped).
- First calls (compile included) are not in the table: e.g. zero-pi 52 s / 92 s (H200), the autodiff fluxonium step
  523 s of wall time on the A100.
