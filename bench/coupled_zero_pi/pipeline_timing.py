"""Timing of the study's Python pipeline (two_qubit_gate_for_zero_pi_qubit/ham_data.py, scqubits 4.3.1, qutip 4.7; its
environment file names the conda env qutip4) for the comparison with run.jl: the default parameter file with
truc_total = 1000 (the pipeline's 3000 reduced to run.jl's ZP_NTOTAL), each stage of save_two_qubit_data timed as the
pipeline runs it, without the file writes:
  build    _build_circuit: the two-qubit scqubits Circuit, hierarchy and coupling g
  qubit 1  _compute_subsystem1: qubit 1's 300 levels and n_theta1
  root     _compute_sweep_point: qubit 2's 300 levels, the kept (connected) sets, the coupled Hamiltonian, eigsh for
           truc_total levels, dressed quantities
(ham_data.py's command line returns after a flux-derivative sweep before the coupled solve, so it is not used.)
The coupled levels are compared with reference/two_qubit_data_default.json (the pipeline's own 3000-level run).

    ~/anaconda3/envs/qutip4/bin/python bench/coupled_zero_pi/pipeline_timing.py [truc_total] [repeats]
                                                                  (writes results/pipeline_timing_<truc_total>.json)
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPE = Path.home() / "src" / "two_qubit_gate_for_zero_pi_qubit"
sys.path.insert(0, str(PIPE))
os.chdir(PIPE)
import ham_data  # noqa: E402

NTOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
params = ham_data.load_params(str(PIPE / "data" / "Two_qubit_data_Sorted_Truc" / "params_truc_default.yaml"))
params["truc_total"] = NTOTAL
ref = json.loads((HERE / "reference" / "two_qubit_data_default.json").read_text())
runs = []
for r in range(REPEATS):
    with tempfile.TemporaryDirectory() as d:
        summary = str(Path(d, "summary.txt"))
        t0 = time.time(); zp, g = ham_data._build_circuit(params, summary); t1 = time.time()
        s1 = ham_data._compute_subsystem1(zp, params); t2 = time.time()
        sp = ham_data._compute_sweep_point(zp, g, s1, params, summary); t3 = time.time()
    E = np.sort(np.asarray(sp["evals_tot"], dtype=float))
    evt = np.asarray(ref["evals_tot"], dtype=float)
    n = min(len(E), len(evt))
    dev = float(np.max(np.abs((E[:n] - E[0]) - (evt[:n] - evt[0]))))
    run = {"build": t1 - t0, "qubit1": t2 - t1, "root": t3 - t2, "total": t3 - t0, "levels": len(E),
           "worst_dev_vs_reference": dev}
    runs.append(run)
    print(f"run {r + 1}: build {run['build']:.1f} s, qubit 1 {run['qubit1']:.1f} s, qubit 2 + coupled solve {run['root']:.1f} s, "
          f"total {run['total']:.1f} s; {len(E)} levels, worst |Δ(E_k - E_0)| vs the pipeline's reference {dev:.1e} GHz",
          flush=True)
out = HERE / "results" / f"pipeline_timing_{NTOTAL}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"truc_total": NTOTAL, "runs": runs, "cpu_count": os.cpu_count(),
                           "env": sys.executable}, indent=1) + "\n")
print("wrote", out)
