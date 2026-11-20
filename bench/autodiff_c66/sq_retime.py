"""Re-timing of SQcircuit on the c66 circuit on a quiet machine: one solve (n_eig 4, values only, best of N_REP) and the
central-difference Jacobian (2 solves per parameter). The autodiff Jacobians against n_eig are sq_neig_sweep.py's (OUT_TAG=_quiet).

    ~/anaconda3/envs/sqcircuit-bench/bin/python bench/autodiff_c66/sq_retime.py     (writes results/sqcircuit_quiet.json)
"""
import json
import time
from pathlib import Path

import numpy as np

import sq_grad

P0, PARAMS = sq_grad.P0, sq_grad.PARAMS
N_REP = 2
ts = []
for _ in range(N_REP):
    t = time.time(); v0 = sq_grad.values(P0); ts.append(time.time() - t)
print(f"one solve: {min(ts):.1f} s (runs {', '.join(f'{x:.1f}' for x in ts)})", flush=True)
step = {k: (1e-4 if k == "flux" else 1e-4 * P0[k]) for k in PARAMS}
t = time.time()
Jfd = np.zeros((len(sq_grad.QNAMES), len(PARAMS)))
for a, k in enumerate(PARAMS):
    pp = dict(P0); pm = dict(P0); pp[k] += step[k]; pm[k] -= step[k]
    Jfd[:, a] = (sq_grad.values(pp) - sq_grad.values(pm)) / (2 * step[k])
t_fd = time.time() - t
print(f"central-difference Jacobian: {t_fd:.1f} s", flush=True)
out = Path(__file__).resolve().parent / "results" / "sqcircuit_quiet.json"
out.write_text(json.dumps({"truncation": sq_grad.TR, "seconds_solve": min(ts), "seconds_solve_runs": ts,
                           "seconds_fd_jacobian": t_fd, "values": v0.tolist(), "jacobian_fd": Jfd.tolist()}, indent=1) + "\n")
print("wrote", out)
