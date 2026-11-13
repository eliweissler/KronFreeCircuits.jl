"""SQcircuit's one solve of the c66 circuit (n_eig 4, values only, truncation [150, 80]): mean and standard deviation
of 10 trials after a warm-up solve, on an idle machine.

    ~/anaconda3/envs/sqcircuit-bench/bin/python bench/autodiff_c66/sq_solve_trials.py   (writes results/sqcircuit_solve_trials.json)
"""
import json
import time
from pathlib import Path

import numpy as np

import sq_grad

N = 10
sq_grad.values(sq_grad.P0)                     # warm-up
ts = []
for i in range(N):
    t = time.time(); sq_grad.values(sq_grad.P0); ts.append(time.time() - t)
    print(f"trial {i + 1}: {ts[-1]:.2f} s", flush=True)
m, s = float(np.mean(ts)), float(np.std(ts, ddof=1))
print(f"one solve: {m:.2f} ± {s:.2f} s ({N} trials)")
out = Path(__file__).resolve().parent / "results" / "sqcircuit_solve_trials.json"
out.write_text(json.dumps({"truncation": sq_grad.TR, "n_eig": sq_grad.N_EIG, "seconds_mean": m, "seconds_std": s,
                           "seconds_runs": ts}, indent=1) + "\n")
print("wrote", out)
