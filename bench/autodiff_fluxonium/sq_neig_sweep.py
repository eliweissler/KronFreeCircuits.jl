"""SQcircuit's autodiff Jacobian of the fluxonium quantities against the number of eigenpairs `diag` computes (n_eig).
SQcircuit's eigenvector gradient (Circuit.get_partial_vec) is a sum over states that runs over the computed eigenpairs
only, so the matrix elements' gradients converge with n_eig; the energies' do not depend on it. Each point's Jacobian
is compared with the central differences of SQcircuit's own values (results/sqcircuit_neig4.json, same truncation;
they do not depend on n_eig) and with KronFree's (results/kronfree.json), each entry against its row's largest entry.

    ~/anaconda3/envs/sqcircuit-bench/bin/python bench/autodiff_fluxonium/sq_neig_sweep.py [smoke] [n_eig ...]
                                                            (writes results/sqcircuit_neig[_smoke].json after every point)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sq_grad  # noqa: E402

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
SMOKE = bool(args) and args[0] == "smoke"
if SMOKE:
    args = args[1:]
    sq_grad.TR = [40]
NS = [int(a) for a in args] or ([4, 8, 16] if SMOKE else [4, 8, 16, 32, 64])   # n_eig < truncation 120
NTRIAL = 10

ref = {}
if not SMOKE:
    s = json.loads((HERE / "results" / "sqcircuit_neig4.json").read_text())
    k = json.loads((HERE / "results" / "kronfree.json").read_text())
    ref["sq_fd"] = np.array(s["jacobian_fd"])
    ref["kronfree"] = np.array(k["jacobian"]).reshape(len(sq_grad.QNAMES), len(sq_grad.PARAMS), order="F")     # JSON3 writes column-major


def rowdev(A, B):
    return np.max(np.abs(A - B) / np.max(np.abs(B), axis=1, keepdims=True), axis=1)


import os
out = HERE / "results" / ("sqcircuit_neig_smoke.json" if SMOKE else f"sqcircuit_neig{os.environ.get('OUT_TAG', '')}.json")
points = []
print(f"truncation {sq_grad.TR}; n_eig {NS}", flush=True)
for n in NS:
    sq_grad.N_EIG = n
    ts = []
    for _ in range(NTRIAL):
        t = time.time(); v, J = sq_grad.jacobian(sq_grad.P0); ts.append(time.time() - t)
    dt = float(np.mean(ts))
    p = {"n_eig": n, "seconds_solve_and_jacobian": dt, "seconds_std": float(np.std(ts, ddof=1)), "seconds_runs": ts, "values": v.tolist(), "jacobian": J.tolist()}
    for name, R in ref.items():
        p["row_deviation_vs_" + name] = rowdev(J, R).tolist()
    points.append(p)
    print(f"n_eig {n:4d}: {dt:.4f} ± {p['seconds_std']:.4f} s ({NTRIAL} trials)   " + "   ".join(
        f"vs {name}: " + " ".join(f"{x:.1e}" for x in p['row_deviation_vs_' + name]) for name in ref), flush=True)
    out.write_text(json.dumps({"truncation": sq_grad.TR, "quantities": sq_grad.QNAMES, "params": sq_grad.PARAMS,
                               "points": points}, indent=1) + "\n")
if SMOKE:
    last = np.array(points[-1]["jacobian"])
    for p in points:
        print(f"smoke n_eig {p['n_eig']}: rows vs n_eig {NS[-1]}:", " ".join(f"{x:.1e}" for x in rowdev(np.array(p['jacobian']), last)))
print("wrote", out)
