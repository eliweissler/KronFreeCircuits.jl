"""Reduce the two-qubit pipeline's npz to the arrays the benchmark compares against.

    ~/anaconda3/envs/cq-z4/bin/python bench/coupled_zero_pi/reference.py [path/to/two_qubit_data_default.npz]

Writes bench/coupled_zero_pi/reference/<npz name>.json with eval1, eval2 (each qubit's lowest levels, GHz), the kept
index sets hspace_1_charge, hspace_2_charge (0-based), g_theta1theta2, and evals_tot (the coupled levels, ground
subtracted). Only these members are read: an npz loads members on access, so the eigenvector arrays stay on disk.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "src/two_qubit_gate_for_zero_pi_qubit/data/Two_qubit_data_Sorted_Truc/two_qubit_data_default.npz"

with np.load(NPZ, allow_pickle=False) as z:
    print("members:", z.files)
    out = {
        "source": str(NPZ),
        "eval1": np.real(z["eval1"]).tolist(),
        "eval2": np.real(z["eval2"]).tolist(),
        "hspace_1_charge": [int(i) for i in z["hspace_1_charge"]],
        "hspace_2_charge": [int(i) for i in z["hspace_2_charge"]],
        "g_theta1theta2": complex(z["g_theta1theta2"]).real,
        "evals_tot": np.real(z["evals_tot"]).tolist(),
    }
for k in ("eval1", "eval2", "hspace_1_charge", "hspace_2_charge", "evals_tot"):
    print(f"{k}: {len(out[k])} entries, first {out[k][:4]}")
print("g_theta1theta2:", out["g_theta1theta2"])
dst = HERE / "reference" / (NPZ.stem + ".json")
dst.parent.mkdir(exist_ok=True)
dst.write_text(json.dumps(out) + "\n")
print("wrote", dst)
