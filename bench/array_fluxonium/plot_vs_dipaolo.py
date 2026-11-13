"""The N = 40 array fluxonium's charge dispersion against Di Paolo et al.'s Fig. 3 (arXiv:1912.01018): our DMRG
(circuit-quantization specs) against their DMRG and their single-mode Eq. 6, both read from the figure's vector PDF
(results/dipaolo_fig3_digitized.json, digitize_fig3.py).

    ~/anaconda3/envs/qutip4/bin/python bench/array_fluxonium/plot_vs_dipaolo.py [sweep-tag]

Reads results/sweep-<tag>-z*.log (latest tag by default). Writes results/dmrg_vs_dipaolo.{png,pdf}.
"""
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results"
logs = sorted(glob.glob(str(R / "sweep-*-z*.log")))
tag = sys.argv[1] if len(sys.argv) > 1 else re.search(r"sweep-(.*)-z", logs[-1]).group(1)
ours = []
for f in sorted(glob.glob(str(R / f"sweep-{tag}-z*.log"))):
    z = float(re.search(r"-z([0-9.]+)\.log", f).group(1))
    w = {float(m.group(1)): float(m.group(2))
         for m in re.finditer(r"n_g = ([0-9.]+),.*?ω01 = ([-0-9.]+)", Path(f).read_text())}
    if 0.0 in w and 0.5 in w:
        ours.append((z, abs(w[0.0] - w[0.5])))
dp = json.loads((R / "dipaolo_fig3_digitized.json").read_text())
theirs = [(p["z"], p["delta_omega01_GHz"]) for p in dp["bottom_full_dmrg"]]
eq6 = [(p["z"], p["delta_omega01_GHz"]) for p in dp["bottom_single_mode_eq6"]]

BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#3d3d3a"
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.semilogy(*zip(*eq6), "--", color=INK, lw=1.5, label="Di Paolo et al., single mode (Eq. 6)")
ax.semilogy(*zip(*theirs), "s", ms=8, mfc="none", mec=ORANGE, mew=1.5, label="Di Paolo et al., DMRG")
ax.semilogy(*zip(*ours), "o", ms=4.5, color=BLUE, label="this work, DMRG")
ax.set_xlabel(r"$z = Z_J / R_Q$")
ax.set_ylabel(r"$|\Delta\omega_{01}|/2\pi$ (GHz)")
ax.grid(True, which="major", color="0.9", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(R / f"dmrg_vs_dipaolo.{ext}", dpi=200)
print("sweep", tag, "z:", [p[0] for p in ours])
print("wrote", R / "dmrg_vs_dipaolo.png")
