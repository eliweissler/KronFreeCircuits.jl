"""Our version of Di Paolo et al.'s Fig. 3 (arXiv:1912.01018): the N = 40 array fluxonium at phi_ext = pi against z.

    ~/anaconda3/envs/qutip4/bin/python bench/array_fluxonium/plot_fig3.py [sweep-tag]

(a) omega_01 from DMRG at n_g = 0 and 1/2 (circuit-quantization specs, full-form offsets) and the single-mode fluxonium;
(b) |Delta omega_01| from DMRG (and the earlier sweep with the on-site-only offset), against the phase-slip formula Eq. 6
    with the single-mode <T> and either the printed epsilon_0 (Eq. 5) or the exact ground-band width of one junction.
Reads results/sweep-<tag>-z*.log (latest tag by default), results/single_mode_N40.json and the old sweep's log.
Writes results/fig3_comparison.png.
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
OLD = HERE.parent.parent / "bench_archive" / "array_fluxonium" / "results" / "dmrg_N40_zsweep.txt"

logs = sorted(glob.glob(str(R / "sweep-*-z*.log")))
tag = sys.argv[1] if len(sys.argv) > 1 else re.search(r"sweep-(.*)-z", logs[-1]).group(1)
dmrg = {}
for f in sorted(glob.glob(str(R / f"sweep-{tag}-z*.log"))):
    z = float(re.search(r"-z([0-9.]+)\.log", f).group(1))
    w = {float(m.group(1)): float(m.group(2))
         for m in re.finditer(r"n_g = ([0-9.]+),.*?ω01 = ([-0-9.]+)", Path(f).read_text())}
    if w:
        dmrg[z] = w
old = {}
if OLD.exists():
    for m in re.finditer(r"\[z=([0-9.]+)\] Δω01 = ([-0-9.e]+)", OLD.read_text()):
        old[float(m.group(1))] = abs(float(m.group(2)))
sm = json.loads((R / "single_mode_N40.json").read_text())["rows"]

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
zs = [r["z"] for r in sm]
a.plot(zs, [r["omega01"] for r in sm], "k--", label="single-mode fluxonium")
for ng, mk, lab in ((0.0, "o", r"DMRG, $n_g = 0$"), (0.5, "s", r"DMRG, $n_g = 1/2$")):
    pts = sorted((z, w[ng]) for z, w in dmrg.items() if ng in w)
    a.plot([p[0] for p in pts], [p[1] for p in pts], mk, label=lab)
a.set_xlabel(r"$z = Z_J/R_Q$"); a.set_ylabel(r"$\omega_{01}/2\pi$ (GHz) at $\Phi_{ext} = \Phi_0/2$")
a.set_title("(a) qubit frequency, $N_J = 40$"); a.legend()

full = sorted((z, abs(w[0.0] - w[0.5])) for z, w in dmrg.items() if 0.0 in w and 0.5 in w)
b.semilogy([p[0] for p in full], [p[1] for p in full], "o", ms=7, label="DMRG (full-form $n_g$, cq spec)")
if old:
    b.semilogy(sorted(old), [old[z] for z in sorted(old)], "x", color="0.6", label="DMRG, on-site $n_g$ (earlier)")
b.semilogy(zs, [r["delta_omega01_eq6_exact"] for r in sm], "k-", label=r"Eq. 6, single mode, exact $\epsilon_0$")
b.semilogy(zs, [r["delta_omega01_eq6_printed"] for r in sm], "k--", label=r"Eq. 6, single mode, printed $\epsilon_0$")
b.set_xlabel(r"$z = Z_J/R_Q$"); b.set_ylabel(r"$|\Delta\omega_{01}|/2\pi$ (GHz)")
b.set_title("(b) charge dispersion"); b.legend(fontsize=8)
fig.tight_layout()
out = R / "fig3_comparison.png"
fig.savefig(out, dpi=150)
print("sweep", tag, "points with both n_g:", [p[0] for p in full])
print("wrote", out)
