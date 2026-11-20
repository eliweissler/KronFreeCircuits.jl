"""Di Paolo et al.'s Fig. 3 (arXiv:1912.01018, source file plot_vlad_v2.pdf) read from its vector paths: the full-DMRG
charge dispersion (bottom panel, light-blue circles) and their single-mode Eq. 6 (bottom panel, black dashed), and omega01 at Phi_ext = Phi0/2 of every top-panel curve
(n_g = 0 to 0.5 in steps of 0.05, light to dark; black dashed: their single-mode model). Axes are calibrated on the
tick marks (bottom: z = 0.04 ... 0.15, 10^-8 ... 10^0 GHz, log; top: Phi_ext/Phi0 = 0.45, 0.50 and 0.25 ... 0.75 GHz).
Needs mutool (MuPDF) on PATH.

    python3 bench/array_fluxonium/digitize_fig3.py [path/to/plot_vlad_v2.pdf]   (writes results/dipaolo_fig3_digitized.json)
"""
import json, subprocess, sys, tempfile, xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
pdf = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "arxiv_sources_kronfree_bench/1912.01018/plot_vlad_v2.pdf")
with tempfile.TemporaryDirectory() as d:
    subprocess.run(["mutool", "draw", "-F", "trace", "-o", f"{d}/t.xml", pdf], check=True, capture_output=True)
    root = ET.parse(f"{d}/t.xml").getroot()


def ap(m, x, y):
    a, b, c, dd, e, f = m
    return (a * x + c * y + e, b * x + dd * y + f)


paths = []
for el in root.iter():
    if el.tag in ("fill_path", "stroke_path"):
        m = [float(v) for v in el.get("transform").split()]
        pts = [ap(m, float(s.get("x")), float(s.get("y"))) if s.tag != "curveto" else ap(m, float(s.get("x3")), float(s.get("y3")))
               for s in el if s.tag in ("moveto", "lineto", "curveto")]
        paths.append((el.get("color"), pts))

# bottom panel: major ticks z = 0.04 at x 44.94, 0.15 at 219.98; 1e-8 at y 154.42, 1e0 at 96.19 (y grows downward)
bottom = []
for col, pts in [p for p in paths if p[0] == ".36 .54 .66"][1:]:          # the first is the legend's marker
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    x, y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    bottom.append({"z": round(0.04 + (x - 44.94) / (219.98 - 44.94) * 0.11, 4),
                   "delta_omega01_GHz": 10 ** (-8 + (154.42 - y) / (154.42 - 96.19) * 8)})
# the bottom panel's one black stroked curve: their single-mode model's Eq. 6
eq6 = [{"z": round(0.04 + (x - 44.94) / (219.98 - 44.94) * 0.11, 4), "delta_omega01_GHz": 10 ** (-8 + (154.42 - y) / (154.42 - 96.19) * 8)}
       for x, y in next(pts for col, pts in paths if col == "0" and len(pts) > 4 and min(q[1] for q in pts) > 80)]
# top panels: Phi0/2 at the right tick of each panel; 0.25 GHz at y 44.97, 0.50 at 34.53
top = []
for k, x0 in enumerate([45.5, 86.21, 126.93, 167.64, 208.36]):
    curves = [p for p in paths if len(p[1]) == 21 and x0 - 2 < min(q[0] for q in p[1]) < x0 + 25]
    w = [0.25 + (44.97 - max(c[1], key=lambda t: t[0])[1]) / (44.97 - 34.53) * 0.25 for c in curves]
    colored = [v for (col, _), v in zip(curves, w) if col != "0"]
    single = [v for (col, _), v in zip(curves, w) if col == "0"]
    top.append({"z": round(0.10 + 0.01 * k, 2), "n_g": [round(0.05 * i, 2) for i in range(len(colored))],
                "omega01_GHz": colored, "single_mode_omega01_GHz": single[0]})
out = HERE / "results" / "dipaolo_fig3_digitized.json"
out.write_text(json.dumps({"source": pdf, "bottom_full_dmrg": bottom, "bottom_single_mode_eq6": eq6, "top_at_half_flux": top}, indent=1) + "\n")
for b in bottom:
    print(f"z {b['z']:.3f}  Δω01 {b['delta_omega01_GHz']:.4e}")
for b in eq6:
    print(f"z {b['z']:.4f}  single-mode Eq. 6 Δω01 {b['delta_omega01_GHz']:.4e}")
for t in top:
    print(f"z {t['z']:.2f}  ω01(n_g=0) {t['omega01_GHz'][0]:.4f}  ω01(½) {t['omega01_GHz'][-1]:.4f}  single mode {t['single_mode_omega01_GHz']:.4f}")
print("wrote", out)
