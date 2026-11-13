"""scqubits (4.3.1, conda env scqubits-bench) solves of the two autodiff circuits for the solve-time comparison with
KronFreeCircuits, in KronFree's coordinates and truncation: the Hamiltonian is read from the same circuit-quantization
spec (H = 4 nᵀ E_C n + ½ φᵀ E_L φ + φᵀ flux_linear − Σ E cos(ωᵀφ + offset)) and handed to scqubits' Circuit as a
symbolic Hamiltonian (Q_i, θ_i for extended modes), with KronFree's bases:
  fluxonium: varphi on a 513-point grid over ±16π, 9-point stencil (KronFree's PhaseGrid order 8), 6 levels;
  c66: hierarchical, varphi1 harmonic (cutoff 60, oscillator length (8 E_C/E_L)^(1/4), KronFree's FockBasis(60))
       truncated to 40, varphi2 on a 1153-point grid over ±24π truncated to 60, the root's 6 levels.
Timed two ways, 10 trials after 3 warm-ups: building the Circuit from the symbolic Hamiltonian and diagonalizing, and
setting new parameter values on a built Circuit (every coefficient is a symbolic parameter) and diagonalizing, which
is the step that matches KronFree's materialize.

    ~/anaconda3/envs/scqubits-bench/bin/python bench/scqubits_solve/scq_solve.py fluxonium|c66 [trials]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scqubits as scq
import sympy as sm

HERE = Path(__file__).resolve().parent
CASES = {
    "fluxonium": dict(spec=HERE.parent / "autodiff_fluxonium" / "specs" / "fluxonium_sqcircuit.json",
                      p0={"E_C": 2.63, "E_J": 2.85, "E_L": 0.051, "varphi_ext": 0.0},
                      kronfree=HERE.parent / "autodiff_fluxonium" / "results" / "kronfree.json"),
    "c66": dict(spec=HERE.parent / "autodiff_c66" / "specs" / "c66_sqcircuit.json",
                p0={"E_J": 28.83436338, "E_L13": 0.10343485, "E_C23": 1.36334654, "E_L23": 0.05167763, "varphi_ext": 0.0},
                kronfree=HERE.parent / "autodiff_c66" / "results" / "kronfree.json"),
}


def julia_expr(s):
    return sm.sympify(s.replace("//", "/").replace("./", "/").replace(".*", "*").replace(".^", "**"))


def coefficients(spec, p):
    """The spec's numeric coefficients at parameters p: E_C, E_L (matrices), flux_linear, cosine energies."""
    def ev(s):
        e = julia_expr(s)
        return float(e.subs({x: p[x.name] for x in e.free_symbols}))
    assert all(md["type"] in ("e", "h") for md in spec["modes"]), spec["modes"]
    assert all(ev(g) == 0 for g in spec["n_g"])
    m = len(spec["modes"])
    vals = {}
    for i in range(m):
        for j in range(i, m):
            vals[f"EC{i + 1}{j + 1}"] = ev(spec["E_C"][i][j])
            vals[f"EL{i + 1}{j + 1}"] = ev(spec["E_L"][i][j])
        vals[f"FL{i + 1}"] = ev(spec["flux_linear"][i])
    for k, c in enumerate(spec["cosines"]):
        vals[f"EJ{k + 1}"] = ev(c["energy"])
    return vals


def hamiltonian(spec, p):
    """The spec's Hamiltonian in scqubits' symbols Q_i, θ_i (i from 1), every coefficient a symbolic parameter
    (EC_ij, EL_ij, FL_i, EJ_k) so that a solve at new parameters only sets their values; with their values at p."""
    m = len(spec["modes"])
    Q = sm.symbols(f"Q1:{m + 1}"); th = sm.symbols(f"θ1:{m + 1}")
    vals = coefficients(spec, p)
    S = {k: sm.Symbol(k) for k in vals}
    H = 0
    for i in range(m):
        for j in range(i, m):
            f = 1 if i == j else 2
            if vals[f"EC{i + 1}{j + 1}"] != 0:
                H += 4 * f * S[f"EC{i + 1}{j + 1}"] * Q[i] * Q[j]
            if vals[f"EL{i + 1}{j + 1}"] != 0:
                H += sm.Rational(f, 2) * S[f"EL{i + 1}{j + 1}"] * th[i] * th[j]
        if vals[f"FL{i + 1}"] != 0:
            H += S[f"FL{i + 1}"] * th[i]
    for k, c in enumerate(spec["cosines"]):
        assert julia_expr(c["offset"]) == 0
        H -= S[f"EJ{k + 1}"] * sm.cos(sum(w * th[i] for i, w in enumerate(c["omega"])))
    return H, vals


def build(case, spec, p):
    H, vals = hamiltonian(spec, p)
    cr = scq.Circuit(symbolic_hamiltonian=H, symbolic_param_dict=vals, ext_basis="discretized", truncated_dim=6)
    if case == "fluxonium":
        cr.cutoff_ext_1 = 513
        cr.set_discretized_phi_range((1,), (-16 * np.pi, 16 * np.pi))
    else:
        cr.cutoff_ext_1 = 60
        cr.cutoff_ext_2 = 1153
        cr.set_discretized_phi_range((2,), (-24 * np.pi, 24 * np.pi))
        cr.configure(system_hierarchy=[[1], [2]], subsystem_trunc_dims=[40, 60], ext_basis=["harmonic", "discretized"])
    return cr


def build_and_solve(case, spec, p):
    return build(case, spec, p).eigenvals(evals_count=6)


def update_and_solve(cr, spec, p):
    """New parameter values on a built circuit (every coefficient set), then the diagonalization."""
    for k, v in coefficients(spec, p).items():
        if hasattr(cr, k):
            setattr(cr, k, v)
    return cr.eigenvals(evals_count=6)


def trials(f, n):
    for _ in range(3):
        f()
    ts = []
    for _ in range(n):
        t = time.time(); E = f(); ts.append(time.time() - t)
    return E, ts


if __name__ == "__main__":
    scq.settings.STENCIL = 9                   # KronFree's PhaseGrid order 8: the same 9-point coefficients
    case = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    c = CASES[case]
    spec = json.loads(c["spec"].read_text())
    H, _ = hamiltonian(spec, c["p0"])
    print(f"scqubits {scq.__version__}; {case}; H =", H, flush=True)
    E, t_build = trials(lambda: build_and_solve(case, spec, c["p0"]), n)
    cr = build(case, spec, c["p0"])
    # alternate between two parameter points so that no cached diagonalization can be reused
    pts = [c["p0"], {k: (v * (1 + 1e-6) if v else v) for k, v in c["p0"].items()}]
    k = [0]
    def step():
        k[0] += 1
        return update_and_solve(cr, spec, pts[k[0] % 2])
    _, t_upd = trials(step, n)
    E = update_and_solve(cr, spec, c["p0"])
    kf = json.loads(c["kronfree"].read_text())["values"]
    w01, a = E[1] - E[0], E[2] - 2 * E[1] + E[0]
    print(f"levels (GHz, from E0): {np.round(E - E[0], 9)}")
    print(f"omega01 {w01:.9f} (KronFree {kf[0]:.9f}, diff {w01 - kf[0]:.1e}); alpha {a:.9f} (KronFree {kf[1]:.9f}, diff {a - kf[1]:.1e})")
    for name, ts in (("build circuit + diagonalize", t_build), ("set parameters + diagonalize", t_upd)):
        print(f"{name}: {np.mean(ts):.4f} ± {np.std(ts, ddof=1):.4f} s ({n} trials after 3 warm-ups; "
              + " ".join(f"{1e3 * x:.0f}" for x in ts) + " ms)")
    out = HERE / "results" / f"scqubits_{case}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"scqubits": scq.__version__, "case": case, "levels": list(map(float, E)),
                               "omega01": float(w01), "alpha": float(a), "kronfree_omega01": kf[0], "kronfree_alpha": kf[1],
                               "build_and_solve_seconds_mean_std": [float(np.mean(t_build)), float(np.std(t_build, ddof=1))],
                               "build_and_solve_seconds_runs": t_build,
                               "update_and_solve_seconds_mean_std": [float(np.mean(t_upd)), float(np.std(t_upd, ddof=1))],
                               "update_and_solve_seconds_runs": t_upd}, indent=1) + "\n")
    print("wrote", out)
