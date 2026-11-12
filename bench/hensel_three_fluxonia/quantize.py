"""The three-fluxonium circuit of arXiv:2611.05627v1 (App. A) in circuit-quantization: local basis, shear, matrices,
and the kronfree-spec export. Adapted from bench_archive/hensel_three_fluxonia/quantize_three_fluxonia.py (2026-11-07):
exact loop inductances, and the shear of every frame mode onto every junction set by the rule of
bench/hensel_two_fluxonia/frame_opt/shear_rule.md (or full, none, or the archive's hand-picked fractions).

    ~/anaconda3/envs/cq-z4/bin/python bench/hensel_three_fluxonia/quantize.py --frame rule|full|none|archive

Writes specs/three_fluxonia_<frame>.json and specs/three_fluxonia_<frame>_shear.json (fractions s*, full shear a*, A).

Local basis (as for the two-fluxonium bench): the five junction phase differences x_1..x_5 = (q1, q2, q3, c1, c2);
six frame directions, zero-sum in node space (so that dropping the global phase fixes the total charge),
C⁻¹-orthogonal to the junction rows and to each other (E_C exactly diagonal), rotated to diagonalize the frame block
of E_L; then the frame modes whose couplings to the junctions shrink under the shear φ_f = x_f + a x_J,
a = -E_L[f,f]⁻¹ E_L[f,J], are sheared (the stiff ones).  The resulting Z goes to circuit-quantization's quantize with
every loop flux on its junction (cos(φ_b + θ_b)), and the Hamiltonian is exported with to_spec.

Writes three_fluxonia.json (modes q1, q2, q3, c1, c2, f1..f6; parameters varphi_q1, varphi_q2, varphi_q3, varphi_c1,
varphi_c2) and prints every matrix.
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sym

import circuit_quantization as cq

HERE = Path(__file__).resolve().parent
np.set_printoptions(precision=4, suppress=True, linewidth=170)
fF, nH = 1e-15, 1e-9
# The table prints L_c = 8.17, L_end = 59.3, L_br = 36.8 nH, rounded from exact loop targets: the coupler loop 2 L_c at
# E_L = 10 GHz, both qubit loops (L_end + L_c + L_4, 2 L_4 + 2 L_c + L_br) at E_L = 2.00 GHz. With the exact values the
# normal modes match Table tab:modes to 1.2 MHz (stiff modes to every digit); with the printed ones, to 3.4 MHz.
_PHI0SQ_OVER_H = (6.62607015e-34 / (2 * 1.602176634e-19) / (2 * np.pi))**2 / 6.62607015e-34   # H·Hz
L4 = 14.3 * nH
Lc = _PHI0SQ_OVER_H / 20e9
Lend = _PHI0SQ_OVER_H / 2e9 - Lc - L4
Lbr = _PHI0SQ_OVER_H / 2e9 - 2 * L4 - 2 * Lc

# ---- the circuit (Table tab:params) ---------------------------------------------------------------------------------
branches = [
    cq.Branch("Jq1", 1, 2, C=10 * fF, EJ=3.70),
    cq.Branch("Jq2", 6, 7, C=10 * fF, EJ=3.70),
    cq.Branch("Jq3", 11, 12, C=10 * fF, EJ=3.70),
    cq.Branch("Jc1", 4, 3, EJ=10.0),                 # no capacitance of its own: C_Jc1 stays symbolic, set to 0 on export
    cq.Branch("Jc2", 8, 9, EJ=10.0),
    cq.Branch("L4a", 2, 3, L=L4), cq.Branch("L4b", 5, 6, L=L4), cq.Branch("L4c", 7, 8, L=L4), cq.Branch("L4d", 10, 11, L=L4),
    cq.Branch("Lc1a", 3, 5, L=Lc), cq.Branch("Lc1b", 4, 5, L=Lc), cq.Branch("Lc2a", 8, 10, L=Lc), cq.Branch("Lc2b", 9, 10, L=Lc),
    cq.Branch("Lend1", 1, 5, L=Lend), cq.Branch("Lend3", 12, 9, L=Lend),
    cq.Branch("Lbr", 4, 10, L=Lbr),
]
cground = {**{n: 70 * fF for n in (1, 2, 6, 7, 11, 12)}, **{n: 25 * fF for n in (3, 4, 8, 9, 5, 10)}}
branches += [cq.Branch(f"Cg{n}", n, 0, C=c) for n, c in sorted(cground.items())]
circuit = cq.Circuit(branches, ground=0)
NN = 12
junctions = [("q1", (1, 2)), ("q2", (6, 7)), ("q3", (11, 12)), ("c1", (4, 3)), ("c2", (8, 9))]
EJ = {"q1": 3.70, "q2": 3.70, "q3": 3.70, "c1": 10.0, "c2": 10.0}

# ---- loops: the five physical loops carry fluxes, the eleven capacitive loops zero -------------------------------
flux_syms = {name: sym.Symbol(f"varphi_{name}", real=True) for name, _ in junctions}
path_from_1 = {2: ("Jq1",), 3: ("Jq1", "L4a"), 4: ("Lend1", "Lc1b"), 5: ("Lend1",), 6: ("Lend1", "L4b"),
               7: ("Lend1", "L4b", "Jq2"), 8: ("Lend1", "L4b", "Jq2", "L4c"), 9: ("Lend1", "L4b", "Jq2", "L4c", "Jc2"),
               10: ("Lend1", "L4b", "Jq2", "L4c", "Lc2a"), 11: ("Lend1", "L4b", "Jq2", "L4c", "Lc2a", "L4d"),
               12: ("Lend1", "L4b", "Jq2", "L4c", "Lc2a", "L4d", "Jq3")}
cycles = {
    "q1": ("Jq1", "L4a", "Lc1a", "Lend1"),
    "c1": ("Jc1", "Lc1a", "Lc1b"),
    "q2": ("Jq2", "L4b", "Lc1b", "Lbr", "Lc2a", "L4c"),
    "c2": ("Jc2", "Lc2a", "Lc2b"),
    "q3": ("Jq3", "L4d", "Lc2b", "Lend3"),
    **{f"Cg{n}": ("Cg1", f"Cg{n}", *path) for n, path in path_from_1.items()},
}
fluxes = {**{name: cq.phase_to_weber(s) for name, s in flux_syms.items()}, **{f"Cg{n}": 0 for n in range(2, 13)}}
loops = cq.Loops.from_branch_dict(circuit, cycles, fluxes)
res0 = cq.quantize(circuit, loops)                     # for the Lagrangian matrices and the automatic classification

C_Jc = [v for v in circuit.variables if v.name.startswith("C_Jc")]
E_CHARGE, H_PLANCK = 1.602176634e-19, 6.62607015e-34
CONSTANTS = {"e": E_CHARGE, "h": H_PLANCK * 1e9, "Phi_0": H_PLANCK / (2 * E_CHARGE)}   # e²/h, Phi_0²/h in GHz


def numeric(M):
    M = sym.sympify(M)
    sub = {s: CONSTANTS[s.name] for s in M.free_symbols if s.name in CONSTANTS}
    sub.update({s: 0 for s in M.free_symbols if s.name.startswith("C_Jc")})
    return M.subs(sub)


def N(M):
    return np.round(np.array(sym.N(numeric(sym.Matrix(M)), 10).tolist(), dtype=float), 8)


C = N(sym.Matrix(res0.lag.C_int) * 10**15) * 1e-15              # F
Linv = N(res0.lag.Lstar_int)                                     # 1/H
print("branches:", circuit.branch_names)
print("automatic classification:", res0.coords.types_full)
print("node capacitance matrix (fF):\n", C * 1e15)
print("node inverse-inductance matrix (1/nH):\n", Linv * 1e-9)

# ---- local basis ---------------------------------------------------------------------------------------------------
e = np.eye(NN)
D = np.array([e[i - 1] - e[j - 1] for _, (i, j) in junctions])          # 5 junction rows
Cinv = np.linalg.inv(C)
ones = np.ones(NN)
# frame rows: zero-sum, C⁻¹-orthogonal to the junction rows: null space of [1ᵀ; D C⁻¹]
constraints = np.vstack([ones, D @ Cinv])
_, _, Vt = np.linalg.svd(constraints)
W = Vt[constraints.shape[0]:]                                             # 6 rows spanning the complement
# C⁻¹-orthonormalize the frame rows (Cholesky of their C⁻¹ Gram matrix), then scale them like a junction row
G = W @ Cinv @ W.T
Lg = np.linalg.cholesky(G)
W = np.linalg.solve(Lg, W)                                                 # now W C⁻¹ Wᵀ = I
W = W * np.sqrt(2 / 25e-15)                                                # wᵀ C⁻¹ w = dᵀ C⁻¹ d of a 25 fF node pair: entries O(1)
A = np.vstack([D, W, ones / NN])
Ainv = np.linalg.inv(A)
K = A @ Cinv @ A.T
Lx = Ainv.T @ Linv @ Ainv
assert np.allclose(K[:11, :11] - np.diag(np.diag(K[:11, :11])), 0, atol=1e-6 * K[0, 0]), "E_C not diagonal"
assert np.max(np.abs(Lx[11])) < 1e-9 * np.max(np.abs(Linv)), "global phase in the inductive energy"
# rotate the frame rows to diagonalize the frame block of E_L (keeps K_ff diagonal since W is C⁻¹-orthonormal up to scale)
lam, R = np.linalg.eigh(Lx[5:11, 5:11])
W = R.T @ W
A = np.vstack([D, W, ones / NN])
Ainv = np.linalg.inv(A)
EC_CONST = E_CHARGE**2 / (2 * H_PLANCK) * 1e-9                             # GHz·F
EL_CONST = (H_PLANCK / (2 * E_CHARGE) / (2 * np.pi))**2 / H_PLANCK * 1e-9  # GHz·H
EC = EC_CONST * (A @ Cinv @ A.T)[:11, :11]
EL = EL_CONST * (Ainv.T @ Linv @ Ainv)[:11, :11]
names = ["q1", "q2", "q3", "c1", "c2"] + [f"f{i}" for i in range(1, 7)]


def coupling_tables(EC, EL):
    s = (8 * np.diag(EC) / np.diag(EL)) ** 0.25 / np.sqrt(2)
    gL = EL * np.outer(s, s)
    gC = EC / np.outer(s, s)
    np.fill_diagonal(gL, 0)
    np.fill_diagonal(gC, 0)
    return s, gL, gC


s, gL, gC = coupling_tables(EC, EL)
print("\n== local basis, unsheared ==")
print("modes:", names)
print("E_C diag (GHz):", np.round(np.diag(EC), 4))
print("E_L (GHz):\n", EL)
print("harmonic frequencies sqrt(8 E_C E_L) (GHz):", np.round(np.sqrt(8 * np.diag(EC) * np.diag(EL)), 3))
print("phi-phi couplings at the zero-point scale (GHz):\n", gL)

# ---- normal modes against the paper (Table tab:modes) ---------------------------------------------------------------
Ltot = Linv.copy()
for name, (i, j) in junctions:
    g = EJ[name] * 1e9 / _PHI0SQ_OVER_H                              # 1/L_J, junction linearized about zero phase
    for a, b, s in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
        Ltot[a - 1, b - 1] += s * g
import scipy.linalg as sla
w = np.sort(np.sqrt(np.maximum(sla.eigh(Ltot, C, eigvals_only=True), 0)) / (2 * np.pi) / 1e9)[1:]
PAPER_OMEGA = [2.267, 3.684, 4.270, 4.934, 5.235, 8.380, 9.151, 17.173, 17.350, 21.166, 21.558]
print("normal modes (GHz):", np.round(w, 3), " max deviation from the paper", np.max(np.abs(w - PAPER_OMEGA)) * 1e3, "MHz")
assert np.max(np.abs(w - PAPER_OMEGA)) < 1.5e-3

# ---- shear: x = S y, S = [I 0; A I], A_fJ = s*_fJ a*_fJ (shear_rule.md) ---------------------------------------------
FRAME = sys.argv[sys.argv.index("--frame") + 1] if "--frame" in sys.argv else "rule"
THETA0 = {"q1": np.pi, "q2": np.pi, "q3": np.pi, "c1": 0.0, "c2": 0.0}
jn = ["q1", "q2", "q3", "c1", "c2"]
astar = np.array([[-EL[5 + f, J] / EL[5 + f, 5 + f] if abs(EL[5 + f, J]) > 1e-9 * np.max(np.abs(EL)) else 0.0
                   for J in range(5)] for f in range(6)])


def sheared(A):
    S = np.eye(11)
    S[5:, :5] = A
    Si = np.linalg.inv(S)
    return S, Si @ EC @ Si.T, S.T @ EL @ S


def leaf_moments(ec, el, ej, th, a=5 * np.pi, n=801):
    """Ground-state <dx^2>, <n^2> of 4 E_C n^2 + 1/2 E_L x^2 - E_J cos(x + th) on a grid (second-order differences)."""
    x = np.linspace(-a, a, n)
    h = x[1] - x[0]
    d, e = np.full(n, 2 / h**2), np.full(n - 1, -1 / h**2)
    lam, vec = sla.eigh_tridiagonal(4 * ec * d + 0.5 * el * x**2 - ej * np.cos(x + th), 4 * ec * e, select="i", select_range=(0, 0))
    psi = vec[:, 0] / np.linalg.norm(vec[:, 0])
    mx = np.sum(psi**2 * x)
    X = np.sum(psi**2 * (x - mx) ** 2)
    n2 = psi @ (d * psi + np.concatenate([e * psi[1:], [0]]) + np.concatenate([[0], e * psi[:-1]]))
    return X, n2


s_star = np.zeros((6, 5))
if FRAME == "rule":
    A = np.zeros((6, 5))
    for it in range(3):                      # the leaves' inductances and Omega_f depend on the shear: iterate
        _, ECs, ELs = sheared(A)
        for J in range(5):
            X, n2 = leaf_moments(ECs[J, J], ELs[J, J], EJ[jn[J]], THETA0[jn[J]])
            for f in range(6):
                if astar[f, J] == 0:
                    continue
                Om = np.sqrt(8 * ECs[5 + f, 5 + f] * ELs[5 + f, 5 + f])
                c = 4 * ECs[J, J] / Om
                P = (8 * ECs[J, J] / Om) ** 2 * n2
                s_star[f, J] = (X + c) / (X + 2 * c + P)
        A = s_star * astar
elif FRAME == "full":
    s_star[astar != 0] = 1.0
elif FRAME == "archive":                     # bench_archive's hand-picked fractions, one per frame mode
    s_star = np.where(astar != 0, np.array([0.35, 0.40, 0.60, 0.80, 0.85, 0.95])[:, None], 0.0)
elif FRAME != "none":
    raise SystemExit(f"unknown --frame {FRAME}")
A = s_star * astar
print(f"frame {FRAME}: shear fractions s* (rows f1..f6, columns q1 q2 q3 c1 c2):\n", np.round(s_star, 3))
S, EC2, EL2 = sheared(A)
chosen, ALPHA = [], {}
s2, gL2, gC2 = coupling_tables(EC2, EL2)
print("\n== local basis, sheared ==")
print("E_C (GHz):\n", EC2)
print("E_L (GHz):\n", EL2)
print("harmonic frequencies (GHz):", np.round(np.sqrt(8 * np.diag(EC2) * np.diag(EL2)), 3))
print("phi-phi couplings at the zero-point scale (GHz):\n", gL2)
print("n-n couplings at the zero-point scale (GHz):\n", gC2)

# ---- circuit-quantization with the final Z (φ = Z x), junction flux gauge, export ------------------------------------
Z = Ainv.copy()
Z[:, :11] = Z[:, :11] @ S                                                  # φ = A⁻¹ x_old, x_old = S x_new on the 11 modes
# circuit-quantization infers a column's type from its exact junction and inductive participation, so the float
# round-off of the inverse is removed where the type depends on it: a frame column has equal entries on the two nodes
# of every junction (zero junction phase), and the global-phase column is exactly the ones vector.
for f in range(5, 11):
    for _, (i, j) in junctions:
        Z[i - 1, f] = Z[j - 1, f] = 0.5 * (Z[i - 1, f] + Z[j - 1, f])
# Likewise a junction column has phase exactly 1 on its own junction and 0 on the others, so that every cosine acts on
# its junction mode alone (to_spec writes omega as Z gives it); the midpoint is rounded to 2⁻⁴⁵ so m ± 1/2 is exact.
for k, (_, (i0, j0)) in enumerate(junctions):
    for _, (i, j) in junctions:
        m, d = 0.5 * (Z[i - 1, k] + Z[j - 1, k]), 0.5 * ((i, j) == (i0, j0))
        m = round(m * 2**45) / 2**45 if d else m
        Z[i - 1, k], Z[j - 1, k] = m + d, m - d
Z[:, 11] = 1.0
types = ("e",) * 5 + ("h",) * 6 + ("fr",)
res = cq.quantize(circuit, loops, sym.Matrix(Z), types)


def junction_flux_B(res, rings):
    G, lnames, bnames = res.loops.G, list(res.loops.names), list(res.circuit.branch_names)
    B = sym.Matrix(res.B)
    caps = [i for i, b in enumerate(res.circuit.branches) if b.type == "C"]
    for loop, shares in rings.items():
        r = lnames.index(loop)
        b = sym.zeros(len(bnames), 1)
        for junction, share in shares.items():
            b[bnames.index(junction)] = share
        y = sym.symbols(f"y0:{len(caps)}")
        target = sym.Matrix([1 if k == r else 0 for k in range(G.rows)]) - G * b
        (solution,) = sym.linsolve((G[:, caps], target), y)
        for i, value in zip(caps, [v.subs({u: 0 for u in y}) for v in solution]):
            b[i] = value
        assert G * b == sym.Matrix([1 if k == r else 0 for k in range(G.rows)])
        B[:, r] = b
    return B


rings = {"q1": {"Jq1": 1}, "q2": {"Jq2": 1}, "q3": {"Jq3": 1}, "c1": {"Jc1": 1}, "c2": {"Jc2": 1}, **{f"Cg{n}": {} for n in range(2, 13)}}
res = cq.quantize(circuit, loops, sym.Matrix(Z), types, B=junction_flux_B(res, rings))
print("\n== circuit-quantization result in the sheared local basis ==")
print("types:", res.coords.types)
ECq, ELq = N(res.ham.E_C), N(res.ham.E_L)
print("E_C (GHz) agrees with numpy to", np.max(np.abs(ECq - EC2)), "; E_L to", np.max(np.abs(ELq - EL2)))
print("loop phases:", list(res.ham.varphi_ext))
H = numeric(res.H)
print("H =", sym.N(sym.nsimplify(H, tolerance=1e-7, rational=False), 5))

# to_spec symmetrizes the float round-off of the basis and prints round-off zeros as 0; the coupler junctions' own
# capacitances C_Jc1, C_Jc2 are set to 0, leaving the five loop fluxes as the only parameters.
data = cq.to_spec(res, subs={c: 0 for c in C_Jc}, mode_names=names, zero_rtol=1e-9)
data["_generator"]["script"] = "bench/hensel_three_fluxonia/quantize.py --frame " + FRAME
data["_comment"] = [
    "Three-fluxonium circuit of arXiv:2611.05627v1 (App. A), nodes 1..12 as in the paper, ground capacitances included.",
    "Local basis: junction phases q1, q2, q3, c1, c2 as single modes; frame modes f1..f6 zero-sum and C^-1-orthogonal"
    " (E_C diagonal before the shear), frame block of E_L diagonalized; frame modes sheared onto the junction coordinates"
    " (x = S y, S[f, J] = s*_fJ a*_fJ, a*_fJ = -E_L[f,J]/E_L[f,f]), frame " + FRAME + "; fractions in the _shear.json beside this file.",
    "Every loop flux sits on its junction: cos(phi_b + varphi_b). Working point of the paper: varphi_q* = pi, varphi_c* = 0.",
    "The coupler junctions have no capacitance of their own (C_Jc1, C_Jc2 = 0).",
]
out = HERE / "specs" / f"three_fluxonia_{FRAME}.json"
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("\nwrote", out, "with modes", [m["name"] for m in data["modes"]], "and parameters", data["parameters"])
(HERE / "specs" / f"three_fluxonia_{FRAME}_shear.json").write_text(json.dumps(
    {"frame": FRAME, "s_star": s_star.tolist(), "a_star": astar.tolist(), "A": A.tolist(),
     "rows": [f"f{i}" for i in range(1, 7)], "columns": jn}, indent=1) + "\n")
