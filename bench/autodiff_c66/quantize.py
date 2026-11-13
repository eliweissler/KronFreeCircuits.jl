"""The top-performing three-node circuit of the enumeration paper (n3_g1_c66: J, L, CL; enumeration_figures/make_figs.py
n3_g1_c66_*) in circuit-quantization, for the autodiff benchmark against SQcircuit.

    ~/anaconda3/envs/cq-z4/bin/python bench/autodiff_c66/quantize.py

Nodes 1, 2, 3 as in the paper's final appendix (Circuit Quantization Details for Top Performing Circuit; enumeration
nodes 0, 1, 2): the junction (E_J, intrinsic capacitance E_CJ = 10 GHz) between 1 and 2, the inductor L13, and C23 in
parallel with L23. Coordinates: the appendix's phi = Z varphi, varphi_1 = (phi_3 - phi_2)/2, varphi_2 = phi_2 - phi_1,
varphi_3 = phi_1 + phi_2 + phi_3.

Two specs:
- specs/c66_appendix.json: the appendix's model (no ground capacitance, coupling capacitance C_c = 0), every element
  symbolic, checked here against the appendix's C~, L~, cosine and flux term;
- specs/c66_sqcircuit.json: the model sircuitenum hands SQcircuit (ground_node=None: every node gets a 20 GHz capacitor
  to ground), with E_J, E_L13, E_C23, E_L23 and the loop flux varphi_ext as parameters (E_CJ = 10, E_CG = 20 GHz fixed),
  the flux in the irrotational gauge (as SQcircuit's).
"""
import json
from pathlib import Path

import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize

HERE = Path(__file__).resolve().parent
(HERE / "specs").mkdir(exist_ok=True)
pos = lambda s: sym.Symbol(s, positive=True)
EJ, EL13, EC23, EL23 = pos("E_J"), pos("E_L13"), pos("E_C23"), pos("E_L23")
ECJ, ECG = pos("E_CJ"), pos("E_CG")
vx = sym.Symbol("varphi_ext", real=True)
Z = sym.Matrix([[-sym.Rational(2, 3), -sym.Rational(2, 3), sym.Rational(1, 3)],
                [-sym.Rational(2, 3), sym.Rational(1, 3), sym.Rational(1, 3)],
                [sym.Rational(4, 3), sym.Rational(1, 3), sym.Rational(1, 3)]])
# the appendix's varphi in terms of the node phases: Z^-1 rows
assert Z.inv() == sym.Matrix([[0, -sym.Rational(1, 2), sym.Rational(1, 2)], [-1, 1, 0], [1, 1, 1]]), Z.inv()


def branches(ground):
    b = [Branch("J", 1, 2, C=cq.ghz_to_farad(ECJ), EJ=EJ),
         Branch("L13", 1, 3, L=cq.ghz_to_henry(EL13)),
         Branch("C23", 2, 3, C=cq.ghz_to_farad(EC23)),
         Branch("L23", 2, 3, L=cq.ghz_to_henry(EL23))]
    if ground:
        b += [Branch(f"Cg{n}", n, 0, C=cq.ghz_to_farad(ECG)) for n in (1, 2, 3)]
    return b


def quantized(ground):
    circuit = Circuit(branches(ground), ground=0 if ground else None)
    assert list(circuit.internal_nodes) == [1, 2, 3], circuit.internal_nodes
    default = Loops.from_circuit(circuit)
    ring = next(n for n in default.names if {"J", "L13", "L23"} <= set(default.branch_dict[n]))
    flux = {n: 0 for n in default.names}
    flux[ring] = cq.phase_to_weber(vx)
    loops = Loops.from_tree(circuit, cq.default_tree(circuit), flux)
    types = ("h", "e", "fr" if ground else "cy")       # varphi1 has no junction (harmonic); varphi2 carries it
    return quantize(circuit, loops, Z, types)


def num(x):
    return sym.nsimplify(x)


# ---- (a) the appendix's model, against its matrices ------------------------------------------------------------------
res = quantized(ground=False)
data = cq.to_spec(res, mode_names=["varphi1", "varphi2"], zero_rtol=1e-12)
EC = sym.Matrix([[sym.sympify(x.replace("//", "/")) for x in r] for r in data["E_C"]])
EL = sym.Matrix([[sym.sympify(x.replace("//", "/")) for x in r] for r in data["E_L"]])
# appendix (C_c = 0): C~ = diag(4 C23, C_J); L~ = [[4(L13+L23)/(L13 L23), -2/L13], [-2/L13, 1/L13]]; in frequency
# units E_C = (e^2/2h) C~^-1 -> diag(E_C23/4, E_CJ); E_L = (phi0^2/h) L~ -> [[4(E_L13+E_L23), -2 E_L13], [-2 E_L13, E_L13]]
EC_app = sym.diag(EC23 / 4, ECJ)
EL_app = sym.Matrix([[4 * (EL13 + EL23), -2 * EL13], [-2 * EL13, EL13]])
syms = {s.name: s for s in EC.free_symbols | EL.free_symbols}
sub = {syms[k]: v for k, v in (("E_C23", EC23), ("E_CJ", ECJ), ("E_L13", EL13), ("E_L23", EL23)) if k in syms}
dEC = sym.simplify(EC.subs(sub) - EC_app)
dEL = sym.simplify(EL.subs(sub) - EL_app)
print("(a) appendix model: E_C =", EC, " E_L =", EL)
print("    E_C - appendix:", dEC, "  E_L - appendix:", dEL)
print("    cosines:", data["cosines"], " flux_linear:", data["flux_linear"])
# The appendix prints L~_12 = -2/L13 (and the flux term (2 Phi_1 - Phi_2) Phi_ext/L13), but its stated Z gives +2/L13:
# by hand, mode 1 has phi_1 - phi_3 = phi_2 - phi_3 = -2, mode 2 has phi_1 - phi_3 = -1, phi_2 - phi_3 = 0, so
# L~_12 = (-2)(-1)/L13 = +2/L13. The printed matrices are those of varphi_2 -> -varphi_2 (phi_1 - phi_2 instead of
# phi_2 - phi_1); the spectrum is unchanged. So the check is up to that sign flip.
D = sym.diag(1, -1)
dEL_flip = sym.simplify(EL.subs(sub) - D * EL_app * D)
print("    E_L - appendix with varphi_2 -> -varphi_2:", dEL_flip)
assert dEC == sym.zeros(2, 2) and dEL_flip == sym.zeros(2, 2), "the appendix's C~ or L~ is not reproduced"
assert data["cosines"] == [{"energy": "E_J", "omega": [0, -1], "offset": "0"}], data["cosines"]   # E_J cos(varphi_2)
data["_generator"]["script"] = "bench/autodiff_c66/quantize.py (appendix model)"
(HERE / "specs" / "c66_appendix.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")

# ---- (b) the model sircuitenum hands SQcircuit -------------------------------------------------------------------------
res = quantized(ground=True)
data = cq.to_spec(res, subs={ECJ: 10, ECG: 20}, mode_names=["varphi1", "varphi2"], zero_rtol=1e-12)
assert sorted(data["parameters"]) == ["E_C23", "E_J", "E_L13", "E_L23", "varphi_ext"], data["parameters"]
data["_generator"]["script"] = "bench/autodiff_c66/quantize.py (SQcircuit model)"
data["_comment"] = ["n3_g1_c66 with sircuitenum's explicit ground (E_CG = 20 GHz on every node) and E_CJ = 10 GHz;"
                    " the appendix's coordinates; irrotational flux."]
(HERE / "specs" / "c66_sqcircuit.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
p0 = {"E_J": 28.83436338, "E_L13": 0.10343485, "E_C23": 1.36334654, "E_L23": 0.05167763, "varphi_ext": 0.0}
def ev(s):
    e = sym.sympify(s.replace("//", "/").replace("./", "/").replace(".*", "*").replace(".^", "**"))
    return complex(e.subs({x: p0[x.name] for x in e.free_symbols}))
print("(b) SQcircuit model at the optimum: E_C =", [[round(ev(x).real, 6) for x in r] for r in data["E_C"]],
      " E_L =", [[round(ev(x).real, 6) for x in r] for r in data["E_L"]])
print("    cosines:", data["cosines"], " flux_linear:", data["flux_linear"], " parameters:", data["parameters"])
