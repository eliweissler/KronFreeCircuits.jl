"""Two capacitively coupled zero-pi qubits (Eli's two-qubit gate study, arXiv:2609.16630) in circuit-quantization,
exported as a kronfree-spec.

    ~/anaconda3/envs/cq-z4/bin/python bench/coupled_zero_pi/quantize.py [params.yaml]

The circuit, element values and coordinate transformation are those of the Python pipeline's parameter file
(default: ~/src/two_qubit_gate_for_zero_pi_qubit/data/Two_qubit_data_Sorted_Truc/params_truc_default.yaml): its zp_yml
netlist (two zero-pis on nodes 1-4 and 5-8, coupler islands 9 and 10 joined to every qubit node by E_cc capacitors,
E_c0 from every qubit node to ground) and its Ztransform_2zeropi (phi_node = Z theta). The zero-pi branches, the ring
loops and the junction flux gauge are those of test/fixtures/generate_specs.py (the package's generated fixtures).
Loop fluxes varphi_ext1, varphi_ext2 and the theta offset charges stay symbolic; every element value is numeric.

Writes bench/coupled_zero_pi/specs/coupled_zero_pi.json (modes theta1, phi1, zeta1, theta2, phi2, zeta2).
"""
import importlib.util
import json
import sys
from pathlib import Path

import ast
import re

import numpy as np
import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PARAMS = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "src/two_qubit_gate_for_zero_pi_qubit/data/Two_qubit_data_Sorted_Truc/params_truc_default.yaml"

# the package's proven zero-pi helpers (importing the module writes nothing)
_spec = importlib.util.spec_from_file_location("kronfree_generate_specs", REPO / "test/fixtures/generate_specs.py")
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)

def read_params(path):
    """The scalar `key: value` entries and the Ztransform_2zeropi rows of the pipeline's parameter file (the env has no
    PyYAML; the file's layout is flat, one entry per line)."""
    P, rows, in_z = {}, [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Ztransform_2zeropi:"):
            in_z = True
            continue
        if in_z:
            m = re.match(r"\s*-\s*(\[.*\])", line)
            if m:
                rows.append(ast.literal_eval(m.group(1)))
                continue
            in_z = False
        m = re.match(r"^([^\s#:][^:]*):\s*([-+0-9.eE]+)\s*(#.*)?$", line)
        if m:
            P[m.group(1).strip()] = float(m.group(2))
    P["Ztransform_2zeropi"] = rows
    assert len(rows) == 10 and all(len(r) == 10 for r in rows), "Ztransform_2zeropi is not 10 x 10"
    return P


P = read_params(PARAMS)
print("parameters from", PARAMS)
for k in ("EJ1", "ECJ1", "EL1", "EC1", "EJ2", "ECJ2", "EL2", "EC2", "Ec0", "Ecc", "Φ1", "Φ2", "ng1", "ng2"):
    print(f"  {k} = {P[k]}")

# ---- the netlist of zp_yml ---------------------------------------------------------------------------------------------
branches = []
for q, nodes in ((1, (1, 2, 3, 4)), (2, (5, 6, 7, 8))):
    # zp_yml: JJ a-b, JJ c-d, L b-c, L a-d, C a-c, C b-d (zero_pi_branches' layout, L d-a the same inductor)
    branches += gs.zero_pi_branches(q, nodes, (P[f"ECJ{q}"],) * 2, (P[f"EC{q}"],) * 2, (P[f"EL{q}"],) * 2, (P[f"EJ{q}"],) * 2)
# coupler: nodes 1, 4, 5, 8 to island 10 and 2, 3, 6, 7 to island 9 (zp_yml's Ecc capacitors)
for q, (a, b, c, d) in ((1, (1, 2, 3, 4)), (2, (5, 6, 7, 8))):
    branches += [Branch(f"Cc{q}_{n}", n, 10, C=cq.ghz_to_farad(P["Ecc"])) for n in (a, d)]
    branches += [Branch(f"Cc{q}_{n}", n, 9, C=cq.ghz_to_farad(P["Ecc"])) for n in (b, c)]
branches += [Branch(f"Cg{n}", n, 0, C=cq.ghz_to_farad(P["Ec0"])) for n in range(1, 9)]
circuit = Circuit(branches, ground=0)
assert list(circuit.internal_nodes) == list(range(1, 11)), circuit.internal_nodes

# ---- coordinates: the pipeline's Ztransform_2zeropi (rows: nodes 1..10, columns: its theta_1..theta_10) ----------------
Z = sym.Matrix(P["Ztransform_2zeropi"]).applyfunc(sym.nsimplify)
# scqubits' classification in the pipeline's summary: periodic 1, 5; extended 3, 4, 7, 8; free 2, 6, 9, 10. Columns 4
# and 8 (zeta) are decoupled oscillators at symmetry: harmonic.
types = ("c", "fr", "e", "h", "c", "fr", "e", "h", "fr", "fr")
vx = {1: gs.real("varphi_ext1"), 2: gs.real("varphi_ext2")}
loops, rings = gs.ring_loops(circuit, vx)
res = gs.quantize_junction_gauge(circuit, loops, rings, Z, types)
print("types:", res.coords.types)

names = ["theta1", "phi1", "zeta1", "theta2", "phi2", "zeta2"]
data = cq.to_spec(res, mode_names=names, zero_rtol=1e-12)
EC = np.array([[float(sym.sympify(x.replace("//", "/"))) for x in row] for row in data["E_C"]])
EL = np.array([[float(sym.sympify(x.replace("//", "/"))) for x in row] for row in data["E_L"]])
print("E_C (GHz):\n", np.round(EC, 6))
print("E_L (GHz):\n", np.round(EL, 6))
# against the pipeline's summary (scqubits' symbolic Hamiltonian in its variables): 4 E_C[i,i] is the coefficient of
# n_i^2 and 8 E_C[theta1,theta2] the coupling g n_theta1 n_theta2
summary = {"4 E_C[theta,theta]": (4 * EC[0, 0], 0.352261), "4 E_C[phi,phi]": (4 * EC[1, 1], 2.130838),
           "4 E_C[zeta,zeta]": (4 * EC[2, 2], 0.342739), "8 E_C[theta1,theta2] = g": (8 * EC[0, 3], 0.030962),
           "E_L[phi,phi]/2": (EL[1, 1] / 2, 0.377), "E_L[zeta,zeta]/2": (EL[2, 2] / 2, 0.377)}
for k, (ours, theirs) in summary.items():
    print(f"  {k}: {ours:.6f}  (pipeline summary {theirs})")

data["_generator"]["script"] = "bench/coupled_zero_pi/quantize.py " + str(PARAMS)
data["_comment"] = [
    "Two capacitively coupled zero-pis, the circuit and Ztransform_2zeropi of the two-qubit gate study's " + PARAMS.name + ".",
    "Modes theta1, phi1, zeta1, theta2, phi2, zeta2 are the pipeline's columns 1, 3, 4, 5, 7, 8; columns 2, 6, 9, 10 are free.",
    "Ring fluxes varphi_ext1, varphi_ext2 on the junctions (test/fixtures/generate_specs.py's gauge).",
]
out = HERE / "specs" / "coupled_zero_pi.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out, "parameters", data["parameters"])
