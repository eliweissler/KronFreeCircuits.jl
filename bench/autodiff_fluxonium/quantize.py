"""The enumeration paper's optimized fluxonium (enumeration_figures/make_figs.py integer_fluxonium, param_set_us: one
edge with C, J, L; E_C = 2.63, E_J = 2.85, E_L = 0.051 GHz; loop flux 0; cj = 0, so no junction capacitance) in
circuit-quantization, as sircuitenum hands it to SQcircuit (ground_node=None: an explicit ground node 0 with a 20 GHz
capacitor to each of the two nodes). Coordinates: varphi = phi_2 - phi_1 (extended) and the free common mode
phi_1 + phi_2, which the spec drops (no potential; its n = 0 sector).

    ~/anaconda3/envs/cq-z4/bin/python bench/autodiff_fluxonium/quantize.py      (writes specs/fluxonium_sqcircuit.json)
"""
import json
from pathlib import Path

import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize

HERE = Path(__file__).resolve().parent
pos = lambda s: sym.Symbol(s, positive=True)
EC, EJ, EL, ECG = pos("E_C"), pos("E_J"), pos("E_L"), pos("E_CG")
vx = sym.Symbol("varphi_ext", real=True)
Zinv = sym.Matrix([[-1, 1], [1, 1]])                    # varphi = phi_2 - phi_1, the common mode phi_1 + phi_2
Z = Zinv.inv()

# circuit-quantization's junction branch carries a capacitance: the shunt C (E_C) sits on it (the same edge)
circuit = Circuit([Branch("J", 1, 2, C=cq.ghz_to_farad(EC), EJ=EJ),
                   Branch("L", 1, 2, L=cq.ghz_to_henry(EL)),
                   Branch("Cg1", 1, 0, C=cq.ghz_to_farad(ECG)),
                   Branch("Cg2", 2, 0, C=cq.ghz_to_farad(ECG))], ground=0)
default = Loops.from_circuit(circuit)
ring = next(n for n in default.names if {"J", "L"} <= set(default.branch_dict[n]))
flux = {n: 0 for n in default.names}
flux[ring] = cq.phase_to_weber(vx)
loops = Loops.from_tree(circuit, cq.default_tree(circuit), flux)
res = quantize(circuit, loops, Z, ("e", "fr"))
data = cq.to_spec(res, subs={ECG: 20}, mode_names=["varphi"], zero_rtol=1e-12)
assert sorted(data["parameters"]) == ["E_C", "E_J", "E_L", "varphi_ext"], data["parameters"]
data["_generator"]["script"] = "bench/autodiff_fluxonium/quantize.py"
data["_comment"] = ["The enumeration's optimized fluxonium with sircuitenum's explicit ground (E_CG = 20 GHz on both"
                    " nodes), no junction capacitance; varphi = phi_2 - phi_1; irrotational flux."]
(HERE / "specs" / "fluxonium_sqcircuit.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
print("E_C =", data["E_C"], " E_L =", data["E_L"])
print("cosines:", data["cosines"], " flux_linear:", data.get("flux_linear"), " parameters:", data["parameters"])
