"""The standard single-mode fluxonium for the array circuit of Di Paolo et al. (arXiv:1912.01018, Fig. 3), in
circuit-quantization: the black-sheep junction (C_Jb = 7.5 fF, E_Jb = 8.9 GHz) shunted by the array treated as a
linear superinductance L = N L_J (E_L = E_J/N) and the array's series capacitance C_J/N. With C_0 = 0 this is the
paper's effective model (superfluxo.tex l. 127-130: phi' = phi, n' = mean n_i) with the array's cos(phi/N) expanded
to the quadratic term, the standard fluxonium. The loop flux sits on the junction.

    ~/anaconda3/envs/cq-z4/bin/python bench/array_fluxonium/quantize_single.py N z [z ...]

Writes bench/array_fluxonium/specs/single_N<N>_z<z>.json (mode phi; parameter varphi_ext).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
_spec = importlib.util.spec_from_file_location("kronfree_generate_specs", REPO / "test/fixtures/generate_specs.py")
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)
sys.path.insert(0, str(HERE))
from quantize import array_junction, CJB, EJB  # noqa: E402  the same element values as the array specs


def build(N, z):
    CJ, EJ = array_junction(z)
    LJ = gs.cq.ghz_to_henry(EJ)                       # E_J = phi0^2/L_J, as an inductance
    circuit = Circuit([Branch("Jb", 1, 2, C=CJB, EJ=EJB),
                       Branch("Ca", 1, 2, C=CJ / N),   # the array's N junction capacitances in series
                       Branch("L", 2, 1, L=N * LJ)])   # the array as a linear inductance, E_L = E_J/N
    vx = sym.Symbol("varphi_ext", real=True)
    default = Loops.from_circuit(circuit)
    loop = next(n for n in default.names if "L" in default.branch_dict[n] and "Jb" in default.branch_dict[n])
    flux = {n: 0 for n in default.names}
    flux[loop] = cq.phase_to_weber(vx)
    loops = Loops.from_tree(circuit, cq.default_tree(circuit), flux)
    Z = sym.Matrix([[sym.Rational(1, 2), 1], [-sym.Rational(1, 2), 1]])   # phi = phi_1 - phi_2, global phase
    types = ("e", "cy")
    for share in (1, -1):                              # the loop flux on the junction (junction gauge)
        try:
            return gs.quantize_junction_gauge(circuit, loops, {loop: {"Jb": share}}, Z, types), CJ, EJ
        except AssertionError:
            continue
    raise RuntimeError("no junction flux assignment satisfies G B = I")


def main():
    N = int(sys.argv[1])
    for z in [float(x) for x in sys.argv[2:]] or [0.10]:
        res, CJ, EJ = build(N, z)
        data = cq.to_spec(res, mode_names=["phi"], zero_rtol=1e-12)
        data["_generator"]["script"] = f"bench/array_fluxonium/quantize_single.py {N} {z}"
        data["_comment"] = [f"Single-mode fluxonium for the N = {N}, z = {z} array: C = C_Jb + C_J/N, E_L = E_J/N,"
                            " junction C_Jb = 7.5 fF, E_Jb = 8.9 GHz; loop flux varphi_ext on the junction."]
        out = HERE / "specs" / f"single_N{N}_z{z:.3f}.json"
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        val = lambda s: float(sym.sympify(s.replace("//", "/")))
        print(f"N {N} z {z}: E_C {val(data['E_C'][0][0]):.5f} GHz, E_L {val(data['E_L'][0][0]):.5f} GHz"
              f" (E_J/N = {EJ / N:.5f}), cosine {data['cosines']}, flux_linear {data['flux_linear']}, wrote {out.name}")


if __name__ == "__main__":
    main()
