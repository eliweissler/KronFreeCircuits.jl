"""Spec VARIANT (not circuit-quantization's output): the full current mirror in all 2N node phases phi_1 .. phi_2N, every
one compact (a charge basis per node), exactly as Weiss et al. write Eq. `Ham` (arXiv:1908.04615):
    H = 4 sum_ij (E_C)_ij n_i n_j - sum_j E_J cos(phi_(j+1) - phi_j - phi_ext / 2N),   E_C = (e^2/2) C^-1,
with the capacitance matrix of Eq. `CapMatrix` (C_ii = C_g + 2 C_J + C_B, -C_J at i +- 1, -C_B at i +- N, mod 2N) and
Tab. `table:parameters` (E_CB = 0.2, E_CJ = 100, E_Cg = 200 GHz; E_J a parameter). C_g > 0 makes C invertible; the
global rotor (all phases together) is then dynamical, its charge Q = sum n_i conserved (H is block-diagonal in Q), and
the Q != 0 blocks sit about 4 (E_Cg / 2N) Q^2 higher. circuit-quantization removes that mode (its type `fr`) and emits
the Q = 0 block in theta_i = phi_i - phi_2N (quantize.py); this variant keeps it, so that every rung (phi_j, phi_(j+N))
is the same leaf and the ring translation and the rung swap act leaf by leaf.

Cross-check against circuit-quantization: the Q = 0 block of this E_C, transformed to theta, equals the E_C of
specs/cm_N<N>.json (quantize.py) exactly (rational arithmetic), and the junction cosines are the same 2N.

    ~/anaconda3/envs/cq-z4/bin/python bench/current_mirror/spec_nodes.py N [N ...]       (writes specs/cm_N<N>_node.json)
"""
import json
import sys
from pathlib import Path

import sympy as sym

HERE = Path(__file__).resolve().parent
ECB, ECJ, ECG = sym.Rational(1, 5), sym.Integer(100), sym.Integer(200)


def jl(x):
    """A rational as the spec's Julia expression."""
    x = sym.Rational(x)
    return str(x.p) if x.q == 1 else f"{x.p} // {x.q}"


def parse_jl(s):
    return sym.Rational(sym.sympify(s.replace("//", "/")))


def spec(N):
    M = 2 * N
    cJ, cB, cg = 1 / ECJ, 1 / ECB, 1 / ECG          # capacitances in units with E_C = 1/C (GHz)
    C = sym.zeros(M, M)
    for i in range(M):
        C[i, i] = cg + 2 * cJ + cB
        for d in (1, -1):
            C[i, (i + d) % M] -= cJ
        C[i, (i + N) % M] -= cB
    EC = C.inv()
    # cross-check with circuit-quantization's Q = 0 block (theta_i = phi_i - phi_2N)
    Z = sym.eye(M)
    for i in range(M - 1):
        Z[i, M - 1] = 1
    block = (Z.inv() * EC * Z.inv().T)[:M - 1, :M - 1]
    cq = json.loads((HERE / "specs" / f"cm_N{N}.json").read_text())
    ECcq = sym.Matrix([[parse_jl(x) for x in r] for r in cq["E_C"]])
    assert block == ECcq, "the Q = 0 block differs from circuit-quantization's E_C"
    assert len(cq["cosines"]) == M
    modes = [{"name": f"phi{i}", "type": "c"} for i in range(1, M + 1)]
    cosines = []
    for i in range(M):                                   # junction i+1: cos(phi_(i+2) - phi_(i+1) - phi_ext / 2N)
        om = [0] * M
        om[(i + 1) % M] += 1
        om[i] -= 1
        cosines.append({"energy": "E_J", "omega": om, "offset": f"-1 // {M} * varphi_ext"})
    data = {"format": "kronfree-spec", "version": 1, "units": "GHz", "parameters": ["E_J", "varphi_ext"],
            "modes": modes, "E_C": [[jl(EC[i, j]) for j in range(M)] for i in range(M)],
            "E_L": [["0"] * M for _ in range(M)], "flux_linear": ["0"] * M, "n_g": ["0"] * M,
            "cosines": cosines, "constant": "0",
            "_generator": {"script": f"bench/current_mirror/spec_nodes.py {N}",
                           "note": "hand-built variant of circuit-quantization's cm_N<N>.json: all 2N node phases kept "
                                   "compact (the island's global rotor included); Q = 0 block checked equal to it"},
            "_comment": [f"Current mirror N = {N}, all 2N = {M} node phases (arXiv:1908.04615 Eq. Ham), E_CB = 0.2, "
                         "E_CJ = 100, E_Cg = 200 GHz; zero offset charges; flux phi_ext / 2N on every junction."]}
    (HERE / "specs" / f"cm_N{N}_node.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"N = {N}: Q = 0 block equal to circuit-quantization's E_C (exact); E_C diagonal {float(EC[0, 0]):.6f}, "
          f"rung {float(EC[0, N]):.6f}, neighbour {float(EC[0, 1]):.6f} GHz; global charging 4 E_Cg/2N = "
          f"{float(4 * ECG / M):.1f} GHz")


for N in map(int, sys.argv[1:]):
    spec(N)
