"""Fluxonium with an explicit junction-array superinductance (Di Paolo et al., arXiv:1912.01018, Fig. 3) in
circuit-quantization, exported as kronfree-specs.

    ~/anaconda3/envs/cq-z4/bin/python bench/array_fluxonium/quantize.py N z [z ...]

The circuit: a ring of N array junctions (C_J, E_J from the array impedance z = Z_J/R_Q at plasma frequency
ω_p/2π = 12.5 GHz: Z_J = z R_Q, C_J = 1/(ω_p Z_J), L_J = Z_J/ω_p) closed by the black-sheep junction (C_Jb = 7.5 fF,
E_Jb = 8.9 GHz), no ground capacitance (C_0 = 0, the circuit floats), as in the figure's caption (superfluxo.tex l. 187).
Coordinates: the array junction phases theta_i = phi_{i-1} - phi_i (nodes 1..N+1 around the ring), every one compact,
and the cyclic global phase. The loop flux varphi_ext sits on the black-sheep junction. circuit-quantization gives each
theta_i its own offset charge; all are set to one parameter n_g (the paper's n_gi = n_g on every site), which the spec
applies in the full form 4 (n - n_g)^T E_C (n - n_g).

Writes bench/array_fluxonium/specs/array_N<N>_z<z>.json (modes t1..tN; parameters n_g, varphi_ext).
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize

HERE = Path(__file__).resolve().parent
E_CHARGE, H_PLANCK = 1.602176634e-19, 6.62607015e-34
RQ = H_PLANCK / (2 * E_CHARGE) ** 2                  # superconducting resistance quantum h/(2e)^2
PHI0R = H_PLANCK / (2 * E_CHARGE) / (2 * np.pi)      # reduced flux quantum
FP = 12.5e9                                          # array plasma frequency, Hz
CJB, EJB = 7.5e-15, 8.9                              # black-sheep junction: F, GHz


def array_junction(z):
    """C_J (F) and E_J (GHz) of an array junction of reduced impedance z at plasma frequency FP."""
    Z = z * RQ
    CJ = 1 / (2 * np.pi * FP * Z)
    LJ = Z / (2 * np.pi * FP)
    return CJ, PHI0R**2 / LJ / H_PLANCK / 1e9


def build(N, z):
    CJ, EJ = array_junction(z)
    nodes = list(range(1, N + 2))
    branches = [Branch(f"J{i}", nodes[i - 1], nodes[i], C=CJ, EJ=EJ) for i in range(1, N + 1)]
    branches.append(Branch("Jb", nodes[N], nodes[0], C=CJB, EJ=EJB))
    circuit = Circuit(branches)                       # floating: C_0 = 0
    assert list(circuit.internal_nodes) == nodes, circuit.internal_nodes
    vx = sym.Symbol("varphi_ext", real=True)
    default = Loops.from_circuit(circuit)
    (ring,) = default.names
    loops = Loops.from_tree(circuit, cq.default_tree(circuit), {ring: cq.phase_to_weber(vx)})
    # phi_k = g - sum_{i<k} theta_i on node k (theta_i = phi_i - phi_{i+1} in node labels 1..N+1); columns
    # theta_1..theta_N, then the cyclic global phase
    Z = sym.zeros(N + 1, N + 1)
    for k in range(N + 1):
        for i in range(k):
            Z[k, i] = -1
        Z[k, N] = 1
    types = ("c",) * N + ("cy",)
    res0 = quantize(circuit, loops, Z, types)
    # the ring flux on the black-sheep junction alone: B's ring column is +-1 on Jb (sign so that G B = I)
    G, names = res0.loops.G, list(res0.circuit.branch_names)
    B = sym.Matrix(res0.B)
    r = list(res0.loops.names).index(ring)
    b = sym.zeros(len(names), 1)
    b[names.index("Jb")] = 1
    if (G * b)[r] != 1:
        b = -b
    assert G * b == sym.Matrix([1 if k == r else 0 for k in range(G.rows)])
    B[:, r] = b
    res = quantize(circuit, loops, Z, types, B=B)
    return res, CJ, EJ


def main():
    N = int(sys.argv[1])
    zs = [float(x) for x in sys.argv[2:]] or [0.10]
    for z in zs:
        res, CJ, EJ = build(N, z)
        ng = sym.Symbol("n_g", real=True)
        subs = {s: ng for s in res.ham.H.free_symbols if s.name.startswith("n_g")} if hasattr(res, "ham") else {}
        data = cq.to_spec(res, subs=subs, mode_names=[f"t{i}" for i in range(1, N + 1)], zero_rtol=1e-12)
        assert sorted(data["parameters"]) == ["n_g", "varphi_ext"], data["parameters"]
        data["_generator"]["script"] = f"bench/array_fluxonium/quantize.py {N} {z}"
        data["_comment"] = [
            f"Array fluxonium, Di Paolo et al. arXiv:1912.01018 Fig. 3: N = {N} array junctions at z = {z}"
            f" (C_J = {CJ:.6e} F, E_J = {EJ:.6f} GHz), black-sheep C_Jb = 7.5 fF, E_Jb = 8.9 GHz, C_0 = 0 (floating).",
            "Modes t_i = phi_(i-1) - phi_i (compact); every theta offset charge set to the one parameter n_g (full form).",
            "Loop flux varphi_ext on the black-sheep junction.",
        ]
        out = HERE / "specs" / f"array_N{N}_z{z:.3f}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        EC = np.array([[float(sym.sympify(x.replace("//", "/"))) for x in row] for row in data["E_C"]])
        print(f"N {N} z {z}: E_J {EJ:.4f} GHz, E_C diag {EC[0, 0]:.6f} off {EC[0, 1]:.6e}, n_g {data['n_g'][:3]}...,"
              f" cosines {len(data['cosines'])}, wrote {out.name}")


if __name__ == "__main__":
    main()
