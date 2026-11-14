"""Kitaev's current mirror (Weiss et al., arXiv:1908.04615, Eq. `Ham`, Tab. `table:parameters`) in circuit-quantization:
2N nodes, a ring of 2N identical junctions (E_J, C_J) between nodes i and i+1 (node 2N back to node 1), rung capacitors
C_B between nodes j and j+N (the Moebius twist), a ground capacitor C_g on every node. E_CB = 0.2, E_CJ = 100,
E_Cg = 200 GHz (numeric), E_J = 19 GHz (a parameter), the loop flux varphi_ext through the ring of junctions.
Coordinates: the junction ring is an island (to ground only through C_g), so its total charge is conserved and one
coordinate is free (cq type fr). theta_i = phi_i - phi_2N (i < 2N, periodic, charge basis; the junction cosines stay local:
cos(theta_(i+1) - theta_i), and the closing junction is cos(theta_1), the one into node 2N cos(theta_(2N-1))) and the
free global phase phi_2N, whose zero-charge sector holds the low spectrum (the spec drops it). The integer charge
lattice is kept (n_theta_i = n_i, n_global = total charge). Checked against the paper's capacitance matrix (Eq.
`CapMatrix`): E_C equals the inverse of C, transformed to theta, restricted to the zero total-charge sector.

Second coordinate choice, `rung` (specs/cm_N<N>_rung.json): per rung j < N the exciton phase a_j = phi_j - phi_(j+N)
and b_j = phi_(j+N) - phi_2N, rung N's a_N = phi_N - phi_2N, the free global phase phi_2N. Their charges: n_a_j = n_j
(an exciton across rung j is n_a_j = 1 alone), n_b_j = n_j + n_(j+N) (the rung's total charge, agiton-like); integer
lattice kept (phi_j = a_j + b_j + g, phi_(j+N) = b_j + g, phi_N = a_N + g, phi_2N = g).

Third, `junction` (specs/cm_N<N>_junction.json): the junction phases psi_k = phi_(k+1) - phi_k (k < 2N) and the free
global phase phi_1 (phi_i = g + sum_(k<i) psi_k). Junction k < 2N is cos(psi_k); the closing junction 2N is
cos(psi_1 + ... + psi_(2N-1)) (the loop). The plaquette between rungs j and j+1 holds junctions j and j+N, kinetically
bound (E_C(psi_j, psi_(j+N)) ~ +28 GHz): their difference is the heavy exciton gradient (~0.2 GHz).

    ~/anaconda3/envs/cq-z4/bin/python bench/current_mirror/quantize.py [--rung | --junction] N [N ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize

HERE = Path(__file__).resolve().parent
ECB, ECJ, ECG = 0.2, 100.0, 200.0
EJ = sym.Symbol("E_J", positive=True)
vx = sym.Symbol("varphi_ext", real=True)


def spec(N, coords="theta"):
    rung = coords == "rung"
    M = 2 * N
    br = [Branch(f"J{i}", i, i % M + 1, C=cq.ghz_to_farad(ECJ), EJ=EJ) for i in range(1, M + 1)]
    br += [Branch(f"CB{j}", j, j + N, C=cq.ghz_to_farad(ECB)) for j in range(1, N + 1)]
    br += [Branch(f"Cg{i}", i, 0, C=cq.ghz_to_farad(ECG)) for i in range(1, M + 1)]
    circuit = Circuit(br, ground=0)
    assert list(circuit.internal_nodes) == list(range(1, M + 1))
    default = Loops.from_circuit(circuit)
    ring = [n for n in default.names if all(b.startswith("J") for b in default.branch_dict[n])]
    assert len(ring) == 1, ring
    flux = {n: 0 for n in default.names}
    flux[ring[0]] = cq.phase_to_weber(vx)
    loops = Loops.from_tree(circuit, cq.default_tree(circuit), flux)
    if coords == "junction":
        Z = sym.zeros(M, M)
        for i in range(M):
            Z[i, M - 1] = 1                               # every node contains g = phi_1
            for k in range(i):
                Z[i, k] = 1                               # phi_i = g + psi_1 + ... + psi_(i-1)
        names = [f"psi{k}" for k in range(1, M)]
    elif not rung:
        Z = sym.eye(M)
        for i in range(M - 1):
            Z[i, M - 1] = 1                               # phi_i = theta_i + theta_2N; phi_2N = theta_2N
        names = [f"theta{i}" for i in range(1, M)]
    else:
        # columns: a_1..a_N, b_1..b_(N-1), g; rows: nodes 1..2N
        Z = sym.zeros(M, M)
        for j in range(1, N + 1):
            Z[j - 1, j - 1] = 1                           # phi_j contains a_j
        for j in range(1, N):
            Z[j - 1, N + j - 1] = 1                       # phi_j contains b_j
            Z[j + N - 1, N + j - 1] = 1                   # phi_(j+N) = b_j + g
        for i in range(M):
            Z[i, M - 1] = 1                               # every node contains g
        assert abs(Z.det()) == 1
        names = [f"a{j}" for j in range(1, N + 1)] + [f"b{j}" for j in range(1, N)]
    res = quantize(circuit, loops, Z, ("c",) * (M - 1) + ("fr",))
    data = cq.to_spec(res, mode_names=names, zero_rtol=1e-12)
    # check E_C against the paper's capacitance matrix: C_ii = Cg + 2CJ + CB, C_i,i+-1 = -CJ, C_i,i+-N = -CB (mod 2N)
    cJ, cB, cg = 1 / ECJ, 1 / ECB, 1 / ECG          # capacitances in units of e^2/(2h GHz): E_C = 1/C
    C = np.zeros((M, M))
    for i in range(M):
        C[i, i] = cg + 2 * cJ + cB
        for d in (1, -1):
            C[i, (i + d) % M] -= cJ
        C[i, (i + N) % M] -= cB
    # n_theta_i = n_i - ... : with phi = Z theta, n_theta = Z^T n_node; E_C(theta) = Z^-1 C^-1 Z^-T; the zero-sector
    # block (n_global = 0) is its upper-left (M-1) x (M-1) block
    Zn = np.array(Z, dtype=float)
    EC_paper = (np.linalg.inv(Zn) @ np.linalg.inv(C) @ np.linalg.inv(Zn).T)[:M - 1, :M - 1]
    EC = np.array([[float(sym.sympify(x.replace("//", "/"))) for x in r] for r in data["E_C"]])
    err = np.max(np.abs(EC - EC_paper)) / np.max(np.abs(EC_paper))
    assert err < 1e-10, err
    assert len(data["cosines"]) == M, data["cosines"]
    assert all(sum(map(abs, c["omega"])) <= {"rung": 4, "theta": 2, "junction": M - 1}[coords] for c in data["cosines"]), data["cosines"]
    data["_generator"]["script"] = f"bench/current_mirror/quantize.py {N}"
    data["_comment"] = [f"Current mirror N = {N} (2N = {M} nodes), arXiv:1908.04615 Eq. Ham; E_CB = 0.2, E_CJ = 100, "
                        "E_Cg = 200 GHz; " + {"rung": "rung coordinates a_j, b_j", "theta": "theta_i = phi_i - phi_2N", "junction": "junction phases psi_k"}[coords] +
                        " (periodic), the free global phase dropped (zero total charge)."]
    (HERE / "specs" / f"cm_N{N}{'' if coords == 'theta' else '_' + coords}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"N = {N} {coords}: E_C vs paper's C^-1: {err:.1e}; parameters {data['parameters']}; cosines {len(data['cosines'])}; "
          f"E_C diagonal {np.round(np.diag(EC), 4)}, largest off-diagonal {np.max(np.abs(EC - np.diag(np.diag(EC)))):.4f} GHz")


args = sys.argv[1:]
COORDS = "rung" if "--rung" in args else "junction" if "--junction" in args else "theta"
for N in (int(a) for a in args if not a.startswith("--")):
    spec(N, COORDS)
