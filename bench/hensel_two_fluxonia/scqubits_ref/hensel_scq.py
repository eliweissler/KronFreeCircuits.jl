"""The Hensel two-fluxonium circuit (arXiv:2611.05627v1 App. A) as a scqubits Circuit, in the coordinates of
../circuit.jl, for a matched hierarchical-diagonalization reference.

Same element list and constants as circuit.jl (read from frames.json, written by Julia):
ground capacitors of 50 fF on nodes 1..7; qubit junctions (1,2), (5,6), E_J = 8.98 GHz, with the 10 fF shunts as their
junction capacitance; coupler junction (3,7), E_J = 3.5 GHz, which has no capacitance in our model: scqubits needs
one, so it gets E_C = 1e12 GHz (2e-11 fF, a 4e-13 relative change); inductors 50 nH on (1,4), (4,5), (2,3), (6,7) and
7.8 nH on (3,4), (4,7). The three loop fluxes sit on the junction branches (closure branches).

Coordinates: circuit.jl's x = A φ (q1, q2, c junction differences, f1, f2, f3, global phase); scqubits takes
φ = T θ, so T = A⁻¹ (local) or A⁻¹ diag(S, 1) (frame modes sheared, x = S y).
"""

import json
import os

import numpy as np
import scqubits as scq

HERE = os.path.dirname(os.path.abspath(__file__))
FR = json.load(open(os.path.join(HERE, "frames.json")))
EC_CONST, EL_CONST = FR["EC_CONST"], FR["EL_CONST"]
EJQ, EJC = FR["EJ"]["q"], FR["EJ"]["c"]
A = np.array(FR["A"])
S_SHEAR = np.array(FR["S_shear_f2f3"])
C_GROUND, C_SHUNT = 50.0, 10.0
INDUCTORS = [((1, 4), 50.0), ((4, 5), 50.0), ((2, 3), 50.0), ((6, 7), 50.0), ((3, 4), 7.8), ((4, 7), 7.8)]
EC_COUPLER_JJ = 1e12


def yaml():
    rows = [f"- [C, 0, {i}, {EC_CONST / C_GROUND:.17g}]" for i in range(1, 8)]
    rows += [f"- [JJ, 1, 2, {EJQ:.17g}, {EC_CONST / C_SHUNT:.17g}]",
             f"- [JJ, 5, 6, {EJQ:.17g}, {EC_CONST / C_SHUNT:.17g}]",
             f"- [JJ, 3, 7, {EJC:.17g}, {EC_COUPLER_JJ:.17g}]"]
    rows += [f"- [L, {i}, {j}, {EL_CONST / L:.17g}]" for (i, j), L in INDUCTORS]
    return "branches:\n" + "\n".join(rows) + "\n"


def transformation(frame="local"):
    Ainv = np.linalg.inv(A)
    if frame == "local":
        return Ainv
    if frame == "shear":
        M = np.eye(7)
        M[:6, :6] = S_SHEAR
        return Ainv @ M
    raise ValueError(frame)


def circuit(frame="local", **kwargs):
    """The circuit with fluxes on the three junction branches, in the given frame."""
    circ = scq.Circuit(yaml(), from_file=False, ext_basis="discretized", **kwargs)
    jj = [b for b in circ.branches if b.type == "JJ"]
    circ.configure(transformation_matrix=transformation(frame), closure_branches=jj)
    return circ
