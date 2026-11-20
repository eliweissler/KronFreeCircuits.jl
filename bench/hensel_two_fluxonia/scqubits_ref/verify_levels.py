"""Same tree, same bases, same truncations: scqubits' hierarchical diagonalization of the Hensel circuit in the local
(unsheared) frame against KronFreeCircuits' binary tree (ours_unsheared_binary.json).

Tree (scqubits variables θ1..θ6 = q1, q2, c, f1, f2, f3): root over M = [P(c, f1), P(q1, f3)] and P(q2, f2);
single-variable leaves. Kept states: q leaves 40 (181-point grid over ±3π, 9-point stencil = our order 8), Fock leaves
20 (c and f1 on 30 levels, f2 and f3 on 20), pair nodes 60, M 240, root 8 levels.
"""

import json
import sys
import time

import numpy as np
import scqubits as scq

from hensel_scq import circuit

scq.settings.STENCIL = 9
FC = [float(x) for x in (sys.argv[1:] or ["0.0", "0.3"])]
ours = json.load(open("ours_unsheared_binary.json"))

t0 = time.time()
c = circuit("local")
hierarchy = [[[[3], [4]], [[1], [6]]], [[2], [5]]]
trunc = [[240, [[60, [20, 20]], [60, [40, 20]]]], [60, [40, 20]]]
basis = [[["harmonic", "harmonic"], ["discretized", "harmonic"]], ["discretized", "harmonic"]]
for i, n in ((1, 181), (2, 181), (3, 30), (4, 30), (5, 22), (6, 22)):   # f2, f3: harmonic leaves; scqubits keeps at most dim - 2, the lowest 20 of 22 = our 20 of 20
    setattr(c, f"cutoff_ext_{i}", n)
c.set_discretized_phi_range((1, 2), (-3 * np.pi, 3 * np.pi))
c.configure(system_hierarchy=hierarchy, subsystem_trunc_dims=trunc, ext_basis=basis)
print(f"setup {time.time() - t0:.1f} s")

for fc in FC:
    c.Φ1, c.Φ2, c.Φ3 = 0.5, 0.5, fc
    t = time.time()
    E = np.sort(c.eigenvals(evals_count=8))
    dt = time.time() - t
    O = np.array(ours[str(fc)])
    w = max(abs((E[j] - E[i]) - (O[j] - O[i])) for i in range(8) for j in range(i + 1, 8))
    print(f"Φc = {fc}: {dt:.1f} s; scqubits ω0j {np.round(E - E[0], 7)}")
    print(f"           ours     ω0j {np.round(O - O[0], 7)}")
    print(f"           worst transition difference {w * 1e6:.3f} kHz; E0 difference {(E[0] - O[0]) * 1e3:.4f} MHz")
