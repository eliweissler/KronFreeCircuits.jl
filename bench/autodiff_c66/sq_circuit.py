"""The n3_g1_c66 circuit in SQcircuit, exactly as sircuitenum builds it (make_sq -> to_SQcircuit, ground_node=None:
an explicit ground node 0 with a 20 GHz capacitor to every node; junction capacitance E_CJ = 10 GHz; enumeration nodes
0, 1, 2 -> 1, 2, 3; the one inductive loop shared by the junction and both inductors, its flux in units of Phi_0).
Shared by the SQcircuit scripts of this folder; run directly it prints the circuit's modes and lowest levels.

    ~/anaconda3/envs/sqcircuit-bench/bin/python bench/autodiff_c66/sq_circuit.py
"""
import numpy as np
import SQcircuit as sq

P0 = {"E_J": 28.83436338, "E_L13": 0.10343485, "E_C23": 1.36334654, "E_L23": 0.05167763, "flux": 0.0}
# make_figs.py n3_g1_c66: trunc_num = [150, 80] on the two qubit modes. Needs SQcircuit's dev-ew branch (sircuitenum's
# README; installed in conda env sqcircuit-bench from ~/src/SQcircuit-dev-ew): there the island's common mode, which has
# no potential (its charge is conserved, the low spectrum is its n = 0 sector), is dropped as a negligible charge
# projection. SQcircuit 1.0.0 keeps it with a garbage flux column (|S| ~ 1e14) and gives a wrong, harmonic spectrum.
TRUNC = [150, 80]
ECJ, ECG = 10.0, 20.0


def build(p=P0, requires_grad=False, trunc=TRUNC):
    """The circuit at parameters p (GHz; flux in Phi_0). With requires_grad the five parameters are leaf tensors."""
    loop = sq.Loop(p["flux"], requires_grad=requires_grad)
    J = sq.Junction(p["E_J"], "GHz", requires_grad=requires_grad, cap=sq.Capacitor(ECJ, "GHz"), loops=[loop])
    L13 = sq.Inductor(p["E_L13"], "GHz", requires_grad=requires_grad, loops=[loop])
    C23 = sq.Capacitor(p["E_C23"], "GHz", requires_grad=requires_grad)
    L23 = sq.Inductor(p["E_L23"], "GHz", requires_grad=requires_grad, loops=[loop])
    elements = {(1, 2): [J], (1, 3): [L13], (2, 3): [C23, L23],
                (0, 1): [sq.Capacitor(ECG, "GHz")], (0, 2): [sq.Capacitor(ECG, "GHz")], (0, 3): [sq.Capacitor(ECG, "GHz")]}
    cr = sq.Circuit(elements)
    if cr.n == len(trunc) + 1:          # the island's common mode kept (the PyTorch engine): its n = 0 sector only
        trunc = list(trunc) + [1]
    assert cr.n == len(trunc), f"SQcircuit found {cr.n} modes for {len(trunc)} cutoffs"
    cr.set_trunc_nums(trunc)
    return cr, {"E_J": J, "E_L13": L13, "E_C23": C23, "E_L23": L23, "flux": loop}


if __name__ == "__main__":
    sq.set_engine("NumPy")
    cr, _ = build()
    print("modes:", cr.n, " truncation:", cr.trunc_nums, " omega:", np.round(np.asarray(cr.omega) / 2 / np.pi / 1e9, 4),
          " (0 = charge mode)")
    print("S (flux, node = S mode):\n", np.round(cr.S, 6))
    print("R (charge, node = R mode):\n", np.round(cr.R, 6))
    efreqs, _ = cr.diag(5)
    print("lowest levels (GHz):", np.round(efreqs, 9))
    print("omega01 = %.9f, alpha = omega12 - omega01 = %.9f GHz" % (efreqs[1] - efreqs[0], efreqs[2] - 2 * efreqs[1] + efreqs[0]))
