"""The enumeration's optimized fluxonium in SQcircuit, exactly as sircuitenum builds it (make_sq -> to_SQcircuit,
ground_node=None: an explicit ground node 0 with a 20 GHz capacitor to each node; cj = 0, so the junction has no
capacitor of its own (SQcircuit gives it a negligible VerySmallCap); enumeration nodes 0, 1 -> 1, 2; the loop of J and
L, its flux in units of Phi_0). Shared by the SQcircuit scripts of this folder; run directly it prints the lowest levels.

    ~/anaconda3/envs/sqcircuit-bench/bin/python bench/autodiff_fluxonium/sq_circuit.py
"""
import numpy as np
import SQcircuit as sq

P0 = {"E_C": 2.63, "E_J": 2.85, "E_L": 0.051, "flux": 0.0}
# make_figs.py integer_fluxonium: trunc_num = 120. SQcircuit's dev-ew branch drops the island's common mode (see
# ../autodiff_c66/sq_circuit.py).
TRUNC = [120]
ECG = 20.0


def build(p=P0, requires_grad=False, trunc=TRUNC):
    """The circuit at parameters p (GHz; flux in Phi_0). With requires_grad the four parameters are leaf tensors."""
    loop = sq.Loop(p["flux"], requires_grad=requires_grad)
    C = sq.Capacitor(p["E_C"], "GHz", requires_grad=requires_grad)
    J = sq.Junction(p["E_J"], "GHz", requires_grad=requires_grad, loops=[loop])
    L = sq.Inductor(p["E_L"], "GHz", requires_grad=requires_grad, loops=[loop])
    elements = {(1, 2): [C, J, L], (0, 1): [sq.Capacitor(ECG, "GHz")], (0, 2): [sq.Capacitor(ECG, "GHz")]}
    cr = sq.Circuit(elements)
    if cr.n == len(trunc) + 1:          # the island's common mode kept (the PyTorch engine): its n = 0 sector only
        trunc = list(trunc) + [1]
    assert cr.n == len(trunc), f"SQcircuit found {cr.n} modes for {len(trunc)} cutoffs"
    cr.set_trunc_nums(trunc)
    return cr, {"E_C": C, "E_J": J, "E_L": L, "flux": loop}


if __name__ == "__main__":
    sq.set_engine("NumPy")
    cr, _ = build()
    print("modes:", cr.n, " truncation:", cr.trunc_nums)
    print("S:\n", np.round(cr.S, 6), "\nR:\n", np.round(cr.R, 6))
    efreqs, _ = cr.diag(5)
    print("lowest levels (GHz):", np.round(efreqs, 9))
    print("omega01 = %.9f, alpha = %.9f GHz" % (efreqs[1] - efreqs[0], efreqs[2] - 2 * efreqs[1] + efreqs[0]))
