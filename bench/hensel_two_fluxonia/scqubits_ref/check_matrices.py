"""Exact check, before any eigensolve: scqubits' variable categories, charging matrix and inductive matrix in our two
frames equal circuit.jl's (E_C, E_L of the local and the f2,f3-sheared frames)."""

import numpy as np

from hensel_scq import FR, circuit

for frame, ec_key, el_key in (("local", "EC_local", "EL_local"), ("shear", "EC_shear", "EL_shear")):
    circ = circuit(frame)
    sc = circ.symbolic_circuit
    print(f"[{frame}] variable categories: {sc.var_categories}")
    EC = np.array(sc._EC_matrix(substitute_params=True), dtype=float)
    ours_EC = np.array(FR[ec_key])
    # inductive matrix in θ: Tᵀ (E_L node matrix) T, from the branch list
    T = np.array(sc.transformation_matrix, dtype=float)
    Lnode = FR["EL_CONST"] * np.array(FR["LINV_GEO_per_nH"])
    EL = (T.T @ Lnode @ T)[:6, :6]
    ours_EL = np.array(FR[el_key])
    print(f"[{frame}] E_C shape {EC.shape}; max |E_C(scqubits)[:6,:6] - ours| = {np.abs(EC[:6, :6] - ours_EC).max():.3e} GHz")
    print(f"[{frame}] max |T^T E_L T - ours| = {np.abs(EL - ours_EL).max():.3e} GHz")
    print(circ.sym_hamiltonian(return_expr=True) if hasattr(circ, "sym_hamiltonian") else "")
