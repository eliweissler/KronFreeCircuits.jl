"""Jacobian and Hessian of the optimized fluxonium's quantities with SQcircuit (dev-ew branch, PyTorch engine), the
reference for kronfree.jl. Same parameters, quantities and coordinates:
  parameters E_C, E_J, E_L (GHz) and the loop flux f (Phi_0);
  quantities omega01, alpha, and |<0|O|1>|^2 for O = n, phi of the mode varphi = phi2 - phi1 (node phases), from
  SQcircuit's own mode operators:
  phi_node = S phi_mode, Q_node = R Q_mode, phase = phi_op 2e/sqrt(hbar), n = Q_op sqrt(hbar)/2e.
Jacobian: autodiff (torch.autograd.grad through SQcircuit's EigenSolver), converted from SQcircuit's SI leaves to the
GHz parameters. Hessian: central differences of the autodiff Jacobian (the EigenSolver is once-differentiable).
Checks: the autodiff Jacobian against central differences of SQcircuit's values.

    ~/anaconda3/envs/sqcircuit-bench/bin/python bench/autodiff_fluxonium/sq_grad.py [smoke]   (writes results/sqcircuit*.json)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import SQcircuit as sq
import torch
from SQcircuit import functions as sqf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sq_circuit import P0, TRUNC, build  # noqa: E402

SMOKE = __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "smoke"
TR = [40] if SMOKE else TRUNC
# eigenpairs diag computes: SQcircuit's eigenvector gradient (get_partial_vec) sums over these states only
import os
N_EIG = int(os.environ.get("N_EIG", "4"))
PARAMS = ["E_C", "E_J", "E_L", "flux"]
QNAMES = ["omega01", "alpha", "|n_01|^2", "|phi_01|^2"]
ZINV = np.array([[-1.0, 1.0]])           # varphi in terms of the node phases 1, 2
ZT = np.array([[-0.5, 0.5]])             # n_varphi = Z^T n_node (Z = [[-1, 1], [1, 1]]^-1)
NM = 1
sq.set_engine("PyTorch")
sq.set_max_eigenvector_grad(3)


def lincomb(coeffs, ops):
    """sum_i c_i op_i over sparse operator tensors (sparse + sparse only; no dense zero to start from)."""
    terms = [c * o for c, o in zip(coeffs, ops) if abs(complex(c)) > 1e-9]
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def as_tensor(op):
    return sqf.qobj_to_tensor(op) if not isinstance(op, torch.Tensor) else op


def quantities(p, grad):
    cr, el = build(p, requires_grad=grad, trunc=TR)
    efreqs, evecs = cr.diag(N_EIG)
    E = efreqs
    q = [E[1] - E[0], E[2] - 2 * E[1] + E[0]]
    S = cr.S if isinstance(cr.S, torch.Tensor) else torch.as_tensor(np.asarray(cr.S))
    R = cr.R if isinstance(cr.R, torch.Tensor) else torch.as_tensor(np.asarray(cr.R))
    e, hbar = sq.unt.e, sq.unt.hbar
    nm = cr.n
    phi_ops = [as_tensor(cr._memory_ops["phi"][i]) for i in range(nm)]
    Q_ops = [as_tensor(cr._memory_ops["Q"][i]) for i in range(nm)]
    psi0, psi1 = evecs[0], evecs[1]
    zinv, zt = torch.as_tensor(ZINV, dtype=S.dtype), torch.as_tensor(ZT, dtype=R.dtype)
    A = zinv @ S                      # varphi_k = sum_i A[k, i] phi_mode_i
    B = zt @ R                        # n_app_k = sum_i B[k, i] n_mode_i
    for k in range(NM):
        On = lincomb([B[k, i] * (np.sqrt(hbar) / (2 * e)) for i in range(nm)], Q_ops)
        q.append(torch.abs(sqf.operator_inner_product(psi0, On, psi1)) ** 2)
    for k in range(NM):
        # the island column of S is the same on every node (~1e12, garbage) and cancels in Z^-1 S's differences
        Op = lincomb([A[k, i] * (2 * e / np.sqrt(hbar)) for i in range(nm)], phi_ops)
        q.append(torch.abs(sqf.operator_inner_product(psi0, Op, psi1)) ** 2)
    return cr, el, q, (A, B)


# d(leaf)/d(parameter): SQcircuit's leaves are SI (junction: angular frequency; L in H; C in F; loop flux internal)
def flux_unit():
    loop = sq.Loop(0.1, requires_grad=True)
    return float(loop.internal_value) / 0.1          # internal per Phi_0


def jacobian(p):
    cr, el, q, _ = quantities(p, grad=True)
    leaves = [el[k].internal_value for k in PARAMS]
    # the leaf of each element at p (not at P0): its conversion factor
    fac = [-float(leaves[0]) / p["E_C"], 2 * np.pi * 1e9, -float(leaves[2]) / p["E_L"], FLUX_UNIT]
    J = np.zeros((len(QNAMES), len(PARAMS)))
    for i, qi in enumerate(q):
        g = torch.autograd.grad(torch.real(qi), leaves, retain_graph=True, allow_unused=True)
        J[i] = [0.0 if gj is None else float(gj) * fac[j] for j, gj in enumerate(g)]
    return np.array([float(torch.real(x).detach()) for x in q]), J


def values(p):
    with torch.no_grad():
        _, _, q, _ = quantities(p, grad=False)
    return np.array([float(torch.real(x)) for x in q])


FLUX_UNIT = flux_unit()


def main():
    print(f"SQcircuit {sq.__file__}; truncation {TR}; n_eig {N_EIG}; loop flux internal per Phi_0 = {FLUX_UNIT}")
    t = time.time(); v0, J = jacobian(P0); t_jac = time.time() - t
    t = time.time(); values(P0); t_val = time.time() - t
    print(f"one solve {t_val:.1f} s; solve + autodiff Jacobian {t_jac:.1f} s")
    for i, n in enumerate(QNAMES):
        print(f"{n:12s} = {v0[i]:+.9e}   J: " + " ".join(f"{x:+.4e}" for x in J[i]))

    step = {k: (1e-4 if k == "flux" else 1e-4 * P0[k]) for k in PARAMS}
    t = time.time()
    Jfd = np.zeros((len(QNAMES), len(PARAMS)))
    for a, k in enumerate(PARAMS):
        pp = dict(P0); pm = dict(P0); pp[k] += step[k]; pm[k] -= step[k]
        Jfd[:, a] = (values(pp) - values(pm)) / (2 * step[k])
    t_fd = time.time() - t
    dev = np.max(np.abs(J - Jfd) / np.max(np.abs(Jfd), axis=1, keepdims=True))
    print(f"autodiff Jacobian vs central differences of values: max deviation {dev:.1e} of each row's largest entry ({t_fd:.0f} s)")

    t = time.time()
    H = np.zeros((len(QNAMES), len(PARAMS), len(PARAMS)))
    for a, k in enumerate(PARAMS):
        pp = dict(P0); pm = dict(P0); pp[k] += step[k]; pm[k] -= step[k]
        H[:, :, a] = (jacobian(pp)[1] - jacobian(pm)[1]) / (2 * step[k])
    H = (H + H.transpose(0, 2, 1)) / 2
    t_hess = time.time() - t
    print(f"Hessian by central differences of the autodiff Jacobian: {t_hess:.0f} s")
    out = Path(__file__).resolve().parent / "results" / (f"sqcircuit_neig{N_EIG}_smoke.json" if SMOKE else f"sqcircuit_neig{N_EIG}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"params": PARAMS, "quantities": QNAMES, "truncation": TR, "n_eig": N_EIG, "values": v0.tolist(),
                               "jacobian": J.tolist(), "jacobian_fd": Jfd.tolist(), "hessian": H.tolist(),
                               "seconds_solve": t_val, "seconds_solve_and_jacobian": t_jac, "seconds_fd_jacobian": t_fd,
                               "seconds_hessian": t_hess}, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
