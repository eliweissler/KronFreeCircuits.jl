"""Matched timing of scqubits' hierarchical diagonalization of the Hensel binary tree (local frame), cold at every
point of the six-point coupler-flux grid.

Solvers as in KronFreeCircuits' tree: dense (scipy eigh) at the leaves and pair nodes; at M and the root, scipy's
LOBPCG with our LOBPCG(preconditioner=:jacobi) settings: block of m + 4 guard columns, product-state start (unit vectors
on the lowest diagonal entries), shifted Jacobi preconditioner 1/(D − min D + 1), tol 1e-6, at most 3000 iterations, real
arithmetic when the node's matrix is real, complex otherwise (scqubits leaves eigenvector phases unfixed). Every solver call is timed, so a point's time splits into solves and the rest
(scqubits' Hamiltonian and operator assembly). Threads: OMP/OPENBLAS/MKL_NUM_THREADS (set to 8 by the caller).

   python matched.py [kM] [solver: lobpcg | arpack | torch | torch-cuda | default] [frame: shear | local]

solver: lobpcg = scipy's LOBPCG at M and the root; arpack = scipy's sparse eigsh there (scqubits' default solver for
those nodes, with the pair nodes forced dense); torch / torch-cuda = torch.lobpcg on the CPU / the GPU there, same
settings, with torch's relative test (residual / (‖AX‖/‖X‖ + |E|), _lobpcg.py) given the tolerance that implies our
absolute 1e-6 GHz residual; torch.lobpcg returns wrong values on complex Hermitian input, so a complex node (scqubits
leaves eigenvector phases unfixed; here at Φc = 0 and 0.5) falls back to scipy's LOBPCG, and each call records which ran;
default = scqubits exactly as shipped: no solver installed anywhere, only the total time per point is measured;
dense-M = dense eigh at M (it computes 120 to 240 of its 3600 states) and ARPACK at the root: scqubits' likely best setting.
MATCHED_FLUXES (comma list) runs only those coupler fluxes.
frame: shear (the default) is the benchmark's frame.
"""

import json
import sys
import time

import numpy as np
import scipy as sp
import scqubits as scq
from scipy.sparse.linalg import LinearOperator, lobpcg
from scqubits.core import diag

from hensel_scq import binary_tree, circuit

scq.settings.STENCIL = 9
KM = int(sys.argv[1]) if len(sys.argv) > 1 else 240
SOLVER = sys.argv[2] if len(sys.argv) > 2 else "lobpcg"
FRAME = sys.argv[3] if len(sys.argv) > 3 else "shear"
import os
FLUXES = [float(x) for x in os.environ.get("MATCHED_FLUXES", "0,0.1,0.2,0.3,0.4,0.5").split(",")]
GUARD, TOL, MAXITER = 4, 1e-6, 3000
LOG = []


COMPLEX = []


def _matrix(m):
    """The node's matrix as scipy CSR, real when it is real. scqubits does not fix the phases of a subsystem's
    eigenvectors, so the operators it passes to a parent are complex Hermitian even when the physics is real; such a
    node is solved in complex arithmetic, as scqubits itself would."""
    m = diag._cast_matrix(m, "sparse")
    if np.iscomplexobj(m.data):
        if abs(m.data.imag).max(initial=0) < 1e-12:
            m = m.real
        else:
            COMPLEX.append(m.shape[0])
    return sp.sparse.csr_matrix(m)


def lobpcg_esys(matrix, evals_count, **kw):
    t = time.perf_counter()
    A = _matrix(matrix)
    n, k = A.shape[0], evals_count
    b = k + GUARD
    D = A.diagonal().real
    X = np.zeros((n, b), dtype=A.dtype)
    X[np.argsort(D, kind="stable")[:b], np.arange(b)] = 1.0
    inv = 1.0 / (D - (D.min() - 1.0))
    M = LinearOperator((n, n), matvec=lambda v: inv * v.ravel(), matmat=lambda V: inv[:, None] * V, dtype=A.dtype)
    w, V, rh = lobpcg(A, X, M=M, tol=TOL, maxiter=MAXITER, largest=False, retResidualNormsHistory=True)
    o = np.argsort(w)[:k]
    LOG.append(dict(kind="lobpcg", n=n, k=k, seconds=time.perf_counter() - t, iterations=len(rh),
                    complex=bool(np.iscomplexobj(A.data))))
    E, V = w[o], V[:, o]
    return (E, diag._convert_evecs_to_qobjs(V, matrix) if isinstance(matrix, scq.utils.misc.qt.Qobj) else V)


def timed(fn, kind):
    def f(matrix, evals_count, **kw):
        t = time.perf_counter()
        r = fn(matrix, evals_count, **kw)
        LOG.append(dict(kind=kind, n=matrix.shape[0], k=evals_count, seconds=time.perf_counter() - t))
        return r
    return f


def lobpcg_evals(matrix, evals_count, **kw):
    return lobpcg_esys(matrix, evals_count, **kw)[0]


def torch_esys(matrix, evals_count, device="cpu", **kw):
    import torch
    A = _matrix(matrix)
    if np.iscomplexobj(A.data):                              # torch.lobpcg is wrong on complex Hermitian input
        E, V = lobpcg_esys(matrix, evals_count)
        LOG[-1]["kind"] = "lobpcg (complex, torch fallback)"
        return E, V
    t = time.perf_counter()
    n, k = A.shape[0], evals_count
    b = k + GUARD
    D = A.diagonal()
    C = A.tocoo()
    At = torch.sparse_coo_tensor(np.vstack([C.row, C.col]), C.data, (n, n), dtype=torch.float64, device=device).coalesce()
    inv = 1.0 / (D - (D.min() - 1.0))
    iK = torch.sparse_coo_tensor(np.vstack([np.arange(n)] * 2), inv, (n, n), dtype=torch.float64, device=device).coalesce()
    X = torch.zeros(n, b, dtype=torch.float64, device=device)
    sel = np.argsort(D, kind="stable")[:b]
    X[sel, np.arange(b)] = 1.0
    A_norm = float(torch.linalg.vector_norm(torch.sparse.mm(At, X))) / float(torch.linalg.vector_norm(X))
    rel_tol = TOL / (A_norm + float(np.abs(D[sel]).max()))  # implies |r| < TOL for every column
    w, V = torch.lobpcg(At, k=b, X=X, iK=iK, largest=False, tol=rel_tol, niter=MAXITER)
    if device == "cuda":
        torch.cuda.synchronize()
    w, V = w.cpu().numpy(), V.cpu().numpy()
    o = np.argsort(w)[:k]
    LOG.append(dict(kind=f"torch-{device}", n=n, k=k, seconds=time.perf_counter() - t, complex=False))
    E, V = w[o], V[:, o]
    return (E, diag._convert_evecs_to_qobjs(V, matrix) if isinstance(matrix, scq.utils.misc.qt.Qobj) else V)


def install(sys_, depth):
    """Dense at the pair nodes (depth 2; P(q2, f2) sits at depth 1 but is a pair node), the chosen iterative solver at
    the root (0) and M (1). The single-variable leaves keep scqubits' default solvers: overriding them makes scqubits
    diagonalize a harmonic leaf's truncated matrix numerically instead of exactly, which shifted the root levels by up
    to 57 MHz in testing (our leaves are exact dense solves, so scqubits' default is the matched choice)."""
    subs = getattr(sys_, "subsystems", []) or []
    for i, sub in enumerate(subs):
        install(sub, depth + 1 if not (depth == 0 and i == 1) else 2)
    if not subs or SOLVER == "default":
        return
    if depth <= 1:
        if SOLVER in ("torch", "torch-cuda"):
            dev = "cuda" if SOLVER == "torch-cuda" else "cpu"
            sys_.esys_method = lambda m, k, **kw: torch_esys(m, k, device=dev)
            sys_.evals_method = lambda m, k, **kw: torch_esys(m, k, device=dev)[0]
        elif SOLVER == "dense-M":
            if depth == 1:
                sys_.evals_method = timed(diag.evals_scipy_dense, "dense M")
                sys_.esys_method = timed(diag.esys_scipy_dense, "dense M")
            else:
                sys_.evals_method = timed(diag.evals_scipy_sparse, "arpack")
                sys_.esys_method = timed(diag.esys_scipy_sparse, "arpack")
        elif SOLVER == "lobpcg":
            sys_.evals_method, sys_.esys_method = lobpcg_evals, lobpcg_esys
        else:
            sys_.evals_method = timed(diag.evals_scipy_sparse, "arpack")
            sys_.esys_method = timed(diag.esys_scipy_sparse, "arpack")
    else:
        sys_.evals_method = timed(diag.evals_scipy_dense, "dense")
        sys_.esys_method = timed(diag.esys_scipy_dense, "dense")


t0 = time.perf_counter()
c = binary_tree(circuit(FRAME), kM=KM)
install(c, 0)
setup = time.perf_counter() - t0
print(f"scqubits {scq.__version__}, frame {FRAME}, kM {KM}, solver {SOLVER}: setup {setup:.1f} s", flush=True)

points = []
for fc in FLUXES:
    c.Φ1, c.Φ2, c.Φ3 = 0.5, 0.5, fc
    LOG.clear()
    t = time.perf_counter()
    E = np.sort(c.eigenvals(evals_count=8))
    total = time.perf_counter() - t
    solves = sum(r["seconds"] for r in LOG)
    by = {}
    for r in LOG:
        key = f'{r["kind"]} n={r["n"]}'
        by[key] = by.get(key, 0.0) + r["seconds"]
    points.append(dict(flux=fc, energies=list(E), seconds=total, solve_seconds=solves, calls=list(LOG)))
    print(f"Φc = {fc}: {total:6.2f} s (solves {solves:6.2f} s, rest {total - solves:6.2f} s)  E1-E0 = {E[1] - E[0]:.6f}  "
          + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:4]), flush=True)

json.dump(dict(scqubits=scq.__version__, frame=FRAME, kM=KM, solver=SOLVER, setup_seconds=setup, points=points),
          open(f"matched-scqubits-{FRAME}-kM{KM}-{SOLVER}" + ("" if len(FLUXES) == 6 else "-" + "_".join(map(str, FLUXES))) + ".json", "w"), indent=1)
