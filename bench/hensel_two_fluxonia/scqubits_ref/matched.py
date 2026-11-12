"""Matched timing of scqubits' hierarchical diagonalization of the Hensel binary tree (local frame), cold at every
point of the six-point coupler-flux grid.

Solvers as in KronFreeCircuits' tree: dense (scipy eigh) at the leaves and pair nodes; at M and the root, scipy's
LOBPCG with our LOBPCG(preconditioner=:jacobi) settings: block of m + 4 guard columns, product-state start (unit vectors
on the lowest diagonal entries), shifted Jacobi preconditioner 1/(D − min D + 1), tol 1e-6, at most 3000 iterations, real
arithmetic when the node's matrix is real, complex otherwise (scqubits leaves eigenvector phases unfixed). Every solver call is timed, so a point's time splits into solves and the rest
(scqubits' Hamiltonian and operator assembly). Threads: OMP/OPENBLAS/MKL_NUM_THREADS (set to 8 by the caller).

   python matched.py [kM] [solver: lobpcg | arpack] [frame: shear | local]
   (arpack: scqubits' default sparse eigsh at M and the root; shear, the default, is the benchmark's frame)
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
FLUXES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
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


def install(sys_, depth):
    """Dense at the pair nodes (depth 2; P(q2, f2) sits at depth 1 but is a pair node), the chosen iterative solver at
    the root (0) and M (1). The single-variable leaves keep scqubits' default solvers: overriding them makes scqubits
    diagonalize a harmonic leaf's truncated matrix numerically instead of exactly, which shifted the root levels by up
    to 57 MHz in testing (our leaves are exact dense solves, so scqubits' default is the matched choice)."""
    subs = getattr(sys_, "subsystems", []) or []
    for i, sub in enumerate(subs):
        install(sub, depth + 1 if not (depth == 0 and i == 1) else 2)
    if not subs:
        return
    if depth <= 1:
        if SOLVER == "lobpcg":
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
          open(f"matched-scqubits-{FRAME}-kM{KM}-{SOLVER}.json", "w"), indent=1)
