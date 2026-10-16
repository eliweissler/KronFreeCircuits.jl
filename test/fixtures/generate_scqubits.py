"""Generate scqubits reference energies for the KronFreeCircuits test suite.

Run once from the repository root:

    python test/fixtures/generate_scqubits.py

and commit the output, test/fixtures/scqubits_reference.json.
All energies are absolute eigenvalues in GHz, ascending.
Parameters are stored under their Julia-side names (see the fixture specs);
scqubits' `flux` (in flux quanta) is stored as `varphi_ext = 2*pi*flux`.
"""

import json
import os
import sys

import numpy as np
import scipy
import scipy.sparse as sparse
import qutip
import scqubits
import scqubits.utils.spectrum_utils as _spectrum_utils

# Compatibility shim: scqubits 4.3.1 with qutip >= 5.3 gets a scipy csc_array (not csc_matrix)
# from Qobj_to_scipy_csc_matrix, which its own convert_operator_to_qobj then rejects inside
# HilbertSpace.  Coerce the result to csc_matrix; the numbers are unchanged.
_qobj_to_csc = _spectrum_utils.Qobj_to_scipy_csc_matrix
_spectrum_utils.Qobj_to_scipy_csc_matrix = lambda q: sparse.csc_matrix(_qobj_to_csc(q))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scqubits_reference.json")


def floats(x):
    return [float(v) for v in np.sort(np.asarray(x, dtype=float))]


def fluxonium():
    p = dict(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=200)
    q = scqubits.Fluxonium(**p, truncated_dim=6)
    return {
        "scqubits_class": "Fluxonium",
        "scqubits_params": p,
        "params": {"E_J": p["EJ"], "E_C": p["EC"], "E_L": p["EL"], "varphi_ext": 2 * np.pi * p["flux"]},
        "energies": floats(q.eigenvals(evals_count=6)),
    }


def transmon():
    p = dict(EJ=30.0, EC=1.2, ng=0.25, ncut=30)
    q = scqubits.Transmon(**p, truncated_dim=5)
    return {
        "scqubits_class": "Transmon",
        "scqubits_params": p,
        "params": {"E_J": p["EJ"], "E_C": p["EC"], "n_g": p["ng"]},
        "energies": floats(q.eigenvals(evals_count=5)),
    }


def two_transmons():
    p1 = dict(EJ=30.0, EC=1.2, ng=0.25, ncut=30)
    p2 = dict(EJ=25.0, EC=1.0, ng=0.0, ncut=30)
    g = 0.3
    t1 = scqubits.Transmon(**p1, truncated_dim=8)
    t2 = scqubits.Transmon(**p2, truncated_dim=8)
    # g (n1 - ng1)(n2 - ng2), from explicit native-basis matrices.
    m1 = t1.n_operator() - p1["ng"] * np.eye(t1.hilbertdim())
    m2 = t2.n_operator() - p2["ng"] * np.eye(t2.hilbertdim())
    term = scqubits.InteractionTerm(g_strength=g, operator_list=[(0, m1), (1, m2)])
    hs = scqubits.HilbertSpace([t1, t2], interaction_list=[term])
    return {
        "scqubits_class": "HilbertSpace[Transmon, Transmon]",
        "scqubits_params": {"transmon1": p1, "transmon2": p2, "g": g, "truncated_dim": 8,
                            "interaction": "g*(n1 - ng1*I) (x) (n2 - ng2*I)"},
        "params": {"E_J1": p1["EJ"], "E_C1": p1["EC"], "n_g1": p1["ng"],
                   "E_J2": p2["EJ"], "E_C2": p2["EC"], "n_g2": p2["ng"], "g": g},
        "energies": floats(hs.eigenvals(evals_count=20)),
    }


ZP = dict(EJ=10.0, EL=0.04, ECJ=20.0, EC=None, ECS=0.5, ng=0.1, flux=0.23, ncut=15)
GRID = dict(min_val=-6 * np.pi, max_val=6 * np.pi, pt_count=1001)


def zp_julia_params(p, suffix=""):
    return {f"E_J{suffix}": p["EJ"], f"E_L{suffix}": p["EL"], f"E_CJ{suffix}": p["ECJ"],
            f"E_CS{suffix}": p["ECS"], f"varphi_ext{suffix}": 2 * np.pi * p["flux"],
            f"n_g{suffix}": p["ng"]}


def zero_pi_obj(truncated_dim):
    return scqubits.ZeroPi(**ZP, grid=scqubits.Grid1d(**GRID), truncated_dim=truncated_dim)


def zero_pi():
    q = zero_pi_obj(8)
    return {
        "scqubits_class": "ZeroPi",
        "scqubits_params": {**ZP, "grid": GRID},
        "params": zp_julia_params(ZP),
        "energies": floats(q.eigenvals(evals_count=8)),
    }


def full_zero_pi():
    p = dict(EJ=10.0, EL=0.04, ECJ=20.0, EC=None, ECS=0.5, dEJ=0.05, dCJ=0.05, dC=0.08,
             dEL=0.1, ng=0.1, flux=0.23, ncut=15, zeropi_cutoff=30, zeta_cutoff=20)
    q = scqubits.FullZeroPi(**p, grid=scqubits.Grid1d(**GRID), truncated_dim=10)
    E_Cz = float(q.EC)
    params = zp_julia_params(p)
    params.update({"E_Cz": E_Cz, "dE_J": p["dEJ"], "dC_J": p["dCJ"], "dC": p["dC"], "dE_L": p["dEL"]})
    return {
        "scqubits_class": "FullZeroPi",
        "scqubits_params": {**p, "grid": GRID},
        "params": params,
        "derived": {"EC": E_Cz, "E_zeta": float(q.E_zeta)},
        "energies": floats(q.eigenvals(evals_count=10)),
    }


def two_zero_pi():
    g = 0.05
    z1, z2 = zero_pi_obj(10), zero_pi_obj(10)
    # g (n_theta1 + ng1)(n_theta2 + ng2), from explicit native-basis matrices.
    m1 = z1.n_theta_operator() + ZP["ng"] * sparse.identity(z1.hilbertdim(), format="csc")
    m2 = z2.n_theta_operator() + ZP["ng"] * sparse.identity(z2.hilbertdim(), format="csc")
    term = scqubits.InteractionTerm(g_strength=g, operator_list=[(0, m1), (1, m2)])
    hs = scqubits.HilbertSpace([z1, z2], interaction_list=[term])
    params = {**zp_julia_params(ZP, "1"), **zp_julia_params(ZP, "2"), "g": g}
    return {
        "scqubits_class": "HilbertSpace[ZeroPi, ZeroPi]",
        "scqubits_params": {"zero_pi1": {**ZP, "grid": GRID}, "zero_pi2": {**ZP, "grid": GRID},
                            "g": g, "truncated_dim": 10,
                            "interaction": "g*(n_theta1 + ng1*I) (x) (n_theta2 + ng2*I)"},
        "params": params,
        "energies": floats(hs.eigenvals(evals_count=20)),
    }


ZP_WIDE = {**ZP, "ncut": 20}
GRID_WIDE = dict(min_val=-10 * np.pi, max_val=10 * np.pi, pt_count=1601)


def zero_pi_wide():
    """Converged zero-pi spectrum (grid wide and fine enough that the stencil no longer matters)."""
    q = scqubits.ZeroPi(**ZP_WIDE, grid=scqubits.Grid1d(**GRID_WIDE), truncated_dim=8)
    return {
        "scqubits_class": "ZeroPi",
        "scqubits_params": {**ZP_WIDE, "grid": GRID_WIDE},
        "params": zp_julia_params(ZP_WIDE),
        "stencil": scqubits.settings.STENCIL,
        "energies": floats(q.eigenvals(evals_count=8)),
    }


# Small grids for the default suite: the same physics and ±6π range as above, small enough to
# diagonalize densely on the Julia side.  scqubits' Grid1d derivative matrices are STENCIL-point
# symmetric finite differences with truncated edge rows on np.linspace(min_val, max_val, pt_count):
# the same discretization as the Julia side's PhaseGrid(min_val, max_val, pt_count; order=STENCIL - 1).
ZP_SMALL = {**ZP, "ncut": 6}
GRID_SMALL = dict(min_val=-6 * np.pi, max_val=6 * np.pi, pt_count=101)


def zero_pi_small():
    q = scqubits.ZeroPi(**ZP_SMALL, grid=scqubits.Grid1d(**GRID_SMALL), truncated_dim=8)
    return {
        "scqubits_class": "ZeroPi",
        "scqubits_params": {**ZP_SMALL, "grid": GRID_SMALL},
        "params": zp_julia_params(ZP_SMALL),
        "stencil": scqubits.settings.STENCIL,
        "energies": floats(q.eigenvals(evals_count=8)),
    }


def full_zero_pi_small():
    p = dict(EJ=10.0, EL=0.04, ECJ=20.0, EC=None, ECS=0.5, dEJ=0.05, dCJ=0.05, dC=0.08,
             dEL=0.1, ng=0.1, flux=0.23, ncut=6, zeropi_cutoff=30, zeta_cutoff=20)
    q = scqubits.FullZeroPi(**p, grid=scqubits.Grid1d(**GRID_SMALL), truncated_dim=10)
    E_Cz = float(q.EC)
    params = zp_julia_params(p)
    params.update({"E_Cz": E_Cz, "dE_J": p["dEJ"], "dC_J": p["dCJ"], "dC": p["dC"], "dE_L": p["dEL"]})
    return {
        "scqubits_class": "FullZeroPi",
        "scqubits_params": {**p, "grid": GRID_SMALL},
        "params": params,
        "derived": {"EC": E_Cz, "E_zeta": float(q.E_zeta)},
        "stencil": scqubits.settings.STENCIL,
        "energies": floats(q.eigenvals(evals_count=10)),
    }


def two_zero_pi_small(ng=ZP_SMALL["ng"]):
    """Two coupled small zero-pi qubits; `ng` (both qubits) defaults to the shared value."""
    g = 0.05
    zp = {**ZP_SMALL, "ng": ng}
    z1, z2 = (scqubits.ZeroPi(**zp, grid=scqubits.Grid1d(**GRID_SMALL), truncated_dim=10) for _ in range(2))
    # g (n_theta1 + ng1)(n_theta2 + ng2), from explicit native-basis matrices.
    m1 = z1.n_theta_operator() + zp["ng"] * sparse.identity(z1.hilbertdim(), format="csc")
    m2 = z2.n_theta_operator() + zp["ng"] * sparse.identity(z2.hilbertdim(), format="csc")
    term = scqubits.InteractionTerm(g_strength=g, operator_list=[(0, m1), (1, m2)])
    hs = scqubits.HilbertSpace([z1, z2], interaction_list=[term])
    params = {**zp_julia_params(zp, "1"), **zp_julia_params(zp, "2"), "g": g}
    return {
        "scqubits_class": "HilbertSpace[ZeroPi, ZeroPi]",
        "scqubits_params": {"zero_pi1": {**zp, "grid": GRID_SMALL}, "zero_pi2": {**zp, "grid": GRID_SMALL},
                            "g": g, "truncated_dim": 10,
                            "interaction": "g*(n_theta1 + ng1*I) (x) (n_theta2 + ng2*I)"},
        "params": params,
        "stencil": scqubits.settings.STENCIL,
        "energies": floats(hs.eigenvals(evals_count=20)),
    }


def main():
    out = {
        "_comment": "Generated by test/fixtures/generate_scqubits.py; energies absolute, GHz, ascending.",
        "scqubits_version": scqubits.__version__,
        "qutip_version": qutip.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scqubits_stencil": scqubits.settings.STENCIL,
        "python_executable": sys.executable,
        "fluxonium": fluxonium(),
        "transmon": transmon(),
        "two_transmons": two_transmons(),
        "zero_pi": zero_pi(),
        "full_zero_pi": full_zero_pi(),
        "two_zero_pi": two_zero_pi(),
        "zero_pi_wide": zero_pi_wide(),
        "zero_pi_small": zero_pi_small(),
        "full_zero_pi_small": full_zero_pi_small(),
        "two_zero_pi_small": two_zero_pi_small(),
        "two_zero_pi_small_ng0": two_zero_pi_small(ng=0.0),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
