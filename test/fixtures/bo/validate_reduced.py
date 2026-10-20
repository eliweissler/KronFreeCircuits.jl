"""Validate test/fixtures/bo/*.json with circuit-quantization, and record Python's numbers for them.

The reduced specs in this directory are written by KronFreeCircuits' `save_spec`
from Born-Oppenheimer fits: compact_reduced.json and oscillators_reduced.json by
test/fixtures/bo/write_reduced.jl, zero_pi_reduced.json (the Di Paolo zero-pi fit)
by bench/born_oppenheimer/zero_pi.jl.
This script loads each with circuit_quantization.load_spec and builds its
Hamiltonian with `.hamiltonian()`, so a file Python's format validator rejects
(or cannot assemble) is caught here, once, and never at test time.

It then evaluates each Hamiltonian numerically from that SymPy expression alone, at the
parameter point in POINTS, and writes test/fixtures/bo_reference/python_reduced_levels.json:
- the lowest LEVELS eigenvalues, in the charge basis |m>, m = -NCUT..NCUT, for a compact
  mode (e^{ik varphi}|m> = |m+k>), and on a spectral (Fourier) grid of GRID points on
  [-HALF_WIDTH, HALF_WIDTH) for an extended mode;
- the Hamiltonian as a function, at a few varphi and at n = n_g (the classical potential V) and at the fixed charge
  n = N_SAMPLE (the kinetic term too).
test/test_bo.jl compares the levels against KronFreeCircuits' spectrum and the function samples against the canonical
form rebuilt from the loaded Spec's fields, so a mismatch in how the two packages read a file (a sign, an offset, a
factor, the sign of n_g) fails a Julia test. The function samples matter: complex conjugation in the charge basis flips
every cosine offset at once, and for one mode parity followed by conjugation maps n_g to -n_g, both leaving the spectrum
unchanged, so eigenvalues alone pin neither the sine-term sign nor the n_g sign. Neither check pins the orientation of
KronFreeCircuits' own charge-basis shift operator (e^{ik varphi}|m> = |m+k> or |m-k>), which the spectrum also cannot see;
the package's operator tests do. The extended-mode path is exercised only by a harmonic oscillator here (no cosines, no
offset charge), so the grid's n and e^{ik varphi} branches are not yet cross-checked.

Run from the repository root, only after regenerating the committed files:

    /opt/anaconda3/envs/circuit_quantization/bin/python test/fixtures/bo/validate_reduced.py

Python is never a test-time dependency of KronFreeCircuits.

Prints `OK <file>` per file; exits nonzero on the first exception.
The format authority is "Python interface and Julia handoff" in
~/src/circuit-quantization/docs/spec-format.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sym

import circuit_quantization as cq

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "bo_reference" / "python_reduced_levels.json"

LEVELS = 6
NCUT = 40                       # compact modes: charge states -NCUT..NCUT
GRID, HALF_WIDTH = 256, 16.0    # extended modes: spectral grid
POTENTIAL_POINTS = [-2.5, -1.0, -0.3, 0.0, 0.4, 1.3, 2.9]
N_SAMPLE = 1.3                  # a charge away from every file's n_g

# The free parameters of each committed file, at a generic point (nonzero offsets and flux).
POINTS = {
    "compact_reduced.json": {"E_Cs": 0.05, "n_gs": 0.1, "varphi_ext": 0.7},
    "oscillators_reduced.json": {"E_Cs": 0.5},
    "zero_pi_reduced.json": {"E_Ctheta": 0.007, "n_g": 0.1, "varphi_ext": 0.7},
}


def _decompose(term, n, phi):
    """A term of the expanded, exponential-form Hamiltonian as (coefficient, power of n, power of phi, k of e^{ik phi})."""
    coeff, a, b, k = sym.Integer(1), 0, 0, 0
    for f in sym.Mul.make_args(term):
        base, power = f.as_base_exp()
        if base == n:
            a += int(power)
        elif base == phi:
            b += int(power)
        elif isinstance(f, sym.exp):
            arg = sym.expand(f.args[0])
            kk = sym.simplify(arg.coeff(phi) / sym.I)
            if not kk.is_number or kk != int(kk):
                raise ValueError(f"exponent {arg} is not i k varphi with integer k")
            k += int(kk)
            coeff *= sym.exp(arg - kk * sym.I * phi)
        elif f.free_symbols & {n, phi}:
            raise ValueError(f"unsupported factor {f}")
        else:
            coeff *= f
    return complex(sym.N(coeff, 30)), a, b, k


def _charge_operator(terms):
    m = np.arange(-NCUT, NCUT + 1)
    H = np.zeros((m.size, m.size), dtype=complex)
    for c, a, b, k in terms:
        if b:
            raise ValueError("a compact mode cannot carry a polynomial in varphi")
        if a and k:
            raise ValueError("a mixed n^a e^{ik varphi} term is ambiguous in a commutative expression")
        if k == 0:
            H += c * np.diag(m.astype(float) ** a)
        else:
            H += c * np.eye(m.size, k=-k)      # <m+k| e^{ik varphi} |m> = 1
    return H


def _grid_operator(terms):
    x = -HALF_WIDTH + 2 * HALF_WIDTH * np.arange(GRID) / GRID
    q = 2 * np.pi * np.fft.fftfreq(GRID, d=2 * HALF_WIDTH / GRID)     # n = -i d/dvarphi in Fourier space
    F = np.fft.fft(np.eye(GRID), axis=0)
    Finv = np.fft.ifft(np.eye(GRID), axis=0)
    N = {a: Finv @ np.diag(q.astype(complex) ** a) @ F for a in (1, 2)}
    H = np.zeros((GRID, GRID), dtype=complex)
    for c, a, b, k in terms:
        if a and (b or k):
            raise ValueError("a mixed n-varphi term is ambiguous in a commutative expression")
        if a:
            H += c * N[a]
        else:
            H += c * np.diag(x ** b * np.exp(1j * k * x))
    return H


def _numbers(spec, point):
    names = [str(p) for p in spec.parameters]
    missing = sorted(set(names) - set(point))
    if missing:
        raise ValueError(f"no value for free parameters {missing}")
    subs = {p: point[str(p)] for p in spec.parameters}
    n, phi = sym.symbols("n_1 varphi_1", real=True)
    H = spec.hamiltonian().subs(subs)
    terms = [_decompose(t, n, phi) for t in sym.Add.make_args(sym.expand(H.rewrite(sym.exp)))]
    compact = spec.modes[0][1] == "c"
    M = _charge_operator(terms) if compact else _grid_operator(terms)
    if np.max(np.abs(M - M.conj().T)) > 1e-10:
        raise ValueError("the assembled operator is not Hermitian")
    levels = np.linalg.eigvalsh((M + M.conj().T) / 2)[:LEVELS]
    ng = complex(sym.N(spec.n_g[0].subs(subs))).real
    V = sym.lambdify(phi, H.subs(n, ng), "numpy")
    potential = [complex(V(x)).real for x in POTENTIAL_POINTS]
    Hn = sym.lambdify(phi, H.subs(n, N_SAMPLE), "numpy")
    at_n = [complex(Hn(x)).real for x in POTENTIAL_POINTS]
    basis = {"charge_cutoff": NCUT} if compact else {"spectral_grid": GRID, "half_width": HALF_WIDTH}
    return {"point": point, "basis": basis, "levels": levels.tolist(),
            "potential": {"varphi": POTENTIAL_POINTS, "V": potential},
            "at_charge": {"n": N_SAMPLE, "H": at_n}}


def main() -> int:
    files = sorted(HERE.glob("*.json"))
    if not files:
        print(f"no *.json files in {HERE}", file=sys.stderr)
        return 1
    results = {}
    for path in files:
        spec = cq.load_spec(path)
        spec.hamiltonian()
        if len(spec.modes) != 1:
            raise ValueError(f"{path.name}: expected a one-mode reduced spec")
        if path.name not in POINTS:
            raise ValueError(f"{path.name}: add its parameter point to POINTS")
        results[path.name] = _numbers(spec, POINTS[path.name])
        print(f"OK {path.name}")
    OUT.write_text(json.dumps({"_comment": "Written by test/fixtures/bo/validate_reduced.py from circuit_quantization; "
                                           "compared against KronFreeCircuits in test/test_bo.jl.",
                               "levels": LEVELS, "files": results}, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(HERE.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
