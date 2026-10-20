"""Validate test/fixtures/bo/*.json with circuit-quantization.

The reduced specs in this directory are written by KronFreeCircuits' `save_spec`
from Born-Oppenheimer fits: compact_reduced.json and oscillators_reduced.json by
test/fixtures/bo/write_reduced.jl, zero_pi_reduced.json (the Di Paolo zero-pi fit)
by bench/born_oppenheimer/zero_pi.jl.
This script loads each with circuit_quantization.load_spec and builds its
Hamiltonian with `.hamiltonian()`, so a file Python's format validator rejects
(or cannot assemble) is caught here, once, and never at test time.

Run from the repository root, only after regenerating the committed files:

    /opt/anaconda3/envs/circuit_quantization/bin/python test/fixtures/bo/validate_reduced.py

Python is never a test-time dependency of KronFreeCircuits.

Prints `OK <file>` per file; exits nonzero on the first exception.
The format authority is "Python interface and Julia handoff" in
~/src/circuit-quantization/docs/spec-format.md.
"""

import sys
from pathlib import Path

import circuit_quantization as cq

HERE = Path(__file__).resolve().parent


def main() -> int:
    files = sorted(HERE.glob("*.json"))
    if not files:
        print(f"no *.json files in {HERE}", file=sys.stderr)
        return 1
    for path in files:
        cq.load_spec(path).hamiltonian()
        print(f"OK {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
