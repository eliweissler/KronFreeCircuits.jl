# Brief: pre-changes to circuit-quantization for the Julia interface (2026-11-10)

This brief is self-contained.
You are working in the Python package at `~/src/circuit-quantization` (editable install in the conda environment `circuit_quantization`; run tests with that environment's `pytest`).
Do not modify any other repository.
Work on a new branch, keep each numbered change in its own commit with its tests, and do not push.

## Why

A Julia solver, KronFreeCircuits.jl, will soon call this package directly through PythonCall instead of reading JSON files written by a separate Python session.
Its main loop varies the residual coordinate transformation `S` (`φ = S φ'`) hundreds of times at fixed numeric parameters and needs the Hamiltonian in each frame cheaply.
Julia hands over `Float64` arrays, and three places in this package assume exact SymPy input or lack the needed public function.
Changes 1 to 3 are required; change 4 is wanted but can be a separate branch if it grows.

## Change 1: accept an integer-valued float compact block

**Now.** `_check_residual` (`src/circuit_quantization/modes.py`, about line 180) requires every entry of the compact block `S_cc` to satisfy `_is_integer`, which is `sym.simplify(x).is_integer`.
For a SymPy `Float` that is `None`, so `Result.transform` with a NumPy float `S` fails with "S_cc must lie in GL(N_c, Z)" even for the identity.

**Wanted.** An entry of the compact block that is a float within a tight absolute tolerance of an integer (suggest `1e-12`) is replaced by that exact integer before the check, and the transformation proceeds with the exact value.
A float that is not near an integer is still rejected, and the message names the entry and its value.
Exact inputs behave exactly as before; in particular `1 + 10**-20` given exactly is still rejected (see the comment on `_is_integer`).

**Tests.** A float identity and a float `S` with a nontrivial unimodular compact block succeed on a circuit with compact modes; `S_cc = 1.0000001` and `S_cc = 0.5` fail; existing tests in `tests/test_modes.py` pass unchanged.

## Change 2: `to_spec` must survive float round-off asymmetry

**Now.** `to_spec` (`src/circuit_quantization/io.py`) re-validates its output through `load_spec`, whose `_check_symmetric` compares `M[i][j]` with `M[j][i]` at three sample points with a relative tolerance of `1e-12` and no absolute floor.
After `numerify` and a float `S`, round-off gives pairs such as `E_L[1,3] = -1.09e-26/pi^2` against `E_L[3,1] = 0`, and `to_spec` raises "must be symmetric".
One downstream script assembles its spec dict by hand to avoid this.

**Wanted.** Fix it on the export side and leave the format rule alone: the loader's check is mirrored by the Julia parser and must not be loosened.
When `E_C` or `E_L` is fully numeric (no free symbols after substitution), `to_spec` symmetrizes it before printing, `(M + Mᵀ)/2`, provided the asymmetry is at round-off level relative to the matrix scale (suggest `|M_ij − M_ji| ≤ 1e-10 · max|M|`); a larger asymmetry is an error that reports the pair.
Entries that are round-off zeros on both sides should print as exactly `0`.
Symbolic matrices are untouched.

**Tests.** The failing case above (`tests/circuits.py` has `coupled_zero_pis`; numerify it and transform by a random admissible float `S`) now exports, and the exported spec loads; a deliberately nonsymmetric numeric matrix is still rejected; existing `tests/test_io.py` passes unchanged.

## Change 3: a public numeric frame change with checks

**Now.** `NumericHamiltonian.transformed(S)` (`src/circuit_quantization/product_state.py`, about line 73) applies
`E_C' = S⁻¹ E_C S⁻ᵀ`, `E_L' = Sᵀ E_L S`, `b' = Sᵀ b`, `n_g' = Sᵀ n_g`, `ω' = Sᵀ ω`.
It does no admissibility check, keeps the old mode `types`, and treats the whole offset-charge vector alike.
`examples/hierarchical_kite.py` reaches into private functions of this module.

**Wanted.** A public, documented way to go from a `Result` (or `HamiltonianSpec`) plus parameter values to a numeric Hamiltonian, and from there to the numeric Hamiltonian in any admissible frame, such that the answer equals what the symbolic route gives.
The symbolic route is `Result.transform(S)` followed by `to_spec` and substitution of the same values.
Specifically:

1. **Admissibility.** The same rules as `_check_residual`: `S` square and invertible, nonzero entries only in the blocks allowed by `_S1b` in `modes.py`, compact block integer (with change 1's tolerance) and unimodular. Violations raise `ValueError` with the offending entry.
2. **Mode types.** The returned object's `types` are those of the new frame, classified by the same rule the symbolic route uses, so a harmonic mode that picks up a junction becomes extended, and any other reclassification the symbolic route produces is reproduced.
3. **Offset charges.** In `hamiltonian.py` the offset charge is a source part (from voltage sources, on every mode) plus a static sector label `n_g{j}` on each compact mode. The symbolic route transforms the source part and issues fresh sector labels on the new frame's compact modes only. The numeric route must match: carry the two parts separately; the source part transforms as `Sᵀ`; the sector values transform with `S_ccᵀ` on compact modes and are zero on noncompact modes (a static offset on a noncompact mode is gauge-removable and must not appear).
4. **Speed.** One frame change stays numeric linear algebra, with no SymPy call, and should take well under a millisecond for a handful of modes. Report the measured time for the KITE in `examples/product_state_demo.py`.
5. Keep `NumericHamiltonian.transformed` working for existing callers, or migrate them; do not leave two functions that disagree.

**Tests.** For the KITE and for `coupled_zero_pis`, with nonzero sector values and an `S` with a nonzero compact-noncompact shear, the numeric route's coefficients, cosine vectors, offsets and types equal the symbolic route's after substitution, to `1e-10` relative. An inadmissible `S` fails on both routes. `tests/test_product_state.py` passes.

## Change 4 (wanted, separable): a run record that rebuilds the circuit

**Wanted.** Two public functions in `io.py`: one that writes, alongside the kronfree-spec fields, `_`-prefixed metadata blocks from which the same `Result` can be rebuilt, and one that reads such a record back to a `Result`.
The format already permits extra keys that start with `_`.

- `_circuit`: branches (name, tail, head, element values or expressions, symbol assumptions), ground node, loops, external fluxes, and the flux assignment (`B`, `B_static`, `static`).
- `_frame`: `Z_full`, the full mode types, and the residual `S` if one was applied.
- `_values`: the parameter values, when the spec is numeric.
- `_generator`: already written; add the git commit when available.

Reading must follow the existing rule in `io.py`: no `eval` and no `sympify` on loaded text; use the module's own expression grammar.

**Tests.** For every fixture circuit in `tests/circuits.py`, write a record, read it back, and check that `to_spec` of the rebuilt `Result` equals `to_spec` of the original. A record with an unknown non-underscore key is rejected as now.

## Done means

- The full test suite passes in the `circuit_quantization` environment.
- `docs/spec-format.md` and the README describe changes 2 to 4 where they touch documented behaviour.
- A short report lists what changed, the measured frame-change time, and anything in this brief that turned out to be wrong about the current code.
