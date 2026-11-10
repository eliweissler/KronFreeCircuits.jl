# Design: a direct Julia interface to circuit-quantization (2026-11-10)

Status: design for Eli's approval; project 1 of 3 before the feature freeze.
Amends architecture spec §2 (anti-goals), §15 (packages) and §20b (the Python–Julia contract).
Followed by project 2 (`2026-11-10-mean-field-augmentation-design.md`) and project 3 (`2026-11-10-perturbative-derivatives-design.md`).
Items marked *(my call)* were chosen by Claude and not discussed; everything else was decided with Eli on 2026-11-10.

## 1. Intent

Today a run needs two sessions.
The circuit and its coordinates are defined in a Python session with circuit-quantization, exported with `save_spec` to a JSON file, and then loaded in a Julia session with `load_spec`.
Changing the circuit or the coordinates means going back to Python, exporting again, and returning to Julia.

This is slow, and it makes provenance hard to keep: the JSON file on disk does not say which circuit, which frame, or which script produced it.
It also made the mean-field coordinate optimization impractical in Julia, because that calculation changes the coordinate transformation on every iteration.

The circuit definition is light compared with the solve.
This project lets one Julia session own the whole run: define the circuit, choose coordinates, solve, and write a single JSON record at the end.
circuit-quantization stays the engine and stays a Python package; it is called from Julia.

## 2. Goals, non-goals, anti-goals

### Goals

1. Define a circuit, its loops and its coordinates from a Julia session, using circuit-quantization's own Python API through PythonCall, and obtain a `Spec` in memory with no file written.
2. Change the secondary coordinate transformation `S` (the paper's residual transformation, `φ = S φ'`) cheaply enough to sit inside an optimization loop, with one Python crossing per iteration.
3. Export at the end: one JSON string or file that holds the spec and enough metadata to rebuild the circuit and frame that produced it.
4. Keep the existing file route working unchanged; committed JSON fixtures stay valid.

### Non-goals

- Porting any of circuit-quantization's symbolic machinery to Julia.
- Differentiating through the coordinate transformation (Eli, 2026-11-10: a nice-to-have, explicitly forgone to keep the Python interface simple).
- Julia wrapper types for every circuit-quantization class or function.
  The user calls the Python API directly; the extension only adds the conversions and the loop handle of §4.
- Putting the tree, bases or truncations into the spec. They stay in Julia code.
- Drawing, the scqubits export, and the `optimize/` symbolic search (these remain reachable through the Python module but get no Julia support).

### Anti-goals

- A Python dependency for the core package.
  `using KronFreeCircuits` without PythonCall must load, pass the default test tier, and never import Python.
- A second definition of the spec format.
  The in-memory route must produce exactly the `Spec` the JSON route produces for the same input.
- An interface that is awkward on the Python side in order to be convenient on the Julia side (Eli's words: keep circuit-quantization "a nice python interface").

## 3. Constraints and current state

- The exporter lives in the other repository: `circuit_quantization/io.py` (`to_spec`, `save_spec`, `load_spec`, `HamiltonianSpec`).
- `to_spec` returns a `dict`, and Julia's `parse_spec(json::AbstractString)` (`src/spec.jl:158`) already accepts a string, so in-memory transfer needs no new format.
- The spec carries neither the tree nor the frame `S`.
- Cost of a new frame, measured during scoping on a 6-mode circuit (coupled zero-pis):
  symbolic parameters with a float `S` took 60 s for `transform` plus 19 s for `to_spec`;
  with parameters substituted first (`numerify`) `transform` took 0.07 s;
  the purely numeric `NumericHamiltonian.transformed(S)` in `product_state.py` takes about 13 µs.
- Two defects in circuit-quantization that a Julia caller hits at once (both observed during scoping):
  1. a float compact block `S_cc` is rejected even for the identity, because `Float(1.0).is_integer` is `None`;
  2. `to_spec`'s symmetry check has no absolute tolerance, so float round-off such as `E_L[1,3] = -1.09e-26` against `E_L[3,1] = 0` raises.
  `bench/hensel_three_fluxonia/quantize_three_fluxonia.py` assembles its spec by hand to avoid the second.
- circuit-quantization keys parameter values by SymPy `Symbol`, and symbol identity includes assumptions (`positive=True`, `real=True`); a bare string key silently fails to match.
- The architecture spec currently says: anti-goal "a runtime dependency on Python" (§2), `PythonCall` "deliberately absent" (§15), and "a PythonCall extension for in-memory spec transfer is an optional convenience, never a dependency" (§20b).
  This design uses exactly that carve-out.

## 4. The design

### 4.1 Where it lives

A package extension, `KronFreeCircuitsPythonCallExt`, loaded when the user has `PythonCall` in their environment, following the existing CUDA extension pattern *(my call)*.
`PythonCall` is a weak dependency only.

The Python interpreter is the user's existing environment that has circuit-quantization installed (today the conda environment `circuit_quantization`, an editable install) *(my call)*.
The extension does not install Python packages.
The README documents how to point PythonCall at that environment.
If circuit-quantization cannot be imported, the first call fails with a message that says which interpreter was used and how to change it.

### 4.2 What the user types

The names below are illustrative; the builder may change them.

```julia
using KronFreeCircuits, PythonCall

cq = circuit_quantization()                 # the Python module, imported once

circuit = cq.Circuit(pylist([cq.Branch("J", 1, 0; C=..., EJ=...), ...]); ground=0)
loops   = cq.Loops.from_branch_dict(circuit, ...)
res     = cq.quantize(circuit, loops, Z, types; B=B)   # or the automatic basis

spec = Spec(res)                            # symbolic parameters, in memory

qc   = QuantizedCircuit(res; values=θ)      # numeric handle for loops
spec = Spec(qc, S)                          # the same circuit in frame S, numeric

record = run_record(spec; circuit=res, frame=S, values=θ)   # JSON string
```

Three things are new on the Julia side.

1. **`Spec(res)`**: a circuit-quantization `Result` becomes a `Spec`.
   It goes through `to_spec` and the same validation as `parse_spec`, so both routes agree by construction.
2. **A numeric circuit handle** (`QuantizedCircuit` above).
   It is built once from a `Result` and a set of parameter values, holds the numeric Hamiltonian in the base frame on the Python side, and returns a `Spec` for any admissible `S`.
   This is the loop route of goal 2.
3. **A run record**: the spec JSON plus `_`-prefixed metadata blocks (§4.5).

Parameter values are passed from Julia by name (a `NamedTuple` or a `Dict` of names), and the extension maps names to the circuit's own symbols.
An unknown name is an error, never a silent no-op.

### 4.3 Two routes to a spec in a new frame

- **Loop route (numeric).**
  Parameters are fixed numbers and `S` changes every iteration.
  The frame change is the numeric linear algebra `E_C' = S⁻¹ E_C S⁻ᵀ`, `E_L' = Sᵀ E_L S`, `b' = Sᵀ b`, `n_g' = Sᵀ n_g`, `ω' = Sᵀ ω`, done in Python.
  The `Spec` is built from the returned arrays directly, without printing and re-parsing expression strings.
  Target *(my call)*: under 5 ms per frame change for the three-mode KITE, measured from Julia and including the crossing; the measured figure goes in the README.
  A numeric spec must not pay a per-coefficient compilation cost, anywhere between the frame change and the end of `materialize`.
  Today `Coefficient(x::Number)` compiles a function (`src/parameters.jl:148`), and so does every arithmetic operation on coefficients used by term expansion, at several milliseconds each on first call; a new numeric spec per iteration would compile afresh every time and grow the generated-function cache without bound.
  Constant coefficients and arithmetic among them are therefore folded to plain numbers in this project *(my call, from the builder check)*.
  This is a core change with tests in the default tier, and its timing criterion is measured through `materialize`, not only through the frame change.
- **Final route (symbolic).**
  `S` is fixed (for example the optimum `S*`) and parameters stay symbolic, so the spec can be swept and, in project 3, differentiated with respect to circuit parameters.
  This uses `Result.transform` and `to_spec` and may take seconds; it runs once.
  The extension passes `S` in a form that avoids the 60 s float-times-symbol path (integers in the compact block, rationals or a substituted intermediate elsewhere); the builder picks the mechanism and records the timing.

Both routes must enforce the same admissibility rules for `S`: zeros below the compact block, `S_cc` integer and unimodular, `S` invertible.
Both must classify mode types in the new frame by the same structural rule, so a change of type (harmonic to extended, or any other the rule produces) is identical on the two routes.

**Offset charges** *(my call, from the builder check)*.
The two routes must give the same spec: the loop-route spec equals the final-route spec with the parameter values substituted.
Today they do not.
The offset charge has two parts in circuit-quantization (`hamiltonian.py`): a source part from gate voltages, on every mode, and a static sector label `n_g{j}` on each compact mode.
The symbolic route transforms the source part and issues fresh sector labels on the new frame's compact modes only.
A literal `n_g' = Sᵀ n_g` on the total instead spills the sector labels onto noncompact modes whenever the compact-noncompact shear is nonzero (as in the KITE's `S*`).
A static offset on a noncompact mode is removable by a gauge transformation, and leaving it in would add a linear charge term that costs those nodes their real arithmetic.
The convention is therefore the symbolic route's: the source part transforms as `Sᵀ`; the sector labels live on compact modes only and transform with `S_ccᵀ`; their removable noncompact components are dropped.
The numeric route is made to follow it, which means it carries the two parts separately.

### 4.4 Changes in circuit-quantization

These land in that repository as their own commits, with tests there.

1. Accept an integer-valued float compact block: values within a tight tolerance of integers are rounded to exact integers; anything else is still rejected with the offending entry named.
2. Give the export symmetry check an absolute tolerance tied to the matrix scale, and symmetrize float matrices that pass it.
   A genuinely nonsymmetric matrix is still rejected (decided 2026-10-19 for both packages).
3. Add the numeric frame change as public API with the constraint checks and type relabelling of §4.3.
   Today `NumericHamiltonian.transformed` has neither, and `examples/hierarchical_kite.py` imports private functions.
4. After (2), `bench/hensel_three_fluxonia` no longer needs its hand-assembled spec.
5. The run record of §4.5 is written and read back by public functions in circuit-quantization, which owns the objects being serialized *(my call, from the builder check)*.
   Reading follows `io.py`'s existing rule of no `eval` and no `sympify` on loaded text.
   The Julia extension calls these functions and adds its own `_generator` entries.

### 4.5 Provenance

The run record is kronfree-spec version 1 plus metadata keys, which the format already allows when they start with `_`.

- `_generator`: package names and versions on both sides, and the git commit of each when available.
- `_circuit`: the branches (name, nodes, element values or symbols with their assumptions), ground node, loops, external fluxes, the flux assignment (`B`, `B_static`), and any substitutions or mode names given to `to_spec`, in a form from which the same `Result` can be rebuilt in Python.
- `_frame`: `Z_full`, the mode types, and `S`.
- `_values`: the parameter values used, when the spec is numeric.

The tree is Julia code and is not serialized here; recording it belongs to plan 6 (export and caching).

## 5. Tests and acceptance

- The default test tier stays Python-free and uses the committed fixtures, as now.
- A separate tier runs when PythonCall and circuit-quantization are available *(my call)*.
  It is not required for the default suite to pass.

The project is done when:

1. For each generated fixture circuit, `Spec(res)` built in memory equals the `Spec` loaded from the committed JSON, coefficient by coefficient at sample parameter values.
2. The KITE is defined, quantized, moved through at least three frames (published, normal-mode, and the paper's `S*`) and solved in one Julia session, and the loop route meets the timing target or the shortfall is explained.
3. A frame change with a float identity `S` on a circuit with compact modes succeeds, and an inadmissible `S` fails with a clear message on both routes.
4. A run record written at the end of a session rebuilds, in a fresh session, a circuit and frame whose spec equals the original.
5. `bench/hensel_three_fluxonia` runs from a spec produced by `to_spec` with no hand assembly.
   The regenerated spec matches the current `three_fluxonia.json` numerically and the bench's smoke run passes on the Mac; the full GPU run is not required *(my call)*.
   The package changes this bench depends on (`within`, the `compute` clamp, the `Auto` rule) were committed by Eli on 2026-11-10 (`057560a`); the bench directory itself is still untracked.
8. On the KITE with a nonzero offset charge and the frame `S*`, the loop-route spec equals the final-route spec with values substituted.
6. `using KronFreeCircuits` in an environment without PythonCall loads and passes the default tier.
7. Architecture spec §2, §15 and §20b are amended to describe the extension, and STATUS.md records it.

## 6. Left to the builder

- All names in §4.2.
- Whether the numeric handle lives on the Python side as a `NumericHamiltonian`, a new small class, or plain arrays.
- The exact schema of the `_circuit` block, provided acceptance item 4 holds.
- How symbolic `S` entries are passed on the final route.
- How generators and stdout-printing helpers in circuit-quantization are handled (collect, silence), where they are touched at all.
