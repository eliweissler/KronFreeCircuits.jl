# Save-file format (version 1)

`save_system(path, sys)` writes a solved `System` to a JLD2 file, and `load_system(path)` reads it back.
JLD2 files are HDF5, and this one holds only numbers, `Bool`, strings and arrays of them, so it can be read without
KronFreeCircuits (see [Reading with h5py](#reading-with-h5py)). Files are uncompressed.

```julia
save_system("kite.jld2", sys; metadata=(note="optimum",))   # retained=false drops the retained eigenvectors
sys2 = load_system("kite.jld2"; on=Array)                   # on picks the device; callbacks come back as nothing
load_metadata("kite.jld2")                                  # Dict("note" => "optimum"), reads no results
system_nodes("kite.jld2")                                   # positions of the solved nodes: [[], [1], [2], ...]
load_node("kite.jld2", [1])                                 # (; energies, kept, vectors, dims, ports, on) of one node
```

## Layout

```
format                  "KronFreeCircuits.System"
format_version          1
package_version         e.g. "0.1.0"
saved_without_retained  true; present only when saved with retained=false
metadata/<key>          your notes (numbers, Bool, String, or arrays of them)
spec                    the spec as JSON text (spec_json); absent for a system built without a spec
constant, constant_deps the constant term as a coefficient string and its parameter names; present only
                        when terms/from_spec is false
terms/from_spec         true: the terms are rebuilt from `spec`, and `terms/*` holds only the extra terms
terms/coefficients      one coefficient string per stored term
terms/deps/<i>          parameter names of coefficient i
terms/factor_term       one entry per factor: its term index
terms/factors/{modes,kinds,ks}
                        one entry per factor: its mode, port kind and port k
theta/names             parameter names
theta/values/<i>        the parameter point, one tagged value each
root/                   the root node (below)
augmented/count         number of augmented children
augmented/<i>/          node_path (child indices from the root), solved/ (a node group), basis, terms/ (as
                        above), theta/, report/labels, report/taken, report/dropped
```

A node group (`root`, `<node>/children/<i>`, `augmented/<i>/solved`):

```
kind                    "node" or "given" (a GivenSystem)
eltype                  "Float64" or "ComplexF64"
dims, energies, vectors, kept, peak_component, peak_amplitudes
ports/count             ports/<i>/{modes,kinds,ks} (one entry each) and ports/<i>/matrix
joint_ports/count       joint_ports/<i>/{modes,kinds,ks} (the factor list) and joint_ports/<i>/matrix
retained/energies, unkept, complete, followed
retained/vectors        absent when nothing was retained, or when saved with retained=false
settings/solver, settings/keep      a struct: `type`, then `fields/<field>`
settings/{compute,phase,arithmetic,retain}   tagged values
settings/ports/{modes,kinds,ks}     the node's port list
children/count          children/<i> is a node group, or a leaf:
                        kind "leaf", mode, basis (a struct)
```

A `given` node has instead `settings/mode`, `settings/given_energies`, `settings/operator_names` and
`settings/operators/<name>`; it has no children.

A struct is `<name>/type` (a solver, selection rule or basis name such as `"Lanczos"`) and `<name>/fields/<field>`,
one tagged value per field. A tagged value is `<name>/kind` plus `<name>/value`, where kind is one of `int`, `float`,
`bool`, `string`, `ints`, `floats`, `strings`, `matrix_real`, `matrix_complex`, `symbol` (stored as a string),
`nothing` (no value), `port` (`value` is the port kind, plus `k`), `modeport` (`mode`, `port_kind`, `k`) or `arraytype`
(the array type's name). The saved device is at `settings/solver/fields/on/value`; callbacks are stored as `nothing`.

## Behaviour

- **Coefficients** are stored as strings in the spec grammar, plus `Complex(re, im)` and `Rational(num, den)` literals.
  `save_system` refuses (with an `ArgumentError`) a coefficient that would not parse back identically.
- **Spec-built systems** store the spec (`spec`) and only the terms added to it afterwards.
- **`retained=false`** sets `/saved_without_retained = true` and `retained/complete = false` on every node whose vectors
  were dropped. Derivatives of such a system take the slower paths for a node without retained vectors and agree with
  the original's to solver tolerance, not bit for bit.
- **Callbacks** are not saved; a loaded solver's callback is `nothing`.
- **Device**: `load_system(path; on=Array)` sets `on` on every solver that has the field. The saved device is only
  recorded: `load_node(path, pos).on` returns it (`nothing` for a node with none).
- **Versions**: `/format_version` is 1. `load_system` reads only the versions it knows and otherwise throws an
  `ArgumentError` naming the file's version; a future format change bumps the number, and a newer package will keep
  reading version 1 files.

## Reading with h5py

```python
import h5py

h5py.get_config().complex_names = ("re", "im")      # complex arrays are stored as (re, im) compounds
with h5py.File("kite.jld2", "r") as f:
    E = f["root/energies"][()]                       # eigenvalues
    V = f["root/vectors"][()]                        # transposed: Julia is column-major, so V[j] is eigenvector j
    kind = f["root/kind"][()].decode()               # strings come back as bytes
    n_leaves = f["root/children/count"][()]
    # Bool comes back as uint8
    complete = bool(f["root/retained/complete"][()])
```

Matrices (`vectors`, `ports/<i>/matrix`) come out transposed relative to Julia; use `V.T` for the Julia orientation.
