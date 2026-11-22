# Saving and loading a solved `System`

## Intent

A solved `System` should be written to one file and read back later as a **fully working** `System`.
That means `port`, `apply`, `decompose`, derivatives, and use as `previous` for a warm-started `materialize` all work
on the loaded object exactly as they did on the original.
The motivation is reproducible paper and thesis figures and long solves that shouldn't be rerun.
Today `Serialization` is the only route. It fails unless `KronFreeCircuits` is loaded first, and its files break across
Julia and package versions.

A second goal is **partial loading**: reading one node's energies or vectors without reading the whole file.
This should work from Julia and from Python (`h5py`).

## Decisions made with Eli

- **Format: JLD2** (pure Julia, no `libhdf5` binary), with a **hand-designed layout of plain arrays, strings and
  numbers**, never an automatic dump of Julia structs.
  The layout outlives changes to the package's structs; JLD2's automatic struct encoding would not.
- **Tree-shaped file:** one JLD2 group per tree node, with child nodes as subgroups, mirroring the `System`.
- **Versioning:** the file records a format version and the package version.
  The current loader only has to read the current format.
  When the layout changes, the format version is bumped and a loader for the older format is added (as needed).
- **Python readability matters but is secondary.**
  A probe on 2026-11-21 (JLD2 0.6.7, h5py 3.16) confirmed that h5py reads this kind of file:
  - Real and integer arrays, floats, strings, and nested groups read as ordinary numpy values.
  - Complex arrays read as `complex128` after `h5py.get_config().complex_names = ("re", "im")`.
    Without that setting they read as a structured array with `re` and `im` fields.
  - Slices of a large dataset read without loading it.
  - Quirks: matrices come out transposed (Julia is column-major), strings as bytes, `Bool` as `uint8`,
    and a `Symbol` as an opaque object (so the layout stores strings, never `Symbol`s).
  - JLD2 doesn't write HDF5 attributes, so all metadata is stored as small datasets.
  - **JLD2-compressed datasets cannot be opened by h5py at all.** Files are written uncompressed.

## Goals

1. `save_system(path, sys)` writes a solved `System`; `load_system(path)` returns an equivalent working `System`.
2. Partial reads in Julia without materializing the whole `System`: list the nodes, and read one node's energies,
   kept indices, vectors or port matrices.
3. A documented layout, so a Python user can read energies, vectors and port matrices with h5py alone.
4. Round-trip exactness: every array is bit-identical after loading, and every scalar setting is equal.

## Non-goals

- Saving `Derivatives`, `Resolvent`, `KronFreeOperator` or any other derived object.
  These are recomputed from the loaded `System`.
- Reading files from older formats now (only the hook for adding older loaders later).
- Compression.
- A Python loader package.
  Python reads with h5py and the documented layout; a convenience reader can come later.
- Writing files that can be read without JLD2's quirks (no attributes, no links).

## Anti-goals

- **Executing code from a file.** Coefficient expressions are stored as text and must pass the same allowlist check
  (`_check_expr` in `parameters.jl`) that spec parsing uses before they are compiled.
  A file must never be able to run arbitrary Julia.
- **Silent loss.** Anything that cannot be stored (see Constraints) is named in the docstring, and loading never
  quietly substitutes a different setting without saying so.
- **A struct dump.** No `jldsave(path; sys)` of Julia objects, even for a part of the system.

## What a `System` holds (as of `main` at `25d3eb4`)

- `System`: `root::MaterializedNode`, `spec::Union{Nothing,Spec}`, `terms::Vector{Term}`, `θ::NamedTuple`,
  `constant::Coefficient`, `augmented::Vector{Any}` (of `AugmentedChild`).
- `MaterializedNode{T}` (T is `Float64` or `ComplexF64`): `node` (a `Node` or `GivenSystem`), `dims`, `energies`,
  `vectors::Matrix{T}`, `kept`, `ports::Dict{Tuple{Symbol,Port},NodeMatrix}`,
  `joint_ports::Dict{Vector{Tuple{Symbol,Port}},NodeMatrix}`, `children::Vector{Any}`, `peak_component`,
  `peak_amplitudes::Matrix{T}`, `retained::Retained{T}`.
- `Retained{T}`: `energies`, `unkept`, `vectors::Union{Nothing,Matrix{T}}`, `complete`, `followed`.
- `Node`: `children` (each a `mode => basis` pair, a `Node` or a `GivenSystem`), `solver`, `compute`, `keep`,
  `phase`, `ports`, `arithmetic`, `retain`.
- Bases: `ChargeBasis`, `PhaseGrid(a, b, N, order)`, `PeriodicGrid(N, order, sectors)`, `FockBasis(N, phi_zpf)`.
- Solvers: `Dense`, `Lanczos`, `LOBPCG`, `ShiftInvert`, `ChebyshevFilter`, `Auto`, each a set of numbers, symbols
  and an `on` field (the array type, `Array` or `CuArray`); `LOBPCG` and `ChebyshevFilter` also have `callback`.
- Selection rules: `Lowest(k)`, `Indices(list)`, `Connected(core, (mode, port), thresh)`,
  `Follow(reference::Union{Nothing,Matrix}, minoverlap)`, `Within(window, max)`.
- `Port(kind::Symbol, k::Float64)`; `Factor(mode, port)`; `Term(coeff, factors)`.
- `Coefficient(ex, deps, f)`: `ex` is an `Expr`, `Symbol` or number; `f` is compiled from `ex` (`_make(ex, deps)`).
- `GivenSystem(mode; energies, operators::Dict{Symbol,Matrix{ComplexF64}})`.
- `AugmentedChild`: `node::Node` (a node of the tree), `solved::MaterializedNode`, `basis::Matrix{ComplexF64}`,
  `terms`, `θ`, `report::UnionReport(labels, taken, dropped)`.
- `Spec` round-trips through the existing `spec_json` / `parse_spec` (both in core).

Everything here is plain data except the solver `callback` and the `on` array type.

## Constraints

- **Callbacks are not saved.** A loaded solver has `callback = nothing`. The docstring says so.
- **The device is chosen at load time, not restored.** `load_system(path; on=Array)` sets `on` on every loaded solver
  that has the field (`Lanczos`, `LOBPCG`, `ChebyshevFilter`, `Auto`); `Dense` and `ShiftInvert` have none and are
  unchanged, so a GPU load of a tree with an explicit `ShiftInvert` leaf keeps that leaf on the CPU, as a freshly built
  tree would. `on` takes the array type, as everywhere else in the package (`on=CuArray` after `using CUDA`).
  The default is the CPU whatever the file was solved on. (Eli's decision, replacing warn-and-load.)
  The original device is still written to the file by name (`settings/on`, `"Array"` or `"CuArray"`), for the record,
  and `load_node` returns it; it is never applied.
  What the choice affects: only computation after loading (derivatives' operators and linear solves, `apply`, and an
  `Auto` node's resolution: an `Auto` leaf resolves to `ShiftInvert` on the CPU and `ChebyshevFilter` on the GPU).
  The stored results are identical either way, since `MaterializedNode` vectors are CPU matrices.
  The round-trip tests compare solver settings with `on` replaced by the requested one.
- Saved vectors are always CPU arrays (they are `Matrix` in `MaterializedNode` already).
- `previous` matches nodes by tree position, not object identity, so a loaded `System` works as `previous` for a fresh
  `materialize` of a tree the user builds again in code.
- Loaded `Node` objects are new objects.
  Inside the loaded `System`, every reference to a node (in particular `AugmentedChild.node`) must point at the
  loaded tree's node at the same position, not at a copy.
- Floating-point values are stored as `Float64` (or `ComplexF64`) arrays; integers as `Int64`.

## Approaches considered

1. **JLD2, hand-designed layout (chosen).** Pure Julia and light; partial loading via `jldopen`; readable by h5py
   apart from compression.
2. **HDF5.jl, hand-designed layout.** The full HDF5 feature set (attributes, compression readable everywhere,
   parallel I/O) at the cost of the `libhdf5` binary dependency and possible clashes with cluster HDF5 builds.
   Eli preferred JLD2's lightness; Python readability is secondary.
3. **JLD2 automatic struct dump.** About one line of code, but files break whenever a struct changes and it cannot
   store the compiled coefficient functions anyway. Rejected.

## Layout (format version 1)

Paths are JLD2 group paths. "String" means a Julia `String`; symbols are always written as strings.
Optional entries are absent rather than empty when they do not apply.

```
/format                      "KronFreeCircuits.System"
/format_version              1                         (Int64)
/package_version             "0.1.0"                   (from pkgversion)
/metadata/...                user metadata (see API), plain values only
/theta/names                 Vector{String}
/theta/values                Vector{Float64} or Vector{ComplexF64}
/spec                        String (spec_json), absent when sys.spec === nothing
/terms/from_spec             Bool: true when the terms are circuit_terms(spec) followed by /terms/... (the extras)
/constant                    String (written only when not from_spec; else constant_term(spec))
/terms/coefficients          Vector{String}, one per written term (all terms, or only the extras when from_spec)
/terms/deps                  Vector{String} per term (the coefficient's parameter names), for the parser
/terms/factor_term           Vector{Int64}   (term index of each factor, 1-based)
/terms/factor_mode           Vector{String}
/terms/factor_port_kind      Vector{String}
/terms/factor_port_k         Vector{Float64}
/root/...                    the root node group (below)
/augmented/count             Int64
/augmented/<i>/...           one AugmentedChild each (below)
```

**Coefficients: reuse the spec grammar.** No new encoding. Two cases:

1. **Spec-built systems** (the normal path, `materialize(root, spec, θ; extra_terms)`): the terms are
   `vcat(circuit_terms(spec), extra_terms)` and the constant is `constant_term(spec)`. The file stores the spec through
   the existing `spec_json`, and only the extras as written terms. `load_system` rebuilds the rest with
   `parse_spec` and `circuit_terms`, the same code that built them. At save time, `save_system` checks that
   `circuit_terms(spec)` equals the first terms of `sys.terms` exactly (see "exact" below); when it doesn't (a term
   list edited after building), it falls back to case 2 for every term and sets `from_spec = false`.
2. **Written terms** (the extras, systems built from a term list, and the terms of each `AugmentedChild`): each
   coefficient is written with the spec writer `expr_string` (specwrite.jl) and read back with the spec parser
   `Coefficient(src, params)` (parameters.jl), which checks every call against the allowlist and folds constants as
   the package does when it first builds coefficients. `params` is the term's stored `deps`.
   `expr_string` refuses complex constants and `im` today; it is extended, minimally, to write a complex constant in a
   form the existing parser accepts and folds back to the same constant (for example `cis`-free `(re + (im * c))`,
   whatever the builder finds round-trips), without widening the parser's allowlist.

**The guarantee is a save-time check.** `save_system` parses back every coefficient it writes, and checks every
rebuilt spec term, and refuses to save, with an `ArgumentError` naming the term and both expressions, if any
comes back different. "Exact" means `ex` equal recursively with the same `typeof` at every literal (Julia's `==` on
`Expr` would accept `1 == 1.0`) and the same `deps`. A file that saved therefore reloads with identical coefficients;
there is no silent mismatch. Loading parses only through `Coefficient(src, params)`, so a hand-edited file can't run
code beyond the allowlist. (Eli chose this over Symbolics.jl, Latexify.jl and a custom typed encoding: it reuses the
tested spec reader and writer.)

A **node group** (`/root`, and recursively `.../children/<i>`):

```
kind                 "node" or "given"
eltype               "Float64" or "ComplexF64"
dims                 Vector{Int64}
energies, kept, peak_component
vectors, peak_amplitudes
settings/solver      String (type name) plus settings/solver_<field> for each stored field
settings/on          "Array" or "CuArray"
settings/compute, settings/phase, settings/arithmetic, settings/retain
settings/keep        String (rule type) plus settings/keep_<field> for each field
                     (Follow's reference matrix is stored when present)
settings/ports       a table: port_mode, port_kind, port_k
ports/count          Int64
ports/<i>/mode, ports/<i>/kind, ports/<i>/k, ports/<i>/matrix
joint_ports/count    Int64
joint_ports/<i>/modes, kinds, ks (the key, in order), joint_ports/<i>/matrix
retained/energies, retained/unkept, retained/complete, retained/followed
retained/vectors     absent when nothing was retained, or when saved with retained=false
children/count       Int64
children/<i>/...     a node group, or a leaf group:
                       kind "leaf", mode String, basis String (type name), basis_<field> for each field
```

A `GivenSystem` group has `kind = "given"`, `mode`, `given_energies` and `operators/<name>` matrices in its settings,
plus the usual result fields of its `MaterializedNode`.

An **augmented child group** (`/augmented/<i>`):

```
node_path            Vector{Int64}   child indices from the root to the augmented node
                     (the augmented node's results are also under /root at that position; storing them
                     twice is accepted in version 1, my call: no code relies on the two being one object)
solved/...           a node group (its MaterializedNode)
basis                Matrix{ComplexF64}
terms/...            as /terms
theta/names, theta/values
report/labels, report/taken, report/dropped
```

Python reading example for the docs (`vectors` transposed back to Julia's orientation):

```python
import h5py
h5py.get_config().complex_names = ("re", "im")
with h5py.File("run.jld2", "r") as f:
    E = f["root/energies"][()]
    V = f["root/children/2/vectors"][()].T
```

## API

```julia
save_system(path::AbstractString, sys::System; retained::Bool=true, metadata=(;)) -> path
load_system(path::AbstractString; on=Array) -> System     # on: the device for every loaded solver
system_nodes(path) -> Vector{Vector{Int}}             # the tree positions in the file, root first
load_node(path, position::AbstractVector{<:Integer}=Int[]) -> NamedTuple
    # (; energies, kept, vectors, dims, ports) of one node, without building a System
```

- `retained = false` leaves out `retained/vectors` (they can be most of the file).
  The file records this (`/saved_without_retained = true`), and the loader sets `retained.complete = false` on every
  node whose vectors were left out; its `retain` setting is kept as it was. This is the one documented exception to
  "every scalar setting is equal".
  (My call. Keeping `complete = true` with no vectors would make `_resolvent` build a resolvent with no unkept states
  and give wrong derivatives silently.)
  Derivatives on such a system then take the paths `_resolvent` already has for a node without its retained vectors:
  a `Dense` node is re-diagonalized from its operator (with an `@info`), an iterative node gets a linear solve on the
  complement, and a node whose `kept` is not `1:k` throws the existing `ArgumentError`.
  The results agree with the original's to solver tolerance, not bit for bit, and are slower. The docstring says so.
- `metadata` takes a `NamedTuple` or dict of plain values (numbers, strings, arrays of them), stored under
  `/metadata` and returned by a `load_metadata(path)` (exact name the builder's call).
- `load_system` checks `/format` and `/format_version`, then dispatches to a loader for that version.
  Today there is only `_load_v1`. An unknown version throws an error naming the file's version and the newest
  version this package reads.
- Exported from the main module, not `Experimental`.

## Dependency

JLD2 becomes a **hard dependency** (pure Julia, light), as Eli chose over a package extension. It is added by
editing `Project.toml` (`[deps]` and `[compat]`) and running `Pkg.resolve()`; no `Pkg.add`.

## Testing

- Round-trip tests that compare every field of the loaded `System` with the original, bit for bit:
  - a real tree and a complex tree (`arithmetic = :complex` or a flux that makes it complex);
  - every basis type, every solver type, every selection rule, including `Follow` with and without a reference matrix;
  - a tree with a `GivenSystem` child;
  - a system built with `augmented` (`AugmentedChild`), checking that `AugmentedChild.node` is the loaded tree's node;
  - a system with `spec === nothing` (from a term list) and one from a spec;
  - `retained = false`.
- Behaviour tests on a loaded system: `port`, `apply`, `decompose` and `derivatives` agree with the original;
  a loaded system as `previous` gives the same warm-started result as the original as `previous`.
- A file whose coefficient text fails the allowlist is refused with an `ArgumentError` (a hand-edited file).
- A spec-built system with `extra_terms` writes only the extras; one whose term list was edited after building falls back to writing every term.
- A complex constant (from `phase_factor`) round-trips through the extended `expr_string` with its type kept, and `save_system` refuses a coefficient that would not round-trip (a test forces one).
- An unknown format version is refused.
- A Python-tier test (the existing `test/python` tier) that opens a saved file with h5py and checks energies and a
  complex vector against Julia's values.
- These go in the fast tier, except the Python one.

## Left to the builder

- Function and file organisation (e.g. `src/saveload.jl`), and helper names.
- The exact field names under `settings/` for each solver and rule, as long as they are documented in one place
  (a docstring or a `docs/` page describing format version 1).
- How the writer walks the tree and how the reader rebuilds it, provided the node-identity constraint holds.
- Whether `load_node` returns port matrices by default or on request.
