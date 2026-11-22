# System Save/Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `save_system` writes a solved `System` to one JLD2 file with a hand-designed, versioned layout, and `load_system` reads it back as a fully working `System`; `system_nodes`, `load_node` and `load_metadata` read parts of a file.

**Architecture:** Three new files under `src/saveload/`: value and settings encoding (`values.jl`), coefficients and terms (`terms.jl`), and the tree walk plus public API (`system.jl`). Coefficients reuse the spec writer (`expr_string`) and parser (`Coefficient(src, params)`), with a save-time check that every written coefficient parses back identical. Settings structs (solvers, rules, bases) are written field by field and rebuilt with their positional constructors from a fixed allowlist of type names.

**Tech Stack:** Julia 1.13, JLD2 0.6 (new hard dependency), the existing test tiers.

**Spec:** `docs/superpowers/specs/2026-11-21-system-save-load-design.md` (read it first; this plan argues from it).

## Global Constraints

- Work in the worktree `~/src/KronFreeCircuits-saveload` (branch `system-save-load`). Julia is `$HOME/.juliaup/bin/julia`.
- **Never run `Pkg.add`, `Pkg.rm`, `Pkg.gc`, `Pkg.update` or `Pkg.develop`.** Add JLD2 by editing `Project.toml` and running `Pkg.resolve()`.
- Run only the fast tier or named files per task: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_saveload_values.jl"])'`. Never set `KRONFREE_SLOW_TESTS=1`; the slow tier runs once after the last task.
- JLD2 compat: `JLD2 = "0.6"`. No compression. The file never contains a Julia `Symbol` or an automatically encoded Julia struct: only numbers, `Bool`, `String`, and arrays of `Int64`, `Float64`, `ComplexF64`, `String`.
- File constants: `/format = "KronFreeCircuits.System"`, `/format_version = 1` (`Int64`), `/package_version = string(pkgversion(KronFreeCircuits))`.
- No file content is ever passed to `eval` or `Meta.parse` except through the coefficient path of Task 1, and type names are looked up only in the fixed allowlist of Task 2.
- Callbacks are not saved (loaded as `nothing`). The device is chosen at load: `load_system(path; on=Array)`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Saving over an existing file, or a save that fails midway**: the old file must be intact after a failed save, and replaced after a successful one. Task 4 writes to `path * ".partial"` and renames it into place; test `"failed save leaves the old file"`.
2. **A path that isn't a save file** (a plain text file, or a JLD2 file without `/format`): an `ArgumentError` naming the path, not a JLD2 `KeyError`. Task 4, test `"not a system file"`.
3. **Unicode mode and parameter names** (`:φ₁`, `:E_J′`): they round-trip as strings. Task 3, test `"unicode names"`.
4. **An empty θ `(;)`** (as `GivenSystem` tests use): round-trips. Task 4, covered by the given-system round trip.
5. **Metadata values that can't be stored** (a function, a struct): `ArgumentError` from `save_system` before anything is written. Task 6, test `"unstorable metadata"`.

---

### Task 1: Coefficient text round trip

**Files:**
- Modify: `src/specwrite.jl` (add `_coefficient_string`)
- Modify: `src/parameters.jl` (add `_restore_literals`, `_saved_coefficient`, `_identical`)
- Test: `test/test_saveload_coefficients.jl`

**Interfaces:**
- Produces:
  - `_coefficient_string(c::Coefficient) -> String`: `expr_string` of `c.ex`, except a `Complex` literal anywhere is written `Complex(re, im)` and a `Rational` literal `Rational(num, den)`, each argument written by `expr_string`; the symbol `im` is written `im`. `spec_json`'s output must not change (it keeps using `expr_string`).
  - `_saved_coefficient(src::AbstractString, deps::Vector{Symbol}) -> Coefficient`: `Meta.parse` → `_restore_literals` → `_normalize_dots` → `_check_expr` → every free symbol must be in `deps` → `_make(ex, deps)` (deps in the stored order).
  - `_restore_literals(ex)`: replaces exactly `call(:Complex, a, b)` and `call(:Rational, a, b)` whose arguments are numeric literals (after unary minus on a literal) with `Complex(a, b)` / `Rational(a, b)`; anything else is left for `_check_expr` to accept or refuse.
  - `_identical(a::Coefficient, b::Coefficient) -> Bool`: `ex` recursively equal with `typeof` equal at every node and literal, and `a.deps == b.deps`.

- [ ] **Step 1: Write the failing tests** in `test/test_saveload_coefficients.jl`:

```julia
using KronFreeCircuits: _coefficient_string, _saved_coefficient, _identical, _make, phase_factor
rt(c) = _saved_coefficient(_coefficient_string(c), copy(c.deps))
@testset "coefficient text round trip" begin
    E_J = Coefficient("E_J", [:E_J]); phi = Coefficient("varphi_ext", [:varphi_ext])
    cases = [Coefficient(2), Coefficient(1.5), Coefficient(-0.25), Coefficient(1.0 + 0.0im), Coefficient(0.3 - 2.0im),
             Coefficient(1 // 3), Coefficient(Int128(2)^70), Coefficient(big(2)^130),
             -E_J, 4 * E_J, (0.5 + 1.0im) * E_J, (1 // 2) * E_J, E_J * phase_factor(Coefficient(0.7), 1),
             phase_factor(phi, -1), Coefficient("sqrt(E_J) * cos(pi * varphi_ext)", [:E_J, :varphi_ext])]
    for c in cases
        @test _identical(rt(c), c)
        @test rt(c)((E_J=3.0, varphi_ext=0.4)) == c((E_J=3.0, varphi_ext=0.4))
    end
    @test !_identical(Coefficient(1), Coefficient(1.0))                     # typeof matters
    @test !_identical(Coefficient(1.0 + 0.0im), Coefficient(1.0))
    @test_throws ArgumentError _saved_coefficient("run(`ls`)", Symbol[])     # allowlist still applies
    @test_throws ArgumentError _saved_coefficient("Complex(E_J, 1.0)", [:E_J])  # only literal arguments
    @test_throws ArgumentError _saved_coefficient("E_J", Symbol[])          # undeclared symbol
    @test spec_json(load_spec(fixture("fluxonium.json"))) == spec_json(load_spec(fixture("fluxonium.json")))
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_saveload_coefficients.jl"])'`
Expected: FAIL with `UndefVarError: _coefficient_string`.

- [ ] **Step 3: Implement the four functions** with the signatures above. `_restore_literals` runs before `_normalize_dots` so the existing `int128"…"`/`big"…"` handling is untouched.

- [ ] **Step 4: Run the test to verify it passes** (same command). Expected: PASS. Then run `Pkg.test(test_args=["test_specwrite.jl", "test_parameters.jl"])`; expected PASS (spec output unchanged).

- [ ] **Step 5: Commit**: `git add src/specwrite.jl src/parameters.jl test/test_saveload_coefficients.jl && git commit -m "Save/load: coefficient text round trip through the spec grammar"`

---

### Task 2: JLD2 dependency and settings encoding

**Files:**
- Modify: `Project.toml` (`[deps]` JLD2 `033835bb-8acc-5ee8-8aae-3f567f8a3819`, `[compat]` `JLD2 = "0.6"`), then `Pkg.resolve()`
- Create: `src/saveload/values.jl`; Modify: `src/KronFreeCircuits.jl` (`using JLD2`; `include("saveload/values.jl")` after `include("augment.jl")`)
- Test: `test/test_saveload_values.jl`

**Interfaces:**
- Produces (all take a JLD2 file or group `g` and a path string `name` relative to it):
  - `_write_value(g, name, v)` / `_read_value(g, name)`: tagged values. Writes `name/kind` and `name/value`. Kinds: `"int"` (Int), `"float"` (Float64), `"symbol"` (stored as String), `"bool"`, `"nothing"` (no value), `"string"`, `"ints"` (Vector{Int}), `"floats"`, `"strings"`, `"port"` (`value` = kind string, plus `name/k` Float64), `"modeport"` (a `Tuple{Symbol,Port}`: `name/mode`, `name/port_kind`, `name/k`), `"matrix_real"`, `"matrix_complex"`, `"arraytype"` (the `on` field: `"Array"` or the type's name; read back as the string, never resolved). Any other value: `ArgumentError` naming the field.
  - `_write_struct(g, name, x)` / `_read_struct(g, name; on=Array)`: writes `name/type` (the type's name) and `_write_value` per field (`name/fields/<field>`). Reading looks the name up in `_SAVE_TYPES::Dict{String,DataType}` (every concrete subtype of `Solver`, `SelectionRule` and `Basis`), reads every field, replaces a field named `on` with the `on` keyword and a field named `callback` with `nothing`, and calls the positional constructor `T(fields...)` (so the inner constructors validate). An unknown type name: `ArgumentError`.

- [ ] **Step 1: Edit Project.toml and resolve**: `julia --project -e 'using Pkg; Pkg.resolve()'`; then `julia --project -e 'using KronFreeCircuits, JLD2'` loads. Commit Project.toml (and Manifest only if tracked: it is ignored).

- [ ] **Step 2: Write the failing tests** in `test/test_saveload_values.jl`:

```julia
using KronFreeCircuits: _write_struct, _read_struct, _write_value, _read_value, _SAVE_TYPES, JLD2
using InteractiveUtils: subtypes
allsub(T) = isabstracttype(T) ? reduce(vcat, map(allsub, subtypes(T)); init=Any[]) : Any[T]
@testset "settings round trip" begin
    xs = Any[Dense(), Lanczos(tol=1e-9, krylovdim=30), LOBPCG(tol=1e-7, preconditioner=:none, callback=x -> nothing),
             ShiftInvert(), ChebyshevFilter(degree=20), Auto(dense_cutoff=500),
             lowest(4), indices([3, 1, 2]), connected([1, 2], (:phi, nport()), 1e-3), follow(), follow([1.0 0; 0 1]; minoverlap=0.3),
             follow(ComplexF64[1 0; 0 1]), within(2.5; max=8),
             ChargeBasis(10), PhaseGrid(-3π, 3π, 201; order=6), PeriodicGrid(64; sectors=2), FockBasis(30; phi_zpf=0.7)]
    path = tempname() * ".jld2"
    JLD2.jldopen(path, "w") do f
        for (i, x) in enumerate(xs); _write_struct(f, "x$i", x); end
    end
    JLD2.jldopen(path, "r") do f
        for (i, x) in enumerate(xs)
            y = _read_struct(f, "x$i")
            @test typeof(y) == typeof(x)
            for n in fieldnames(typeof(x))
                n === :callback ? (@test getfield(y, n) === nothing) : (@test isequal(getfield(y, n), getfield(x, n)))
            end
        end
    end
    # every concrete solver, rule and basis type is in the allowlist (a new type must be added to it)
    @test Set(keys(_SAVE_TYPES)) == Set(string(nameof(T)) for T in vcat(allsub(Solver), allsub(SelectionRule), allsub(Basis)))
    @test_throws ArgumentError JLD2.jldopen(f -> _write_value(f, "bad", x -> x), tempname() * ".jld2", "w")
end
@testset "the on keyword replaces the saved device" begin
    path = tempname() * ".jld2"
    JLD2.jldopen(f -> _write_struct(f, "s", LOBPCG(on=Array)), path, "w")
    JLD2.jldopen(path, "r") do f
        @test _read_struct(f, "s").on === Array
        @test _read_struct(f, "s"; on=Vector).on === Vector       # any array type passes through unchanged
        @test f["s/fields/on/value"] == "Array"                    # the saved name, for the record
    end
end
```

(`allsub` must skip abstract intermediate types and test-only subtypes; if a test file defines its own `SelectionRule`, exclude it by module: keep `T` with `parentmodule(T) === KronFreeCircuits`.)

- [ ] **Step 3: Run to verify it fails.** Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_saveload_values.jl"])'`. Expected: FAIL, `_write_struct` not defined.

- [ ] **Step 4: Implement `src/saveload/values.jl`** with the interfaces above.

- [ ] **Step 5: Run to verify it passes** (same command). Expected: PASS.

- [ ] **Step 6: Commit**: `git add Project.toml src/KronFreeCircuits.jl src/saveload/values.jl test/test_saveload_values.jl && git commit -m "Save/load: JLD2 dependency and settings encoding"`

---

### Task 3: Terms, θ and the spec

**Files:**
- Create: `src/saveload/terms.jl` (include after `values.jl`)
- Test: `test/test_saveload_terms.jl`

**Interfaces:**
- Consumes: `_coefficient_string`, `_saved_coefficient`, `_identical` (Task 1); `_write_value`/`_read_value` (Task 2).
- Produces:
  - `_write_terms(g, name, terms::Vector{Term})` / `_read_terms(g, name) -> Vector{Term}`: the layout of the spec (`coefficients`, `deps` as one `Vector{String}` per term under `deps/<i>`, `factor_term`, `factor_mode`, `factor_port_kind`, `factor_port_k`). `_write_terms` parses back every coefficient it writes and throws `ArgumentError("term $i: coefficient $(c.ex) does not round-trip; written as \"$s\"")` when `!_identical`.
  - `_write_theta(g, name, θ::NamedTuple)` / `_read_theta(g, name) -> NamedTuple`: names as strings, each value through `_write_value` (so `Int` stays `Int`).
  - `_write_system_terms(f, sys::System)` / `_read_system_terms(f) -> (spec, terms, constant)`: implements the spec's two cases. `from_spec` is true when `sys.spec !== nothing`, `rebuilt = circuit_terms(parse_spec(spec_json(sys.spec)))` has `length(rebuilt) <= length(sys.terms)`, every `rebuilt[i]` equals `sys.terms[i]` (coefficients `_identical`, factors `==`), and `constant_term` of the re-parsed spec is `_identical` to `sys.constant`; then only `sys.terms[length(rebuilt)+1:end]` are written. Otherwise all terms and `/constant` are written (`/constant` uses `_coefficient_string`, with `/constant_deps`).
  - `_terms_identical(a::Vector{Term}, b::Vector{Term}) -> Bool` (used by tests and Task 5).

- [ ] **Step 1: Write the failing tests** in `test/test_saveload_terms.jl`:
  - `"spec-built terms are rebuilt"`: for `c = dc_two_transmons()` and `sys = materialize(c.root, c.spec, c.θ)`, write with `_write_system_terms`, then `f["terms/from_spec"] == true`, `length(f["terms/coefficients"]) == 0`, and `_read_system_terms(f)` gives `spec_equal(spec, c.spec)`, `_terms_identical(terms, sys.terms)`, `_identical(constant, sys.constant)`.
  - `"extra terms are written"`: the given-system tree of `test_given.jl`'s second testset (copy its construction), built with `materialize(root, spec, θ; extra_terms=[t])` for one coupling term `t`; only that term is written (`length(f["terms/coefficients"]) == 1`) and the read terms are identical.
  - `"edited term list falls back"`: `materialize(root, vcat(circuit_terms(spec)[2:end], circuit_terms(spec)[1:1]), θ; constant=constant_term(spec))` (spec is `nothing` there) and a `System` whose `spec` is set but whose terms were reordered (construct with `KronFreeCircuits.System(sys.root, sys.spec, reverse(sys.terms), sys.θ, sys.constant)`): both write every term, `from_spec == false`, and read back identical.
  - `"refuses a coefficient that would not round-trip"`: a term whose coefficient is `_make(:(E_J * $(Float32(0.5))), [:E_J])` (a `Float32` literal, which the grammar can't keep): `@test_throws "does not round-trip" _write_terms(...)`.
  - `"unicode names"`: θ `(φ₁=0.3, E_J′=2)` and a term on mode `:φ₁` round-trip, `typeof(θ′.E_J′) == Int`.

- [ ] **Step 2: Run to verify it fails.** `Pkg.test(test_args=["test_saveload_terms.jl"])`. Expected: FAIL, `_write_terms` not defined.

- [ ] **Step 3: Implement `src/saveload/terms.jl`.** If the fallback test's `Float32` case turns out to round-trip, pick another literal type the grammar can't keep (e.g. `Float16`) and note it in the test.

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit**: `git commit -m "Save/load: terms, theta and the spec"` (with the two files).

---

### Task 4: The tree, `save_system` and `load_system`

**Files:**
- Create: `src/saveload/system.jl` (include after `terms.jl`); Modify: `src/KronFreeCircuits.jl` export list (add `save_system, load_system, system_nodes, load_node, load_metadata`)
- Modify: `test/helpers.jl` (add `assert_same_system`)
- Test: `test/test_saveload_system.jl`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces:
  - `save_system(path::AbstractString, sys::System; retained::Bool=true, metadata=(;)) -> path`: writes the layout of the spec to `path * ".partial"`, then `mv(...; force=true)` onto `path`; on any error removes the partial file and rethrows.
  - `load_system(path::AbstractString; on=Array) -> System`: checks `/format` (else `ArgumentError("$path is not a KronFreeCircuits system file")`, also when the file isn't JLD2 at all), dispatches on `/format_version` to `_load_v1(f; on)`; unknown version: `ArgumentError` naming the file's version and `1`.
  - `_write_node(g, mn::MaterializedNode, retained::Bool)`, `_read_node(g; on) -> (node, mn)`: one node group as in the spec, children recursively. Node objects are built first (with the keyword `Node` constructor from the saved settings), and each child `MaterializedNode.node` is the same object as the parent `Node.children[i]` (`===`). A `GivenSystem` group rebuilds the `GivenSystem` once and uses it both as the node and as `children = Any[g]`. A leaf child stores `mode` and the basis (`_write_struct`); its `MaterializedNode.children[i]` entry is the basis, and the `Node.children[i]` entry is `mode => basis`.
  - When `retained = false` and a node has `retained.vectors !== nothing`, the vectors are left out, `/saved_without_retained = true`, and the loader sets `complete = false` on those nodes (the `retain` setting unchanged).
  - `assert_same_system(a::System, b::System; on=Array)` in `test/helpers.jl`: `@test`s that every array (energies, vectors, kept, dims, peak fields, ports, joint ports, retained) is `isequal` with equal `typeof`, node settings equal field by field (solver compared with its `on` replaced by `on` and `callback` by `nothing`), θ `===`-equal, terms and constant identical, `spec_equal` specs, and the tree shape equal, recursively.

- [ ] **Step 1: Write the failing tests** in `test/test_saveload_system.jl`, each a `@testset`, each saving to `tempname() * ".jld2"` and loading back:
  - `"real leaf"`: `dc_transmon()` materialized; `assert_same_system(load_system(p), sys)`.
  - `"complex leaf"`: `dc_transmon(periodic=true)`; also `eltype(load_system(p).root.vectors) == ComplexF64`.
  - `"two-level and three-level trees"`: `dc_two_transmons()` and `dc_chain3()`; plus node identity: for every non-leaf node `n` of the loaded tree and child index `i`, `materialized child .node === n.children[i]`.
  - `"given system"`: the first testset of `test_given.jl` (θ `(;)`, term-list build), round trip identical.
  - `"retained = false"`: `dc_fluxonium(compute=8, keep=lowest(5), retain=:always)`; loaded with `retained=false`: `root.retained.vectors === nothing`, `root.retained.complete == false`, `root.node.retain === :always`, and everything else identical; the file without retained vectors is smaller.
  - `"device chosen at load"`: a two-transmon tree with `solver=LOBPCG()` at the root and `compute=6`: `load_system(p).root.node.solver.on === Array`; `load_system(p; on=Vector).root.node.solver.on === Vector` (any array type passes through; the GPU itself is not exercised here).
  - `"failed save leaves the old file"`: save `sys` to `p`; then `@test_throws ArgumentError save_system(p, sys; metadata=(f=sin,))`; `load_system(p)` still equals `sys`; `!isfile(p * ".partial")`. (Task 6 adds metadata; until then, force the failure with a system whose θ holds an unstorable value, e.g. `(x="a"=>1,)`.)
  - `"not a system file"`: a text file and a JLD2 file holding only `x = 1`: `@test_throws "not a KronFreeCircuits system file" load_system(...)`.
  - `"unknown format version"`: copy a saved file, overwrite `/format_version` with `2` via `JLD2.jldopen(p, "r+")` (delete and rewrite the key), `@test_throws "version 2" load_system(p)`.

- [ ] **Step 2: Run to verify it fails.** `Pkg.test(test_args=["test_saveload_system.jl"])`. Expected: FAIL, `save_system` not defined.

- [ ] **Step 3: Implement `src/saveload/system.jl`** (augmented children are Task 5: until then `save_system` throws `ArgumentError("saving a system with augmented children is not supported yet")` when `!isempty(sys.augmented)`).

- [ ] **Step 4: Run to verify it passes.** Expected: PASS. Then the fast tier: `julia --project -e 'using Pkg; Pkg.test()'`; expected PASS (the export list grew).

- [ ] **Step 5: Commit**: `git commit -m "Save/load: save_system and load_system"` (the four files).

---

### Task 5: Augmented children

**Files:**
- Modify: `src/saveload/system.jl`
- Test: `test/test_saveload_augmented.jl`

**Interfaces:**
- Consumes: `_write_node`, `_read_node`, `_write_terms`, `_read_terms`, `_write_theta`, `_read_theta`.
- Produces: `/augmented/count` and `/augmented/<i>/{node_path, solved, basis, terms, theta, report/labels, report/taken, report/dropped}`. On load each `AugmentedChild(node, solved, basis, terms, θ, report)` has `node ===` the loaded tree's node at `node_path` (child indices from the root, a `Vector{Int}`). The tree position itself is read as an ordinary node group (results stored twice, as the spec accepts).

- [ ] **Step 1: Write the failing test** `"augmented child"`: build `augB` and `sysB` exactly as `test_augment.jl` lines 56–68 do (copy the setup, including its expected `@test_logs` warnings), save `sysB`, load it; `assert_same_system(loaded, sysB)`; `length(loaded.augmented) == 1`; `loaded.augmented[1].node === loaded.root.node.children[2]` (B's position); `isequal(loaded.augmented[1].basis, augB.basis)`; report fields equal; `_terms_identical(loaded.augmented[1].terms, augB.terms)`.

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL with "augmented children is not supported yet".

- [ ] **Step 3: Implement** writing and reading `/augmented`, removing the Task 4 refusal. `assert_same_system` gains the augmented comparison (node by tree position, solved node recursively, basis, terms, θ, report).

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit**: `git commit -m "Save/load: augmented children"`.

---

### Task 6: Partial reads and metadata

**Files:**
- Modify: `src/saveload/system.jl`
- Test: `test/test_saveload_partial.jl`

**Interfaces:**
- Produces:
  - `system_nodes(path) -> Vector{Vector{Int}}`: positions of node and given groups (not leaves), depth first, root (`Int[]`) first.
  - `load_node(path, position::AbstractVector{<:Integer}=Int[]) -> NamedTuple` `(; energies, kept, vectors, dims, ports, on)`: `ports::Dict{Tuple{Symbol,Port},Matrix}` and `on` the saved device name (a String). A position that names a leaf or no child: `ArgumentError` listing the valid positions.
  - `load_metadata(path) -> Dict{String,Any}`; `save_system`'s `metadata` (a `NamedTuple` or `AbstractDict` with `Symbol` or `String` keys) is validated before anything is written: values must be numbers, `Bool`, `String`, or arrays of numbers or strings, else `ArgumentError` naming the key.

- [ ] **Step 1: Write the failing tests**:
  - `"positions and one node"`: `dc_chain3()` system saved; `system_nodes(p) == [Int[], [1], [1, 1], [1, 2], [2]]`; `load_node(p, [1, 2]).energies == sys.root.children[1].children[2].energies` (and vectors, kept, dims, ports equal); `load_node(p).on == "Array"`; `@test_throws ArgumentError load_node(p, [3])`.
  - `"metadata"`: `save_system(p, sys; metadata=(note="fig 3", grid=[1, 2, 3], flux=0.25))`; `load_metadata(p) == Dict("note" => "fig 3", "grid" => [1, 2, 3], "flux" => 0.25)`.
  - `"unstorable metadata"`: `@test_throws "metadata" save_system(p2, sys; metadata=(f=sin,))` and `!isfile(p2)`.
  - Switch Task 4's `"failed save leaves the old file"` to the metadata failure.

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL, `system_nodes` not defined.

- [ ] **Step 3: Implement** the three functions and the metadata write/validation.

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit**: `git commit -m "Save/load: partial reads and metadata"`.

---

### Task 7: A loaded system works

**Files:**
- Test: `test/test_saveload_behaviour.jl` (no source changes expected; a failure here is a bug in Tasks 4–5)

- [ ] **Step 1: Write the tests** (each compares the loaded system `L` with the original `S`):
  - `"port, apply, decompose"`: `dc_two_transmons()`: `port(L, :varphi_1, nport()) == port(S, :varphi_1, nport())`; `decompose(L, L.root.node, 2) == decompose(S, S.root.node, 2)`; `bare_labels(L) == bare_labels(S)`.
  - `"derivatives"`: `dc_two_transmons()` with `retain=:always`: `derivatives(L; wrt=:E_J1)` energies `==` those of `S` (bitwise). With `retained=false`: equal to within `rtol=1e-8` (the re-diagonalization path), and the `@info` about re-diagonalizing is emitted (`@test_logs (:info, r"re-diagonalized") match_mode=:any`).
  - `"previous"`: `dc_two_transmons(keep=follow())` leaves kept `lowest(4)`, root keep `follow()`: `materialize(c.root, c.spec, θ2; previous=L)` gives energies and kept indices `==` to `previous=S`, for `θ2 = merge(c.θ, (g=0.31,))` (build `c.root` afresh for the new solve, as the tests do with `dc_two_transmons()` again).

- [ ] **Step 2: Run them.** `Pkg.test(test_args=["test_saveload_behaviour.jl"])`. Expected: PASS. If one fails, fix the cause in `src/saveload/` (not the test) and add a round-trip test in the owning task's file that would have caught it.

- [ ] **Step 3: Commit**: `git commit -m "Save/load: a loaded system works like the original"`.

---

### Task 8: Docs and the Python reader test

**Files:**
- Create: `docs/save-format.md` (format version 1: the layout from the spec, kept in sync with the code, and the Python example)
- Modify: `README.md` (a short "Saving a solved system" section linking to it), docstrings of the five public functions in `src/saveload/system.jl`
- Create: `test/python/test_saveload_h5py.jl`

**Interfaces:**
- Consumes: the public API.

- [ ] **Step 1: Write the Python-tier test**: save `dc_transmon(periodic=true)`'s system (complex vectors); with PythonCall, `h5py = pyimport("h5py")`; if the import fails, `@test_skip` with the message `"h5py is not installed in this Python; pip install h5py to run this test"`. Else set `h5py.get_config().complex_names = ("re", "im")`, read `root/energies` and `root/vectors`, and check `pyconvert(Vector{Float64}, E) == sys.root.energies` and `permutedims(pyconvert(Matrix{ComplexF64}, V)) == sys.root.vectors`.

- [ ] **Step 2: Run the Python tier**: `JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/home/eweissler/anaconda3/envs/cq-z4/bin/python julia --project=test/python test/python/runtests.jl`. Expected: PASS, with the h5py test skipped if `cq-z4` lacks h5py (it does as of 2026-11-21; do not install it: report the skip).

- [ ] **Step 3: Write `docs/save-format.md`, the README section and the docstrings.** The docstrings state: callbacks are not saved; `on` picks the device and the saved device is only recorded; `retained=false` makes derivatives slower and only tolerance-equal; files are uncompressed; a save that can't reproduce a coefficient exactly refuses. The format page includes the h5py snippet (`complex_names`, transposing matrices, decoding strings).

- [ ] **Step 4: Run the fast tier** `julia --project -e 'using Pkg; Pkg.test()'`. Expected: PASS.

- [ ] **Step 5: Commit**: `git commit -m "Save/load: format docs, README and the h5py test"`.

---

### After the last task

- [ ] Run the slow tier once: `KRONFREE_SLOW_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'` (about 10 minutes; 20 known `fix_phases!` warnings). Expected: PASS.
