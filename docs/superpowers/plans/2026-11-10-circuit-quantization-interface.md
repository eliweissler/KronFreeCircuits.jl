# circuit-quantization Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One Julia session defines a circuit with circuit-quantization, changes its coordinate frame inside a loop, solves, and writes a reproducible JSON record, with no file hand-off between languages.

**Architecture:** The core package gains two Python-free pieces: constant coefficients that never compile, and a `numeric_spec` constructor that builds a `Spec` from plain arrays. A package extension loaded with PythonCall adds the conversions from circuit-quantization objects to `Spec`, a numeric circuit handle for frame loops, and run records. circuit-quantization itself is changed separately under its own brief.

**Tech Stack:** Julia 1.13.2, PythonCall 0.9.29 or later (registry compat `julia = "1.10.0 - 1"`), the conda environment `circuit_quantization` (Python 3.12, SymPy 1.14) with circuit-quantization as an editable install.

**Spec:** `docs/superpowers/specs/2026-11-10-circuit-quantization-interface-design.md`

## Global Constraints

- `using KronFreeCircuits` without PythonCall must load and pass the default test tier; no file under `src/` may mention PythonCall or Python.
- The default test tier stays Python-free and fast; every Python-dependent test lives under `test/python/` and is never included by `test/runtests.jl`.
- The in-memory route must produce the same `Spec` as `parse_spec` on the JSON for the same input; do not add a second validation path for symbolic specs.
- The kronfree-spec format (version 1) does not change; metadata keys start with `_`.
- The loader's symmetry rule (relative `1e-12`, no absolute floor, mirrored in Python) is not loosened.
- Both routes to a framed spec enforce the same admissibility rules for `S` and the offset-charge convention of the spec's §4.3.
- Python interpreter for all Python-tier commands: `$CQ_PYTHON`, selected with `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE=<that path>`. The extension installs nothing.
- Work on a branch `cq-interface` off `main`; commit per task; do not push.
- Markdown and docstrings follow the repository's style: one sentence per line.

## Prerequisite: the circuit-quantization pre-changes

Tasks 5 to 9 need the changes in `docs/superpowers/plans/2026-11-10-circuit-quantization-prechanges.md`, done by another agent in `~/src/circuit-quantization`.
That brief leaves function names to its implementer, so this plan refers to them by role.
Before Task 5, read that work's report and fill in this table in this file; every later task uses these names.

| Role | Used as | Actual name (fill in) |
|---|---|---|
| Numeric Hamiltonian from a `Result` and named values | `NUMERIC(res, values)` | `cq.numeric_hamiltonian(res, values)` |
| Frame change with checks, new types, offset-charge convention | `FRAME(h, S)` | `h.transformed(S)` (`NumericHamiltonian.transformed`) |
| Write a run record | `TO_RECORD(res, ...)` | `cq.to_record(res, path=None, *, subs=None, **kwargs)` |
| Read a run record back to a `Result` | `FROM_RECORD(record)` | `cq.load_record(source)` |

Filled in on 2026-11-10 from circuit-quantization `9c26fd2` and its report, `docs/julia-interface-prechanges-report.md` in that repository.

Tasks 1 to 4 do not depend on the pre-changes and can start at once.

## Review Focus

1. **Wrong or missing Python environment.** A user without the environment variables set gets PythonCall's own managed Python, where circuit-quantization is absent; they should get a message naming the interpreter in use and the two variables to set, not a bare `ModuleNotFoundError`. (Task 3)
2. **A misspelled or missing parameter name in `values`.** The user should get an error naming the offending or missing parameter and listing the circuit's parameters, never a spec that silently keeps a symbol or drops a value. (Task 5)
3. **`S` that is not a plain `Matrix{Float64}`.** An integer matrix, an adjoint, a view, or an `SMatrix`-like array should work; a wrong size or a `NaN` entry should fail in Julia before any Python call, with the expected size in the message. (Task 5)
4. **Round-off asymmetry in arrays coming back from Python.** `numeric_spec` should symmetrize at round-off level and reject a real asymmetry, naming the pair. (Task 2)
5. **A numeric spec whose literal zeros matter.** A numeric `n_g` of exactly `0.0` must keep a grid node in real arithmetic exactly as the literal `"0"` does in a JSON spec; a folded coefficient must not change a node's arithmetic class. (Task 1)

---

### Task 1: Constant coefficients never compile

**Files:**
- Modify: `src/parameters.jl` (the `Coefficient` constructors and arithmetic, about lines 35–170)
- Test: `test/test_parameters.jl`, `test/test_terms.jl`

**Interfaces:**
- Produces: `KronFreeCircuits._Constant <: Function` with field `value::ComplexF64`, callable on any `NamedTuple`; `Coefficient(x::Number)` whose `f` is a `_Constant` and whose `ex` is the number; arithmetic (`*`, `+`, unary `-`, `Number * Coefficient`, `phase_factor`) that returns a constant `Coefficient` whenever every operand's `ex isa Number`.

- [ ] **Step 1: Write the failing tests** in `test/test_parameters.jl`

```julia
@testset "constant coefficients fold and never compile" begin
    K = KronFreeCircuits
    a, b = Coefficient(2.0), Coefficient(3)
    @test a.f isa K._Constant && isempty(a.deps)
    for c in (a * b, a + b, -a, 4 * a, K.phase_factor(Coefficient(0.5), -1))
        @test c.ex isa Number && c.f isa K._Constant
    end
    @test (a * b).ex == 6 && (a + b).ex == 5 && (-a).ex == -2 && (4 * a).ex == 8
    @test K.phase_factor(Coefficient(0.5), -1).ex ≈ exp(-0.5im)
    @test (a * b)((;)) === ComplexF64(6)
    @test (a * b)((x=1.0,)) === ComplexF64(6)          # extra parameters are ignored
    p = Coefficient("x", [:x])
    @test (a * p).ex isa Expr && !((a * p).f isa K._Constant)   # mixed stays symbolic
    @test K.iszero_expr(Coefficient(0.0) * a)
end
```

and in `test/test_terms.jl`

```julia
@testset "a numeric spec expands to constant terms and keeps its arithmetic class" begin
    spec = load_spec(fixture("fluxonium.json"))
    θ = (E_C=1.0, E_L=0.5, E_J=4.0, varphi_ext=0.3)      # use the fixture's actual parameter names
    num = substitute(spec, θ)
    @test all(t.coeff.f isa KronFreeCircuits._Constant for t in circuit_terms(num))
    root = Node([spec.modes[1].name => PhaseGrid(-12.0, 12.0, 201)]; solver=Dense(), keep=lowest(6))
    a, b = materialize(root, spec, θ), materialize(root, num, (;))
    @test energies(a) ≈ energies(b) rtol=1e-12
    @test arithmetic(a) == arithmetic(b)
end
```

Take the parameter names and a valid `PhaseGrid` call from the existing fluxonium tests in the same files.

- [ ] **Step 2: Run them and confirm they fail**

Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_parameters.jl","test_terms.jl"])'`
Expected: FAIL on `a.f isa K._Constant`.

- [ ] **Step 3: Implement**

`_make(ex::Number, deps)` builds `Coefficient(ex, Symbol[], _Constant(ComplexF64(ex)))` with no `RuntimeGeneratedFunction`.
Each arithmetic method and `phase_factor` checks whether all operand `ex` are `Number` and, if so, computes the number and returns `Coefficient(number)`; otherwise it behaves as now.
`ex` may then hold a complex number: check `src/specwrite.jl` (`expr_string`), `src/character.jl` (`classify`, `_test_points`) and `src/terms.jl` (`_iszero_ex`) and make each handle a `Complex` literal; a complex literal with nonzero imaginary part counts as complex character.

- [ ] **Step 4: Run the default tier and the slow tier**

Run: `julia --project -e 'using Pkg; Pkg.test()'` then `KRONFREE_SLOW_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS, with no change in any existing expectation.

- [ ] **Step 5: Commit** (`Coefficient: constants fold and never compile`)

---

### Task 2: `numeric_spec`, a `Spec` from plain arrays

**Files:**
- Modify: `src/spec.jl` (new function after `parse_spec`), `src/KronFreeCircuits.jl` (export `numeric_spec`)
- Test: `test/test_spec.jl`

**Interfaces:**
- Consumes: Task 1's constant `Coefficient(x::Number)`.
- Produces:

```julia
numeric_spec(; modes::AbstractVector{<:Pair{Symbol,Symbol}},      # name => :c | :e | :h
               E_C::AbstractMatrix{<:Real}, E_L::AbstractMatrix{<:Real},
               flux_linear::AbstractVector{<:Real}, n_g::AbstractVector{<:Real},
               cosines::AbstractVector = Tuple[],                 # (energy, omega, offset) each
               constant::Real = 0.0, symmetry_tol::Real = 1e-10) -> Spec
```

The result has `parameters == Symbol[]`, no charge cosines, and constant coefficients throughout.

- [ ] **Step 1: Write the failing tests** in `test/test_spec.jl`

```julia
@testset "numeric_spec" begin
    modes = [:a => :c, :b => :e]
    EC = [1.0 0.2; 0.2 0.5]; EL = [0.0 0.0; 0.0 0.3]
    cos1 = (4.0, [1, 2.5], 0.1)
    s = numeric_spec(; modes, E_C=EC, E_L=EL, flux_linear=[0.0, 0.07], n_g=[0.25, 0.0], cosines=[cos1])
    json = """{"format":"kronfree-spec","version":1,"parameters":[],
      "modes":[{"name":"a","type":"c"},{"name":"b","type":"e"}],
      "E_C":[["1.0","0.2"],["0.2","0.5"]],"E_L":[["0","0"],["0","0.3"]],
      "flux_linear":["0","0.07"],"n_g":["0.25","0"],
      "cosines":[{"energy":"4.0","omega":[1,2.5],"offset":"0.1"}]}"""
    p = parse_spec(json)
    @test [m.name => m.type for m in s.modes] == modes
    @test all(s.E_C[i,j]((;)) == p.E_C[i,j]((;)) && s.E_L[i,j]((;)) == p.E_L[i,j]((;)) for i in 1:2, j in 1:2)
    @test s.cosines[1].omega == p.cosines[1].omega && s.cosines[1].energy((;)) == 4
    @test KronFreeCircuits.iszero_expr(s.n_g[2])                          # literal zero survives (Review Focus 5)
    # round-off asymmetry is symmetrized (Review Focus 4)
    s2 = numeric_spec(; modes, E_C=EC, E_L=[0.0 1e-26; 0.0 0.3], flux_linear=[0,0], n_g=[0,0])
    @test s2.E_L[1,2]((;)) == s2.E_L[2,1]((;))
    @test_throws "E_C[1,2]" numeric_spec(; modes, E_C=[1.0 0.2; 0.3 0.5], E_L=EL, flux_linear=[0,0], n_g=[0,0])
    @test_throws "integer" numeric_spec(; modes, E_C=EC, E_L=EL, flux_linear=[0,0], n_g=[0,0], cosines=[(1.0, [0.5, 1], 0.0)])
    @test_throws "type" numeric_spec(; modes=[:a => :x, :b => :e], E_C=EC, E_L=EL, flux_linear=[0,0], n_g=[0,0])
    @test_throws ArgumentError numeric_spec(; modes, E_C=[1.0 NaN; NaN 0.5], E_L=EL, flux_linear=[0,0], n_g=[0,0])
    @test_throws ArgumentError numeric_spec(; modes, E_C=ones(3,3), E_L=EL, flux_linear=[0,0], n_g=[0,0])
end
```

- [ ] **Step 2: Run and confirm failure** (`UndefVarError: numeric_spec`)

- [ ] **Step 3: Implement `numeric_spec`**

Validation, in order: shapes against `length(modes)`; mode types in `_MODE_TYPES`; all entries finite; for `E_C` and `E_L`, an asymmetry `|M[i,j] − M[j,i]| ≤ symmetry_tol · max|M|` is replaced by the mean on both sides, a larger one throws naming `E_C[i,j]`; an `omega` entry on a compact mode within `_INTEGER_TOL` of an integer is rounded, otherwise it throws.
Reuse the existing helpers of `parse_spec` where they apply; do not go through strings.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_spec.jl"])`, expected PASS

- [ ] **Step 5: Commit** (`numeric_spec: a Spec from plain arrays`)

---

### Task 3: Extension scaffold and the Python test tier

**Files:**
- Modify: `Project.toml` (`[weakdeps]` PythonCall, `[extensions]` `KronFreeCircuitsPythonCallExt = "PythonCall"`, `[compat]` `PythonCall = "0.9.29"`)
- Create: `src/pyinterface.jl` (stubs and docstrings), included and exported from `src/KronFreeCircuits.jl`
- Create: `ext/KronFreeCircuitsPythonCallExt.jl`
- Create: `test/python/Project.toml`, `test/python/runtests.jl`, `test/python/test_import.jl`

**Interfaces:**
- Produces, in the core, exported:

```julia
function circuit_quantization end      # () -> the Python module
struct QuantizedCircuit{H}             # filled by the extension
    handle::H                          # Python numeric Hamiltonian, base frame
    values::NamedTuple                 # the parameter values used
end
function run_record end
function load_record end
```

A `MethodError` on `circuit_quantization`, `run_record` or `load_record` with no extension loaded carries the hint `load PythonCall to use the circuit-quantization interface` (`Base.Experimental.register_error_hint` in `__init__`).
- Produces, in the extension: `circuit_quantization()` returning the cached `pyimport("circuit_quantization")`.

- [ ] **Step 1: Verify the stack before writing code**

Run: `JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("PythonCall"); using PythonCall; println(pyimport("circuit_quantization").__version__)'`
Expected: prints `0.2.0` (or the current version) on Julia 1.13.2.
If PythonCall does not load on 1.13.2, stop and report; the design's fallback is a separate glue package, which is Eli's decision.

- [ ] **Step 2: Write the tests** in `test/python/test_import.jl`

```julia
@testset "module import" begin
    cq = circuit_quantization()
    @test pyconvert(Bool, pyhasattr(cq, "quantize"))
    @test circuit_quantization() === cq                         # cached
    msg = Base.get_extension(KronFreeCircuits, :KronFreeCircuitsPythonCallExt)._import_help("/some/python")
    @test occursin("/some/python", msg) && occursin("JULIA_PYTHONCALL_EXE", msg) && occursin("JULIA_CONDAPKG_BACKEND", msg)
end
```

`test/python/runtests.jl` does `using Test, KronFreeCircuits, PythonCall` and includes every `test_*.jl` in its directory; `test/python/Project.toml` lists `KronFreeCircuits` (dev path `../..`), `PythonCall`, `Test`, `JSON3`, `LinearAlgebra`.
Also add to `test/test_helpers.jl` (default tier): `@test_throws "load PythonCall" circuit_quantization()`.

- [ ] **Step 3: Implement** the stubs, the hint, the extension module, and `_import_help(exe::AbstractString)::String`; a failed import rethrows as an `ErrorException` whose message is `_import_help` of the interpreter PythonCall is using.

- [ ] **Step 4: Run both tiers**

Run: `julia --project -e 'using Pkg; Pkg.test()'` (PASS, no Python), then
`JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=test/python test/python/runtests.jl` (PASS).
The second command is "the Python tier" in later tasks.

- [ ] **Step 5: Commit** (`PythonCall extension scaffold and Python test tier`)

---

### Task 4: `Spec(res)`, the in-memory symbolic route

**Files:**
- Modify: `ext/KronFreeCircuitsPythonCallExt.jl`
- Create: `test/python/circuits.jl` (builders), `test/python/test_spec_from_result.jl`

**Interfaces:**
- Produces: `KronFreeCircuits.Spec(res::Py; subs=nothing, simplify::Bool=true, mode_names=nothing) -> Spec`, implemented as `parse_spec` on the JSON text of `cq.to_spec(res; ...)`; `subs` is a Julia `Dict` or `NamedTuple` of name ⇒ value or name ⇒ name, mapped to the circuit's own SymPy symbols by name.
- Produces: `test/python/circuits.jl` with `fixture_result(name::String) -> (res::Py, kwargs::NamedTuple)` for each name in `test/fixtures/generated/`, obtained by importing `test/fixtures/generate_specs.py` as a module and calling its builders.

- [ ] **Step 1: Write the failing test**

```julia
@testset "in-memory Spec equals the committed JSON: $name" for name in
        ("transmon", "fluxonium", "two_transmons", "three_transmons", "zero_pi", "full_zero_pi", "two_zero_pi")
    res, kw = fixture_result(name)
    a = Spec(res; kw...)
    b = load_spec(joinpath(@__DIR__, "..", "fixtures", "generated", name * ".json"))
    @test a.parameters == b.parameters
    @test [(m.name, m.type) for m in a.modes] == [(m.name, m.type) for m in b.modes]
    θ = NamedTuple{Tuple(a.parameters)}(Tuple(0.5 .+ rand(length(a.parameters))))
    same(x, y) = isapprox(x(θ), y(θ); rtol=1e-12, atol=1e-14)
    @test all(same.(a.E_C, b.E_C)) && all(same.(a.E_L, b.E_L))
    @test all(same.(a.flux_linear, b.flux_linear)) && all(same.(a.n_g, b.n_g)) && same(a.constant, b.constant)
    @test length(a.cosines) == length(b.cosines)
    @test all(c.omega == d.omega && same(c.energy, d.energy) && same(c.offset, d.offset) for (c, d) in zip(a.cosines, b.cosines))
end
```

- [ ] **Step 2: Run the Python tier and confirm failure** (no method `Spec(::Py)`)

- [ ] **Step 3: Implement.** If `generate_specs.py` runs its generation at import, refactor it so the builders are importable without writing files, and confirm `git diff test/fixtures/generated` is empty after re-running it.

- [ ] **Step 4: Run the Python tier**, expected PASS (acceptance 1)

- [ ] **Step 5: Commit** (`Spec(res): circuit-quantization Result to Spec in memory`)

---

### Task 5: `QuantizedCircuit` and the loop route

**Files:**
- Modify: `ext/KronFreeCircuitsPythonCallExt.jl`
- Create: `test/python/kite.jl` (the KITE built through `circuit_quantization()`), `test/python/test_loop_route.jl`

**Interfaces:**
- Consumes: `NUMERIC`, `FRAME` (prerequisite table); `numeric_spec` (Task 2).
- Produces:

```julia
QuantizedCircuit(res::Py; values) -> QuantizedCircuit{Py}       # values: NamedTuple or AbstractDict of name => Real
KronFreeCircuits.Spec(qc::QuantizedCircuit{Py}) -> Spec          # base frame
KronFreeCircuits.Spec(qc::QuantizedCircuit{Py}, S::AbstractMatrix{<:Real}) -> Spec
```

- Produces: `test/python/kite.jl` with `kite() -> (res::Py, θ::NamedTuple, frames::NamedTuple)`, where `frames` has `published` (identity), `normal` and `optimum`. Port the circuit from `kite` in `~/src/circuit-quantization/examples/product_state_demo.py` (parameters `E_J = 16.83`, `E_CJ = 4.82`, `E_CS = 0.072`, `E_L = 1.27`, asymmetry 0.03, half flux), take `optimum` from `S_OPT` in `examples/hierarchical_kite.py`, and `normal` from `normal_mode_transformation`.

- [ ] **Step 1: Write the failing tests**

```julia
res, θ, frames = kite()
qc = QuantizedCircuit(res; values=θ)

@testset "loop route equals the symbolic route" for S in (frames.published, frames.normal, frames.optimum)
    a = Spec(qc, S)
    b = substitute(Spec(res.transform(to_sympy_frame(S))), θ)     # helper from Task 6; until then build b by hand in Python
    @test [m.type for m in a.modes] == [m.type for m in b.modes]
    @test all(isapprox.(evalc.(a.E_C), evalc.(b.E_C); rtol=1e-10, atol=1e-12))   # evalc(c) = real(c((;)))
    # likewise E_L, flux_linear, n_g, and each cosine's energy, omega, offset
end

@testset "offset charges follow the sector convention" begin   # acceptance 8
    θg = merge(θ, (n_g1 = 0.3,))                              # the KITE's compact sector label
    a = Spec(QuantizedCircuit(res; values=θg), frames.optimum)
    @test all(iszero(evalc(a.n_g[j])) for j in eachindex(a.modes) if a.modes[j].type != :c)
end

@testset "S inputs" begin                                     # acceptance 3, Review Focus 3
    I3 = [1.0 0 0; 0 1 0; 0 0 1]
    @test Spec(qc, I3).E_C[1,1]((;)) == Spec(qc).E_C[1,1]((;))
    @test Spec(qc, Int[1 0 0; 0 1 0; 0 0 1]) isa Spec
    @test Spec(qc, permutedims(frames.optimum)') isa Spec      # an Adjoint wrapper
    @test Spec(qc, @view frames.optimum[:, :]) isa Spec
    @test_throws "3×3" Spec(qc, ones(2, 2))
    @test_throws ArgumentError Spec(qc, [NaN 0 0; 0 1 0; 0 0 1])
    bad = copy(frames.optimum); bad[2, 1] = 0.1               # a noncompact row feeding the compact column
    @test_throws "S" Spec(qc, bad)
    half = copy(I3); half[1, 1] = 0.5
    @test_throws "S" Spec(qc, half)                            # compact block not unimodular
end

@testset "parameter names" begin                               # Review Focus 2
    @test_throws "E_JJ" QuantizedCircuit(res; values=merge(θ, (E_JJ = 1.0,)))
    @test_throws "E_L" QuantizedCircuit(res; values=Base.structdiff(θ, (E_L = 0,)))
end
```

Use the KITE's actual parameter names from `kite()` in place of `E_JJ` and `E_L`.

- [ ] **Step 2: Run the Python tier and confirm failure**

- [ ] **Step 3: Implement.**
The size and finiteness checks run in Julia before any Python call; admissibility is checked by `FRAME` and its `ValueError` is rethrown as an `ArgumentError` with the Python message.
`S` is passed as a NumPy `float64` array made from `Matrix{Float64}(S)`.
The returned arrays are converted with `pyconvert` and handed to `numeric_spec`; mode names come from the Python object.
An unknown name in `values`, or a circuit parameter with no value, throws an `ArgumentError` that names it and lists the circuit's parameters.

- [ ] **Step 4: Run the Python tier**, expected PASS

- [ ] **Step 5: Measure and record**

Time `Spec(qc, S)` for the KITE after warm-up (`@elapsed` over 1000 calls) and record the per-call figure in the commit message; the target is under 5 ms.
If it misses, record where the time goes (crossing, Python, `numeric_spec`) before changing anything.

- [ ] **Step 6: Commit** (`QuantizedCircuit and the numeric frame-change route`)

---

### Task 6: The final symbolic route in a fixed frame

**Files:**
- Modify: `ext/KronFreeCircuitsPythonCallExt.jl`
- Test: `test/python/test_final_route.jl`

**Interfaces:**
- Produces: the keyword `frame` on `Spec(res::Py; frame::Union{Nothing,AbstractMatrix{<:Real}}=nothing, ...)`, and the internal `to_sympy_frame(S, types) -> Py` (a SymPy matrix).

`to_sympy_frame` converts each entry of the compact block to the nearest integer (it must be within `1e-12` of one) and every other entry to an exact rational from its shortest round-trip decimal string (`sympy.Rational(repr(x))`); exact zeros stay exact zeros.

- [ ] **Step 1: Write the failing test**

```julia
res, θ, frames = kite()
@testset "final route keeps parameters symbolic in a fixed frame" begin
    t = @elapsed s = Spec(res; frame=frames.optimum)
    @test !isempty(s.parameters)
    n = Spec(QuantizedCircuit(res; values=θ), frames.optimum)
    @test all(isapprox(s.E_C[i,j](θ), n.E_C[i,j]((;)); rtol=1e-10, atol=1e-12) for i in 1:3, j in 1:3)
    # likewise E_L, flux_linear, n_g and the cosines
    @test t < 60                                              # a regression guard, not a target
    @test_throws ArgumentError Spec(res; frame=ones(2, 2))
end
```

- [ ] **Step 2: Run and confirm failure**; **Step 3: implement**; **Step 4: run the Python tier**, expected PASS, and replace the hand-built `b` in Task 5's test with this route.

- [ ] **Step 5: Record the timing** for the KITE and for `coupled_zero_pis` (from circuit-quantization's `tests/circuits.py`) in the commit message; the design expects seconds, against 60 s for a float `S`.

- [ ] **Step 6: Commit** (`Spec(res; frame): symbolic spec in a fixed frame`)

---

### Task 7: Run records

**Files:**
- Modify: `ext/KronFreeCircuitsPythonCallExt.jl`
- Test: `test/python/test_run_record.jl`

**Interfaces:**
- Consumes: `TO_RECORD`, `FROM_RECORD`.
- Produces:

```julia
run_record(res::Py; frame=nothing, values=nothing, metadata::NamedTuple=(;)) -> String   # JSON text
load_record(source::AbstractString) -> (; result::Py, spec::Spec, frame, values)          # JSON text or a path
```

The JSON is a kronfree-spec version 1 document (so `parse_spec` accepts it) with `_circuit`, `_frame`, `_values` and `_generator`; the extension adds `_generator.kronfree` with this package's version and git commit, and merges `metadata` under `_metadata`.

- [ ] **Step 1: Write the failing test**

```julia
res, θ, frames = kite()
@testset "a run record rebuilds the circuit and frame" begin     # acceptance 4
    text = run_record(res; frame=frames.optimum, values=θ, metadata=(note="kite test",))
    @test parse_spec(text) isa Spec                               # still a valid spec
    back = load_record(text)
    @test back.frame ≈ frames.optimum && back.values == θ
    a, b = Spec(res; frame=frames.optimum), Spec(back.result; frame=back.frame)
    @test a.parameters == b.parameters
    @test all(isapprox(a.E_C[i,j](θ), b.E_C[i,j](θ); rtol=1e-12) for i in 1:3, j in 1:3)
    path = joinpath(mktempdir(), "run.json"); write(path, text)
    @test load_record(path).values == θ
    @test occursin("\"_generator\"", text) && occursin("kronfree", text)
end
```

- [ ] **Step 2: Run and confirm failure**; **Step 3: implement**; **Step 4: run the Python tier**, expected PASS.

- [ ] **Step 5: Commit** (`run_record and load_record`)

---

### Task 8: The KITE in three frames, in one session

**Files:**
- Create: `bench/kite_frames/run.jl`, `bench/kite_frames/Project.toml`, `bench/kite_frames/README.md`

**Interfaces:**
- Consumes: everything above. Produces no API.

- [ ] **Step 1: Write `run.jl`**

One script that defines the KITE through `circuit_quantization()` (reuse `test/python/kite.jl`), and for each of the three frames builds the spec by the loop route, solves a three-leaf tree (`ChargeBasis` for the compact mode, `PhaseGrid` for the two extended modes, a `Dense` root keeping the lowest 4), and prints the four lowest energies.
It then times, after warm-up, 200 iterations of "random admissible `S` near the optimum → `Spec(qc, S)` → `materialize`" and prints the mean time of the frame change alone and of the frame change through the end of `materialize`'s setup (everything before the first eigensolve).
It ends by writing a run record for the optimum frame.

- [ ] **Step 2: Run it**

Run: `JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=bench/kite_frames bench/kite_frames/run.jl`
Expected: the three frames' lowest energies agree with each other to the discretization error of the chosen grids (the paper reports agreement to `1e-5` GHz between frames with converged bases), the ground energy is near `−9.524` GHz, and the timings print.

- [ ] **Step 3: Check the compilation claim**

In the timing loop, assert that the number of compiled coefficient functions does not grow: every `t.coeff.f` of `circuit_terms(spec)` is a `KronFreeCircuits._Constant`.
Record both timings in the README (acceptance 2); if the frame change exceeds 5 ms, explain where the time goes.

- [ ] **Step 4: Commit** (`bench/kite_frames: define, reframe and solve in one session`)

---

### Task 9: The three-fluxonium bench without hand assembly

**Files:**
- Modify: `bench/hensel_three_fluxonia/quantize_three_fluxonia.py` (about lines 228–263), its `README.md`

`bench/hensel_three_fluxonia/` is untracked as of 2026-11-10; confirm with Eli that it may be edited and committed before starting.

- [ ] **Step 1: Save the current outputs** (`three_fluxonia.json`, `three_fluxonia_partial.json`) under a temporary name for comparison.

- [ ] **Step 2: Replace the hand-assembled dict with `cq.to_spec`** (now tolerant of round-off asymmetry), keeping the same metadata and file names.

- [ ] **Step 3: Regenerate and compare**

Run the script with the environment's Python, then check in Julia that the old and new specs agree: same modes, and every coefficient equal to `1e-12` relative at the bench's parameter point.

- [ ] **Step 4: Smoke run**

Run: `HENSEL_SMOKE=1 julia --project bench/hensel_three_fluxonia/run.jl`
Expected: completes, with the same levels as the last recorded smoke result to the precision the README states (acceptance 5).

- [ ] **Step 5: Commit** (`hensel_three_fluxonia: spec from to_spec, no hand assembly`)

---

### Task 10: Documentation and the Python-free guarantee

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` (§2 anti-goals, §15, new §20j), `README.md`, `docs/superpowers/STATUS.md`
- Test: `test/test_helpers.jl`

- [ ] **Step 1: Write the guard test** (default tier)

```julia
@testset "the core never mentions Python" begin
    src = joinpath(pkgdir(KronFreeCircuits), "src")
    for f in readdir(src; join=true)
        text = read(f, String)
        @test !occursin("PythonCall", text) && !occursin("pyimport", text)
    end
    @test Base.get_extension(KronFreeCircuits, :KronFreeCircuitsPythonCallExt) === nothing
end
```

The hint text in `src/pyinterface.jl` says "load PythonCall"; exempt that one string by matching `using PythonCall` and `PythonCall.` in the test instead, if needed.

- [ ] **Step 2: Run the default tier in a fresh environment without PythonCall**, expected PASS (acceptance 6).

- [ ] **Step 3: Amend the architecture spec**

§2: the anti-goal becomes "a runtime dependency on Python *in the core*".
§15: move PythonCall from "deliberately absent" to "weak dependency, extension only".
New §20j (dated): the extension, the two routes, the offset-charge convention, constant folding, `numeric_spec`, run records, the measured timings from Tasks 5, 6 and 8, and the superseded sentence of §20b.

- [ ] **Step 4: README and STATUS**

README: a section "Defining circuits from Julia" with the environment variables, the session of the spec's §4.2 using the real names, and the measured timings.
STATUS.md: record the project as done, the Python tier command, and that the mean-field proposal and plan of 2026-10-28 are superseded by the 2026-11-10 design.

- [ ] **Step 5: Run every tier once more** (default, slow, Python) and **commit** (`Docs: circuit-quantization interface; core stays Python-free`)

---

## Acceptance map

| Spec §5 item | Task |
|---|---|
| 1. In-memory spec equals committed JSON | 4 |
| 2. KITE in three frames in one session, timing | 5, 8 |
| 3. Float identity succeeds, inadmissible `S` fails on both routes | 5, 6 |
| 4. Run record rebuilds circuit and frame | 7 |
| 5. Three-fluxonium bench without hand assembly | 9 |
| 6. Core loads and passes without PythonCall | 3, 10 |
| 7. Architecture spec and STATUS amended | 10 |
| 8. Loop route equals final route with nonzero offset charge | 5, 6 |
