# Plan 4: Hamiltonian save and load in circuit-quantization (the kronfree-spec format)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. This plan spans two repositories: `~/src/circuit-quantization` (Python, Tasks 1–4) and `~/src/KronFreeCircuits.jl` (Julia, Tasks 5–7). Work on a branch in each, never on `main`; Eli merges.

**Goal:** `circuit-quantization` gains a general save/load module for its quantized Hamiltonians, in the `kronfree-spec` JSON format, which it uses itself (save a quantization, load it back into a SymPy Hamiltonian without re-quantizing) and which KronFreeCircuits reads, and KronFreeCircuits' test fixtures are checked against files the exporter generates, so the circuit → Python → JSON → Julia path has provenance instead of hand transcription.

**Why now (Eli, 2026-10-19):** the plan order moved the exporter ahead of automatic differentiation; every spec fixture in KronFreeCircuits' `test/fixtures/` is transcribed by hand from the scqubits source, and the round trip has never run.

**Architecture:** One new Python module, `src/circuit_quantization/io.py` (Eli, 2026-10-19: a general save/load module the package uses itself, not a KronFreeCircuits-specific exporter), with three public functions: `to_spec(res, …) -> dict` (reads `res.ham`, `res.lag`, `res.coords`, `res.B`), `save_spec(res, path, …)` (writes JSON), and `load_spec(source) -> HamiltonianSpec` (a frozen dataclass whose `hamiltonian()` rebuilds the SymPy `H` in GHz). The module's own correctness test is the round trip `load_spec(to_spec(res)).hamiltonian() == res.H`. The format identifier stays `"kronfree-spec"` (KronFreeCircuits' loader requires it); the names in Python are neutral. On the Julia side, the coefficient parser accepts SymPy's dotted operators, and a provenance test compares exporter-generated specs with the hand-written fixtures.

**Tech stack:** Python ≥ 3.11, SymPy 1.14 (`julia_code` printer, `parse_expr`), the package's existing pytest suite; Julia 1.13, KronFreeCircuits' `parse_spec`/`Coefficient`.

**Spec:** the format is §5 of `~/src/KronFreeCircuits.jl/docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` (§5.1 source, §5.2 content, §5.3 loading), with §20a (dotted operators parked as load-bearing), §20b (the later `operators`/`noise_channels` sections) and §20f "Real inputs" (no `im`, `cis`, `cispi` in spec expressions). The Hamiltonian it serializes is `circuit-quantization`'s `hamiltonian()` (`src/circuit_quantization/hamiltonian.py`) and its notes (`docs/mathematical-contract.md`).

## Decisions to confirm with Eli (assumed here; the executing agent asks before Task 1)

The brainstorm stopped before these were settled; each is the planner's call and is easy to change before work starts.

1. **"Exportable and loadable by itself, sharing the same spec"** is read as: the Python package both writes and reads the format, without Julia; the round trip `load_spec(to_spec(res)).hamiltonian() == res.H` is the module's test; the format is documented once, in a format document both packages cite (Task 1). If Eli meant something else (e.g. a JSON Schema file shipped by both packages), adjust Task 1.
2. **Where the format document lives:** `~/src/circuit-quantization/docs/spec-format.md` (the producer owns it); KronFreeCircuits' spec §5.2 gets a one-line pointer to it with the format version. Alternative: keep it in the KronFreeCircuits spec and have the Python package point there.
3. **Parameters and units:** the exporter writes whatever free symbols remain after `subs_constants` and an optional user `subs` mapping, in first-appearance order. Circuits built with SI symbols (`C_J` in farads) export valid but badly scaled coefficients; the docs recommend the GHz parametrization the README already uses (`C=ghz_to_farad(E_C)`), and `subs` lets a user convert an SI-built circuit (`{C_J: ghz_to_farad(E_CJ)}`).
4. **Scope:** no `charge_cosines` (circuit-quantization has no phase-slip element), and no `operators`/`noise_channels` sections (§20b; a later plan). Format version stays `1`; provenance goes into an underscore key (`_generator`: package name, version, and the circuit's branch list), which the Julia loader already ignores.
5. **Dotted operators:** SymPy's `julia_code` prints `x .^ 2 .* y`; the Julia `Coefficient` parser normalizes `.* ./ .^ .+ .-` to their scalar forms (the parked §20a item), and the exporter writes `julia_code` output verbatim.

## Global constraints

- The format is exactly §5.2's: keys `format` (`"kronfree-spec"`), `version` (`1`), `units` (`"GHz"`), `parameters`, `modes` (`name`, `type` ∈ `c`, `e`, `h`), `E_C`, `E_L` (square matrices of expression strings), `flux_linear`, `n_g` (vectors of expression strings), `cosines` (`energy`, `omega`, `offset`), optional `charge_cosines`, optional `constant`; unknown keys only with a leading underscore.
- The Hamiltonian the file encodes is `4 (n − n_g)ᵀ E_C (n − n_g) + ½ φᵀ E_L φ + φᵀ flux_linear − Σ_k energy_k cos(omega_kᵀ φ + offset_k) + constant`, identical to `hamiltonian.py`'s `H` (charging, inductive, inductive-flux cross term, Josephson).
- `omega` entries are numbers (no free symbols; refuse to export otherwise, §5.1), integers on compact modes (refuse otherwise, naming the junction and mode).
- No spec expression may contain `im`, `cis` or `cispi` (§20f "Real inputs"); the exporter refuses such an expression naming the field.
- Every expression string uses only what KronFreeCircuits' `Coefficient` accepts: numbers, declared parameters, `pi`/`π`/`ℯ`, and calls of `+ - * / ^ // sqrt exp cos sin tan cosh sinh tanh log abs sign conj real imag` (plus the dotted operators after Task 5). Parameter names are valid Julia identifiers and not reserved (`pi`, `π`, `im`, `ℯ`, `e`, or an allowed function name).
- Only dynamical modes (`c`, `e`, `h`) are exported, in `res.coords.types` order, named `varphi_1…varphi_N` as in `hamiltonian.py`.
- The new Python code lives in the new file `src/circuit_quantization/io.py` (inside the package, so `import io` elsewhere still resolves to the standard library under absolute imports; use relative imports within the package); existing modules change only to export the public names from `__init__.py`.
- Python is never a runtime or test-time dependency of KronFreeCircuits: generated fixtures are committed files; the generator script runs only when fixtures are regenerated.
- Prose one sentence per line (both repositories' style); docstrings follow `circuit-quantization`'s reST `:param:` style.
- Commits end with a `Co-Authored-By:` line naming the model that wrote them.

## Review focus

1. A circuit whose automatic basis gives a non-integer or symbolic `omega` on a compact mode must be refused with a message naming the junction and the mode, never written. (Task 2)
2. Voltage-source circuits put `V` symbols into `n_g`; external fluxes enter `offset` and `flux_linear`. Both must round-trip exactly, including the sign conventions of `n_g` and `B_J φ_ext`. (Tasks 2–3)
3. A circuit with no junctions, no inductors, or a single mode must export valid JSON (empty `cosines`, zero `E_L`, 1×1 matrices). (Task 2)
4. A float coefficient written by `julia_code` must survive the round trip to within 1 ulp-level relative error, and an exact Rational must survive exactly. (Task 3)
5. A spec written by the exporter must load in KronFreeCircuits' `parse_spec` (Task 6 runs every generated file through it).

---

### Task 1: The format document

**Files:** Create `~/src/circuit-quantization/docs/spec-format.md`; modify `~/src/KronFreeCircuits.jl/docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` (§5.2: one line pointing to the document and its version) — the Julia edit is committed in Task 7 with the rest of the Julia side.

- [ ] Write the document from §5.2–§5.3 and §20f "Real inputs": every key, its type and meaning, the Hamiltonian formula above, mode types, the integer rule for compact modes, the expression grammar (the allowlist above, dotted operators accepted), underscore metadata keys, units (GHz; phases dimensionless; charge in units of 2e), and a complete small example (the fluxonium). One sentence per line.
- [ ] Commit (circuit-quantization branch).

### Task 2: `to_spec` and `save_spec`

**Files:** Create `src/circuit_quantization/io.py`; modify `src/circuit_quantization/__init__.py` (exports); Test `tests/test_io.py`.

**Interfaces:**
- `to_spec(res, *, subs=None, simplify=True) -> dict`: the JSON-ready dict. `subs` is applied before `subs_constants`. Fields come from `res.ham.E_C`, `res.ham.E_L`, `res.ham.n_g`, the cross-term vector `res.coords.omega_L_tilde * res.ham.E_Lb * B_L * res.ham.varphi_ext` (`B_L = res.B[list(res.lag.L), :]`), and per junction `k`: `energy = res.lag.EJ[k]`, `omega = res.coords.omega_J_tilde[:, k]`, `offset = (B_J * varphi_ext)[k]` (`B_J = res.B[list(res.lag.J), :]`); `constant = "0"`. Each expression is `subs_constants`-substituted, `sym.cancel`-ed (and `sym.nsimplify`-free: keep exact Rationals exact), then printed with `sym.julia_code`. `parameters` is the sorted-by-first-appearance list of free symbols over all fields. `_generator` records `{"package": "circuit-quantization", "version": …, "branches": […]}`.
- `save_spec(res, path, **kwargs) -> dict`: writes `to_spec(res, **kwargs)` with `json.dump(…, indent=2)` and returns the dict.
- Raises `ValueError` for: a symbolic `omega` entry; a non-integer `omega` on a compact mode; an expression containing `I`, `cis`-like `exp(I·…)` forms, or a call outside the allowlist after printing; an invalid or reserved parameter name.

- [ ] **Failing tests first:** the fluxonium of the README exports the dict with one `e` mode, `E_C [["E_C"]]`-equivalent (compare as SymPy after loading the string, not as text), one cosine with `omega == [1]`, `flux_linear` linear in `varphi_ext`; `zero_pi()` and `harmonium()` from `tests/circuits.py` export with integer `omega` on compact modes; a circuit forced into a basis with a half-integer compact `omega` raises `ValueError` naming the junction; an LC oscillator (no junction) exports `cosines == []`; every printed string matches the allowlist regex.
- [ ] Implement; run `pytest tests/test_io.py`; then the full `pytest`.
- [ ] Commit.

### Task 3: `load_spec` and the round trip

**Files:** `src/circuit_quantization/io.py`; `tests/test_io.py`.

**Interfaces:**
- `@dataclass(frozen=True) class HamiltonianSpec`: `parameters: tuple[sym.Symbol, …]`, `modes: tuple[tuple[str, str], …]`, `E_C`, `E_L` (`sym.ImmutableMatrix`), `flux_linear`, `n_g` (`sym.ImmutableMatrix` columns), `cosines: tuple[tuple[sym.Expr, tuple[sym.Rational | float, …], sym.Expr], …]`, `constant: sym.Expr`; `hamiltonian(self) -> sym.Expr` builds the Global-constraints formula with `varphi_j`, `n_j` symbols named and assumed exactly as `hamiltonian.py` does (real).
- `load_spec(source) -> HamiltonianSpec` where `source` is a path, a JSON string or a dict; it validates like KronFreeCircuits' `parse_spec` (format, version, units, keys, mode types, integer `omega` on compact modes, square matrices of the right size, every expression's symbols declared) and parses each expression string with a restricted parser: dotted operators mapped to scalar, `^` to `**`, `a//b` to `Rational(a, b)`, `pi`/`π` to `sym.pi`, `ℯ` to `sym.E`, and only the allowlisted functions in the local namespace (`sym.parsing.sympy_parser.parse_expr` with an explicit `local_dict` and `global_dict`; no `eval` of arbitrary names).
- Parameter symbols are created `real=True` (Eli's real-inputs rule).

- [ ] **Failing tests first:** for every circuit in `tests/circuits.py` that quantizes without error (harmonium, zero_pi, zero_pi_asymmetric, fluxonium_pair, transmon_fluxonium, kite, rhombus, coupled, coupled_zero_pis, and `random_circuit(seed)` for a few seeds), with GHz parametrization where the circuit has SI symbols: `sym.simplify(load_spec(to_spec(res)).hamiltonian() - subs_constants(res.H)) == 0`, or, where `simplify` is too slow, equality at five random real parameter points to 1e-12 relative (state which in the test). Float and Rational coefficients round-trip (Review focus 4). A loader test per validation error, and a test that a malicious string (`"run(\`ls\`)"`, `"__import__('os')"`) is rejected.
- [ ] Implement; run the focused tests and the full suite.
- [ ] Commit.

### Task 4: Python documentation

**Files:** `~/src/circuit-quantization/README.md` (a short "Export to KronFreeCircuits" section with the fluxonium example and the GHz-parametrization advice); module docstrings.

- [ ] Write it; one sentence per line; commit.

### Task 5: Julia `Coefficient` accepts SymPy's dotted operators

**Files:** `~/src/KronFreeCircuits.jl/src/parameters.jl` (`_check_expr`/parsing normalizes `.* ./ .^ .+ .-` call heads and broadcast forms to `* / ^ + -`); `test/test_parameters.jl`.

- [ ] **Failing test first:** `Coefficient("x .^ 2 .* y / 3 + x / 2 + pi * sqrt(x)", [:x, :y])` evaluates equal to the scalar form at a few points; `Coefficient("x .+ y", …)` and `"x ./ y"`, `"x .- y"` likewise; a dotted call of a non-allowlisted function is still rejected. Check how `Meta.parse` represents each dotted form in Julia 1.13 (a `:call` with a dotted symbol or a `:.` broadcast expression) and normalize both.
- [ ] Implement; run the default suite; commit (KronFreeCircuits branch).

### Task 6: Generated fixtures and the provenance test

**Files:** Create `~/src/KronFreeCircuits.jl/test/fixtures/generate_specs.py` and `test/fixtures/generated/*.json`; create `test/test_generated_specs.jl` (picked up by `runtests.jl`'s `test_*.jl` rule).

- [ ] For each hand-written fixture (`transmon`, `two_transmons`, `three_transmons`, `fluxonium`, `cos2phi` where expressible as a circuit — otherwise document why not — `zero_pi`, `full_zero_pi`, `two_zero_pi`), build the same circuit in circuit-quantization with the coordinate choice that matches the hand fixture's modes (explicit `Z_full`/`types` in `quantize`, from the fixture's `_comment` and scqubits' conventions), substitute the parameter names the fixture uses (`subs`), and export into `test/fixtures/generated/<name>.json`. The script records the circuit-quantization commit in `_generator`.
- [ ] `test_generated_specs.jl`: every generated file loads with `parse_spec` (Review focus 5); for each pair, the two specs define the same Hamiltonian — same mode names and types, and `E_C`, `E_L`, `flux_linear`, `n_g`, `constant` equal as functions at three random real parameter points (1e-12 relative), cosines equal as a multiset of `(omega, energy, offset)` up to the `cos(−x) = cos(x)` sign flip of `omega` and `offset` together; intended differences are asserted explicitly with their reason (e.g. scqubits' `2 E_J` constant in the zero-pi fixtures, the `n_g` sign convention noted in `zero_pi.json`'s comment). Then `materialize` both specs on the same small tree (Dense) and require equal energies to 1e-12.
- [ ] Any disagreement that is not a documented convention is a finding: stop and report it to Eli rather than editing a fixture to match.
- [ ] Run the default suite; commit.

### Task 7: Julia documentation

**Files:** KronFreeCircuits `README.md` (the exporter as the source of specs; the generated fixtures), spec §5.2 pointer to the format document (Task 1), a dated execution-notes line in the spec.

- [ ] Write; commit; hand Eli the merge commands for both branches.

## Verification (both repositories)

- circuit-quantization: full `pytest` passes, including the round trip over every test circuit.
- KronFreeCircuits: the default suite passes; the gated suite passes if any gated file changed; the device suite is unaffected (no `src/` change outside `parameters.jl`, which the device suite exercises only through specs, so run it once if time allows).
- A final whole-branch review on the most capable model covers both branches together, with adversarial probes of the round trip (circuits with voltage sources, several loops with flux, a basis giving a refused `omega`, an expression that tries to escape the parser).
