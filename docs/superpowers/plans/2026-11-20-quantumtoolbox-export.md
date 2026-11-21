# QuantumToolbox Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a materialized `System` into QuantumToolbox objects (static and driven Hamiltonians, operators, initial
states) in the dressed basis or the matrix-free product basis, in the lab frame or the interaction picture, on CPU or
GPU, with several initial states evolved at once as a partial propagator.

**Architecture:** Core gains `port_operator` (a port as a `KronFreeOperator` on a node's product basis), the `Frame`
type with `to_lab`, and exported stubs. A package extension `KronFreeCircuitsQuantumToolboxExt`, triggered by
`QuantumToolbox` alone, defines the methods: dressed objects are ordinary `Qobj`/`QobjEvo`; matrix-free ones are
SciMLOperators `FunctionOperator`s (reached as `QuantumToolbox.SciMLOperators`) inside a `QobjEvo`.

**Tech Stack:** Julia 1.13, QuantumToolbox 0.52, SciMLOperators 1.30 (via QuantumToolbox), CUDA/cuTENSOR through the
existing CUDA extension.

**Spec:** `docs/superpowers/specs/2026-11-20-quantumtoolbox-export-design.md`

## Global Constraints

- Never run `Pkg.rm`, `Pkg.gc`, `Pkg.update`, `Pkg.add` or `Pkg.develop`. Add dependencies by editing `Project.toml`,
  then `Pkg.resolve()` / `Pkg.instantiate()` in that environment. Set `JULIA_PKG_GC_AUTO=false` for every Julia command.
- Julia is `$HOME/.juliaup/bin/julia`. Work only in the worktree `~/src/KronFreeCircuits-qt` (branch
  `quantumtoolbox-ext`). One Julia process at a time.
- Fast tier per task: `julia --project -e 'using Pkg; Pkg.test()'` (~90 s). Never run `KRONFREE_SLOW_TESTS=1` before
  Task 5's last step.
- QuantumToolbox tier: `julia --project=test/quantumtoolbox test/quantumtoolbox/runtests.jl`.
- Scratch scripts: `julia --project` from the repo root plus `include("test/helpers.jl")`. Never `Pkg.develop` to load
  the package.
- QuantumToolbox must stay a weak dependency; the core must load and pass its tests without it.
- Units: time in ns; Hamiltonians in rad/ns, `H/ħ = 2π [H0 − E₀ + Σ_k f_k(t) O_k]`, `f_k` in GHz, `E₀` the node's
  lowest kept energy. `qt_operator` returns the plain operator (no 2π), for analysis and `e_ops`.
- Interaction picture frame: the diagonal of H0 in the evolved basis. Dressed: exact, Ω = 2π(Eᵢ − E₀).
  Matrix-free: Ω = 2π(real(diagonal(H0)) − E₀); the remainder `2π(H0 − E₀) − diag Ω` stays and is rotated.
- The dressed basis lives on the host; `on=CuArray` applies to the matrix-free basis only.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A complex node (two transmons with `n_g1 = 0.25`): H0 and ports are complex; the real-block apply must not be used.
  Pinned in Tasks 2–4 (every test runs on a real and a complex system).
- A real node driven through a complex-character port (`expiphi(1)` on `:varphi_1`): `port_operator` and the
  matrix-free apply must come out complex. Pinned in Task 1 and Task 3.
- An empty drive list: the lab frame is H0 alone; the dressed interaction picture is the zero Hamiltonian, and the
  state must stay put. Pinned in Task 4.
- Bad requests: `n` larger than the kept count, a label with no dressed state, `basis`/`frame` not in the allowed set,
  `on=CuArray` with the dressed basis. Each must be an `ArgumentError` with a message naming the bad value. Pinned in
  Task 2 and Task 3.
- A drive on a port the node (dressed) or its child (matrix-free) did not form: an `ArgumentError` listing the
  available ports (the one `port` / `_childop` already throw), not a `KeyError`. Pinned in Task 1 and Task 3.

---

## File Structure

- `src/portoperator.jl` (create): `port_operator`.
- `src/quantumtoolbox.jl` (create): `Frame`, `to_lab` on arrays, and the stubs with docstrings for `qt_hamiltonian`,
  `qt_operator`, `qt_ket`, `to_dressed`, `drive_hamiltonian` (pattern: `src/pyinterface.jl`, including its hint when the
  extension is not loaded).
- `ext/KronFreeCircuitsQuantumToolboxExt.jl` (create): every method that needs QuantumToolbox.
- `src/KronFreeCircuits.jl` (modify): include both files after `dressed.jl`; export the new names.
- `Project.toml` (modify): `[weakdeps] QuantumToolbox = "6c2fb7c5-b903-41d2-bc5e-5a7c320b9fab"`,
  `[extensions] KronFreeCircuitsQuantumToolboxExt = "QuantumToolbox"`, `[compat] QuantumToolbox = "0.52"`.
- `test/test_port_operator.jl` (create): fast tier.
- `test/quantumtoolbox/{Project.toml,runtests.jl,test_quantumtoolbox.jl}` (create): deps `KronFreeCircuits`
  (`[sources] path = "../.."`), `QuantumToolbox`, `CUDA`, `cuTENSOR`, `LinearAlgebra`, `Test`, UUIDs as in
  `test/cuda/Project.toml` plus QuantumToolbox's above.
- `README.md` (modify): a section after the CUDA paragraph (line ~102).

Shared test systems (define once at the top of `test/quantumtoolbox/test_quantumtoolbox.jl`, reuse everywhere):

```julia
spec = load_spec(fixture("two_transmons.json"))
leaf(i) = Node([Symbol("varphi_$i") => ChargeBasis(15)]; keep=lowest(6),
               ports=[(Symbol("varphi_$i"), nport()), (Symbol("varphi_$i"), expiphi(1))])
top() = Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(36), ports=[(:varphi_1, nport())])
sys_c = materialize(top(), spec, two_transmons_params(g=0.3))                       # complex (n_g1 = 0.25)
sys_r = materialize(top(), spec, merge(two_transmons_params(g=0.3), (n_g1=0.0,)))  # real
```

All 36 product states are kept at the top, so the dressed basis with `n = 36` spans the same space as the matrix-free
basis and the two must agree to solver tolerance. Drive for every evolution test: `key = (:varphi_1, nport())`,
`f(p, t) = 0.05 * cos(ωd * t)` with `ωd = 2π * (E[2] − E[1])` of that system, `tlist = range(0, 20; length=21)`,
`sesolve(...; reltol=1e-10, abstol=1e-12, progress_bar=Val(false))`.

---

### Task 1: `port_operator` in core

**Files:**
- Create: `src/portoperator.jl`
- Modify: `src/KronFreeCircuits.jl` (include after `dressed.jl`, export `port_operator`)
- Test: `test/test_port_operator.jl`

**Interfaces:**
- Produces: `port_operator(sys::System, key, node::Node=sys.root.node; on=Array) -> KronFreeOperator`, `key` a
  `Tuple{Symbol,Port}` or an `AbstractVector{Tuple{Symbol,Port}}` (joint, factors on distinct modes). Block capacity 1
  (the apply grows its buffers).

- [ ] **Step 1: Write the failing tests** (`@testset "port_operator"`, systems as in File Structure but built in this
  file, since the fast tier has no QuantumToolbox):
  - `Matrix(port_operator(sys_r, (:varphi_1, nport())) * I(36)) ≈ kron(I(6), port(sys_r, :varphi_1, nport(), c1))`
    with `c1 = top.children[1]` (axis 1 fastest); same for `:varphi_2` with `kron(P2, I(6))`.
  - Joint key `[(:varphi_1, nport()), (:varphi_2, nport())]` ≈ `kron(P2, P1)`.
  - `eltype(port_operator(sys_r, (:varphi_1, nport()))) == Float64`;
    `eltype(port_operator(sys_r, (:varphi_1, expiphi(1)))) == ComplexF64` and it equals its kron reference;
    on `sys_c` the `nport` operator equals its reference whatever its eltype.
  - `@test_throws ArgumentError port_operator(sys_r, (:varphi_1, n2port()))` (not formed on the child), and the message
    contains `"lacks port"`.
  - `@test_throws ArgumentError port_operator(sys_r, (:nosuch, nport()))`.

- [ ] **Step 2: Run** `julia --project -e 'using Pkg; Pkg.test(test_args=["test_port_operator.jl"])'`.
  Expected: FAIL, `port_operator` not defined.

- [ ] **Step 3: Implement** in `src/portoperator.jl`: normalize the key with `_sortkey!`, check every mode is in
  `modes(node)` (`ArgumentError` naming the mode otherwise), reject an augmented node as `operator` does, then
  `KronFreeOperator(mn.dims, [AxisTerm(1.0, [(a, op) for (a, op, _) in _axis_ops(node, mn.children, K)])]; on)`.
  The element type follows the operators (a complex `DenseOp` makes it complex).

- [ ] **Step 4: Run** the same command. Expected: PASS. Then the whole fast tier: PASS.

- [ ] **Step 5: Commit** `port_operator: a port or joint port on a node's product basis`.

---

### Task 2: Extension skeleton, static objects and states

**Files:**
- Create: `src/quantumtoolbox.jl`, `ext/KronFreeCircuitsQuantumToolboxExt.jl`, `test/quantumtoolbox/*`
- Modify: `Project.toml`, `src/KronFreeCircuits.jl` (include, export `Frame, to_lab, qt_hamiltonian, qt_operator,
  qt_ket, to_dressed, drive_hamiltonian`)

**Interfaces:**
- Consumes: `port_operator` (Task 1), `operator`, `energies`, `eigenvectors`, `port`, `dressed_index`.
- Produces (all `node::Node=sys.root.node`, `basis ∈ (:dressed, :free)`, `n::Union{Nothing,Int}=nothing` meaning all
  kept, `on=Array`):
  - `qt_hamiltonian(sys, node; basis=:dressed, n, on, batch=false)`: dressed → `Qobj(2π Diagonal(E[1:n] .- E₀))`;
    free → `QobjEvo` of a constant `FunctionOperator` applying `2π(H0 − E₀)`.
  - `qt_operator(sys, key, node; basis=:dressed, n, on, batch=false)`: dressed → `Qobj(port(sys, key, node)[1:n, 1:n])`;
    free → `QobjEvo` of a constant `FunctionOperator` of `port_operator`. No 2π.
  - `qt_ket(sys, label_or_labels, node; basis=:dressed, n, on)`: a `Tuple` label → `Ket` (dressed: index from
    `dressed_index`; free: `LinearIndices(Tuple(dims))[label...]`); a vector of labels → `Operator` of size N×m with
    `dims=((N,), (m,))`, one unit column per label. Free kets live on `on`.
  - `to_dressed(sys, ψ, node; n)`: `Vn' * ψ` with `Vn = eigenvectors(sys, node)[:, 1:n]`, ψ a `Ket` or N×m `Operator`
    (moved to the host), returned with the same type and dims `n` / `((n,), (m,))`.
  - Internal (ext): `_apply_op(H::KronFreeOperator; scale, shift, on, batch) -> FunctionOperator`. Kwargs to
    `FunctionOperator`: `u=proto, p=NullParameters(), T=ComplexF64, islinear=true, has_mul5=true, ishermitian`
    as appropriate, `batch`; `proto = adapt(on, zeros(ComplexF64, N, 1))` when `batch` else `zeros(ComplexF64, N)`.
    `u=` is required: without it SciMLOperators fails in `update_coefficients!` ("cannot convert a value to nothing").
    For a real `H` the apply packs `[Re X | Im X]` into one preallocated real `N × 2m` block (reallocated when `m`
    changes) and applies once, as `RealBlockApply` in `~/src/kronfree-propagate-probe/propagate.jl`; for a complex
    `H` it applies `mul!` directly.

- [ ] **Step 1: Set up the environment.** Edit `Project.toml` (File Structure); create
  `test/quantumtoolbox/Project.toml`; run `julia --project=test/quantumtoolbox -e 'using Pkg; Pkg.resolve();
  Pkg.instantiate()'`. Expected: resolves with QuantumToolbox 0.52.x from the depot. `runtests.jl`: `using Test,
  LinearAlgebra, CUDA, cuTENSOR, QuantumToolbox, KronFreeCircuits`, `include("../helpers.jl")`,
  `include("test_quantumtoolbox.jl")`.

- [ ] **Step 2: Write the failing tests** (`@testset "static"`, looped over `sys in (sys_r, sys_c)`):
  - `qt_hamiltonian(sys).data ≈ 2π * Diagonal(E .- E[1])`; with `n=10` its size is `(10, 10)`.
  - Free: for a random complex `x` (vector, and a 36×3 block with `batch=true`), the `FunctionOperator` applied to `x`
    ≈ `2π * (Matrix(operator(sys) * I(36)) - E[1] * I) * x`. Reach the operator as `qt_hamiltonian(...).data`.
  - `qt_operator(sys, (:varphi_1, nport())).data ≈ port(sys, :varphi_1, nport())`.
  - `to_dressed(sys, qt_ket(sys, (1, 1); basis=:free)).data ≈ conj(eigenvectors(sys)[LinearIndices((6, 6))[1, 1], :])`.
  - `qt_ket(sys, [(1, 1), (2, 1)])` is an `Operator` of size `(36, 2)` whose columns are `qt_ket(sys, (1, 1))` and
    `qt_ket(sys, (2, 1))`.
  - `ArgumentError`s, each with a message containing the bad value: `n=37`; `basis=:bogus`; `qt_hamiltonian(sys;
    on=CuArray)` with the dressed basis (only when `CUDA.functional()`); `qt_ket(sys, (7, 1))`.
  - Without the extension: in `test/test_port_operator.jl` (fast tier, no QuantumToolbox loaded),
    `@test_throws ErrorException qt_hamiltonian(sys_r)` with the hint "load QuantumToolbox" in the message.

- [ ] **Step 3: Run** the QuantumToolbox tier. Expected: FAIL, methods missing.

- [ ] **Step 4: Implement** the stubs (`src/quantumtoolbox.jl`, hint as in `src/pyinterface.jl`) and the extension
  methods. Validation lives in one internal `_check_request(sys, node, basis, n, on)` in the extension.

- [ ] **Step 5: Run** the QuantumToolbox tier and the fast tier. Expected: PASS both.

- [ ] **Step 6: Commit** `QuantumToolbox extension: Hamiltonians, operators, kets in the dressed and matrix-free bases`.

---

### Task 3: Lab-frame drives and partial propagators

**Files:**
- Modify: `ext/KronFreeCircuitsQuantumToolboxExt.jl`, `src/quantumtoolbox.jl` (docstring)
- Test: `test/quantumtoolbox/test_quantumtoolbox.jl`

**Interfaces:**
- Consumes: Task 2's builders; `Frame` (Task 4 fills it in; here return `Frame(nothing)` for the lab).
- Produces: `drive_hamiltonian(sys, drives, node=root; basis=:dressed, frame=:lab, n=nothing, on=Array, batch=false)
  -> (H::QobjEvo, frame::Frame)`, `drives::AbstractVector{<:Pair}` of `key => f`, `f(p, t)` in GHz.
  Dressed lab: `QobjEvo((qt_hamiltonian(...), (Qobj(2π * Pk[1:n, 1:n]), f_k)...))`. Free lab:
  `QobjEvo(F0 + Σ_k ScalarOperator(0.0; update_func=(a, u, p, t) -> f_k(p, t)) * Fk)` with `F0` scaled by 2π and
  shifted by E₀, each `Fk` the `port_operator` scaled by 2π, all built with the same `batch`.

- [ ] **Step 1: Write the failing tests** (`@testset "lab drives"`, over `sys_r`, `sys_c`; drive as in File Structure):
  - Dressed vs free: `ψd = sesolve(Hd, qt_ket(sys, (2, 1)), tlist).states[end]` with `Hd` dressed, `n=36`;
    `ψf` free from `qt_ket(sys, (2, 1); basis=:free)`; `to_dressed(sys, ψf).data ≈ ψd.data` within `atol=1e-7`.
  - Partial propagator: free `batch=true`, initial `qt_ket(sys, [(1, 1), (2, 1), (1, 2)]; basis=:free)`; each column of
    the final block ≈ the single-ket free solve of that label (`atol=1e-7`). Same for the dressed basis (its `Qobj`s
    take either).
  - Complex port on a real node: `drives = [(:varphi_1, expiphi(1)) => f]`, free vs dressed on `sys_r` agree
    (`atol=1e-7`). This needs `ports` on `top()` to include `(:varphi_1, expiphi(1))` for the dressed side: add it.
  - `@test_throws ArgumentError drive_hamiltonian(sys, [(:varphi_2, n2port()) => f])` for each basis; `frame=:bogus`
    is an `ArgumentError`.

- [ ] **Step 2: Run** the QuantumToolbox tier. Expected: FAIL.

- [ ] **Step 3: Implement** `drive_hamiltonian` for `frame=:lab`.

- [ ] **Step 4: Run** the QuantumToolbox tier. Expected: PASS.

- [ ] **Step 5: Commit** `drive_hamiltonian: lab frame, both bases, partial propagators`.

---

### Task 4: Interaction picture

**Files:**
- Modify: `src/quantumtoolbox.jl` (`Frame`, `to_lab`), `ext/KronFreeCircuitsQuantumToolboxExt.jl`
- Test: `test/quantumtoolbox/test_quantumtoolbox.jl`

**Interfaces:**
- Produces: `struct Frame; ω::Union{Nothing,Vector{Float64}}; end` (rad/ns, host). `to_lab(frame, ψ, t)`:
  `cis.(-ω .* t) .* ψ` row-wise for an array (vector or N×m, host or device; `ω` adapted to ψ's storage), identity when
  `ω === nothing`; the extension adds the method for a `QuantumObject` (same type and dims back).
  `drive_hamiltonian(...; frame=:interaction)` returns `Frame(Ω)`.
  - Dressed: `Ω = 2π(E[1:n] .- E₀)`; one time-dependent `FunctionOperator` (`isconstant=false`) applying
    `Σ_k f_k(p, t) · conj(ph) .* (Pk * (ph .* v))`, `ph = cis.(-Ω t)`, `Pk` the 2π-scaled dense ports.
  - Free: `Ω = 2π(real(diagonal(H0)) .- E₀)`; one time-dependent `FunctionOperator` applying
    `conj(ph) .* (2π(H0 − E₀)(ph .* v) − Ω .* ph .* v + Σ_k f_k(p, t) · 2π Ok (ph .* v))`, the phase vector formed once
    per call, the real-block packing of Task 2 reused for each real operator.

- [ ] **Step 1: Write the failing tests** (`@testset "interaction picture"`, over `sys_r`, `sys_c`, both bases, Ket and
  a 3-column `batch=true` block):
  - `to_lab(frame, ψI_final, tlist[end]).data ≈ ψlab_final.data`, `atol=1e-7`, where `ψlab_final` comes from the Task 3
    lab solve of the same initial state.
  - Empty drives, dressed interaction picture: the final state equals the initial state (`atol=1e-12`).
  - `to_lab(Frame(nothing), ψ, 3.0) === ψ`, and on a plain array `to_lab(Frame([0.0, 1.0]), [1.0, 1.0], π)` ≈
    `[1, -1]`.

- [ ] **Step 2: Run** the QuantumToolbox tier. Expected: FAIL.

- [ ] **Step 3: Implement** as in Interfaces.

- [ ] **Step 4: Run** the QuantumToolbox tier and the fast tier. Expected: PASS.

- [ ] **Step 5: Commit** `drive_hamiltonian: interaction picture in the frame of H0's diagonal`.

---

### Task 5: GPU, docs, final runs

**Files:**
- Modify: `test/quantumtoolbox/test_quantumtoolbox.jl`, `README.md`, any extension fixes the GPU test exposes

- [ ] **Step 1: Write the GPU tests** (`@testset "GPU"`, `if CUDA.functional()`, else `@test_skip`): free basis,
  `on=CuArray`, for `sys_r` and `sys_c`, lab and interaction, Ket and `batch=true` block: the final state (brought to
  the host with `Array`) ≈ the CPU result, `atol=1e-7`; `to_lab` on the device state matches the host one.

- [ ] **Step 2: Run** the QuantumToolbox tier on the GPU machine. Fix what fails (expected suspects: a host `ω` or
  `E₀` broadcast against a `CuArray`, a host prototype). Expected: PASS.

- [ ] **Step 3: README.** A section "Time evolution with QuantumToolbox" after the CUDA paragraph: loading
  (`using QuantumToolbox` beside the package), the two bases, the units line from Global Constraints, the frame rule,
  and one example of a driven 3-state partial propagator in the free basis with `to_lab` and `to_dressed`. The example
  must run: paste it into a scratch script against `sys_r` and run it.

- [ ] **Step 4: Final runs.** Fast tier; QuantumToolbox tier; then once,
  `KRONFREE_SLOW_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'` (~10 min, 20 known `fix_phases!` warnings).
  Expected: all PASS. Report the counts.

- [ ] **Step 5: Commit** `QuantumToolbox extension: GPU tests and README`.
