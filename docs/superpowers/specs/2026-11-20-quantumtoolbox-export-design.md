# QuantumToolbox export: design

## Intent

Hand a materialized `System` to QuantumToolbox.jl so time evolution (`sesolve`) runs on KronFreeCircuits' operators,
in either of two bases:

- **dressed**: the `n` lowest kept eigenstates of a node, dense matrices (the standard workflow, e.g. the zero-π CZ
  study with n = 1000);
- **matrix-free**: a node's product basis of its children's kept states, with the Hamiltonian applied by the
  `KronFreeOperator` on CPU or GPU, never formed.

The drive Hamiltonian can be built in the lab frame or in the interaction picture. This is the feature the code paper's
dense-vs-matrix-free timing comparison rests on: the probe in `~/src/kronfree-propagate-probe` showed matrix-free
23 870 states at 918 s (CPU) / 490 s (GPU) lab frame and 156 s (GPU) interaction picture, against dense crossovers at
roughly 1450 / 1140 retained dressed states.

## Goals

- Static Hamiltonian and port operators of any node as QuantumToolbox objects, in either basis.
- A drive builder: `H(t) = H0 + Σ_k f_k(t) O_k`, where `O_k` is a port or joint-port key, returned as a `QobjEvo` in
  the lab frame or the interaction picture.
- Helpers for initial states and for reading results: a ket from a bare label, rotating an interaction-picture state
  back to the lab frame, and projecting a matrix-free state onto the dressed states.
- GPU for the matrix-free basis, through the existing `_on_storage` / CUDA extension.

## Non-goals

- `mesolve` / collapse operators in the matrix-free basis (it needs superoperators on N² entries). The dressed
  `Qobj`s work with `mesolve` as they stand.
- Stacking several initial states into one solve. The probe got 1.5× from it, but QuantumToolbox's `sesolve` takes one
  state; this can come later as a standalone propagator.
- Pulse shapes, RWA, gate-fidelity routines. The user supplies `f_k(t)`.

## Anti-goals

- QuantumToolbox becoming a hard dependency: it is heavy (OrdinaryDiffEq, SciML). It must be a weak dependency.
- A matrix-free path that silently densifies or hand-rolls the operator instead of using the `KronFreeOperator`.

## Approaches considered

1. **Package extension on `QuantumToolbox` (chosen).** Core gets a QuantumToolbox-free `port_operator`. The extension
   defines the conversion functions, whose stubs and docstrings live in core (as the CUDA and PythonCall extensions do).
   The matrix-free operators are SciMLOperators `FunctionOperator`s inside `QobjEvo`.
2. A separate package (KronFreeQuantumToolbox.jl): cleaner dependency story, but a second package to version and cite
   for one file of glue.
3. Export dense matrices only and let users wrap them: no matrix-free path, which is the point.

## Decisions (agreed)

- **Interaction picture frame**: the diagonal of H0 *in the basis being evolved*.
  - Dressed: H0 is diagonal there, so the frame is exact and H0 drops out: `H_I(t) = Σ_k f_k(t) e^{iΩt} O_k e^{-iΩt}`.
  - Matrix-free: Ω = `diagonal(H0)` (fused diagonal plus the diagonals of the terms); the off-diagonal part of H0 stays
    and is rotated with the drives:
    `H_I(t) = e^{iΩt} (H0 − diag Ω + Σ_k f_k(t) O_k) e^{-iΩt}`.
    This avoids needing the higher node's eigenstates. The phase vector `e^{-iΩt}` is formed once per right-hand side.
- QuantumToolbox's `sesolve` drives both bases; no custom ODE loop.

## API (builder's call where marked)

Core, exported (no QuantumToolbox needed):

- `port_operator(sys, key, node=root; on=Array) -> KronFreeOperator`: the port `(mode, p)` or joint key
  `[(mode, p), ...]` on `node`'s product basis (identity on the other axes), built from the children's operators with
  the existing `_axis_ops`. Real exactly when the node is real and the key's character is real.

Extension `KronFreeCircuitsQuantumToolboxExt` (weakdep `QuantumToolbox`; SciMLOperators reached through it if possible,
else a second weakdep), stubs exported from core:

- `qt_hamiltonian(sys, node=root; basis=:dressed, n=nothing, on=Array)`: H0 as a `Qobj` (dressed, `n` lowest, default all
  kept) or a `QobjEvo` around a constant `FunctionOperator` (matrix-free).
- `qt_operator(sys, key, node=root; basis, n, on)`: a port / joint port, same two forms.
- `drive_hamiltonian(sys, drives, node=root; basis, frame=:lab | :interaction, n, on) -> (H::QobjEvo, frame::Frame)`,
  with `drives` a vector of `key => f`, `f(p, t)` the QuantumToolbox coefficient signature.
- `qt_ket(sys, label, node=root; basis, n)`: a ket of the dressed state with that bare label (`dressed_index`) or of the
  product state itself.
- `to_lab(frame, ψ, t)`: rotate an interaction-picture state back (identity for `:lab`).
- `to_dressed(sys, ψ, node=root; n)`: dressed-basis amplitudes `V† ψ` of a matrix-free state.

Names are provisional (builder's call; `qt_` prefix to avoid colliding with QuantumToolbox exports).

**Units (builder's call)**: time in ns, operators in rad/ns: `H/ħ = 2π [H0 − E₀ + Σ_k f_k(t) O_k]`, `f_k` in GHz
(the package's units), E₀ the node's lowest energy, so the ground state has zero phase in both bases.

**Device**: `on=CuArray` puts the matrix-free operator and the `QobjEvo`'s prototype vector on the GPU (needs CUDA and
cuTENSOR loaded, as for the solvers); the dressed basis stays on the host.

## Tests (builder's call)

A separate test environment `test/quantumtoolbox/` (like `test/cuda`, `test/python`), so the fast tier does not load
OrdinaryDiffEq. A small two-mode system built from an existing spec fixture:

- `port_operator` equals the dense Kronecker product; real/complex type as stated.
- Dressed vs matrix-free `sesolve` of the same drive agree in dressed populations to solver tolerance.
- Interaction picture vs lab frame agree after `to_lab`, in both bases.
- GPU matches CPU when `CUDA.functional()`; skipped otherwise.

`port_operator` also gets a fast-tier test in the main suite.

## Constraints

- No `Pkg.rm`, `Pkg.gc`, `Pkg.update`, `Pkg.develop`; dependencies are edited in `Project.toml` and resolved.
- Work in the `quantumtoolbox-ext` worktree, not the main checkout.
- QuantumToolbox 0.52 / SciMLOperators 1.30 are what the probe used; compat should admit them.
