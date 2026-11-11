# KronFreeCircuits.jl: architecture design

Date: 2026-10-15.
Status: approved by Eli Weissler on 2026-10-15.
Author of record: Eli Weissler, with Claude as brainstorming partner.

This document describes a Julia package for the numerical modeling of superconducting circuits by recursive hierarchical diagonalization, without ever forming a Kronecker-product Hamiltonian.
A builder should be able to plan the implementation from this document alone.
Decisions marked **(decided)** were made in conversation and are not open.
Decisions marked **(my call)** were made by Claude as reasonable defaults and may be overridden.
Anything in "Left to the builder" is the builder's to settle.

## 1. Intent

Eli's thesis work on zero-pi and fluxonium qubits is limited by slow numerics in two places.
Optimizing a two-qubit gate between two zero-pi qubits takes about a day per iteration.
Optimizing a circuit design by enumeration takes about a month.
Both bottlenecks are the same operation: diagonalize a composite of a few strongly coupled, high-dimensional modes, keep a chosen set of states per subsystem, and compose the subsystems.
Existing tools (scqubits, SQcircuit, QuTiP, QuantumToolbox.jl) build the composite Hamiltonian as a sparse Kronecker product.
For dense local operators this is infeasible: the two-zero-pi composite at 300 kept states per qubit needs roughly 8e9 nonzeros, about 130 GB, while the equivalent kron-free product costs 3 ms per application (Section 17).

The package applies Roth's column lemma (the "vec trick", thesis Chapter 2) recursively: every operator is a sum of products of single-axis operators, applied to a state held as a dense tensor by acting along one axis at a time.
It runs on CPU and GPU from the same code, and every output is differentiable with respect to circuit parameters and external fluxes.

Symbolic quantization is not part of this package.
Eli's Python package `circuit-quantization` (`~/src/circuit-quantization`) produces the Hamiltonian in the irrotational gauge; this package consumes a structured export of it (Section 5).

## 2. Goals, non-goals, anti-goals

### Goals

- Ingest a circuit Hamiltonian from `circuit-quantization` and build a numerical model with no hand-written operators.
- Recursive hierarchical diagonalization on an arbitrary tree, where a leaf may contain several coupled modes, and each node keeps an arbitrary *set* of eigenstates, not only the lowest k.
- Never form the Kronecker product at any level; every level is matrix-free.
- Same code on CPU and NVIDIA GPU; the device is selected by the array type of the data.
- Gradients of energies and matrix elements with respect to every circuit parameter and every external flux, through the whole tree.
- Export of the final (or any intermediate) node to QuantumToolbox.jl or QuantumOptics.jl objects for downstream work, including dynamics.
- Extensible in two directions without touching the core: new single-mode bases, and new local-operator structure types.
- Friendly to a later Born-Oppenheimer workflow (Section 13.1): diagonalize a node over a subset of its modes while treating the others' coordinates as parameters, producing scalar energy surfaces.
- Acceptance goal: reproduce the two-zero-pi results of `~/src/two_qubit_gate_for_zero_pi_qubit` (spectra, selected subspaces, and, after export to QuantumToolbox.jl, the gate dynamics) from this package. Dynamics move to Julia; nothing downstream stays in Python.
- Stretch goal: an NPZ export of energies and ports with the documented phase convention (Section 12), because it is cheap and Python users will want it.

### Non-goals (for the first version)

- Time dynamics inside this package. The matrix-free node supports in-place multiplication, so wrapping a node as a solver operator is a thin later layer. Dynamics are done after export.
- Gradients with respect to pulse parameters.
- Tensor-network (MPS, DMRG, TDVP) methods. The apply-layer interface leaves room for an ITensor backend later, but nothing is built for it.
- A periodic phase-grid basis for compact modes. The charge basis is sufficient for now.
- Time-dependent offset charge from a time-dependent flux. The spec is static.
- Forward-mode differentiation. Reverse mode first.
  Amended 2026-11-11 (Section 20l): derivatives are computed upward (forward) by perturbation theory through the tree, to second order; a reverse (adjoint) sweep is now the non-goal.
- Float32 or mixed precision. Everything is Float64 / ComplexF64.

### Anti-goals (would make this a failure even if it worked)

- A "mishmash" of device-specific backends. There must be one apply interface whose implementations are chosen by operator *structure*, never by device.
- Any code path that silently falls back to generic indexing of a lazy matrix (the failure mode of the `mat_free_hacking` AbstractMatrix experiment).
- A runtime dependency on Python *in the core*, or on ITensor.
  Amended 2026-11-10 (Section 20j): the anti-goal was "a runtime dependency on Python"; Python is now reachable through an optional PythonCall extension, and `using KronFreeCircuits` without PythonCall still loads, passes the default tests and never imports Python.
- Parameters hidden inside basis matrices, which would make every local operator parameter-dependent and break the autodiff design.

## 3. Constraints

- Julia 1.13 (current stable on both machines). Package follows standard layout with `Project.toml`, extensions for optional dependencies, and tests.
- Development on a Mac (M2, no FP64 GPU). CUDA development on `ewdesktop` (RTX A4500, 20 GB, FP64 rate about 0.37 TFLOPS). Final timings on data-center NVIDIA GPUs with full FP64.
- Repository is private at first. License undecided.
- Units follow `circuit-quantization`: energies and the Hamiltonian are $E/h$ in GHz; phases dimensionless; offset charge in units of $2e$.
- The thesis chapter "Matrix Free Methods for Simulating Circuit QED on GPUs" will describe this package. Its empty "Julia Implementation" section should be writable from the final architecture, so the architecture should match the chapter's concepts (vec trick, mode-k product, stencils, hierarchical diagonalization).

## 4. The core abstraction (decided)

Everything is a **subsystem**: an object with a finite basis of dimension $d$ that can hand out, on request, a **port** for each named operator it supports, as a local operator of some structure type (Section 7).
Port names are `n`, `n2`, `phi`, `phi2`, `expiphi(k)`, `exp2pin(k)` for every mode a subsystem contains, where `k` is a numeric multiplier.
Two kinds of subsystem exist:

- A **mode** is a primitive subsystem: one circuit coordinate in one basis (Section 6). Its ports are analytic and depend only on basis hyperparameters, never on circuit parameters.
- A **node** holds child subsystems and a Hamiltonian over their ports (Section 7). A node that holds only modes is a **leaf**; a node that holds nodes is a **branch**. The code does not distinguish them; "leaf" and "branch" are words for people.

A node is **materialized** (Section 10) by diagonalizing its Hamiltonian matrix-free, selecting a set of eigenstates, and re-expressing every port needed above it as a dense matrix in the selected eigenbasis.
After materialization the node is itself a subsystem of dimension $k$ (the number of kept states) whose ports are dense.
This is the whole recursion: a branch's children are subsystems with dense ports, exactly as a leaf's children are modes with structured ports.

Terms route themselves (decided): a product term of the circuit Hamiltonian belongs to the lowest node that contains every mode the term touches.
Ports are inferred (decided): a node materializes a port for every operator that appears in a term routed above it, plus any port named by a selection rule or requested by the user.

## 5. Ingest: the spec (decided)

### 5.1 Source

`circuit-quantization` gets one new function, `export_julia_spec(result, path)`, that writes a JSON file.
As built (plan 4, Section 20g), the exporter is `circuit-quantization`'s `to_spec` and `save_spec` (`circuit_quantization.io`), which supersede `export_julia_spec`.
This is in scope for this project even though the Python package otherwise stays as it is.
Coefficients are written as Julia source strings produced by SymPy's Julia code printer, so they remain symbolic in the circuit parameters.
The function must refuse to export if any cosine coefficient vector (Section 5.2, `omega`) or any charge-cosine vector (`kappa`) contains a free symbol, because parameter-dependent vectors would put parameters into local operators (Section 11).

### 5.2 Content

The shared format document is `circuit-quantization`'s `docs/spec-format.md` (kronfree-spec version 1).

The Hamiltonian produced by `circuit-quantization` always has the form

$$
H = 4\,(\mathbf n - \mathbf n_g)^{\mathsf T} E_C\,(\mathbf n - \mathbf n_g)
  + \tfrac12\,\boldsymbol\varphi^{\mathsf T} E_L\,\boldsymbol\varphi
  + \boldsymbol\varphi^{\mathsf T}\mathbf b(\varphi_{\rm ext})
  - \sum_k E_{J,k}\cos\!\big(\boldsymbol\omega_k^{\mathsf T}\boldsymbol\varphi + \delta_k(\varphi_{\rm ext})\big),
$$

with $\mathbf b$ the inductive flux cross term (linear in $\varphi$).
The spec carries exactly these ingredients plus two optional generalizations, and nothing else:

```json
{
  "format": "kronfree-spec",
  "version": 1,
  "units": "GHz",
  "parameters": ["E_J1", "E_C1", "E_L1", "varphi_ext1", "n_g1"],
  "modes": [
    {"name": "varphi_1", "type": "e"},
    {"name": "varphi_2", "type": "c"}
  ],
  "E_C":  [["E_C1", "0"], ["0", "E_C1/2"]],
  "E_L":  [["E_L1", "0"], ["0", "0"]],
  "flux_linear": ["E_L1*varphi_ext1", "0"],
  "n_g":  ["0", "n_g1"],
  "cosines": [
    {"energy": "E_J1", "omega": [1, 1], "offset": "varphi_ext1/2"},
    {"energy": "E_J1", "omega": [1, -1], "offset": "-varphi_ext1/2"}
  ],
  "charge_cosines": [
    {"energy": "E_S1", "kappa": [1, 0], "offset": "0"}
  ],
  "constant": "0"
}
```

- `parameters`: every free symbol in the file, in a fixed order. The Julia side binds them by name.
- `modes`: name and type, `c` compact, `e` extended, `h` harmonic, in the order used by all matrices and vectors.
- `E_C`, `E_L`: square matrices of expression strings. `flux_linear`: the coefficient of $\varphi_i$ from the inductive flux cross term. `n_g`: offset charge per mode (zero for non-compact modes).
- `cosines`: Josephson terms, $-\text{energy}\cdot\cos(\boldsymbol\omega^{\mathsf T}\boldsymbol\varphi + \text{offset})$. Entries of `omega` are numbers; for compact modes they are integers.
- `charge_cosines` (optional): phase-slip terms, $-\text{energy}\cdot\cos(2\pi\,\boldsymbol\kappa^{\mathsf T}(\mathbf n - \mathbf n_g) + \text{offset})$, with numeric `kappa`. **(my call)** on the sign and the $\mathbf n_g$ inside; the exporter and loader must agree and the builder should check the convention against `circuit-quantization`'s notes when the QPS element is added there. Such a term is only meaningful on a non-compact mode: on a compact mode in the charge basis $e^{i2\pi n}$ is the identity, and the loader should reject it.
- Arbitrary $2\pi$-periodic potentials are expressed as additional `cosines` entries with integer multiples in `omega` and their own energies, so no separate field is needed.
- `constant`: an energy offset, kept so exported spectra match the source Hamiltonian exactly.

### 5.3 Loading

The loader parses each expression string with `Meta.parse` and compiles it into a function of a named parameter tuple using RuntimeGeneratedFunctions.jl **(my call; alternative is Symbolics.jl, rejected as too heavy)**.
Each compiled coefficient records which parameters it depends on (its free symbols), which Section 10.4 uses for caching.
The loader expands the quadratic forms into product terms:
$4E_{C,ii}\,n_i^2$ uses the `n2` port; $8E_{C,ij}\,n_i n_j$ for $i<j$ uses `n` on two axes; the linear terms $-8(E_C\mathbf n_g)_i\,n_i$ use `n`; the constant $4\mathbf n_g^{\mathsf T}E_C\mathbf n_g$ joins `constant`.
Likewise for $E_L$ with `phi2` and `phi`.
Each cosine becomes two Hermitian-conjugate product terms with complex coefficients $-\tfrac12 E\,e^{\pm i\,\text{offset}}$ and factors `expiphi(±omega_j)` on every mode $j$ with nonzero $\omega_j$; the builder may instead represent the pair as a single Hermitian term if that simplifies the adjoint (Section 11).
Charge cosines become the same with `exp2pin(±kappa_j)` and the $\mathbf n_g$ offset folded into the phase.

## 6. Bases (decided unless marked)

A basis is a type implementing: its dimension, the mode type it serves, and a method returning each port as a local operator (Section 7.2).
Adding a basis is adding one such type; nothing above the mode changes.

| Basis | For | `n` | `n2` | `phi`, `phi2`, potentials | `expiphi(k)` | `exp2pin(k)` |
|---|---|---|---|---|---|---|
| `ChargeBasis(ncut)` | compact | diagonal | diagonal | not available | shift by $k$ (integer), truncated at the edge | diagonal phase |
| `PhaseGrid(a, b, N; order)` | extended | banded, antisymmetric first-derivative stencil | banded, second-derivative stencil | diagonal | diagonal | shift by $2\pi k/\Delta\varphi$ points, requires that to be an integer |
| `FockBasis(N; phi_zpf)` | harmonic (and extended if chosen) | banded, two off-diagonals | banded | diagonal / banded | dense displacement matrix | dense |
| `PeriodicGrid(N; order, sectors)` | compact (chosen by the user) | banded cyclic antisymmetric first-derivative stencil (wraps around the circle), imaginary | banded cyclic second-derivative stencil, real | `phi`, `phi2` not available (φ is multivalued on the circle) | diagonal, integer $k$ | not available |

`PeriodicGrid` (added 2026-10-17) is specified in Section 20d; it works in the periodic gauge since 2026-10-30 (the offset charge enters only through term construction).

Defaults per mode type: charge for compact, grid for extended, Fock for harmonic.
Every mode's basis is overridable by the user.

- The grid uses the central finite-difference weights of thesis Table `tab:fd_stencils`, orders 2 through 8, default 8.
- The grid charging cross term $n_i n_j$ between two grid modes is the product of two first-derivative stencils, while $n_i^2$ is the second-derivative stencil **(decided convention)**. These are different discretizations of consistent order; the design accepts this and tests must use the same convention for reference values.
- The grid must provide an **edge-support diagnostic**: for each kept eigenvector, the norm of its components on the outer $w$ grid points on each side, with a warning above a threshold. Default $w$ equals the stencil half-width **(my call)**.
- `FockBasis` takes `phi_zpf` as a fixed hyperparameter, not derived from circuit parameters. Any value defines a valid basis. If it is not given, the default is the oscillator value $(2E_C/E_L)^{1/4}$ from the spec evaluated at the parameters of the *first* `materialize` call, after which it is frozen and logged; later calls with other parameters reuse it **(my call)**.
- The closed-form displacement matrix elements (Laguerre form, thesis Chapter 2) are used for `expiphi` in the Fock basis, not matrix exponentiation of truncated operators.

## 7. Operators

### 7.1 Sum of products (decided)

A node's Hamiltonian is a vector of terms.
A term is a coefficient and a list of (axis, local operator) factors, at most one factor per axis.
The coefficient is a compiled function of the parameter tuple (Section 5.3), evaluated to a `ComplexF64` at materialization time.
Applying the Hamiltonian to a state applies each term in sequence and accumulates: a term is applied as a chain of single-axis applications through two scratch buffers, then added with its coefficient.

### 7.2 Local operator structure types (decided)

A local operator carries its structure as its type.
The first version has four:

- `DenseOp`: a dense `ComplexF64` matrix. Always complex, never real, because BLAS has no mixed real-complex gemm and the generic fallback is tens of times slower. (Amended by §20f: real nodes hold real `DenseOp`s; complex nodes still convert every factor to complex.)
- `DiagonalOp`: a vector.
- `BandedOp`: a set of offset diagonals, each with its own weight vector. Constant-weight stencils are the special case where every weight vector is constant; the Fock $a$ and $a^\dagger$ with weights $\sqrt m$ are the general case. Periodic wrap-around is *not* supported (no periodic grid in this version).
- `ShiftOp`: a shift by an integer number of positions with a scalar factor, truncating at the edges.

A structure type implements exactly four methods: apply along axis $k$ of a tensor (Section 8), its adjoint (as another local operator), its diagonal (a vector, for the preconditioner), and materialization to `DenseOp`.
Nothing above the apply layer dispatches on structure type.
Adding a structure type (a blocked port for symmetry sectors, a low-rank factor) is implementing these four.

### 7.3 Ports of a materialized node

After a node keeps $k$ eigenvectors $V$ ($d\times k$), each port it must expose is $V^\dagger O V$, a `DenseOp`.
It is computed matrix-free: apply $O$ along its axis to each column of $V$ (one block application), then one gemm with $V^\dagger$.

## 8. The apply layer (decided)

One function: apply local operator $A$ along axis $k$ of tensor $X$ into tensor $Y$, where $X$ and $Y$ have one axis per child of the node plus, optionally, one trailing block axis for a block of vectors.
Implementations, chosen by structure type, never by device:

| Structure | Implementation | Why |
|---|---|---|
| `DenseOp` | reshape to (left, $d_k$, right) and call gemm: a single gemm when left or right is 1, a loop of gemm on CPU, one strided-batched cuBLAS gemm on GPU (in the CUDA extension) | Benchmarked: fastest on both devices, Section 17 |
| `DiagonalOp` | broadcast with the vector reshaped to a singleton on every other axis | No kernel needed |
| `BandedOp`, `ShiftOp` | KernelAbstractions kernel, one work-item per output element, reading the few inputs at stride | Memory-bound; gemm would do $d_k$ multiplies where the stencil does a handful |

The device is carried by the array type (`Array` or `CuArray`) of the state and the operator data; `Adapt.jl` moves a node's data between devices.
All three implementations run in order on the task's CUDA stream without explicit synchronization.
The block axis needs no separate code path: gemm absorbs it into its right dimension, the kernels index by stride, and the broadcast singleton covers it.

ITensor is **not** a runtime backend.
It was benchmarked (Section 17) and its GPU speed is exactly "permute, one gemm, permute back", which the batched gemm matches within 1 percent while being 3 to 4 times faster on CPU.
It is a test-only dependency used as an independent reference implementation of apply.
The permute-then-gemm variant is kept in the package as the portable fallback for array types without a batched gemm **(my call)**.

**Workspace contract (2026-10-16 decision).** Keep LOBPCGEigensolver as the planned block eigensolver dependency. The operator adapter implements block `mul!(Y, H, X)` into solver-owned output, with two preallocated state/block scratch arrays reused across all terms. Reserve the maximum operator-input block width before iteration; shrinking active blocks as states lock uses views into that capacity, without replacing backing arrays. Each simultaneous solve owns its workspace. Apply diagonal energies and the Jacobi preconditioner by in-place broadcasts. An optional retained-space ITensor adapter must use output/scratch-backed contractions and accumulation rather than allocating a result for every factor or term. Verify warmed GPU allocations, including permutations and library workspaces, separately from numerical correctness; the `mul!` spelling alone is not evidence of zero allocation. This requirement covers state-sized multiplication buffers, not every small metadata allocation or the eigensolver's other workspace.

**Superseded (2026-10-16).** The "ITensor is not a runtime backend" paragraph above is reversed by the retained-space ITensor audit (`docs/spikes/2026-10-16-itensor-audit/`): dense ports at branch nodes are applied through ITensor contractions (cuTENSOR on GPU), structured leaf operators stay native. The decision and its boundary are recorded in Section 20c.

Fusion of all structured terms of a leaf into one kernel pass is a recognized later optimization, not part of the first version.

## 9. Diagonalization (decided)

Amended by §20e (2026-10-17): the solver set and the default rule below supersede the first version's three-solver table.

One internal interface: given a node's apply and preconditioner, return the lowest $m$ eigenpairs.
Six solvers, all available at every node, chosen per node by setting or by `Auto()`:

| Solver | When | Implementation |
|---|---|---|
| `Auto(; dense_cutoff=2500, on=Array)` | the default of every node; an explicit solver always wins | resolves at materialization by node kind, product dimension and storage: `Dense()` at most `dense_cutoff` product states (a placeholder, to be benchmarked); above it, at a leaf `ShiftInvert()` on `Array` storage and `ChebyshevFilter(on=on)` on device storage, and at a branch `LOBPCG(preconditioner=:ritz, seed=:product, on=on)`; `compute = 0` is valid only where it resolves to `Dense` |
| `Dense()` | product dimension at most `dense_cutoff`, and whenever every eigenpair is wanted (`compute = 0`); warns above 10 000 states | the explicit matrix, `LinearAlgebra.eigen` (LAPACK) on the host; returns the lowest `compute` pairs (all when `compute = 0`) |
| `Lanczos()` | chosen explicitly, for a few states of a large node | KrylovKit `eigsolve` with `Lanczos`, vector-at-a-time, on the host; no warm start, and exactly repeated eigenvalues are not resolved |
| `LOBPCG(; preconditioner=:ritz, seed=:product, guard=4, on)` | a branch above the cutoff; a leaf only with a warm start | LOBPCGEigensolver.jl (JuliaMolSim, extracted from DFTK, GPU-capable, MIT), block, with locking; adaptive per-Ritz Jacobi from the fused diagonal; a cold block seeded from `product_seed`, a warm start from the previous materialization's kept vectors |
| `ShiftInvert()` | a leaf above the cutoff on `Array` storage | KrylovKit Lanczos on $(H - \sigma I)^{-1}$, with $H - \sigma I$ factored by sparse Cholesky from `sparse(H)` and $\sigma$ below the lowest eigenvalue (LU fallback); real arithmetic when the assembled matrix is exactly real; a mandatory true-residual check; optional spectrum slicing across threads and block size 2; an inertia-count completeness certificate, on by default for lowest-m solves (`certify=:auto`), opt-out with `certify=false` |
| `ChebyshevFilter(; on)` | a leaf above the cutoff on device storage (it also runs on the host) | Chebyshev-filtered block subspace iteration started from `product_seed`, with $\lambda_{\max}$ from `gershgorin_bound`, degree capped near 100, a rescaled recurrence, SVQB re-orthogonalization with drop-and-refill, and true-residual locking |

Preconditioner: Jacobi, $(\operatorname{diag}H - \sigma)^{-1}$.
The diagonal of a sum of products is the sum of products of the local diagonals, so it is available at every node from the operator representation.
At a branch it is exactly $E_i^{(1)} + E_j^{(2)} + \dots$, and the couplings are a perturbation on it, which is why LOBPCG converges in tens of iterations there.
Applying it is a broadcast.

Chunked LOBPCG with deflation (decided): when `chunk` is smaller than $m$, solve for the lowest `chunk` states, then the next `chunk` orthogonal to those, and so on.
LOBPCGEigensolver has no constraint argument, so deflation is done on the operator side by adding $\sigma\,P_{\rm found}$ to the operator, one extra block inner product per application.
Memory: LOBPCGEigensolver holds roughly ten blocks of $M$ vectors; at 90 000 dimensions a single 2000-vector block is about 29 GB (fits a data-center card, not the A4500), while chunks of 500 plus the 2000 found vectors are about 10 GB.
The default is a single chunk when it fits in device memory **(my call)**.

Every node carries two counts: `compute`, how many eigenpairs the solver returns, and the selection rule (Section 9.1), which picks the kept subset.
The dense solver computes every eigenpair but returns only the lowest `compute` when `compute > 0` (`compute = 0` means all), so that selection ports are not formed on every vector of a large dense node **(amended 2026-10-16 after implementation; the original text said Dense ignores `compute`)**.

### 9.1 Selection (decided)

The primitive is an index list: a node keeps exactly the eigenstates at these indices of the solver's output.
A selection rule is any function from (energies, eigenvectors in the node's product basis, ports in the eigenbasis, and the node's previous materialization if one exists) to an index list.
Four ship with the package:

- `lowest(k)`: the default.
- `indices(list)`: explicit.
- `connected(core, port, thresh)`: Eli's rule from `two_qubit_gate_for_zero_pi_qubit/truncation_estimate.py`, `trunc_by_thresh`: starting from the core state indices, add every state connected to any already-included state by a matrix element of the named port with modulus above `thresh`, iterating to closure, and return the sorted list.
- `follow(reference)` (decided): adiabatic following. `reference` is a set of eigenvectors in the node's full product basis, by default the kept eigenvectors of the node's previous materialization along a sweep, or an explicit matrix for the first point. The rule forms the overlap matrix $|V_{\rm ref}^\dagger V|$ and assigns each reference state to the current eigenstate of largest overlap, one-to-one (greedy by overlap, with the optimal assignment as an option), returning those indices in reference order. It warns when a best overlap falls below a threshold, which is the signature of an avoided crossing the step size did not resolve. Overlap moduli are phase-independent, so the assignment needs no phase convention.

Rules may be composed: `connected` to pick the subspace at the first point of a sweep, `follow` to track it thereafter.

Because a rule may need ports in the eigenbasis, materialization computes `compute` eigenpairs, forms the ports the rule names on all of them, runs the rule, then slices eigenvalues, eigenvectors, and ports to the kept set.
Selection is frozen for differentiation (Section 11.3).

**Phase convention (decided).** Each kept eigenvector's global phase is fixed by one of two conventions, chosen per node:

- `:positive_sector` (the default): among the components of the eigenvector in the node's product basis whose coordinates are all nonnegative, the component of largest modulus is made real and positive. "Coordinate" means the grid value $\varphi_i \ge 0$ for a `PhaseGrid` axis and the charge $n \ge 0$ for a `ChargeBasis` axis; a `FockBasis` axis and a child-node axis (whose index is an eigenstate label) impose no restriction. For a symmetric potential the global maximum is degenerate between $\pm\varphi$ and would make the sign a coin flip; the positive sector (the positive quadrant in two grid dimensions) makes it unique. This is Eli's long-standing convention.
- `:follow`: the phase maximizing $\mathrm{Re}\,\langle v_{\rm ref}|v\rangle$ against the reference eigenvector, which gives continuous matrix elements along a sweep and is the default whenever `follow` is the selection rule.

The convention is applied to a node's kept eigenvectors before its ports are formed, so every exported matrix element carries it.

## 10. Materialization (decided)

### 10.1 Procedure

`materialize(tree, spec, θ)` is one function.
For each node, children first:

1. Evaluate every coefficient of the node's terms at $\theta$.
2. Build the preconditioner diagonal.
3. Run the node's solver for `compute` eigenpairs.
4. Form, in the eigenbasis, every port named by the selection rule.
5. Run the selection rule; slice energies and eigenvectors to the kept indices.
6. Form every remaining port needed above (inferred) or requested, as `DenseOp` in the kept eigenbasis.

The result is a tree of materialized nodes, each holding energies, kept eigenvectors, kept indices, and ports.
Energies are reported as absolute (not ground-subtracted) in GHz.

### 10.2 Outputs

- `energies(sys, node=top)`: kept eigenvalues.
- `port(sys, name, mode, node=top)`: the dense port in that node's kept eigenbasis.
- `eigenvectors(sys, node)`: kept eigenvectors in the node's product basis, for diagnostics and the edge-support check.
- The dressed and bare state interface of Section 10.3.
- Matrix elements are provided both with modulus (gauge-invariant) and with the node's phase convention (Section 9.1).

### 10.3 Dressed and bare states (decided, first version)

At any node the kept eigenvectors are columns in the product basis of the children's kept states, so a **bare label** is a tuple of child indices and the dressed-bare overlap matrix is $|V|^2$ with no further computation.
Modeled on scqubits' lookup interface, the node exposes:

- `bare_labels(sys, node)`: the product-basis labels as tuples of child indices, with a `depth` option that expands each child index recursively into that child's own dominant bare label, down to mode basis states.
- `decompose(sys, node, i; thresh)`: for dressed state `i`, the list of (bare label, amplitude) pairs with $|{\rm amplitude}|^2$ above `thresh`, sorted by weight. This is `dressed_state_decomp` of the zero-pi project.
- `dressed_index(sys, node, bare_label)`: the kept dressed state of largest overlap with that product state, with a warning when the largest overlap is below one half, where the assignment is ambiguous, and `nothing` when the product state has no kept partner.
- `bare_overlaps(sys, node)`: the full $|V|^2$ matrix, kept dressed states by product states.

At a leaf the same functions label by mode basis states (grid points, charge states, Fock levels), which is mainly useful for Fock-basis leaves.
Amplitudes carry the node's phase convention (Section 9.1).

### 10.4 Caching (decided)

Each node records the union of parameters its terms depend on, and its children's.
`materialize` memoizes each node's result keyed by the values of exactly those parameters, so in a flux sweep that moves only one qubit's flux, the other qubit's leaf is computed once.
The cache lives on the tree object and is invalidated by any change of tree structure, basis, solver, or selection settings.

## 11. Automatic differentiation (decided)

Amended 2026-11-11 (Section 20l): the three ChainRules primitives of Section 11.2 and Zygote as the engine are replaced by the upward perturbative pass of Section 20l (`derivatives`), with an `rrule` and an `frule` of one outer function (`spectrum`) for optimizers; Section 11.1 stands.

### 11.1 Where parameters live

Circuit parameters and external fluxes enter only through term coefficients.
Basis hyperparameters (grid range and spacing, cutoffs, `phi_zpf`) are fixed and never differentiated.
Therefore at a **leaf**, $\partial H/\partial\theta_p = \sum_t (\partial c_t/\partial\theta_p)\,O_t$ is a sum of products over the same, parameter-free local operators.

At a **branch** the local operators are the children's materialized ports, $V^\dagger O V$ and the children's energy diagonals, and these do depend on $\theta$ through the children's eigenvectors and eigenvalues.
So the gradient path through a branch has three legs: the branch's own coefficients, the dense port operators, and the energy diagonals.
The last two carry the cotangent down into the children.
A design that only propagated coefficient cotangents would silently return zero for every parameter that enters only through a child (for example $\partial E_{\rm top}/\partial E_{J1}$); this must be a tested failure mode.

### 11.2 Three primitives with ChainRules rules

Amended 2026-11-11 (Section 20l): not built; replaced by the per-node first- and second-order rules carried upward through the tree, with linear solves on the complement of the computed states at iterative nodes, and by the rules of `spectrum` in a ChainRulesCore extension.

Everything in `materialize` is ordinary array code except three operations, which get `ChainRulesCore` rules so that the mutating kernels beneath them are never seen by the engine:

- **apply**: linear in the state, in the coefficients, and in the local operators. Pullback: the state cotangent is apply with adjoint factors; the coefficient cotangent for term $t$ is $\langle \bar y, O_t x\rangle$; the operator cotangent for a `DenseOp` or `DiagonalOp` factor on axis $k$ is the partial contraction of $\bar y$ with the other factors applied to $x$, summed over all axes but $k$ (a $d_k\times d_k$ matrix, or its diagonal). Structured operators at leaves are parameter-free and receive no cotangent.
- **port formation** $P = V^\dagger O V$ (and the energy diagonal $\operatorname{diag}(\lambda)$): pullback gives $\bar V = O V \bar P^{\dagger} + O^\dagger V \bar P$ (through apply, matrix-free) and $\bar\lambda$ from the diagonal's cotangent. This is where a branch's cotangent enters its children.
- **eigenpairs**: returns the selected $(\lambda_i, v_i)$. Pullback, given cotangents $\bar\lambda_i$ and $\bar v_i$:
  energies by Hellmann-Feynman, $\bar H \mathrel{+}= \bar\lambda_i\, v_i v_i^\dagger$;
  eigenvectors by a projected linear solve, $(H-\lambda_i)\,w_i = -(I - v_i v_i^\dagger)\,\bar v_i$, then $\bar H \mathrel{+}= w_i v_i^\dagger + \text{h.c.}$ as appropriate for the real loss.
  When the node was solved densely, the full-spectrum formula $w_i = \sum_{j\ne i} v_j \frac{v_j^\dagger \bar v_i}{\lambda_i-\lambda_j}$ is used instead of linear solves.
  The linear solves use KrylovKit `linsolve` with the node's Jacobi preconditioner.
  $\bar H$ is never materialized: its contraction with each term, $\bar c_t = \sum_i \bar\lambda_i\langle v_i|O_t|v_i\rangle + \sum_i 2\,\mathrm{Re}\langle w_i|O_t|v_i\rangle$, and the operator cotangents of the term's factors, are computed through apply.

This one eigenpair rule serves all three solvers, since it depends only on $H$ and the converged pairs.
KrylovKit's own `eigsolve` rule exists (version 0.8 and later) and may be used as a cross-check in tests but is not relied upon.

**Cost of the backward pass (decided).**
The eigenvector pullback at a node costs one preconditioned linear solve per eigenvector with a nonzero cotangent, not one per kept state.
The first version targets losses built from a handful of energies and matrix elements among a handful of states, which cost a handful of solves at the top node, comparable to a few forward LOBPCG iterations, and run on the A4500.
A loss that depends on an entire $2000\times2000$ top port would cost 2000 solves of dimension 90 000 per backward pass, as much as or more than the forward solve; the rule must not break for such a loss, but making it fast (blocked solves with the chunking and memory discipline of Section 9) is later work.
At a densely solved leaf the full-spectrum formula needs all eigenvectors kept alive for the backward pass, about 1.6 GB per leaf at the 10 000 threshold; these may be held in host memory when device memory is tight **(my call)**, with projected linear solves as the alternative if that proves awkward.

Default engine: Zygote, through ChainRules **(my call)**.
Enzyme is a later option; KernelAbstractions kernels are Enzyme-compatible if rules ever need to move below the primitives.

### 11.3 Frozen selection

Amended 2026-11-11 (Section 20l): the warning exists as the selection-boundary test of the derivative pass (a kept state close to an unkept computed state warns and is recorded in `Derivatives.boundary`); the selection stays fixed, as here.

The selection rule is a discrete function of $\theta$ and is treated like an argmax: the forward pass records the kept indices and the derivative flows only through the selected energies, eigenvectors, and ports.
This is the exact derivative wherever one exists.
The package must warn when a kept and an unkept state are within a tolerance of each other in energy at a node, since the derivative is unreliable there.
Pointwise derivatives in a sweep are exact at each point; only finite differences across points can mix selections.
Intermediate truncations are robust to relabeling inside the kept set, because the parent only sees the kept subspace.

### 11.4 Near-degenerate kept states

Amended 2026-11-11 (Section 20l): the explicit cluster handling is the near-degenerate block report (`blocks`, `block_levels`): a pair of kept states is flagged when its coupling times the caller's scale of the parameter exceeds a fraction of its detuning, and each block carries an effective Hamiltonian for finite displacements; point derivatives stay the true derivatives, and an exactly degenerate coupled pair gives `NaN` with a report.
The cluster-invariant observables listed below were not built as separate functions.
The warning below "when two *kept* states at a node are within the tolerance of each other" was not built either: near-degenerate kept pairs are reported only as blocks.

Zero-pi qubits have ground doublets split by MHz or less while the spectrum spans GHz.
Within such a cluster the individual eigenvectors are only resolved to about (solver tolerance)/(splitting) in the forward pass, and the eigenvector pullback amplifies cotangents by $1/(\lambda_i-\lambda_j)$.
This is the true sensitivity, not an error, but it means that quantities which are not invariant under rotations inside the cluster, such as a single matrix element between the two doublet states, are ill-conditioned both forward and backward.
Intended losses do involve such matrix elements (the zero-pi doublet is the logical qubit), so this is first-version work **(decided)**:
the package warns when two *kept* states at a node are within the tolerance of each other;
it offers cluster-invariant observables, the energies, a cluster's mean energy and splitting, and the singular values or operator norm of a port's block within a cluster;
the `follow` selection rule and `:follow` phase convention (Section 9.1) give consistent labels and phases across a sweep so that non-invariant matrix elements are at least continuous;
and the eigenpair pullback must handle clusters explicitly, treating states within the tolerance as a subspace whose internal rotations are projected out of the cotangent when the loss is invariant to them, and warning when it is not.
Gradient tests must include a near-degenerate pair.

## 12. Export (decided)

Through package extensions, loaded only when the user loads QuantumToolbox.jl or QuantumOptics.jl:

- `to_quantumtoolbox(sys, node=top)`: the node's kept Hamiltonian as a diagonal `Qobj` and each port as a dense `Qobj`, with `dims` equal to the kept dimension of the node.
- `to_quantumtoolbox(sys, node; basis=:product)`: the node's Hamiltonian and ports in the *product* basis of its children's kept states, as dense or sparse `Qobj` with `dims` equal to the tuple of children's kept dimensions. This reproduces the existing zero-pi workflow (`H1⊗I + I⊗H2 + g n1⊗n2`) and keeps subsystem addressing (`ptrace`, `tensor`) available downstream. It is only offered when the product dimension is small enough to materialize.
- The same two for QuantumOptics.jl.
- Stretch (decided as a goal, not first-version): `save_npz(path, sys, node)` through an NPZ.jl extension, writing energies, kept indices, and ports in the node's phase convention, with a metadata entry naming the convention.

## 13. User-facing shape (decided)

```julia
spec = load_spec("two_zero_pi.json")
θ = (E_J1=4.0, E_C1=0.2, varphi_ext1=π, ...)      # one entry per spec.parameters

grid = PhaseGrid(-6π, 6π, 400; order=8)
q1 = Node([:varphi_1 => grid, :varphi_2 => ChargeBasis(15)];
          solver=Dense(), compute=500,
          keep=connected(core=[1, 3], port=(:n, :varphi_2), thresh=1e-2))
q2 = Node([:varphi_3 => grid, :varphi_4 => ChargeBasis(15)]; ...)
top = Node([q1, q2]; solver=LOBPCG(chunk=500), compute=2000, keep=lowest(2000))

sys = materialize(top, spec, θ)
energies(sys)
port(sys, :n, :varphi_2)
H, ops = to_quantumtoolbox(sys)

loss(θ) = energies(materialize(top, spec, θ))[3] - energies(materialize(top, spec, θ))[1]
Zygote.gradient(loss, θ)
```

Moving to the GPU is `top = adapt(CuArray, top)` or a keyword on `materialize`; the builder picks one.
Amended 2026-11-11 (Section 20l): derivatives are `derivatives(sys; wrt)`, and a loss is differentiated by Zygote through the `rrule` of `spectrum(top, spec, θ; ports)`, not of `materialize`.

### 13.1 Future-friendliness: Born-Oppenheimer surfaces (decided requirements, not first-version work)

A Born-Oppenheimer surface treats a slow mode's coordinate as a classical parameter: for each grid point of the slow mode, diagonalize the fast modes and record the energies $E_k(\varphi_{\rm slow})$.
In this design every operator of a grid mode other than its kinetic term is a `DiagonalOp`, so fixing the slow coordinate is a transformation on the term list, and the fast problem is an ordinary node whose coefficients depend on one extra named parameter.
The surface is `energies(materialize(fast, spec′, (θ..., φ_slow=x)))` swept over `x`; its slope is the Hellmann-Feynman derivative the eigenpair rule already gives, and derivative couplings later are the eigenvector derivative from the same rule.
Amended 2026-11-11 (Section 20l): there is no eigenpair rule; the slope and the eigenvector derivative come from the derivative pass, `energies(d, :x)` and `eigenvectors(d, :x)` of `d = derivatives(sys; wrt = :x)`.
Three requirements keep this cheap to add and are binding on the first version:

- **The term list is a first-class, user-constructible object**, not a loader internal: it can be built, inspected, filtered, and transformed from user code.
- **A named "promote mode to parameter" transformation on term lists** is part of the design: diagonal factors on that axis become scalar coefficient functions of a new parameter; the axis's own kinetic terms are dropped with a logged note; any non-diagonal factor on that axis inside a cross term (for example a capacitive coupling to the slow mode) is rejected with a clear error, never silently approximated. The first version need not implement it, but nothing may make it hard.
- **`DiagonalOp` can be constructed from data**, so a computed surface feeds back into a node as a potential on the slow grid.

The parameter tuple is open to any name, and per-parameter caching (Section 10.4) means a sweep over `φ_slow` recomputes only what depends on it.
A batched small-matrix eigensolver over all slow grid points at once is a later GPU optimization.

### 13.2 Future-friendliness: adiabatic labeling through a coupling ramp (later goal)

A cleaner dressed-state labeling than largest overlap is to turn the coupling off, where the eigenstates *are* the product states and labels are exact, and follow them adiabatically as the coupling ramps to full strength.
This design supports it with the pieces already present: a term-list transformation that multiplies the terms routed to a node (its cross terms) by a new parameter `g`, a sweep of `g` from 0 to 1, and the `follow` selection rule with the `:follow` phase convention (Section 9.1).
The one requirement on the first version is the one already stated in Section 13.1: the term list is first-class and transformable from user code, including scaling a selected subset of terms by a named parameter.

## 14. Numerical conventions (decided)

- Column-major, first axis fastest, matching `reshape`. Kronecker-product reference constructions in tests must therefore use `kron` in *reverse* axis order relative to QuantumToolbox's `tensor`.
- Energies in GHz ($h=1$); phases dimensionless; charges in units of $2e$.
- Grid stencils: thesis Table `tab:fd_stencils`; $n = -i\,\partial_\varphi$ uses the antisymmetric central first-derivative stencil of the same order.
- All state and dense-operator element types are `ComplexF64`. (Amended by §20f: `Float64` at nodes whose structure guarantees a real Hamiltonian.)

## 15. Packages (decided)

Core dependencies: `LinearAlgebra`, `SparseArrays` (stdlib); `KernelAbstractions`; `KrylovKit`; `LOBPCGEigensolver`; `ChainRulesCore`; `Adapt`; `GPUArraysCore`; `JSON3`; `RuntimeGeneratedFunctions`.

Extensions (weak dependencies): `CUDA` (strided-batched gemm, cuSOLVER dense eigen); `QuantumToolbox`; `QuantumOptics`; `PythonCall` (the circuit-quantization interface, extension only; Section 20j).

Test-only: `ITensors` (reference apply), `FiniteDifferences` and `ChainRulesTestUtils` (rules), `Zygote` (engine under test), `Test`.
Amended 2026-11-11 (Section 20l): `ChainRulesCore` is a weak dependency that loads the extension `KronFreeCircuitsChainRulesCoreExt`, not a core dependency (it is loaded in practice through `ITensors`, so the extension is always active); `Zygote` is test-only, in the separate environment `test/chainrules`; `FiniteDifferences` and `ChainRulesTestUtils` are not used.

Deliberately absent: `NNlib` (the batched gemm is ten lines in the CUDA extension), `ITensors` at runtime, `IterativeSolvers` (unmaintained), `Symbolics`, `QuantumToolbox` as the operator container.
Amended 2026-11-10 (Section 20j): `PythonCall` was listed here as deliberately absent; it is now a weak dependency that loads the extension `KronFreeCircuitsPythonCallExt`, and nothing in the core imports it.

Python side: `circuit-quantization` gains `export_julia_spec`, using the standard-library `json` module and SymPy's `julia_code` printer.
As built, that exporter is `to_spec` and `save_spec` (Section 20g), superseding `export_julia_spec`.

## 16. Testing strategy (decided)

- **Differential tests of apply** against the explicit Kronecker product for small systems, for every structure type and every axis position, with and without a block axis, on CPU; the same on GPU when run on a CUDA machine.
- **Solver tests**: all three solvers agree with each other and with the dense reference on small nodes; LOBPCG chunked equals LOBPCG unchunked.
- **Physics tests** against reference values from scqubits for the fluxonium, the transmon, and the zero-pi (single qubit and the two-zero-pi composite in the product basis), committed as data files under `test/fixtures` with the generating Python script alongside; no Python at test time.
- **Rule tests** with `ChainRulesTestUtils` for apply and eigenpairs, and end-to-end finite-difference checks of `Zygote.gradient` of energies and matrix elements with respect to a circuit parameter and a flux.
  Amended 2026-11-11 (Section 20l): the derivative tests compare every derivative with the package's own finite differences of `materialize`, and the rules of `spectrum` are tested directly and through Zygote in `test/chainrules`.
- **ITensor reference** for apply in the test suite.
- **Edge diagnostics**: a test that the grid edge-support warning fires for a deliberately too-small grid.
- CI runs CPU tests on GitHub Actions; GPU tests are run manually on `ewdesktop` before releases.

## 17. Evidence: the apply benchmark

A throwaway spike (`docs/spikes/2026-10-15-apply-benchmark/bench.jl`, results in `results.md` alongside) measured one Hamiltonian application for three cosine-coupled modes with dense local operators (A) and for two leaves with dense $k\times k$ ports (B), in milliseconds:

| Case | sparse Kronecker | gemm loop | KA dense kernel | permute+gemm | batched gemm | ITensor |
|---|---|---|---|---|---|---|
| A, $d=20$, Mac CPU | 1.50 | 0.40 | 0.67 | | | 0.45 |
| A, $d=80$, desktop CPU (16 threads) | infeasible | 22.5 | 122.6 | 67.0 | 19.8 | 79.0 |
| A, $d=80$, RTX A4500 | infeasible | 19.0 | 12.3 | 9.4 | 9.3 | 9.3 |
| A, $d=100$, RTX A4500 ($10^6$ states) | infeasible | 40.1 | 30.1 | 27.5 | 28.2 | 27.3 |
| B, $k=300$, desktop CPU | infeasible (8e9 nnz) | 2.6 | 31.0 | 3.8 | 2.4 | 4.1 |
| B, $k=500$, RTX A4500 | infeasible | 10.4 | 16.7 | 10.7 | 10.5 | 10.6 |

All methods agree to $10^{-14}$.
Conclusions that shaped this design: sparse Kronecker is out once ports are dense; batched gemm is the fastest dense method on both devices; ITensor equals it on GPU and loses by 3 to 4 times on CPU; the naive dense kernel loses on CPU by 5 to 20 times; and the A4500 numbers are bounded by its FP64 rate, so data-center GPUs should be roughly 25 times faster for the same operation.

## 18. Repository and process

- Repository `~/src/KronFreeCircuits.jl`, private, branch `main`. Nothing is committed until Eli approves this document.
- Layout **(my call)**: `src/` with `spec.jl`, `parameters.jl`, `bases/`, `operators/` (structure types and apply), `tree.jl`, `solvers/`, `selection.jl`, `materialize.jl`, `rules.jl`; `ext/` for CUDA, QuantumToolbox, QuantumOptics; `test/`; `docs/superpowers/` for specs and plans; `docs/spikes/` for throwaway evidence.
- Implementation follows the superpowers workflow: a written plan from this spec, test-driven development, subagent-driven execution where tasks are independent.
- Suggested build order **(my call)**: spec loader and parameters; bases and structure types with the apply layer on CPU; dense solver and single-leaf materialization validated against scqubits fluxonium; branch materialization and ports validated against the two-zero-pi product-basis construction; selection rules; LOBPCG and Lanczos; CUDA extension; ChainRules rules and gradient tests; export extensions; caching.
  Amended 2026-11-11 (Section 20l): the ChainRules rules and gradient tests became the perturbative derivatives of Section 20l, built before export and caching.

## 19. Left to the builder

- Internal type and function names beyond those used above.
- Exact JSON field names beyond those listed, and the loader's validation and error messages.
- How the dense solver assembles the explicit matrix (applying to basis vectors versus direct Kronecker assembly), and the threshold defaults.
- KernelAbstractions workgroup sizes and the CPU backend's threading interaction with BLAS threads.
- The deflation shift $\sigma$ and the chunk-size default rule.
- The near-degeneracy tolerance for the selection warning.
- Whether a cosine is stored as two conjugate terms or one Hermitian term.
- The memoization key representation and cache eviction, if any.
- Documentation tooling.

## 20. Open questions

None.
The three questions raised by the builder check were answered by Eli on 2026-10-15 and folded into the text:
first-version losses are a handful of energies and matrix elements among a handful of states (Section 11.2);
losses do involve near-degenerate clusters, so cluster handling, the `follow` rule, and the `:follow` phase convention are first-version work (Sections 9.1 and 11.4);
dynamics move to Julia with the two-zero-pi project as the acceptance goal, and an NPZ export is a stretch goal (Sections 2 and 12).

## 20a. Amendments after implementation (plan 1, 2026-10-16)

Recorded here so the document stays the authority; each is also in the plan-1 ledger with its reasoning.

- Section 9: Dense returns the lowest `compute` pairs when `compute > 0` (above). `Lanczos` has `maxiter` (default 1000; KrylovKit's 100 did not converge a 9k-dimensional zero-pi leaf) and `verbosity` fields.
- Section 9.1: `:follow` phase with no previous materialization (the first point of a sweep) falls back to `:positive_sector` instead of erroring; a node whose rule is `follow` defaults to the `:follow` phase.
- Section 4 / 7.3: composite ports (a term routed to a branch with two factors inside the same child) were not supported in plan 1 and raised a clear error; multi-mode port keys were design debt for a later plan, since paid by the joint ports of Section 20i. `Factor` carries only a single-mode `Port`, so a data-valued operator (Section 13.1's surface fed back as a potential) has no entry into a `Term` yet; the mechanism is to be designed before plan 2 wraps `materialize` in differentiation rules.
  Amended 2026-11-11 (Section 20l): derivatives do not wrap `materialize`, so the data-valued `DiagonalOp` entry is no longer blocked by them; it is still open, as its own item.
- Section 5.3: coefficient expressions are restricted to an allowlist of arithmetic and elementary functions and may not execute arbitrary code; the allowlist must include SymPy's dotted operators (`.* ./ .^ .+ .-`), which plan 1's implementation still lacks (ledger: parked, load-bearing for plan 5).
  Amended 2026-11-11 (Section 20l): the dotted operators are accepted and read as the scalar ones (`src/parameters.jl`), and the derivative pass differentiates the stored expressions symbolically over the whole grammar.
- Section 6: the grid `exp2pin(k)` shift has the opposite sign from the plan's first draft (`ShiftOp(N, -2πk/Δ)`), fixed during implementation and pinned by a plane-wave test; the Fock `n` port is the charge operator $i n_{\rm zpf}(a^\dagger - a)$, not the number operator.
- Section 16: the two-zero-pi acceptance test is gated behind `KRONFREE_SLOW_TESTS=1`; the ITensor reference implementation is deferred to plan 3. The scqubits references were generated with scqubits 4.3.1 from `/opt/anaconda3/envs/circuit_quantization`; the zero-pi leaf test does not pin port sign conventions (the symmetric spectrum is invariant under θ→−θ and φ→−φ), so matrix-element sign tests belong in plan 4.

### 20b. Amendments and decisions of 2026-10-16 (plan 1b and design discussion)

- Section 9.1, branch phase convention: at a branch, `:positive_sector` fixes each kept state's phase by its exact full-space amplitude at the joint peak of its dominant product-state component, computed from the children's stored peak points and k×k peak-amplitude matrices (a child's amplitude at every other kept state's peak). This is invariant to a child state's sign flip, recursive, and costs k² per node with no grid-sized arrays above the leaves. A diagnostic warns when the amplitude at the chosen point is below one half of the dominant component's own contribution (heavy hybridization). Known limitation: an antisymmetric exchange combination of two identical subsystems can have vanishing amplitude at that point, leaving its phase arbitrary (observed for state 15 of the two-zero-pi test system); resolution pending, either a fallback to the joint peaks of the next few components or the tabled exact search.
- Tabled: the exact full-space positive-sector maximum by Cauchy–Schwarz branch-and-bound over sector grid points (plan 1b Task 3), with the candidate set cut by 2ⁿ for n grid axes by enumerating only all-positive coordinates; expected to terminate quickly when peaks line up, capped with a fallback otherwise.
- Section 16, testing philosophy: the default suite compares against scqubits on identical small discretizations (same range and point count, scqubits' 7-point stencil = our `order=6`), where agreement is at the 1e-12 level and independent of physical convergence; physical-correctness and convergence comparisons (different stencils on a shared grid, wide converged grids) are gated slow tests to be grown over time. Composite-system comparisons with nonzero offset charge differ from scqubits' construction by the truncation order of the linear n_g cross terms (routed into our leaves before truncation); exact hierarchical agreement is pinned with offset charges set to zero.
- Section 11, flux derivatives: external flux enters only the cosine offsets and the inductive linear term, so ∂H/∂φ_ext is the same operator pair with differentiated coefficients and Hellmann–Feynman gives ∂E/∂φ_ext = −E δ′⟨sin(ωφ+δ)⟩ with no basis-dependent pieces; coefficient derivatives will be computed with ForwardDiff inside a ChainRules rule for coefficient evaluation (the compiled coefficients are generated functions); second derivatives for flux-noise dephasing via forward-over-reverse are a plan-2 target.
  Amended 2026-11-11 (Section 20l): coefficient derivatives are symbolic (`differentiate`), not ForwardDiff inside a rule, and the second derivatives for dephasing are the mixed second derivatives of the upward pass.
- Section 5.2, noise operators (for plan 5 and a later noise plan): the spec gains an `operators` section of named general term lists (coefficient expression plus (mode, port) factors) carrying analytic derivative operators (∂H/∂φ_ext as sine terms, ∂H/∂E_J, ∂H/∂n_g) and element operators such as the charge on a lossy capacitor, the flux across a lossy inductor, and sin(φ_J/2) across a junction; plus a `noise_channels` table mapping elements and loops to those operators and parameters. Julia derives the same derivative operators automatically for cross-checking and implements the rate formulas.
- Section 6, half-integer ports: sin(φ_J/2) needs e^{±iω_jφ_j/2}; legal on grids and in Fock; on compact modes it requires a doubled charge basis on half-integer charges (the odd-parity sector is the same circuit at n_g shifted by one half, and the operator connects the sectors, as in quasiparticle tunneling); to be added as a basis variant only when a quasiparticle channel is requested.
- Architecture decisions reaffirmed: the JSON spec remains the Python–Julia contract (the topological, parameter-independent content from Python; parameter-dependent coefficients compiled in Julia); a PythonCall extension for in-memory spec transfer is an optional convenience, never a dependency; porting the symbolic quantization machinery to Julia is rejected unless parameter-dependent coordinate choices become first-class, in which case only that numeric step moves.
  Superseded in part on 2026-11-10 by Section 20j: the PythonCall extension now exists and does more than in-memory spec transfer (numeric frame changes and run records), it is still optional and never a dependency of the core, and the numeric frame change stays in circuit-quantization rather than moving to Julia.

### 20c. Backend and solver decisions of 2026-10-16 (ITensor audit; plan 2)

Source: Eli's audit spike `docs/spikes/2026-10-16-itensor-audit/` (REPORT.md, HANDOFF.md, CORRECTNESS-NOTES.md and the Ritz, packing, real-arithmetic and precision follow-ups). Eli's decision after the spike: ITensors for dense parents, LOBPCGEigensolver with adaptive per-Ritz Jacobi. These amend Sections 8, 9, 9.1, 11 and 15. Plan 2 implements the solver and backend; automatic differentiation moves to the plan after it.

**Evidence the decisions rest on** (all on the retained two-zero-pi parent, 300×300 kept states, dimension 90 000, unless stated):

- At the retained-parent boundary a buffered ITensor adapter with ordinary-array solver vectors matches native CPU speed (10.0 s against 10.4 s for a six-state solve); the original 2.3× ITensor penalty came from densified diagonal contractions and conjugating inner products, both avoidable. ITensor applied to *primitive* leaf operators is 10× slower than the banded kernels and is rejected.
- On the H200, stock LOBPCG with fixed Jacobi solves 500 / 1 000 / 2 000 states in 3.9 / 12.9 / 49.5 s against 62 / 198 / 834 s on nine CPU threads. cuTENSOR beats a hand-written strided cuBLAS pair by 17–26 % per application and 5 % per solve on the A4500.
- Adaptive per-Ritz Jacobi (cached Ritz values through the public `precondprep!`/`ldiv!`/callback hooks) gives 3.2× on the A4500 and 2.8× on CPU at 120 states, 2.7× at 500, by locking most states within four iterations. Measured on a 1 % coupling step with a warm start; cold-start and strong-mixing robustness are not established.
- Basis packing inside the solver gives 2.3–2.7× more on the H200 at 500–2 000 states but needs a private patch of LOBPCGEigensolver; it is slower on CPU and on the A4500 at 120 states.
- Real arithmetic (Float64 where the representation is real) gives 3.4× on the A4500 and 2.4× on CPU at 120 states. Aligned secant extrapolation along a smooth sweep gives a further 1.5×. An FP32 first stage gives 5–8 % and failed to converge below 1e-4; not adopted.
- The leaf is now the bottleneck: the 300-state zero-pi leaf (dimension 9 331) took 90 s with single-vector Lanczos on CPU, against 1–4 s for the parent on a GPU. Coupling sweeps reuse leaves; flux and circuit-parameter sweeps do not.

**Decisions.**

1. *Backend boundary (amends Section 8).* Structured leaf operators (`BandedOp`, `ShiftOp`, `DiagonalOp`) stay native: KernelAbstractions kernels and broadcasts. `DenseOp` factors, which arise as children's ports at branches, are applied through ITensor contractions on scratch-backed, offset-view-safe storage; on GPU this is cuTENSOR through NDTensors. Solver vectors are ordinary arrays (`Array` or `CuArray`), never ITensors. The pure-diagonal part of a node's operator (children's energy diagonals, diagonal potentials, the constant) is fused into one state-shaped vector applied by a single broadcast, which is also the Jacobi diagonal. This is two layers with one boundary, not a mix of backends: what crosses it is kept energies, dense port matrices and the phase/following metadata. The permute-then-gemm `apply_axis!` for `DenseOp` is kept as the reference the ITensor path is tested against.
2. *Solvers (amends Section 9).* `LOBPCG` is the stock JuliaMolSim LOBPCGEigensolver, pinned at the tested minor version, with a block `mul!` adapter honoring the workspace contract of Section 8. Preconditioners: `:ritz` (default; per-column `R ./ hypot.(D .- θ_j, η)` with θ initialized from the diagonal expectation and thereafter the previous iteration's Ritz values from the solver callback, locked prefix removed; `η` fixed per solve, Eli's decision, default 0.01 in the Hamiltonian's energy units and documented as fixture-derived), `:jacobi` (fixed, `(D - σ)^{-1}`, `σ = min D - 1`) and `:none`. The callback and locking order the `:ritz` cache depends on is pinned by a test, and the dependency's compat is held to the tested version until that test is rerun against a newer one. Warm start: a node's previous kept eigenvectors are the initial block when a previous materialization exists (raw coefficient reuse; the audit measured that transporting them physically is a worse seed). Chunked deflation (Section 9) and guarded batching are deferred: batching gave 13–15 % for four times the work. Basis packing is not carried as a private patch; it is proposed upstream as an optional workspace strategy. `Lanczos` remains available with its documented limitation: a single-vector Krylov space does not resolve multiplicity (KrylovKit documents this), which matters for exact charge-parity sectors under cos(2φ) and for symmetric composites; `LOBPCG` and `Dense` do, and a repeated-eigenvalue regression pins it.
3. *Leaves get the block solver too.* `LOBPCG` with the Ritz preconditioner runs on leaves over the native kernels, on CPU or GPU, which is where the 90 s now goes. The full asymmetric zero-pi leaf (31×301×20 = 186 k) has no dense option.
4. *Device.* A CUDA package extension moves a node's operator data (ports, energies, fused diagonal, kernel weights, scratch) and the initial block to the device; the solve runs there; eigenvalues and eigenvectors return to host arrays, so `MaterializedNode` is unchanged in plan 2. Forming ports on the device while the eigenvectors are resident is the next step (2 000 kept states × 90 000 on CPU is tens of seconds). Whole-hierarchy device residency needs a storage abstraction and is later work.
5. *Following across changing child bases (correctness; amends Section 9.1).* The `follow` rule and the `:follow` phase compared coefficient vectors `R' V`, which is the physical overlap only when the child bases are unchanged; at a branch whose children were re-diagonalized (any flux or circuit-parameter step) it is wrong. The physical overlap is `S_parent = V_old' (⊗_a S_a) V_new` with `S_a = ⟨old child a | new child a⟩`, recursive to the leaves where `S = I`. Materialization transports the previous kept vectors into the current product basis once per node, `R̃ = (⊗_a S_a') V_old`, and hands `R̃` to both consumers; selection and phase code are unchanged. Kept counts per child must match between the two materializations (already required). Pinned by a test that permutes a child's kept order between two materializations.
6. *Dependencies (amends Section 15).* Add `ITensors` and `LOBPCGEigensolver`; `CUDA` (and `cuTENSOR`, loaded by the user for the fast path) as weak dependencies of the extension. `KernelAbstractions` compat widens to 0.9 and 0.10: CUDA 6.4 requires 0.9, and 0.10's CPU backend runs kernels through POCL, which produced a kernel-link failure in the spike's environment; the suite is run under both.
7. *Real arithmetic.* Its own plan after plan 2 and before autodiff: ports carry a character (real, imaginary, general) established by construction from each basis's operators and the term coefficients, propagated through real eigenvector projection; a parent is real when all its couplings are; complex observables and derivative operators stay complex. Never inferred from n_g or flux values or from small imaginary parts. Plan 2 keeps `ComplexF64` but writes the new solver and backend code without literal element types.
8. *Sweep layer (later).* Optional aligned secant prediction from two previous points, with Procrustes alignment inside near-degenerate clusters, as a sweep-level feature independent of the solver. Not an initial Rayleigh–Ritz step, which cost more than it saved.
9. *Consequences for the autodiff plan (Section 11).* The eigenpair rule is ours, as planned: stock LOBPCG under Zygote crashed the compiler, and the spike's working toy reused a private KrylovKit helper, usable only as a cross-check. The apply rule over the ITensor path needs an owned array/ITensor conversion rule with explicit index order. The phase factor `conj(a)/|a|` and the transport maps are differentiated with their discrete choices frozen. A loss on a few top states still induces a dense cotangent on the child ports, hence response solves for many kept child states; the cost discipline of Section 11.2 stands.
   Amended 2026-11-11 (Section 20l): no eigenpair rule, apply rule or ITensor conversion rule was built; the upward pass of Section 20l differentiates through the stored ports and phase convention with the discrete choices fixed, and costs one pass per parameter rather than solves per cotangent.
10. *Not adopted:* mixed precision; packing as a private patch; ITensor on primitive operators; ITensor vectors as solver vectors; ITensor's own eigen/SVD AD paths (failed on the tested versions and not needed).
11. *Given systems (Eli's request of 2026-10-16; amends Section 4).* A tree may contain pre-diagonalized subsystems: `GivenSystem(mode; energies, operators)` is a third kind of child beside `Node` and `mode => Basis`, at the tree level. `energies` are its eigenvalues in ascending order and `operators` a dictionary from port kind (any `Symbol`, the standard kinds included) to a k×k matrix in that eigenbasis. It materializes to a node with identity eigenvectors, all states kept, its own energies as the diagonal term on its axis, and its operators as ports; it imposes no positive-sector restriction (its axis is an eigenstate label, like a node child), its peak amplitudes are the identity, and for following its overlap with its previous self is the identity, because its labels are the user's. Its own Hamiltonian is its energies; a term whose factors all lie on its mode (a drive or Stark shift) is applied at its parent through its ports, since routing never descends into it. Coupling terms are ordinary `Term`s naming its mode and a port kind it provides; the spec-form `materialize` gains an `extra_terms` keyword for them, since they are not in the circuit spec. Use cases: a resonator or a qubit from another tool, a subsystem exported from scqubits, a fixed model of a neighbor.
12. *Pathological-case catalogue.* `docs/spikes/2026-10-16-pathological-test-catalogue/README.md` (Eli, 2026-10-16) lists the solver, operator, transport, following and projection cases with corrected expectations; plan 2 maps its case identifiers onto tasks and adds a pre-acceptance regression task for the cases that need the whole pipeline.

**Execution notes (plan 2, 2026-10-17).** Recorded from the SDD ledger; each amends the decision it names.

- *Decision 2, guards and warm starts.* With LOBPCGEigensolver's prefix locking, checking convergence on the lowest `m` only let an exact excited seed lock on the first iteration and return before a guard column found a lower state (seen on the free-rotor crossing). The guard columns are therefore converged along with the block (`n_conv_check = m + g`) and the lowest `m` of the sorted block returned. A supplied warm-start block receives a deterministic relative perturbation of `sqrt(eps)` per column, because an exactly zero residual column makes the library's orthogonalization produce NaN. Consequence: with guards, a warm start reduces H columns (the active block is tiny after the seeded columns lock) but not necessarily iterations, since the guards converge from random either way; on stiff leaves the guards can run the solve to `maxiter` after the requested states have converged, so the guard policy needs a stopping rule (open). Default `maxiter` is 1000, matching `Lanczos`.
- *Decision 3, leaves.* Measured on the 300-state symmetric zero-pi leaf (dimension 9 331, ewdesktop): `LOBPCG(:ritz)` does not converge in 1000 iterations (291/300 after 266 s CPU, 294/300 after 192 s GPU; 294 s / 217 s with `maxiter = 3000`), `Lanczos` 84 s, `Dense` 69 s on eight BLAS threads. The leaf solver of record at this size is `Dense` (the Section 9 threshold of 10 000 stands). The likely, untested explanation is that a diagonal preconditioner captures the children's energies at a branch but not the kinetic stencil on a grid leaf; the next leaf item is a grid-aware preconditioner (kinetic or shift-invert), which the 186 k-dimensional asymmetric leaf also needs. `docs/spikes/2026-10-17-plan2-benchmark/results.md` holds the measurements; the parent solve reproduces the audit (`:ritz` iteration loop 3.01 s on the A4500 against 3.20 s, identical 514 H columns).
- *Decision 1, one device-dispatched hook.* NDTensors' cuTENSOR extension copies an operand at a nonzero offset into a block-sized temporary on every contraction; `_contraction_operand` stages such an operand into free scratch and its device method dispatches on `CuArray`. This is a storage-staging exception to "dispatch on structure, never on device": it does not choose how an operator is applied.
- *Decision 4, ports.* Port formation after a device solve runs on the host (1.24 s for two charge ports on 120 kept states of a 90 000-dimensional parent, one BLAS thread); device-side port formation is the next parent-side item.
- *Decision 5, transport.* Implemented as specified; child overlaps are recomputed at every node that uses `prev` (quadratic in depth, unmemoized). A `GivenSystem` child's overlap with its previous self is the identity by construction; its basis identity is mode and dimension.
- *Selection and near-degenerate clusters (new finding, decision pending).* On the `cos 2φ` fixture at `n_g = 1/2`, Dense mixes the near-degenerate parity pairs and `connected` keeps one member of a pair without its partner, so the kept span is not parity invariant (leakage 0.014). Selection rules do not close over near-degenerate clusters; the proposed fix is cluster closure by an energy tolerance in `select`, consistent with Section 11.4. Recorded in the catalogue's `outcomes.md` as a contract gap.
- *Test suite.* The default suite runs about 108 s (solver and extension compilation dominate), superseding the "about 60 s" target; the gated suite about 8.5 min; the device suite (`test/cuda/`, run on a CUDA machine) 303 assertions with zero device allocation after warm-up.

Plan order after this amendment: 2 solver and ITensor parent backend with CUDA extension (this amendment); 3 real arithmetic; 4 automatic differentiation; 5 export, caching and NPZ; 6 Python exporter with `operators` and `noise_channels`; then the sweep layer and the noise plan. The two-zero-pi acceptance test and the tabled antisymmetric-state phase fallback are unchanged.
Amended 2026-10-19 (Eli): the Python exporter (`export_julia_spec` in `circuit-quantization`) moves ahead of automatic differentiation and is the next plan after plan 3, for the provenance of the test fixtures, which are so far transcribed by hand from the scqubits source.
Plan order as of 2026-10-19 (after plan 4): 1, 1b, 2, 2b, 3 (real arithmetic) and 4 (spec files from `circuit-quantization`: its `to_spec`/`save_spec`/`load_spec`, with generated cross-check fixtures) are done; next is 5, automatic differentiation; then 6, export, caching and NPZ; then the sweep layer and the noise plan, which needs the format's later `operators` and `noise_channels` sections (§20b) on both sides.
That exporter became `circuit-quantization`'s `to_spec` and `save_spec` (Section 20g), superseding `export_julia_spec`.

### 20d. Proposed design: a periodic phase grid for compact modes (2026-10-17, for Eli's review)

Motivation (Eli, 2026-10-17): non-standard compact modes (cos 2φ, higher harmonics, numerically supplied 2π-periodic potentials) are numerically finicky in the charge basis; a wraparound phase grid is preferred, accepting complex arithmetic at generic offset charge. The convention is the thesis listing (`code-snippets/chapter_0.py`, `transmon_phase_fd`): grid on [−π, π) without the endpoint, the plain real second-derivative stencil, and the offset charge entering only through the twisted boundary ψ(φ+2π) = e^{−2πi n_g} ψ(φ), realized as the phase e^{±2πi n_g} on the stencil entries that wrap around.

- **Basis.** `PeriodicGrid(N; order=8)` is a `Basis` for `:c` modes beside `ChargeBasis`, chosen by the user (the charge basis stays the default for a single cos φ). Coordinates φ_i = −π + 2π i/N, spacing 2π/N, central stencils of the given order wrapping around; no `n_g` in the basis, so basis hyperparameters stay parameter-free.
- **Twist without parameters in operator data (the rule of Section 11.1 is kept).** A wrapped derivative stencil splits into three parameter-free local operators: the interior band and the two wrap corners W₊, W₋. A derivative term on a twisted mode expands into three terms with coefficients multiplied by {1, e^{2πi n_g}, e^{−2πi n_g}}. In the twisted gauge every occurrence of (n_m − n_g,m) in the canonical quadratic form 4 Σ_ij (n_i − n_g,i) E_C[i,j] (n_j − n_g,j) is replaced by the twisted derivative D_m (so the diagonal kinetic term is 4E_C[m,m] times the twisted second derivative, and a cross term with another mode j is 8E_C[m,j] D_m ⊗ (n_j − n_g,j)); the canonical expansion into linear and constant pieces is performed only for non-twisted modes. The physical charge port exposed for couplings is n = −i D₁(twisted) + n_g I, so a coupling factor on a twisted mode expands into the three derivative pieces (parameter-free ports the leaf forms in its eigenbasis) plus an identity piece whose coefficient n_g multiplies the remaining factors of the term; a convenience accessor assembles the physical charge matrix V†D₁V + n_g I for the user. `e^{ikφ}` for integer k is diagonal and unaffected; half-integer k (quasiparticle ports, Section 20b) connects twist sectors and stays a later item. Hermitian by construction: W₋ is the adjoint of W₊ and the coefficients are conjugate.
- **Where the expansion lives.** Term construction becomes basis-aware for compact modes on a `PeriodicGrid`: `circuit_terms(spec, bases)` (or a post-pass over the tree's bases in `materialize`) performs the expansion; `localop(::PeriodicGrid, port)` serves the three derivative pieces under distinct port kinds (`:d2`, `:d2wrap` with k = ±1, `:d1`, `:d1wrap`) and the diagonal φ-function ports. Coefficient derivatives with respect to n_g flow through the e^{±2πi n_g} coefficients exactly as other parameters do.
- **Sectors (Eli's request).** Custom 2π-periodic elements are supplied as Fourier series, i.e. as cosine terms with integer harmonics (a numerically supplied potential is expanded on the Python side), so the periodicity is known a priori. The loader requires N divisible by every integer harmonic ω appearing in a cosine on that mode (from `CosineSpec.omega`), so the translation by 2π/ω maps grid to grid and the ω charge-parity sectors are exact; an informative error otherwise. An optional `sectors=` argument on `PeriodicGrid` adds a divisibility requirement beyond the declared harmonics. Non-integer harmonics on a compact mode are rejected at load time.
- **Arithmetic and structure.** Complex except at n_g ∈ {0, ½}, where the twist phase is ±1 and the operators are real (relevant to the real-arithmetic plan). The wrapped corners make the mode's operators sparse but not banded; a leaf solver that relies on bandedness (Section 20c notes, the leaf-solver spike) must use the sparse path for such leaves.
- **Validation.** Against the charge basis on identical physics: the free rotor, cos φ, pure cos 2φ with both sectors, cos 2φ with a weak cos φ, at n_g ∈ {0, ½, generic}, including convergence in N and the exact half-period translation symmetry on even N; ports and gauge convention cross-checked against the charge basis (the audit's REAL-WARM report lists these cases).

**Decided 2026-10-30: the periodic gauge (design `2026-10-30-periodic-gauge-design.md`).**
The twisted gauge above is retired; the bullets above it are historical.
`PeriodicGrid` serves `n`, `n2` and integer `expiphi(k)` as parameter-free cyclic operators with `PhaseGrid`'s characters, wavefunctions are strictly periodic, and the offset charge enters only through the canonical quadratic form, so term construction is basis-independent (`circuit_terms(spec)`, `constant_term(spec)`) and `nport()` on a grid mode is an ordinary port in both forms of `materialize`.
Reasons: fidelity to the paper's quantization, and charge driving — the twisted gauge is `U(t) = e^{-i n_g(t) φ}` applied to the periodic-gauge state, whose time derivative for an AC gate voltage adds the generator `ṅ_g φ`, not an operator on the circle; in the periodic gauge a drive is `f(t) n̂` with a fixed `n̂`.
Costs, accepted by Eli (2026-10-30/31): a grid node is real only when the linear `n` term is structurally absent (literal `n_g = "0"`), not at `"0.5"`; and the `n_g = ½` sweet-spot pairs are degenerate only to `O(Δ^order)` on the grid (order 8, free rotor: 1.1e-8 at N = 32, 4.4e-11 at 64, roundoff at 128; pure cos 2φ at E_2 = 15 E_C, lowest two pairs: 4.5e-9 and 1.1e-7 at N = 64, 1.8e-11 and 4.3e-10 at 128), pinned by `test_periodic.jl`.

### 20e. Leaf and branch solver policy, `Auto()`, and the solver extension API (decided 2026-10-17)

Source: the leaf-solver spike `docs/spikes/2026-10-17-leaf-solvers/results.md` (every row validated against exact references) and the discussion of 2026-10-17. Amends Sections 8 and 9 and the execution notes of 20c.

**The principle.** The best eigensolver follows the operator's cost structure, not the device. At a primitive leaf, applying H is cheap (structured kernels, memory-bound) and the spectrum is wide (the stencil's 16 760 GHz against 110 GHz of wanted states), so spend applications to avoid orthogonalization and Rayleigh–Ritz steps, and never use a diagonal preconditioner (the order-8 stencil puts a constant +7 212 GHz on every diagonal entry, which made `:ritz` a no-op). At a branch, applying H is a dense contraction per port and the fused diagonal of children's energies is nearly the answer, so spend a preconditioner to avoid applications.

**Measured (2D zero-pi leaf, 9 331 states, lowest 300, ewdesktop; branch: the 90 000-state parent, lowest 120, A4500):**

| Where | Method | Time |
|---|---|---|
| leaf, CPU | sparse Cholesky shift-invert Lanczos, real arithmetic | 2.5 s (3.2 s with assembly); spectrum slicing on 8 threads 1.2 s |
| leaf, GPU | Chebyshev-filtered block iteration, seeded with the lowest product states of the separable 1D eigenbases, degree 100 | 2.5 s with the package operator; 1.2 s in real arithmetic |
| leaf, both | `Dense` 69 s; plain Lanczos 82 s / 155 s; `LOBPCG(:ritz)` nonconvergent; separable-preconditioned LOBPCG 31 s GPU; one-node-per-mode hierarchy 23 s (needs 220 of 301 φ states to be exact); FEAST 23 s | not chosen |
| branch, GPU | `LOBPCG(:ritz)` seeded with the lowest product states by summed child energy | 5.9 s (cold random start 41 s; the filter 35–42 s) |

**Decisions.**

1. *One operator everywhere.* The `KronFreeOperator` with its KernelAbstractions kernels stays the single apply for every node and device (portability; the term-wise structure the autodiff rules and port cotangents need; every `LocalOp` including the periodic grid). A cached sparse matrix was measured no faster on the GPU at block widths (both are bandwidth-bound) and 1.6–2.9× faster on the CPU, where the leaf default does not apply H at all; the gain that matters is real arithmetic (2× everywhere), which is the real-arithmetic plan's item.
2. *Derived quantities, on demand and memoized, never built by default:* `sparse(H)` (the sparse Kronecker sum of the structured factors, axis 1 fastest; refused above a nonzero cap so a branch cannot ask for it), `gershgorin_bound(H)` (a rigorous upper bound on λ_max from the fused diagonal and the factors' ∞-norms; 0.2% above the truth on the leaf, computed in milliseconds), `separable_eigenbases(H)` (per-axis eigendecompositions of the single-axis terms and their summed energies), `product_seed(H, k)` (the lowest k product states by summed 1D energy, or by the fused diagonal when the axes are already eigenbases, i.e. at a branch).
3. *Leaf solvers.*
   `ShiftInvert` (CPU): sparse Cholesky of H − σI with σ below the lowest eigenvalue, LU fallback when not positive definite, real arithmetic when the assembled matrix is exactly real, KrylovKit Lanczos on the inverse with a tight inner tolerance and a mandatory true-residual check (the Krylov tolerance applies to the inverted operator), optional spectrum slicing across threads, optional block size 2 as degeneracy insurance (block size 4 reported false convergence in KrylovKit 0.10.4 and is not offered), and a completeness certificate by eigenvalue counting from the factorization's inertia, on by default for lowest-m solves (`certify=:auto`), opt-out with `certify=false`.
   `ChebyshevFilter` (GPU, also CPU): block filtered subspace iteration started from `product_seed`, λ_max from `gershgorin_bound`, degree capped near 100, the recurrence rescaled every ~20 steps, a rank-revealing re-orthogonalization (SVQB with drop-and-refill) and true-residual locking.
   Neither relies on a diagonal preconditioner.
4. *Branch solver.* `LOBPCG(:ritz)` with its cold start seeded from `product_seed` (default `seed=:product`); a previous materialization still supplies the warm start when present. This is a 7× improvement on the cold start for free.
5. *`Auto()` is the default solver* of every node and resolves at materialization from the node kind, product dimension, requested count and storage: `Dense` when the product dimension is at most 2 500 (Eli's placeholder, to be benchmarked at the end; the existing 10 000 remains the warning threshold for a user who picks `Dense` explicitly on a large node); otherwise at a leaf `ShiftInvert` on `Array` storage and `ChebyshevFilter` on device storage, and at a branch `LOBPCG(:ritz, seed=:product)`. An explicit solver always wins.
6. *Composable pieces and an extension API.* Preconditioners, spectral transformations and seeds are arguments, not solver types: the defaults use the same objects a user can pass. The `Solver` contract is documented: `solve(s, H, m; X0=nothing)` returns ascending energies and orthonormal eigenvectors as host arrays with true residuals below the solver's tolerance; `storage(s)` names the array type the operator is built on; the operator guarantees `mul!` on blocks, `diagonal(H)`, `sparse(H)`, `gershgorin_bound(H)`, `separable_eigenbases(H)` and `product_seed(H, k)`. A user-defined solver needs only the first method.
7. *Banded factorizations* are not adopted: the Kronecker-sum band is 95% zeros (3.5 s against 2.5 s sparse), has no GPU library support, and a twisted periodic grid breaks the band.
8. *3D leaves are tabled.* Best measured: one node per mode on the GPU 83 s, the filter 174 s, sparse Cholesky 254 s with an 11.6 GB factor. The first structural change to try when resumed is the (θ,φ)+ζ hierarchy, the plan-1 test tree, whose child is the 2D problem above.

Plan 2b implements decisions 2–6 together with the periodic grid of Section 20d; real arithmetic follows as plan 3, then autodiff.

### 20f. Real arithmetic by port character (decided 2026-10-18; plan 3)

Source: the plan-3 brainstorm with Eli on 2026-10-18, refining Section 20c decision 7.
Amends Sections 7.2, 9, 9.1, 10, 14 and 20e.
The measured motivation is Section 20e decision 1 and the audit (Section 20c): real arithmetic is worth about 2× at the leaf (`ShiftInvert` 2.5 s against 4.9 s, `ChebyshevFilter` 1.2 s against 2.5 s on the 9 331-state zero-pi leaf) and 2.4–3.4× at the branch.
`ShiftInvert` already factors in real arithmetic when the assembled matrix is exactly real, so on the benchmark leaf plan 3's gain is in `ChebyshevFilter`, `Dense`, the branch `LOBPCG`, port formation and the apply kernels, not in `ShiftInvert`.

**The rule (decided).** A node runs in real arithmetic only when its structure guarantees that its Hamiltonian is real, at every θ.
Realness is decided once from the term list and the tree, never from the parameter values at which a node is materialized, and never from small or exactly zero imaginary parts of evaluated data.
A circuit that is complex by structure but real at special parameter values (a `PeriodicGrid` mode at a parameter `n_g` that equals 0, a charge-basis SQUID at zero flux) stays complex; using such special values is a recorded later item, not part of plan 3 (Eli: "a small benefit for what I see as a big headache").
A spec literal is structure: a `PeriodicGrid` mode whose `n_g` entry is the number `"0"` has no linear charge term and qualifies (periodic gauge, §20d 2026-10-30; `"0.5"` does not).

**Port character (decided).** Every basis declares, for every port it serves, how complex conjugation acts on its operator: `conj(O_p) = σ_p O_{π(p)}` with a sign `σ_p = ±1` and an involution `π` on ports.
A port is *real* (`σ = +1`, `π(p) = p`), *imaginary* (`σ = −1`, `π(p) = p`), or *paired* (`σ = +1`, `π(p) ≠ p`, its conjugate partner); anything else is *general* and has no relation.
Section 20c's three characters (real, imaginary, general) gain *paired*, because a single `expiphi(k)` on a grid is complex while the cosine it belongs to is real.

| Basis | real | imaginary | paired (`p ↦ π(p)`) |
|---|---|---|---|
| `ChargeBasis` | `n`, `n2`, `expiphi(k)` | | `exp2pin(k) ↦ exp2pin(−k)` |
| `PhaseGrid` | `n2`, `phi`, `phi2`, `exp2pin(k)` | `n` | `expiphi(k) ↦ expiphi(−k)` |
| `PeriodicGrid` | `n2` | `n` | `expiphi(k) ↦ expiphi(−k)` |
| `FockBasis` | `phi`, `phi2`, `n2`, `exp2pin(k)` | `n` | `expiphi(k) ↦ expiphi(−k)` |
| `GivenSystem` | an operator whose data is real | one whose data is purely imaginary | (none) |

A `GivenSystem`'s operators are fixed data with no θ dependence, so their character is read from the data exactly; an operator with both parts nonzero is general.
A test pins every declared character against the operator data exactly, for every basis and port.
**(my call)** For this the Fock displacement matrix computes its phases exactly (`iᵃ` and `(−1)ᵃ` for imaginary and real `α`) instead of through `cis(a·angle(α))`, which today leaves imaginary residues near 1e-15 in `exp2pin` and breaks `conj(expiphi(k)) == expiphi(−k)` at the same level; the change moves Fock matrix elements by round-off only.
Building a real operator still takes only the components a port's character declares nonzero.

**The node test (decided).** Group a node's routed terms by their factor key (the sorted `(mode, port)` list) and sum the coefficient functions in each group, `C_K(θ)`; the root's constant is the group of the empty key.
Conjugation maps key `K` to `π(K)` (each port to its partner) with sign `σ_K = Π σ_p`.
The node is real when every factor has a real, imaginary or paired character and, for every key, `C_{π(K)}(θ) = σ_K conj(C_K(θ))` holds as an identity in θ.
A node is real only when every node below it is real (decided in the brainstorm; the builder check showed that a real node over a complex child would need complex phases and would mislabel the characters of the ports it passes up).
A real node child's ports carry the characters of the same ports in the leaf basis of their mode.
**Real inputs (decided, Eli 2026-10-18).** All system parameters are real: an entry of θ that is not of a `Real` Julia type raises an `ArgumentError` naming it (a complex drive amplitude is passed as two real parameters).
The spec's charging and inductive matrices (`E_C`, `E_L`), `flux_linear`, `n_g`, and every cosine's and charge cosine's `energy` and `offset`, and `constant`, are real-valued expressions: the loader rejects any that uses `im`, `cis` or `cispi`; the multipliers `omega` and `kappa` (from the incidence structure) are numbers already.
Complex structure in a circuit therefore enters only through the package's own phase factor (`phase_factor`) and through the bases' operators.
**(my call, from the plan-3 Task 3 review)** The three-point test proves an identity for every θ only for coefficients that are real-analytic in θ, so a coefficient that calls `abs` or `sign`, or applies `sqrt`, `log` or a non-integer power to a possibly complex-valued argument (possible in user-built terms), fails the test and its node is complex.
**(my call)** The identity is established by evaluating both sides at three fixed pseudo-random parameter points (each parameter drawn in [0.5, 2] from a fixed seed), never at the user's θ, with agreement required to 1e-12 relative to the larger modulus; a coefficient that throws at a test point fails the test (complex, safe).
This is a test of an identity between functions, so its answer is the same at every θ.
It finds the cross-cosine pairing that makes the symmetric zero-pi real: the conjugate of the `(θ: expiphi(1), φ: expiphi(1))` term, coefficient `−½E_J e^{−iφ_ext/2}`, is the `(θ: expiphi(1), φ: expiphi(−1))` term of the other cosine, coefficient `−½E_J e^{+iφ_ext/2}`.

Consequences on the fixtures: the transmon, fluxonium, three-transmon and symmetric zero-pi leaves and the two-zero-pi parent are real at every flux and offset charge; the asymmetric zero-pi (`dE_J`, `dC_J`, `dC` terms with `sin θ sin φ` and `n_θ n_φ` structure) is complex; a `PeriodicGrid` mode whose `n_g` is a parameter is complex.

**The `arithmetic` option (my call).** `Node(...; arithmetic=:auto)` follows the test.
`:complex` forces complex arithmetic for that node (and so for every ancestor whose terms touch it), for comparisons against the complex path and for later autodiff checks.
`:real` asserts the test and raises an `ArgumentError` naming the first term whose conjugate image is missing.
On `:auto`, a node that tests complex logs that term at debug level.
An accessor reports each materialized node's arithmetic (name left to the builder).

**The real operator (decided in outline; details the builder's).** A real node's `KronFreeOperator` is built from real data only.
Since `H = Re H` by structure, each term contributes `Re(c ⊗_a O_a)`: with `O_a = R_a + i I_a` split into real structured operators of the same type (an imaginary factor has `R_a = 0`, a real one `I_a = 0`), the product expands into the real terms `Re(c · i^{|S|}) ⊗_{a∈S} I_a ⊗_{a∉S} R_a` over the subsets `S` of the factors with a nonzero imaginary part, dropping zero coefficients.
A conjugate pair produces the same real terms twice, so pairs are merged (`2 Re(c P)` once); the imaginary parts of evaluated coefficients are never consulted beyond this expansion.
Diagonal terms fuse into the real `diag` as today.
The local operators become element-type parametric where they are not already: `DenseOp{T}` holds a real or complex matrix (Section 7.2's "always complex" stays true for complex nodes, which still convert every factor to `ComplexF64`, since BLAS has no mixed real–complex gemm), and `AxisTerm` keeps its coefficient's type.
A `KronFreeOperator`'s element type is the promotion of its coefficients' and factors' element types, so an operator built by hand from real data is real (a rule on Julia types, not values).

**Outputs (decided: native types).** A real node stores `Float64` eigenvectors, energies and peak amplitudes, and `eigenvectors(sys, node)` returns them as stored.
Ports are returned in their natural type: a real-character port of a real node is a `Matrix{Float64}`; an imaginary or paired port (a grid charge, `expiphi(k)` on a grid) is a `Matrix{ComplexF64}`, because its matrix in the real eigenbasis is purely imaginary or general.
Internally an imaginary port's matrix is formed from its real component and the parent uses that component directly; the stored complex matrix's real and imaginary parts are the components exactly, since they are computed separately.
Port formation on a real node runs in real arithmetic (`Vᵀ R V` and `Vᵀ I V`).
Complex nodes are unchanged: `ComplexF64` everywhere.
The README documents the types.

**Phases, selection and following (my call).** On real vectors both phase conventions reduce to signs: `:positive_sector` makes the peak entry positive (as before, since a real vector's peak angle is 0 or π), the branch rule multiplies by the sign of the real joint-peak amplitude, and `:follow` by the sign of the real overlap, so a real node never acquires complex vectors.
Selection contexts, the `follow` reference and the transported previous vectors accept either element type and promote in overlaps.
A warm start whose element type differs from the operator's (only possible when `arithmetic` or the term list changed between calls) is converted when real into complex and dropped (cold start) when complex into real.

**Solvers (decided).** Every solver runs in the operator's element type and returns eigenvectors of that type; the `Solver` contract of Section 20e decision 6 is amended accordingly (`V` has `eltype(H)`; a user solver must accept real operators).
`Dense` diagonalizes a real symmetric matrix; `Lanczos` starts from a random vector of the operator's type; `LOBPCG` and `ChebyshevFilter` are already type-generic and run real on the host and on the device.
`ShiftInvert` follows the operator's element type only: its assembled-matrix realness check (`isreal_assembled`, value-based) is retired from the solver and from the exports (Eli, 2026-10-18), and tests that used it check the operator's element type instead.
The CUDA extension needs no new device code beyond confirming the real path (cuTENSOR and cuBLAS real contractions, the expectation kernel already generic).

**What stays complex.** Complex nodes; imaginary and paired ports in their natural type; `charge_port` on a `PeriodicGrid` mode; derivative operators and the autodiff pullbacks of plan 4, which promote to complex where a cotangent or a derivative operator is complex (amended 2026-11-11: there are no autodiff pullbacks; the derivative operators of Section 20l are built in the node's own arithmetic, and the outer rule's tangents take the promoted element type of a port and its derivative).

**Validation (decided).** For every fixture with a real node, materializing with `arithmetic=:auto` and with `arithmetic=:complex` gives energies equal to about 1e-12 with `Dense` and to the solver tolerance otherwise, ports equal up to the per-state signs (inside an exactly degenerate level, where the two paths may pick different bases, the comparison uses rotation-invariant quantities such as the cluster projector and the port's block norms), and the expected arithmetic per node (the classification above is a test).
Tests that relied on `isreal_assembled` change meaning: the `PeriodicGrid` transmon at a parameter `n_g` of 0 is now complex by structure, so those assertions become element-type checks (the exact-zero imaginary part of its assembled matrix at `n_g = 0` can still be asserted as a property of the operator data; at ½ it is nonzero in the periodic gauge, §20d 2026-10-30).
A solver change is reviewed with adversarial completeness probes compared against `Dense` (degenerate levels, invariant-seed traps, near crossings), not residual checks alone.
The default, gated (`KRONFREE_SLOW_TESTS=1`) and device suites pass, and the plan-2b end-to-end benchmark reruns on ewdesktop in both arithmetics (a new `BENCH_ARITHMETIC=auto|complex` knob).

**Later items.** Real arithmetic at special parameter values, or for parameters declared constant by a substitution helper; mixed real–complex trees where a real child feeds a complex parent without conversion.

**Left to the builder.** Names (the character type, the accessor, internal helpers); where the characters are computed (once per `materialize` after routing is natural); the merge of conjugate pairs; the storage of a real node's `MaterializedNode` fields (parametric or abstractly typed); the test-point draw.

**Execution notes (plan 3, 2026-10-19).**
The analyticity gate came from the Task 3 review, whose probes found false-real classifications with coefficients that call `abs` or `sign` and with complex branch cuts of `sqrt`, `log` and non-integer powers; the gate is a conservative syntactic check in `classify`, and it over-rejects the analytic `(E + im*x)^0.5` (pinned by a test).
Eli's real-inputs rule is enforced in two places: `materialize` and `classify` raise an `ArgumentError` for a non-`Real` entry of θ (`_check_real_parameters`), and the spec loader rejects `im`, `cis` and `cispi` in the listed expressions.
The merge of conjugate pairs in the real operator is keyed on (axis, mode, canonical port, component); the plan's key omitted the mode, so in the Task 4 review a multi-mode node child collapsed couplings from different modes onto one operator (0.1 GHz errors on a three-transmon chain), and a regression test pins it.
`lobpcg_solve` converts a complex `X0` whose imaginary parts are all exactly zero to the real operator's type and raises an `ArgumentError` for any other complex `X0`, rather than silently dropping imaginary parts or restarting at random; `materialize` already drops such a block itself.
`ShiftInvert(block=2)` intermittently raises a KrylovKit block-Lanczos breakdown ("blocklength must be >(0)") on small operators with few distinct levels, in both arithmetics; this predates plan 3 and is an open item (the failure is loud, and a deflation fallback is the likely fix).
On exchange-symmetric trees (the symmetric two-zero-pi) the branch peak argmax ties between exchange-symmetric components, so the global signs of some root states differ between arithmetics; energies and port magnitudes agree, and the tests compare rotation-invariant quantities there (cluster block norms, the physical overlap).
Measured gains (`docs/spikes/2026-10-18-plan3-benchmark/results.md`): the two-zero-pi tree takes 9.9 s on CPU and 5.0 s on the A4500 in real arithmetic, against 22.3 s and 13.2 s in complex arithmetic of the same plan (2.3× and 2.6×) and 15.4 s and 12.6 s in plan 2b (1.55× and 2.5×); the parent solve gains 2.5× on CPU and 3.3× on the A4500, port formation about 4×, and the `ShiftInvert` leaf on CPU nothing against plan 2b.
Default-suite cost: the default suite takes 178 s against 139 s for the base tree `b92bce0` (two alternating runs each, 178.1 and 178.5 s against 138.4 and 138.8 s, at load about 4–7; +40 s against the +15 s budget).
A second pass of the same files in one process takes 50.4 s against 50.6 s, so the whole excess is compilation of the `Float64` methods (the operator, the solvers, port formation and the ITensor paths) in addition to the complex ones, not steady-state cost.
The two slowest new real-against-complex equivalence testsets (the two-zero-pi real branch, about 6 s, and the solver element-type equivalence, about 3 s) moved to the gated suite, which keeps them; moving them saved only about 3 s of wall time because their compilation then falls to the next test that exercises those methods.

### 20g. Execution notes (plan 4, 2026-10-19)

The spec exporter lives in `circuit-quantization` (`circuit_quantization.io`: `to_spec`, `save_spec`, `load_spec`); the Julia side adds provenance tests.

**Parser.** The `Coefficient` allowlist accepts SymPy's dotted operators (`.+ .- .* ./ .^`), normalized at parse time, as well as two-argument `log`, scientific literals and `ℯ`.
`Meta.parse` turns an integer literal beyond `typemax(Int64)` (SI-valued exports, loop fluxes in webers) into an `@int128_str` or `@big_str` macrocall; the parser folds exactly these, with a signed decimal string, into the `Int128` or `BigInt` value before the allowlist check, and every other macrocall stays rejected.
A declared parameter name is normalized as a Julia identifier (`Meta.parse`), so `µ` (U+00B5) matches its use as `μ` (U+03BC).
Nonsymmetric `E_C` and `E_L` matrices are rejected (non-physical; Eli, 2026-10-19): in `parse_spec`, identical entries `[i,j]` and `[j,i]` pass, a pair with an entry that fails the real-analyticity gate of Section 20f is rejected (agreement at test points cannot prove it equal), and an analytic pair must agree at the three test points; the `ArgumentError` names the matrix, the pair and both expressions.

**Provenance test.** `test/fixtures/generate_specs.py` builds each hand fixture's Hamiltonian as a lumped circuit, exports it with `to_spec`, and writes `test/fixtures/generated/*.json`; `test/test_generated_specs.jl` shows that each pair defines the same Hamiltonian (coefficients at random real points, cosines up to the joint sign flip of `omega` and offset, energies on a small Dense tree).
The cos2phi fixture has no generated counterpart, since it is not a lumped circuit (junctions with phases 2φ and φ have non-parallel incidence rows).
Python is not a test dependency: the generated files are committed.
The test asserts these convention differences:
- fluxonium: the hand spec keeps the constant `E_L varphi_ext²/2`, which circuit-quantization omits;
- zero-pi: the constant `2E_J` (scqubits' potential), and a decoupled ζ mode in the generated spec that scqubits' ZeroPi drops;
- full zero-pi: the constant `2E_J − sqrt(2 E_Cz E_L)`, and 4 cosines in the hand spec against 2 in the generated one;
- two zero-pi: the constant `2E_J1 + 2E_J2` and two decoupled ζ modes.

**full_zero_pi finding.** scqubits' FullZeroPi charging matrix is a first-order model in the capacitive asymmetries, and no circuit with nonnegative capacitances reproduces it for any nonzero (dC_J, dC), since an exact realization needs a negative capacitance in a b→c or d→a branch.
The two specs agree exactly at zero capacitive asymmetry; at nonzero asymmetry the test asserts the exact second-order coefficients of the difference in `E_C`, and the second-order difference of the θ offset charge (`n_g/D` in the hand spec against the circuit's `n_g`, coefficient `−n_g (E_CS a² + (E_CJ − E_CS) b²)/(4E_CJ)` along (dC_J, dC) = ε (a, b)).
At an asymmetric point the test also asserts that `E_L`, `flux_linear` and the merged cosines are equal, that both `E_C` matrices are symmetric, that the hand spec's offset vector ν solves `E_C ν = (E_CS n_g/2, 0, 0)`, and that the constants differ by exactly `2E_CS n_g² (1 − 1/D)` in addition to the symmetric-point conventions.

**Sign of dC_J (decided by Eli, 2026-10-19).** scqubits 4.3.1's source sign (`−2 E_CΣ dC_J (i∂φ) n_θ`) contradicts its documented definitions of dE_J and dC_J, which refer to the same junction; the docstring's `+2E_CΣ dC_J ∂φ∂θ` is the consistent one.
With the source sign, the junction with the larger E_J has the smaller C_J when dE_J and dC_J have the same sign.
The effect is observable, about 0.02 GHz at dE_J = dC_J = 0.3, and vanishes at dE_J = 0.
The fixtures keep the source sign.
Decision (Eli, 2026-10-19): `full_zero_pi.json` keeps scqubits 4.3.1's dC_J sign, which is opposite to Groszkowski et al., arXiv:1708.02886v1, eq. `eq:Hzeropi1` (`H_0π = H_sym + 2E_CΣ dC_J ∂φ∂θ + E_J dE_J sin θ sin(φ − φ_ext/2)`, `dX = (X1 − X2)/X`), so dC_J(fixture) = −dC_J(paper, `circuit-quantization`); dE_J agrees.

**Performance workaround (removed).** `to_spec(res, subs=...)` did not finish on two_zero_pi (over 10 minutes), because `sympy.cancel` stalled on the substituted expression, while `sympy.cancel(sympy.together(expr))` takes 0.05 s.
The generator applied `together` to every expression before `to_spec(res)` for that fixture only.
`circuit-quantization` commit 58ee6ef applies `together` before `cancel` in `to_spec` (0.3 s on two_zero_pi), so the generator now calls `to_spec(res, subs=..., mode_names=...)` for every fixture; the regenerated files differ from the previous ones only in `_generator.commit` and in two_zero_pi's `_comment`.

### 20h. Born–Oppenheimer reduction (2026-10-20)

The Born–Oppenheimer workflow of Section 13.1 is designed in `docs/superpowers/specs/2026-10-19-born-oppenheimer-design.md` and implemented in `src/spectransform.jl` (`substitute`, `promote_mode`) and `src/bo.jl` (`bo_surface`, `fourier_coefficients`, `slow_spec`, `fit_surfaces`).
Promotion is spec-level: `promote_mode` turns a mode of a `Spec` into a parameter and returns the fast `Spec`, so the term list, `materialize` and the tree are unchanged; a charge coupling between the slow and fast modes is rejected by expression, never by value.
The surface reaches the slow stage only as a reduced one-mode spec (a Fourier `slow_spec` or a linear least-squares `fit_surfaces`), written with `save_spec`; the data-valued `DiagonalOp` entry into a `Term` (Section 20a) is still deferred to plan 5 (amended 2026-11-11: plan 5 was replaced by Section 20l, which does not wrap `materialize`; the entry is now an open item of its own, STATUS.md).
Acceptance: the gated `test/test_born_oppenheimer_reference.jl` reproduces Di Paolo et al.'s E_α and E_γ (arXiv:1811.07921v1, Appendix B) to 0.2% with E_Cφ derived from the plasma-frequency identity (decided 2026-10-20; `bench/born_oppenheimer/results.md`).

### 20i. Composite ports (2026-10-26)

A term routed to a branch with several factors inside one child is now applied through a joint port of that child, the operator `V† (⊗ₐ Oₐ) V` of those factors, formed in the child's product basis before its projection onto the kept states.
A product of separately projected single ports, `(V† A V)(V† B V)`, inserts the projector between the factors and drops excursions through discarded states; it is not an approximation of the joint port at a truncated child.
The requirements and the motivating Z4 three-way hierarchy are in `docs/superpowers/specs/2026-10-26-composite-ports-requirements.md`; the plan is `docs/superpowers/plans/2026-10-26-composite-ports.md`.
Decided 2026-10-26.

- **Key.** A joint port is keyed by the sorted list of its `(mode, port)` factors (`_FactorKey`, `src/terms.jl`), stored in `MaterializedNode.joint_ports`; every key has two or more factors, and single-factor ports stay in `ports`.
  A term's factors inside a node form one key, because all of them lie inside the one child of the ancestor that holds the term; only the node that forms the port splits the key by its own axes (`_axis_groups`, `src/tree.jl`).
- **Request.** `required_joint_ports` (`src/tree.jl`) lists, for each node, the keys of terms routed to strict ancestors that have two or more factors inside the node.
  `required_ports` no longer requests single ports for factors that belong to such a group.
- **Forming.** `_joint_port` (`src/materialize.jl`) takes each child's operator for its group of the key (`_axis_ops`: a basis child's local operator, a node child's single port, or a node child's own joint port when the group has several factors, so joint ports nest), applies them in increasing axis order, and contracts with `V'` once (`_port_matrix`).
  Forming needs two scratch blocks of `prod(dims) × kept`, and the stored result is a dense `kept × kept` matrix like every port.
  The same `_axis_ops` feeds `_node_operator`, so a node's operator applies a joint port as one factor on its axis.
- **Character.** `_key_character` (`src/character.jl`) gives a key the character `:general` if any factor has it, else `:paired` if any factor does, else `:real` or `:imaginary` by the product of the factors' signs σ.
  A paired key's partner is the sorted key of the factors' partners; the σ sign of the non-paired factors is not carried in the partner, and any future consumer of the partner must recompute it.
- **Real path.** On a real node a joint port is formed from real components by the subset expansion of Section 20f (`_real_expansion`, coefficient 1 for the real part and −i for the imaginary part, on each group's own character), so a real node never forms a complex product-basis block.
  A real joint port is `Matrix{Float64}`; an imaginary, paired or general one is `Matrix{ComplexF64}`, and `_check_eltype` asserts this.
  At the ancestor a joint factor enters the real operator with its own stored components, and conjugate joint pairs are not merged: each term's `Re(c O)` expansion is a correct real operator by itself, and merging would need the partner's sign.
  This costs a few extra axis terms at the root, judged negligible.
- **Which nodes are real.** A cosine mixing a charge-basis mode with a grid or Fock mode has no conjugate image in the term list, so its node is complex; a cosine over charge-basis modes only, or over paired bases only (`PhaseGrid`, `PeriodicGrid`, `FockBasis`), is real.
  The three- and four-charge-mode test trees are real, and the mixed `ChargeBasis`/`PhaseGrid`/`FockBasis` tree is complex.
- **Still rejected.** Two factors on the same mode in one term (`_check_duplicate_modes`: "has several factors on mode"), since nothing forms the product of two operators of one mode.
- **Not implemented (a later item, for the matrix-element query work).** User-requested joint ports in `Node(ports=...)` and in a selection rule's `ports_needed` stay single-mode keys.
  `port(sys, key, node)` reads a joint port that a routed term caused to be formed, in any order of its factors, and throws otherwise.
  The adjoint of the `s = +1` joint port is not reused for the `s = −1` term of a cosine; both are formed, which is cheap.
- **Tests.** `test/test_composite.jl` (fast tier: the multi-axis sandwich, forming on real and complex leaves, the truncated-child test where separate projection demonstrably fails, real versus complex agreement); `test/test_composite_slow.jl` (slow tier: mixed bases at both signs and at π and generic offsets, a `PeriodicGrid` real-path case against its complex twin, nesting through a grandchild); and a solver-agreement testset in `test/test_solvers.jl`.
  The last compares `Dense` with `LOBPCG` on a degenerate spectrum, and all three solvers on a copy whose offset charges lift the degeneracy, because single-vector `Lanczos` does not resolve multiplets (see the `Node` docstring in `src/tree.jl` and the `Lanczos` docstring).
- **Z4 pilot.** The hierarchy `[[1,4,5],[2,6,7],[3]]` is now accepted.
  The rerun of the pilot with the root-operator Hermiticity check, and the convergence sweep (retentions, then primitive cutoffs), are pending on ewdesktop (Eli).

### 20j. The circuit-quantization interface (2026-11-10)

One Julia session can now define a circuit with circuit-quantization, choose its coordinates, build a `Spec` in memory, solve it, and write a single JSON run record, with no file passed between a Python and a Julia session.
circuit-quantization stays a Python package and does all the quantization; Julia calls its Python API through PythonCall.
The design is `docs/superpowers/specs/2026-11-10-circuit-quantization-interface-design.md` and the plan `docs/superpowers/plans/2026-11-10-circuit-quantization-interface.md`.
This amends Sections 2 (anti-goals), 15 (packages) and 20b.

**What it supersedes.**
Section 20b said: "the JSON spec remains the Python–Julia contract (the topological, parameter-independent content from Python; parameter-dependent coefficients compiled in Julia); a PythonCall extension for in-memory spec transfer is an optional convenience, never a dependency; porting the symbolic quantization machinery to Julia is rejected unless parameter-dependent coordinate choices become first-class, in which case only that numeric step moves."
Four statements replace it.
The kronfree-spec version 1 format remains the contract, and every route through the extension produces either that document or the `Spec` that `parse_spec` builds from it.
The extension does more than in-memory transfer, since it also changes coordinate frames numerically inside a loop and writes and reads run records.
It is still optional and never a dependency of the core.
The numeric frame change, which a frame optimization now needs on every iteration, stays in circuit-quantization (NumPy) instead of moving to Julia.
Nothing of the symbolic machinery is ported.

**Where it lives.**
The core declares the interface without any Python: `src/pyinterface.jl` holds the function stubs `circuit_quantization`, `record_json`, `save_record` and `load_record`, the struct `QuantizedCircuit{H}` (fields `handle`, `values`, `mode_names`), and an error hint, so that calling a stub without the extension says "load PythonCall to use the circuit-quantization interface".
The methods live in the package extension `KronFreeCircuitsPythonCallExt` (`ext/KronFreeCircuitsPythonCallExt.jl`), which Julia loads when `PythonCall` (a weak dependency, compat 0.9.29) is loaded beside the package, as the CUDA extension is.
The Python interpreter is the user's own environment with circuit-quantization installed; PythonCall is pointed at it with `JULIA_CONDAPKG_BACKEND=Null` and `JULIA_PYTHONCALL_EXE=<that interpreter>`.
The extension installs nothing.
When the module `circuit_quantization` is not found, the first call fails with a message naming the interpreter in use and the two variables; any other import failure (a missing dependency, an error inside circuit-quantization) fails with Python's own message.
A default-tier test (`test/test_helpers.jl`, "the core never mentions Python") checks that no file under `src/` contains `using PythonCall`, `import PythonCall`, `PythonCall.` or `pyimport`, and that the extension is not loaded in the default tier.
The default tier's environment (`test/Project.toml`) does not list PythonCall, so it runs without it.

**The two routes to a spec in a new frame.**
A frame is the secondary coordinate transformation `φ = S φ′` over the retained modes (the paper's residual transformation).
`S` is admissible when it is invertible, mixes no noncompact mode into a compact one (zeros below the compact block), and has an integer, unimodular compact block.
Both routes check the size and finiteness of `S` in Julia first, before any Python call, and leave admissibility to circuit-quantization, turning its `ValueError` into an `ArgumentError` that quotes its message and says that the matrix indices in it are circuit-quantization's, counted from 0.
The symbolic route also rejects in Julia, before any Python call, a compact-block entry that is not within `1e-12` of an integer, with an `ArgumentError` that names the entry by Julia's indices, counted from 1.
Both retype the modes in the new frame by circuit-quantization's structural rule.

- *The symbolic route*, `Spec(res; frame=S, subs, simplify, mode_names, zero_rtol, zero_atol)` for a circuit-quantization `Result` `res`, keeps the parameters symbolic.
  It applies `res.transform` to an exact copy of `S` (`to_sympy_frame`), then circuit-quantization's `to_spec`, `json.dumps` and `parse_spec`, so it is by construction the `Spec` that `load_spec` reads from the file that `save_spec` writes.
  Without `frame` it is the spec of `res` in its own coordinates.
  The frame reaches SymPy as exact integers in the compact block (each entry within `1e-12` of an integer) and as the exact rational of each entry's shortest round-trip decimal string elsewhere.
  Passing floats straight through is not only slow in general (Section 3 of the design measured 60 s on a six-mode circuit) but fails: round-off leaves the KITE's `E_L` at its optimum frame asymmetric by about `2e-16`, and the exporter's symbolic symmetry check rejects it.
  `subs` maps parameter names to numbers, to other names or to SymPy expressions, matched to the circuit's own SymPy symbols (symbol identity includes assumptions, so a fresh symbol of the same name would silently not match); only the symbols that the exported spec can contain are accepted (for the flux offsets `varphi_b`, those of the inductive branches and junctions that `to_spec` exports), so a name that would substitute nothing is an `ArgumentError` listing the accepted names.
- *The loop route* evaluates the circuit once and then changes frames numerically.
  `QuantizedCircuit(res; values)` takes a value for every parameter of `Spec(res)` by name, in the frame of `res` (a `NamedTuple`, or a `Dict` keyed by `Symbol`s or strings), and holds circuit-quantization's `numeric_hamiltonian` of `to_spec(res)` in that base frame, with the values stored in the order of `Spec(res).parameters`.
  The same validator checks the `values` of `record_json` and `save_record`.
  A `values` of another type, a key that is not a name, a name given twice, or a value that is not a finite real number is an `ArgumentError` naming the problem; an unknown name or a parameter left without a value is an `ArgumentError` that also lists the circuit's parameters, in the order of `Spec(res).parameters`.
  `Spec(qc)` is its spec of constants in that frame, and `Spec(qc, S)` the spec in the frame `S`, from the arrays of circuit-quantization's NumPy frame change `transformed(S)` through `numeric_spec`, with no symbolic work and no expression strings.
  The modes keep the names of `Spec(res)`; the numeric Hamiltonian carries no names, so `QuantizedCircuit` stores them (`mode_names`).

**The offset-charge convention.**
The offset charge has two parts in circuit-quantization: a source part from gate voltages, on every mode, and a static sector label on each compact mode.
In a new frame the source part transforms as `Sᵀ`, the sector labels live on the compact modes only and transform with `S_ccᵀ`, and their components on noncompact modes are dropped, because a static offset on a noncompact mode is removable by a gauge transformation.
The symbolic route issues a fresh label `n_g{j}` for each compact mode `j` of the new frame; the loop route carries the two parts separately and applies the same rule.
The fresh label reuses the name of the base frame's label but is a different parameter: the value that describes the same circuit is `S_ccᵀ n_g`, with `n_g` the base frame's sector values, not the base label's own value.
`values`, given to `QuantizedCircuit` or to `record_json`, are always in the base frame, and the record's `_values` block gives the fresh labels of the framed `Result` the transformed sector offsets that circuit-quantization's numeric frame change computes (its `n_g_sector`).
With a nonzero offset charge on the KITE at its optimum frame, the loop-route spec equals the symbolic-route spec with the values substituted (`test/python/test_final_route.jl`).
On the two-transmon fixture, whose two compact modes allow `S_cc = [1 1; 0 1]` (where `S_ccᵀ`, `S_cc⁻¹` and `S_cc` differ), the loop route's offset charges are `Sᵀ n_g`, the symbolic route agrees when its fresh labels take `S_ccᵀ n_g`, and the six lowest levels in the new frame equal the base frame's to 1e-11 GHz on a full charge basis (`test/python/test_loop_route.jl`); a run record of that run carries the solved spec, and circuit-quantization's `load_record` of it rebuilds a circuit with the same numbers (`test/python/test_save_record.jl`).

**Constant coefficients never compile (core).**
A numeric spec is built on every iteration of a frame loop, so a coefficient that compiled a function would cost milliseconds per coefficient and grow the generated-function cache without bound.
A constant `Coefficient` (from `Coefficient(x::Number)`, or from an expression string that parses to a number) therefore carries a `_Constant` (`src/parameters.jl`) as its function and compiles nothing, and arithmetic among constants (`*`, `+`, unary `-`, a number times a constant, `phase_factor`) folds to a constant.
A folded constant may be complex (a phase factor), and both `expr_string` and the real-arithmetic classification of Section 20f accept it, so a numeric spec keeps the arithmetic class of the symbolic spec it was evaluated from (`test/test_terms.jl`).

**`numeric_spec` (core).**
`numeric_spec(; modes, E_C, E_L, flux_linear, n_g, cosines, constant, symmetry_tol, zero_rtol, zero_atol)` (`src/spec.jl`) builds a `Spec` with no parameters from plain numeric arrays, the same `Spec` that `parse_spec` builds from a JSON document of the same numbers, with every coefficient a constant and an exact zero a literal zero, as `"0"` is in a JSON spec.
`E_C` and `E_L` are symmetrized at round-off level: with `scale` the largest magnitude in the matrix, a pair differing by at most `symmetry_tol * scale` (default `1e-10`) becomes its mean, and a larger asymmetry is an `ArgumentError` naming the pair.
An `omega` entry on a compact mode within `1e-12` of an integer is rounded to it, and any other is rejected.
The loop route calls it on the arrays that come back from Python.

**Round-off zeroing is an explicit option.**
A circuit quantized in a floating-point coordinate basis exports entries that should be zero as round-off numbers (in the three-fluxonium bench, `E_C` and `E_L` entries around `1e-14` to `1e-11` at matrix scales of 1 to 10 GHz, and cosine `omega` entries around `1e-17`).
The package treats every nonzero entry as a real term, so such entries change which modes are coupled.
Eli's decision (2026-11-10): zeroing at a relative tolerance is acceptable but must be explicit; the default is `1e-12` relative, and a bench may pass `1e-9`.
The keywords `zero_rtol = 1e-12` and `zero_atol = 0.0` are taken by `numeric_spec`, by `Spec(qc[, S])` (forwarded to `numeric_spec`), and by `Spec(res; ...)` (forwarded to circuit-quantization's `to_spec`), with the same meaning on both sides; `record_json` and `save_record` record them with the spec and forward them to `to_record`.
After symmetrization, an entry of a fully numeric `E_C` or `E_L` (diagonal included) of magnitude at most `max(zero_rtol * scale, zero_atol)` becomes exactly zero, where `scale` is the largest magnitude in that matrix.
In each cosine, an `omega` entry of magnitude at most `zero_rtol` times that cosine's largest `omega` magnitude becomes exactly zero, on a compact mode at most `zero_rtol` times the smaller of that magnitude and 1 (relative only: `zero_atol` does not apply to `omega`).
`zero_rtol = 0, zero_atol = 0` turns zeroing off.
A negative or non-finite tolerance is an `ArgumentError` naming the keyword.
This replaces a hard-coded rule (pairs below `1e-15` of the scale, off-diagonal only) that both sides had used before.
The two sides still differ in three ways: the exporter zeroes only floating-point residue and skips a matrix with a symbolic entry, where `numeric_spec` zeroes any entry within the threshold; the exporter takes the scale before symmetrization, `numeric_spec` after it; and the exporter requires an exact integer for a compact `omega` entry, where `numeric_spec` rounds one within `1e-12` of an integer.
Zeroing a diagonal entry can in principle make a positive-definite matrix indefinite when its scales are separated by more than `1/zero_rtol`; no circuit here comes near that.

**Run records.**
`save_record(path, spec; circuit=res, frame, values, metadata, zero_rtol, zero_atol, check=true)` writes the record of a run that solved `spec`, and `record_json(spec; ...)` returns its text, as `save_spec` and `spec_json` do for a spec.
The text is at once a kronfree-spec version 1 document and a circuit-quantization record (its `load_record` rebuilds the circuit).
Its spec fields are those of `spec`, written by `spec_json`; nothing in them is recomputed from the circuit, the frame or the values, so `parse_spec` of the record gives `spec` back coefficient for coefficient, exact zeros included (`spec_json` prints each number by its shortest round-trip decimal string).
The provenance blocks come from circuit-quantization's `to_record` of `res` in the run's frame: `_circuit` (branches, ground, loops, flux assignment and the SymPy symbols with their assumptions), `_frame` (`Z_full`, the full mode types and `S`), `_values` (the parameter values, when given) and `_generator`; the spec fields that `to_record` also writes are dropped.
`values` are in the frame of `res`, exactly as `QuantizedCircuit(res; values)` takes them; in the framed record `_values` gives each fresh sector label the transformed sector offset `S_ccᵀ n_g` and every other parameter its given value.
`zero_rtol` and `zero_atol` are provenance: the tolerances the spec was built with, forwarded to `to_record` and recorded so that the run can be rebuilt.
Julia adds three keys: `_generator.kronfree` (this package's version and git commit, `null` when not available), `_metadata` (the caller's `metadata`, which must be JSON-serializable), and the Julia-owned `_kronfree`, which holds the frame of `res` as it was passed (`base_frame`, in the encoding of `_frame`), the `frame` argument, the base-frame `values` in the order of `Spec(res).parameters`, and `zero_rtol` and `zero_atol`.
With `check = true` (the default) the spec is first rebuilt from the provenance and compared with `spec`: by the loop route, `Spec(QuantizedCircuit(res; values), frame; zero_rtol, zero_atol)`, when there are values, and as `Spec(res; frame, zero_rtol, zero_atol)` when there are none.
The comparison covers the parameters, the mode names and types, the cosine vectors and every coefficient (at a fixed sample point of the parameters when there are any), each number to a relative `1e-9` of a scale: the largest magnitude of the numbers compared together, in either spec, but at least a floor.
An entry of `E_C` or `E_L` is compared against its matrix with no floor, an entry of `flux_linear` against its vector with the spec's largest energy (the largest magnitude among `E_C`, `E_L` and the cosine energies) as floor, an entry of `n_g`, an omega or a kappa entry against its vector with floor 1 (they are dimensionless), a cosine's energy against the two energies with no floor, its offset with floor 1, and the constant with the largest energy as floor.
So a spec zeroed at a different tolerance than the one recorded still passes when the entries involved lie below `1e-9` of their scale, and a vector of round-off passes against exact zeros.
A mismatch is an `ArgumentError` naming the first field that differs, with both values; a coefficient that cannot be evaluated at the sample point is an `ArgumentError` naming it.
A spec with mode names other than the circuit's defaults, or built with `subs` that replaces some parameters, fails the check and needs `check = false`.
`check = false` skips the comparison, for a spec edited by hand or a loop that writes many records.
Whatever `check` is, the assembled document is then validated by circuit-quantization's `load_spec`, which `circuit_quantization.load_record`, and so `load_record`, applies first; a spec it rejects (one with `charge_cosines`, for example) is an `ArgumentError` carrying its message, and no record is returned or written.
`load_record(source)`, from JSON text or a file, returns `(; result, spec, frame, values, zero_rtol, zero_atol)`: `spec` is the spec the run solved, `result` the symbolic `Result` that was passed as `circuit`, rebuilt in its own frame, `frame` the frame that was applied, and `values`, `zero_rtol` and `zero_atol` those of the run.
So `Spec(result; frame, zero_rtol, zero_atol)` is the symbolic spec of the run, and `Spec(QuantizedCircuit(result; values), frame; zero_rtol, zero_atol)` rebuilds its numeric spec.
Nothing in a record is evaluated: numbers are JSON numbers and expressions go through circuit-quantization's restricted parser.
The tree is Julia code and is not recorded; that belongs to the export and caching plan.

*Amendment, 2026-11-10, after the final review.*
The record function was first built as `run_record(res; frame, values, metadata, zero_rtol, zero_atol)`: it never saw the spec that was solved, and wrote circuit-quantization's `to_record` of `res` in the frame, whose spec fields the symbolic route recomputed from `res`, the frame and the values.
A record therefore equalled the run only while the symbolic route and the loop route agreed, and the final review found a case where they did not (offset charges under a frame mixing two compact modes; fixed, with the test of the two-transmon fixture above).
The interface design's own sketch had been to record the solved spec with the circuit, frame and values as provenance, and the owner chose that, with a name to match `save_spec`, `load_spec` and `load_record` ("it's a record of the run, not running a record").
So `run_record` was removed, not deprecated (the branch was unreleased), in favour of `record_json` and `save_record`, which take the solved spec and check that the provenance describes it; `load_record` is unchanged.

**Measurements.**
On ewdesktop (Intel i7-11700K, Julia 1.13.2, one thread), 2026-11-10.

- Loop route, `Spec(qc, S)` for the three-mode KITE at its optimum frame, after warm-up, mean of 1000 calls: 0.134 ms per call, of which circuit-quantization's NumPy frame change is 0.086 ms and the array conversion plus `numeric_spec` 0.030 ms.
  The design's target was 5 ms.
- Symbolic route, `Spec(res; frame=S)`, after one warm-up call: 0.37 s for the KITE at its optimum frame, and 0.80 s for the six-mode coupled zero-pis with a float `S`.
- The KITE in three frames in one session (`bench/kite_frames`; figures from `bench/kite_frames/README.md`, which a later run of the bench may revise): means of 200 iterations of a loop over random admissible frames near the optimum, in the run of 2026-11-10 on commit `d3f338c`, were 0.355 ms for the frame change alone, 0.569 ms for the frame change through the end of `materialize`'s setup (everything before the first eigensolve), and 1194 ms for a full `materialize`, dominated by the eigensolves.
  No coefficient compiled anywhere in the loop: every coefficient was a `_Constant`, the generated-function cache stayed empty, and Julia's cumulative compile time did not increase (the bench fails on compile time between a frame change and the first eigensolve, and warns on it after that).
  The three frames' four lowest energies agree to 7.2e-6 GHz, and the ground energy is −9.523981 GHz.
  They lie about 1.4e-5 GHz above the circuit-quantization paper's reference levels; solving the KITE in Julia on the reference calculation's own grids (charge cutoff 24, 160-point phase grids with steps of about 0.28 to 0.32) reproduced the paper's four levels to within 5e-7 GHz, so the offset is the reference grid's discretization error, not a difference in the Hamiltonian (`bench/kite_frames/README.md`; `run.jl` reproduces this with `KITE_FRAMES_REFGRID=1`).

**Deviations from the plan.**
`QuantizedCircuit` carries a third field, `mode_names`, which the plan did not have: circuit-quantization's numeric Hamiltonian has no mode names, so they are taken once from the spec of the `Result` and kept in every frame.
The KITE bench's root uses the `Auto()` solver, which resolves to LOBPCG there, instead of the `Dense` root of the plan: with 24 to 40 states kept per leaf the root has 24³ to 40³ product states, beyond what `Dense` diagonalizes in reasonable time (the reasoning and the measurements are in `bench/kite_frames/README.md`).
The leaves are `Dense`.

**circuit-quantization.**
The interface needs that repository at commit `c8c8714` or later.
Its commits of 2026-11-10 on `main` (at `9c26fd2`) bring an integer-valued float compact block accepted, the export symmetry check with round-off repair, the numeric frame change with its checks as public API, and run records.
The zero-tolerance keywords of `to_spec` are in `c8c8714`, on its local branch `to-spec-zero-tolerance`, which is not yet merged into `main`.
`Spec(res)` always passes those keywords, and `record_json` and `save_record` forward them, so all three need that commit, and so does the Python tier; `QuantizedCircuit` and `Spec(qc[, S])` do not.
Against an older circuit-quantization, `Spec(res)`, `record_json` and `save_record` throw an `ArgumentError` that names the required commit and the file the module was imported from, decided once from the signature of `to_spec`; there is no fallback, since a fallback would zero by a different rule.

**Tests.**
The default tier stays Python-free.
The Python tier, `test/python/` with its own environment, is never included by `test/runtests.jl`.
It runs as `JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=<python> julia --project=test/python test/python/runtests.jl`.
It covers the module import and its error messages, the error for a circuit-quantization without the zero-tolerance keywords, `Spec(res)` against every committed generated fixture, the symbolic and loop routes and their agreement (including the nonzero offset charge), the offset-charge convention with a two-by-two compact block, inadmissible frames on both routes, and run records written and read back (with two compact modes, with zero tolerances, and once in a fresh Julia session).

### 20k. Mean-field basis augmentation (2026-11-11)

A child of a node can now be solved in the mean field of its siblings, and the mean-field states of several self-consistent solutions can become the basis through which the child is presented to its parent.
The parent's operator is never changed: only the basis the parent sees changes, so nothing is approximated beyond the truncation to that basis.
This is the optimized-coordinate vibrational self-consistent field method of the circuit-quantization paper (its Appendix C), whose prototype is circuit-quantization's `product_state.py`.
The design is `docs/superpowers/specs/2026-11-10-mean-field-augmentation-design.md` and the plan `docs/superpowers/plans/2026-11-10-mean-field-augmentation.md`; the work is on `main` (local; not pushed as of 2026-11-11).

**The pieces.**
They are exported, so that the self-consistent loop and the basis construction are a few readable lines (README, "Mean-field bases").

- *Mean-field terms* (`src/meanfield.jl`).
  `straddling_terms(parent, terms)` are the terms `route` places at `parent`; `child_keys(parent, terms, child)` lists the factor keys those terms need inside `child`, one key per group of factors that sit in the child, so two factors in one child form a joint key whose value is the child's joint expectation, not a product of single ones.
  An `Expectations` table maps each key to a complex number: `expectations(solved, keys; state, weights)` takes it in one kept state of a solved child or averages over the kept states, and `pinned_expectations(keys, displacement)` and `delocalized_expectations(keys)` make starting tables that correspond to no solved state.
  `mean_field_terms(parent, terms, θ, child, others; outside)` reduces each straddling term to the child's own factors, with a constant complex coefficient equal to the term's coefficient at `θ` times the other children's expectations of their keys; terms that reach outside `parent` enter only when `outside` supplies their outside expectation.
- *A mean-field child.*
  `bare_child(parent, terms, θ, child)` solves the child on its own terms, as the root of its own tree, in complex arithmetic, with the ports of `child_keys`.
  `mean_field_child(parent, terms, θ, child, bare, others; ...)` solves the child's own node again, on its bare terms plus the mean-field terms, reusing the children of `bare`, and returns a `MeanFieldChild` (`node`, `solved`, `terms`, `bare_energy`, `expectations`), whose energies are labelled mean-field energies.
  `product_energy(parent, terms, θ, bare_energies, tables; constant)` is the expectation of the parent's operator in the product state, with no double counting.
- *The self-consistent loop.*
  `self_consistent(parent, terms, θ, bare; start, tol=1e-11, maxiter=300, mixing=0.5, ...)` updates all children together from the previous iteration's tables, damps with `mixing` after the energy has risen twice, and converges when the relative energy change is below `tol` and no table entry changed by `sqrt(tol)` or more.
  It returns a `MeanFieldSolution` (`energy`, `children`, `iterations`, `converged`, `start`); a run that reaches `maxiter` is returned with `converged = false`.
  `mean_field_solutions(parent, terms_or_spec, θ; starts=default_starts(parent), distinct=1e-3, bare=nothing, ...)` runs the loop from every start, with the bare solves formed once (or taken from `bare`), and returns the distinct solutions, lowest product energy first.
  The solutions' states are coordinates on the children of those bare solves, so a caller who goes on to `augment` passes `bare` and gives `augment` the same solves.
  `default_starts(parent)` is the delocalized start (`nothing`) and every combination of `0`, `π` and `-π` over the modes under `parent` (28 for three modes), refused above `max_starts = 81`.
- *The union basis* (`src/augment.jl`).
  A `BasisSource(label, vectors, energies; cap, window)` is one source of vectors on the child's own product space; `union_basis(sources; m, drop=1e-6, allow_fewer=false)` merges them level by level by classical Gram–Schmidt applied twice, drops a candidate whose remainder is at most `drop`, stops at `m`, and returns the basis with a `UnionReport` of the vectors taken and dropped per source.
  Ending with fewer than `m` vectors is an `ArgumentError` that prints the report unless `allow_fewer = true`.
- *The augmented child.*
  `augment(parent, terms, θ, child, bare, U, report)` projects the child's bare operator on `U`, diagonalizes the projection densely, keeps every column, and returns an `AugmentedChild` (`node`, `solved`, `basis`, `terms`, `θ`, `report`); `bare` must be the solve on which the columns of `U` are expressed.
- *The wrapper.*
  `mean_field_augment(parent, spec, θ; m, window=0.5, max_solutions=4, bare=false, cap=nothing, starts, allow_fewer, ...)` runs `mean_field_solutions`, forms each child's bare solve once and passes it to `mean_field_solutions` and to `augment`, uses the solutions within `window` of the lowest (at most `max_solutions`), builds one source per used solution for each child (plus the bare ladder with `bare = true`), and returns `(; children, solutions, used, energy, unconverged)`; it adds no behaviour of its own.
- *Grids per frame* (`src/suggest.jl`).
  `suggest_bases(spec, θ; charge_cutoff=16, max_points=120, extent_scale=1.0, order=8, when_capped=:warn)` gives `ChargeBasis(charge_cutoff)` on a compact mode and a symmetric `PhaseGrid` sized from the zero-point width and the junction energies on any other mode, by the formula of the prototype's `default_bases`.
  When the formula asks for more than `max_points` points the grid keeps its extent with a larger step, so an energy on it is no longer an upper bound of the resolved problem's; `when_capped` makes that a warning (`:warn`), an `ArgumentError` (`:error`) or silent (`:allow`, for the KITE's deliberate 120-point cap), and the bases are the same in all three cases.
  `edge_weight(solved)` is the PhaseGrid edge warning's measure for the lowest kept state of a solved grid leaf.
  Growing a grid when a state reaches its edge, which the design's §4.6 puts in the loop, is done by the frame-optimization script (`bench/kite_meanfield/optimize.jl`), not by the package.

**One level deep.**
Mean field acts among the children of one node.
A child that is itself a branch is solved as a whole, with the mean-field terms routed into its subtree as ordinary terms and its own children solved once by `bare_child` and reused; self-consistency nested inside a child is not provided.
Every child of the mean-field node must be a `Node`: a bare `mode => basis` or `GivenSystem` child has no kept states to take an expectation in, and is an `ArgumentError`.

**What an augmented child stores, and why.**
An `AugmentedChild` keeps its union basis `U` on the child's own product space and the child's bare terms (those `route` places at the child, with no mean-field terms), beside the projected solve.
The perturbative-derivatives design (`docs/superpowers/specs/2026-11-10-perturbative-derivatives-design.md`) differentiates such a child as the bare operator projected on a fixed stored basis, which needs both; a child reduced to numeric energies and port matrices would report zero derivatives.
The `solved` node's kept vectors are `U W`, with `W` the eigenvectors of the projection, and it carries the ports and joint ports of `child_keys`.

**`materialize(...; augmented)`.**
Both forms of `materialize` take `augmented`, a vector of `AugmentedChild`.
Such a node is not solved; its `solved` is used as that child of its parent, and a port or joint port the tree needs that it lacks is formed from its kept vectors.
The terms are routed as usual, so the terms inside an augmented node are not used.
An augmented child is used as built, at the terms and `θ` that `augment` was given; a call whose `θ` differs from the child's in a parameter its bare terms depend on is an `ArgumentError` that says to rebuild the child, while the terms themselves and the parameters of the terms inside a branch child are not checked.
The `System` records its augmented children, which `augmented(sys)` returns.
Refused with an `ArgumentError`: an augmented node that is not in the tree or is the root, a node given twice, an augmented node inside another augmented node (Ruling 7 of the plan's ledger), `augmented` together with `previous`, and an augmented node or an ancestor of one with `arithmetic = :real`.
`KronFreeCircuits.operator(sys, node)` (not exported) throws for an augmented node (Ruling 7): its stored states diagonalize its bare operator projected on its union basis, which is not an operator on the node's product space; an ancestor's operator is built on the augmented child's states as usual.

**Complex arithmetic.**
The mean-field coefficients are complex in general (the outside expectations `⟨e^{i k φ}⟩`), so `bare_child`, `mean_field_child` and `augment` run complex, and an augmented node and every ancestor of it are complex in `materialize`, whatever the classification of Section 20f says.
Real arithmetic for mean-field children is a non-goal of the design.

**Measured: the KITE (design §6.1).**
The results below are compared with two other codes.
The prototype is circuit-quantization's `product_state.py`, which `test/python/generate_kite_fixtures.jl` runs to produce the fixtures `test/fixtures/kite/`.
The reference calculation is the owner's script that produced the figure (`mean_field_ref_dont_commit/kite_reference/`, untracked); its numbers are hard-coded in the benches and in `test/test_meanfield_kite.jl` with comments naming their source.

- *Intermediate numbers (steps 1 and 2).*
  `test/python/generate_kite_fixtures.jl` runs the prototype and writes `test/fixtures/kite/{published,normal,optimum,meanfield_reference}.json`.
  `suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5)` equals the prototype's grids exactly in all three frames.
  Against the prototype's wells, the product energies agree to 3.4e-13 (normal modes), 1.7e-13 (optimum) and 1.1e-11 GHz (published), and the three lowest mean-field levels of each mode to 5.9e-13, 5.9e-13 and 2.50e-6 GHz.
  Fed the prototype's own expectation values, `mean_field_child` reproduces the prototype's levels to 5e-13 in every frame (asserted to 1e-9 in `test/test_meanfield_kite.jl`).
- *The reference ground energy (step 3).*
  On the reference calculation's grid (`ChargeBasis(24)` and 160-point grids, `suggest_bases(spec; charge_cutoff=24, max_points=160, extent_scale=2.0)`) with 56 states kept per leaf, the published frame's four lowest levels are −9.5239953, −9.5105249, −8.4537903 and −8.3378621 GHz, within 3.2e-7 of the paper's `REF`; raising the kept count from 48 to 56 moved them by at most 6.5e-9 GHz.
- *Figure 8(b) (step 4), `bench/kite_meanfield/levels.jl`.*
  Two-well mean-field leaves keeping 20 states, no bare ladder, the root over the three augmented leaves; the error is `max(levels[1:4] - REF)` in GHz, with the ratio to the reference calculation's error:

  | frame | m = 8 | m = 12 | m = 16 | m = 20 |
  |---|---|---|---|---|
  | published | 6.9179e-01 (1.0000) | 2.2134e-01 (1.0000) | 3.7649e-02 (0.9986) | 6.5900e-04 (0.9726) |
  | normal modes | 6.2176e-01 (1.0000) | 1.9890e-01 (1.0000) | 3.6005e-02 (1.0000) | 8.0124e-04 (1.0000) |
  | optimum | 3.6677e-02 (1.0000) | 2.2679e-03 (1.0000) | 2.6093e-04 (1.0000) | 2.0717e-05 (1.0000) |

  Both targets hold: every error is within a factor of two of the reference's, and the ordering of the frames is the reference's at every `m`.
  In the normal-mode and optimum frames the eight lowest levels agree with the reference calculation's to 1.5e-13 GHz; in the published frame to 2.3e-6 and 5.6e-7 at `m = 8` and `12`, and 2.9e-4 and 6.5e-5 at `m = 16` and `20`.
  The run, its wells, union reports, Appendix C.4 variants and warnings are in `bench/kite_meanfield/README.md`.
- *The frame optimization (step 5), `bench/kite_meanfield/optimize.jl`.*
  From the identity, BFGS on `E_prod + 1e-4 F_NM` over circuit-quantization's six-parameter admissible family, with forward-difference gradients.
  The recorded run ended on its 150-iteration cap, not by converging (3290 cost evaluations, 149 min on one thread).
  Its final frame, with the noncompact columns scaled to `(Sᵀ C S)_jj = 1`, is within 1.03e-4 of the paper's `S_OPT` in every entry (largest: 0.696303 against 0.6962; then 0.734002 against 0.7339), so it does not agree to the four printed decimals of `S_OPT`.
  The run's `iterations.csv`, `optimum_S.txt` and `optimum_record.json` are tracked in `bench/kite_meanfield/recorded/`.
  On the figure's bases from the 33 starts it has two wells at −9.429473654505 and −9.429473654501 GHz, against −9.429473599561 and −9.429473599557 for the paper's optimum frame on the same bases, so the pass criterion (two converged wells within 2e-3 GHz of −9.4295) holds.
  On the score's grids, doubling the points changed the final frame's product energy by 4.7e-7 GHz; scaling the noncompact columns by (1.3, 0.8) changed it by −1.3e-7 GHz and flipping the sign of column 2 by −2.3e-13 GHz.

**Where the build departed from the design, and why.**
The rulings are those of the plan's ledger (`.superpowers/sdd/2026-11-10-mean-field-augmentation/progress.md`), a local file that is not tracked (`.superpowers/` is git-ignored); each ruling cited here is stated where it is cited.

1. *The figure's grids and starts.*
   The design's §6.1 assumed the derivation notebook's grids (33 charge states, about 117 and 116 grid points) and its 27 displacement starts plus the delocalized start.
   The script that produced the figure, which the owner supplied before the plan was written, uses `default_bases(charge_cutoff=16, max_points=120, extent_scale=1.5)`, that is 33 charge states and 120 points on each extended mode in every frame, and 33 starts (the compact mode at `0` or `π`, each extended mode at `0`, `π/2`, `π` or `−π/2`, and the delocalized start).
   The tests and benches use those, through `suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5)`, because they are what reproduces the published figure.
   The package's `default_starts` keeps the design's rule (`0`, `π`, `−π` on every mode and the delocalized start).
2. *The published frame.*
   The design's target for the mean-field levels was 1e-6 GHz; in the published frame the third level of mode 3 differs from the prototype's by 2.50e-6, and that one comparison is made to 1e-5 in `test/test_meanfield_kite.jl`, with the cause in a comment (Ruling 10).
   The cause was traced: the prototype's convergence test also includes ⟨φ⟩ and ⟨n⟩ of every mode, while this loop tests only its tables, the quantities the mean field depends on; in the published frame ⟨φ⟩ is not among them, so this loop stops at iteration 220 and the prototype at 221.
   It is a different stopping point, not a different fixed point: at equal inputs the two codes' mean-field operators agree to 5e-13.
   The convergence test was kept on the tables, as the general rule for multi-mode children.
   The Figure 8(b) ratios 0.9986 and 0.9726 at `m = 16` and `20` in the published frame have an inferred cause, not a proven one: on `varphi_2` the second well's Gram–Schmidt remainders against the first (7.5e-8 to 4.9e-6, one at 3.5e-4) straddle the `1e-6` drop threshold, and which directions are admitted then depends on round-off and on where each code's loop stopped; the reference calculation's own union is not recorded in its result files, so this was not checked against it.
   Those remainders are the loop's convergence residue, not differences between the wells, as measured in the final review with a separate script that is not in the repository: tightening the loop's `tol` from 1e-11 to 1e-13 lowers the distance of the second well's states 1 to 8 from the span of the first well's twelve on `varphi_2` tenfold (1.2e-8…1.8e-6 to 1.1e-9…1.7e-7) and leaves the other modes unchanged.
   So the two wells have the same states on that mode, and what the union admits there at `drop = 1e-6` is residue of a loop whose tables converge only to about `sqrt(tol)` ≈ 3e-6.
3. *A solution's stored states (Ruling 9).*
   The loop ends with one more solve of every child at the final tables, and that solve is what a solution's `solved` holds; its `expectations` and `bare_energy` remain those the energy was computed from.
   The plan's own detail kept the last iteration's solve, which is one iteration older than the tables (up to 6.8e-5 GHz off in the KITE's levels), while the prototype builds its levels from the operator at the converged expectation values.
   This costs one extra solve per child per run.
4. *The loop's tests and records.*
   The plan's two-fluxonium test assumed four fixed points; with a positive φφ coupling the same-well configurations are not fixed points of the together-update, so the test asserts the opposite-well pair and returns every run with its converged flag and iteration count, without asserting what the same-well starts end as, which round-off in the rise count decides (Ruling 4).
   The same round-off can make mirror-image starts end differently in symmetric problems, as in the prototype.
   A warm-started solution stores `start = :warm`, not the solution it came from, so that a frame sweep does not keep the whole chain of earlier solutions alive (Ruling 5).
   `operator(sys, node)` throws for an augmented node, and nested and `:real` augmented nodes are errors (Ruling 7, above).
5. *The frame optimization (Rulings 12 to 15).*
   The plan scored frames on `suggest_bases(spec; charge_cutoff=16, max_points=76)`, the prototype example's cap.
   On those grids the score is not variational in stretched frames: the optimizer reached the paper's well, then descended by under-resolving a stretched mode (−9.43001 GHz on 76 points against −9.42815 on 400 for one frame) and finally below the true ground level of −9.524 GHz (−13.16 GHz).
   The score now uses `max_points = 400`, circuit-quantization's own default, and a frame whose formula asks for more than 400 points on a mode scores `Inf` (Ruling 12).
   The optimizer stops at the first of a gradient norm below 1e-6, a cost change below 1e-9 GHz, a failed line search, or 150 iterations (Ruling 13).
   The warm start stays in the score although it makes the score history-dependent: the same `x` scores differently by about 3e-10 GHz depending on the evaluation before it, a floor of about 3e-4 on the forward-difference gradient, so the 1e-6 tolerance was out of reach and the run ended on the cap (Ruling 14).
   The recorded run predates a later fix to the gradient next to a non-finite score; it is not rerun because that case never occurred in it (its seven non-finite evaluations were one whole stencil), and the other changes were logging and wording (Ruling 15).
6. *The Harmonium benchmark* (design §6.2) was not built.
   It is exploratory, depends on what the KITE work measured, and needs choices about sizes and the GPU; it is deferred to its own plan.
   Multi-mode children, which it would exercise, are built; `mean_field_augment` is unit-tested on a parent whose child is a branch of two one-mode leaves and on one whose child is a single leaf on two modes (`test/test_augment.jl`): the product energy is an upper bound of the exact ground energy, the augmented parent's ground energy is variational and, with the default `bare = false`, no worse than bare children at equal `m` (with `bare = true` at a small `m` it was measured worse, because the union then holds fewer bare states than the plain child), and with the bare ladder and `m` equal to the child's full dimension the parent reproduces the plain spectrum.
   Caps and supplied starts are unit-tested on one-mode leaves.
7. *Derivatives* through the mean-field bases are not provided, as the design's non-goals say; the augmented child keeps its basis and bare terms for the perturbative-derivatives project, which holds the basis fixed.

**Known limitations.**

- `operator(sys, node)` throws for an augmented node; `augmented(sys)` gives its basis and bare terms.
- A warm start must be a solution of the same parent with the same table keys: a solution whose tables lack a key (same nodes, different terms, as in a frame change, where the keys `expiphi(k)` move with `S`) fails with a bare `KeyError` once mixing is on; `optimize.jl` rebuilds the tables by mode name for this reason.
- Passing `constant` to the spec form of `mean_field_solutions` fails with Julia's duplicate-keyword error, since that form sets it from the spec.
- An augmented child is checked against the `θ` of the `materialize` call that uses it only in the parameters its bare terms depend on; the terms are not checked, nor the parameters of the terms inside a branch child (those of its bare solve).
- The grid growth in `optimize.jl` (up to three times by 1.5) can take a grid past the 400-point cap that the score otherwise enforces.
- A leaf that keeps many states prints `fix_phases!` notes (the second state of each degenerate charge pair of a compact leaf solved with no cosine expectation) and PhaseGrid edge warnings for its highest states; `bench/kite_meanfield/README.md` shows that none of them concerns a state that a union reaches in the figure.

**Tests.**
`test/test_meanfield.jl` (fast tier: the mean-field terms against hand-built tables, the child and product energy, the loop on zero-pi, two coupled fluxonia, the final-tables solve, the product energy's invariance under a rescaling and a sign flip of a mode, and the KITE's optimum frame from two starts), `test/test_augment.jl` (fast tier: the union basis, augmented leaf and branch children, the `θ` check, `mean_field_augment` with measured errors, on a branch child and a two-mode leaf, with unconverged runs, and `edge_weight`), a `suggest_bases` testset in `test/test_bases.jl` (fast tier, with the binding cap), and `test/test_meanfield_kite.jl` (slow tier: all three KITE frames from the 33 starts against the fixtures).

### 20l. Derivatives by perturbation theory through the tree (2026-11-11)

First and second derivatives of every kept energy, kept state, port and joint port of a materialized tree, with respect to named spec parameters, at the root and at every node.
They are computed going upward through the tree, children first, by perturbation theory written by hand; no autodiff engine is used in the core, and nothing is re-solved.
This replaces the reverse mode of Section 11 (Section 11's three primitives were never built).
The design is `docs/superpowers/specs/2026-11-10-perturbative-derivatives-design.md` (it gives the reasons for going upward: one to about ten parameters, state derivatives as direct outputs, second order as the same recipe one order higher, and the near-degeneracy test needs the perturbation of a specific parameter); the plan is `docs/superpowers/plans/2026-11-11-perturbative-derivatives.md`, whose "Decisions and their costs" (D1–D13, P1–P9) and "Disagreements with the controller's decisions" are cited below by name.
The rulings cited are those of the run's ledger, a local file that is not tracked; the content of each is stated where it is cited.
Where the work stands (branch, merge, push) is in `docs/superpowers/STATUS.md`.

**Sources of the formulas.**
The first-order rules are those of arXiv:2408.12704v3 (Rajabzadeh et al.), `main.tex`: `eq:eig_val_grad` and `eq:eig_vec_grad` (lines 283–284), with the sum over states made complete; the second energy derivative is its `eq:eig_val_secondgrad` (line 826).
All three were verified against the paper's LaTeX source by this project, and the tests check that the rules reduce to them at a single dense node (`test/test_derivatives_node.jl`, `test/test_derivatives_second.jl`).
The second-order state, its normalization and the second-order gauge were derived for this project, since the owner's notes (`tight_binding.tex`, lines 287–360, which the design cites) are not on the machine that built it.
They were checked numerically on random Hermitian families before the plan was written (agreement 1e-11 at first order and 1e-8 at second), by a reviewer who re-derived them term by term, and by the finite-difference suite; comparing them with the notes remains to be done.

**What a derivative is.**
The derivatives are those of the calculation as performed: the selection of kept states, every basis, grid and cutoff, the coordinate frame, the mean-field bases, the solver of each node, its real or complex arithmetic and its discrete phase choice (its peak indices, `peak_component`) are held fixed.
They agree with finite differences of `materialize` wherever those are themselves well defined (away from exact degeneracies, symmetric points, kept/unkept crossings and changes of a peak index inside the stencil), to within the finite differences' own error in every finite-difference measurement below.
At an iterative node the derivatives agree with the full-spectrum `Dense` route, including weakly coupled children under an iterative parent (the weak-coupling testset of `test/test_derivatives_iterative.jl`, per kept column, couplings 0.3 to 3e-4; added after the final review found that case failing).
A node's `∂H/∂x` has three contributions: the derivatives of the coefficients of the terms routed to it (and, at the root, of the constant), its children's port and joint-port derivatives, and its children's energy derivatives on the diagonal (Section 11.1, which stands).
Removing either of the last two makes a test fail (`test/test_derivoperator.jl`, `test/test_derivatives_tree.jl`).
Coefficients are differentiated symbolically over the whole expression grammar, `differentiate(c, x)` (`src/diffcoeff.jl`); the result is an ordinary `Coefficient`, so `∂H/∂x` is applied by the existing apply layer.
The derivative operator is built from the terms themselves, beside the forward operator and never from it (Decision 4): the forward real operator drops a term whose coefficient vanishes at `θ`, whose derivative need not vanish (a coupling at `g = 0`, a flux term at `varphi_ext = 0`).
A node none of whose terms (in its subtree, and the constant at the root) depends on `x` is not visited and reads as zeros (P7).
At the root, a parameter of `wrt` that nothing in the system depends on (no term, no augmented child's bare term, not the constant) is an `ArgumentError` when the system has a spec that does not declare it (a `θ` key the spec never named, or a parameter `substitute` replaced by a number), and otherwise reads as zeros with one warning; the rules of `spectrum` differentiate only the parameters something depends on and neither throw nor warn for the others.

**First order at one node** (plan Task 4; `src/derivatives.jl`, `_node_first_order`).
`mn` is the node's materialized data (a `MaterializedNode`), `V` its kept vectors (columns `ψ_j`), `E` its kept energies, `A = mn.peak_amplitudes`, and `H1 = ∂H/∂x`; `dx` holds, per axis of the node, the child's first-order derivative data along `x` (a `NodeDerivative`) or `nothing` for a basis or a child that does not depend on `x`.
In physical terms: Hellmann–Feynman for the energies, first-order perturbation theory over every other state for the states, then a phase term that keeps the phase convention.

1. `R = H1 V`, `C = V† R`, and `∂E_j = Re C[j, j]` (the paper's `eq:eig_val_grad`): the energy moves by the expectation value of the perturbation.
2. `W, clashes = _apply_resolvent(res, R, E, [[j] for j])`: `w_j = Σ_{l ≠ j} ψ_l ⟨ψ_l|H1|ψ_j⟩ / (E_j − E_l)` over all states of the node's space, so `(H − E_j) w_j = −(H1 − ∂E_j) ψ_j` and `w_j ⟂ ψ_j` (the paper's `eq:eig_vec_grad`, with the sum complete): the state mixes in each other state in proportion to their coupling over their detuning.
3. The gauge.
   The phase convention makes `ℓ_j · ψ_j` real and positive, where `ℓ_j · X = _peak_map(mn, X, _peak_amps(mn))[j, :]` (at a leaf it is the entry of `X` at the state's peak index; at a branch the contraction with the children's peak amplitudes).
   Its derivative must therefore be real, which fixes the one free component of `∂ψ_j`, along `iψ_j`: `a_j = −Im( L1[j, j] + L0[j, j] ) / Re A[j, j]`, with `L0 = _peak_map(mn, W, _peak_amps(mn))` and `L1` the sum, over the axes `k` with `dx[k] !== nothing`, of `_peak_map(mn, V, amps)` with `amps[k]` replaced by `dx[k].peak_amplitudes`.
   On a real node `a_j = 0` and is not computed.
4. `∂ψ_j = w_j + i a_j ψ_j`; `gauge = a`.
5. `∂A = _peak_map(mn, ∂V, _peak_amps(mn)) + L1`.
6. Each port and joint port `P = V† O V` on a key `K`, with `ops = _axis_ops(mn.node, mn.children, K)`:
   `∂P = _sandwich(mn, ∂V, V, K, ops) + _sandwich(mn, V, ∂V, K, ops) + Σ_g _sandwich(mn, V, V, K, ops with entry g replaced by _derivative_op(dx[k_g], K_g))`, the last sum over the entries `g = (k_g, op_g, K_g)` whose child has derivative data.
   `O` need not be Hermitian, so the first two terms are computed separately.
7. A column with a clash gets `∂E_j = NaN` and `∂ψ_j = NaN` (and `NaN` flows into the ports and peak amplitudes), with one warning per call that names the states.
8. A column of `R` that is not finite comes from a child whose derivative is undefined (P5): it is set to zero before the resolvent (so it reaches no linear solve, no selection-boundary test and no block search), and every output of that state is `NaN`, its gauge included on a real node, with one warning per call naming the children ("…undefined (NaN) because a child's derivative is undefined…"), not a degeneracy warning.

The phase convention is one functional at every node, "`ℓ_j · ψ_j` is real and positive", so leaves and branches share one code path and each node hands the derivative of its peak amplitudes to its parent (P4).
On a complex node the eigenvector rule differs from the paper's by the term along `ψ_j` itself (step 4), which the paper sets to zero.
Energies and diagonal port elements are blind to that term; the off-diagonal elements of a non-Hermitian port (`expiphi`) on complex nodes are not, and the tests compare them at a complex leaf and a complex branch, to both orders.

**Second order at one node** (plan Task 7; `_node_second_order`).
The operator `∂²H/∂x∂λ`, for a term with coefficient `c` and operators `ops`, with `g`, `h` running over the entries of `ops`: `c_xλ(θ)` on `ops`; `c_x(θ)` on `ops[g → dλ]` and `c_λ(θ)` on `ops[g → dx]`, for every `g` whose child has that data; `c(θ)` on `ops[g → dxλ]`; `c(θ)` on `ops[g → dx, h → dλ]` for every ordered pair `g ≠ h`; then the children's second energy derivatives on the diagonal and the constant's second derivative.
With `x == λ` the same list gives the factor 2 of the cross terms.
Per kept state `j`, with `W^x = ∂_xV − i V diag(a^x)` (likewise `W^λ`), the part of the first derivative orthogonal to the state (columns `w^x_j`), and `dx`, `dλ`, `dxλ` the children's first- and second-order data per axis as above.
In physical terms: second-order perturbation theory driven by the first-order responses, then the normalization and the phase convention, each differentiated once more.

1. `RHS = (Hx W^λ − W^λ diag(∂_xE)) + (Hλ W^x − W^x diag(∂_λE)) + Hxλ V`.
2. `∂²E_j = Re⟨ψ_j|RHS_j⟩` (the curvature of the level: the second-order operator's expectation plus the level repulsion of the first-order mixing), which equals `⟨ψ_j|Hxλ|ψ_j⟩ + 2 Re⟨w^x_j|Hλ|ψ_j⟩`, the paper's `eq:eig_val_secondgrad`.
3. `G, clashes = _apply_resolvent(res, RHS, E, [[j] for j])` and `U = G + i W^x diag(a^λ) + i W^λ diag(a^x)`, whose column `u_j` is the part of `∂²ψ_j` orthogonal to `ψ_j`.
4. Normalization (the state stays a unit vector to second order): `Re⟨ψ_j|∂²ψ_j⟩ = −Re⟨∂_xψ_j|∂_λψ_j⟩ = −Re⟨w^x_j|w^λ_j⟩ − a^x_j a^λ_j =: n_j`.
5. Gauge (complex nodes; `b_j = 0` on a real node): the second derivative of `ℓ_j · ψ_j` is real, so `b_j = −Im( ℓ^{xλ}_j·ψ_j + ℓ^x_j·∂_λψ_j + ℓ^λ_j·∂_xψ_j + ℓ_j·u_j ) / Re A[j, j]`, where `ℓ^x` replaces one axis's amplitudes by the child's first derivative (summed over the axes with data), and `ℓ^{xλ}` is the sum over axes of the child's second derivative replaced, plus the sum over ordered pairs of axes `k ≠ m` with `k → dx[k]`, `m → dλ[m]` replaced together.
6. `∂²ψ_j = u_j + (n_j + i b_j) ψ_j`; `gauge = b`.
7. `∂²A = ℓ·∂²V + ℓ^x·∂_λV + ℓ^λ·∂_xV + ℓ^{xλ}·V` (all rows).
8. Each port and joint port, with `S(W, V; O)` for `_sandwich`, `O_x` the operator with one entry replaced by the child's `x`-derivative (summed over entries), and `O_xλ` built like `ℓ^{xλ}`:
   `∂²P = S(∂²V, V; O) + S(V, ∂²V; O) + S(∂_xV, ∂_λV; O) + S(∂_λV, ∂_xV; O) + S(∂_xV, V; O_λ) + S(V, ∂_xV; O_λ) + S(∂_λV, V; O_x) + S(V, ∂_λV; O_x) + S(V, V; O_xλ)`.
9. A column with a clash in step 3, or already `NaN` at first order, is `NaN` in energy and state, with one warning.
   When a child's first- or second-order data is undefined, the columns of `RHS` that are not finite and the node's own first-order `NaN` states are inherited from the child: they are zeroed before the resolvent, are `NaN` (gauge included), and are reported by the warning of step 8 of the first order, not as "degenerate at first order".

The tree pass runs one first-order pass per parameter and, with `order = 2`, one second-order pass per parameter and per mixed pair; each node's resolvent is formed once and shared by all passes.
A second-order pass rebuilds `∂H/∂x`, `∂H/∂λ` and `∂²H/∂x∂λ` at each node; only first-order data and resolvents are kept between passes.

**What is retained, and its cost** (Decision 2, P1, P2; `Retained` in `src/materialize.jl`, the `retain` keyword in `src/tree.jl`).
The sum over states needs the states a node computed but did not keep.
Every node now retains them (`Node(...; retain = :auto)`), except a `Dense` node above 1000 product states (`_RETAIN_DENSE_MAX`, the `Auto` dense cutoff), where it is opt-in (`retain = :always`); `retain = :never` restores the old behaviour per node.
The solves whose results are never differentiated do not retain whatever the node's setting: the re-solves inside the mean-field loop (`mean_field_child`) and the solves of `bo_surface` (an internal override of `_solve_node`, added after the final review; `bare_child` keeps the node's setting, since the derivative pass visits its children below an augmented node).
`Retained` holds every computed energy, the vectors of the computed states that were not kept (the kept ones are not stored twice), and the flags `complete` and `followed`.
A retaining `Dense` node with a `compute` limit computes its full decomposition, and still hands only the lowest `compute` pairs to `keep`.
A `Dense` node without its complete spectrum is re-diagonalized by the derivative pass, with one `@info` per node naming the remedy `retain = :always`.
Where its selection boundary cuts an exactly degenerate cluster, the re-diagonalization's partners of a kept state are replaced by an orthonormal basis of the cluster with the stored kept vectors projected out, so that the other columns do not count the kept state twice (measured on the `cos 2φ` element with the pair at 16 GHz cut: the derivative of the ground state, which couples to the pair, equals a retaining twin's to 1e-12, where before it put the pair's whole contribution, 0.0625, on one charge state instead of 0.03125 on each).
A retaining compute-limited `Dense` node now computes the full decomposition where it computed a partial one, so the rotation inside an exactly degenerate pair of kept states may differ from before (noted when retention was built; energies and the kept subspace are unchanged).
Measured cost of the default (`bench/retention_cost.jl`, minimum of five `materialize` timings and `Base.summarysize`, on the loaded build machine):

| tree | `retain = :never` | `retain = :auto` |
|---|---|---|
| transmon leaf, `ChargeBasis(400)` (801 states), `Dense`, `compute = 10` | 0.019 s, 73 KB | 0.091 s (×4.8), 5.2 MB |
| fluxonium leaf, `PhaseGrid` of 901 points, `Dense`, `compute = 10` | 0.059 s, 83 KB | 0.111 s (×1.9), 6.5 MB |
| the KITE tree of `test/test_meanfield_kite.jl`, optimum frame | 0.0064 s, 30 KB | 0.0111 s, 0.51 MB |

Ruling 7 kept the default despite this cost: the absolute cost is under 0.1 s per leaf, the owner prefers retention (design §4.5), and `retain = :never` restores the old behaviour; it costs slower forward solves in loops that re-solve small `Dense` leaves with a `compute` limit many times.
That ground understates the cost of keeping results: the final review measured the `System`s of a 21-point sweep of one 901-point leaf at 130 MB with retention against 0.81 MB without (`retain = :never` per node restores the old size).
Peak memory at a retaining `Dense` node with `compute` below its dimension is about 4N² elements while it is solved (a reviewer's reading of the code, not measured).

**How fast the sum over states converges** (design §4.5 asked for this before the default is fixed; `bench/derivative_sums.jl`).
In three cases on two single `Dense` leaves (the fluxonium for `varphi_ext` and for `E_J`, and the transmon), the first-order state derivatives were recomputed with the resolvent's other states cut to the `M` lowest above the kept ones; the table gives the largest error relative to the largest element of the full result, for the state derivatives and for each port derivative:

| leaf, parameter | M | states | port 1 | port 2 |
|---|---|---|---|---|
| fluxonium, 601 points, 5 kept, 596 others; `varphi_ext` (ports `n`, `phi`) | 0 | 0.094 | 0.059 | 0.082 |
| | 5 | 1.1e-3 | 9.2e-5 | 4.6e-5 |
| | 10 | 7.0e-5 | 9.9e-7 | 4.5e-7 |
| | 20 | 2.0e-6 | 1.4e-9 | 7.7e-10 |
| | 40 | 5.3e-9 | 2.9e-14 | 1.1e-14 |
| | 80 | 5.0e-13 | 2.8e-16 | 2.3e-16 |
| the same leaf; `E_J` | 0 | 0.33 | 0.71 | 0.27 |
| | 5 | 0.088 | 0.06 | 2.1e-3 |
| | 10 | 0.016 | 1.4e-3 | 7.8e-5 |
| | 20 | 1.3e-3 | 2.4e-5 | 1.2e-7 |
| | 40 | 1.9e-5 | 4.1e-9 | 8.6e-12 |
| | 80 | 6.2e-9 | 2.9e-16 | 1.4e-16 |
| transmon, 61 charge states, 4 kept, 57 others; `n_g` (ports `expiphi(1.0)`, `n`) | 0 | 0.69 | 1.3 | 3.1 |
| | 5 | 1.2e-4 | 2.3e-6 | 4.4e-7 |
| | 10 | 2.5e-9 | 5.7e-15 | 5.5e-16 |
| | 20 | 1.1e-19 | 0 | 0 |

What it implies: on these leaves a sum over the nearest few tens of states would reach 1e-8 in the state derivatives (40 to 80 of the fluxonium's 596, 10 of the transmon's 57), and the port derivatives, which are what a parent consumes, converge faster (20 to 40 states on the fluxonium).
So a controlled truncation (nearest states until a convergence criterion holds, with the count reported) could replace full retention at much lower memory.
This is measured at first order on single leaves only; second order, branches and iterative nodes were not measured, and the rate depends on the operator (`E_J` converges more slowly than `varphi_ext` on the same leaf).
The default was not changed; the choice is the owner's (STATUS.md).

**Iterative nodes: the linear solve** (Decision 1, P3, Ruling 9; `src/resolvent.jl`, `_complement_solve`).
At a node without its complete spectrum (`LOBPCG`, `Lanczos`, `ShiftInvert`, `ChebyshevFilter`), the part of each state derivative orthogonal to the node's known states (kept and retained unkept) solves `Q (H − E_j) Q x = −Q R_j`, with `Q` the projector off the known states, and the sum over the known states gives the rest; truncating to the known states without the solve is never done (a test asserts that it would be wrong).
The solver is a hand-written conjugate gradient, all columns in lockstep through the operator's block apply, projected on the complement, preconditioned by the Jacobi diagonal `1 / hypot(D − E_j, 0.01)` that `:ritz` uses; it uses only `mul!`, broadcasts and column sums, so it runs on a device unchanged.
Its right-hand side is formed on the host as `R_j` minus its components on every known state, from the overlaps the sum over the known states already uses (Decision 1's `R_j − ψ_j ∂E_j`, with the other known components subtracted too, in one cancellation).
Stopping rule (as implemented after the final review's re-review): each column's target is its own scale, a recurrence residual at most `lintol ‖Q R_j‖` (`lintol = 1e-10`, `linmaxiter = 5000`).
A column that cannot reach it because of round-off is accepted at the round-off floor: once its recurrence residual is at most the cap `1000 eps ‖R_j‖` without meeting its target, its true residual (recomputed with a fresh application of `H`) is checked at every iteration, and when it has not improved by a factor of 2 over 5 successive checks and is at most the cap, the column stops there.
The cap is chosen from the floors measured on the weak-coupling cases, 5e-16 to 4e-14 of `‖R_j‖` (2 to 180 eps), a factor of 5 above the largest.
After the loop every true residual is recomputed: a column that met its target must have one at most `10 lintol ‖Q R_j‖`, or else at most the cap (it is then counted at the floor); a column at the floor must have one at most the cap; a column that misses both, in the loop, at `linmaxiter` or in the final check, is a "did not converge" error.
The positivity check is not applied to a column whose recurrence residual is at or below the cap, where round-off can make `⟨p, (H − E_j) p⟩` non-positive; above it a non-positive value is an error ("a state below or degenerate with the kept state was not computed (or was not retained)").
A column whose complement part is at most the cap (zero included) is converged from the start and solved as exactly zero; this subsumes Ruling 9's rule (`‖Q B_j‖ ≤ lintol ‖B_j‖`).
Each solve is recorded in `Derivatives.linear_solves` with `residual` (the largest true residual relative to `‖Q R_j‖` over the columns that met `lintol`), `from_start`, `at_floor` (the columns stopped at the floor) and `floor_residual` (their largest true residual relative to `‖R_j‖`).
Why: for a parameter internal to a weakly coupled child, `R_j ≈ ψ_j ∂E_child` lies almost inside the parent's computed states, and the round-off of the projection, about `eps ‖R_j‖`, can exceed `lintol ‖Q R_j‖`.
Measured on two transmons under an `LOBPCG` root (12 states per leaf, 8 computed, 6 kept), for `E_J2`: `‖Q R_j‖/‖R_j‖` was 5e-5 to 2e-3 at `g = 0.03`, 5e-6 to 2e-4 at 0.003 and 5e-7 to 2e-5 at 3e-4, while the conjugate gradient's floor, with the right-hand side projected inside the solve (before the host-side cancellation), stayed at a true residual of 5e-16 to 9e-15 of `‖R_j‖`, that is 1e-10 to 3e-9 of `‖Q R_j‖`.
The original code (a target `lintol ‖Q R_j‖` and no floor) threw "did not converge" at `g = 0.03` (`E_J2`) and at 0.003 (`E_J2`, `E_C1`), and "not positive on the complement" at 3e-4 on periodic grids, where the iteration continued past its round-off floor.
The final review's first fix measured convergence against `lintol ‖R_j‖`: every case finished, but a column whose derivative is small stopped early, measured per column at up to 2.8e-6 of the column's own norm (`E_J2`, `g = 3e-4`, a column of norm 4.9e-8) and 9.9e-9 at `g = 0.3`; a matrix-wide comparison could not see it.
With the stopping rule above, all 32 cases of that comparison (charge bases and periodic grids, `g` = 0.3, 0.03, 0.003, 3e-4, `E_J2`, `E_C1`, `n_g1`, `g`) finish in 3 to 7 iterations, and every kept column's state derivative agrees with the `Dense` route to within 4.4e-10 of its own norm at `g` from 0.03 to 3e-4 and to within 4.4e-11 at 0.3, at first and second order (measured per column, columns of norm at least 1e-12); where the original code converged, the per-column errors are within 8% of its (ratios 0.93 to 1.02 over 49 comparable rows; for example 4.6e-12 against 4.9e-12, `E_J2`, `g = 0.3`).
At `g = 3e-4` the remaining error is the eigensolver's: a tighter `lintol` (1e-13) leaves it unchanged.
Where accuracy degrades: a column that stops at the round-off floor, or is converged from the start, is accurate to `1000 eps ‖R_j‖ / gap_c` in absolute terms, with `gap_c` the distance from `E_j` to the lowest state of the complement; that exceeds 1e-6 of `‖ψ′_j‖` when `‖ψ′_j‖ gap_c / ‖R_j‖ < 2.2e-7`.
On the two-transmon test tree that is `E_J2`, kept state 1 (`‖R‖ = 0.86`, `gap_c = 41.8`): the bound is 9.3e-8 of `‖ψ′_1‖` at `g = 3e-4`, 9.3e-7 at 3e-5 and 9.3e-6 at 3e-6.
Measured by the fix's independent verifier (scripts not in the repository): down to `g = 3e-6` no column stopped at the floor (the host-side cancellation left true residuals of 1e-17 to 1e-22 of `‖R_j‖`), and the per-column errors there, 1.1e-7 at `g = 3e-5` and 4.2e-8 at 3e-6 for `E_J2`, 2.7e-8 and 1.6e-8 for `E_C1` (first order, default `lintol`, identical at `lintol = 1e-13`), are the eigensolver's (`LOBPCG(tol = 1e-11)`), not the floor's.
In these cases the host-side cancellation of the right-hand side brings the true residual below `lintol ‖Q R_j‖` (5e-14 at `g = 3e-4`), so no column stopped at the floor; the floor path is exercised by a unit test whose right-hand side is `1e-9` off the known states (every column stops at about 7e-15 of `‖B_j‖` and is still the solution to 2.4e-6).
The slow tier's weak-coupling testset asserts the per-column accuracy (1e-7 at couplings 0.03 to 3e-4, 1e-9 at 0.3, both orders), and the fast tier one case; the default solvers against the `Dense` route (`Auto()`, `ShiftInvert` leaf and `LOBPCG` root at their default tolerances, both orders and mixed) are unchanged at 5.1e-11 at worst.
The `ShiftInvert` factorization is not reused: plan P3 chose one hand-written conjugate gradient that runs unchanged on any device, and gives up that reuse as its cost (`KrylovKit.linsolve`, the other candidate, has no preconditioner hook and works one column at a time).
A node that keeps every state of its space needs no solve, and a node with `retain = :never` projects out its kept states alone, which must then be its lowest computed ones; it records no selection-boundary pairs (its unkept computed states are not known).
The iteration count grows roughly linearly with a grid leaf's point count (measured: 744, 1505 and 3055 iterations at 3001, 6001 and 12001 points at the default settings), so `linmaxiter` may need raising for very fine grids.
Measured: the largest iteration count in the tests is 190 (`test/test_derivatives_iterative.jl`); at the KITE root (`LOBPCG`, 6 computed, 4 kept) 15 solves took 49 to 60 iterations with true residuals at most 9.8e-11 (both before the change of scale).
The iterative tests tighten the eigensolver tolerance (`tol = 1e-11`, Decision 11); a reviewer measured that at the default `LOBPCG` tolerance (1e-8) the derivatives still match the dense route to 2.8e-12 (vectors) and 4.3e-12 (energies), at `g = 0.3` only.

**Degeneracy, the selection boundary and near-degenerate blocks** (design §4.4; Decision 7, P5, P6, P8; `src/derivblocks.jl`).
- Exact degeneracy.
  A pair within `degenerate_tol = 1e-9` GHz is exactly degenerate; with a zero coupling by the per-column rule (`|⟨ψ_l|R_j⟩| ≤ 1e-10 ‖R_j‖`, the one definition of a zero coupling throughout) it is left out of the sum, otherwise the column is `NaN`, with a warning and a block (or a selection-boundary record when the partner is not kept).
  A `NaN` child makes every ancestor `NaN` (P5): a parent whose own derivatives exist despite a degenerate coupled pair in a child reports `NaN`, and the block report names the child.
  This holds at a `Dense` and at an iterative parent, at both orders, and when the child clashes only at second order (tested in `test/test_derivatives_blocks.jl`); until the final review an iterative parent threw "not positive on the complement" instead, and a `Dense` parent's second-order warning called its states "degenerate at first order".
  The cost at symmetric operating points: a non-degenerate root whose derivative exists reports `NaN` because a child keeps both members of an exactly degenerate, coupled pair; a parallel-transport gauge inside the child would give correct parent energies (not built; the owner's decision, STATUS.md).
- The selection boundary (the warning of Section 11.3).
  At every visited node, a kept state close to an unkept computed state warns ("selection boundary") and is recorded in `Derivatives.boundary`, since the derivative at a fixed selection is fragile there.
  "Close" is the block test below when `scale` has an entry for the parameter, else a detuning at most `1e-6` times the states' distance from the node's lowest computed energy (P8); an exact degeneracy with an unkept state it couples to, at either order, is always recorded.
  States beyond a node's `compute` were never computed and cannot be tested.
- Near-degenerate blocks (the cluster handling of Section 11.4).
  A pair of kept states is flagged along `x` when `|⟨ψ_l|∂H/∂x|ψ_i⟩| δx > r |E_i − E_l|`, with `r = 0.1` and `δx` the caller's `scale` for `x`, or when it is an exact degeneracy with a nonzero coupling at first or second order; a block is a connected component of the flagged pairs, per node, per parameter and per mixed pair (the union of the blocks of its two parameters and of its mixed pass).
  Point derivatives are always the true derivatives and are not changed by the blocks (bitwise identical with and without `scale`, measured by a reviewer).
  A block is a validity report: the second-order Taylor expansion of its levels is not trusted over `δx`.
  It carries the effective Hamiltonian, second order in the displacement (first order when `order = 1`), P6:
  `H_eff(δ) = diag(E_B) + Σ_x δ_x C^x + ½ Σ_{x,y} δ_x δ_y (⟨H_xy⟩ + F^{xy} + F^{yx})`, with `E_B` the block's energies, `C^x = H1[x] = ⟨ψ_b|∂H/∂x|ψ_b′⟩` and `⟨H_xy⟩ = ⟨ψ_b|∂²H/∂x∂y|ψ_b′⟩` over the block's states `b`, `b′`, the states outside the block folded in through `F^{xy} = ½ (M^{xy} + (M^{yx})†)`, `M^{xy}[b, b′] = ⟨(∂H/∂x) ψ_b|Y^y_{b′}⟩` and `Y^y_{b′} = Σ_{l ∉ B} ψ_l ⟨ψ_l|∂H/∂y|ψ_{b′}⟩ / (E_{b′} − E_l)` over every state outside the block, kept or not.
  `block_levels(b, displacement)` diagonalizes it densely (an `ArgumentError` when it is not finite, which happens when a block state is exactly degenerate with a state outside the kept set that it couples to, a selection-boundary record); its eigenvalues differ from a re-solve by `O(|δ|³ v³ / g²)` (`v` the block's largest coupling to outside states, `g` its gap to them).
  Measured on the `cos 2φ` doublet (detuning 0.0593 GHz, coupling 0.7456, `test/test_derivatives_blocks.jl`), against independent re-solves, by a reviewer:

  | δ | 0.05 | 0.02 | 0.01 | 0.005 |
  |---|---|---|---|---|
  | largest level error (GHz) | 1.155e-7 | 4.24e-9 | 2.87e-10 | 1.85e-11 |

  The error falls as δ⁴ at this symmetric point (the odd orders vanish there), so no second-order term is missing.
  On the KITE doublet (below) it is 9.9e-9 at δ = 0.01 and 1.1e-9 at δ = 0.005.
  Forming a block's second-order pieces costs two resolvent applications per pass on as many columns as the block has states.
  For a block flagged in more than one way `coupling` is the largest of quantities that may have different units, for reporting only; `tr(H1[x])` and `tr(H2[(x, x)])` are the block-invariant sums of the first and second derivatives of the block's energies, which exist even when the per-state derivatives are `NaN`.

**Stored-data children** (design §4.8, Decision 8).
An augmented child (Section 20k) is differentiated as its bare operator projected on its fixed stored basis: its `∂H/∂x` is the derivative of its stored bare terms projected on the basis, plus its node children's port and energy derivatives; its stored states are the whole spectrum of the projected problem, so its resolvent is the sum over them, with no solve, no re-diagonalization and no selection boundary.
This is the derivative of the calculation with the union basis held fixed, not of a re-run of the mean-field loop.
`materialize` checks an augmented child's `θ` only in the parameters of its own bare terms; the derivative pass repeats the check over every term in the augmented subtree (the plan's disagreement on D8).
A `GivenSystem` child is an `ArgumentError` unless `given_independent = true` declares its numbers independent of every parameter; it then reads as zeros.
Building this found and fixed a defect: a complex child beside a real sibling at second order (always the case next to an augmented child) threw a `MethodError`, because the element type of a collection of children's data was inferred from its first element.
A Born–Oppenheimer reduced spec is an ordinary spec and needs no special handling (see the departures below).

**The API** (Decision 6, P9).

- `derivatives(sys::System; wrt, order=1, mixed=(), scale=(;), r=0.1, degenerate_tol=1e-9, lintol=1e-10, linmaxiter=5000, given_independent=false) -> Derivatives`.
  `wrt` is a parameter or a collection; `degenerate_tol ≥ 0`, `lintol > 0` and `linmaxiter > 0` are checked; `mixed` holds pairs of different parameters of `wrt` (both must be in `wrt`, the plan's disagreement on D6) and needs `order = 2`; `scale` is a `NamedTuple` of positive displacements of parameters in `wrt`.
- Read through new methods of the existing generics, at the root or at any `node`: `energies(d, x[, node])`, `eigenvectors(d, x[, node])`, `port(d, x, mode, p[, node])`, its spec-form spelling `port(d, x, kind, mode[, node]; k)` (as `port(sys, kind, mode, node; k)`), and `port(d, x, key[, node])` for a joint port, and the same with two parameters `(d, x, λ, ...)` for second derivatives (symmetric in `x`, `λ`).
- `blocks(d)` returns the `NearDegenerateBlock`s (`node`, `parameters`, `states`, `energies`, `detuning`, `coupling`, `exact`, `order`, `H1`, `H2`); `block_levels(b, displacement)` gives a block's levels and states at a finite displacement.
- `Derivatives` also holds `boundary` (the selection-boundary records) and `linear_solves`.
- `spectrum(root, spec, θ; ports, derivative_options) -> (; energies, ports)`, defined in the core with no autodiff code, is the one function the outer rule covers; a zero cotangent or tangent gives zeros without running the derivative pass.

**The outer rule** (design §4.6, Decision 9; `ext/KronFreeCircuitsChainRulesCoreExt.jl`).
An `rrule` and an `frule` of `spectrum`, each running the first-order passes of `derivatives` once and contracting with the cotangent or tangent; the `rrule` follows ChainRules' convention for a real loss of complex port elements (the cotangent of `P` is `∂L/∂Re P + i ∂L/∂Im P`).
`ChainRulesCore` is a weak dependency with this extension; the core's `[deps]` are unchanged.
In practice the extension is always active, because `ChainRulesCore` is loaded transitively through `ITensors` in every session.
`Zygote` is test-only (a user adds it to their own environment), in `test/chainrules` with its own environment (Zygote 0.7.16 and ChainRules 1.74.13 resolved and work on Julia 1.13.2); `FiniteDifferences` and `ChainRulesTestUtils` are not used, the tests comparing against the package's own Richardson-extrapolated central differences.
Differentiating twice through the rule (Zygote over Zygote) fails with an obscure "Flatten" `ArgumentError`; second derivatives come from `derivatives(...; order = 2)`, as the docstring says.
Measured (`bench/derivative_design/design.jl`, two coupled transmons, `x = (E_J1, E_J2, g)`, a least-squares loss on `[E_2 − E_1, E_3 − E_1, |n_1[1, 2]|]`): BFGS on the Zygote gradient converged on its gradient tolerance (`‖∇L‖∞ = 1.78e-11`) in 13 iterations, 16 loss and 16 gradient evaluations, 1.6 s, reaching the planted design to about 1e-11 with final loss 3.8e-23; the gradient agreed with finite differences of the loss to 1.9e-9 relative at the start, and at the end both were about 1.8e-11.
A reviewer checked the Zygote gradient of a loss of complex off-diagonal port elements on the complex periodic two-transmon tree against central differences for all seven parameters (about 1e-8 relative), and the `frule` against the `rrule` by a dot-product test.

**Devices** (design §4.7, Decision 12; `test/cuda/test_derivatives_device.jl`).
The derivative pass ran on the GPU with no change to the code at a real `LOBPCG(on = CuArray)` root of two transmons (144 states, 8 computed, 6 kept): its first and second derivatives along `g` and `n_g1`, with the linear solve on the complement, equal the host's `Dense` route at the test's tolerances, 10 of 10 tests on an RTX A4500, with the results as host arrays.
The test asserts that the root's resolved solver and the forward operator of its linear solve are on `CuArray` (added after the final review; before, it would also have passed on the host).
The device tier as a whole had 13 errors, all in older testsets of `test/cuda/test_device.jl` that solve complex `LOBPCG` on the device, none in the new file.
Diagnosed on the build machine: cuBLAS's complex-double triangular solve (`cublasZtrsm_v2_64`, cuBLAS 13.8.0.4) fails on every call with `CUBLAS_STATUS_EXECUTION_FAILED`, even on a 2×2 matrix, while `zgemm`, `ztrmm`, `zherk`, the device Cholesky, `ComplexF32` and real triangular solves work; LOBPCGEigensolver's `ortho!` uses that solve, catches the error as a Cholesky failure and gives up ("badly failing to keep the vectors normalized").
It is not caused by this project or by dependency versions: `origin/main` fails identically with the new environment, this code fails identically with the old one (CUDA 6.4.2, GPUArrays 11.5.16), and a copy of the environment that passed this tier on 2026-10-18 fails today on the minimal case (if that log was indeed from this machine and depot: the 2026-10-18 log may come from another machine or depot, which is the second unproven suspect).
Unproven: why it passed on 2026-10-18 (no relevant depot file, Julia install or driver module changed since 2026-10-15), and whether the CUDA runtime artifact is damaged (its tree hash does not match its name; its files are unmodified since installation).
Recommended to the owner, not done: no compat bound (the old stack fails the same way); reboot or reset the GPU and rerun a 2×2 `CUBLAS.trsm!` with `ComplexF64`; if it still fails, re-download the CUDA runtime artifact; if it persists, report the 2×2 reproduction to JuliaGPU/CUDA.jl.
Consequently a complex iterative node's derivative pass, mixed pairs and blocks are untested on the device.

**Measured on the KITE** (design acceptance 7 and 8; `test/test_derivatives_kite.jl`, slow tier).
The symbolic KITE in the optimum frame (`test/fixtures/kite/optimum_symbolic.json`, generated once through the Python tier by `test/python/generate_kite_symbolic.jl`), three `Dense` leaves under an `LOBPCG` root (6 computed, 4 kept), at the sweet spot of the flux offset `delta`:
- The doublet's detuning is Δ = 0.0135992 GHz; the root reports the block of states [1, 2] for `delta` (coupling 1.73193, not exact), so the block search runs at a branch and through the complement solve.
- The first derivative in `delta` agrees with finite differences to 4.8e-9 relative and the second to 6.0e-7 (tolerances 1e-5 and 1e-4, Decision 11).
- The curvature `E2[2] − E2[1]` is 882.28074 against `4c²/Δ` = 882.27962 (1.3e-6 relative).
- `block_levels` reproduces a full re-solve within 9.9e-9 GHz at δ = 0.01, where the second-order Taylor expansion misses by 1.17e-2, and within 1.1e-9 at δ = 0.005.
- The `LOBPCG` route agrees with a `Dense`, `retain = :always` route to 3.2e-12 at first order and 1.3e-12 at second; given `scale`, the `Dense` route reports the same blocks, and the two routes' `block_levels` agree to 4.6e-14 (asserted to 1e-9 since the final review).
- At `delta = 0`, `n_g1 = 0`, the symbolic fixture and the numeric `test/fixtures/kite/optimum.json` give the same four lowest levels on one tree with thirteen states on the compact leaf (measured 1.0e-13, asserted 1e-10); the fixture was generated with circuit-quantization at `c8c8714` (branch `to-spec-zero-tolerance`).
- Off the sweet spot (`delta = 0.05`) the mixed derivative `∂²f_01/∂E_J∂delta` is 0.0427752105, within 1.7e-7 relative of finite differences.

**Finite-difference agreement** (the reviewers' independent checks at parameter values the tests do not use; relative to the largest element unless stated).
- First order at one node (Task 4): off-diagonal `expiphi` and `n` port derivatives of a complex leaf 4e-8 to 8e-8; a complex branch with real child data 3e-9 to 7e-8 (removing the gauge term gives errors of 0.11 to 1.0).
- First order through trees (Task 5): the three-level chain along `E_J3` 6.5e-9 and along `g23` 4.3e-8; the periodic two transmons along `E_J2` 6.8e-9; the three-mode tree with joint ports along `E_C` 2.4e-10 (worst over every node's energies, vectors, ports and joint ports).
- Second order at one node (Task 7): a complex leaf's `expiphi` port second derivatives 4.4e-8, 2.8e-9 and 5.0e-9 (removing the gauge term gives 0.33, 4.7e-3 and 7.3e-2, removing the normalization 5.9, 4.5e-2 and 0.69); the closed form `eq:eig_val_secondgrad` to 1e-14.
- Second order through trees (Task 8), a complex two-level tree: `(E_J1, g)` energies 2.1e-8, ports 2.8e-7, vectors 3.5e-6; `(n_g1, g)` energies 5.3e-8, ports 8e-8; symmetric to 2e-16 under swapped roles.
- Augmented children with bases that are not eigen-subspaces (Task 10): a branch child at first order 2e-9, at second order 4e-7 (the finite difference's own error 3e-7); a leaf child at first order 1e-7, at second order within the finite difference's error.
- The final review's cross-feature experiments, each against Richardson finite differences at every node's energies, vectors, ports and joint ports (worst relative error, with the finite difference's own error in parentheses; first order / second order), except E10, which compares with the `Dense` route:
  E1, a three-level tree with complex leaves, an `LOBPCG` interior node and mixed pairs entering at different levels: 3.7e-7 (4.4e-7) / 2.8e-4 (2.1e-4);
  E2, an augmented branch child whose basis is not an eigen-subspace, under an `LOBPCG` root: 8.6e-9 (2.0e-8) / 1.0e-5 (8.0e-6);
  E3, an iterative interior node and root keeping `indices` sets that skip states: 3.3e-6 (4.1e-6) / 8.8e-5 (8.1e-5);
  E4, a parameter entering only through a cosine offset of a periodic-grid leaf, alone and mixed: 4.4e-8 (5.5e-8) / 2.0e-4 (3.4e-3);
  E5, `retain = :never` leaves under an iterative parent: 7.6e-9 (7.8e-9) / 7.3e-5 (5.8e-5);
  E6, `within` windows as the selection of the root and of an interior node: 1.2e-8 (2.0e-8) / 7.5e-6 (7.9e-6);
  E10, the default solvers (`Auto()`: a 1501-point `ShiftInvert` leaf, a 1156-state `LOBPCG` root) at their default tolerances against the `Dense` route: at most 5.1e-11, before and after the change of scale of the linear solve;
  E11, the Zygote gradient through `spectrum` on a tree with an iterative complex root: at most 1.5e-7 (1.4e-7).
  In every finite-difference case the worst error is within a factor of about 1.5 of the finite difference's own error, so these measure the finite differences more than the derivatives.
The tests assert `1e-6` relative where the finite difference is itself good to that (Decision 11); second differences reach `1e-8` to `8e-6` self-consistency on the test circuits, so second-order tests use `rtol = 1e-5` except where `1e-6` was measured to hold, and the KITE doublet tests use `1e-5` and `1e-4`.
Several tolerances are looser than the agreement measured, a note for the owner, not changed: iterative and device vectors and ports 1e-6 / 1e-5 against 1e-11 to 1e-12 measured; KITE `block_levels` 1e-4 against 1e-8, and its curvature 1e-2 against 1.3e-6; trees 1e-6 against at most 4.3e-8; stored-data children 1e-6 / 1e-5 against 2e-9 / 4e-7; the rules 1e-6 against about 1e-8.

**Where the build departed from the design, and why.**

1. *The call* is `derivatives(sys; wrt, ...)`, not the design's illustrative `derivatives(sys, spec, θ; ...)`: the `System` already carries its terms, `θ` and constant (Decision 6).
2. *`scale` is a keyword `NamedTuple` of `δx` per parameter* (Decision 7, Ruling 3); the design marks how `δx` is supplied as not confirmed by the owner, and its own illustrative call does it this way; a change costs a keyword rename.
3. *The `:follow` restriction is narrowed* (the plan's disagreement on D5, Ruling 2): only a node whose phases were actually fixed by following a previous solution (`Retained.followed`) is refused, with an `ArgumentError` whose workaround is to re-materialize at the point without `previous`; a `:follow` node materialized without `previous` used the positive-sector rule and is differentiated, and a `follow` selection is a discrete choice held fixed like any other when its phase is `:positive_sector` given explicitly, or without `previous`; note that `keep = follow(...)` defaults to `phase = :follow`.
   A followed sweep cannot be differentiated directly.
4. *What is retained* (the plan's disagreement on D2): all computed energies, but only the vectors of the unkept states, so a node that keeps all it computes stores nothing extra.
5. *The linear solve* is a hand-written, Jacobi-preconditioned conjugate gradient that runs on any device and does not reuse the `ShiftInvert` factorization (P3, above); each column converges on its own scale `‖Q R_j‖` or stops at the round-off floor (the final review's fixes, above).
6. *A mixed pair needs both members in `wrt`* (the plan's disagreement on D6): the pair needs both first-order responses.
7. *Design §4.8 on Born–Oppenheimer reduced specs is partly wrong* (the plan's disagreement on D8, Ruling 2).
   It says a derivative with respect to a parameter the reduced spec no longer carries is an error by construction; that holds for a `slow_spec`, but a `fit_surfaces` spec keeps the fitted-over parameters symbolically in its fit model, so derivatives with respect to them are derivatives of the fit, not an error.
   No special handling was added.
8. *The KITE fixture* (Decision 10): the plan assumed that the symbolic fixture and `test/fixtures/kite/optimum.json` differ by a flux gauge and pinned the symbolic one by its levels; a reviewer showed they are the same circuit (their coefficients coincide at `delta = 0`, and their four lowest levels agree to 1e-13).
   The level pin stays.
9. *Exact degeneracy* gives `NaN` per-state derivatives with a block, and the `NaN` propagates to every ancestor (P5); the design asks that per-state derivatives be reported as undefined, and says nothing about ancestors.
10. *Accuracy* (Decision 11): the design's target is `1e-6` relative where the finite difference is trustworthy; second-order and KITE tests are looser where it is not (above), and the iterative tests tighten the eigensolver tolerance, so the default `LOBPCG` tolerance is exercised only by the reviewer's check above.
11. *The outer rule* targets ChainRulesCore in an extension (left to the builder by the design), which is always active in practice (above).
12. *Retention is on by default* (Ruling 7, Decision 2), despite its measured cost and the convergence table above; only the solves that are never differentiated (mean-field loop, `bo_surface`) do not retain.
13. *The selection-boundary threshold without `scale`* is a detuning at most `1e-6` times the states' distance from the node's lowest computed energy (P8), a choice of the plan; the design gives no threshold without a scale.
14. *A `wrt` parameter nothing depends on* is an error when the spec does not declare it and zeros with a warning otherwise (ruled after the final review); the design does not mention the case.
STATUS.md lists the same decisions under "Decisions still Eli's".

**Not built, and not tested.**
- Not built, by the design's non-goals: a reverse (adjoint) sweep, third and higher order, derivatives with respect to the frame `S`, and derivatives through basis construction (grids, cutoffs, mean-field bases).
- A truncated, convergence-checked sum over the nearest states (design §4.5's note): measured, not built.
- For the owner, not built: a pairs-only second-order request (today `order = 2` runs the single-parameter pass for every `wrt` parameter); reuse of the `ShiftInvert` factorization; `show` methods for `Derivatives` and `NearDegenerateBlock`; `extra_terms` in `spectrum`; a clear error for differentiating twice through the rule.
- Untested paths a user could reach: a `NaN` child column entering a parent's right-hand side through more than one level (one level, child to parent, is tested at a `Dense` and an iterative parent since the final review); a near-degenerate block at an augmented node; a pair that clashes only in a mixed second-order pass (no available circuit produces one); the terms routed below an augmented node are not checked against those of its bare solve (an `extra_terms` entry routed there would differentiate a different operator than the bare solve used).
- Known limitations: a followed node or an undeclared `GivenSystem` cannot reach `spectrum` (it has no `previous` and takes a `Spec`), so the rules' pass-through of those errors is untested.
  Fixed after the final review: `block_levels` on a block whose effective Hamiltonian is not finite throws an `ArgumentError` naming the cause instead of `eigen`'s "matrix contains Infs or NaNs", and a solver that returns fewer than `compute` pairs gives an `ArgumentError` in `_solve_node` instead of a `BoundsError`.
- Devices: only a real iterative node is tested (above).

**Tests.**
Fast tier: `test_diffcoeff.jl`, `test_retention.jl`, `test_derivoperator.jl`, `test_derivatives_node.jl`, `test_derivatives_tree.jl`, `test_derivatives_complement.jl`, `test_derivatives_second.jl`, `test_derivatives_second_tree.jl`, `test_derivatives_blocks.jl`, `test_derivatives_stored.jl`, `test_spectrum.jl`.
Slow tier (`SLOW_FILES`): `test_derivatives_iterative.jl` (iterative nodes against the dense route and finite differences, both orders, real and complex) and `test_derivatives_kite.jl`.
Own tiers: `test/chainrules/runtests.jl` (the rules, directly and through Zygote) and `test/cuda/test_derivatives_device.jl`.
The finite-difference helpers (`fd_derivative`, `fd_second`, `fd_mixed`, `fd_pair`, `check_fd`, in `test/helpers.jl`) use Richardson-extrapolated central differences and assert both that the difference agrees with itself at two steps and that the derivative agrees with it.
Benches: `bench/retention_cost.jl`, `bench/derivative_sums.jl` and `bench/derivative_design/` (its own environment; not part of any test tier).

## 21. Builder check

A fresh reviewer read the document as a builder on 2026-10-15.
It found one gap that was fixed in the text, Section 11.1 and 11.2 had omitted the gradient path from a branch's ports into its children, and raised three questions, since answered (Section 20), plus one scoping question answered in Section 11.2.
Three minor points it raised were settled as calls: the `phi_zpf` default is fixed at the first `materialize`, the near-degeneracy warning covers kept pairs too, and phase-slip terms are rejected on compact modes.
