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
- Float32 or mixed precision. Everything is Float64 / ComplexF64.

### Anti-goals (would make this a failure even if it worked)

- A "mishmash" of device-specific backends. There must be one apply interface whose implementations are chosen by operator *structure*, never by device.
- Any code path that silently falls back to generic indexing of a lazy matrix (the failure mode of the `mat_free_hacking` AbstractMatrix experiment).
- A runtime dependency on Python or on ITensor.
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
| `PeriodicGrid(N; order, sectors)` | compact (chosen by the user) | not served: term construction assembles it from `d1`, `d1wrap(±1)` and the offset charge | not served: assembled from `d2`, `d2wrap(±1)` | `phi`, `phi2` not available (φ is multivalued on the circle) | diagonal, integer $k$ | not available |

`PeriodicGrid` (added 2026-10-17) is specified in Section 20d.

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

### 11.1 Where parameters live

Circuit parameters and external fluxes enter only through term coefficients.
Basis hyperparameters (grid range and spacing, cutoffs, `phi_zpf`) are fixed and never differentiated.
Therefore at a **leaf**, $\partial H/\partial\theta_p = \sum_t (\partial c_t/\partial\theta_p)\,O_t$ is a sum of products over the same, parameter-free local operators.

At a **branch** the local operators are the children's materialized ports, $V^\dagger O V$ and the children's energy diagonals, and these do depend on $\theta$ through the children's eigenvectors and eigenvalues.
So the gradient path through a branch has three legs: the branch's own coefficients, the dense port operators, and the energy diagonals.
The last two carry the cotangent down into the children.
A design that only propagated coefficient cotangents would silently return zero for every parameter that enters only through a child (for example $\partial E_{\rm top}/\partial E_{J1}$); this must be a tested failure mode.

### 11.2 Three primitives with ChainRules rules

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

The selection rule is a discrete function of $\theta$ and is treated like an argmax: the forward pass records the kept indices and the derivative flows only through the selected energies, eigenvectors, and ports.
This is the exact derivative wherever one exists.
The package must warn when a kept and an unkept state are within a tolerance of each other in energy at a node, since the derivative is unreliable there.
Pointwise derivatives in a sweep are exact at each point; only finite differences across points can mix selections.
Intermediate truncations are robust to relabeling inside the kept set, because the parent only sees the kept subspace.

### 11.4 Near-degenerate kept states

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

### 13.1 Future-friendliness: Born-Oppenheimer surfaces (decided requirements, not first-version work)

A Born-Oppenheimer surface treats a slow mode's coordinate as a classical parameter: for each grid point of the slow mode, diagonalize the fast modes and record the energies $E_k(\varphi_{\rm slow})$.
In this design every operator of a grid mode other than its kinetic term is a `DiagonalOp`, so fixing the slow coordinate is a transformation on the term list, and the fast problem is an ordinary node whose coefficients depend on one extra named parameter.
The surface is `energies(materialize(fast, spec′, (θ..., φ_slow=x)))` swept over `x`; its slope is the Hellmann-Feynman derivative the eigenpair rule already gives, and derivative couplings later are the eigenvector derivative from the same rule.
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

Extensions (weak dependencies): `CUDA` (strided-batched gemm, cuSOLVER dense eigen); `QuantumToolbox`; `QuantumOptics`.

Test-only: `ITensors` (reference apply), `FiniteDifferences` and `ChainRulesTestUtils` (rules), `Zygote` (engine under test), `Test`.

Deliberately absent: `NNlib` (the batched gemm is ten lines in the CUDA extension), `ITensors` at runtime, `PythonCall`, `IterativeSolvers` (unmaintained), `Symbolics`, `QuantumToolbox` as the operator container.

Python side: `circuit-quantization` gains `export_julia_spec`, using the standard-library `json` module and SymPy's `julia_code` printer.
As built, that exporter is `to_spec` and `save_spec` (Section 20g), superseding `export_julia_spec`.

## 16. Testing strategy (decided)

- **Differential tests of apply** against the explicit Kronecker product for small systems, for every structure type and every axis position, with and without a block axis, on CPU; the same on GPU when run on a CUDA machine.
- **Solver tests**: all three solvers agree with each other and with the dense reference on small nodes; LOBPCG chunked equals LOBPCG unchunked.
- **Physics tests** against reference values from scqubits for the fluxonium, the transmon, and the zero-pi (single qubit and the two-zero-pi composite in the product basis), committed as data files under `test/fixtures` with the generating Python script alongside; no Python at test time.
- **Rule tests** with `ChainRulesTestUtils` for apply and eigenpairs, and end-to-end finite-difference checks of `Zygote.gradient` of energies and matrix elements with respect to a circuit parameter and a flux.
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
- Section 4 / 7.3: composite ports (a term routed to a branch with two factors inside the same child) are not supported in plan 1 and raise a clear error; multi-mode port keys are design debt for a later plan. `Factor` carries only a single-mode `Port`, so a data-valued operator (Section 13.1's surface fed back as a potential) has no entry into a `Term` yet; the mechanism is to be designed before plan 2 wraps `materialize` in differentiation rules.
- Section 5.3: coefficient expressions are restricted to an allowlist of arithmetic and elementary functions and may not execute arbitrary code; the allowlist must include SymPy's dotted operators (`.* ./ .^ .+ .-`), which plan 1's implementation still lacks (ledger: parked, load-bearing for plan 5).
- Section 6: the grid `exp2pin(k)` shift has the opposite sign from the plan's first draft (`ShiftOp(N, -2πk/Δ)`), fixed during implementation and pinned by a plane-wave test; the Fock `n` port is the charge operator $i n_{\rm zpf}(a^\dagger - a)$, not the number operator.
- Section 16: the two-zero-pi acceptance test is gated behind `KRONFREE_SLOW_TESTS=1`; the ITensor reference implementation is deferred to plan 3. The scqubits references were generated with scqubits 4.3.1 from `/opt/anaconda3/envs/circuit_quantization`; the zero-pi leaf test does not pin port sign conventions (the symmetric spectrum is invariant under θ→−θ and φ→−φ), so matrix-element sign tests belong in plan 4.

### 20b. Amendments and decisions of 2026-10-16 (plan 1b and design discussion)

- Section 9.1, branch phase convention: at a branch, `:positive_sector` fixes each kept state's phase by its exact full-space amplitude at the joint peak of its dominant product-state component, computed from the children's stored peak points and k×k peak-amplitude matrices (a child's amplitude at every other kept state's peak). This is invariant to a child state's sign flip, recursive, and costs k² per node with no grid-sized arrays above the leaves. A diagnostic warns when the amplitude at the chosen point is below one half of the dominant component's own contribution (heavy hybridization). Known limitation: an antisymmetric exchange combination of two identical subsystems can have vanishing amplitude at that point, leaving its phase arbitrary (observed for state 15 of the two-zero-pi test system); resolution pending, either a fallback to the joint peaks of the next few components or the tabled exact search.
- Tabled: the exact full-space positive-sector maximum by Cauchy–Schwarz branch-and-bound over sector grid points (plan 1b Task 3), with the candidate set cut by 2ⁿ for n grid axes by enumerating only all-positive coordinates; expected to terminate quickly when peaks line up, capped with a fallback otherwise.
- Section 16, testing philosophy: the default suite compares against scqubits on identical small discretizations (same range and point count, scqubits' 7-point stencil = our `order=6`), where agreement is at the 1e-12 level and independent of physical convergence; physical-correctness and convergence comparisons (different stencils on a shared grid, wide converged grids) are gated slow tests to be grown over time. Composite-system comparisons with nonzero offset charge differ from scqubits' construction by the truncation order of the linear n_g cross terms (routed into our leaves before truncation); exact hierarchical agreement is pinned with offset charges set to zero.
- Section 11, flux derivatives: external flux enters only the cosine offsets and the inductive linear term, so ∂H/∂φ_ext is the same operator pair with differentiated coefficients and Hellmann–Feynman gives ∂E/∂φ_ext = −E δ′⟨sin(ωφ+δ)⟩ with no basis-dependent pieces; coefficient derivatives will be computed with ForwardDiff inside a ChainRules rule for coefficient evaluation (the compiled coefficients are generated functions); second derivatives for flux-noise dephasing via forward-over-reverse are a plan-2 target.
- Section 5.2, noise operators (for plan 5 and a later noise plan): the spec gains an `operators` section of named general term lists (coefficient expression plus (mode, port) factors) carrying analytic derivative operators (∂H/∂φ_ext as sine terms, ∂H/∂E_J, ∂H/∂n_g) and element operators such as the charge on a lossy capacitor, the flux across a lossy inductor, and sin(φ_J/2) across a junction; plus a `noise_channels` table mapping elements and loops to those operators and parameters. Julia derives the same derivative operators automatically for cross-checking and implements the rate formulas.
- Section 6, half-integer ports: sin(φ_J/2) needs e^{±iω_jφ_j/2}; legal on grids and in Fock; on compact modes it requires a doubled charge basis on half-integer charges (the odd-parity sector is the same circuit at n_g shifted by one half, and the operator connects the sectors, as in quasiparticle tunneling); to be added as a basis variant only when a quasiparticle channel is requested.
- Architecture decisions reaffirmed: the JSON spec remains the Python–Julia contract (the topological, parameter-independent content from Python; parameter-dependent coefficients compiled in Julia); a PythonCall extension for in-memory spec transfer is an optional convenience, never a dependency; porting the symbolic quantization machinery to Julia is rejected unless parameter-dependent coordinate choices become first-class, in which case only that numeric step moves.

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
That exporter became `circuit-quantization`'s `to_spec` and `save_spec` (Section 20g), superseding `export_julia_spec`.

### 20d. Proposed design: a periodic phase grid for compact modes (2026-10-17, for Eli's review)

Motivation (Eli, 2026-10-17): non-standard compact modes (cos 2φ, higher harmonics, numerically supplied 2π-periodic potentials) are numerically finicky in the charge basis; a wraparound phase grid is preferred, accepting complex arithmetic at generic offset charge. The convention is the thesis listing (`code-snippets/chapter_0.py`, `transmon_phase_fd`): grid on [−π, π) without the endpoint, the plain real second-derivative stencil, and the offset charge entering only through the twisted boundary ψ(φ+2π) = e^{−2πi n_g} ψ(φ), realized as the phase e^{±2πi n_g} on the stencil entries that wrap around.

- **Basis.** `PeriodicGrid(N; order=8)` is a `Basis` for `:c` modes beside `ChargeBasis`, chosen by the user (the charge basis stays the default for a single cos φ). Coordinates φ_i = −π + 2π i/N, spacing 2π/N, central stencils of the given order wrapping around; no `n_g` in the basis, so basis hyperparameters stay parameter-free.
- **Twist without parameters in operator data (the rule of Section 11.1 is kept).** A wrapped derivative stencil splits into three parameter-free local operators: the interior band and the two wrap corners W₊, W₋. A derivative term on a twisted mode expands into three terms with coefficients multiplied by {1, e^{2πi n_g}, e^{−2πi n_g}}. In the twisted gauge every occurrence of (n_m − n_g,m) in the canonical quadratic form 4 Σ_ij (n_i − n_g,i) E_C[i,j] (n_j − n_g,j) is replaced by the twisted derivative D_m (so the diagonal kinetic term is 4E_C[m,m] times the twisted second derivative, and a cross term with another mode j is 8E_C[m,j] D_m ⊗ (n_j − n_g,j)); the canonical expansion into linear and constant pieces is performed only for non-twisted modes. The physical charge port exposed for couplings is n = −i D₁(twisted) + n_g I, so a coupling factor on a twisted mode expands into the three derivative pieces (parameter-free ports the leaf forms in its eigenbasis) plus an identity piece whose coefficient n_g multiplies the remaining factors of the term; a convenience accessor assembles the physical charge matrix V†D₁V + n_g I for the user. `e^{ikφ}` for integer k is diagonal and unaffected; half-integer k (quasiparticle ports, Section 20b) connects twist sectors and stays a later item. Hermitian by construction: W₋ is the adjoint of W₊ and the coefficients are conjugate.
- **Where the expansion lives.** Term construction becomes basis-aware for compact modes on a `PeriodicGrid`: `circuit_terms(spec, bases)` (or a post-pass over the tree's bases in `materialize`) performs the expansion; `localop(::PeriodicGrid, port)` serves the three derivative pieces under distinct port kinds (`:d2`, `:d2wrap` with k = ±1, `:d1`, `:d1wrap`) and the diagonal φ-function ports. Coefficient derivatives with respect to n_g flow through the e^{±2πi n_g} coefficients exactly as other parameters do.
- **Sectors (Eli's request).** Custom 2π-periodic elements are supplied as Fourier series, i.e. as cosine terms with integer harmonics (a numerically supplied potential is expanded on the Python side), so the periodicity is known a priori. The loader requires N divisible by every integer harmonic ω appearing in a cosine on that mode (from `CosineSpec.omega`), so the translation by 2π/ω maps grid to grid and the ω charge-parity sectors are exact; an informative error otherwise. An optional `sectors=` argument on `PeriodicGrid` adds a divisibility requirement beyond the declared harmonics. Non-integer harmonics on a compact mode are rejected at load time.
- **Arithmetic and structure.** Complex except at n_g ∈ {0, ½}, where the twist phase is ±1 and the operators are real (relevant to the real-arithmetic plan). The wrapped corners make the mode's operators sparse but not banded; a leaf solver that relies on bandedness (Section 20c notes, the leaf-solver spike) must use the sparse path for such leaves.
- **Validation.** Against the charge basis on identical physics: the free rotor, cos φ, pure cos 2φ with both sectors, cos 2φ with a weak cos φ, at n_g ∈ {0, ½, generic}, including convergence in N and the exact half-period translation symmetry on even N; ports and gauge convention cross-checked against the charge basis (the audit's REAL-WARM report lists these cases).

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
A circuit that is complex by structure but real at special parameter values (a `PeriodicGrid` mode at a parameter `n_g` that equals 0 or ½, a charge-basis SQUID at zero flux) stays complex; using such special values is a recorded later item, not part of plan 3 (Eli: "a small benefit for what I see as a big headache").
A spec literal is structure: a `PeriodicGrid` mode whose `n_g` entry is the number `"0.5"` has the constant twist phase −1 and qualifies.

**Port character (decided).** Every basis declares, for every port it serves, how complex conjugation acts on its operator: `conj(O_p) = σ_p O_{π(p)}` with a sign `σ_p = ±1` and an involution `π` on ports.
A port is *real* (`σ = +1`, `π(p) = p`), *imaginary* (`σ = −1`, `π(p) = p`), or *paired* (`σ = +1`, `π(p) ≠ p`, its conjugate partner); anything else is *general* and has no relation.
Section 20c's three characters (real, imaginary, general) gain *paired*, because a single `expiphi(k)` on a grid is complex while the cosine it belongs to is real.

| Basis | real | imaginary | paired (`p ↦ π(p)`) |
|---|---|---|---|
| `ChargeBasis` | `n`, `n2`, `expiphi(k)` | | `exp2pin(k) ↦ exp2pin(−k)` |
| `PhaseGrid` | `n2`, `phi`, `phi2`, `exp2pin(k)` | `n` | `expiphi(k) ↦ expiphi(−k)` |
| `PeriodicGrid` | `d2`, `d2wrap(±1)` | `d1`, `d1wrap(±1)` | `expiphi(k) ↦ expiphi(−k)` |
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
Complex structure in a circuit therefore enters only through the package's own phase factors (`phase_factor`, `twist_factor`) and through the bases' operators.
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

**What stays complex.** Complex nodes; imaginary and paired ports in their natural type; `charge_port` on a `PeriodicGrid` mode; derivative operators and the autodiff pullbacks of plan 4, which promote to complex where a cotangent or a derivative operator is complex.

**Validation (decided).** For every fixture with a real node, materializing with `arithmetic=:auto` and with `arithmetic=:complex` gives energies equal to about 1e-12 with `Dense` and to the solver tolerance otherwise, ports equal up to the per-state signs (inside an exactly degenerate level, where the two paths may pick different bases, the comparison uses rotation-invariant quantities such as the cluster projector and the port's block norms), and the expected arithmetic per node (the classification above is a test).
Tests that relied on `isreal_assembled` change meaning: the `PeriodicGrid` transmon at a parameter `n_g` of 0 or ½ is now complex by structure, so those assertions become element-type checks (the exact-zero imaginary part of its assembled matrix can still be asserted as a property of the operator data).
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
Nonsymmetric `E_C` and `E_L` matrices are rejected (non-physical; Eli, 2026-10-19): `parse_spec` requires entries `[i,j]` and `[j,i]` to be the same expression or to agree at the three test points of the node test (Section 20f), and its `ArgumentError` names the matrix, the pair and both expressions.

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

**Performance workaround.** `to_spec(res, subs=...)` does not finish on two_zero_pi (over 10 minutes), because `sympy.cancel` stalls on the substituted expression, while `sympy.cancel(sympy.together(expr))` takes 0.05 s.
The generator applies `together` to every expression before `to_spec(res)` for that fixture only.
`circuit-quantization`'s fix (`together` before `cancel` in `to_spec`) is written but not yet committed as of 2026-10-19; once it is committed, the generator can drop the workaround and regenerate every file against the new commit.

## 21. Builder check

A fresh reviewer read the document as a builder on 2026-10-15.
It found one gap that was fixed in the text, Section 11.1 and 11.2 had omitted the gradient path from a branch's ports into its children, and raised three questions, since answered (Section 20), plus one scoping question answered in Section 11.2.
Three minor points it raised were settled as calls: the `phi_zpf` default is fixed at the first `materialize`, the near-degeneracy warning covers kept pairs too, and phase-slip terms are rejected on compact modes.
