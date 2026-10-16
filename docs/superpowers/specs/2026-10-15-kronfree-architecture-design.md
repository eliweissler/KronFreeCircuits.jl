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
This is in scope for this project even though the Python package otherwise stays as it is.
Coefficients are written as Julia source strings produced by SymPy's Julia code printer, so they remain symbolic in the circuit parameters.
The function must refuse to export if any cosine coefficient vector (Section 5.2, `omega`) or any charge-cosine vector (`kappa`) contains a free symbol, because parameter-dependent vectors would put parameters into local operators (Section 11).

### 5.2 Content

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

- `DenseOp`: a dense `ComplexF64` matrix. Always complex, never real, because BLAS has no mixed real-complex gemm and the generic fallback is tens of times slower.
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

Fusion of all structured terms of a leaf into one kernel pass is a recognized later optimization, not part of the first version.

## 9. Diagonalization (decided)

One internal interface: given a node's apply and preconditioner, return the lowest $m$ eigenpairs.
Three implementations, all available at every node, chosen per node by setting or by heuristic:

| Solver | When (default heuristic) | Implementation |
|---|---|---|
| `Dense()` | product dimension $\le$ threshold (default 10 000 **(my call)**) | build the explicit matrix by applying $H$ to basis vectors or by direct Kronecker assembly of the small factors; `LinearAlgebra.eigen` (LAPACK) on CPU, cuSOLVER through the CUDA extension on GPU; returns *all* eigenpairs |
| `Lanczos()` | fewer than about 50 states wanted **(my call)** and dimension above threshold | KrylovKit `eigsolve` with `Lanczos`, vector-at-a-time |
| `LOBPCG(; chunk)` | otherwise | LOBPCGEigensolver.jl (JuliaMolSim, extracted from DFTK, GPU-capable, MIT), block preconditioned, with locking |

Preconditioner: Jacobi, $(\operatorname{diag}H - \sigma)^{-1}$.
The diagonal of a sum of products is the sum of products of the local diagonals, so it is available at every node from the operator representation.
At a branch it is exactly $E_i^{(1)} + E_j^{(2)} + \dots$, and the couplings are a perturbation on it, which is why LOBPCG converges in tens of iterations there.
Applying it is a broadcast.

Chunked LOBPCG with deflation (decided): when `chunk` is smaller than $m$, solve for the lowest `chunk` states, then the next `chunk` orthogonal to those, and so on.
LOBPCGEigensolver has no constraint argument, so deflation is done on the operator side by adding $\sigma\,P_{\rm found}$ to the operator, one extra block inner product per application.
Memory: LOBPCGEigensolver holds roughly ten blocks of $M$ vectors; at 90 000 dimensions a single 2000-vector block is about 29 GB (fits a data-center card, not the A4500), while chunks of 500 plus the 2000 found vectors are about 10 GB.
The default is a single chunk when it fits in device memory **(my call)**.

Every node carries two counts: `compute`, how many eigenpairs the solver returns, and the selection rule (Section 9.1), which picks the kept subset.
The dense solver ignores `compute` and returns everything.

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
- All state and dense-operator element types are `ComplexF64`.

## 15. Packages (decided)

Core dependencies: `LinearAlgebra`, `SparseArrays` (stdlib); `KernelAbstractions`; `KrylovKit`; `LOBPCGEigensolver`; `ChainRulesCore`; `Adapt`; `GPUArraysCore`; `JSON3`; `RuntimeGeneratedFunctions`.

Extensions (weak dependencies): `CUDA` (strided-batched gemm, cuSOLVER dense eigen); `QuantumToolbox`; `QuantumOptics`.

Test-only: `ITensors` (reference apply), `FiniteDifferences` and `ChainRulesTestUtils` (rules), `Zygote` (engine under test), `Test`.

Deliberately absent: `NNlib` (the batched gemm is ten lines in the CUDA extension), `ITensors` at runtime, `PythonCall`, `IterativeSolvers` (unmaintained), `Symbolics`, `QuantumToolbox` as the operator container.

Python side: `circuit-quantization` gains `export_julia_spec`, using the standard-library `json` module and SymPy's `julia_code` printer.

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

## 21. Builder check

A fresh reviewer read the document as a builder on 2026-10-15.
It found one gap that was fixed in the text, Section 11.1 and 11.2 had omitted the gradient path from a branch's ports into its children, and raised three questions, since answered (Section 20), plus one scoping question answered in Section 11.2.
Three minor points it raised were settled as calls: the `phi_zpf` default is fixed at the first `materialize`, the near-degeneracy warning covers kept pairs too, and phase-slip terms are rejected on compact modes.
