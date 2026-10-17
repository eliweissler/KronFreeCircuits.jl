# KronFreeCircuits.jl

Kron-free hierarchical diagonalization of superconducting circuits.
A circuit is a tree of nodes: each leaf diagonalizes a few modes (charge basis, phase grid, or Fock basis), and each parent combines its children's kept eigenstates.
Operators are never assembled as Kronecker products; they are applied axis by axis on the tensor of the mode basis, so composite systems with large Hilbert spaces stay tractable.
Hamiltonian terms come from a JSON circuit spec, which is exported by the `circuit-quantization` package.
Until that exporter exists, the specs in `test/fixtures` are written by hand.

## Sketch

```julia
using KronFreeCircuits

spec = load_spec("two_transmons.json")
θ = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.0, g=0.3)   # one entry per spec parameter

# each leaf keeps its 8 lowest states; for a phase-grid mode use PhaseGrid(-6π, 6π, 301),
# for a harmonic mode FockBasis(20; phi_zpf=0.5)
leaf(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(8))
top = Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(20), ports=[(:varphi_1, nport())])
# larger leaves can use solver=Lanczos(), compute=10, and keep=connected(...) to follow a port

sys = materialize(top, spec, θ)
energies(sys)                          # kept eigenvalues of the top node
port(sys, :varphi_1, nport())          # charge operator of mode varphi_1 in the kept eigenbasis
decompose(sys, top, 1)                 # bare product-state content of dressed state 1
dressed_index(sys, top, (2, 1))        # dressed state that best matches bare state (2, 1)
```

Selection rules are `lowest`, `indices`, `connected`, and `follow`.
By default each kept state's phase makes its full-space amplitude real and positive at one point: at a leaf, its largest entry with nonnegative coordinates; at a branch, the joint peak of its dominant product component.
The spec's scalar constant is added at the root only, so the `energies` of child nodes exclude it.
Bare-state bookkeeping (labels, overlaps) is available through `bare_labels` and `bare_overlaps`.

A pre-diagonalized subsystem (a resonator, a qubit from another tool, a subsystem exported from scqubits) enters the tree as a `GivenSystem`, a child beside nodes and `mode => basis` pairs.
It takes its ascending energies and a dictionary from port kind to its matrix in that eigenbasis; it keeps all its states, with identity eigenvectors.
Coupling terms name its mode and a port kind it provides, and the spec-form `materialize` takes them as `extra_terms`:

```julia
using LinearAlgebra
spec = load_spec("transmon.json")                             # mode varphi_1
θ = (E_J=30.0, E_C=1.2, n_g=0.25)
a = diagm(1 => sqrt.(1.0:4.0))                                # resonator annihilation operator, 5 levels
res = GivenSystem(:res; energies=6.0 .* (0:4), operators=Dict(:x => a + a'))
qubit = Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(6))
top = Node([qubit, res]; solver=Dense(), keep=lowest(12))
coupling = [Term(Coefficient(0.05), [Factor(:varphi_1, nport()), Factor(:res, Port(:x, 0))])]   # 0.05 n (a + a†)
sys = materialize(top, spec, θ; extra_terms=coupling)
```

A term whose factors all lie on the given mode (a drive or a Stark shift) is applied at its parent through the given operators.

Each node has one of three solvers.
`Dense()` diagonalizes the full product-space matrix: use it for nodes up to about 10 000 states (it warns above that) and whenever every eigenpair is wanted (`compute = 0`).
`Lanczos()` is KrylovKit's single-vector Krylov solver on the host: it needs no tuning, but it has no warm start and does not resolve exactly degenerate eigenvalues.
`LOBPCG(; tol=1e-8, maxiter=1000, preconditioner=:ritz, eta=0.01, guard=4, on=Array, callback=nothing)` is a block solver: it resolves degenerate clusters, takes a warm start, and runs on a GPU with `on=CuArray`.
Its `preconditioner` is `:ritz` (adaptive per-Ritz Jacobi), `:jacobi` (fixed shifted Jacobi) or `:none`.
`eta` regularizes `:ritz` and is in the Hamiltonian's energy units (GHz for these specs), so it must be set again for another energy scale.
`callback` is called with each iteration's `info` named tuple, and its return value is ignored.
`guard` extra random columns are converged with the block and then discarded, which makes a missed lower state unlikely without guaranteeing completeness.
Both preconditioners use only the diagonal of `H`.
They work well at a branch, whose diagonal holds its children's energies, and poorly on a phase-grid leaf, plausibly (this is untested) because the kinetic stencil dominates there: in the plan-2 benchmark a cold 300-state zero-pi leaf (9331 states) did not converge within the default `maxiter`, while `Dense()` took 69 s and `Lanczos()` 84 s.
Use `Dense` for nodes up to about 10 000 states, `LOBPCG` for branches (and for leaves only with a warm start or a future grid-aware preconditioner), and `Lanczos` for a few states of a large, stiff leaf.

`materialize(root, spec, θ; previous=sys)` warm-starts every `LOBPCG` node from the node at the same tree position in `sys`: its previous kept vectors become the initial block (a node whose product dimensions changed starts cold).
Only kept vectors are stored, so a node with `compute > keep` fills its other `compute − keep` columns randomly and is only partially warm.
`previous` also feeds the `follow` rule and the `:follow` phase convention, which compare physical overlaps with the previous kept states.

On a CUDA machine, `using CUDA, cuTENSOR` loads the package extension, and `LOBPCG(on=CuArray)` stores a node's operator and block on the device.
Energies, eigenvectors and ports return as host arrays; ports are formed on the host after the solve.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

Setting `KRONFREE_SLOW_TESTS=1` adds the two-coupled-zero-pi acceptance test against scqubits, which takes about a minute.

The device tests have their own environment and need a CUDA device:

```bash
julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
```

Two benchmark scripts take their settings from environment variables, documented at the top of each file.
`bench/parent_solve.jl` solves the audit's stored 300 × 300 parent cold and after a 1 % coupling step, with both preconditioners; `bench/end_to_end.jl` materializes two coupled zero-pis and times each stage separately.
Results and the exact commands are in `docs/spikes/2026-10-17-plan2-benchmark/results.md`.

```bash
BENCH_SMOKE=1 julia --project=. -t 1 bench/end_to_end.jl                                  # reduced sizes, CPU
BENCH_FIXTURE=path/to/larger-real300-operators.bin julia --project=. -t 1 bench/parent_solve.jl
BENCH_DEVICE=gpu BENCH_FIXTURE=path/to/larger-real300-operators.bin julia --project=test/cuda -t 1 bench/parent_solve.jl
```

## Status and plans

The design is in `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`.
Plan 1 (core CPU pipeline: spec loading, term construction, kron-free application, Dense and Lanczos solvers, selection rules, materialization, dressed/bare interface) is done.
Plan 2 (ITensor-backed operator application, LOBPCG with adaptive Jacobi preconditioning, warm starts and physical-overlap transport, the CUDA extension) is done; its benchmark names the leaf solver and device-side port formation as the next stages to optimize.
KernelAbstractions compat is `0.9, 0.10`: CUDA 6 requires 0.9, so a CUDA environment resolves to 0.9, while under 0.10 the CPU backend runs the kernels through POCL.
Later plans:

1. Automatic differentiation rules.
2. Export, caching, and NPZ output.
3. The Python exporter in `circuit-quantization`.
