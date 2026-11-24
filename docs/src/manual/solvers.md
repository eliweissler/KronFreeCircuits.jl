# Solvers

Each node has a solver.
The default, `Auto()`, picks one when the node is materialized; an explicit solver always wins.

## Auto

`Auto(; dense_cutoff=1000, on=Array)` resolves to:

- `Dense()` for a node of at most 1000 product states;
- `ShiftInvert()` for a larger leaf on `Array` storage;
- `ChebyshevFilter(on=on)` for a larger leaf on device storage;
- `LOBPCG(preconditioner=:ritz, seed=:product, on=on)` for a larger branch.

`compute = 0` (all eigenpairs) needs a node whose solver is, or resolves to, `Dense`.

## The solvers

**`Dense()`** diagonalizes the full product-space matrix.
Use it for nodes up to about 10 000 states (it warns above that) and whenever every eigenpair is wanted.

**`ShiftInvert()`** is the CPU leaf solver: Lanczos on `(H − σI)⁻¹`, with `H − σI` factored by sparse Cholesky and `σ` below the lowest eigenvalue, in the operator's element type (real for a real node, see [Real arithmetic](real_arithmetic.md)).
It offers spectrum slicing across threads and a completeness certificate by inertia counts, on by default with the automatic shift, which recovers the copies of a degenerate level that single-vector Lanczos misses.

**`ChebyshevFilter(; on=Array)`** is the leaf solver for a GPU (it also runs on the host): Chebyshev-filtered block subspace iteration, started from the lowest product states of the separable part with a `1e-2` random admixture and guard columns, so a lower state outside the seed span usually surfaces.
It has no completeness certificate.
Its defaults were tested against invariant-seed traps, weak filters, crossing sweeps, degenerate blocks and deep references without a wrong result, but this is tested, not proven: use `tol ≤ 1e-6` and the default or a larger `extra`, and `ShiftInvert()` or `Dense()` when completeness matters.

**`LOBPCG(; tol=1e-8, maxiter=1000, preconditioner=:ritz, eta=0.01, guard=4, seed=:product, on=Array, callback=nothing)`** is a block solver: it resolves degenerate clusters, takes a warm start, and runs on a GPU with `on=CuArray`.

- `preconditioner` is `:ritz` (adaptive per-Ritz Jacobi), `:jacobi` (fixed shifted Jacobi) or `:none`. Both preconditioners use only the diagonal of `H`: they work well at a branch, whose diagonal holds its children's energies, and poorly on a phase-grid leaf, where the kinetic stencil puts the same large constant on every diagonal entry.
- `eta` regularizes `:ritz` and is in the Hamiltonian's energy units (GHz for these specs), so it must be set again for another energy scale.
- `guard` extra columns are converged with the block and then discarded, which makes a missed lower state unlikely without guaranteeing completeness. On a stiff grid leaf the guard columns can set the iteration count; `guard=0` avoids this for warm leaf sweeps at the cost of that safety.
- `seed=:product` starts a cold block from the lowest product states of the separable part (`product_seed`), perturbed by `sqrt(eps)`; `seed=:random` starts from a random block.
- `callback` is called with each iteration's `info` named tuple.

**`Lanczos()`** is KrylovKit's single-vector Krylov solver on the host: it needs no tuning, but it has no warm start and does not resolve exactly degenerate eigenvalues.

## Choosing

| node | solver |
|:-----|:-------|
| up to about 10 000 states, or all eigenpairs wanted | `Dense()` |
| a large, stiff leaf on the CPU | `ShiftInvert()` |
| a large leaf on a GPU | `ChebyshevFilter(on=CuArray)` |
| a branch, or a leaf with a warm start | `LOBPCG()` |
| a few states, when neither leaf solver fits | `Lanczos()` |

A zero-pi qubit whose `(θ, φ)` leaf has 15 531 product states, beside its decoupled harmonic mode `ζ`.
`KronFreeCircuits.node_timings` records, for each node solved, the solver `Auto()` chose and the time it took:

```@example solvers
using KronFreeCircuits
spec = load_spec("zero_pi.json")
θ = (E_J=10.0, E_L=0.04, E_CJ=20.0, E_CS=0.5, varphi_ext=2π * 0.23, n_g=0.1)
zpf = (2real(spec.E_C[3, 3](θ)) / real(spec.E_L[3, 3](θ)))^(1 / 4)          # zero-point width of ζ
leaf = Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-10π, 10π, 501)]; compute=6, keep=lowest(6))
zeta = Node([:zeta => FockBasis(10; phi_zpf=zpf)]; keep=lowest(2))
sys, records = KronFreeCircuits.node_timings(() -> materialize(Node([leaf, zeta]; keep=lowest(6)), spec, θ))
[(r.modes, r.dim, r.solver) for r in records]
```

## GPUs

On a CUDA machine, `using CUDA, cuTENSOR` loads the package extension, and `LOBPCG(on=CuArray)` or `ChebyshevFilter(on=CuArray)` stores a node's operator and block on the device.
`Auto(on=CuArray)` does so only for a node it resolves to a device solver; a node of at most `dense_cutoff` states resolves to `Dense()` on the host.
Energies, eigenvectors and ports return as host arrays; ports are formed on the host after the solve.

## Your own solver

A solver is a subtype of `KronFreeCircuits.Solver` with one method; see [Extending](@ref).
