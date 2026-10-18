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
# larger leaves can keep the default solver Auto() with compute=10, and keep=connected(...) to follow a port

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

A given system's `energies` must not include a constant that the parent also adds (for example a circuit spec's root constant), because the spec-form `materialize` adds `constant_term(spec)` once.
If the given system is built from a materialized `System`, subtract it: `energies(sys) .- real(sys.constant(sys.θ))`.

A term whose factors all lie on the given mode (a drive or a Stark shift) is applied at its parent through the given operators.

Each node has a solver; the default, `Auto()`, picks one when the node is materialized.
`Auto(; dense_cutoff=2500, on=Array)` resolves to `Dense()` for a node of at most 2 500 product states; above that, to `ShiftInvert()` at a leaf on `Array` storage, `ChebyshevFilter(on=on)` at a leaf on device storage, and `LOBPCG(preconditioner=:ritz, seed=:product, on=on)` at a branch.
An explicit solver always wins, and `compute = 0` (all eigenpairs) needs a node whose solver is, or resolves to, `Dense`.
`ShiftInvert()` is the CPU leaf solver: Lanczos on `(H − σI)⁻¹`, with `H − σI` factored by sparse Cholesky from `sparse(H)` and `σ` below the lowest eigenvalue, in real arithmetic when the assembled matrix is exactly real; it offers spectrum slicing across threads and a completeness certificate by inertia counts.
`ChebyshevFilter(; on=Array)` is the leaf solver for a device (it also runs on the host): Chebyshev-filtered block subspace iteration started from the lowest product states of the separable part, with the filter's upper end from `gershgorin_bound(H)`.
Chosen explicitly, `Dense()` diagonalizes the full product-space matrix: use it for nodes up to about 10 000 states (it warns above that) and whenever every eigenpair is wanted (`compute = 0`).
`Lanczos()` is KrylovKit's single-vector Krylov solver on the host: it needs no tuning, but it has no warm start and does not resolve exactly degenerate eigenvalues.
`LOBPCG(; tol=1e-8, maxiter=1000, preconditioner=:ritz, eta=0.01, guard=4, seed=:product, on=Array, callback=nothing)` is a block solver: it resolves degenerate clusters, takes a warm start, and runs on a GPU with `on=CuArray`.
Its `preconditioner` is `:ritz` (adaptive per-Ritz Jacobi), `:jacobi` (fixed shifted Jacobi) or `:none`.
`eta` regularizes `:ritz` and is in the Hamiltonian's energy units (GHz for these specs), so it must be set again for another energy scale.
`callback` is called with each iteration's `info` named tuple, and its return value is ignored.
`guard` extra columns are converged with the block and then discarded, which makes a missed lower state unlikely without guaranteeing completeness.
With `seed=:product` (the default), a cold block and a warm start's guard columns are the lowest product states of the separable part (`product_seed`: by summed 1D energy at a leaf, by the children's energies at a branch), perturbed by `sqrt(eps)`; `seed=:random` keeps plan 2's random block and random guards.
On a stiff grid leaf the guard columns can set the iteration count and emit the guard warning even on a warm start.
`guard=0` avoids this for warm leaf sweeps at the documented completeness cost; a guard stopping rule is an open item (spec §20c execution notes).
Both preconditioners use only the diagonal of `H`.
They work well at a branch, whose diagonal holds its children's energies, and poorly on a phase-grid leaf, where the kinetic stencil puts the same large constant on every diagonal entry (spec §20e): in the plan-2 benchmark a cold 300-state zero-pi leaf (9331 states) did not converge within the default `maxiter`, while `Dense()` took 69 s and `Lanczos()` 84 s.
When choosing explicitly, use `ShiftInvert` (CPU) or `ChebyshevFilter` (device) at a large, stiff leaf, `LOBPCG` for branches (and for leaves only with a warm start), `Dense` for nodes up to about 10 000 states, and `Lanczos` for a few states when neither leaf solver fits.

A user-defined solver subtypes `Solver` and needs one method, `KronFreeCircuits.solve(s, H, m)` (it may also accept `X0=nothing`, which only `LOBPCG` receives today), returning the lowest `m` energies in ascending order and orthonormal eigenvectors as host arrays, with true residuals below its tolerance.
The `Solver` docstring states the full contract and what `H` provides (`mul!` on blocks, `diagonal`, `sparse`, `gershgorin_bound`, `separable_eigenbases`, `product_seed`).

```julia
using LinearAlgebra
struct MySolver <: Solver end
function KronFreeCircuits.solve(::MySolver, H, m)          # qualified: `solve` is not exported
    F = eigen(Hermitian(Matrix(H)), 1:m)                      # exact, so the residuals are at round-off
    return F.values, F.vectors
end
leaf = Node([:varphi_1 => ChargeBasis(30)]; solver=MySolver(), compute=6, keep=lowest(6))
```

Define the method as `KronFreeCircuits.solve` (or after `import KronFreeCircuits: solve`): an unqualified `function solve(...)` after `using KronFreeCircuits` creates a new function instead of adding a method.

`materialize(root, spec, θ; previous=sys)` warm-starts every `LOBPCG` node (including an `Auto()` branch) from the node at the same tree position in `sys`: its previous kept vectors become the initial block (a node whose product dimensions changed starts cold).
Only kept vectors are stored, so a node with `compute > keep` fills its other `compute − keep` columns randomly and is only partially warm.
`previous` also feeds the `follow` rule and the `:follow` phase convention, which compare physical overlaps with the previous kept states.

On a CUDA machine, `using CUDA, cuTENSOR` loads the package extension, and `LOBPCG(on=CuArray)` or `ChebyshevFilter(on=CuArray)` stores a node's operator and block on the device.
`Auto(on=CuArray)` does so only for a node it resolves to a device solver; a node of at most `dense_cutoff` (2 500) states resolves to `Dense()` on the host.
Energies, eigenvectors and ports return as host arrays; ports are formed on the host after the solve.

## Periodic phase grid

`PeriodicGrid(N; order=8, sectors=1)` is a phase grid for a compact mode, on `[-π, π)` without the endpoint, with central stencils of order 2 to 8 that wrap around the circle.
Use it for non-standard compact modes: a cos 2φ element, higher harmonics, or a numerically supplied 2π-periodic potential given to the spec as a Fourier series of cosines.
The charge basis stays the default, and the better choice, for a single cos φ.
The basis is parameter-free: the offset charge enters only through the coefficients `e^{±2πi n_g}` that term construction puts on the stencil's two wrap-corner pieces (the twisted boundary `ψ(φ + 2π) = e^{-2πi n_g} ψ(φ)`).
`N` must be divisible by every integer harmonic of a cosine on the mode and by `sectors`, so that translations by `2π/ω` map the grid to itself and the charge-parity sectors are exact; a non-integer harmonic on a compact mode is rejected when the spec is loaded.
The operators are complex except at `n_g ∈ {0, ½}`, where the twist phase is ±1.

```julia
spec = load_spec("cos2phi.json")                     # 4 E_C (n - n_g)^2 - E_2 cos 2φ - E_1 cos φ, mode phi
θ = (E_C=1.0, n_g=0.3, E_1=0.5, E_2=10.0)
sys = materialize(Node([:phi => PeriodicGrid(128)]; keep=lowest(8)), spec, θ)
energies(sys)
KronFreeCircuits.charge_port(sys, :phi)              # the charge n in the kept eigenbasis (8 × 8, complex)
```

The basis serves no `:n` port, because the charge is assembled from the derivative pieces and the offset charge.
So `ports=[(:phi, nport())]`, or a selection rule that needs `nport()`, on a `PeriodicGrid` mode throws; use `KronFreeCircuits.charge_port(sys, mode)` (not exported) for the physical charge matrix `V'(D₁)V + n_g I`.

Above the leaf, request the three derivative pieces as ports and assemble the charge from them (here `ng` is the mode's offset charge):

```julia
ports = [(:phi, Port(:d1, 0)), (:phi, Port(:d1wrap, 1)), (:phi, Port(:d1wrap, -1))]   # on the ancestor node
n = port(sys, :phi, Port(:d1, 0)) + cispi(2ng) * port(sys, :phi, Port(:d1wrap, 1)) +
    cispi(-2ng) * port(sys, :phi, Port(:d1wrap, -1)) + ng * I
```

A hand-written coupling `g n ⊗ A` on a twisted mode becomes three terms, whose coefficients carry the twist factors from the exported `twist_factor(ng, ±1)` (with `ng` the offset charge as a `Coefficient`), plus the `n_g I` share folded into the remaining factors:

```julia
c = Coefficient(g); ngc = Coefficient("n_g", [:n_g]); A = Factor(:res, Port(:x, 0))
coupling = [Term(c, [Factor(:phi, Port(:d1, 0)), A]),
            Term(c * twist_factor(ngc, 1), [Factor(:phi, Port(:d1wrap, 1)), A]),
            Term(c * twist_factor(ngc, -1), [Factor(:phi, Port(:d1wrap, -1)), A]),
            Term(c * ngc, [A])]
```

The automatic expansion of `nport()` in `ports=` and `extra_terms` is a planned follow-up.
The design is in spec §20d.

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
Results and the exact commands are in `docs/spikes/2026-10-17-plan2-benchmark/results.md`, and the rerun with the plan-2b leaf solvers (`BENCH_LEAF_SOLVER=shiftinvert`, `chebyshev` or `auto`) is in `docs/spikes/2026-10-17-plan2b-benchmark/results.md`.

```bash
BENCH_SMOKE=1 julia --project=. -t 1 bench/end_to_end.jl                                  # reduced sizes, CPU
BENCH_FIXTURE=path/to/larger-real300-operators.bin julia --project=. -t 1 bench/parent_solve.jl
BENCH_DEVICE=gpu BENCH_FIXTURE=path/to/larger-real300-operators.bin julia --project=test/cuda -t 1 bench/parent_solve.jl
```

## Status and plans

The design is in `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`.
Plan 1 (core CPU pipeline: spec loading, term construction, kron-free application, Dense and Lanczos solvers, selection rules, materialization, dressed/bare interface) is done.
Plan 2 (ITensor-backed operator application, LOBPCG with adaptive Jacobi preconditioning, warm starts and physical-overlap transport, the CUDA extension) is done; its benchmark names the leaf solver and device-side port formation as the next stages to optimize.
Plan 2b (derived operator quantities, the `ShiftInvert` and `ChebyshevFilter` leaf solvers, product-state seeding for `LOBPCG`, the `Auto()` default, the solver extension contract, and `PeriodicGrid`) is done; in its benchmark the two-zero-pi tree materializes in 15 s on CPU and 12.4 s on the A4500, against plan 2's 237 s (`docs/spikes/2026-10-17-plan2b-benchmark/results.md`).
`Auto()`'s `dense_cutoff=2500` is a placeholder still to be tuned, and 3D leaves are tabled (spec §20e decision 8).
KernelAbstractions compat is `0.9, 0.10`: CUDA 6 requires 0.9, so a CUDA environment resolves to 0.9, while under 0.10 the CPU backend runs the kernels through POCL.
Later plans:

1. Automatic differentiation rules.
2. Export, caching, and NPZ output.
3. The Python exporter in `circuit-quantization`.
