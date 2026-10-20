# Status

This is a personal research project and largely implemented using Claude Code. Use at your own risk!

# KronFreeCircuits.jl

Kron-free hierarchical diagonalization of superconducting circuits.
A circuit is a tree of nodes: each leaf diagonalizes a few modes (charge basis, phase grid, or Fock basis), and each parent combines its children's kept eigenstates.
Operators are never assembled as Kronecker products; they are applied axis by axis on the tensor of the mode basis, so composite systems with large Hilbert spaces stay tractable.
Hamiltonian terms come from a JSON circuit spec, which is exported by the `circuit-quantization` package.
The hand-written specs in `test/fixtures` remain as the test inputs.

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
`Auto(; dense_cutoff=1000, on=Array)` resolves to `Dense()` for a node of at most 1 000 product states; above that, to `ShiftInvert()` at a leaf on `Array` storage, `ChebyshevFilter(on=on)` at a leaf on device storage, and `LOBPCG(preconditioner=:ritz, seed=:product, on=on)` at a branch.
An explicit solver always wins, and `compute = 0` (all eigenpairs) needs a node whose solver is, or resolves to, `Dense`.
`ShiftInvert()` is the CPU leaf solver: Lanczos on `(H − σI)⁻¹`, with `H − σI` factored by sparse Cholesky from `sparse(H)` and `σ` below the lowest eigenvalue, in the operator's element type (real for a real node, see "Real arithmetic" below); it offers spectrum slicing across threads and a completeness certificate by inertia counts, on by default with the automatic shift, which recovers the copies of a degenerate level single-vector Lanczos misses by deflation.
`ChebyshevFilter(; on=Array)` is the leaf solver for a device (it also runs on the host): Chebyshev-filtered block subspace iteration started from the lowest product states of the separable part (with a `1e-2` random admixture and guard columns, so a lower state outside the seed span usually surfaces; there is no completeness certificate, which `ShiftInvert` provides), with the filter's upper end from `gershgorin_bound(H)`.
Its defaults were tested against invariant-seed traps, weak filters, crossing sweeps, degenerate blocks and deep references without a wrong result, but this is tested, not proven: a state outside the product-seed span is found only if the filter lifts its `1e-2/√N` admixture to O(1) before the wanted and guard columns lock, which can fail silently with a weak filter and a loose `tol`, or at a small `extra`.
Use `tol ≤ 1e-6` and the default or a larger `extra`, and `ShiftInvert()` or `Dense()` when completeness matters.
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

A user-defined solver subtypes `Solver` and needs one method, `KronFreeCircuits.solve(s, H, m)` (it may also accept `X0=nothing`, which only `LOBPCG` receives today), returning the lowest `m` energies in ascending order and orthonormal eigenvectors of the operator's element type (`eltype(H)`, real for a real node) as host arrays, with true residuals below its tolerance.
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
`Auto(on=CuArray)` does so only for a node it resolves to a device solver; a node of at most `dense_cutoff` (1 000) states resolves to `Dense()` on the host.
Energies, eigenvectors and ports return as host arrays; ports are formed on the host after the solve.

## Spec files

Specs come from `circuit-quantization`'s `save_spec`, which writes a `kronfree-spec` version 1 file; its `docs/spec-format.md` is the authority on the format.

```python
from circuit_quantization.io import save_spec
save_spec(res, "my_circuit.json")   # res: a quantization result
```

The hand-written fixtures are cross-checked against specs generated from the corresponding lumped circuits (`test/test_generated_specs.jl`, using the committed files in `test/fixtures/generated/`).
`test/fixtures/generate_specs.py` regenerates them; run it with circuit-quantization's Python environment only when regenerating, since Python is not a test dependency.
The documented convention differences between the two sets are asserted in the test (spec §20g).

## Periodic phase grid

`PeriodicGrid(N; order=8, sectors=1)` is a phase grid for a compact mode, on `[-π, π)` without the endpoint, with central stencils of order 2 to 8 that wrap around the circle.
Use it for non-standard compact modes: a cos 2φ element, higher harmonics, or a numerically supplied 2π-periodic potential given to the spec as a Fourier series of cosines.
The charge basis stays the default, and the better choice, for a single cos φ.
The basis is parameter-free: the offset charge enters only through the coefficients `e^{±2πi n_g}` that term construction puts on the stencil's two wrap-corner pieces (the twisted boundary `ψ(φ + 2π) = e^{-2πi n_g} ψ(φ)`).
`N` must be divisible by every integer harmonic of a cosine on the mode and by `sectors`, so that translations by `2π/ω` map the grid to itself and the charge-parity sectors are exact; a non-integer harmonic on a compact mode is rejected when the spec is loaded.
The operators are complex, except when the spec gives `n_g` as the literal `"0.5"` (the twist phase is then the constant −1; see "Real arithmetic").

```julia
spec = load_spec("cos2phi.json")                     # 4 E_C (n - n_g)^2 - E_2 cos 2φ - E_1 cos φ, mode phi
θ = (E_C=1.0, n_g=0.3, E_1=0.5, E_2=10.0)
sys = materialize(Node([:phi => PeriodicGrid(128)]; keep=lowest(8), ports=[(:phi, nport())]), spec, θ)
energies(sys)
port(sys, :phi, nport())                             # the charge n in the kept eigenbasis (8 × 8, complex)
```

The basis serves no `:n` port of its own, because the charge is assembled from the derivative pieces and the offset charge.
The spec form of `materialize` expands `nport()` on a `PeriodicGrid` mode automatically wherever a charge is requested: in `ports=` on any node, in a selection rule such as `connected`, and as a factor of an `extra_terms` term.
It expands into `n = d1 + e^{2πi n_g} d1wrap₊ + e^{-2πi n_g} d1wrap₋ + n_g I` on the ports `Port(:d1, 0)` and `Port(:d1wrap, ±1)`, a term's `n_g I` share multiplying its remaining factors.
At the leaf, `port(sys, mode, nport())` equals `KronFreeCircuits.charge_port(sys, mode)` (not exported), which forms the same matrix on demand.
The term-list form of `materialize` has no offset charge, so `nport()` on a twisted mode throws there; write the pieces out instead, with the twist factors `twist_factor(ng, ±1)` as coefficients.
`n2port()` is not served on a `PeriodicGrid`.
The design is in spec §20d.

## Real arithmetic

A node runs in `Float64` only when its structure guarantees a real Hamiltonian at every θ.
This is decided once per `materialize` from the term list and the tree, never from parameter values or from the size of imaginary parts of evaluated data.
A node is real when every factor port has a declared character and every coefficient function satisfies the conjugation identity of its factor key; the identity is checked at three fixed pseudo-random parameter points, and a coefficient that is not real-analytic in θ (for example one that calls `abs` or `sign`) makes the node complex.
A node is real only when every node below it is real.

All entries of θ must be `Real`; a complex entry raises an `ArgumentError`, so a complex amplitude is passed as two real parameters.
The spec expressions for `E_C`, `E_L`, `flux_linear`, `n_g`, energies, offsets and `constant` may not use `im`, `cis` or `cispi`; the loader rejects them.

Each basis declares how complex conjugation acts on the ports it serves:

| Basis | real | imaginary | paired |
|---|---|---|---|
| `ChargeBasis` | `n`, `n2`, `expiphi(k)` | | `exp2pin(k)` with `exp2pin(-k)` |
| `PhaseGrid` | `n2`, `phi`, `phi2`, `exp2pin(k)` | `n` | `expiphi(k)` with `expiphi(-k)` |
| `PeriodicGrid` | `d2`, `d2wrap(±1)` | `d1`, `d1wrap(±1)` | `expiphi(k)` with `expiphi(-k)` |
| `FockBasis` | `phi`, `phi2`, `n2`, `exp2pin(k)` | `n` | `expiphi(k)` with `expiphi(-k)` |
| `GivenSystem` | operator with real data | operator with purely imaginary data | none (an operator with both real and imaginary parts nonzero is general and makes its node complex) |

In practice the transmon and fluxonium leaves, the symmetric zero-pi at any flux and `n_g`, and the two-zero-pi parent are real.
The asymmetric zero-pi and a `PeriodicGrid` mode whose `n_g` is a parameter are complex; a literal `"0.5"` in the spec is real.
A circuit that is complex by structure but real at special parameter values (a charge-basis SQUID at zero flux) stays complex.

`Node(...; arithmetic=:auto)` follows the classification; `:complex` forces complex arithmetic for the node and every ancestor, and `:real` asserts the classification and raises an `ArgumentError` naming the node's modes and the first reason it is not real (for example a term with no conjugate image, a general-character factor, a non-analytic or throwing coefficient, or a complex child).
`arithmetic(sys, node)` returns `:real` or `:complex` for a materialized node.

Outputs have the node's native type.
`eigenvectors` of a real node are `Matrix{Float64}`.
`port` returns `Matrix{Float64}` for a real-character port of a real node and `Matrix{ComplexF64}` for an imaginary port (a grid or Fock `n`) or a paired port (`expiphi` on a grid).
`charge_port` and `decompose` follow the node in the same way, and a complex node returns `ComplexF64` throughout.

`ShiftInvert` follows the operator's element type (the value-based `isreal_assembled` test is gone).
A direct `solve(LOBPCG(), H, m; X0)` with a complex `X0` on a real `H` converts an exactly real block and raises an `ArgumentError` for a complex one.

In the benchmark (`docs/spikes/2026-10-18-plan3-benchmark/results.md`), the full two-zero-pi tree takes 9.9 s on CPU and 5.0 s on the A4500 in real arithmetic, against 22.3 s and 13.2 s in complex arithmetic (2.3× and 2.6×) and 15.4 s and 12.6 s in plan 2b (1.55× and 2.5×).
The `ShiftInvert` leaf on CPU gains nothing against plan 2b, which already factored the assembled real matrix; the gains are in the parent solve (2.5× CPU, 3.3× A4500), the device leaf (2.5×) and port formation (about 4×).

## Born–Oppenheimer reduction

A slow mode can be promoted to a parameter, the fast modes diagonalized on a grid of its coordinate, and the resulting surface fitted to a one-mode reduced spec.
The zero-pi of Di Paolo et al. (arXiv:1811.07921v1, Appendix B), with θ slow and φ fast:

```julia
using KronFreeCircuits

p = promote_mode(load_spec("test/fixtures/zero_pi_bo.json"), :theta)        # fast spec over φ; θ is a parameter
leaf = Node([:phi => PhaseGrid(-12π, 12π, 481)]; solver=Dense(), compute=2, keep=lowest(2))
θ = (E_Cphi=40 / (16 * 0.165), E_Ctheta=0.007, E_J=6.6, E_L=0.04)           # GHz
S = [bo_surface(leaf, p, merge(θ, (varphi_ext=2π * f,)), periodic_grid(32; start=-π/2)) for f in 0:0.125:0.5]
F = fit_surfaces(S, [:constant => ["1", "cos(varphi_ext)", "cos(2*varphi_ext)", "cos(3*varphi_ext)", "cos(4*varphi_ext)"],
                     (:cos, 2) => ["1", "cos(varphi_ext)"], (:cos, 1) => ["cos(varphi_ext/2)"]])
F.constants, F.stderr, F.rms_residual                                        # E_α = c[6], E_β = −c[7], E_γ = c[8] in GHz
save_spec("zero_pi_reduced.json", F.spec; metadata=F.metadata)              # a kronfree-spec with its _bo provenance
```

`slow_spec(S[1])` gives the slow spec of one surface from its Fourier coefficients instead of a fit.
Promotion is a spec-to-spec transformation; a charge coupling between slow and fast modes is rejected, and only the single lowest surface is used (no derivative couplings).
The reproduction (`bench/born_oppenheimer/zero_pi.jl`, results in `bench/born_oppenheimer/results.md`) matches the published E_α and E_γ to 0.12% with E_Cφ derived from the plasma-frequency identity; the design is `docs/superpowers/specs/2026-10-19-born-oppenheimer-design.md`.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                                   # fast tier, under a minute
KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'             # every file
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_bo.jl"])'           # only the named files
```

The default runs the fast tier: the unit-level files, about 45 s on an M2.
Setting `KRONFREE_SLOW_TESTS=1` runs every file, about 10 minutes in all on a loaded M2:
- the apply, end-to-end, solver, catalogue, character, generated-spec and Born–Oppenheimer files (`SLOW_FILES` in `test/runtests.jl`);
- two plan-3 real-versus-complex equivalence testsets gated inside those files;
- last, the published-reference reproductions: the two-coupled-zero-pi acceptance test against scqubits (about a minute) and the zero-pi Born–Oppenheimer reproduction (`test/test_born_oppenheimer_reference.jl`, about 15 s).

Run the slow tier before merging.
A named-file run skips the testsets gated inside those files unless `KRONFREE_SLOW_TESTS=1` is also set.

The [science reference catalogue](docs/science/README.md) maps package capabilities to
versioned literature sources and reproduction targets. The default suite includes
saved scqubits spectra and matrix elements for CPB/transmon, fluxonium, and three
seeded circuits with irrotational flux assignments. Python is only needed to
regenerate those fixtures. The [Appendix B BO reference](docs/science/born-oppenheimer.md)
is implemented by the Born–Oppenheimer reduction above.
The [KITE references](docs/science/kite.md) prepare both the inductively shunted
experiment and the capacitively shunted protected circuit for future spectra and
matrix-element benchmarks alongside the four-mode fluxonium coupler.

The device tests have their own environment and need a CUDA device:

```bash
julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
```

Two benchmark scripts take their settings from environment variables, documented at the top of each file.
`bench/parent_solve.jl` solves the audit's stored 300 × 300 parent cold and after a 1 % coupling step, with both preconditioners; `bench/end_to_end.jl` materializes two coupled zero-pis and times each stage separately.
Results and the exact commands are in `docs/spikes/2026-10-17-plan2-benchmark/results.md`, and the rerun with the plan-2b leaf solvers (`BENCH_LEAF_SOLVER=shiftinvert`, `chebyshev` or `auto`) is in `docs/spikes/2026-10-17-plan2b-benchmark/results.md`.
`BENCH_ARITHMETIC=auto|complex` sets `arithmetic=` on every node of `bench/end_to_end.jl`'s tree; the plan-3 results and commands are in `docs/spikes/2026-10-18-plan3-benchmark/results.md`.

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
`Auto()`'s `dense_cutoff` was benchmarked (`bench/dense_cutoff/results.md`): the iterative solvers win from about 150 to 1000 states, and it was lowered from 2500 to 1000 once `ShiftInvert`'s spurious residual-check failures were fixed; 3D leaves are tabled (spec §20e decision 8).
KernelAbstractions compat is `0.9, 0.10`: CUDA 6 requires 0.9, so a CUDA environment resolves to 0.9, while under 0.10 the CPU backend runs the kernels through POCL.
Plan 3 (real arithmetic by port character: `Float64` nodes, native output types, the `arithmetic` option) is done; its benchmark is in `docs/spikes/2026-10-18-plan3-benchmark/results.md`.
Plan 4 (spec files from `circuit-quantization`, Julia side: the loader and generated cross-check fixtures) is done.
The Born–Oppenheimer reduction (`promote_mode`, `bo_surface`, `slow_spec`, `fit_surfaces`, `save_spec`) is done; its gated test reproduces Di Paolo et al.'s E_α and E_γ.
The next plan (5) is automatic differentiation rules.
Later plans:

1. Export, caching, and NPZ output.
2. The sweep layer (aligned secant prediction along parameter sweeps).
3. The noise plan, which needs the spec format's `operators` and `noise_channels` sections on both the Python and the Julia side.

Current state, open items and decisions pending are in `docs/superpowers/STATUS.md`.
