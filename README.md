# Status

This is a personal research project and largely implemented using Claude Code. Use at your own risk!

# KronFreeCircuits.jl

Kron-free hierarchical diagonalization of superconducting circuits.
A circuit is a tree of nodes: each leaf diagonalizes a few modes (charge basis, phase grid, or Fock basis), and each parent combines its children's kept eigenstates.
Operators are never assembled as Kronecker products; they are applied axis by axis on the tensor of the mode basis, so composite systems with large Hilbert spaces stay tractable.
Hamiltonian terms come from a JSON circuit spec, which is exported by the `circuit-quantization` package, or from a `Spec` built in memory from a circuit-quantization session in Julia (see "Defining circuits from Julia").
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

A term with several factors inside one child (a phase factor over several modes of a KITE, say) is applied through that child's joint port, formed in the child's product basis before its truncation (spec §20i); `port(sys, [(:varphi_1, expiphi(1)), (:varphi_4, expiphi(-1))], node)` reads one.
Selection rules are `lowest`, `indices`, `connected`, `follow`, and `within`.
`within(window; max=0)` keeps the computed states within `window` (in the Hamiltonian's energy units) of the node's lowest, at most `max` of them: an energy cutoff instead of a count, which spends kept states where a node's spectrum is dense and makes nodes of different size comparable; `compute` must be large enough to close the window, and the rule warns when it is not.
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

## Defining circuits from Julia

A circuit can also be defined, quantized and solved in one Julia session, with no spec file passed between Python and Julia.
circuit-quantization still does the quantization: you call its Python API from Julia through [PythonCall](https://github.com/JuliaPy/PythonCall.jl), and this package turns the result into a `Spec` in memory.
The interface is a package extension, loaded only when PythonCall is loaded beside this package; without PythonCall the package never imports Python, and calling `circuit_quantization()` says "load PythonCall to use the circuit-quantization interface".
The design is in spec §20j and `docs/superpowers/specs/2026-11-10-circuit-quantization-interface-design.md`.

You need a Python environment with circuit-quantization installed (Python ≥ 3.11, SymPy ≥ 1.13), and PythonCall in the Julia environment you work in (`Pkg.add("PythonCall")`).
circuit-quantization must be at commit `c8c8714` or later, which is on its branch `to-spec-zero-tolerance` until that branch is merged into its `main`: `Spec(res)` always passes the round-off zeroing keywords of that commit to circuit-quantization's `to_spec`, and `record_json` and `save_record` forward them, so all three need it, and so does the Python test tier.
With an older circuit-quantization, `Spec(res)`, `record_json` and `save_record` stop with an `ArgumentError` that names the required commit and the file the module was imported from; `QuantizedCircuit` and `Spec(qc[, S])` still work.
The extension installs no Python packages.
Point PythonCall at that environment's interpreter before PythonCall is loaded, with two environment variables:

```bash
export JULIA_CONDAPKG_BACKEND=Null                        # use an existing Python, do not let CondaPkg make one
export JULIA_PYTHONCALL_EXE=/path/to/env/bin/python       # the Python that has circuit-quantization installed
```

If the module `circuit_quantization` is not found, the first call fails with a message naming the interpreter PythonCall used and these two variables; if it is found but fails to import (a missing dependency, say), the message is Python's own.

A session that defines the KITE (the cos 2φ transmon of circuit-quantization's `examples/product_state_demo.py`), builds its spec in two frames, solves one of them and writes a run record:

```julia
using KronFreeCircuits, PythonCall

cq = circuit_quantization()                                   # the Python module, imported once
sympy = pyimport("sympy")

E_J, E_CJ, E_CS, E_L = sympy.symbols("E_J E_CJ E_CS E_L"; positive=true)
eps, delta = sympy.Symbol("epsilon"; real=true), sympy.Symbol("delta"; real=true)
L(E) = cq.ghz_to_henry(E)
C(E) = cq.ghz_to_farad(E)
circuit = cq.Circuit(pylist([
    cq.Branch("L1", 1, 2; L=L(2 * E_L)),
    cq.Branch("L2", 1, 3; L=L(2 * E_L)),
    cq.Branch("CS", 1, 6; C=C(E_CS)),
    cq.Branch("J1", 2, 4; C=C(E_CJ) * (1 + eps), EJ=E_J * (1 + eps)),
    cq.Branch("J2", 3, 5; C=C(E_CJ) * (1 - eps), EJ=E_J * (1 - eps)),
    cq.Branch("L3", 5, 6; L=L(2 * E_L)),
    cq.Branch("L4", 4, 6; L=L(2 * E_L)),
]); ground=6)
cycles = pydict(Dict("kite" => pydict(Dict("L1" => 1, "J1" => 1, "L4" => 1, "L2" => -1, "J2" => -1, "L3" => -1)),
                     "cb" => pydict(Dict("CS" => 1, "L1" => -1, "J1" => -1, "L4" => -1))))
fluxes = pydict(Dict("kite" => cq.phase_to_weber(sympy.pi + delta), "cb" => cq.phase_to_weber(0)))
loops = cq.Loops.from_branch_dict(circuit, cycles, fluxes)
q, h = sympy.Rational(1, 4), sympy.Rational(1, 2)
shares = pydict(Dict("kite" => pydict(Dict("L1" => q, "L4" => q, "L2" => -q, "L3" => -q, "CS" => h)),
                     "cb" => pydict(Dict("CS" => 1))))
B = cq.B_from_shares(circuit, loops, shares)
Z = sympy.Matrix(pylist([pylist(r) for r in ([1, 0, 0, 0, 0], [1, -h, h, 1, 0], [1, -h, -h, 0, 1],
                                             [0, h, -h, 1, 0], [0, h, h, 0, 1])]))
res = cq.quantize(circuit, loops, Z, pytuple(("c", "e", "e", "fz", "fz")); B)   # modes varphi_1 (c), varphi_2, varphi_3 (e)

spec = Spec(res)                                              # symbolic parameters, in memory

θ = (E_J=16.83, E_CJ=4.82, E_CS=0.072, E_L=1.27, epsilon=0.03, delta=0.0, n_g1=0.0)
S = [1.0 0.2354 0.2306; 0.0 0.6962 0.7339; 0.0 -0.5719 0.629]   # the paper's optimum frame, φ = S φ′

qc = QuantizedCircuit(res; values=θ)                          # evaluated once, for many frames
spec_S = Spec(qc, S)                                          # numeric, in the frame S: the loop route
spec_Ssym = Spec(res; frame=S)                                # symbolic, in the frame S: the final route

# One leaf per mode; each phase grid is centered at the minimum of the extended modes' quadratic potential.
e = findall(m -> m.type === :e, spec_S.modes)
value(c) = real(c((;)))                                       # a constant coefficient's value
center = Dict(zip(e, -([value(spec_S.E_L[i, j]) for i in e, j in e] \ [value(spec_S.flux_linear[i]) for i in e])))
basis(i, m) = m.type === :c ? ChargeBasis(10) : PhaseGrid(center[i] - 2.5π, center[i] + 2.5π, 121)
leaves = [Node([m.name => basis(i, m)]; solver=Dense(), compute=16, keep=lowest(16)) for (i, m) in enumerate(spec_S.modes)]
sys = materialize(Node(leaves; compute=4, keep=lowest(4)), spec_S, (;))
energies(sys)                                                 # the four lowest levels, in GHz

save_record("kite_optimum.json", spec_S; circuit=res, frame=S, values=θ, metadata=(note="KITE at the optimum frame",))
rec = load_record("kite_optimum.json")                        # (; result, spec, frame, values, zero_rtol, zero_atol)
```

The same circuit, with its parameter values and three frames, is `kite()` in `test/python/kite.jl`.
The tree above is small, for illustration: its four levels lie 1e-4 to 7e-4 GHz above the converged ones, and `materialize` prints warnings about the kept leaf states (phase grids too small for them, and phase fixing), which `bench_archive/kite_frames/README.md` explains.
`bench_archive/kite_frames/run.jl` solves the three frames on converged trees, and its README shows that the four lowest levels agree across frames to 7e-6 GHz.

There are two routes to a spec in a new frame `φ = S φ′`.
`S` is a square real matrix over the retained modes; it must be invertible, mix no noncompact mode into a compact one (zeros below the compact block), and have an integer, unimodular compact block, and an inadmissible `S` is an `ArgumentError` on both routes.
- The final route, `Spec(res; frame=S, subs=nothing, simplify=true, mode_names=nothing)`, keeps the parameters symbolic: it is circuit-quantization's `transform` and `to_spec` followed by `parse_spec`, so it is exactly the `Spec` that `load_spec` reads from the file `save_spec` writes.
  It takes about a second (below), so it suits a frame fixed once, for a sweep over circuit parameters.
  `subs` replaces parameters by name with numbers, other names or SymPy expressions, and `mode_names` names the modes (default `varphi_1`, `varphi_2`, ...).
- The loop route, `QuantizedCircuit(res; values=θ)` once and then `Spec(qc, S)` per frame, evaluates the circuit at fixed values and changes frames with NumPy, with no symbolic work; it suits a loop that optimizes the frame.
  `values` must give every parameter of `Spec(res)` by name, in the frame of `res`; an unknown or missing name is an `ArgumentError` listing the circuit's parameters, and `record_json` checks its `values` by the same rule.
  Every coefficient of the result is a constant, so nothing is compiled anywhere from the frame change through `materialize`.

Both routes apply the same rule to offset charges: the gate-voltage part transforms as `Sᵀ`, and each compact mode carries its own static offset label, which noncompact modes never do.
In the new frame the static labels are fresh parameters that reuse the names `n_g1`, `n_g2`, ... of the base frame: the value of the new label `n_g{j}` that describes the same circuit is the `j`-th entry of `S_ccᵀ n_g`, with `S_cc` the compact block of `S` and `n_g` the base frame's static offsets, not the base label's own value.
`Spec(qc, S)` applies that rule itself; with `Spec(res; frame=S)` the caller substitutes the transformed values.
`numeric_spec(; modes, E_C, E_L, flux_linear, n_g, cosines, constant)` builds a `Spec` of constants from plain arrays directly, without Python.

Entries that should be zero often come out of a floating-point coordinate change as round-off, and the package would treat each as a real coupling.
Zeroing them is explicit: `numeric_spec`, `Spec(qc[, S])` and `Spec(res; ...)` take `zero_rtol = 1e-12` and `zero_atol = 0.0`.
An `E_C` or `E_L` entry no larger than `max(zero_rtol * scale, zero_atol)`, with `scale` the largest magnitude in that matrix, becomes exactly zero, and so does a cosine's `omega` entry no larger than `zero_rtol` times that cosine's largest `omega` entry (on a compact mode, times the smaller of that entry and 1); `zero_rtol = 0, zero_atol = 0` turns zeroing off.
For example, `Spec(qc, S; zero_rtol=1e-9)` zeroes entries below `1e-9` of the matrix scale.
On the final route the keywords are passed to circuit-quantization's `to_spec` (which needs the commit named above), and that exporter zeroes only floating-point residue, and only in a fully numeric matrix: with symbolic parameters left in `E_C` or `E_L`, nothing in that matrix is zeroed.

`save_record(path, spec; circuit=res, frame, values, metadata, zero_rtol, zero_atol, check=true)` writes the record of a run to `path`, and `record_json(spec; ...)`, with the same keywords, returns its JSON text.
The record is one JSON document for the end of a run.
Its spec fields are those of `spec`, the spec the run solved, written as `spec_json` writes them, so `parse_spec` and `load_spec` read back exactly that spec, coefficient for coefficient.
Beside them it holds what rebuilds the run: circuit-quantization's `to_record` blocks of the circuit `res` in the run's frame (so circuit-quantization's `load_record` rebuilds the circuit in Python), this package's version and git commit under `_generator.kronfree`, the caller's `metadata` under `_metadata`, and a `_kronfree` block with the frame, the values and the two zeroing tolerances the spec was built with.
`values` are in the frame of `res`, as `QuantizedCircuit(res; values)` takes them; in the framed circuit the record gives each fresh static offset label its transformed value `S_ccᵀ n_g`, and every other parameter its given value.
With `check=true` (the default) the spec is first rebuilt from `res`, the frame, the values and the tolerances (by the loop route when there are values, by `Spec(res; frame)` when there are none) and compared with `spec`, to a relative `1e-9` of each matrix's scale; a mismatch, such as a spec of one frame recorded with another, is an `ArgumentError` that names the first field that differs, and `check=false` writes the record without comparing.
`load_record` returns the spec of the run, the symbolic `Result` that was passed as `circuit`, the frame, the base-frame values (in the order of `Spec(res).parameters`) and the tolerances, so `Spec(rec.result; frame=rec.frame)` is the symbolic spec of the run and `Spec(QuantizedCircuit(rec.result; values=rec.values), rec.frame; zero_rtol=rec.zero_rtol, zero_atol=rec.zero_atol)` rebuilds its numeric spec.

Measured on an Intel i7-11700K with Julia 1.13.2 on one thread (2026-11-10): `Spec(qc, S)` for the three-mode KITE takes 0.134 ms per call in a tight loop, against a design target of 5 ms; `Spec(res; frame=S)` takes 0.37 s for the KITE and 0.80 s for six-mode coupled zero-pis.
`bench_archive/kite_frames/README.md` holds the KITE bench's figures, a loop over random frames with a full solve per frame: in its run of 2026-11-10 on commit `d3f338c`, 0.355 ms for the frame change, 0.569 ms through `materialize`'s setup, and no coefficient compiled in the loop.

The Python tests are a separate tier, never run by `Pkg.test()`; see "Tests" below.

## Periodic phase grid

`PeriodicGrid(N; order=8, sectors=1)` is a phase grid for a compact mode, on `[-π, π)` without the endpoint, with central stencils of order 2 to 8 that wrap around the circle.
Use it for non-standard compact modes: a cos 2φ element, higher harmonics, or a numerically supplied 2π-periodic potential given to the spec as a Fourier series of cosines.
The charge basis stays the default, and the better choice, for a single cos φ.
The basis is parameter-free and strictly periodic: it serves `nport()` (the cyclic `-i ∂_φ`), `n2port()` (the cyclic `-∂_φ²`) and `expiphi(k)` for integer `k`, and the offset charge enters only through the coefficients of `4 (n - n_g)ᵀ E_C (n - n_g)`, as on every other basis.
`N` must be divisible by every integer harmonic of a cosine on the mode and by `sectors`, so that translations by `2π/ω` map the grid to itself and the charge-parity sectors are exact; a non-integer harmonic on a compact mode is rejected when the spec is loaded.
The operators are complex unless the spec gives `n_g` as the literal `"0"` (the imaginary linear charge term is then absent; see "Real arithmetic").

```julia
spec = load_spec("cos2phi.json")                     # 4 E_C (n - n_g)^2 - E_2 cos 2φ - E_1 cos φ, mode phi
θ = (E_C=1.0, n_g=0.3, E_1=0.5, E_2=10.0)
sys = materialize(Node([:phi => PeriodicGrid(128)]; keep=lowest(8), ports=[(:phi, nport())]), spec, θ)
energies(sys)
port(sys, :phi, nport())                             # the charge n in the kept eigenbasis (8 × 8, complex)
```

`nport()` on a `PeriodicGrid` mode is an ordinary port in both forms of `materialize`: in `ports=` on any node, in a selection rule such as `connected`, and as a factor of a term; at the leaf it equals `KronFreeCircuits.charge_port(sys, mode)` (not exported), which forms the same matrix on demand.
For a gate drive `n_g(t) = n_g + n_g^{AC}(t)` on a single mode, the Hamiltonian is `H₀ - 8 E_C n_g^{AC}(t) n̂` up to a c-number (for several modes, the drive on mode `i` is `-4 Σ_j (E_C[i,j] + E_C[j,i]) n_g,j^{AC}(t) n̂_i`), so the kept-basis `nport()` matrix is the drive operator.
Two costs of this gauge: a grid node is real only at the literal `n_g = "0"`, and at `n_g = ½` the sweet-spot pairs are degenerate only to the discretization error, `O(Δ^order)` (order 8, free rotor: 1.1e-8 at `N = 32`, 4.4e-11 at 64, roundoff at 128; pure cos 2φ at `E_2 = 15 E_C`: 1.8e-11 and 4.3e-10 for the lowest two pairs at `N = 128`).
The design is in spec §20d (2026-10-30) and `docs/superpowers/specs/2026-10-30-periodic-gauge-design.md`.

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
| `PeriodicGrid` | `n2` | `n` | `expiphi(k)` with `expiphi(-k)` |
| `FockBasis` | `phi`, `phi2`, `n2`, `exp2pin(k)` | `n` | `expiphi(k)` with `expiphi(-k)` |
| `GivenSystem` | operator with real data | operator with purely imaginary data | none (an operator with both real and imaginary parts nonzero is general and makes its node complex) |

In practice the transmon and fluxonium leaves, the symmetric zero-pi at any flux and `n_g`, and the two-zero-pi parent are real.
The asymmetric zero-pi and a `PeriodicGrid` mode whose `n_g` is a parameter or the literal `"0.5"` are complex; the literal `"0"` is real.
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
The reproduction (`bench_archive/born_oppenheimer/zero_pi.jl`, results in `bench_archive/born_oppenheimer/results.md`) matches the published E_α and E_γ to 0.12% with E_Cφ derived from the plasma-frequency identity; the design is `docs/superpowers/specs/2026-10-19-born-oppenheimer-design.md`.

## Mean-field bases

In a frame that mixes modes, a Josephson cosine straddles several children, so a bare leaf never sees a junction and its eigenstates are a poor basis for the parent.
A child can instead be solved in the mean field of its siblings, and the mean-field states of several self-consistent solutions (one per well) can become the basis through which the child is presented to its parent.
The parent's operator is unchanged: it is projected on the new basis, so only the truncation changes.
This is the self-consistent product-state method of the circuit-quantization paper's Appendix C.

The pieces are exported, and the loop is a few lines.
The KITE in the paper's optimum frame (`test/fixtures/kite/optimum.json`, a spec of constants), with one leaf per mode keeping 20 states and two wells feeding each leaf's basis:

```julia
using KronFreeCircuits

spec = load_spec("test/fixtures/kite/optimum.json")
θ = (;)
terms = circuit_terms(spec)
leaves = [Node([m => b]; solver=Dense(), compute=20, keep=lowest(20), arithmetic=:complex)
          for (m, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5, when_capped=:allow)]
root = Node(leaves; solver=LOBPCG(tol=1e-10), compute=8, keep=lowest(8))

# the self-consistent loop from each start; the distinct solutions, lowest product energy first
names = [m.name for m in spec.modes]
starts = [Dict(zip(names, (0.0, 0.0, 0.0))), Dict(zip(names, (π, -π/2, π))), nothing]   # nothing: delocalized
bare = IdDict(leaf => bare_child(root, terms, θ, leaf) for leaf in leaves)    # each leaf's bare solve, formed once
sols = mean_field_solutions(root, spec, θ; starts, bare)
wells = filter(s -> s.energy - sols[1].energy <= 0.5, sols)                               # two, at -9.4294736 GHz

# each leaf's basis: the wells' m lowest mean-field states, merged level by level; its bare operator projected on it
m = 12
aug = map(leaves) do leaf
    sources = [BasisSource("well $i", s.children[leaf].solved.vectors[:, 1:m], s.children[leaf].solved.energies[1:m])
               for (i, s) in enumerate(wells)]
    U, report = union_basis(sources; m)                                                     # report: taken and dropped
    augment(root, terms, θ, leaf, bare[leaf], U, report)
end
sys = materialize(root, spec, θ; augmented=aug)
energies(sys)
```

The wrapper does the same in one call (the solutions within `window = 0.5` GHz of the lowest, at most four, each child's first `m` mean-field states per solution; `bare = true` adds the bare ladder and `cap` limits each solution's share):

```julia
r = mean_field_augment(root, spec, θ; m=12, starts)    # (; children, solutions, used, energy, unconverged)
sys = materialize(root, spec, θ; augmented=r.children)
```

The same bare solves go to `mean_field_solutions` and to `augment`: the solutions' states are coordinates on the bare solve's children, so `U` must be read on that solve, not on a new one.
Both give a largest error of the four lowest levels of 2.2679e-3 GHz against the paper's reference levels, the value of the table below (run on 2026-11-11).
The solves print `fix_phases!` notes for the compact leaf's degenerate charge pairs and PhaseGrid edge warnings for kept states 18 to 20, which no union of 12 states reaches.

`suggest_bases` sizes each grid from the mode's zero-point width and junction energies, and caps it at `max_points`.
A binding cap (the formula asks for more points) keeps the extent with a larger step, so the mode is under-resolved and an energy on that grid is no longer an upper bound of the resolved problem's: an optimizer can then go below the true ground level.
By default it warns; `when_capped = :error` refuses, and `when_capped = :allow` caps silently, as here, where 120 points are the figure's own choice.

The other pieces: `self_consistent` (one run of the loop, also from a previous solution as a warm start), `default_starts`, `mean_field_child`, `bare_child`, `mean_field_terms`, `product_energy`, `expectations`, `pinned_expectations` and `delocalized_expectations`, and `edge_weight` for a grid leaf's edge support.
An augmented child (`AugmentedChild`) keeps its union basis, its bare terms and the `θ` it was built at, and `augmented(sys)` returns those a system was built with; `materialize` refuses it at a `θ` that differs in a parameter its bare terms depend on.
Mean field acts one level deep, among the children of one node, and mean-field and augmented nodes run in complex arithmetic.
`KronFreeCircuits.operator(sys, node)` throws for an augmented node, and derivatives through the mean-field bases are not provided.

Figure 8(b) of the paper, reproduced by `bench_archive/kite_meanfield/levels.jl`: the largest error of the four lowest levels, in GHz, with two-well leaves, against the reference calculation's (ratio in brackets):

| frame | m = 8 | m = 12 | m = 16 | m = 20 |
|---|---|---|---|---|
| published | 6.9179e-01 (1.0000) | 2.2134e-01 (1.0000) | 3.7649e-02 (0.9986) | 6.5900e-04 (0.9726) |
| normal modes | 6.2176e-01 (1.0000) | 1.9890e-01 (1.0000) | 3.6005e-02 (1.0000) | 8.0124e-04 (1.0000) |
| optimum | 3.6677e-02 (1.0000) | 2.2679e-03 (1.0000) | 2.6093e-04 (1.0000) | 2.0717e-05 (1.0000) |

In the normal-mode and optimum frames the levels match the reference calculation's to 1.5e-13 GHz.
In the published frame the two wells' states coincide on one mode up to the loop's convergence residue, whose Gram–Schmidt remainders straddle the `1e-6` drop threshold, so the union depends on where the loop stopped and on round-off; that is the inferred cause of the ratios 0.9986 and 0.9726.
`bench_archive/kite_meanfield/optimize.jl` optimizes the frame from the identity on the product energy and ends at a frame as good as the paper's optimum (two wells at −9.429473654505 GHz against −9.429473599561 on the same bases), after 150 iterations, the cap, without meeting its gradient tolerance.
Both scripts, their runs and the measured details are in `bench_archive/kite_meanfield/README.md`; the design is `docs/superpowers/specs/2026-11-10-mean-field-augmentation-design.md`, and spec §20k records the build and where it departed from the design.

## Derivatives

`derivatives` gives the first and second derivatives of every kept energy, kept state and port of a materialized tree with respect to named parameters, at the root and at every node.
They are computed by perturbation theory carried upward through the tree, children first, with no autodiff engine and no re-solve.
Two coupled transmons (`test/fixtures/two_transmons.json`):

```julia
using KronFreeCircuits

spec = load_spec("test/fixtures/two_transmons.json")
θ = (E_J1=30.0, E_C1=1.2, n_g1=0.21, E_J2=25.0, E_C2=1.0, n_g2=0.09, g=0.3)
q1, q2 = (Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(4)) for i in 1:2)
root = Node([q1, q2]; solver=Dense(), keep=lowest(6), ports=[(:varphi_1, nport())])
sys = materialize(root, spec, θ)

d = derivatives(sys; wrt=(:E_J1, :g), order=2, mixed=((:E_J1, :g),), scale=(g=0.5,))
energies(d, :E_J1)                  # ∂E/∂E_J1 of the root's kept levels (GHz per GHz)
energies(d, :E_J1, :g)              # the mixed second derivative ∂²E/∂E_J1∂g
energies(d, :E_J1, q1)              # the same at a child node
eigenvectors(d, :g)                 # ∂ψ/∂g, one column per kept state, in the root's product basis
port(d, :g, :g, :varphi_1, nport()) # ∂²/∂g² of the root's charge port
b = first(blocks(d))                # states [2, 3]: their Taylor expansion is not trusted over δg = 0.5
block_levels(b, (g=0.2,))           # their levels at g + 0.2, from the block's effective Hamiltonian
```

The derivatives are those of the calculation as performed: the selection of kept states, every basis and grid, the coordinate frame, the solvers, each node's real or complex arithmetic and its discrete phase choice are held fixed.
They agree with finite differences of `materialize` wherever those are themselves well defined (measured by the tests and by the final review's experiments on three-level, complex, augmented and iterative trees, to within the finite differences' own error).
At an iterative node they agree with the full-spectrum (`Dense`) route: for weakly coupled children (couplings 0.03 to 3e-4, `test/test_derivatives_iterative.jl`) each kept state's derivative to within 5e-10 of its own norm, at first and second order.
Each node's `∂H/∂x` has three parts: the derivatives of its own coefficients, its children's port derivatives and its children's energy derivatives.
At a node with its complete spectrum the sum over states is exact; at an iterative node (`LOBPCG`, `Lanczos`, `ShiftInvert`, `ChebyshevFilter`) the part outside the computed states comes from a preconditioned conjugate-gradient solve.
Each column converges to `lintol = 1e-10` relative to its own part outside the computed states, or, where that is below round-off, stops at the round-off floor (at most `1000 eps` of the whole right-hand side); the true residuals are recorded in `d.linear_solves`.
Such a column, like one whose part outside the computed states is below that cap from the start, is accurate only to about `1000 eps ‖(∂H/∂x) ψ_j‖ / gap`, with `gap` the distance from `E_j` to the lowest state outside the computed ones; that exceeds 1e-6 of the state's derivative when `‖∂ψ_j‖ · gap / ‖(∂H/∂x) ψ_j‖ < 2.2e-7`, which on the two-transmon test tree means couplings below about 3e-5 for a parameter internal to one transmon (measured: no column stopped at the floor down to `g = 3e-6`, and the errors there, at most 1.1e-7, are the eigensolver's).

`scale` gives a typical displacement `δx` for a parameter.
A pair of kept states whose coupling `|⟨ψ_l|∂H/∂x|ψ_i⟩| δx` exceeds `r = 0.1` times their detuning forms a near-degenerate block.
The point derivatives are the true derivatives whatever the blocks; a block says that the second-order Taylor expansion of its levels is not trusted over `δx`, and `block_levels` gives its levels at a finite displacement from a small effective Hamiltonian, second order in the displacement.
Without `scale` only exact degeneracies are looked for.
A kept state degenerate (within `degenerate_tol = 1e-9` GHz) with another kept state it couples to has `NaN` derivatives, with a warning and a block; degenerate with an unkept computed state it couples to, `NaN` with a warning and a selection-boundary record in `d.boundary`.
At an iterative node, a kept state degenerate with or above a state that was not computed is an error ("not positive on the complement"); raise `compute`.
A child whose derivative is `NaN` makes the derivatives of its ancestors' states `NaN`, at a `Dense` or an iterative parent, with one warning per node naming the child; the child's block stays available.
A kept state close to an unkept computed state warns ("selection boundary") and is recorded in `d.boundary`, since the derivative at a fixed selection is fragile there.

Every node now retains the eigenvectors of the states its solver computed but did not keep (`Node(...; retain = :auto)`), because the sum over states needs them; the solves inside the mean-field loop and in `bo_surface`, whose results are never differentiated, do not.
The exception is a `Dense` node above 1000 product states, where retention is opt-in (`retain = :always`); without it such a node is re-diagonalized by the derivative pass, with a notice.
A retaining `Dense` node with a `compute` limit computes its full decomposition: measured on a 801-state transmon leaf with `compute = 10`, `materialize` took 0.091 s against 0.019 s and the system 5.2 MB against 73 KB (`bench_archive/retention_cost.jl`).
`retain = :never` restores the old behaviour per node.
The cost adds up in stored sweeps: the `System`s of a 21-point sweep of one 901-point leaf take 130 MB with retention against 0.81 MB without (measured by the final review).
At a retaining, compute-limited `Dense` node the rotation inside an exactly degenerate pair of kept states may differ from the one before retention (the full decomposition replaces the partial one).

What `derivatives` refuses or does not do:
- A node whose phases followed a previous solution (`phase = :follow` with `previous`) is an `ArgumentError`; re-materialize at the point without `previous`.
  A `follow` selection is fine with `phase = :positive_sector` given explicitly, or without `previous`; note that `keep = follow(...)` defaults to `phase = :follow`.
- A `wrt` parameter that nothing in the system depends on is an `ArgumentError` when the spec does not declare it (a `θ` key the spec never named, or a parameter `substitute` replaced by a number), and otherwise gives zeros with a warning.
- A `GivenSystem` child is an `ArgumentError` unless `given_independent = true` declares its numbers independent of every parameter.
- An augmented child is differentiated with its union basis held fixed: the derivative of the projected problem, not of a re-run of the mean-field loop.
- No third order, no derivatives with respect to the frame, the bases or the grids, and no reverse sweep: the cost is one pass per parameter and per mixed pair.

`port(d, x, :n, :varphi_1)` and `port(d, x, λ, :n, :varphi_1)` are the spec-form spellings, as `port(sys, :n, :varphi_1)` is for a `System`.

`spectrum(root, spec, θ; ports)` returns the root's `(; energies, ports)`, and the ChainRulesCore extension (loaded with the package, through its dependencies) gives it an `rrule` and an `frule`, so a loss written over energies and port elements can be differentiated by Zygote:

Zygote is not a dependency of this package: add it to your own environment (the session below was checked in `test/chainrules`, with Zygote 0.7.16).

```julia
using Zygote
L(θ) = (y = spectrum(root, spec, θ; ports=[(:varphi_1, nport())]); y.energies[2] - y.energies[1] + abs2(y.ports[1][1, 2]))
Zygote.gradient(L, θ)[1]            # a NamedTuple: ∂L/∂E_J1, …, ∂L/∂g
```

The rules run `derivatives` to first order; second derivatives come from `derivatives(...; order = 2)`, not from differentiating the rule twice.
`bench_archive/derivative_design` solves a three-parameter design problem with BFGS on this gradient.
The design is `docs/superpowers/specs/2026-11-10-perturbative-derivatives-design.md`; spec §20l records the equations, the measurements and where the build departed from the design.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                                   # fast tier, a few minutes
KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'             # every file
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_bo.jl"])'           # only the named files
```

The default runs the fast tier: the unit-level files, about 45 s on an M2 before the derivative files were added, now several minutes (5m50s on ewdesktop at load average 5 to 6.6, 2026-11-11; 26 minutes earlier that day under heavier load).
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

The Python tier tests the circuit-quantization interface and has its own environment, `test/python`; it needs the Python environment of "Defining circuits from Julia" (with circuit-quantization at commit `c8c8714` or later) and is never included by `Pkg.test()`.
Set the two environment variables for the `Pkg.instantiate()` too: precompiling the extension without them can make CondaPkg install a Python of its own.

```bash
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/path/to/env/bin/python julia --project=test/python -e 'using Pkg; Pkg.instantiate()'
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/path/to/env/bin/python julia --project=test/python test/python/runtests.jl
```

The default tier checks that the package itself stays free of Python: no file under `src/` loads or calls PythonCall, and the extension is not loaded.

The rules tier tests the `rrule` and `frule` of `spectrum` (the ChainRulesCore extension), directly and through Zygote; it has its own environment, `test/chainrules`, and is never included by `Pkg.test()`:

```bash
julia --project=test/chainrules -e 'using Pkg; Pkg.instantiate()'
julia --project=test/chainrules test/chainrules/runtests.jl
```

The device tests have their own environment and need a CUDA device:

```bash
julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
```

Two benchmark scripts take their settings from environment variables, documented at the top of each file.
`bench_archive/parent_solve.jl` solves the audit's stored 300 × 300 parent cold and after a 1 % coupling step, with both preconditioners; `bench_archive/end_to_end.jl` materializes two coupled zero-pis and times each stage separately.
Results and the exact commands are in `docs/spikes/2026-10-17-plan2-benchmark/results.md`, and the rerun with the plan-2b leaf solvers (`BENCH_LEAF_SOLVER=shiftinvert`, `chebyshev` or `auto`) is in `docs/spikes/2026-10-17-plan2b-benchmark/results.md`.
`BENCH_ARITHMETIC=auto|complex` sets `arithmetic=` on every node of `bench_archive/end_to_end.jl`'s tree; the plan-3 results and commands are in `docs/spikes/2026-10-18-plan3-benchmark/results.md`.

```bash
BENCH_SMOKE=1 julia --project=. -t 1 bench_archive/end_to_end.jl                                  # reduced sizes, CPU
BENCH_FIXTURE=path/to/larger-real300-operators.bin julia --project=. -t 1 bench_archive/parent_solve.jl
BENCH_DEVICE=gpu BENCH_FIXTURE=path/to/larger-real300-operators.bin julia --project=test/cuda -t 1 bench_archive/parent_solve.jl
```

## Status and plans

The design is in `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`.
Plan 1 (core CPU pipeline: spec loading, term construction, kron-free application, Dense and Lanczos solvers, selection rules, materialization, dressed/bare interface) is done.
Plan 2 (ITensor-backed operator application, LOBPCG with adaptive Jacobi preconditioning, warm starts and physical-overlap transport, the CUDA extension) is done; its benchmark names the leaf solver and device-side port formation as the next stages to optimize.
Plan 2b (derived operator quantities, the `ShiftInvert` and `ChebyshevFilter` leaf solvers, product-state seeding for `LOBPCG`, the `Auto()` default, the solver extension contract, and `PeriodicGrid`) is done; in its benchmark the two-zero-pi tree materializes in 15 s on CPU and 12.4 s on the A4500, against plan 2's 237 s (`docs/spikes/2026-10-17-plan2b-benchmark/results.md`).
`Auto()`'s `dense_cutoff` was benchmarked (`bench_archive/dense_cutoff/results.md`): the iterative solvers win from about 150 to 1000 states, and it was lowered from 2500 to 1000 once `ShiftInvert`'s spurious residual-check failures were fixed; 3D leaves are tabled (spec §20e decision 8).
KernelAbstractions compat is `0.9, 0.10`: CUDA 6 requires 0.9, so a CUDA environment resolves to 0.9, while under 0.10 the CPU backend runs the kernels through POCL.
Plan 3 (real arithmetic by port character: `Float64` nodes, native output types, the `arithmetic` option) is done; its benchmark is in `docs/spikes/2026-10-18-plan3-benchmark/results.md`.
Plan 4 (spec files from `circuit-quantization`, Julia side: the loader and generated cross-check fixtures) is done.
The Born–Oppenheimer reduction (`promote_mode`, `bo_surface`, `slow_spec`, `fit_surfaces`, `save_spec`) is done; its gated test reproduces Di Paolo et al.'s E_α and E_γ.
The circuit-quantization interface (2026-11-10: a PythonCall extension with the symbolic and loop routes to a spec in a new frame, `numeric_spec`, constant coefficients that never compile, and run records; spec §20j) is on `main` (local; not pushed as of 2026-11-11); what is done and what is open is in `docs/superpowers/STATUS.md`.
Mean-field basis augmentation (2026-11-11: mean-field children, the self-consistent loop, union bases and augmented children; spec §20k) is on `main` (local; not pushed as of 2026-11-11); its Harmonium benchmark (design §6.2) is deferred to its own plan.
Derivatives by perturbation theory through the tree (2026-11-11: first and second derivatives, near-degenerate blocks, and the outer rule of `spectrum`; spec §20l) replace the automatic-differentiation rules that were to be plan 5; see "Derivatives" above and `docs/superpowers/STATUS.md`.
Later plans:

1. Export, caching, and NPZ output.
2. The sweep layer (aligned secant prediction along parameter sweeps).
3. The noise plan, which needs the spec format's `operators` and `noise_channels` sections on both the Python and the Julia side.

Current state, open items and decisions pending are in `docs/superpowers/STATUS.md`.
