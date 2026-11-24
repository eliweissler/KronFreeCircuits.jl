# Building circuits

A circuit reaches this package as a `Spec`: its modes, its parameters and the coefficient matrices of
`H = 4 (n - n_g)ᵀ E_C (n - n_g) + ½ φᵀ E_L φ + flux_linear ⋅ φ - Σ E_k cos(ω_k ⋅ φ + offset_k) + constant`,
each entry a function of the parameters.
[circuit-quantization](https://github.com/eliweissler/circuit-quantization) derives it from a lumped-element circuit, and there are two ways to hand it over: a spec file, or a Julia session that calls circuit-quantization through PythonCall.

## Spec files

circuit-quantization's `save_spec` writes a `kronfree-spec` version 1 file. Its `docs/spec-format.md` is the authority on the format.

```python
from circuit_quantization.io import save_spec
save_spec(res, "my_circuit.json")   # res: a quantization result
```

`load_spec` reads the file:

```@example circuits
using KronFreeCircuits
spec = load_spec("fluxonium.json")
spec.modes
```

Each mode has a type: `:c` (compact, a charge-like island), `:e` (extended) or `:h` (harmonic, decoupled from junctions).
The parameters are the names `θ` must supply to [`materialize`](@ref):

```@example circuits
spec.parameters
```

`substitute` replaces parameters by numbers or expressions, and `save_spec` / `spec_json` write a spec back out.
`numeric_spec(; modes, E_C, E_L, flux_linear, n_g, cosines, constant)` builds a `Spec` of constants from plain arrays, without Python.

## Mode bases

Each mode of a leaf gets a basis:

| basis | for | holds |
|:------|:----|:------|
| `ChargeBasis(ncut)` | a compact mode | the charges `-ncut..ncut` |
| `PhaseGrid(a, b, N; order=8)` | an extended mode | `N` points on `[a, b]`, finite-difference stencils of order 2 to 8 |
| `PeriodicGrid(N; order=8, sectors=1)` | a compact mode with a non-standard potential | `N` points on `[-π, π)` |
| `FockBasis(N; phi_zpf)` | a harmonic or weakly anharmonic mode | the `N` lowest oscillator states of zero-point width `phi_zpf` |

A `PhaseGrid` must be wide enough for every kept state: `materialize` warns when a kept state has weight at the grid's edges ("PhaseGrid on mode … is too small").
Widen the grid until the warning goes away.
`suggest_bases(spec; ...)` proposes a basis for each mode from its zero-point width and junction energies.

### Periodic phase grid

`PeriodicGrid(N; order=8, sectors=1)` is a phase grid for a compact mode, on `[-π, π)` without the endpoint, with central stencils of order 2 to 8 that wrap around the circle.
Use it for non-standard compact modes: a cos 2φ element, higher harmonics, or a numerically supplied 2π-periodic potential given to the spec as a Fourier series of cosines.
The charge basis stays the default, and the better choice, for a single cos φ.

The basis is parameter-free and strictly periodic: it serves `nport()` (the cyclic `-i ∂_φ`), `n2port()` (the cyclic `-∂_φ²`) and `expiphi(k)` for integer `k`, and the offset charge enters only through the coefficients of `4 (n - n_g)ᵀ E_C (n - n_g)`, as on every other basis.
`N` must be divisible by every integer harmonic of a cosine on the mode and by `sectors`, so that translations by `2π/ω` map the grid to itself and the charge-parity sectors are exact; a non-integer harmonic on a compact mode is rejected when the spec is loaded.
The operators are complex unless the spec gives `n_g` as the literal `"0"` (see [Real arithmetic](real_arithmetic.md)).

```julia
spec = load_spec("cos2phi.json")                     # 4 E_C (n - n_g)^2 - E_2 cos 2φ - E_1 cos φ, mode phi
θ = (E_C=1.0, n_g=0.3, E_1=0.5, E_2=10.0)
sys = materialize(Node([:phi => PeriodicGrid(128)]; keep=lowest(8), ports=[(:phi, nport())]), spec, θ)
port(sys, :phi, nport())                             # the charge n in the kept eigenbasis (8 × 8, complex)
```

At the leaf, `nport()` equals `KronFreeCircuits.charge_port(sys, mode)`, which forms the same matrix on demand.
For a gate drive `n_g(t) = n_g + n_g^{AC}(t)` on a single mode, the Hamiltonian is `H₀ - 8 E_C n_g^{AC}(t) n̂` up to a c-number (for several modes, the drive on mode `i` is `-4 Σ_j (E_C[i,j] + E_C[j,i]) n_g,j^{AC}(t) n̂_i`), so the kept-basis `nport()` matrix is the drive operator.
Two costs of this gauge: a grid node is real only at the literal `n_g = "0"`, and at `n_g = ½` the sweet-spot pairs are degenerate only to the discretization error, `O(Δ^order)` (order 8, free rotor: 1.1e-8 at `N = 32`, 4.4e-11 at 64, roundoff at 128).

## Defining circuits from Julia

A circuit can also be defined, quantized and solved in one Julia session, with no spec file passed between Python and Julia.
circuit-quantization still does the quantization: you call its Python API through [PythonCall](https://github.com/JuliaPy/PythonCall.jl), and this package turns the result into a `Spec` in memory.
The interface is a package extension, loaded only when PythonCall is loaded beside this package; without PythonCall the package never imports Python, and calling `circuit_quantization()` says "load PythonCall to use the circuit-quantization interface".

You need a Python environment with circuit-quantization installed (Python ≥ 3.11, SymPy ≥ 1.13), and PythonCall in the Julia environment you work in.
circuit-quantization must be at commit `c8c8714` or later (on its branch `to-spec-zero-tolerance` until that branch is merged): `Spec(res)`, `record_json` and `save_record` pass it round-off zeroing keywords, and with an older version they stop with an `ArgumentError` naming the required commit. `QuantizedCircuit` and `Spec(qc[, S])` work with any version.
The extension installs no Python packages.
Point PythonCall at that environment's interpreter before PythonCall is loaded:

```bash
export JULIA_CONDAPKG_BACKEND=Null                        # use an existing Python, do not let CondaPkg make one
export JULIA_PYTHONCALL_EXE=/path/to/env/bin/python       # the Python that has circuit-quantization installed
```

If the module `circuit_quantization` is not found, the first call fails with a message naming the interpreter PythonCall used and these two variables.

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

The tree above is small, for illustration: its four levels lie 1e-4 to 7e-4 GHz above the converged ones, and `materialize` warns that the phase grids are too small for the kept leaf states.

### Changing frames

There are two routes to a spec in a new frame `φ = S φ′`.
`S` is a square real matrix over the retained modes; it must be invertible, mix no noncompact mode into a compact one (zeros below the compact block), and have an integer, unimodular compact block. An inadmissible `S` is an `ArgumentError` on both routes.

- The final route, `Spec(res; frame=S, subs=nothing, simplify=true, mode_names=nothing)`, keeps the parameters symbolic: it is circuit-quantization's `transform` and `to_spec` followed by `parse_spec`, so it is exactly the `Spec` that `load_spec` reads from the file `save_spec` writes.
  It takes about a second, so it suits a frame fixed once, for a sweep over circuit parameters.
  `subs` replaces parameters by name with numbers, other names or SymPy expressions, and `mode_names` names the modes (default `varphi_1`, `varphi_2`, ...).
- The loop route, `QuantizedCircuit(res; values=θ)` once and then `Spec(qc, S)` per frame, evaluates the circuit at fixed values and changes frames with NumPy, with no symbolic work; it suits a loop that optimizes the frame (about 0.13 ms per frame for the KITE).
  `values` must give every parameter of `Spec(res)` by name, in the frame of `res`; an unknown or missing name is an `ArgumentError` listing the circuit's parameters, and `record_json` checks its `values` by the same rule.
  Every coefficient of the result is a constant, so nothing is compiled anywhere from the frame change through `materialize`.

Both routes apply the same rule to offset charges: the gate-voltage part transforms as `Sᵀ`, and each compact mode carries its own static offset label, which noncompact modes never do.
In the new frame the static labels are fresh parameters that reuse the names `n_g1`, `n_g2`, ... of the base frame: the value of the new label `n_g{j}` that describes the same circuit is the `j`-th entry of `S_ccᵀ n_g`, with `S_cc` the compact block of `S` and `n_g` the base frame's static offsets, not the base label's own value.
`Spec(qc, S)` applies that rule itself; with `Spec(res; frame=S)` the caller substitutes the transformed values.

### Round-off zeroing

Entries that should be zero often come out of a floating-point coordinate change as round-off, and the package would treat each as a real coupling.
`numeric_spec`, `Spec(qc[, S])` and `Spec(res; ...)` take `zero_rtol = 1e-12` and `zero_atol = 0.0`.
An `E_C` or `E_L` entry no larger than `max(zero_rtol * scale, zero_atol)`, with `scale` the largest magnitude in that matrix, becomes exactly zero, and so does a cosine's `omega` entry no larger than `zero_rtol` times that cosine's largest `omega` entry (on a compact mode, times the smaller of that entry and 1); `zero_rtol = 0, zero_atol = 0` turns zeroing off.
For example, `Spec(qc, S; zero_rtol=1e-9)` zeroes entries below `1e-9` of the matrix scale.
On the final route the keywords are passed to circuit-quantization's `to_spec`, which zeroes only floating-point residue, and only in a fully numeric matrix: with symbolic parameters left in `E_C` or `E_L`, nothing in that matrix is zeroed.

### Run records

`save_record(path, spec; circuit=res, frame, values, metadata, zero_rtol, zero_atol, check=true)` writes the record of a run, and `record_json(spec; ...)` returns its JSON text.
The record holds the spec the run solved (so `load_spec` reads it back exactly), circuit-quantization's record of the circuit in the run's frame, this package's version and git commit, your `metadata`, and the frame, values and tolerances the spec was built with.
`values` are in the frame of `res`; in the framed circuit the record gives each fresh static offset label its transformed value `S_ccᵀ n_g`.
With `check=true` (the default) the spec is first rebuilt from the circuit, frame, values and tolerances (by the loop route when there are values, by `Spec(res; frame)` when there are none) and compared with `spec`, to a relative `1e-9` of each matrix's scale; a mismatch, such as a spec of one frame recorded with another, is an `ArgumentError` naming the first field that differs, and `check=false` writes the record without comparing.
`load_record` returns `(; result, spec, frame, values, zero_rtol, zero_atol)`: `Spec(rec.result; frame=rec.frame)` is the symbolic spec of the run, and `Spec(QuantizedCircuit(rec.result; values=rec.values), rec.frame; zero_rtol=rec.zero_rtol, zero_atol=rec.zero_atol)` rebuilds its numeric spec.
