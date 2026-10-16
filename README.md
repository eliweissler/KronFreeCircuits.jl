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
The spec's scalar constant is added at the root only, so the `energies` of child nodes exclude it.
Bare-state bookkeeping (labels, overlaps) is available through `bare_labels` and `bare_overlaps`.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

Setting `KRONFREE_SLOW_TESTS=1` adds the two-coupled-zero-pi acceptance test against scqubits, which takes about a minute.

## Status and plans

The design is in `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`.
Plan 1 (core CPU pipeline: spec loading, term construction, kron-free application, Dense and Lanczos solvers, selection rules, materialization, dressed/bare interface) is done.
Later plans:

1. Automatic differentiation rules.
2. LOBPCG and CUDA.
3. Export, caching, and NPZ output.
4. The Python exporter in `circuit-quantization`.
