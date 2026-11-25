[![CI](https://github.com/eliweissler/KronFreeCircuits.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/eliweissler/KronFreeCircuits.jl/actions/workflows/CI.yml)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://eliweissler.github.io/KronFreeCircuits.jl/dev/)

# KronFreeCircuits.jl

Hierarchical diagonalization of superconducting circuits without Kronecker products.

A circuit is a tree of nodes: each leaf diagonalizes a few modes (charge basis, phase grid, or Fock basis), and each parent combines its children's kept eigenstates.
Operators are never assembled as Kronecker products; they are applied axis by axis on the tensor of the mode basis, so composite systems with large Hilbert spaces stay tractable.
Hamiltonians come from circuit specs written by the `circuit-quantization` package, as a JSON file or in memory from a Julia session.

**Status:** this is a personal research project, largely implemented with Claude Code. Use it at your own risk.

## Documentation

The [documentation](https://eliweissler.github.io/KronFreeCircuits.jl/dev/) has a getting-started guide, a manual (building circuits, trees and truncation, solvers, real arithmetic, derivatives, wavefunctions, saving and loading, time evolution with QuantumToolbox), tutorials on a soft zero-pi qubit and on two coupled soft zero-pi qubits, and the API reference.

## Installation

The package is not registered yet. It needs Julia 1.11 or later.

```julia
using Pkg
Pkg.add(url="https://github.com/eliweissler/KronFreeCircuits.jl")
```

## Sketch

```julia
using KronFreeCircuits

spec = load_spec("two_transmons.json")
θ = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.0, g=0.3)   # one entry per spec parameter, GHz

leaf(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))   # each transmon keeps its 8 lowest states
top = Node([leaf(1), leaf(2)]; keep=lowest(20), ports=[(:varphi_1, nport())])

sys = materialize(top, spec, θ)
energies(sys)                          # kept eigenvalues of the top node
port(sys, :varphi_1, nport())          # charge operator of mode varphi_1 in the kept eigenbasis
decompose(sys, top, 2)                 # bare product-state content of dressed state 2
println(dressed_state_string(sys, 1))  # the same state: |ψ₁⟩ = 0.996|0,1⟩ − 0.093|1,0⟩ (physics indices from 0)
```

## Development

Tests, test tiers and benchmarks are described on the documentation's [Development](https://eliweissler.github.io/KronFreeCircuits.jl/dev/development/) page.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                       # fast tier
KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()' # every file
julia --project=docs docs/make.jl                                  # build the documentation
```
