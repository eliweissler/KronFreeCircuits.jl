# KronFreeCircuits.jl

Hierarchical diagonalization of superconducting circuits without Kronecker products.

A circuit is a tree of nodes.
Each leaf diagonalizes a few modes in a charge basis, a phase grid or a Fock basis, and each parent combines the states its children keep.
No operator is ever assembled as a Kronecker product: it is applied axis by axis to the tensor of the mode bases, so composite systems with large Hilbert spaces stay tractable.

The Hamiltonian comes from a circuit spec written by the [`circuit-quantization`](https://github.com/eliweissler/circuit-quantization) package, as a JSON file or in memory from a Julia session.

!!! note "Status"
    This is a personal research project, largely implemented with Claude Code. Use it at your own risk.

## Installation

The package is not registered yet. Add it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/eliweissler/KronFreeCircuits.jl")
```

It needs Julia 1.11 or later.

## A first example

A transmon, from the spec that circuit-quantization writes for it.
The example specs ship with the package, in its `test/fixtures/generated` directory; change to it to run the examples in this documentation as written:

```julia
cd(joinpath(pkgdir(KronFreeCircuits), "test", "fixtures", "generated"))
```


```@example index
using KronFreeCircuits

spec = load_spec("transmon.json")
θ = (E_J=30.0, E_C=1.2, n_g=0.25)                      # one entry per spec parameter, in GHz
leaf = Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(6), ports=[(:varphi_1, nport())])
sys = materialize(leaf, spec, θ)
energies(sys) .- energies(sys)[1]                       # transition energies in GHz
```

```@example index
port(sys, :varphi_1, nport())                           # the charge operator in the kept eigenbasis
```

## Where to go next

- [Getting started](@ref) builds a transmon and two coupled transmons and reads their states.
- The manual explains [building circuits](manual/circuits.md), [trees and truncation](manual/trees.md) and [solvers](manual/solvers.md), then the features built on a solved tree: [derivatives](manual/derivatives.md), [wavefunctions](manual/wavefunctions.md), [saving](manual/saveload.md) and [time evolution](manual/quantumtoolbox.md).
- [Tutorial: a soft zero-pi qubit](@ref) solves the device of Gyenis et al. and redraws its wavefunctions; [Tutorial: two coupled soft zero-pi qubits](@ref) couples two of them as in Liu, Weissler and Combes and works through a three-level tree.
- [Extending](@ref) shows how to plug in your own solver or selection rule.
