# Getting started

This page solves a transmon, then two coupled transmons, and shows how to read energies, operators and states.
The specs come from [`circuit-quantization`](https://github.com/eliweissler/circuit-quantization); [Building circuits](manual/circuits.md) explains how to make your own.
The example specs ship with the package, in its `test/fixtures/generated` directory; change to it to run the examples in this documentation as written:

```julia
cd(joinpath(pkgdir(KronFreeCircuits), "test", "fixtures", "generated"))
```

## One transmon

A spec lists the circuit's modes and parameters:

```@example start
using KronFreeCircuits

spec = load_spec("transmon.json")
spec.modes, spec.parameters
```

The transmon has one compact mode, `varphi_1`.
A tree with a single leaf describes how to solve it: the mode's basis, how many states to keep, and which operators (ports) to form in the kept eigenbasis.

```@example start
leaf = Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(6), ports=[(:varphi_1, nport())])
θ = (E_J=30.0, E_C=1.2, n_g=0.25)
sys = materialize(leaf, spec, θ)
energies(sys)
```

`ChargeBasis(30)` holds the charges `-30..30`.
`materialize` solves the tree at the parameter values `θ`, in GHz, and returns a `System`.
The kept charge operator is a 6 × 6 matrix:

```@example start
n = port(sys, :varphi_1, nport())
round.(abs.(n); digits=3)
```

## Two coupled transmons

Two transmons coupled through their charges have a tree with one leaf per transmon and a parent that couples them.
Each leaf keeps its lowest 8 states, and the parent diagonalizes the 64-dimensional product of the kept states:

```@example start
spec = load_spec("two_transmons.json")
θ = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.0, g=0.3)

transmon(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
root = Node([transmon(1), transmon(2)]; keep=lowest(20), ports=[(:varphi_1, nport())])
sys = materialize(root, spec, θ)
energies(sys)[1:6] .- energies(sys)[1]
```

Each node solves with `Auto()` unless told otherwise; here every node is small, so `Auto()` picks exact dense diagonalization.
[Solvers](manual/solvers.md) lists the others and when to use them.

### Reading states

A dressed state of the parent is a superposition of products of its children's kept states.
`decompose` gives that content for dressed state 2 (Julia indices from 1), largest first:

```@example start
decompose(sys, root, 2)
```

`dressed_state_string` prints the same state for a paper or a log, with physics indices that start at 0, both for the dressed state and for each child's states:

```@example start
println(dressed_state_string(sys, 1))   # the same state as decompose(sys, root, 2)
println(dressed_state_string(sys, 2))
```

`dressed_index` goes the other way, from a bare product label to the dressed state that overlaps it most:

```@example start
dressed_index(sys, root, (2, 1)), dressed_index(sys, root, (1, 2))
```

The labels are 1-based indices into each child's kept states, so `(2, 1)` is the first transmon's first excited state with the second transmon in its ground state.

A child's own results are read by passing the child node:

```@example start
energies(sys, root.children[1]) .- energies(sys, root.children[1])[1]
```

## What next

- [Trees and truncation](manual/trees.md): selection rules other than `lowest`, ports, and given subsystems such as a resonator.
- [Tutorial: two coupled zero-pi qubits](@ref): a larger tree whose leaves have two modes each.
