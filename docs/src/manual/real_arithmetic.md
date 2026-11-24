# Real arithmetic

A node runs in `Float64` only when its structure guarantees a real Hamiltonian at every θ.
This is decided once per `materialize` from the term list and the tree, never from parameter values or from the size of imaginary parts of evaluated data.
A node is real when every factor port has a declared character and every coefficient function satisfies the conjugation identity of its factor key; the identity is checked at three fixed pseudo-random parameter points, and a coefficient that is not real-analytic in θ (for example one that calls `abs` or `sign`) makes the node complex.
A node is real only when every node below it is real.

Real arithmetic is worth having: on the two-zero-pi benchmark tree it is 2.3× faster on CPU and 2.6× faster on an A4500 GPU than complex arithmetic, mostly in the parent solve, the device leaf and port formation.

## Inputs

All entries of θ must be `Real`; a complex entry raises an `ArgumentError`, so a complex amplitude is passed as two real parameters.
The spec expressions for `E_C`, `E_L`, `flux_linear`, `n_g`, energies, offsets and `constant` may not use `im`, `cis` or `cispi`; the loader rejects them.

## Port characters

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

## Controlling it

`Node(...; arithmetic=:auto)` follows the classification; `:complex` forces complex arithmetic for the node and every ancestor, and `:real` asserts the classification and raises an `ArgumentError` naming the node's modes and the first reason it is not real (for example a term with no conjugate image, a general-character factor, a non-analytic or throwing coefficient, or a complex child).
`arithmetic(sys, node)` returns `:real` or `:complex` for a materialized node:

```@example real
using KronFreeCircuits
spec = load_spec("transmon.json")
leaf = Node([:varphi_1 => ChargeBasis(20)]; keep=lowest(4))
sys = materialize(leaf, spec, (E_J=30.0, E_C=1.2, n_g=0.25))
arithmetic(sys), eltype(eigenvectors(sys))
```

## Outputs

Outputs have the node's native type.
`eigenvectors` of a real node are `Matrix{Float64}`.
`port` returns `Matrix{Float64}` for a real-character port of a real node and `Matrix{ComplexF64}` for an imaginary port (a grid or Fock `n`) or a paired port (`expiphi` on a grid).
`charge_port` and `decompose` follow the node in the same way, and a complex node returns `ComplexF64` throughout.
`ShiftInvert` follows the operator's element type.
A direct `solve(LOBPCG(), H, m; X0)` with a complex `X0` on a real `H` converts an exactly real block and raises an `ArgumentError` for a complex one.
