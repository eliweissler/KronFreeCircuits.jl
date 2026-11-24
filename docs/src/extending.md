# Extending

Two interfaces are public for extension: solvers and selection rules.
Their names are public but not exported, so write `KronFreeCircuits.Solver` or `import KronFreeCircuits: solve`.

## A custom solver

A solver subtypes `KronFreeCircuits.Solver` and needs one method, `KronFreeCircuits.solve(s, H, m)`, returning the lowest `m` energies in ascending order and orthonormal eigenvectors of the operator's element type (`eltype(H)`, real for a real node) as host arrays, with true residuals below its tolerance.
It may also accept `X0=nothing`, a warm-start block, which only `LOBPCG` receives today.
The [`KronFreeCircuits.Solver`](@ref) docstring states the full contract and what `H` provides (`mul!` on blocks, `diagonal`, `sparse`, `gershgorin_bound`, `separable_eigenbases`, `product_seed`).

```@example ext
using KronFreeCircuits, LinearAlgebra

struct MySolver <: KronFreeCircuits.Solver end
function KronFreeCircuits.solve(::MySolver, H, m)          # qualified: `solve` is not exported
    F = eigen(Hermitian(Matrix(H)), 1:m)                      # exact, so the residuals are at round-off
    return F.values, F.vectors
end

spec = load_spec("transmon.json")
leaf = Node([:varphi_1 => ChargeBasis(30)]; solver=MySolver(), compute=6, keep=lowest(6))
energies(materialize(leaf, spec, (E_J=30.0, E_C=1.2, n_g=0.25)))
```

Define the method as `KronFreeCircuits.solve` (or after `import KronFreeCircuits: solve`): an unqualified `function solve(...)` after `using KronFreeCircuits` creates a new function instead of adding a method.

A system solved with a custom solver saves and loads without the solver's code; see [Saving and loading](manual/saveload.md).

## A custom selection rule

A rule subtypes `KronFreeCircuits.SelectionRule` and implements `KronFreeCircuits.select(rule, ctx)`, returning the indices of the kept states among the computed ones, in the order they are to be kept.
`ctx` is a [`KronFreeCircuits.SelectionContext`](@ref): the computed `energies` and `vectors`, the ports in the eigenbasis, and the previous solution of a sweep.
A rule that reads ports also implements `KronFreeCircuits.ports_needed(rule)`, returning the `(mode, port)` pairs it needs.

A rule that keeps only every other state:

```@example ext
struct EveryOther <: KronFreeCircuits.SelectionRule end
KronFreeCircuits.select(::EveryOther, ctx::KronFreeCircuits.SelectionContext) = collect(1:2:length(ctx.energies))

leaf = Node([:varphi_1 => ChargeBasis(30)]; compute=8, keep=EveryOther())
energies(materialize(leaf, spec, (E_J=30.0, E_C=1.2, n_g=0.25)))
```

## Operators

The operator machinery (`LocalOp`, `DenseOp`, `DiagonalOp`, `BandedOp`, `ShiftOp`, `apply_axis!`, `KronFreeOperator`) is public for reading and for writing solvers; see the [API reference](@ref "Public, not exported").
