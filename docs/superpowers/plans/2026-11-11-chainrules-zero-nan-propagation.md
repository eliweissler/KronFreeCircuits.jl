# Prevent zero AD weights from propagating undefined spectral derivatives

## Scope

Fix the ChainRules extension of `spectrum` so that an output derivative which is
undefined (`NaN`) does not contaminate a scalar pullback or a directional
derivative when its corresponding cotangent or parameter tangent is exactly
zero.

The review that found this issue did not change production code. Its regression
cases are in `test/chainrules/test_pathological.jl` and are included by
`test/chainrules/runtests.jl`. Three assertions are deliberately marked
`@test_broken`; make them ordinary passing `@test` assertions as part of the
fix.

The separate Fock-displacement numerical issue is outside this task; it has
already been handled independently.

## Confirmed behavior

Use the exactly degenerate charge-basis spectrum from `cos2phi.json` at

```julia
θ = (E_C=1.0, E_1=0.0, E_2=0.0, n_g=0.0)
```

Its ground-state derivative with respect to `E_2` is well-defined and zero.
The excited pair is exactly degenerate and coupled by `E_2`, so the core
derivative API intentionally reports the individual excited-state derivatives
as `NaN`.

The following pullback asks only for the ground-state energy:

```julia
y, back = ChainRulesCore.rrule(spectrum, root, spec, θ)
grad = back((energies=[1.0, 0.0, 0.0],))[4]
```

`grad[:E_2]` is currently `NaN`; it must be `0.0`. Direct re-solves at
`E_2 = ±0.01` confirm that the ground energy remains zero.

A pushforward also depends incorrectly on how a mathematically identical
tangent is represented:

```julia
Δsparse   = (E_C=1.0,)
Δexplicit = (E_C=1.0, E_1=0.0, E_2=0.0, n_g=0.0)
```

The sparse tangent produces `[0, 4, 4]`, whereas the explicit tangent produces
`[0, NaN, NaN]`. They must agree.

The same pullback problem occurs for a partially populated port cotangent. A
cotangent with only the ground-state diagonal element nonzero currently becomes
`NaN` because zero cotangent entries multiply undefined derivative entries in
the excited-state rows and columns.

## Cause

There are two related zero-handling problems in
`ext/KronFreeCircuitsChainRulesCoreExt.jl`.

1. `_contract` calls `dot` on a whole energy or port array. IEEE arithmetic
   gives `0 * NaN == NaN`, so undefined derivative entries contribute even when
   their cotangent entries are exactly zero.
2. The `frule` loop skips only structural zeros using `_iszero(δ)`. An explicit
   numeric `0.0` is therefore multiplied into the derivative arrays and turns
   their `NaN` entries into `NaN`.

The core derivative API's `NaN` values are intentional. Do not replace or mask
them globally: a nonzero cotangent or tangent that selects an undefined
derivative must still produce `NaN`.

## Required behavior

- In the pullback, contract only entries whose cotangent is nonzero. This must
  work elementwise for energy vectors and port matrices, including complex port
  cotangents under the convention documented by `spectrum`.
- A nonzero cotangent on an undefined entry must continue to return `NaN`.
- In the pushforward, skip explicit numeric zero parameter components as well as
  structural zeros.
- Preserve lazy construction and reuse of the derivative pass.
- Preserve the current fast path in which an entirely zero tangent or
  cotangent avoids running the derivative pass.
- Do not alter `derivatives`, its degeneracy warnings, or its use of `NaN` to
  represent undefined per-state quantities.

A small helper that performs a zero-aware contraction is preferable to
sanitizing derivative arrays. Treat `-0.0` and complex zero as zero. A `NaN`
cotangent is not zero and should propagate.

## Regression tests

The existing pathological file covers:

- a ground-energy-only pullback beside an undefined excited doublet;
- a nonzero cotangent on the undefined doublet, which must remain `NaN`;
- sparse and explicitly zero-filled forward tangents;
- a partially populated complex port cotangent whose zero entries overlap
  undefined derivative entries.

After the fix, convert the three `@test_broken` assertions to `@test`. Add a
focused unit test for the zero-aware contraction helper if one is introduced,
including real and complex arrays, structural zeros, numeric zeros, and a
nonzero weight on `NaN`.

## Validation

Run the focused regression and the optional ChainRules tier:

```sh
julia --startup-file=no --compiled-modules=existing --project \
  -e 'using KronFreeCircuits; include("test/chainrules/test_pathological.jl")'

julia --project=test/chainrules test/chainrules/runtests.jl
```

Then run the normal fast tier:

```sh
julia --startup-file=no --project test/runtests.jl
```

The implementation report should state which tests were actually run and
should separately identify any optional Python or GPU tiers that were not run.

