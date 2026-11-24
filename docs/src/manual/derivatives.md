# Derivatives

`derivatives` gives the first and second derivatives of every kept energy, kept state and port of a materialized tree with respect to named parameters, at the root and at every node.
They are computed by perturbation theory carried upward through the tree, children first, with no autodiff engine and no re-solve.

Two coupled transmons:

```@example deriv
using KronFreeCircuits

spec = load_spec("two_transmons.json")
θ = (E_J1=30.0, E_C1=1.2, n_g1=0.21, E_J2=25.0, E_C2=1.0, n_g2=0.09, g=0.3)
q1, q2 = (Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(4)) for i in 1:2)
root = Node([q1, q2]; solver=Dense(), keep=lowest(6), ports=[(:varphi_1, nport())])
sys = materialize(root, spec, θ)

d = derivatives(sys; wrt=(:E_J1, :g), order=2, mixed=((:E_J1, :g),), scale=(g=0.5,))
energies(d, :E_J1)                  # ∂E/∂E_J1 of the root's kept levels (GHz per GHz)
```

```@example deriv
energies(d, :E_J1, :g)              # the mixed second derivative ∂²E/∂E_J1∂g
```

```@example deriv
energies(d, :E_J1, q1)              # ∂E/∂E_J1 at a child node
```

The other results:

```julia
eigenvectors(d, :g)                 # ∂ψ/∂g, one column per kept state, in the root's product basis
port(d, :g, :g, :varphi_1, nport()) # ∂²/∂g² of the root's charge port
```

`scale=(g=0.5,)` says that `g` will be moved by about 0.5.
Kept states that mix strongly over that displacement form a near-degenerate block, whose second-order Taylor expansion is not trusted; `block_levels` gives their levels at a finite displacement instead:

```@example deriv
b = first(blocks(d))
b.states, block_levels(b, (g=0.2,))  # the block's levels at g + 0.2, from its effective Hamiltonian
```

## What is held fixed

The derivatives are those of the calculation as performed: the selection of kept states, every basis and grid, the coordinate frame, the solvers, each node's real or complex arithmetic and its discrete phase choice are held fixed.
They agree with finite differences of `materialize` wherever those are themselves well defined (checked on three-level, complex, augmented and iterative trees, to within the finite differences' own error).
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

Every node retains the eigenvectors of the states its solver computed but did not keep (`Node(...; retain = :auto)`), because the sum over states needs them; the solves inside the mean-field loop and in `bo_surface`, whose results are never differentiated, do not.
The exception is a `Dense` node above 1000 product states, where retention is opt-in (`retain = :always`); without it such a node is re-diagonalized by the derivative pass, with a notice.
A retaining `Dense` node with a `compute` limit computes its full decomposition: measured on a 801-state transmon leaf with `compute = 10`, `materialize` took 0.091 s against 0.019 s and the system 5.2 MB against 73 KB (`bench_archive/retention_cost.jl`).
`retain = :never` turns retention off per node.
The cost adds up in stored sweeps: the `System`s of a 21-point sweep of one 901-point leaf take 130 MB with retention against 0.81 MB without.
At a retaining, compute-limited `Dense` node the rotation inside an exactly degenerate pair of kept states may differ from the one before retention (the full decomposition replaces the partial one).

What `derivatives` refuses or does not do:
- A node whose phases followed a previous solution (`phase = :follow` with `previous`) is an `ArgumentError`; re-materialize at the point without `previous`.
  A `follow` selection is fine with `phase = :positive_sector` given explicitly, or without `previous`; note that `keep = follow(...)` defaults to `phase = :follow`.
- A `wrt` parameter that nothing in the system depends on is an `ArgumentError` when the spec does not declare it (a `θ` key the spec never named, or a parameter `substitute` replaced by a number), and otherwise gives zeros with a warning.
- A `GivenSystem` child is an `ArgumentError` unless `given_independent = true` declares its numbers independent of every parameter.
- An augmented child is differentiated with its union basis held fixed: the derivative of the projected problem, not of a re-run of the mean-field loop.
- No third order, no derivatives with respect to the frame, the bases or the grids, and no reverse sweep: the cost is one pass per parameter and per mixed pair.

`port(d, x, :n, :varphi_1)` and `port(d, x, λ, :n, :varphi_1)` are the spec-form spellings, as `port(sys, :n, :varphi_1)` is for a `System`.


## Gradients through ChainRules

`spectrum(root, spec, θ; ports)` returns the root's `(; energies, ports)`, and the ChainRulesCore extension (loaded with the package, through its dependencies) gives it an `rrule` and an `frule`, so a loss written over energies and port elements can be differentiated by Zygote (not a dependency of this package: add it to your own environment; checked with Zygote 0.7.16):

```julia
using Zygote
L(θ) = (y = spectrum(root, spec, θ; ports=[(:varphi_1, nport())]); y.energies[2] - y.energies[1] + abs2(y.ports[1][1, 2]))
Zygote.gradient(L, θ)[1]            # a NamedTuple: ∂L/∂E_J1, …, ∂L/∂g
```

The rules run `derivatives` to first order; second derivatives come from `derivatives(...; order = 2)`, not from differentiating the rule twice.
[`bench_archive/derivative_design`](https://github.com/eliweissler/KronFreeCircuits.jl/tree/main/bench_archive/derivative_design) solves a three-parameter design problem with BFGS on this gradient.
