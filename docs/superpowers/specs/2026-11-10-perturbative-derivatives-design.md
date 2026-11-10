# Design: derivatives by perturbation theory through the tree (2026-11-10)

Status: design for Eli's approval; project 3 of 3 before the feature freeze, to be built after projects 1 and 2.
Replaces the approach of architecture spec §11 (reverse mode through Zygote and ChainRules) and is the content of what STATUS.md calls "plan 5".
Items marked *(my call)* were chosen by Claude and not discussed; everything else was decided with Eli on 2026-11-10.

## 1. Intent

Circuit design optimization needs derivatives of energies, wavefunctions and matrix elements with respect to circuit parameters (`C`, `L`, `E_J`) and control parameters (`n_g`, `φ_ext`), to second order.
Second order is needed because dephasing rates go as a first derivative of a transition frequency with respect to a noise parameter, so the gradient of a dephasing-aware loss contains a mixed second derivative.

The eigensolvers in this package cannot be differentiated by a general autodiff engine (stock LOBPCG under Zygote crashed in the 2026-10-16 spike).
The derivatives are known analytically as perturbation theory, so the rules are written by hand.

### Sources

- Eli's notes, `tight_binding.tex` (overleaf `rotor_codes/0 notes`), lines 287–360: non-degenerate Rayleigh–Schrödinger sums, wavefunctions to second order and energies to third.
- arXiv:2408.12704v3 (Rajabzadeh et al.), `main.tex`:
  energies by Hellmann–Feynman (`eq:eig_val_grad`);
  eigenvectors by a sum over the other *computed* states (`eq:eig_vec_grad`), with the remark that the number of computed states must be large enough for accuracy;
  second derivatives of energies only (`eq:eig_val_secondgrad`, appendix `app:T2_gradient`), in a separate hand-written node, with `∂H/∂x` and the nonvanishing `∂²H/∂x∂λ` tabulated.

### Why upward and not reverse

Architecture spec §11 chose reverse mode first and listed forward mode as a non-goal.
Eli chose to change the spec and do the right thing over the quick thing (2026-11-10).
The deciding facts:

- A design optimization varies 1 to 3 parameters typically and about 10 at most; noise calculations use one parameter at a time.
  Reverse mode's only advantage is cost independent of the parameter count.
- Wavefunction derivatives are a direct output going upward and unavailable as vectors in reverse.
- Second order going upward is the same recipe one order higher.
  In reverse it needs the adjoint of the first-order rules, a second and harder set to derive and test.
- The near-degeneracy test Eli wants (§4.4) needs the perturbation for a specific parameter, which exists going upward and not in reverse.
- No autodiff engine is needed in the core, so the in-place buffers and concrete element types are not an obstacle.

## 2. Goals, non-goals, anti-goals

### Goals

1. First and second derivatives of every kept energy, kept state and port (matrix elements of the requested operators among kept states) at the root and at any node, with respect to named spec parameters.
2. Mixed second derivatives for chosen pairs of parameters, typically one design parameter against one noise parameter.
3. Derivatives that are exact for the calculation as performed: they agree with finite differences of `materialize` at fixed selection, bases and discrete choices.
4. Near-degenerate levels handled explicitly, never by dividing by a small detuning.
5. An ordinary gradient for an optimizer, through a thin outer rule (§4.6).

### Non-goals

- A reverse (adjoint) sweep. It may be added later if parameter counts grow; nothing here should prevent it.
- Third and higher order.
- Derivatives with respect to the coordinate transformation `S` (project 1 non-goal).
- Derivatives through basis construction: grids, cutoffs, and mean-field bases are held fixed (project 2; Eli, 2026-11-10: acceptable for the first and likely the final version).
- Pulse or time-dependent parameters.

### Anti-goals

- A derivative that is silently wrong because a contribution was truncated or omitted: a sum over only the computed states at an iterative node, or a parent that ignores how its children's bases move.
- A Taylor expansion used silently outside its range of validity near a small detuning (§4.4).
- Zero derivatives reported for a child whose basis or operators are stored numeric data (§4.8).
- Derivatives of a different calculation from the one `materialize` ran (different selection, different phases).

## 3. What a derivative means here

Parameters enter only through term coefficients; bases are parameter-free (architecture spec §11.1, unchanged).
A node's operator is `H = Σ_t c_t(θ) · (product of child ports and local operators)` plus the children's energies on the diagonal.

Differentiating along a parameter `x`, the node's `H⁽¹⁾` has three contributions:

1. the coefficients' own derivatives `∂c_t/∂x`;
2. the children's port derivatives, because the children's kept eigenvectors move;
3. the children's energy derivatives.

Omitting (2) or (3) gives a plausible but wrong answer, so a test must fail when either is removed (architecture spec §11.1, kept).

Held fixed while differentiating: which states are kept (selection), which solver ran, the real or complex classification, the phase convention's discrete choices, and every basis.
A warning is raised when a kept and an unkept state are close in energy at a selection boundary (architecture spec §11.3, kept), since the fixed-selection derivative is then fragile.

## 4. The design

### 4.1 Derivatives of the coefficients

`∂c_t/∂x` and the second derivatives are obtained by differentiating the spec's coefficient expressions symbolically on the Julia side *(my call)*; a `Coefficient` already keeps its expression.
The result is the same list of terms with differentiated coefficients, so `H⁽¹⁾` and `H⁽²⁾` are applied by the existing apply layer.

To differentiate with respect to element values, the spec must carry them as parameters.
Project 1's symbolic route provides this: the frame is fixed and the element values stay symbolic.

### 4.2 The per-node rule, first order

For a kept state `i` of a node with energy `E_i` and vector `ψ_i`:

- `E_i⁽¹⁾ = ⟨ψ_i| H⁽¹⁾ |ψ_i⟩`.
- `ψ_i⁽¹⁾ = Σ_{l≠i} |ψ_l⟩ ⟨ψ_l| H⁽¹⁾ |ψ_i⟩ / (E_i − E_l)`, summed over *all* states of the node's space.

How the sum is evaluated depends on what the node has:

- **Dense node**: the full spectrum is in hand, so the sum is exact and costs only matrix products.
- **Iterative node** (LOBPCG, shift-invert, Chebyshev): the part along the other computed states uses the sum; the part orthogonal to the computed states comes from a linear solve `(H − E_i) w = −Q H⁽¹⁾ ψ_i`, with `Q` the projector off the computed states.
  The existing preconditioner, and the shift-invert factorization where one exists, can be reused.
  Truncating to the computed states without the solve is not allowed.

The node's port derivatives follow by the product rule: for a port `P = V† O V`, `P⁽¹⁾ = V⁽¹⁾† O V + V† O⁽¹⁾ V + V† O V⁽¹⁾`, where `O⁽¹⁾` is nonzero only if `O` is itself built from children's ports.
These, with `E⁽¹⁾`, are what the parent consumes.

### 4.3 Second order

The same structure one order higher, using the second-order formulas of the notes (lines 346–360) for the state and the standard second-order energy.
Two points need care and are requirements, not suggestions:

- **Normalization.**
  The notes' formulas are for unnormalized states (intermediate normalization).
  Ports are matrix elements between normalized kept states, so the normalization correction at second order must be included.
- **Phase.**
  `materialize` fixes each kept vector's phase by a convention.
  Matrix elements between different states depend on relative phase, so the derivative of the state must be consistent with that convention, to both orders.
  The test is agreement with finite differences of the actual outputs, including off-diagonal ports of complex nodes.

A mixed derivative with respect to `(x, λ)` uses the first-order responses to `x` and to `λ` and one further solve per kept state for the pair.
The root's second energy derivative needs only first-order states at the root, but children must supply second-order ports.

Cost, with `P` parameters and `M` mixed pairs: `P` first-order passes over the tree, plus one second-order pass per parameter or pair requested.
A pass costs about one linear solve per kept state per iterative node and only matrix products at dense nodes.
Parameters are processed one at a time, so memory grows by a small multiple of the kept vectors, independent of `P`.

### 4.4 Near-degenerate levels

The non-degenerate formulas divide by `E_i − E_l`.
The protected circuits this package targets have near-degenerate doublets by design (the KITE's is 13.5 MHz), so this case is the norm.

Eli's rule: compare the detuning with the coupling.
For each pair of states, if `|E_i − E_l|` is smaller than a fraction `r` of `|⟨ψ_l| H⁽¹⁾ |ψ_i⟩| · δx`, and that matrix element is nonzero, the pair belongs to a near-degenerate block.
`r` defaults to 1/10 and is configurable down to 1/100.
`δx` is a scale for the parameter, supplied per parameter by the caller, since a detuning and a derivative matrix element have different units and cannot be compared without one *(the need for `δx` is my observation; Eli has not yet confirmed how it is supplied)*.

A matrix element that vanishes by symmetry removes the pair from consideration, as Eli specified, however small the detuning.

**What a block means** (proposed after the builder check; confirmed by Eli, 2026-11-10).
Eli's summary: the small parameter can always be made smaller, so a derivative near a small detuning is not wrong, it is only not trusted over the scale of interest.
The builder check pointed out that the first draft of this section was inconsistent.
For a doublet with splitting `Δ` and coupling `c` to a parameter `λ`, the transition frequency is `f = sqrt(Δ² + 4c²λ²)`, whose curvature at the sweet spot is `4c²/Δ`.
That is a true derivative, it is exactly what second-order dephasing needs, and it scales as the inverse detuning.
A rule that suppresses such values would return wrong derivatives, and a rule that passes non-derivative "block" values up to a parent would break agreement with finite differences everywhere above a doublet.

The proposed resolution separates two things.

1. **Point derivatives are always the true derivatives.**
   Whenever the detuning is nonzero they exist, the per-node rules of §4.2 and §4.3 give them, and they are what a parent consumes.
   Agreement with finite differences (at a step small against `Δ/c`) holds throughout the tree.
2. **The block test is a validity report, plus the tool for when the expansion is not valid.**
   A flagged block means the second-order Taylor expansion is not trustworthy over the scale `δx` for those states.
   For each flagged block the result carries the block's effective Hamiltonian: the block of `H`, `H⁽¹⁾` and `H⁽²⁾` in the block's states, with states outside the block folded in perturbatively.
   Diagonalizing that small matrix densely gives the block's energies and states at any finite displacement up to the scale `δx`; this is the "dense numerical diagonalization" of Eli's rule.
   A dephasing estimate over a noise amplitude uses it in place of the Taylor expansion.

When the detuning is exactly zero (to rounding) and the coupling is not, per-state point derivatives do not exist; they are reported as undefined, and only block-invariant derivatives and the effective Hamiltonian are returned.

The threshold follows from this reading and replaces the inequality at the top of this section: a pair is flagged when the coupling over `δx` exceeds a fraction `r` of the detuning, `|⟨ψ_l| H⁽¹⁾ |ψ_i⟩| · δx > r · |E_i − E_l|`, with `r` = 1/10 by default and 1/100 the stricter setting.

Smaller rules *(my calls, from the builder check)*:

- `scale` is required for every parameter for which the validity report is wanted; without it only the exact-degeneracy guard runs.
- For a mixed pair `(x, λ)` the blocks are the union of the blocks for `x` and for `λ`.
- A block with a member outside the kept set raises the selection-boundary warning of §3, using the same test; it does not force the selection to close over the block.
  Selection cluster closure remains Eli's open decision in STATUS.md and is not part of this project.

Every block found is reported with its states, detuning and coupling.

### 4.5 What the user calls

Names are illustrative.

```julia
sys = materialize(root, spec, θ)

d = derivatives(sys, spec, θ;
                wrt    = (:E_J1, :phi_ext),
                order  = 2,
                mixed  = ((:E_J1, :phi_ext),),
                scale  = (E_J1 = 0.1, phi_ext = 1e-3))

energies(d, :E_J1)              # dE/dE_J1 for every kept level
energies(d, :E_J1, :phi_ext)    # mixed second derivative
port(d, :phi_ext, :varphi_1, nport())   # derivative of a matrix-element table
eigenvectors(d, :E_J1)          # derivative of the kept states
blocks(d)                       # near-degenerate blocks that were found
```

Derivatives are available at interior nodes as well as the root.

The derivative pass reuses the stored kept vectors and ports and rebuilds node operators on demand, which is cheap.
A dense node does not currently keep its full spectrum: only kept vectors and energies are stored, and `Dense` with a `compute` limit finds only the lowest pairs.
Eli prefers retaining the spectrum (2026-11-10).
A dense node of dimension `N` retains its full eigenvector matrix: `8 N²` bytes real, `16 N²` complex, so 8 or 16 MB at `N = 1000` (the `Auto` dense cutoff), 200 or 400 MB at 5000, and 0.8 or 1.6 GB at 10 000 (the existing large-dense warning).
Retention is on by default for dense nodes up to a size cap, and opt-in above it *(the cap is my call; the `Auto` cutoff is the natural default)*.
A dense node without a retained spectrum is re-diagonalized by the derivative pass, with a notice.
**To keep in mind (Eli, 2026-11-10; a note, not a directive).**
The sum over states probably should not run over the whole spectrum in practice: contributions fall off with detuning, so summing over the nearest states, or as many as are needed to converge, may be enough and would cut both the retained memory and the work.
This is compatible with the anti-goal on silent truncation only if the truncation is controlled: the neglected remainder is bounded or checked against a convergence criterion, and the number of states used is reported.
The builder should measure how fast the sums converge on the test circuits before fixing the retention default.

Where a node was solved with a `compute` limit, retention means the forward solve computes the full decomposition there instead of the lowest pairs, which costs forward time as well as memory; this is measured and reported when built.

### 4.6 The outer rule for optimizers

A single differentiation rule on the user-facing "parameters to energies and ports" function, implemented by running the upward passes and contracting with the optimizer's cotangent.
With it, a loss written in plain Julia over energies and matrix elements can be handed to a gradient-based optimizer without the user calling `derivatives` by hand.
Second derivatives are requested explicitly through §4.5; a dephasing-aware loss uses the mixed derivatives as inputs.

Whether this rule lives in the core or in an extension, and which rule interface it targets, is left to the builder; the core must not depend on a particular autodiff engine *(my call)*.

### 4.7 Devices

CPU is implemented and verified first.
The derivative pass uses the same apply layer and storage as the forward solve, so it should run on a device unchanged; a device test follows once the CPU tests pass *(my call)*.

### 4.8 Children whose basis or operators are stored data *(my calls, from the builder check)*

- **Augmented children from project 2.**
  The basis is fixed, so the child is "its bare operator projected on a stored basis".
  Its derivatives come from projecting the differentiated bare operator on the same basis and applying the per-node rule to the projected problem.
  Project 2 is required to retain the basis vectors and bare terms for this (its §4.4).
- **`GivenSystem` children.**
  Their energies and operators are user-supplied numbers with no parameter dependence.
  `derivatives` raises an error for a tree containing one, unless the caller declares it parameter-independent.
- **Born–Oppenheimer reduced specs.**
  Their coefficients are numerical fits, so the dependence on circuit parameters is lost.
  `derivatives` with respect to a parameter that the reduced spec no longer carries is an error by construction (the parameter is not in the spec); no special handling is added.

## 5. Acceptance

Agreement with finite differences is the primary evidence for this project and is an explicit, named test suite, not a by-product of other tests (Eli, 2026-11-10).

1. **Per-node rules against finite differences**, for dense and iterative nodes, real and complex, first and second order, including off-diagonal ports.
   Target *(my call)*: relative agreement of `1e-6` or better where the finite difference itself is trustworthy.
2. **Whole-tree derivatives against finite differences** of `materialize` at fixed selection, on a two-level and a three-level tree.
3. **The three-contribution test**: removing the children's port derivatives or energy derivatives makes a test fail.
4. **Analytic checks** on single-node cases: an oscillator's frequency against its inductive energy, a transmon's charge dispersion derivative, a fluxonium's flux derivative.
5. **Equivalence with the paper's formulas** at a single dense node: the first-order rules reduce to `eq:eig_val_grad` and `eq:eig_vec_grad`, and the second energy derivative to `eq:eig_val_secondgrad`.
   On complex nodes the eigenvector rule agrees up to the phase-convention term along the state itself, which the paper's formula sets to zero.
6. **Iterative against dense**: on a problem small enough for both, the linear-solve route matches the full-spectrum route.
7. **Near-degeneracy** (under the proposed reading of §4.4): on the KITE doublet the block is detected and reported; the point derivatives match finite differences taken at a step small against the detuning, including the sweet-spot curvature `4c²/Δ`; and the block's effective Hamiltonian, diagonalized at finite displacements up to `δx`, reproduces a full re-solve at those displacements.
8. **A mixed derivative for dephasing**: `∂²f_{01}/∂x ∂φ_ext` against finite differences on a fluxonium, and on the KITE at a flux point off the sweet spot where the mixed derivative does not vanish by symmetry.
   The KITE tests use ordinary leaves; one further test differentiates a tree with an augmented child and checks it against finite differences at fixed basis.
9. **An optimization demonstration**: a small design problem (1 to 3 parameters) run to convergence with the outer rule, with the gradient checked against finite differences at the start and the end.
10. Architecture spec §11, §15 and §2 (the forward-mode non-goal) are amended, and STATUS.md records the change of approach.

## 6. Left to the builder

- All names and the file layout.
- The linear solver and preconditioner at iterative nodes, and when to reuse a factorization.
- How symbolic differentiation of coefficient expressions is implemented, provided the allowed expression grammar is covered.
- The exact form of the phase-consistent first- and second-order state corrections, provided acceptance items 1 and 2 hold.
- Storage of derivative data (per parameter, per node) and whether intermediate results are cached between first- and second-order passes.
- The data-valued `DiagonalOp` term entry, which was deferred to "plan 5" only because the old approach wrapped `materialize`; it is no longer blocked by this project and is not part of it.
