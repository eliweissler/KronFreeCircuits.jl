# Design: mean-field basis augmentation (2026-11-10)

Status: design for Eli's approval; project 2 of 3 before the feature freeze.
Supersedes `2026-10-28-mean-field-leaves-proposal.md` and `plans/2026-10-28-mean-field-leaves.md`, neither of which was implemented.
Depends on project 1 (`2026-11-10-circuit-quantization-interface-design.md`) for the frame loop.
Items marked *(my call)* were chosen by Claude and not discussed; everything else was decided with Eli on 2026-11-10.

## 1. Intent

A child node diagonalizes only the terms routed to it.
In a frame that mixes modes, every Josephson cosine straddles several children, so a bare leaf never sees a junction and its eigenstates are a poor basis for the parent.

The fix is the optimized-coordinate vibrational self-consistent field method the circuit-quantization paper adapts (its Appendix C, `apx:meanfield`): each child is solved in the mean field of the others, the product energy scores a coordinate frame, and the mean-field eigenstates become the child's basis for the correlated solve.
The parent's operator is never changed, so nothing is approximated; only the basis the parent sees changes.

A working prototype exists in Python (NumPy and SciPy): `circuit_quantization/product_state.py` and `examples/hierarchical_kite.py`.
This project builds the capability into the tree, reproduces the paper's result with it, and then tries it on the Harmonium.

### Why the earlier plan is replaced

- It was written before the periodic-gauge merge and cites code that no longer exists (a `charges` argument to `_materialize`, stale line numbers).
- It hid the whole calculation inside `materialize` behind one `MeanField(...)` option.
  Eli wants the pieces exposed, so the self-consistent loop is a few readable lines that can be changed.
- It left out the two lessons its own proposal recorded: appending the bare ladder, and capping how many states each well contributes.

## 2. Goals, non-goals, anti-goals

### Goals

1. Mean field as a capability of the tree, in four separable pieces (§4.1 to §4.4), plus a convenience wrapper.
2. A frame score: the self-consistent product-state ground energy of a spec, usable as the cost in an optimization over the secondary transformation `S`.
3. Reproduce Figure 8(b) of the circuit-quantization paper, or explain any disagreement to the point where it is clear which calculation is right (§6.1).
4. Find out whether augmented bases speed up the convergence of Harmonium simulations (§6.2).

### Non-goals

- Differentiating through the self-consistent loop.
  Project 3 holds mean-field bases fixed while differentiating (Eli, 2026-11-10: acceptable for the first and likely the final version).
- Gradients of the frame score with respect to `S` by anything other than finite differences or a derivative-free optimizer.
- A mean-field seed for `product_seed`, and the automatic shear of cosine-free modes; both stay deferred.
- Real arithmetic for mean-field children.
  The mean-field cosine weights are complex in general, so these children run complex in the first version.

### Anti-goals

- Changing the parent's operator, or double counting the mean-field terms in the parent.
- A calculation that can only be run as a black box.
- A basis construction that silently returns fewer states than requested, or silently drops a source.

## 3. The calculation, as the paper states it

In a frame `S` the Hamiltonian is `H' = 4 n'ᵀ E_C' n' + ½ φ'ᵀ E_L' φ' + b'ᵀ φ' − Σ_k E_Jk cos(ω'_kᵀ φ' + δ_k)` (paper Eq. `H_frame`).
The ansatz is a product of normalized states, one factor per child.
For single-mode children the mean-field Hamiltonian is the paper's Eq. `mean_field_h`:

`h_i = 4 E_C'_ii n_i² + ½ E_L'_ii φ_i² + (b'_i + Σ_{j≠i} E_L'_ij ⟨φ_j⟩) φ_i + 8 Σ_{j≠i} E_C'_ij ⟨n_j⟩ n_i − Σ_k E_Jk |z_ki| cos(ω'_ki φ_i + arg z_ki)`,

with `z_ki = e^{iδ_k} Π_{j≠i} X_kj` and `X_kj = ⟨e^{i ω'_kj φ_j}⟩`.

In tree terms this is one rule: for every term that straddles a child, keep the child's own factors and replace each outside factor by its expectation value in the given state of the child that owns it.
The rule covers multi-mode children as written, using the existing joint ports for factors that sit together in one outside child.

Three facts from the prototype shape the design:

- **The fixed point is not a minimizer.**
  The loop converges to different solutions from different starts, including saddles (the KITE's delocalized start lands at +3.83 GHz against wells at −8.08 GHz in the published frame).
  Several starts are required and the lowest energy is kept.
- **A product state sits in one well.**
  For the KITE's doublets each mean-field solution describes one of two wells, so leaves from one well miss the doublet partner (E₁ wrong by about 1 GHz at small `m`).
  The leaf basis has to combine several solutions.
- **The cost is scale-free.**
  The product energy does not change under column scaling of `S`, and in the harmonic limit it is flat along the compact-noncompact shear.
  Grids must be sized per frame, and the frame cost needs a small tie-break.

## 4. The pieces

Names are illustrative; the builder may change them.

### 4.1 Mean-field terms from given states

Given a node, the terms routed at or above it, and for each *other* child a state, return the extra terms for one child: each straddling term reduced to that child's factors, with a numeric complex coefficient equal to the original coefficient times the outside expectation values.

"A state" for another child is any of *(my call on the exact forms)*:

- a materialized child and the index of one of its kept states;
- a materialized child and a set of weights over its kept states (a state average);
- a bare table of expectation values, for starting points that correspond to no solved state (a mode pinned at a displacement, or fully delocalized with every `X = 0`).

Expectation values come from the child's ports and joint ports, so nothing new is needed below the node.

**Depth** *(my call, from the builder check)*.
Mean field acts among the children of one node, one level deep.
If a child is itself a branch, its own children stay bare: the mean-field terms are routed down into its subtree as ordinary terms, and the child is solved as a whole.
Every well's vectors for a child then live on the same space, which is what the union of §4.4 needs.
Self-consistency nested inside a child is not part of this project.

The node need not be the root.
For terms routed above the node, whose factors lie outside it altogether, the caller supplies the outside expectation values as a table; the wrapper of §4.5 offers this only when asked and otherwise treats such terms as absent from the mean field.

### 4.2 Diagonalizing a mean-field child

The child is materialized through the existing machinery with its bare routed terms plus the terms from §4.1.
Solvers, selection and ports work as for any node.
The child's reported energies in this step are mean-field energies, and the result is labelled as such so it cannot be mistaken for a bare solve.

### 4.3 The self-consistent loop

Built from §4.1 and §4.2:

- all children are updated together from the same set of expectation values (Jacobi), as the prototype does;
- convergence is on the product energy and on the change in the expectation values;
- after the energy rises twice, updates are damped with mixing 0.5;
- several starts are run and each distinct converged solution is returned with its product energy, not only the lowest.

The product energy is the expectation of the full node operator in the product state, computed without double counting.

Starts: the delocalized start, and displacement starts with each mode, compact ones included, at `0` or `±π` (27 for the three-mode KITE, as in the notebook).
The loop and the frame score use each child's ground state by default; weights over kept states are available but not the default.
Which converged solutions feed the union of §4.4 is an explicit, reported argument, defaulting to those within a small energy window of the lowest.
All combinations are enumerated when they are few; otherwise the caller supplies starts or a cap, and the cap is reported.
A solution from a previous, nearby frame can be passed as a warm start.

Defaults follow the prototype (relative energy tolerance `1e-11`, at most 300 iterations) unless the builder finds a reason to change them, which is then recorded.

### 4.4 The union basis

The child's final basis is assembled from several sources of vectors on the child's own space:

- the lowest eigenstates of the child's mean-field Hamiltonian for each retained solution (one set per well);
- optionally the bare ladder, the eigenstates of the child's own routed terms alone.

Each source can carry a cap: a number of states, or an energy window above that source's lowest state.
The sources are merged level by level (first state of each source, then the second of each, and so on), orthonormalized by Gram–Schmidt applied twice, and a vector whose remainder has norm below `1e-6` is dropped.
Merging stops at the requested `m`; when sources are capped, the uncapped source (normally the bare ladder) fills the remainder.

The child's *bare* operator is then projected onto this basis and diagonalized, and the child is handed to the parent with energies and ports in that basis.
From the parent's point of view it is an ordinary child.

The augmented child keeps its union basis vectors (on the child's own space) and its bare routed terms, not only the projected matrices.
Project 3 differentiates such a child as "the bare operator projected on a fixed stored basis", which needs both; a child reduced to numeric energies and port matrices would silently report zero derivatives.

The function reports, per child, how many vectors came from each source and how many were dropped.
If fewer than `m` independent vectors exist, that is an error unless the caller allowed it.

### 4.5 The wrapper

One call that runs §4.3 and §4.4 for a node with sensible defaults and returns the augmented children, the solutions found, and the product energy.
It is a convenience over the pieces and adds no behaviour of its own.

### 4.6 Grids per frame

A function that proposes a phase grid (extent and number of points) or a charge cutoff for each mode of a spec at given parameter values, from the mode's zero-point width and the junction energies it sees, as `default_bases` does in the prototype.
The self-consistent loop can enlarge a grid whose ground state reaches the edge (the prototype grows by 1.5 times, up to three times, when the edge weight exceeds `1e-8`).

## 5. The frame optimization

The loop Eli described, in one Julia session:

1. Define the circuit and take a default frame (automatic or normal-mode) through project 1.
2. For a trial `S`, get the numeric spec in that frame through project 1's loop route.
3. Run the self-consistent loop and take the lowest product energy.
4. Return the cost `E_prod(S) + 10⁻⁴ F_NM(S)`, where `F_NM` is the normal-mode cost used as a tie-break.
5. An outer optimizer varies `S` using finite-difference gradients or a derivative-free method.

`S` is varied only within the admissible family: the compact block is fixed (integer, unimodular, enumerated separately), and the free variables are the compact-noncompact shear and the noncompact block in polar form `e^A e^B` with `A` antisymmetric and `B` symmetric.
The parameterization and `F_NM` come from circuit-quantization through the interface *(my call)*, so the Julia side does not reimplement them.

The optimizer is not a core dependency; it lives in the benchmark or test environment *(my call)*.
The loop itself is a short script in `bench/`, written against the pieces, so it can be read and changed.

## 6. Acceptance

### 6.1 The KITE figure (pass or explain)

The reference is the derivation notebook `appendix_c_product_states.ipynb` in the paper's repository, which reproduces every number in Appendix C.
No committed script generates `figs/kite_levels.pdf`, and `examples/hierarchical_kite.py` uses different grids and a smaller reference solve, so neither is the reference.

Setup: the cos 2φ KITE with `E_J = 16.83`, `E_CJ = 4.82`, `E_CS = 0.072`, `E_L = 1.27` per arm (GHz), 3% junction asymmetry, half flux.
Three frames: the published coordinates, the normal modes, and the product-state optimum `S*` (paper Eq. `kite_S_opt`).
Leaves: two-well mean-field leaves, no bare ladder, as in the figure.
Quantity: the largest error among the four lowest levels against the reference energies, for `m = 8, 12, 16, 20`.

**The reference is incomplete as found (builder check, 2026-11-10).**
The notebook computes only two of the figure's twelve points (optimum frame, `m = 8` and `12`), and has no KITE numbers in the normal-mode frame.
Eli will locate the source of the figure's data before this project starts.
If it cannot be found, the fallback is a committed Python script that extends the notebook's functions to all twelve points, cross-checked against the values the paper quotes in its text; the steps below are written against whichever reference results.

The work proceeds in the order that makes a disagreement traceable:

1. Pin the notebook's intermediate numbers as fixtures, generated by a committed script: the transformed coefficients in each frame, the converged product energy of each well, and the lowest mean-field energies per mode.
2. Match those with the same grids (charge cutoff 33 states, about 117 and 116 grid points, order-8 stencils).
   Target *(my call)*: `1e-6` GHz on product energies and mean-field levels.
3. Reproduce the reference ground energy `E_0 = −9.523995` GHz independently with this package's own solver at a size comparable to the notebook's 1,254,400-state solve.
4. Reproduce the twelve points of Figure 8(b).
   Target *(my call)*: each error within a factor of two of the notebook's value, with the same ordering of the three frames at every `m`.
5. Separately, run the frame optimization of §5 from the identity and confirm it reaches a frame whose product energy matches the notebook's optimum (both wells near −9.43 GHz); `S*` itself is only defined up to column scale and sign.

Where a target is missed, the project is not done until the cause is identified and written down.
Known differences to check first:

- the paper says modes are updated in turn, the code updates them together;
- the paper says all `3^N` starts, the examples use five hand-picked ones, the notebook uses 27 plus the delocalized start;
- the paper's `F_NM` sums squared raw entries, the package's is scale-free, and the notebook normalizes columns first.

### 6.2 The Harmonium benchmark (exploratory)

No accuracy target; the result is reported whichever way it comes out *(my call, agreed in outline with Eli)*.

On the Z4 three-way Harmonium layouts already used for the pilot, in the pilot's coordinates as exported, compare at equal kept dimension: bare children, well-only children, and wells plus the bare ladder with and without a cap on states per well.
Running the frame optimization of §5 on the seven-mode Harmonium is a separate, larger step and is attempted only if this first comparison shows a gain *(my call, from the builder check)*.
Report level errors against the best available reference and the cost of each construction.

The prototype numbers suggest what to expect: on the Harmonium, full well ladders lost to bare leaves above the ground doublet, and two states per well plus the bare ladder did best.
This benchmark is what exercises multi-mode children, the caps of §4.4, and start selection when wells cannot be enumerated.

The pilot's scripts and data live outside this repository; Eli supplies their location when this step starts.

### 6.3 Other tests

- Each piece has unit tests on small cases with known answers: two coupled oscillators (mean field is exact for a product Hamiltonian), a fluxonium pair, and a case with a compact mode.
- The product energy is an upper bound on the exact ground energy in every test.
- The parent spectrum with augmented children converges to the same values as with bare children as `m` grows.
- The frame-invariance properties from the prototype's tests hold: the product energy is unchanged under column scaling and sign of `S`.

## 7. Left to the builder

- All names, and the file layout (`src/meanfield.jl` is the natural place).
- How the union child is delivered to the parent, provided ports and joint ports reach the parent correctly and the retention requirement of §4.4 holds (which rules out a plain `GivenSystem` as it stands).
- Sequencing against project 1: the pieces of §4 and steps 1 to 4 of §6.1 can be built against committed JSON fixtures; only §5 and step 5 need the interface.
- How warm starts are stored between frames.
- How mean-field children interact with the real-arithmetic classification, given that they run complex.
- Memory handling when many starts are run on a device.
