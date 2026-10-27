# Z4 three-way Harmonium: missing backend support

## Requested calculation and present outcome

Use the circuit-quantization object from `symmetry_derivations/four_well_harmonium.ipynb`, including its Z4 coordinates, B, B_static and static biases. The requested full-system hierarchy is `[[1,4,5],[2,6,7],[3]]`: one retained left KITE, one retained right KITE, and the fluxonium, followed by a three-way combination. No BO projection is involved.

The notebook CQ object was constructed and exported successfully. The native exported symbolic Hamiltonian agreed with CQ at all 48 sampled phase/charge points (maximum observed residual zero). Its provenance and numerical settings are saved in `data/harmonium_2d/z4_threeway/pilot_v1/export/export.json`.

KronFreeCircuits rejects the hierarchy before solving because composite/multimode ports are unsupported. Native scqubits 4.3.1 executes this hierarchy, but an independent test confirms that its separate phase-factor projection does not equal the required joint projection. Neither backend currently provides a validated spectrum for this exact retained-block partition through the tested native path.

## Coordinate and parameter conventions

Columns are `(theta_L,theta_R,theta_f,theta_SigmaL,theta_DeltaL,theta_SigmaR,theta_DeltaR)`. Rows follow CQ's `circuit.internal_nodes`; reordering rows into `(L,R,C,a_plus,a_minus,b_plus,b_minus)` yields the notebook's displayed Z4. No charge-decoupling shear is subsequently applied.

The original numerical candidate is retained: fluxonium EJ=28, total EC=8, EL=11 GHz; nominal KITE junction EJ=35, EC=21.9906 GHz and branch EL=11 GHz; 3000 fF outer shunts; original fractional Josephson disorders 0.012 and 0.008. All three physical loop biases are pi and all offsets and offset charges are zero.

The notebook has separate central junction and shunt capacitors. Both were assigned EC=16 GHz, giving total EC=8 GHz in parallel. This split is explicit; only the combined capacitance enters this static calculation with zero capacitive-return-loop flux. Static and offset flux allocations are copied from the notebook without replacement by a backend-selected gauge.

## Why the hierarchy requires a joint operator

The exported left-plus Josephson term is

`-35.21 cos(theta_L - theta_f - theta_SigmaL - theta_DeltaL - pi)` GHz.

At the final node it separates between the KITE and fluxonium sectors. The KITE factor is the joint operator

`A_L = exp[i(theta_L - theta_SigmaL - theta_DeltaL)]`.

If V_L contains the retained KITE eigenvectors, the correct operator passed to the final node is `V_L† A_L V_L`. Computing the primitive factors jointly before this single projection is essential. In general,

`V† A B C V != (V† A V)(V† B V)(V† C V)`.

The latter inserts retained-subspace projectors between factors and discards excursions through omitted states. The primitive phase factors commute, but their compressed operators need not. Equality is recovered for a complete child basis, or special invariant subspaces; it is not generally valid at a retained cutoff.

The right KITE has the same need. This obstruction is independent of the root eigensolver, its tolerance, and the choice of Lanczos versus LOBPCG.

## KronFreeCircuits: observed limitation

The tested native source is frozen under `pilot_v1/sources/KronFreeCircuits.jl`; its exact commit is in `SOURCE_COMMIT` and `pilot_v1/run.json`.

`src/materialize.jl::_check_composite` rejects a routed interaction with multiple factors belonging to one child node. The tiny test failed with:

> a term on modes [:varphi_1, :varphi_3, :varphi_4, :varphi_5] has several factors inside the child node with modes [:varphi_1, :varphi_4, :varphi_5]; composite (multi-mode) ports are not yet supported

Reproduction from the repository root:

```bash
julia +1.13.1 --project=data/harmonium_2d/z4_threeway/pilot_v1/sources/env -t 1 \
  data/harmonium_2d/z4_threeway/pilot_v1/sources/kronfree_harmonium_layouts.jl \
  data/harmonium_2d/z4_threeway/pilot_v1/export/circuit.json \
  data/harmonium_2d/z4_threeway/pilot_v1/settings.json \
  /tmp/z4-threeway-result.json
```

Choose a fresh result path. The saved pilot log records the original failure. No eigenvalues were computed by this Julia pilot.

## Required native Julia implementation

1. Represent a composite port with a canonical descriptor containing all its local mode factors and coefficients, rather than only one mode/operator pair. Include conjugation and coefficient/sign conventions in its identity.
2. Route each interaction to its common ancestor, collecting factors by child. A group with several local modes requests one joint child port.
3. Form the joint operator in the child's existing native product representation and project once with that child's retained eigenvectors. Reuse native tensor application rather than assembling the full seven-mode matrix. Do not replace this operation with multiplication of already compressed single-mode ports.
4. Propagate composite requests recursively through nested children. If several factors lie in one descendant, preserve them as a joint request until they can be applied together before projection.
5. Preserve conjugate/exponential phase conventions, scalar coefficients, static flux offsets and arithmetic type. Construct the final cosine from the joint exponential and its adjoint so Hermiticity is retained.
6. Cache by the complete port descriptor and the retained basis identity. Estimate memory costs explicitly; a retained composite operator may be dense even when primitive factors are sparse.
7. Replace `_check_composite` rejection only for cases fully implemented. Continue to reject unsupported combinations clearly.

This is a package feature, not a correction to the CQ Hamiltonian. No package patch, custom solver, or substitute Hamiltonian was introduced in this study.

## scqubits: requested hierarchy was executed as a diagnostic

The installed version is 4.3.1. A single native `scqubits.Circuit` imports the exact CQ-exported symbolic Hamiltonian, preserving its gauge and constant. Native subsystems use exactly `[[1,4,5],[2,6,7],[3]]`, retaining `[12,12,5]` states. Primitive charge cutoff attributes are 2 and the five extended grids have 9 points on ±4pi. This deliberately tiny calculation tests implementation, not convergence.

Its symbolic Hamiltonian remained unchanged. The eight computed eigenpairs have maximum native residual 1.29e-10 GHz. However, `Circuit._evaluate_matrix_cosine_terms` identity-wraps each phase factor separately into its root-child eigenbasis before multiplying the factors.

An independent validation-only calculation compared the joint native primitive KITE cosine projected once with the product of separately projected factors:

- Difference in operator 2-norm: **0.054062570334933474** (dimensionless).
- Correct projected cosine 2-norm: **0.8304666483664804**.
- Correct joint-projection check: **failed**.

A small eigenvector residual verifies the matrix that scqubits actually constructed; it does not repair this Hamiltonian-projection discrepancy. The raw native energies are preserved but explicitly labeled diagnostic, and must not be used as a physical reference or convergence point.

Evidence: `data/harmonium_2d/z4_threeway/scqubits_pilot_v1/run.json`, `joint_projection_control.npz`, and the saved installed cosine-evaluation source. Reproduction:

```bash
OPENBLAS_NUM_THREADS=1 /home/eweissler/anaconda3/envs/nphikite/bin/python \
  validation/z4_scqubits_projection_check.py \
  --spec data/harmonium_2d/z4_threeway/pilot_v1/export/circuit.json \
  --out /tmp/z4-scqubits-new-diagnostic
```

This script confines the independent projection comparison to validation. It does not supply a custom projected Hamiltonian to production calculations.

The corresponding scqubits fix would group cosine phase factors by root child, evaluate their joint operator within that child's native primitive/hierarchical representation, and identity-wrap/project it once. Nested hierarchies require the same recursive treatment. A native supported API for this operation would also suffice; the tested path does not provide it correctly.

## Acceptance tests for either backend

- Native flat versus untruncated hierarchical small systems: compare all low eigenvalues and the Hamiltonian action under the basis transformation.
- Genuinely truncated child: directly compare each joint projected operator against `V† A_joint V`; include a case where separate projection demonstrably fails (the saved pilot).
- Test both KITE signs, static pi offsets, nonzero flux offsets, and canonical coordinate ordering. Check Hermiticity and adjoints.
- Test joint operators spanning two and three modes, including descendants nested within a child. Retain regressions for existing singleton-port layouts.
- Compare the final native matrix/operator action against an independent small reference, then compare Lanczos and a block/Dense solver at the same retained Hamiltonian.
- Only after those implementation checks, vary each KITE and fluxonium retention independently, then primitive charge/grid/range cutoffs. Compare both absolute energies and the lowest eight transition energies across coordinate representations.

Current state: report and both backend pilots exported; no validated production Z4 spectrum; no numerical jobs left running by these pilots. Earlier campaigns remain paused.
