# Born–Oppenheimer reduction: design

Date: 2026-10-19.
Status: draft for Eli's review; acceptance criterion and the `DiagonalOp` deferral decided 2026-10-20.
Author of record: Eli Weissler, with Claude as brainstorming partner.

This document describes a single-surface Born–Oppenheimer (BO) reduction for KronFreeCircuits.jl, and its first acceptance target, the zero-pi one-dimensional model of Di Paolo et al.
It extends Section 13.1 of the architecture design (`2026-10-15-kronfree-architecture-design.md`), whose three binding requirements it builds on.
Decisions marked **(decided)** were made in conversation with Eli and are not open.
Decisions marked **(my call)** were made by Claude as reasonable defaults and may be overridden.
Anything under "Left to the builder" is the builder's to settle.

## 1. Intent

A BO reduction treats a slow mode's coordinate as a classical parameter, diagonalizes the remaining (fast) modes at each value of it, and quantizes the slow mode again with the resulting energy surface as its potential.
Eli wants this to reproduce published reduced models and to produce his own: a one-mode effective Hamiltonian whose coefficients are fitted functions of circuit parameters such as the external flux.

The first target is Di Paolo et al., [arXiv:1811.07921v1](https://arxiv.org/abs/1811.07921v1), Appendix B ("One-dimensional effective model"): the symmetric zero-pi with the extended mode φ fast and the compact mode θ slow.
Its parameter mapping, published coefficients and an acceptance protocol are already written up in `docs/science/born-oppenheimer.md`; that note is part of this design and is not repeated here.

The second target, deferred **(decided)**, is the capacitively shunted KITE of Smith et al., [arXiv:1905.01206v2](https://arxiv.org/abs/1905.01206v2), with the compact mode varphi slow and the two extended modes fast (Section 9).

**Ordering (decided).** This feature goes ahead of plan 5 (automatic differentiation).
Nothing here needs derivatives.

## 2. Goals, non-goals, anti-goals

### Goals

- Turn a mode of a spec into a named parameter, giving a spec for the fast modes alone.
- Compute BO surfaces: the lowest few fast energies as functions of the slow coordinate, at given values of the other parameters.
- Quantize a compact slow mode with a computed surface, as a Fourier series through the existing `cosines` interface **(decided)**.
- Fit surfaces computed at many parameter values to a model **linear in its unknown constants** **(decided)**, by linear least squares, and return the constants, their residuals, and a reduced one-mode spec whose coefficients are expressions in the remaining parameters.
  Quadratic polynomials in one or more parameters around a working point are the expected common case **(decided)**; Di Paolo's forms, such as `E_α − E_β cos φ_ext`, are another.
- Reproduce Di Paolo's published zero-pi model (Section 7).

### Non-goals

- Derivative couplings, Born–Huang corrections, and multi-surface (coupled-band) models.
  Single-surface BO only, as in the source.
- Nonlinear fits, where a fitted constant sits inside a nonlinear function (for example a phase δ in `E cos(φ_ext/2 + δ)`) **(decided)**.
- Automatic discovery of a model's functional form; the user supplies the basis functions **(decided)**.
- Charge coupling between the slow mode and a fast mode.
  It is rejected with an error in this feature; the KITE needs it removed first, by a shear (Section 9).
- A batched small-matrix eigensolver over all slow grid points; this is a later GPU optimization (architecture Section 13.1).
- Automatic differentiation of surfaces or fits (plan 5 and later).
- A data-valued `DiagonalOp` entry into a `Term`, which quantizing an extended slow mode with an unfitted surface needs (Section 4.3).
  It is deferred to plan 5 **(decided)**, because autodiff wraps `materialize` and the entry should be designed with it (STATUS.md item 9 on `plan4-julia-integration`).
  Until then an extended slow mode is reduced only through a fit.

### Anti-goals

- Silently approximating anything.
  A term that cannot be carried over exactly when a mode becomes a parameter is an error naming the term, never dropped (architecture Section 13.1).
- Passing off the published fit coefficients as results of a run, or reporting a reduced model without its fit residuals and truncation error (`docs/science/born-oppenheimer.md`).
- Requiring a BO spectrum to equal the full spectrum.
  The BO error is measured and recorded, separately from discretization and solver error.
- New surface plumbing that would make the existing kron-free path, real-arithmetic selection or the spec format harder to change.

## 3. Constraints

- Julia, inside KronFreeCircuits.jl; no new heavy dependencies.
  The Fourier transform of a surface is small (tens to hundreds of points) and may be a direct sum; a dependency on FFTW is not justified **(my call)**.
- Spec files follow kronfree-spec version 1, whose format authority is `~/src/circuit-quantization/docs/spec-format.md`.
  A spec written by this package must load in both packages.
- Energies in GHz; phases dimensionless; charges in units of 2e (architecture Section 14).
- Default-suite time is already over budget (STATUS.md, open item 1).
  New default-suite tests should cost seconds; convergence studies and the full reproduction go in the gated suite (`KRONFREE_SLOW_TESTS=1`).

## 4. The pieces

### 4.1 Promote a mode to a parameter

A transformation from a `Spec` and a mode name to a new `Spec` over the remaining modes, with one new parameter holding the slow coordinate (named by the caller, defaulting to the mode's name).
It works on the `Spec`, not on the term list **(my call)**: the fast problem is then an ordinary spec that every existing path (`materialize`, real-arithmetic detection, saving) accepts unchanged, and the KITE shear (Section 9) is also a spec-to-spec transformation.
The term-list requirement of architecture Section 13.1 stays in force; this does not replace it.

How each part of the spec is carried over, for slow mode `s` and fast modes `f`:

| Spec part | Carried over as | When it is an error |
| --- | --- | --- |
| `E_C[s,s]`, `n_g[s]` | dropped from the fast spec, with a logged note; kept for the slow stage | never |
| `E_C[s,f]` (charge coupling) | — | always, naming the entry |
| `E_L[s,s]` | constant term `½ E_L[s,s] s²` | never |
| `E_L[s,f]` | `flux_linear[f]` gains `E_L[s,f] s` (symmetrized as in `circuit_terms`) | never |
| `flux_linear[s]` | constant term `flux_linear[s] s` | never |
| a cosine with `omega[s] ≠ 0` | same cosine over the fast modes, `offset` gains `omega[s] s` | never |
| a cosine with only `omega[s] ≠ 0` | constant term `−energy cos(omega[s] s + offset)` | never |
| a charge cosine with `kappa[s] ≠ 0` | — | always (non-diagonal in the slow coordinate) |
| the offset-charge constant `4 n_gᵀ E_C n_g` | its `(s,s)` part belongs to the slow stage, not the fast spec | never |

Every one of these is exact; the two error rows are the only places an exact carry-over does not exist.

The error checks are structural: they look at the coefficient expressions, not at parameter values **(my call)**.
A coupling that vanishes only at particular values (for example `full_zero_pi.json` at `dCJ = 0`) is removed first with a substitution helper, part of this feature, that replaces named parameters of a spec by numbers and drops the entries that become exactly zero.

### 4.2 Surfaces

A function that takes a fast tree (an ordinary `Node`), the fast spec, the remaining parameter values, and a grid of slow-coordinate values, and returns a surface object.
At each grid point it calls `materialize` and records the lowest `bands` energies, including the spec constant so that absolute energies are kept.
The surface object holds the grid, an energies matrix (bands × points), the parameter values it was computed at, and enough metadata to reproduce it.
`bands` defaults to 2 **(my call)**: band 1 is the surface used, and band 2 gives the gap to the first fast excitation, which the acceptance protocol asks for.
A band is the k-th lowest energy at each point, an envelope, not a branch followed through crossings **(my call)**; catalogue test L6 must show the difference explicitly.

For a compact slow mode the grid is uniform and periodic (M points on a half-open 2π interval), so the surface's Fourier coefficients up to harmonic M/2 are exact for the sampled data.
For an extended slow mode the grid is any set of points covering the region the fit should describe; the surface reaches the slow stage only through a fit (Section 4.3).

### 4.3 The slow stage

A reduced one-mode spec for the slow mode: its `E_C[s,s]` and `n_g[s]` from the original spec, plus the surface as potential.

- **Compact slow mode (decided).** The surface's harmonics become `cosines` entries in charge basis.
  Each harmonic k contributes two entries, `cos(kθ)` (offset 0) and `cos(kθ − π/2)` (that is, `sin kθ`), so their energies are linear in the surface and no amplitude–phase split is needed **(my call)**.
  The zeroth harmonic is the constant.
  Harmonics are kept up to a caller-chosen `K`, or until the dropped harmonics' total amplitude falls below a tolerance; the dropped amplitude is reported **(my call)**.
  With all harmonics up to M/2 kept, this is the "full numerically computed potential" of the source.
- **Extended slow mode (decided, deferred).** The intended representation is the surface on the slow `PhaseGrid` as a data-valued `DiagonalOp` potential.
  Today a `Factor` carries only a symbolic `Port`, so a data-valued operator has no way into a `Term` (architecture Section 20, plan-1 debt); that entry is designed with plan 5, not here.
  In this feature an extended slow mode is reduced only through a fit (Section 4.4), whose `x` and `x²` terms become `flux_linear` and `E_L` in the reduced spec.

### 4.4 Fits

Input: surfaces computed at a set of parameter samples (for zero-pi, flux values), and a model.
A model assigns to each slow-coordinate basis function a list of parameter basis functions, each a scalar expression string in the spec expression grammar; the fitted quantity is the band-1 surface, and the model is `Σ c_ij g_i(slow) h_ij(params)` with unknown constants `c_ij`.

- For a compact slow mode, the slow basis functions are `1`, `cos kθ` and `sin kθ`.
  Di Paolo's model is `cos 2θ → ["1", "cos(varphi_ext)"]`, `cos θ → ["cos(varphi_ext/2)"]`, `1 → [...]`.
- For an extended slow mode, the slow basis functions are those a spec can express: `x`, `x²`, `cos(kx)` and `sin(kx)` **(my call)**.

The fit is one linear least-squares solve over all (slow point, parameter sample) pairs, or equivalently per slow basis function after projecting each surface, since the compact basis is orthogonal on a uniform grid.
Output: the constants, per-constant residual or standard error, the overall residual, and the reduced spec, whose coefficient expressions are the model's expressions with the fitted constants written in as numbers **(my call)**.
Parameters held fixed while computing the surfaces (for zero-pi, `E_J`, `E_L`, `E_CJ`) are substituted by their values everywhere in the reduced spec, including the slow stage's `E_C[s,s]` and `n_g[s]`, with the substitution helper of Section 4.1 **(my call)**.
A parameter stays free only if it is fitted over (it appears in the model's parameter basis functions) or the fast spec does not depend on it, as with `n_g` on the slow mode; both are exact.
Otherwise a fitted potential would silently assume one value of a parameter that stays a free knob elsewhere.

### 4.5 Saving a reduced spec

Julia has `load_spec` but no writer.
A writer for kronfree-spec version 1 is part of this feature, so a reduced spec can be saved, loaded, swept, and later differentiated like any other.
It records provenance in underscore metadata: the source spec, the slow mode, grids and cutoffs, parameter samples, the model, residuals, truncation error and package version (`_generator` as in the format document).

## 5. Representation of the zero-pi target

The fast spec is the symmetric zero-pi spec with θ promoted.
Use a spec without the `2 E_J` constant, or subtract it once, as `docs/science/born-oppenheimer.md` explains; note the factor-of-two charging-energy mapping there.
`n_g = 0` is the recorded baseline assumption.
The fast leaf is φ on a `PhaseGrid` or `FockBasis`; its range and spacing (or cutoff) are refined independently of the θ grid and the slow charge cutoff.

## 6. Tests

- **L6 (catalogue).** A small conditional Hamiltonian with an independently calculable surface: two harmonic oscillators coupled through `E_L[s,f]`, whose fast surface is exactly quadratic in the slow coordinate, for the extended path through a fit; and a two-level crossing and avoided crossing, to separate the band envelope from a followed branch.
  Repeated parameter points, state ordering and surface insertion against an explicit small-space reference, as the catalogue describes.
- **Promotion.** Each row of the table in Section 4.1, including both errors, against a hand-built fast spec; fixing the slow coordinate in the full spec's Hamiltonian at a grid point reproduces the fast spec's spectrum exactly.
- **Fourier path.** A surface with known harmonics round-trips through the reduced spec to the same slow spectrum as a dense reference built on a periodic grid with the surface on its diagonal.
- **Fit.** Synthetic surfaces from known constants are recovered to round-off; a quadratic model in two parameters is recovered; the reduced spec saves and loads in Julia.
- **Interchange.** A committed reduced-spec fixture is validated once by a Python script beside `test/fixtures/generate_specs.py`, which loads it with circuit-quantization; Python is never a test-time dependency **(my call)**, as for the existing generators.
- **Independent BO model.** Protocol step 3 of the reference note is in scope: the test builds H_φ(θ) at each grid point from plain dense matrices, outside the spec, term and `materialize` path, and compares the surface and slow spectrum **(my call)**.
  A throwaway spike on 2026-10-19 matched such a reference to 2e-12 (STATUS.md item 9 on `plan4-julia-integration`).
- **Zero-pi reproduction (gated).** Section 7.

## 7. Acceptance: the published zero-pi test

Follow the protocol in `docs/science/born-oppenheimer.md` at flux samples `0, 0.125, 0.25, 0.375, 0.5`.
Passing means **(decided)**:

- The published input ratios (0.001, 0.378, 0.000175, 0.165 of ħω_p, with ω_p/2π = 40 GHz) are treated as exact.
  They are not replaced by values inferred from the plasma-frequency identity.
- E_α and E_γ agree with the published values (1.8608e-2 and 2.6625e-5 of ħω_p) to 0.1% relative, as a hard assertion in the gated suite.
  Numerical convergence (grids, cutoffs, solver tolerance) must be demonstrably tighter than 0.1%.
  A miss is a bug to be chased until it is explained, not a recorded finding.
- The flux dependence matches the source qualitatively, and the fast ground energy is π-periodic in θ at half flux.
- E_β is reported with an error estimate and is not required to match; it is small enough to be lost in numerical error.
- The slow spectrum is compared with the full 2D zero-pi at the same parameters, and the per-level BO error is recorded, not bounded.

A fixture from this run identifies its producer as `independent BO numerics` and carries the metadata the reference note lists.

Work starts from `plan4-julia-integration`, which is `main` plus the STATUS note on that spike **(my call)**.

## 8. Left to the builder

- Function and type names, and whether surfaces and fits are plain structs or richer objects.
- Whether the surface sweep reuses the previous point's materialization (`previous`) and whether it is threaded.
- The exact provenance keys in a saved reduced spec, within the format's underscore-metadata rules.
- Grid sizes, cutoffs and tolerances for the reproduction, established from refinement data.
- Where the reproduction script and fixture live (`bench/`, `test/science/`, or both).

## 9. Deferred: the KITE (decided)

After the zero-pi test passes, the KITE of Smith et al. (`eq:sym_ham`) is the next target, with varphi slow and (φ, θ) fast as one two-mode fast node.
Its kinetic term `½(N − N_g − η)²` couples the slow charge N to the fast charge η, which Section 4.1 rejects.
The plan is a shear transformation that removes the charge–charge coupling, as circuit-quantization does, applied as a spec-to-spec transformation before promotion.
The paper has no published BO surface: its cos 2φ model (`eq:2m_ham`) comes from an instanton path and Fourier fit, so the KITE check is BO against the full three-mode spectrum, with `eq:2m_ham` as a qualitative comparison.
The shear's effect on varphi's periodicity must be checked when that work starts.
