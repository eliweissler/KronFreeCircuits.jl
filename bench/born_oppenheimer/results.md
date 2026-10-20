# Zero-pi Born–Oppenheimer reproduction (Di Paolo et al., arXiv:1811.07921v1, Appendix B)

Written by `bench/born_oppenheimer/zero_pi.jl` on 2026-10-20 09:35 (Julia 1.13.2, KronFreeCircuits 0.1.0, 8 threads); run time 75 s.

Inputs (decided 2026-10-20, design §7), ratios to ħω_p with ω_p/2π = 40 GHz: E_Cφ = 1/(16·0.165) = 0.378788, derived from the source's ω_p = sqrt(8 E_J E_CJ)/ħ with C_φ = 2C_J (E_Cφ = E_CJ/2), instead of the printed 0.378;
E_L = 0.001, E_J = 0.165, E_Cθ = 0.000175 as printed.
In GHz: `E_Cphi = 15.15151515, E_Ctheta = 0.007, E_J = 6.6, E_L = 0.04`; n_g = 0 (it does not enter the surface).
Fluxes `[0.0, 0.125, 0.25, 0.375, 0.5]` with `varphi_ext = 2π·flux`; θ̃ on `periodic_grid(M; start=-π/2)`, M uniform points of [−π/2, 3π/2).
Fast stage: `test/fixtures/zero_pi_bo.json` promoted at θ, φ on a `PhaseGrid`, `Dense` (2 lowest pairs) per point.
Model (`fit_surfaces`): `:constant => ["1", "cos(varphi_ext)", "cos(2*varphi_ext)", "cos(3*varphi_ext)", "cos(4*varphi_ext)"]` (exact per flux on five fluxes; on a uniform θ grid it is orthogonal to the cosines, so it does not bias the coefficients),
`(:cos, 2) => ["1", "cos(varphi_ext)"]`, `(:cos, 1) => ["cos(varphi_ext/2)"]`; E_α = c[cos2,1], E_β = −c[cos2,cos φ_ext], E_γ = c[cos1], each / 40 GHz.

**Acceptance (design §7: 0.2% on E_α and E_γ, derived E_Cφ): PASS.**
With the printed E_Cφ = 0.378 (superseded), E_γ misses by +0.96% (rounding-extremes table below).

## Result at the converged setting

Setting: PhaseGrid(±12π, 481; order=8), Δ=0.1571, M = 32; edge warnings: none.

| coefficient | obtained / ħω_p | fit standard error | published / ħω_p | relative deviation | 0.2% criterion |
| --- | ---: | ---: | ---: | ---: | --- |
| E_α | 1.858685e-02 | 8.3e-06 | 1.8608e-2 | -1.137e-03 | pass |
| E_β | 1.004665e-08 | 1.1e-05 | 1.0073e-8 | -2.616e-03 | reported only |
| E_γ | 2.659240e-05 | 1.2e-05 | 2.6625e-5 | -1.224e-03 | pass |

The fit standard errors come from the residuals, which are the omitted harmonics (cos 3θ, cos 4θ, …), not numerical noise;
the numerical error of the base setting (its change to the adjacent refinement in each block, below) is 3.2e-08 for E_α and 2.1e-08 for E_γ, relative; E_β's estimate is the round-off floor 2.1e-06 (below).
Fit residuals: rms 2.884e-03 GHz, max 4.147e-03 GHz.
Fast gap min(band 2 − band 1) over θ and flux: 2.890294 GHz.
π-periodicity at half flux, max |U(θ+π) − U(θ)|: 4.49e-12 GHz (bound used by the gated test: 1e-5·E_γ = 1.06e-08 GHz).
Solver check (flux 0.125, θ̃ = 0.3): ShiftInvert(tol=1e-10) vs Dense, max |ΔE| over the two bands = 2.8e-12 GHz.
The gated test `test/test_born_oppenheimer_reference.jl` recomputes the five surfaces and the fit at this setting, single-threaded:
about 15 s run alone, 4 s inside the gated suite (measured 2026-10-20).
Flux dependence: a_2 varies by 8.0e-07 GHz across the fluxes (E_β), a_1 follows cos(φ_ext/2) and vanishes at half flux, as in the source.

Fourier cosine coefficients a_k of U(θ) = a_0 + Σ a_k cos kθ + b_k sin kθ (GHz):

| flux | a_0 | a_1 | a_2 | a_3 | a_4 | a_5 | a_6 | max abs b_k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.809398705 | -1.063696e-03 | -7.434736e-01 | -9.643726e-05 | 4.078504e-03 | -1.905707e-06 | -4.641077e-05 | 3.5e-13 |
| 0.125 | 0.809398606 | -9.827270e-04 | -7.434737e-01 | -8.909627e-05 | 4.078485e-03 | -1.760613e-06 | -4.641214e-05 | 5.6e-13 |
| 0.25 | 0.809398365 | -7.521461e-04 | -7.434740e-01 | -6.819106e-05 | 4.078438e-03 | -1.347457e-06 | -4.641546e-05 | 1.8e-13 |
| 0.375 | 0.809398125 | -4.070583e-04 | -7.434743e-01 | -3.690459e-05 | 4.078390e-03 | -7.292076e-07 | -4.641877e-05 | 4.7e-13 |
| 0.5 | 0.809398026 | -3.783328e-13 | -7.434744e-01 | 1.951911e-14 | 4.078371e-03 | -4.978934e-13 | -4.642015e-05 | 4.2e-13 |

## Refinement study

Each block varies one setting about the base (L = 12π, N = 481, order 8, M = 32); Δ is held fixed in the L block.
`ΔE_α`, `ΔE_γ`, `ΔE_β` are relative changes from the previous row; the chosen row is the first without an edge warning
whose next refinement changes E_α and E_γ by less than 1e-5. The base setting is at or beyond every chosen row.

### phi range L

| setting | M | E_α | E_γ | E_β | ΔE_α | ΔE_γ | ΔE_β | edge warning | time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PhaseGrid(±6π, 241; order=8), Δ=0.1571 | 32 | 1.859045e-02 | 2.642247e-05 | -8.41204e-08 |  |  |  | yes | 6.0 |
| PhaseGrid(±8π, 321; order=8), Δ=0.1571 | 32 | 1.858686e-02 | 2.659217e-05 | 9.96320e-09 | 1.9e-04 | 6.4e-03 | 1.1e+00 | yes | 1.1 |
| PhaseGrid(±10π, 401; order=8), Δ=0.1571 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00467e-08 | 2.8e-07 | 8.7e-06 | 8.4e-03 | yes | 0.9 |
| PhaseGrid(±12π, 481; order=8), Δ=0.1571 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 | 4.4e-11 | 1.5e-09 | 7.6e-07 | no | 1.1 |
| PhaseGrid(±16π, 641; order=8), Δ=0.1571 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 | 7.9e-13 | 3.3e-10 | 4.2e-07 | no | 1.7 |

Chosen: row 4.

### phi points N (L = 12π)

| setting | M | E_α | E_γ | E_β | ΔE_α | ΔE_γ | ΔE_β | edge warning | time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PhaseGrid(±12π, 241; order=8), Δ=0.3142 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 |  |  |  | no | 0.6 |
| PhaseGrid(±12π, 481; order=8), Δ=0.1571 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 | 5.3e-08 | 5.8e-08 | 2.7e-07 | no | 1.1 |
| PhaseGrid(±12π, 721; order=8), Δ=0.1047 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 | 2.1e-10 | 1.5e-09 | 1.0e-06 | no | 2.0 |
| PhaseGrid(±12π, 961; order=8), Δ=0.0785 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00467e-08 | 7.1e-12 | 2.2e-09 | 2.1e-06 | no | 4.1 |

Chosen: row 1.

### stencil order

| setting | M | E_α | E_γ | E_β | ΔE_α | ΔE_γ | ΔE_β | edge warning | time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PhaseGrid(±12π, 481; order=4), Δ=0.1571 | 32 | 1.858697e-02 | 2.659248e-05 | 1.00467e-08 |  |  |  | no | 1.1 |
| PhaseGrid(±12π, 481; order=6), Δ=0.1571 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 | 6.7e-06 | 2.9e-06 | 8.1e-06 | no | 1.1 |
| PhaseGrid(±12π, 481; order=8), Δ=0.1571 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 | 3.2e-08 | 2.1e-08 | 8.1e-07 | no | 1.1 |

Chosen: row 1.

### theta samples M

| setting | M | E_α | E_γ | E_β | ΔE_α | ΔE_γ | ΔE_β | edge warning | time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| PhaseGrid(±12π, 481; order=8), Δ=0.1571 | 8 | 1.858801e-02 | 2.659185e-05 | 1.01639e-08 |  |  |  | no | 0.6 |
| PhaseGrid(±12π, 481; order=8), Δ=0.1571 | 16 | 1.858685e-02 | 2.659240e-05 | 1.00467e-08 | 6.2e-05 | 2.1e-05 | 1.2e-02 | no | 0.7 |
| PhaseGrid(±12π, 481; order=8), Δ=0.1571 | 32 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 | 3.9e-12 | 6.3e-11 | 7.0e-07 | no | 1.1 |
| PhaseGrid(±12π, 481; order=8), Δ=0.1571 | 64 | 1.858685e-02 | 2.659240e-05 | 1.00466e-08 | 5.2e-14 | 5.1e-11 | 1.1e-07 | no | 1.9 |

Chosen: row 2.

E_β's changes plateau at about 2e-06 relative (round-off): it is a 4e-7 GHz flux variation of a_2 extracted from energies of order 1 GHz, and in the N block its change grows as the grid is refined. Its recorded error estimate is at least this floor.

## Rounding extremes of the inputs

Each input at the ends of its printed precision, the others at a baseline (derived E_Cφ = 0.378788 or printed 0.378); E_L's printed `10^{-3}`
is read both as 1.0e-3 (± 0.05e-3) and as one figure (± 0.5e-3). The corners take the derived reading, E_Cφ = 1/(16 E_J) following E_J.
Same numerical setting as above; deviations relative to the published coefficients.

| inputs (ratios to ħω_p) | E_L | E_Cφ | E_J | E_α | dev | E_β | dev | E_γ | dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| derived baseline (acceptance inputs) | 1.00e-03 | 0.378788 | 0.1650 | 1.858685e-02 | -1.137e-03 | 1.0047e-08 | -2.616e-03 | 2.659240e-05 | -1.224e-03 |
| printed baseline, E_Cφ = 0.378 | 1.00e-03 | 0.378000 | 0.1650 | 1.862507e-02 | +9.172e-04 | 1.0277e-08 | +2.030e-02 | 2.687976e-05 | +9.569e-03 |
| E_Cφ = 0.3775 (either baseline) | 1.00e-03 | 0.377500 | 0.1650 | 1.864940e-02 | +2.225e-03 | 1.0427e-08 | +3.513e-02 | 2.706391e-05 | +1.648e-02 |
| E_Cφ = 0.3785 (either baseline) | 1.00e-03 | 0.378500 | 0.1650 | 1.860080e-02 | -3.871e-04 | 1.0130e-08 | +5.693e-03 | 2.669700e-05 | +2.704e-03 |
| derived baseline, E_J = 0.1645 | 1.00e-03 | 0.378788 | 0.1645 | 1.847679e-02 | -7.051e-03 | 9.9414e-09 | -1.307e-02 | 2.646678e-05 | -5.942e-03 |
| derived baseline, E_J = 0.1655 | 1.00e-03 | 0.378788 | 0.1655 | 1.869722e-02 | +4.795e-03 | 1.0153e-08 | +7.931e-03 | 2.671846e-05 | +3.510e-03 |
| derived baseline, E_L = 0.95e-3 | 9.50e-04 | 0.378788 | 0.1650 | 1.855898e-02 | -2.634e-03 | 6.4768e-09 | -3.570e-01 | 2.093692e-05 | -2.136e-01 |
| derived baseline, E_L = 1.05e-3 | 1.05e-03 | 0.378788 | 0.1650 | 1.861423e-02 | +3.346e-04 | 1.5086e-08 | +4.977e-01 | 3.319888e-05 | +2.469e-01 |
| derived baseline, E_L = 0.50e-3 | 5.00e-04 | 0.378788 | 0.1650 | 1.827605e-02 | -1.784e-02 | 8.5330e-12 | -9.992e-01 | 5.926936e-07 | -9.777e-01 |
| derived baseline, E_L = 1.50e-3 | 1.50e-03 | 0.378788 | 0.1650 | 1.884370e-02 | +1.267e-02 | 2.1740e-07 | +2.058e+01 | 1.442864e-04 | +4.419e+00 |
| printed baseline, E_J = 0.1645 | 1.00e-03 | 0.378000 | 0.1645 | 1.851479e-02 | -5.009e-03 | 1.0170e-08 | +9.606e-03 | 2.675271e-05 | +4.797e-03 |
| printed baseline, E_J = 0.1655 | 1.00e-03 | 0.378000 | 0.1655 | 1.873565e-02 | +6.860e-03 | 1.0386e-08 | +3.110e-02 | 2.700727e-05 | +1.436e-02 |
| printed baseline, E_L = 0.95e-3 | 9.50e-04 | 0.378000 | 0.1650 | 1.859711e-02 | -5.851e-04 | 6.6297e-09 | -3.418e-01 | 2.116948e-05 | -2.049e-01 |
| printed baseline, E_L = 1.05e-3 | 1.05e-03 | 0.378000 | 0.1650 | 1.865253e-02 | +2.393e-03 | 1.5424e-08 | +5.312e-01 | 3.354835e-05 | +2.600e-01 |
| printed baseline, E_L = 0.50e-3 | 5.00e-04 | 0.378000 | 0.1650 | 1.831330e-02 | -1.584e-02 | 8.8123e-12 | -9.991e-01 | 6.019665e-07 | -9.774e-01 |
| printed baseline, E_L = 1.50e-3 | 1.50e-03 | 0.378000 | 0.1650 | 1.888275e-02 | +1.477e-02 | 2.2144e-07 | +2.098e+01 | 1.455388e-04 | +4.466e+00 |
| corner E_L = 0.95e-3, E_J = 0.1645, E_Cφ = 1/(16 E_J) = 0.379939 | 9.50e-04 | 0.379939 | 0.1645 | 1.839395e-02 | -1.150e-02 | 6.1934e-09 | -3.851e-01 | 2.050332e-05 | -2.299e-01 |
| corner E_L = 0.95e-3, E_J = 0.1655, E_Cφ = 1/(16 E_J) = 0.377644 | 9.50e-04 | 0.377644 | 0.1655 | 1.872494e-02 | +6.284e-03 | 6.7723e-09 | -3.277e-01 | 2.137825e-05 | -1.971e-01 |
| corner E_L = 1.05e-3, E_J = 0.1645, E_Cφ = 1/(16 E_J) = 0.379939 | 1.05e-03 | 0.379939 | 0.1645 | 1.844862e-02 | -8.565e-03 | 1.4456e-08 | +4.351e-01 | 3.254343e-05 | +2.223e-01 |
| corner E_L = 1.05e-3, E_J = 0.1655, E_Cφ = 1/(16 E_J) = 0.377644 | 1.05e-03 | 0.377644 | 0.1655 | 1.878075e-02 | +9.284e-03 | 1.5742e-08 | +5.628e-01 | 3.386534e-05 | +2.719e-01 |

Within the printed precision (E_L read as 1.0e-3) E_α ranges over -1.15% to +0.93% and E_γ over -22.99% to +27.19% of the published values; with E_L read as one figure E_γ ranges over -98% to +447%. The input rounding, not the numerics, sets the attainable agreement with the published coefficients.

## Checks of the mapping and the fit

- Charging energies: the source's q²/(2C) with q = 2e n and E_C = e²/(2C) is 4 E_C n² (eq:symm_qubit_Hamiltonian); the spec's `4 n E_C n` makes `E_C[φ,φ] = E_Cphi` (zero_pi_bo.json).
- Inductive term: the source's E_L φ² is the spec's `E_L[φ,φ] = 2 E_L` (½ φ E_L φ form).
- No 2E_J constant (Appendix B adds none); a constant would only shift a_0, which does not enter E_α, E_β, E_γ.
- E_β sign: E_2 = E_α − E_β cos φ_ext and the spec cosine energy is c₁ + c₂ cos φ_ext, so E_β = −c₂; the obtained E_β is positive, as published.
- The surface agrees with an independent dense reference to 1e-10 (test/test_bo.jl, "zero-pi BO against an independent dense BO").
- An independent discretization outside the package (sinc-DVR kinetic matrix, φ ∈ ±12π, 601 points, plain `eigvals`, same θ grid):

| flux | a_1 sinc-DVR | a_1 package | a_2 sinc-DVR | a_2 package |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | -1.0636964023e-03 | -1.0636964017e-03 | -7.4347359854e-01 | -7.4347359870e-01 |
| 0.25 | -7.5214614922e-04 | -7.5214614871e-04 | -7.4347400041e-01 | -7.4347400057e-01 |

Alternative fits on the converged surfaces (on a uniform θ grid the harmonics are orthogonal, so adding terms or changing the
constant's basis leaves E_α, E_β, E_γ unchanged):

| model | E_α | dev | E_γ | dev | E_β |
| --- | ---: | ---: | ---: | ---: | ---: |
| U_0 one constant | 1.858685e-02 | -1.137e-03 | 2.659240e-05 | -1.224e-03 | 1.0047e-08 |
| U_0 per flux (5 basis functions), the acceptance model | 1.858685e-02 | -1.137e-03 | 2.659240e-05 | -1.224e-03 | 1.0047e-08 |
| acceptance model plus cos 3θ, cos 4θ terms | 1.858685e-02 | -1.137e-03 | 2.659240e-05 | -1.224e-03 | 1.0047e-08 |

## Slow stage against the full 2D zero-pi

Slow stage: the reduced spec (`F.spec`, the fitted model; test/fixtures/bo/zero_pi_reduced.json) and the full numerical potential
(`slow_spec` of each surface, all harmonics), each on `ChargeBasis(20)` with `Dense`.
Full 2D: `test/fixtures/zero_pi_bo.json` on `[:theta => ChargeBasis(20), :phi => PhaseGrid(±12π, 481; order=8), Δ=0.1571]` (the BO's φ grid), `ShiftInvert(tol=1e-9)`, 6 levels.
Energies are absolute (GHz). No bound is asserted on the BO error.

Charge-cutoff refinement, max |ΔE| over the 6 levels against the next cutoff (GHz):

| slow-stage ncut | max over fluxes |
| ---: | ---: |
| 5 → 10 | 1.0e-01 |
| 10 → 15 | 5.2e-05 |
| 15 → 20 | 1.4e-10 |
| 20 → 30 | 3.2e-15 |

| 2D ncut | flux 0 | flux 0.5 |
| ---: | ---: | ---: |
| 10 → 15 | 4.6e-05 | 4.6e-05 |
| 15 → 20 | 9.0e-11 | 9.2e-11 |
| 20 → 30 | 5.0e-13 | 7.3e-13 |

Per-level comparison (GHz): E_2D, and the BO errors E_fit − E_2D (reduced spec) and E_U − E_2D (full numerical potential).

| flux | level | E_2D | E_fit − E_2D | E_U − E_2D |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1 | 0.263904748 | -2.229e-03 | -4.952e-05 |
| 0.0 | 2 | 0.266092881 | -2.367e-03 | -4.855e-05 |
| 0.0 | 3 | 0.639085002 | +3.696e-04 | -1.510e-04 |
| 0.0 | 4 | 0.641006613 | +3.347e-04 | -1.485e-04 |
| 0.0 | 5 | 0.980706856 | +1.399e-03 | -2.586e-04 |
| 0.0 | 6 | 0.983338945 | +1.443e-03 | -2.520e-04 |
| 0.125 | 1 | 0.263987820 | -2.234e-03 | -4.948e-05 |
| 0.125 | 2 | 0.266009390 | -2.362e-03 | -4.859e-05 |
| 0.125 | 3 | 0.639157757 | +3.683e-04 | -1.509e-04 |
| 0.125 | 4 | 0.640933541 | +3.361e-04 | -1.486e-04 |
| 0.125 | 5 | 0.980744989 | +1.400e-03 | -2.585e-04 |
| 0.125 | 6 | 0.983300578 | +1.442e-03 | -2.520e-04 |
| 0.25 | 1 | 0.264224478 | -2.249e-03 | -4.938e-05 |
| 0.25 | 2 | 0.265771720 | -2.347e-03 | -4.870e-05 |
| 0.25 | 3 | 0.639364877 | +3.645e-04 | -1.507e-04 |
| 0.25 | 4 | 0.640725657 | +3.398e-04 | -1.489e-04 |
| 0.25 | 5 | 0.980842150 | +1.403e-03 | -2.584e-04 |
| 0.25 | 6 | 0.983202855 | +1.439e-03 | -2.521e-04 |
| 0.375 | 1 | 0.264578911 | -2.271e-03 | -4.922e-05 |
| 0.375 | 2 | 0.265416274 | -2.324e-03 | -4.885e-05 |
| 0.375 | 3 | 0.639674061 | +3.588e-04 | -1.503e-04 |
| 0.375 | 4 | 0.640415707 | +3.455e-04 | -1.493e-04 |
| 0.375 | 5 | 0.980948114 | +1.406e-03 | -2.583e-04 |
| 0.375 | 6 | 0.983096329 | +1.435e-03 | -2.522e-04 |
| 0.5 | 1 | 0.264996262 | -2.298e-03 | -4.904e-05 |
| 0.5 | 2 | 0.264998503 | -2.298e-03 | -4.904e-05 |
| 0.5 | 3 | 0.639992607 | +3.523e-04 | -1.499e-04 |
| 0.5 | 4 | 0.640096844 | +3.519e-04 | -1.496e-04 |
| 0.5 | 5 | 0.980995221 | +1.408e-03 | -2.583e-04 |
| 0.5 | 6 | 0.983048989 | +1.434e-03 | -2.523e-04 |

The doublet splittings (levels 2 − 1, 4 − 3, 6 − 5) are set mainly by E_γ cos(φ_ext/2) and shrink at half flux to the tunneling splitting.
E_fit − E_2D includes the truncation of the two-harmonic model (omitted a_3, a_4, …, with a_4 ≈ 4e-3 GHz); E_U − E_2D, from the full numerical potential,
is the single-surface BO error itself.

