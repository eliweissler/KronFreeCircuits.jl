# Fluxonium with an explicit junction array (Di Paolo et al., arXiv:1912.01018, Fig. 3)

Not yet a suite case: the circuit-quantization version of the 2026-11-10/11 spike (`../../bench_archive/array_fluxonium/`,
untracked), whose hand-written `array_spec` put the offset charge on the on-site terms only, as Di Paolo's printed
Hamiltonian reads. That is the full quadratic form with offset r·n_g, r = 1 + (N − 1) C_Jb/C_J, and it made Δω01 3 to
14 times too small at z = 0.09, 0.10, 0.13, 0.14 (see the spike's README and
`~/arxiv_sources_kronfree_bench/notes/2510.08680_vs_1912.01018.md`). Here the circuit goes through circuit-quantization,
which gives every junction phase its own offset charge in the full form 4 (n − n_g)ᵀ E_C (n − n_g); all are set to n_g.

## Files

- `quantize.py N z [z ...]` (conda env `cq-z4`): the ring of N array junctions (C_J, E_J from z at ω_p/2π = 12.5 GHz) and
  the black-sheep junction (C_Jb = 7.5 fF, E_Jb = 8.9 GHz), C_0 = 0 (floating); coordinates the junction phases
  θ_i = φ_(i−1) − φ_i (compact) and the cyclic global phase; the loop flux on the black-sheep junction. Writes
  `specs/array_N<N>_z<z>.json` (parameters n_g, varphi_ext).
- `dmrg/dmrg_spec.jl` (its own environment: ITensors, ITensorMPS): DMRG on a spec, every term read from it (E_C, the
  offsets, the cosines); site eigenbases from KronFreeCircuits or the raw charge basis; excited state by penalty.

## Checks (2026-11-12)

- The N = 4, z = 0.10 spec against a dense Hamiltonian built by hand from the closed form E_C = E_CJ (I − a 11ᵀ),
  a = C_Jb/(C_J + N C_Jb), and the full-form offsets: E_C to 2e-15 GHz; the lowest ten levels at (n_g, φ_ext) =
  (0, π), (½, π), (¼, π), (½, 0.3π) to 1.3e-12 GHz.
- `dmrg_spec.jl` against the package's flat solve of the N = 6, z = 0.10 spec (ncut 3): E0, E1 and ω01 at n_g = 0 and ½
  identical to every printed digit (ω01 1.366659993 and 1.413859769 GHz) in both the charge and the eigen site basis.
- The spike's DMRG with its offsets switched to the full form (`NGMODE=full`) at N = 40, z = 0.10, D = 10:
  Δω01 = 2.309e-2 GHz against 1.660e-3 with the on-site form, on Di Paolo's curve.

## The N = 40 sweep against the single-mode model (2026-11-12)

`results/sweep-20261112-1518-z*.log`: DMRG on the circuit-quantization specs, z = 0.05 to 0.15, eigen site basis D = 10
(ncut 8), bond cap 150, cutoff and energy tolerance 1e-12, penalty weight 100, one CPU core per point (about 3.5 to 6 h).
`single_mode.jl` (specs from `quantize_single.py`: the black-sheep junction, C_J/N, and E_L = E_J/N; grid ±10π at
spacing 2π/40, no range warning) gives the single-mode ω01 and Di Paolo's Eq. 6, |Δω01| = 2 N ε0 |⟨T⟩1 − ⟨T⟩0|, with
T = e^{−2πin} from the package's `exp2pin(-1)` port. `plot_fig3.py` draws `results/fig3_comparison.png`.

| z | DMRG Δω01 (GHz) | Eq. 6, exact ε0 | Eq. 6, printed ε0 | DMRG ω01(0), ω01(½) | single-mode ω01 |
|---|---|---|---|---|---|
| 0.05 | 5.1e-7 | 8.8e-8 | 2.2e-7 | 0.4594, 0.4594 | 0.4976 |
| 0.06 | 2.19e-5 | 6.0e-6 | 1.43e-5 | 0.4037, 0.4036 | 0.4391 |
| 0.07 | 1.68e-4 | 1.21e-4 | 2.81e-4 | 0.3665, 0.3664 | 0.4000 |
| 0.08 | 1.24e-3 | 1.14e-3 | 2.60e-3 | 0.3406, 0.3393 | 0.3722 |
| 0.09 | 6.28e-3 | 6.45e-3 | 1.45e-2 | 0.3233, 0.3170 | 0.3514 |
| 0.10 | 2.31e-2 | 2.56e-2 | 5.72e-2 | 0.3163, 0.2932 | 0.3352 |
| 0.11 | 6.66e-2 | 7.85e-2 | 0.175 | 0.3257, 0.2591 | 0.3224 |
| 0.12 | 0.160 | 0.199 | 0.441 | 0.3625, 0.2022 | 0.3119 |
| 0.13 | 0.335 | 0.434 | 0.962 | 0.4415, 0.1064 | 0.3031 |
| 0.14 | 0.534 | 0.844 | 1.871 | 0.5806, 0.0469 | 0.2958 |
| 0.15 | 0.521 | 1.495 | 3.324 | 0.7989, 0.2774 | 0.2895 |

- The single-mode Eq. 6 has the same exponential divergence. With the exact ε0 (the ground-band width E0(½) − E0(0) of
  one junction's transmon, from the spec's site E_C and E_J) it lies on the DMRG at z = 0.08-0.10 (within 3-11 %) and
  above it beyond, the large-z breakdown Di Paolo reports. With the printed Eq. 5, which is √π times Koch et al.'s
  asymptotic band width and about 2.3 times the exact one, it runs about 2 times high; Di Paolo's own dashed line looks
  consistent with the correct band width (read off their figure), so Eq. 5 as printed is probably a typo.
- The single-mode ω01 lies between the n_g = 0 and ½ DMRG curves at large z, as in Di Paolo's top panel, and 15-40 MHz
  above both at small z (the array treated as a linear inductor).
- z = 0.05: at the DMRG's precision (5e-7 GHz against energies of order 1e3 GHz); z = 0.06-0.07 may be too.
- z ≥ 0.13 at n_g = ½, rerun with a third level (`NLEV=3`, 1500-sweep cap; `results/rerun3-z*.log`, 2026-11-13): the
  sweep stopped at the 400-sweep cap for the excited state at z = 0.13 and the ground state at z = 0.14, and ω01(½)
  falls 0.202, 0.106, 0.047 GHz then rises to 0.277 at z = 0.15, which could have been the penalty method finding a
  different second state (the risk Di Paolo names, superfluxo.tex l. 93). The reruns converge every state (314/52,
  499/50 and 137/57 sweeps), reproduce ω01(½) to 1e-6 GHz (0.106436, 0.046892, 0.277379), and put the third level,
  penalized against both, 9.17, 9.04 and 8.90 GHz above the ground state: no state was missed below it, and the dip is
  the spectrum's.
