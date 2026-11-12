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
