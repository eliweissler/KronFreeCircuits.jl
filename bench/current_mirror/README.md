# The full current-mirror circuit (Weiss et al., arXiv:1908.04615) — SHELVED, work in progress

The full 2N-node circuit of Eq. `Ham` (not the paper's effective model), Tab. `table:parameters`: E_CB = 0.2,
E_CJ = 100, E_Cg = 200, E_J = 19 GHz, zero flux and offset charges. The paper publishes no full-circuit spectra (exact
diagonalization "feasible for N ≤ 3", full-model DMRG deferred); its numbers are effective-model ones (Fig. 4). Shelved
on 2026-11-13 before a symmetry-adapted root (below); nothing here is a suite case yet.

## Files

- `quantize.py [--rung | --junction] N ...` (conda env `cq-z4`): circuit-quantization specs. The junction ring is an
  island (to ground only through C_g), so cq removes its global rotor (type `fr`) and keeps the zero total-charge
  sector. Coordinates: `theta_i = phi_i - phi_2N` (default, `specs/cm_N<N>.json`), rung coordinates `a_j, b_j`
  (`_rung`; the same per-rung space as theta), or junction phases `psi_k` (`_junction`). E_C is checked against the
  inverse of the paper's capacitance matrix (Eq. `CapMatrix`) to 1e-14.
- `spec_nodes.py N ...`: a hand-built spec VARIANT (`specs/cm_N<N>_node.json`) with all 2N node phases compact, the
  island's global rotor kept, exactly Eq. `Ham`; C_g > 0 makes C invertible, the total charge Q is conserved and the
  Q ≠ 0 blocks sit 4 E_Cg/2N Q² higher (133 GHz at N = 3). Its Q = 0 block is checked equal to cq's E_C (rational
  arithmetic). Every rung (phi_j, phi_(j+N)) is then the same leaf. A cq option for user-given coordinates with an
  invertible capacitance matrix would replace it.
- `cm.jl` (`CM_N`, `CM_COORDS`, `CM_TREE` = flat | rungs | pairs, `CM_K`, `CM_K2`, `CM_ROOT` = lobpcg | lanczos | dense):
  KronFree solves. `momentum.jl`: symmetry labels of the exact flat eigenvectors. `dmrg_cm.jl` (env
  `../array_fluxonium/dmrg`): DMRG on a spec, raw charge basis, penalty method. `mf.jl`: mean-field rung leaves
  (EXPERIMENT: needs the uncommitted `src/meanfield.jl`, `src/augment.jl`).

## Findings (charge cutoff 10 per mode unless noted)

- Exact levels (from E0, GHz). N = 2 (flat 9 261 states; cutoff 13 identical): 3.958337 (×2), 7.685296, 7.747341,
  7.850589, 11.236907 (×2), 11.539931 (×2). N = 3 (flat 4.1 M states, LOBPCG, 958 s, 21 GB): 2.276451 (×2), 4.412031,
  4.480204 (×2), 4.494658, 6.448449 (×2), 6.535654 (×2), 6.587552.
- Lanczos (one start vector) missed partners of degenerate levels (N = 2: the 11.2369 and 11.5399 pairs); LOBPCG did
  not and is the default.
- Rung leaves under one root (cq theta): N = 2 exact at K = 160; N = 3 K = 40 off by up to 110 MHz (13 s), K = 80
  within 0.4 MHz (74 s), K = 120 exact to 1e-6 GHz (194 s). N = 4: K = 40 (27 min, 8.4 GB) and K = 60 (1 h 43 min,
  21 GB) agree to about 1 MHz (lowest levels 1.478, 1.498 GHz; whether that pair is a split ±k doublet is open);
  K = 80 (41 M root, 43 GB) was running when shelved.
- Pairwise tree (rungs 1–2 paired, then rung 3; cq theta, LOBPCG at the pair node): K2 = 200 / 400 / 800 off by
  0.1–0.4 GHz / up to 1 GHz / 4–146 MHz; K2 = 400 worse than 200 suggests LOBPCG mis-solving hundreds of pair levels.
- Plaquette leaves (junction coordinates, junctions j and j+N): fail (N = 2 ground state 6–14 GHz high): the frozen
  rung (agiton) charges are a cross-leaf condition in junction coordinates. Junction coordinates need cutoff 14.
- Mean-field rung leaves (with and without the bare ladder): no help. The product state is 0.85 GHz (N = 2) and
  6.6 GHz (N = 3) above the exact ground state; the mean-field states displace useful bare ones.
- DMRG (`dmrg_cm.jl`, Krylov dimension 20, 4 iterations per local solve; ITensorMPS' defaults left 0.57 GHz errors):
  N = 2 exact to 1e-8 GHz in 73 s; N = 3 levels 1–4 exact to 1e-6 GHz, level 4 alone 74 min (stopped at 2 h).
- Symmetry (`momentum.jl`, N = 2 and N = 3 at cutoff 8): ring translation T (node i → i+1) exact to 1e-12; every
  degenerate doublet is a ±k pair (counter-propagating collective modes); T^N = (-1)^m and C = T^N for every level
  checked, i.e. the low spectrum is even under C·T^N (the agiton flip n_i → -n_(i+N)).
- Node-phase spec (all 2N phases): N = 2 flat and K = 160 rung tree reproduce the cq levels exactly
  (E0 = -21.931314005 GHz); N = 3 with identical rung leaves, K = 80: every ±k doublet exactly degenerate, levels
  0.5–1.2 MHz low, E0 3.4 MHz high.

## Where it stopped

The next step is a symmetry-adapted root: with identical node-phase rung leaves the product basis is closed under T,
so the root splits into 2N momentum sectors (and the agiton-parity block), about 4N smaller per sector and with ±k
partners separated. That needs a sector projection at the root in KronFreeCircuits (package work).
