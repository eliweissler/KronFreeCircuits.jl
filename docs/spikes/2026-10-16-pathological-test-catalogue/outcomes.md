# Catalogue outcomes

Date: 2026-10-17.
Recorded against the case identifiers of [README.md](README.md), with the statuses of its Section 8: implemented/passing, implemented/failing, unsupported-by-contract, deferred.
Package: `src/` at commit `21acda2` (Plan 2 Task 9); Julia 1.13.2, LOBPCGEigensolver 0.1.4, ITensors 0.9, KrylovKit 0.10.4; macOS (Apple silicon, 8 threads), CPU only.
Device repeats (Task 7) live in `test/cuda/test_device.jl` and run only on a CUDA machine; they were not rerun for this record.

Every reference below is assembled without the apply routine under test: explicit Kronecker products of the bases' declared operators (`primitive_hamiltonian` in `test/helpers.jl`), closed forms, or analytic spectra.
The default suite runs with `julia --project=. -e 'using Pkg; Pkg.test()'`; the slow cases with `KRONFREE_SLOW_TESTS=1` added (`test/test_zero_pi_reference.jl`).
Work counts are H columns (`n_matvec`); they are observations, not acceptance criteria.

## Summary

| ID | Status | Test file: testset |
| --- | --- | --- |
| O1 | implemented/passing | `test/test_apply.jl`: "O1: axis identity, equal sizes, and complex factor order (ITensor dense path)"; device: `test/cuda/test_device.jl`, same name |
| O2 | implemented/passing | `test/test_apply.jl`: "O2: shrinking/growing blocks, views, and stale scratch (ITensor dense path)"; device: same name |
| O3 | implemented/passing | `test/test_apply.jl`: "O3: extreme shifts and unusual bands" |
| O4 | implemented/passing | `test/test_apply.jl`: "O4: fused diagonal and cancellation"; device: same name |
| O5 | implemented/passing | `test/test_apply.jl`: "O5: workspace ownership"; transport buffers: `test/test_selection.jl`: "transport through two changed children; input untouched; basis change rejected" |
| S1 | implemented/passing | `test/test_solvers.jl`: "S1: exact and wrong-subspace seeds; bounded recovery"; device: same name |
| S2 | implemented/passing | `test/test_solvers.jl`: "S2: zero, duplicate, nearly dependent and badly scaled seeds" |
| S3 | implemented/passing | `test/test_solvers.jl`: "S3: H = 0, H = cI, tight clusters and a cutoff inside a cluster"; device: same name |
| S4 | implemented/passing | `test/test_solvers.jl`: "S4: exact resonance and invalid regularization" |
| S5 | implemented/passing | `test/test_solvers.jl`: "S5: no locking, several locks at once, one active column, complete locking"; device: same name |
| S6 | implemented/passing | `test/test_solvers.jl`: "S6: aH + bI, scaled tolerance and eta; finite shift below the diagonal" |
| S7 | implemented/passing | `test/test_solvers.jl`: "S7: constant diagonal, strong off-diagonal coupling" |
| S8 | implemented/passing | `test/test_solvers.jl`: "S8: dimensions, arguments and explicit nonconvergence" |
| S9 | implemented/passing | `test/test_materialize.jl`: "S9: partial, resized and reordered warm starts" |
| F1 | implemented/passing | `test/test_solvers.jl`: "F1: free rotor through n_g = 1/2 (operator level)"; `test/test_materialize.jl`: "F1: free rotor through n_g = 1/2, warm-started through materialize" |
| F2 | implemented/passing; the `n_g = 1/2` selection is implemented/failing by contract gap (selection rules do not close over near-degenerate clusters; decision pending, Eli) | `test/test_solvers.jl`: "F2: cos 2φ charge-parity sectors (operator level)"; `test/test_materialize.jl`: "F2: cos 2φ charge-parity sectors, sector-confined warm start through materialize"; `test/test_catalogue.jl`: "F2: connected with a parity-preserving port (public path)" |
| F3 | implemented/passing | `test/test_materialize.jl`: "F3: identical uncoupled zero-pis (small fixture)"; device: same name; slow: "F3 (slow): identical uncoupled zero-pis, larger leaves, several seeds" |
| F4 | implemented/passing | `test/test_catalogue.jl`: "F4: exchange symmetry of identical, weakly coupled transmons" |
| F5 | implemented/passing | `test/test_materialize.jl`: "F5: sweep through a resonance: fine steps, one jump, a step landing on it" |
| F6 | implemented/passing (`:jacobi` leaf solves fail explicitly in the slow run; recorded) | `test/test_catalogue.jl`: "F6: symmetric to asymmetric full zero-pi (small discretization)"; slow: "F6 (slow): symmetric to asymmetric full zero-pi, cold and warm, :ritz and :jacobi" |
| F7 | implemented/passing | `test/test_apply.jl`: "F7: complex three-mode cosine as an adjoint pair (operator level)"; device and public flat tree: `test/cuda/test_device.jl`: "F7: complex three-mode cosine as an adjoint pair (device operator and LOBPCG on the flat tree)" |
| F8 | implemented/passing | `test/test_materialize.jl`: "F8: deliberately inadequate hierarchical truncation"; `test/test_catalogue.jl`: "F8: hierarchical vs Dense on the same projection vs the primitive reference, cutoffs increased"; slow: "F8 (slow): inadequate truncation of a strongly coupled full zero-pi, both cutoffs increased" |
| F9 | implemented/passing | `test/test_catalogue.jl`: "F9: undersized and changing phase grids (small)"; slow: "F9 (slow): zero-pi on undersized and changing phase grids"; rejection across intervals: `test/test_materialize.jl`: "T2: follow with a changed primitive basis of equal dimension is rejected" |
| F10 | implemented/passing | `test/test_materialize.jl`: "F10: a raw basis child beside a materialized branch, reversed order, an extra level" (extended in Task 10) |
| T1 | implemented/passing | `test/test_selection.jl`: "T1: overlaps through several levels against explicit primitive vectors; gauge covariance"; `test/test_materialize.jl`: "T1: follow through a permuted grandchild and two changed children (three levels)" |
| T2 | implemented/passing (rejection is the contract) | `test/test_selection.jl`: "T2: basis compatibility"; `test/test_materialize.jl`: "T2: follow with a changed primitive basis of equal dimension is rejected" |
| T3 | implemented/passing | `test/test_selection.jl`: "T3: following collisions, ties and vanishing overlap" |
| T4 | implemented/passing | `test/test_catalogue.jl`: "T4: phase-anchor cancellation" |
| P1 | implemented/passing (host); device repeat deferred | `test/test_catalogue.jl`: "P1: projected square versus square of the projection (Fock)" |
| P2 | implemented/passing (host); device parity deferred | `test/test_catalogue.jl`: "P2: port covariance under V -> V U (permutations, complex phases)" |
| B1 | implemented (benchmark, not a test) | `bench/` and [the Plan 2 benchmark results](../2026-10-17-plan2-benchmark/results.md) (Task 8) |
| L1 | deferred | real-arithmetic milestone after Plan 2 |
| L2 | deferred | autodiff milestone |
| L3 | deferred | autodiff milestone, after T1–T4 and P1–P2 |
| L4 | deferred | optional: only when a wraparound compact phase basis is implemented |
| L5 | deferred | optional sweep optimization after ordinary warm starts and physical alignment |
| L6 | implemented/passing (2026-10-20, BO plan) | `test/test_bo.jl`: "BO surfaces" (two-level crossing at n_g = ½ and the avoided crossing at 0.45, envelope ordering, repeated points), "fits" (two oscillators coupled through `E_L[s,f]`, exactly quadratic surface through `:x2`), "compact slow stage" (surface insertion against a dense periodic-grid reference), "zero-pi BO against an independent dense BO"; gated: `test/test_born_oppenheimer_reference.jl`: "zero-pi BO: Di Paolo E_alpha and E_gamma (gated)". The surface enters the slow stage through a fitted or Fourier reduced spec, not a data-valued `DiagonalOp` (still deferred to plan 5). |

## Cases added or extended in Task 10

### F2 (public-path `connected`)

Configuration: `4 E_C (n − n_g)² − E_2 cos 2φ`, `E_C = 1.2`, `E_2 = 3`, `ChargeBasis(20)`, Dense, `compute = 12`, `connected([1], (:varphi_1, expiphi(2)), 0.05)`.
Reference: the explicit primitive matrix and its two parity blocks diagonalized separately.

- `n_g = 0.2`: implemented/passing. The kept states (computed indices 1, 5, 9) each lie in the ground state's sector to `1e-20`, and their energies belong to that sector's spectrum.
- `n_g = 1/2`: **implemented/failing by contract gap.** Selection rules do not close over near-degenerate clusters; the decision is pending (Eli).
  - Cause: the even and odd levels pair up to rounding, because the wavefunctions never reach the edge of the window. `Dense` may return any rotation inside each pair, and the public path has no way to request parity-resolved vectors.
  - Effect: `connected` follows those rotations across both sectors and keeps one member of a pair without its partner. On this machine the kept indices are `[1, 2, 3, 4, 5, 6, 8, 9, 10, 11]`: the partners 7 (pair at 57.62) and 12 (pair at 144.01) are left out, and the parity leakage of the kept span is `‖[Π, VV']‖ = 0.0138`.
  - Minimal reproduction: the inline `cos 2φ` spec in the testset (`E_C = 1.2`, `E_2 = 3`, `n_g = 0.5`), `ChargeBasis(20)`, Dense with `compute = 12`, `keep = connected([1], (:varphi_1, expiphi(2)), 0.05)`.
  - What the test asserts: the outcome depends on how the BLAS rotates each pair, so the test asserts the diagnosis instead, which holds on every machine.
    The clusters come from the independently diagonalized parity blocks: the eigenvalues within `1e-8` (relative) of a kept energy.
    The kept span lies inside those clusters, and it leaks parity (`‖[Π, VV']‖ ≥ 1e-8`) only if it is rank-deficient in at least one of them.
    An `@info` line names the split clusters and the leakage; here, the pairs at E = 57.62 and 144.01, leakage 0.0138.
  - Proposed fix: cluster closure by an energy tolerance in `select`. Every rule would keep or drop a near-degenerate cluster whole, consistent with spec §11.4 (near-degenerate kept states).
  - No change was made to the package.

### F3 (slow)

Configuration: `two_zero_pi.json`; identical leaves `ChargeBasis(8)` × `PhaseGrid(-6π, 6π, 151; order=8)` (2567 states), Dense, keep 12; the root operator (144 states) solved by `LOBPCG(tol=1e-10)` with seeds 101–103.
Oracle at `g = 0`: the sorted pairwise sums and explicit product-state projectors.
Elsewhere: the parent assembled from the leaf energies and `V' n_θ V` (not the ports), and the leaf residuals against the primitive leaf matrices.

- `g = 0`: the cutoffs ending on a complete cluster are 3, 4, 6, 8, 10, 12, 14 and 16; those inside an exchange pair are 2, 5, 7, 9, 11, 13 and 15.
  At `m` = 3, 16 (complete) and 2 (cut), the energies and complete-cluster projectors match.
  For the cut, the lower complete subspace matches and the boundary vector lies in the degenerate pair; all three seeds take 68–164 H columns.
- Seeds of exchange-symmetric product states only (`m = 16`), a 1 % detuning warm-started from the identical-leaf vectors, and `g = 0.05` seeded from symmetric eigenstates only (three seeds, 168 H columns each) all recover the complete lowest subspace.

### F4

Configuration: `two_transmons.json` with identical transmons (`E_J = 30`, `E_C = 1.2`, `n_g = 0.25` on both), `g = 0.05`; leaves `ChargeBasis(8)` keep 6; root `LOBPCG(tol=1e-10)`, `compute = 5`.
Reference: the 289-dimensional primitive matrix and its exchange operator; the projected Hamiltonian `W' H W` with `W = V₂ ⊗ V₁`.

- The primitive Hamiltonian commutes with exchange (`1e-12`).
- Every kept state has definite exchange parity (`1e-8`), matching the primitive reference.
  Each excitation pair has one symmetric and one antisymmetric member.
- In each single-excitation state, the Bell-like coefficients on `|1,0⟩` and `|0,1⟩` have equal moduli, by symmetry to `1e-8`.
  Those moduli are 0.707105 in state 2 and 0.707104 in state 3: 1/√2, not 1/2.
- The antisymmetric states vanish on the exchange diagonal (`1e-9`).
- The energies match the projected reference to `1e-8` and the primitive reference to `1.8e-7` (the retained-subspace error).
- No diagnostics are emitted. Anchor cancellation is not triggered, which the catalogue says must not be relied on; T4 covers it deterministically.

### F6

Small configuration (default suite): `full_zero_pi.json`; zero-pi leaf `ChargeBasis(6)` × `PhaseGrid(-6π, 6π, 101; order=6)`, Dense, keep 8; raw `FockBasis(6)` zeta; root `LOBPCG(tol=1e-9)`, `compute = 4`.
The asymmetries `dE_J, dC_J, dC, dE_L` are each turned on alone, starting from the symmetric point, and then all together (a large step).
A small step follows (`n_g + 1e-3`), warm-started from the all-asymmetries point.
`n_g = 0.1` and `varphi_ext = 0.46π` throughout.

- The charging matrix is positive definite at every point.
- Leaf: true residuals against the independent primitive leaf matrix stay below `1e-9`. The leaf operator matrix equals that primitive matrix at the all-asymmetries point.
- Root: the projected Hamiltonian `W' H W` is assembled from the 7878-dimensional primitive matrix. It is Hermitian to `1e-10`, its spectrum matches to `1e-8`, and the residuals stay below `1e-8`.
  The root operator under test (`operator(sys)`, as a dense matrix) is Hermitian and equals that projection to `1e-9`; in the slow version, to `1e-8`.
  The projector onto the complete lowest 4 matches.
  The projected ports (`n_θ`, `φ`, `n_ζ`) match their primitive counterparts on the explicit states and are Hermitian.
- `:ritz` and `:jacobi` were run on the same root operator, cold and warm.
  The test requires `:ritz` to converge, and allows `:jacobi` either to converge or to fail with an explicit nonconvergence error; here all of them converged.
  The work in H columns below was measured separately, with the same configuration:

| point | `:ritz` cold | `:ritz` warm | `:jacobi` cold | `:jacobi` warm |
| --- | --- | --- | --- | --- |
| symmetric | 57 | 44 | 122 | 95 |
| dE_J | 57 | 44 | 122 | 95 |
| dC_J | 57 | 44 | 122 | 95 |
| dC | 78 | 60 | 124 | 109 |
| dE_L | 81 | 69 | 122 | 114 |
| all | 87 | 75 | 122 | 112 |

(Warm means warm from the symmetric root vectors; seed 1.)

Slow configuration: zero-pi leaf `ChargeBasis(10)` × `PhaseGrid(-8π, 8π, 201; order=8)` (4221 states), `LOBPCG()` (default `tol = 1e-8`, `maxiter = 1000`), `compute = 10`; raw `FockBasis(10)` zeta; root `LOBPCG()`, `compute = 6`.
For each point the leaf is solved alone, cold and warm, and the tree is materialized warm-started from its predecessor.
Leaf validation: residuals against the primitive matrix below `1e-7`, and the energies match a dense primitive reference at the symmetric and all-asymmetries points.
Tree validation: as in the small case, with the 42 210-dimensional primitive matrix.

| point | solve | outcome | H columns | time (s) |
| --- | --- | --- | --- | --- |
| symmetric | leaf `:ritz` cold | converged | 9240 | 9.8 |
| symmetric | leaf `:jacobi` cold | **explicit failure**: 0 of 10 pairs converged in 1000 iterations, max residual 2.1e-4 | 14014 | 13.0 |
| symmetric | tree cold (root columns) | converged | 91 | 8.5 |
| dE_J | leaf `:ritz` cold / warm | converged / converged | 9337 / 6506 | 8.7 / 5.9 |
| dC_J | leaf `:ritz` cold / warm | converged / converged | 9645 / 6356 | 8.0 / 4.9 |
| dC | leaf `:ritz` cold / warm | converged / converged | 9240 / 3346 | 7.2 / 2.4 |
| dE_L | leaf `:ritz` cold / warm | converged / converged | 9240 / 3346 | 7.3 / 2.6 |
| all (large step) | leaf `:ritz` cold / warm | converged / converged | 9445 / 5794 | 7.8 / 4.6 |
| all (large step) | leaf `:jacobi` cold | **explicit failure**: 0 of 10 converged, max residual 4.7e-4 | 14014 | 11.1 |
| all (large step) | leaf `:jacobi` warm | **explicit failure**: 6 of 10 converged, max residual 1.6e-7 | 12363 | 9.4 |
| all, small step | leaf `:ritz` cold / warm | converged / converged | 9102 / 6236 | 7.2 / 4.8 |
| tree warm (each asymmetric point) | root columns | converged | 55–85 | 2.6–5.5 |

Most `:ritz` leaf solves also emit the documented warning "k of 4 guard columns did not converge within 1000 iterations".
The 10 kept pairs converge, and they match the dense reference where it was computed, so no lower state was missed at those points.
On this stiff grid leaf the guards are the slowest columns, and they set the iteration count.
The fixed-Jacobi failures are explicit `ErrorException`s; the test asserts the error, as the catalogue's "record explicit failures" asks.
They are not reported as a defect: the contract only asks for valid output or an informative failure.
(The `:jacobi` leaf failure also occurs on the small 1313-state leaf, where 3 of 8 pairs converge at `tol = 1e-9`; the small default-suite case therefore uses a Dense leaf and compares the preconditioners at the root.)
Times are single runs on this machine, with the BLAS thread count left at its default.

### F8

Small extension (default suite): `two_transmons.json`, `g = 4`, `n_g1 = 0.25`, `n_g2 = 0.4` (both linear pieces routed to the leaves, the constant at the root), `ChargeBasis(8)` leaves keeping `k` states, `m = 4`.
Reference: the 289-dimensional primitive matrix, and the projection `W' H W` assembled without ports or apply.
For every `k`, Dense on the tree and `LOBPCG` on the tree (for `k ≥ 5`) both equal the projection to `1e-8`; the hierarchical and Dense solves never disagree with each other.
The error against the primitive reference is entirely retained-subspace error:

| k | 3 | 5 | 8 | 12 | 17 (complete) |
| --- | --- | --- | --- | --- | --- |
| max(E_proj − E_prim) | 1.35 | 0.141 | 2.5e-4 | 3.0e-9 | 3.6e-13 |

The projected energies are variational (never below the primitive ones) and nonincreasing in `k` (the retained spaces are nested).

Slow: `full_zero_pi.json` with strong couplings `dC = 0.8`, `dE_L = 0.5` (the charging matrix stays positive definite); zero-pi leaf as in the small F6 (Dense, keep `kz`) and a zeta leaf `FockBasis(8)` keeping `kζ`; `m = 6`.
Primitive reference: the lowest 6 eigenpairs of the 10 504-dimensional primitive matrix, by shift-invert Lanczos on its sparse LU factorization.
At every cutoff, Dense and LOBPCG on the tree equal the projection to `1e-8`; the projections are variational and monotone.

| (kz, kζ) | (4, 2) | (8, 3) | (16, 4) | (32, 6) | (64, 8) |
| --- | --- | --- | --- | --- | --- |
| max(E_proj − E_prim) | 1.31 | 0.099 | 0.012 | 3.3e-4 | 1.9e-4 |

### F9

Small (default suite): fluxonium (`E_J = 8.9`, `E_C = 2.5`, `E_L = 0.5`, half flux), Dense, 4 kept states, compared with the scqubits fluxonium reference.
At every size the test checks two things.
The edge warning appears exactly when the recomputed edge support exceeds `_EDGE_TOL = 1e-6`.
The true residual against the independent primitive matrix stays below `1e-9`, whatever the discretization error.

| grid | max edge support | max energy error | edge warning |
| --- | --- | --- | --- |
| ±2π, 81 points (Δ = π/20) | 0.217 | 6.82 | yes |
| ±4π, 161 points | 7.3e-3 | 3.8e-3 | yes |
| ±6π, 241 points | 5.0e-7 | 2.8e-8 | no |
| ±10π, 401 points | 6.5e-16 | 2.8e-8 | no |
| ±10π, 201 points | 2.0e-16 | 6.7e-6 | no |
| ±10π, 801 points | 6.6e-16 | 1.1e-10 | no |

Slow: zero-pi (`zero_pi.json`, `ChargeBasis(10)`, Lanczos, 8 states) against the converged wide scqubits reference.

| grid | max edge support | max energy error |
| --- | --- | --- |
| ±6π, 151 points (Δ = 2π/25) | 6.3e-3 | 2.4e-3 |
| ±8π, 201 points | 1.4e-4 | 6.7e-7 |
| ±10π, 251 points | 7.1e-7 | 1.1e-7 |
| ±12π, 301 points | 9.8e-10 | 1.1e-7 |
| ±10π, 201 points | 1.0e-6 (warns, just above the tolerance) | 6.3e-7 |
| ±10π, 401 points | 3.5e-7 | 2.5e-9 |
| ±10π, 801 points | 1.3e-7 | 5.3e-11 |

Widening at fixed spacing changes the energies by 2.4e-3, then 7.7e-7, then 1.6e-11.
Once the edges clear, the remaining error is the spacing error, which refinement at the fixed interval removes.
A previous solution from a different interval with the same point count is rejected (T2, referenced, not repeated).

### F10 (extension)

Added to the existing testset: the coupling ports the zero-pi nodes expose to the root (`n_θ` for `dC`, `φ` for `dE_L`).
In both child orders, they equal the closed-form primitive operators on the nodes' explicit (θ, φ) states.
For both trees, every root state's `:positive_sector` anchor is resolved to a primitive (θ, φ, ζ) point through the raw basis indices and the node children's peak components.
The amplitude there is real and positive, and equals the stored peak amplitude.
On the raw grid axis the anchor lies in the positive sector.

### P1

A `FockBasis` fluxonium leaf, `(N, kept) = (30, 4)` and `(6, 3)`.
The reference operators come from the ladder matrix `a` on the window, not from `localop`.
`n²` and `φ²` are the closed-form bands of the infinite-space squares; they differ from the squares of the truncated generators on the last level, by `N/(4φ_zpf²)` for `n²`.
The `n2port`/`phi2port` ports equal `V' O² V` to `1e-12` and differ from `(V' O V)²` by more than `1e-2` in operator norm.
For `N = 6` they also differ from `V' (O_trunc)² V` by more than `1e-3`, a second truncation the reference must not make.
Through a branch (a transmon coupled to an oscillator leaf with a mismatched `φ_zpf`, the leaf keeping 3 of 10 Fock states), the root's `n²`/`φ²` ports equal `Ψ' O² Ψ` on the explicit primitive states and differ from the square of the root's `n`/`φ` port.
The device repeat is deferred until port projection moves to the GPU.

### P2

A fluxonium grid leaf (`PhaseGrid(-10π, 10π, 201)`, `varphi_ext = 0.8π`, Dense `compute = 12`, keep 5).
A custom `SelectionRule` inspects `(:varphi_1, n)`, so that port is formed on all computed vectors before slicing, and keeps a given permutation.
The requested ports `φ`, `n²`, `e^{±iφ/2}` are formed after slicing.
Three transformations `V → V U` were tested: a permutation (the selection), complex phases (`:follow` from a previous solution whose vectors were multiplied by random phases), and both.
For each, the vectors equal `V₀ U`, and every port of both kinds equals `U' P₀ U` to `1e-10`.
The ports also equal the declared operators on the kept vectors, the Hermitian ones stay Hermitian, and `e^{iφ/2}` stays the adjoint of `e^{-iφ/2}`.
At a branch (two transmons), the root selection port, the root's required ports, and the `e^{±iφ}` ancestor ports the leaf forms after its own slicing transform the same way under `U`.
They match the primitive operators on the explicit states, and the conjugate pair stays adjoint.
The device parity is deferred (B1 covers device port timing).

### T4

`branch_peak_phase` is tested on explicit tensors.
The child has three points and states `s₁ = (0.8, √0.18, √0.18)`, `s₂ = (0.6, −0.4√2, −0.4√2)` and `s₃ = (0, 1, −1)/√2`; `s₁` and `s₂` both peak at point 1.
A basis child of dimension 2 completes the product.

| case | result |
| --- | --- |
| exact cancellation, `C = (0.6, −0.8, 0) ⊗ e₁` | the dominant component is `s₂`, the anchor amplitude is below 1e-15, and the denominator is 0.48, so the ratio is below `_HYBRID_RATIO`; output finite, and no unique phase is asserted |
| near cancellation (1e-7) | `0 < |α| < 1e-6`, equal to the explicit amplitude at the anchor; the fixed phase makes it real and positive |
| tied dominant coefficients | either tie-break gives the same anchor, because `s₁` and `s₂` share their peak point |
| empty mask, zero-weight mask | both fall back to the unmasked argmax |
| arbitrary global phase | `α` rotates with the phase and the phase-fixed state does not (for `α ≠ 0`); the leaf rule is phase invariant |

Through `materialize`, a child node over a given three-point system with `H = S diag(0, 1, 2) S'` is coupled to a given two-state system, so that the parent ground state is exactly `ψ ⊗ |b₁⟩` with `ψ = (ε, 1, 1)/‖·‖`.

- `ε = 0`: the hybridization warning is emitted, and the anchor amplitude is below 1e-14 on the explicit full-space state.
- `ε = 1e-3`: the hybridization warning is emitted, and the anchor amplitude is real, positive and equal to `ε/‖·‖`.
- A grandparent over that state emits the vanishing-denominator warning.
- A branch whose kept state lies only at charge −1 emits the branch empty-mask warning.
- A grid leaf entirely at negative phase emits the leaf empty-mask warnings.

All outputs are finite. Warnings are captured with `@test_logs`.

## Suite times

Default suite (`Pkg.test()`, wall clock including precompilation checks) on this machine: 96.3 s before Task 10 and 108.2 s after it (+11.9 s, within the ~15 s allowance).
The added testsets' in-suite times include compilation that later testsets no longer pay; see the Task 10 report for the per-testset figures.
In the full gated run (`KRONFREE_SLOW_TESTS=1 Pkg.test()`, 7 min 35 s in total), the slow catalogue testsets take F3 20.5 s, F6 2 min 48 s, F8 9.2 s and F9 19.8 s.
The gated file prints its measurements (`CATALOGUE_SLOW`) at the end of the run; the F3, F6, F8 and F9 numbers above come from that record. The work counts reproduce exactly between runs; the F6 times in the table are from an earlier run of the same testsets and vary by about 10 %.
