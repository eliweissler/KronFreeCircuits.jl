# Plan 1b: Matched-Grid Agreement and Full-Tree Phase Fixing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the in-suite scqubits agreement for the grid systems from 1e-4-level to 1e-10-level by comparing on identical small discretizations, and make branch phase fixing physical: each branch state's phase is set by its exact full-space amplitude at the joint peak of its dominant product component, computed from the children's stored peak amplitudes without expansion.

**Architecture:** Task 1 changes only tests and the reference generator: scqubits' 7-point stencil on its 1001-point grid is exactly our `PhaseGrid(…, 1001; order=6)`, so the two codes build the same matrices. Task 2 changes the branch behavior of `:positive_sector`: every materialized node stores, per kept state, its peak point and a k×k matrix of exact full-space amplitudes at the other kept states' peaks; a branch uses its children's matrices to evaluate each state at the joint peak of its dominant product component and makes that amplitude real positive. Task 3 (tabled) is the exact sector maximum by branch-and-bound.

**Tech Stack:** as plan 1. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`, Sections 9.1 (phase conventions) and 16 (tests). Eli's request of 2026-10-16: push numerical agreement higher through shared grids; keep the per-node positive-sector rule but also allow full-dimensional positive-quadrant phase fixing by going all the way down the tree.

## Global Constraints

- As plan 1: Julia 1.13, `ComplexF64`, column-major axis 1 fastest, structure-dispatched apply, no new dependencies, commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Physics conventions and fixture files are unchanged. Only the reference generator may change, and only additively (new entries, recorded metadata); existing entries keep their values.
- scqubits' phase-grid second derivative uses `scqubits.settings.STENCIL = 7` (record it in the reference file), i.e. a 7-point central stencil, which is our `order=6`; its first derivative (the `n_φ` operator and the `n_θ n_φ` cross term) uses the matching 7-point first-derivative stencil. Edge handling is truncation, as ours.
- The default phase convention stays `:positive_sector`; its branch behavior becomes the joint-peak rule of Task 2 (no new convention symbol).

## Review Focus

1. A matched-discretization test must fail if either code's stencil, range, or point count differs: pin `order=6` and `N=201` explicitly and assert agreement at `1e-9` or tighter, never looser. Pinned in Task 1.
2. Leaf behavior of `:positive_sector` is unchanged by Task 2; leaf `peak_amplitudes` equal the vectors at the peaks. Pinned in Task 2.
3. The branch rule must be invariant to a child state's sign flip (coefficients and stored amplitudes flip together). Pinned in Task 2's unit test.
4. The amplitude used for the phase must equal the brute-force full-space value at the joint peak, through two levels of tree. Pinned in Task 2 by Kronecker reconstruction oracles.
5. Edge-support warnings on the shared ±6π grid are expected in Task A's tests and must be captured with `@test_logs`, never left as noise.

---

### Task 1: Matched-discretization scqubits comparisons on small grids; convergence tests gated

**Files:**
- Modify: `test/fixtures/generate_scqubits.py` (add entries, record `STENCIL`, numpy/scipy versions, interpreter), regenerate `test/fixtures/scqubits_reference.json` (existing entries byte-identical)
- Modify: `test/test_materialize.jl` (replace the 301-point zero-pi and full zero-pi comparisons with small matched ones), `test/test_zero_pi_reference.jl` (gated: the moved 301-point comparisons, the existing two-zero-pi test, a matched two-zero-pi, and a wide-grid converged zero-pi), `test/helpers.jl` if a helper is needed

**Interfaces:**
- Consumes: `reference(name)`, `zero_pi_params()`, `full_zero_pi_params()`, `two_zero_pi_params(; g)`, `PhaseGrid(a, b, N; order)`, `Dense()`, `Lanczos(; …)`, `materialize`, `energies`.
- Produces: reference entries, all with the same physics parameters as their existing counterparts and the ±6π range unless stated:
  `zero_pi_small` (`pt_count=201`, `ncut=10`, lowest 8), `full_zero_pi_small` (`pt_count=201`, `ncut=10`, `zeropi_cutoff=30`, `zeta_cutoff=20`, lowest 10), `two_zero_pi_small` (`pt_count=201`, `ncut=10`, `truncated_dim=10`, `g=0.05`, lowest 20), `zero_pi_wide` (range ±10π, `pt_count=1601`, `ncut=20`, lowest 8). Metadata records `scqubits.settings.STENCIL` (7), numpy and scipy versions, and `sys.executable`.

Eli's direction (2026-10-16): the default suite checks agreement with a trusted tool on identical discretizations and small bases; physical correctness and convergence are a few gated slow tests to be grown later.

- [ ] **Step 1: Write the failing tests.** In `test/test_materialize.jl`, replacing the existing "zero-pi leaf vs scqubits (Lanczos on a 9k-dimensional leaf)" and "full zero-pi: all three mode types …" testsets (which move to the gated file):

```julia
@testset "zero-pi leaf vs scqubits, identical discretization (7-point stencil, 201 points)" begin
    ref = reference("zero_pi_small"); θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    leaf = Node([:theta => ChargeBasis(10), :phi => PhaseGrid(-6π, 6π, 201; order=6)]; solver=Dense(), keep=lowest(8), ports=[(:theta, nport())])
    sys = @test_logs (:warn, r"edge") match_mode=:any materialize(leaf, spec_zp, θ)
    @test maximum(abs.(energies(sys) .- ref.energies)) < 1e-9
    @test port(sys, :theta, nport()) ≈ port(sys, :theta, nport())'
    lan = Node([:theta => ChargeBasis(10), :phi => PhaseGrid(-6π, 6π, 201; order=6)]; solver=Lanczos(tol=1e-12), compute=8, keep=lowest(8))
    @test energies(@test_logs (:warn, r"edge") match_mode=:any materialize(lan, spec_zp, θ)) ≈ energies(sys) atol=1e-9   # Lanczos = Dense
end
@testset "full zero-pi vs scqubits, identical discretization" begin
    ref = reference("full_zero_pi_small"); θ = full_zero_pi_params()
    zp = Node([:theta => ChargeBasis(10), :phi => PhaseGrid(-6π, 6π, 201; order=6)]; solver=Dense(), keep=lowest(30))
    zeta = Node([:zeta => FockBasis(20; phi_zpf=(2θ.E_Cz/θ.E_L)^(1/4))]; keep=lowest(20))
    full = Node([zp, zeta]; solver=Dense(), keep=lowest(10))
    sys = @test_logs (:warn, r"edge") match_mode=:any materialize(full, load_spec(fixture("full_zero_pi.json")), θ)
    @test maximum(abs.(energies(sys) .- ref.energies)) < 1e-9
    @test modes(full) == [:theta, :phi, :zeta]
end
```
In `test/test_zero_pi_reference.jl` (gated by `KRONFREE_SLOW_TESTS=1`): keep the existing two-zero-pi test; add `two_zero_pi_small` with `ChargeBasis(10)`, `PhaseGrid(-6π, 6π, 201; order=6)` leaves (`Dense`, `keep=lowest(10)`), top `Dense`, `keep=lowest(20)`, elementwise `< 1e-9`; move the two removed 301-point testsets here under a `@testset "convergence (different stencils, shared grid)"` with their existing tolerances; add the wide-grid converged check: `zero_pi_wide` against `Node([:theta => ChargeBasis(20), :phi => PhaseGrid(-10π, 10π, 1601; order=8)]; solver=Lanczos(tol=1e-12), compute=8, keep=lowest(8))`, no edge warning expected, elementwise `< 1e-6`.

- [ ] **Step 2: Run to verify they fail** (missing reference entries).
- [ ] **Step 3: Extend the generator**, run it with `/opt/anaconda3/envs/circuit_quantization/bin/python test/fixtures/generate_scqubits.py`, confirm existing entries byte-identical.
- [ ] **Step 4: Run both suites → PASS.** If a matched comparison is worse than 1e-9, find the discretization difference (first-derivative stencil for `n_φ`, edge rows, `range` vs `linspace` endpoints, the `n_θ n_φ` cross term in the full zero-pi) rather than loosening; report observed maxima in the commit body. Report the default suite's wall time before and after.
- [ ] **Step 5: Commit** `"Compare grid systems to scqubits on identical small discretizations; gate convergence tests"`.

---

### Task 2: Branch phase fixing at the joint peak of the dominant product component

**Decision (Eli, 2026-10-16).** Keep the per-node positive-sector rule at leaves. At a branch, fix each kept state's phase by its exact full-space amplitude at the joint peak of its dominant product-state component, computed through the children's stored peak amplitudes. This is child-convention-independent (a leaf state flipping sign between parameter points flips the coefficients and the stored amplitudes together, leaving the physical amplitude unchanged), recursive, and costs k² per node with no grid-sized arrays. The exact full-space sector maximum via a Cauchy–Schwarz branch-and-bound is tabled (Task 3).

**Files:**
- Modify: `src/materialize.jl` (`MaterializedNode` gains `peak_component::Vector{Int}` and `peak_amplitudes::Matrix{ComplexF64}`; the branch phase step), `src/phase.jl` (a pure function for the branch rule), `src/tree.jl` (docstring), `test/test_materialize.jl`, `test/test_phase.jl` (new, unit tests of the pure function), `README.md` (one sentence)

**Interfaces:**
- Consumes: `MaterializedNode` (`children`, `vectors`, `dims`, `kept`), `positive_mask`, `fix_phases!`, the existing selection-port rotation code.
- Produces:
  - `MaterializedNode.peak_component::Vector{Int}`: for kept state `j`, the linear index into the node's own product basis (`dims`) of its peak point. For a leaf this is the positive-sector argmax already used by `:positive_sector`. For a branch it is the linear index of the dominant coefficient `(a*, b*, …) = argmax |c|` of the kept vector; the physical point it denotes is resolved recursively through the children's `peak_component`.
  - `MaterializedNode.peak_amplitudes::Matrix{ComplexF64}` of size `k × k`: `A[p, j]` is the exact full-space amplitude of kept state `j` at the physical peak point of kept state `p`. Leaf: `A = V[peak_component, :]`. Branch: `A[p, j] = Σ_{ab…} c^{(j)}_{ab…} A₁[a_p, a] A₂[b_p, b] …`, i.e. the kept vector `j` reshaped to `dims`, contracted on each child axis with the child's peak-amplitude row for `p`'s component on that axis. Implement the contraction with `apply_axis!` and a `DenseOp` of the single row (or a direct `dot`), not by expansion.
  - `branch_peak_phase(C::AbstractArray{ComplexF64}, rows::Vector{<:AbstractVector})::Tuple{Int,ComplexF64}` in `src/phase.jl`: given a kept vector reshaped to the node's dims and, per child axis, that child's peak-amplitude ROWS indexed by the child's kept index (i.e. the child's full `peak_amplitudes` matrix is passed and the function selects row `a*` for the dominant component), returns `(linear index of argmax |C|, exact amplitude at the joint peak)`. Pure, unit-testable.
  - Branch phase step in `_materialize`: for each kept `j`, `(comp, α) = branch_peak_phase(...)`; multiply column `j` (and rotate already-formed ports, as the existing code does) by `exp(-im*angle(α))`; store `comp` and compute the node's `peak_amplitudes`. Diagnostic: when `abs(α) < 0.1` (the amplitude at the chosen point is small relative to the unit-norm state, i.e. heavy hybridization), `@warn maxlog=10` naming the node's modes and state index.
  - `:positive_sector` keeps its name; the branch behavior changes from "largest coefficient real positive" to the above. `:follow` is unchanged.

- [ ] **Step 1: Write the failing tests**

`test/test_phase.jl` (unit, synthetic):
```julia
@testset "branch_peak_phase" begin
    # two children with 3 and 2 kept states; synthetic peak-amplitude matrices and a coefficient tensor
    A1 = ComplexF64[0.9 0.1 0.2; 0.3 0.8 0.1; 0.2 0.2 0.7]; A2 = ComplexF64[0.95 0.2; 0.1 0.9]
    C = zeros(ComplexF64, 3, 2); C[2, 1] = 0.8im; C[1, 2] = 0.5; C[3, 2] = 0.2
    comp, α = branch_peak_phase(C, [A1, A2])
    @test comp == LinearIndices((3, 2))[2, 1]                       # dominant component
    # exact amplitude at the joint peak (row 2 of A1, row 1 of A2): Σ_ab C[a,b] A1[2,a] A2[1,b]
    @test α ≈ sum(C[a, b] * A1[2, a] * A2[1, b] for a in 1:3, b in 1:2)
    # child sign flip invariance: flipping child-1 state 2 flips C[2,:] and A1[:,2] together; α unchanged
    C2 = copy(C); C2[2, :] .*= -1; A1f = copy(A1); A1f[:, 2] .*= -1
    @test branch_peak_phase(C2, [A1f, A2])[2] ≈ α
    # three children recursion shape
    @test branch_peak_phase(randn(ComplexF64, 2, 3, 2), [randn(ComplexF64, 2, 2), randn(ComplexF64, 3, 3), randn(ComplexF64, 2, 2)]) isa Tuple{Int,ComplexF64}
end
```
`test/test_materialize.jl` (end to end, two transmons and three-level tree):
```julia
@testset "branch positive-sector phase at the dominant joint peak" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(g=0.3)
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    q1n, q2n = mk(1), mk(2); top = Node([q1n, q2n]; keep=lowest(20), ports=[(:varphi_1, nport())])
    sys = materialize(top, spec, θ); q1, q2, t = materialized(sys, q1n), materialized(sys, q2n), materialized(sys, top)
    V = t.vectors; mask1 = positive_mask(ChargeBasis(30))
    for j in 1:20
        ψ = kron(q2.vectors, q1.vectors) * V[:, j]                                # brute-force full-space oracle, axis 1 fastest
        a, b = Tuple(CartesianIndices((8, 8))[t.peak_component[j]])
        p = LinearIndices((61, 61))[q1.peak_component[a], q2.peak_component[b]]
        @test abs(imag(ψ[p])) < 1e-12 && real(ψ[p]) > 0                            # real positive at the joint peak
        @test ψ[p] ≈ t.peak_amplitudes[j, j]
        @test mask1[q1.peak_component[a]] && mask1[q2.peak_component[b]]           # the joint peak lies in the positive sector
        @test (a, b) == Tuple(argmax(abs.(reshape(V[:, j], 8, 8))))               # dominant component
    end
    # leaf peak amplitudes equal the vectors at the peaks
    @test q1.peak_amplitudes == q1.vectors[q1.peak_component, :]
    # ports and energies unchanged in modulus relative to a run with the old rule are not testable directly; check Hermiticity and spectrum
    @test port(sys, :varphi_1, nport()) ≈ port(sys, :varphi_1, nport())'
    # three-level tree: peak amplitudes of the outer node agree with a brute-force expansion through two levels
    spec3 = load_spec(fixture("three_transmons.json")); θ3 = three_transmons_params(g=0.3)
    inner = Node([mk(1), mk(2)]; keep=lowest(12)); third = Node([:varphi_3 => ChargeBasis(10)]; keep=lowest(4))
    outer = Node([inner, third]; keep=lowest(10)); sys3 = materialize(outer, spec3, θ3)
    o, inn, th = materialized(sys3, outer), materialized(sys3, inner), materialized(sys3, third)
    l1, l2 = materialized(sys3, inner.children[1]), materialized(sys3, inner.children[2])
    Vinner_full = kron(l2.vectors, l1.vectors) * inn.vectors                       # inner kept states in the leaf basis
    for j in 1:10
        ψ = kron(th.vectors, Vinner_full) * o.vectors[:, j]
        ai, c = Tuple(CartesianIndices((12, 4))[o.peak_component[j]])
        a, b = Tuple(CartesianIndices((8, 8))[inn.peak_component[ai]])
        p = LinearIndices((61, 61, 21))[l1.peak_component[a], l2.peak_component[b], th.peak_component[c]]
        @test abs(imag(ψ[p])) < 1e-12 && real(ψ[p]) > 0
        @test ψ[p] ≈ o.peak_amplitudes[j, j]
    end
end
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** `branch_peak_phase`, the new fields (leaf and branch), the branch phase step with the diagnostic, docstrings, README sentence.
- [ ] **Step 4: Run both suites → PASS.**
- [ ] **Step 5: Commit** `"Fix branch phases at the dominant joint peak via children's stored peak amplitudes"`.

---

### Task 3 (tabled): exact full-space sector maximum by Cauchy–Schwarz branch-and-bound

Not scheduled. Recorded so the design is not lost. For a node whose children are leaves, with `W = V₁ C V₂ᵀ` the (never formed) full-space matrix of one kept state: compute `r_i = ‖(V₁C)[i,:]‖` for sector rows `i` and `s_j = ‖V₂[j,:]‖` for sector columns `j` (cost `d·k` each); enumerate sector pairs `(i, j)` lazily in decreasing `r_i s_j` with a heap over `i` against `s` sorted descending; evaluate `W_ij` exactly (`k₁k₂`) for each; stop with proof when the next bound is below the best exact `|W_ij|` found. Restricting to the all-positive sector cuts the candidate set by `2ⁿ` for `n` grid axes. Cap the number of evaluations (default 10⁴) with a `@warn` and fall back to Task 2's joint peak. Deeper trees need recursive bounds or fall back to Task 2. Eli's expectation: in practice the peaks line up, so the search should terminate after a handful of evaluations; the cap guards the delocalized case.

---

## Self-review notes
- Task 1 touches only tests and the generator; Task 2 touches `materialize.jl`, `phase.jl`, `tree.jl`. They are independent and may be dispatched in order 1 then 2.
- Names: `:positive_sector_full`, `full_space_vector`, `full_space_mask`, `expanded_size`, `full_phase_max_entries`.
