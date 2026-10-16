# Plan 1b: Matched-Grid Agreement and Full-Tree Phase Fixing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the in-suite scqubits agreement for the grid systems from 1e-4-level to 1e-10-level by comparing on identical discretizations, and add a phase convention that fixes a branch eigenvector's phase in the full leaf coordinate basis by expanding down the tree.

**Architecture:** Task 1 changes only tests and the reference generator: scqubits' 7-point stencil on its 1001-point grid is exactly our `PhaseGrid(…, 1001; order=6)`, so the two codes build the same matrices. Task 2 adds `:positive_sector_full` beside `:positive_sector`: the kept vectors of a node are expanded one child axis at a time with the children's kept-vector matrices (via `apply_axis!` with a `DenseOp`), recursively to the leaves, restricted to the positive leaf coordinates, and the largest component there is made real and positive.

**Tech Stack:** as plan 1. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`, Sections 9.1 (phase conventions) and 16 (tests). Eli's request of 2026-10-16: push numerical agreement higher through shared grids; keep the per-node positive-sector rule but also allow full-dimensional positive-quadrant phase fixing by going all the way down the tree.

## Global Constraints

- As plan 1: Julia 1.13, `ComplexF64`, column-major axis 1 fastest, structure-dispatched apply, no new dependencies, commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Physics conventions and fixture files are unchanged. Only the reference generator may change, and only additively (new entries, recorded metadata); existing entries keep their values.
- scqubits' phase-grid second derivative uses `scqubits.settings.STENCIL = 7` (record it in the reference file), i.e. a 7-point central stencil, which is our `order=6`; its first derivative (the `n_φ` operator and the `n_θ n_φ` cross term) uses the matching 7-point first-derivative stencil. Edge handling is truncation, as ours.
- The default phase convention stays `:positive_sector` (per-node). `:positive_sector_full` is opt-in per node.

## Review Focus

1. A matched-discretization test must fail if either code's stencil, range, or point count differs: pin `order=6` and `N=201` explicitly and assert agreement at `1e-9` or tighter, never looser. Pinned in Task 1.
2. `:positive_sector_full` on a leaf must give exactly the same result as `:positive_sector`. Pinned in Task 2.
3. `:positive_sector_full` must estimate the expanded size and `@warn` above a threshold (default 2e8 complex entries, about 3.2 GB) before allocating. Pinned in Task 2.
4. The expansion must use the children's kept vectors AFTER their own phase fixing, so that the full-space vector is the physical state; the convention changes only the global phase of the branch vector. Pinned in Task 2 by comparing against a direct Kronecker reconstruction.
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

### Task 2: `:positive_sector_full` phase convention

**Files:**
- Modify: `src/phase.jl`, `src/materialize.jl`, `src/tree.jl` (allow the new symbol; docstrings), `test/test_materialize.jl`, `test/test_selection.jl` or a new `test/test_phase.jl`

**Interfaces:**
- Consumes: `fix_phases!`, `MaterializedNode` (`children`, `vectors`, `dims`), `positive_mask(b)`, `apply_axis!`, `DenseOp`.
- Produces: `Node(...; phase=:positive_sector_full)` accepted; `full_space_vector(mn::MaterializedNode, j::Int)::Array{ComplexF64}` returning kept state `j` expanded to the leaf coordinate basis as a tensor with one axis per leaf mode (in `modes(node)` order); `full_space_mask(mn)::Vector{Bool}` the Kronecker product of all leaf `positive_mask`s; `expanded_size(mn)::Int`; the warning threshold keyword `full_phase_max_entries` on `materialize` (default `2_000_000_000 ÷ 10 = 200_000_000`).
- Expansion algorithm: start from `reshape(V[:, j], dims...)`; for each child axis `a` that is a `MaterializedNode` child `c`, apply `DenseOp(c.vectors)` along axis `a` (this maps the child's kept index to the child's product index, growing the axis from `length(c.kept)` to `prod(c.dims)`), then reshape that axis into `c.dims`; recurse into `c`'s node children. Use `apply_axis!` so the kron-free machinery does the work; a basis child needs no expansion.
- Phase rule: for each kept vector, expand, take the entry of largest modulus among `full_space_mask`, multiply the branch vector (and its already-formed ports, as the existing rotation code does) by `exp(-im*angle(entry))`. Leaves: identical to `:positive_sector` (the expansion is the identity).

- [ ] **Step 1: Write the failing tests**

```julia
@testset ":positive_sector_full" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(g=0.3)
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    top_full = Node([mk(1), mk(2)]; keep=lowest(20), phase=:positive_sector_full, ports=[(:varphi_1, nport())])
    top_node = Node([mk(1), mk(2)]; keep=lowest(20), ports=[(:varphi_1, nport())])
    sf = materialize(top_full, spec, θ); sn = materialize(top_node, spec, θ)
    @test energies(sf) ≈ energies(sn)
    Vf = eigenvectors(sf, top_full); Vn = eigenvectors(sn, top_node)
    @test all(abs.(abs.(Vf) .- abs.(Vn)) .< 1e-12)                          # same states up to phase
    q1, q2 = materialized(sf, mk1), materialized(sf, mk2)                      # use the actual child Node objects
    for j in 1:20
        ψ = kron(q2.vectors, q1.vectors) * Vf[:, j]                             # explicit full-space vector, axis 1 fastest
        mask = kron(positive_mask(ChargeBasis(30)), positive_mask(ChargeBasis(30)))
        i = argmax(abs.(ψ) .* mask)
        @test imag(ψ[i]) ≈ 0 atol=1e-12 && real(ψ[i]) > 0
        @test vec(full_space_vector(q_top, j)) ≈ ψ
    end
    @test port(sf, :varphi_1, nport()) ≈ port(sn, :varphi_1, nport()) atol=1e-10 broken=false  # ports differ only by the diagonal phase rotation; compare |·|
    @test abs.(port(sf, :varphi_1, nport())) ≈ abs.(port(sn, :varphi_1, nport()))
    # leaf: identical to :positive_sector
    leaf_full = Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(5), phase=:positive_sector_full)
    leaf_node = Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(5))
    @test eigenvectors(materialize(leaf_full, load_spec(fixture("transmon.json")), (E_J=30.0, E_C=1.2, n_g=0.25)), leaf_full) ≈
          eigenvectors(materialize(leaf_node, load_spec(fixture("transmon.json")), (E_J=30.0, E_C=1.2, n_g=0.25)), leaf_node)
    # three-level tree: expansion recurses
    ... (three_transmons.json: inner = two coupled, outer = inner + third; compare full_space_vector against kron(V3, kron(V2, V1)) * Vinner-expanded)
    # size warning
    @test_logs (:warn, r"expanded") materialize(top_full, spec, θ; full_phase_max_entries=10)
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** `full_space_vector`, `full_space_mask`, `expanded_size`, the new convention in `fix_phases!`'s caller in `materialize.jl` (the convention needs the materialized children, so it is handled in `materialize.jl`, not in `phase.jl`'s column-local function), accept the symbol in `Node`, update the `Node` docstring and the README's phase sentence.
- [ ] **Step 4: Run tests → PASS**, including the full suite.
- [ ] **Step 5: Commit** `"Add :positive_sector_full phase fixing by expanding kept vectors down the tree"`.

---

## Self-review notes
- Task 1 touches only tests and the generator; Task 2 touches `materialize.jl`, `phase.jl`, `tree.jl`. They are independent and may be dispatched in order 1 then 2.
- Names: `:positive_sector_full`, `full_space_vector`, `full_space_mask`, `expanded_size`, `full_phase_max_entries`.
