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

1. A matched-discretization test must fail if either code's stencil, range, or point count differs: pin `order=6` and `N=1001` explicitly and assert agreement at `1e-9` or tighter, never looser. Pinned in Task 1.
2. `:positive_sector_full` on a leaf must give exactly the same result as `:positive_sector`. Pinned in Task 2.
3. `:positive_sector_full` must estimate the expanded size and `@warn` above a threshold (default 2e8 complex entries, about 3.2 GB) before allocating. Pinned in Task 2.
4. The expansion must use the children's kept vectors AFTER their own phase fixing, so that the full-space vector is the physical state; the convention changes only the global phase of the branch vector. Pinned in Task 2 by comparing against a direct Kronecker reconstruction.
5. Edge-support warnings on the shared ±6π grid are expected in Task A's tests and must be captured with `@test_logs`, never left as noise.

---

### Task 1: Matched-discretization and converged scqubits comparisons

**Files:**
- Modify: `test/fixtures/generate_scqubits.py` (add entries, record `STENCIL`, numpy/scipy versions, interpreter), regenerate `test/fixtures/scqubits_reference.json`
- Modify: `test/test_materialize.jl` (zero-pi leaf and full zero-pi), `test/test_zero_pi_reference.jl` (two zero-pi), `test/helpers.jl` if a helper is needed

**Interfaces:**
- Consumes: `reference(name)`, `zero_pi_params()`, `full_zero_pi_params()`, `two_zero_pi_params(; g)`, `PhaseGrid(a, b, N; order)`, `Lanczos(; …)`, `materialize`, `energies`.
- Produces: reference entries `zero_pi_matched`, `full_zero_pi_matched`, `two_zero_pi_matched` (same physics parameters as the existing entries, same ±6π range, `pt_count=1001`, `ncut=15`; the stencil recorded), and `zero_pi_wide` (range ±10π, `pt_count=1601`, `ncut=20`, lowest 8) for the converged check. The existing entries are unchanged.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "zero-pi leaf, matched discretization (order 6 on scqubits' 1001 points)" begin
    ref = reference("zero_pi_matched"); θ = zero_pi_params()
    leaf = Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-6π, 6π, 1001; order=6)];
                solver=Lanczos(tol=1e-12), compute=8, keep=lowest(8))
    sys = @test_logs (:warn, r"edge") match_mode=:any materialize(leaf, load_spec(fixture("zero_pi.json")), θ)
    @test maximum(abs.(energies(sys) .- ref.energies)) < 1e-9
end
@testset "zero-pi leaf, converged on a wide grid" begin
    ref = reference("zero_pi_wide"); θ = zero_pi_params()
    leaf = Node([:theta => ChargeBasis(20), :phi => PhaseGrid(-10π, 10π, 1601; order=8)]; solver=Lanczos(tol=1e-12), compute=8, keep=lowest(8))
    sys = materialize(leaf, load_spec(fixture("zero_pi.json")), θ)     # no edge warning expected
    @test maximum(abs.(energies(sys) .- ref.energies)) < 1e-6            # different stencils, both converged
end
@testset "full zero-pi, matched discretization" begin
    ref = reference("full_zero_pi_matched"); θ = full_zero_pi_params()
    zp = Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-6π, 6π, 1001; order=6)]; solver=Lanczos(tol=1e-12), compute=30, keep=lowest(30))
    zeta = Node([:zeta => FockBasis(20; phi_zpf=(2θ.E_Cz/θ.E_L)^(1/4))]; keep=lowest(20))
    sys = @test_logs (:warn, r"edge") match_mode=:any materialize(Node([zp, zeta]; keep=lowest(10)), load_spec(fixture("full_zero_pi.json")), θ)
    @test maximum(abs.(energies(sys) .- ref.energies)) < 1e-9
end
```
and in `test/test_zero_pi_reference.jl` the same pattern for `two_zero_pi_matched` with `PhaseGrid(-6π, 6π, 1001; order=6)` leaves, `Lanczos(tol=1e-12), compute=10, keep=lowest(10)`, `atol` on the elementwise maximum `< 1e-9`. Keep the existing `two_zero_pi` test as is.

- [ ] **Step 2: Run to verify they fail** (missing reference entries).
- [ ] **Step 3: Extend the generator**: add the four entries; record `scqubits.settings.STENCIL`, `numpy.__version__`, `scipy.__version__`, and `sys.executable` in the metadata; run with `/opt/anaconda3/envs/circuit_quantization/bin/python test/fixtures/generate_scqubits.py`; confirm existing entries are byte-identical (diff the old and new JSON for those keys).
- [ ] **Step 4: Run tests → PASS.** If the matched comparison is worse than 1e-9, find the discretization difference (first-derivative stencil for `n_φ`, edge rows, the `n_θ n_φ` cross term in the full zero-pi) rather than loosening; report the observed maxima in the commit body.
- [ ] **Step 5: Commit** `"Compare grid systems to scqubits on matched and converged discretizations"`.

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
