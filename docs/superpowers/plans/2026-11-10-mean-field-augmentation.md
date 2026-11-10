# Mean-Field Basis Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each child of a tree node can be solved in the mean field of its siblings, the product energy scores a coordinate frame, and the mean-field eigenstates of several self-consistent solutions become the child's basis for the correlated solve, reproducing Figure 8(b) of the circuit-quantization paper.

**Architecture:** Mean field is built from separable pieces in the core package: expectation tables, mean-field terms, a child solved with those terms, a self-consistent loop, a union basis, and an augmented child that the parent consumes through `materialize(...; augmented=...)`.
The parent's operator is never changed; only the basis a child presents changes.
The KITE figure and the frame optimization are scripts in `bench/` written against the pieces, and the frame optimization reaches circuit-quantization through the PythonCall extension.

**Tech Stack:** Julia 1.13.2; no new core dependency; `Optim.jl` in the bench environment only; circuit-quantization at `c8c8714` or later through the PythonCall extension for fixtures generation and the frame optimization.

**Spec:** `docs/superpowers/specs/2026-11-10-mean-field-augmentation-design.md`

## Global Constraints

- The parent's operator is never changed, and mean-field terms are never double counted in the parent: they never enter `System.terms`, and an augmented child is the child's bare operator projected on its union basis.
- The pieces of the design's §4.1 to §4.4 are separately callable; the wrapper adds no behaviour of its own.
- No basis construction silently returns fewer states than requested or silently drops a source: every drop is counted in a report, and fewer than `m` independent vectors is an error unless the caller allows it.
- Mean-field children and augmented children run in complex arithmetic in this version; an ancestor of an augmented child is complex.
- Mean field acts among the children of one node, one level deep: when a child is a branch, its own children stay bare and are solved once.
- An augmented child keeps its union basis vectors on the child's own product space and its bare terms, not only projected matrices.
- `using KronFreeCircuits` without PythonCall must load and pass the default test tier; no code under `src/` imports or calls PythonCall; the optimizer is not a core dependency.
- Defaults of the self-consistent loop: relative energy tolerance `1e-11`, at most 300 iterations, mixing 0.5 after the energy has risen twice, Gram–Schmidt applied twice, drop threshold `1e-6`.
- `mean_field_ref_dont_commit/` is the owner's untracked reference and is never committed, copied into the repository, or edited; numbers taken from it are hard-coded with a comment naming their source.
- Per task, run only the fast (default) test tier and the focused test files named in the task; the slow tier runs once, in the last task.
- Work on the branch `mean-field` off `main`; commit per task; do not push.
- Markdown and docstrings follow the repository's style: one sentence per line.

## What the reference is (settled 2026-11-10)

The design's §6.1 says the reference for Figure 8(b) was incomplete.
The owner has since supplied it, in `mean_field_ref_dont_commit/kite_reference/`.
Its script `kite_levels.py` produced the figure's twelve points, and it differs from what §6.1 assumed in two ways that this plan follows:

- Grids: `default_bases(charge_cutoff=16, max_points=120, extent_scale=1.5)`, which gives 33 charge states and 120 grid points on each extended mode in every frame (not the notebook's 117 and 116 points).
- Starts: 33 (compact mode at `0` or `π`; each extended mode at `0`, `π/2`, `π`, `-π/2`; plus the delocalized start), not the notebook's 27.

The figure's error is `max(levels[1:4] - REF)` with `REF = [-9.523995, -9.510525, -8.453790, -8.337862]` GHz.
The twelve values, from the owner's result files:

| frame | m=8 | m=12 | m=16 | m=20 |
|---|---|---|---|---|
| published | 6.918e-1 | 2.213e-1 | 3.770e-2 | 6.776e-4 |
| normal modes | 6.218e-1 | 1.989e-1 | 3.600e-2 | 8.012e-4 |
| optimum | 3.668e-2 | 2.268e-3 | 2.609e-4 | 2.072e-5 |

A rerun of the reference on ewdesktop reproduces the stored levels of the normal-mode and optimum frames to about `1e-10` GHz, and those of the published frame only to a few percent of the error.
In the published frame the two wells' eigenvectors on one mode coincide, so Gram–Schmidt remainders of `1e-7` to `2e-5` sit on both sides of the `1e-6` drop threshold and the admitted vectors depend on round-off.
The published-frame curve is therefore compared with a tolerance, not to digits.

The Harmonium benchmark of the design's §6.2 is not in this plan.
It is exploratory, depends on what the KITE work measures, and needs choices about sizes and the GPU; it gets its own plan after this one lands.
Multi-mode children, caps, and supplied starts, which that benchmark exercises, are built and unit-tested here.

## Review Focus

1. **Two outside factors of one term in the same sibling.** Their expectation is the sibling's joint expectation, not the product of two single expectations; a three-mode case with a two-mode sibling must show the difference. (Task 2)
2. **A parent whose children are not all `Node`s.** A `mode => basis` or `GivenSystem` child under the mean-field node gets an `ArgumentError` that names the child and says to wrap the mode in a `Node`, never a silently skipped child. (Task 2)
3. **A run that does not converge.** A solution that hits the iteration limit is returned with `converged == false`, the wrapper reports how many such runs it used, and nothing treats it as converged silently. (Task 4)
4. **Coinciding wells.** When two solutions give the same vectors on a child, the union reports the dropped vectors per source and still returns `m` vectors if other sources can fill them, and errors otherwise unless `allow_fewer = true`. (Task 5)
5. **An augmented child that does not belong.** An augmented child whose node is not in the tree, that is given twice, or that is passed together with `previous=`, is an `ArgumentError`; an ancestor with `arithmetic = :real` is an `ArgumentError` naming the node. (Task 6)

---

### Task 1: Solve one node on given children, and a subtree on a term list

**Files:**
- Modify: `src/materialize.jl` (`_materialize`, about lines 560–690, and `_materialize_terms`, about lines 134–148)
- Test: `test/test_materialize.jl` (slow-tier file; run it by name)

**Interfaces:**
- Produces, internal to the package (not exported), used by Tasks 3 to 6:

```julia
# Steps of `_materialize` after the recursion into children: operator, solve, selection, phase, peaks, ports.
# `children` are bases (for a leaf) or already materialized children; `terms` are the terms acting at this node.
_solve_node(node::Node, children::Vector{Any}, terms::Vector{Term}, θ::NamedTuple;
            constant::Number = 0.0, ports::Vector{Tuple{Symbol,Port}} = Tuple{Symbol,Port}[],
            joint_ports::Vector{_FactorKey} = _FactorKey[], real::Bool = false, prev = nothing) -> MaterializedNode

# Phase fixing, peak amplitudes, ports and joint ports for kept vectors that are given, not solved for.
_assemble_node(node::Node, children::Vector{Any}, energies::Vector{Float64}, vectors::AbstractMatrix;
               ports, joint_ports, real::Bool = false) -> MaterializedNode

# `node` as the root of its own tree, from the terms of `terms` whose modes all lie inside it.
# `keys` are factor keys inside `node` that an outside caller will need: a one-factor key becomes a port,
# a longer one a joint port, and the nodes below form whatever those need.
_solve_subtree(node::Node, terms::Vector{Term}, θ::NamedTuple; keys::Vector{_FactorKey} = _FactorKey[],
               complex::Bool = false, modetypes::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}()) -> MaterializedNode

# The terms of `terms` that `route` places at `node` itself when `node` is the root of its own tree.
_terms_at(node::Node, terms::Vector{Term}) -> Vector{Term}
```

`_materialize` itself becomes the recursion plus a call to `_solve_node`, with no change in behaviour.

- [ ] **Step 1: Write the failing tests** in `test/test_materialize.jl`

```julia
@testset "a subtree solved on its own terms equals the same node inside the full tree" begin
    K = KronFreeCircuits
    spec = load_spec(fixture("zero_pi.json")); θ = zero_pi_params()
    A = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=6, keep=lowest(6))
    B = Node([:phi => PhaseGrid(-12.0, 12.0, 121)]; solver=Dense(), compute=6, keep=lowest(6))
    root = Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))
    sys = materialize(root, spec, θ)
    terms = circuit_terms(spec)
    keys = [[(:phi, expiphi(1.0))], [(:phi, phiport())]]
    b = K._solve_subtree(B, terms, θ; keys, complex=true)
    @test b.energies ≈ materialized(sys, B).energies rtol=1e-12
    @test all(haskey(b.ports, only(k)) for k in keys)
    # one node on given children, with an extra term appended at the node
    extra = Term(Coefficient(0.3 - 0.2im), [Factor(:phi, expiphi(1.0))])
    extra_c = Term(Coefficient(0.3 + 0.2im), [Factor(:phi, expiphi(-1.0))])
    at = K._terms_at(B, terms)
    c = K._solve_node(B, b.children, vcat(at, [extra, extra_c]), θ; ports=[only(k) for k in keys])
    H = primitive_hamiltonian(vcat(at, [extra, extra_c]), θ, [:phi => PhaseGrid(-12.0, 12.0, 121)])
    @test c.energies ≈ eigvals(Hermitian(Matrix(H)))[1:6] rtol=1e-10
    # given vectors: the same kept vectors come back with the same ports
    d = K._assemble_node(B, b.children, b.energies, b.vectors; ports=[only(k) for k in keys])
    @test d.ports[(:phi, phiport())] ≈ b.ports[(:phi, phiport())]
end
```

Take the mode names, the parameter helper and the grid sizes from the existing zero-pi tests in the same file, and `primitive_hamiltonian`'s argument forms from `test/helpers.jl`.

- [ ] **Step 2: Run and confirm failure**

Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_materialize.jl"])'`
Expected: FAIL with `UndefVarError: _solve_subtree`.

- [ ] **Step 3: Implement.**
Move the body of `_materialize` after its child loop into `_solve_node`, and its tail (phase fixing, peak amplitudes, the needed ports and joint ports, the edge warning, the element-type check) into `_assemble_node`, which `_solve_node` calls.
`_solve_subtree` filters `terms` to those whose modes lie in `node`, routes them with `route(node, ·)`, adds `keys` to what `required_ports` and `required_joint_ports` return for `node`, classifies (every node complex when `complex` is true), and runs `_materialize`.

- [ ] **Step 4: Run the files that cover `_materialize`**

Run: `julia --project -e 'using Pkg; Pkg.test(test_args=["test_materialize.jl","test_composite.jl","test_selection.jl","test_phase.jl","test_tree.jl","test_given.jl"])'`, then the default tier `julia --project -e 'using Pkg; Pkg.test()'`.
Expected: PASS, with no change to any existing expectation.

- [ ] **Step 5: Commit** (`materialize: solve one node on given children, and a subtree on a term list`)

---

### Task 2: Expectation tables and mean-field terms

**Files:**
- Create: `src/meanfield.jl` (included after `materialize.jl` in `src/KronFreeCircuits.jl`)
- Test: `test/test_meanfield.jl` (fast tier)

**Interfaces:**
- Consumes: `_solve_subtree` (Task 1); `route`, `_inside_key`, `_axis_groups`, `_node_port`, `_joint_port` (existing).
- Produces, exported:

```julia
const Expectations = Dict{_FactorKey,ComplexF64}      # one child's expectation value of each of its factor keys

# The terms routed at `parent` (those that straddle its children), and the factor keys they need inside `child`.
straddling_terms(parent::Node, terms::Vector{Term}) -> Vector{Term}
child_keys(parent::Node, terms::Vector{Term}, child::Node) -> Vector{_FactorKey}

# Expectation values in a solved child: its `state`-th kept state, or an average with `weights` over kept states.
expectations(solved::MaterializedNode, keys::Vector{_FactorKey}; state::Int = 1,
             weights::Union{Nothing,AbstractVector{<:Real}} = nothing) -> Expectations

# Starting points that correspond to no solved state.
pinned_expectations(keys::Vector{_FactorKey}, displacement::AbstractDict{Symbol,<:Real}) -> Expectations
delocalized_expectations(keys::Vector{_FactorKey}) -> Expectations

# The extra terms for `child`: each straddling term reduced to the child's own factors, its coefficient at `θ`
# times the other children's expectation values; terms with the same factors are merged.
mean_field_terms(parent::Node, terms::Vector{Term}, θ::NamedTuple, child::Node,
                 others::AbstractDict{Node,Expectations}; outside::Union{Nothing,Expectations} = nothing) -> Vector{Term}
```

`parent` need not be the root of the tree that `terms` describes.
A term with a factor on a mode outside `parent` is treated as absent when `outside` is `nothing`.
When `outside` is given, it must hold the expectation value of each such term's outside factors (as one key), and the term is then reduced like any other, with that value as one more factor of its coefficient; a missing key is an `ArgumentError` naming it.

Rules:
`pinned_expectations` gives, for a key, the product over its factors of: `exp(im*k*φ⁰)` for `expiphi(k)`; `φ⁰` for `phi`; `φ⁰^2` for `phi2`; `0` for `n` and `n2`; `1` for `exp2pin`; a mode absent from `displacement` has `φ⁰ = 0`.
`delocalized_expectations` gives `0` for every key.
Every child of `parent` must be a `Node` (Review Focus 2).
A one-factor key reads the child's port and a longer key its joint port, computed from the kept vectors when the solved child did not form it.
With `weights`, they must be nonnegative, sum to one within `1e-12`, and have one entry per kept state.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "mean-field terms" begin
    K = KronFreeCircuits
    quad(m) = [Term(Coefficient(4.0), [Factor(m, n2port())]), Term(Coefficient(0.5), [Factor(m, phi2port())])]
    t1 = Term(Coefficient(2.0), [Factor(:a, phiport()), Factor(:b, phiport())])
    t2 = Term(Coefficient(0.5 + 0.25im), [Factor(:a, expiphi(1.0)), Factor(:b, expiphi(-2.0)), Factor(:c, expiphi(1.0))])
    t3 = Term(Coefficient(0.7), [Factor(:b, phiport()), Factor(:c, phiport())])           # inside B: not straddling
    terms = vcat(quad(:a), quad(:b), quad(:c), [t1, t2, t3])
    grid = PhaseGrid(-6.0, 6.0, 61)
    A = Node([:a => grid]; solver=Dense(), compute=4, keep=lowest(4))
    B = Node([:b => grid, :c => grid]; solver=Dense(), compute=4, keep=lowest(4))
    root = Node([A, B]; solver=Dense(), compute=2, keep=lowest(2))
    @test Set(straddling_terms(root, terms)) == Set([t1, t2])
    kb, kbc = [(:b, phiport())], K._sortkey!([(:b, expiphi(-2.0)), (:c, expiphi(1.0))])
    @test Set(child_keys(root, terms, B)) == Set([kb, kbc])
    @test Set(child_keys(root, terms, A)) == Set([[(:a, phiport())], [(:a, expiphi(1.0))]])
    others = IdDict{Node,Expectations}(B => Expectations(kb => 0.3, kbc => 0.2 - 0.1im))
    mf = mean_field_terms(root, terms, (;), A, others)
    val(key) = only(t.coeff((;)) for t in mf if K._factor_key(t) == key)
    @test length(mf) == 2
    @test val([(:a, phiport())]) ≈ 2.0 * 0.3
    @test val([(:a, expiphi(1.0))]) ≈ (0.5 + 0.25im) * (0.2 - 0.1im)
    # the joint expectation of B is not the product of its single ones (Review Focus 1)
    b = K._solve_subtree(B, terms, (;); keys=[kb, kbc, [(:b, expiphi(-2.0))], [(:c, expiphi(1.0))]], complex=true)
    e = expectations(b, [kbc, [(:b, expiphi(-2.0))], [(:c, expiphi(1.0))]])
    @test abs(e[kbc] - e[[(:b, expiphi(-2.0))]] * e[[(:c, expiphi(1.0))]]) > 1e-4
    @test expectations(b, [kb]; weights=[1, 0, 0, 0])[kb] ≈ expectations(b, [kb])[kb]
    @test_throws ArgumentError expectations(b, [kb]; weights=[0.5, 0.2])
    # starting tables
    @test pinned_expectations([kb, kbc], Dict(:b => 0.5, :c => -1.0)) ==
          Expectations(kb => 0.5, kbc => cis(-2.0 * 0.5) * cis(1.0 * -1.0))
    @test all(iszero, values(delocalized_expectations([kb, kbc])))
    # a term reaching outside the parent is absent unless its outside expectation is supplied
    t4 = Term(Coefficient(3.0), [Factor(:a, phiport()), Factor(:z, phiport())])
    with_z = vcat(terms, quad(:z), [t4])
    @test length(mean_field_terms(root, with_z, (;), A, others)) == 2
    out = Expectations([(:z, phiport())] => 0.1)
    mfz = mean_field_terms(root, with_z, (;), A, others; outside=out)
    @test only(t.coeff((;)) for t in mfz if K._factor_key(t) == [(:a, phiport())]) ≈ 2.0 * 0.3 + 3.0 * 0.1
    @test_throws ArgumentError mean_field_terms(root, with_z, (;), A, others; outside=Expectations())
    # a parent with a bare mode child is refused (Review Focus 2)
    bad = Node([A, :b => grid, :c => grid]; solver=Dense(), compute=2, keep=lowest(2))
    @test_throws "Node" straddling_terms(bad, terms)
end
```

- [ ] **Step 2: Run and confirm failure** (`UndefVarError: straddling_terms`)

- [ ] **Step 3: Implement** in `src/meanfield.jl`, and export the names above.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_meanfield.jl"])` and the default tier, expected PASS

- [ ] **Step 5: Commit** (`Mean-field terms from expectation tables`)

---

### Task 3: A mean-field child, and the product energy

**Files:**
- Modify: `src/meanfield.jl`
- Test: `test/test_meanfield.jl`

**Interfaces:**
- Consumes: Task 1's `_solve_subtree`, `_solve_node`, `_terms_at`; Task 2.
- Produces, exported:

```julia
# A child solved with its bare terms plus mean-field terms; its energies are mean-field energies.
struct MeanFieldChild
    node::Node
    solved::MaterializedNode          # kept states of the mean-field operator, with the ports of `child_keys`
    terms::Vector{Term}               # the mean-field terms that were added
    bare_energy::Float64              # expectation of the child's bare operator in the state used for `expectations`
    expectations::Expectations        # of that state, for every key of `child_keys`
end

# The child's bare solve: its own terms only, every node complex, forming the ports of `child_keys`.
bare_child(parent::Node, terms::Vector{Term}, θ::NamedTuple, child::Node) -> MaterializedNode

# One mean-field solve of `child`: the children of `child` are taken from `bare` and not solved again.
mean_field_child(parent::Node, terms::Vector{Term}, θ::NamedTuple, child::Node, bare::MaterializedNode,
                 others::AbstractDict{Node,Expectations}; state::Int = 1, weights = nothing) -> MeanFieldChild

# The expectation of the parent's full operator in the product state: no double counting.
product_energy(parent::Node, terms::Vector{Term}, θ::NamedTuple, bare_energies::AbstractDict{Node,Float64},
               tables::AbstractDict{Node,Expectations}; constant::Number = 0.0) -> Float64
```

`bare_energy` is the mean-field energy of the state (or the weighted average) minus the expectation of the mean-field terms in it.
`product_energy` is `constant + Σ bare_energies + real(Σ over straddling terms of coeff(θ) · Π over children of the child's expectation of its key)`.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "mean-field child and product energy" begin
    spec = coupled_oscillators_spec()
    grid = PhaseGrid(-10.0, 10.0, 161)
    S = Node([:s => grid]; solver=Dense(), compute=6, keep=lowest(6))
    F = Node([:f => grid]; solver=Dense(), compute=6, keep=lowest(6))
    root = Node([S, F]; solver=Dense(), compute=4, keep=lowest(4))
    # uncoupled: the product state is exact, so one mean-field solve gives the exact ground energy
    θ0 = (a=1.0, b=4.0, c=0.0, E_Cf=1.0, E_Cs=0.5); terms = circuit_terms(spec)
    bs, bf = bare_child(root, terms, θ0, S), bare_child(root, terms, θ0, F)
    es = expectations(bs, child_keys(root, terms, S)); ef = expectations(bf, child_keys(root, terms, F))
    E = product_energy(root, terms, θ0, IdDict(S => bs.energies[1], F => bf.energies[1]), IdDict(S => es, F => ef))
    @test E ≈ energies(materialize(root, spec, θ0))[1] rtol=1e-10
    # coupled: a mean-field child is the bare child plus a linear term, and the product energy is an upper bound
    θ = (a=1.0, b=4.0, c=0.8, E_Cf=1.0, E_Cs=0.5)
    shifted = IdDict{Node,Expectations}(F => pinned_expectations(child_keys(root, terms, F), Dict(:f => 0.4)))
    ms = mean_field_child(root, terms, θ, S, bare_child(root, terms, θ, S), shifted)
    @test ms.solved.energies[1] ≈ 0.5 * sqrt(8 * 0.5 * 1.0) - (0.8 * 0.4)^2 / (2 * 1.0) atol=1e-6   # ½ω − (c φ_f)²/2a
    @test ms.expectations[[(:s, phiport())]] ≈ -0.8 * 0.4 / 1.0 atol=1e-6
    @test ms.bare_energy ≈ ms.solved.energies[1] - 0.8 * 0.4 * real(ms.expectations[[(:s, phiport())]]) atol=1e-9
    exact = energies(materialize(Node([:s => grid, :f => grid]; solver=LOBPCG(), compute=2, keep=lowest(2)), spec, θ))[1]
    mf = mean_field_child(root, terms, θ, F, bare_child(root, terms, θ, F), IdDict{Node,Expectations}(S => ms.expectations))
    Ep = product_energy(root, terms, θ, IdDict(S => ms.bare_energy, F => mf.bare_energy),
                        IdDict(S => ms.expectations, F => mf.expectations))
    @test Ep >= exact - 1e-9
end
```

- [ ] **Step 2: Run and confirm failure**; **Step 3: implement**; **Step 4: run** `test_meanfield.jl` and the default tier, expected PASS.

- [ ] **Step 5: Commit** (`Mean-field child and product energy`)

---

### Task 4: The self-consistent loop

**Files:**
- Modify: `src/meanfield.jl`
- Test: `test/test_meanfield.jl`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces, exported:

```julia
struct MeanFieldSolution
    energy::Float64                                   # the product energy
    children::IdDict{Node,MeanFieldChild}             # every child, solved in the final expectation values
    iterations::Int
    converged::Bool
    start                                             # the start it came from: `nothing`, a displacement Dict, or a solution
end

# A start is `nothing` (delocalized), a `Dict{Symbol,Float64}` of displacements, or a `MeanFieldSolution` (warm start).
default_starts(parent::Node; displacements = (0.0, π, -π), max_starts::Int = 81) -> Vector{Any}

self_consistent(parent::Node, terms::Vector{Term}, θ::NamedTuple, bare::AbstractDict{Node,MaterializedNode};
                start = nothing, tol::Real = 1e-11, maxiter::Int = 300, mixing::Real = 0.5,
                constant::Number = 0.0, state::Int = 1, weights = nothing, outside = nothing) -> MeanFieldSolution

mean_field_solutions(parent::Node, terms::Vector{Term}, θ::NamedTuple; starts = default_starts(parent),
                     distinct::Real = 1e-3, kwargs...) -> Vector{MeanFieldSolution}     # distinct, lowest energy first
mean_field_solutions(parent::Node, spec::Spec, θ::NamedTuple; kwargs...)                # terms and constant from the spec
```

The loop, per iteration: every child is solved from the same tables (all children together); the new tables and bare energies are averaged with the previous ones when mixing is on; the product energy is computed from them.
Mixing switches on for the rest of the run once the energy has risen (by more than `1e-12`) twice.
Convergence: `|E − E_prev| < tol·max(1, |E|)` and the largest change of any table entry below `sqrt(tol)`.
A run that reaches `maxiter` is returned with `converged = false`.
`mean_field_solutions` sorts by energy and keeps a solution when, against every one already kept, some table entry differs by more than `distinct`.
`default_starts` returns the delocalized start and every combination of `displacements` over the modes under `parent`; more than `max_starts` combinations is an `ArgumentError` that gives the count and says to pass `starts` or raise `max_starts`.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "self-consistent loop" begin
    spec = load_spec(fixture("zero_pi.json")); θ = zero_pi_params(); terms = circuit_terms(spec)
    A = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=4, keep=lowest(4))
    B = Node([:phi => PhaseGrid(-12.0, 12.0, 121)]; solver=Dense(), compute=4, keep=lowest(4))
    root = Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))
    sols = mean_field_solutions(root, spec, θ)
    @test issorted([s.energy for s in sols]) && all(s.converged for s in sols)
    exact = energies(materialize(Node([:theta => ChargeBasis(8), :phi => PhaseGrid(-12.0, 12.0, 121)];
                                      solver=Dense(), compute=1, keep=lowest(1)), spec, θ))[1]
    @test sols[1].energy >= exact - 1e-9                                  # an upper bound
    # a fixed point: one more solve of each child from the converged tables changes nothing
    s = sols[1]; tables = IdDict{Node,Expectations}(n => c.expectations for (n, c) in s.children)
    bare = IdDict(n => bare_child(root, terms, θ, n) for n in (A, B))
    again = mean_field_child(root, terms, θ, A, bare[A], IdDict{Node,Expectations}(B => tables[B]))
    @test maximum(abs(again.expectations[k] - tables[A][k]) for k in keys(tables[A])) < 1e-5
    # a warm start from a solution converges at once to the same energy
    w = self_consistent(root, terms, θ, bare; start=s, constant=real(KronFreeCircuits.constant_term(spec)(θ)))
    @test w.energy ≈ s.energy atol=1e-9 && w.iterations <= 3
    # the iteration limit is reported, not hidden (Review Focus 3)
    short = self_consistent(root, terms, θ, bare; start=Dict(:phi => π), maxiter=1)
    @test !short.converged && short.iterations == 1
    @test length(default_starts(root)) == 3^2 + 1
    @test_throws "max_starts" default_starts(root; max_starts=4)
end

@testset "two coupled fluxonia at half flux: one solution per pair of wells" begin
    pair = numeric_spec(; modes=[:a => :e, :b => :e], E_C=[1.0 0.0; 0.0 1.0], E_L=[1.0 0.1; 0.1 1.0],
                        flux_linear=[0.0, 0.0], n_g=[0.0, 0.0], cosines=[(4.0, [1.0, 0.0], π), (4.0, [0.0, 1.0], π)])
    grid = PhaseGrid(-12.0, 12.0, 161)
    A = Node([:a => grid]; solver=Dense(), compute=4, keep=lowest(4))
    B = Node([:b => grid]; solver=Dense(), compute=4, keep=lowest(4))
    root = Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))
    starts = [Dict(:a => x, :b => y) for x in (2.0, -2.0), y in (2.0, -2.0)]
    sols = mean_field_solutions(root, pair, (;); starts=vec(starts))
    @test length(sols) == 4                                               # the four combinations of wells are distinct
    @test sols[1].energy ≈ sols[2].energy atol=1e-8                       # opposite wells: lower, and a degenerate pair
    @test sols[3].energy ≈ sols[4].energy atol=1e-8 && sols[3].energy > sols[1].energy + 1e-4
    exact = energies(materialize(Node([:a => grid, :b => grid]; solver=LOBPCG(), compute=2, keep=lowest(2)), pair, (;)))[1]
    @test sols[1].energy >= exact - 1e-9
end
```

Use the fixture's own mode names; if the zero-pi well structure makes a start fail to converge within 300 iterations, keep the assertion on the lowest solution and assert the count of unconverged runs instead, with the reason in a comment.

- [ ] **Step 2: Run and confirm failure**; **Step 3: implement**; **Step 4: run** `test_meanfield.jl` and the default tier, expected PASS.

- [ ] **Step 5: Commit** (`Self-consistent mean-field loop with several starts`)

---

### Task 5: The union basis

**Files:**
- Create: `src/augment.jl` (included after `meanfield.jl`)
- Test: `test/test_augment.jl` (fast tier)

**Interfaces:**
- Produces, exported:

```julia
struct BasisSource
    label::String
    vectors::Matrix{ComplexF64}                 # columns on the child's own product space, lowest state first
    energies::Vector{Float64}                   # one per column, used by `window`
    cap::Union{Nothing,Int}                     # at most this many columns are offered
    window::Union{Nothing,Float64}              # only columns within this energy of the source's first
end
BasisSource(label, vectors, energies; cap = nothing, window = nothing)

struct UnionReport
    labels::Vector{String}
    taken::Vector{Int}                          # vectors accepted from each source
    dropped::Vector{Int}                        # vectors offered and rejected as dependent
end

union_basis(sources::Vector{BasisSource}; m::Int, drop::Real = 1e-6, allow_fewer::Bool = false)
    -> (U::Matrix{ComplexF64}, report::UnionReport)
```

Merging is level by level: the first offered column of each source in order, then the second of each, and so on.
Each candidate is orthogonalized against the accepted vectors by classical Gram–Schmidt applied twice and accepted when its remainder has norm above `drop`; it is then normalized.
Merging stops at `m` accepted vectors.
A capped or windowed source stops offering at its limit, so an unlimited source fills the remainder.
Fewer than `m` accepted vectors is an `ArgumentError` that prints the report, unless `allow_fewer` is true.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "union basis" begin
    e(i, n=8) = (v = zeros(ComplexF64, n); v[i] = 1; v)
    src(label, cols; kw...) = BasisSource(label, reduce(hcat, cols), collect(1.0:length(cols)); kw...)
    a = src("well 1", [e(1), e(2), e(3)]); b = src("well 2", [e(4), e(1), e(5)])
    U, r = union_basis([a, b]; m=4)
    @test U ≈ hcat(e(1), e(4), e(2), e(3)) && r.taken == [3, 1] && r.dropped == [0, 1]     # level by level; e(1) twice
    @test U' * U ≈ I
    # a cap leaves the remainder to the uncapped source
    U2, r2 = union_basis([src("well 1", [e(1), e(2), e(3)]; cap=1), src("bare", [e(6), e(7), e(8)])]; m=4)
    @test r2.taken == [1, 3]
    # a window in energy
    @test union_basis([src("w", [e(1), e(2), e(3)]; window=1.5), src("bare", [e(6), e(7)])]; m=4)[2].taken == [2, 2]
    # coinciding sources: reported, and an error when m cannot be reached (Review Focus 4)
    @test_throws "dropped" union_basis([a, a]; m=4)
    U3, r3 = union_basis([a, a]; m=4, allow_fewer=true)
    @test size(U3, 2) == 3 && r3.dropped == [0, 3]
    # nearly dependent: a remainder below the threshold is dropped, above it is kept
    near(δ) = normalize(e(1) + δ * e(2))
    @test union_basis([src("x", [e(1)]), src("y", [near(1e-8)])]; m=2, allow_fewer=true)[2].dropped == [0, 1]
    @test size(union_basis([src("x", [e(1)]), src("y", [near(1e-3)])]; m=2)[1], 2) == 2
end
```

- [ ] **Step 2: Run and confirm failure**; **Step 3: implement**; **Step 4: run** `Pkg.test(test_args=["test_augment.jl"])` and the default tier, expected PASS.

- [ ] **Step 5: Commit** (`union_basis: merge sources level by level, with caps and a drop report`)

---

### Task 6: The augmented child, and a parent that consumes it

**Files:**
- Modify: `src/augment.jl`, `src/materialize.jl` (the public `materialize` methods, `_materialize_terms`, the child loop of `_materialize`), `src/character.jl` (`classify`)
- Test: `test/test_augment.jl`

**Interfaces:**
- Consumes: Task 1's `_assemble_node`, `_terms_at`; Task 3's `bare_child`; Task 5.
- Produces, exported:

```julia
struct AugmentedChild
    node::Node
    solved::MaterializedNode          # the bare operator projected on `basis` and diagonalized: vectors = basis * W
    basis::Matrix{ComplexF64}         # the union basis U, on the child's own product space
    terms::Vector{Term}               # the child's bare terms
    report::UnionReport
end

# Project the bare operator of `bare` on `U`, diagonalize, and form the ports of `child_keys(parent, terms, child)`.
augment(parent::Node, terms::Vector{Term}, θ::NamedTuple, child::Node, bare::MaterializedNode,
        U::AbstractMatrix, report::UnionReport) -> AugmentedChild

materialize(root::Node, spec::Spec, θ::NamedTuple; augmented::AbstractVector{AugmentedChild} = AugmentedChild[], kwargs...)
materialize(root::Node, terms::Vector{Term}, θ::NamedTuple; augmented = AugmentedChild[], kwargs...)
```

In `materialize`, a node named by an entry of `augmented` is not solved: its `solved` is used as that child, after forming from its stored vectors any port or joint port the tree needs that it lacks.
`classify` treats such a node as complex.
Errors (Review Focus 5), all `ArgumentError`: an augmented node that is not in the tree; the same node twice; `augmented` together with `previous`; an ancestor of an augmented node with `arithmetic = :real`, naming the ancestor.

- [ ] **Step 1: Write the failing tests**

```julia
@testset "augmented child" begin
    spec = load_spec(fixture("zero_pi.json")); θ = zero_pi_params(); terms = circuit_terms(spec)
    mk() = (A = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=17, keep=lowest(17));
            B = Node([:phi => PhaseGrid(-12.0, 12.0, 121)]; solver=Dense(), compute=12, keep=lowest(12));
            (A, B, Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))))
    A, B, root = mk()
    plain = energies(materialize(root, spec, θ))
    bareA, bareB = bare_child(root, terms, θ, A), bare_child(root, terms, θ, B)
    # the bare vectors as the union basis reproduce the bare child exactly
    rep(n) = UnionReport(["bare"], [n], [0])
    augB = augment(root, terms, θ, B, bareB, Matrix{ComplexF64}(bareB.vectors), rep(12))
    @test augB.solved.energies ≈ bareB.energies rtol=1e-10
    @test energies(materialize(root, spec, θ; augmented=[augB])) ≈ plain rtol=1e-9
    # a rotated basis of the same span gives the same parent spectrum
    Q = Matrix(qr(randn(ComplexF64, 12, 12)).Q)
    augQ = augment(root, terms, θ, B, bareB, bareB.vectors * Q, rep(12))
    @test energies(materialize(root, spec, θ; augmented=[augQ])) ≈ plain rtol=1e-9
    # the whole space of A: the parent no longer depends on A's truncation
    full = augment(root, terms, θ, A, bareA, Matrix{ComplexF64}(I, 17, 17), rep(17))
    @test energies(materialize(root, spec, θ; augmented=[full])) ≈ plain rtol=1e-9
    @test augB.basis ≈ bareB.vectors && augB.terms == KronFreeCircuits._terms_at(B, terms)
    @test arithmetic(materialize(root, spec, θ; augmented=[augB]), root) == :complex
    # Review Focus 5
    A2, B2, root2 = mk()
    @test_throws ArgumentError materialize(root2, spec, θ; augmented=[augB])              # node of another tree
    @test_throws ArgumentError materialize(root, spec, θ; augmented=[augB, augB])
    @test_throws ArgumentError materialize(root, spec, θ; augmented=[augB], previous=materialize(root, spec, θ))
    Ar = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=4, keep=lowest(4))
    Br = Node([:phi => PhaseGrid(-12.0, 12.0, 121)]; solver=Dense(), compute=4, keep=lowest(4))
    rootr = Node([Ar, Br]; solver=Dense(), compute=4, keep=lowest(4), arithmetic=:real)
    bBr = bare_child(rootr, terms, θ, Br)
    augr = augment(rootr, terms, θ, Br, bBr, Matrix{ComplexF64}(bBr.vectors), rep(4))
    @test_throws "real" materialize(rootr, spec, θ; augmented=[augr])
end
```

Use the fixture's actual mode names and the accessor names of the existing tests (`arithmetic`, `energies`, `materialized`).

- [ ] **Step 2: Run and confirm failure**; **Step 3: implement.**
The projected matrix is `U' * (H * U)` with `H` the child's bare node operator on the children of `bare`; the vectors of `solved` are `U * W`, assembled with `_assemble_node`.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_augment.jl","test_materialize.jl","test_character.jl"])` and the default tier, expected PASS.

- [ ] **Step 5: Commit** (`Augmented children: a projected bare child that the parent consumes`)

---

### Task 7: The wrapper, and grids per frame

**Files:**
- Modify: `src/augment.jl`
- Create: `src/suggest.jl` (included after `bases.jl`)
- Test: `test/test_augment.jl`, `test/test_bases.jl`

**Interfaces:**
- Consumes: Tasks 3 to 6.
- Produces, exported:

```julia
# Runs the loop and the union for every child of `parent`, with nothing of its own.
mean_field_augment(parent::Node, spec::Spec, θ::NamedTuple; m::Int, window::Real = 0.5, max_solutions::Int = 4,
                   bare::Bool = false, cap = nothing, starts = default_starts(parent), allow_fewer::Bool = false,
                   kwargs...) -> (; children::Vector{AugmentedChild}, solutions::Vector{MeanFieldSolution},
                                   used::Vector{Int}, energy::Float64, unconverged::Int)

# A basis for each mode of `spec` at `θ`: a charge cutoff for a compact mode, a symmetric phase grid otherwise.
suggest_bases(spec::Spec, θ::NamedTuple = (;); charge_cutoff::Int = 16, max_points::Int = 120,
              extent_scale::Real = 1.0, order::Int = 8) -> Vector{Pair{Symbol,Basis}}

# The largest weight of a solved leaf's lowest kept state on the first or last `order ÷ 2` grid points.
edge_weight(solved::MaterializedNode) -> Float64
```

`mean_field_augment` uses the solutions within `window` of the lowest energy, at most `max_solutions`, and returns their indices in `used` and the number of them that did not converge in `unconverged`.
Each used solution is one source per child (its `m` lowest mean-field states, limited by `cap`); `bare = true` appends the child's bare ladder as an unlimited last source.
A child that kept fewer mean-field states than the union may ask of it is an `ArgumentError` naming the node and the count needed.
`suggest_bases`, for an extended or harmonic mode `i` with `E_C = E_C[i,i]` and `E_L = E_L[i,i]` at `θ`: `zpf = (2 E_C / E_L)^(1/4)`, `ω = sqrt(8 E_C E_L)`, `extent = extent_scale · max(6 zpf, sqrt(2 (2 Σ E_J + 12 ω + 4 E_C) / E_L))` with the sum over the cosines that touch the mode, `step = min(zpf / 6, π / (6 k_max))` with `k_max` the largest `|omega|` on the mode, `points = min(ceil(2 extent / step) + 1, max_points)`, and the grid is `PhaseGrid(-extent, extent, points; order)`.
A mode with `E_L[i,i] == 0` that is not compact is an `ArgumentError`.

- [ ] **Step 1: Write the failing tests**

In `test/test_bases.jl`, with the three KITE specs of Task 8 not yet available, test the formula on a hand-built numeric spec:

```julia
@testset "suggest_bases" begin
    s = numeric_spec(; modes=[:a => :c, :b => :e], E_C=[0.072 0.072; 0.072 2.484170953858], E_L=[0.0 0.0; 0.0 2.54],
                     flux_linear=[0.0, 0.0], n_g=[0.0, 0.0], cosines=[(17.3349, [1, -1.0], 0.0), (16.3251, [1, -1.0], 0.0)])
    b = Dict(suggest_bases(s; charge_cutoff=16, max_points=120, extent_scale=1.5))
    @test b[:a] == ChargeBasis(16)
    zpf = (2 * 2.484170953858 / 2.54)^0.25; ω = sqrt(8 * 2.484170953858 * 2.54)
    ext = 1.5 * max(6zpf, sqrt(2 * (2 * (17.3349 + 16.3251) + 12ω + 4 * 2.484170953858) / 2.54))
    @test b[:b] == PhaseGrid(-ext, ext, 120)                       # capped: the formula asks for more than 120 points
    @test length(Dict(suggest_bases(s; max_points=10_000))[:b]) == ceil(Int, 2 * (ext / 1.5) / min(zpf / 6, π / 6)) + 1
end
```

Use the package's way of reading a grid's point count in place of `length` if it differs, and compare the grids field by field if `PhaseGrid` defines no `==`.

In `test/test_augment.jl`:

```julia
@testset "mean_field_augment" begin
    spec = load_spec(fixture("zero_pi.json")); θ = zero_pi_params()
    A = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=10, keep=lowest(10))
    B = Node([:phi => PhaseGrid(-12.0, 12.0, 121)]; solver=Dense(), compute=10, keep=lowest(10))
    root = Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))
    exact = energies(materialize(Node([:theta => ChargeBasis(8), :phi => PhaseGrid(-12.0, 12.0, 121)];
                                      solver=Dense(), compute=4, keep=lowest(4)), spec, θ))
    err(m; kw...) = (r = mean_field_augment(root, spec, θ; m, kw...);
                     maximum(abs, energies(materialize(root, spec, θ; augmented=r.children)) .- exact))
    @test err(10; bare=true) <= err(6; bare=true) + 1e-12                  # converges as m grows
    r = mean_field_augment(root, spec, θ; m=8, bare=true, cap=2)
    @test all(c.report.taken[end] >= 8 - 2 * length(r.used) for c in r.children)   # the bare ladder fills the rest
    @test r.energy == r.solutions[1].energy && r.unconverged == 0
    @test all(size(c.basis, 2) == 8 for c in r.children)
    small = Node([Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=3, keep=lowest(3)), B];
                 solver=Dense(), compute=4, keep=lowest(4))
    @test_throws "kept" mean_field_augment(small, spec, θ; m=8)
end
```

- [ ] **Step 2: Run and confirm failure**; **Step 3: implement**; **Step 4: run** `Pkg.test(test_args=["test_augment.jl","test_bases.jl"])` and the default tier, expected PASS.

- [ ] **Step 5: Commit** (`mean_field_augment, suggest_bases and edge_weight`)

---

### Task 8: KITE fixtures and the reference's intermediate numbers

**Files:**
- Create: `test/python/generate_kite_fixtures.jl` (a generator, not a `test_*.jl` file)
- Create: `test/fixtures/kite/published.json`, `normal.json`, `optimum.json`, `meanfield_reference.json` (generated, committed)
- Create: `test/test_meanfield_kite.jl`; register it in `SLOW_FILES` of `test/runtests.jl`
- Modify: `test/test_meanfield.jl` (one fast check)

**Interfaces:**
- Consumes: `kite()` of `test/python/kite.jl`; `Spec(qc, S)` and `save_spec`; circuit-quantization's `product_state` module (`default_bases`, `_scf`, `_initial`, `_mode_hamiltonian`) for the reference numbers; Tasks 4 and 7.
- Produces: the three numeric specs, and `meanfield_reference.json` with, per frame: the grid extents and point counts of `default_bases(h, charge_cutoff=16, max_points=120, extent_scale=1.5)`; for each distinct solution of the 33 starts within 0.5 GHz of the lowest, its product energy, its iteration count, its start, and the three lowest eigenvalues of each mode's mean-field operator.

The 33 starts are: the delocalized start, and every combination of the compact mode at `0` or `π` with each extended mode at `0`, `π/2`, `π`, `-π/2`.
The generator states in its header how to run it (the Python-tier environment variables) and that the fixtures are regenerated only on purpose.
Expected values, from a run of the reference on this machine (product energies in GHz): published `-8.083188250415` twice; normal modes `-8.466359372280` and `-8.466359372269`; optimum `-9.429473599561` and `-9.429473599557`; two solutions within 0.5 GHz in every frame.
If the generator's numbers differ from these beyond `1e-9`, stop and report.

- [ ] **Step 1: Write the generator and run it**

Run: `JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=test/python test/python/generate_kite_fixtures.jl`
Expected: the four files written; the printed product energies equal the values above to `1e-9`.

- [ ] **Step 2: Write the failing tests**

In `test/test_meanfield.jl` (fast): the optimum frame from two starts.

```julia
@testset "KITE optimum frame: both wells from two starts" begin
    ref = KronFreeCircuits.JSON3.read(read(fixture("kite/meanfield_reference.json"), String)).optimum
    spec = load_spec(fixture("kite/optimum.json"))
    leaves = [Node([m => b]; solver=Dense(), compute=3, keep=lowest(3), arithmetic=:complex)
              for (m, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5)]
    root = Node(leaves; compute=4, keep=lowest(4))
    names = [m.name for m in spec.modes]
    starts = [Dict(zip(names, (0.0, 0.0, 0.0))), Dict(zip(names, (π, -π/2, π)))]
    sols = mean_field_solutions(root, spec, (;); starts)
    @test length(sols) == 2 && all(s.converged for s in sols)
    for (s, r) in zip(sols, ref.solutions)
        @test s.energy ≈ r.energy atol=1e-6
        for (leaf, levels) in zip(leaves, r.levels)
            @test s.children[leaf].solved.energies[1:3] ≈ collect(levels) atol=1e-6
        end
    end
end
```

In `test/test_meanfield_kite.jl` (slow): for each of the three frames, the grids of `suggest_bases` equal the fixture's extents to `1e-12` relative and its point counts exactly; `mean_field_solutions` with the 33 starts gives exactly two solutions within 0.5 GHz of the lowest, with product energies equal to the fixture's to `1e-6` and mean-field levels to `1e-6`; and the product energy of every solution is at or above `-9.523995`.

- [ ] **Step 3: Run** `Pkg.test(test_args=["test_meanfield.jl","test_meanfield_kite.jl"])`

Expected: PASS.
If a product energy or level misses `1e-6`, do not loosen it: find which operator differs (compare one mode's mean-field matrix elements against `_mode_hamiltonian` at the fixture's expectation values) and report the cause; the design requires an explanation, not a pass.

- [ ] **Step 4: Record the cost.**
Time `mean_field_solutions` with the 33 starts in the optimum frame and one `self_consistent` run from a warm start, and put both in the commit message; Task 10's loop depends on them.

- [ ] **Step 5: Commit** (`KITE fixtures and the reference's product energies and mean-field levels`)

---

### Task 9: Figure 8(b)

**Files:**
- Create: `bench/kite_meanfield/levels.jl`, `bench/kite_meanfield/Project.toml`, `bench/kite_meanfield/README.md`

**Interfaces:**
- Consumes: the fixtures of Task 8; `mean_field_augment`; `materialize(...; augmented)`. Produces no API.

- [ ] **Step 1: Write `levels.jl`**

For each frame (published, normal modes, optimum): the bases of `suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5)`, one leaf per mode keeping 20 states, the 33 starts of Task 8, the solutions within 0.5 GHz of the lowest (two in every frame), no bare ladder.
For `m` in 8, 12, 16, 20: the union of the two wells' `m` lowest mean-field states per leaf, the augmented children, the root solved for its eight lowest levels to a residual below `1e-9` GHz, and the error `max(levels[1:4] - REF)` with `REF = [-9.523995, -9.510525, -8.453790, -8.337862]`.
The script prints the twelve errors beside the twelve reference values of this plan's table (hard-coded, with a comment naming `mean_field_ref_dont_commit/kite_reference/results/` as their source), their ratios, the union reports, and the product-state gaps `E_prod - REF[1]` beside the paper's `1.44` and `0.09` GHz.
It also prints, for the optimum frame at `m = 8, 12, 16, 20`, the error with one well only and with the bare ladder alone, the two variants of the paper's Appendix C.4.
It exits nonzero if any error is outside a factor of two of its reference value, or if the ordering of the three frames differs from the reference at any `m`.

- [ ] **Step 2: Run it**

Run: `julia --project=bench/kite_meanfield bench/kite_meanfield/levels.jl` (single-threaded; no GPU).
Expected: all checks pass; in the normal-mode and optimum frames the errors agree with the reference to well under one percent; in the published frame to a few percent (see "What the reference is").

- [ ] **Step 3: Compare the reference solve.**
Solve the published frame with this package on `ChargeBasis(24)` and the 160-point grids of `suggest_bases(spec; charge_cutoff=24, max_points=160, extent_scale=2.0)`, with a tree converged to below `1e-6` GHz, and print the four lowest levels beside `REF` (the design's §6.1 step 3; `bench/kite_frames/run.jl` has this as its opt-in reference-grid block and may be reused).

- [ ] **Step 4: Write the README**: the command, the table of twelve errors against the reference, the union reports, the two variants, the reference solve, wall times, and for every target that is missed the identified cause.

- [ ] **Step 5: Commit** (`bench/kite_meanfield: Figure 8(b) from mean-field leaves`)

---

### Task 10: The frame optimization

**Files:**
- Create: `bench/kite_meanfield/optimize.jl`; modify `bench/kite_meanfield/Project.toml` (add `Optim`, `PythonCall`) and `README.md`

**Interfaces:**
- Consumes: `QuantizedCircuit`, `Spec(qc, S)`, `circuit_quantization()`; circuit-quantization's `numerical.parameterization` and `numerical.cost_normal_modes`; `suggest_bases`, `edge_weight`, `mean_field_solutions`, `self_consistent`. Produces no API.

- [ ] **Step 1: Write `optimize.jl`**

A function `frame_score(x) -> Float64` built from the pieces, readable top to bottom:
`S` from circuit-quantization's parameterization of the admissible family (compact block fixed at the identity; the compact–noncompact shear and the noncompact block in polar form; six parameters for the KITE, `x = 0` the identity); the numeric spec `Spec(qc, S)`; bases from `suggest_bases(spec; charge_cutoff=16, max_points=76)`, each grid grown by 1.5 times in extent and points, up to three times, while the lowest solution's `edge_weight` exceeds `1e-8`; the lowest product energy over a fixed list of starts (all modes at zero; the compact mode at `π`; the last extended mode at `π`; both; and the delocalized start) plus the previous call's solution as a warm start; and the cost `E_prod + 1e-4 · F_NM(S)` with `F_NM` circuit-quantization's `cost_normal_modes`.
A singular `S`, or one circuit-quantization rejects, scores `Inf`.
The optimizer is `Optim.BFGS` with finite-difference gradients (step `1e-6`, gradient tolerance `1e-6`, at most 500 iterations), started at `x = 0`.
The script prints the cost and product energy per iteration, the final `S` with its noncompact columns rescaled so that `(Sᵀ C S)_jj = 1`, and then, on the grids of Task 9 in the final frame, the product energies of the solutions within 0.5 GHz of the lowest.
It exits nonzero unless two such solutions are found, both within `2e-3` GHz of `-9.4295`.

- [ ] **Step 2: Run it**

Run: `JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=bench/kite_meanfield bench/kite_meanfield/optimize.jl`
Expected: the optimizer ends in a frame whose two wells have product energy near `-9.4295` GHz; `S` itself is defined only up to the scale and sign of its columns and is not compared.

- [ ] **Step 3: Check the invariances** in the same script: the product energy at the final frame is unchanged, to `1e-6` GHz on grids resized by `suggest_bases`, when the noncompact columns of `S` are scaled by `(1.3, 0.8)` and when one of them changes sign.

- [ ] **Step 4: README**: the loop in words, the iteration log's first and last lines, the number of cost evaluations and the wall time, the final frame, the invariance check, and the three known differences between sources (update order; number of starts; the three definitions of `F_NM`) with which one this script uses.

- [ ] **Step 5: Commit** (`bench/kite_meanfield: frame optimization on the product energy`)

---

### Task 11: Documentation and the slow tier

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` (new dated §20 subsection), `README.md`, `docs/superpowers/STATUS.md`

- [ ] **Step 1: Architecture spec.**
A dated subsection after the last §20 entry: the pieces and their names, the one-level depth rule, what an augmented child stores and why (the perturbative-derivatives design needs the basis and the bare terms), `materialize(...; augmented)`, complex arithmetic, the measured results of Tasks 8 to 10, and every place the build departed from the design, each with its reason (at least: the figure's grids and starts; the published-frame tolerance; the Harmonium benchmark deferred to its own plan).

- [ ] **Step 2: README.**
A section "Mean-field bases" with the self-consistent loop written out in a few lines from the pieces, the wrapper call, and the KITE table.

- [ ] **Step 3: STATUS.md.**
Record what is done, what was measured, what was missed and why, and that the Harmonium benchmark is the next plan.

- [ ] **Step 4: Run every tier once**

Run: the default tier; `KRONFREE_SLOW_TESTS=1 julia --project -e 'using Pkg; Pkg.test()'`; the Python tier.
Expected: PASS; the only warnings of the slow tier are the 20 `fix_phases!` warnings that `main` already prints.

- [ ] **Step 5: Commit** (`Docs: mean-field basis augmentation`)

---

## Acceptance map

| Design item | Task |
|---|---|
| §4.1 mean-field terms from given states | 2 |
| §4.2 diagonalizing a mean-field child | 1, 3 |
| §4.3 the self-consistent loop | 4 |
| §4.4 the union basis and the augmented child | 5, 6 |
| §4.5 the wrapper | 7 |
| §4.6 grids per frame | 7 |
| §5 the frame optimization | 10 |
| §6.1 steps 1 and 2: intermediate numbers | 8 |
| §6.1 step 3: reference ground energy | 9 |
| §6.1 step 4: the twelve points | 9 |
| §6.1 step 5: the optimum from the identity | 10 |
| §6.2 the Harmonium benchmark | a later plan |
| §6.3 other tests | 2, 3, 4, 6, 7, 10 |
