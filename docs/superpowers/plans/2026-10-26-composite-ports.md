# Composite (Joint) Ports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work happens on branch `composite-ports` (off `main`), never on `main`.

**Goal:** A term routed to a branch with several factors inside one child node is applied through one *joint port*, `V† (O₁ ⊗ O₂ ⊗ ⋯) V`, formed in the child's product basis before its single projection onto the kept states, so hierarchies such as the Z4 three-way `[[1,4,5],[2,6,7],[3]]` materialize instead of being rejected.

**Architecture:** The term list already writes every cosine as products of single-mode `expiphi` factors (`circuit_terms`), so nothing changes in ingest or routing. `required_ports` learns to group a term's factors by the child they fall in: a group of one is today's port, a group of two or more is a joint port keyed by the sorted factor list (`_FactorKey`, which already exists). A child forms a joint port by applying each factor along its own axis of the reshaped kept-vector block (ping-pong buffers, as `_transport_previous` does) and contracting once; a factor group inside a grandchild is looked up as that grandchild's joint port, so nesting recurses. The ancestor's operator then sees one local operator per axis, as it does today. A real child forms the joint port from components with the subset expansion `_real_expansion` already used for real operators; the joint factor's conjugation character is the product of its factors' characters.

**Tech Stack:** Julia 1.13, LinearAlgebra. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-26-composite-ports-requirements.md` (the Z4 three-way requirements, section "Required native Julia implementation" and "Acceptance tests"), with the architecture spec `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` §7.3, §20a (the composite-port debt) and §20f (real arithmetic by port character). Executors read the requirements document's "Why the hierarchy requires a joint operator" before any task.

## Global Constraints

- A joint port is formed only as `V† (⊗ₐ Oₐ) V`, the product applied to the kept vectors in the child's product basis before projection; never as a product of separately projected single-mode ports (requirements §"Why the hierarchy requires a joint operator", item 3).
- No seven-mode (full-space) matrix is ever assembled inside the package; joint ports use `apply_axis!` on the reshaped kept-vector block (item 3).
- A joint port's identity is its complete sorted factor list, `(mode, Port)` pairs sorted by `(mode, kind, k)` with `_sortkey!`; `k`'s sign (conjugation) is part of it (item 1).
- Conjugate pairs stay explicit: the `s = −1` cosine term's joint port is the adjoint of the `s = +1` one and both are formed from the terms as routed; Hermiticity of the final operator is checked by test, not enforced by symmetrization (item 5).
- Natural element types (spec §20f): a joint port on a real node is a `Matrix{Float64}` exactly when its character is `:real`, else a `Matrix{ComplexF64}` whose real and imaginary parts are separately computed components; `_check_eltype` asserts this on every joint port.
- Two factors on the same mode in one term remain an `ArgumentError` ("has several factors on mode"); only the multi-mode-child rejection is lifted (item 7).
- Memory: a joint port is a dense `kept × kept` matrix, like every port; forming it needs two scratch blocks of `prod(dims) × kept` (today's `_port_matrix` uses one). No other allocation class is introduced; the docstring states this (item 6).
- Selection rules, `Node(ports=...)` and `SelectionContext` keep single-mode `(mode, Port)` keys; joint ports are requested only by routed terms of strict ancestors, and are formed after selection and phase fixing on the kept vectors (so they need no phase rotation).
- Default test tier: at most +10 s over the baseline measured in Task 1 Step 0; the new `test/test_composite.jl` is in the fast tier and uses `Dense` on tiny bases only; solver comparisons go to `test/test_solvers.jl` (slow tier).
- Running one test file: `julia --project=. -e 'using Test, LinearAlgebra, SparseArrays, Random, KronFreeCircuits; include("test/helpers.jl"); include("test/<file>.jl")'`. "Run the test file" below means this command.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (or the model that wrote them).

## Review Focus

1. A term whose factors inside a child fall on the *same* mode twice (e.g. `n ⊗ n2` on one charge mode) must still throw "has several factors on mode", never be silently formed as a joint port. Pinned in Task 4 (the rewritten `test_character.jl` case).
2. A joint group whose factors all sit inside one grandchild (so the child's own axis group has one entry that is itself a node) must request the grandchild's joint port and form the child's joint port from it; a missing grandchild joint port must be an internal error naming the key, not a `KeyError`. Pinned in Task 5 (nested test).
3. A real child forming a `:paired` joint port must return the same matrix (to roundoff) as the same tree materialized with `arithmetic=:complex` at the child, in `Matrix{ComplexF64}`, and a `:real` joint port (all charge-basis `expiphi`) a `Matrix{Float64}`. Pinned in Task 4 (real vs complex testset).
4. A `PeriodicGrid` mode inside a joint group: the cosine's `expiphi(k)` factor is an ordinary port of the grid, and a twisted `nport()` factor never reaches a joint key because `_expand_twisted_charge` expands it before routing; materialization must succeed and agree with the flat reference. Pinned in Task 5 (mixed-bases test).
5. `required_ports` must stop requesting the single ports of factors that belong to a joint group, otherwise the child forms ports nobody consumes; and the public `port(sys, key)` must give a clear "not formed; available: …" error for an absent joint key. Pinned in Task 1 and Task 4.

---

### Task 1: Group ancestor factors by child in `tree.jl`

**Files:**
- Modify: `src/tree.jl:507-545` (`required_ports` and its docstring; add `required_joint_ports` after it)
- Modify: `src/KronFreeCircuits.jl` (nothing exported; `required_joint_ports` is internal like `required_ports`)
- Test: `test/test_tree.jl` (extend the `"route and required_ports"` testset or add `"joint port requests"`)

**Interfaces:**
- Consumes: `route`, `_all_nodes`, `_subnodes`, `modes`, `_FactorKey`, `_sortkey!`, `_factor_key`. These three live in `src/character.jl:98-101`, which is included *after* `tree.jl`: move them (with `_key_string`, `_term_string`) to the end of `src/terms.jl`, unchanged.
- Produces:
  - `_axis_groups(node::Node, K::_FactorKey) -> Vector{Tuple{Int,_FactorKey}}`: the key's entries grouped by `axis_of(node, mode)`, each group a sorted `_FactorKey`, groups in increasing axis order; `_axis_groups(node, t::Term) = _axis_groups(node, _factor_key(t))`. Used by Tasks 3 and 4 for the operator and for forming.
  - `required_ports(root, routed)`: unchanged signature and return type; a factor of an ancestor term is requested as a single port **only when it is the only factor of that term inside the node**.
  - `required_joint_ports(root::Node, routed::IdDict) -> IdDict{Node,Vector{_FactorKey}}`: per node, in order of first appearance and deduplicated, the sorted keys of the ancestor-term factor groups inside the node that have two or more entries. Nodes with no joint request map to an empty vector.

- [ ] **Step 0: Baseline.** Create branch `composite-ports` off `main`. Run `julia --project=. -e 'using Pkg; Pkg.test()'` and record the wall time in the task report; the default-tier budget is that time + 10 s.

- [ ] **Step 1: Write the failing tests** in `test/test_tree.jl`:

```julia
@testset "joint port requests" begin
    using KronFreeCircuits: required_joint_ports, _axis_groups, Term, Factor, Coefficient, expiphi
    ab = Node([:a => ChargeBasis(3), :b => ChargeBasis(3)]); abc = Node([ab, :c => ChargeBasis(3)])
    root = Node([abc, Node([:d => ChargeBasis(3)])])
    cosine(s) = Term(Coefficient("E", [:E]), [Factor(:a, expiphi(s)), Factor(:b, expiphi(-s)), Factor(:c, expiphi(s)), Factor(:d, expiphi(-s))])
    single = Term(Coefficient("g", [:g]), [Factor(:b, nport()), Factor(:d, nport())])
    r = route(root, [cosine(1), cosine(-1), single])
    @test _axis_groups(root, cosine(1)) == [(1, [(:a, expiphi(1)), (:b, expiphi(-1)), (:c, expiphi(1))]), (2, [(:d, expiphi(-1))])]
    jp = required_joint_ports(root, r)
    @test jp[abc] == [[(:a, expiphi(1)), (:b, expiphi(-1)), (:c, expiphi(1))], [(:a, expiphi(-1)), (:b, expiphi(1)), (:c, expiphi(-1))]]
    @test jp[ab] == [[(:a, expiphi(1)), (:b, expiphi(-1))], [(:a, expiphi(-1)), (:b, expiphi(1))]]
    @test isempty(jp[root]) && isempty(jp[root.children[2]])
    rp = required_ports(root, r)
    @test rp[root.children[2]] == [(:d, expiphi(-1)), (:d, expiphi(1)), (:d, nport())]   # singles are still requested
    @test rp[ab] == [(:b, nport())]                                                     # grouped factors are not
    @test rp[abc] == [(:b, nport())]
end
```

- [ ] **Step 2: Run the test file.** Expected: FAIL, `required_joint_ports` not defined.

- [ ] **Step 3: Implement `_axis_groups`, `required_joint_ports`, and the grouping in `required_ports`** in `src/tree.jl` as specified in Interfaces. Both request functions share one loop over `(ancestor, term)` pairs that calls `_axis_groups(node, t)` restricted to the term's factors with modes inside `node` (write a helper `_inside_groups(node, t)` returning the groups of the factors inside `node`, grouped by the child of `node` they fall in). Update the `required_ports` docstring: "a factor of an ancestor term alone in its child; the groups of two or more are joint ports, see `required_joint_ports`".

- [ ] **Step 4: Run the test file.** Expected: PASS, including the existing `"route and required_ports"` testset.

- [ ] **Step 5: Commit**

```bash
git add src/tree.jl src/terms.jl test/test_tree.jl
git commit -m "required_ports groups an ancestor term's factors by child; joint requests"
```

---

### Task 2: Multi-axis `_port_matrix`

**Files:**
- Modify: `src/materialize.jl:357-363` (`_port_matrix`)
- Test: `test/test_composite.jl` (new file; starts with `using KronFreeCircuits: _port_matrix, DenseOp, ShiftOp, DiagonalOp, dense` as needed)

**Interfaces:**
- Consumes: `apply_axis!`, `LocalOp`, `NodeMatrix`.
- Produces: `_port_matrix(V::NodeMatrix, dims::Vector{Int}, ops::Vector{Tuple{Int,LocalOp}}) -> Matrix`: `V' * (⊗ₐ Oₐ) V` with each `Oₐ` applied along its axis `a` of `reshape(V, dims..., k)` in increasing axis order, alternating two scratch blocks (`apply_axis!` forbids aliasing), one contraction at the end. The existing `_port_matrix(V, dims, axis, op)` becomes `_port_matrix(V, dims, [(axis, op)])`. Axes are distinct (checked with an `ArgumentError` "one operator per axis").

- [ ] **Step 1: Write the failing test** in `test/test_composite.jl`:

```julia
@testset "_port_matrix over several axes equals the Kronecker sandwich" begin
    rng = MersenneTwister(11); dims = [3, 4, 2]
    V = randn(rng, ComplexF64, prod(dims), 5)
    A1 = randn(rng, ComplexF64, 3, 3); A3 = randn(rng, ComplexF64, 2, 2)
    ops = [(1, DenseOp(A1)), (3, DenseOp(A3))]
    P = _port_matrix(V, dims, ops)
    @test P ≈ V' * kron(A3, Matrix(I, 4, 4), A1) * V atol=1e-12 * norm(V)^2 * norm(A1) * norm(A3)
    @test _port_matrix(V, dims, 1, DenseOp(A1)) == _port_matrix(V, dims, [(1, DenseOp(A1))])
    Vr = randn(rng, prod(dims), 5)
    @test _port_matrix(Vr, dims, [(1, DenseOp(real(A1))), (3, DenseOp(real(A3)))]) isa Matrix{Float64}
    @test_throws ArgumentError _port_matrix(V, dims, [(1, DenseOp(A1)), (1, DenseOp(A1))])
end
```

- [ ] **Step 2: Run the test file.** Expected: FAIL, no method for a vector of operators.

- [ ] **Step 3: Implement** the vector method in `src/materialize.jl`; make the four-argument method delegate to it.

- [ ] **Step 4: Run the test file.** Expected: PASS. Also run `test/test_materialize.jl` (slow tier, uses single ports throughout). Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/materialize.jl test/test_composite.jl
git commit -m "_port_matrix applies several axes before one contraction"
```

---

### Task 3: Joint-key character, `joint_ports` storage, and forming a joint port

**Files:**
- Modify: `src/character.jl` (after `_factor_character`; and `_real_expansion`)
- Modify: `src/materialize.jl:8-50` (`MaterializedNode` docstring, struct, convenience constructor), `:278-281` (`_materialize_given`), `:364-391` (after `_node_port`)
- Test: `test/test_character.jl` (unit tests of `_key_character` and the sign-carrying `_real_expansion`), `test/test_composite.jl`

**Interfaces:**
- Consumes: `_factor_character`, `PortCharacter`, `σ`, `_canonical_port`, `_real_expansion`, `realpart`, `imagpart`, `_childop`, `_port_matrix` (Task 2), `_axis_groups` (Task 1).
- Produces:
  - `_key_character(leaves::Dict{Symbol,Any}, K::_FactorKey) -> NamedTuple{(:kind, :partner)}`: `:general` if any factor is; else `:paired` with `partner = _sortkey!([(m, char.partner) ...])` if any factor is paired; else `:real` when the product of the factors' `σ` is `+1`, `:imaginary` when `−1`, with `partner = K`. For a singleton key this agrees with `_factor_character` (`kind` and `[partner]`).
  - `_real_expansion(c::Number, factors::Vector{Tuple{Int,LocalOp,Symbol,Int}})`: the core, taking `(axis, op, kind, s)` with `s` the imaginary-component sign relative to the canonical factor; the existing `Vector{Tuple{Int,LocalOp,PortCharacter}}` method maps each factor to `(a, op, ch.kind, last(_canonical_port(ch)))` and delegates. Behaviour on the old signature is unchanged (existing tests pin it).
  - `MaterializedNode{T}` gains the field `joint_ports::Dict{_FactorKey,NodeMatrix}` after `ports`; the convenience constructor takes it as keyword `joint_ports=Dict{_FactorKey,NodeMatrix}()`; `_materialize_given` passes an empty dictionary. Invariant: every key has length ≥ 2.
  - `_jointop(mn::MaterializedNode, K::_FactorKey) -> DenseOp`: `mn.joint_ports[K]`, or `error("internal error: child node on modes ... lacks the joint port <key string>")`.
  - `_axis_ops(node::Node, children, K::_FactorKey) -> Vector{Tuple{Int,LocalOp,_FactorKey}}`: for each `(a, Ka)` of `_axis_groups(node, K)`: `length(Ka) == 1 ? _childop(children[a], Ka[1]...) : _jointop(children[a], Ka)`, with its sub-key.
  - `_joint_port(node::Node, leaves, children, dims::Vector{Int}, V::NodeMatrix, K::_FactorKey) -> NodeMatrix`: on complex `V`, `_port_matrix(V, dims, [(a, op) ...])` with any real `DenseOp` converted to `ComplexF64` first (as the single-port method does). On real `V` (`Matrix{Float64}`): with `ch = _key_character(leaves, K)`, the real component `R = Σ r.coeff · _port_matrix(V, dims, r.factors)` over `_real_expansion(1.0, fs)` and, unless `ch.kind === :real`, the imaginary component `I` likewise over `_real_expansion(-im, fs)`, with `fs = [(a, op, _key_character(leaves, Ka).kind, 1) ...]`; returns `R` (`Matrix{Float64}`) for `:real`, `complex.(zero(I), I)` for `:imaginary`, `complex.(R, I)` otherwise.

- [ ] **Step 1: Write the failing tests.** In `test/test_character.jl`:

```julia
@testset "_key_character" begin
    using KronFreeCircuits: _key_character, _mode_leaves, expiphi, nport, phiport
    leaves = _mode_leaves(Node([Node([:c => ChargeBasis(3), :e => PhaseGrid(-2.0, 2.0, 8; order=2)]), :h => FockBasis(4; phi_zpf=0.6)]))
    @test _key_character(leaves, [(:c, expiphi(1)), (:c, expiphi(1))]).kind === :real          # charge-basis expiphi is real
    kc = _key_character(leaves, [(:c, expiphi(1)), (:e, expiphi(-1))])
    @test kc.kind === :paired && kc.partner == [(:c, expiphi(1)), (:e, expiphi(1))]
    @test _key_character(leaves, [(:e, nport()), (:h, phiport())]).kind === :imaginary
    @test _key_character(leaves, [(:e, nport()), (:h, nport())]).kind === :real
    @test _key_character(leaves, [(:e, nport()), (:h, expiphi(1))]).kind === :paired
end
```

In `test/test_composite.jl`, a direct forming test on a materialized leaf (no joint request is routed yet, so the port is formed by hand). `ce_json` is an inline two-mode spec at the top of the file: modes `c` (compact) and `e` (extended), `E_C` diagonal, `E_L` on `e` only, one cosine with `omega [1, 1]` and `offset "0"`, parameters `E_C, E_J, E_L`.

```julia
@testset "_joint_port on a leaf equals the primitive sandwich, real and complex" begin
    using KronFreeCircuits: _joint_port, _mode_leaves, dense, localop
    bc, be = ChargeBasis(3), PhaseGrid(-3.0, 3.0, 7; order=2)
    spec = KronFreeCircuits.parse_spec(ce_json); θ = (E_C=1.0, E_J=2.0, E_L=0.5)
    leaf = Node([:c => bc, :e => be]; solver=Dense(), keep=lowest(5))
    sys = materialize(leaf, spec, θ)
    @test arithmetic(sys) === :real
    mn = sys.root; V = eigenvectors(sys)
    sandwich(V, sgn) = V' * kron(dense(localop(be, expiphi(-sgn))), dense(localop(bc, expiphi(sgn)))) * V
    for sgn in (1, -1)
        P = _joint_port(leaf, _mode_leaves(leaf), mn.children, mn.dims, V, [(:c, expiphi(sgn)), (:e, expiphi(-sgn))])
        @test P isa Matrix{ComplexF64}                 # charge expiphi (real) ⊗ grid expiphi (paired): paired
        @test P ≈ sandwich(V, sgn) atol=1e-12
    end
    leafc = Node([:c => bc, :e => be]; solver=Dense(), keep=lowest(5), arithmetic=:complex)
    sysc = materialize(leafc, spec, θ); mc = sysc.root; Vc = eigenvectors(sysc)
    Pc = _joint_port(leafc, _mode_leaves(leafc), mc.children, mc.dims, Vc, [(:c, expiphi(1)), (:e, expiphi(-1))])
    @test Pc isa Matrix{ComplexF64} && Pc ≈ sandwich(Vc, 1) atol=1e-12
end
```

- [ ] **Step 2: Run both test files.** Expected: FAIL, `_key_character` / `_joint_port` not defined.

- [ ] **Step 3: Implement** `_key_character` and the `(axis, op, kind, s)` core of `_real_expansion` in `src/character.jl`; the `joint_ports` field, `_jointop`, `_axis_ops` and `_joint_port` in `src/materialize.jl`. Update the `MaterializedNode` docstring: `joint_ports[key]` is the operator `V† (⊗ Oₐ) V` of a sorted factor list spanning several of the node's modes, in the kept eigenbasis, natural type by its key character; memory `kept²` per key plus two scratch blocks while forming.

- [ ] **Step 4: Run** `test/test_character.jl`, `test/test_composite.jl`, `test/test_given.jl`, `test/test_selection.jl`. Expected: PASS (every hand-built `MaterializedNode` in tests goes through the keyword constructor; fix any positional call the suite reveals).

- [ ] **Step 5: Commit**

```bash
git add src/character.jl src/materialize.jl test/test_character.jl test/test_composite.jl
git commit -m "Joint ports: key character, storage on MaterializedNode, forming on real and complex nodes"
```

---

### Task 4: Wire joint ports through materialization; lift the rejection

**Files:**
- Modify: `src/materialize.jl:137-156` (`_materialize_terms`: compute `needed_joint`), `:227-240` (`_check_composite` → `_check_duplicate_modes`), `:451-480` (`_node_operator`), `:482-514` (`_real_node_terms`), `:560-651` (`_materialize`: form joint ports at step (8), pass to the constructor), `:656-668` (`_check_eltype`), `:716-731` (`port`), the `materialize` docstring paragraph on composite ports
- Modify: `test/test_materialize.jl:194-210`, `test/test_apply.jl:413-429`, `test/test_character.jl:111-114`
- Test: `test/test_composite.jl`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `_check_duplicate_modes(routed)`: throws `ArgumentError("a term on modes ... has several factors on mode :m")` when a term names a mode twice; nothing else is rejected.
  - `_node_operator` and `_real_node_terms` build each routed term's factors with `_axis_ops(node, children, _factor_key(t))` (one local operator per axis). In `_real_node_terms` the merge key becomes `Vector{Tuple{Int,_FactorKey,Bool}}`: a singleton group keeps today's `(a, [(m, canonical port)], inS)` with sign from `_canonical_port`; a joint group is its own canonical key with sign `+1` (conjugate joint pairs are not merged; each term's expansion is a correct real operator on its own).
  - `_materialize` forms `joint_ports[K] = _joint_port(node, leaves, children, dims, V, K)` for `K in needed_joint[node]` after step (8), on the phase-fixed kept `V`, and passes `joint_ports` to the constructor; `_check_eltype(node, real, V, peak_amps, ports, joint_ports, leaves)` checks each joint port's type by `_key_character` (`Float64` exactly for a `:real` key on a real node).
  - `port(sys::System, key::AbstractVector{Tuple{Symbol,Port}}, node::Node=sys.root.node)`: length 1 delegates to the single-port method; otherwise returns `joint_ports[_sortkey!(collect(key))]` or throws `ArgumentError("joint port <key string> was not formed for this node; available: ...")`.

- [ ] **Step 1: Rewrite the three rejection tests.** `test/test_materialize.jl` `"composite ports are rejected"` → `"composite ports: untruncated hierarchy equals the flat spectrum"`: same JSON and tree, child `Node([:a => ChargeBasis(3), :b => ChargeBasis(3)]; solver=Dense())` (keeps all 9), root `Dense()`; assert `energies(sys) ≈ eigvals(Hermitian(Matrix(primitive_hamiltonian(circuit_terms(spec), θ, [:a => ChargeBasis(3), :b => ChargeBasis(3), :c => ChargeBasis(3)]))))` to `1e-10`. `test/test_apply.jl:413-429` → the mixed-basis tree now materializes; assert `energies(sys)` equals the flat `primitive_hamiltonian` eigenvalues to `1e-9` relative for the lowest 6, with the child keeping all states. `test/test_character.jl:111-114`: keep as is (it tests the same-mode duplicate, still rejected with "several factors").

- [ ] **Step 2: Write the failing acceptance tests** in `test/test_composite.jl` (the 3-charge-mode spec `abc_json` from the old rejection test, `θ = (E_C=1.0, E_J=1.0)`):

```julia
@testset "truncated child: joint port is V† A_joint V, separate projection is not" begin
    spec = KronFreeCircuits.parse_spec(abc_json); θ = (E_C=1.0, E_J=1.0)
    ba = ChargeBasis(4); ab = Node([:a => ba, :b => ba]; solver=Dense(), keep=lowest(6)); c = Node([:c => ba]; solver=Dense())
    root = Node([ab, c]; solver=Dense())
    sys = materialize(root, spec, θ)
    Vab = eigenvectors(sys, ab)                        # 81 × 6, phase-fixed kept vectors
    Ea = KronFreeCircuits.dense(KronFreeCircuits.localop(ba, expiphi(1)))
    Ajoint = kron(Ea, Ea)                              # kron(B_b, A_a): axis 1 (a) fastest
    Pj = port(sys, [(:a, expiphi(1)), (:b, expiphi(1))], ab)
    @test Pj ≈ Vab' * Ajoint * Vab atol=1e-12
    @test port(sys, [(:a, expiphi(-1)), (:b, expiphi(-1))], ab) ≈ Pj' atol=1e-12          # adjoint pair
    Psep = (Vab' * kron(I(9), Ea) * Vab) * (Vab' * kron(Ea, I(9)) * Vab)   # ChargeBasis(4) has 9 states
    @test norm(Psep - Pj) > 1e-3 * norm(Pj)            # the separate projection demonstrably differs
    # the root's spectrum is that of the independently projected flat Hamiltonian
    Hflat = primitive_hamiltonian(circuit_terms(spec), θ, [:a => ba, :b => ba, :c => ba])
    P = kron(_primitive(KronFreeCircuits.materialized(sys, c)), _primitive(KronFreeCircuits.materialized(sys, ab)))
    Hred = Hermitian(P' * Hflat * P)
    @test norm(Hred - Hred') <= 1e-12 * norm(Hred)
    @test energies(sys) ≈ eigvals(Hred) atol=1e-10
    @test_throws ArgumentError port(sys, [(:a, expiphi(2)), (:b, expiphi(2))], ab)
    @test port(sys, [(:a, expiphi(1))], ab) == port(sys, :a, expiphi(1), ab)             # a singleton key delegates to the (mode, Port) method
end

@testset "real and complex arithmetic agree on a joint-port tree" begin
    spec = KronFreeCircuits.parse_spec(abc_json); θ = (E_C=1.0, E_J=1.0); ba = ChargeBasis(4)
    mk(arith) = Node([Node([:a => ba, :b => ba]; solver=Dense(), keep=lowest(6), arithmetic=arith), Node([:c => ba]; solver=Dense())]; solver=Dense())
    sr, sc = materialize(mk(:auto), spec, θ), materialize(mk(:complex), spec, θ)
    @test arithmetic(sr, sr.root.node.children[1]) === :real && arithmetic(sc, sc.root.node.children[1]) === :complex
    @test port(sr, [(:a, expiphi(1)), (:b, expiphi(1))], sr.root.node.children[1]) isa Matrix{Float64}     # charge-basis expiphi: real character
    @test energies(sr) ≈ energies(sc) atol=1e-10
end
```

- [ ] **Step 3: Run the test file.** Expected: FAIL on "composite (multi-mode) ports are not yet supported".

- [ ] **Step 4: Implement** the wiring listed in Interfaces. In `_materialize_terms`, `needed_joint = required_joint_ports(root, routed)` is threaded to `_materialize` alongside `needed`. Replace the `materialize` docstring's last paragraph: a term with several factors inside one child is applied through that child's joint port (`joint_ports`), formed as `V† (⊗ Oₐ) V` before projection; two factors on one mode still throw.

- [ ] **Step 5: Run** `test/test_composite.jl`, `test/test_materialize.jl`, `test/test_apply.jl`, `test/test_character.jl`, `test/test_tree.jl`. Expected: PASS.

- [ ] **Step 6: Run the fast tier** (`Pkg.test()`) and compare the wall time to the Task 1 baseline. Expected: PASS within +10 s.

- [ ] **Step 7: Commit**

```bash
git add src/materialize.jl test/test_composite.jl test/test_materialize.jl test/test_apply.jl test/test_character.jl
git commit -m "Composite ports: a term's factors inside one child are applied through a joint port"
```

---

### Task 5: Mixed bases, both signs, static offsets, nesting, solver agreement

**Files:**
- Test: `test/test_composite.jl`, `test/test_solvers.jl`

**Interfaces:**
- Consumes: Task 4. No new production code expected; any defect found is fixed in the file that owns it and noted in the task report.

- [ ] **Step 1: Write the tests** in `test/test_composite.jl`:

```julia
@testset "mixed bases, both KITE signs, π and generic offsets" begin
    # ceh_json: modes c (compact), e (extended), h (harmonic); E_L on e and h; one cosine with `omega` and `offset` substituted
    for (omega, offset) in (([1, 1, -1], "pi"), ([1, -1, -1], "pi"), ([1, 1, -1], "0.37*pi"), ([1, -1, 1], "0"))
        spec = KronFreeCircuits.parse_spec(ceh_json(omega, offset)); θ = (E_C=1.0, E_J=2.0, E_L=0.5)
        bc, be, bh = ChargeBasis(3), PhaseGrid(-4.0, 4.0, 9; order=2), FockBasis(4; phi_zpf=0.6)
        for keepn in (0, 8)                                   # untruncated, then genuinely truncated
            ce = Node([:c => bc, :e => be]; solver=Dense(), keep=lowest(keepn)); root = Node([ce, Node([:h => bh]; solver=Dense())]; solver=Dense())
            sys = materialize(root, spec, θ)
            Hflat = primitive_hamiltonian(circuit_terms(spec), θ, [:c => bc, :e => be, :h => bh])
            P = kron(_primitive(KronFreeCircuits.materialized(sys, root.children[2])), _primitive(KronFreeCircuits.materialized(sys, ce)))
            @test energies(sys) ≈ eigvals(Hermitian(P' * Hflat * P)) atol=1e-9
            K = [(:c, expiphi(omega[1])), (:e, expiphi(omega[2]))]
            Vce = eigenvectors(sys, ce)
            @test port(sys, K, ce) ≈ Vce' * kron(dense(localop(be, expiphi(omega[2]))), dense(localop(bc, expiphi(omega[1])))) * Vce atol=1e-11
            @test port(sys, [(:c, expiphi(-omega[1])), (:e, expiphi(-omega[2]))], ce) ≈ port(sys, K, ce)' atol=1e-11
        end
    end
    # a PeriodicGrid compact mode inside the joint group (Review Focus 4): untruncated hierarchy equals the flat leaf,
    # and the truncated joint port equals the sandwich of the grid's own expiphi operator
    spec = KronFreeCircuits.parse_spec(ceh_json([1, 1, -1], "pi")); θ = (E_C=1.0, E_J=2.0, E_L=0.5)
    bp, be, bh = PeriodicGrid(8; order=2), PhaseGrid(-4.0, 4.0, 9; order=2), FockBasis(4; phi_zpf=0.6)
    flat = materialize(Node([:c => bp, :e => be, :h => bh]; solver=Dense()), spec, θ)
    full = materialize(Node([Node([:c => bp, :e => be]; solver=Dense()), Node([:h => bh]; solver=Dense())]; solver=Dense()), spec, θ)
    @test energies(full) ≈ energies(flat) atol=1e-9
    ce = Node([:c => bp, :e => be]; solver=Dense(), keep=lowest(8))
    sys = materialize(Node([ce, Node([:h => bh]; solver=Dense())]; solver=Dense()), spec, θ)
    Vce = eigenvectors(sys, ce)
    @test port(sys, [(:c, expiphi(1)), (:e, expiphi(1))], ce) ≈ Vce' * kron(dense(localop(be, expiphi(1))), dense(localop(bp, expiphi(1)))) * Vce atol=1e-11
end

@testset "nested joint ports: a group inside a grandchild" begin
    # abcd_json: four compact modes, one cosine with omega [1, -1, 1, -1], offset "pi"
    spec = KronFreeCircuits.parse_spec(abcd_json); θ = (E_C=1.0, E_J=1.0); b = ChargeBasis(3)
    ab = Node([:a => b, :b => b]; solver=Dense(), keep=lowest(5)); abc = Node([ab, :c => b]; solver=Dense(), keep=lowest(7))
    root = Node([abc, Node([:d => b]; solver=Dense())]; solver=Dense())
    sys = materialize(root, spec, θ)
    @test haskey(KronFreeCircuits.materialized(sys, ab).joint_ports, [(:a, expiphi(1)), (:b, expiphi(-1))])
    @test haskey(KronFreeCircuits.materialized(sys, abc).joint_ports, [(:a, expiphi(1)), (:b, expiphi(-1)), (:c, expiphi(1))])
    E1 = dense(localop(b, expiphi(1))); Em = dense(localop(b, expiphi(-1)))
    Pabc = _primitive(KronFreeCircuits.materialized(sys, abc))          # 27 × 7 primitive-space vectors
    @test port(sys, [(:a, expiphi(1)), (:b, expiphi(-1)), (:c, expiphi(1))], abc) ≈ Pabc' * kron(E1, Em, E1) * Pabc atol=1e-11
    Hflat = primitive_hamiltonian(circuit_terms(spec), θ, [:a => b, :b => b, :c => b, :d => b])
    P = kron(_primitive(KronFreeCircuits.materialized(sys, root.children[2])), Pabc)
    @test energies(sys) ≈ eigvals(Hermitian(P' * Hflat * P)) atol=1e-10
    # untruncated nesting equals the flat spectrum
    full = Node([Node([Node([:a => b, :b => b]; solver=Dense()), :c => b]; solver=Dense()), Node([:d => b]; solver=Dense())]; solver=Dense())
    @test energies(materialize(full, spec, θ)) ≈ eigvals(Hermitian(Matrix(Hflat))) atol=1e-10
end
```

`ceh_json(omega, offset)` and `abcd_json` are string-building helpers at the top of the file, in the style of the existing inline specs (`E_C` diagonal, `E_L` on the `e`/`h` modes only, `flux_linear` and `n_g` zero).

- [ ] **Step 2: Run the test file.** Expected: PASS (Task 4 implemented nesting through `_axis_ops`); any failure is a defect to fix with `superpowers:systematic-debugging`, not a tolerance to loosen.

- [ ] **Step 3: Write the solver-agreement test** in `test/test_solvers.jl` (slow tier): the Task 4 `abc_json` tree with `ChargeBasis(8)` leaves, child `keep=lowest(12)`, root `compute=8` under `Dense()`, `LOBPCG()` and `Lanczos()`; assert the eight energies agree pairwise to `1e-8` and that `operator(sys, root)` applied to a random block equals `Matrix(operator(sys, root))` times it to `1e-10` relative.

- [ ] **Step 4: Run** `test/test_solvers.jl`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/test_composite.jl test/test_solvers.jl
git commit -m "Composite ports: mixed bases, both signs, offsets, nesting and solver agreement tests"
```

---

### Task 6: Documentation and the Z4 pilot rerun

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` (new `### 20i. Composite ports (2026-10-26)` after §20h; strike the §20a composite-debt sentence by a forward reference), `docs/superpowers/STATUS.md` (plans done; the §20a debt line under "Proposed by Eli"), `README.md` (one sentence where ports are introduced: a term with several factors in one child uses a joint port)
- Modify: `src/materialize.jl` docstrings if any still say composite ports throw (grep `not yet supported`, `composite`).

- [ ] **Step 1: Write §20i** (10–15 lines): the joint-port key, where it is requested (`required_joint_ports`), how it is formed (`_joint_port`: product before projection, two scratch blocks, dense `kept²` result), the real-arithmetic components (character product; conjugate joint pairs not merged), what is still rejected (two factors on one mode), and the not-yet-supported user-requested joint ports in `Node(ports=...)` and selection rules (recorded as a later item for the matrix-element query plan).

- [ ] **Step 2: Update STATUS.md and README.md** as listed; `grep -rn "not yet supported\|composite" src README.md docs/superpowers/STATUS.md` must show no stale claim that composite ports throw.

- [ ] **Step 3: Run the full suite** (`KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'`). Expected: PASS; record the time.

- [ ] **Step 4: Commit**

```bash
git add docs README.md src/materialize.jl
git commit -m "Composite ports: spec §20i, status and README"
```

- [ ] **Step 5: Z4 pilot rerun (Eli, on ewdesktop; reported, not asserted).** Pull `composite-ports` into the frozen source checkout the pilot uses (or point the pilot environment at this branch), then run, with a fresh result path, the command from the requirements document:

```bash
julia +1.13.1 --project=data/harmonium_2d/z4_threeway/pilot_v1/sources/env -t 1 data/harmonium_2d/z4_threeway/pilot_v1/sources/kronfree_harmonium_layouts.jl data/harmonium_2d/z4_threeway/pilot_v1/export/circuit.json data/harmonium_2d/z4_threeway/pilot_v1/settings.json /tmp/z4-threeway-composite-v1.json
```

Expected: the hierarchy `[[1,4,5],[2,6,7],[3]]` materializes with no `ArgumentError`; the result file holds eigenvalues; the root operator is Hermitian to roundoff (`norm(M - M') / norm(M)` on `Matrix(operator(sys))` at the pilot's small retention). Record the lowest eight energies and the run time in the pilot's `run.json` next to the scqubits diagnostic; the convergence sweep (retentions, then primitive cutoffs; requirements document, last acceptance item) is a separate study and not part of this plan.
