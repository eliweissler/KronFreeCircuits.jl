# Plan 3: Real Arithmetic by Port Character Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Implementation happens in the worktree `.claude/worktrees/plan-3-real-arithmetic` on branch `worktree-plan-3-real-arithmetic`, never on `main`.

**Goal:** Nodes whose structure guarantees a real Hamiltonian materialize entirely in `Float64` (operator, solver, eigenvectors, real-character ports), with energies equal to the complex path, on the host and on the device.

**Architecture:** Every basis declares a conjugation character for each port (real, imaginary, paired with a partner port, or general). Once per `materialize`, after routing, each node is classified real or complex by a θ-independent identity test on its term groups (coefficients compared as functions at fixed pseudo-random parameter points). A real node's operator is built from the real and imaginary components of its factors by expanding `Re(c ⊗ O)`; the local operators and `KronFreeOperator` become element-type generic, solvers run in `eltype(H)`, and the materialized node stores `Float64` vectors and natural-type ports, so the characters propagate to the parent through real eigenvector projection.

**Tech Stack:** Julia 1.13, LinearAlgebra/SparseArrays, KernelAbstractions, ITensors (cuTENSOR on device), KrylovKit 0.10.4, LOBPCGEigensolver 0.1.4, CUDA extension. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`, Section 20f (with 20c decision 7 and 20e). Executors read 20f in full before their task.

## Global Constraints

- Real arithmetic only when the structure guarantees it, decided once per materialization from the term list and the tree, never from the values in θ and never from small or exactly zero imaginary parts of evaluated data (§20f "The rule").
- The identity test evaluates coefficients at three fixed pseudo-random parameter points, each parameter drawn in [0.5, 2] from a fixed seed, never at the user's θ; agreement to 1e-12 relative to the larger modulus; a coefficient that throws at a test point fails the test (§20f "The node test").
- A node is real only when every node below it is real; when any entry of θ is not of a `Real` Julia type, every node of that materialization is complex (§20f).
- Complex nodes are unchanged: `ComplexF64` operator, vectors and ports, and every `DenseOp` factor converted to `ComplexF64` (no mixed real–complex gemm).
- Outputs in native types (Eli, 2026-10-18): a real node's `eigenvectors` are `Matrix{Float64}`; a real-character port of a real node is `Matrix{Float64}`; imaginary and paired ports are `Matrix{ComplexF64}`; documented in the README.
- `ShiftInvert` follows `eltype(H)` only; `isreal_assembled` is removed from the solver and the exports (Eli, 2026-10-18).
- Column-major, axis 1 fastest; apply dispatch on structure, never on device; the `KronFreeOperator` is the only apply. Solver and backend code carries no literal element type.
- Every eigensolver result is validated by true residuals against the package operator; a solver change is reviewed with adversarial completeness probes compared against `Dense` (degenerate levels, invariant-seed traps, near crossings), not residual checks alone (memory note `sdd-review-silent-wrong-answers`).
- Existing tests keep passing; a test may change only where it asserts a `ComplexF64` type or `isreal_assembled`, and each such change is listed in the task report with the reason.
- Default suite: at most +15 s over the baseline of 142.4 s wall measured on this branch before Task 1 (`Pkg.test()` on the Mac); slower checks are gated behind `KRONFREE_SLOW_TESTS=1`. Device tests live in `test/cuda/` and run on `ssh ewdesktop` by `rsync --delete` into the disposable `~/kronfree-plan2-ci/` only (recipe: `docs/spikes/2026-10-17-plan2-benchmark/results.md`, "Preparing `ewdesktop`").
- Commits end with a `Co-Authored-By:` line naming the model that wrote them (`Claude Opus 5.5 <noreply@anthropic.com>` or `Claude Sonnet 5.5 <noreply@anthropic.com>`).

## Review Focus

1. A circuit that is complex by structure but whose coefficients are exactly real at the given θ (a charge-basis SQUID cosine with offset `varphi_ext` at `varphi_ext = 0`; a `PeriodicGrid` mode with a parameter `n_g = 0.5`) must materialize complex. Pinned in Task 3.
2. A node over a child forced complex (`arithmetic=:complex`) must be complex even when none of its terms touches that child, and must materialize to the same energies as the all-complex tree. Pinned in Tasks 3 and 7.
3. A θ with a `ComplexF64` entry (a drive `Ω·a + conj(Ω)·a†` on a `GivenSystem` with real `:a`, `:adag` operators) must run complex and give the Hermitian spectrum, never silently drop the imaginary part. Pinned in Task 3.
4. `previous=` from a materialization in the other arithmetic (the user toggled `arithmetic` between calls) must not throw: a complex warm start into a real operator is dropped, a real one into a complex operator converted, `follow` overlaps promoted. Pinned in Task 7.
5. A port of general character requested on a real node (a complex `GivenSystem` operator in `ports=`, or a `connected` selection on a grid `expiphi` port) must come back complex and equal to the complex path's. Pinned in Task 6.

---

### Task 1: Element-type-generic local operators and `KronFreeOperator`

**Files:**
- Modify: `src/localops.jl` (`DenseOp{T}`; `diagonal`, `dense`, `sparse` keep the operator's element type; `realpart`, `imagpart`), `src/apply.jl` (`AxisTerm{T}`, `_OperatorTerm{T}`, element-type promotion in `KronFreeOperator`, `DenseOp` data converted to `eltype(H)`, `Matrix(H)`, mixed real-operator/complex-vector application)
- Test: `test/test_apply.jl`, `test/test_localops.jl`

**Interfaces:**
- Consumes: the current `LocalOp` types, `KronFreeOperator`, `kron_reference`, `three_oscillators` (test/helpers.jl).
- Produces:
  - `DenseOp{T<:Number}` with field `M::Matrix{T}`; `DenseOp(A::AbstractMatrix{T})` keeps `T` (float of an integer `T`). `eltype(op::LocalOp)` is the data's element type for all four types.
  - `diagonal(op)`, `dense(op)`, `sparse(op)` return `eltype(op)` data.
  - `realpart(op::LocalOp)` and `imagpart(op::LocalOp)`: the same structure type with `Float64` data, `real.` / `imag.` of the data (for `ShiftOp`, of `factor`), so `dense(realpart(op)) + im * dense(imagpart(op)) == dense(op)` exactly. Both are defined for real data (`imagpart` then has zero data).
  - `AxisTerm{T}(coeff::T, factors)`; `AxisTerm(coeff::Number, factors)` keeps `typeof(coeff)` (no conversion to `ComplexF64`).
  - `KronFreeOperator(dims, terms; constant=0.0, …)` has `eltype` = `promote_type` of every term's coefficient type, every factor's `eltype`, and `typeof(constant)`; `DenseOp` factor data is stored converted to that type (`_factor_data`), banded/diagonal data as is.
  - `mul!(y, H, x)` with `eltype(H) <: Real` and complex `x`, `y`: applies `H` to `real(x)` and `imag(x)` through two real temporaries (allocating; documented as a convenience path that solvers never take). `H * x` allocates its result with `promote_type(eltype(H), eltype(x))`.
  - `Matrix(H)` returns `Matrix{eltype(H)}`.

- [ ] **Step 0: Baseline.** The controller measured the default suite on this branch before any change: 142.4 s wall (`Pkg.test()`, Mac M2); the budget is 157 s.

- [ ] **Step 1: Write the failing tests** (`test/test_localops.jl`, `test/test_apply.jl`):

```julia
@testset "element-type-generic local operators" begin
    rng = MersenneTwister(301)
    A = randn(rng, 5, 5)
    @test DenseOp(A).M isa Matrix{Float64} && eltype(DenseOp(A)) === Float64
    @test eltype(DenseOp(complex.(A))) === ComplexF64
    for op in (DenseOp(randn(rng, ComplexF64, 5, 5)), DiagonalOp(randn(rng, ComplexF64, 5)),
               BandedOp(5, [-1, 2], [randn(rng, ComplexF64, 5), randn(rng, ComplexF64, 5)]), ShiftOp(5, 2, 0.3 - 0.7im))
        @test eltype(realpart(op)) === Float64 && eltype(imagpart(op)) === Float64
        @test dense(realpart(op)) + im * dense(imagpart(op)) == dense(op)
        @test typeof(realpart(op)).name === typeof(op).name          # same structure type
    end
    @test eltype(dense(BandedOp(4, [1], [ones(4)]))) === Float64 && eltype(diagonal(DiagonalOp(ones(3)))) === Float64
end

@testset "KronFreeOperator element type follows its data" begin
    rng = MersenneTwister(302)
    terms = [AxisTerm(1.0, [(1, DiagonalOp(collect(0.0:3)))]), AxisTerm(0.5, [(1, BandedOp(4, [1, -1], [ones(4), ones(4)])), (2, DenseOp(randn(rng, 3, 3) |> x -> x + x'))])]
    H = KronFreeOperator([4, 3], terms)
    @test eltype(H) === Float64 && eltype(Matrix(H)) === Float64
    D, B, C = dense(terms[1].factors[1][2]), dense(terms[2].factors[1][2]), dense(terms[2].factors[2][2])
    @test Matrix(H) ≈ kron(Matrix(1.0I, 3, 3), D) + 0.5 * kron(C, B) atol=1e-14          # axis 1 fastest
    Hc = KronFreeOperator([4, 3], [AxisTerm(ComplexF64(t.coeff), t.factors) for t in terms])
    @test eltype(Hc) === ComplexF64 && Matrix(Hc) ≈ Matrix(H) atol=1e-14
    @test all(t -> all(f -> !(f[2] isa DenseOp) || eltype(f[3]) === ComplexF64, t.factors), Hc.terms)   # dense data converted
    X = randn(rng, ComplexF64, 12, 3)
    @test H * X ≈ Matrix(H) * X atol=1e-13                                                              # mixed convenience path
    Y = similar(X); mul!(Y, H, X); @test Y ≈ Matrix(H) * X atol=1e-13
    @test eltype(three_oscillators(4)) === ComplexF64                                                   # complex data stays complex
end
```

- [ ] **Step 2: Run** `julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_localops.jl", "test_apply.jl"])'` if the runner supports file selection, else the whole default suite; expect failures on `realpart`, `DenseOp{Float64}` and `eltype(H) === Float64`.

- [ ] **Step 3: Implement** the Interfaces above. `_add_term_diagonal!`, `_device_diagonal`, `sparse(H)`, `gershgorin_bound`, `separable_eigenbases`, `product_seed` already take `T = eltype(H)`; keep them so. ITensor contractions in `_apply_dense!` take the stored data's element type unchanged.

- [ ] **Step 4: Run the default suite.** Every existing test passes; any test changed because it asserted `ComplexF64` is listed in the report.

- [ ] **Step 5: Commit** (`Element-type-generic local operators and KronFreeOperator`).

---

### Task 2: Port characters per basis, and exact Fock phases

**Files:**
- Create: `src/character.jl` (included after `given.jl`)
- Modify: `src/bases.jl` (`displacement` with exact phases), `src/KronFreeCircuits.jl` (include; export `PortCharacter`, `character`)
- Test: `test/test_character.jl` (new)

**Interfaces:**
- Consumes: `Basis` types, `localop`, `Port`, `GivenSystem`, `dense` (Task 1).
- Produces:
  - `struct PortCharacter; kind::Symbol; partner::Port; end` with `kind ∈ (:real, :imaginary, :paired, :general)`; `σ(c::PortCharacter)::Int` is `-1` for `:imaginary`, else `+1`; `partner` is the port itself except for `:paired`.
  - `character(b::Basis, p::Port)::PortCharacter` per the table of §20f (ChargeBasis, PhaseGrid, PeriodicGrid, FockBasis); throws the same `ArgumentError` as `localop` for a port the basis does not serve.
  - `character(g::GivenSystem, p::Port)::PortCharacter` read exactly from `g.operators[p.kind]`: `:real` when every entry's imaginary part is zero, `:imaginary` when every real part is zero (and the matrix is not all zero), else `:general`.
  - `displacement(N, α)` computes the phase factor `cis(a·θ)` exactly as `(±i)^a` or `(±1)^a` when `α` is purely imaginary or purely real (`a` the level difference), and as today otherwise.

- [ ] **Step 1: Write the failing test** `test/test_character.jl`:

```julia
@testset "port characters match the operator data exactly" begin
    bases = [ChargeBasis(4), PhaseGrid(-4π, 4π, 41; order=6), PeriodicGrid(12; order=4), FockBasis(9; phi_zpf=0.7)]
    ports = [nport(), n2port(), phiport(), phi2port(), expiphi(1), expiphi(-2), exp2pin(1), exp2pin(-1),
             Port(:d1, 0), Port(:d2, 0), Port(:d1wrap, 1), Port(:d1wrap, -1), Port(:d2wrap, 1), Port(:d2wrap, -1)]
    nchecked = 0
    for b in bases, p in ports
        op = try localop(b, p) catch e; e isa ArgumentError || rethrow(); continue end
        c = character(b, p)
        M = dense(op)
        if c.kind === :real
            @test all(iszero, imag(M))
        elseif c.kind === :imaginary
            @test all(iszero, real(M)) && !all(iszero, M)
        else
            @test c.kind === :paired && c.partner != p && conj(M) == dense(localop(b, c.partner))
        end
        nchecked += 1
    end
    @test nchecked == 30          # served pairs: ChargeBasis 6, PhaseGrid 8 (2π/Δ = 10, so exp2pin(±1) is served), PeriodicGrid 8, FockBasis 8
    @test character(PhaseGrid(-4π, 4π, 41; order=6), expiphi(2)).partner == expiphi(-2)
    @test character(ChargeBasis(3), expiphi(1)).kind === :real && character(ChargeBasis(3), exp2pin(1)).partner == exp2pin(-1)
    @test character(FockBasis(9; phi_zpf=0.7), exp2pin(1)).kind === :real
    @test_throws ArgumentError character(PeriodicGrid(12; order=4), nport())
    g = GivenSystem(:r; energies=[0.0, 1.0, 2.0], operators=Dict(:a => [0 1 0; 0 0 1.4; 0 0 0], :y => im * [0 1 0; -1 0 0; 0 0 0], :z => [1 1im 0; -1im 1 0; 0 0 1]))
    @test character(g, Port(:a, 0)).kind === :real && character(g, Port(:y, 0)).kind === :imaginary && character(g, Port(:z, 0)).kind === :general
end
```

(If `nchecked` differs, the implementer counts the served (basis, port) pairs from `localop` and reports the number; the test pins that count.)

- [ ] **Step 2: Run** the suite; expect `character` undefined and the Fock exactness assertions to fail on today's `displacement`.

- [ ] **Step 3: Implement** `src/character.jl` and the exact `displacement` phases. Check `test/test_bases.jl` Fock tests still pass (values move by round-off only).

- [ ] **Step 4: Run the default suite**; all pass.

- [ ] **Step 5: Commit** (`Port characters per basis; exact Fock displacement phases`).

---

### Task 3: Node classification and the `arithmetic` option

**Files:**
- Modify: `src/character.jl` (classification), `src/tree.jl` (`Node` field `arithmetic`), `src/materialize.jl` (classify once after routing; store the result for `_materialize` and `operator`), `src/KronFreeCircuits.jl` (export `arithmetic`)
- Test: `test/test_character.jl`

**Interfaces:**
- Consumes: `character` (Task 2), `route`, `_tree_bases`, `Coefficient`, `Term`, `GivenSystem`.
- Produces:
  - `Node(children; …, arithmetic::Symbol=:auto)` accepting `:auto`, `:complex`, `:real` (else `ArgumentError`); stored as `node.arithmetic`.
  - `classify(root::Node, routed::IdDict, constant::Coefficient, θ::NamedTuple)::IdDict{Node,Bool}` (internal; `true` = real): bottom-up; a node is real iff its `arithmetic` is not `:complex`, every node below it is real, every factor of its routed terms has a non-`:general` character, and for every factor key `K` (sorted `(mode, port)` pairs) the summed coefficient function `C_K` satisfies `C_{π(K)}(θr) ≈ σ_K conj(C_K(θr))` at the three test points (§20f values); the constant joins the root's empty key. A factor's character: on a basis child, `character(basis, port)`; on a `GivenSystem` child, `character(g, port)`; on a node child, the character of that port in the basis of the leaf holding the mode. Every node is complex when any value of θ is not `<: Real`.
  - The test points: `_TEST_POINTS::Vector{NamedTuple}`-like values built per call from the union of parameters the coefficients depend on, drawn from `MersenneTwister(0x6b66)` uniformly in [0.5, 2]; the same three points for every node.
  - `:real` on a node that tests complex throws `ArgumentError` naming the node's modes and the first term (its factors and coefficient expression) whose conjugate image is missing; on `:auto` that term is logged with `@debug`.
  - `arithmetic(sys::System, node=sys.root.node)::Symbol` returns `:real` or `:complex` for a materialized node (from its stored vectors' element type, Task 6).

- [ ] **Step 1: Write the failing tests** in `test/test_character.jl` (classification only; use `KronFreeCircuits.classify` with `route` and the spec's terms, so nothing needs to materialize real yet):

```julia
classified(root, spec, θ) = (b = KronFreeCircuits._tree_bases(root); terms = circuit_terms(spec, b);
                             r = KronFreeCircuits.route(root, terms); KronFreeCircuits.classify(root, r, constant_term(spec, b), θ))

@testset "structural classification of the fixtures" begin
    zp = load_spec(fixture("zero_pi.json")); θzp = zero_pi_params()
    leaf = Node([:theta => ChargeBasis(4), :phi => PhaseGrid(-6π, 6π, 41; order=6)]; solver=Dense(), keep=lowest(4))
    @test classified(leaf, zp, θzp)[leaf]                                                   # real at flux 0.23, n_g 0.1
    @test classified(leaf, zp, (θzp..., varphi_ext=0.0, n_g=0.0))[leaf]                      # same answer at every θ
    tz = load_spec(fixture("two_zero_pi.json")); θtz = two_zero_pi_params(g=0.05)
    q1 = Node([:theta1 => ChargeBasis(3), :phi1 => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4))
    q2 = Node([:theta2 => ChargeBasis(3), :phi2 => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4))
    top = Node([q1, q2]; keep=lowest(4))
    c = classified(top, tz, θtz); @test c[top] && c[q1] && c[q2]
    fz = load_spec(fixture("full_zero_pi.json"))     # dE_J, dC_J, dC are parameters: complex by structure at every θ
    zpn = Node([:theta => ChargeBasis(3), :phi => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4))
    ftop = Node([zpn, :zeta => FockBasis(4; phi_zpf=0.5)]; keep=lowest(4))
    fθ = merge(full_zero_pi_params(), (dE_J=0.0, dC_J=0.0, dC=0.0, dE_L=0.0))                # symmetric values, same answer
    cf = classified(ftop, fz, fθ); @test !cf[zpn] && !cf[ftop]
    tr = load_spec(fixture("transmon.json")); θtr = (E_J=20.0, E_C=1.0, n_g=0.3)
    @test classified(Node([:varphi_1 => ChargeBasis(10)]), tr, θtr) |> d -> all(values(d))
    pg = Node([:varphi_1 => PeriodicGrid(16; order=4)])
    @test !classified(pg, tr, θtr)[pg]
    @test !classified(pg, tr, (θtr..., n_g=0.5))[pg]                                         # Review Focus 1: exact value, still complex
    lit = parse_spec(replace(read(fixture("transmon.json"), String), "\"n_g\": [\"n_g\"]" => "\"n_g\": [\"0.5\"]"))
    @test classified(pg, lit, (E_J=20.0, E_C=1.0))[pg]                                       # a spec literal is structure
    @test classified(Node([:varphi_1 => FockBasis(12; phi_zpf=0.6)]), load_spec(fixture("fluxonium.json")), (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π)) |> d -> all(values(d))
end

@testset "classification: forced complex, descendants, complex θ, :real assertion" begin
    tz = load_spec(fixture("two_zero_pi.json")); θtz = two_zero_pi_params(g=0.05)
    q1 = Node([:theta1 => ChargeBasis(3), :phi1 => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4), arithmetic=:complex)
    q2 = Node([:theta2 => ChargeBasis(3), :phi2 => PhaseGrid(-6π, 6π, 31; order=6)]; keep=lowest(4))
    top = Node([q1, q2]; keep=lowest(4))
    c = classified(top, tz, θtz); @test !c[q1] && c[q2] && !c[top]                         # Review Focus 2
    @test all(!, values(classified(top, tz, (θtz..., g=0.05 + 0im))))                       # Review Focus 3 (complex-typed θ)
    @test_throws ArgumentError Node([:x => ChargeBasis(2)]; arithmetic=:fast)
    pg = Node([:varphi_1 => PeriodicGrid(16; order=4)]; arithmetic=:real)
    err = try classified(pg, load_spec(fixture("transmon.json")), (E_J=20.0, E_C=1.0, n_g=0.3)); nothing catch e; e end
    @test err isa ArgumentError && occursin("d2wrap", err.msg)
    # a SQUID-like cosine with a flux offset on a charge mode: complex even at zero flux (Review Focus 1)
    sq = parse_spec("""{"format":"kronfree-spec","version":1,"units":"GHz","parameters":["E_J","E_C","x"],
        "modes":[{"name":"q","type":"c"}],"E_C":[["E_C"]],"E_L":[["0"]],"flux_linear":["0"],"n_g":["0"],
        "cosines":[{"energy":"E_J","omega":[1],"offset":"x"}],"constant":"0"}""")
    nq = Node([:q => ChargeBasis(5)])
    @test !classified(nq, sq, (E_J=1.0, E_C=0.2, x=0.0))[nq]
    # a complex drive on a given system with real operators (Review Focus 3, value side): real when θ is real-typed
    g = GivenSystem(:r; energies=[0.0, 1.0, 2.0], operators=Dict(:a => diagm(1 => [1.0, sqrt(2)]), :adag => diagm(-1 => [1.0, sqrt(2)])))
    drive = [Term(Coefficient("W", [:W]), [Factor(:r, Port(:a, 0))]), Term(Coefficient("conj(W)", [:W]), [Factor(:r, Port(:adag, 0))])]
    root = Node([Node([:q => ChargeBasis(3)]; keep=lowest(2)), g]; keep=lowest(3))
    terms = vcat(circuit_terms(sq), drive)
    r = KronFreeCircuits.route(root, terms)
    @test KronFreeCircuits.classify(root, r, Coefficient(0), (E_J=1.0, E_C=0.2, x=0.0, W=0.3 + 0.2im)) |> d -> !d[root]
end
```

(`zero_pi_params`, `two_zero_pi_params` and `full_zero_pi_params` are in test/helpers.jl. Add the real-typed drive case too: with `W = 0.3` the images match, since the coefficient `conj(W)` is the conjugate of `W` as a function, so the root classifies real.)

- [ ] **Step 2: Run** the suite; expect `classify` and the `arithmetic` keyword undefined.

- [ ] **Step 3: Implement** `classify`, the `Node` field, and the call in `materialize(root, terms, θ; …)` right after `route` (the result is passed down to `_materialize`; until Task 4 it is unused there). Error messages name the node's modes.

- [ ] **Step 4: Run the default suite**; all pass.

- [ ] **Step 5: Commit** (`Structural node classification and the arithmetic option`).

---

### Task 4: The real node operator

**Files:**
- Modify: `src/character.jl` (the expansion), `src/materialize.jl` (`_node_operator` builds real terms for a real node; `operator(sys, node)` reproduces it)
- Test: `test/test_character.jl`

**Interfaces:**
- Consumes: `classify` (Task 3), `realpart`/`imagpart` and generic `AxisTerm`/`KronFreeOperator` (Task 1), `character` (Task 2).
- Produces:
  - `_real_axisterms(c::Number, factors::Vector{Tuple{Int,LocalOp,PortCharacter}})::Vector{AxisTerm{Float64}}`: the expansion of §20f, `Re(c · i^{|S|}) ⊗_{a∈S} imagpart ⊗_{a∉S} realpart` over the subsets `S` of the factors whose character is `:imaginary` or `:paired` (an `:imaginary` factor is always in `S`, a `:real` one never), terms with a zero coefficient dropped.
  - Conjugate pairs merged: each paired port is expressed through its orbit's canonical port (the one with the larger `k`, or the port itself), using `realpart(O_{π(p)}) = realpart(O_p)` and `imagpart(O_{π(p)}) = −imagpart(O_p)` (`σ = +1` for paired), and real terms with identical `(axis, port, component)` keys are summed.
  - `_node_operator(node, solver, children, terms, θ, constant; real::Bool)`: for a real node, `KronFreeOperator` with `eltype === Float64` built from `_real_axisterms` and `real(constant)`; for a complex node, as today.

- [ ] **Step 1: Write the failing tests:**

```julia
@testset "real node operator equals the complex one" begin
    zp = load_spec(fixture("zero_pi.json")); θ = zero_pi_params()
    children = Any[:theta => ChargeBasis(4), :phi => PhaseGrid(-6π, 6π, 41; order=6)]
    for (spec, ch, θ_) in ((zp, children, θ),
                           (load_spec(fixture("fluxonium.json")), Any[:varphi_1 => FockBasis(14; phi_zpf=0.6)], (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π)),
                           (load_spec(fixture("transmon.json")), Any[:varphi_1 => ChargeBasis(8)], (E_J=20.0, E_C=1.0, n_g=0.3)))
        node = Node(ch; solver=Dense(), keep=lowest(4))
        b = KronFreeCircuits._tree_bases(node); terms = circuit_terms(spec, b); r = KronFreeCircuits.route(node, terms)
        Hr = KronFreeCircuits._node_operator(node, Dense(), Any[c.second for c in ch], r[node], θ_, constant_term(spec, b)(θ_); real=true)
        Hc = KronFreeCircuits._node_operator(node, Dense(), Any[c.second for c in ch], r[node], θ_, constant_term(spec, b)(θ_); real=false)
        @test eltype(Hr) === Float64 && eltype(Hc) === ComplexF64
        A, B = Matrix(Hr), Matrix(Hc)
        @test maximum(abs, A - B) <= 1e-13 * max(1, maximum(abs, B))
        @test length(Hr.terms) <= length(Hc.terms)                                     # conjugate pairs merged
    end
end
```

Plus one three-factor case: three `FockBasis(6; phi_zpf=0.5)` modes with a term list holding `cos(φ1 + φ2 − φ3 + δ)` as two conjugate `expiphi` product terms (coefficients `−½E e^{±iδ}`), so every factor is a dense paired operator and the expansion runs over all subsets; `Matrix` of the real build equals the complex one to 1e-13.

- [ ] **Step 2: Run**; expect `_node_operator` to reject the `real` keyword.

- [ ] **Step 3: Implement.** `_materialize` passes `real = classification[node]`; until Task 6 a real classification must not reach `_materialize` (keep `_materialize` building the complex operator, i.e. call with `real=false`, until Task 6 switches it).

- [ ] **Step 4: Run the default suite**; all pass.

- [ ] **Step 5: Commit** (`Real node operator by expansion of Re(c ⊗ O)`).

---

### Task 5: Solvers in the operator's element type, with completeness probes

**Files:**
- Modify: `src/solvers.jl` (`Dense`, `Lanczos`, the `Solver` contract docstring), `src/shiftinvert.jl` (retire `isreal_assembled`), `src/lobpcg.jl` and `src/chebyshev.jl` (only where a literal type or a complex assumption remains), `src/KronFreeCircuits.jl` (drop the export)
- Test: `test/test_solvers.jl`, `test/test_periodic.jl` (the `isreal_assembled` assertions become element-type checks)

**Interfaces:**
- Consumes: real `KronFreeOperator`s (Task 1).
- Produces: `solve(s, H::KronFreeOperator, m; …)` returns `(E::Vector{Float64}, V::Matrix{eltype(H)})` for every package solver; `ShiftInvert` factors in real arithmetic exactly when `eltype(H) <: Real`; `isreal_assembled` and `_isreal_assembled` no longer exist.

- [ ] **Step 1: Write the failing tests.** A helper in `test/helpers.jl`, `complexified(H::KronFreeOperator)`, rebuilds `H` with every coefficient converted to `ComplexF64` (same factors). Then, in `test/test_solvers.jl`, for each solver in `(Dense(), Lanczos(), LOBPCG(), ShiftInvert(), ShiftInvert(block=2), ChebyshevFilter())`:

```julia
@testset "solvers run in the operator's element type; real equals complex" begin
    for (name, H) in (("three real oscillators", real_three_oscillators(6)),        # 216 states, real cos couplings
                      ("degenerate pair", two_identical_real_oscillators(8)))      # exactly 2-fold degenerate levels
        Hc = complexified(H)
        Ed, Vd = solve(Dense(), H, 0); Edc, _ = solve(Dense(), Hc, 0)
        @test eltype(Vd) === Float64 && Ed ≈ Edc atol=1e-12
        for s in (Lanczos(), LOBPCG(), ShiftInvert(), ShiftInvert(block=2), ChebyshevFilter())
            name == "degenerate pair" && s isa Lanczos && continue                  # documented: single-vector Krylov misses copies
            E, V = solve(s, H, 10)
            @test eltype(V) === Float64
            @test E ≈ Ed[1:10] atol=1e-8
            check_eigenpairs(H, E, V; tol=1e-8)
            # completeness: the kept span equals Dense's on whole degenerate levels
            cut = findlast(<(Ed[10] + 1e-6), Ed); cut == 10 && @test V * V' ≈ Vd[:, 1:10] * Vd[:, 1:10]' atol=1e-6
        end
    end
end
```

`real_three_oscillators(d)` is `three_oscillators` with `cos_op` built real (`real.(…)` of the same matrix, which is real); `two_identical_real_oscillators(d)` couples nothing between two equal real oscillators so every level above the ground is degenerate. Add three adversarial probes in real arithmetic, each against `Dense` on the same operator: (a) a 4-fold degenerate ground level (two identical uncoupled real double wells), `ShiftInvert()` with the default certificate, `LOBPCG()` and `ChebyshevFilter()` returning all four copies; (b) the ChebyshevFilter invariant-seed trap from the plan-2b fixes (a parity-symmetric real operator whose `product_seed` spans one parity sector only), lowest 6 equal to `Dense`'s; (c) a near crossing (two levels 1e-7 apart at m and m+1), the returned span's projector matching `Dense` on the cluster. Replace every `isreal_assembled` use in `test/test_solvers.jl` and `test/test_periodic.jl` by `eltype(operator(sys)) === Float64` / `ComplexF64` per §20f (the periodic transmon at a parameter `n_g` of 0 or ½ is now `ComplexF64`; its `maximum(abs, imag(Matrix(H))) == 0` assertion on the operator data stays).

- [ ] **Step 2: Run**; expect failures in `Dense`/`Lanczos` (`ComplexF64` literals) and the `isreal_assembled` references.

- [ ] **Step 3: Implement.** `Dense`: `eigen!(Hermitian(Matrix(H)), …)` returns `Matrix{eltype(H)}`. `Lanczos`: `randn(eltype(H), n)`. `ShiftInvert`: `realpath = eltype(H) <: Real`; remove the assembled check and its docstring sentence; the `@debug` names the arithmetic. Update the `Solver` docstring: `V` has `eltype(H)`; a user solver must accept real operators.

- [ ] **Step 4: Run the default suite**; all pass.

- [ ] **Step 5: Commit** (`Solvers in the operator's element type; retire isreal_assembled`).

---

### Task 6: Real leaves materialize in `Float64`

**Files:**
- Modify: `src/materialize.jl` (`MaterializedNode{T}`, real operator for real nodes, signs as phases, port formation by components, natural port types, `charge_port`, `arithmetic` accessor), `src/phase.jl` (`fix_phases!` on real vectors), `src/selection.jl` (`SelectionContext` and `Follow` accept either element type), `src/dressed.jl` (`decompose` element type)
- Test: `test/test_materialize.jl`, `test/test_character.jl`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `MaterializedNode{T<:Union{Float64,ComplexF64}}` with `vectors::Matrix{T}`, `peak_amplitudes::Matrix{T}`, `ports::Dict{Tuple{Symbol,Port},Union{Matrix{Float64},Matrix{ComplexF64}}}`, other fields unchanged; `T = Float64` exactly when the node is classified real.
  - `_port_matrix(V::Matrix{Float64}, dims, axis, op, ch::PortCharacter)`: `:real` → `Vᵀ·realpart(op)·V` as `Matrix{Float64}`; `:imaginary` → `im .* (Vᵀ·imagpart(op)·V)`; `:paired` and `:general` → `Vᵀ R V + im·Vᵀ I V` as `Matrix{ComplexF64}`; each product applied with real kernels (`apply_axis!` on real data). The complex-node method is unchanged.
  - Phase on real vectors: multiply by `sign(x)` of the peak entry (leaf), of the real joint-peak amplitude (branch), or of `real(⟨ref|v⟩)` (`:follow`; `+1` when it is zero); a real node never holds complex vectors.
  - `arithmetic(sys, node)` (Task 3) reads `T`. `eigenvectors`, `port`, `decompose` return stored types.

- [ ] **Step 1: Write the failing tests** (`test/test_materialize.jl`):

```julia
@testset "real leaves: Float64 throughout and equal to the complex path" begin
    cases = [(load_spec(fixture("transmon.json")), [:varphi_1 => ChargeBasis(15)], (E_J=20.0, E_C=1.0, n_g=0.3), [(:varphi_1, nport())]),
             (load_spec(fixture("fluxonium.json")), [:varphi_1 => PhaseGrid(-6π, 6π, 151; order=8)], (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π), [(:varphi_1, nport()), (:varphi_1, phiport()), (:varphi_1, expiphi(1))]),
             (load_spec(fixture("zero_pi.json")), [:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)], zero_pi_params(), [(:theta, nport()), (:phi, nport())])]
    for (spec, ch, θ, ps) in cases
        sr = materialize(Node(ch; solver=Dense(), keep=lowest(8), ports=ps), spec, θ)
        sc = materialize(Node(ch; solver=Dense(), keep=lowest(8), ports=ps, arithmetic=:complex), spec, θ)
        @test arithmetic(sr) === :real && arithmetic(sc) === :complex
        @test eltype(eigenvectors(sr)) === Float64
        H = KronFreeCircuits.operator(sc)
        @test energies(sr) ≈ energies(sc) atol=1e-12 * max(1, gershgorin_bound(H))
        s = sign.(real(diag(eigenvectors(sc)' * eigenvectors(sr))))                         # per-state signs (nondegenerate here)
        @test eigenvectors(sr) ≈ real(eigenvectors(sc)) .* s' atol=1e-9
        for (m, p) in ps
            Pr, Pc = port(sr, m, p), port(sc, m, p)
            ch_ = character(only(b for (k, b) in ch if k === m), p)
            @test eltype(Pr) === (ch_.kind === :real ? Float64 : ComplexF64)
            @test Pr ≈ s .* Pc .* s' atol=1e-9
        end
    end
end
```

Also: a real leaf with `keep=connected(core=[1], port=(:phi, expiphi(1)), thresh=1e-3)` on the zero-pi leaf (a paired selection port) and a `GivenSystem` child with a `:general` operator requested in `ports=` at its parent (Review Focus 5) return `ComplexF64` ports equal to the `arithmetic=:complex` path's up to the signs; `decompose(sr, 1)` amplitudes are `Float64`; the existing PhaseGrid edge-support warning still fires on a too-small real leaf; `charge_port` on a `ChargeBasis` mode of a real leaf is `Matrix{Float64}`.

- [ ] **Step 2: Run**; expect `arithmetic(sr) === :complex` (nothing real materializes yet).

- [ ] **Step 3: Implement**: `_materialize` builds the real operator for real nodes, the solver returns `Float64` vectors, phases are signs, ports formed by `_port_matrix` with the character of the factor (the character table lookup of Task 3), peak amplitudes real. Selection contexts and `Follow` store `AbstractMatrix` data of either type.

- [ ] **Step 4: Run the default suite**; all pass, and list every pre-existing test that changed (type assertions only).

- [ ] **Step 5: Commit** (`Real leaves materialize in Float64 with natural-type ports`).

---

### Task 7: Real branches, transport, following and warm starts

**Files:**
- Modify: `src/materialize.jl` (branch phase sign, `_transport_previous` promotion, `_initial_block` element-type rule, child ports into the real parent through `realpart`/`imagpart` of the stored matrices)
- Test: `test/test_materialize.jl`, `test/test_catalogue.jl` (the two-zero-pi small fixture), `test/test_zero_pi_reference.jl` (gated)

**Interfaces:**
- Consumes: Task 6.
- Produces: real branches whose node children are real; `_initial_block(prev, dims, m, T)` returns `nothing` when `eltype(prev.vectors)` is complex and `T <: Real`, converts when real into complex; `_transport_previous` returns `Matrix{promote_type(…)}` of the previous vectors and every child overlap; `follow` and `:follow` work with either element type on either side.

- [ ] **Step 1: Write the failing tests:**

```julia
@testset "real branch: two coupled zero-pis equal the complex path" begin
    tz = load_spec(fixture("two_zero_pi.json")); θ = two_zero_pi_params(g=0.05)
    mk(ar) = (q1 = Node([:theta1 => ChargeBasis(4), :phi1 => PhaseGrid(-6π, 6π, 61; order=6)]; solver=Dense(), keep=lowest(10), arithmetic=ar);
              q2 = Node([:theta2 => ChargeBasis(4), :phi2 => PhaseGrid(-6π, 6π, 61; order=6)]; solver=Dense(), keep=lowest(10), arithmetic=ar);
              Node([q1, q2]; solver=Dense(), keep=lowest(12), ports=[(:theta1, nport())], arithmetic=ar))
    tr, tc = mk(:auto), mk(:complex)
    sr = @test_logs (:warn, r"edge") match_mode=:any materialize(tr, tz, θ)
    sc = @test_logs (:warn, r"edge") match_mode=:any materialize(tc, tz, θ)
    @test arithmetic(sr) === :real && eltype(eigenvectors(sr)) === Float64
    @test energies(sr) ≈ energies(sc) atol=1e-11
    Pr, Pc = port(sr, :theta1, nport()), port(sc, :theta1, nport())
    @test eltype(Pr) === Float64 && abs.(Pr) ≈ abs.(Pc) atol=1e-9                        # invariant to per-state signs
    # LOBPCG at the branch (Auto's branch solver) in real arithmetic equals Dense
    tl = mk(:auto); tl.solver = LOBPCG(); tl.compute = 12
    @test energies(materialize(tl, tz, θ)) ≈ energies(sr) atol=1e-8
end
```

Also, in the same testset file: (Review Focus 2) a parent over `q1` with `arithmetic=:complex` and `q2` real materializes complex with energies equal to `sc`'s; (Review Focus 4) `materialize(tr, tz, θ′; previous=sc)` and `materialize(tc, tz, θ′; previous=sr)` with `θ′` at `g = 0.06` and `keep=follow()` on the root both run and match the same-arithmetic results; the branch `:positive_sector` phase on the real root returns real vectors whose peak amplitude is positive. Extend the gated `test_zero_pi_reference.jl` two-zero-pi acceptance test with an `arithmetic=:complex` rerun compared at 1e-10 (the gated suite's runtime grows by that one rerun).

- [ ] **Step 2: Run**; expect the branch to be complex or to fail in transport/phase on mixed types.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run the default suite and the gated suite** (`KRONFREE_SLOW_TESTS=1`); all pass.

- [ ] **Step 5: Commit** (`Real branches; transport, following and warm starts across element types`).

---

### Task 8: The device path in real arithmetic

**Files:**
- Modify: `ext/KronFreeCircuitsCUDAExt.jl` (only if a complex assumption appears), `test/cuda/test_device.jl`
- Test: `test/cuda/test_device.jl` (run on ewdesktop)

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: real operators, `LOBPCG(on=CuArray)` and `ChebyshevFilter(on=CuArray)` on `CuArray{Float64}` data, with ITensor/cuTENSOR real contractions and zero device allocation after warm-up.

- [ ] **Step 1: Write the device tests:** a real version of "operator on the device equals host; no device allocation after warm-up" (a real operator with banded, shift, diagonal and dense factors; `CUDA.@allocated mul!(…) == 0` after warm-up on a block and on an offset column view); "LOBPCG and ChebyshevFilter on the device in real arithmetic equal Dense on the host" (real three oscillators, and the degenerate-pair probe of Task 5); "F3 small two-zero-pi fixture: real tree with `LOBPCG(on=CuArray)` at the root equals the `arithmetic=:complex` tree" (energies to 1e-8, `eltype(eigenvectors) === Float64`).

- [ ] **Step 2: Run on ewdesktop** with the recipe (`rsync -a --delete … ./ ewdesktop:~/kronfree-plan2-ci/`, `Pkg.develop(path=".")`, `Pkg.instantiate()`, then `~/.juliaup/bin/julia --project=test/cuda test/cuda/runtests.jl`), foreground, `timeout 1200`; save the log to `docs/spikes/2026-10-18-plan3-benchmark/cuda-tests.log`. Expect the new tests to fail only if the device path has a complex assumption.

- [ ] **Step 3: Fix** any device-path complex assumption found.

- [ ] **Step 4: Rerun the device suite**; all pass (previous 332 plus the new assertions).

- [ ] **Step 5: Commit** (`Device path in real arithmetic; device suite log`).

---

### Task 9: Documentation, benchmark rerun, and verification

**Files:**
- Modify: `README.md` (real arithmetic: the rule, the character table in short, `arithmetic=`, native output types, `ShiftInvert`'s arithmetic sentence), `bench/end_to_end.jl` (`BENCH_ARITHMETIC=auto|complex`, applied to every node of the bench tree, recorded in the JSON), docstrings touched by earlier tasks
- Create: `docs/spikes/2026-10-18-plan3-benchmark/results.md` and its JSON outputs
- Modify: `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` (an "Execution notes (plan 3)" paragraph at the end of §20f, from the ledger)

- [ ] **Step 1: Add the knob** and check it with `BENCH_SMOKE=1 BENCH_ARITHMETIC=complex julia --project=. -t 1 bench/end_to_end.jl` and `BENCH_ARITHMETIC=auto`; both finish, and the JSON records the arithmetic.

- [ ] **Step 2: Run the benchmark on ewdesktop**, each in both arithmetics, with the plan-2b configurations of `docs/spikes/2026-10-17-plan2b-benchmark/results.md`: CPU `BENCH_LEAF_SOLVER=auto BENCH_STAGES=leaf,tree,parts` and GPU `BENCH_DEVICE=gpu BENCH_LEAF_SOLVER=auto BENCH_STAGES=leaf,tree,parts` (plus `BENCH_LEAF_SOLVER=chebyshev` on CPU, where the real-arithmetic leaf gain is expected). Write `results.md`: a table of stage times real vs complex vs the plan-2b numbers, the exact commands, and the commit.

- [ ] **Step 3: Update the README and the spec execution notes.**

- [ ] **Step 4: Full verification:** default suite (time against the baseline), gated suite, device suite log from Task 8 rerun if any source changed since.

- [ ] **Step 5: Commit** (`Plan 3: README, benchmark rerun, spec execution notes`).
