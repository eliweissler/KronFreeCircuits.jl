# Plan 2b: Leaf Solvers, Product-State Seeding and the Periodic Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Eli's instruction (2026-10-17): implementation happens in a git worktree on a branch, not on `main`.

**Goal:** Make leaf diagonalization fast and correct by its own cost structure (sparse shift-invert on the CPU, a seeded Chebyshev filter on the GPU), seed every LOBPCG cold start with product states, give nodes an `Auto()` default that picks the solver from the node's shape, document the solver extension contract, and add a periodic phase grid for compact modes with the twisted boundary and exact charge-parity sectors.

**Architecture:** The `KronFreeOperator` stays the one apply everywhere. It gains on-demand, memoized derived quantities (`sparse(H)`, `gershgorin_bound(H)`, `separable_eigenbases(H)`, `product_seed(H, k)`) that the new solvers consume: `ShiftInvert` factors `sparse(H) − σI` and runs KrylovKit on the inverse with a true-residual check; `ChebyshevFilter` runs a filtered block iteration from `product_seed` with a rigorous spectral bound; `LOBPCG` starts from `product_seed` when no previous solution exists; `Auto()` resolves to one of these at materialization. `PeriodicGrid` is a `Basis` whose derivative ports are three parameter-free pieces (interior band and two wrap corners); term construction becomes basis-aware so the twist phases `e^{±2πi n_g}` enter as coefficients and the canonical quadratic form is expanded per mode.

**Tech Stack:** Julia 1.13, SparseArrays (stdlib; CHOLMOD/UMFPACK), KrylovKit 0.10.4, LOBPCGEigensolver 0.1.4, KernelAbstractions, ITensors, CUDA extension as in plan 2. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`, Sections 20d (periodic grid) and 20e (solver policy, `Auto()`, extension API), with 20c's execution notes. Reference implementations (throwaway, validated): `docs/spikes/2026-10-17-leaf-solvers/` — `a1_shiftinvert.jl`, `a3_slicing.jl`, `inertia.jl` (shift-invert, slicing, inertia counts), `c_lobpcg_sep.jl` (separable eigenbases and product seeds), `e_chebyshev.jl` (the filter, Gershgorin bound, rescaling, SVQB), `branch_check.jl` (branch seeding); the thesis listing `~/overleaf/thesis/code-snippets/chapter_0.py` (`transmon_phase_fd`) for the twisted boundary.

## Global Constraints

- Column-major, axis 1 fastest; apply dispatch on structure, never on device; the `KronFreeOperator` is the only apply. Derived quantities are computed when asked for and memoized on the operator in `Ref` fields that start empty; nothing is assembled at construction.
- Element type stays `ComplexF64` in the operator; solver code carries no literal element type (take it from the data). `ShiftInvert` may run in real arithmetic only when the assembled sparse matrix has exactly zero imaginary part (an exact check on assembled data, not a threshold); it must say so in its result/log.
- Every eigensolver result is validated by a true-residual check against the solver's `tol` using the package operator before it is returned; a Krylov or FEAST-style inner tolerance is never the acceptance criterion. Every solver test uses `check_eigenpairs` and projectors where eigenvalues repeat.
- LOBPCGEigensolver and KrylovKit are used through public interfaces only. KrylovKit `BlockLanczos` with block size 4 is not offered (false convergence observed at 0.10.4).
- Parameters enter only through term coefficients (spec §11.1): the periodic grid's twist phases are coefficients on parameter-free wrap-corner operators; `PeriodicGrid` has no `n_g`.
- `Auto()` becomes the `Node` default; it must reproduce the existing test suite's results (small nodes resolve to `Dense`), and an explicit solver always wins.
- Default suite budget: at most +20 s over the current ~108 s; physics convergence studies and anything slower are gated (`KRONFREE_SLOW_TESTS=1`); device tests live in `test/cuda/` and run on ewdesktop (`rsync --delete` only into the disposable `~/kronfree-plan2-ci/`, never a regular checkout; keep fixtures outside the synced tree).
- Physics conventions and existing fixtures unchanged; new fixtures are additive. Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A branch operator asked for `sparse(H)` would build billions of nonzeros: the call must refuse above the nonzero cap with a message naming the cap, before allocating. Pinned in Task 1.
2. A loose Krylov inner tolerance can report convergence at true residuals of 1e-5: `ShiftInvert` must raise (not return) when the true residual exceeds `tol`. Pinned in Task 2.
3. A user-supplied `λ_max` below the true maximum makes the Chebyshev recurrence diverge: the filter must detect non-finite values and raise naming the bound; with `bound=:gershgorin` it must never happen. Pinned in Task 4.
4. A `PeriodicGrid` whose size is not divisible by a harmonic present on its mode, or a non-integer harmonic on a compact mode, must be rejected at materialization with a message naming the mode, the harmonic and the size. Pinned in Task 8.
5. Switching the `Node` default to `Auto()` must not change any existing test's numbers; a node of product dimension 2 500 resolves to `Dense`, 2 501 does not. Pinned in Task 6.

---

### Task 1: Derived quantities on the operator

**Files:**
- Modify: `src/apply.jl` (`KronFreeOperator` gains memo fields; `SparseArrays.sparse`, `gershgorin_bound`, `separable_eigenbases`, `product_seed`), `src/localops.jl` (`infnorm(op)`, `sparse(op)` per `LocalOp`), `src/KronFreeCircuits.jl` (exports)
- Test: `test/test_apply.jl`

**Interfaces:**
- Consumes: `KronFreeOperator` (fields `dims`, `diag`, `terms` with `(axis, op, data)` factor records, `on`), `LocalOp` types, `dense(op)`, `kron_reference`, `three_oscillators` (test/helpers.jl).
- Produces:
  - `SparseArrays.sparse(op::LocalOp)::SparseMatrixCSC{ComplexF64}` for every `LocalOp` (banded and shift from their data, not via `dense`); `infnorm(op::LocalOp)::Float64` = maximum absolute row sum (`sum(abs, weights)` per band row for `BandedOp`, `abs(factor)` for `ShiftOp`, `maximum(abs, d)` for `DiagonalOp`, `opnorm(M, Inf)` for `DenseOp`).
  - `SparseArrays.sparse(H::KronFreeOperator; maxnnz=10^8)::SparseMatrixCSC{ComplexF64}`: the sparse Kronecker sum `Σ_t c_t ⊗_k O_{t,k}` (identity on absent axes, axis 1 fastest) plus the fused diagonal; estimated nonzeros `Σ_t |c_t|·Π_k nnz(O_{t,k})·Π_{absent} d` checked against `maxnnz` before assembly (`ArgumentError` naming the estimate and the cap). Memoized in `H.sparse::Ref{Any}`; the memo is a host matrix regardless of `H.on`.
  - `gershgorin_bound(H)::Float64` = `maximum(real(H.diag)) + Σ_t |c_t| Π_k infnorm(O_{t,k})`, a rigorous upper bound on the largest eigenvalue; memoized.
  - The operator keeps the pure-diagonal `AxisTerm`s it fused in a new field `diagterms::Vector{AxisTerm}` (introspection only; `mul!` still uses `diag`).
  - `separable_eigenbases(H)::NamedTuple` with `U::Vector{Matrix{ComplexF64}}` and `λ::Vector{Vector{Float64}}` per axis: for axis k, every term (fused or not) with exactly one factor, on axis k, summed into a `d_k × d_k` Hermitian matrix (`c_t · dense(O)`; the constant is added to axis 1's energies) and diagonalized with `eigen!`; multi-axis terms, diagonal or not, are coupling and are ignored here; an axis with no single-axis term gets the identity and zero energies; memoized.
  - `product_seed(H, k::Int)::Matrix{ComplexF64}` of size `(prod(dims), k)`: the `k` product states `⊗_a U_a[:, i_a]` with the smallest summed energies `Σ_a λ_a[i_a]` (`partialsortperm` over the Kronecker sum of the `λ_a`), built column by column without forming the full Kronecker sum when `prod(dims)` is large (`k` is small). When every `U_a` is the identity (a branch, whose axes are already eigenbases), the columns are unit vectors picked by the fused diagonal. Columns are orthonormal by construction.

- [ ] **Step 1: Write the failing tests** in `test/test_apply.jl`:

```julia
@testset "sparse(H), gershgorin_bound, separable_eigenbases, product_seed" begin
    rng = MersenneTwister(71)
    H = three_oscillators(5)                                   # 125 states, structured factors only
    S = sparse(H)
    @test S ≈ Matrix(H) && S === sparse(H)                    # exact and memoized
    @test gershgorin_bound(H) >= maximum(eigvals(Hermitian(Matrix(H))))
    @test gershgorin_bound(H) <= 1.5 * maximum(eigvals(Hermitian(Matrix(H))))   # not absurdly loose on this fixture
    sep = separable_eigenbases(H)
    @test length(sep.U) == 3 && all(U -> U' * U ≈ I, sep.U)
    X = product_seed(H, 6)
    @test size(X) == (125, 6) && X' * X ≈ I
    # the seed's Rayleigh quotients sit below the median of the spectrum
    E = eigvals(Hermitian(Matrix(H)))
    @test all(real(diag(X' * (Matrix(H) * X))) .< E[end ÷ 2])
    # a branch-like operator (dense factors, no single-axis structure) seeds with unit vectors by the diagonal
    Hb = KronFreeOperator([4, 3], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(1:4)))]), AxisTerm(1.0, [(2, DiagonalOp(ComplexF64.([0.0, 5.0, 10.0])))]),
                                   AxisTerm(0.1, [(1, DenseOp(randn(rng, ComplexF64, 4, 4))), (2, DenseOp(randn(rng, ComplexF64, 3, 3)))])])
    Xb = product_seed(Hb, 3)
    @test all(count(!iszero, Xb[:, j]) == 1 for j in 1:3) && sort(findall(!iszero, Xb[:, 1]))[1] == argmin(real(Hb.diag))
    # the nonzero cap refuses before assembling
    Hbig = KronFreeOperator([300, 300], [AxisTerm(1.0, [(1, DenseOp(randn(rng, ComplexF64, 300, 300))), (2, DenseOp(randn(rng, ComplexF64, 300, 300)))])])
    err = try sparse(Hbig; maxnnz=10^6); nothing catch e; e end
    @test err isa ArgumentError && occursin("maxnnz", err.msg)
    @test infnorm(ShiftOp(5, 1, 0.3)) == 0.3 && infnorm(DiagonalOp(ComplexF64[1, -2, 0])) == 2
end
```

- [ ] **Step 2: Run** the focused file; confirm `sparse(::KronFreeOperator)`, `gershgorin_bound`, `separable_eigenbases`, `product_seed`, `infnorm` are undefined.
- [ ] **Step 3: Implement** in `src/localops.jl` and `src/apply.jl`; export the four operator functions (`sparse` extends `SparseArrays.sparse`).
- [ ] **Step 4: Run the full default suite.** Expected: pass.
- [ ] **Step 5: Commit.**

---

### Task 2: `ShiftInvert` leaf solver

**Files:**
- Create: `src/shiftinvert.jl` (included after `lobpcg.jl`)
- Modify: `src/tree.jl` (`struct ShiftInvert <: Solver`), `src/KronFreeCircuits.jl` (export)
- Test: `test/test_solvers.jl`

**Interfaces:**
- Consumes: `sparse(H)`, `gershgorin_bound(H)`, `check_eigenpairs`, KrylovKit (`KronFreeCircuits.KrylovKit`), `SparseArrays.cholesky`/`lu`/`ldlt`.
- Produces:
  - `ShiftInvert(; tol=1e-8, shift=:auto, shift_margin=5.0, inner_tol=1e-13, krylovdim=0, maxiter=100, block=1, slices=1, certify=false, maxnnz=10^8)`; `block ∈ (1, 2)` else `ArgumentError`; `slices ≥ 1` (slicing itself is Task 3; `slices > 1` raises `ArgumentError("slicing arrives in Task 3")` until then). `shift_margin` is in the Hamiltonian's energy units (documented like `eta`).
  - `solve(s::ShiftInvert, H::KronFreeOperator, m)`: `A = sparse(H; maxnnz=s.maxnnz)`; real path when `iszero(imag(A))` and the fused diagonal is real (then `A = real(A)`); `σ = s.shift === :auto ? E_min − s.shift_margin : s.shift` with `E_min` from a short KrylovKit Lanczos (1 state, tol 1e-6) on `A`; `F = cholesky(A − σI)` falling back to `lu` on `PosDefException` (and always `lu` when `σ` is interior); KrylovKit `eigsolve(x -> F \ x, x0, m, :LM; ishermitian=true, tol=s.inner_tol, krylovdim=max(2m + 10, s.krylovdim), maxiter=s.maxiter)` (`BlockLanczos` with block 2 when `s.block == 2`); energies `σ + 1/μ` sorted ascending; the true residual of every returned pair computed with the package operator and compared to `s.tol`, raising `ErrorException` naming the worst residual when it fails; `certify=true` additionally counts eigenvalues below `E_m + gap/2` by the inertia of `ldlt(A − τI)` and raises when the count differs from `m`. Returns `(E::Vector{Float64}, V::Matrix{ComplexF64})` on the host (real eigenvectors promoted).
  - `storage(::ShiftInvert) = Array`; `isreal_assembled(H)::Bool` (exact zero imaginary part of `sparse(H)` and of the fused diagonal) is the predicate the real path uses, exported for tests.

- [ ] **Step 1: Write the failing tests** in `test/test_solvers.jl`:

```julia
@testset "ShiftInvert equals Dense; real path; residual check; certificate" begin
    Random.seed!(81)
    H = three_oscillators(6)                                   # complex coefficients? check: if real, the real path runs
    Ed, Vd = solve(Dense(), H, 0)
    E, V = solve(ShiftInvert(tol=1e-9), H, 12)
    @test E ≈ Ed[1:12] atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
    @test V * V' ≈ Vd[:, 1:12] * Vd[:, 1:12]' atol=1e-7
    E2, V2 = solve(ShiftInvert(tol=1e-9, block=2), H, 12); @test E2 ≈ Ed[1:12] atol=1e-8; check_eigenpairs(H, E2, V2; tol=1e-9)
    E3, _ = solve(ShiftInvert(tol=1e-9, certify=true), H, 12); @test E3 ≈ Ed[1:12] atol=1e-8
    # the zero-pi leaf on the small grid (1313 states) against Dense, real arithmetic expected
    θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    leaf = Node([:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(8))
    sd = @test_logs (:warn, r"edge") match_mode=:any materialize(leaf, spec_zp, θ)
    si = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(leaf.children; solver=ShiftInvert(tol=1e-10), compute=8, keep=lowest(8)), spec_zp, θ)
    @test energies(si) ≈ energies(sd) atol=1e-8
    @test KronFreeCircuits.isreal_assembled(KronFreeCircuits.operator(si))        # the symmetric zero-pi leaf takes the real path
    check_eigenpairs(KronFreeCircuits.operator(si), energies(si), eigenvectors(si); tol=1e-9)
    # a loose inner tolerance must raise, not return
    err = try solve(ShiftInvert(tol=1e-10, inner_tol=1e-4), H, 12); nothing catch e; e end
    @test err isa ErrorException && occursin("residual", err.msg)
    @test_throws ArgumentError ShiftInvert(block=4)
    @test_throws ArgumentError solve(ShiftInvert(slices=2), H, 4)
end
```

- [ ] **Step 2: Run**; confirm `ShiftInvert` undefined.
- [ ] **Step 3: Implement** `src/shiftinvert.jl` following `a1_shiftinvert.jl` and `a1_block.jl`; the inertia count follows `inertia.jl`.
- [ ] **Step 4: Run the full default suite.** Expected: pass.
- [ ] **Step 5: Commit.**

---

### Task 3: Spectrum slicing for `ShiftInvert`

**Files:**
- Modify: `src/shiftinvert.jl`
- Test: `test/test_solvers.jl`

**Interfaces:**
- Consumes: Task 2's solver and the inertia count.
- Produces: `slices::Int > 1` (or `slices=:auto` = `Threads.nthreads()` when greater than 1): the target range `[E_min, E_m]` is located by inertia bisection (`ldlt(A − τI)` negative-pivot counts), split into `slices` contiguous intervals of roughly equal state counts, each solved by shift-invert Lanczos with the shift at the interval's center (`lu`, since interior shifts are indefinite) and `krylovdim = 2S + 10` for `S` states in the slice, in parallel over `Threads.@threads` with one factorization per slice; the union is de-duplicated by energy within `10·tol` and re-orthogonalized if two slices returned the same state; the inertia counts certify that every slice is complete (a slice short of its count raises). Reference: `a3_slicing.jl`.

- [ ] **Step 1: Write the failing test:**

```julia
@testset "ShiftInvert slicing equals the single-shift solve" begin
    Random.seed!(82)
    H = three_oscillators(7)                                   # 343 states
    E1, V1 = solve(ShiftInvert(tol=1e-9), H, 40)
    E3, V3 = solve(ShiftInvert(tol=1e-9, slices=3), H, 40)
    @test E3 ≈ E1 atol=1e-8 && V3 * V3' ≈ V1 * V1' atol=1e-7
    check_eigenpairs(H, E3, V3; tol=1e-9)
end
```

- [ ] **Step 2: Run**; confirm the Task 2 placeholder error. **Step 3: Implement. Step 4: Full suite. Step 5: Commit.**

---

### Task 4: `ChebyshevFilter` block solver

**Files:**
- Create: `src/chebyshev.jl` (included after `shiftinvert.jl`)
- Modify: `src/tree.jl` (`struct ChebyshevFilter <: Solver`), `src/KronFreeCircuits.jl` (export), `test/cuda/test_device.jl` (one device testset)
- Test: `test/test_solvers.jl`

**Interfaces:**
- Consumes: `product_seed`, `gershgorin_bound`, `KronFreeOperator` block `mul!` with reserved capacity, `with_block_capacity`, `check_eigenpairs`.
- Produces:
  - `ChebyshevFilter(; tol=1e-8, degree=100, extra=40, maxiter=50, rescale_every=20, ortho=:svqb, bound=:gershgorin, seed=:product, on=Array, callback=nothing)`; `degree ∈ 1:200`, `ortho ∈ (:svqb, :cholqr)`, `bound ∈ (:gershgorin, <:Real)`, `seed ∈ (:product, :random)`.
  - `solve(s::ChebyshevFilter, H, m)`: block width `nb = m + extra`; `X0 = product_seed(H, nb)` (or random), adapted to `s.on`; `λ_max = gershgorin_bound(H)` or the given number; `λ_cut` from the current Ritz values (the `m`-th plus a margin, initialized from the seed's Rayleigh quotients); each iteration applies the degree-`degree` Chebyshev polynomial on the interval `[λ_cut, λ_max]` mapped to `[−1, 1]` through the three-term recurrence with the block renormalized every `rescale_every` steps, then orthonormalizes (`:svqb`: eigendecomposition of the Gram matrix with a relative cutoff, dropping and refilling directions with random vectors; `:cholqr`: Cholesky-QR), performs Rayleigh–Ritz on the `nb`-wide block, locks columns whose true residual (package operator) is below `tol`, and stops when the lowest `m` are locked; raises `ErrorException` naming the bound when any value is non-finite (a bound below the true `λ_max`), and when `maxiter` is exhausted. Returns the lowest `m` as host arrays. `storage(s) = s.on`. Reference: `e_chebyshev.jl`.

- [ ] **Step 1: Write the failing tests:**

```julia
@testset "ChebyshevFilter equals Dense; seeds; bound safety" begin
    Random.seed!(91)
    H = three_oscillators(6)
    Ed, Vd = solve(Dense(), H, 0)
    for (seed, deg) in ((:product, 50), (:random, 50), (:product, 200))
        E, V = solve(ChebyshevFilter(tol=1e-9, degree=deg, extra=12, seed=seed), H, 8)
        @test E ≈ Ed[1:8] atol=1e-7
        check_eigenpairs(H, E, V; tol=1e-8)
        @test V * V' ≈ Vd[:, 1:8] * Vd[:, 1:8]' atol=1e-6
    end
    θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    leaf = Node([:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(8))
    sd = @test_logs (:warn, r"edge") match_mode=:any materialize(leaf, spec_zp, θ)
    sc = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(leaf.children; solver=ChebyshevFilter(tol=1e-9), compute=8, keep=lowest(8)), spec_zp, θ)
    @test energies(sc) ≈ energies(sd) atol=1e-7
    # a user bound below the true λ_max is detected, not silently wrong
    λtrue = maximum(Ed)
    err = try solve(ChebyshevFilter(tol=1e-9, degree=100, bound=0.5 * λtrue), H, 8); nothing catch e; e end
    @test err isa ErrorException && occursin("bound", err.msg)
    @test_throws ArgumentError ChebyshevFilter(degree=0)
end
```

And in `test/cuda/test_device.jl`: the two-zero-pi small leaf (`ChargeBasis(6)`, 101 points) with `ChebyshevFilter(on=CuArray)` for 8 states against `Dense`, `check_eigenpairs`, and `CUDA.@allocated == 0` for one filtered block apply after warm-up.

- [ ] **Step 2: Run**; confirm failure. **Step 3: Implement** (host first; the device path reuses the operator and `product_seed` adapted to `CuArray`; run the device test on ewdesktop with plan 2's commands). **Step 4: Full default suite; device suite on ewdesktop. Step 5: Commit** (with the device log under `docs/spikes/2026-10-17-plan2b-benchmark/cuda-tests.log`).

---

### Task 5: Product-state seeding of `LOBPCG`

**Files:**
- Modify: `src/tree.jl` (`LOBPCG` gains `seed::Symbol`), `src/lobpcg.jl` (`lobpcg_solve` uses `product_seed` when `X0 === nothing` and `seed === :product`), `src/materialize.jl` (no change in `_initial_block`; verify the previous block still takes precedence)
- Test: `test/test_solvers.jl`, `test/test_materialize.jl`

**Interfaces:**
- Produces: `LOBPCG(; …, seed=:product)` with `seed ∈ (:product, :random)`; with `:product` a cold start's whole block of `m + g` columns is `product_seed(H, m + g)` (unperturbed, no random columns), and the guard count still sets `n_conv_check = m + g` and the returned lowest `m`; with `:random` the behavior of plan 2 is unchanged; `X0` from a previous materialization wins over either seed.

- [ ] **Step 1: Write the failing tests:**

```julia
@testset "LOBPCG product-state seed cuts the cold start" begin
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    work = Dict{Symbol,Int}()
    tree(seed) = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))];
                      solver=LOBPCG(tol=1e-10, seed=seed, callback=info -> (work[seed] = info.n_matvec; nothing)), compute=6, keep=lowest(6))
    Random.seed!(5)
    sp = materialize(tree(:product), spec, θ); sr = materialize(tree(:random), spec, θ)
    @test energies(sp) ≈ energies(sr) atol=1e-9
    @test work[:product] < work[:random]
    check_eigenpairs(KronFreeCircuits.operator(sp), energies(sp), eigenvectors(sp); tol=1e-9)
end
```

- [ ] **Step 2: Run**; confirm `seed` keyword unknown. **Step 3: Implement. Step 4: Full suite (the Task 4/5 warm-start tests of plan 2 must still pass: previous vectors win). Step 5: Commit.**

---

### Task 6: `Auto()` default and the solver extension contract

**Files:**
- Modify: `src/tree.jl` (`struct Auto <: Solver`, `Node` default `solver=Auto()`), `src/materialize.jl` (`resolve(::Solver, info) = solver`, `resolve(::Auto, info)` at the top of `_node_operator`/`_solve`), `src/solvers.jl` (docstring of `Solver` with the contract), `README.md` (Solvers section), `src/KronFreeCircuits.jl` (export `Auto`)
- Test: `test/test_tree.jl`, `test/test_materialize.jl`

**Interfaces:**
- Produces:
  - `Auto(; dense_cutoff=2500, on=Array)`; `resolve(a::Auto, (; isleaf, dims, m))::Solver`: `Dense()` when `prod(dims) ≤ dense_cutoff`; else leaf → `ShiftInvert()` when `a.on === Array`, `ChebyshevFilter(on=a.on)` otherwise; branch → `LOBPCG(:ritz, seed=:product, on=a.on)`. `storage(a::Auto) = a.on`. `Node` docstring lists `Auto()` as the default and the rule; `compute == 0` with `Auto()` is allowed only when it resolves to `Dense` (otherwise `ArgumentError` at materialization naming the resolved solver).
  - The `Solver` docstring states the contract (spec §20e.6): `solve(s, H, m; X0=nothing)` → ascending `E::Vector{Float64}`, orthonormal `V::Matrix` on the host, true residuals below `tol`; `storage(s)`; what `H` guarantees (`mul!` on blocks, `diagonal`, `sparse`, `gershgorin_bound`, `separable_eigenbases`, `product_seed`); a user solver needs only `solve`. README gets a short example of a user-defined solver.

- [ ] **Step 1: Write the failing tests:**

```julia
@testset "Auto resolves by node shape and keeps small nodes Dense" begin
    using KronFreeCircuits: resolve, Auto
    @test resolve(Auto(), (isleaf=true, dims=[50, 50], m=10)) isa Dense                # 2500: Dense
    @test resolve(Auto(), (isleaf=true, dims=[50, 51], m=10)) isa ShiftInvert           # 2550: leaf, CPU
    struct FakeDevice end                                                                 # any non-Array storage marker
    @test resolve(Auto(on=FakeDevice), (isleaf=true, dims=[50, 51], m=10)) isa ChebyshevFilter
    @test resolve(Auto(), (isleaf=false, dims=[60, 60], m=10)) isa LOBPCG
    @test Node([:a => ChargeBasis(3)]).solver isa Auto
    # the default tree of the existing suite resolves to Dense and gives the same numbers
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    t_auto = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))]; keep=lowest(6))
    t_dense = Node([Node([:varphi_1 => ChargeBasis(8)]; solver=Dense(), keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; solver=Dense(), keep=lowest(6))]; solver=Dense(), keep=lowest(6))
    @test energies(materialize(t_auto, spec, θ)) == energies(materialize(t_dense, spec, θ))
    @test_throws ArgumentError materialize(Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-6π, 6π, 301; order=8)]; compute=0, keep=lowest(10)), load_spec(fixture("zero_pi.json")), zero_pi_params())   # Auto → ShiftInvert needs compute > 0
end
```

(Define a trivial `struct FakeDevice end` storage marker in the test for the device-resolution assertion; `resolve` must dispatch on `a.on === Array` only.)

- [ ] **Step 2: Run**; confirm failure. **Step 3: Implement**, including the docstrings and README section. **Step 4: Full default suite (every existing test unchanged in numbers). Step 5: Commit.**

---

### Task 7: `PeriodicGrid` basis and its operators

**Files:**
- Modify: `src/bases.jl` (`PeriodicGrid`, `dimension`, `coordinates`, `positive_mask`, `modetype`, `spacing`, `localop`, `check_sectors`), `src/terms.jl` (`port_of_kind` accepts the new kinds), `src/materialize.jl` (`_check_bases`: `:c` allows `PeriodicGrid`), `src/KronFreeCircuits.jl` (export `PeriodicGrid`)
- Test: `test/test_bases.jl`

**Interfaces:**
- Produces:
  - `PeriodicGrid(N; order=8, sectors=1)`: `N ≥ order + 1`, `N % sectors == 0`; `coordinates` = `−π .+ 2π .* (0:N−1) ./ N`, `spacing = 2π/N`, `modetype = :c`, `positive_mask = coordinates .>= 0`.
  - Port kinds, all parameter-free: `Port(:d1, 0)` the interior first-derivative band (`−i`·antisymmetric stencil weights `/Δ`, as `PhaseGrid`'s `:n`, rows and offsets that do not wrap); `Port(:d1wrap, +1)` / `Port(:d1wrap, −1)` the upper / lower wrap corners (the stencil entries that cross the boundary, as `BandedOp`s with offsets `±(N − j)` for `j = 1..order/2` and weight vectors nonzero only in the corner rows); `Port(:d2, 0)`, `Port(:d2wrap, ±1)` likewise for `−∂²`; `expiphi(k)` for integer `k` diagonal `exp(i k φ)`; `:phi`, `:phi2`, `:n`, `:n2`, `:exp2pin` unavailable with a message explaining why (`:n`/`:n2` are assembled by term construction from the pieces; `φ` is multivalued on a circle; `:exp2pin` is a later item). Hermiticity: `d2 + e^{iα} d2wrap₊ + e^{−iα} d2wrap₋` is Hermitian for every real `α` (`d2wrap₋ == d2wrap₊'`), and likewise for `d1` with the `−i` convention.
  - `check_sectors(b::PeriodicGrid, harmonics::AbstractVector{<:Integer}, mode::Symbol)`: `ArgumentError` naming the mode, the harmonic and `N` unless `N % ω == 0` for every `ω` and `N % b.sectors == 0`.

- [ ] **Step 1: Write the failing tests** in `test/test_bases.jl`:

```julia
@testset "PeriodicGrid: twisted stencil equals the thesis construction; sectors" begin
    using KronFreeCircuits: check_sectors
    for order in (2, 4, 6, 8), N in (16, 24)
        b = PeriodicGrid(N; order=order)
        Δ = 2π / N; w = KronFreeCircuits.fd_weights(order, 2); h = order ÷ 2
        for ng in (0.0, 0.25, 0.5)
            # thesis listing: roll-based shifts with phases on the wrapped block, D2 = Σ_j w_j (S_j + S_-j) + w_0 I, divided by Δ²
            shift(j) = (S = circshift(Matrix{ComplexF64}(I, N, N), (j, 0)); j > 0 ? (S[1:j, :] .*= cis(2π * ng); S) : j < 0 ? (S[end+j+1:end, :] .*= cis(-2π * ng); S) : S)
            D2 = -(w[h+1] * I + sum(w[j+h+1] * (shift(j) + shift(-j)) for j in 1:h)) / Δ^2
            ours = dense(localop(b, Port(:d2, 0))) + cis(2π * ng) * dense(localop(b, Port(:d2wrap, 1))) + cis(-2π * ng) * dense(localop(b, Port(:d2wrap, -1)))
            @test ours ≈ D2
            @test ours ≈ ours'
            d1 = dense(localop(b, Port(:d1, 0))) + cis(2π * ng) * dense(localop(b, Port(:d1wrap, 1))) + cis(-2π * ng) * dense(localop(b, Port(:d1wrap, -1)))
            @test d1 ≈ d1'
        end
        @test dense(localop(b, expiphi(2))) ≈ Diagonal(cis.(2 .* coordinates(b)))
    end
    @test_throws ArgumentError localop(PeriodicGrid(16), phiport())
    @test_throws ArgumentError localop(PeriodicGrid(16), expiphi(0.5))
    @test_throws ArgumentError check_sectors(PeriodicGrid(15), [2], :phi)
    @test check_sectors(PeriodicGrid(16; sectors=4), [2, 4], :phi) === nothing
    @test_throws ArgumentError PeriodicGrid(15; sectors=2)
    @test KronFreeCircuits.modetype(PeriodicGrid(16)) === :c && KronFreeCircuits.spacing(PeriodicGrid(16)) ≈ 2π / 16
end
```

(The real-at-half-integer property, `cis(2π·0.5) = −1`, is covered by the `ng = 0.5` loop iteration; assert `maximum(abs, imag(ours)) == 0` there.)

- [ ] **Step 2: Run**; confirm `PeriodicGrid` undefined. **Step 3: Implement. Step 4: Full suite. Step 5: Commit.**

---

### Task 8: Basis-aware term construction, materialization and physics tests

**Files:**
- Modify: `src/terms.jl` (`circuit_terms(spec, bases::Dict{Symbol,Basis})`), `src/spec.jl` (reject non-integer harmonics on compact modes at load), `src/materialize.jl` (collect bases from the tree, call the basis-aware construction, run `check_sectors`, `charge_port(sys, mode)` accessor), `src/dressed.jl` only if labels need the `:c` handling
- Create: `test/fixtures/cos2phi.json` (one compact mode, `cosines: [{energy: "E_2", omega: [2], offset: "0"}, {energy: "E_1", omega: [1], offset: "0"}]`, parameters `E_C, n_g, E_1, E_2`), `test/test_periodic.jl` (default suite) and additions to `test/test_zero_pi_reference.jl` (gated convergence)
- Test: as above

**Interfaces:**
- Produces:
  - `circuit_terms(spec, bases)`: for each compact mode `m` whose basis is a `PeriodicGrid`, the canonical quadratic form `4 Σ_ij (n_i − n_g,i) E_C[i,j] (n_j − n_g,j)` is expanded per mode: a twisted mode contributes `D_m` in place of `(n_m − n_g,m)`, where the first derivative `D_m = d1 + e^{2πi n_g,m} d1wrap₊ + e^{−2πi n_g,m} d1wrap₋` and `D_m² = d2 + e^{2πi n_g,m} d2wrap₊ + e^{−2πi n_g,m} d2wrap₋` (the twisted second derivative is used directly for the diagonal kinetic term; the cross term with mode `j` is `8E_C[m,j] D_m ⊗ (n_j − n_g,j)` expanded into the three pieces times `n_j` and times `−n_g,j`); the twist phases are `Coefficient`s built with `phase_factor(2π·n_g,m, ±1)`; non-twisted modes expand as today; the constant term contains only the non-twisted modes' `4 n_gᵀ E_C n_g` share. Cosines on a twisted mode with integer `ω` use `expiphi(ω)` as today. `circuit_terms(spec)` (no bases) is unchanged and means "all compact modes in the charge basis".
  - `materialize(root, spec, θ; …)` collects `mode => basis` over the whole tree, calls `circuit_terms(spec, bases)` and `constant_term(spec, bases)`, runs `check_sectors(b, harmonics on that mode, mode)` for every `PeriodicGrid`, and raises at load (`parse_spec`) when a compact mode has a non-integer harmonic.
  - `charge_port(sys, mode)`: the physical charge matrix in the kept eigenbasis of the leaf holding `mode`: `V†(D₁)V + n_g I` assembled from the three piece ports (which `required_ports` must therefore expose when a coupling or the user asks) plus the identity; for a charge-basis mode it is `port(sys, mode, nport())`.

- [ ] **Step 1: Write the failing tests** in `test/test_periodic.jl`:

```julia
@testset "free rotor on the periodic grid vs the exact (n - n_g)^2 spectrum" begin
    spec = load_spec(fixture("transmon.json"))
    for ng in (0.0, 0.3, 0.5), N in (32, 64)
        sys = materialize(Node([:varphi_1 => PeriodicGrid(N; order=8)]; solver=Dense(), keep=lowest(6)), spec, (E_J=0.0, E_C=1.0, n_g=ng))
        exact = sort([4 * (n - ng)^2 for n in -8:8])[1:6]
        @test energies(sys) ≈ exact atol=(N == 32 ? 1e-4 : 1e-6)
        H = KronFreeCircuits.operator(sys); A = Matrix(H)
        ng == 0.5 || ng == 0.0 ? (@test maximum(abs, imag(A)) == 0) : (@test maximum(abs, imag(A)) > 0)   # twist phase ±1 is real
    end
end

@testset "transmon: periodic grid vs charge basis; physical charge port" begin
    spec = load_spec(fixture("transmon.json")); θ = (E_J=20.0, E_C=1.0, n_g=0.3)
    sc = materialize(Node([:varphi_1 => ChargeBasis(20)]; solver=Dense(), keep=lowest(6), ports=[(:varphi_1, nport())]), spec, θ)
    sp = materialize(Node([:varphi_1 => PeriodicGrid(128; order=8)]; solver=Dense(), keep=lowest(6)), spec, θ)
    @test energies(sp) ≈ energies(sc) atol=1e-7
    @test abs.(KronFreeCircuits.charge_port(sp, :varphi_1)) ≈ abs.(port(sc, :varphi_1, nport())) atol=1e-6
end

@testset "cos 2φ: both parity sectors, exact half-period translation, divisibility" begin
    spec = load_spec(fixture("cos2phi.json"))
    θ2 = (E_C=1.0, n_g=0.2, E_1=0.0, E_2=15.0)
    sc = materialize(Node([:phi => ChargeBasis(20)]; solver=Dense(), keep=lowest(8)), spec, θ2)
    N = 64; b = PeriodicGrid(N; order=8)
    sp = materialize(Node([:phi => b]; solver=Dense(), keep=lowest(8)), spec, θ2)
    @test energies(sp) ≈ energies(sc) atol=1e-7
    # the half-period translation T: φ -> φ + π is an exact grid translation on the twisted function space, so it must
    # carry the twist phase on the half that wraps (the listing's shift(N ÷ 2) construction, same convention as Task 7's test);
    # then T² = cis(±2π n_g) I, T commutes with H, and the kept-span projector commutes with T
    V = eigenvectors(sp)
    T = circshift(Matrix{ComplexF64}(I, N, N), (N ÷ 2, 0)); T[1:N÷2, :] .*= cis(2π * θ2.n_g)
    phase = (T^2)[1, 1]; @test T^2 ≈ phase * I
    @test norm(T * (V * V') - (V * V') * T) < 1e-9
    # the sectors are the two eigenvalues ±√phase of T; both are present among the kept states
    parity = [real(dot(V[:, j], T * V[:, j]) / cis(angle(phase) / 2)) for j in 1:8]
    @test all(abs.(abs.(parity) .- 1) .< 1e-6) && any(>(0.5), parity) && any(<(-0.5), parity)
    @test_throws ArgumentError materialize(Node([:phi => PeriodicGrid(63; order=8)]; solver=Dense(), keep=lowest(8)), spec, θ2)   # 63 not divisible by 2
    # a weak cos φ mixes the sectors: T no longer commutes with the kept span
    sm = materialize(Node([:phi => b]; solver=Dense(), keep=lowest(8)), spec, (E_C=1.0, n_g=0.2, E_1=0.5, E_2=15.0))
    Vm = eigenvectors(sm)
    @test norm(T * (Vm * Vm') - (Vm * Vm') * T) > 1e-3
end

@testset "a twisted mode coupled through its charge: two transmons" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(; g=0.1)
    tc = Node([Node([:varphi_1 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8)), Node([:varphi_2 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8))]; solver=Dense(), keep=lowest(10))
    tp = Node([Node([:varphi_1 => PeriodicGrid(96; order=8)]; solver=Dense(), keep=lowest(8)), Node([:varphi_2 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8))]; solver=Dense(), keep=lowest(10))
    @test energies(materialize(tp, spec, θ)) ≈ energies(materialize(tc, spec, θ)) atol=1e-6
    # the twist phases are coefficients: n_g1 is a free symbol of the wrap terms, and no operator data depends on it
    terms = circuit_terms(spec, Dict(:varphi_1 => PeriodicGrid(96; order=8), :varphi_2 => ChargeBasis(10)))
    wrap = filter(t -> any(f -> f.port.kind in (:d1wrap, :d2wrap), t.factors), terms)
    @test !isempty(wrap) && all(t -> :n_g1 in KronFreeCircuits.free_symbols(t.coeff), wrap)
end

@testset "non-integer harmonic on a compact mode is rejected at load" begin
    txt = replace(read(fixture("cos2phi.json"), String), "\"omega\": [2]" => "\"omega\": [1.5]")
    @test_throws ArgumentError parse_spec(txt)
end
```

Gated (`test/test_zero_pi_reference.jl`): convergence of the transmon and cos 2φ spectra with `N ∈ (32, 64, 128, 256)` at order 8 against the charge basis, reporting the error decay; the symmetric zero-pi with θ on a `PeriodicGrid` against the charge-basis tree on identical φ grids.

- [ ] **Step 2: Run**; confirm failures (`circuit_terms(spec, bases)` undefined; `PeriodicGrid` rejected by `_check_bases` until Task 7's change is wired).
- [ ] **Step 3: Implement.** Keep `circuit_terms(spec)` byte-for-byte in behavior (existing fixtures pin it).
- [ ] **Step 4: Full default suite and the gated suite.** Expected: pass.
- [ ] **Step 5: Commit.**

---

### Task 9: Documentation and the leaf benchmark rerun

**Files:**
- Modify: `README.md` (solver policy and `Auto()`, `ShiftInvert`, `ChebyshevFilter`, `PeriodicGrid`, the extension contract example), `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` Section 9's table (one row per solver with the `Auto()` rule; mark the Section 8/9 text amended by 20e)
- Modify: `bench/end_to_end.jl` (leaf solver options `shiftinvert` and `chebyshev`)
- Create: `docs/spikes/2026-10-17-plan2b-benchmark/results.md`

**Interfaces:**
- Consumes: the plan-2 benchmark scripts and ewdesktop workflow; the spike's numbers as expectations (2D leaf, 300 states: `ShiftInvert` ≈ 3 s all-in on the CPU; `ChebyshevFilter(on=CuArray)` ≈ 2.5 s in complex arithmetic).
- Produces: the end-to-end two-zero-pi tree (plan 2 part B) rerun with `Auto()` leaves on CPU and on GPU, reporting leaf time, parent time (now seeded) and total against plan 2's 237 s full tree; the `dense_cutoff=2500` placeholder noted as the quantity to tune at the end.

- [ ] **Step 1: Update README and the spec table. Step 2: Extend the benchmark script and run it on ewdesktop (CPU with 8 threads, then GPU), one configuration per ssh call. Step 3: Write `results.md` with how each number was obtained. Step 4: Commit.**
