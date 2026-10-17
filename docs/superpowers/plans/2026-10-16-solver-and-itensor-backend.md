# Plan 2: Block Solver and ITensor Parent Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every node the stock LOBPCGEigensolver block solver with fixed and adaptive per-Ritz Jacobi preconditioning and warm starts, apply dense ports at branches through ITensor contractions (cuTENSOR on GPU) behind the existing operator interface, run a node's solve on a CUDA device through a package extension, make `follow` physically correct across changing child bases, and admit pre-diagonalized subsystems as tree children.

**Architecture:** `KronFreeOperator` becomes the single device-aware operator: it owns device copies of its factor data, fuses every pure-diagonal term into one state-shaped vector applied by a broadcast, applies `DenseOp` factors through scratch-backed ITensor contractions and structured factors through the existing KernelAbstractions kernels. `LOBPCG <: Solver` wraps `LOBPCGEigensolver.lobpcg` through its public `mul!`/`ldiv!`/`precondprep!`/callback interfaces only. Materialization passes a node's previous kept eigenvectors as the initial block and transports them into the current product basis for selection and phase. The CUDA extension adds only what the base cannot express generically (a fused diagonal-expectation kernel); everything else moves to the device through `Adapt`.

**Tech Stack:** Julia 1.13, ITensors 0.9 (NDTensors), LOBPCGEigensolver 0.1.4, KernelAbstractions 0.9/0.10, KrylovKit 0.10, Adapt; CUDA 6.4 and cuTENSOR as weak dependencies; tests on the Mac (CPU) and on `ewdesktop` (RTX A4500) over ssh.

**Spec:** `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`, Sections 8, 9, 9.1 as amended by Section 20c (decisions 1–6, 9, 11 and 12). Evidence and reference implementations: `docs/spikes/2026-10-16-itensor-audit/` (REPORT.md §5, §7; `buffer-layout.jl`, `native-gemm-parent.jl` for the buffered ITensor adapter; `ritz-preconditioner.jl` for the adaptive preconditioner; CORRECTNESS-NOTES.md for the two bugs Tasks 4 and 6 fix).

## Global Constraints

- Julia 1.13. Column-major, axis 1 fastest, one axis per child plus an optional trailing block axis. Apply dispatch is on operator structure, never on device.
- Element type stays `ComplexF64` throughout plan 2, but new solver and backend code carries no literal element type: take it from the data (`eltype(X)`, `real(eltype(X))`), so the real-arithmetic plan can follow without rewriting it.
- LOBPCGEigensolver is used only through `lobpcg`, `precondprep!`, `ldiv!`, and the `callback` keyword; no method is defined on its internal types and no source is copied. Compat is pinned to `"=0.1.4"` until the callback-contract test (Task 4) is rerun against a newer release.
- ITensors compat `"0.9"`; KernelAbstractions compat `"0.9, 0.10"`; KrylovKit `"0.10.4"` unchanged. CUDA and cuTENSOR are weak dependencies only; the default test suite runs without a GPU and stays under about 60 s.
- Workspace contract (spec §8): block `mul!(Y, H, X)` writes into caller-owned output through two preallocated scratch arrays reserved for the solver's maximum block width; shrinking active blocks reuse contiguous prefixes; after warm-up no state-sized allocation occurs on CPU or device. Small metadata allocations are allowed.
- User callbacks handed to a solver may return anything; the solver's own bookkeeping never depends on their return value.
- Every solver test validates independently: true residuals `H v − λ v` and orthonormality through `check_eigenpairs` (Task 4), and subspace projectors wherever eigenvalues repeat. Energies alone never establish that independent eigenvectors were recovered.
- Physics conventions, fixtures and the scqubits reference file are unchanged.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A block solver locks states, so `mul!` receives offset column views (`view(X, :, 3:7)`) into solver-owned arrays. The ITensor path must give the same result as the reference and allocate nothing state-sized on them. Pinned in Task 3.
2. `LOBPCG` on a node too small for its block (`N < 3m + 5`) must raise an `ArgumentError` naming `Dense`, not a cryptic library error. Pinned in Task 4.
3. An exactly repeated eigenvalue (`diag(0, 0, 1, …)`) must come back twice from `LOBPCG` and `Dense`; the single-vector `Lanczos` limitation is documented, not silently inherited by the default path. Pinned in Task 4.
4. A `previous` system whose child kept order differs from the current one must give the same branch energies and ports under `follow`/`:follow`. Pinned in Task 6.
5. A mixed branch (a `FockBasis` child beside a node child, as in the full zero-pi) runs the fused diagonal, an ITensor dense factor and a banded kernel inside one operator and still equals the Kronecker reference. Pinned in Task 2 and Task 3.

## Catalogue coverage

`docs/spikes/2026-10-16-pathological-test-catalogue/README.md` (Eli, 2026-10-16) lists pathological cases by identifier with corrected expectations. Each task below carries a "Catalogue cases" line; the implementer adds one testset per listed case, named by the identifier, following that case's oracle and expected behavior in the catalogue (a case whose contract is an informative rejection asserts the error). The cases that need the whole pipeline are Task 10.

| Task | Cases |
|---|---|
| 2, 3 | O1–O5 (O5's transport part in Task 6); F7 as a term-and-application test |
| 4 | S1–S8; F1 and F2 at operator level |
| 5 | F1–F3, F8 (small), S9 |
| 6 | T1–T3, F5, F10 (T2's changed-basis rejection; T4 in Task 10) |
| 7 | O1–O5 and S1–S8 representative cases on the device; F3, F7 small |
| 8 | B1; larger F3, F6, F8, F9 as measurements, not gates |
| 9 | none; given systems are new |
| 10 | F4, F6 (small), F8, F10, P1, P2, T4; S3, S4, S6, S7 if Task 4 left them; gated larger F3, F6, F8, F9 |


---

### Task 1: Dependencies, compat and the extension skeleton

**Files:**
- Modify: `Project.toml` (deps, weakdeps, extensions, compat), `test/Project.toml` (add `Adapt` if tests need it)
- Create: `ext/KronFreeCircuitsCUDAExt.jl` (empty module body for now), `test/cuda/Project.toml`, `test/cuda/runtests.jl` (placeholder that loads `CUDA`, `cuTENSOR`, `KronFreeCircuits` and runs an empty testset)
- Modify: `src/KronFreeCircuits.jl` (`using ITensors: ITensors`, `using LOBPCGEigensolver: LOBPCGEigensolver`, `using Adapt: adapt`)

**Interfaces:**
- Produces: the resolved environment later tasks build on. Nothing else.

- [ ] **Step 1: Add dependencies.** `Pkg.add(["ITensors", "LOBPCGEigensolver", "Adapt"])` in the package environment. In `Project.toml` set compat `ITensors = "0.9"`, `LOBPCGEigensolver = "=0.1.4"`, `Adapt = "4"`, `KernelAbstractions = "0.9, 0.10"`. Add `[weakdeps] CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"`, `cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"`, `[extensions] KronFreeCircuitsCUDAExt = ["CUDA", "cuTENSOR"]`, compat `CUDA = "6"`, `cuTENSOR = "6"`.

- [ ] **Step 2: Verify resolution with CUDA present.** In a scratch environment: `Pkg.develop(path=".")`, `Pkg.add(["CUDA", "cuTENSOR"])`, `Pkg.resolve()`. Expected: resolves (KernelAbstractions lands on 0.9.x). Record the resolved versions of KernelAbstractions, CUDA, cuTENSOR, ITensors, NDTensors in the commit message. Then `test/cuda/Project.toml` gets `KronFreeCircuits` (dev path `../..`), `CUDA`, `cuTENSOR`, `Test`, `LinearAlgebra`, `Random`.

- [ ] **Step 3: Run the default suite under both KernelAbstractions lines.** `Pkg.test()` with the package Manifest at KA 0.10 (current), then `Pkg.add(name="KernelAbstractions", version="0.9")` and `Pkg.test()` again, then restore 0.10 (or leave the Manifest at whichever passes cleanly; it is git-ignored). Expected: both pass. If 0.10 fails with a POCL `CLError`, note it in the ledger and leave the Manifest at 0.9.

- [ ] **Step 4: Commit** `Project.toml`, `ext/`, `test/cuda/`, `src/KronFreeCircuits.jl`.

---

### Task 2: Device-owned factor data and a fused diagonal in `KronFreeOperator`

**Files:**
- Modify: `src/apply.jl` (`KronFreeOperator`, `mul!`, `diagonal`, `Matrix`), `src/localops.jl` only if a helper is needed
- Test: `test/test_apply.jl`

**Interfaces:**
- Consumes: `AxisTerm(coeff, factors)` unchanged; `LocalOp` types unchanged; `kron_reference` from `test/helpers.jl`.
- Produces:
  - `KronFreeOperator(dims, terms::AbstractVector{AxisTerm}; constant=0.0, block_capacity=1, on=Array)`. `on` is the storage array type; every array the operator owns is `adapt(on, …)` at construction. Fields: `dims::Vector{Int}`, `diag` (state-shaped vector, length `prod(dims)`, the sum over all terms whose factors are all `DiagonalOp` of `coeff * ⊗_k d_k` with ones on absent axes, plus `constant`), `terms::Vector{<:Any}` (the remaining terms as device-ready records: coefficient plus, per factor, `(axis, op::LocalOp, data)` where `data` is the adapted array the apply needs: the matrix of a `DenseOp`, the weight matrix and offsets of a `BandedOp`, nothing for `ShiftOp`), `buf1`, `buf2` (shape `(dims..., block_capacity)`, storage `on`), `on`. The `constant` field is removed; `constant` is folded into `diag`.
  - `LinearAlgebra.mul!(y, H, x)`: `Y .= reshape(diag) .* X` first (replacing `fill!`), then the remaining terms accumulate as before, through the same structure-dispatched application now reading factor data from the term record (Task 3 replaces the `DenseOp` branch of that dispatch). `H * x` allocates on `on`.
  - `diagonal(H)::Vector` = `Array(H.diag) + Σ_remaining terms (coeff * ⊗ diagonal(op))`, returned on the host. `Base.Matrix(H)` unchanged in behavior (host only). `Base.eltype(H) = eltype(H.diag)`.

**Catalogue cases:** O3 (extreme shifts and unusual bands), O4 (fused diagonal and cancellation, including exactly cancelling terms and complex coefficients with a real Hermitian diagonal), O5 (workspace ownership: independent instances do not share scratch; factor data never mutated).

- [ ] **Step 1: Write the failing tests** in `test/test_apply.jl`:

```julia
@testset "fused diagonal part" begin
    rng = MersenneTwister(31)
    dims = [3, 4, 2]
    d1 = DiagonalOp(ComplexF64.(1:3)); d2 = DiagonalOp(ComplexF64.(randn(rng, 4))); d3 = DiagonalOp(ComplexF64.(randn(rng, 2)))
    M = DenseOp(randn(rng, ComplexF64, 3, 3)); B = BandedOp(4, [-1, 1], [ones(4), ones(4)])
    terms = [AxisTerm(0.5, [(1, d1)]), AxisTerm(-1.5, [(2, d2), (3, d3)]),    # pure diagonal: fused
             AxisTerm(0.7, [(1, M), (2, B)]), AxisTerm(0.2, [(3, d3), (1, M)])] # not fused
    H = KronFreeOperator(dims, terms; constant=0.3)
    ref = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms]) + 0.3I
    @test length(H.terms) == 2
    @test H.diag ≈ 0.5 .* kron(ones(2), ones(4), d1.d) .- 1.5 .* kron(d3.d, d2.d, ones(3)) .+ 0.3
    @test diagonal(H) ≈ diag(ref)
    x = randn(rng, ComplexF64, 24); @test H * x ≈ ref * x
    X = randn(rng, ComplexF64, 24, 3); @test mul!(similar(X), H, X) ≈ ref * X
    @test Matrix(H) ≈ ref
    Hd = KronFreeOperator(dims, terms[1:2]; constant=0.3)             # diagonal-only operator
    refd = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms[1:2]]) + 0.3I
    @test isempty(Hd.terms) && Hd * x ≈ refd * x && Hd.diag ≈ diag(refd)
    @test eltype(H) == ComplexF64
    @test_throws ArgumentError KronFreeOperator(dims, terms; block_capacity=0)
end

@testset "operator owns adapted factor data" begin
    H = KronFreeOperator([3, 2], [AxisTerm(1.0, [(1, DenseOp(ComplexF64[1 2 0; 0 1 0; 0 0 3]))])]; on=Array)
    @test H.on === Array && H.buf1 isa Array && H.diag isa Vector
    @test H.terms[1].factors[1][3] isa Matrix   # the dense factor's data lives in the operator
end
```

- [ ] **Step 2: Run** `julia --project -e 'using Pkg; Pkg.test()'` (or include the file) and confirm the new testsets fail (no `diag` field, no `on` keyword).

- [ ] **Step 3: Implement** the new `KronFreeOperator` layout in `src/apply.jl`. Keep `_active_buffer` and the typed `fill!` behavior (the broadcast into the typed `y` replaces `fill!`). Grep for `H.constant` and `.terms` users across `src/` and `test/` and update them (`Matrix(H)`, `diagonal`, any materialize use).

- [ ] **Step 4: Run the full default suite.** Expected: all pass, including the existing workspace and allocation testsets (the allocation bound `< sizeof(X)` must still hold).

- [ ] **Step 5: Commit** `src/apply.jl`, `test/test_apply.jl` (and any touched callers).

---

### Task 3: ITensor application of `DenseOp` factors inside the operator

**Files:**
- Modify: `src/apply.jl` (the `DenseOp` branch of the term chain; cached `ITensors.Index` objects on the operator)
- Test: `test/test_apply.jl`

**Interfaces:**
- Consumes: `KronFreeOperator` from Task 2 (`terms` records with the dense matrix as adapted data; `buf1`, `buf2`; `dims`).
- Produces: inside `KronFreeOperator`, two new fields: `inds::Vector{ITensors.Index}` (one `Index(dims[k])` per axis, created at construction) and `blockinds::Dict{Int,ITensors.Index}` (created on demand per block width). The chain's `DenseOp` application is `_apply_dense!(dst, data, src, k, H, b)` using `ITensors.itensor(array, inds...)` wrappers over `src`, `dst` (and scratch) and the matrix (`itensor(data, inds[k]', inds[k])`), with `ITensors.mul!` / contraction into the wrapped destination; permutations, when the contraction needs them, go through explicit `permutedims!` into the operator's scratch (`docs/spikes/2026-10-16-itensor-audit/buffer-layout.jl` is the reference layout for a middle axis). The public `apply_axis!(Y, ::DenseOp, X, k)` keeps its gemm implementation and is the reference the ITensor path is tested against.

**Catalogue cases:** O1 (axis identity with equal dimensions `(4,4,4)`, singleton axes `(1,5,3)`, distinct nonsymmetric complex factors per axis), O2 (block widths `8 → 3 → 1 → 12 → 2` with sentinels and stale scratch), F7 (the complex three-mode cosine as an adjoint pair of terms, Hermiticity of the sum, three-axis chain; assert the composite-port rejection for the unsupported grouping).

- [ ] **Step 1: Write the failing tests** in `test/test_apply.jl`:

```julia
@testset "ITensor dense factors match the gemm reference" begin
    rng = MersenneTwister(41)
    dims = [5, 6, 4]
    ops = [DenseOp(randn(rng, ComplexF64, d, d)) for d in dims]
    for k in 1:3, width in (1, 3)      # first, middle and last axis; vector and block
        H = KronFreeOperator(dims, [AxisTerm(1.0, [(k, ops[k])])]; block_capacity=3)
        X = randn(rng, ComplexF64, prod(dims), width)
        Y = similar(X); Yref = similar(X)
        apply_axis!(reshape(Yref, dims..., width), ops[k], reshape(X, dims..., width), k)
        @test mul!(Y, H, X) ≈ Yref
    end
    H2 = KronFreeOperator(dims, [AxisTerm(0.3 + 0.1im, [(1, ops[1]), (3, ops[3])]), AxisTerm(-0.8, [(2, ops[2]), (3, ops[3])])])
    ref = kron_reference(dims, [(0.3 + 0.1im, [(1, dense(ops[1])), (3, dense(ops[3]))]), (-0.8, [(2, dense(ops[2])), (3, dense(ops[3]))])])
    X = randn(rng, ComplexF64, prod(dims), 2)
    @test mul!(similar(X), H2, X) ≈ ref * X
    @test Matrix(H2) ≈ ref
end

@testset "ITensor dense path: offset views, input preserved, no state-sized allocation" begin
    rng = MersenneTwister(42)
    dims = [60, 50]
    H = KronFreeOperator(dims, [AxisTerm(1.0, [(1, DenseOp(randn(rng, ComplexF64, 60, 60))), (2, DenseOp(randn(rng, ComplexF64, 50, 50)))])]; block_capacity=8)
    Xp = randn(rng, ComplexF64, 3000, 8); Yp = fill(ComplexF64(7), 3000, 8)
    X = view(Xp, :, 3:7); Y = view(Yp, :, 3:7)
    saved = copy(Xp)
    mul!(Y, H, X); mul!(Y, H, X)
    @test (@allocated mul!(Y, H, X)) < 64 * 1024          # 240 kB state block; metadata only
    @test Xp == saved
    @test all(==(7), Yp[:, [1, 2, 8]])
    @test Y ≈ H * copy(X)
end

@testset "mixed branch operator: fused diagonal + ITensor dense + banded kernel" begin
    rng = MersenneTwister(43)
    dims = [4, 7]
    E = DiagonalOp(ComplexF64.(sort(randn(rng, 4)))); P = DenseOp(randn(rng, ComplexF64, 4, 4))
    B = BandedOp(7, [-1, 0, 1], [ones(7), 2 .* ones(7), ones(7)])
    terms = [AxisTerm(1.0, [(1, E)]), AxisTerm(0.1, [(1, P), (2, B)])]
    H = KronFreeOperator(dims, terms)
    ref = kron_reference(dims, [(t.coeff, [(a, dense(op)) for (a, op) in t.factors]) for t in terms])
    X = randn(rng, ComplexF64, 28, 3)
    @test mul!(similar(X), H, X) ≈ ref * X
    @test diagonal(H) ≈ diag(ref)
end
```

- [ ] **Step 2: Run** the new testsets. The correctness ones may already pass through Task 2's gemm chain; confirm that the allocation bound fails or that `H.inds` is undefined, so the ITensor path is observably absent before Step 3.

- [ ] **Step 3: Implement** `_apply_dense!` and the index caching. Make `_accumulate_term!` call it for `DenseOp` factors and keep the kernels/broadcasts for the others.

- [ ] **Step 4: Run the full default suite.** Expected: pass; the existing `test_materialize.jl` branch tests exercise the path through `materialize`.

- [ ] **Step 5: Commit** `src/apply.jl`, `test/test_apply.jl`.

---

### Task 4: `LOBPCG` solver with fixed and adaptive per-Ritz Jacobi

**Files:**
- Create: `src/lobpcg.jl` (preconditioners and the solver wrapper), included after `solvers.jl`
- Modify: `src/tree.jl` (`struct LOBPCG <: Solver`), `src/solvers.jl` (`solve(::LOBPCG, …)`, `_dense_too_large` message), `src/KronFreeCircuits.jl` (export `LOBPCG`)
- Test: `test/test_solvers.jl`

**Interfaces:**
- Consumes: `KronFreeOperator` (`mul!`, `diagonal`, `size`, `eltype`, `on`), `LOBPCGEigensolver.lobpcg`, `LOBPCGEigensolver.precondprep!`.
- Produces:
  - `struct LOBPCG <: Solver` with fields `tol::Float64`, `maxiter::Int`, `preconditioner::Symbol` (`:ritz`, `:jacobi`, `:none`), `eta::Float64`, `guard::Int`, `on::Any`, `callback::Any`; constructor `LOBPCG(; tol=1e-8, maxiter=500, preconditioner=:ritz, eta=0.01, guard=4, on=Array, callback=nothing)`. `eta` is in the Hamiltonian's energy units; the docstring says the default is taken from the audit fixture (GHz) and must be set for other scales. `guard` extra random columns are appended to every block (cold or warm), convergence is checked on the lowest `m` only (`n_conv_check=m`), and the lowest `m` converged pairs are returned; the docstring states that guards make exact excited-state seeds and sector-confined seeds recoverable in practice and are not a completeness guarantee.
  - `struct FixedJacobi{V}` with field `inv::V` (= `1 ./ (D .- σ)`, `σ = minimum(D) - 1`, real) and `LinearAlgebra.ldiv!(P::FixedJacobi, R)` = `R .*= P.inv`.
  - `mutable struct RitzJacobi{V,W}` with fields `D::V` (real diagonal), `theta::W` (real, length = block capacity), `eta`, `locked::Int`, `width::Int`, `ready::Bool`, `trace::Union{Nothing,Vector{Any}}`; `RitzJacobi(D, capacity; eta, trace=false)`; `LOBPCGEigensolver.precondprep!(P::RitzJacobi, X)` (on first call, `theta[j] = Σ_i D[i]|X[i,j]|² / Σ_i |X[i,j]|²` per active column through `_diagonal_expectation!(theta, D, X)`, a column loop on host arrays; afterwards no-op except width bookkeeping; when `trace` is on, push `(width, copy(theta[1:width]))`), `LinearAlgebra.ldiv!(P::RitzJacobi, R)` = `R ./= hypot.(P.D .- transpose(view(P.theta, 1:width)), P.eta)`, and `_ritz_update!(P::RitzJacobi, info)` (copy the active Ritz values from `info.λs` into `P.theta`, dropping the newly locked prefix; update `P.locked`). The exact copy logic follows `docs/spikes/2026-10-16-itensor-audit/ritz-preconditioner.jl` (`LO.precondprep!`, `ldiv!`, the callback block) and `RITZ-PRECONDITIONER-REPORT.md` ("Question and implementation").
  - `solver_callback(P, user)` returns the single closure handed to `lobpcg`: `info -> begin; user === nothing || user(info); P isa RitzJacobi && _ritz_update!(P, info); nothing; end`. Both callbacks receive the original `info`; return values are ignored. Never compose with `∘`, which would feed the user's return value to the cache update.
  - `_full_diagonal(H::KronFreeOperator)` returns the real diagonal of `H` as an array on `H.on` (fused part plus the diagonals of the remaining terms, computed on the device); `diagonal(H)` keeps returning a host `Vector` and is `Array(_full_diagonal(H))`. The fused `H.diag` used by `mul!` never includes diagonals of terms that stay in the contraction sum (Task 2); only `_full_diagonal` adds them.
  - `lobpcg_solve(s::LOBPCG, H::KronFreeOperator, m::Int; X0=nothing) -> (E::Vector{Float64}, V::Matrix, result)` where `result` is `lobpcg`'s named tuple (`λ, X, residual_norms, n_matvec, …`) for the full block; `X0` is an `N×m` host block or `nothing` (then `randn(Random.default_rng(), eltype(H), N, m)`); the block handed to `lobpcg` is `adapt(s.on, hcat(X0, randn(N, g)))` with `g = min(s.guard, (N - 5) ÷ 3 - m)` (the guard shrinks to fit `N ≥ 3(m + g) + 5`); `H` is replaced by `with_block_capacity(H, m + g)` when its capacity is smaller; `miniter` stays at the library default of 1. Throws `ArgumentError("LOBPCG needs N ≥ 3m + 5 … use Dense()")` when `size(H, 1) < 3m + 5`, `ArgumentError` for `m < 1`.
    Recovery policy (bounded): if `lobpcg` throws with a warm-start block (the library's orthogonalization can fail on a block whose residual is exactly zero), warn once and retry from a random block; a second failure propagates. Nonconvergence (`any(result.residual_norms[1:m] .> s.tol)`) raises an `ErrorException` naming the unconverged count and the maximum residual, like `Lanczos`; it is never retried silently. `E` and `V` return as host arrays, ascending, the lowest `m` of the block.
  - `solve(s::LOBPCG, H, m)` = the first two outputs of `lobpcg_solve`, same return types as the other solvers.
  - `with_block_capacity(H::KronFreeOperator, m::Integer)::KronFreeOperator` in `src/apply.jl`: the same operator (shared factor data and `diag`) with scratch reserved for `m` columns.
  - `check_eigenpairs(H, E, V; tol)` in `test/helpers.jl`: asserts `norm(H * V[:, j] - E[j] * V[:, j]) < tol` for every column and `norm(V' * V - I) < 1e-10`; used by every solver test from here on.
  - `_dense_too_large` warning text in `materialize.jl` changes "Lanczos is advised" to "LOBPCG is advised".
  - `Lanczos` docstring gains the sentence: "A single-vector Krylov space does not resolve repeated eigenvalues (KrylovKit documents this); use `LOBPCG` or `Dense` where exact degeneracies are expected."

**Catalogue cases:** S1–S8 beyond the testsets above: S3 (exact degeneracy, `H = 0`, `H = cI`, tight clusters with a cutoff inside a cluster; projector of the complete lower subspace and membership in the boundary eigenspace), S4 (exact resonance `D[i] == θ_j` gives the finite `R/η`; zero, negative, NaN and infinite `η` rejected before iterating), S5 (complete locking and a single remaining active column; no zero-width backend call), S6 (`aH + bI` with scaled tolerances; the shift finite and strictly below the diagonal), S7 (constant diagonal with strong off-diagonal coupling: valid output or explicit nonconvergence for all three preconditioners), S8 (`m = 1`, `N` at and just below `3m + 5`, malformed `X0` dimensions); F1 and F2 at operator level (free rotor through `n_g = 1/2`; `cos 2φ` sectors with a seed confined to one parity, using the oracle and window correction the catalogue gives).

- [ ] **Step 1: Write the failing tests** in `test/test_solvers.jl` (and `check_eigenpairs` in `test/helpers.jl`; add `LOBPCGEigensolver` to `test/Project.toml`):

```julia
using KronFreeCircuits: LOBPCG, lobpcg_solve, FixedJacobi, RitzJacobi, solver_callback, with_block_capacity, _full_diagonal
import LOBPCGEigensolver

@testset "LOBPCG agrees with Dense; all preconditioners; independent validation" begin
    Random.seed!(7)
    H = three_oscillators(6)                       # N = 216
    Ed, Vd = solve(Dense(), H, 0)
    for pc in (:none, :jacobi, :ritz)
        E, V = solve(LOBPCG(tol=1e-9, preconditioner=pc), H, 8)
        @test E ≈ Ed[1:8] atol=1e-7
        check_eigenpairs(H, E, V; tol=1e-8)
        @test V * V' ≈ Vd[:, 1:8] * Vd[:, 1:8]' atol=1e-7       # same subspace
        @test issorted(E) && typeof((E, V)) == Tuple{Vector{Float64},Matrix{ComplexF64}}
    end
    @test Array(_full_diagonal(H)) ≈ diagonal(H)
end

@testset "LOBPCG argument errors and nonconvergence" begin
    H = three_oscillators(3)                       # N = 27
    @test_throws ArgumentError solve(LOBPCG(), H, 8)          # 3*8+5 > 27
    err = try solve(LOBPCG(), H, 8); nothing catch e; e end
    @test occursin("Dense", err.msg)
    @test_throws ArgumentError solve(LOBPCG(), H, 0)
    @test_throws ArgumentError LOBPCG(preconditioner=:magic)
    @test_throws ArgumentError LOBPCG(guard=-1)
    H6 = three_oscillators(6)
    err = try solve(LOBPCG(tol=1e-14, maxiter=1), H6, 5); nothing catch e; e end
    @test err isa ErrorException && occursin("converged", err.msg) && occursin("residual", err.msg)
    # the guard shrinks to fit: N = 27, m = 7 leaves room for g = 0
    E, V = solve(LOBPCG(tol=1e-10, guard=4), H, 7)
    @test E ≈ solve(Dense(), H, 7)[1] atol=1e-8
end

@testset "repeated eigenvalue is recovered twice (projector, not just energies)" begin
    H = KronFreeOperator([12], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))])])
    Random.seed!(3)
    E, V = solve(LOBPCG(tol=1e-10, guard=0), H, 2)
    Ed, Vd = solve(Dense(), H, 2)
    @test E ≈ [0.0, 0.0] atol=1e-8 && Ed ≈ [0.0, 0.0] atol=1e-12
    @test V * V' ≈ Vd * Vd' atol=1e-8
    check_eigenpairs(H, E, V; tol=1e-9)
end

@testset "user callbacks of any return type run with the original info" begin
    Random.seed!(9)
    H = three_oscillators(6)
    seen = Int[]
    for cb in (info -> push!(seen, info.n_iter), info -> nothing, info -> info)
        E, _ = solve(LOBPCG(tol=1e-9, preconditioner=:ritz, callback=cb), H, 6)
        @test E ≈ solve(Dense(), H, 6)[1] atol=1e-7
    end
    @test length(seen) >= 2                          # the first callback recorded iterations
end

@testset "warm starts: exact lowest, exact excited, rank-deficient and sector-confined seeds" begin
    Random.seed!(11)
    H = three_oscillators(6)
    Ed, Vd = solve(Dense(), H, 0)
    E, V, cold = lobpcg_solve(LOBPCG(tol=1e-9), H, 6)
    _, _, warm = lobpcg_solve(LOBPCG(tol=1e-9), H, 6; X0=V)
    @test warm.n_matvec < cold.n_matvec ÷ 2
    for X0 in (Vd[:, 1:6],                          # exact lowest eigenvectors
               Vd[:, 3:8],                          # exact excited subspace: guards must bring the lowest two back
               repeat(Vd[:, 1:1], 1, 6))            # rank-deficient block
        E2, V2 = solve(LOBPCG(tol=1e-10, guard=4), H, 6; X0=X0)
        @test E2 ≈ Ed[1:6] atol=1e-8
        @test V2 * V2' ≈ Vd[:, 1:6] * Vd[:, 1:6]' atol=1e-7
        check_eigenpairs(H, E2, V2; tol=1e-9)
    end
    # sector-confined seed: a diagonal operator whose even sites hold the ground state; N = 40 leaves
    # room for the full guard (g = min(4, (40 - 5) ÷ 3 - 3) = 4)
    d = zeros(40); d[1:2:40] .= 2:2:40; d[2:2:40] .= 0.5:2:38.5
    Hs = KronFreeOperator([40], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(d)))])])
    X0 = Matrix{ComplexF64}(I, 40, 40)[:, 1:2:5]     # exact eigenvectors of odd sites (energies 2, 4, 6): zero residual
    Es, Vs = solve(LOBPCG(tol=1e-10, guard=4), Hs, 3; X0=X0)
    @test Es ≈ [0.5, 2.0, 2.5] atol=1e-8               # the guards must find site 2 (0.5) and site 4 (2.5)
    check_eigenpairs(Hs, Es, Vs; tol=1e-9)
end

@testset "Ritz cache stays aligned with the solver's active columns, through locking" begin
    Random.seed!(5)
    H = three_oscillators(6); m = 10
    D = real.(diagonal(H))
    P = RitzJacobi(D, m; eta=0.05, trace=true)
    seen = Any[]
    cb = info -> push!(seen, (info.n_iter, info.n_locked, copy(info.λs)))
    X0 = randn(ComplexF64, size(H, 1), m)
    Hb = with_block_capacity(H, m)
    res = LOBPCGEigensolver.lobpcg(Hb, X0, I, P, 1e-9, 300; callback=solver_callback(P, cb))
    @test res.λ ≈ solve(Dense(), H, m)[1] atol=1e-7
    @test length(seen) >= 5                                   # enough iterations to mean something
    @test any(w < m for (w, _) in P.trace)                    # locking actually shrank the active block
    @test seen[end][2] == m
    # From the second prepare onward, the cached values for the active columns are exactly the
    # previous iteration's Ritz values for those same columns (locked prefix removed).
    for t in 2:min(length(P.trace), length(seen))
        width, theta = P.trace[t]
        _, _, λs = seen[t - 1]
        @test theta == λs[end - width + 1:end]
    end
end
```

- [ ] **Step 2: Run** and confirm failures (`LOBPCG` undefined).

- [ ] **Step 3: Implement** `src/lobpcg.jl` and the `LOBPCG` type. In `lobpcg_solve`: `D = _full_diagonal(H)` on `s.on`; choose the preconditioner by `s.preconditioner`; build the guarded block; call `LOBPCGEigensolver.lobpcg(Hb, Xdev, I, P, s.tol, s.maxiter; n_conv_check=m, callback=solver_callback(P, s.callback))` inside the recovery policy; validate; return `Vector{Float64}(res.λ[1:m])`, `Matrix(Array(res.X[:, 1:m]))`.

- [ ] **Step 4: Run the full default suite.** Expected: pass.

- [ ] **Step 5: Commit** `src/lobpcg.jl`, `src/tree.jl`, `src/solvers.jl`, `src/KronFreeCircuits.jl`, `test/test_solvers.jl`.

---

### Task 5: `LOBPCG` nodes in materialization with warm starts

**Files:**
- Modify: `src/materialize.jl` (operator construction with capacity and storage; initial block from `previous`), `src/tree.jl` (Node docstring: `solver` is `Dense()`, `Lanczos()` or `LOBPCG()`)
- Test: `test/test_materialize.jl`

**Interfaces:**
- Consumes: `lobpcg_solve`, `LOBPCG.on`, `KronFreeOperator(…; block_capacity, on)`, `prev::MaterializedNode` (`vectors`, `dims`).
- Produces: in `_materialize`, `storage(::Solver) = Array`, `storage(s::LOBPCG) = s.on`; `H = KronFreeOperator(dims, terms; constant, block_capacity=(node.solver isa LOBPCG ? m + node.solver.guard : 1), on=storage(node.solver))`; `_initial_block(prev, dims, m)` = `nothing` when `prev === nothing` or `prev.dims != dims`, else a copy of `prev.vectors` padded on the right with `randn` columns to `m` (truncated to `m` columns when it has more); `solve(::LOBPCG, …)` receives it as `X0` (the other solvers ignore it). The guess is the raw previous coefficients, never the transported reference of Task 6 (spec §20c, decision 2). Documented limitation: only the *kept* vectors survive in a `MaterializedNode`, so when `compute > keep` the remaining columns start random and the block is only partially warm; retaining the whole computed block in an optional cache is future work. The solver's guard columns and its warm-start recovery policy (Task 4) apply unchanged.

**Catalogue cases:** F1 and F2 through the public path with `previous` across the crossing; F3 (identical uncoupled zero-pis at `g = 0` on the small fixture: sorted pairwise sums, exchange degeneracy, a cutoff inside a cluster); F8 (deliberately inadequate truncation: hierarchical result equals Dense on the same projected Hamiltonian while both differ from the primitive reference); S9 (reordered `indices` selections, increased and decreased `compute`).

- [ ] **Step 1: Write the failing tests** in `test/test_materialize.jl`:

```julia
@testset "LOBPCG at a leaf and at a branch equals Dense" begin
    θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    children = [:theta => ChargeBasis(6), :phi => PhaseGrid(-6π, 6π, 101; order=6)]
    dense = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(children; solver=Dense(), keep=lowest(8)), spec_zp, θ)
    Random.seed!(21)
    lob = @test_logs (:warn, r"edge") match_mode=:any materialize(Node(children; solver=LOBPCG(tol=1e-10), compute=8, keep=lowest(8)), spec_zp, θ)
    @test energies(lob) ≈ energies(dense) atol=1e-7
    # branch: the two-zero-pi small fixture tree used in test_zero_pi_reference.jl, with the top solver swapped
    θ2 = two_zero_pi_params(; g=0.05); spec2 = load_spec(fixture("two_zero_pi.json"))
    leaf(i) = Node([Symbol("theta$i") => ChargeBasis(6), Symbol("phi$i") => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(10))
    top_dense = Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(12))
    top_lob = Node([leaf(1), leaf(2)]; solver=LOBPCG(tol=1e-10), compute=12, keep=lowest(12))
    sd = @test_logs (:warn, r"edge") match_mode=:any materialize(top_dense, spec2, θ2)
    sl = @test_logs (:warn, r"edge") match_mode=:any materialize(top_lob, spec2, θ2)
    @test energies(sl) ≈ energies(sd) atol=1e-7
end

@testset "previous solution warm-starts a LOBPCG node" begin
    iters = Int[]
    cb = info -> (push!(iters, info.n_iter); info)
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    tree() = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))];
                  solver=LOBPCG(tol=1e-10, callback=cb), compute=6, keep=lowest(6))
    Random.seed!(2)
    s1 = materialize(tree(), spec, θ); cold = maximum(iters); empty!(iters)
    s2 = materialize(tree(), spec, two_transmons_params(; g=0.101); previous=s1); warm = maximum(iters)
    @test warm < cold
    @test energies(s2) ≈ energies(materialize(Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))]; solver=Dense(), keep=lowest(6)), spec, two_transmons_params(; g=0.101))) atol=1e-7
end
```

(Check the mode names in `two_transmons.json` and `two_zero_pi.json` before writing; use the fixture's names.)

Add, in the same file:

```julia
@testset "warm start: unchanged Hamiltonian, partial block, exact seeds" begin
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    iters = Int[]; cb = info -> push!(iters, info.n_iter)          # returns a Vector, not info: must be tolerated
    leaves() = [Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(6)), Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(6))]
    tree(; keep) = Node(leaves(); solver=LOBPCG(tol=1e-10, callback=cb), compute=8, keep=keep)
    Random.seed!(4)
    s1 = materialize(tree(; keep=lowest(8)), spec, θ); cold = maximum(iters); empty!(iters)
    s_same = materialize(tree(; keep=lowest(8)), spec, θ; previous=s1)                 # exact eigenvectors as the seed
    @test energies(s_same) ≈ energies(s1) atol=1e-9 && maximum(iters) < cold; empty!(iters)
    s_part = materialize(tree(; keep=lowest(3)), spec, θ)                              # compute 8, keep 3: partial warm start
    s_part2 = materialize(tree(; keep=lowest(3)), spec, two_transmons_params(; g=0.101); previous=s_part)
    @test energies(s_part2) ≈ energies(materialize(Node(leaves(); solver=Dense(), keep=lowest(3)), spec, two_transmons_params(; g=0.101))) atol=1e-7
    V = eigenvectors(s1)                                                               # 36 × 8 host matrix
    check_eigenpairs(KronFreeCircuits.operator(s1), energies(s1), V; tol=1e-9)
end
```

(`operator(sys)` rebuilds the root's `KronFreeOperator` from the system; add it to `materialize.jl` if absent, or build the operator in the test through the term-list entry point. `check_eigenpairs` is the helper added in Task 4.)

- [ ] **Step 2: Run** and confirm failure (`LOBPCG` not accepted by `_materialize`, no `X0`).

- [ ] **Step 3: Implement** the storage/capacity construction and the initial block. Update the `Node` docstring and the large-dense warning text.

- [ ] **Step 4: Run the full default suite** and the slow suite (`KRONFREE_SLOW_TESTS=1`). Expected: pass.

- [ ] **Step 5: Commit** `src/materialize.jl`, `src/tree.jl`, `test/test_materialize.jl`.

---

### Task 6: Physical overlap transport for `follow` and `:follow` at branches

**Files:**
- Modify: `src/materialize.jl` (transport of the previous kept vectors), `src/selection.jl` (`SelectionContext` gains `reference`; `Follow` reads it)
- Test: `test/test_selection.jl`, `test/test_materialize.jl`

**Interfaces:**
- Consumes: `prev::MaterializedNode` and the current `children` (materialized nodes or bases), `apply_axis!` with a `DenseOp`.
- Produces:
  - `_child_overlap(prev::MaterializedNode, cur::MaterializedNode)::Matrix{ComplexF64}` = `⟨prev kept i | cur kept j⟩` in the physical space: at a leaf `prev.vectors' * cur.vectors`; at a branch `_transport_previous(prev, cur.children, cur.dims)' * cur.vectors`. Requires `prev.dims == cur.dims` and, on every basis axis, `prev.children[a] == cur.children[a]` (bases are immutable isbits structs, so `==` compares range, point count, order, cutoff, `phi_zpf`); otherwise throws `ArgumentError` saying that physical overlaps across different primitive bases are not implemented. Equal dimensions alone never establish the same basis.
  - `_transport_previous(prev::MaterializedNode, children, dims)::Matrix{ComplexF64}` = `(⊗_a S_a') prev.vectors`, applying `DenseOp(Matrix(S_a'))` along each axis `a` whose child is a `MaterializedNode` (`S_a = _child_overlap(prev.children[a], children[a])`), identity on basis axes (after the same basis-equality check). `apply_axis!` forbids aliasing, so the chain copies `prev.vectors` into a fresh array and alternates between two distinct work arrays, one application per transformed axis; `prev.vectors` is never written. At a leaf it returns a copy of `prev.vectors`.
  - `SelectionContext(energies, vectors, ports, previous, reference)`; the four-argument constructor keeps working with `reference = previous === nothing ? nothing : previous.vectors`. `select(::Follow, ctx)` uses `ctx.reference` where it used `ctx.previous.vectors`.
  - In `_materialize`, when `uses_prev` and `prev !== nothing`: `R̃ = _transport_previous(prev, children, dims)` once; it is the `reference` of the `SelectionContext` and the reference handed to `fix_phases!(V, :follow, mask, R̃)`. The basis-equality check runs at every node that uses `prev`, leaves included: a `follow` at a leaf whose grid changed is as wrong as at a branch.

**Catalogue cases:** T1 (gauge covariance through two levels with phases, permutations and rotations of exactly degenerate clusters, two children changed at once, unequal depths), T2 (the rejection tests above), T3 (competing assignments, ties, near-zero assigned overlap, more references than computed states), F5 (a sweep through resonance: fine steps, one jump, a step landing on resonance; the low-overlap warning actually fires), F10 (a raw basis child beside a materialized branch, reversed order, an extra level on one side).

- [ ] **Step 1: Write the failing tests.** In `test/test_selection.jl` a unit test against brute force on hand-built nodes (construct `MaterializedNode` positionally; `ports` empty, `peak_component`/`peak_amplitudes` arbitrary valid values):

```julia
@testset "physical overlap through a changed child basis" begin
    using KronFreeCircuits: MaterializedNode, _child_overlap, _transport_previous, Node, ChargeBasis, lowest, Port
    rng = MersenneTwister(51)
    orth(n, k) = Matrix(qr(randn(rng, ComplexF64, n, k)).Q)[:, 1:k]
    leafnode = Node([:a => ChargeBasis(2)]; keep=lowest(3))          # basis dim 5
    mk(node, dims, V, children) = MaterializedNode(node, dims, zeros(size(V, 2)), V, collect(1:size(V, 2)),
                                                  Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}(), children, ones(Int, size(V, 2)), Matrix{ComplexF64}(I, size(V, 2), size(V, 2)))
    Vold = orth(5, 3); Vnew = Vold[:, [2, 1, 3]] .* [1, -1, im]'       # permuted and rephased leaf basis
    lo, ln = mk(leafnode, [5], Vold, Any[ChargeBasis(2)]), mk(leafnode, [5], Vnew, Any[ChargeBasis(2)])
    S = _child_overlap(lo, ln)
    @test S ≈ Vold' * Vnew
    bnode = Node([leafnode, :b => ChargeBasis(1)]; keep=lowest(4))     # dims [3, 3]
    Cold = orth(9, 4)
    bo = mk(bnode, [3, 3], Cold, Any[lo, ChargeBasis(1)])
    R̃ = _transport_previous(bo, Any[ln, ChargeBasis(1)], [3, 3])
    @test R̃ ≈ kron(Matrix{ComplexF64}(I, 3, 3), S') * Cold            # axis 1 fastest: kron(I_b, S') acts on axis 1
    Cnew = kron(Matrix{ComplexF64}(I, 3, 3), S') * Cold               # the same physical states in the new child basis
    bn = mk(bnode, [3, 3], Cnew, Any[ln, ChargeBasis(1)])
    @test _child_overlap(bo, bn) ≈ Matrix{ComplexF64}(I, 4, 4) atol=1e-12
    @test_throws ArgumentError _child_overlap(bo, mk(bnode, [3, 2], Cold[1:6, :], Any[ln, ChargeBasis(1)]))
end
```

Add to the same file a second unit test with two transformed children and the contracts above:

```julia
@testset "transport through two changed children; input untouched; basis change rejected" begin
    using KronFreeCircuits: MaterializedNode, _child_overlap, _transport_previous, Node, ChargeBasis, PhaseGrid, FockBasis, lowest, Port
    rng = MersenneTwister(52)
    orth(n, k) = Matrix(qr(randn(rng, ComplexF64, n, k)).Q)[:, 1:k]
    mk(node, dims, V, children) = MaterializedNode(node, dims, zeros(size(V, 2)), V, collect(1:size(V, 2)),
                                                  Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}(), children, ones(Int, size(V, 2)), Matrix{ComplexF64}(I, size(V, 2), size(V, 2)))
    la = Node([:a => ChargeBasis(2)]; keep=lowest(3)); lb = Node([:b => ChargeBasis(3)]; keep=lowest(2))   # dims 5 and 7
    Ua, Ub = orth(5, 3), orth(7, 2)
    Wa = Ua * Matrix(qr(randn(rng, ComplexF64, 3, 3)).Q); Wb = Ub * Matrix(qr(randn(rng, ComplexF64, 2, 2)).Q)   # same spans, rotated
    ao, an = mk(la, [5], Ua, Any[ChargeBasis(2)]), mk(la, [5], Wa, Any[ChargeBasis(2)])
    bo, bn = mk(lb, [7], Ub, Any[ChargeBasis(3)]), mk(lb, [7], Wb, Any[ChargeBasis(3)])
    Sa, Sb = _child_overlap(ao, an), _child_overlap(bo, bn)
    top = Node([la, lb, :c => ChargeBasis(1)]; keep=lowest(5))        # dims [3, 2, 3]
    C = orth(18, 5); Csaved = copy(C)
    po = mk(top, [3, 2, 3], C, Any[ao, bo, ChargeBasis(1)])
    R̃ = _transport_previous(po, Any[an, bn, ChargeBasis(1)], [3, 2, 3])
    @test R̃ ≈ kron(Matrix{ComplexF64}(I, 3, 3), Sb', Sa') * C         # axis 1 fastest
    @test po.vectors == Csaved                                        # input untouched
    @test_throws ArgumentError _transport_previous(po, Any[an, bn, FockBasis(3; phi_zpf=1.0)], [3, 2, 3])   # same dimension, different basis on axis 3
    lg = Node([:g => PhaseGrid(-2π, 2π, 9; order=2)]; keep=lowest(2)); lg2 = Node([:g => PhaseGrid(-3π, 3π, 9; order=2)]; keep=lowest(2))
    V9 = orth(9, 2)
    @test_throws ArgumentError _child_overlap(mk(lg, [9], V9, Any[PhaseGrid(-2π, 2π, 9; order=2)]), mk(lg2, [9], V9, Any[PhaseGrid(-3π, 3π, 9; order=2)]))
end
```

In `test/test_materialize.jl` the end-to-end test:

```julia
@testset "follow across a permuted child kept order" begin
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    leaf1(order) = Node([:varphi_1 => ChargeBasis(8)]; keep=indices(order))
    leaf2 = Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(4))
    top(order) = Node([leaf1(order), leaf2]; solver=Dense(), compute=8, keep=follow(), ports=[(:varphi_1, nport())])
    s1 = materialize(Node([leaf1([1, 2, 3, 4]), leaf2]; solver=Dense(), keep=lowest(8), ports=[(:varphi_1, nport())]), spec, θ)
    s2 = materialize(top([2, 1, 4, 3]), spec, θ; previous=s1)
    @test energies(s2) ≈ energies(s1)                      # same physical states, same order
    @test port(s2, :varphi_1, nport()) ≈ port(s1, :varphi_1, nport()) atol=1e-10   # :follow phases agree physically
end
```

- [ ] **Step 2: Run** and confirm failures (`_child_overlap` undefined; the end-to-end test mislabels or mismatches ports).

- [ ] **Step 3: Implement** the two functions and the `SelectionContext` field; wire `R̃` into `_materialize`.

- [ ] **Step 4: Run the full default suite.** Expected: pass; the existing "phase :follow with a previous solution" and "follow sweep over n_g" tests (leaf and unchanged-basis cases) are unaffected.

- [ ] **Step 5: Commit** `src/materialize.jl`, `src/selection.jl`, both test files.

---

### Task 7: CUDA extension and device solves

**Files:**
- Modify: `ext/KronFreeCircuitsCUDAExt.jl` (fused diagonal expectation; anything CPU-generic code cannot do), `src/lobpcg.jl` (`_diagonal_expectation!` generic fallback stays), `test/cuda/runtests.jl`
- Create: `test/cuda/test_device.jl`

**Interfaces:**
- Consumes: everything above with `on=CuArray`; `Adapt`; CUDA's KernelAbstractions backend; NDTensors' CUDA/cuTENSOR extensions (loaded by `using CUDA, cuTENSOR`).
- Produces: `LOBPCG(; on=CuArray)` works end to end through `materialize`, returning host arrays; `KronFreeCircuitsCUDAExt` defines `KronFreeCircuits._diagonal_expectation!(theta::CuVector, D::CuVector, X::CuMatrix)` as one fused reduction kernel (one block per column, FP64 accumulators; `docs/spikes/2026-10-16-itensor-audit/ritz-preconditioner.jl` lines 22–50 are the reference) so the Ritz initialization allocates no `N×k` temporary. Tests live in `test/cuda/` and run only there.

**Catalogue cases:** O1, O2, O4 and S1, S3, S5 on `CuArray` as small correctness cases; F3 and F7 small through `LOBPCG(on=CuArray)`; `CUDA.@allocated` bounds only after warm-up and synchronization (catalogue Section 2).

- [ ] **Step 1: Write the device tests** in `test/cuda/test_device.jl` (run with `julia --project=test/cuda test/cuda/runtests.jl`, which includes `../helpers.jl`):

```julia
using Test, LinearAlgebra, Random, CUDA, cuTENSOR, KronFreeCircuits
using KronFreeCircuits: LOBPCG, lobpcg_solve, RitzJacobi, _diagonal_expectation!

@testset "operator on the device equals host; no device allocation after warm-up" begin
    rng = MersenneTwister(61)
    dims = [40, 30, 3]
    terms = [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64.(1:40)))]),
             AxisTerm(0.2, [(1, DenseOp(randn(rng, ComplexF64, 40, 40))), (2, DenseOp(randn(rng, ComplexF64, 30, 30)))]),
             AxisTerm(0.1, [(3, BandedOp(3, [-1, 1], [ones(3), ones(3)])), (2, ShiftOp(30, 2, 0.5))])]
    Hh = KronFreeOperator(dims, terms; block_capacity=6)
    Hd = KronFreeOperator(dims, terms; block_capacity=6, on=CuArray)
    Xh = randn(rng, ComplexF64, 3600, 6); Xd = CuArray(Xh)
    Yd = similar(Xd); Yv = view(Yd, :, 2:5); Xv = view(Xd, :, 2:5)
    @test Array(mul!(Yd, Hd, Xd)) ≈ mul!(similar(Xh), Hh, Xh)
    mul!(Yv, Hd, Xv); mul!(Yv, Hd, Xv)
    @test (CUDA.@allocated mul!(Yv, Hd, Xv)) < 4096
    @test diagonal(Hd) ≈ diagonal(Hh)
end

@testset "fused diagonal expectation" begin
    rng = MersenneTwister(62)
    D = CuArray(randn(rng, 5000)); X = CuArray(randn(rng, ComplexF64, 5000, 7)); theta = CUDA.zeros(Float64, 7)
    _diagonal_expectation!(theta, D, X)
    @test Array(theta) ≈ vec(sum(Array(D) .* abs2.(Array(X)); dims=1) ./ sum(abs2.(Array(X)); dims=1))
    @test (CUDA.@allocated _diagonal_expectation!(theta, D, X)) == 0
end

@testset "LOBPCG on the device equals Dense on the host" begin
    Random.seed!(63)
    H = three_oscillators(6)
    Ed, _ = solve(Dense(), H, 0)
    for pc in (:jacobi, :ritz)
        E, V = solve(LOBPCG(tol=1e-9, preconditioner=pc, on=CuArray), H, 8)
        @test E ≈ Ed[1:8] atol=1e-7 && V isa Matrix{ComplexF64}
        check_eigenpairs(H, E, V; tol=1e-8)
    end
    θ = two_zero_pi_params(; g=0.05); spec = load_spec(fixture("two_zero_pi.json"))
    leaf(i) = Node([Symbol("theta$i") => ChargeBasis(6), Symbol("phi$i") => PhaseGrid(-6π, 6π, 101; order=6)]; solver=Dense(), keep=lowest(10))
    sd = materialize(Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(12)), spec, θ)
    sg = materialize(Node([leaf(1), leaf(2)]; solver=LOBPCG(tol=1e-10, on=CuArray), compute=12, keep=lowest(12)), spec, θ)
    @test energies(sg) ≈ energies(sd) atol=1e-7
end
```

- [ ] **Step 2: Run on ewdesktop.** Sync and run:

```bash
rsync -a --delete --exclude .git --exclude Manifest.toml --exclude 'test/cuda/Manifest.toml' --exclude docs/spikes /Users/eweissler/src/KronFreeCircuits.jl/ ewdesktop:~/kronfree-plan2-ci/
ssh ewdesktop 'cd ~/kronfree-plan2-ci && ~/.juliaup/bin/julia --project=test/cuda -e "using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()" && ~/.juliaup/bin/julia --project=test/cuda test/cuda/runtests.jl'
```

`~/kronfree-plan2-ci/` is a disposable copy used only by this plan; never sync into a regular checkout on ewdesktop, since `--delete` removes whatever is not in the source tree. Expected on the first run: failures in the fused-expectation test (not defined for `CuArray`) and possibly in the allocation bound.

- [ ] **Step 3: Implement** the extension kernel and fix whatever the device run reveals (a host-only `Vector` constructor, a `Matrix(...)` on a `CuArray`, an `ITensors.Index` path that allocates). The CPU suite must stay green after each fix.

- [ ] **Step 4: Rerun Step 2.** Expected: all pass. Save the run log to `docs/spikes/2026-10-17-plan2-benchmark/cuda-tests.log` (create the directory).

- [ ] **Step 5: Commit** `ext/`, `src/lobpcg.jl`, `test/cuda/`, the log.

---

### Task 8: Benchmark against the audit and documentation

**Files:**
- Create: `bench/parent_solve.jl` (part A: stored-fixture regression), `bench/end_to_end.jl` (part B: full materialization with separated timings), `docs/spikes/2026-10-17-plan2-benchmark/results.md`
- Modify: `README.md` (solvers, devices, how to run the device tests and the benchmarks)

**Interfaces:**
- Consumes: the package as built; `two_zero_pi.json`; the audit's stored retained parent `docs/spikes/2026-10-16-itensor-audit/profiling/larger-real300-operators.bin` (a `Serialization` file holding a one-element vector of `Dict`s with `"dims"`, `"constant"`, `"terms" => [(coeff, [(axis, "diag" | "dense", M)])]`, `"x"`, `"y"`; untracked, so the script takes its path from `BENCH_FIXTURE` and skips part A when unset); the audit's numbers in `docs/spikes/2026-10-16-itensor-audit/RITZ-CPU-PACKING-REPORT.md` (120 states on that parent: A4500 fixed 10.18 s, ritz 3.20 s; CPU eight threads fixed 15.67 s, ritz 5.72 s) and REPORT.md §3 (native 300-state leaf solve 90.4 s with single-vector Lanczos).
- Produces:
  - Part A, `bench/parent_solve.jl`: rebuilds the stored parent as a `KronFreeOperator` (checking `H * x ≈ y`), solves the stored coupling point cold for `BENCH_NEV` states, then the coupling scaled by 1.01 warm-started from those vectors, with `:jacobi` and `:ritz`, timing only the resident solve (operator and block already on the device; the audit's boundary), `BENCH_REPEATS` repetitions after a warm-up, validating true residuals, orthonormality and agreement between the two preconditioners. This is the regression against the audit's rows.
  - Part B, `bench/end_to_end.jl`: materializes the two coupled symmetric zero-pis (`ChargeBasis(15)`, `PhaseGrid(-6π, 6π, 301; order=8)`, 300 kept states per leaf with `LOBPCG(:ritz)`, then the parent for `BENCH_NEV` states), reporting separately: leaf solves, operator construction and transfers, parent solve, and port formation (which runs on the host after a device solve in plan 2; see spec §20c decision 4). The cold 300-state adaptive leaf solve is a new measurement with no audit counterpart.
  - Both scripts are driven by `BENCH_DEVICE=cpu|gpu`, `BENCH_NEV` (default 120), `BENCH_REPEATS` (default 3), `BENCH_SMOKE=1` (reduced sizes), print one line per configuration and write a JSON file of all samples next to `results.md`.

**Catalogue cases:** B1 (both parts above, reporting setup, transfers, leaf solve, parent solve, selection/transport and port formation separately, with true residuals, orthogonality, iterations and cumulative H columns); larger F3, F6, F8 and F9 as cold/warm robustness measurements, recorded with explicit failures and timeouts, never as speed gates.

- [ ] **Step 1: Write both scripts.**

- [ ] **Step 2: Smoke-check on the Mac** with `BENCH_SMOKE=1 BENCH_DEVICE=cpu` (part B at 101 points, `ncut=6`, 30 kept states, 12 parent states; part A at `BENCH_NEV=6` if the fixture is present) and confirm both complete and validate.

- [ ] **Step 3: Run on ewdesktop** in the disposable directory of Task 7, CPU with eight BLAS threads then GPU, sequentially, with the fixture copied alongside. Expectations to investigate, not gates: part A `:ritz` on the A4500 within 1.5× of 3.20 s and at least 2× faster than `:jacobi` on both devices; part B's leaf solve well under the audit's 90 s; port formation after the device solve visible as its own line. Residuals below tolerance and host/device agreement to 1e-9 are requirements.

- [ ] **Step 4: Write `results.md`** with two tables (part A against the audit's rows; part B's stage breakdown), the exact commands, package versions, and one paragraph on each expectation met or missed, naming the stage to optimize next (device-side port formation, leaf solver) if the breakdown shows it.

- [ ] **Step 5: Update `README.md`**: the three solvers and when to use each, `LOBPCG(; preconditioner, eta, guard, on)`, warm starts through `previous` and their partial nature when `compute > keep`, the device test command, the benchmark commands, the KernelAbstractions 0.9/0.10 note.

- [ ] **Step 6: Commit** `bench/`, `docs/spikes/2026-10-17-plan2-benchmark/`, `README.md`.


---

### Task 9: Given systems (pre-diagonalized subsystems as tree children)

**Files:**
- Create: `src/given.jl` (included after `tree.jl`)
- Modify: `src/tree.jl` (`Node` accepts a `GivenSystem` child; `_child_modes`), `src/materialize.jl` (materialization of a given child; `extra_terms`; transport treats given children), `src/KronFreeCircuits.jl` (export `GivenSystem`)
- Test: `test/test_given.jl` (new; auto-included by `runtests.jl`)

**Interfaces:**
- Consumes: `Node`, `Term`, `Factor`, `Port`, `Coefficient(x::Number)`, `MaterializedNode`, `_child_overlap` and `_transport_previous` from Task 6, `materialize` in both forms.
- Produces (spec §20c decision 11):
  - `struct GivenSystem` with fields `mode::Symbol`, `energies::Vector{Float64}`, `operators::Dict{Symbol,Matrix{ComplexF64}}`; constructor `GivenSystem(mode; energies, operators)` requiring ascending `energies` (`ArgumentError` otherwise), every operator square of size `length(energies)`, and at least one energy. `dimension(g) = length(g.energies)`.
  - `Node` accepts `GivenSystem` children; `_child_modes(g) = [g.mode]`; `axis_of`, `modes`, `route` and `required_ports` work unchanged (routing never descends into a given system, so a term with all factors on its mode is applied at the parent through the given ports).
  - `MaterializedNode.node::Union{Node,GivenSystem}`; `materialized(sys, g::GivenSystem)` finds it.
  - In `_materialize`, a `GivenSystem` child becomes `MaterializedNode(g, [k], g.energies, Matrix{ComplexF64}(I, k, k), 1:k, Dict((g.mode, Port(kind, 0)) => M for (kind, M) in g.operators), Any[g], collect(1:k), Matrix{ComplexF64}(I, k, k))`. The parent's `_childop` on a given axis looks up `p.kind` in the given ports and requires `p.k == 0`; a missing kind throws `ArgumentError` naming the kind and the mode. Everything downstream (energy diagonal term, joint-peak phase with identity amplitudes, dressed labels) is unchanged.
  - Task 6's basis-equality check gains `_same_basis(a::GivenSystem, b::GivenSystem) = a.mode == b.mode && length(a.energies) == length(b.energies)`; the overlap of a given system with its previous self is the identity by construction (identity vectors), as the spec states.
  - `materialize(root, spec, θ; previous, extra_terms::Vector{Term}=Term[])` appends `extra_terms` to `circuit_terms(spec)`.
  - `route`'s "in no term of the spec" message becomes "in no term".

- [ ] **Step 1: Write the failing tests** in `test/test_given.jl`:

```julia
using KronFreeCircuits: GivenSystem, Term, Factor, Coefficient
@testset "two given systems: explicit Kronecker reference" begin
    E1 = [0.0, 1.0, 2.5]; X1 = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
    E2 = [0.0, 0.8];     X2 = ComplexF64[0 1; 1 0]
    g1 = GivenSystem(:a; energies=E1, operators=Dict(:x => X1))
    g2 = GivenSystem(:b; energies=E2, operators=Dict(:x => X2))
    top = Node([g1, g2]; solver=Dense(), keep=lowest(6), ports=[(:a, Port(:x, 0))])
    terms = [Term(Coefficient(0.1), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])]
    sys = materialize(top, terms, (;))
    Href = kron(Matrix{ComplexF64}(I, 2, 2), Diagonal(E1)) + kron(Diagonal(E2), Matrix{ComplexF64}(I, 3, 3)) + 0.1 * kron(X2, X1)
    @test energies(sys) ≈ eigvals(Hermitian(Href))
    @test bare_labels(sys) == [Tuple(Int(i) for i in Tuple(c)) for c in CartesianIndices((3, 2))]
    @test port(sys, :a, Port(:x, 0)) ≈ port(sys, :a, Port(:x, 0))'
    # a single-mode term on a given mode is applied at the parent through the given port
    sys2 = materialize(top, vcat(terms, [Term(Coefficient(0.3), [Factor(:b, Port(:x, 0))])]), (;))
    @test energies(sys2) ≈ eigvals(Hermitian(Href + 0.3 * kron(X2, Matrix{ComplexF64}(I, 3, 3))))
end

@testset "a materialized transmon re-entered as a given system matches the node tree" begin
    θ = two_transmons_params(; g=0.1); spec = load_spec(fixture("two_transmons.json"))
    leaf1 = Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(5), ports=[(:varphi_1, nport())])
    leaf2 = Node([:varphi_2 => ChargeBasis(8)]; keep=lowest(5))
    full = materialize(Node([leaf1, leaf2]; solver=Dense(), keep=lowest(8)), spec, θ)
    one = materialize(leaf1, load_spec(fixture("transmon.json")), (E_J=θ.E_J1, E_C=θ.E_C1, n_g=θ.n_g1))   # check the transmon fixture's names
    g = GivenSystem(:q; energies=energies(one), operators=Dict(:n => port(one, :varphi_1, nport())))
    # the coupling term of two_transmons.json, written by hand on the given mode
    coupling = [Term(Coefficient("g", [:g]), [Factor(:q, Port(:n, 0)), Factor(:varphi_2, nport())])]
    # the spec's own terms on varphi_2 come from a one-transmon spec with renamed parameters: use the term-list entry point
    terms2 = [t for t in circuit_terms(spec) if all(f.mode == :varphi_2 for f in t.factors)]
    sys = materialize(Node([g, leaf2]; solver=Dense(), keep=lowest(8)), vcat(terms2, coupling), θ; modetypes=Dict(:varphi_2 => :c))
    @test energies(sys) ≈ energies(full) atol=1e-10
    @test_throws ArgumentError materialize(Node([g, leaf2]; solver=Dense(), keep=lowest(8)),
                                           [Term(Coefficient(1.0), [Factor(:q, Port(:phi, 0)), Factor(:varphi_2, nport())])], θ)   # :phi not provided
end

@testset "GivenSystem validation, extra_terms, follow with a given child" begin
    @test_throws ArgumentError GivenSystem(:a; energies=[1.0, 0.0], operators=Dict(:x => zeros(ComplexF64, 2, 2)))
    @test_throws ArgumentError GivenSystem(:a; energies=[0.0, 1.0], operators=Dict(:x => zeros(ComplexF64, 3, 3)))
    @test_throws ArgumentError GivenSystem(:a; energies=Float64[], operators=Dict{Symbol,Matrix{ComplexF64}}())
    θ = (E_J=30.0, E_C=1.2, n_g=0.25); spec = load_spec(fixture("transmon.json"))
    r = GivenSystem(:r; energies=[0.0, 5.0, 10.0], operators=Dict(:x => ComplexF64[0 1 0; 1 0 √2; 0 √2 0]))
    tree() = Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(4)), r]; solver=Dense(), compute=6, keep=follow())   # check the mode name
    extra = [Term(Coefficient(0.2), [Factor(:varphi_1, nport()), Factor(:r, Port(:x, 0))])]
    s1 = materialize(Node([Node([:varphi_1 => ChargeBasis(8)]; keep=lowest(4)), r]; solver=Dense(), keep=lowest(6)), spec, θ; extra_terms=extra)
    s2 = materialize(tree(), spec, θ; previous=s1, extra_terms=extra)
    @test energies(s2) ≈ energies(s1)
    @test materialized(s2, r).vectors == I
end
```

- [ ] **Step 2: Run** and confirm failure (`GivenSystem` undefined).

- [ ] **Step 3: Implement** `src/given.jl` and the tree/materialize changes. Check the actual mode names in `transmon.json` and `two_transmons.json` and the two-transmon coupling form before finalizing the second test; the assertion is that the given-system tree reproduces the node tree exactly, since both keep the lowest five states of the same leaf.

- [ ] **Step 4: Run the full default suite.** Expected: pass.

- [ ] **Step 5: Document** `GivenSystem` in `README.md` (a short example coupling a given resonator to a circuit leaf through `extra_terms`).

- [ ] **Step 6: Commit** `src/given.jl`, `src/tree.jl`, `src/materialize.jl`, `src/KronFreeCircuits.jl`, `test/test_given.jl`, `README.md`.

---

### Task 10: Catalogue regressions before acceptance

**Files:**
- Create: `test/test_catalogue.jl` (default suite; each testset named by its catalogue identifier), additions to `test/test_zero_pi_reference.jl` for the larger gated versions
- Create: `docs/spikes/2026-10-16-pathological-test-catalogue/outcomes.md`

**Interfaces:**
- Consumes: the whole pipeline as built by Tasks 1–9; the catalogue `docs/spikes/2026-10-16-pathological-test-catalogue/README.md`, whose per-case "Oracle" and "Expected" paragraphs are the requirements (read the case before writing its test; do not restate it here).
- Produces: default-suite testsets for F4, F6 (small), F8, F10, P1, P2, T4 and, where Task 4 did not already cover them, S3, S4, S6, S7; gated slow testsets for the larger F3, F6, F8 and F9 versions. `outcomes.md` records, per case identifier, the status the catalogue's Section 8 asks for (implemented/passing, implemented/failing, unsupported-by-contract, deferred), the test file and testset name, and any minimal reproduction of a failure.

- [ ] **Step 1: Write the testsets** in `test/test_catalogue.jl`, one per case, following each case's oracle; keep every default-suite case small enough for an independent dense reference built without the apply routine under test (catalogue Section 2). A case whose documented contract is an informative rejection asserts the error.

- [ ] **Step 2: Run the default suite**; the suite must stay under about 60 s, so move anything slower to the gated file.

- [ ] **Step 3: Run the gated suite** (`KRONFREE_SLOW_TESTS=1`) and record outcomes. A failing case is recorded as implemented/failing with its reproduction and reported in the task report; it is not deleted or weakened to pass.

- [ ] **Step 4: Write `outcomes.md`.**

- [ ] **Step 5: Commit** both test files and `outcomes.md`.
