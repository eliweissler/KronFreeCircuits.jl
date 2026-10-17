# Plan 2: Block Solver and ITensor Parent Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every node the stock LOBPCGEigensolver block solver with fixed and adaptive per-Ritz Jacobi preconditioning and warm starts, apply dense ports at branches through ITensor contractions (cuTENSOR on GPU) behind the existing operator interface, run a node's solve on a CUDA device through a package extension, and make `follow` physically correct across changing child bases.

**Architecture:** `KronFreeOperator` becomes the single device-aware operator: it owns device copies of its factor data, fuses every pure-diagonal term into one state-shaped vector applied by a broadcast, applies `DenseOp` factors through scratch-backed ITensor contractions and structured factors through the existing KernelAbstractions kernels. `LOBPCG <: Solver` wraps `LOBPCGEigensolver.lobpcg` through its public `mul!`/`ldiv!`/`precondprep!`/callback interfaces only. Materialization passes a node's previous kept eigenvectors as the initial block and transports them into the current product basis for selection and phase. The CUDA extension adds only what the base cannot express generically (a fused diagonal-expectation kernel); everything else moves to the device through `Adapt`.

**Tech Stack:** Julia 1.13, ITensors 0.9 (NDTensors), LOBPCGEigensolver 0.1.4, KernelAbstractions 0.9/0.10, KrylovKit 0.10, Adapt; CUDA 6.4 and cuTENSOR as weak dependencies; tests on the Mac (CPU) and on `ewdesktop` (RTX A4500) over ssh.

**Spec:** `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`, Sections 8, 9, 9.1 as amended by Section 20c (decisions 1–6 and 9). Evidence and reference implementations: `docs/spikes/2026-10-16-itensor-audit/` (REPORT.md §5, §7; `buffer-layout.jl`, `native-gemm-parent.jl` for the buffered ITensor adapter; `ritz-preconditioner.jl` for the adaptive preconditioner; CORRECTNESS-NOTES.md for the two bugs Tasks 4 and 6 fix).

## Global Constraints

- Julia 1.13. Column-major, axis 1 fastest, one axis per child plus an optional trailing block axis. Apply dispatch is on operator structure, never on device.
- Element type stays `ComplexF64` throughout plan 2, but new solver and backend code carries no literal element type: take it from the data (`eltype(X)`, `real(eltype(X))`), so the real-arithmetic plan can follow without rewriting it.
- LOBPCGEigensolver is used only through `lobpcg`, `precondprep!`, `ldiv!`, and the `callback` keyword; no method is defined on its internal types and no source is copied. Compat is pinned to `"=0.1.4"` until the callback-contract test (Task 4) is rerun against a newer release.
- ITensors compat `"0.9"`; KernelAbstractions compat `"0.9, 0.10"`; KrylovKit `"0.10.4"` unchanged. CUDA and cuTENSOR are weak dependencies only; the default test suite runs without a GPU and stays under about 60 s.
- Workspace contract (spec §8): block `mul!(Y, H, X)` writes into caller-owned output through two preallocated scratch arrays reserved for the solver's maximum block width; shrinking active blocks reuse contiguous prefixes; after warm-up no state-sized allocation occurs on CPU or device. Small metadata allocations are allowed.
- Physics conventions, fixtures and the scqubits reference file are unchanged.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A block solver locks states, so `mul!` receives offset column views (`view(X, :, 3:7)`) into solver-owned arrays. The ITensor path must give the same result as the reference and allocate nothing state-sized on them. Pinned in Task 3.
2. `LOBPCG` on a node too small for its block (`N < 3m + 5`) must raise an `ArgumentError` naming `Dense`, not a cryptic library error. Pinned in Task 4.
3. An exactly repeated eigenvalue (`diag(0, 0, 1, …)`) must come back twice from `LOBPCG` and `Dense`; the single-vector `Lanczos` limitation is documented, not silently inherited by the default path. Pinned in Task 4.
4. A `previous` system whose child kept order differs from the current one must give the same branch energies and ports under `follow`/`:follow`. Pinned in Task 6.
5. A mixed branch (a `FockBasis` child beside a node child, as in the full zero-pi) runs the fused diagonal, an ITensor dense factor and a banded kernel inside one operator and still equals the Kronecker reference. Pinned in Task 2 and Task 3.

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
  - `struct LOBPCG <: Solver` with fields `tol::Float64`, `maxiter::Int`, `preconditioner::Symbol` (`:ritz`, `:jacobi`, `:none`), `eta::Float64`, `on::Any`, `callback::Any`; constructor `LOBPCG(; tol=1e-8, maxiter=500, preconditioner=:ritz, eta=0.01, on=Array, callback=identity)`. `eta` is in the Hamiltonian's energy units; the docstring says the default is taken from the audit fixture (GHz) and must be set for other scales.
  - `struct FixedJacobi{V}` with field `inv::V` (= `1 ./ (D .- σ)`, `σ = minimum(D) - 1`, real) and `LinearAlgebra.ldiv!(P::FixedJacobi, R)` = `R .*= P.inv`.
  - `mutable struct RitzJacobi{V,W}` with fields `D::V` (real diagonal), `theta::W` (real, length = block capacity), `eta`, `locked::Int`, `width::Int`, `ready::Bool`, `trace::Union{Nothing,Vector{Any}}`; `RitzJacobi(D, capacity; eta, trace=false)`; `LOBPCGEigensolver.precondprep!(P::RitzJacobi, X)` (on first call, `theta[j] = Σ_i D[i]|X[i,j]|² / Σ_i |X[i,j]|²` per active column through `_diagonal_expectation!(theta, D, X)`, a column loop on host arrays; afterwards no-op except width bookkeeping; when `trace` is on, push `(width, copy(theta[1:width]))`), `LinearAlgebra.ldiv!(P::RitzJacobi, R)` = `R ./= hypot.(P.D .- transpose(view(P.theta, 1:width)), P.eta)`, and `ritz_callback(P::RitzJacobi)` returning the function `info -> (copy the active Ritz values from info.λs into P.theta, dropping the newly locked prefix; update P.locked)` composed with the user's callback. The exact copy logic follows `docs/spikes/2026-10-16-itensor-audit/ritz-preconditioner.jl` (`LO.precondprep!`, `ldiv!`, the callback block) and `RITZ-PRECONDITIONER-REPORT.md` ("Question and implementation").
  - `lobpcg_solve(s::LOBPCG, H::KronFreeOperator, m::Int; X0=nothing) -> (E::Vector{Float64}, V::Matrix, result)` where `result` is `lobpcg`'s named tuple (`λ, X, residual_norms, n_matvec, …`); `X0` is an `N×m` initial block on the host or `nothing` (then `randn(rng, eltype(H), N, m)` with `Random.default_rng()`); `H` is rebuilt or checked to have `block_capacity ≥ m` and storage `s.on`; the block is `adapt(s.on, X0)`; `E` and `V` return as host arrays sorted ascending. Throws `ArgumentError("LOBPCG needs N ≥ 3m + 5 … use Dense()")` when `size(H, 1) < 3m + 5`, `ArgumentError` for `m < 1`.
  - `solve(s::LOBPCG, H, m)` = the first two outputs of `lobpcg_solve`, same return types as the other solvers.
  - `with_block_capacity(H::KronFreeOperator, m::Integer)::KronFreeOperator` in `src/apply.jl`: the same operator (shared factor data and `diag`) with scratch reserved for `m` columns; `lobpcg_solve` uses it when `H`'s capacity is below `m`.
  - `_dense_too_large` warning text in `materialize.jl` changes "Lanczos is advised" to "LOBPCG is advised".
  - `Lanczos` docstring gains the sentence: "A single-vector Krylov space does not resolve repeated eigenvalues (KrylovKit documents this); use `LOBPCG` or `Dense` where exact degeneracies are expected."

- [ ] **Step 1: Write the failing tests** in `test/test_solvers.jl`:

```julia
using KronFreeCircuits: LOBPCG, lobpcg_solve, FixedJacobi, RitzJacobi, ritz_callback, with_block_capacity
import LOBPCGEigensolver   # add to test/Project.toml

@testset "LOBPCG agrees with Dense; all preconditioners" begin
    Random.seed!(7)
    H = three_oscillators(6)                       # N = 216
    Ed, Vd = solve(Dense(), H, 0)
    for pc in (:none, :jacobi, :ritz)
        E, V = solve(LOBPCG(tol=1e-9, preconditioner=pc), H, 8)
        @test E ≈ Ed[1:8] atol=1e-7
        @test all(abs.(diag(V' * Vd[:, 1:8])) .≈ 1)
        @test issorted(E) && typeof((E, V)) == Tuple{Vector{Float64},Matrix{ComplexF64}}
    end
end

@testset "LOBPCG argument errors" begin
    H = three_oscillators(3)                       # N = 27
    @test_throws ArgumentError solve(LOBPCG(), H, 8)          # 3*8+5 > 27
    err = try solve(LOBPCG(), H, 8); nothing catch e; e end
    @test occursin("Dense", err.msg)
    @test_throws ArgumentError solve(LOBPCG(), H, 0)
    @test_throws ArgumentError LOBPCG(preconditioner=:magic)
end

@testset "repeated eigenvalue is recovered twice" begin
    H = KronFreeOperator([12], [AxisTerm(1.0, [(1, DiagonalOp(ComplexF64[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))])])
    Random.seed!(3)
    E, _ = solve(LOBPCG(tol=1e-10), H, 2)
    @test E ≈ [0.0, 0.0] atol=1e-8
    @test solve(Dense(), H, 2)[1] ≈ [0.0, 0.0] atol=1e-12
end

@testset "warm start reduces work" begin
    Random.seed!(11)
    H = three_oscillators(6)
    E, V, cold = lobpcg_solve(LOBPCG(tol=1e-9), H, 6)
    _, _, warm = lobpcg_solve(LOBPCG(tol=1e-9), H, 6; X0=V)
    @test warm.n_matvec < cold.n_matvec ÷ 2
end

@testset "Ritz cache stays aligned with the solver's active columns" begin
    Random.seed!(5)
    H = three_oscillators(6); m = 10
    D = real.(diagonal(H))
    P = RitzJacobi(D, m; eta=0.05, trace=true)
    seen = Any[]
    cb = info -> (push!(seen, (info.n_iter, info.n_locked, copy(info.λs))); info)
    X0 = randn(ComplexF64, size(H, 1), m)
    Hb = with_block_capacity(H, m)
    res = LOBPCGEigensolver.lobpcg(Hb, X0, I, P, 1e-9, 300; callback=ritz_callback(P) ∘ cb)   # cb runs first, then the cache update
    @test res.λ ≈ solve(Dense(), H, m)[1] atol=1e-7
    # From the second prepare onward, the cached values for the active columns are exactly the
    # previous iteration's Ritz values for those same columns (locked prefix removed).
    for t in 2:min(length(P.trace), length(seen))
        width, theta = P.trace[t]
        n_iter, n_locked, λs = seen[t - 1]
        @test theta == λs[end - width + 1:end]
    end
end
```

- [ ] **Step 2: Run** and confirm failures (`LOBPCG` undefined).

- [ ] **Step 3: Implement** `src/lobpcg.jl` and the `LOBPCG` type. In `lobpcg_solve`: build `D = real.(diagonal(H))` on `s.on`; choose the preconditioner by `s.preconditioner`; call `LOBPCGEigensolver.lobpcg(H, X0dev, I, P, s.tol, s.maxiter; callback=ritz_callback(P) ∘ s.callback)` (`callback=s.callback` for the other two); check `all(res.residual_norms .< s.tol)` or raise `ErrorException` naming the unconverged count and the maximum residual, like `Lanczos`; return `Vector{Float64}(res.λ)`, `Matrix(Array(res.X))`.

- [ ] **Step 4: Run the full default suite.** Expected: pass.

- [ ] **Step 5: Commit** `src/lobpcg.jl`, `src/tree.jl`, `src/solvers.jl`, `src/KronFreeCircuits.jl`, `test/test_solvers.jl`.

---

### Task 5: `LOBPCG` nodes in materialization with warm starts

**Files:**
- Modify: `src/materialize.jl` (operator construction with capacity and storage; initial block from `previous`), `src/tree.jl` (Node docstring: `solver` is `Dense()`, `Lanczos()` or `LOBPCG()`)
- Test: `test/test_materialize.jl`

**Interfaces:**
- Consumes: `lobpcg_solve`, `LOBPCG.on`, `KronFreeOperator(…; block_capacity, on)`, `prev::MaterializedNode` (`vectors`, `dims`).
- Produces: in `_materialize`, `storage(::Solver) = Array`, `storage(s::LOBPCG) = s.on`; `H = KronFreeOperator(dims, terms; constant, block_capacity=(node.solver isa LOBPCG ? m : 1), on=storage(node.solver))`; `_initial_block(prev, dims, m)` = `nothing` when `prev === nothing` or `prev.dims != dims`, else `prev.vectors` padded on the right with `randn` columns to `m` (and truncated to `m` columns when it has more); `solve(::LOBPCG, …)` receives it as `X0` (the other solvers ignore it). The guess is the raw previous coefficients, never the transported reference of Task 6 (spec §20c, decision 2).

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
  - `_child_overlap(prev::MaterializedNode, cur::MaterializedNode)::Matrix{ComplexF64}` = `⟨prev kept i | cur kept j⟩` in the physical space: at a leaf `prev.vectors' * cur.vectors`; at a branch `_transport_previous(prev, cur.children, cur.dims)' * cur.vectors`. Requires `prev.dims == cur.dims` (throws `ArgumentError` otherwise).
  - `_transport_previous(prev::MaterializedNode, children, dims)::Matrix{ComplexF64}` = `(⊗_a S_a') prev.vectors`, applying `DenseOp(Matrix(S_a'))` along each axis `a` whose child is a `MaterializedNode` (`S_a = _child_overlap(prev.children[a], children[a])`), identity on basis axes. At a leaf it returns `prev.vectors` unchanged.
  - `SelectionContext(energies, vectors, ports, previous, reference)`; the four-argument constructor keeps working with `reference = previous === nothing ? nothing : previous.vectors`. `select(::Follow, ctx)` uses `ctx.reference` where it used `ctx.previous.vectors`.
  - In `_materialize`, when `uses_prev` and `prev !== nothing`: `R̃ = _transport_previous(prev, children, dims)` once; it is the `reference` of the `SelectionContext` and the reference handed to `fix_phases!(V, :follow, mask, R̃)`.

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
rsync -a --delete --exclude .git --exclude Manifest.toml --exclude 'test/cuda/Manifest.toml' --exclude docs/spikes /Users/eweissler/src/KronFreeCircuits.jl/ ewdesktop:~/src/KronFreeCircuits.jl/
ssh ewdesktop 'cd ~/src/KronFreeCircuits.jl && ~/.juliaup/bin/julia --project=test/cuda -e "using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()" && ~/.juliaup/bin/julia --project=test/cuda test/cuda/runtests.jl'
```

Expected on the first run: failures in the fused-expectation test (not defined for `CuArray`) and possibly in the allocation bound.

- [ ] **Step 3: Implement** the extension kernel and fix whatever the device run reveals (a host-only `Vector` constructor, a `Matrix(...)` on a `CuArray`, an `ITensors.Index` path that allocates). The CPU suite must stay green after each fix.

- [ ] **Step 4: Rerun Step 2.** Expected: all pass. Save the run log to `docs/spikes/2026-10-17-plan2-benchmark/cuda-tests.log` (create the directory).

- [ ] **Step 5: Commit** `ext/`, `src/lobpcg.jl`, `test/cuda/`, the log.

---

### Task 8: Benchmark against the audit and documentation

**Files:**
- Create: `bench/parent_solve.jl`, `docs/spikes/2026-10-17-plan2-benchmark/results.md`
- Modify: `README.md` (solvers, devices, how to run the device tests and the benchmark)

**Interfaces:**
- Consumes: the package as built; `two_zero_pi.json`; the audit's numbers in `docs/spikes/2026-10-16-itensor-audit/RITZ-CPU-PACKING-REPORT.md` (120 states, 300×300 retained parent: A4500 fixed 10.18 s, ritz 3.20 s; CPU eight threads fixed 15.67 s, ritz 5.72 s) and REPORT.md §3 (native 300-state leaf solve 90.4 s).
- Produces: a script that materializes the two coupled symmetric zero-pis with `ChargeBasis(15)`, `PhaseGrid(-6π, 6π, 301; order=8)`, 300 kept states per leaf (leaf solver from the command line: `Dense` is out of reach at dimension 9 331 only by time, `LOBPCG` with `:ritz` is the one to time), then the 300×300 parent for 120 states with `:jacobi` and `:ritz`, cold and warm-started from a 1 % coupling step, on `Array` with eight BLAS threads and on `CuArray`; it prints leaf and parent times, iteration counts and `n_matvec`, and checks residuals and host/device energy agreement.

- [ ] **Step 1: Write the script**, driven by environment variables `BENCH_DEVICE=cpu|gpu`, `BENCH_NEV=120`, `BENCH_REPEATS=3`, printing one line per configuration and a JSON file of all samples next to the results.

- [ ] **Step 2: Run on the Mac (CPU) for a smoke check** at reduced size (`BENCH_SMOKE=1`: 101 points, `ncut=6`, 30 kept states, 12 parent states) and confirm it completes and validates.

- [ ] **Step 3: Run on ewdesktop**, CPU (eight threads) then GPU, sequentially, with the rsync command of Task 7. Expected: parent 120-state `:ritz` on the A4500 within 1.5× of 3.20 s; `:ritz` at least 2× faster than `:jacobi` on both devices; residuals below tolerance; host and device energies agree to 1e-9. Record the leaf solve time for `LOBPCG(:ritz)` against the audit's 90.4 s for single-vector Lanczos.

- [ ] **Step 4: Write `results.md`** with the table (configuration, median seconds, iterations, `n_matvec`, validation), the exact commands, package versions, and one paragraph each on what met and what missed the expectations above.

- [ ] **Step 5: Update `README.md`**: the three solvers and when to use each, `LOBPCG(; preconditioner, eta, on)`, warm starts through `previous`, the device test command, the benchmark command, the KernelAbstractions 0.9/0.10 note.

- [ ] **Step 6: Commit** `bench/`, `docs/spikes/2026-10-17-plan2-benchmark/`, `README.md`.
