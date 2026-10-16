# KronFreeCircuits Core CPU Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Julia package that loads a circuit spec, builds kron-free Hamiltonians over a tree of nodes, diagonalizes leaves and branches on CPU with dense and Lanczos solvers, selects kept states by rule, and exposes energies, ports, and dressed/bare labels, validated against explicit Kronecker products and scqubits.

**Architecture:** Every operator is a list of product terms whose factors are structured local operators (dense, diagonal, banded, shift) applied one axis at a time to a state tensor. A node is diagonalized matrix-free, a set of eigenstates is kept, and its ports are re-expressed as dense matrices so the node becomes a child of the next node up. Parameters enter only through term coefficients.

**Tech Stack:** Julia 1.13; LinearAlgebra, SparseArrays (stdlib); KernelAbstractions.jl (banded and shift kernels); KrylovKit.jl (Lanczos); JSON3.jl (spec); RuntimeGeneratedFunctions.jl (coefficients). Test-only: Test, SparseArrays, scqubits-generated fixtures committed as JSON.

**Spec:** `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`

**Plan sequence:** This is plan 1 of 5. Later plans: autodiff rules (spec Section 11); LOBPCG and the CUDA extension (Sections 8, 9); export, caching, NPZ (Sections 10.4, 12); the Python exporter in circuit-quantization (Section 5.1). Nothing in this plan may make those harder: in particular the term list (Task 4) and `Node` (Task 8) are public types.

## Global Constraints

- Julia `1.13`; package name `KronFreeCircuits`, repository root `~/src/KronFreeCircuits.jl`, branch `main`.
- All state vectors and `DenseOp` matrices are `ComplexF64`. A `DenseOp` is never stored real (spec 7.2).
- Column-major: first axis fastest. Test Kronecker references use `kron(A_N, ..., A_1)` so that axis 1 is fastest (spec 14).
- Energies in GHz with $h=1$; phases dimensionless; charge in units of $2e$ (spec 3).
- Circuit parameters enter only through term coefficients; basis hyperparameters are fixed (spec 11.1). No basis method may take `θ`.
- One apply interface, `apply_axis!`, dispatching on local-operator structure type, never on device (spec 8). Dense goes to gemm, diagonal to broadcast, banded and shift to KernelAbstractions kernels.
- No runtime dependency on Python, ITensors, QuantumToolbox, NNlib, Symbolics, IterativeSolvers (spec 15).
- Grid stencils are the central weights of spec Section 6 (thesis table), orders 2, 4, 6, 8; $n_i^2$ is the second-derivative stencil and $n_i n_j$ the product of first-derivative stencils (spec 6).
- Phase conventions `:positive_sector` (default) and `:follow` exactly as spec 9.1.
- Commit after every task with the attribution line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A parameter tuple `θ` missing a symbol the spec uses: `materialize` must throw an `ArgumentError` naming the missing parameter, not a `FieldError` from deep inside a generated function. Pinned in Task 2.
2. A spec whose cosine `omega` has a non-integer entry on a compact mode: `parse_spec` must reject it with the mode name, since a non-integer shift is meaningless in the charge basis. Pinned in Task 3.
3. `exp2pin(k)` on a `PhaseGrid` whose spacing does not divide $2\pi k$: `localop` must throw rather than round silently. Pinned in Task 6.
4. `mul!` called with a vector of the wrong length, or a block whose first dimension is wrong: `DimensionMismatch`, never a silent reshape. Pinned in Task 7.
5. A selection rule returning indices outside `1:compute`, duplicates, or an empty list: `materialize` must throw naming the rule, not produce a node with garbage ports. Pinned in Task 10.

---

## File structure

```
Project.toml
src/KronFreeCircuits.jl      module, includes, exports
src/parameters.jl            Coefficient: compiled expression of named parameters
src/spec.jl                  Spec, ModeSpec, CosineSpec, ChargeCosineSpec; load_spec, parse_spec, validation
src/terms.jl                 Port, Factor, Term; circuit_terms(spec) expansion
src/localops.jl              LocalOp, DenseOp, DiagonalOp, BandedOp, ShiftOp; adjoint, diagonal, dense
src/bases.jl                 Basis, ChargeBasis, PhaseGrid, FockBasis; localop, fd_weights, displacement
src/apply.jl                 apply_axis! per structure type; AxisTerm, KronFreeOperator, mul!, diagonal
src/tree.jl                  Solver types, Node, modes, route, required_ports
src/solvers.jl               solve(::Dense | ::Lanczos, H, m)
src/selection.jl             SelectionRule, SelectionContext, lowest, indices, connected, follow, select, ports_needed
src/phase.jl                 fix_phases!
src/materialize.jl           MaterializedNode, System, materialize, energies, port, eigenvectors
src/dressed.jl               bare_labels, decompose, dressed_index, bare_overlaps
test/runtests.jl             includes every test file below
test/test_parameters.jl … test/test_dressed.jl   one per source file
test/fixtures/fluxonium.json, transmon.json, zero_pi.json, two_zero_pi.json   hand-written specs
test/fixtures/generate_scqubits.py                reference generator (run once, outputs committed)
test/fixtures/scqubits_reference.json             reference energies
test/helpers.jl                                   kron_reference(dims, axisterms) and fixture loaders
```

---

### Task 1: Package skeleton and test harness

**Files:**
- Create: `Project.toml`, `src/KronFreeCircuits.jl`, `test/runtests.jl`, `test/helpers.jl`, `test/Project.toml`

**Interfaces:**
- Produces: module `KronFreeCircuits`; test helper `kron_reference(dims::Vector{Int}, terms::Vector{<:Tuple{ComplexF64,Vector{Tuple{Int,<:AbstractMatrix}}}})::Matrix{ComplexF64}` that builds $\sum_t c_t\, A_{t,N}\otimes\cdots\otimes A_{t,1}$ with identities on absent axes, first axis fastest.

- [ ] **Step 1: Write the failing test** in `test/test_helpers.jl`

```julia
@testset "kron_reference" begin
    A = ComplexF64[1 2; 3 4]; B = ComplexF64[0 1; 1 0]
    H = kron_reference([2, 2], [(1.0 + 0im, [(1, A)]), (2.0 + 0im, [(2, B)])])
    @test H ≈ kron(I(2), A) + 2 * kron(B, I(2))   # axis 1 fastest
end
```

- [ ] **Step 2: Run to verify it fails**: `julia --project=. -e 'using Pkg; Pkg.test()'` → fails, module missing.
- [ ] **Step 3: Create `Project.toml`** with `name = "KronFreeCircuits"`, a fresh UUID, `version = "0.1.0"`, `[compat] julia = "1.13"`, deps `LinearAlgebra`, `SparseArrays`; `test/Project.toml` with `Test`, `LinearAlgebra`, `SparseArrays`. Create `src/KronFreeCircuits.jl` defining the empty module with `using LinearAlgebra, SparseArrays`. Create `test/runtests.jl` that includes `helpers.jl` and every `test_*.jl`, and `test/helpers.jl` with `kron_reference`.
- [ ] **Step 4: Run tests**: `Pkg.test()` → PASS.
- [ ] **Step 5: Commit**: `git add -A && git commit -m "Add package skeleton and kron_reference test helper"`.

---

### Task 2: Coefficients

**Files:**
- Create: `src/parameters.jl`, `test/test_parameters.jl`
- Modify: `Project.toml` (add `RuntimeGeneratedFunctions`), `src/KronFreeCircuits.jl`

**Interfaces:**
- Produces:
  - `struct Coefficient; ex::Union{Expr,Symbol,Number}; deps::Vector{Symbol}; f::Function; end`
  - `Coefficient(src::AbstractString, params::AbstractVector{Symbol})::Coefficient` parses with `Meta.parse`, checks every free symbol is in `params`, compiles `f(θ::NamedTuple)::ComplexF64` with `@RuntimeGeneratedFunction`.
  - `Coefficient(x::Number)::Coefficient` (constant).
  - `(c::Coefficient)(θ::NamedTuple)::ComplexF64`; throws `ArgumentError("parameter :E_J missing from θ")` when a dep is absent (check `haskey` before calling `f`).
  - `free_symbols(ex)::Vector{Symbol}`: symbols in value position, excluding call heads and `:pi, :π, :im, :ℯ, :e`.
  - Algebra producing new `Coefficient`s at expression level: `Base.:*(::Number, ::Coefficient)`, `Base.:*(::Coefficient, ::Coefficient)`, `Base.:+(::Coefficient, ::Coefficient)`, `Base.:-(::Coefficient)`, `phase_factor(offset::Coefficient, sign::Int)::Coefficient` = $e^{\,\mathrm{sign}\cdot i\cdot\mathrm{offset}}$.
  - `iszero_expr(c::Coefficient)::Bool` true when `ex` is the literal `0` or `0.0`.

- [ ] **Step 1: Write failing tests**

```julia
@testset "Coefficient" begin
    c = Coefficient("E_J*cos(varphi_ext/2) + 1/2", [:E_J, :varphi_ext])
    @test c.deps == [:E_J, :varphi_ext]
    @test c((E_J=2.0, varphi_ext=0.0)) ≈ 2.5
    @test_throws ArgumentError Coefficient("E_J*E_X", [:E_J])
    @test_throws ArgumentError c((E_J=2.0,))                  # missing varphi_ext, Review Focus 1
    @test (2 * c)((E_J=2.0, varphi_ext=0.0)) ≈ 5.0
    @test phase_factor(Coefficient("varphi_ext", [:varphi_ext]), -1)((varphi_ext=π/2,)) ≈ exp(-im*π/2)
    @test iszero_expr(Coefficient("0", Symbol[])) && !iszero_expr(c)
    @test free_symbols(Meta.parse("E_J*cos(pi*x) + im")) == [:E_J, :x]
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `src/parameters.jl`.** Generated function body: `θ -> begin E_J = θ.E_J; varphi_ext = θ.varphi_ext; ComplexF64($ex) end`, bound to the module with `RuntimeGeneratedFunctions.init(@__MODULE__)`.
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add Coefficient: compiled expressions of named parameters"`.

---

### Task 3: Spec loading and fixtures

**Files:**
- Create: `src/spec.jl`, `test/test_spec.jl`, `test/fixtures/fluxonium.json`, `test/fixtures/transmon.json`, `test/fixtures/two_transmons.json`, `test/fixtures/three_transmons.json`, `test/fixtures/zero_pi.json`, `test/fixtures/full_zero_pi.json`, `test/fixtures/two_zero_pi.json`
- Modify: `Project.toml` (add `JSON3`)

**Interfaces:**
- Produces:
  - `struct ModeSpec; name::Symbol; type::Symbol; end` with `type ∈ (:c, :e, :h)`.
  - `struct CosineSpec; energy::Coefficient; omega::Vector{Float64}; offset::Coefficient; end`
  - `struct ChargeCosineSpec; energy::Coefficient; kappa::Vector{Float64}; offset::Coefficient; end`
  - `struct Spec; parameters::Vector{Symbol}; modes::Vector{ModeSpec}; E_C::Matrix{Coefficient}; E_L::Matrix{Coefficient}; flux_linear::Vector{Coefficient}; n_g::Vector{Coefficient}; cosines::Vector{CosineSpec}; charge_cosines::Vector{ChargeCosineSpec}; constant::Coefficient; end`
  - `parse_spec(json::AbstractString)::Spec`, `load_spec(path::AbstractString)::Spec`.
  - Validation throws `ArgumentError` for: `format != "kronfree-spec"`, `version != 1`, matrix or vector sizes not matching `length(modes)`, a cosine `omega` entry on a compact mode farther than `1e-12` from an integer (message includes the mode name), a `charge_cosines` `kappa` nonzero on a compact mode, an expression using a symbol not in `parameters`. Missing `charge_cosines` or `constant` default to empty and `"0"`.
- Fixture contents (hand-written, modes in the given order; these are the ground truth plan 5's exporter must reproduce):
  - `fluxonium.json`: parameters `[E_J, E_C, E_L, varphi_ext]`; modes `[varphi_1: e]`; `E_C=[["E_C"]]`, `E_L=[["E_L"]]`, `flux_linear=["E_L*varphi_ext"]`, `n_g=["0"]`, cosines `[{E_J, [1], "0"}]`, constant `"E_L*varphi_ext^2/2"`. This is $4E_Cn^2 - E_J\cos\varphi + \tfrac12E_L(\varphi+\varphi_{\rm ext})^2$.
  - `transmon.json`: parameters `[E_J, E_C, n_g]`; modes `[varphi_1: c]`; `E_C=[["E_C"]]`, `E_L=[["0"]]`, `flux_linear=["0"]`, `n_g=["n_g"]`, cosines `[{E_J, [1], "0"}]`.
  - `zero_pi.json` (scqubits `ZeroPi` convention with `dEJ = dCJ = 0`): parameters `[E_J, E_L, E_CJ, E_CS, varphi_ext, n_g]`; modes `[theta: c, phi: e]`; `E_C=[["E_CS/2","0"],["0","E_CJ/2"]]`, `E_L=[["0","0"],["0","2*E_L"]]`, `flux_linear=["0","0"]`, `n_g=["n_g","0"]`, cosines `[{E_J, [1, 1], "-varphi_ext/2"}, {E_J, [1, -1], "varphi_ext/2"}]`. This is $2E_{CJ}n_\varphi^2 + 2E_{CS}(n_\theta-n_g)^2 - 2E_J\cos\theta\cos(\varphi-\varphi_{\rm ext}/2) + E_L\varphi^2$.
  - `two_transmons.json`: parameters `[E_J1, E_C1, n_g1, E_J2, E_C2, n_g2, g]`; modes `[varphi_1: c, varphi_2: c]`; `E_C=[["E_C1","g/8"],["g/8","E_C2"]]`, `E_L` zero, `flux_linear` zero, `n_g=["n_g1","n_g2"]`, cosines `[{E_J1, [1, 0], "0"}, {E_J2, [0, 1], "0"}]`. The off-diagonal `g/8` yields the coupling $g\,n_1 n_2$ after expansion (Task 4). This is the cheap composite used for all branch and dressed-state tests.
  - `three_transmons.json`: `two_transmons.json` plus a third compact mode `varphi_3` with parameters `E_J3, E_C3, n_g3`, uncoupled (its `E_C` row and column are zero off the diagonal), cosine `{E_J3, [0, 0, 1], "0"}`. Used by the three-level tree test (Task 12).
  - `full_zero_pi.json`: the asymmetric zero-pi with its zeta mode, modes `[theta: c, phi: e, zeta: h]`, transcribed by the implementer from scqubits' `FullZeroPi.hamiltonian` source (not from memory) with parameters `[E_J, E_L, E_CJ, E_CS, E_Cz, dE_J, dC_J, dC, dE_L, varphi_ext, n_g]`. The $\sin\theta\sin(\varphi-\varphi_{\rm ext}/2)$ asymmetry term is written as two cosines with energies `"E_J*dE_J"` and `"-E_J*dE_J"`; charge asymmetries become off-diagonal `E_C` entries and the inductive asymmetry an off-diagonal `E_L` entry. The file carries a `"_comment"` field recording the mapping from scqubits' expression, term by term. This fixture exercises all three mode types and a two-level tree (Task 12).
  - `two_zero_pi.json`: two copies of `zero_pi.json` with suffixes `1`, `2` on modes and parameters, plus parameter `g` and `E_C[theta1, theta2] = E_C[theta2, theta1] = "g/8"`, the coupling $g\,n_{\theta1}n_{\theta2}$. Used only by the final reference task (Task 14).

- [ ] **Step 1: Write failing tests**

```julia
@testset "Spec" begin
    s = load_spec(joinpath(@__DIR__, "fixtures", "zero_pi.json"))
    @test [m.name for m in s.modes] == [:theta, :phi] && s.modes[1].type == :c
    @test size(s.E_C) == (2, 2) && length(s.cosines) == 2
    @test s.cosines[1].omega == [1.0, 1.0]
    @test s.E_C[1,1]((E_CS=1.0,)) ≈ 0.5
    bad = replace(read(joinpath(@__DIR__, "fixtures", "zero_pi.json"), String), "[1, 1]" => "[0.5, 1]")
    @test_throws ArgumentError parse_spec(bad)                 # Review Focus 2
    @test_throws ArgumentError parse_spec(replace(read(..., String), "\"E_CS/2\"" => "\"E_XX/2\""))
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Write the seven fixture files and `src/spec.jl`.** For `full_zero_pi.json`, read the scqubits source in the `zp2q` conda environment: `conda run -n zp2q python -c 'import scqubits, inspect; print(inspect.getsource(scqubits.FullZeroPi.hamiltonian))'`.
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add Spec loading with validation and hand-written circuit fixtures"`.

---

### Task 4: Terms

**Files:**
- Create: `src/terms.jl`, `test/test_terms.jl`

**Interfaces:**
- Produces:
  - `struct Port; kind::Symbol; k::Float64; end`; constructors `nport() = Port(:n, 0)`, `n2port()`, `phiport()`, `phi2port()`, `expiphi(k::Real)`, `exp2pin(k::Real)`. `Base.:(==)` and `hash` compare `kind` and `k` (so ports work as `Dict` keys).
  - `struct Factor; mode::Symbol; port::Port; end`
  - `struct Term; coeff::Coefficient; factors::Vector{Factor}; end`; `modes(t::Term)::Vector{Symbol}`.
  - `circuit_terms(spec::Spec)::Vector{Term}`: the expansion of spec 5.3, exactly: for each mode $i$, `4*E_C[i,i]` on `n2port()`; for $i<j$, `4*(E_C[i,j] + E_C[j,i])` on `nport()` at modes $i, j$; for each $i$, `-8*Σ_j E_C[i,j]*n_g[j]` on `nport()` at $i$ (skipping when the result is identically zero); `E_L[i,i]/2` on `phi2port()`; `(E_L[i,j]+E_L[j,i])/2` on `phiport()` at $i,j$; `flux_linear[i]` on `phiport()`; for each cosine and each `s in (1, -1)`, coefficient `-(1/2)*energy*phase_factor(offset, s)` with factors `expiphi(s*omega[j])` for every $j$ with `omega[j] != 0`; for each charge cosine likewise with `exp2pin(s*kappa[j])` and offset `offset - 2π*Σ_j kappa[j]*n_g[j]`. Terms whose coefficient `iszero_expr` are dropped.
  - `constant_term(spec::Spec)::Coefficient` = `spec.constant + 4*n_gᵀ E_C n_g`.

- [ ] **Step 1: Write failing tests**

```julia
@testset "circuit_terms" begin
    θ = (E_J=8.0, E_C=1.0, E_L=0.5, varphi_ext=π)
    ts = circuit_terms(load_spec(fixture("fluxonium.json")))
    kinds = Dict((t.factors[1].port.kind, t.factors[1].port.k) => t.coeff(θ) for t in ts)
    @test kinds[(:n2, 0.0)] ≈ 4.0 && kinds[(:phi2, 0.0)] ≈ 0.25 && kinds[(:phi, 0.0)] ≈ 0.5π
    @test kinds[(:expiphi, 1.0)] ≈ -4.0 && kinds[(:expiphi, -1.0)] ≈ -4.0
    @test length(ts) == 5
    zp = circuit_terms(load_spec(fixture("two_zero_pi.json")))
    cross = only(filter(t -> Set(modes(t)) == Set([:theta1, :theta2]), zp))
    @test cross.coeff((g=0.1,)) ≈ 0.1
    @test constant_term(load_spec(fixture("transmon.json")))((E_C=0.3, n_g=0.25)) ≈ 4*0.3*0.25^2
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `src/terms.jl`.**
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add Port, Factor, Term and circuit_terms expansion"`.

---

### Task 5: Local operator structure types

**Files:**
- Create: `src/localops.jl`, `test/test_localops.jl`

**Interfaces:**
- Produces:
  - `abstract type LocalOp end`
  - `struct DenseOp <: LocalOp; M::Matrix{ComplexF64}; end` with `DenseOp(A::AbstractMatrix{<:Number})` converting to `ComplexF64`.
  - `struct DiagonalOp{T<:Number} <: LocalOp; d::Vector{T}; end`
  - `struct BandedOp{T<:Number} <: LocalOp; n::Int; offsets::Vector{Int}; weights::Vector{Vector{T}}; end` meaning $Y[i] = \sum_b \mathrm{weights}[b][i]\, X[i + \mathrm{offsets}[b]]$ for $1 \le i+\mathrm{offsets}[b] \le n$; `length(weights[b]) == n`.
  - `struct ShiftOp{T<:Number} <: LocalOp; n::Int; s::Int; factor::T; end` meaning $Y[i+s] = \mathrm{factor}\cdot X[i]$ for valid indices, i.e. $\mathrm{factor}\sum_i |i+s\rangle\langle i|$; in the charge basis $e^{i\varphi}$ is `ShiftOp(n, 1, 1)`.
  - `dimension(op::LocalOp)::Int`; `Base.adjoint(op::LocalOp)::LocalOp` returning the same family (`BandedOp` adjoint has `offsets .= -offsets` and shifted, conjugated weights; `ShiftOp` adjoint has `-s` and `conj(factor)`); `diagonal(op)::Vector{ComplexF64}`; `dense(op)::Matrix{ComplexF64}`.

- [ ] **Step 1: Write failing tests**

```julia
@testset "LocalOp" begin
    n = 7
    ops = [DenseOp(randn(n, n)), DiagonalOp(randn(ComplexF64, n)),
           BandedOp(n, [-2, 0, 1], [randn(ComplexF64, n) for _ in 1:3]), ShiftOp(n, 2, 0.5im)]
    for op in ops
        @test dimension(op) == n
        @test dense(op') ≈ dense(op)'
        @test diagonal(op) ≈ diag(dense(op))
        @test eltype(dense(op)) == ComplexF64
    end
    @test eltype(DenseOp(randn(3, 3)).M) == ComplexF64            # real input stored complex
    S = dense(ShiftOp(4, 1, 1.0)); @test S[2, 1] == 1 && S[1, 4] == 0  # truncates at edge
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `src/localops.jl`.**
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add local operator structure types with adjoint, diagonal, dense"`.

---

### Task 6: Bases

**Files:**
- Create: `src/bases.jl`, `test/test_bases.jl`

**Interfaces:**
- Consumes: `Port` (Task 4), `LocalOp` types (Task 5).
- Produces:
  - `abstract type Basis end`
  - `struct ChargeBasis <: Basis; ncut::Int; end`, states $n = -n_{\rm cut},\dots,n_{\rm cut}$, dimension $2n_{\rm cut}+1$.
  - `struct PhaseGrid <: Basis; a::Float64; b::Float64; N::Int; order::Int; end`, `PhaseGrid(a, b, N; order=8)`, points `range(a, b; length=N)`, `order ∈ (2,4,6,8)`.
  - `struct FockBasis <: Basis; N::Int; phi_zpf::Float64; end`, `FockBasis(N; phi_zpf)` (keyword required in this plan; the spec's default-from-first-materialize belongs to plan 4 with caching).
  - `dimension(b::Basis)::Int`, `modetype(b)::Symbol` (`:c`, `:e`, `:h`), `coordinates(b)::Vector{Float64}` (grid values; charge integers; Fock `0:N-1`), `positive_mask(b)::Vector{Bool}` (`coordinates .>= 0` for grid and charge; all `true` for Fock).
  - `fd_weights(order::Int, derivative::Int)::Vector{Float64}` central weights for `derivative ∈ (1, 2)`, length `order + 1`, indexed from $-order/2$ to $+order/2$; second-derivative weights must equal the thesis table (order 8: $w_0 = -205/72$, $w_{\pm1} = 8/5$, $w_{\pm2} = -1/5$, $w_{\pm3} = 8/315$, $w_{\pm4} = -1/560$).
  - `localop(b::Basis, p::Port)::LocalOp` per spec Section 6's table:
    - Charge: `n` → `DiagonalOp(-ncut:ncut)`, `n2` → squares, `expiphi(k)` → `ShiftOp(dim, Int(k), 1)` (throws unless `k` is within `1e-12` of an integer), `exp2pin(k)` → `DiagonalOp(exp.(2πim*k .* n))`, `phi`/`phi2` → `ArgumentError`.
    - Grid: `n` → `BandedOp` with offsets $-h..h$ and constant weights $-i\,w^{(1)}_j/\Delta$; `n2` → constant weights $-w^{(2)}_j/\Delta^2$; `phi` → `DiagonalOp(φ)`, `phi2` → `DiagonalOp(φ.^2)`, `expiphi(k)` → `DiagonalOp(exp.(im*k .* φ))`; `exp2pin(k)` → `ShiftOp(N, s, 1)` with $s = 2\pi k/\Delta$, throwing `ArgumentError` unless $|s - \mathrm{round}(s)| < 10^{-8}$ (Review Focus 3).
    - Fock: `n` → `DiagonalOp(0:N-1)`, `n2` → squares, `phi` → `BandedOp` offsets $\pm1$ with weights $\varphi_{\rm zpf}\sqrt{m}$, `phi2` → `BandedOp` offsets $0,\pm2$ equal to `dense(phi)^2` restricted to those bands, `expiphi(k)` → `DenseOp(displacement(N, im*k*phi_zpf))`, `exp2pin(k)` → `DenseOp(displacement(N, -2π*k*n_zpf))` with $n_{\rm zpf} = 1/(2\varphi_{\rm zpf})$.
  - `displacement(N::Int, α::Number)::Matrix{ComplexF64}` closed form: for $n \ge m$, $\sqrt{m!/n!}\,\alpha^{n-m}e^{-|\alpha|^2/2}L_m^{(n-m)}(|\alpha|^2)$, with $\langle n|D|m\rangle = \overline{\langle m|D(-\alpha)|n\rangle}$ for $n<m$; associated Laguerre by the three-term recurrence; factorial ratios in log space.
  - `edge_support(b::PhaseGrid, V::AbstractMatrix, w::Int)::Vector{Float64}`: per column, $\sqrt{\sum |V_i|^2}$ over the first and last `w` grid points.

- [ ] **Step 1: Write failing tests**

```julia
@testset "Bases" begin
    @test fd_weights(8, 2) ≈ [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
    g = PhaseGrid(-10, 10, 801; order=8); φ = coordinates(g); Δ = φ[2]-φ[1]
    ψ = exp.(-φ.^2 ./ 2)
    @test dense(localop(g, n2port())) * ψ ≈ -(φ.^2 .- 1) .* ψ atol=1e-7      # -∂²ψ, interior accuracy
    @test dense(localop(g, nport())) * ψ ≈ im .* φ .* ψ atol=1e-7           # -i∂ψ
    @test dense(localop(g, nport()))' ≈ dense(localop(g, nport()))          # Hermitian
    @test_throws ArgumentError localop(PhaseGrid(-1, 1, 10), exp2pin(1))     # Review Focus 3
    c = ChargeBasis(3)
    @test dense(localop(c, expiphi(1))) ≈ diagm(-1 => ones(6))               # |n+1><n|
    @test_throws ArgumentError localop(c, expiphi(0.5))
    @test_throws ArgumentError localop(c, phiport())
    f = FockBasis(40; phi_zpf=0.7); a = diagm(1 => sqrt.(1:39))
    @test dense(localop(f, phiport())) ≈ 0.7 * (a + a')
    D = displacement(40, 0.3im); @test (D * D')[1:20, 1:20] ≈ I(20) atol=1e-10
    @test D[1:10, 1:10] ≈ exp(0.3im*(a + a'))[1:10, 1:10] atol=1e-8
    @test positive_mask(g) == (φ .>= 0) && all(positive_mask(f))
    @test edge_support(g, reshape(ψ, :, 1), 4)[1] < 1e-15
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `src/bases.jl`.**
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add ChargeBasis, PhaseGrid, FockBasis with port operators"`.

---

### Task 7: Apply layer and KronFreeOperator

**Files:**
- Create: `src/apply.jl`, `test/test_apply.jl`
- Modify: `Project.toml` (add `KernelAbstractions`)

**Interfaces:**
- Consumes: `LocalOp` types (Task 5), `kron_reference` (Task 1).
- Produces:
  - `apply_axis!(Y::AbstractArray, op::LocalOp, X::AbstractArray, k::Int)::typeof(Y)`: `X` and `Y` have identical size, `size(X, k) == dimension(op)`; any number of axes, including a trailing block axis. Implementations: `DenseOp` → reshape to `(left, d_k, right)`, single `mul!` when `left == 1` or `right == 1`, else a loop over `right` of `mul!` with `transpose(op.M)`; `DiagonalOp` → broadcast with `reshape(d, ntuple(i -> i == k ? d_k : 1, ndims(X)))`; `BandedOp`, `ShiftOp` → `@kernel` functions over `length(Y)` work-items computing `(i_k, base)` from the linear index and `stride_k = prod(size(X)[1:k-1])`, launched on `get_backend(Y)` with workgroup 256, followed by `KernelAbstractions.synchronize`.
  - `struct AxisTerm; coeff::ComplexF64; factors::Vector{Tuple{Int,LocalOp}}; end` (axes strictly increasing, at most one per axis).
  - `mutable struct KronFreeOperator; dims::Vector{Int}; terms::Vector{AxisTerm}; constant::ComplexF64; buf1::Array{ComplexF64}; buf2::Array{ComplexF64}; end`, `KronFreeOperator(dims, terms; constant=0.0)`; buffers reallocated when a call's block width changes.
  - `Base.size(H)`, `Base.eltype(::KronFreeOperator) = ComplexF64`, `LinearAlgebra.ishermitian(H) = true` (documented assumption).
  - `LinearAlgebra.mul!(y::AbstractVecOrMat, H::KronFreeOperator, x::AbstractVecOrMat)`: throws `DimensionMismatch` unless `size(x, 1) == prod(dims)` and `size(y) == size(x)`; applies each term as a chain through the two buffers, accumulating `coeff * result` into `y`, then adds `constant * x`.
  - `Base.:*(H, x)`, `diagonal(H::KronFreeOperator)::Vector{ComplexF64}` = $\sum_t c_t \bigotimes \mathrm{diag}$ (ones on absent axes) + constant; `Base.Matrix(H)` = `H * I` via one block `mul!`.

- [ ] **Step 1: Write failing tests**

```julia
@testset "apply_axis! vs kron" begin
    dims = [3, 4, 2]
    rng = MersenneTwister(1)
    mk(d) = [DenseOp(randn(rng, ComplexF64, d, d)), DiagonalOp(randn(rng, ComplexF64, d)),
             BandedOp(d, [-1, 1], [randn(rng, ComplexF64, d) for _ in 1:2]), ShiftOp(d, 1, 2.0)]
    for k in 1:3, op in mk(dims[k])
        X = randn(rng, ComplexF64, dims...); Y = similar(X)
        apply_axis!(Y, op, X, k)
        @test vec(Y) ≈ kron_reference(dims, [(1.0+0im, [(k, dense(op))])]) * vec(X)
        Xb = randn(rng, ComplexF64, dims..., 5); Yb = similar(Xb)         # block axis
        apply_axis!(Yb, op, Xb, k)
        @test reshape(Yb, :, 5) ≈ kron_reference(dims, [(1.0+0im, [(k, dense(op))])]) * reshape(Xb, :, 5)
    end
end
@testset "KronFreeOperator" begin
    dims = [3, 4, 2]; ops = ...  # one op per axis as above
    terms = [AxisTerm(0.5+0im, [(1, ops[1]), (3, ops[3])]), AxisTerm(-1.0im, [(2, ops[2])])]
    H = KronFreeOperator(dims, terms; constant=0.25)
    ref = kron_reference(dims, [(0.5+0im, [(1, dense(ops[1])), (3, dense(ops[3]))]), (-1.0im, [(2, dense(ops[2]))])]) + 0.25I
    x = randn(ComplexF64, 24)
    @test H * x ≈ ref * x
    @test Matrix(H) ≈ ref
    @test diagonal(H) ≈ diag(ref)
    @test mul!(zeros(ComplexF64, 24, 3), H, randn(ComplexF64, 24, 3)) ≈ ref * ...   # block
    @test_throws DimensionMismatch H * randn(ComplexF64, 23)                        # Review Focus 4
    @test_throws DimensionMismatch mul!(zeros(ComplexF64, 24, 2), H, randn(ComplexF64, 24, 3))
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `src/apply.jl`.** Note that the `BandedOp` kernel reads `weights[b][i_k]` and `X[base + (i_k + off - 1)*stride_k + 1]` guarded by `1 ≤ i_k + off ≤ n`.
- [ ] **Step 4: Run tests → PASS.** Also run with `julia -t 4` to exercise the KA CPU backend threaded.
- [ ] **Step 5: Commit** `"Add apply layer: structure-dispatched apply_axis! and KronFreeOperator"`.

---

### Task 8: Tree, routing, required ports

**Files:**
- Create: `src/tree.jl`, `test/test_tree.jl`

**Interfaces:**
- Consumes: `Basis` (Task 6), `Term`, `Port` (Task 4).
- Produces:
  - `abstract type Solver end`; `struct Dense <: Solver end`; `struct Lanczos <: Solver; tol::Float64; krylovdim::Int; end` with `Lanczos(; tol=1e-10, krylovdim=0)` where `0` means "chosen by the solver" (Task 9).
  - `abstract type SelectionRule end` (implemented in Task 10; `tree.jl` only declares it).
  - `mutable struct Node; children::Vector{Any}; solver::Solver; compute::Int; keep::SelectionRule; phase::Symbol; ports::Vector{Tuple{Symbol,Port}}; end`
  - `Node(children::AbstractVector; solver=Dense(), compute=0, keep=lowest(0), phase=:positive_sector, ports=Tuple{Symbol,Port}[])` where each child is `Pair{Symbol,<:Basis}` or `Node`; mixing is allowed. `compute == 0` means all eigenpairs (only valid with `Dense`); `lowest(0)` means keep all computed. `phase ∈ (:positive_sector, :follow)`.
  - `modes(node::Node)::Vector{Symbol}` recursive, axis order; `isleaf(node)::Bool`; `axis_of(node, mode::Symbol)::Int` the child axis containing that mode.
  - `route(root::Node, terms::Vector{Term})::IdDict{Node,Vector{Term}}`: each term to the lowest node whose `modes` contain all of `modes(term)`; `ArgumentError` naming the mode if a term's mode is not in the tree, and if a spec mode is absent from the tree (every spec mode must appear exactly once).
  - `required_ports(root::Node, routed::IdDict)::IdDict{Node,Vector{Tuple{Symbol,Port}}}`: for every non-root node, the distinct `(mode, port)` pairs appearing in terms routed to any ancestor and touching a mode inside this node, plus `ports_needed(node.keep)` (Task 10) and `node.ports`; for the root, `ports_needed(root.keep)` and `root.ports` only.

- [ ] **Step 1: Write failing tests**

```julia
@testset "route and required_ports" begin
    g = PhaseGrid(-6π, 6π, 50); c = ChargeBasis(4)
    q1 = Node([:theta1 => c, :phi1 => g]); q2 = Node([:theta2 => c, :phi2 => g])
    top = Node([q1, q2])
    @test modes(top) == [:theta1, :phi1, :theta2, :phi2] && axis_of(top, :phi2) == 2
    ts = circuit_terms(load_spec(fixture("two_zero_pi.json")))
    r = route(top, ts)
    @test length(r[top]) == 1 && Set(modes(r[top][1])) == Set([:theta1, :theta2])
    @test all(t -> issubset(modes(t), [:theta1, :phi1]), r[q1])
    rp = required_ports(top, r)
    @test rp[q1] == [(:theta1, nport())] && rp[q2] == [(:theta2, nport())] && isempty(rp[top])
    @test_throws ArgumentError route(Node([:theta1 => c]), ts)
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `src/tree.jl`.**
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add Node tree with term routing and required-port inference"`.

---

### Task 9: Solvers

**Files:**
- Create: `src/solvers.jl`, `test/test_solvers.jl`
- Modify: `Project.toml` (add `KrylovKit`)

**Interfaces:**
- Consumes: `KronFreeOperator` (Task 7), `Dense`, `Lanczos` (Task 8).
- Produces: `solve(::Dense, H::KronFreeOperator, m::Int)::Tuple{Vector{Float64},Matrix{ComplexF64}}` via `eigen(Hermitian(Matrix(H)))`, all eigenpairs ascending (ignores `m`); `solve(::Lanczos, H, m)` via `KrylovKit.eigsolve(v -> H*v, randn(ComplexF64, n), m, :SR; ishermitian=true, tol, krylovdim=max(2m+10, krylovdim))`, returning exactly `m` ascending pairs as a matrix, throwing `ErrorException` with the residual info when fewer than `m` converged.

- [ ] **Step 1: Write failing tests**

```julia
@testset "solvers agree" begin
    H = three_oscillators(6)    # helper, see below
    Ed, Vd = solve(Dense(), H, 0)
    El, Vl = solve(Lanczos(tol=1e-10), H, 5)
    @test El ≈ Ed[1:5] atol=1e-8
    @test all(abs.(diag(Vl' * Vd[:, 1:5])) .≈ 1)   # same vectors up to phase
    @test Ed ≈ eigvals(Hermitian(kron_reference(...)))
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Add `three_oscillators(d)::KronFreeOperator` to `test/helpers.jl`**: `dims = [d, d, d]`; terms `ω_i * n_i` with `ω = (1.0, 1.1, 1.2)` and `n_i = DiagonalOp(0:d-1)`; couplings `0.2 * x_i x_j` for `(i,j) in ((1,2),(2,3),(1,3))` with `x_i = DenseOp(cos(0.5*(a + a')))` built from the truncated `a` (the spike's `cos_op`). Also add `kron_three_oscillators(d)::Matrix` for the reference, using `kron_reference`. Then implement `src/solvers.jl`.
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add Dense and Lanczos solvers behind one solve interface"`.

---

### Task 10: Selection rules and phase conventions

**Files:**
- Create: `src/selection.jl`, `src/phase.jl`, `test/test_selection.jl`

**Interfaces:**
- Consumes: `SelectionRule` (Task 8), `Port` (Task 4).
- Produces:
  - `struct SelectionContext; energies::Vector{Float64}; vectors::Matrix{ComplexF64}; ports::Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}; previous::Any; end` where `previous` is `nothing` or the node's previous `MaterializedNode` (Task 11).
  - `lowest(k::Int)`, `indices(v::AbstractVector{Int})`, `connected(core::Vector{Int}, port::Tuple{Symbol,Port}, thresh::Float64)`, `follow(reference::Union{Nothing,AbstractMatrix}; minoverlap=0.5)` all `<: SelectionRule`.
  - `select(rule, ctx::SelectionContext)::Vector{Int}`; `ports_needed(rule)::Vector{Tuple{Symbol,Port}}` (`connected` → its port; others empty).
  - `connected` semantics (spec 9.1, `trunc_by_thresh`): start with `core`; iterate over the growing list; for each state `s` add every `j` with `abs(P[s, j]) > thresh` not yet included; return `sort(list)`. `lowest(0)` returns `1:length(energies)`.
  - `follow`: reference `R` (columns) is `rule.reference`, else `ctx.previous.vectors`, else `ArgumentError`; overlap `O = abs.(R' * V)`; greedy assignment in reference order, each reference taking its largest-overlap unassigned column; `@warn` listing reference indices whose chosen overlap is below `minoverlap`; returns indices in reference order.
  - `validate_selection(idx, compute::Int, rule)::Vector{Int}`: throws `ArgumentError` naming `typeof(rule)` for empty, duplicate, or out-of-range indices (Review Focus 5).
  - `fix_phases!(V::AbstractMatrix, convention::Symbol, mask::AbstractVector{Bool}, reference)`: `:positive_sector` multiplies each column by $e^{-i\arg V[i^*,j]}$ where $i^*$ is the index of largest modulus among `mask`; `:follow` by $e^{-i\arg(\langle r_j|v_j\rangle)}$ with `reference` columns (throws `ArgumentError` when `reference === nothing`). Returns `V`.

- [ ] **Step 1: Write failing tests**

```julia
@testset "selection rules" begin
    P = zeros(ComplexF64, 6, 6); P[1,4] = P[4,1] = 0.5; P[4,6] = P[6,4] = 0.2; P[2,3] = P[3,2] = 0.9
    ctx = SelectionContext(collect(1.0:6), Matrix{ComplexF64}(I, 6, 6), Dict((:m, nport()) => P), nothing)
    @test select(connected([1, 2], (:m, nport()), 0.1), ctx) == [1, 2, 3, 4, 6]
    @test ports_needed(connected([1], (:m, nport()), 0.1)) == [(:m, nport())]
    @test select(lowest(3), ctx) == [1, 2, 3] && select(lowest(0), ctx) == 1:6
    @test_throws ArgumentError validate_selection([1, 1], 6, lowest(2))     # Review Focus 5
    @test_throws ArgumentError validate_selection(Int[], 6, lowest(0))
    @test_throws ArgumentError validate_selection([7], 6, indices([7]))
    V = Matrix{ComplexF64}(I, 6, 6)[:, [3, 1, 2, 4, 5, 6]]                   # permuted eigenvectors
    @test select(follow(Matrix{ComplexF64}(I, 6, 6)[:, 1:3]), SelectionContext(ctx.energies, V, ctx.ports, nothing)) == [2, 3, 1]
    @test_logs (:warn,) select(follow(Matrix{ComplexF64}(I, 6, 6)[:, 1:1]), SelectionContext(ctx.energies, V .* 0.1 .+ 0.05, ctx.ports, nothing))
end
@testset "fix_phases!" begin
    v = [0.3, -0.8im, 0.5]; mask = [false, true, true]
    W = fix_phases!(reshape(copy(v), :, 1), :positive_sector, mask, nothing)
    @test W[2, 1] ≈ 0.8 && abs(W[1, 1]) ≈ 0.3                                # largest in positive sector real positive
    r = reshape(exp(0.4im) .* v, :, 1)
    @test fix_phases!(reshape(copy(v), :, 1), :follow, mask, r) ≈ r
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `src/selection.jl` and `src/phase.jl`.**
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add selection rules (lowest, indices, connected, follow) and phase conventions"`.

---

### Task 11: Leaf materialization against scqubits

**Files:**
- Create: `src/materialize.jl`, `test/test_materialize.jl`, `test/fixtures/generate_scqubits.py`, `test/fixtures/scqubits_reference.json`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `struct MaterializedNode; node::Node; dims::Vector{Int}; energies::Vector{Float64}; vectors::Matrix{ComplexF64}; kept::Vector{Int}; ports::Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}; children::Vector{Any}; end` where `children` holds `Basis` or `MaterializedNode` in axis order and `vectors` is `prod(dims) × length(kept)` in the node's product basis.
  - `struct System; root::MaterializedNode; spec::Spec; θ::NamedTuple; end`
  - `materialize(root::Node, spec::Spec, θ::NamedTuple; previous::Union{Nothing,System}=nothing)::System`. Per node, children first: build `AxisTerm`s from the routed `Term`s (coefficients evaluated at `θ`, factors from `localop(basis, port)` for a basis child or `DenseOp(child.ports[(mode, port)])` for a node child), plus one `AxisTerm(1, [(axis, DiagonalOp(child.energies))])` per node child, plus `constant_term(spec)` at the root only; `solve`; `fix_phases!` with `mask` = Kronecker product of children's `positive_mask` (all-true for node children) and the previous node's vectors as reference; form the ports in `ports_needed(keep)` on all computed vectors as `V' * apply(O, V)`; `select`; `validate_selection`; slice; form the remaining required ports. Missing `θ` entries surface as the `ArgumentError` of Task 2 (Review Focus 1).
  - `energies(sys::System, node::Node=sys.root.node)::Vector{Float64}`, `port(sys, mode::Symbol, p::Port, node=root)::Matrix{ComplexF64}` (throws `KeyError` with the available ports listed if absent), `eigenvectors(sys, node)::Matrix{ComplexF64}`, `materialized(sys, node::Node)::MaterializedNode` by tree position.
  - A grid leaf logs `@warn` when `edge_support(grid, kept vectors, order ÷ 2)` exceeds `1e-6` for any kept vector.
- Test helpers added to `test/helpers.jl` in this task: `reference(name::String)` reads `scqubits_reference.json` and returns a NamedTuple with `energies::Vector{Float64}` and the parameters; `fixture(name)` returns the path under `test/fixtures`; `two_transmons_params(; g)` returns `(E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.0, g=g)`; `three_transmons_params(; g)` adds `E_J3=20.0, E_C3=0.8, n_g3=0.1`; `zero_pi_params()` returns `(E_J=10.0, E_L=0.04, E_CJ=20.0, E_CS=0.5, varphi_ext=2π*0.23, n_g=0.1)`; `full_zero_pi_params()` adds `E_Cz` (from the reference file), `dE_J=0.05, dC_J=0.05, dC=0.08, dE_L=0.1`; `two_zero_pi_params(; g)` returns `zero_pi_params()` duplicated with suffixes `1` and `2` plus `g`.
- Reference generator `test/fixtures/generate_scqubits.py` (run once with `conda run -n zp2q python test/fixtures/generate_scqubits.py`; its output `scqubits_reference.json` is committed): fluxonium `EJ=8.9, EC=2.5, EL=0.5, flux=0.5` with `cutoff=200`, lowest 6; transmon `EJ=30, EC=1.2, ng=0.25`, `ncut=30`, lowest 5; two transmons `EJ1=30, EC1=1.2, ng1=0.25, EJ2=25, EC2=1.0, ng2=0.0`, `ncut=30`, `truncated_dim=8` each, `InteractionTerm(g=0.3, n_operator ⊗ n_operator)`, lowest 20 of the `HilbertSpace`; zero-pi `EJ=10, EL=0.04, ECJ=20, ECS=0.5 (pass EC=None, ECS=0.5), ng=0.1, flux=0.23`, φ grid `-6π..6π` with `pt_count=1001`, `ncut=15`, lowest 8; full zero-pi `FullZeroPi` with the same `EJ, EL, ECJ, ECS, ng, flux`, `dEJ=0.05, dCJ=0.05, dC=0.08, dEL=0.1`, `ncut=15`, same grid, `zeropi_cutoff=30, zeta_cutoff=20`, lowest 10; two zero-pi: two `ZeroPi` subsystems as above with `truncated_dim=10`, `InteractionTerm(g=0.05, n_theta_operator ⊗ n_theta_operator)`, lowest 20 of the `HilbertSpace`. All energies absolute, GHz, stored with the parameters used, including scqubits' derived zeta charging energy for the full zero-pi so the Julia side can pass the same `E_Cz`. The two-zero-pi entry is consumed only by Task 14.

- [ ] **Step 1: Write failing tests**

```julia
@testset "fluxonium leaf vs scqubits" begin
    ref = reference("fluxonium"); θ = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π)
    leaf = Node([:varphi_1 => PhaseGrid(-10π, 10π, 1201; order=8)]; solver=Dense(), keep=lowest(6))
    sys = materialize(leaf, load_spec(fixture("fluxonium.json")), θ)
    @test energies(sys) ≈ ref.energies atol=1e-6
    @test_throws ArgumentError materialize(leaf, load_spec(fixture("fluxonium.json")), (E_J=8.9,))   # Review Focus 1
end
@testset "transmon leaf vs scqubits (same discretization)" begin
    ref = reference("transmon"); θ = (E_J=30.0, E_C=1.2, n_g=0.25)
    sys = materialize(Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(5)), load_spec(fixture("transmon.json")), θ)
    @test energies(sys) ≈ ref.energies atol=1e-10
end
@testset "zero-pi leaf vs scqubits (Lanczos on a 9k-dimensional leaf)" begin
    ref = reference("zero_pi"); θ = zero_pi_params(); spec_zp = load_spec(fixture("zero_pi.json"))
    leaf = Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-6π, 6π, 301; order=8)];
                solver=Lanczos(), compute=8, keep=lowest(8), ports=[(:theta, nport())])
    sys = materialize(leaf, spec_zp, θ)
    @test energies(sys) ≈ ref.energies atol=1e-4            # scqubits uses a 3-point stencil on 1001 points
    @test port(sys, :theta, nport()) ≈ port(sys, :theta, nport())'
    small = Node([:theta => ChargeBasis(8), :phi => PhaseGrid(-6π, 6π, 151)]; keep=lowest(8))
    smalll = Node([:theta => ChargeBasis(8), :phi => PhaseGrid(-6π, 6π, 151)]; solver=Lanczos(), compute=8, keep=lowest(8))
    @test energies(materialize(smalll, spec_zp, θ)) ≈ energies(materialize(small, spec_zp, θ)) atol=1e-8   # Lanczos = Dense
end
@testset "requested port and edge warning" begin
    θ = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=π); spec = load_spec(fixture("fluxonium.json"))
    sys = materialize(Node([:varphi_1 => PhaseGrid(-10π, 10π, 401)]; keep=lowest(6), ports=[(:varphi_1, nport())]), spec, θ)
    P = port(sys, :varphi_1, nport()); @test size(P) == (6, 6) && P ≈ P'
    @test_logs (:warn,) materialize(Node([:varphi_1 => PhaseGrid(-2, 2, 41)]; keep=lowest(6)), spec, θ)   # too-small grid warns
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Write `generate_scqubits.py`, run it, commit its output; implement `src/materialize.jl`.**
- [ ] **Step 4: Run tests → PASS.** If the zero-pi comparison fails at the 1e-4 level, first check the flux convention (`flux = varphi_ext / 2π`) and the `E_L` factor of 2 against scqubits' `ZeroPi.hamiltonian` source before touching the solver.
- [ ] **Step 5: Commit** `"Add leaf materialization with scqubits reference tests"`.

---

### Task 12: Branch materialization and the two-transmon composite

**Files:**
- Modify: `src/materialize.jl`, `test/test_materialize.jl`

**Interfaces:**
- Consumes: Task 11.
- Produces: branch handling inside `materialize` (node children as `DenseOp` ports and energy diagonals, as specified in Task 11); nothing new in the public API.

- [ ] **Step 1: Write failing tests**

```julia
@testset "branch equals explicit product-basis construction" begin
    θ = two_transmons_params(g=0.3); spec = load_spec(fixture("two_transmons.json"))
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20), ports=[(:varphi_1, nport())])
    sys = materialize(top, spec, θ)
    q1, q2 = materialized(sys, top.children[1]), materialized(sys, top.children[2])
    Hprod = kron(I(8), Diagonal(q1.energies)) + kron(Diagonal(q2.energies), I(8)) +
            0.3 * kron(q2.ports[(:varphi_2, nport())], q1.ports[(:varphi_1, nport())])
    @test energies(sys) ≈ eigvals(Hermitian(Hprod))[1:20] atol=1e-10
    @test size(port(sys, :varphi_1, nport())) == (20, 20)
end
@testset "two transmons vs scqubits HilbertSpace" begin
    @test energies(sys) ≈ reference("two_transmons").energies atol=1e-8     # identical discretization
end
@testset "lanczos at the top matches dense" begin
    top_l = Node([mk(1), mk(2)]; solver=Lanczos(), compute=6, keep=lowest(6))
    @test energies(materialize(top_l, spec, θ)) ≈ energies(sys)[1:6] atol=1e-8
end
@testset "full zero-pi: all three mode types in a two-level tree vs scqubits FullZeroPi" begin
    ref = reference("full_zero_pi"); θ = full_zero_pi_params()
    zp = Node([:theta => ChargeBasis(15), :phi => PhaseGrid(-6π, 6π, 301; order=8)]; solver=Lanczos(), compute=30, keep=lowest(30))
    zeta = Node([:zeta => FockBasis(20; phi_zpf=(2θ.E_Cz/θ.E_L)^(1/4))]; keep=lowest(20))   # scqubits' oscillator basis for zeta
    full = Node([zp, zeta]; solver=Dense(), keep=lowest(10))
    sys = materialize(full, load_spec(fixture("full_zero_pi.json")), θ)
    @test energies(sys) ≈ ref.energies atol=1e-4      # same truncations as scqubits (zeropi_cutoff=30, zeta_cutoff=20)
    @test modes(full) == [:theta, :phi, :zeta]
end
@testset "three-level tree" begin
    spec3 = load_spec(fixture("three_transmons.json")); θ3 = three_transmons_params(g=0.3)
    inner = Node([mk(1), mk(2)]; keep=lowest(12))
    third = Node([:varphi_3 => ChargeBasis(10)]; keep=lowest(4))
    outer = Node([inner, third]; keep=lowest(10))
    sys3 = materialize(outer, spec3, θ3)
    Ein = materialized(sys3, inner).energies; Eth = materialized(sys3, third).energies
    @test energies(sys3) ≈ sort(vec(Ein .+ Eth'))[1:10] atol=1e-10        # uncoupled third mode: energies add
    @test Ein ≈ energies(sys)[1:12] atol=1e-10                            # inner node equals the two-transmon top
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement the branch path.**
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add branch materialization validated against product-basis and scqubits two-transmon composites"`.

---

### Task 13: Dressed and bare states

**Files:**
- Create: `src/dressed.jl`, `test/test_dressed.jl`

**Interfaces:**
- Consumes: `System`, `MaterializedNode`, `materialized` (Task 11).
- Produces (spec 10.3):
  - `bare_labels(sys::System, node::Node=root; depth::Int=1)::Vector{Tuple}`: `prod(dims)` tuples of child kept-indices (positions `1:length(child.kept)`), axis 1 fastest; with `depth > 1` each node-child index is replaced by that child's dominant bare label (its `decompose(...)[1]`) recursively, basis children contribute their basis index.
  - `bare_overlaps(sys, node)::Matrix{Float64}` of size `length(kept) × prod(dims)`, entries $|V_{pi}|^2$.
  - `decompose(sys, node, i::Int; thresh=1e-3)::Vector{Pair{Tuple,ComplexF64}}` sorted by descending weight, amplitudes with the node's phase convention.
  - `dressed_index(sys, node, label::Tuple)::Union{Int,Nothing}`: index of the kept state with largest overlap with that product state; `@warn` when that overlap is below `0.5`; `nothing` when the product state's total weight over kept states is below `1e-12`.

- [ ] **Step 1: Write failing tests** (on the two-transmon system of Task 12 with `g=0.0` and `g=0.3`)

```julia
@testset "dressed/bare" begin
    sys0 = materialize(top, spec, two_transmons_params(g=0.0))
    @test length(bare_labels(sys0, top)) == 64 && bare_labels(sys0, top)[2] == (2, 1)
    @test all(maximum(bare_overlaps(sys0, top); dims=2) .≈ 1)           # uncoupled: exact product states
    @test dressed_index(sys0, top, (1, 1)) == 1
    sys = materialize(top, spec, two_transmons_params(g=0.3))
    d = decompose(sys, top, 1)
    @test d[1].first == (1, 1) && sum(abs2, last.(d)) ≈ 1 atol=1e-3
    @test dressed_index(sys, top, d[1].first) == 1
    @test dressed_index(sys, top, (8, 8)) === nothing || isa(dressed_index(sys, top, (8, 8)), Int)
    @test length(bare_labels(sys, top; depth=2)[1]) == 2 && bare_labels(sys, top; depth=2)[1][1] isa Tuple
end
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `src/dressed.jl`.**
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** `"Add dressed/bare state interface"`.

---

### Task 14: Zero-pi final reference tests (gated)

**Files:**
- Create: `test/test_zero_pi_reference.jl`
- Modify: `test/runtests.jl` to include it only when `get(ENV, "KRONFREE_SLOW_TESTS", "0") == "1"`

**Interfaces:**
- Consumes: everything above; the `zero_pi` and `two_zero_pi` entries of `scqubits_reference.json` (Task 11).

This is the project's acceptance test against the system that motivated it: two coupled zero-pi qubits. It is slow and runs only when the flag is set, so the normal suite stays fast. Eli's instruction: do not let it drive the schedule; everything else lands first. The single zero-pi and the full zero-pi are already covered in Tasks 11 and 12.

- [ ] **Step 1: Write the tests**

```julia
@testset "two zero-pi vs scqubits HilbertSpace" begin
    θ = two_zero_pi_params(g=0.05)
    mk(i) = Node([Symbol("theta$i") => ChargeBasis(15), Symbol("phi$i") => PhaseGrid(-6π, 6π, 301)]; solver=Lanczos(), compute=10, keep=lowest(10))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20))
    sys = materialize(top, load_spec(fixture("two_zero_pi.json")), θ)
    @test energies(sys) ≈ reference("two_zero_pi").energies atol=1e-4
    @test length(bare_labels(sys, top)) == 100
end
```

- [ ] **Step 2: Run with the flag**: `KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'` → PASS. Record the wall time of this run in the commit message.
- [ ] **Step 3: Run without the flag** and confirm the file is skipped.
- [ ] **Step 4: Commit** `"Add gated two-zero-pi acceptance test against scqubits"`.

---

### Task 15: Exports, README, full test run

**Files:**
- Modify: `src/KronFreeCircuits.jl` (exports), create `README.md`

- [ ] **Step 1: Export** `Spec, load_spec, parse_spec, Coefficient, Port, nport, n2port, phiport, phi2port, expiphi, exp2pin, Factor, Term, circuit_terms, LocalOp, DenseOp, DiagonalOp, BandedOp, ShiftOp, dense, diagonal, Basis, ChargeBasis, PhaseGrid, FockBasis, localop, apply_axis!, KronFreeOperator, Node, Dense, Lanczos, lowest, indices, connected, follow, materialize, System, energies, port, eigenvectors, materialized, bare_labels, bare_overlaps, decompose, dressed_index`.
- [ ] **Step 2: Write `README.md`**: one paragraph of purpose, the user-facing sketch from spec Section 13 adapted to what this plan builds (no LOBPCG, no gradient yet), the `KRONFREE_SLOW_TESTS=1` flag for the zero-pi acceptance tests, a pointer to the design document and the plan sequence.
- [ ] **Step 3: Run the full suite** `julia --project=. -e 'using Pkg; Pkg.test()'` and `julia -t 8 --project=. -e 'using Pkg; Pkg.test()'` → all PASS.
- [ ] **Step 4: Commit** `"Export public API and add README"`.

---

## Self-review notes

- Spec coverage for this plan's scope: Sections 4, 5.2–5.3, 6, 7, 8 (CPU), 9 (Dense, Lanczos, selection), 9.1 phase conventions, 10.1–10.3, 13 (shape), 14, 16 (CPU parts) are covered by Tasks 2–14. Per Eli: the two-transmon composite drives Tasks 12 and 13; the single zero-pi (Task 11) and the full asymmetric zero-pi with all three mode types (Task 12) are in the main line; only the two coupled zero-pis are the gated final acceptance test (Task 14). Deferred by design to later plans: 5.1, 8 (CUDA), 9 (LOBPCG), 10.4, 11, 12, 13.1, 13.2, the `FockBasis` default `phi_zpf`.
- Names used consistently across tasks: `apply_axis!`, `AxisTerm`, `KronFreeOperator`, `localop`, `dimension`, `diagonal`, `dense`, `nport()` etc., `route`, `required_ports`, `ports_needed`, `select`, `validate_selection`, `fix_phases!`, `MaterializedNode`, `System`, `materialized`, `energies`, `port`.
- Review Focus items are pinned: 1 → Tasks 2 and 11; 2 → Task 3; 3 → Task 6; 4 → Task 7; 5 → Task 10.
