# Born–Oppenheimer Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work happens on branch `born-oppenheimer` (off `plan4-julia-integration`), never on `main`.

**Goal:** Single-surface BO reduction of a spec over a compact (or, through fits, extended) slow mode, with linear least-squares fits to a reduced one-mode spec, accepted by reproducing Di Paolo et al.'s zero-pi E_α and E_γ to 0.1%.

**Architecture:** Everything is spec-to-spec. `promote_mode` turns a mode into a parameter and returns the fast spec plus the slow mode's kinetic data; `bo_surface` sweeps that parameter with ordinary `materialize` calls; `slow_spec` turns a compact-mode surface's Fourier coefficients into `cosines`; `fit_surfaces` fits surfaces at many parameter values to a model linear in its constants and emits a reduced spec, with held-fixed parameters substituted by value. A Julia spec writer saves the result in kronfree-spec version 1.

**Tech Stack:** Julia 1.13, LinearAlgebra, JSON3 (already a dependency). No new dependencies; Fourier coefficients are direct sums.

**Spec:** `docs/superpowers/specs/2026-10-19-born-oppenheimer-design.md`, with `docs/science/born-oppenheimer.md` (parameter mapping, published targets, protocol). Executors read both before their task.

## Global Constraints

- Single-surface BO only; no derivative couplings, Born–Huang terms or coupled bands (design §2).
- Nothing is silently approximated: a spec part with no exact carry-over is an `ArgumentError` naming it, never dropped (design §2, §4.1).
- Charge coupling `E_C[s,f]` and charge cosines with `kappa[s] ≠ 0` are rejected structurally (by expression, `_iszero` in `src/terms.jl`), never by value (design §4.1).
- No data-valued `DiagonalOp` entry into a `Term`; it is deferred to plan 5 (design §2, §4.3). An extended slow mode reaches a reduced spec only through a fit.
- Fits are linear least squares in the unknown constants only; the user supplies all basis functions (design §2).
- Parameters held fixed while computing surfaces are substituted by value in every reduced spec; a parameter stays free only if it is fitted over or the fast spec does not depend on it (design §4.4).
- Written specs are kronfree-spec version 1 per `~/src/circuit-quantization/docs/spec-format.md`: explicit `*` (no implicit multiplication), only the grammar's operations and constants, underscore keys for metadata, `_generator` for provenance.
- Python is never a test-time dependency; Python runs only in a committed validation script, with `/opt/anaconda3/envs/circuit_quantization/bin/python`.
- Default suite: at most +10 s over the baseline measured in Task 1 Step 0; the zero-pi reproduction and refinement run only under `KRONFREE_SLOW_TESTS=1`.
- Acceptance (design §7, decided): published input ratios treated as exact; E_α and E_γ to 0.1% relative as a hard gated assertion; a miss is a bug to chase, never a loosened tolerance. If a miss cannot be explained, stop and report to Eli with the refinement data.
- Running one test file: `julia --project=. -e 'using Test, LinearAlgebra, SparseArrays, Random, KronFreeCircuits; include("test/helpers.jl"); include("test/<file>.jl")'` (the pattern `docs/science/README.md` uses). "Run the test file" below means this command.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the model that wrote them).

## Review Focus

1. A promoted parameter name that collides with an existing spec parameter, a reserved name (`pi`, `im`, `e`, …) or a remaining mode name must throw an `ArgumentError` naming it, not shadow silently. Pinned in Task 3.
2. Surfaces passed to `fit_surfaces` that disagree in a held-fixed parameter (two runs at different `E_J`) must throw naming the parameter, since the reduced spec would otherwise pick one value silently. Pinned in Task 5.
3. `slow_spec` on a non-uniform or non-periodic grid (the endpoint 2π included, so a point is duplicated) must throw rather than return wrong Fourier coefficients. Pinned in Task 4.
4. A spec expression with a negative literal, a power, nested unary minus or `π` must survive the writer and parse back to the same function (`2E_J` printed as implicit multiplication would be rejected by Python). Pinned in Task 1.
5. A model term naming a harmonic or slow basis function that does not fit the slow mode's type (`:x2` on a compact mode, non-integer `k` on a compact mode) must throw, since the reduced spec could not express it. Pinned in Task 5.

---

### Task 1: Spec writer

**Files:**
- Create: `src/specwrite.jl`
- Modify: `src/KronFreeCircuits.jl` (include after `spec.jl`; export `spec_json`, `save_spec`)
- Test: `test/test_specwrite.jl`

**Interfaces:**
- Consumes: `Spec`, `ModeSpec`, `CosineSpec`, `ChargeCosineSpec`, `Coefficient` (`ex` field), `parse_spec`.
- Produces:
  - `expr_string(ex::Union{Expr,Symbol,Number}) -> String`: grammar-conforming text; every binary operation parenthesized, explicit `*`, `Float64` via `repr` (shortest round-trip), negative literals parenthesized, `π`/`pi` as `pi`, `ℯ` as `ℯ`. Internal (not exported).
  - `spec_json(spec::Spec; metadata::AbstractDict{String}=Dict{String,Any}()) -> String`: `format`, `version`, `units = "GHz"`, every required key, `charge_cosines` only when nonempty, then each metadata entry, whose keys must start with `_` (else `ArgumentError`).
  - `save_spec(path::AbstractString, spec::Spec; metadata=Dict{String,Any}()) -> path`.

- [ ] **Step 0: Baseline.** On `born-oppenheimer`, run `julia --project=. -e 'using Pkg; Pkg.test()'` and record the wall time in the task report; the default-suite budget is that time + 10 s.

- [ ] **Step 1: Write the failing tests** in `test/test_specwrite.jl`:

```julia
@testset "spec writer" begin
    # every hand-written and generated fixture round-trips to the same functions
    for f in vcat(fixture.(filter(endswith(".json"), readdir(joinpath(@__DIR__, "fixtures")))),
                  fixture.("generated/" .* filter(endswith(".json"), readdir(joinpath(@__DIR__, "fixtures", "generated")))))
        s = load_spec(f)
        @test spec_equal(parse_spec(spec_json(s)), s)
    end
    # hard expressions (Review Focus 4)
    for src in ("-2*E_J", "E_J^2/(-3.5)", "-(-(E_J))", "pi*E_J/2", "2*E_J", "1e-300*E_J", "1 // 3")
        s = parse_spec(minimal_spec_json(constant=src))      # one compact mode, parameter E_J
        @test !occursin(r"\d(?![eE][-+]?\d)[A-Za-z_]", expr_text(spec_json(s), "constant"))   # no implicit multiplication
        @test spec_equal(parse_spec(spec_json(s)), s)
    end
    @test_throws ArgumentError spec_json(load_spec(fixture("transmon.json")); metadata=Dict("comment" => 1))
    md = Dict{String,Any}("_generator" => Dict("package" => "KronFreeCircuits"))
    @test KronFreeCircuits.JSON3.read(spec_json(load_spec(fixture("transmon.json")); metadata=md))._generator.package == "KronFreeCircuits"
end
```

`spec_equal(a::Spec, b::Spec)` and `minimal_spec_json(; constant)` go in `test/helpers.jl`: `spec_equal` requires equal modes, equal parameter sets, equal `omega`/`kappa`, and every coefficient pair agreeing to 1e-12 relative at three fixed pseudo-random points in [0.5, 2] (seeded, sorted parameter order). `expr_text(json, key)` reads one expression string back out of the JSON.

- [ ] **Step 2: Run the test file.** Expected: FAIL, `spec_json` not defined.

- [ ] **Step 3: Implement `expr_string`, `spec_json`, `save_spec`** in `src/specwrite.jl` as specified in Interfaces. Use JSON3 for the object; key order as in the format document's complete example.

- [ ] **Step 4: Run the test file.** Expected: PASS.

- [ ] **Step 5: Commit** `src/specwrite.jl src/KronFreeCircuits.jl test/test_specwrite.jl test/helpers.jl` with message `Spec writer: spec_json and save_spec in kronfree-spec version 1`.

### Task 2: Parameter substitution

**Files:**
- Create: `src/spectransform.jl`
- Modify: `src/KronFreeCircuits.jl` (include after `specwrite.jl`; export `substitute`)
- Test: `test/test_spectransform.jl`

**Interfaces:**
- Consumes: `Spec`, `Coefficient`, `_make` and `_iszero` (internal), `spec_equal` (test helper).
- Produces:
  - `substitute(c::Coefficient, values::NamedTuple) -> Coefficient`: every symbol in `keys(values)` replaced by its `Float64` value, then every subtree with no free symbols folded to one `Float64` literal (evaluated as a `ComplexF64` and required to have zero imaginary part, else `ArgumentError`); `deps` recomputed.
  - `substitute(spec::Spec, values::NamedTuple) -> Spec`: every coefficient substituted; substituted names removed from `parameters`; cosines and charge cosines whose `energy` is `_iszero` after substitution removed. Values must be `Real` (`ArgumentError` otherwise); an unknown name is an `ArgumentError` naming it.

- [ ] **Step 1: Write the failing tests** in `test/test_spectransform.jl`:

```julia
@testset "substitute" begin
    full = load_spec(fixture("full_zero_pi.json"))       # modes theta (c), phi (e), zeta (h)
    s = substitute(full, (dC_J=0.0,))
    @test :dC_J ∉ s.parameters
    @test KronFreeCircuits._iszero(s.E_C[1, 2])          # E_CS*dC_J/4 -> 0
    @test !KronFreeCircuits._iszero(s.E_C[1, 3])         # -sqrt(2)*E_CS*dC/4 untouched
    @test coefficients_agree(s, full, (dC_J=0.0,))
    c = substitute(Coefficient("E_J*cos(varphi_ext) + 2", [:E_J, :varphi_ext]), (varphi_ext=0.0,))
    @test c.deps == [:E_J] && c((E_J=3.0,)) ≈ 5.0
    @test substitute(Coefficient("(1-1)*E_J", [:E_J]), NamedTuple()) |> KronFreeCircuits._iszero
    @test_throws ArgumentError substitute(full, (nonexistent=1.0,))
    @test_throws ArgumentError substitute(full, (dCJ=1.0im,))
end
```

`coefficients_agree(s, full, values)` (helpers): every coefficient of `s` at three seeded random points in [0.5, 2] equals the matching coefficient of `full` at the same points merged with `values`, to 1e-12 relative.

- [ ] **Step 2: Run the test file.** Expected: FAIL, `substitute` not defined.
- [ ] **Step 3: Implement both `substitute` methods** in `src/spectransform.jl`.
- [ ] **Step 4: Run the test file.** Expected: PASS.
- [ ] **Step 5: Commit** with message `substitute: replace spec parameters by values and drop entries that vanish`.

### Task 3: Promote a mode to a parameter

**Files:**
- Modify: `src/spectransform.jl`, `src/KronFreeCircuits.jl` (export `promote_mode`, `PromotedSpec`)
- Test: `test/test_spectransform.jl`

**Interfaces:**
- Consumes: `Spec`, `substitute` is not used here (structural checks only).
- Produces:

```julia
struct PromotedSpec
    fast::Spec            # the remaining modes; parameters = source parameters ∪ [parameter]
    source::Spec
    mode::ModeSpec        # the slow mode (name, type)
    parameter::Symbol     # name of the new parameter holding the slow coordinate
    E_C::Coefficient      # source E_C[s,s]
    n_g::Coefficient      # source n_g[s]
end
promote_mode(spec::Spec, mode::Symbol; parameter::Symbol=mode) -> PromotedSpec
```

Carry-over exactly per design §4.1's table: `E_C[s,s]`, `n_g[s]` dropped from `fast` (an `@info` names them); `E_L[s,s]` → constant `½ E_L[s,s] p²`; `E_L[s,f]` → `flux_linear[f] += E_L[s,f] p`; `flux_linear[s]` → constant `flux_linear[s] p`; a cosine with `omega[s] ≠ 0` keeps its other multipliers and gets `offset + omega[s] p`; a cosine on `s` alone → constant `−energy cos(omega[s] p + offset)`. The offset-charge `(s,s)` constant needs no code: dropping `E_C[s,·]` and `n_g[s]` removes it from `constant_term(fast)`, and the slow stage's own spec restores it (tested below). Errors (`ArgumentError`, naming entry and mode): `E_C[s,f]` not `_iszero`; a charge cosine with `kappa[s] ≠ 0`; `parameter` already in `spec.parameters`, equal to a remaining mode name, or rejected by `check_parameter_names`; a one-mode spec (nothing left to be fast).

- [ ] **Step 1: Write the failing tests:**

```julia
@testset "promote_mode" begin
    # one spec exercising every table row: modes s (e) and f (e) plus c (c), hand-written in the test
    p = promote_mode(table_spec(), :s; parameter=:x)
    @test spec_equal(p.fast, table_spec_expected_fast())   # hand-built expected fast spec, parameter :x added
    @test p.mode == KronFreeCircuits.ModeSpec(:s, :e) && p.parameter == :x
    # zero-pi: fast spec at θ = θ0 equals the full Hamiltonian with θ fixed, against a dense φ-only build
    zp = promote_mode(load_spec(fixture("zero_pi.json")), :theta)
    @test zp.fast.modes == [KronFreeCircuits.ModeSpec(:phi, :e)]
    # offset-charge constant: constant_term(full) == constant_term(fast) + 4 E_C[s,s] n_g[s]^2 at a test point
    # errors (Review Focus 1 and design §4.1)
    full = load_spec(fixture("full_zero_pi.json"))
    @test_throws ArgumentError promote_mode(full, :theta)                         # E_C[theta,phi], E_C[theta,zeta] ≠ 0 structurally
    @test promote_mode(substitute(full, (dC_J=0.0, dC=0.0)), :theta).fast.modes == [KronFreeCircuits.ModeSpec(:phi, :e), KronFreeCircuits.ModeSpec(:zeta, :h)]
    @test_throws ArgumentError promote_mode(load_spec(fixture("zero_pi.json")), :theta; parameter=:E_J)
    @test_throws ArgumentError promote_mode(load_spec(fixture("zero_pi.json")), :theta; parameter=:phi)
    @test_throws ArgumentError promote_mode(load_spec(fixture("zero_pi.json")), :theta; parameter=:pi)
    @test_throws ArgumentError promote_mode(load_spec(fixture("transmon.json")), :varphi_1)   # nothing left to be fast
    @test_throws ArgumentError promote_mode(charge_cosine_spec(), :s)
end
```

The zero-pi check: at `theta = 0.7` and the zero-pi test parameters, `energies(materialize(Node([:phi => PhaseGrid(-6π, 6π, 201)]; solver=Dense(), keep=lowest(4)), zp.fast, merge(θ, (theta=0.7,))))` equals the lowest four eigenvalues of a dense matrix built in the test from `4E_C[φφ]` times the grid's finite-difference `n²`, `E_L φ²`, and `−2E_J cos(0.7) cos(φ − varphi_ext/2)` on the same grid, plus `2E_J`, to 1e-10.

- [ ] **Step 2: Run the test file.** Expected: FAIL, `promote_mode` not defined.
- [ ] **Step 3: Implement `PromotedSpec` and `promote_mode`.** Build new coefficients with the internal `_make`; the parameter symbol itself is `_make(parameter, [parameter])`.
- [ ] **Step 4: Run the test file.** Expected: PASS.
- [ ] **Step 5: Commit** with message `promote_mode: turn a spec mode into a parameter (design §4.1)`.

### Task 4: Surfaces and the compact slow stage

**Files:**
- Create: `src/bo.jl`
- Modify: `src/KronFreeCircuits.jl` (include after `materialize.jl`; export `periodic_grid`, `BOSurface`, `bo_surface`, `fourier_coefficients`, `slow_spec`)
- Test: `test/test_bo.jl`

**Interfaces:**
- Consumes: `PromotedSpec`, `promote_mode`, `substitute`, `materialize`, `energies`.
- Produces:

```julia
periodic_grid(M::Integer; start::Real=0.0) -> Vector{Float64}     # start .+ 2π .* (0:M-1) ./ M

struct BOSurface
    promoted::PromotedSpec
    grid::Vector{Float64}
    energies::Matrix{Float64}    # bands × length(grid), absolute (root energies include the fast constant)
    θ::NamedTuple                # the parameters the surface was computed at, without the slow parameter
end
bo_surface(fast_root::Node, p::PromotedSpec, θ::NamedTuple, grid::AbstractVector{<:Real}; bands::Integer=2) -> BOSurface

fourier_coefficients(s::BOSurface; band::Integer=1) -> (a0::Float64, a::Vector{Float64}, b::Vector{Float64})
slow_spec(s::BOSurface; band::Integer=1, K::Union{Nothing,Integer}=nothing, tol::Union{Nothing,Real}=nothing)
    -> (spec::Spec, dropped::Float64)
```

`bo_surface`: one `materialize(fast_root, p.fast, merge(θ, (p.parameter => x,)))` per grid point, recording the lowest `bands` root energies (`real`); `bands` larger than the root's kept count is an `ArgumentError`. A band is the k-th lowest energy (envelope). `θ` must not contain `p.parameter`.

`fourier_coefficients` (algorithm fixed here): requires `s.promoted.mode.type === :c` and a uniform periodic grid — consecutive spacings equal to `2π/M` within 1e-12 and no point repeated modulo 2π (else `ArgumentError`, Review Focus 3). With `U = s.energies[band, :]`, `θ_j = s.grid`: `a0 = mean(U)`; for `1 ≤ k < M/2`, `a_k = (2/M) Σ U_j cos(kθ_j)`, `b_k = (2/M) Σ U_j sin(kθ_j)`; for even `M`, `a_{M/2} = (1/M) Σ U_j cos(Mθ_j/2)`, `b_{M/2} = 0`. Then `U(θ) = a0 + Σ a_k cos kθ + b_k sin kθ` exactly at the grid points.

`slow_spec`: a one-mode spec for `s.promoted.mode` with `E_C = [p.E_C]`, `n_g = [p.n_g]`, `E_L = [0]`, `flux_linear = [0]`, constant `a0`, and for each kept `k` the cosines `(energy = −a_k, omega = [k], offset = 0)` and `(energy = −b_k, omega = [k], offset = −π/2)` (zero-energy entries omitted). Keep `k ≤ K` when `K` is given; else the smallest `K` whose dropped amplitude `Σ_{k>K} hypot(a_k, b_k)` is `≤ tol`; else all harmonics. `dropped` is that sum. Parameters: substitute into `p.E_C` and `p.n_g` every parameter of `s.θ` that `p.fast` depends on (design §4.4); the rest stay free.

- [ ] **Step 1: Write the failing tests** in `test/test_bo.jl`:

```julia
@testset "BO surfaces" begin
    # L6 crossing: fast compact mode f, slow compact s, H = 4E_C (n_f - n_g)^2 - E_J cos(s) cos(φ_f)
    # (two cosines, omega [1,1] and [1,-1], energy E_J/2). At n_g = 0.5 the two lowest bands cross at s = π/2.
    p = promote_mode(crossing_spec(), :s)
    leaf = Node([:f => ChargeBasis(4)]; solver=Dense(), keep=lowest(4))
    g = periodic_grid(64)
    S = bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0), g; bands=2)
    @test S.energies ≈ dense_crossing_reference(g; E_C=1.0, E_J=0.2, n_g=0.5, ncut=4)[1:2, :] atol=1e-12
    @test all(S.energies[1, :] .<= S.energies[2, :])                          # envelope, not a followed branch
    @test S.energies[1, 17] ≈ S.energies[2, 17] atol=1e-12                     # g[17] == π/2: exact crossing
    Sa = bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.45, n_gs=0.0), g)        # avoided crossing
    @test minimum(Sa.energies[2, :] .- Sa.energies[1, :]) > 1e-3
    Sr = bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0), [0.3, 0.3])
    @test Sr.energies[:, 1] == Sr.energies[:, 2]                               # repeated point
    @test_throws ArgumentError bo_surface(leaf, p, (E_C=1.0, E_J=0.2, n_g=0.5, n_gs=0.0, s=0.1), g)
end

@testset "compact slow stage" begin
    # known harmonics: a surface built by hand round-trips through slow_spec to a dense periodic-grid reference
    S = synthetic_surface(periodic_grid(32), θ -> 0.3 - 0.7cos(2θ) + 0.05sin(θ) + 0.01cos(5θ))
    a0, a, b = fourier_coefficients(S)
    @test a0 ≈ 0.3 && a[2] ≈ -0.7 && b[1] ≈ 0.05 && a[5] ≈ 0.01 && maximum(abs, a[[1,3,4]]) < 1e-14
    spec, dropped = slow_spec(S; tol=1e-3)
    @test dropped < 1e-14
    spec2, dropped2 = slow_spec(S; K=2)
    @test dropped2 ≈ 0.01
    @test energies(materialize(Node([:s => ChargeBasis(20)]; solver=Dense(), keep=lowest(6)), spec, (E_Cs=0.05, n_gs=0.1))) ≈
          dense_periodic_reference(S, E_C=0.05, n_g=0.1)[1:6] rtol=1e-8
    @test_throws ArgumentError fourier_coefficients(synthetic_surface(range(0, 2π; length=33), cos))   # endpoint repeated
    @test_throws ArgumentError fourier_coefficients(synthetic_surface(sort(rand(MersenneTwister(1), 32)) .* 2π, cos))
end

@testset "zero-pi BO against an independent dense BO" begin
    zp = promote_mode(load_spec(fixture("zero_pi_bo.json")), :theta)   # created in this task, see Step 3
    θ = (E_Cphi=15.12, E_Ctheta=0.007, E_J=6.6, E_L=0.04, varphi_ext=0.5, n_g=0.0)
    leaf = Node([:phi => PhaseGrid(-8π, 8π, 241)]; solver=Dense(), keep=lowest(2))
    S = bo_surface(leaf, zp, Base.structdiff(θ, (n_g=0,)), periodic_grid(16))
    @test S.energies ≈ dense_zero_pi_bo_reference(θ, periodic_grid(16), PhaseGrid(-8π, 8π, 241)) rtol=1e-10
    spec, _ = slow_spec(S)
    @test Set(spec.parameters) == Set([:E_Ctheta, :n_g])               # fast-dependent parameters substituted
end
```

Helpers (in `test/helpers.jl`): `crossing_spec()` (hand-written JSON, modes `f` (c) and `s` (c), parameters `E_C, E_J, n_g, E_Cs, n_gs`, `E_C = [["E_C","0"],["0","E_Cs"]]`, `n_g = ["n_g","n_gs"]`); `dense_crossing_reference` (charge-basis matrix built directly, eigenvalues per grid point); `synthetic_surface(grid, U)` (a `BOSurface` around a one-compact-mode-plus-dummy promoted spec with `E_C[s,s] = E_Cs`, `n_g[s] = n_gs`, filled from `U.(grid)`); `dense_periodic_reference(S; E_C, n_g)` (charge-basis matrix `4E_C (n − n_g)²` plus the Fourier potential, built by hand); `dense_zero_pi_bo_reference(θ, grid, basis)` (Appendix B's `H_φ(θ)` from plain matrices on the same finite-difference grid as `basis`, lowest two eigenvalues per point; no spec, term or `materialize` code). Keep each default-suite testset under 2 s.

- [ ] **Step 2: Run the test file.** Expected: FAIL, `periodic_grid` not defined.
- [ ] **Step 3: Write `test/fixtures/zero_pi_bo.json`**: Appendix B in the spec's form, modes `theta` (c), `phi` (e); parameters `E_Cphi, E_Ctheta, E_J, E_L, varphi_ext, n_g`; `E_C = [["E_Ctheta","0"],["0","E_Cphi"]]` (4 n E_C n ⇒ entries equal the 4E_C coefficients); `E_L = [["0","0"],["0","2*E_L"]]`; `n_g = ["n_g","0"]`; the two cosines of `zero_pi.json`; constant `"0"` (no `2E_J`). A `_comment` cites arXiv:1811.07921v1 `eq:phi_Hamiltonian`, `eq:theta_Hamiltonian` and `docs/science/born-oppenheimer.md`.
- [ ] **Step 4: Implement `periodic_grid`, `BOSurface`, `bo_surface`, `fourier_coefficients`, `slow_spec`** in `src/bo.jl`.
- [ ] **Step 5: Run the test file.** Expected: PASS.
- [ ] **Step 6: Commit** with message `BO surfaces and the compact slow stage (design §4.2-4.3)`.

### Task 5: Fits and reduced specs

**Files:**
- Modify: `src/bo.jl`, `src/KronFreeCircuits.jl` (export `BOFit`, `fit_surfaces`)
- Test: `test/test_bo.jl`

**Interfaces:**
- Consumes: `BOSurface`, `PromotedSpec`, `substitute`, `spec_json` metadata conventions.
- Produces:

```julia
struct BOFit
    terms::Vector{Tuple{Any,String}}   # (slow basis key, parameter expression), in model order
    constants::Vector{Float64}
    stderr::Vector{Float64}
    rms_residual::Float64
    max_residual::Float64
    spec::Spec                         # the reduced one-mode spec
    metadata::Dict{String,Any}         # "_bo" provenance, ready for save_spec
end
fit_surfaces(surfaces::AbstractVector{BOSurface}, model::AbstractVector{<:Pair}; band::Integer=1) -> BOFit
```

A model is `[key => ["expr", ...], ...]`; each expression is parsed with `Coefficient(expr, params)` against the source spec's parameters. Slow basis keys and their meaning, chosen so each fitted constant multiplies a spec entry directly:

| key | basis value b(s) | reduced-spec entry gains `c·h(params)` in | allowed slow mode type |
| --- | --- | --- | --- |
| `:constant` | 1 | `constant` | any |
| `(:cos, k)` | −cos(k s) | cosine `energy`, `omega=[k]`, `offset=0` | any (`k` integer on `:c`) |
| `(:sin, k)` | −sin(k s) | cosine `energy`, `omega=[k]`, `offset=−π/2` | any (`k` integer on `:c`) |
| `:x` | s | `flux_linear` | `:e`, `:h` |
| `:x2` | s² | `E_L` (as `2·c·h`, since the form is `½ E_L s²`) | `:e`, `:h` |

The model is then `U(s; params) = Σ c_i h_i(params) b_i(s)`, fitted to `surface.energies[band, :]` over every (surface, grid point) by one least-squares solve (`A \ y`, QR); `stderr_i = sqrt(σ² (AᵀA)⁻¹_ii)` with `σ² = RSS/(n − p)`. Fitted-over parameters are the free symbols of the model's expressions. Every surface must share one `PromotedSpec` source and slow mode and must agree exactly on every other entry of `θ` (else `ArgumentError` naming the parameter, Review Focus 2). A key not allowed for the slow mode's type is an `ArgumentError` (Review Focus 5). Reduced spec: parameters = fitted-over parameters ∪ those of `p.E_C`, `p.n_g` that `p.fast` does not depend on; every other parameter of `p.E_C`, `p.n_g` substituted from the shared `θ`. Coefficient text is the model expression times the constant written as a number. `metadata["_bo"]` holds the slow mode, the fast spec's parameter values, each surface's grid size and fitted-over values, the model, constants, standard errors and residuals; `metadata["_generator"] = Dict("package" => "KronFreeCircuits", "version" => <package version>)`.

- [ ] **Step 1: Write the failing tests:**

```julia
@testset "fits" begin
    # synthetic recovery to round-off: Di Paolo's form on a compact slow mode
    mk(φ) = synthetic_surface(periodic_grid(32), θ -> 1.1 - (0.74 - 4e-4cos(φ))cos(2θ) - 0.0011cos(φ/2)cos(θ); θfix=(varphi_ext=φ,))
    surfs = [mk(φ) for φ in 2π .* (0:0.125:0.5)]
    model = [:constant => ["1"], (:cos, 2) => ["1", "cos(varphi_ext)"], (:cos, 1) => ["cos(varphi_ext/2)"]]
    F = fit_surfaces(surfs, model)
    @test F.constants ≈ [1.1, 0.74, -4e-4, 0.0011] atol=1e-13
    @test F.rms_residual < 1e-13
    @test :varphi_ext in F.spec.parameters
    @test spec_equal(parse_spec(spec_json(F.spec; metadata=F.metadata)), F.spec)
    # quadratic model in two parameters around a working point
    # (surfaces over a 3×3 grid of (varphi_ext, n_g1) with a hidden quadratic; recover all six constants)
    # L6 oscillators, extended slow mode through :x2: U(s) = ω/2 + ½(a − c²/b) s², ω = sqrt(8 E_Cf b)
    p = promote_mode(coupled_oscillators_spec(), :s)
    leaf = Node([:f => PhaseGrid(-12, 12, 301)]; solver=Dense(), keep=lowest(1))
    θ = (a=1.0, b=2.0, c=0.6, E_Cf=0.5, E_Cs=0.1)
    S = bo_surface(leaf, p, θ, range(-1, 1; length=21); bands=1)
    Fo = fit_surfaces([S], [:constant => ["1"], :x2 => ["1"]])
    @test Fo.constants ≈ [sqrt(8 * 0.5 * 2.0) / 2, (1.0 - 0.6^2 / 2.0) / 2] rtol=1e-7
    @test Fo.spec.parameters == [:E_Cs]                                  # a, b, c, E_Cf substituted
    # errors
    @test_throws ArgumentError fit_surfaces([surfs[1], mk_other_fixed_param()], model)   # Review Focus 2
    @test_throws ArgumentError fit_surfaces(surfs, [:x2 => ["1"]])                         # Review Focus 5
    @test_throws ArgumentError fit_surfaces(surfs, [(:cos, 1.5) => ["1"]])
end
```

`synthetic_surface` gains a `θfix` keyword (the surface's `θ`, plus a fixed dummy fast parameter so the substitution rule is exercised); `coupled_oscillators_spec()` is hand-written JSON with modes `s` (e), `f` (e), `E_C = [["E_Cs","0"],["0","E_Cf"]]`, `E_L = [["a","c"],["c","b"]]`; `mk_other_fixed_param()` is a surface like `surfs[1]` with a different value of the dummy fixed parameter. Fill in the two-parameter quadratic case concretely (hidden constants of your choice, recovered to 1e-12).

- [ ] **Step 2: Run the test file.** Expected: FAIL, `fit_surfaces` not defined.
- [ ] **Step 3: Implement `BOFit` and `fit_surfaces`.**
- [ ] **Step 4: Run the test file.** Expected: PASS.
- [ ] **Step 5: Commit** with message `fit_surfaces: linear least-squares fits to a reduced spec (design §4.4)`.

### Task 6: Interchange with circuit-quantization

**Files:**
- Create: `test/fixtures/bo/oscillators_reduced.json`, `test/fixtures/bo/compact_reduced.json`, `test/fixtures/bo/write_reduced.jl`, `test/fixtures/bo/validate_reduced.py`
- Test: `test/test_bo.jl`

**Interfaces:**
- Consumes: `fit_surfaces`, `save_spec`, the Task 5 test constructions.
- Produces: two committed reduced specs (extended slow mode via `:x2`; compact slow mode with `(:cos, k)`, `(:sin, k)` terms and a free `varphi_ext`), validated once by Python.

- [ ] **Step 1: Write the failing test:** for each committed file, `load_spec` succeeds and `spec_equal` holds against the spec `write_reduced.jl` would produce now (rebuilt in the test from the same construction).
- [ ] **Step 2: Run.** Expected: FAIL, files missing.
- [ ] **Step 3: Write `write_reduced.jl`** (`julia --project=. test/fixtures/bo/write_reduced.jl` regenerates both files) **and `validate_reduced.py`**, which loads each file with `circuit_quantization.load_spec`, builds its Hamiltonian, and exits nonzero on any failure; its docstring follows `test/fixtures/generate_specs.py` (run command, never a test-time dependency).
- [ ] **Step 4: Run** the Julia writer, then `/opt/anaconda3/envs/circuit_quantization/bin/python test/fixtures/bo/validate_reduced.py`. Expected: both files written; the script prints one OK line per file and exits 0. Then run the test file. Expected: PASS.
- [ ] **Step 5: Commit** with message `BO reduced specs validated against circuit-quantization`.

### Task 7: Zero-pi reproduction, acceptance and documentation

**Files:**
- Create: `bench/born_oppenheimer/zero_pi.jl`, `bench/born_oppenheimer/results.md`, `test/science/fixtures/zero_pi_bo_reference.json`, `test/fixtures/bo/zero_pi_reduced.json`, `test/test_born_oppenheimer_reference.jl`
- Modify: `test/runtests.jl` (exclude `test_born_oppenheimer_reference.jl` from the default loop and include it under `KRONFREE_SLOW_TESTS=1`, beside `test_zero_pi_reference.jl`), `test/fixtures/bo/validate_reduced.py` (add the zero-pi file), `README.md`, `docs/science/born-oppenheimer.md`, `docs/science/README.md`, `docs/spikes/2026-10-16-pathological-test-catalogue/outcomes.md` (L6), `docs/superpowers/STATUS.md`, `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` (a short §20h pointing to the BO design and recording that promotion is spec-level)

**Interfaces:**
- Consumes: everything above; `test/fixtures/zero_pi_bo.json` (Task 4).
- Produces: the gated acceptance test and a reference fixture whose producer is `independent BO numerics`.

Published values (ratios to ħω_p, ω_p/2π = 40 GHz, inputs exact): E_Cφ = 0.378, E_Cθ = 0.000175, E_J = 0.165, E_L = 0.001, so in GHz `E_Cphi=15.12, E_Ctheta=0.007, E_J=6.6, E_L=0.04`; `n_g = 0`; flux samples `0, 0.125, 0.25, 0.375, 0.5` with `varphi_ext = 2π·flux`. Targets: `E_α = 1.8608e-2`, `E_γ = 2.6625e-5` (ratios to ħω_p); `E_β = 1.0073e-8` reported only. Model: `(:cos, 2) => ["1", "cos(varphi_ext)"]`, `(:cos, 1) => ["cos(varphi_ext/2)"]`, plus a `:constant` term whose parameter basis is the builder's choice (on a uniform θ grid the constant is orthogonal to the cosine terms, so it does not bias E_α, E_β, E_γ); then `E_α = c[cos2,"1"]`, `E_β = −c[cos2,"cos(varphi_ext)"]`, `E_γ = c[cos1,…]`, each divided by 40 GHz.

- [ ] **Step 1: Refinement study.** `zero_pi.jl` computes surfaces at the five flux values and refines, independently, the φ range, the φ grid size and order (or a `FockBasis` cutoff), and the θ sample count `M`, until E_α, E_β, E_γ change by less than 1e-5 relative between refinements (100× below the 0.1% tolerance). It writes `results.md` (the refinement table, chosen settings, fitted constants with standard errors, residuals, the fast gap `min(band 2 − band 1)` over θ and flux, π-periodicity at half flux as `max |U(θ+π) − U(θ)|`, and run time) and `zero_pi_bo_reference.json` (producer `independent BO numerics`; units, sign conventions, constants, grids, cutoffs, Julia and package versions, the source `1811.07921v1`, error estimates). Run: `julia --project=. bench/born_oppenheimer/zero_pi.jl`.
- [ ] **Step 2: Full-2D comparison.** The same script quantizes the slow mode from the reduced spec at each flux (charge cutoff refined like the rest) and compares its lowest six levels with the full 2D `zero_pi_bo.json` at matched settings, recording per-level BO error in `results.md` and the fixture. No bound is asserted.
- [ ] **Step 3: Check the acceptance criterion.** If E_α or E_γ misses 0.1%: do not change the tolerance or the inputs; check the parameter mapping (factor of 2 in charging energies, the absent `2E_J`), the sign of `E_β`, convergence, and the fit; if still unexplained, stop and report to Eli with `results.md`.
- [ ] **Step 4: Write the gated test** `test/test_born_oppenheimer_reference.jl`: at the chosen converged settings, recompute the five surfaces and the fit, and assert `abs(E_α/1.8608e-2 − 1) < 1e-3`, `abs(E_γ/2.6625e-5 − 1) < 1e-3`, half-flux `max |U(θ+π) − U(θ)|` below the convergence level from Step 1, and agreement with `zero_pi_bo_reference.json` to its recorded error estimate. Save the fit's reduced spec as `test/fixtures/bo/zero_pi_reduced.json` with its metadata; add it to `validate_reduced.py` and run that script (Expected: OK lines, exit 0).
- [ ] **Step 5: Run** `KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'`. Expected: PASS, including the new gated file. Run the default suite as well and confirm the wall time is within the Task 1 budget.
- [ ] **Step 6: Documentation.** README: a short "Born–Oppenheimer reduction" section with the zero-pi example (promote, surface, fit, save). `docs/science/born-oppenheimer.md` and `docs/science/README.md`: status changed from deferred to implemented, pointing to `results.md` and stating that the published coefficients are compared, not reproduced as fixture values. Catalogue `outcomes.md`: L6 implemented/passing with the test names. STATUS.md: BO done, KITE and the data-valued `DiagonalOp` entry listed as next with their design references. Architecture §20h as listed under Files.
- [ ] **Step 7: Commit** with message `Zero-pi BO reproduction: Di Paolo E_alpha and E_gamma to 0.1% (gated)`.
