# Periodic Gauge for `PeriodicGrid` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the `PeriodicGrid` basis from the twisted gauge (offset charge in the boundary condition) to the periodic gauge (fixed cyclic `n̂`, `n̂²`; offset charge only in the canonical quadratic form), and remove the twisted-gauge machinery.

**Architecture:** `PeriodicGrid` becomes an ordinary basis that serves `nport()`, `n2port()` and integer `expiphi(k)` as parameter-free cyclic `BandedOp`s with `PhaseGrid`'s characters. Term construction then has nothing basis-specific left: the one-argument `circuit_terms(spec)` and `constant_term(spec)` are the only expansion, and the twisted branches in `terms.jl`, `materialize.jl`, `tree.jl`, the `twist_factor` export and the derivative-piece port kinds are deleted. Tests are rewritten around the periodic-gauge discretization formula, including a pinned sweet-spot splitting.

**Tech Stack:** Julia, the package's own `BandedOp`/`DiagonalOp` local operators, `Test`, `Pkg.test(; test_args=[...])` for single files.

**Spec:** `docs/superpowers/specs/2026-10-30-periodic-gauge-design.md` (amends `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md` §20d and §20f).

## Global Constraints

- Operator data stays parameter-free; `n_g` lives only in term coefficients (architecture §11.1).
- Spec expressions stay real-valued; the only complex structure from this basis is the `-i` in the `:n` stencil and the `expiphi` diagonals.
- Realness is decided by structure from the term list (§20f); no new special cases. A literal `n_g = "0"` is real, `"0.5"` is complex, a parameter `n_g` is complex.
- `check_sectors`, `sectors`, the integer-harmonic check and the mode-type check are unchanged.
- Test tiers are unchanged: `test_periodic.jl`, `test_bases.jl`, `test_terms.jl` fast; `test_character.jl`, `test_composite_slow.jl`, `test_zero_pi_reference.jl` slow or reference.
- Work on a branch `periodic-gauge` off `main`; merging is Eli's. One commit per task, each ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run a single test file with `julia --project -e 'using Pkg; Pkg.test(; test_args=["<file>"])'` from the repository root; the fast tier with `julia --project -e 'using Pkg; Pkg.test()'`.

## Review Focus

1. A term list built by hand with `n2port()` on a `PeriodicGrid` mode (previously refused) must now materialize and agree with the charge basis — Task 2 (term-list form test).
2. `nport()` requested at an ancestor of a `PeriodicGrid` leaf must be the plain routed port, with no spec needed, in both `materialize` forms — Task 2 (ancestor test).
3. A spec with the literal `n_g = "0.5"` on a `PeriodicGrid` must classify complex and still agree with the charge basis — Task 2 (literal test).
4. A real node (`n_g = "0"`) with a `PeriodicGrid` leaf must return `Matrix{ComplexF64}` from `port(sys, mode, nport())` and `charge_port` (imaginary port of a real node) and a real `Matrix{Float64}` operator — Task 2 (literal test).
5. The error for `arithmetic=:real` on a `PeriodicGrid` transmon with parameter `n_g` must name the mode and say the summed coefficient of `n` must be purely imaginary — Task 3 (character test).

---

### Task 1: `PeriodicGrid` serves `n`, `n2` and `expiphi` as cyclic operators

**Files:**
- Modify: `src/bases.jl:39-95` (docstring, `_is_twisted` line), `src/bases.jl:213-262` (`_PERIODIC_*` constants, `_periodic_stencil`, `localop(::PeriodicGrid, ::Port)`)
- Modify: `src/character.jl:46-51` (`character(::PeriodicGrid, ::Port)`)
- Modify: `src/materialize.jl:835-860` (`charge_port`: the `PeriodicGrid` path would otherwise request the removed `:d1` ports)
- Test: `test/test_bases.jl:78-115`, `test/test_character.jl:1-24`

**Interfaces:**
- Produces: `localop(b::PeriodicGrid, nport()) :: BandedOp{ComplexF64}` (cyclic `-i ∂_φ`), `localop(b::PeriodicGrid, n2port()) :: BandedOp{Float64}` (cyclic `-∂_φ²`), `localop(b::PeriodicGrid, expiphi(k))` unchanged; `character(b::PeriodicGrid, nport()).kind === :imaginary`, `character(b::PeriodicGrid, n2port()).kind === :real`, `expiphi(k)` paired. Ports of kind `:d1, :d2, :d1wrap, :d2wrap`, `:phi, :phi2, :exp2pin` throw `ArgumentError`. `_is_twisted(::PeriodicGrid)` no longer exists (so `_twisted` in `terms.jl` is all-false until Task 3 deletes it). `charge_port(sys, mode)` is `_port_matrix(V, dims, a, localop(b, nport()), character(b, nport()))` for every basis and needs no spec.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b periodic-gauge main
```

- [ ] **Step 2: Replace the twisted-stencil testset in `test/test_bases.jl`**

Replace the testset `"PeriodicGrid: twisted stencil equals the thesis construction; sectors"` (lines 78–115) with:

```julia
@testset "PeriodicGrid: cyclic stencils of -i∂ and -∂²; plane-wave symbols; sectors" begin
    using KronFreeCircuits: check_sectors
    for order in (2, 4, 6, 8), N in (16, 24)
        b = PeriodicGrid(N; order=order)
        Δ = 2π / N; h = order ÷ 2
        w1 = KronFreeCircuits.fd_weights(order, 1); w2 = KronFreeCircuits.fd_weights(order, 2)
        # the cyclic matrices built directly with mod1
        D1 = zeros(ComplexF64, N, N); D2 = zeros(N, N)
        for i in 1:N, j in -h:h
            D1[i, mod1(i + j, N)] += -im * w1[j+h+1] / Δ
            D2[i, mod1(i + j, N)] += -w2[j+h+1] / Δ^2
        end
        n = dense(localop(b, nport())); n2 = dense(localop(b, n2port()))
        @test n ≈ D1 && n2 ≈ D2
        @test n ≈ n' && maximum(abs, real(n)) == 0          # Hermitian, purely imaginary
        @test n2 ≈ n2' && eltype(n2) <: Real                 # real symmetric
        # plane waves diagonalize both, with the symbols σ₁(k) = Σ_j w¹_j sin(jkΔ)/Δ and σ₂(k) = -Σ_j w²_j cos(jkΔ)/Δ²
        φ = coordinates(b)
        for k in (0, 1, -2, 3)
            ψ = cis.(k .* φ)
            σ1 = sum(w1[j+h+1] * sin(j * k * Δ) for j in -h:h) / Δ
            σ2 = -sum(w2[j+h+1] * cos(j * k * Δ) for j in -h:h) / Δ^2
            @test n * ψ ≈ σ1 .* ψ atol=1e-12
            @test n2 * ψ ≈ σ2 .* ψ atol=1e-12
        end
        @test dense(localop(b, expiphi(2))) ≈ Diagonal(cis.(2 .* coordinates(b)))
        @test sparse(localop(b, nport())) ≈ n && sparse(localop(b, n2port())) ≈ n2
        for p in (Port(:d1, 0), Port(:d2, 0), Port(:d1wrap, 1), Port(:d2wrap, -1), Port(:n, 1), Port(:n2, 1), Port(:exp2pin, 1))
            @test_throws ArgumentError localop(b, p)
        end
    end
    @test KronFreeCircuits._check_bases(Node([:phi => PeriodicGrid(16)]), Dict(:phi => :c)) === nothing
    @test_throws ArgumentError KronFreeCircuits._check_bases(Node([:phi => PeriodicGrid(16)]), Dict(:phi => :e))
    @test_throws ArgumentError localop(PeriodicGrid(16), phiport())
    @test_throws ArgumentError localop(PeriodicGrid(16), phi2port())
    @test_throws ArgumentError localop(PeriodicGrid(16), expiphi(0.5))
    @test_throws ArgumentError check_sectors(PeriodicGrid(15), [2], :phi)
    @test check_sectors(PeriodicGrid(16; sectors=4), [2, 4], :phi) === nothing
    @test_throws ArgumentError PeriodicGrid(15; sectors=2)
    @test KronFreeCircuits.modetype(PeriodicGrid(16)) === :c && KronFreeCircuits.spacing(PeriodicGrid(16)) ≈ 2π / 16
end
```

- [ ] **Step 3: Update the character sweep in `test/test_character.jl`**

In the testset `"port characters match the operator data exactly"`: drop the six `Port(:d…)` entries from `ports`; change `@test nchecked == 30` to `@test nchecked == 26` with the comment `# served pairs: ChargeBasis 6, PhaseGrid 8 (2π/Δ = 10, so exp2pin(±1) is served), PeriodicGrid 4 (n, n2, expiphi(±)), FockBasis 8`; replace `@test_throws ArgumentError character(PeriodicGrid(12; order=4), nport())` with

```julia
    @test character(PeriodicGrid(12; order=4), nport()).kind === :imaginary
    @test character(PeriodicGrid(12; order=4), n2port()).kind === :real
    @test_throws ArgumentError character(PeriodicGrid(12; order=4), Port(:d2, 0))
```

- [ ] **Step 4: Run both test files to verify they fail**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["test_bases.jl"])'`
Expected: FAIL in the new testset (`localop(b, nport())` throws "not available").

- [ ] **Step 5: Implement the cyclic operators in `src/bases.jl`**

- Delete the line `_is_twisted(::PeriodicGrid) = true`.
- Replace `_PERIODIC_PIECES`, `_PERIODIC_CHARGE_ROUTES` and `_PERIODIC_UNAVAILABLE` with a single `_PERIODIC_UNAVAILABLE` holding only `:phi`, `:phi2` (existing messages) and `:exp2pin => "exp2pin on a periodic grid is not implemented yet"`.
- Replace `_periodic_stencil(N, h, c, corner)` by `_cyclic_stencil(N::Int, h::Int, c::Vector{T}) where {T} -> BandedOp{T}`: the cyclic matrix `Y[i] = Σ_{j=-h..h} c[j+h+1] X[mod1(i + j, N)]` as one `BandedOp` whose offsets are every `j` with nonzero `c` plus `j + N` for `j < 0` and `j - N` for `j > 0`, each weight vector zero on the rows where `1 ≤ i + offset ≤ N` fails (the present per-corner construction, concatenated).
- `localop(b::PeriodicGrid, p)`: `:n` with `k == 0` → `_cyclic_stencil(N, h, ComplexF64.(-im .* fd_weights(order, 1) ./ Δ))`; `:n2` with `k == 0` → `_cyclic_stencil(N, h, -fd_weights(order, 2) ./ Δ^2)`; nonzero `k` on either → `ArgumentError("PeriodicGrid: port $(p.kind) takes k = 0, got k = $(p.k)")`; `:expiphi` unchanged; a key of `_PERIODIC_UNAVAILABLE` → its message; anything else → `_unavailable(b, p)`.
- Rewrite the `PeriodicGrid` docstring: periodic boundary condition, parameter-free operators, the ports table (`nport()`: cyclic `-i ∂_φ`; `n2port()`: cyclic `-∂_φ²`; `expiphi(k)` integer `k`), the offset charge "enters only through the coefficients of the canonical quadratic form `4 (n - n_g)ᵀ E_C (n - n_g)` (see `circuit_terms`)", `:phi`/`:phi2` multivalued, `:exp2pin` later; drop every twisted sentence.

- [ ] **Step 6: Implement the characters in `src/character.jl`**

`character(b::PeriodicGrid, p::Port)`: `localop(b, p)` to validate; `p.kind === :n2 && return _real_char(p)`; `p.kind === :n && return _imag_char(p)`; `return _paired_char(p)`.

- [ ] **Step 7: Simplify `charge_port` in `src/materialize.jl`**

Make the `ChargeBasis` line unconditional for every basis and delete the `PeriodicGrid` branch (the `sys.spec === nothing` check, the `findfirst` and the `_assembled_charge` call). Docstring: "`V′ n V` in the leaf's kept eigenbasis, in the port's natural type (real for a real-character `n` on a real node, complex for a grid's imaginary `n`)"; drop the `PeriodicGrid` paragraph.

- [ ] **Step 8: Run the two test files and verify they pass**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["test_bases.jl", "test_character.jl"])'`
Expected: `test_bases.jl` PASS. In `test_character.jl` the sweep testset passes; the testsets that assert `"d2wrap"` in an error message and that the literal `"0.5"` classifies real now FAIL (they are fixed in Task 3). Note the failing testset names in the commit message.

- [ ] **Step 9: Commit**

```bash
git add src/bases.jl src/character.jl src/materialize.jl test/test_bases.jl test/test_character.jl
git commit -m "PeriodicGrid: periodic gauge; serve n, n2 and expiphi as cyclic operators

The offset charge now enters only through the canonical quadratic form; the
twisted-gauge derivative pieces are no longer served. Term construction and
the old periodic tests still reference the twisted path (next tasks).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Rewrite `test/test_periodic.jl` for the periodic gauge

**Files:**
- Rewrite: `test/test_periodic.jl`

**Interfaces:**
- Consumes: Task 1's ports, characters and `charge_port`. With `_is_twisted(::PeriodicGrid)` gone, `circuit_terms(spec)` already expands every mode in the charge-basis form and `materialize` no longer expands or refuses `nport()` on `PeriodicGrid` modes (`_twisted` is all-false); the tests here must pass both before and after Task 3, so they use only public behaviour and the one-argument `circuit_terms`/`constant_term`.
- Produces: the periodic-gauge reference tests; the helper `periodic_symbols(order) -> (σ1, σ2)` defined at the top of the file, `σ1(k, Δ)`, `σ2(k, Δ)`.

- [ ] **Step 1: Write the new file**

Replace the whole file with the following testsets (fixtures and helpers as in the present file: `fixture`, `two_transmons_params`, `load_spec`, `parse_spec`).

Header helper:

```julia
# Plane-wave symbols of the order-`order` cyclic stencils (design §4.5): n e^{ikφ} = σ₁(k) e^{ikφ}, n² e^{ikφ} = σ₂(k) e^{ikφ}
function periodic_symbols(order)
    w1 = KronFreeCircuits.fd_weights(order, 1); w2 = KronFreeCircuits.fd_weights(order, 2); h = order ÷ 2
    σ1(k, Δ) = sum(w1[j+h+1] * sin(j * k * Δ) for j in -h:h) / Δ
    σ2(k, Δ) = -sum(w2[j+h+1] * cos(j * k * Δ) for j in -h:h) / Δ^2
    return σ1, σ2
end
```

Testset 1, `"free rotor on the periodic grid: continuum spectrum, discrete spectrum, imaginary part"`: as the present first testset, with `σ1, σ2 = periodic_symbols(8)` before the loop, `disc = sort([4 * (σ2(n, Δ) - 2ng * σ1(n, Δ) + ng^2) for n in -8:8])[1:6]` and `@test energies(sys) ≈ disc atol=1e-10`; the imaginary-part assertion becomes `ng == 0.0 ? (@test maximum(abs, imag(A)) == 0) : (@test maximum(abs, imag(A)) > 0)` with the comment `# the linear term -8 E_C n_g n is imaginary on the grid`.

Testset 2, `"n_g = ½: the sweet-spot pairs split by the discretization, at order 8 (design §4.5)"`:

```julia
    spec = load_spec(fixture("transmon.json")); σ1, σ2 = periodic_symbols(8)
    split = Float64[]
    for N in (32, 64, 128)
        E = energies(materialize(Node([:varphi_1 => PeriodicGrid(N; order=8)]; solver=Dense(), keep=lowest(4)), spec, (E_J=0.0, E_C=1.0, n_g=0.5)))
        Δ = 2π / N
        pred(k) = 4 * (σ2(k, Δ) - σ2(1 - k, Δ) - σ1(k, Δ) + σ1(1 - k, Δ))     # E(k) - E(1-k) of the pair (k, 1-k)
        @test abs(E[2] - E[1]) ≈ abs(pred(0)) atol=1e-12                     # pair (0, 1)
        @test abs(E[4] - E[3]) ≈ abs(pred(-1)) atol=1e-12                    # pair (-1, 2)
        push!(split, abs(E[2] - E[1]))
    end
    @test split[1] ≈ 1.1e-8 rtol=0.1                 # builder check, 2026-10-30
    @test split[1] / split[2] > 100 && split[2] / split[3] > 50      # about 2^8 per doubling, the last near roundoff
    @test split[3] < 1e-12
```

Testset 3, `"transmon: periodic grid vs charge basis; charge port; literals 0 and 0.5"`: keep the present second testset's first seven assertions (energies to 1e-7, `abs.(charge_port)` and its real diagonal to 1e-6, charge-basis `charge_port` equals the port, sparse equals dense, `charge_port(sp, :nope)` throws). Replace the imaginary-part loop with:

```julia
    # structure decides: a parameter n_g is complex; the literal "0" is real; the literal "0.5" is complex (design §4.4)
    leaf(b; kw...) = Node([:varphi_1 => b]; solver=Dense(), keep=lowest(6), kw...)
    lit(v) = parse_spec(replace(read(fixture("transmon.json"), String), "\"n_g\": [\"n_g\"]" => "\"n_g\": [\"$v\"]"))
    s0 = materialize(leaf(PeriodicGrid(64); ports=[(:varphi_1, nport())]), lit("0"), (E_J=20.0, E_C=1.0))
    @test arithmetic(s0) === :real && s0.root isa KronFreeCircuits.MaterializedNode{Float64}
    @test eltype(KronFreeCircuits.operator(s0)) === Float64
    @test port(s0, :varphi_1, nport()) isa Matrix{ComplexF64}                 # an imaginary port of a real node
    @test KronFreeCircuits.charge_port(s0, :varphi_1) ≈ port(s0, :varphi_1, nport()) atol=1e-12
    @test energies(s0) ≈ energies(materialize(leaf(ChargeBasis(20)), lit("0"), (E_J=20.0, E_C=1.0))) atol=1e-7
    sh = materialize(leaf(PeriodicGrid(64); ports=[(:varphi_1, nport())]), lit("0.5"), (E_J=20.0, E_C=1.0))
    @test arithmetic(sh) === :complex
    @test maximum(abs, imag(Matrix(KronFreeCircuits.operator(sh)))) > 0
    shc = materialize(leaf(ChargeBasis(20); ports=[(:varphi_1, nport())]), lit("0.5"), (E_J=20.0, E_C=1.0))
    @test energies(sh) ≈ energies(shc) atol=1e-7
    @test abs.(port(sh, :varphi_1, nport())) ≈ abs.(port(shc, :varphi_1, nport())) atol=1e-6
    @test arithmetic(materialize(leaf(PeriodicGrid(64)), spec, θ)) === :complex
```

Testset 4, `"cos 2φ: both parity sectors, exact half-period translation, sweet-spot degeneracy, divisibility"`: as the present third testset with `T = circshift(Matrix{Float64}(I, N, N), (N ÷ 2, 0))`, `@test T^2 == I`, `parity = [real(dot(V[:, j], T * V[:, j])) for j in 1:8]`, the same parity and divisibility assertions and the weak-cos φ mixing assertion; add

```julia
    # at n_g = ½ the two sectors are degenerate (k ↦ 1 - k); on the grid only to O(Δ⁸) (design §1, "What it costs")
    E½ = energies(materialize(Node([:phi => b]; solver=Dense(), keep=lowest(8)), spec, (E_C=1.0, n_g=0.5, E_1=0.0, E_2=15.0)))
    @test abs(E½[2] - E½[1]) < 1e-10 && abs(E½[4] - E½[3]) < 1e-10
```

and record the measured splittings in a comment next to it.

Testset 5, `"every expiphi harmonic on a PeriodicGrid is checked: extra_terms and the term-list form"`: as the present one, with `terms = vcat(circuit_terms(spec), cos2)`.

Testset 6, `"the term-list form: n2port() and nport() on a PeriodicGrid mode need no spec"`:

```julia
    spec = load_spec(fixture("transmon.json")); θ = (E_J=20.0, E_C=1.0, n_g=0.3)
    leaf(b; kw...) = Node([:varphi_1 => b]; solver=Dense(), keep=lowest(6), kw...)
    terms = circuit_terms(spec)
    @test any(t -> any(f -> f.port == n2port(), t.factors), terms) && any(t -> any(f -> f.port == nport(), t.factors), terms)
    @test all(f.port.kind in (:n, :n2, :expiphi) for t in terms for f in t.factors)
    sp = materialize(leaf(PeriodicGrid(128); ports=[(:varphi_1, nport()), (:varphi_1, n2port())]), terms, θ; constant=constant_term(spec), modetypes=Dict(:varphi_1 => :c))
    sc = materialize(leaf(ChargeBasis(20); ports=[(:varphi_1, nport()), (:varphi_1, n2port())]), spec, θ)
    @test energies(sp) ≈ energies(sc) atol=1e-7
    @test abs.(port(sp, :varphi_1, nport())) ≈ abs.(port(sc, :varphi_1, nport())) atol=1e-6
    @test port(sp, :varphi_1, n2port()) isa Matrix{ComplexF64}      # a real port of a complex node
    @test abs.(port(sp, :varphi_1, n2port())) ≈ abs.(port(sc, :varphi_1, n2port())) atol=1e-6
    # a selection rule on the charge selects the charge basis's states (closure from the ground state at 0.5: states 1-6)
    rule = connected([1], (:varphi_1, nport()), 0.5)
    skp = materialize(Node([:varphi_1 => PeriodicGrid(128)]; solver=Dense(), compute=10, keep=rule), spec, θ)
    skc = materialize(Node([:varphi_1 => ChargeBasis(20)]; solver=Dense(), compute=10, keep=rule), spec, θ)
    @test skp.root.kept == skc.root.kept == 1:6
    @test energies(skp) ≈ energies(skc) atol=1e-7
    # a linear charge term c n (a static gate drive)
    lin = [Term(Coefficient(0.7), [Factor(:varphi_1, nport())])]
    @test energies(materialize(leaf(PeriodicGrid(128)), spec, θ; extra_terms=lin)) ≈ energies(materialize(leaf(ChargeBasis(20)), spec, θ; extra_terms=lin)) atol=1e-7
```

Testset 7, `"two transmons: a PeriodicGrid mode coupled through its charge; n_g only in coefficients"`: the present `"a twisted mode coupled through its charge"` energies assertion, then

```julia
    terms = circuit_terms(spec)
    with_ng1 = filter(t -> :n_g1 in KronFreeCircuits.free_symbols(t.coeff), terms)
    @test !isempty(with_ng1) && all(t -> length(t.factors) == 1 && t.factors[1].port == nport(), with_ng1)   # only the linear charge terms
    @test :n_g1 in KronFreeCircuits.free_symbols(constant_term(spec))
```

Testset 8, `"nport() on a PeriodicGrid mode above its leaf: ancestor ports, selection rules, coupling terms"`: the present testset without the pieces assembly (`sp = materialize(top(PeriodicGrid(96); ports=[(:varphi_1, nport())]), spec, θ)` and the `abs.` comparison to the charge basis to 1e-5), the selection-rule block unchanged, the `plain` coupling against the charge basis to 1e-5 and with both modes on `PeriodicGrid(48)` at `θ2`, and the term-list form at the root: `materialize(top(PeriodicGrid(96); ports=[(:varphi_1, nport())]), vcat(circuit_terms(spec), plain), θ; constant=constant_term(spec))` has energies equal to the spec form's to 1e-12. Drop the `_expand_twisted_charge` and `"0.5"`-pair blocks.

Testset 9, `"two transmons with both offset charges nonzero: the sign of the offset, one or both modes on the grid"`: the present testset through the one-leaf comparison; replace the bookkeeping block with `@test constant_term(spec)(θ) ≈ 4 * (θ.E_C1 * θ.n_g1^2 + θ.E_C2 * θ.n_g2^2 + 2 * θ.g / 8 * θ.n_g1 * θ.n_g2)`.

Drop the testset `"circuit_terms(spec) is the charge-basis expansion: identical to an empty basis dictionary"`. Keep `"non-integer harmonic on a compact mode is rejected at load"`.

- [ ] **Step 2: Run the file and verify it passes**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["test_periodic.jl"])'`
Expected: PASS. If the sweet-spot `rtol=0.1` or the cos 2φ `1e-10` assertion fails, record the measured value in the test's comment and set the tolerance to twice the measurement; do not loosen the `1e-12` agreement with the §4.5 formula.

- [ ] **Step 3: Run the fast tier**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: every fast file passes except `test_terms.jl`'s `port_of_kind(:d2)` assertions (fixed in Task 3).

- [ ] **Step 4: Commit**

```bash
git add test/test_periodic.jl
git commit -m "test_periodic: periodic-gauge references, sweet-spot splitting pinned

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Remove the twisted-gauge machinery

**Files:**
- Modify: `src/terms.jl:1-55` (Port docstring, `port_of_kind`), `src/terms.jl:107-150` (delete `_is_twisted`, `_twisted`, `_twisted_pieces`, `_times`, `_expand_twisted_charge`), `src/terms.jl:155-245` (`circuit_terms`, `constant_term`)
- Modify: `src/materialize.jl:76-200` (docstring, `materialize` spec form, `_materialize_terms`, delete `_check_offset_charges`), `src/materialize.jl:398-407` (`_node_port`), `src/materialize.jl:453-460` (delete `_assembled_charge`), `src/materialize.jl:608-715` (`_materialize`: drop `charges`)
- Modify: `src/tree.jl:525-560` (`required_ports`)
- Modify: `src/parameters.jl:173-180` (delete `twist_factor`), `src/KronFreeCircuits.jl:24` (drop the export)
- Test: `test/test_terms.jl:27-29`, `test/test_character.jl:48-66`

**Interfaces:**
- Produces: `circuit_terms(spec::Spec) -> Vector{Term}` and `constant_term(spec::Spec) -> Coefficient` as the only methods; `_materialize_terms(root, terms, θ, constant, previous, modetypes)`; `_node_port(node, leaves, children, dims, V, mode, p)`; `_materialize(node, θ, routed, needed, needed_joint, constant, prev, realnodes)`.

- [ ] **Step 1: Update the tests that pin the removed surface**

`test/test_terms.jl`: replace lines 27–29 with

```julia
    for (kind, k) in ((:d2, 0), (:d1wrap, -1), (:d1, 0), (:d2wrap, 1))
        @test_throws ArgumentError port_of_kind(kind, k)
    end
```

`test/test_character.jl`, `classified` helper: `classified(root, spec, θ) = (r = KronFreeCircuits.route(root, circuit_terms(spec)); KronFreeCircuits.classify(root, r, constant_term(spec), θ))`.
In `"classification …"` (lines 50–52): keep `@test !classified(pg, tr, θtr)[pg]` and the `n_g=0.5` line; change the `lit` assertion to `@test !classified(pg, lit, (E_J=20.0, E_C=1.0))[pg]   # the literal ½ is complex in the periodic gauge (its linear n term is imaginary)` and add

```julia
    lit0 = parse_spec(replace(read(fixture("transmon.json"), String), "\"n_g\": [\"n_g\"]" => "\"n_g\": [\"0\"]"))
    @test classified(pg, lit0, (E_J=20.0, E_C=1.0))[pg]                                      # a spec literal is structure: the linear term is dropped
```

Line 66: `@test err isa ArgumentError && occursin(":varphi_1", err.msg) && occursin("purely imaginary", err.msg)`.

- [ ] **Step 2: Run `test_terms.jl` to verify it fails**

Run: `julia --project -e 'using Pkg; Pkg.test(; test_args=["test_terms.jl"])'`
Expected: FAIL (`port_of_kind(:d2, 0)` returns a Port instead of throwing).

- [ ] **Step 3: Delete the twisted path in `src/terms.jl`**

- `Port` docstring: only the six ordinary kinds. `port_of_kind`: remove the `:d1/:d2` and `:d1wrap/:d2wrap` branches; the unknown-kind message lists `:n, :n2, :phi, :phi2, :expiphi, :exp2pin`.
- Delete `_is_twisted`, `_twisted`, `_twisted_pieces`, `_times`, `_expand_twisted_charge`.
- `circuit_terms(spec::Spec)`: the present body with `tw` all false, written plainly (diagonal `n2port()`, cross `nport() ⊗ nport()`, linear `nport()` with `-4 Σ_j (E_C[i,j] + E_C[j,i]) n_g[j]`, inductive, flux, cosines, charge cosines). Delete the two-argument method. Docstring: the expansion is the same for every basis; "the offset charge enters only through the linear and constant pieces; a `PeriodicGrid` mode is expanded like a charge-basis one (periodic gauge, design 2026-10-30)".
- `constant_term(spec::Spec) = spec.constant + 4 * _sum(spec.n_g[i] * spec.E_C[i, j] * spec.n_g[j] for i in 1:N, j in 1:N)`; delete the two-argument method.
- Header comment (line 6 area): drop the derivative pieces.

- [ ] **Step 4: Delete the twisted path in `src/materialize.jl`, `src/tree.jl`, `src/parameters.jl`, `src/KronFreeCircuits.jl`**

- `materialize` spec form: `_materialize_terms(root, vcat(circuit_terms(spec), extra_terms), θ, constant_term(spec), previous, modetypes)`; delete `bases`, `tw`, `charges`.
- `_materialize_terms(root, terms, θ, constant, previous, modetypes)`: delete the `_check_offset_charges` and `_expand_twisted_charge` lines; the term-list form's public method passes no `charges`.
- Delete `_check_offset_charges` and `_assembled_charge`.
- `_node_port(node, leaves, children, dims, V, mode, p)`: `form(p)` only.
- `_materialize`: drop the `charges` parameter and argument at its call sites (lines 608, 611, 650, 713).
- `materialize` docstring: delete the two `PeriodicGrid` charge paragraphs and the "(so a compact mode on a PeriodicGrid gets its twisted derivative pieces)" clause; keep the harmonic check sentence.
- `tree.jl` `required_ports`: delete `twisted`, `pieces` and the `if m in twisted …` branch (always `push!(req, (m, p))`); docstring: drop the last sentence.
- `parameters.jl`: delete `twist_factor` and its docstring. `KronFreeCircuits.jl`: remove `twist_factor` from the export list.

- [ ] **Step 5: Verify nothing references the removed names**

Run: `grep -rn 'twist\|_twisted\|d1wrap\|d2wrap\|_PERIODIC_PIECES\|_assembled_charge\|_check_offset_charges' src test README.md | grep -v MersenneTwister`
Expected: no output from `src/` and `test/`; `README.md` lines only (Task 4).

- [ ] **Step 6: Run the fast tier and the slow character file**

Run: `julia --project -e 'using Pkg; Pkg.test()'` then `julia --project -e 'using Pkg; Pkg.test(; test_args=["test_character.jl"])'`
Expected: PASS, both.

- [ ] **Step 7: Commit**

```bash
git add src/terms.jl src/materialize.jl src/tree.jl src/parameters.jl src/KronFreeCircuits.jl test/test_terms.jl test/test_character.jl
git commit -m "Remove the twisted-gauge machinery: twist_factor, derivative-piece ports, charge assembly

circuit_terms and constant_term take only the spec; nport() on a PeriodicGrid
mode is an ordinary port in both materialize forms; charge_port is V'nV on every basis.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Slow-tier verification and documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md:574-584` (§20d), `:626` and `:652` (§20f), the `PeriodicGrid` row of the character table at `:641`
- Modify: `README.md:118-141` ("Periodic phase grid"), `README.md:159` (character table row), `README.md:164-165` ("Real arithmetic" paragraph)
- Modify: `docs/superpowers/STATUS.md` (header date, item 6, item 9's "twisted" phrase, a "Decided recently" line)
- Test: `test/test_zero_pi_reference.jl`, `test/test_composite_slow.jl` (rerun only)

- [ ] **Step 1: Run the slow files that use a `PeriodicGrid`**

Run: `KRONFREE_SLOW_TESTS=1 julia --project -e 'using Pkg; Pkg.test(; test_args=["test_composite_slow.jl", "test_zero_pi_reference.jl"])'` (in the background; about 10 minutes)
Expected: PASS. If a `PeriodicGrid` tolerance fails, compute the periodic-gauge discretization error from §4.5 at that `N`, set the tolerance to twice it, and say so in the test's comment.

- [ ] **Step 2: Amend spec §20d**

Append to §20d, after the "Validation" bullet:

> **Decided 2026-10-30: the periodic gauge (design `2026-10-30-periodic-gauge-design.md`).** The twisted gauge above is retired. `PeriodicGrid` serves `n`, `n2` and integer `expiphi(k)` as parameter-free cyclic operators with `PhaseGrid`'s characters, wavefunctions are strictly periodic, and the offset charge enters only through the canonical quadratic form, so term construction is basis-independent and `nport()` on a grid mode is an ordinary port. Reasons: fidelity to the paper's quantization, and charge driving — the twisted gauge is `U(t) = e^{-i n_g(t) φ}` applied to the periodic-gauge state, whose time derivative for an AC gate voltage adds the generator `ṅ_g φ`, not an operator on the circle; in the periodic gauge a drive is `f(t) n̂` with a fixed `n̂`. Costs, accepted by Eli (2026-10-30/31): a grid node is real only when the linear `n` term is structurally absent (literal `n_g = "0"`), not at `"0.5"`; and the `n_g = ½` sweet-spot pairs are degenerate only to `O(Δ^order)` on the grid (order 8: 1.1e-8 at N = 32, 4.4e-11 at 64, 9e-14 at 128), pinned by `test_periodic.jl`. The twisted-gauge bullets above are historical.

§20f: change the sentence `A spec literal is structure: a `PeriodicGrid` mode whose `n_g` entry is the number `"0.5"` has the constant twist phase −1 and qualifies.` to `A spec literal is structure: a `PeriodicGrid` mode whose `n_g` entry is the number `"0"` has no linear charge term and qualifies (periodic gauge, §20d 2026-10-30; `"0.5"` does not).`; in the table row for `PeriodicGrid` put `n2` under real, `n` under imaginary; change `(`phase_factor`, `twist_factor`)` to `(`phase_factor`)`.

- [ ] **Step 3: Rewrite the README sections**

"Periodic phase grid": keep the first three sentences; replace the twisted sentence with `The basis is parameter-free and strictly periodic: it serves `nport()` (the cyclic `-i ∂_φ`), `n2port()` (the cyclic `-∂_φ²`) and `expiphi(k)` for integer `k`, and the offset charge enters only through the coefficients of `4 (n - n_g)ᵀ E_C (n - n_g)`, as on every other basis.`; keep the divisibility sentence; replace the operators sentence with `The operators are complex unless the spec gives `n_g` as the literal `"0"` (the imaginary linear charge term is then absent; see "Real arithmetic").`; keep the code example; replace the remaining paragraph with:

> `nport()` on a `PeriodicGrid` mode is an ordinary port in both forms of `materialize`: in `ports=` on any node, in a selection rule such as `connected`, and as a factor of a term; at the leaf it equals `KronFreeCircuits.charge_port(sys, mode)`.
> For a gate drive `n_g(t) = n_g + n_g^{AC}(t)`, the Hamiltonian is `H₀ - 8 E_C n_g^{AC}(t) n̂` up to a c-number, so the kept-basis `nport()` matrix is the drive operator.
> Two costs of this gauge: a grid node is real only at the literal `n_g = "0"`, and at `n_g = ½` the sweet-spot pairs are degenerate only to the discretization error, `O(Δ^order)` (order 8: 1.1e-8 at `N = 32`, 4.4e-11 at 64, 9e-14 at 128).
> The design is in spec §20d (2026-10-30) and `docs/superpowers/specs/2026-10-30-periodic-gauge-design.md`.

"Real arithmetic": table row `| `PeriodicGrid` | `n2` | `n` | `expiphi(k)` with `expiphi(-k)` |`; the sentence `The asymmetric zero-pi and a `PeriodicGrid` mode whose `n_g` is a parameter are complex; a literal `"0.5"` in the spec is real.` becomes `The asymmetric zero-pi and a `PeriodicGrid` mode whose `n_g` is a parameter or the literal `"0.5"` are complex; the literal `"0"` is real.`

- [ ] **Step 4: Update STATUS.md**

Header `# Status, 2026-11-01`. Item 6: `Real arithmetic at special parameter values (a parameter `n_g` at 0 on a `PeriodicGrid`, zero flux in a charge-basis SQUID) is a recorded later item (§20f); such circuits run complex. Since the periodic gauge (2026-10-30) a grid node is real only at the literal `n_g = "0"`.` Item 9: `the twisted `PeriodicGrid` kinetic term` → `the `PeriodicGrid` kinetic term`. Under "Decided recently": `- `PeriodicGrid` in the periodic gauge (2026-10-30, design `2026-10-30-periodic-gauge-design.md`): fixed cyclic `n`, `n2`; offset charge only in the quadratic form; `twist_factor` and the derivative-piece ports removed; the sweet-spot pairs degenerate to `O(Δ^order)` on the grid.` Under "Where things stand": `The periodic-gauge change is on the branch `periodic-gauge`, off `main`; merging and pushing are Eli's.`

- [ ] **Step 5: Final grep and fast tier**

Run: `grep -rn 'twist' src test README.md | grep -v MersenneTwister` — Expected: no output.
Run: `julia --project -e 'using Pkg; Pkg.test()'` — Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md docs/superpowers/specs/2026-10-30-periodic-gauge-design.md docs/superpowers/plans/2026-11-01-periodic-gauge.md README.md docs/superpowers/STATUS.md
git commit -m "Periodic gauge: spec §20d/§20f amendments, README, status; design and plan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
