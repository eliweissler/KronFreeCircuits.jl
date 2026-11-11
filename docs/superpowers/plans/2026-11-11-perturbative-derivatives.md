# Perturbative Derivatives Through The Tree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First and second derivatives of every kept energy, kept state and port of a materialized tree with respect to named parameters, by perturbation theory carried upward through the tree, agreeing with finite differences of `materialize` at fixed selection, bases and discrete choices.

**Architecture:** A derivative pass walks the materialized tree children first, one parameter (or pair) at a time.
At each node it builds `∂H/∂x` as a `KronFreeOperator` from three contributions (differentiated coefficients, the children's port derivatives, the children's energy derivatives), applies a per-node rule (Hellmann–Feynman energies; states through one resolvent that is a sum over the retained computed states plus, at iterative nodes, a linear solve on their orthogonal complement; a gauge term fixed by the package's phase convention), and hands the node's energy, port and peak-amplitude derivatives to its parent.
Second order repeats the pass with right-hand sides built from first-order results; near-degenerate blocks are reported with an effective Hamiltonian; an extension on ChainRulesCore gives optimizers a gradient.

**Tech Stack:** Julia 1.13.2; no new core dependency (LinearAlgebra, SparseArrays, the existing apply layer); `ChainRulesCore` as a weak dependency with an extension; `Zygote` and `Optim` only in `test/chainrules/` and `bench/derivative_design/`; circuit-quantization through the Python tier only to generate one fixture.

**Spec:** `docs/superpowers/specs/2026-11-10-perturbative-derivatives-design.md` (binding).
Formulas of arXiv:2408.12704v3 are quoted from its LaTeX source (`main.tex`, `eq:eig_val_grad`, `eq:eig_vec_grad` at lines 283–284, `eq:eig_val_secondgrad` at line 826); the owner's notes `tight_binding.tex` are not on this machine, so the second-order state formulas below were derived for this plan and checked numerically on random Hermitian families before the plan was written.

## Global Constraints

- Julia is `$HOME/.juliaup/bin/julia` (1.13.2), not on PATH. Default tier: `$HOME/.juliaup/bin/julia --project -e 'using Pkg; Pkg.test()'`; one file: `Pkg.test(test_args=["test_x.jl"])`; slow tier: `KRONFREE_SLOW_TESTS=1`. Per task run the focused files and the default tier; the slow tier and the Python tier run once, in the last task.
- New test files are in the fast tier unless listed in `SLOW_FILES`; a fast-tier file should take under ~30 s.
- Never push, never open a pull request, never switch branches. Commit with explicit paths (`git add <paths>`), never `git add -A`.
- Never add, commit, edit or delete `bench/hensel_three_fluxonia/`, `bench/hensel_two_fluxonia/`, `bench/array_fluxonium/`, `mean_field_ref_dont_commit/`.
- No `Pkg.rm`, `Pkg.gc`, `Pkg.update`; `Pkg.add`/`Pkg.develop` only in the tasks that say so.
- Kill only your own processes; never `pkill julia`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
  (A worker whose own session instructions give a different attribution line for its model follows its session and says so in its report.)
- The core package (`src/`) gains no dependency.
- Results are reported as they are: a failing test with its output, a skipped step as skipped.

From the design, binding on every task:

- Derivatives are exact for the calculation as performed: they agree with finite differences of `materialize` at fixed selection, fixed bases, fixed solver, fixed real/complex classification and fixed discrete phase choices (the stored `peak_component` of every node).
- A node's `∂H/∂x` has three contributions: the coefficients' derivatives, the children's port derivatives, the children's energy derivatives. None may be omitted, and a test must fail when either of the last two is removed.
- The sum over states is never silently truncated: at a node without its complete spectrum the part orthogonal to the computed states comes from a linear solve whose true residual is checked.
- Point derivatives are always the true derivatives; nothing divides by a detuning below `degenerate_tol` (default `1e-9` GHz): such a pair with nonzero coupling gives `NaN` per-state derivatives and a reported block (a boundary record when the partner state is not kept), and with zero coupling the pair is left out of the sum; "zero coupling" has one definition throughout, the per-column rule of Task 4.
- A pair of states is flagged near-degenerate for a parameter `x` when `|⟨ψ_l|∂H/∂x|ψ_i⟩| · δx > r · |E_i − E_l|`, with `r = 1/10` by default and `δx` the caller's `scale` for `x`.
- Bases, grids, cutoffs, the coordinate frame and mean-field bases are held fixed; there is no reverse sweep and no third order.
- Finite-difference agreement is `1e-6` relative (to the largest element of the compared array) where the finite difference is itself trustworthy.
  The helper `check_fd` asserts both: that the finite difference agrees with itself at two step sizes to `rtol`, and that the derivative agrees with it to `rtol`.
  When the first assertion fails, the step `h` is wrong for that parameter: change `h` by at most a factor of 4 either way and say so in the report. `rtol` and `atol` are never loosened beyond the values written in this plan.
- Finite-difference tests avoid exact degeneracies, symmetric points (`n_g = 0` or `1/2`, flux `0` or `π` unless the test says so), kept sets whose boundary lies near a level crossing, and points where a node's discrete phase choice (its `peak_component`) changes inside the stencil.
  The shared test circuits (`dc_*` in `test/helpers.jl`, Task 3) were chosen and measured for this: every tree comparison of Tasks 5 and 8 was run as a finite-difference self-consistency check, at the steps written in the tests, when the plan was revised.
  When a node's `peak_component` does change inside a stencil, the finite differences of that node's vectors and off-diagonal port elements, and of the vectors of every node above it, are meaningless; energies at every node and the ports of the nodes above it remain well defined.
- By the plan's estimate no fast-tier file of this plan takes more than about 30 s on this machine; the derivative test files that solve nodes with `LOBPCG`, `Lanczos` or `ShiftInvert` are in `SLOW_FILES`, because the first call of `LOBPCG` alone compiles for 35 to 40 s per element type here.
  The one exception is `test_retention.jl` (Task 2), whose single real `LOBPCG` solve adds nothing to the default tier, where `test_meanfield.jl` already compiles `LOBPCG` on a real operator; run alone by name, that file takes about 40 s longer.
  A fast-tier file that is measured above 30 s is reported with its time, and the task says which testset moves to a slow file.
- Markdown and docstrings follow the repository's style: one sentence per line.

## The code this plan is written against

The branch `mean-field` as merged into `main`, including the fix wave announced by the controller on 2026-11-11, which changes three things under `src/`:
`AugmentedChild` has a field `θ::NamedTuple` (the `θ` given to `augment`), and `materialize` throws an `ArgumentError` when its `θ` differs from it in a parameter the stored bare terms depend on;
`mean_field_solutions` has a keyword `bare`, and `mean_field_augment` forms the bare solves once;
`suggest_bases` has `when_capped::Symbol = :warn`.
Line numbers quoted in the tasks are those of commit `6853aaf` and may have moved by a few lines; find the code by the names given.
A task that meets a positional construction of `AugmentedChild` or `MaterializedNode` not listed here treats it like the listed ones and says so in its report.

## Decisions and their costs

The controller's decisions (D1–D13), then the ones this plan adds (P1–P9).

- **Decision 1: formulation** — per node and kept state, `(H − E_i) w_i = −(H⁽¹⁾ − E_i⁽¹⁾) ψ_i` with `w_i ⟂ ψ_i`, plus the gauge term of the phase convention; second order with the same resolvent, the normalization condition and a second-order gauge condition; one resolvent interface for both node kinds — it is the textbook formulation and the one the design describes — cost if wrong: Tasks 4 and 7 are rewritten; the finite-difference suite is the arbiter.
- **Decision 2: retention** — `MaterializedNode` gains one field `retained`; computed pairs are retained at every node except a `Dense` node above 1000 states, where it is opt-in; a `Dense` node with a `compute` limit computes its full decomposition when retaining — the owner prefers retention (design §4.5) — cost if wrong: forward time and memory rise for every user (Task 2 measures both); the keyword `retain = :never` restores the old behaviour per node.
  The design (§4.5) asks for the convergence of the sums to be measured before the default is fixed; the per-node rule needs the retained pairs to exist, so Task 2 sets the default, Task 4 measures, and Task 15 puts the measurement before the owner as an open decision.
- **Decision 3: coefficients** — symbolic differentiation of the stored `Expr` over the whole grammar, real-parameter rules for the non-analytic calls — no new dependency, and the result is an ordinary `Coefficient` — cost if wrong: one file (`src/diffcoeff.jl`) is replaced.
- **Decision 4: the derivative operator** — a new builder beside `_node_operator`, never a mutation of the forward operator, never dropping a term because its coefficient vanishes at `θ` — the forward real operator drops such terms (`materialize.jl:585`) — cost if wrong: confined to `src/derivoperator.jl`.
- **Decision 5: phase** — supported for the default rules (leaf `:positive_sector`, branch joint-peak) in real and complex arithmetic with discrete choices fixed; a node whose phase was fixed by following a previous solution is an `ArgumentError` — cost: a followed sweep cannot be differentiated directly (see "Disagreements" for the narrowing).
- **Decision 6: API** — `derivatives(sys::System; wrt, order = 1, mixed = (), scale = (;), r = 0.1, ...)` returning a `Derivatives` object read through new methods of `energies`, `eigenvectors`, `port`, and `blocks(d)` — no new verbs for the same quantities — cost if wrong: a rename.
- **Decision 7: near-degeneracy** — exactly the design's §4.4 resolution; `scale` is a keyword `NamedTuple` — the design marks how `δx` is supplied as unconfirmed by the owner — cost if wrong: a keyword rename; the block test itself is unaffected.
- **Decision 8: stored-data children** — an augmented child is its bare operator projected on its fixed stored basis; its `θ` is recorded in the `AugmentedChild` and checked by `materialize` (the controller's update of 2026-11-11), and the derivative pass repeats the check over every term inside the augmented node's subtree; a `GivenSystem` is an error unless declared parameter-independent; reduced specs get no special handling — cost: the union basis itself is held fixed, so the derivative is that of the projected problem, not of a re-run of the mean-field loop.
- **Decision 9: outer rule** — a ChainRulesCore extension with an `rrule` and an `frule` for one function, tests in their own environment with Zygote, the demonstration in a bench directory — the core stays free of autodiff — cost if wrong: an extension file.
- **Decision 10: KITE** — a committed JSON fixture of the symbolic KITE in the optimum frame, generated once through the Python tier — the default and slow tiers need no Python — cost: the fixture goes stale if circuit-quantization changes; its generator and a level regression pin it.
- **Decision 11: accuracy** — `1e-6` relative where the finite difference is trustworthy; Dense nodes or tightened solver tolerances in tests — cost: tests of iterative nodes do not exercise the default `1e-8` tolerance.
- **Decision 12: devices** — one late device test, skipped (and reported as skipped) when another process is using the GPU.
- **Decision 13: scope at second order** — energies, kept states and ports all get second derivatives; split over Tasks 7 and 8.

- **Decision P1: what is retained** — `Retained` holds every computed energy, the vectors of the computed states that were *not* kept, and two flags; the kept vectors are not stored twice — a node that keeps all it computes retains nothing extra — cost: the resolvent reads two arrays instead of one.
- **Decision P2: where `retain` lives** — a `Node` keyword `retain ∈ (:auto, :always, :never)`, not a `materialize` keyword — it then reaches every path that solves a node (`_solve_subtree`, mean-field children) without plumbing — cost: no tree-wide switch; a user sets it per node.
- **Decision P3: the linear solver** — a hand-written preconditioned conjugate gradient, all kept states in lockstep through the operator's block apply, projected off the computed states, preconditioned by the Jacobi diagonal `1 / hypot(D − E_i, 0.01)` that `RitzJacobi` uses — `KrylovKit.linsolve` has no preconditioner hook and applies one column at a time; the hand-written loop uses only `mul!`, broadcasts and column sums, so it runs on a device unchanged — cost: no reuse of the `ShiftInvert` factorization (the design allows but does not require it); slow convergence on stiff grid leaves solved by `ChebyshevFilter` is reported, not hidden (`Derivatives.linear_solves`).
- **Decision P4: gauge through one functional** — the phase convention at any node is "`ℓ_j · ψ_j` is real and positive", with `ℓ_j` the peak functional the node already stores implicitly (`peak_component`, the children's `peak_amplitudes`); leaf and branch share one code path, and each node's `peak_amplitudes` derivative is propagated upward — cost: one more array per node per parameter.
- **Decision P5: exact degeneracy** — a pair within `degenerate_tol = 1e-9` GHz is exactly degenerate; with overlap `|⟨ψ_l|r_j⟩| ≤ 1e-10 ‖r_j‖` it is skipped, otherwise the column is `NaN`; a `NaN` child makes every ancestor `NaN` — cost: a parent whose own derivatives exist despite a degenerate coupled pair in a child reports `NaN`; the block report names the child.
- **Decision P6: order of the effective Hamiltonian** — second order in the displacement (first order when `order = 1`): `H_eff(δ) = E_B + Σ_x δ_x C^x + ½ Σ_{x,y} δ_x δ_y (⟨H_xy⟩ + F^{xy} + F^{yx})`, with outside states folded in through `F`; its eigenvalues differ from a re-solve by `O(|δ|³ v³ / g²)` (`v` the block's largest coupling to outside states, `g` its gap to them).
- **Decision P7: independent subtrees** — a node none of whose terms (in its subtree) depends on `x` has no derivative data (`nothing`), is not visited, and reads as zeros.
- **Decision P8: selection-boundary test without `scale`** — a kept state `i` and an unkept computed state `l` are close when `|E_i − E_l| ≤ 1e-6 · max(|E_i − E_1|, |E_l − E_1|)`, with `E_1` the node's lowest computed energy; with `scale` it is the block test.
- **Decision P9: the outer function** — `spectrum(root, spec, θ; ports) -> (; energies, ports)`, defined in the core with no autodiff, is the one function the rules cover.

## Review Focus

The five ways this feature is most likely to give a silently wrong derivative; each has its test in the task named.

1. **A term whose coefficient vanishes at `θ` but whose derivative does not** (a coupling at `g = 0`, a flux term at `varphi_ext = 0`): the forward operator has no such term, and the derivative must. (Task 3: the flux leaf; Task 5: the tree at `g = 0`.)
2. **The root constant** (`4 n_gᵀ E_C n_g`, `E_L varphi_ext²/2`): its derivative shifts every root energy alike, so transition frequencies hide its omission; absolute energies are compared, and a child's energies must not receive it. (Task 4 at a leaf; Task 5 at a tree; Task 8 at second order.)
3. **The phase-convention term on complex nodes**: energies and diagonal elements are blind to it; the off-diagonal elements of a non-Hermitian port (`expiphi`) are compared at a complex leaf and a complex branch, to both orders. (Tasks 4, 5, 8.)
4. **A kept set that is not the lowest computed states** (`indices([1, 3, 4])`): the unkept computed state between them belongs in every sum. (Task 4 dense; Task 6 iterative.)
5. **A mixed second derivative of parameters that enter at different levels of the tree** (one inside a leaf, one in a root coupling): the cross terms `∂c/∂x · ∂P/∂λ` of the product rule. (Task 8.)

---

### Task 1: Derivatives of coefficients

**Files:**
- Create: `src/diffcoeff.jl` (included after `character.jl` in `src/KronFreeCircuits.jl`; it uses `_maybe_complex`)
- Modify: `src/KronFreeCircuits.jl` (include, export `differentiate`), `test/helpers.jl` (finite-difference helpers)
- Test: `test/test_diffcoeff.jl` (fast tier)

**Interfaces:**
- Consumes: `Coefficient`, `_make`, `free_symbols`, `iszero_expr`, `phase_factor` (`src/parameters.jl`); `_maybe_complex` (`src/character.jl`).
- Produces, exported:

```julia
# ∂c/∂x as a new Coefficient: its `ex` is the differentiated expression, lightly simplified; a constant (a Number `ex`,
# so `iszero_expr` is true for a zero) when `c` does not depend on `x`.  Second derivatives are two calls.
differentiate(c::Coefficient, x::Symbol) -> Coefficient
```

- Produces, in `test/helpers.jl`, used by every later test file:

```julia
quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())        # run `f()` with logging silenced
shifted(θ::NamedTuple, x::Symbol, δ::Real) = merge(θ, NamedTuple{(x,)}((θ[x] + δ,)))
# Each returns `(value, err)`, by one formula.  With `D(s)` the plain central difference at step `s` and
# `R(s) = (4 D(s/2) − D(s)) / 3` its Richardson extrapolation:   value = R(h/2),   err = ‖R(h) − R(h/2)‖∞
# (the largest absolute element), so `D` is evaluated at h, h/2 and h/4.  `f(θ)` returns a number or an array.
fd_derivative(f, θ::NamedTuple, x::Symbol; h::Real)                 # D(s) = (f(θ + s) − f(θ − s)) / 2s
fd_second(f, θ::NamedTuple, x::Symbol; h::Real)                     # D(s) = (f(θ + s) − 2 f(θ) + f(θ − s)) / s²
# D(s) = (f(+,+) − f(+,−) − f(−,+) + f(−,−)) / (4 s² hx hλ) at the displacements ±s·hx of x and ±s·hλ of λ, s playing the
# part of the step: value = R(1/2), err = ‖R(1) − R(1/2)‖∞, so s = 1, 1/2 and 1/4.
fd_mixed(f, θ::NamedTuple, x::Symbol, λ::Symbol; hx::Real, hλ::Real)
fd_pair(f, θ, x::Symbol, λ::Symbol; hx::Real, hλ::Real = hx) = x === λ ? fd_second(f, θ, x; h=hx) : fd_mixed(f, θ, x, λ; hx, hλ)
# Two `@test`s: `err ≤ rtol·‖value‖∞ + atol` (the finite difference is trustworthy) and
# `‖analytic − value‖∞ ≤ rtol·‖value‖∞ + atol`, with `‖·‖∞` the largest absolute element.
check_fd(analytic, fd::Tuple; rtol::Real = 1e-6, atol::Real = 0.0)
```

Rules of `differentiate`, for real parameters, with `f′` the derivative of a subexpression:
a number, a named constant (`pi π im ℯ`) or another symbol gives `0`, and `x` gives `1`;
`+`, `-` (unary and binary) termwise; n-ary `*` by the product rule; `a / b` and `a // b` as `a′/b − a·b′/b^2`;
`f^g` as `g·f^(g−1)·f′` when `g′` is `0` (the exponent `g − 1` folded when `g` is a number, so an integer literal stays an integer literal), as `f^g·log(f)·g′` when `f′` is `0`, and as `f^g·(g′·log(f) + g·f′/f)` otherwise;
`sqrt(f)` → `f′/(2·sqrt(f))`; `exp` → `exp(f)·f′`; `log` → `f′/f`; `cos` → `−sin(f)·f′`; `sin` → `cos(f)·f′`; `tan` → `f′/cos(f)^2`; `cosh` → `sinh(f)·f′`; `sinh` → `cosh(f)·f′`; `tanh` → `f′/cosh(f)^2`;
`cis(f)` → `im·cis(f)·f′`; `cispi(f)` → `im·pi·cispi(f)·f′`;
`conj(f)` → `conj(f′)`; `real(f)` → `real(f′)`; `imag(f)` → `imag(f′)`; `abs(f)` → `real(conj(f)·f′)/abs(f)`;
`sign(f)` → `0`, except that `f′ ≠ 0` with `_maybe_complex(f)` true is an `ArgumentError` naming the expression (the sign of a complex value is not piecewise constant).
Simplification, applied as each node is built and nowhere else: a zero factor makes a product `0`; zero terms and unit factors are dropped; `f^1` is `f`, and `f^0` and `1^g` are `1`; an operation whose arguments are all numbers is evaluated (left as it is if the evaluation throws); a one-argument `+` or `*` is its argument.
So `x^2` gives `2 * x`, then `2`, then `0`.
The result goes through `_make(ex, free_symbols(ex))`, so a number becomes a constant coefficient.

- [ ] **Step 1: Add the helpers to `test/helpers.jl` and write the failing tests** in `test/test_diffcoeff.jl`

```julia
using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: Coefficient, differentiate, phase_factor, iszero_expr, circuit_terms, constant_term

@testset "differentiate: every grammar element against finite differences" begin
    P = [:x, :y]
    cases = ["3.5", "y", "x", "x + y", "x - y", "-x", "x * y * x", "x / y", "y / x", "x^3", "x^(-2)", "x^2.5", "x^y", "y^x",
             "(x*y)^(x+1)", "1//2 * x", "sqrt(x)", "exp(x*y)", "log(x)", "cos(x)", "sin(2*x)", "tan(x/2)", "cosh(x)",
             "sinh(x)", "tanh(x)", "abs(x - 1)", "abs(x - 3)", "cis(x)", "cispi(x)", "conj(cis(x) * y)", "real(cis(x)*x)",
             "imag(cis(x)*x)", "abs(cis(x) * (x + 2))", "sign(y) * x", "sign(x - 3) * x", "exp(im * x) * pi + ℯ^x",
             "sqrt(cis(x))", "log(cis(x) + 2)", "x .* y .+ x .^ 2"]
    for src in cases, θ in [(x=0.7, y=1.3), (x=1.9, y=0.6), (x=1.2, y=2.4)], v in P
        c = Coefficient(src, P)
        d1 = differentiate(c, v)
        check_fd(d1(θ), fd_derivative(t -> c(t), θ, v; h=2e-3); rtol=1e-8, atol=1e-10)
        check_fd(differentiate(d1, v)(θ), fd_second(t -> c(t), θ, v; h=1e-2); rtol=1e-6, atol=1e-7)
        w = v === :x ? :y : :x
        check_fd(differentiate(d1, w)(θ), fd_mixed(t -> c(t), θ, v, w; hx=1e-2, hλ=1e-2); rtol=1e-6, atol=1e-7)
    end
end

@testset "differentiate: constants, folding, literals of any type, errors" begin
    c = Coefficient("x^2", [:x])
    @test differentiate(c, :x)((x=1.7,)) ≈ 3.4
    d2 = differentiate(differentiate(c, :x), :x)
    @test d2.ex isa Number && d2.ex == 2 && isempty(d2.deps)
    @test iszero_expr(differentiate(d2, :x))
    @test iszero_expr(differentiate(Coefficient("cos(y)", [:x, :y]), :x))             # no dependence: a structural zero
    # coefficients built by the package's algebra carry ComplexF64 literals and exp(±im*offset)
    E, φ = Coefficient("E_J", [:E_J, :phi]), Coefficient("phi + 0.3", [:E_J, :phi])
    k = (0.5 - 0.25im) * (E * phase_factor(φ, -1)) + Coefficient(2.0im)
    θ = (E_J=3.1, phi=0.9)
    @test differentiate(k, :E_J)(θ) ≈ (0.5 - 0.25im) * exp(-im * 1.2)
    @test differentiate(k, :phi)(θ) ≈ (0.5 - 0.25im) * 3.1 * (-im) * exp(-im * 1.2)
    @test differentiate(differentiate(k, :phi), :phi)(θ) ≈ -(0.5 - 0.25im) * 3.1 * exp(-im * 1.2)
    @test differentiate(k, :E_J).deps == [:phi]
    nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args[2:end]; init=0) : 1
    @test nodes(differentiate(differentiate(k, :phi), :phi).ex) <= 40                 # second derivatives stay small
    # sign of a possibly complex argument that depends on x
    @test_throws ArgumentError differentiate(Coefficient("sign(cis(x)) * x", [:x, :y]), :x)
    @test differentiate(Coefficient("sign(cis(y)) * x", [:x, :y]), :x)((x=1.0, y=0.2)) ≈ sign(cis(0.2))
end

@testset "differentiate: every coefficient of the JSON fixtures" begin
    JSON3 = KronFreeCircuits.JSON3
    files = [joinpath(dir, f) for (dir, _, fs) in walkdir(fixture("")) for f in fs if endswith(f, ".json")]
    ntested = 0
    failed = String[]                 # coefficients that cannot be evaluated at the test point: reported, never skipped silently
    for file in files
        doc = JSON3.read(read(file, String))
        doc isa JSON3.Object && get(doc, :format, nothing) == "kronfree-spec" || continue
        spec = load_spec(file)
        ps = spec.parameters
        isempty(ps) && continue
        θ = NamedTuple{Tuple(ps)}(Tuple(0.6 .+ 0.37 .* (1:length(ps)) ./ length(ps)))
        for c in vcat([t.coeff for t in circuit_terms(spec)], constant_term(spec)), x in c.deps
            try c(θ) catch; push!(failed, "$file: $(c.ex)"); continue end
            d1 = differentiate(c, x)
            check_fd(d1(θ), fd_derivative(t -> c(t), θ, x; h=1e-3); rtol=1e-7, atol=1e-9)
            check_fd(differentiate(d1, x)(θ), fd_second(t -> c(t), θ, x; h=1e-2); rtol=1e-5, atol=1e-6)
            ntested += 1
        end
    end
    @test ntested > 50
    @test isempty(failed)             # a failure here prints the files and expressions
end
```

- [ ] **Step 2: Run and confirm failure**

Run: `$HOME/.juliaup/bin/julia --project -e 'using Pkg; Pkg.test(test_args=["test_diffcoeff.jl"])'`
Expected: FAIL with `UndefVarError: differentiate`.

- [ ] **Step 3: Implement `differentiate` in `src/diffcoeff.jl`**, a recursive walk over `c.ex` with the rules above, and export it.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_diffcoeff.jl"])`, then the default tier.
Expected: PASS.
A fixture coefficient that cannot be evaluated at the test point (a domain error) fails the last assertion, which prints its file and expression; report it and do not change the test point.

- [ ] **Step 5: Commit** (`Coefficients: symbolic derivatives over the whole expression grammar`), with `src/diffcoeff.jl src/KronFreeCircuits.jl test/helpers.jl test/test_diffcoeff.jl`.

---

### Task 2: Retaining the computed pairs

**Files:**
- Modify: `src/tree.jl` (`Node`: field and keyword `retain`; its docstring, which also still says "`Dense()` up to 2 500": the cutoff is 1000), `src/materialize.jl` (`MaterializedNode`, its constructors, `_materialize_given`, `_solve_node`, `_assemble_node`), `src/augment.jl` (`_augmented_node`)
- Create: `bench/retention_cost.jl`
- Test: `test/test_retention.jl` (fast tier; estimated 10 s within the default tier, see the Global Constraints on its `LOBPCG` solve)

**Interfaces:**
- Produces:

```julia
# What a node's solve computed beyond its kept vectors.
struct Retained{T<:Union{Float64,ComplexF64}}
    energies::Vector{Float64}             # every computed eigenvalue, in the solver's (ascending) order; `kept` indexes it
    unkept::Vector{Int}                   # the indices among them that were not kept, increasing
    vectors::Union{Nothing,Matrix{T}}     # the unkept eigenvectors, one column per entry of `unkept` (as the solver returned
                                          # them, no phase fixed); `nothing` when the node does not retain
    complete::Bool                        # `vectors !== nothing` and kept ∪ unkept is the whole spectrum of the node's space
    followed::Bool                        # the kept vectors' phases were fixed by a previous solution (`:follow` with a reference)
end

# New last field of MaterializedNode{T}:
#     retained::Retained{T}
# New Node keyword and field:
#     Node(children; ..., retain::Symbol = :auto)       # :auto, :always or :never; anything else is an ArgumentError
const _RETAIN_DENSE_MAX = 1000
_retains(node::Node, solver::Solver, N::Int)::Bool =
    node.retain === :always || (node.retain === :auto && !(solver isa Dense && N > _RETAIN_DENSE_MAX))
```

Behaviour:
- A node that retains keeps the vectors of its computed-but-unkept states.
  A `Dense` node that retains computes its full decomposition whatever `compute` is (`solve(Dense(), H, 0)`), hands only the lowest `_compute_count(node, dims)` pairs to the selection rule exactly as before (so `kept`, `lowest(0)`, `within`, `connected` behave as they did), and retains all the others: `complete == true`.
- A node that does not retain stores the computed energies and `unkept`, with `vectors === nothing` and `complete == false`; the solve is the one of today.
- `followed` is true exactly when `_assemble_node` took the branch `node.phase === :follow && reference !== nothing`.
- `_assemble_node` gains the keyword `retained = nothing`; when `nothing` it stores the default: the given energies, no unkept states, an `N × 0` matrix, `complete = (number of columns == prod(dims))`, `followed` as above.
  `augment` and `_materialize_given` get this default (a given system is complete).
- Every existing ten-argument construction keeps working: add the method `MaterializedNode{T}(node, dims, energies, vectors, kept, ports, joint_ports, children, peak_component, peak_amplitudes) where {T}` that fills the same default, used by the outer constructor and `_materialize_given` in `materialize.jl`, by the tests that build nodes by hand (`test_catalogue.jl:336`, `test_selection.jl:75/95/117`, and any other ten-argument construction a grep for `MaterializedNode` finds) and by `bench/end_to_end.jl:372`, none of which is edited.
  `_assemble_node` passes the real `Retained`, and `_augmented_node` in `augment.jl` passes `s.retained` explicitly.
- The `Node` docstring gains one entry for `retain`, and the `materialize` docstring's sentence "retaining the whole computed block is future work" is replaced by what is retained now.

- [ ] **Step 0: Before changing anything, run the default tier** and record its wall time; Step 4 compares with it.

- [ ] **Step 1: Write the failing tests** in `test/test_retention.jl`

```julia
using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: Retained, GivenSystem, Term, Factor, Coefficient, Port, bare_child, augment, UnionReport

@testset "retention of computed pairs" begin
    spec = load_spec(fixture("transmon.json")); θ = (E_J=30.0, E_C=1.2, n_g=0.25)
    b = :varphi_1 => ChargeBasis(30)                                               # 61 states
    full = materialize(Node([b]; solver=Dense()), spec, θ).root
    @test full.retained.complete && isempty(full.retained.unkept) && size(full.retained.vectors) == (61, 0)
    # Dense with a compute limit: the whole spectrum is retained, and the selection still sees `compute` pairs
    mn = materialize(Node([b]; solver=Dense(), compute=10, keep=lowest(4)), spec, θ).root
    r = mn.retained
    @test r isa Retained{Float64} && r.complete && !r.followed
    @test length(r.energies) == 61 && r.energies ≈ full.energies atol=1e-10
    @test mn.kept == 1:4 && r.unkept == 5:61 && size(r.vectors) == (61, 57)
    @test norm(r.vectors' * mn.vectors) < 1e-12 && r.vectors' * r.vectors ≈ I
    @test length(materialize(Node([b]; solver=Dense(), compute=10, keep=lowest(0)), spec, θ).root.kept) == 10
    mi = materialize(Node([b]; solver=Dense(), compute=6, keep=indices([1, 3, 4])), spec, θ).root
    @test mi.kept == [1, 3, 4] && mi.retained.unkept == vcat(2, 5:61)
    # :never keeps the energies and nothing else
    nv = materialize(Node([b]; solver=Dense(), compute=10, keep=lowest(4), retain=:never), spec, θ).root.retained
    @test nv.vectors === nothing && !nv.complete && nv.unkept == 5:10 && nv.energies ≈ full.energies[1:10] atol=1e-10
    @test_throws ArgumentError Node([b]; retain=:sometimes)
    # a Dense node above 1000 states retains only when asked
    fl = load_spec(fixture("fluxonium.json")); θf = (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=2.0)
    grid = :varphi_1 => PhaseGrid(-10π, 10π, 1201; order=8)
    big = materialize(Node([grid]; solver=Dense(), compute=6, keep=lowest(4)), fl, θf).root.retained
    @test big.vectors === nothing && !big.complete && length(big.energies) == 6
    always = materialize(Node([grid]; solver=Dense(), compute=6, keep=lowest(4), retain=:always), fl, θf).root.retained
    @test always.complete && size(always.vectors) == (1201, 1197)
    # an iterative node retains its computed-but-unkept pairs; a given system is complete
    sp2 = load_spec(fixture("two_transmons.json")); θ2 = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.15, g=0.3)
    leaf(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(12))
    it = materialize(Node([leaf(1), leaf(2)]; solver=LOBPCG(tol=1e-10), compute=8, keep=lowest(6)), sp2, θ2).root
    @test !it.retained.complete && it.retained.unkept == [7, 8] && size(it.retained.vectors) == (144, 2)
    @test length(it.retained.energies) == 8 && norm(it.retained.vectors' * it.vectors) < 1e-8
    g = GivenSystem(:a; energies=[0.0, 1.0], operators=Dict(:x => ComplexF64[0 1; 1 0]))
    gs = materialize(Node([g]; solver=Dense()), [Term(Coefficient(0.1), [Factor(:a, Port(:x, 0))])], (;))
    @test gs.root.children[1].retained.complete
    # `followed` records that a previous solution fixed the phases
    tr(ng) = (E_J=30.0, E_C=1.2, n_g=ng)
    fnode = Node([b]; solver=Dense(), keep=lowest(3), phase=:follow)
    s0 = materialize(fnode, spec, tr(0.20))                                       # no previous: the positive-sector rule
    s1 = materialize(fnode, spec, tr(0.21); previous=s0)
    @test !s0.root.retained.followed && s1.root.retained.followed
end
```

- [ ] **Step 2: Run and confirm failure** (`UndefVarError: Retained`, or a `MethodError` on the `retain` keyword).

- [ ] **Step 3: Implement** as described under Behaviour.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_retention.jl","test_materialize.jl","test_selection.jl","test_augment.jl","test_catalogue.jl","test_meanfield.jl","test_given.jl"])`, then the default tier.
Expected: PASS with no existing expectation changed.
Record the wall time of the default tier beside the one of Step 0.

- [ ] **Step 5: Measure the cost of the default.**
Write `bench/retention_cost.jl`: for each of (a) a transmon leaf on `ChargeBasis(400)` (801 states), `Dense`, `compute=10, keep=lowest(10)`; (b) a fluxonium leaf on `PhaseGrid(-10π, 10π, 901; order=8)`, `Dense`, `compute=10, keep=lowest(10)`; (c) the KITE tree that `kite_tree` of `test/test_meanfield_kite.jl` builds, written out in the script, on `test/fixtures/kite/optimum.json` (with `when_capped = :allow` on `suggest_bases` if it warns); it prints the minimum of five timings of `materialize` and `Base.summarysize(sys)` with every node at `retain=:never` and at `retain=:auto`.
Run: `$HOME/.juliaup/bin/julia --project bench/retention_cost.jl`.
Put the six times, the six sizes and the two default-tier wall times in the commit message body.

- [ ] **Step 6: Commit** (`Nodes retain their computed pairs; Node(...; retain)`), with `src/tree.jl src/materialize.jl src/augment.jl test/test_retention.jl bench/retention_cost.jl`.

---

### Task 3: The derivative operator and two-sided ports

**Files:**
- Create: `src/derivoperator.jl` (included after `augment.jl` in `src/KronFreeCircuits.jl`)
- Modify: `test/helpers.jl` (the shared test circuits)
- Test: `test/test_derivoperator.jl` (fast tier)

**Interfaces:**
- Consumes: `Retained`, `Node(...; retain)` (Task 2); `differentiate` (Task 1); `fd_derivative`, `check_fd`, `quiet` (Task 1); existing `_axis_ops`, `_real_expansion`, `_key_character`, `_mode_leaves`, `_port_matrix`, `AxisTerm`, `KronFreeOperator`, `DenseOp`, `DiagonalOp`.
- Produces, internal (not exported):

```julia
# The derivative data of one node along one parameter (first order) or one pair (second order); it mirrors MaterializedNode.
struct NodeDerivative{T<:Union{Float64,ComplexF64}}
    energies::Vector{Float64}                      # one per kept state
    vectors::Matrix{T}                             # prod(dims) × nkept
    gauge::Vector{Float64}                         # Im⟨ψ_j|∂ψ_j⟩ per kept state (zero on a real node)
    ports::Dict{Tuple{Symbol,Port},NodeMatrix}     # same keys and element types as the node's `ports`
    joint_ports::Dict{_FactorKey,NodeMatrix}       # same keys and element types as the node's `joint_ports`
    peak_amplitudes::Matrix{T}
    children::Vector{Any}                          # per axis: `nothing` (a basis, or a child with no dependence) or its NodeDerivative
end

# ∂H/∂x of the node `mn` on its product basis, in the node's arithmetic (Float64 when `mn isa MaterializedNode{Float64}`).
# `terms` are the terms acting at this node, `dx[a]` the derivative data of the child on axis `a` (or `nothing`),
# `constant` the root's scalar (zero below the root).  `parts` exists for the three-contribution test only.
_derivative_operator(mn::MaterializedNode, terms::Vector{Term}, θ::NamedTuple, x::Symbol, dx::Vector{Any};
                     constant::Coefficient = Coefficient(0), on = Array,
                     parts = (:coefficients, :ports, :energies)) -> KronFreeOperator

# W' (⊗ ops) V, in the natural type of the key K at `mn`: Float64 for a real-character key of a real node, else ComplexF64.
# `ops` has the form `_axis_ops(mn.node, mn.children, K)` returns, one `(axis, op, subkey)` per child the key touches;
# a caller replaces an entry's `op` by `DenseOp` of a child's derivative to differentiate the operator.
_sandwich(mn::MaterializedNode, W::AbstractMatrix, V::AbstractMatrix, K::_FactorKey,
          ops::Vector{Tuple{Int,LocalOp,_FactorKey}}) -> NodeMatrix

# The child's derivative of its port (one factor) or joint port (several) on the sub-key Ka, as an operator.
_derivative_op(d::NodeDerivative, Ka::_FactorKey) -> DenseOp
```

- Produces, in `test/helpers.jl`, the test circuits and the scaffolding every later test file uses: `dc_transmon`, `dc_fluxonium`, `dc_oscillator`, `dc_two_transmons`, `dc_chain3`, `dc_zero_pi`, `dc_abc`, `dc_cos2phi`; `dc_solve(c, θ = c.θ)`, `dc_memo(c)`; `fd_child(at, c, child, x; h)` and `fd_child(at, c, child, x, λ; hx, hλ)`.
No test file defines its own copy of these (the names `solve` and `fd_child` of a test file's own would collide in `Main`, where `runtests.jl` includes every file, and `solve` is a package function some files import).
Each `dc_*` call builds a fresh tree, so `materialized(sys, node)` finds its nodes by identity; write them exactly:

```julia
const chain3_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_J1", "E_J2", "E_J3", "g12", "g23", "n_g2"],
 "modes": [{"name": "varphi_1", "type": "c"}, {"name": "varphi_2", "type": "c"}, {"name": "varphi_3", "type": "c"}],
 "E_C": [["1.2", "g12/8", "0"], ["g12/8", "1.0", "g23/8"], ["0", "g23/8", "0.8"]],
 "E_L": [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]],
 "flux_linear": ["0", "0", "0"], "n_g": ["0.21", "n_g2", "0.1"],
 "cosines": [{"energy": "E_J1", "omega": [1, 0, 0], "offset": "0"}, {"energy": "E_J2", "omega": [0, 1, 0], "offset": "0"},
             {"energy": "E_J3", "omega": [0, 0, 1], "offset": "0"}]}
"""
const abc_deriv_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_C", "E_J", "E_J3", "n_ga", "varphi_ext"],
 "modes": [{"name": "a", "type": "c"}, {"name": "b", "type": "c"}, {"name": "c", "type": "c"}],
 "E_C": [["E_C", "0", "0"], ["0", "1.3*E_C", "0"], ["0", "0", "0.8*E_C"]],
 "E_L": [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]],
 "flux_linear": ["0", "0", "0"], "n_g": ["n_ga", "0.13", "0.37"],
 "cosines": [{"energy": "E_J", "omega": [1, 0, 0], "offset": "0"}, {"energy": "0.7*E_J", "omega": [0, 1, 0], "offset": "0"},
             {"energy": "1.2*E_J", "omega": [0, 0, 1], "offset": "0"}, {"energy": "E_J3", "omega": [1, 1, 1], "offset": "varphi_ext"}]}
"""
const oscillator_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz", "parameters": ["E_C", "E_L"],
 "modes": [{"name": "phi", "type": "e"}], "E_C": [["E_C"]], "E_L": [["E_L"]], "flux_linear": ["0"], "n_g": ["0"],
 "cosines": [], "constant": "0"}
"""

# One transmon leaf: real on a charge basis, complex on a periodic grid (its `n` is imaginary there).
function dc_transmon(; periodic::Bool=false, keep=lowest(4), compute::Int=0, solver=Dense(), retain::Symbol=:auto,
                     θ=(E_J=20.0, E_C=1.0, n_g=0.2))
    b = periodic ? PeriodicGrid(64; order=8) : ChargeBasis(30)
    root = Node([:varphi_1 => b]; solver, compute, keep, retain, ports=[(:varphi_1, nport()), (:varphi_1, expiphi(1.0))])
    return (; root, spec=load_spec(fixture("transmon.json")), θ)
end
# One fluxonium leaf (real), 601 grid points, off its sweet spots.
function dc_fluxonium(; varphi_ext::Real=2.0, E_J::Real=8.9, solver=Dense(), compute::Int=8, keep=lowest(5), retain::Symbol=:auto)
    root = Node([:varphi_1 => PhaseGrid(-10π, 10π, 601; order=8)]; solver, compute, keep, retain,
                ports=[(:varphi_1, phiport()), (:varphi_1, nport())])
    return (; root, spec=load_spec(fixture("fluxonium.json")), θ=(E_J=E_J, E_C=2.5, E_L=0.5, varphi_ext=varphi_ext))
end
# One harmonic oscillator on a Fock basis matched to θ: H is diagonal there, so closed forms hold to round-off.
function dc_oscillator()
    θ = (E_C=0.5, E_L=1.3)
    root = Node([:phi => FockBasis(40; phi_zpf=(2 * θ.E_C / θ.E_L)^0.25)]; solver=Dense(), keep=lowest(6))
    return (; root, spec=KronFreeCircuits.parse_spec(oscillator_json), θ)
end
# Two coupled transmons, a two-level tree: real on charge bases, complex (leaves and root) on periodic grids.
# The offset charges 0.21 and 0.09 (here and in `dc_chain3`) are deliberate: a transmon leaf with E_J/E_C = 25 on a charge
# basis changes the peak index of one of its four lowest states at n_g ≈ 0.024, 0.16 and 0.258 (measured), and a finite
# difference across such a change is meaningless.
function dc_two_transmons(; periodic::Bool=false, solver=Dense(), compute::Int=0, keep=lowest(6), leafkeep::Int=4, g::Real=0.3,
                          retain::Symbol=:auto)
    b() = periodic ? PeriodicGrid(64; order=8) : ChargeBasis(30)
    leaves = [Node([Symbol("varphi_$i") => b()]; solver=Dense(), keep=lowest(leafkeep)) for i in 1:2]
    root = Node(leaves; solver, compute, keep, retain, ports=[(:varphi_1, nport()), (:varphi_1, expiphi(1.0))])
    θ = (E_J1=30.0, E_C1=1.2, n_g1=0.21, E_J2=25.0, E_C2=1.0, n_g2=0.09, g=g)
    return (; root, leaves, spec=load_spec(fixture("two_transmons.json")), θ)
end
# Three transmons in a chain, a three-level tree (real): `inner` holds modes 1 and 2, and `g23` couples it to the third.
function dc_chain3()
    leaves = [Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(4)) for i in 1:2]
    inner = Node(leaves; solver=Dense(), keep=lowest(10))
    third = Node([:varphi_3 => ChargeBasis(10)]; solver=Dense(), keep=lowest(4))
    root = Node([inner, third]; solver=Dense(), keep=lowest(6), ports=[(:varphi_1, nport())])
    θ = (E_J1=30.0, E_J2=25.0, E_J3=20.0, g12=0.3, g23=0.2, n_g2=0.09)
    return (; root, inner, third, leaves, spec=KronFreeCircuits.parse_spec(chain3_json), θ)
end
# The zero-pi as a two-level tree: every node is real, but the root's terms carry complex coefficients on paired ports.
# Leaf A holds bare charge states and warns about the phase mask; wrap its materializations in `quiet`.
function dc_zero_pi()
    A = Node([:theta => ChargeBasis(8)]; solver=Dense(), keep=lowest(8))
    B = Node([:phi => PhaseGrid(-12π, 12π, 201; order=6)]; solver=Dense(), compute=10, keep=lowest(10))
    root = Node([A, B]; solver=Dense(), keep=lowest(6), ports=[(:theta, nport()), (:phi, phiport())])
    return (; root, A, B, spec=load_spec(fixture("zero_pi.json")), θ=zero_pi_params())
end
# Three compact modes with one cosine on all three: `ab` is a two-mode leaf, so the root applies its joint ports.
# The root is complex (the cosine's offset), its children real.
function dc_abc()
    ab = Node([:a => ChargeBasis(5), :b => ChargeBasis(5)]; solver=Dense(), keep=lowest(8))
    c = Node([:c => ChargeBasis(5)]; solver=Dense(), keep=lowest(5))
    root = Node([ab, c]; solver=Dense(), keep=lowest(6))
    θ = (E_C=1.0, E_J=6.0, E_J3=0.6, n_ga=0.21, varphi_ext=0.7)
    return (; root, ab, c, spec=KronFreeCircuits.parse_spec(abc_deriv_json), θ)
end
# The cos 2φ element near its parity degeneracy (at n_g = 1/2, E_1 = 0): a doublet with detuning 0.0593 GHz at n_g = 0.49,
# coupled by the parameter E_1 with matrix element 0.7456.
function dc_cos2phi(; n_g::Real=0.49, keep=lowest(4), basis=ChargeBasis(20), E_2::Real=5.0)
    root = Node([:phi => basis]; solver=Dense(), keep, ports=[(:phi, expiphi(1.0))])
    return (; root, spec=load_spec(fixture("cos2phi.json")), θ=(E_C=1.0, n_g=n_g, E_1=0.0, E_2=E_2))
end

# A circuit solved at θ with logging silenced, and the same memoized on θ: the finite differences of many quantities of
# one circuit then share one set of solves, which is what keeps the tree tests fast.
dc_solve(c, θ=c.θ) = quiet(() -> materialize(c.root, c.spec, θ))
function dc_memo(c)
    cache = Dict{Any,Any}()
    return θ -> get!(() -> dc_solve(c, θ), cache, θ)
end

# The derivative data of the node `child` of circuit `c` by finite differences, holding only what the operator builder
# reads (energies, ports, joint ports; zero vectors, gauge and peak amplitudes; no children).  `at` is `dc_memo(c)`.
# Three arguments: the first derivative along `x`.  Four: the second derivative along `x` and `λ` (`fd_pair`).
function fd_child(at, c, child, x::Symbol; h)
    return _fd_child(at, c, child, f -> first(fd_derivative(f, c.θ, x; h)))
end
function fd_child(at, c, child, x::Symbol, λ::Symbol; hx, hλ=hx)
    return _fd_child(at, c, child, f -> first(fd_pair(f, c.θ, x, λ; hx, hλ)))
end
function _fd_child(at, c, child, fd)
    K = KronFreeCircuits
    m = K.materialized(at(c.θ), child)
    ports = Dict{Tuple{Symbol,K.Port},K.NodeMatrix}(k => fd(θ -> K.materialized(at(θ), child).ports[k]) for k in keys(m.ports))
    joint = Dict{Vector{Tuple{Symbol,K.Port}},K.NodeMatrix}(k => fd(θ -> K.materialized(at(θ), child).joint_ports[k]) for k in keys(m.joint_ports))
    return K.NodeDerivative{eltype(m.vectors)}(fd(θ -> K.materialized(at(θ), child).energies), zero(m.vectors), zeros(length(m.kept)),
                                               ports, joint, zero(m.peak_amplitudes), Any[nothing for _ in m.children])
end
```

How `_derivative_operator` builds `∂H/∂x`.
For each term `t` with coefficient `c`, let `ops = _axis_ops(mn.node, mn.children, _factor_key(t))` (one operator per child axis the term touches, a port or a joint port).
The term contributes these variants, each a coefficient value and a list of operators:

1. `(∂c/∂x)(θ)` on `ops` (part `:coefficients`), when that value is not zero; whether `c(θ)` itself is zero plays no role;
2. for each entry `(a, op, Ka)` of `ops` whose child has derivative data (`dx[a] !== nothing`): `c(θ)` on `ops` with that entry's operator replaced by `_derivative_op(dx[a], Ka)` (part `:ports`), when `c(θ)` is not zero.

On a complex node a variant is `AxisTerm(ComplexF64(value), [(a, op), …])`.
On a real node it is the real expansion of `Re(value ⊗ ops)`: `_real_expansion(value, [(a, op, kind, 1) …])` with `kind` the character kind of each entry's sub-key (`_key_character(_mode_leaves(mn.node), Ka).kind`), one `AxisTerm(r.coeff, r.factors)` per returned term; nothing is merged across terms and nothing else is dropped.
A child's derivative of a port has the character of the port, so the same expansion applies to a replaced operator.
Then (part `:energies`) for every child with derivative data, `AxisTerm(1.0, [(a, DiagonalOp(dx[a].energies))])`.
The operator's `constant` is `differentiate(constant, x)(θ)` (its real part on a real node).
Never derive this operator from the forward operator's stored terms: the forward real operator drops terms whose coefficient is zero at `θ`.

`_sandwich` on a complex node converts real dense operators to complex and calls `_port_matrix`-style application with `W'` on the left.
On a real node it forms the real and imaginary components separately, as `_joint_port` does (`_real_expansion(1.0, fs)` and `_real_expansion(-im, fs)` with `s = 1`), and returns them in the key's natural type.

- [ ] **Step 1: Add the circuits to `test/helpers.jl` and write the failing tests** in `test/test_derivoperator.jl`

```julia
using Test, LinearAlgebra, Random, KronFreeCircuits
using KronFreeCircuits: _derivative_operator, _sandwich, _axis_ops, operator, route, materialized

@testset "_sandwich reproduces every stored port and joint port, and is two-sided" begin
    rng = MersenneTwister(3)
    for c in (dc_two_transmons(periodic=true), dc_zero_pi(), dc_abc(), dc_chain3())
        sys = dc_solve(c)
        for node in KronFreeCircuits._all_nodes(c.root)
            mn = materialized(sys, node); V = mn.vectors; k = size(V, 2)
            Q = Matrix(qr(randn(rng, eltype(V), k, k)).Q)
            for (K, P) in vcat([[key] => P for (key, P) in mn.ports], collect(mn.joint_ports))
                ops = _axis_ops(node, mn.children, K)
                S = _sandwich(mn, V, V, K, ops)
                @test typeof(S) === typeof(P) && S ≈ P atol=1e-12
                @test _sandwich(mn, V * Q, V, K, ops) ≈ Q' * P atol=1e-11          # a different left argument
                @test _sandwich(mn, V, V * Q, K, ops) ≈ P * Q atol=1e-11           # a different right argument
            end
        end
    end
end

@testset "∂H/∂x at a leaf equals the finite difference of the forward operator" begin
    for (c, xs) in ((dc_transmon(), (:n_g, :E_J, :E_C)), (dc_transmon(periodic=true), (:n_g, :E_J)),
                    (dc_fluxonium(), (:varphi_ext, :E_L, :E_J)))
        at = dc_memo(c); sys = at(c.θ)
        for x in xs
            H1 = _derivative_operator(sys.root, route(c.root, sys.terms)[c.root], c.θ, x, Any[nothing]; constant=sys.constant)
            @test eltype(H1) === eltype(operator(sys))
            check_fd(Matrix(H1), fd_derivative(θ -> Matrix(operator(at(θ))), c.θ, x; h=1e-3); rtol=1e-8, atol=1e-8)
        end
    end
    # Review Focus 1: at varphi_ext = 0 the flux term's coefficient E_L*varphi_ext vanishes and the forward operator has
    # no such term; its derivative is E_L φ.
    c = dc_fluxonium(varphi_ext=0.0); sys = dc_solve(c)
    H1 = Matrix(_derivative_operator(sys.root, route(c.root, sys.terms)[c.root], c.θ, :varphi_ext, Any[nothing]; constant=sys.constant))
    @test H1 ≈ c.θ.E_L * Diagonal(KronFreeCircuits.coordinates(PhaseGrid(-10π, 10π, 601; order=8))) atol=1e-12
    @test norm(H1) > 1
end

@testset "∂H/∂x at a branch: coefficients, the children's ports and their energies" begin
    for (c, xs) in ((dc_two_transmons(), (:g, :n_g1, :E_J1, :E_J2)), (dc_two_transmons(periodic=true), (:g, :n_g1)),
                    (dc_zero_pi(), (:varphi_ext, :n_g, :E_J)), (dc_abc(), (:varphi_ext, :E_J3, :n_ga, :E_J)))
        at = dc_memo(c); sys = at(c.θ)
        terms = route(c.root, sys.terms)[c.root]
        for x in xs
            dx = Any[fd_child(at, c, ch, x; h=1e-3) for ch in c.root.children]
            H1 = _derivative_operator(sys.root, terms, c.θ, x, dx; constant=sys.constant)
            @test eltype(H1) === eltype(operator(sys))
            check_fd(Matrix(H1), fd_derivative(θ -> Matrix(operator(at(θ))), c.θ, x; h=1e-3); rtol=1e-6, atol=1e-7)
        end
    end
    # each contribution matters on this circuit
    c = dc_two_transmons(); at = dc_memo(c); sys = at(c.θ); terms = route(c.root, sys.terms)[c.root]
    op(x, parts) = Matrix(_derivative_operator(sys.root, terms, c.θ, x, Any[fd_child(at, c, ch, x; h=1e-3) for ch in c.root.children];
                                               constant=sys.constant, parts))
    all3 = (:coefficients, :ports, :energies)
    @test norm(op(:E_J2, all3) - op(:E_J2, (:coefficients, :ports))) > 0.1          # the children's energy derivatives
    @test norm(op(:E_J1, all3) - op(:E_J1, (:coefficients, :energies))) > 1e-3      # the children's port derivatives
    @test norm(op(:g, all3) - op(:g, (:ports, :energies))) > 0.1                    # the coefficients' derivatives
end
```

- [ ] **Step 2: Run and confirm failure** (`UndefVarError: NodeDerivative`).

- [ ] **Step 3: Implement** `NodeDerivative`, `_derivative_op`, `_derivative_operator` (first-order form) and `_sandwich` in `src/derivoperator.jl`.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivoperator.jl"])`, then the default tier. Expected: PASS.

- [ ] **Step 5: Commit** (`The derivative operator of a node, and two-sided ports`), with `src/derivoperator.jl src/KronFreeCircuits.jl test/helpers.jl test/test_derivoperator.jl`.

---

### Task 4: The first-order rule at one node

**Files:**
- Create: `src/resolvent.jl`, `src/derivatives.jl` (both included after `derivoperator.jl`), `bench/derivative_sums.jl`
- Modify: `test/helpers.jl` (`node_setup`, `node_first_order`)
- Test: `test/test_derivatives_node.jl` (fast tier; estimated 20 s, most of it compilation)

**Interfaces:**
- Consumes: `NodeDerivative`, `_derivative_operator`, `_sandwich`, `_derivative_op` (Task 3); `Retained` (Task 2); the `dc_*` circuits and finite-difference helpers; existing `_contract_rows`, `_resolve_node_solver`, `operator`.
- Produces, internal:

```julia
# How (s − H)⁻¹ is applied at one node, away from chosen kept states.
struct Resolvent{T<:Union{Float64,ComplexF64}}
    energies::Vector{Float64}          # the kept energies
    vectors::Matrix{T}                 # the kept vectors, as stored (phase-fixed)
    other_energies::Vector{Float64}    # the other known eigenpairs: the retained unkept ones, or those of a re-diagonalization
    other_vectors::Matrix{T}
    complement::Any                    # `nothing` when kept ∪ other span the whole space of the node's eigenproblem;
                                       # otherwise the linear solver of Task 6
    degenerate_tol::Float64
end

# `H` is the node's forward operator, needed only when the node lacks its complete spectrum.
# `projected = true` declares the kept states the whole space (an augmented node, Task 10).
_resolvent(mn::MaterializedNode, H::Union{Nothing,KronFreeOperator} = nothing; degenerate_tol::Real = 1e-9,
           projected::Bool = false) -> Resolvent

# W[:, j] = Σ_l ψ_l ⟨ψ_l|R[:, j]⟩ / (shifts[j] − E_l), over every state l of the node's space except the kept states
# (column indices of the node's vectors) listed in exclude[j].
# clashes[j] lists the states that make column j undefined (degenerate with shifts[j], with a nonzero overlap): a kept
# state by its column index, any other state as 0.  Column j is defined exactly when clashes[j] is empty; an undefined
# column of W is NaN.
_apply_resolvent(res::Resolvent, R::AbstractMatrix, shifts::Vector{Float64}, exclude::Vector{Vector{Int}})
    -> (W::Matrix, clashes::Vector{Vector{Int}})

# The node's peak functionals applied to the columns of X: out[p, j] = Σ X[(i₁,…,iₙ), j] · amps[1][c₁(p), i₁] ⋯ amps[n][cₙ(p), iₙ],
# with (c₁(p), …) the product-basis index `mn.peak_component[p]` of kept state p's peak; `amps[a] === nothing` is the identity.
_peak_map(mn::MaterializedNode, X::AbstractMatrix, amps::Vector{Any}) -> Matrix
_peak_amps(mn::MaterializedNode) -> Vector{Any}     # per axis: `nothing` for a basis child, else the child's `peak_amplitudes`

# The first derivative of every port and joint port of `mn`, given the kept vectors' derivative V1.
_port_derivatives(mn::MaterializedNode, V1::AbstractMatrix, dx::Vector{Any}) -> (ports::Dict, joint_ports::Dict)

# The per-node rule.  `H1` is ∂H/∂x on any storage: a block is moved to the operator's storage for each application
# (`Array(H1 * adapt(H1.on, X))`) and every result is a host array.  `dx` is the children's derivative data, as given to
# `_derivative_operator`.
# The result's `children` field is `dx` itself, so the tree of results mirrors the materialized tree.
_node_first_order(mn::MaterializedNode, H1::KronFreeOperator, res::Resolvent, dx::Vector{Any}) -> NodeDerivative
```

- Produces, in `test/helpers.jl` (used by Tasks 4, 7 and the bench script):

```julia
# What the per-node rules need at the root of a one-node system: the node, "no child data", its terms, its resolvent.
function node_setup(sys)
    K = KronFreeCircuits
    mn = sys.root
    res = K._resolvent(mn, mn.retained.complete ? nothing : K.operator(sys))
    return (; mn, none=Any[nothing for _ in mn.children], terms=K.route(mn.node, sys.terms)[mn.node], res)
end
# The first-order rule on a one-node system.
function node_first_order(sys, x::Symbol; setup=node_setup(sys))
    K = KronFreeCircuits
    H1 = K._derivative_operator(setup.mn, setup.terms, sys.θ, x, setup.none; constant=sys.constant)
    return K._node_first_order(setup.mn, H1, setup.res, setup.none)
end
```

The equations.
`V` are the kept vectors (columns `ψ_j`), `E` the kept energies, `A = mn.peak_amplitudes`, and `H1 = ∂H/∂x`.

1. `R = H1 V`, `C = V† R`, and `∂E_j = Re C[j, j]` (the paper's `eq:eig_val_grad`).
2. `W, clashes = _apply_resolvent(res, R, E, [[j] for j])`: `w_j = Σ_{l ≠ j} ψ_l ⟨ψ_l|H1|ψ_j⟩ / (E_j − E_l)` over all states of the node's space, so `(H − E_j) w_j = −(H1 − ∂E_j) ψ_j` and `w_j ⟂ ψ_j` (the paper's `eq:eig_vec_grad`, with the sum complete).
3. The gauge.
   The phase convention makes `ℓ_j · ψ_j` real and positive, where `ℓ_j · X = _peak_map(mn, X, _peak_amps(mn))[j, :]` (at a leaf it is the entry of `X` at the state's peak index; at a branch the contraction with the children's peak amplitudes).
   Its derivative must be real: `a_j = −Im( L1[j, j] + L0[j, j] ) / Re A[j, j]`, with `L0 = _peak_map(mn, W, _peak_amps(mn))` and `L1` the sum, over the axes `k` with `dx[k] !== nothing`, of `_peak_map(mn, V, amps)` with `amps[k]` replaced by `dx[k].peak_amplitudes`.
   (`a_j` is the gauge of state `j`; axes are numbered `k` here to keep the two apart.)
   On a real node `a_j = 0` and is not computed.
4. `∂ψ_j = w_j + i a_j ψ_j`; `gauge = a`.
5. `∂A = _peak_map(mn, ∂V, _peak_amps(mn)) + L1`.
6. Each port and joint port `P = V† O V` on a key `K`, with `ops = _axis_ops(mn.node, mn.children, K)`:
   `∂P = _sandwich(mn, ∂V, V, K, ops) + _sandwich(mn, V, ∂V, K, ops) + Σ_g _sandwich(mn, V, V, K, ops with entry g replaced by _derivative_op(dx[k_g], K_g))`, the last sum over the entries `g = (k_g, op_g, K_g)` whose child has derivative data.
   `O` need not be Hermitian, so the first two terms are computed separately.
7. A column with a clash gets `∂E_j = NaN` and `∂ψ_j = NaN` (and `NaN` flows into the ports and peak amplitudes).
   One `@warn` per call of the rule, whose text contains the word "degenerate", names the node's modes, the undefined states and the states they clash with.
   (From Task 9 on, every such clash is also reported as a block, or as a boundary record when the partner is an unkept state, so a `NaN` never appears without a report.)

`_apply_resolvent`, for column `j` and each state `l` it sums (kept states not in `exclude[j]`, then the other states): when `|shifts[j] − E_l| ≤ degenerate_tol`, the term is skipped if `|⟨ψ_l|R[:, j]⟩| ≤ 1e-10 ‖R[:, j]‖`, and otherwise `l` is a clash of column `j`; every other term is included whatever the size of its detuning.
This per-column rule is the plan's one definition of a coupling that is zero: the overlap of state `l` with the column being resolved, against `1e-10` times that column's norm.
Tasks 6 and 9 use the same rule and no other.
With `complement === nothing` nothing else is added.

`_resolvent`: when `projected`, or `mn.retained.complete`, the other states are the retained ones and `complement` is `nothing`.
Otherwise, when the node's solver (`_resolve_node_solver(mn.node, mn.children)`) is `Dense`, the node is re-diagonalized from `H` (`eigen(Hermitian(Matrix(H)))`), the other states are the eigenpairs at the positions not in `mn.kept`, and an `@info` whose text contains "re-diagonalized" says so, naming the node's modes and dimension and the remedy `retain = :always`.
Otherwise, until Task 6, it throws an `ArgumentError` whose text contains "linear solve".

- [ ] **Step 1: Write the failing tests** in `test/test_derivatives_node.jl`

```julia
using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: _derivative_operator, _peak_map, _peak_amps, route, materialized

# energies, vectors, both ports and the peak amplitudes against finite differences
function check_node(c, x; h=1e-3, rtol=1e-6)
    at = dc_memo(c); sys = at(c.θ); d = node_first_order(sys, x)
    @test eltype(d.vectors) === eltype(eigenvectors(sys)) && eltype(d.peak_amplitudes) === eltype(d.vectors)
    @test d.children == Any[nothing]
    check_fd(d.energies, fd_derivative(θ -> energies(at(θ)), c.θ, x; h); rtol)                   # absolute energies: Review Focus 2
    check_fd(d.vectors, fd_derivative(θ -> eigenvectors(at(θ)), c.θ, x; h); rtol)
    check_fd(d.peak_amplitudes, fd_derivative(θ -> at(θ).root.peak_amplitudes, c.θ, x; h); rtol)
    for (key, P) in sys.root.ports                                                               # off-diagonals of expiphi: Review Focus 3
        @test eltype(d.ports[key]) === eltype(P)
        check_fd(d.ports[key], fd_derivative(θ -> at(θ).root.ports[key], c.θ, x; h); rtol)
    end
    return d
end

@testset "first order at a dense leaf against finite differences" begin
    for x in (:n_g, :E_J, :E_C); check_node(dc_transmon(), x); end                               # real
    for x in (:n_g, :E_J); d = check_node(dc_transmon(periodic=true), x)                         # complex
        x === :n_g && @test maximum(abs, d.gauge) > 0.1                                          # the gauge term is really there
    end
    for x in (:varphi_ext, :E_J, :E_L); check_node(dc_fluxonium(), x); end
    @test all(iszero, check_node(dc_transmon(), :E_J).gauge)
    # Review Focus 4: the kept states are not the lowest computed ones
    for x in (:n_g, :E_J); check_node(dc_transmon(compute=8, keep=indices([1, 3, 4])), x); end
    for x in (:n_g, :E_J); check_node(dc_transmon(periodic=true, compute=8, keep=indices([1, 3, 4])), x); end
    # a dense node that did not retain is re-diagonalized, with a notice, to the same result
    a = node_first_order(dc_solve(dc_transmon(compute=8)), :n_g)
    sn = dc_solve(dc_transmon(compute=8, retain=:never))
    b = @test_logs (:info, r"re-diagonalized") match_mode=:any node_first_order(sn, :n_g)
    @test b.energies ≈ a.energies rtol=1e-10 && b.vectors ≈ a.vectors rtol=1e-9
end

@testset "the peak map reproduces the stored peak amplitudes" begin
    for c in (dc_transmon(periodic=true), dc_chain3(), dc_abc(), dc_two_transmons(periodic=true))
        sys = dc_solve(c)
        for node in KronFreeCircuits._all_nodes(c.root)
            mn = materialized(sys, node)
            @test _peak_map(mn, mn.vectors, _peak_amps(mn)) ≈ mn.peak_amplitudes atol=1e-12
        end
    end
end

@testset "first order reduces to eq:eig_val_grad and eq:eig_vec_grad of arXiv:2408.12704v3" begin
    # ∂f_i/∂x = ⟨f_i|∂H/∂x|f_i⟩ and ∂|f_i⟩/∂x = Σ_{m≠i} ⟨f_m|∂H/∂x|f_i⟩ / (f_i − f_m) |f_m⟩, the sum over all states of a dense
    # node.  On a complex node the rule adds i a_i |f_i⟩, the phase-convention term the paper's formula sets to zero.
    spec = load_spec(fixture("transmon.json")); θ = (E_J=20.0, E_C=1.0, n_g=0.2)
    for (basis, x, iscomplex) in ((ChargeBasis(8), :E_J, false), (PeriodicGrid(16; order=8), :n_g, true))
        leaf = Node([:varphi_1 => basis]; solver=Dense())                     # every state kept
        sys = quiet(() -> materialize(leaf, spec, θ)); d = node_first_order(sys, x)
        E, V = energies(sys), eigenvectors(sys)
        H1 = Matrix(_derivative_operator(sys.root, route(leaf, sys.terms)[leaf], θ, x, Any[nothing]; constant=sys.constant))
        @test d.energies ≈ [real(V[:, i]' * H1 * V[:, i]) for i in eachindex(E)] rtol=1e-12
        paper = reduce(hcat, [sum(V[:, m] * (V[:, m]' * H1 * V[:, i]) / (E[i] - E[m]) for m in eachindex(E) if m != i) for i in eachindex(E)])
        along = [V[:, i]' * d.vectors[:, i] for i in eachindex(E)]
        @test d.vectors - V * Diagonal(along) ≈ paper rtol=1e-9
        @test maximum(abs, real.(along)) < 1e-12 && imag.(along) ≈ d.gauge atol=1e-12
        @test iscomplex ? maximum(abs, d.gauge) > 0.1 : all(iszero, d.gauge)
    end
end

@testset "analytic checks at single nodes" begin
    # oscillator: E_k = (k + 1/2) sqrt(8 E_C E_L), so ∂E_k/∂E_L = (k + 1/2) sqrt(2 E_C / E_L) and ∂E_k/∂E_C = (k + 1/2) sqrt(2 E_L / E_C)
    c = dc_oscillator(); k = 0:5
    @test node_first_order(dc_solve(c), :E_L).energies ≈ (k .+ 0.5) .* sqrt(2 * c.θ.E_C / c.θ.E_L) rtol=1e-10
    @test node_first_order(dc_solve(c), :E_C).energies ≈ (k .+ 0.5) .* sqrt(2 * c.θ.E_L / c.θ.E_C) rtol=1e-10
    # transmon charge dispersion: E_m(n_g) = Ē_m − (ε_m/2) cos(2π n_g) to tight-binding accuracy, with ε_m = E_m(1/2) − E_m(0),
    # so ∂E_m/∂n_g at n_g = 1/4 is π ε_m; at n_g = 0 it vanishes by symmetry
    tm(ng) = dc_transmon(θ=(E_J=30.0, E_C=1.2, n_g=ng))
    ε = energies(dc_solve(tm(0.5))) - energies(dc_solve(tm(0.0)))
    dq = node_first_order(dc_solve(tm(0.25)), :n_g).energies
    @test dq[1] ≈ π * ε[1] rtol=1e-6
    @test dq[2] ≈ π * ε[2] rtol=1e-5
    @test maximum(abs, node_first_order(dc_solve(tm(0.0)), :n_g).energies[1:2]) < 1e-9
    # fluxonium: at varphi_ext = π every level is stationary by symmetry; with E_J = 0 the flux only displaces an oscillator
    @test maximum(abs, node_first_order(dc_solve(dc_fluxonium(varphi_ext=π)), :varphi_ext).energies) < 1e-7
    @test maximum(abs, node_first_order(dc_solve(dc_fluxonium(varphi_ext=1.0, E_J=0.0)), :varphi_ext).energies) < 1e-7
end

@testset "exact degeneracy: undefined with a coupling, skipped without one" begin
    # 4 E_C n² on the charges −1, 0, 1 with every cosine off: energies 0, 4, 4, the pair |−1⟩, |+1⟩ exactly degenerate
    c = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(0), E_2=0.0)
    sys = dc_solve(c)
    @test energies(sys) ≈ [0.0, 4.0, 4.0] atol=1e-14
    d2 = @test_logs (:warn, r"degenerate") match_mode=:any node_first_order(sys, :E_2)    # −cos 2φ couples the pair directly
    @test isnan.(d2.energies) == [false, true, true] && all(isnan, d2.vectors[:, 2:3]) && all(isfinite, d2.vectors[:, 1])
    dn = node_first_order(sys, :n_g)                                                      # −8 E_C n is diagonal: no coupling
    @test all(isfinite, dn.vectors) && sort(dn.energies) ≈ [-8.0, 0.0, 8.0] atol=1e-12
    d1 = node_first_order(sys, :E_1)                                                      # −cos φ couples the pair only through |0⟩
    @test all(isfinite, d1.vectors) && maximum(abs, d1.energies) < 1e-12
end
```

- [ ] **Step 2: Run and confirm failure** (`UndefVarError: _resolvent`).

- [ ] **Step 3: Implement** `Resolvent`, `_resolvent`, `_apply_resolvent` in `src/resolvent.jl`, and `_peak_map`, `_peak_amps`, `_port_derivatives`, `_node_first_order` in `src/derivatives.jl`, by the equations above.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivatives_node.jl"])`, then the default tier. Expected: PASS.

- [ ] **Step 5: Measure how fast the sum over states converges** (the design's §4.5 note asks for this before the retention default is taken as final; see Decision 2).
Write `bench/derivative_sums.jl`: for `dc_fluxonium()` wrt `:varphi_ext` and `:E_J`, and `dc_transmon()` wrt `:n_g`, it runs `_node_first_order` with a `Resolvent` whose other states are cut to the nearest `M` above the kept ones (`M` = 0, 5, 10, 20, 40, 80, 160, all; build the cut `Resolvent` by hand from the full one's fields) and prints, per `M`, the largest relative error of the state derivatives and of each port derivative against the full sum.
It loads `Test`, `LinearAlgebra` and `KronFreeCircuits` and then includes `test/helpers.jl` for the circuits (the helpers use `@test`).
Run: `$HOME/.juliaup/bin/julia --project bench/derivative_sums.jl`.
Put the table in the commit message body; Task 15 copies it into the architecture spec.
This step changes no default.

- [ ] **Step 6: Commit** (`First-order derivatives at one node: resolvent, gauge term, ports`), with `src/resolvent.jl src/derivatives.jl src/KronFreeCircuits.jl test/helpers.jl test/test_derivatives_node.jl bench/derivative_sums.jl`.

---

### Task 5: First order through the whole tree, and the API

**Files:**
- Modify: `src/derivatives.jl`, `src/KronFreeCircuits.jl` (export `derivatives`, `Derivatives`)
- Test: `test/test_derivatives_tree.jl` (fast tier; estimated 25 s, most of it compilation: the finite differences share memoized solves)

**Interfaces:**
- Consumes: `_derivative_operator` (with `parts`), `_resolvent`, `_node_first_order` (whose result's `children` is the children's data it was given), `NodeDerivative` (Tasks 3, 4); `dc_memo`, `dc_solve` (Task 3); `route`, `materialized`, `operator`, `System`, `Retained.followed`.
- Produces, exported:

```julia
struct Derivatives
    sys::System
    wrt::Vector{Symbol}
    order::Int
    mixed::Vector{Tuple{Symbol,Symbol}}
    first::Dict{Symbol,Any}                    # parameter => the root's NodeDerivative (children nested), or `nothing`
    second::Dict{Tuple{Symbol,Symbol},Any}     # `_pairkey(x, λ)` => the root's second-order NodeDerivative, or `nothing` (Task 8)
    blocks::Vector{Any}                        # near-degenerate blocks (Task 9)
    boundary::Vector{Any}                      # selection-boundary records (Task 6)
    linear_solves::Vector{Any}                 # one record per linear solve (Task 6)
end

derivatives(sys::System; wrt, order::Int = 1, mixed = (), scale::NamedTuple = (;), r::Real = 0.1,
            degenerate_tol::Real = 1e-9, lintol::Real = 1e-10, linmaxiter::Int = 5000,
            given_independent::Bool = false) -> Derivatives

# New methods of the existing exported generics; `node` defaults to the root, as for a `System`.
energies(d::Derivatives, x::Symbol, node::Node = d.sys.root.node) -> Vector{Float64}
eigenvectors(d::Derivatives, x::Symbol, node::Node = d.sys.root.node) -> Matrix
port(d::Derivatives, x::Symbol, mode::Symbol, p::Port, node::Node = d.sys.root.node) -> Matrix
port(d::Derivatives, x::Symbol, key::AbstractVector{Tuple{Symbol,Port}}, node::Node = d.sys.root.node) -> Matrix
```

- Produces, internal: `_pairkey(x::Symbol, λ::Symbol) = x <= λ ? (x, λ) : (λ, x)`; `_node_derivative(root_data, sys::System, node::Node)`, which walks the materialized tree and the derivative tree together and returns the node's `NodeDerivative` or `nothing`; and the internal keyword `_parts` of `derivatives`, passed to every `_derivative_operator` call, which exists for the three-contribution test and is not documented.

Behaviour:
- The pass for a parameter `x` visits the materialized tree children first.
  At a node it gathers its children's data (`nothing` for a basis child), builds `∂H/∂x` from the terms `route(sys.root.node, sys.terms)` places at the node (with `sys.constant` at the root only, and `Coefficient(0)` below it), and calls `_node_first_order`.
  It uses `sys.terms`, `sys.constant` and `sys.θ`; it does not re-solve anything.
- A node none of whose terms, in its whole subtree (and, at the root, the constant), has `x` among its coefficient's `deps` is not visited: its data is `nothing`, and every accessor returns zeros of the stored quantity's size and element type for it.
- One `Resolvent` per node is built on first use and reused for every parameter of the call; the node's forward operator is rebuilt (as `operator(sys, node)` does) only when `_resolvent` needs it.
- Parameters are processed one at a time, in the order of `wrt`.
- Validation, each an `ArgumentError` that names the offender: `wrt` is a `Symbol` or a collection of `Symbol`s (duplicates removed), each a key of `sys.θ` (the message lists the keys); `order` is 1 (2 arrives in Task 8, and `mixed` must be empty until then); the keys of `scale` are in `wrt` and its values positive and finite; `r > 0`; a node whose `retained.followed` is true (the message names its modes and contains "re-materialize at this point without `previous`"); a tree with a `GivenSystem` child or an augmented child (until Task 10).
- An accessor asked for a parameter not in `wrt`, or for a port or joint port the node did not form, throws an `ArgumentError` listing what is available (as `port(sys, …)` does).
- `energies(d, x)` at the root includes the derivative of the constant; a child's does not.
- Docstrings for `derivatives`, `Derivatives` and the new methods state: what is held fixed (selection, bases, solver, arithmetic, peak indices), that derivatives are those of the calculation as performed, the element types (those of the stored quantities), and the meaning of every keyword.

- [ ] **Step 1: Write the failing tests** in `test/test_derivatives_tree.jl`

```julia
using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: materialized, _all_nodes

# every node's energies, vectors, ports and joint ports against finite differences of `materialize`
function check_tree(c, x; h=1e-3, rtol=1e-6, atol=1e-9, kwargs...)
    at = dc_memo(c); sys = at(c.θ); d = derivatives(sys; wrt=x, kwargs...)
    for node in _all_nodes(c.root)
        mn = materialized(sys, node)
        check_fd(energies(d, x, node), fd_derivative(θ -> energies(at(θ), node), c.θ, x; h); rtol, atol)
        check_fd(eigenvectors(d, x, node), fd_derivative(θ -> eigenvectors(at(θ), node), c.θ, x; h); rtol, atol)
        for (mode, p) in keys(mn.ports)
            @test eltype(port(d, x, mode, p, node)) === eltype(mn.ports[(mode, p)])
            check_fd(port(d, x, mode, p, node), fd_derivative(θ -> port(at(θ), mode, p, node), c.θ, x; h); rtol, atol)
        end
        for K in keys(mn.joint_ports)
            check_fd(port(d, x, K, node), fd_derivative(θ -> port(at(θ), K, node), c.θ, x; h); rtol, atol)
        end
    end
    return d
end

@testset "two-level trees against finite differences" begin
    for x in (:g, :n_g1, :E_J2, :E_C1); check_tree(dc_two_transmons(), x); end                 # real
    for x in (:g, :n_g1); check_tree(dc_two_transmons(periodic=true), x); end                  # complex leaves and root
    for x in (:varphi_ext, :n_g, :E_J); check_tree(dc_zero_pi(), x); end                       # real, complex coefficients on paired ports
    check_tree(dc_zero_pi(), :E_L; h=2.5e-4)                                                   # E_L = 0.04: a step of 1e-3 is too coarse (measured)
    for x in (:varphi_ext, :E_J3, :n_ga, :E_J); check_tree(dc_abc(), x); end                   # joint ports; complex root on real children
    @test maximum(abs, derivatives(dc_solve(dc_abc()); wrt=:varphi_ext).first[:varphi_ext].gauge) > 1e-3
end

@testset "a three-level tree against finite differences" begin
    for x in (:g12, :g23, :E_J1, :n_g2); check_tree(dc_chain3(), x); end
end

@testset "Review Focus 1: a coupling switched off at θ still has a derivative" begin
    c = dc_two_transmons(g=0.0)
    d = check_tree(c, :g)
    @test maximum(abs, eigenvectors(d, :g)) > 0.5                                # the root states do move with g
    @test maximum(abs, energies(d, :g, c.leaves[1])) > 0.01                      # and so do the leaves, through -g n_g2 n_1
    # (the root energies' own derivative is of order 1e-6 here; `check_tree`'s atol = 1e-9 covers its finite difference)
end

@testset "the three contributions" begin
    c = dc_chain3(); sys = dc_solve(c)
    fd = first(fd_derivative(θ -> energies(dc_solve(c, θ)), c.θ, :E_J1; h=1e-3))
    @test norm(fd, Inf) > 0.5
    # E_J1 enters only inside a leaf: with coefficients alone, every node above the leaf sees nothing
    @test all(iszero, energies(derivatives(sys; wrt=:E_J1, _parts=(:coefficients,)), :E_J1))
    # without the children's energy derivatives
    @test norm(energies(derivatives(sys; wrt=:E_J1, _parts=(:coefficients, :ports)), :E_J1) - fd, Inf) > 0.1
    # without the children's port derivatives (n_g1 moves the leaf's charge operator)
    c2 = dc_two_transmons(); s2 = dc_solve(c2)
    fd2 = first(fd_derivative(θ -> energies(dc_solve(c2, θ)), c2.θ, :n_g1; h=1e-3))
    @test norm(energies(derivatives(s2; wrt=:n_g1, _parts=(:coefficients, :energies)), :n_g1) - fd2, Inf) > 1e-2
    @test norm(energies(derivatives(s2; wrt=:n_g1), :n_g1) - fd2, Inf) < 1e-7
end

@testset "the API" begin
    c = dc_chain3(); sys = dc_solve(c)
    d = derivatives(sys; wrt=(:g12, :E_J1))
    @test d isa Derivatives && d.wrt == [:g12, :E_J1] && d.order == 1
    @test derivatives(sys; wrt=[:g12, :E_J1, :g12]).wrt == [:g12, :E_J1]
    @test energies(derivatives(sys; wrt=:g12), :g12) == energies(d, :g12)
    # Review Focus 2: the constant 4 n_gᵀ E_C n_g moves with n_g2 by 8 (1.0·0.09 + (0.3/8)·0.21 + (0.2/8)·0.1) = 0.803 here.
    # `check_tree(dc_chain3(), :n_g2)` above compares absolute energies at every node; here the pass's root derivative is
    # shown to be the derivative of the constant-free energies plus exactly that shift, and the children's to carry none.
    at = dc_memo(c); dn = derivatives(sys; wrt=:n_g2)
    bare(θ) = (t = at(θ); energies(t) .- real(t.constant(t.θ)))
    check_fd(energies(dn, :n_g2) .- 0.803, fd_derivative(bare, c.θ, :n_g2; h=1e-3); rtol=1e-6)
    check_fd(energies(dn, :n_g2, c.inner), fd_derivative(θ -> energies(at(θ), c.inner), c.θ, :n_g2; h=1e-3); rtol=1e-6)
    # a subtree with no dependence on the parameter reads as zeros of the right shape
    m3 = materialized(sys, c.third)
    @test all(iszero, energies(d, :g12, c.third)) && size(eigenvectors(d, :g12, c.third)) == size(m3.vectors)
    @test all(iszero, port(d, :g12, :varphi_3, nport(), c.third)) && d.first[:g12].children[2] === nothing
    @test_throws ArgumentError energies(d, :g23)                                 # not in `wrt`
    @test_throws ArgumentError port(d, :g12, :varphi_2, n2port())                # not formed
    @test_throws ArgumentError derivatives(sys; wrt=:nope)
    @test_throws ArgumentError derivatives(sys; wrt=:g12, scale=(E_J1=0.1,))
    @test_throws ArgumentError derivatives(sys; wrt=:g12, scale=(g12=-1.0,))
    @test_throws ArgumentError derivatives(sys; wrt=:g12, r=0.0)
    # a node whose phases followed a previous solution cannot be differentiated; the same node solved fresh can
    spec = load_spec(fixture("transmon.json")); θ(ng) = (E_J=30.0, E_C=1.2, n_g=ng)
    leaf = Node([:varphi_1 => ChargeBasis(30)]; solver=Dense(), keep=lowest(3), phase=:follow)
    s0 = materialize(leaf, spec, θ(0.20))
    @test_throws "previous" derivatives(materialize(leaf, spec, θ(0.21); previous=s0); wrt=:n_g)
    @test all(isfinite, energies(derivatives(s0; wrt=:n_g), :n_g))
end
```

- [ ] **Step 2: Run and confirm failure** (`UndefVarError: derivatives`).

- [ ] **Step 3: Implement** the tree pass, `Derivatives`, `derivatives` and the accessors in `src/derivatives.jl`; export `derivatives` and `Derivatives`.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivatives_tree.jl","test_derivatives_node.jl"])`, then the default tier. Expected: PASS.

- [ ] **Step 5: Commit** (`derivatives(sys; wrt): first order through the tree`), with `src/derivatives.jl src/KronFreeCircuits.jl test/test_derivatives_tree.jl`.

---

### Task 6: Iterative nodes: the linear solve on the complement, and the selection-boundary warning

**Files:**
- Modify: `src/resolvent.jl` (`_resolvent`, `_apply_resolvent`, new `_complement_solve`), `src/derivatives.jl` (the pass: operators for iterative nodes, the boundary test, the records), `test/runtests.jl` (add `"test_derivatives_iterative.jl"` to `SLOW_FILES`)
- Test: `test/test_derivatives_complement.jl` (fast tier; estimated 15 s: the solver on its own, real and complex, and the boundary warning, with no iterative eigensolver), `test/test_derivatives_iterative.jl` (slow tier: nodes solved by `LOBPCG`, `Lanczos` and `ShiftInvert`, whose first calls compile for 35 to 40 s per element type on this machine)

**Interfaces:**
- Consumes: `Resolvent`, `_resolvent`, `_apply_resolvent` with its `clashes` (Task 4); `derivatives`, `Derivatives.boundary`, `Derivatives.linear_solves` (Task 5); `dc_*`, `dc_solve`, `dc_memo` (Task 3); existing `_node_operator`, `_resolve_node_solver`, `storage`, `with_block_capacity`, `_full_diagonal`, `_colnorms`, `_to_storage`, `adapt`.
- Produces, internal:

```julia
# X[:, j] solves Q (H − shifts[j]) Q x = Q B[:, j] with x ⟂ the columns of `basis`, Q = 1 − basis basis'.
# `basis` (orthonormal columns) and `B` live on H's storage; `D` is H's real diagonal there.
# `residuals[j]` is the relative true residual ‖Q (H − shifts[j]) X[:, j] − Q B[:, j]‖ / ‖Q B[:, j]‖, recomputed with a
# fresh application of H after the loop (0 for a column whose right-hand side vanishes).
# Throws when a column does not converge, or when H − shifts[j] is not positive on the complement.
_complement_solve(H::KronFreeOperator, D::AbstractVector, basis::AbstractMatrix, B::AbstractMatrix, shifts::Vector{Float64};
                  tol::Real, maxiter::Int, eta::Real = 0.01)
    -> (X::AbstractMatrix, iterations::Int, residuals::Vector{Float64})

# `_resolvent` gains: lintol::Real = 1e-10, linmaxiter::Int = 5000, log = nothing (a vector that receives one record per
# solve), and truncate::Bool = false (the sum over the known states alone, for one test; the pass never sets it).
# `derivatives` gains the internal, undocumented keyword `_truncate::Bool = false`, passed to every `_resolvent`.
```

- Records in `Derivatives.linear_solves`: `(; modes::Vector{Symbol}, columns::Int, iterations::Int, residual::Float64)`, with `residual` the largest relative true residual of the solve's columns (`maximum(residuals)`); a solve that returns has `residual ≤ 10 · lintol`.
- Records in `Derivatives.boundary`: `(; node::Node, parameter::Symbol, kept::Int, unkept::Int, detuning::Float64, coupling::Float64)`, with `kept` a kept-state index (column of the node's vectors), `unkept` an index among the node's computed pairs, and `coupling` `NaN` when it was not computed.

The complement.
At a node whose spectrum is not complete and whose solver is not `Dense`, `_resolvent` keeps: the node's forward operator `H` on its solver's storage (`storage(solver)`), with scratch for as many columns as kept states (`with_block_capacity`); `Φ`, the kept vectors and the retained unkept vectors side by side on that storage; and `D = _full_diagonal(H)`.
`_apply_resolvent` then adds, to the sum over the known states, the solution `X` of `Q (H − s_j) Q x_j = −Q R_j`: this is `(s_j − H)⁻¹` on the orthogonal complement of all the known states, where the sum cannot reach.
`R` is moved to the storage for the solve and `X` back to the host.
At such a node the pass also builds the derivative operators with `on = storage(solver)`, so no operator of a device node is applied on the host.
With all computed states (the lowest of the spectrum) in `Φ`, `H − s_j` is positive on the complement for every kept `j`, so conjugate gradients apply.
A node with `retained.vectors === nothing` (`retain = :never`) uses the kept vectors alone as `Φ`, which is valid only when `mn.kept == 1:length(mn.kept)`; otherwise `_resolvent` throws an `ArgumentError` that names the node's modes and contains "retain".

The solver, all columns in lockstep, each with its own scalars (the algorithm is fixed; write it with `mul!`, broadcasts and column sums only, so that it runs on any storage):

```
Minv[:, j] = 1 ./ hypot.(D .- s_j, eta)                  # the Jacobi diagonal RitzJacobi uses
P_Q(Y)     = Y − Φ (Φ' Y)
X = 0;  Rr = P_Q(B);  b_j = ‖Rr_j‖;  Z = P_Q(Minv .* Rr);  P = Z;  ρ_j = Re⟨Rr_j, Z_j⟩
repeat at most maxiter times:
    AP  = P_Q(H P − P diag(s))
    κ_j = Re⟨P_j, AP_j⟩;  an unconverged column with κ_j ≤ 0 → error: not positive on the complement
    α_j = ρ_j / κ_j  (0 for a converged column);  X += P diag(α);  Rr −= AP diag(α)
    column j is converged when ‖Rr_j‖ ≤ tol · b_j  (a column with b_j = 0 is converged from the start, with x_j = 0)
    stop when every column is converged
    Z = P_Q(Minv .* Rr);  ρ′_j = Re⟨Rr_j, Z_j⟩;  P = Z + P diag(ρ′ ./ ρ);  ρ = ρ′
after the loop: residuals[j] = ‖P_Q(H X − X diag(s)) − P_Q(B)‖_j / b_j with a fresh application of H,
    and must be ≤ 10 · tol for every column
```

A failure throws an error (`error`, not a warning) that names the node's modes (the caller adds them), the column, the residual reached and the iteration count.
Its text contains "did not converge" for a column that misses the tolerance, in the loop or in the final check, and "not positive on the complement" for `κ_j ≤ 0`; the second also says that a kept state may be degenerate with a state that was not computed, and to raise `compute`.
Nothing is returned from a failed solve.

The selection-boundary test (design §3; architecture spec §11.3), run at every visited node for every parameter, over each kept state `i` and each other known state `l` of the node's `Resolvent`.
The coupling of the pair is zero by the per-column rule of Task 4: `|⟨ψ_l|R[:, i]⟩| ≤ 1e-10 ‖R[:, i]‖`, with `R = (∂H/∂x) V`.
- When `scale` has an entry `δx` for the parameter: the pair is recorded when its coupling is not zero and `|⟨ψ_l|R[:, i]⟩| · δx > r · |E_i − E_l|`.
- Otherwise: when `|E_i − E_l| ≤ 1e-6 · max(|E_i − E_1|, |E_l − E_1|)`, with `E_1` the lowest computed energy of the node (`coupling` is then `NaN`).
- In either case, a clash of column `i` with a state that is not kept (an entry `0` in `clashes[i]`, at first or second order) is recorded, with its coupling: the state's derivative is `NaN`, and this record is its report.
- A pair is recorded once per parameter; a clash record, with its coupling, replaces a plain one of the same pair.
One `@warn` per node and parameter, whose text starts with "selection boundary:", names the node's modes, the kept and unkept states, the smallest detuning and the largest coupling, and says the fixed-selection derivative is fragile there.
States beyond `compute` were never computed and cannot be tested; the docstring of `derivatives` says so.

- [ ] **Step 1: Write the failing tests.**

`test/test_derivatives_complement.jl` (fast):

```julia
using Test, LinearAlgebra, Random, KronFreeCircuits
using KronFreeCircuits: _complement_solve, _full_diagonal, with_block_capacity, operator

@testset "the projected solve against the sum over the remaining states, real and complex" for periodic in (false, true)
    c = dc_two_transmons(; periodic, leafkeep=8); sys = dc_solve(c)                  # root dimension 64, real or complex
    H = with_block_capacity(operator(sys), 4); M = Matrix(operator(sys)); F = eigen(Hermitian(M))
    Φ = F.vectors[:, 1:6]; s = F.values[1:4]                                         # the six lowest states projected out
    B = randn(MersenneTwister(1), eltype(M), 64, 4)
    X, iterations, residuals = _complement_solve(H, _full_diagonal(H), Φ, B, s; tol=1e-12, maxiter=5000)
    for j in 1:4
        ref = sum(F.vectors[:, l] * (F.vectors[:, l]' * B[:, j]) / (F.values[l] - s[j]) for l in 7:64)
        @test norm(X[:, j] - ref) <= 1e-8 * norm(ref)
    end
    @test norm(Φ' * X) < 1e-10 && maximum(residuals) <= 1e-11 && iterations < 5000           # 10 · tol
    @test all(iszero, _complement_solve(H, _full_diagonal(H), Φ, Φ[:, 1:4], s; tol=1e-10, maxiter=10)[1])   # Q B = 0: x = 0
    @test_throws "did not converge" _complement_solve(H, _full_diagonal(H), Φ, B, s; tol=1e-10, maxiter=2)
end

@testset "selection-boundary warning" begin
    # the doublet of the cos 2φ element with only its lower state kept: detuning 0.0593, coupling 0.7456 to E_1
    c = dc_cos2phi(keep=lowest(1)); sys = dc_solve(c)
    d = @test_logs (:warn, r"^selection boundary:") match_mode=:any derivatives(sys; wrt=:E_1, scale=(E_1=0.05,))
    b = only(d.boundary)
    @test b.node === c.root && b.parameter === :E_1 && b.kept == 1 && b.unkept == 2
    @test b.detuning ≈ 0.0593 atol=1e-3 && b.coupling ≈ 0.7456 atol=1e-3
    @test isempty(derivatives(sys; wrt=:E_1, scale=(E_1=0.001,)).boundary)           # 0.7456·0.001 < 0.1·0.0593
    @test isempty(derivatives(sys; wrt=:E_1).boundary)                               # no scale: the plain relative-gap test
    # without a scale, an exactly degenerate kept/unkept pair is reported whatever its coupling
    c0 = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(2), E_2=0.0); s0 = dc_solve(c0)   # energies 0, 4 | 4
    d0 = @test_logs (:warn, r"^selection boundary:") match_mode=:any derivatives(s0; wrt=:n_g)
    @test only(d0.boundary).kept == 2 && only(d0.boundary).unkept == 3 && isnan(only(d0.boundary).coupling)
    @test all(isfinite, energies(d0, :n_g))                                          # -8 E_C n does not couple the pair
    # a kept state degenerate with an unkept one and coupled to it: NaN, and the boundary record is its report
    dE = quiet(() -> derivatives(s0; wrt=:E_2))                                      # -cos 2φ couples |−1⟩ and |+1⟩
    @test isnan(energies(dE, :E_2)[2]) && isfinite(energies(dE, :E_2)[1])
    @test only(dE.boundary).kept == 2 && only(dE.boundary).unkept == 3 && only(dE.boundary).coupling ≈ 0.5 atol=1e-12
end
```

`test/test_derivatives_iterative.jl` (slow; add it to `SLOW_FILES`):

```julia
using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: materialized

# the root's derivatives of two trees that differ only in the root's solver
function same_derivatives(ci, cd, x; kwargs...)
    di, dd = derivatives(dc_solve(ci); wrt=x, kwargs...), derivatives(dc_solve(cd); wrt=x)
    @test energies(di, x) ≈ energies(dd, x) rtol=1e-8 atol=1e-9
    @test norm(eigenvectors(di, x) - eigenvectors(dd, x), Inf) <= 1e-6 * norm(eigenvectors(dd, x), Inf)
    for (mode, p) in keys(materialized(dc_solve(cd), cd.root).ports)
        @test norm(port(di, x, mode, p) - port(dd, x, mode, p), Inf) <= 1e-6 * norm(port(dd, x, mode, p), Inf)
    end
    return di
end
tree(solver; kw...) = dc_two_transmons(; leafkeep=12, solver, compute=8, keep=lowest(6), kw...)     # root dimension 144

@testset "the linear-solve route equals the full-spectrum route" begin
    worst = 0
    for x in (:g, :n_g1, :E_J2)
        di = same_derivatives(tree(LOBPCG(tol=1e-11)), tree(Dense()), x)
        @test !isempty(di.linear_solves)
        @test all(s -> s.iterations < 5000 && s.residual <= 1e-9, di.linear_solves)       # 10 · lintol
        worst = max(worst, maximum(s.iterations for s in di.linear_solves))
    end
    # a complex node (design §5.1): periodic grids make the leaves and the root complex; the ports include expiphi
    for x in (:g, :n_g1)
        di = same_derivatives(tree(LOBPCG(tol=1e-11); periodic=true), tree(Dense(); periodic=true), x)
        worst = max(worst, maximum(s.iterations for s in di.linear_solves))
    end
    same_derivatives(tree(Lanczos(tol=1e-12)), tree(Dense()), :g)
    # a leaf solved by shift-invert
    fl(solver) = dc_fluxonium(; solver, compute=6, keep=lowest(4))
    for x in (:varphi_ext, :E_J)
        di = same_derivatives(fl(ShiftInvert(tol=1e-11)), fl(Dense()), x)
        worst = max(worst, maximum(s.iterations for s in di.linear_solves))
    end
    @info "largest iteration count of the linear solves in this testset" worst
    # and against finite differences of the iterative calculation itself
    c = tree(LOBPCG(tol=1e-11)); at = dc_memo(c); d = derivatives(at(c.θ); wrt=:g)
    check_fd(energies(d, :g), fd_derivative(θ -> energies(at(θ)), c.θ, :g; h=1e-3); rtol=1e-6)
    check_fd(eigenvectors(d, :g), fd_derivative(θ -> eigenvectors(at(θ)), c.θ, :g; h=1e-3); rtol=1e-6)
    # Review Focus 4: kept states that are not the lowest computed ones
    sel(solver) = dc_two_transmons(; leafkeep=12, solver, compute=4, keep=indices([1, 3]))
    same_derivatives(sel(LOBPCG(tol=1e-11)), sel(Dense()), :g)
    # without retained vectors the kept states alone are projected out, which needs them to be the lowest
    same_derivatives(tree(LOBPCG(tol=1e-11); retain=:never), tree(Dense()), :g)
    never = dc_two_transmons(; leafkeep=12, solver=LOBPCG(tol=1e-11), compute=4, keep=indices([1, 3]), retain=:never)
    @test_throws "retain" derivatives(dc_solve(never); wrt=:g)
end

@testset "the sum over the computed states alone is not the derivative; a solve that fails is an error" begin
    sys = dc_solve(tree(LOBPCG(tol=1e-11)))
    full = eigenvectors(derivatives(sys; wrt=:g), :g)
    cut = eigenvectors(derivatives(sys; wrt=:g, _truncate=true), :g)
    @test norm(cut - full, Inf) > 1e-3 * norm(full, Inf)
    @test_throws "did not converge" derivatives(sys; wrt=:g, linmaxiter=2)
end
```

- [ ] **Step 2: Run and confirm failure**: `Pkg.test(test_args=["test_derivatives_complement.jl","test_derivatives_iterative.jl"])` fails with `UndefVarError: _complement_solve`, the `ArgumentError` of Task 4 for an iterative node, and no `boundary` records.

- [ ] **Step 3: Implement** `_complement_solve`, the complement branch of `_resolvent` and `_apply_resolvent`, the records, the boundary test and the two internal keywords.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivatives_complement.jl","test_derivatives_iterative.jl","test_derivatives_tree.jl","test_derivatives_node.jl"])`, then the default tier (which must include the first file and not the second). Expected: PASS.
Report the time of each of the two new files and the iteration count the `@info` prints.

- [ ] **Step 5: Commit** (`Derivatives at iterative nodes: a projected linear solve; the selection-boundary warning`), with `src/resolvent.jl src/derivatives.jl test/runtests.jl test/test_derivatives_complement.jl test/test_derivatives_iterative.jl`.

---

### Task 7: The second-order rule at one node

**Files:**
- Modify: `src/derivoperator.jl` (the second-order form of `_derivative_operator`), `src/derivatives.jl` (`_node_second_order`, `_port_second_derivatives`), `test/helpers.jl` (`node_second_order`)
- Test: `test/test_derivatives_second.jl` (fast tier; estimated 20 s, most of it compilation)

**Interfaces:**
- Consumes: Tasks 3 and 4 (`NodeDerivative` with its `gauge` and `children`, `_derivative_operator`, `_resolvent`, `_apply_resolvent` returning `(W, clashes)`, `_peak_map`, `_peak_amps`, `_sandwich`, `_derivative_op`, `_node_first_order`); `differentiate` twice; `node_setup`, `node_first_order`, `fd_child`, `fd_pair`, `dc_memo` of `test/helpers.jl`.
- Produces, internal:

```julia
# ∂²H/∂x∂λ of the node (x == λ allowed).  `dx`, `dλ` are the children's first-order data, `dxλ` their second-order data.
_derivative_operator(mn::MaterializedNode, terms::Vector{Term}, θ::NamedTuple, x::Symbol, λ::Symbol,
                     dx::Vector{Any}, dλ::Vector{Any}, dxλ::Vector{Any};
                     constant::Coefficient = Coefficient(0), on = Array,
                     parts = (:coefficients, :ports, :energies)) -> KronFreeOperator

# The node's second-order data for the pair (x, λ): `dX`, `dΛ` are its own first-order data (their `children` hold the
# children's), `dxλ` the children's second-order data.  The result's `gauge` is Im⟨ψ_j|∂²ψ_j⟩, and its `children`
# field is `dxλ` itself.
_node_second_order(mn::MaterializedNode, Hx::KronFreeOperator, Hλ::KronFreeOperator, Hxλ::KronFreeOperator, res::Resolvent,
                   dX::NodeDerivative, dΛ::NodeDerivative, dxλ::Vector{Any}) -> NodeDerivative
```

- Produces, in `test/helpers.jl`:

```julia
# The second-order rule on a one-node system, with the first-order data and the operators it used.
function node_second_order(sys, x::Symbol, λ::Symbol)
    K = KronFreeCircuits
    u = node_setup(sys)
    op(args...) = K._derivative_operator(u.mn, u.terms, sys.θ, args...; constant=sys.constant)
    Hx, Hλ, Hxλ = op(x, u.none), op(λ, u.none), op(x, λ, u.none, u.none, u.none)
    dX, dΛ = K._node_first_order(u.mn, Hx, u.res, u.none), K._node_first_order(u.mn, Hλ, u.res, u.none)
    return (; d=K._node_second_order(u.mn, Hx, Hλ, Hxλ, u.res, dX, dΛ, u.none), dX, dΛ, Hλ, Hxλ)
end
```

The operator.
For a term with coefficient `c` and operators `ops` (as in the first-order form), with `g`, `h` running over the entries of `ops`:
`c_xλ(θ)` on `ops`;
`c_x(θ)` on `ops[g → dλ]` and `c_λ(θ)` on `ops[g → dx]`, for every `g` whose child has that data;
`c(θ)` on `ops[g → dxλ]`;
`c(θ)` on `ops[g → dx, h → dλ]` for every ordered pair `g ≠ h`.
A variant whose coefficient value is zero is left out; real nodes take the real expansion of each variant as before.
Then `DiagonalOp(dxλ[a].energies)` per child with second-order data, and the constant `differentiate(differentiate(constant, x), λ)(θ)`.
With `x == λ` the same list applies (`dx` and `dλ` are the same data), which gives the factor 2 of the cross terms.

The equations, per kept state `j`, with `W^x = ∂_xV − i V diag(a^x)` recovered from `dX.vectors` and `dX.gauge` (likewise `W^λ`):

1. `RHS = (Hx W^λ − W^λ diag(∂_xE)) + (Hλ W^x − W^x diag(∂_λE)) + Hxλ V`.
2. `∂²E_j = Re⟨ψ_j|RHS_j⟩`, which equals `⟨ψ_j|Hxλ|ψ_j⟩ + 2 Re⟨w^x_j|Hλ|ψ_j⟩`, the paper's `eq:eig_val_secondgrad`.
3. `G, clashes = _apply_resolvent(res, RHS, E, [[j] for j])` and `U = G + i W^x diag(a^λ) + i W^λ diag(a^x)`: the part of `∂²ψ_j` orthogonal to `ψ_j`.
4. Normalization: `Re⟨ψ_j|∂²ψ_j⟩ = −Re⟨∂_xψ_j|∂_λψ_j⟩ = −Re⟨w^x_j|w^λ_j⟩ − a^x_j a^λ_j =: n_j`.
5. Gauge (complex nodes; `b_j = 0` on a real node): the second derivative of `ℓ_j · ψ_j` is real, so
   `b_j = −Im( ℓ^{xλ}_j·ψ_j + ℓ^x_j·∂_λψ_j + ℓ^λ_j·∂_xψ_j + ℓ_j·u_j ) / Re A[j, j]`,
   where `ℓ_j·X` is row `j` of `_peak_map(mn, X, _peak_amps(mn))`, `ℓ^x` replaces one axis's amplitudes by the child's first derivative (summed over the axes with data), and `ℓ^{xλ}` is the sum over axes of the child's second derivative replaced, plus the sum over ordered pairs of axes `k ≠ m` with `k → dx[k]`, `m → dλ[m]` replaced together.
   (`a^x`, `a^λ`, `b` are gauges of states; axes are numbered `k`, `m`.)
6. `∂²ψ_j = u_j + (n_j + i b_j) ψ_j`; `gauge = b`.
7. `∂²A = ℓ·∂²V + ℓ^x·∂_λV + ℓ^λ·∂_xV + ℓ^{xλ}·V` (all rows).
8. Each port and joint port `P = V† O V`, with `S(W, V; O)` standing for `_sandwich`, `O_x` the operator with one entry replaced by the child's `x`-derivative (summed over entries), and `O_xλ` built like `ℓ^{xλ}`:
   `∂²P = S(∂²V, V; O) + S(V, ∂²V; O) + S(∂_xV, ∂_λV; O) + S(∂_λV, ∂_xV; O) + S(∂_xV, V; O_λ) + S(V, ∂_xV; O_λ) + S(∂_λV, V; O_x) + S(V, ∂_λV; O_x) + S(V, V; O_xλ)`.
9. A column with a clash in step 3, or already `NaN` at first order, is `NaN` in energy and state, with the one `@warn` containing "degenerate" as at first order.
   The clashes are handed to the block search (Task 9) and the boundary test (Task 6) like the first-order ones.

- [ ] **Step 1: Write the failing tests** in `test/test_derivatives_second.jl`

```julia
using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: _derivative_operator, operator, route

function check_node2(c, x, λ; hx, hλ=hx, rtol=1e-5, atol=1e-8)
    at = dc_memo(c); sys = at(c.θ); r = node_second_order(sys, x, λ); d = r.d
    fd(f) = fd_pair(f, c.θ, x, λ; hx, hλ)
    @test d.children == Any[nothing]
    check_fd(d.energies, fd(θ -> energies(at(θ))); rtol, atol)                              # absolute: Review Focus 2
    check_fd(d.vectors, fd(θ -> eigenvectors(at(θ))); rtol, atol)
    check_fd(d.peak_amplitudes, fd(θ -> at(θ).root.peak_amplitudes); rtol, atol)
    for key in keys(sys.root.ports)                                                         # Review Focus 3
        check_fd(d.ports[key], fd(θ -> at(θ).root.ports[key]); rtol, atol)
    end
    V = eigenvectors(sys)                                                                   # the normalization condition
    @test [real(dot(V[:, j], d.vectors[:, j])) for j in axes(V, 2)] ≈
          [-real(dot(r.dX.vectors[:, j], r.dΛ.vectors[:, j])) for j in axes(V, 2)] atol=1e-10
    # eq:eig_val_secondgrad of arXiv:2408.12704v3: ∂²f_i = ⟨f_i|∂²H|f_i⟩ + 2 Re⟨∂_x f_i|∂_λH|f_i⟩
    Hλ, Hxλ = Matrix(r.Hλ), Matrix(r.Hxλ)
    @test d.energies ≈ [real(V[:, j]' * Hxλ * V[:, j]) + 2 * real(r.dX.vectors[:, j]' * Hλ * V[:, j]) for j in axes(V, 2)] rtol=1e-9 atol=1e-10
    return d
end

@testset "second order at a dense leaf against finite differences" begin
    check_node2(dc_transmon(), :E_J, :E_J; hx=2e-2)
    check_node2(dc_transmon(), :n_g, :n_g; hx=5e-3)
    check_node2(dc_transmon(), :E_J, :n_g; hx=2e-2, hλ=5e-3)
    d = check_node2(dc_transmon(periodic=true), :n_g, :n_g; hx=5e-3, rtol=1e-6)
    @test maximum(abs, d.gauge) > 1e-3
    check_node2(dc_transmon(periodic=true), :E_J, :E_J; hx=2e-2)
    check_node2(dc_transmon(periodic=true), :E_J, :n_g; hx=2e-2, hλ=5e-3)
    check_node2(dc_fluxonium(), :varphi_ext, :varphi_ext; hx=1e-2, rtol=1e-6, atol=1e-6)
    check_node2(dc_fluxonium(), :E_J, :E_J; hx=2e-2)
    check_node2(dc_fluxonium(), :E_J, :varphi_ext; hx=1e-2, atol=1e-6)
    check_node2(dc_transmon(compute=8, keep=indices([1, 3, 4])), :E_J, :n_g; hx=2e-2, hλ=5e-3)   # Review Focus 4
end

@testset "analytic second derivatives" begin
    # oscillator, E_k = (k + 1/2) sqrt(8 E_C E_L): ∂²E_k/∂E_L² = −(k + 1/2) sqrt(2 E_C) / (2 E_L^(3/2)), ∂²E_k/∂E_C∂E_L = (k + 1/2) / sqrt(2 E_C E_L)
    c = dc_oscillator(); k = 0:5; (; E_C, E_L) = c.θ
    @test node_second_order(dc_solve(c), :E_L, :E_L).d.energies ≈ -(k .+ 0.5) .* sqrt(2E_C) ./ (2 * E_L^1.5) rtol=1e-9
    @test node_second_order(dc_solve(c), :E_C, :E_L).d.energies ≈ (k .+ 0.5) ./ sqrt(2 * E_C * E_L) rtol=1e-9
    # transmon at n_g = 0: the cosine law of the charge dispersion gives ∂²E_0/∂n_g² = 2π² ε_0
    tm(ng) = dc_transmon(θ=(E_J=30.0, E_C=1.2, n_g=ng))
    ε = energies(dc_solve(tm(0.5))) - energies(dc_solve(tm(0.0)))
    @test node_second_order(dc_solve(tm(0.0)), :n_g, :n_g).d.energies[1] ≈ 2π^2 * ε[1] rtol=1e-4
    # fluxonium with E_J = 0: the flux displaces an oscillator, so every level's curvature vanishes (E_L from the constant
    # cancels the sum over states)
    @test maximum(abs, node_second_order(dc_solve(dc_fluxonium(varphi_ext=1.0, E_J=0.0)), :varphi_ext, :varphi_ext).d.energies) < 1e-6
    # exact degeneracy at second order: on the charges −1, 0, 1, −cos φ couples |−1⟩ and |+1⟩ through |0⟩
    c1 = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(0), E_2=0.0)
    d = quiet(() -> node_second_order(dc_solve(c1), :E_1, :E_1).d)
    @test d.energies[1] ≈ -0.25 atol=1e-12 && all(isnan, d.energies[2:3]) && all(isnan, d.vectors[:, 2:3])
end

@testset "∂²H/∂x∂λ at a branch equals the second difference of the forward operator" begin
    for (c, pairs) in ((dc_two_transmons(), ((:g, :g), (:g, :E_J1), (:E_J1, :E_J2), (:E_J1, :E_J1))),
                       (dc_abc(), ((:varphi_ext, :varphi_ext), (:E_J, :varphi_ext), (:E_J3, :E_J))))
        at = dc_memo(c); sys = at(c.θ); terms = route(c.root, sys.terms)[c.root]
        for (x, λ) in pairs
            first_of(p) = Any[fd_child(at, c, ch, p; h=1e-3) for ch in c.root.children]
            second = Any[fd_child(at, c, ch, x, λ; hx=2e-2) for ch in c.root.children]
            H2 = _derivative_operator(sys.root, terms, c.θ, x, λ, first_of(x), first_of(λ), second; constant=sys.constant)
            check_fd(Matrix(H2), fd_pair(θ -> Matrix(operator(at(θ))), c.θ, x, λ; hx=2e-2); rtol=1e-5, atol=1e-6)
        end
    end
end
```

- [ ] **Step 2: Run and confirm failure** (`UndefVarError: _node_second_order`).

- [ ] **Step 3: Implement** the second-order form of `_derivative_operator`, `_port_second_derivatives` and `_node_second_order`, by the equations above.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivatives_second.jl","test_derivoperator.jl","test_derivatives_node.jl"])`, then the default tier. Expected: PASS.

- [ ] **Step 5: Commit** (`Second-order derivatives at one node: energies, states, normalization, gauge, ports`), with `src/derivoperator.jl src/derivatives.jl test/helpers.jl test/test_derivatives_second.jl`.

---

### Task 8: Second order through the tree, and mixed pairs

**Files:**
- Modify: `src/derivatives.jl` (the second-order pass, `derivatives(...; order, mixed)`, the two-parameter accessors)
- Test: `test/test_derivatives_second_tree.jl` (fast tier; estimated 25 s, most of it compilation: every comparison shares memoized solves), and one testset appended to `test/test_derivatives_iterative.jl` (slow tier, Task 6's file)

**Interfaces:**
- Consumes: `_node_second_order` (whose result's `children` is the children's second-order data), the second-order `_derivative_operator` (Task 7); the first-order pass, `Derivatives`, `_pairkey`, `_node_derivative` (Task 5); `fd_pair`, `dc_memo`, `dc_solve`, the `dc_*` circuits; `tree` and `same_derivatives`'s pattern in `test/test_derivatives_iterative.jl` (Task 6).
- Produces, exported methods:

```julia
# order = 2: the second derivative with respect to each parameter of `wrt` alone, and to each pair of `mixed`.
derivatives(sys; wrt, order = 2, mixed = ((:E_J, :varphi_ext),), ...)
energies(d::Derivatives, x::Symbol, λ::Symbol, node::Node = d.sys.root.node) -> Vector{Float64}      # x == λ allowed; symmetric in (x, λ)
eigenvectors(d::Derivatives, x::Symbol, λ::Symbol, node::Node = d.sys.root.node) -> Matrix
port(d::Derivatives, x::Symbol, λ::Symbol, mode::Symbol, p::Port, node::Node = d.sys.root.node) -> Matrix
port(d::Derivatives, x::Symbol, λ::Symbol, key::AbstractVector{Tuple{Symbol,Port}}, node::Node = d.sys.root.node) -> Matrix
```

Behaviour:
- With `order = 2`, after the first-order pass of every parameter, one second-order pass runs for `(x, x)` for each `x` of `wrt` and one for each pair of `mixed`, children first, storing the root's data in `d.second[_pairkey(x, λ)]`.
  A pass rebuilds `∂H/∂x`, `∂H/∂λ` and `∂²H/∂x∂λ` at each node; nothing but the first-order `NodeDerivative`s is kept between passes.
- A node whose first-order data for `x` or for `λ` is `nothing` has second-order data `nothing`, read as zeros.
- Validation, each an `ArgumentError`: `order` is 1 or 2; `mixed` is a collection of pairs of two different symbols, both in `wrt`; `mixed` nonempty needs `order = 2`; an accessor asked for a pair that was not computed lists the pairs that were.
- The docstring of `derivatives` states the cost: one first-order pass per parameter and one second-order pass per parameter and per mixed pair, each about one resolvent application per kept state per node.

- [ ] **Step 1: Write the failing tests.**

`test/test_derivatives_second_tree.jl` (fast):

```julia
using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: materialized, _all_nodes

# every node's second derivatives (energies, vectors, ports, joint ports) against finite differences of `materialize`
function check_tree2(c, x, λ; hx, hλ=hx, rtol=1e-5, atol=1e-7)
    at = dc_memo(c); sys = at(c.θ)
    d = derivatives(sys; wrt=unique([x, λ]), order=2, mixed=x === λ ? () : ((x, λ),))
    fd(f) = fd_pair(f, c.θ, x, λ; hx, hλ)
    for node in _all_nodes(c.root)
        mn = materialized(sys, node)
        check_fd(energies(d, x, λ, node), fd(θ -> energies(at(θ), node)); rtol, atol)
        check_fd(eigenvectors(d, x, λ, node), fd(θ -> eigenvectors(at(θ), node)); rtol, atol)
        for (mode, p) in keys(mn.ports)
            check_fd(port(d, x, λ, mode, p, node), fd(θ -> port(at(θ), mode, p, node)); rtol, atol)
        end
        for K in keys(mn.joint_ports)
            check_fd(port(d, x, λ, K, node), fd(θ -> port(at(θ), K, node)); rtol, atol)
        end
    end
    return d
end

@testset "second order through trees against finite differences" begin
    check_tree2(dc_two_transmons(), :g, :g; hx=1e-2)
    check_tree2(dc_two_transmons(), :E_J2, :E_J2; hx=2e-2)
    check_tree2(dc_two_transmons(), :g, :E_J2; hx=1e-2, hλ=2e-2)
    # complex leaves and root (Review Focus 3).  On the periodic grids the leaves' energies are of order 1e3 and their
    # second derivative in g of order 1e-3, so its finite difference carries round-off of 1e-7: hence atol = 1e-6.
    check_tree2(dc_two_transmons(periodic=true), :g, :g; hx=1e-2, atol=1e-6)
    check_tree2(dc_two_transmons(periodic=true), :g, :n_g1; hx=1e-2, hλ=5e-3)
    check_tree2(dc_chain3(), :g12, :g12; hx=1e-2)
    check_tree2(dc_chain3(), :g23, :g23; hx=1e-2)
    check_tree2(dc_chain3(), :E_J1, :g23; hx=2e-2, hλ=1e-2)
end

@testset "second order through trees: paired ports with complex coefficients, and joint ports" begin
    check_tree2(dc_zero_pi(), :varphi_ext, :varphi_ext; hx=1e-2)
    check_tree2(dc_zero_pi(), :E_J, :varphi_ext; hx=2e-2, hλ=1e-2)
    check_tree2(dc_abc(), :varphi_ext, :varphi_ext; hx=1e-2)
    check_tree2(dc_abc(), :E_J3, :varphi_ext; hx=1e-2)
end

@testset "Review Focus 5: a mixed pair whose parameters enter at different levels" begin
    # n_g1 and n_g2 enter inside leaves, g and g23 in a coupling above them; E_J of dc_abc inside the leaves, varphi_ext at the root
    check_tree2(dc_two_transmons(), :g, :n_g1; hx=1e-2, hλ=5e-3)
    check_tree2(dc_abc(), :E_J, :varphi_ext; hx=2e-2, hλ=1e-2)
    d = check_tree2(dc_chain3(), :n_g2, :g23; hx=5e-3, hλ=1e-2)
    @test norm(energies(d, :n_g2, :g23), Inf) > 1e-3
    # the result does not depend on the order in which the pair is given
    sys = dc_solve(dc_chain3())
    d1 = derivatives(sys; wrt=(:n_g2, :g23), order=2, mixed=((:n_g2, :g23),))
    d2 = derivatives(sys; wrt=(:g23, :n_g2), order=2, mixed=((:g23, :n_g2),))
    @test energies(d1, :n_g2, :g23) ≈ energies(d2, :n_g2, :g23) rtol=1e-10
    @test eigenvectors(d1, :g23, :n_g2) ≈ eigenvectors(d2, :n_g2, :g23) rtol=1e-9
end

@testset "a mixed derivative for dephasing; the two-parameter API" begin
    # ∂²f_01/∂E_J∂varphi_ext of the fluxonium off its sweet spots
    c = dc_fluxonium(); at = dc_memo(c); sys = at(c.θ)
    d = derivatives(sys; wrt=(:E_J, :varphi_ext), order=2, mixed=((:E_J, :varphi_ext),))
    f01(E) = E[2] - E[1]
    @test f01(energies(d, :E_J, :varphi_ext)) ≈ -0.053526672 rtol=1e-6          # measured by finite differences when the plan was written
    check_fd(f01(energies(d, :E_J, :varphi_ext)), fd_mixed(θ -> f01(energies(at(θ))), c.θ, :E_J, :varphi_ext; hx=1e-2, hλ=1e-2); rtol=1e-6)
    @test haskey(d.second, (:E_J, :E_J)) && haskey(d.second, (:E_J, :varphi_ext)) && haskey(d.second, (:varphi_ext, :varphi_ext))
    @test_throws ArgumentError energies(derivatives(sys; wrt=(:E_J, :varphi_ext), order=2), :E_J, :varphi_ext)   # pair not requested
    @test_throws ArgumentError derivatives(sys; wrt=:E_J, order=2, mixed=((:E_J, :varphi_ext),))                 # not in wrt
    @test_throws ArgumentError derivatives(sys; wrt=(:E_J, :varphi_ext), mixed=((:E_J, :varphi_ext),))            # needs order = 2
    @test_throws ArgumentError derivatives(sys; wrt=:E_J, order=3)
    # a subtree that does not depend on one of the two parameters has a zero mixed derivative
    c3 = dc_chain3(); d3 = derivatives(dc_solve(c3); wrt=(:g12, :E_J3), order=2, mixed=((:g12, :E_J3),))
    @test all(iszero, energies(d3, :g12, :E_J3, c3.inner)) && all(iszero, energies(d3, :g12, :E_J3, c3.third))
    @test norm(energies(d3, :g12, :E_J3), Inf) > 0
end
```

Appended to `test/test_derivatives_iterative.jl` (slow), the second order at iterative nodes, real and complex (design §5.1):

```julia
@testset "second order: the linear-solve route equals the full-spectrum route" begin
    for (periodic, x, λ) in ((false, :g, :g), (false, :g, :E_J2), (true, :g, :g), (true, :g, :n_g1))
        wrt = unique([x, λ]); mixed = x === λ ? () : ((x, λ),)
        di = derivatives(dc_solve(tree(LOBPCG(tol=1e-11); periodic)); wrt, order=2, mixed)
        dd = derivatives(dc_solve(tree(Dense(); periodic)); wrt, order=2, mixed)
        @test energies(di, x, λ) ≈ energies(dd, x, λ) rtol=1e-6 atol=1e-8
        @test norm(eigenvectors(di, x, λ) - eigenvectors(dd, x, λ), Inf) <= 1e-5 * norm(eigenvectors(dd, x, λ), Inf)
        for p in (nport(), expiphi(1.0))
            @test norm(port(di, x, λ, :varphi_1, p) - port(dd, x, λ, :varphi_1, p), Inf) <= 1e-5 * norm(port(dd, x, λ, :varphi_1, p), Inf)
        end
    end
end
```

- [ ] **Step 2: Run and confirm failure** (the `ArgumentError` of Task 5 for `order = 2`).

- [ ] **Step 3: Implement** the second-order pass, the keywords and the accessors.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivatives_second_tree.jl","test_derivatives_second.jl","test_derivatives_tree.jl","test_derivatives_iterative.jl"])`, then the default tier. Expected: PASS.
Report the time of `test_derivatives_second_tree.jl`.
If it is above 30 s, move its second testset ("paired ports with complex coefficients, and joint ports") with `check_tree2` to a new file `test/test_derivatives_second_slow.jl`, list that file in `SLOW_FILES`, and report both times.

- [ ] **Step 5: Commit** (`derivatives(sys; order = 2, mixed): second order through the tree`), with `src/derivatives.jl test/test_derivatives_second_tree.jl test/test_derivatives_iterative.jl` (and `test/runtests.jl test/test_derivatives_second_slow.jl` if Step 4 moved a testset).

---

### Task 9: Near-degenerate blocks and their effective Hamiltonian

**Files:**
- Create: `src/derivblocks.jl` (included after `derivatives.jl`)
- Modify: `src/derivatives.jl` (the passes call the block search), `src/KronFreeCircuits.jl` (export `blocks`, `NearDegenerateBlock`, `block_levels`)
- Test: `test/test_derivatives_blocks.jl` (fast tier; estimated 15 s)

**Interfaces:**
- Consumes: `Derivatives.blocks`, the passes and their per-node operators (Tasks 5, 8); `_apply_resolvent` with a block in `exclude`, and the `clashes` it returns at first and second order (Tasks 4, 7); `scale`, `r`, `degenerate_tol`; `dc_cos2phi`, `dc_solve`, `shifted`.
- Produces, exported:

```julia
struct NearDegenerateBlock
    node::Node
    parameters::Tuple{Vararg{Symbol}}       # (x,), or `_pairkey(x, λ)` for a mixed pair
    states::Vector{Int}                     # kept-state indices at `node` (columns of its vectors), increasing
    energies::Vector{Float64}               # their energies at θ
    detuning::Float64                       # the smallest |E_i − E_l| among the flagged pairs
    coupling::Float64                       # the largest |⟨ψ_l|∂H/∂x|ψ_i⟩| among the flagged pairs
    exact::Bool                             # a flagged pair lies within `degenerate_tol`: its per-state derivatives are NaN
    order::Int                              # the order of the effective Hamiltonian in the displacement: 1 or 2
    H1::Dict{Symbol,Matrix{ComplexF64}}                    # x => ⟨ψ_b|∂H/∂x|ψ_b′⟩ in the block
    H2::Dict{Tuple{Symbol,Symbol},Matrix{ComplexF64}}      # `_pairkey(x, y)` => second-order piece; empty when order == 1
end

blocks(d::Derivatives) -> Vector{NearDegenerateBlock}

# The block's energies and states (coefficients in the block's states at θ) at a finite displacement of its parameters,
# e.g. `block_levels(b, (phi_ext = 2e-3,))`; a parameter not named is not displaced; a name that is not one of
# `b.parameters` is an ArgumentError.
block_levels(b::NearDegenerateBlock, displacement::NamedTuple) -> (; energies::Vector{Float64}, states::Matrix{ComplexF64})
```

The rules (design §4.4).
At every visited node and for every parameter `x`, with `R = (∂H/∂x) V`, `C = V† R` and the kept energies `E`, a pair `(i, l)` of kept states is flagged in two cases.
- **Exact**: `l` is a clash of column `i`, or `i` of column `l`, as `_apply_resolvent` reported it in the pass (Task 4's per-column rule: a detuning within `degenerate_tol` and an overlap above `1e-10` times the column's norm).
  This is always tested, and it is the same event that makes the state's derivative `NaN`, so a `NaN` never appears without a block (or, when the partner is not kept, without the boundary record of Task 6).
  A clash that first appears in a second-order pass (a pair coupled only through other states) flags the pair for the parameters of that pass: `(x,)` for the pass `(x, x)`, `_pairkey(x, λ)` for a mixed pass.
- **Near**: `scale` has `δx` for `x`, the coupling is not zero by the same per-column rule in either direction (`|C[l, i]| > 1e-10 ‖R[:, i]‖` or `|C[i, l]| > 1e-10 ‖R[:, l]‖`), and `|C[l, i]| · δx > r · |E_i − E_l|`.
A pair whose coupling vanishes is never flagged, however small its detuning.
The blocks of `x` are the connected components (of two or more states) of the flagged pairs.
For a mixed pair `(x, λ)` the blocks are the connected components of the union of the flagged pairs of `x`, of `λ` and of the mixed pass, reported with `parameters == _pairkey(x, λ)`.
Every block found is in `blocks(d)`: one per parameter and node, and one per mixed pair and node.
Point derivatives are not changed by any of this; they are what Tasks 4 to 8 compute.

When the pieces are formed.
The first-order flags of every parameter are known after the first-order passes, so the blocks of every parameter and of every mixed pair (as far as first order decides them) exist before any second-order pass; each gets its `H1` then.
During the second-order pass of a pair `(y, z)`, including `(x, x)`, at each node, every block at that node whose `parameters` contain both `y` and `z` gets `H2[_pairkey(y, z)]`, formed there from the operators the pass has just built (`∂H/∂y`, `∂H/∂z`, `∂²H/∂y∂z`) and the block-excluded resolvent.
A mixed block `(x, λ)` therefore receives `H2[(x, x)]` in the pass `(x, x)`, `H2[(λ, λ)]` in the pass `(λ, λ)` and `H2[_pairkey(x, λ)]` in the mixed pass, each for the mixed block's own states (which may be more than either parameter's block).
A block created or enlarged by a second-order clash gets its pieces in that same pass, and the pieces of the passes already run are formed again for it from rebuilt operators.

The effective Hamiltonian of a block `B` (states `b, b′`), in the displacement `δ` of its parameters:

```
H_eff(δ) = diag(E_B) + Σ_x δ_x H1[x] + ½ Σ_x δ_x² H2[(x, x)] + δ_x δ_λ H2[(x, λ)]          (the last term for a mixed block)
H1[x]      = C^x restricted to B
H2[(x, y)] = ⟨ψ_b|∂²H/∂x∂y|ψ_b′⟩ + F^{xy} + F^{yx},      F^{xy} = ½ (M^{xy} + (M^{yx})†)
M^{xy}[b, b′] = ⟨(∂H/∂x) ψ_b | Y^y_{b′}⟩,   Y^y_{b′} = Σ_{l ∉ B} ψ_l ⟨ψ_l|∂H/∂y|ψ_{b′}⟩ / (E_{b′} − E_l)
```

`Y^y` is the first result of `_apply_resolvent(res, (∂H/∂y) V[:, B], E[B], [B for each column])`: the states outside the block, kept or not, folded in to second order in the displacement, which is what "folded in perturbatively" means here.
With `order = 1` only `H1` is filled and `b.order == 1`.
`block_levels` diagonalizes `H_eff(δ)` densely (`eigen(Hermitian(…))`).
Its energies differ from a re-solve at fixed selection by `O(|δ|³ v³ / g²)`, with `v` the block's largest coupling to outside states and `g` its gap to them (first order: `O(|δ|² v² / g)`); the docstring says so.

- [ ] **Step 1: Write the failing tests** in `test/test_derivatives_blocks.jl`

```julia
using Test, LinearAlgebra, KronFreeCircuits

@testset "the doublet of the cos 2φ element: report, true point derivatives, effective Hamiltonian" begin
    # detuning Δ = 0.05932237 GHz, coupling c = |⟨1|∂H/∂E_1|2⟩| = 0.74562858, so c·δx/Δ = 0.63 at δx = 0.05
    c = dc_cos2phi(keep=lowest(2)); sys = dc_solve(c); E0 = energies(sys)
    d = derivatives(sys; wrt=:E_1, order=2, scale=(E_1=0.05,))
    b = only(blocks(d))
    @test b isa NearDegenerateBlock && b.node === c.root && b.parameters == (:E_1,) && b.states == [1, 2] && !b.exact && b.order == 2
    @test b.detuning ≈ 0.05932237 atol=1e-6 && b.coupling ≈ 0.74562858 atol=1e-6 && b.energies ≈ E0
    # the point derivatives are the true ones: E_1 = 0 is a sweet spot with curvature 4c²/Δ of the transition frequency
    E1, E2 = energies(d, :E_1), energies(d, :E_1, :E_1)
    @test maximum(abs, E1) < 1e-9
    # measured when the plan was written: the finite-difference curvature is 37.48731 and 4c²/Δ is 37.48751, 5e-6 apart
    # (the doublet's couplings to the levels 9 GHz away)
    @test E2[2] - E2[1] ≈ 4 * b.coupling^2 / b.detuning rtol=1e-4
    check_fd(E2, fd_second(θ -> energies(dc_solve(c, θ)), c.θ, :E_1; h=2e-3); rtol=1e-5)          # a step small against Δ/c = 0.08
    # at δ = 0.05 the second-order Taylor expansion is off by more than 1e-3 GHz, and the effective Hamiltonian is not
    for (δ, tol) in ((0.05, 1e-6), (0.02, 5e-8))
        exact = energies(dc_solve(c, shifted(c.θ, :E_1, δ)))
        @test block_levels(b, (E_1=δ,)).energies ≈ exact atol=tol
    end
    @test maximum(abs, E0 .+ 0.05 .* E1 .+ 0.5 * 0.05^2 .* E2 .- energies(dc_solve(c, shifted(c.θ, :E_1, 0.05)))) > 1e-3
    @test_throws ArgumentError block_levels(b, (n_g=0.01,))
    # first order only
    b1 = only(blocks(derivatives(sys; wrt=:E_1, scale=(E_1=0.05,))))
    @test b1.order == 1 && isempty(b1.H2)
    @test block_levels(b1, (E_1=0.01,)).energies ≈ energies(dc_solve(c, shifted(c.θ, :E_1, 0.01))) atol=1e-5
    # the threshold: |c| δx > r Δ
    @test isempty(blocks(derivatives(sys; wrt=:E_1, scale=(E_1=0.001,))))                 # 7.5e-4 < 0.1 · 0.0593
    @test length(blocks(derivatives(sys; wrt=:E_1, scale=(E_1=0.001,), r=0.01))) == 1     # 7.5e-4 > 0.01 · 0.0593
    @test isempty(blocks(derivatives(sys; wrt=:E_1)))                                     # no scale: only the exact guard
    # a coupling that vanishes by symmetry removes the pair: ∂H/∂n_g conserves the Cooper-pair parity of the doublet
    @test isempty(blocks(derivatives(sys; wrt=:n_g, scale=(n_g=0.01,))))
end

@testset "a mixed pair: the union of the two parameters' blocks" begin
    c = dc_cos2phi(keep=lowest(2)); sys = dc_solve(c)
    d = derivatives(sys; wrt=(:E_1, :n_g), order=2, mixed=((:E_1, :n_g),), scale=(E_1=0.05, n_g=0.01))
    @test sort([b.parameters for b in blocks(d)]) == [(:E_1,), (:E_1, :n_g)]
    bm = only(b for b in blocks(d) if b.parameters == (:E_1, :n_g))
    @test bm.states == [1, 2] && haskey(bm.H2, (:E_1, :n_g)) && haskey(bm.H2, (:n_g, :n_g)) && haskey(bm.H1, :n_g)
    θd = merge(c.θ, (E_1=0.03, n_g=0.492))
    @test block_levels(bm, (E_1=0.03, n_g=0.002)).energies ≈ energies(dc_solve(c, θd)) atol=1e-6
end

@testset "exact degeneracy with a coupling: undefined per state, defined for the block" begin
    c = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(0), E_2=0.0); sys = dc_solve(c)     # energies 0, 4, 4
    d = quiet(() -> derivatives(sys; wrt=:E_2))
    b = only(blocks(d))
    @test b.exact && b.states == [2, 3] && b.detuning <= 1e-9 && b.coupling ≈ 0.5 atol=1e-12
    @test isnan.(energies(d, :E_2)) == [false, true, true]
    @test abs(tr(b.H1[:E_2])) < 1e-12                                                    # the block-invariant first derivative
    @test block_levels(b, (E_2=0.1,)).energies ≈ energies(dc_solve(c, shifted(c.θ, :E_2, 0.1)))[2:3] atol=1e-12
    @test isempty(blocks(derivatives(sys; wrt=:n_g)))                                    # degenerate but uncoupled: not a block
    # −cos φ couples |−1⟩ and |+1⟩ only through |0⟩: no block at first order, an exact block at second order, whose
    # effective Hamiltonian is 4 + (δ²/16) [1 1; 1 1], with levels 4 and 4 + δ²/8
    @test isempty(blocks(derivatives(sys; wrt=:E_1)))
    d2 = quiet(() -> derivatives(sys; wrt=:E_1, order=2))
    b2 = only(blocks(d2))
    @test b2.exact && b2.states == [2, 3] && b2.parameters == (:E_1,) && b2.order == 2
    @test isnan.(energies(d2, :E_1, :E_1)) == [false, true, true] && energies(d2, :E_1, :E_1)[1] ≈ -0.25 atol=1e-12
    @test all(isfinite, energies(d2, :E_1))
    @test block_levels(b2, (E_1=0.1,)).energies ≈ [4.0, 4.0 + 0.1^2 / 8] atol=1e-12
    @test block_levels(b2, (E_1=0.1,)).energies ≈ energies(dc_solve(c, shifted(c.θ, :E_1, 0.1)))[2:3] atol=1e-6
end
```

- [ ] **Step 2: Run and confirm failure** (`UndefVarError: blocks`).

- [ ] **Step 3: Implement** the block search, the effective Hamiltonian pieces, `blocks` and `block_levels`; export the three names.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivatives_blocks.jl","test_derivatives_second.jl","test_derivatives_iterative.jl"])`, then the default tier. Expected: PASS.

- [ ] **Step 5: Commit** (`Near-degenerate blocks: report and effective Hamiltonian`), with `src/derivblocks.jl src/derivatives.jl src/KronFreeCircuits.jl test/test_derivatives_blocks.jl`.

---

### Task 10: Children whose basis or operators are stored data

**Files:**
- Modify: `src/derivatives.jl` (the passes: augmented nodes, given systems; the two `ArgumentError`s of Task 5 are replaced)
- Test: `test/test_derivatives_stored.jl` (fast tier; estimated 20 s)

**Interfaces:**
- Consumes: `AugmentedChild` with its fields `node`, `solved`, `basis`, `terms`, `θ` (`src/augment.jl`, after the fix wave); `augment`, `bare_child`, `UnionReport`, `sys.augmented`; `_resolvent(mn; projected = true)` (Task 4); `given_independent` (Task 5).
- Produces: no new name.

Behaviour:
- **Augmented node.**
  Its stored states diagonalize its bare operator projected on its fixed union basis, so they are the whole spectrum of that projected problem.
  The pass treats it as a node on `a.solved.children` with the terms `a.terms` (not the tree's routed terms at that node) and `_resolvent(mn; projected = true)`: the sum over its own kept states, no complement, no re-diagonalization.
  Everything else is the ordinary rule: `∂H/∂x` is the derivative of the bare operator (its terms' coefficients and, when its children are nodes, their port and energy derivatives, computed by the ordinary pass on the terms the tree routes to them), and the gauge is the node's own phase rule.
  This is the derivative of "the bare operator projected on the stored basis"; the basis coefficients `a.basis` do not move.
- Before the first pass, for each augmented child `a` and every parameter that a term inside its subtree depends on (`a.terms` and the routed terms of every node below it), `a.θ` and `sys.θ` must hold the same value; otherwise an `ArgumentError` names the parameter and the node's modes.
  (`materialize` checks only the parameters of `a.terms`.)
- **`GivenSystem` child.**
  Without `given_independent = true`, `derivatives` throws an `ArgumentError` naming the given system's mode and the keyword: its energies and operators are numbers whose dependence on the parameters is unknown.
  With it, the child has no derivative data (`nothing`), like a basis.
- The docstring of `derivatives` gains both rules, and one sentence on Born–Oppenheimer reduced specs: a reduced spec is an ordinary spec, a `slow_spec` carries no dependence on the fast parameters, and a `fit_surfaces` spec carries the fitted-over parameters through its fit model, so derivatives with respect to them are derivatives of the fit.

- [ ] **Step 1: Write the failing tests** in `test/test_derivatives_stored.jl`

```julia
using Test, LinearAlgebra, Random, KronFreeCircuits
using KronFreeCircuits: GivenSystem, Term, Factor, Coefficient, Port, UnionReport

@testset "an augmented leaf child: derivatives at fixed basis" begin
    c = dc_zero_pi(); terms = circuit_terms(c.spec); rep = UnionReport(["bare"], [10], [0])
    bare(θ) = quiet(() -> bare_child(c.root, terms, θ, c.B))
    U = bare(c.θ).vectors * Matrix(qr(randn(MersenneTwister(5), ComplexF64, 10, 10)).Q)       # fixed for every θ
    cache = Dict{Any,Any}()                                                                   # one set of solves for every comparison
    at(θ) = get!(() -> quiet(() -> materialize(c.root, c.spec, θ; augmented=[augment(c.root, terms, θ, c.B, bare(θ), U, rep)])), cache, θ)
    sys = at(c.θ)
    for (x, h) in ((:E_L, 2.5e-4), (:E_CJ, 1e-2), (:E_J, 1e-3), (:varphi_ext, 1e-3))          # E_L = 0.04 and E_CJ = 20 set their steps
        d = derivatives(sys; wrt=x)
        check_fd(energies(d, x), fd_derivative(θ -> energies(at(θ)), c.θ, x; h); rtol=1e-6, atol=1e-9)
        check_fd(energies(d, x, c.B), fd_derivative(θ -> energies(at(θ), c.B), c.θ, x; h); rtol=1e-6, atol=1e-9)
        check_fd(eigenvectors(d, x, c.B), fd_derivative(θ -> eigenvectors(at(θ), c.B), c.θ, x; h); rtol=1e-6, atol=1e-9)
        check_fd(port(d, x, :phi, phiport()), fd_derivative(θ -> port(at(θ), :phi, phiport()), c.θ, x; h); rtol=1e-6, atol=1e-9)
    end
    d2 = derivatives(sys; wrt=(:E_L, :E_J), order=2, mixed=((:E_J, :E_L),))
    check_fd(energies(d2, :E_J, :E_L), fd_mixed(θ -> energies(at(θ)), c.θ, :E_J, :E_L; hx=2e-2, hλ=5e-4); rtol=1e-5, atol=1e-6)
end

@testset "an augmented branch child, whose own children move" begin
    c = dc_chain3(); terms = circuit_terms(c.spec); rep = UnionReport(["bare"], [10], [0])
    bare(θ) = quiet(() -> bare_child(c.root, terms, θ, c.inner))
    U = Matrix{ComplexF64}(bare(c.θ).vectors)                                                 # 16 × 10 coefficients, fixed
    aug(θ) = augment(c.root, terms, θ, c.inner, bare(θ), U, rep)
    cache = Dict{Any,Any}()
    at(θ) = get!(() -> quiet(() -> materialize(c.root, c.spec, θ; augmented=[aug(θ)])), cache, θ)
    sys = at(c.θ)
    for x in (:g12, :g23, :E_J1)
        d = derivatives(sys; wrt=x)
        check_fd(energies(d, x), fd_derivative(θ -> energies(at(θ)), c.θ, x; h=1e-3); rtol=1e-6, atol=1e-9)
        check_fd(port(d, x, :varphi_1, nport()), fd_derivative(θ -> port(at(θ), :varphi_1, nport()), c.θ, x; h=1e-3); rtol=1e-6, atol=1e-9)
    end
    # the augmented child was built at another value of a parameter that only a node below it uses: materialize does not
    # see that (it checks the child's own bare terms); the derivative pass must refuse
    θ1 = shifted(c.θ, :E_J1, 0.5)
    @test_throws ArgumentError derivatives(quiet(() -> materialize(c.root, c.spec, θ1; augmented=[aug(c.θ)])); wrt=:g23)
end

@testset "a given system is refused unless declared parameter-independent" begin
    g1 = GivenSystem(:a; energies=[0.0, 1.0, 2.5], operators=Dict(:x => ComplexF64[0 1 0; 1 0 √2; 0 √2 0]))
    g2 = GivenSystem(:b; energies=[0.0, 0.8], operators=Dict(:x => ComplexF64[0 1; 1 0]))
    top = Node([g1, g2]; solver=Dense(), keep=lowest(6), ports=[(:a, Port(:x, 0))])
    terms = [Term(Coefficient("g", [:g]), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])]
    gsolve(θ) = materialize(top, terms, θ)
    θ = (g=0.1,); sys = gsolve(θ)
    @test_throws "given_independent" derivatives(sys; wrt=:g)
    d = derivatives(sys; wrt=:g, order=2, given_independent=true)
    check_fd(energies(d, :g), fd_derivative(t -> energies(gsolve(t)), θ, :g; h=1e-3); rtol=1e-6)
    check_fd(energies(d, :g, :g), fd_second(t -> energies(gsolve(t)), θ, :g; h=1e-2); rtol=1e-5)
    check_fd(port(d, :g, :a, Port(:x, 0)), fd_derivative(t -> port(gsolve(t), :a, Port(:x, 0)), θ, :g; h=1e-3); rtol=1e-6)
end
```

- [ ] **Step 2: Run and confirm failure** (the `ArgumentError`s of Task 5).

- [ ] **Step 3: Implement** the three rules.

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivatives_stored.jl","test_augment.jl","test_given.jl"])`, then the default tier. Expected: PASS.

- [ ] **Step 5: Commit** (`Derivatives through augmented children and declared-constant given systems`), with `src/derivatives.jl test/test_derivatives_stored.jl`.

---

### Task 11: The symbolic KITE fixture, and the KITE acceptance tests

**Files:**
- Create: `test/python/generate_kite_symbolic.jl`, `test/fixtures/kite/optimum_symbolic.json` (generated, committed), `test/test_derivatives_kite.jl`
- Modify: `test/runtests.jl` (add `"test_derivatives_kite.jl"` to `SLOW_FILES`)

**Interfaces:**
- Consumes: `derivatives`, `blocks`, `block_levels`, the accessors (Tasks 5 to 9); `test/python/kite.jl` (`kite()`); `Spec(res; frame, subs)` of the PythonCall extension; `save_spec`.
- Produces: the fixture, with parameters `delta` (the kite flux offset from half flux, in radians), `n_g1` and `E_J`, in the paper's optimum frame.

- [ ] **Step 1: Write the generator** `test/python/generate_kite_symbolic.jl`, on the pattern of `test/python/generate_kite_fixtures.jl` (a generator, not a test file; its header says how to run it and that the fixture is regenerated only on purpose):

```julia
using KronFreeCircuits, PythonCall, LinearAlgebra
include("kite.jl")
res, θ, frames = kite()
spec = Spec(res; frame=frames.optimum, subs=(E_CJ=4.82, E_CS=0.072, E_L=1.27, epsilon=0.03))
save_spec(normpath(joinpath(@__DIR__, "..", "fixtures", "kite", "optimum_symbolic.json")), spec)
println("parameters: ", spec.parameters)
# The four lowest levels on the tree of test/test_derivatives_kite.jl at a generic point, to be pinned in that file.
bases = [:varphi_1 => ChargeBasis(16), :varphi_2 => PhaseGrid(-18.915907192616967, 18.915907192616967, 120; order=8),
         :varphi_3 => PhaseGrid(-17.714011073078225, 17.714011073078225, 120; order=8)]
root = Node([Node([m => b]; solver=Dense(), keep=lowest(12)) for (m, b) in bases]; solver=Dense(), compute=6, keep=lowest(4))
println("levels at (E_J = 16.83, delta = 0.0, n_g1 = 0.1): ", repr(energies(materialize(root, spec, (E_J=16.83, delta=0.0, n_g1=0.1)))))
```

- [ ] **Step 2: Run it** (the Python tier's interpreter; nothing may be installed into that conda environment):

`JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/home/eweissler/anaconda3/envs/cq-z4/bin/python $HOME/.juliaup/bin/julia --project=test/python test/python/generate_kite_symbolic.jl`

Expected: it prints `parameters: [:delta, :n_g1, :E_J]`, writes a file of about 1.7 kB whose `flux_linear` entries contain `delta` and whose two cosines have energies `(1.03 * E_J)` and `(0.97 * E_J)`, and prints four levels.
This fixture carries the flux on the inductors (`flux_linear`), while the numeric `kite/optimum.json` carries it in another, equivalent gauge, so the two are not compared coefficient by coefficient; Step 3's first testset pins the fixture by its levels instead.
The pinned levels are the ones this run prints: copy them, to all printed digits, into `KITE_SYM_LEVELS` of the test file, with the comment naming this command.
They are taken at `n_g1 = 0.1` on purpose: at `n_g1 = 0` the compact leaf's kept charge states are degenerate pairs and its twelfth kept state is one of a pair, so the levels there would pin an arbitrary choice.
For comparison only: on a scratch copy of the fixture the plan's author measured `[-9.510804644179908, -9.497205434969919, -8.425747834344323, -8.308985980043891]`; a difference above `1e-6` from what the generator prints is reported, and the generator's numbers are the ones committed.
If the interpreter or circuit-quantization is unavailable, stop and report; do not write the fixture by hand.

- [ ] **Step 3: Write the tests** in `test/test_derivatives_kite.jl` and list the file in `SLOW_FILES`

```julia
using Test, LinearAlgebra, KronFreeCircuits

# The KITE at the paper's optimum frame with the flux offset `delta`, the offset charge `n_g1` and `E_J` symbolic
# (test/python/generate_kite_symbolic.jl).  The tree: three leaves on the reference calculation's bases, twelve states
# each, and a root of dimension 1728.  Twelve states per leaf leave the levels 13 to 33 MHz above the paper's, which does
# not matter here: the derivatives are those of this calculation.
const KITE_SYM = load_spec(fixture("kite/optimum_symbolic.json"))
function kite_deriv_tree(solver; retain=:auto)
    bases = [:varphi_1 => ChargeBasis(16), :varphi_2 => PhaseGrid(-18.915907192616967, 18.915907192616967, 120; order=8),
             :varphi_3 => PhaseGrid(-17.714011073078225, 17.714011073078225, 120; order=8)]
    return Node([Node([m => b]; solver=Dense(), keep=lowest(12)) for (m, b) in bases]; solver, compute=6, keep=lowest(4), retain)
end
ksolve(root, θ) = quiet(() -> materialize(root, KITE_SYM, θ))

# The four lowest levels at (E_J = 16.83, delta = 0.0, n_g1 = 0.1) on this tree, as printed by
#     JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=test/python test/python/generate_kite_symbolic.jl
# when the fixture was generated; regenerate both together.
const KITE_SYM_LEVELS = [#= the four numbers the generator printed =#]

@testset "the symbolic KITE fixture" begin
    @test KITE_SYM.parameters == [:delta, :n_g1, :E_J]
    E = energies(ksolve(kite_deriv_tree(Dense()), (E_J=16.83, delta=0.0, n_g1=0.1)))
    @test length(KITE_SYM_LEVELS) == 4 && E ≈ KITE_SYM_LEVELS atol=1e-7           # the fixture is the one that was generated
    @test maximum(abs, E - [-9.523995, -9.510525, -8.453790, -8.337862]) < 0.04   # evidence: the paper's levels, to this tree's truncation
end

@testset "the KITE doublet: block, point derivatives, effective Hamiltonian (design acceptance 7)" begin
    θ = (E_J=16.83, delta=0.0, n_g1=0.1)            # n_g1 = 0 would make the compact leaf's kept charge states degenerate pairs
    root = kite_deriv_tree(LOBPCG(tol=1e-11)); sys = ksolve(root, θ); E0 = energies(sys)
    Δ = E0[2] - E0[1]
    @test 0.010 < Δ < 0.020                         # the 13.5 MHz doublet, on this tree
    d = derivatives(sys; wrt=(:delta, :E_J), order=2, mixed=((:E_J, :delta),), scale=(delta=0.01,))
    b = only(b for b in blocks(d) if b.node === root && b.parameters == (:delta,) && 1 in b.states)
    @test b.states == [1, 2] && !b.exact && b.detuning ≈ Δ atol=1e-9 && b.coupling > 1.0
    # point derivatives against finite differences at a step small against Δ/c ≈ 8e-3; with so small a step the finite
    # difference itself is good to about 1e-6 at first order and 1e-5 at second (measured), hence rtol = 1e-5 and 1e-4
    cache = Dict{Any,Any}(); f(t) = get!(() -> energies(ksolve(root, t)), cache, t)
    check_fd(energies(d, :delta), fd_derivative(f, θ, :delta; h=2e-4); rtol=1e-5)
    E2 = energies(d, :delta, :delta)
    check_fd(E2, fd_second(f, θ, :delta; h=2e-4); rtol=1e-4)
    @test E2[2] - E2[1] ≈ 4 * b.coupling^2 / b.detuning rtol=1e-2          # the curvature 4c²/Δ of the transition frequency
    # the effective Hamiltonian at displacements up to δx against a full re-solve; the Taylor expansion fails there
    for δ in (0.01, 0.005)
        @test block_levels(b, (delta=δ,)).energies ≈ f(shifted(θ, :delta, δ))[1:2] atol=1e-4
    end
    taylor = E0 .+ 0.01 .* energies(d, :delta) .+ 0.5 * 0.01^2 .* E2
    @test maximum(abs, (taylor - f(shifted(θ, :delta, 0.01)))[1:2]) > 1e-3
    # the full-spectrum route gives the same derivatives
    dd = derivatives(ksolve(kite_deriv_tree(Dense(); retain=:always), θ); wrt=:delta, order=2)
    @test energies(dd, :delta) ≈ energies(d, :delta) rtol=1e-6
    @test energies(dd, :delta, :delta) ≈ E2 rtol=1e-5
    @info "KITE linear solves: largest iteration count" maximum(s.iterations for s in d.linear_solves)
end

@testset "a mixed derivative for dephasing off the sweet spot (design acceptance 8)" begin
    θ = (E_J=16.83, delta=0.05, n_g1=0.1)
    root = kite_deriv_tree(Dense(); retain=:always); sys = ksolve(root, θ)
    d = derivatives(sys; wrt=(:E_J, :delta), order=2, mixed=((:E_J, :delta),))
    f01(E) = E[2] - E[1]
    m = f01(energies(d, :E_J, :delta))
    @test m ≈ 0.04277521 rtol=1e-5                                       # measured by finite differences when the plan was written
    check_fd(m, fd_mixed(t -> f01(energies(ksolve(root, t))), θ, :E_J, :delta; hx=1e-2, hλ=1e-3); rtol=1e-5)
end
```

- [ ] **Step 4: Run** `Pkg.test(test_args=["test_derivatives_kite.jl"])`, then the default tier (which must not include the new file).
Expected: PASS; the file takes one to three minutes.
A tolerance of the second testset that fails is reported with the measured number; it is not loosened.

- [ ] **Step 5: Commit** (`KITE: a symbolic fixture; the doublet's derivatives, block and effective Hamiltonian`), with `test/python/generate_kite_symbolic.jl test/fixtures/kite/optimum_symbolic.json test/test_derivatives_kite.jl test/runtests.jl`.

---

### Task 12: The outer rule: `spectrum` and the ChainRulesCore extension

**Files:**
- Create: `src/spectrum.jl` (included after `derivblocks.jl`), `ext/KronFreeCircuitsChainRulesCoreExt.jl`, `test/chainrules/Project.toml`, `test/chainrules/runtests.jl`
- Modify: `Project.toml` (weak dependency, extension, compat), `src/KronFreeCircuits.jl` (include, export `spectrum`)
- Test: `test/test_spectrum.jl` (fast tier, no autodiff), `test/chainrules/runtests.jl` (its own tier)

**Interfaces:**
- Consumes: `materialize`, `energies`, `port`, `derivatives` and its accessors.
- Produces, exported, in the core with no autodiff code:

```julia
# The kept energies of the root and the requested root ports at θ, as copies: the one function the differentiation rules cover.
# `ports` are (mode, port) pairs that the root forms (list them in the root's `ports`); `ports[i]` of the result is
# `port(sys, ports[i]...)`.  `derivative_options` are keywords passed to `derivatives` by the rules (`lintol`, `given_independent`, …).
spectrum(root::Node, spec::Spec, θ::NamedTuple; ports::AbstractVector = Tuple{Symbol,Port}[],
         derivative_options::NamedTuple = (;)) -> (; energies::Vector{Float64}, ports::Vector{<:Matrix})
```

- Produces, in the extension: `ChainRulesCore.rrule(::typeof(spectrum), root::Node, spec::Spec, θ::NamedTuple; kwargs...)` and `ChainRulesCore.frule((_, _, _, Δθ), ::typeof(spectrum), root, spec, θ; kwargs...)`.
  Both materialize once and call `derivatives(sys; wrt = <the keys of θ that some term or the constant depends on>, derivative_options...)` once (the pullback lazily, on its first call).
  The pullback returns `(NoTangent(), NoTangent(), NoTangent(), Tangent{typeof(θ)}(; …))` with, for each key `x` of `θ`, `Σ_i Ē_i ∂E_i/∂x + Σ_p Re Σ_{ij} conj(P̄_p[i, j]) ∂P_p[i, j]/∂x` (zero for a key nothing depends on); a zero or missing cotangent of `energies`, of `ports` or of one port contributes nothing.
  The pushforward returns `Tangent` of the output with `energies = Σ_x Δθ_x ∂E/∂x` and `ports[p] = Σ_x Δθ_x ∂P_p/∂x`.
  Second derivatives are not differentiated through: a loss that needs them calls `derivatives` itself (design §4.6).

- [ ] **Step 1: Write the failing core test** `test/test_spectrum.jl`

```julia
using Test, KronFreeCircuits
@testset "spectrum" begin
    c = dc_two_transmons()
    y = spectrum(c.root, c.spec, c.θ; ports=[(:varphi_1, nport())])
    sys = materialize(c.root, c.spec, c.θ)
    @test y.energies == energies(sys) && y.ports[1] == port(sys, :varphi_1, nport()) && y.ports[1] !== port(sys, :varphi_1, nport())
    @test_throws ArgumentError spectrum(c.root, c.spec, c.θ; ports=[(:varphi_2, n2port())])     # not formed at the root
end
```

Run `Pkg.test(test_args=["test_spectrum.jl"])` and confirm it fails with `UndefVarError: spectrum`.

- [ ] **Step 2: Implement `spectrum`**, export it, run `Pkg.test(test_args=["test_spectrum.jl"])`. Expected: PASS.

- [ ] **Step 3: Declare the weak dependency.**
ChainRulesCore is already loaded in every session of this package, as a dependency of ITensors (it is in the Manifest and in `Base.loaded_modules` after `using KronFreeCircuits`), so the extension loads without any action by a user; the core's own `[deps]` still do not name it.
Edit the root `Project.toml` by hand: under `[weakdeps]` add `ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"`; under `[extensions]` add `KronFreeCircuitsChainRulesCoreExt = "ChainRulesCore"`; under `[compat]` add `ChainRulesCore = "1"`.
`[deps]` is not touched.
Then run `$HOME/.juliaup/bin/julia --project -e 'using Pkg; Pkg.resolve()'` and report what it printed.
Append to `test/test_spectrum.jl` the test that the core's own dependencies name no autodiff package (it reads the file, so it can fail):

```julia
@testset "the core depends on no autodiff package" begin
    text = read(joinpath(pkgdir(KronFreeCircuits), "Project.toml"), String)
    deps = match(r"\[deps\]\n(.*?)\n\["s, text * "\n[").captures[1]           # the [deps] table alone
    weak = match(r"\[weakdeps\]\n(.*?)\n\["s, text * "\n[").captures[1]
    @test !occursin("ChainRulesCore", deps) && !occursin("Zygote", text) && !occursin("ForwardDiff", text)
    @test occursin("ChainRulesCore", weak)
end
```


- [ ] **Step 4: Create the test environment.**
Write `test/chainrules/Project.toml` by hand:

```toml
[deps]
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
KronFreeCircuits = "6ef25c9c-5ef5-4b26-8a4c-126bf0368fad"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[sources]
KronFreeCircuits = {path = "../.."}
```

Run `$HOME/.juliaup/bin/julia --project=test/chainrules -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'` and report everything Pkg printed, including the versions installed.
These two commands (and the `Pkg.resolve()` of Step 3) are the only Pkg operations of this task: no `Pkg.add`, `Pkg.rm`, `Pkg.gc` or `Pkg.update` (a `Pkg.rm` on this machine once triggered a collection that deleted 151 package installations).
Zygote is not in this machine's depot, so the instantiate downloads it.
If Zygote cannot be resolved or precompiled on Julia 1.13.2: delete the line `Zygote = …` from `test/chainrules/Project.toml` by hand, delete the generated, untracked `test/chainrules/Manifest.toml`, run the same command again, mark the Zygote testset of Step 5 with `@test_skip`, and report the error text; the direct tests of the rules stay.

- [ ] **Step 5: Write the failing tests** `test/chainrules/runtests.jl` (header comment: the command of Step 7, and that `Pkg.test()` never includes this directory)

```julia
using Test, LinearAlgebra, KronFreeCircuits, ChainRulesCore, Zygote
include("../helpers.jl")                     # it loads JSON3 and Random at top level, hence the two entries in the Project

@test Base.get_extension(KronFreeCircuits, :KronFreeCircuitsChainRulesCoreExt) !== nothing
key = [(:varphi_1, nport()), (:varphi_1, expiphi(1.0))]
loss(y) = (y.energies[2] - y.energies[1] - 5.0)^2 + 0.3 * y.energies[4] + abs2(y.ports[1][1, 2]) + real(y.ports[2][2, 3]) - 2 * imag(y.ports[2][1, 3])

@testset "rrule and frule of spectrum against finite differences" for periodic in (false, true)
    c = dc_two_transmons(; periodic)
    F(θ) = loss(spectrum(c.root, c.spec, θ; ports=key))
    fd = [first(fd_derivative(F, c.θ, x; h=1e-3)) for x in keys(c.θ)]
    # the pullback, fed the loss's cotangent by hand
    y, back = ChainRulesCore.rrule(spectrum, c.root, c.spec, c.θ; ports=key)
    Ē = zeros(length(y.energies)); Ē[2] = 2 * (y.energies[2] - y.energies[1] - 5.0); Ē[1] = -Ē[2]; Ē[4] = 0.3
    P̄1 = zeros(ComplexF64, size(y.ports[1])); P̄1[1, 2] = 2 * y.ports[1][1, 2]
    P̄2 = zeros(ComplexF64, size(y.ports[2])); P̄2[2, 3] = 1; P̄2[1, 3] = -2im
    θ̄ = back(Tangent{typeof(y)}(; energies=Ē, ports=[P̄1, P̄2]))[4]
    @test [θ̄[x] for x in keys(c.θ)] ≈ fd rtol=1e-6
    # the pushforward along each parameter
    for (i, x) in enumerate(keys(c.θ))
        Δθ = Tangent{typeof(c.θ)}(; NamedTuple{(x,)}((1.0,))...)
        _, ẏ = ChainRulesCore.frule((NoTangent(), NoTangent(), NoTangent(), Δθ), spectrum, c.root, c.spec, c.θ; ports=key)
        @test ẏ.energies ≈ first(fd_derivative(θ -> spectrum(c.root, c.spec, θ; ports=key).energies, c.θ, x; h=1e-3)) rtol=1e-6 atol=1e-9
    end
    # an engine, on a loss written in plain Julia
    g = Zygote.gradient(F, c.θ)[1]
    @test [g[x] for x in keys(c.θ)] ≈ fd rtol=1e-6
end
```

- [ ] **Step 6: Implement the extension.**

- [ ] **Step 7: Run** `$HOME/.juliaup/bin/julia --project=test/chainrules test/chainrules/runtests.jl`, then the default tier.
Expected: PASS in both; the default tier's environment is not changed (no entry is added to `test/Project.toml`).

- [ ] **Step 8: Commit** (`spectrum, and its rrule and frule in a ChainRulesCore extension`), with `Project.toml src/spectrum.jl src/KronFreeCircuits.jl ext/KronFreeCircuitsChainRulesCoreExt.jl test/test_spectrum.jl test/chainrules/Project.toml test/chainrules/runtests.jl` (no `Manifest.toml`: they are ignored by `.gitignore`).

---

### Task 13: The optimization demonstration

**Files:**
- Create: `bench/derivative_design/Project.toml`, `bench/derivative_design/design.jl`, `bench/derivative_design/README.md`

**Interfaces:**
- Consumes: `spectrum` and its `rrule` through Zygote (Task 12). Produces no API.

- [ ] **Step 1: Create the environment.**
Write `bench/derivative_design/Project.toml` by hand with `[deps]` `KronFreeCircuits = "6ef25c9c-5ef5-4b26-8a4c-126bf0368fad"`, `LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"`, `Optim = "429524aa-4258-5aef-a3af-852621145aeb"`, `Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"`, `Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"`, and `[sources]` `KronFreeCircuits = {path = "../.."}`.
Run `$HOME/.juliaup/bin/julia --project=bench/derivative_design -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'` and report what Pkg printed.
No other Pkg operation; no `Pkg.rm`, `Pkg.gc`, `Pkg.update`.
If Task 12 had to drop Zygote, drop it here the same way, and compute the gradient in Step 2 from `ChainRulesCore.rrule(spectrum, …)` with the loss's cotangent written by hand (add `ChainRulesCore` to `[deps]` instead); say so in the README.

- [ ] **Step 2: Write `design.jl`**, readable top to bottom.

The problem: recover a planted design of two coupled transmons from its observables, three parameters.
The tree is two leaves on `ChargeBasis(30)` keeping 4 states and a `Dense` root keeping 6 with `ports = [(:varphi_1, nport())]`, on `test/fixtures/two_transmons.json`, with `E_C1 = 1.2, E_C2 = 1.0, n_g1 = 0.21, n_g2 = 0.09` fixed.
The free parameters are `x = (E_J1, E_J2, g)`.
The features of a point are `φ(x) = [E_2 − E_1, E_3 − E_1, |n_1[1, 2]|]` from `spectrum`, the targets are `φ(x*)` at the planted `x* = (28.0, 24.0, 0.25)`, and the loss is `L(x) = Σ ((φ(x) − φ(x*)) ./ [1.0, 1.0, 0.1]).^2`.
The start is `x0 = (30.0, 25.0, 0.3)`.
The optimizer is `Optim.BFGS()` with the gradient from `Zygote.gradient` of `L` (through the `rrule`), `g_tol = 1e-10`, at most 200 iterations.
The script prints the loss and `x` per iteration, checks the gradient against central finite differences (step `1e-4`, once extrapolated) at `x0` and at the final point, and exits nonzero unless: the gradient at `x0` agrees to `1e-6` relative; the final loss is below `1e-12`; the final `x` is within `1e-4` of `x*` in each component; and at the final point the gradient and its finite difference are both below `1e-5` in the largest component.

- [ ] **Step 3: Run it**: `$HOME/.juliaup/bin/julia --project=bench/derivative_design bench/derivative_design/design.jl`.
Expected: exit code 0.
If it converges to another point with a loss below `1e-12` (the features need not determine `x` uniquely), report that point and its features; do not change the targets to make the check pass.

- [ ] **Step 4: README**: the command, the problem in three lines, the first and last lines of the iteration log, the two gradient checks with their numbers, the number of loss and gradient evaluations, and the wall time.

- [ ] **Step 5: Commit** (`bench/derivative_design: a design problem solved with the outer rule`), with the three files.

---

### Task 14: A device test of the derivative pass

**Files:**
- Create: `test/cuda/test_derivatives_device.jl`
- Modify: `test/cuda/runtests.jl` (include the new file), and `src/resolvent.jl` or `src/derivatives.jl` only where the device run exposes host-only code in the complement solve

**Interfaces:**
- Consumes: `derivatives` at an iterative node (Task 6) with `order = 2` and the two-parameter accessors (Task 8), `LOBPCG(on = CuArray)`, the `dc_*` circuits.

- [ ] **Step 1: Write the test** `test/cuda/test_derivatives_device.jl`

```julia
@testset "derivatives at an LOBPCG node on the device equal the host's full-spectrum route" begin
    tree(solver) = dc_two_transmons(; leafkeep=12, solver, compute=8, keep=lowest(6))            # root dimension 144
    cg, ch = tree(LOBPCG(tol=1e-11, on=CuArray)), tree(Dense())
    sg, sh = materialize(cg.root, cg.spec, cg.θ), materialize(ch.root, ch.spec, ch.θ)
    for x in (:g, :n_g1)
        dg = derivatives(sg; wrt=x, order=2); dh = derivatives(sh; wrt=x, order=2)
        @test energies(dg, x) ≈ energies(dh, x) rtol=1e-8 atol=1e-9
        @test energies(dg, x, x) ≈ energies(dh, x, x) rtol=1e-6 atol=1e-8
        @test norm(eigenvectors(dg, x) - eigenvectors(dh, x), Inf) <= 1e-6 * norm(eigenvectors(dh, x), Inf)
        @test norm(port(dg, x, :varphi_1, nport()) - port(dh, x, :varphi_1, nport()), Inf) <= 1e-6 * norm(port(dh, x, :varphi_1, nport()), Inf)
        @test !isempty(dg.linear_solves) && eigenvectors(dg, x) isa Matrix                      # results are host arrays
    end
end
```

- [ ] **Step 2: Check the GPU before running anything on it.**
Run `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv`.
If it lists any process that is not your own, do not run the device tier: commit the test file as written, and report this task's run as **skipped**, with the output of the command.
Never kill a process that is not your own.

- [ ] **Step 3: Run the device tier** (only when Step 2 found the GPU free): `$HOME/.juliaup/bin/julia --project=test/cuda test/cuda/runtests.jl`.
If the environment is not instantiated, `$HOME/.juliaup/bin/julia --project=test/cuda -e 'using Pkg; Pkg.instantiate()'` is allowed, and nothing else from Pkg.
Expected: PASS, including every existing device testset.
Where the complement solve fails on device arrays, fix the host-only operation in `src/` (the solve is specified with `mul!`, broadcasts and column sums only), rerun the default tier, and say what was changed.

- [ ] **Step 4: Commit** (`Device test: the derivative pass at an LOBPCG node on the GPU`), with `test/cuda/test_derivatives_device.jl test/cuda/runtests.jl` and any `src/` file changed.

---

### Task 15: Documentation, and one run of every tier

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md`, `README.md`, `docs/superpowers/STATUS.md`

- [ ] **Step 1: Architecture spec, amended in place with dated notes** (the pattern of the existing "Amended 2026-11-10 (Section 20j): …" lines; old text stays, the note says what replaces it):
  - §2, the non-goal "Forward-mode differentiation. Reverse mode first.": derivatives are computed upward (forward) by perturbation theory; a reverse sweep is the non-goal now.
  - §11, at its head and in §11.2, §11.3, §11.4: the three ChainRules primitives and Zygote are replaced by the upward pass of §20l; §11.1 stands; the §11.3 warning exists (the selection-boundary test); §11.4 is the block report.
  - §15: `ChainRulesCore` is a weak dependency with an extension, not a core dependency; `Zygote` is test-only, in `test/chainrules`; `FiniteDifferences` and `ChainRulesTestUtils` are not used.
  - One-line dated pointers to §20l at the other places that describe the old approach: §13 (the `Zygote.gradient(loss, θ)` usage), §13.1 (the surface slope), §16 (rule tests), §18 (build order), §20a (the allowlist and the data-valued `DiagonalOp`, which this project no longer blocks), §20b (flux derivatives through ForwardDiff), §20c decision 9.
  - A new subsection `### 20l. Derivatives by perturbation theory through the tree (2026-11-11)` after §20k and before §21: the equations of Tasks 4 and 7 (first and second order, gauge, normalization), the three contributions, what is retained and the measured cost (Task 2), the measured convergence of the sum over states (Task 4), the linear solve and its measured iteration counts (Tasks 6, 11), the block rule and the order and error of the effective Hamiltonian, stored-data children, the API, the outer rule, the measured results of Tasks 11 and 13, and every place the build departed from the design, each with its reason: at least the API form `derivatives(sys; …)`, `scale` as a keyword, the narrowing of the `:follow` restriction, and design §4.8's statement on reduced specs (a `fit_surfaces` spec keeps the fitted-over parameters, so derivatives with respect to them are derivatives of the fit, not an error).

- [ ] **Step 2: README.**
A section "Derivatives" before "Tests": a session of a dozen lines (`materialize`, `derivatives` with `wrt`, `order`, `mixed`, `scale`; `energies`, `eigenvectors`, `port`, `blocks`, `block_levels`), what is held fixed, the `retain` keyword and its cost, the restrictions (`:follow` phases, given systems, exact degeneracies as `NaN`), and `spectrum` with a three-line Zygote example.
In "Tests": the `test/chainrules` tier with its two commands.

- [ ] **Step 3: STATUS.md.**
Record what is done, what was measured, what was missed and why; replace the "Plan 5, automatic differentiation" entry and the lines that defer work to plan 5; under "Decisions still Eli's" keep selection cluster closure (not part of this project) and add: how `scale` is supplied (design §4.4), whether a sum over only the nearest states should replace full retention (design §4.5, with the measured table), and whether an augmented child's union basis should ever move.

- [ ] **Step 4: Run every tier once**, each to completion, and report each result as it is:
  - the default tier: `$HOME/.juliaup/bin/julia --project -e 'using Pkg; Pkg.test()'`;
  - the slow tier: `KRONFREE_SLOW_TESTS=1 $HOME/.juliaup/bin/julia --project -e 'using Pkg; Pkg.test()'`;
  - the Python tier: `JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/home/eweissler/anaconda3/envs/cq-z4/bin/python $HOME/.juliaup/bin/julia --project=test/python test/python/runtests.jl`;
  - the rules tier: `$HOME/.juliaup/bin/julia --project=test/chainrules test/chainrules/runtests.jl`;
  - the device tier, under the GPU rule of Task 14 (skipped and reported as skipped when another process is using the GPU);
  - `bench/derivative_design/design.jl` and `bench/retention_cost.jl`.
Expected: PASS; the slow tier's warnings are those `main` already prints plus the notices and warnings this plan introduces on purpose in its own tests.

- [ ] **Step 5: Commit** (`Docs: derivatives by perturbation theory through the tree`), with the three files.

---

## Acceptance map

| Design item | Task |
|---|---|
| §4.1 derivatives of the coefficients | 1 |
| §4.2 the per-node rule, first order (dense and iterative) | 4, 6 |
| §4.3 second order: normalization, phase, mixed pairs | 7, 8 |
| §4.4 near-degenerate levels | 4 (exact guard), 6 (boundary), 9 |
| §4.5 the API; retention and its cost; convergence of the sums | 5, 8; 2; 4 |
| §4.6 the outer rule | 12, 13 |
| §4.7 devices | 14 |
| §4.8 stored-data children | 10 |
| §5.1 per-node rules against finite differences (dense, iterative, real, complex, both orders, off-diagonal ports) | 4, 7 (dense); 6, 8 (iterative, real and complex, both orders: against the dense route and against finite differences, slow tier; the solver itself in the fast tier) |
| §5.2 whole-tree derivatives, two- and three-level trees | 5, 8 |
| §5.3 the three-contribution test | 3, 5 |
| §5.4 analytic checks (oscillator, transmon dispersion, fluxonium flux) | 4, 7 |
| §5.5 equivalence with the paper's formulas | 4 (first order), 7 (second energy derivative) |
| §5.6 iterative against dense | 6, 11, 14 |
| §5.7 near-degeneracy on the KITE doublet | 11 (and 9 on the cos 2φ doublet, fast tier) |
| §5.8 mixed derivative for dephasing: fluxonium, KITE off the sweet spot, augmented child | 8, 11, 10 |
| §5.9 the optimization demonstration | 13 |
| §5.10 architecture spec §11, §15, §2 and STATUS.md | 15 |

## Names this plan introduces

Exported: `differentiate` (Task 1); `derivatives`, `Derivatives` (Task 5); `blocks`, `NearDegenerateBlock`, `block_levels` (Task 9); `spectrum` (Task 12); new methods of `energies`, `eigenvectors`, `port` on a `Derivatives` (Tasks 5, 8); the `Node` keyword `retain` (Task 2).
Internal, used across tasks: `Retained`, `_retains`, `_RETAIN_DENSE_MAX` (2); `NodeDerivative`, `_derivative_operator` (two forms: 3, 7), `_sandwich`, `_derivative_op` (3); `Resolvent`, `_resolvent`, `_apply_resolvent` (returning `(W, clashes)`), `_peak_map`, `_peak_amps`, `_port_derivatives`, `_node_first_order` (4); `_pairkey`, `_node_derivative`, the keyword `_parts` (5); `_complement_solve`, the keyword `_truncate` (6); `_node_second_order`, `_port_second_derivatives` (7).
Test helpers in `test/helpers.jl`: `quiet`, `shifted`, `fd_derivative`, `fd_second`, `fd_mixed`, `fd_pair`, `check_fd` (1); `dc_transmon`, `dc_fluxonium`, `dc_oscillator`, `dc_two_transmons`, `dc_chain3`, `dc_zero_pi`, `dc_abc`, `dc_cos2phi`, `dc_solve`, `dc_memo`, `fd_child` (3); `node_setup`, `node_first_order` (4); `node_second_order` (7).
Test files: fast `test_diffcoeff.jl` (1), `test_retention.jl` (2), `test_derivoperator.jl` (3), `test_derivatives_node.jl` (4), `test_derivatives_tree.jl` (5), `test_derivatives_complement.jl` (6), `test_derivatives_second.jl` (7), `test_derivatives_second_tree.jl` (8), `test_derivatives_blocks.jl` (9), `test_derivatives_stored.jl` (10), `test_spectrum.jl` (12); slow (`SLOW_FILES`) `test_derivatives_iterative.jl` (6, 8), `test_derivatives_kite.jl` (11); own tiers `test/chainrules/runtests.jl` (12), `test/cuda/test_derivatives_device.jl` (14).

## Disagreements with the controller's decisions

- **D5, narrowed.**
  The decision refuses every node with `phase === :follow` or `keep isa Follow`.
  The plan refuses only a node whose phases were actually fixed by a previous solution, which Task 2 records (`Retained.followed`).
  A `:follow` node materialized without `previous` used the positive-sector rule and is differentiated correctly, and that is the decision's own workaround, which could not be told apart otherwise; a `follow` selection is a discrete choice held fixed like any other and needs no restriction.
- **D2, in what is stored.**
  The field holds all computed energies but only the vectors of the unkept computed states; the kept vectors are already in `vectors`, and storing them twice would double the memory of every iterative node that keeps all it computes.
- **D6, one added constraint.**
  Both members of a `mixed` pair must be in `wrt`: the pair needs both first-order responses, and returning them only for some parameters would be a surprise.
- **D8, as updated by the controller on 2026-11-11.**
  `materialize` checks an augmented child's `θ` only for the parameters of its own bare terms; the derivative pass also needs the nodes below it to be at `sys.θ`, so Task 10 repeats the check over the whole augmented subtree.
- **D8 / design §4.8, reduced specs.**
  The design says a derivative with respect to a parameter a reduced spec no longer carries is an error by construction, which holds for a `slow_spec`.
  A `fit_surfaces` spec keeps the fitted-over parameters through its fit model, so derivatives with respect to them are not an error: they are derivatives of the fit.
  The plan adds no special handling (as decided) and documents this (Tasks 10 and 15).
- **D10, a finding rather than a disagreement.**
  The symbolic fixture cannot be checked against `kite/optimum.json` coefficient by coefficient: generated through `Spec(res; frame, subs)` it carries the flux on the inductors, the numeric fixture in another gauge (the cosine coefficients differ in sign at the same values).
  Task 11 pins it by its levels on a fixed tree instead.
- **D9, two facts the decision did not have.**
  ChainRulesCore is already a transitive dependency (through ITensors) and is loaded in every session, so the extension is always active; the core's `[deps]` are unchanged, which is what the decision protects.
  Zygote is not installed in this machine's depot (ChainRulesCore, Optim and ForwardDiff are), and its support for Julia 1.13.2 is untested here; Tasks 12 and 13 state the fallback (the rules tested directly, the bench driven by the `rrule`).
- **D11, where the finite difference is not trustworthy.**
  Measured when the plan was written and again when it was revised: second differences reach `1e-8` to `8e-6` self-consistency on the test circuits, so second-order tests use `rtol = 1e-5` except where `1e-6` was measured to hold.
  On the KITE doublet the step must be small against `Δ/c ≈ 8e-3`, which leaves the finite difference good to about `1e-6` at first order and `1e-5` at second, so Task 11 uses `1e-5` and `1e-4` there.
  At symmetric points (`n_g = 0`), across a kept/unkept crossing, and where a node's peak index changes inside the stencil, finite differences of states are meaningless.
  The last was found on the first version of the test circuits (a transmon leaf with `E_J/E_C = 25` changes a peak index at `n_g ≈ 0.024, 0.16, 0.258`, and the circuits sat at `0.15` and `0.25`); the circuits now sit at `0.09` and `0.21`, and every tree comparison was re-measured there.
