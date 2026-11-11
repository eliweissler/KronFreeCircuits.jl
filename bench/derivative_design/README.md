# A design problem solved with the outer rule

`design.jl` recovers a planted design of two coupled transmons from three of their observables by minimizing a least-squares loss with BFGS, on the gradient that Zygote takes through the `rrule` of `spectrum` (the package's ChainRulesCore extension).
It checks that gradient against finite differences of the loss at the start and at the end, and exits nonzero if a pass criterion fails.

## Command

From the repository root (Julia 1.13; instantiate the project once with `julia --project=bench/derivative_design -e 'using Pkg; Pkg.instantiate()'`):

```
julia --project=bench/derivative_design bench/derivative_design/design.jl
```

The project depends on this package by path (`[sources]`), on Optim, Zygote, LinearAlgebra and Printf.
No `Manifest.toml` is committed (`Manifest*.toml` is ignored by git); the run below resolved Optim 2.3.2 and Zygote 0.7.16.
The script writes no files and has no options: everything it reports goes to standard output, and the full output of the recorded run is copied at the end of this file.
It sets one BLAS thread and does not use the GPU.
The bench is not part of any test tier: neither `Pkg.test()` (with or without `KRONFREE_SLOW_TESTS=1`) nor the Python, rules or device tiers run it.

## The problem

Two transmons, `Σ_i [4 E_Ci (n_i − n_gi)² − E_Ji cos φ_i] + g (n_1 − n_g1)(n_2 − n_g2)` (`test/fixtures/two_transmons.json`), with `E_C1 = 1.2`, `E_C2 = 1.0`, `n_g1 = 0.21`, `n_g2 = 0.09` fixed, as a tree of two leaves on `ChargeBasis(30)` keeping 4 states each and a `Dense` root keeping 6 with the port `(:varphi_1, nport())`.
The free parameters are `x = (E_J1, E_J2, g)`, the features are `φ(x) = [E_2 − E_1, E_3 − E_1, |n_1[1, 2]|]` from `spectrum`, and the loss is `L(x) = Σ ((φ(x) − φ(x*)) ./ [1, 1, 0.1])²` with the planted `x* = (28, 24, 0.25)`.
BFGS (`Optim.BFGS()`, `g_tol = 1e-10`, at most 200 iterations) starts at `x0 = (30, 25, 0.3)` with the gradient `Zygote.gradient(L, x)`.

The finite difference of the gradient is the central difference of `L` along each component with steps `h = 1e-4` and `h/2`, extrapolated once: `(4 D(h/2) − D(h)) / 3`.
The pass criteria: the gradient at `x0` agrees with the finite difference to `1e-6` relative to the largest component of the finite difference; the final loss is below `1e-12`; the final `x` is within `1e-4` of `x*` in each component; at the final `x` the largest component of the gradient and of its finite difference are both below `1e-5`.
How the optimizer stopped is printed and is not a criterion.

## Result

All numbers below are from one run, the command above on 2026-11-11 at 15:06 UTC, on a machine loaded by another 8-thread job, which ended with `all checks passed` and exit status 0.
An earlier run of the same script, before only the wording of two printed lines changed, printed the same numbers to every digit (the objective is deterministic).

The optimizer converged: it stopped on its gradient tolerance (termination code `GradientNorm`, `‖∇L‖∞ = 1.782e-11 ≤ 1e-10`) after 13 iterations, not on the iteration limit, a failed line search or a non-finite value.
It reached the planted design, not another point with the same features: the final `x − x* = (+1.302e-11, +1.341e-11, +7.870e-13)`, with final loss `3.796e-23`.

The first and last lines of the iteration log (iteration, loss, `x = (E_J1, E_J2, g)`, `‖∇L‖∞`):

```
     0  4.317770e-01  (30.0000000000, 25.0000000000, 0.3000000000)  6.709e-01
     1  4.107249e-01  (29.9780841615, 24.9874754337, 0.2536923654)  3.305e-01
   ...
    12  1.933407e-19  (28.0000000009, 24.0000000003, 0.2499999999)  2.123e-09
    13  3.795995e-23  (28.0000000000, 24.0000000000, 0.2500000000)  1.782e-11
```

The two gradient checks:

| point | gradient (Zygote) | finite difference | max difference | relative to max \|fd\| |
|---|---|---|---|---|
| `x0` | (3.1753360043e-01, 1.8146559321e-01, 6.7094078691e-01) | (3.1753360098e-01, 1.8146559258e-01, 6.7094078821e-01) | 1.299e-09 | 1.936e-09 |
| final `x` | (1.9034028511e-12, 2.7575282477e-12, 1.7823583065e-11) | (1.9195833127e-12, 2.7622083466e-12, 1.7935112198e-11) | 1.115e-13 | 6.218e-03 |

At `x0` the gradient agrees with the finite difference to `1.9e-9` relative, well inside `1e-6`.
At the final point both are about `1.8e-11` in the largest component, well inside `1e-5`; their relative difference there (`6.2e-3`) is not a criterion, since a finite difference of a loss of `4e-23` at step `1e-4` is dominated by round-off of the loss (about `1e-16` / `1e-4`).

Evaluations: Optim counted 16 evaluations of the loss and 16 of the gradient (each gradient also runs the loss's forward pass once, so the script counted 32 forward evaluations of the loss during the optimization).
The whole script made 59 evaluations of the loss (the 24 of the two finite-difference checks included) and 18 gradients.
Wall time: the optimization itself took 1.6 s; the script took 39.9 s by its own clock and 42.4 s from the shell, almost all of it compiling the package, Zygote's pullbacks and Optim on the first calls.

## The recorded run's full output

```
derivative_design: two coupled transmons, three parameters, BFGS on the gradient of the outer rule
Julia 1.13.2, 1 thread(s), BLAS threads 1
planted x* = [28.0, 24.0, 0.25]  start x0 = [30.0, 25.0, 0.3]
targets φ(x*) = [12.750006247879, 15.100236975967, 0.081077471207]
gradient check at x0, x = [30.0, 25.0, 0.3]
  gradient (Zygote)   = [+3.1753360043e-01, +1.8146559321e-01, +6.7094078691e-01]
  finite difference   = [+3.1753360098e-01, +1.8146559258e-01, +6.7094078821e-01]
  max |difference|    = 1.299e-09, relative to max |fd| = 1.936e-09
loss at x0 = 4.3177701484e-01

iterations (loss, x = (E_J1, E_J2, g), ‖gradient‖∞):
     0  4.317770e-01  (30.0000000000, 25.0000000000, 0.3000000000)  6.709e-01
     1  4.107249e-01  (29.9780841615, 24.9874754337, 0.2536923654)  3.305e-01
     2  2.550702e-01  (29.5517476676, 24.7770868427, 0.2512947756)  2.629e-01
     3  7.241746e-04  (27.9174218078, 23.9718145315, 0.2514772980)  4.989e-02
     4  9.340967e-06  (27.9993485946, 24.0097230301, 0.2493736298)  7.049e-03
     5  4.922278e-06  (27.9970030807, 24.0069288039, 0.2497628563)  1.196e-03
     6  2.565530e-06  (27.9968429383, 24.0042832118, 0.2499174901)  2.049e-03
     7  2.966064e-07  (27.9985474435, 24.0004454667, 0.2500534106)  1.949e-03
     8  1.932252e-08  (27.9996796594, 23.9998474494, 0.2500262591)  5.842e-04
     9  2.008763e-10  (27.9999913550, 23.9999574560, 0.2500029767)  4.045e-05
    10  9.274532e-13  (28.0000011211, 23.9999969183, 0.2500001164)  5.234e-07
    11  6.520661e-16  (28.0000000669, 23.9999999620, 0.2499999988)  7.306e-08
    12  1.933407e-19  (28.0000000009, 24.0000000003, 0.2499999999)  2.123e-09
    13  3.795995e-23  (28.0000000000, 24.0000000000, 0.2500000000)  1.782e-11

how the optimizer stopped: termination code GradientNorm; gradient tolerance met: true; x change zero: false; f change zero: false; iteration limit reached: false
the optimizer CONVERGED: ‖gradient‖∞ ≤ g_tol = 1e-10
iterations 13; Optim's evaluations: 16 of the loss, 16 of the gradient; optimization wall time 1.6 s
evaluations counted by the script during the optimization: 32 of the loss (one inside each gradient), 16 of the gradient
final x = [28.000000000013024, 24.000000000013408, 0.250000000000787]
final x - x* = [+1.302e-11, +1.341e-11, +7.870e-13]
final loss = 3.795995e-23
features at the final x = [12.750006247883, 15.100236975971, 0.081077471207]
gradient check at the final x, x = [28.000000000013024, 24.000000000013408, 0.250000000000787]
  gradient (Zygote)   = [+1.9034028511e-12, +2.7575282477e-12, +1.7823583065e-11]
  finite difference   = [+1.9195833127e-12, +2.7622083466e-12, +1.7935112198e-11]
  max |difference|    = 1.115e-13, relative to max |fd| = 6.218e-03

PASS  gradient at x0 agrees with finite differences to 1e-6 relative (1.936e-09)
PASS  final loss below 1e-12 (3.796e-23)
PASS  final x within 1e-4 of x* in each component (max 1.341e-11)
PASS  at the final x, max |gradient| (1.782e-11) and max |finite difference| (1.794e-11) below 1e-5

total evaluations: 59 of the loss (one inside each gradient), 18 of the gradient; total wall time 39.9 s
all checks passed
```
