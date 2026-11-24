# Development

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                                   # fast tier, a few minutes
KRONFREE_SLOW_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'             # every file
julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_bo.jl"])'           # only the named files
```

The default runs the fast tier, the unit-level files, in several minutes.
Setting `KRONFREE_SLOW_TESTS=1` runs every file:
- the apply, end-to-end, solver, catalogue, character, generated-spec and Born–Oppenheimer files (`SLOW_FILES` in `test/runtests.jl`);
- two real-versus-complex equivalence testsets gated inside those files;
- last, the published-reference reproductions: the two-coupled-zero-pi acceptance test against scqubits (about a minute) and the zero-pi Born–Oppenheimer reproduction (`test/test_born_oppenheimer_reference.jl`, about 15 s).

CI runs the fast tier on every push and pull request (on Julia 1.11 and the latest release, and at the lowest compatible dependency versions), the ChainRules, ITensorMPS and QuantumToolbox tiers below, and the slow tier weekly.
The CUDA and Python tiers need a GPU and circuit-quantization, so they run locally.

Run the slow tier before merging.
A named-file run skips the testsets gated inside those files unless `KRONFREE_SLOW_TESTS=1` is also set.

The Python tier tests the circuit-quantization interface and has its own environment, `test/python`; it needs the Python environment of "Defining circuits from Julia" (with circuit-quantization at commit `c8c8714` or later) and is never included by `Pkg.test()`.
Set the two environment variables for the `Pkg.instantiate()` too: precompiling the extension without them can make CondaPkg install a Python of its own.

```bash
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/path/to/env/bin/python julia --project=test/python -e 'using Pkg; Pkg.instantiate()'
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/path/to/env/bin/python julia --project=test/python test/python/runtests.jl
```

The default tier checks that the package itself stays free of Python: no file under `src/` loads or calls PythonCall, and the extension is not loaded.

The rules tier tests the `rrule` and `frule` of `spectrum` (the ChainRulesCore extension), directly and through Zygote; it has its own environment, `test/chainrules`, and is never included by `Pkg.test()`:

```bash
julia --project=test/chainrules -e 'using Pkg; Pkg.instantiate()'
julia --project=test/chainrules test/chainrules/runtests.jl
```

The ITensorMPS tier tests `Experimental.dmrg_levels` (the ITensorMPS extension) against exact diagonalization of a small chain; it has its own environment, `test/itensormps`, and is never included by `Pkg.test()`:

```bash
julia --project=test/itensormps -e 'using Pkg; Pkg.instantiate()'
julia --project=test/itensormps test/itensormps/runtests.jl
```

The device tests have their own environment and need a CUDA device:

```bash
julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/cuda test/cuda/runtests.jl
```


## Test fixtures

The tests use hand-written specs in `test/fixtures`.
They are cross-checked against specs that circuit-quantization generates from the corresponding lumped circuits (`test/test_generated_specs.jl`, using the committed files in `test/fixtures/generated/`, which this documentation's examples also use).
`test/fixtures/generate_specs.py` regenerates them; run it with circuit-quantization's Python environment only when regenerating, since Python is not a test dependency.
The documented convention differences between the two sets are asserted in the test.

## Benchmarks

Two benchmark scripts take their settings from environment variables, documented at the top of each file.
`bench_archive/parent_solve.jl` solves the audit's stored 300 × 300 parent cold and after a 1 % coupling step, with both preconditioners; `bench_archive/end_to_end.jl` materializes two coupled zero-pis and times each stage separately.
`BENCH_LEAF_SOLVER=shiftinvert`, `chebyshev` or `auto` picks the leaf solver.
`BENCH_ARITHMETIC=auto|complex` sets `arithmetic=` on every node of `bench_archive/end_to_end.jl`'s tree.

```bash
BENCH_SMOKE=1 julia --project=. -t 1 bench_archive/end_to_end.jl                                  # reduced sizes, CPU
BENCH_FIXTURE=path/to/larger-real300-operators.bin julia --project=. -t 1 bench_archive/parent_solve.jl
BENCH_DEVICE=gpu BENCH_FIXTURE=path/to/larger-real300-operators.bin julia --project=test/cuda -t 1 bench_archive/parent_solve.jl
```

## Building this documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The examples run during the build, with the working directory set to `test/fixtures/generated`, the specs circuit-quantization generates for the test circuits.
The site is in `docs/build`; the `Documentation` workflow builds it on every push and deploys `main` and each release to GitHub Pages.
