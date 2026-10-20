using Test, LinearAlgebra, SparseArrays, KronFreeCircuits

include("helpers.jl")

# Two tiers. The default runs the fast unit-level files (under a minute). KRONFREE_SLOW_TESTS=1 runs every file: also the
# end-to-end, solver, catalogue and Born–Oppenheimer files, and last the published-reference reproductions.
# `Pkg.test(test_args=["test_bo.jl", ...])` runs only the named files, in either tier.
const SLOW_FILES = ("test_apply.jl", "test_bo.jl", "test_catalogue.jl", "test_character.jl", "test_generated_specs.jl",
                    "test_materialize.jl", "test_solvers.jl")
const REFERENCE_FILES = ("test_zero_pi_reference.jl", "test_born_oppenheimer_reference.jl")

slow = get(ENV, "KRONFREE_SLOW_TESTS", "0") == "1"
regular = sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl") && f ∉ REFERENCE_FILES, readdir(@__DIR__)))
files = !isempty(ARGS) ? collect(ARGS) :
        slow ? [regular; collect(REFERENCE_FILES)] : filter(∉(SLOW_FILES), regular)
for f in files
    isfile(joinpath(@__DIR__, f)) || error("no test file $f in $(@__DIR__)")
end
println(isempty(ARGS) ? (slow ? "KRONFREE_SLOW_TESTS=1: running every test file" :
                                "KRONFREE_SLOW_TESTS not set: running the fast tier ($(length(files)) files)") :
                        "running the named test files")
for f in files
    include(f)
end
