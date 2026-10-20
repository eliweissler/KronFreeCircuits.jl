using Test, LinearAlgebra, SparseArrays, KronFreeCircuits

include("helpers.jl")

for f in sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl") && f ∉ ("test_zero_pi_reference.jl", "test_born_oppenheimer_reference.jl"), readdir(@__DIR__)))
    include(f)
end

if get(ENV, "KRONFREE_SLOW_TESTS", "0") == "1"
    println("KRONFREE_SLOW_TESTS=1: including slow zero-pi tests")
    include("test_zero_pi_reference.jl")
    include("test_born_oppenheimer_reference.jl")
else
    println("KRONFREE_SLOW_TESTS not set: skipping slow zero-pi tests")
end
