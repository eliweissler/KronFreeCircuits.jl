using Test, LinearAlgebra, SparseArrays, KronFreeCircuits

include("helpers.jl")

for f in sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(@__DIR__)))
    include(f)
end
