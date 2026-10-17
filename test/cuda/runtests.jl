using Test, LinearAlgebra, Random
using CUDA, cuTENSOR
using KronFreeCircuits

include("../helpers.jl")

@testset verbose=true "CUDA device" begin
    include("test_device.jl")
end
