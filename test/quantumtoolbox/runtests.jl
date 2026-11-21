using Test, LinearAlgebra
using CUDA, cuTENSOR
using QuantumToolbox
using KronFreeCircuits

include("../helpers.jl")

@testset verbose=true "QuantumToolbox extension" begin
    include("test_quantumtoolbox.jl")
end
