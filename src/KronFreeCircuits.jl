module KronFreeCircuits

using LinearAlgebra, SparseArrays
using RuntimeGeneratedFunctions
using JSON3
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, get_backend

RuntimeGeneratedFunctions.init(@__MODULE__)

include("parameters.jl")
include("spec.jl")
include("terms.jl")
include("localops.jl")
include("bases.jl")
include("apply.jl")

end # module
