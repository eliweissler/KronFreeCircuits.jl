module KronFreeCircuits

using LinearAlgebra, SparseArrays
using RuntimeGeneratedFunctions
using JSON3

RuntimeGeneratedFunctions.init(@__MODULE__)

include("parameters.jl")
include("spec.jl")
include("terms.jl")
include("localops.jl")

end # module
