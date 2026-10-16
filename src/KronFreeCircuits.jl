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
include("tree.jl")
include("solvers.jl")
include("selection.jl")
include("phase.jl")
include("materialize.jl")

end # module
