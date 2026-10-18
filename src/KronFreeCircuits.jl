module KronFreeCircuits

using LinearAlgebra, SparseArrays
using RuntimeGeneratedFunctions
using JSON3
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, get_backend
using ITensors: ITensors
using LOBPCGEigensolver: LOBPCGEigensolver
using Adapt: adapt
using Random: Random, randn!

RuntimeGeneratedFunctions.init(@__MODULE__)

export Spec, load_spec, parse_spec, Coefficient, Port, nport,
       n2port, phiport, phi2port, expiphi, exp2pin, Factor,
       Term, circuit_terms, LocalOp, DenseOp, DiagonalOp, BandedOp,
       ShiftOp, dense, diagonal, Basis, ChargeBasis, PhaseGrid,
       FockBasis, localop, apply_axis!, KronFreeOperator, Node, Dense,
       Lanczos, LOBPCG, lowest, indices, connected, follow, materialize,
       System, energies, port, eigenvectors, materialized, bare_labels,
       bare_overlaps, decompose, dressed_index, constant_term, SelectionRule,
       SelectionContext, GivenSystem, gershgorin_bound,
       separable_eigenbases, product_seed, infnorm

include("parameters.jl")
include("spec.jl")
include("terms.jl")
include("localops.jl")
include("bases.jl")
include("apply.jl")
include("tree.jl")
include("given.jl")
include("solvers.jl")
include("lobpcg.jl")
include("selection.jl")
include("phase.jl")
include("materialize.jl")
include("dressed.jl")

end # module
