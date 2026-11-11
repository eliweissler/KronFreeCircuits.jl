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

export Spec, load_spec, parse_spec, numeric_spec, spec_json, save_spec, substitute, promote_mode, PromotedSpec,
       periodic_grid, BOSurface, bo_surface, fourier_coefficients, slow_spec, BOFit, fit_surfaces, Coefficient, Port, nport,
       n2port, phiport, phi2port, expiphi, exp2pin, Factor,
       Term, circuit_terms, LocalOp, DenseOp, DiagonalOp, BandedOp,
       ShiftOp, dense, diagonal, realpart, imagpart, Basis, ChargeBasis, PhaseGrid, PeriodicGrid,
       FockBasis, localop, apply_axis!, KronFreeOperator, Node, Auto, Dense,
       Lanczos, LOBPCG, ShiftInvert, ChebyshevFilter, lowest, indices, connected, follow, within, materialize,
       System, energies, port, eigenvectors, materialized, bare_labels,
       bare_overlaps, decompose, dressed_index, constant_term, Solver, SelectionRule,
       SelectionContext, GivenSystem, gershgorin_bound,
       separable_eigenbases, product_seed, infnorm,
       PortCharacter, character, arithmetic,
       circuit_quantization, QuantizedCircuit, record_json, save_record, load_record,
       Expectations, straddling_terms, child_keys, expectations, pinned_expectations, delocalized_expectations,
       mean_field_terms, MeanFieldChild, bare_child, mean_field_child, product_energy,
       MeanFieldSolution, default_starts, self_consistent, mean_field_solutions,
       BasisSource, UnionReport, union_basis, AugmentedChild, augment, augmented, mean_field_augment,
       suggest_bases, edge_weight, differentiate

include("parameters.jl")
include("spec.jl")
include("specwrite.jl")
include("spectransform.jl")
include("terms.jl")
include("localops.jl")
include("bases.jl")
include("apply.jl")
include("tree.jl")
include("given.jl")
include("character.jl")
include("diffcoeff.jl")
include("solvers.jl")
include("lobpcg.jl")
include("shiftinvert.jl")
include("chebyshev.jl")
include("selection.jl")
include("phase.jl")
include("materialize.jl")
include("suggest.jl")
include("meanfield.jl")
include("augment.jl")
include("derivoperator.jl")
include("resolvent.jl")
include("derivatives.jl")
include("bo.jl")
include("dressed.jl")
include("pyinterface.jl")

end # module
