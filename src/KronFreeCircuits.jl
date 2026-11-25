module KronFreeCircuits

using LinearAlgebra, SparseArrays
using RuntimeGeneratedFunctions
using JSON3
using JLD2: JLD2
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, get_backend
using ITensors: ITensors
using LOBPCGEigensolver: LOBPCGEigensolver
using Adapt: adapt
using Random: Random, randn!
using FFTW: FFTW
using GSL: GSL

RuntimeGeneratedFunctions.init(@__MODULE__)

# The everyday workflow (see the documentation).
export Spec, load_spec, parse_spec, numeric_spec, spec_json, save_spec, substitute, promote_mode, Coefficient, Port,
       nport, n2port, phiport, phi2port, expiphi, exp2pin, Factor, Term, circuit_terms, ChargeBasis, PhaseGrid,
       PeriodicGrid, FockBasis, Node, Auto, Dense, Lanczos, LOBPCG, ShiftInvert, ChebyshevFilter, lowest, indices,
       connected, follow, within, materialize, System, energies, port, eigenvectors, materialized, bare_labels,
       bare_overlaps, decompose, dressed_index, dressed_state_string, constant_term, GivenSystem, character,
       arithmetic, circuit_quantization, QuantizedCircuit, record_json, save_record, load_record, suggest_bases,
       derivatives, blocks, block_levels, spectrum, phase_wavefunction, charge_wavefunction, port_operator,
       qt_hamiltonian, qt_operator, qt_ket, to_dressed, drive_hamiltonian, Frame, to_lab, save_system, load_system,
       system_nodes, load_node, load_metadata
# Supported but not exported (write `KronFreeCircuits.name`): the operator machinery, the custom-solver interface
# (see the `Solver` docstring), basis augmentation, and types and helpers rarely named directly.
public LocalOp, DenseOp, DiagonalOp, BandedOp, ShiftOp, localop, apply_axis!, dense, diagonal, realpart, imagpart,
       KronFreeOperator, infnorm, Solver, solve, resolve, gershgorin_bound, separable_eigenbases, product_seed,
       SavedSolver, SelectionRule, SelectionContext, select, ports_needed, Basis, PromotedSpec, PortCharacter, Derivatives,
       NearDegenerateBlock, differentiate, node_timings, edge_weight, augment, union_basis, BasisSource,
       AugmentedChild, augmented, charge_port, operator

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
include("augment.jl")
include("saveload/values.jl")
include("saveload/terms.jl")
include("saveload/system.jl")
include("experimental/meanfield.jl")
include("experimental/mean_field_augment.jl")
include("derivoperator.jl")
include("resolvent.jl")
include("complement_capture.jl")
include("derivatives.jl")
include("derivblocks.jl")
include("spectrum.jl")
include("experimental/bo.jl")
include("dressed.jl")
include("portoperator.jl")
include("quantumtoolbox.jl")
include("wavefunctions.jl")
include("pyinterface.jl")
include("experimental/Experimental.jl")

end # module
