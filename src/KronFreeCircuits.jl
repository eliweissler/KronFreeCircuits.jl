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

export Spec, load_spec, parse_spec, numeric_spec, spec_json, save_spec, substitute, promote_mode, PromotedSpec,
       Coefficient, Port, nport, n2port, phiport, phi2port, expiphi, exp2pin, Factor, Term, circuit_terms, LocalOp,
       DenseOp, DiagonalOp, BandedOp, ShiftOp, dense, diagonal, realpart, imagpart, Basis, ChargeBasis, PhaseGrid,
       PeriodicGrid, FockBasis, localop, apply_axis!, KronFreeOperator, Node, Auto, Dense, Lanczos, LOBPCG,
       ShiftInvert, ChebyshevFilter, lowest, indices, connected, follow, within, materialize, System, energies, port,
       eigenvectors, materialized, node_timings, bare_labels, bare_overlaps, decompose, dressed_index, constant_term,
       Solver, SelectionRule, SelectionContext, GivenSystem, gershgorin_bound, separable_eigenbases, product_seed,
       infnorm, PortCharacter, character, arithmetic, circuit_quantization, QuantizedCircuit, record_json,
       save_record, load_record, suggest_bases, edge_weight, differentiate, derivatives, Derivatives, blocks,
       NearDegenerateBlock, block_levels, spectrum, phase_wavefunction, charge_wavefunction, port_operator,
       qt_hamiltonian, qt_operator, qt_ket, to_dressed, drive_hamiltonian, Frame, to_lab, save_system, load_system,
       system_nodes, load_node, load_metadata

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
