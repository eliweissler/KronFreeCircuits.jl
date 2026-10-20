# Regenerate test/fixtures/bo/*.json: the reduced specs of the Born–Oppenheimer tests, written by `save_spec`.
# Run from the repository root, only to regenerate the committed files:
#
#     julia --project=. test/fixtures/bo/write_reduced.jl
#
# test/test_bo.jl compares each committed file with a fresh reconstruction (`spec_equal`; metadata is not compared);
# validate_reduced.py checks them with circuit-quantization. The constructions live in reduced_specs.jl.

using Test, LinearAlgebra, SparseArrays, KronFreeCircuits

include(joinpath(@__DIR__, "..", "..", "helpers.jl"))
include(joinpath(@__DIR__, "reduced_specs.jl"))

for (name, F) in bo_reduced_fits()
    path = save_spec(joinpath(@__DIR__, name), F.spec; metadata=F.metadata)
    println("wrote ", path)
end
