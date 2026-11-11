# Generator of test/fixtures/kite/optimum_symbolic.json: the KITE at the paper's optimum frame with the flux offset
# `delta`, the offset charge `n_g1` and the junction energy `E_J` left symbolic, for the derivative tests of
# test/test_derivatives_kite.jl.
# A generator, not a test file: test/python/runtests.jl does not include it.
#
# Run it with the interpreter that has circuit-quantization installed, from the repository root:
#     JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=test/python test/python/generate_kite_symbolic.jl
# The fixture is committed and regenerated only on purpose (a change of circuit-quantization or of the settings below),
# never as part of a test run; regenerating it means copying the levels it prints into `KITE_SYM_LEVELS` of
# test/test_derivatives_kite.jl.
using KronFreeCircuits, PythonCall, LinearAlgebra
include("kite.jl")
res, θ, frames = kite()
spec = Spec(res; frame=frames.optimum, subs=(E_CJ=4.82, E_CS=0.072, E_L=1.27, epsilon=0.03))
save_spec(normpath(joinpath(@__DIR__, "..", "fixtures", "kite", "optimum_symbolic.json")), spec)
println("parameters: ", spec.parameters)
# The four lowest levels on the tree of test/test_derivatives_kite.jl at a generic point, to be pinned in that file.
bases = [:varphi_1 => ChargeBasis(16), :varphi_2 => PhaseGrid(-18.915907192616967, 18.915907192616967, 120; order=8),
         :varphi_3 => PhaseGrid(-17.714011073078225, 17.714011073078225, 120; order=8)]
root = Node([Node([m => b]; solver=Dense(), keep=lowest(12)) for (m, b) in bases]; solver=Dense(), compute=6, keep=lowest(4))
println("levels at (E_J = 16.83, delta = 0.0, n_g1 = 0.1): ", repr(energies(materialize(root, spec, (E_J=16.83, delta=0.0, n_g1=0.1)))))
