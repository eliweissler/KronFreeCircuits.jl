# The Python tier: every test that needs PythonCall and circuit-quantization.
# Run with the interpreter that has circuit-quantization installed, for example
#     JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=test/python test/python/runtests.jl
# The default tier (test/runtests.jl) never includes this directory.
using Test, KronFreeCircuits, PythonCall

Base.get_extension(KronFreeCircuits, :KronFreeCircuitsPythonCallExt) === nothing &&
    error("the PythonCall extension of KronFreeCircuits did not load; the Manifest of test/python may be stale: " *
          "run Pkg.resolve() and Pkg.instantiate() with --project=test/python")

@testset verbose=true "Python tier" begin
    for f in sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(@__DIR__)))
        include(f)
    end
end
