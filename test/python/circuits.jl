# The circuits of test/fixtures/generated, rebuilt in memory by test/fixtures/generate_specs.py's builders.
# A helper, not a test file: test files include it.
using PythonCall

@isdefined(_GENERATE_SPECS) || const _GENERATE_SPECS = PythonCall.pynew()   # several test files include this helper

# test/fixtures/generate_specs.py as a Python module, imported once; importing it writes nothing.
function generate_specs()
    if PythonCall.pyisnull(_GENERATE_SPECS)
        circuit_quantization()                    # the import error with its help, if circuit-quantization is missing
        util = pyimport("importlib.util")
        path = joinpath(@__DIR__, "..", "fixtures", "generate_specs.py")
        spec = util.spec_from_file_location("kronfree_generate_specs", path)
        mod = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        PythonCall.pycopy!(_GENERATE_SPECS, mod)
    end
    return _GENERATE_SPECS
end

"""
    fixture_result(name::String) -> (res::Py, kwargs::NamedTuple)

The circuit-quantization `Result` of the generated fixture `name` and the keywords (`subs`, `mode_names`) that make
`Spec(res; kwargs...)` the fixture's spec.
`subs` is a `Dict` keyed by symbol name; a value that is a bare SymPy symbol is given by its name, any other stays a
SymPy expression.
The builders' progress lines on Python's standard output are discarded.
"""
function fixture_result(name::String)
    contextlib, io = pyimport("contextlib"), pyimport("io")
    fx = pywith(contextlib.redirect_stdout(io.StringIO())) do _
        generate_specs().BUILDERS[name]()
    end
    subs = if pyis(fx.subs, pybuiltins.None)
        nothing
    else
        sympy = pyimport("sympy")
        Dict{String,Any}(pyconvert(String, k.name) => (pyisinstance(v, sympy.Symbol) ? pyconvert(String, v.name) : v)
                         for (k, v) in fx.subs.items())
    end
    return fx.res, (; subs, mode_names=pyconvert(Vector{String}, fx.mode_names))
end
