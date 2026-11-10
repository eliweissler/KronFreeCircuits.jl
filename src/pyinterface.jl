# The circuit-quantization interface: stubs whose methods live in the PythonCall extension
# (ext/KronFreeCircuitsPythonCallExt.jl), so the core loads and runs without Python.

"""
    circuit_quantization()

Return the Python module `circuit_quantization`, imported once and cached.
Defined by the PythonCall extension: load PythonCall to use the circuit-quantization interface.
The Python interpreter is the one PythonCall uses; select it with the environment variables `JULIA_PYTHONCALL_EXE` and `JULIA_CONDAPKG_BACKEND=Null`.
"""
function circuit_quantization end

"""
    QuantizedCircuit{H}

A circuit quantized by circuit-quantization at fixed parameter values.
`handle` is the Python numeric Hamiltonian in the base frame, and `values` is the `NamedTuple` of parameter values used, as `Float64` in the circuit's parameter order.
`mode_names` names the retained modes in order, as the circuit-quantization spec of the circuit does; the numeric Hamiltonian carries no names, so they are taken once, at construction, and every frame keeps them.
Constructed by the PythonCall extension, as `QuantizedCircuit(res; values)`.
"""
struct QuantizedCircuit{H}
    handle::H
    values::NamedTuple
    mode_names::Vector{Symbol}
end

"""
    run_record

Write a record of a solver run on a circuit from circuit-quantization.
Defined by the PythonCall extension: load PythonCall to use the circuit-quantization interface.
"""
function run_record end

"""
    load_record

Read a record written by [`run_record`](@ref).
Defined by the PythonCall extension: load PythonCall to use the circuit-quantization interface.
"""
function load_record end

const _PYINTERFACE_HINT = "load PythonCall to use the circuit-quantization interface"

# A MethodError on an interface function with no method means the extension is not loaded; say how to load it.
function _pyinterface_hint(io::IO, exc::MethodError, argtypes, kwargs)
    exc.f in (circuit_quantization, run_record, load_record) || return
    Base.get_extension(@__MODULE__, :KronFreeCircuitsPythonCallExt) === nothing || return
    print(io, "\n", _PYINTERFACE_HINT)
end

function __init__()
    Base.Experimental.register_error_hint(_pyinterface_hint, MethodError)
end
