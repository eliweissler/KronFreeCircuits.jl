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
    run_record(res; frame=nothing, values=nothing, metadata=(;)) -> String

The run record of a circuit-quantization `Result` in the frame `frame`, as JSON text: a kronfree-spec document that `parse_spec` reads and `circuit_quantization.load_record` rebuilds, with this package's version and git commit under `_generator.kronfree` and `metadata` under `_metadata`.
Defined by the PythonCall extension: load PythonCall to use the circuit-quantization interface.
"""
function run_record end

"""
    load_record(source) -> (; result, spec, frame, values)

Read a record written by [`run_record`](@ref), from JSON text or a file.
`spec` is the spec of the run, and `result` is the symbolic `Result` that was passed to `run_record`, in its own frame, so `Spec(result; frame)` is the symbolic spec of the run.
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
