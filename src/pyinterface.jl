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
    record_json(spec::Spec; circuit, frame=nothing, values=nothing, metadata=(;), zero_rtol=1e-12, zero_atol=0.0,
                check=true) -> String

The record of a run that solved `spec`, as JSON text: a kronfree-spec document whose spec fields are those of `spec`, written as [`spec_json`](@ref) writes them, so `parse_spec` of the text gives `spec` back.
The document also carries what rebuilds the run: `circuit`, the circuit-quantization `Result` the spec came from, in the frame `frame`, written by `circuit_quantization.to_record` so that `circuit_quantization.load_record` rebuilds it; the parameter values `values` in the frame of `circuit`, as `QuantizedCircuit(circuit; values)` takes them; and the round-off zeroing tolerances `zero_rtol` and `zero_atol` the spec was built with.
This package's version and git commit are written under `_generator.kronfree`, and `metadata` under `_metadata`.
With `check = true` the spec rebuilt from `circuit`, `frame`, `values` and the tolerances must agree with `spec`, and a mismatch is an `ArgumentError`; `check = false` writes the record without comparing.
Defined by the PythonCall extension: load PythonCall to use the circuit-quantization interface.
"""
function record_json end

"""
    save_record(path::AbstractString, spec::Spec; circuit, frame=nothing, values=nothing, metadata=(;),
                zero_rtol=1e-12, zero_atol=0.0, check=true) -> path

Write [`record_json`](@ref) of `spec` to `path`, followed by a newline, and return `path`; [`load_record`](@ref) reads it back.
Defined by the PythonCall extension: load PythonCall to use the circuit-quantization interface.
"""
function save_record end

"""
    load_record(source) -> (; result, spec, frame, values, zero_rtol, zero_atol)

Read a record written by [`save_record`](@ref) or [`record_json`](@ref), from JSON text or a file.
`spec` is the spec the run solved, and `result` is the symbolic `Result` that was passed as `circuit`, in its own frame, so `Spec(result; frame)` is the symbolic spec of the run.
`values` are the parameter values that were passed, in the frame of `result`, and `zero_rtol` and `zero_atol` the run's round-off zeroing tolerances.
Defined by the PythonCall extension: load PythonCall to use the circuit-quantization interface.
"""
function load_record end

const _PYINTERFACE_HINT = "load PythonCall to use the circuit-quantization interface"

# A MethodError on an interface function with no method means the extension is not loaded; say how to load it.
function _pyinterface_hint(io::IO, exc::MethodError, argtypes, kwargs)
    exc.f in (circuit_quantization, record_json, save_record, load_record) || return
    Base.get_extension(@__MODULE__, :KronFreeCircuitsPythonCallExt) === nothing || return
    print(io, "\n", _PYINTERFACE_HINT)
end

function __init__()
    Base.Experimental.register_error_hint(_pyinterface_hint, MethodError)
end
