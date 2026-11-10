module KronFreeCircuitsPythonCallExt

using KronFreeCircuits, PythonCall
import KronFreeCircuits: circuit_quantization

# What to do when circuit_quantization does not import: name the interpreter PythonCall is using and how to choose another.
function _import_help(exe::AbstractString)::String
    return """
        could not import the Python module circuit_quantization with the interpreter $exe.
        Point PythonCall at a Python that has circuit-quantization installed, before loading PythonCall:
            JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/path/to/python julia ..."""
end

const _CQ = PythonCall.pynew()         # the module, filled on first use
const _CQ_LOCK = ReentrantLock()

function circuit_quantization()
    lock(_CQ_LOCK) do
        if PythonCall.pyisnull(_CQ)
            cq = try
                pyimport("circuit_quantization")
            catch err
                err isa PythonCall.PyException || rethrow()
                exe = pyconvert(String, pyimport("sys").executable)
                throw(ErrorException(_import_help(exe)))
            end
            PythonCall.pycopy!(_CQ, cq)
        end
    end
    return _CQ
end

end
