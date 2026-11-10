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

# The parameter symbols of res, by name: the free symbols of every expression `to_spec` exports, except the physical
# constants circuit-quantization evaluates itself.
function _circuit_symbols(res::Py)::Dict{String,Py}
    sympy, units = pyimport("sympy"), pyimport("circuit_quantization.units")
    constants = pyset([units.e, units.h, units.hbar, units.Phi0, units.h_GHz])
    ham, found = res.ham, pyset()
    for x in (ham.E_C, ham.E_L, ham.E_Lb, ham.varphi_b, ham.n_g, res.coords.omega_L_tilde, res.lag.EJ...)
        found.update(sympy.sympify(x).free_symbols)
    end
    return Dict(pyconvert(String, s.name) => s for s in found if !pyin(s, constants))
end

# A `subs` value as SymPy: a number, the name of a symbol (the circuit's own, or else a new one with the assumptions of
# the symbol it replaces), or a SymPy expression passed through.
function _subs_value(v, symbols::Dict{String,Py}, replaced::Py)
    sympy = pyimport("sympy")
    v isa Py && return v
    v isa Bool && throw(ArgumentError("subs: a value must be a number, a name or a SymPy expression, got $v"))
    v isa Integer && return sympy.Integer(string(v))
    v isa Rational && return sympy.Rational(string(numerator(v)), string(denominator(v)))
    v isa Real && return sympy.Float(Float64(v))
    if v isa Union{AbstractString,Symbol}
        name = string(v)
        assumptions = (Symbol(pyconvert(String, a)) => x for (a, x) in replaced.assumptions0.items())
        return get(() -> sympy.Symbol(name; assumptions...), symbols, name)
    end
    throw(ArgumentError("subs: a value must be a number, a name or a SymPy expression, got $(repr(v))"))
end

# Julia `subs` (name ⇒ value) as the SymPy substitution `to_spec` takes, keyed by the circuit's own symbols:
# SymPy matches symbols by name and assumptions, so a fresh symbol of the same name would silently not match.
_pysubs(res::Py, ::Nothing) = nothing
_pysubs(res::Py, subs::NamedTuple) = _pysubs(res, pairs(subs))
function _pysubs(res::Py, subs)
    symbols = _circuit_symbols(res)
    out = pydict()
    for (k, v) in subs
        name = string(k)
        haskey(symbols, name) ||
            throw(ArgumentError("subs: $name is not a symbol of the circuit; its symbols are " *
                                join(sort!(collect(keys(symbols))), ", ")))
        out[symbols[name]] = _subs_value(v, symbols, symbols[name])
    end
    return out
end

const _FRAME_TOL = 1e-12               # how far a compact-block entry of a frame may lie from an integer

# A frame entry as an exact SymPy number: integers and rationals as they are, a float by its shortest round-trip
# decimal string. SymPy never sees a float: float entries leave round-off in the symbolic coefficients (in the KITE at
# its optimum frame, an `E_L` that `to_spec` rejects as asymmetric by 2e-16) and slow the symbolic algebra.
_sympy_number(sympy::Py, x::Integer) = sympy.Integer(string(BigInt(x)))     # BigInt: `string(true)` is "true"
_sympy_number(sympy::Py, x::Rational) = sympy.Rational(string(numerator(x)), string(denominator(x)))
_sympy_number(sympy::Py, x::Real) = iszero(x) ? sympy.Integer(0) : sympy.Rational(repr(Float64(x)))

"""
    to_sympy_frame(S::AbstractMatrix{<:Real}, types) -> Py

`S` as an exact SymPy matrix, for the modes of `types` (`"c"`, `"e"`, `"h"`, in mode order).
Each entry of the compact block (rows and columns of the `"c"` modes) becomes the nearest integer, and must lie within
`1e-12` of it; every other entry becomes the exact rational of its shortest round-trip decimal string, and an exact zero
stays an exact zero.
An `ArgumentError` names a compact-block entry that is not an integer.
"""
function to_sympy_frame(S::AbstractMatrix{<:Real}, types)::Py
    sympy = pyimport("sympy")
    compact = [string(t) == "c" for t in types]
    function entry(i, j, in_block)
        x = S[i, j]
        in_block || return _sympy_number(sympy, x)
        k = round(x)
        abs(x - k) <= _FRAME_TOL ||
            throw(ArgumentError("frame: the compact-block entry S[$i, $j] = $x must be an integer"))
        return sympy.Integer(string(BigInt(k)))
    end
    rows = [pylist([entry(i, j, compact[a] && compact[b]) for (b, j) in enumerate(axes(S, 2))])
            for (a, i) in enumerate(axes(S, 1))]
    return sympy.Matrix(pylist(rows))
end

# `res` in the frame φ = S φ′: the size and finiteness of `S` are checked here, admissibility by `Result.transform`,
# whose `ValueError` becomes an `ArgumentError`.
function _in_frame(res::Py, S::AbstractMatrix{<:Real})::Py
    types = pyconvert(Vector{String}, pylist(res.coords.types))
    _check_frame(S, length(types))
    return _frame_call(() -> res.transform(to_sympy_frame(S, types)))
end

"""
    Spec(res::Py; frame=nothing, subs=nothing, simplify::Bool=true, mode_names=nothing) -> Spec

The spec of the circuit-quantization `Result` `res`, built in memory: `parse_spec` of the JSON text of
`circuit_quantization.to_spec(res; subs, simplify, mode_names)`, so it is the spec that `load_spec` reads from the file
`save_spec` writes.
`frame`, a square real matrix `S` over the retained modes, first takes `res` to the coordinates `φ = S φ′` with
`res.transform`, keeping the parameters symbolic; `to_sympy_frame` makes `S` exact before it reaches SymPy.
`S` must be invertible, mix no noncompact mode into a compact one (zeros below the compact block), and have an integer,
unimodular compact block; an inadmissible `S` is an `ArgumentError`.
In the new frame each compact mode `j` carries a fresh static offset-charge label `n_g{j}`, and the noncompact modes
carry none.
`subs` is a `Dict` or `NamedTuple` from a parameter name (`String` or `Symbol`) to a number, to the name of another
parameter (an existing one, or a new one with the same SymPy assumptions), or to a SymPy expression.
Names are matched to the circuit's own SymPy symbols (in the new frame, when `frame` is given); a name that matches none
is an `ArgumentError`.
`mode_names` labels the retained modes in order (default `varphi_1`, ..., `varphi_N`).
"""
function KronFreeCircuits.Spec(res::Py; frame::Union{Nothing,AbstractMatrix{<:Real}}=nothing, subs=nothing,
                               simplify::Bool=true, mode_names=nothing)
    frame === nothing || (res = _in_frame(res, frame))
    names = mode_names === nothing ? nothing : pylist(String[string(m) for m in mode_names])
    data = circuit_quantization().to_spec(res; subs=_pysubs(res, subs), simplify, mode_names=names)
    text = pyconvert(String, pyimport("json").dumps(data; allow_nan=false))
    return parse_spec(text)
end

# ---- The loop route: a numeric Hamiltonian made once at fixed values, and its spec in any frame without SymPy.

# The size and finiteness of a frame `S` over `N` retained modes, checked in Julia before any Python call.
function _check_frame(S::AbstractMatrix, N::Int)
    size(S) == (N, N) ||
        throw(ArgumentError("frame: S must be $(N)×$(N) for the $N retained modes, got $(join(size(S), "×"))"))
    all(isfinite, S) || throw(ArgumentError("frame: every entry of S must be finite"))
    return nothing
end

# `f()`, a circuit-quantization frame change, with the `ValueError` of an inadmissible frame as an `ArgumentError`
# carrying the Python message; any other exception passes through.
function _frame_call(f)
    return try
        f()
    catch err
        err isa PythonCall.PyException && pyisinstance(err.v, pybuiltins.ValueError) || rethrow()
        throw(ArgumentError("frame: " * pyconvert(String, pystr(err.v))))
    end
end

# `values` (a NamedTuple, or a Dict keyed by Symbols or strings) as Float64s in the order of the circuit's parameters
# `names`; an unknown, repeated or missing name, or a value that is not a finite real number, is an `ArgumentError`.
function _parameter_values(values, names::Vector{Symbol})::NamedTuple
    values isa Union{NamedTuple,AbstractDict} ||
        throw(ArgumentError("values must be a NamedTuple or a Dict from parameter name to number, got $(typeof(values))"))
    listing = "; the circuit's parameters are " * join(names, ", ")
    given = Dict{Symbol,Float64}()
    for (k, v) in pairs(values)
        k isa Union{Symbol,AbstractString} ||
            throw(ArgumentError("values: a parameter name must be a Symbol or a string, got $(repr(k))$listing"))
        name = Symbol(k)
        name in names || throw(ArgumentError("values: $name is not a parameter of the circuit$listing"))
        haskey(given, name) && throw(ArgumentError("values: $name is given more than once"))
        (v isa Real && !(v isa Bool) && isfinite(v)) ||
            throw(ArgumentError("values: $name = $(repr(v)) must be a finite real number"))
        given[name] = Float64(v)
    end
    absent = filter(n -> !haskey(given, n), names)
    isempty(absent) || throw(ArgumentError("values: no value for $(join(absent, ", "))$listing"))
    return NamedTuple{Tuple(names)}(Tuple(given[n] for n in names))
end

"""
    QuantizedCircuit(res::Py; values) -> QuantizedCircuit{Py}

The circuit-quantization `Result` `res` evaluated once at the parameter values `values`, for building specs in many
frames with no symbolic work: `circuit_quantization.numeric_hamiltonian` of the spec `to_spec(res)`.
`values` is a `NamedTuple`, or an `AbstractDict` keyed by `Symbol`s or strings, giving a finite real number for every
parameter of the circuit; an unknown or missing name is an `ArgumentError` that names it and lists the circuit's
parameters, raised before anything is evaluated.
The modes are named as in `Spec(res)`.
"""
function KronFreeCircuits.QuantizedCircuit(res::Py; values)
    cq = circuit_quantization()
    spec = cq.load_spec(cq.to_spec(res))
    names = [Symbol(pyconvert(String, p.name)) for p in spec.parameters]
    θ = _parameter_values(values, names)
    mode_names = [Symbol(pyconvert(String, m[0])) for m in spec.modes]
    handle = cq.numeric_hamiltonian(spec, pydict(Dict(string(k) => v for (k, v) in pairs(θ))))
    return QuantizedCircuit(handle, θ, mode_names)
end

# The spec of constants of the numeric Hamiltonian `h`, with the modes named `mode_names`.
function _numeric_spec(h::Py, mode_names::Vector{Symbol})::Spec
    types = pyconvert(Vector{String}, h.types)
    cosines = [(pyconvert(Float64, E), pyconvert(Vector{Float64}, w), pyconvert(Float64, d)) for (E, w, d) in h.cosines]
    return numeric_spec(; modes=[n => Symbol(t) for (n, t) in zip(mode_names, types)],
                        E_C=pyconvert(Matrix{Float64}, h.E_C), E_L=pyconvert(Matrix{Float64}, h.E_L),
                        flux_linear=pyconvert(Vector{Float64}, h.flux_linear), n_g=pyconvert(Vector{Float64}, h.n_g),
                        cosines, constant=pyconvert(Float64, h.constant))
end

"""
    Spec(qc::QuantizedCircuit{Py}) -> Spec
    Spec(qc::QuantizedCircuit{Py}, S::AbstractMatrix{<:Real}) -> Spec

The spec of constants of `qc` in its base frame, or in the frame `φ = S φ′`: [`numeric_spec`](@ref) of the arrays of
`circuit_quantization`'s NumPy frame change `transformed(S)`, with no symbolic work.
`S` follows the rules of `Spec(res; frame=S)`, and the modes are named and retyped as there; its size and finiteness
are checked in Julia, its admissibility by circuit-quantization, and an inadmissible `S` is an `ArgumentError`.
In the new frame each compact mode's offset charge is the source part transformed by `Sᵀ` plus the static sector part
transformed by `S_ccᵀ`, and a noncompact mode carries no sector offset.
"""
KronFreeCircuits.Spec(qc::QuantizedCircuit{Py}) = _numeric_spec(qc.handle, qc.mode_names)

function KronFreeCircuits.Spec(qc::QuantizedCircuit{Py}, S::AbstractMatrix{<:Real})
    _check_frame(S, length(qc.mode_names))
    S_np = pyimport("numpy").array(Matrix{Float64}(S))
    return _numeric_spec(_frame_call(() -> qc.handle.transformed(S_np)), qc.mode_names)
end

end
