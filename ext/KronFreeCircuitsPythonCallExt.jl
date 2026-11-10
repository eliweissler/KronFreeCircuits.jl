module KronFreeCircuitsPythonCallExt

using KronFreeCircuits, PythonCall
import KronFreeCircuits: circuit_quantization, _ZERO_RTOL, _ZERO_ATOL, _check_zero_tolerance

# What to do when circuit_quantization is not installed for the interpreter PythonCall is using: name the interpreter
# and how to choose another.
function _import_help(exe::AbstractString)::String
    return """
        could not import the Python module circuit_quantization with the interpreter $exe.
        Point PythonCall at a Python that has circuit-quantization installed, before loading PythonCall:
            JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/path/to/python julia ..."""
end

# The Python module `name`, imported.
# When the module itself is not found, the error is the help of `_import_help` (with `name` in place of
# circuit_quantization); any other import failure, such as a missing dependency or an error raised while the module
# runs, is an error carrying Python's own message.
function _import_module(name::AbstractString)::Py
    try
        return pyimport(name)
    catch err
        err isa PythonCall.PyException || rethrow()
        exe = pyconvert(String, pyimport("sys").executable)
        if pyisinstance(err.v, pyimport("builtins").ModuleNotFoundError) && !pyis(err.v.name, pybuiltins.None) &&
           pyconvert(String, err.v.name) == name
            throw(ErrorException(replace(_import_help(exe), "circuit_quantization" => name)))
        end
        python = pyconvert(String, pytype(err.v).__name__) * ": " * pyconvert(String, pystr(err.v))
        throw(ErrorException("importing the Python module $name with the interpreter $exe failed: $python"))
    end
end

const _CQ = PythonCall.pynew()         # the module, filled on first use
const _CQ_LOCK = ReentrantLock()

function circuit_quantization()
    lock(_CQ_LOCK) do
        PythonCall.pyisnull(_CQ) && PythonCall.pycopy!(_CQ, _import_module("circuit_quantization"))
    end
    return _CQ
end

const _NUMPY = PythonCall.pynew()      # numpy, filled on first use

function _numpy()
    lock(_CQ_LOCK) do
        PythonCall.pyisnull(_NUMPY) && PythonCall.pycopy!(_NUMPY, pyimport("numpy"))
    end
    return _NUMPY
end

# Why the circuit-quantization function `to_spec`, from the module file `origin`, cannot take the round-off zeroing
# keywords `zero_rtol` and `zero_atol`, or `nothing` when it can; read from the function's signature.
function _zero_keywords_missing(to_spec::Py, origin::AbstractString)::Union{Nothing,String}
    parameters = pyimport("inspect").signature(to_spec).parameters
    all(k -> pyin(k, parameters), ("zero_rtol", "zero_atol")) && return nothing
    return "the circuit-quantization imported from $origin has no zero_rtol and zero_atol keywords in to_spec; " *
           "circuit-quantization at commit c8c8714 or later (branch to-spec-zero-tolerance until it is merged) is required"
end

const _ZERO_KEYWORDS = Ref{Union{Missing,Nothing,String}}(missing)   # `_zero_keywords_missing`, checked once

# Throw an `ArgumentError` unless the installed circuit-quantization's `to_spec` takes `zero_rtol` and `zero_atol`.
function _require_zero_keywords()
    lock(_CQ_LOCK) do
        if ismissing(_ZERO_KEYWORDS[])
            cq = circuit_quantization()
            _ZERO_KEYWORDS[] = _zero_keywords_missing(cq.to_spec, pyconvert(String, cq.__file__))
        end
    end
    message = _ZERO_KEYWORDS[]
    message === nothing || throw(ArgumentError(message))
    return nothing
end

# The parameter symbols of res, by name: the free symbols of every expression `to_spec` exports (of `varphi_b`, only the
# rows of the inductive branches `lag.L` and the junctions `lag.J`), except the physical constants circuit-quantization
# evaluates itself.
function _circuit_symbols(res::Py)::Dict{String,Py}
    sympy, units = pyimport("sympy"), pyimport("circuit_quantization.units")
    constants = pyset([units.e, units.h, units.hbar, units.Phi0, units.h_GHz])
    ham, found = res.ham, pyset()
    columns = pylist(0:pyconvert(Int, ham.varphi_b.cols)-1)
    exported_rows(branches) = ham.varphi_b.extract(pylist(branches), columns)
    for x in (ham.E_C, ham.E_L, ham.E_Lb, exported_rows(res.lag.L), exported_rows(res.lag.J), ham.n_g,
              res.coords.omega_L_tilde, res.lag.EJ...)
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

# ---- Frame checks, shared by the symbolic route (`Spec(res; frame)`, `record_json`) and the loop route
# (`Spec(qc, S)`).

# The size and finiteness of a frame `S` over `N` retained modes, checked in Julia before any Python call.
function _check_frame(S::AbstractMatrix, N::Int)
    size(S) == (N, N) ||
        throw(ArgumentError("frame: S must be $(N)×$(N) for the $N retained modes, got $(join(size(S), "×"))"))
    all(isfinite, S) || throw(ArgumentError("frame: every entry of S must be finite"))
    return nothing
end

# `f()`, a circuit-quantization frame change, with the `ValueError` of an inadmissible frame as an `ArgumentError`
# carrying the Python message, whose matrix indices are circuit-quantization's, counted from 0; any other exception
# passes through.
function _frame_call(f)
    return try
        f()
    catch err
        err isa PythonCall.PyException && pyisinstance(err.v, pybuiltins.ValueError) || rethrow()
        throw(ArgumentError("frame: " * pyconvert(String, pystr(err.v)) *
                            " (circuit-quantization's message: its matrix indices count from 0)"))
    end
end

# A frame entry as an exact SymPy number: integers and rationals as they are, a float by its shortest round-trip
# decimal string.
# SymPy never sees a float: float entries leave round-off in the symbolic coefficients (in the KITE at its optimum frame,
# an `E_L` that `to_spec` rejects as asymmetric by 2e-16) and slow the symbolic algebra.
_sympy_number(sympy::Py, x::Integer) = sympy.Integer(string(BigInt(x)))     # BigInt: `string(true)` is "true"
_sympy_number(sympy::Py, x::Rational) = sympy.Rational(string(numerator(x)), string(denominator(x)))
_sympy_number(sympy::Py, x::Real) = iszero(x) ? sympy.Integer(0) : sympy.Rational(repr(Float64(x)))

"""
    to_sympy_frame(S::AbstractMatrix{<:Real}, types) -> Py

`S` as an exact SymPy matrix, for the modes of `types` (`"c"`, `"e"`, `"h"`, in mode order).
Each entry of the compact block (rows and columns of the `"c"` modes) becomes the nearest integer, and must lie within `1e-12` of it.
Every other entry becomes the exact rational of its shortest round-trip decimal string, and an exact zero stays an exact zero.
An `ArgumentError` names a compact-block entry that is not an integer, with Julia's indices (counted from 1).
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

# ---- The symbolic route.

"""
    Spec(res::Py; frame=nothing, subs=nothing, simplify::Bool=true, mode_names=nothing, zero_rtol=1e-12,
         zero_atol=0.0) -> Spec

The spec of the circuit-quantization `Result` `res`, built in memory.
It is `parse_spec` of the JSON text of `circuit_quantization.to_spec(res; subs, simplify, mode_names, zero_rtol, zero_atol)`, so it is the spec that `load_spec` reads from the file `save_spec` writes.

`frame`, a square real matrix `S` over the retained modes, first takes `res` to the coordinates `φ = S φ′` with `res.transform`, keeping the parameters symbolic; `to_sympy_frame` makes `S` exact before it reaches SymPy.
`S` must be invertible, mix no noncompact mode into a compact one (zeros below the compact block), and have an integer, unimodular compact block; an inadmissible `S` is an `ArgumentError`.
In the new frame each compact mode `j` carries a fresh static offset-charge label `n_g{j}`, and the noncompact modes carry none.
The framed spec's parameter `n_g{j}` is therefore the new frame's sector label, not the base frame's, although the name is the same: the value that describes the same circuit is `S_ccᵀ n_g`, with `S_cc` the compact block of `S` and `n_g` the base frame's sector values, which is what `Spec(qc, S)` gives it and what `record_json` records.

`subs` is a `Dict` or `NamedTuple` from a parameter name (`String` or `Symbol`) to a number, to the name of another parameter (an existing one, or a new one with the same SymPy assumptions), or to a SymPy expression.
Names are matched to the circuit's own SymPy symbols (in the new frame, when `frame` is given), among the symbols that the exported spec can contain; a name that matches none is an `ArgumentError` listing them.
`mode_names` labels the retained modes in order (default `varphi_1`, ..., `varphi_N`).

`zero_rtol` and `zero_atol` are `to_spec`'s round-off zeroing tolerances, the rule of [`numeric_spec`](@ref).
In a fully numeric `E_C` or `E_L`, a floating-point entry at most `max(zero_rtol * scale, zero_atol)` in magnitude is written as an exact zero, and in each cosine an `omega` entry at most `zero_rtol` times that `omega`'s largest magnitude.
A negative or non-finite tolerance is an `ArgumentError` naming it, raised before any Python call.
The keywords need circuit-quantization at commit c8c8714 or later (branch `to-spec-zero-tolerance` until it is merged); with an older circuit-quantization every call is an `ArgumentError` that says so and names the file the module was imported from.
"""
function KronFreeCircuits.Spec(res::Py; frame::Union{Nothing,AbstractMatrix{<:Real}}=nothing, subs=nothing,
                               simplify::Bool=true, mode_names=nothing,
                               zero_rtol::Real=_ZERO_RTOL, zero_atol::Real=_ZERO_ATOL)
    zero_rtol = _check_zero_tolerance(zero_rtol, :zero_rtol)
    zero_atol = _check_zero_tolerance(zero_atol, :zero_atol)
    _require_zero_keywords()
    frame === nothing || (res = _in_frame(res, frame))
    names = mode_names === nothing ? nothing : pylist(String[string(m) for m in mode_names])
    data = circuit_quantization().to_spec(res; subs=_pysubs(res, subs), simplify, mode_names=names, zero_rtol,
                                          zero_atol)
    text = pyconvert(String, pyimport("json").dumps(data; allow_nan=false))
    return parse_spec(text)
end

# ---- Parameter values, shared by `QuantizedCircuit` and `record_json`.

# The circuit-quantization spec of `res` in its own frame, as circuit-quantization loads it, with its parameters
# (`spec.parameters`, in the order `Spec(res)` lists them) and its modes.
_exported_spec(res::Py) = (cq = circuit_quantization(); cq.load_spec(cq.to_spec(res)))

_parameter_names(spec::Py) = [Symbol(pyconvert(String, p.name)) for p in spec.parameters]

# `values` (a NamedTuple, or a Dict keyed by Symbols or strings) as Float64s in the order of the circuit's parameters
# `names`, the parameters of the exported spec.
# A `values` of another type, a key that is not a name, an unknown name, a name given twice, a value that is not a
# finite real number, or a parameter left without a value is an `ArgumentError`; for an unknown or a missing name it
# lists the circuit's parameters.
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

# Parameter values as the Python dict, keyed by name, that circuit-quantization's numeric functions take.
_pyvalues(θ::NamedTuple) = pydict(Dict(string(k) => v for (k, v) in pairs(θ)))

# ---- The loop route: a numeric Hamiltonian made once at fixed values, and its spec in any frame without SymPy.

"""
    QuantizedCircuit(res::Py; values) -> QuantizedCircuit{Py}

The circuit-quantization `Result` `res` evaluated once at the parameter values `values`, for building specs in many frames with no symbolic work: `circuit_quantization.numeric_hamiltonian` of the spec `to_spec(res)`.
`values` is a `NamedTuple`, or an `AbstractDict` keyed by `Symbol`s or strings, giving a finite real number for every parameter of `Spec(res)`, in the frame of `res`; a compact mode's `n_g{j}` is that frame's static sector offset.
An unknown or missing name is an `ArgumentError` that names it and lists the circuit's parameters, raised before anything is evaluated; `record_json` checks `values` the same way.
The values are stored in the circuit's parameter order, and the modes are named as in `Spec(res)`.
"""
function KronFreeCircuits.QuantizedCircuit(res::Py; values)
    spec = _exported_spec(res)
    θ = _parameter_values(values, _parameter_names(spec))
    mode_names = [Symbol(pyconvert(String, m[0])) for m in spec.modes]
    handle = circuit_quantization().numeric_hamiltonian(spec, _pyvalues(θ))
    return QuantizedCircuit(handle, θ, mode_names)
end

# The spec of constants of the numeric Hamiltonian `h`, with the modes named `mode_names`; `zero` holds the round-off
# zeroing keywords of `numeric_spec`.
function _numeric_spec(h::Py, mode_names::Vector{Symbol}; zero...)::Spec
    types = pyconvert(Vector{String}, h.types)
    cosines = [(pyconvert(Float64, E), pyconvert(Vector{Float64}, w), pyconvert(Float64, d)) for (E, w, d) in h.cosines]
    return numeric_spec(; modes=[n => Symbol(t) for (n, t) in zip(mode_names, types)],
                        E_C=pyconvert(Matrix{Float64}, h.E_C), E_L=pyconvert(Matrix{Float64}, h.E_L),
                        flux_linear=pyconvert(Vector{Float64}, h.flux_linear), n_g=pyconvert(Vector{Float64}, h.n_g),
                        cosines, constant=pyconvert(Float64, h.constant), zero...)
end

# The numeric Hamiltonian `h` in the frame φ = S φ′, by circuit-quantization's NumPy frame change, whose `ValueError`
# becomes an `ArgumentError`; the caller has checked the size and finiteness of `S`.
function _transformed(h::Py, S::AbstractMatrix{<:Real})::Py
    S_np = _numpy().array(Matrix{Float64}(S))
    return _frame_call(() -> h.transformed(S_np))
end

"""
    Spec(qc::QuantizedCircuit{Py}; zero_rtol=1e-12, zero_atol=0.0) -> Spec
    Spec(qc::QuantizedCircuit{Py}, S::AbstractMatrix{<:Real}; zero_rtol=1e-12, zero_atol=0.0) -> Spec

The spec of constants of `qc` in its base frame, or in the frame `φ = S φ′`: [`numeric_spec`](@ref) of the arrays of `circuit_quantization`'s NumPy frame change `transformed(S)`, with no symbolic work.
`S` follows the rules of `Spec(res; frame=S)`, and the modes are named and retyped as there.
The size and finiteness of `S` are checked in Julia, its admissibility by circuit-quantization, and an inadmissible `S` is an `ArgumentError`.
In the new frame each compact mode's offset charge is the source part transformed by `Sᵀ` plus the static sector part transformed by `S_ccᵀ`, and a noncompact mode carries no sector offset.
`zero_rtol` and `zero_atol` are passed to `numeric_spec`, whose round-off zeroing they set as in `Spec(res)`; a negative or non-finite tolerance is an `ArgumentError` naming it, raised before any Python call.
"""
function KronFreeCircuits.Spec(qc::QuantizedCircuit{Py}; zero_rtol::Real=_ZERO_RTOL,
                               zero_atol::Real=_ZERO_ATOL)
    zero_rtol = _check_zero_tolerance(zero_rtol, :zero_rtol)
    zero_atol = _check_zero_tolerance(zero_atol, :zero_atol)
    return _numeric_spec(qc.handle, qc.mode_names; zero_rtol, zero_atol)
end

function KronFreeCircuits.Spec(qc::QuantizedCircuit{Py}, S::AbstractMatrix{<:Real};
                               zero_rtol::Real=_ZERO_RTOL, zero_atol::Real=_ZERO_ATOL)
    zero_rtol = _check_zero_tolerance(zero_rtol, :zero_rtol)
    zero_atol = _check_zero_tolerance(zero_atol, :zero_atol)
    _check_frame(S, length(qc.mode_names))
    return _numeric_spec(_transformed(qc.handle, S), qc.mode_names; zero_rtol, zero_atol)
end

# --- Run records -----------------------------------------------------------------------------------------------------
# A record is the spec a run solved, written by `spec_json`, with provenance blocks that rebuild the run.
# `_circuit`, `_frame`, `_values` and `_generator` come from `circuit_quantization.to_record` of the circuit in the
# frame `frame`, so they mean to a Python reader what they mean here; the spec fields that `to_record` also writes are
# dropped.
# The Julia-owned key `_kronfree` adds what that record cannot say: the frame of the circuit itself (`base_frame`, in
# the matrix encoding of `_frame`, `null` when no `frame` was applied), the `frame` argument, the `values` in the frame
# of the circuit as JSON numbers, and the round-off zeroing tolerances `zero_rtol` and `zero_atol`.

# This package's git commit, or `nothing` when it is not a git checkout or git fails or takes over 2 s.
function _git_commit()::Union{Nothing,String}
    dir = pkgdir(KronFreeCircuits)
    try
        out = Pipe()
        proc = run(pipeline(ignorestatus(`git -C $dir rev-parse --show-toplevel HEAD`); stdout=out, stderr=devnull);
                   wait=false)
        close(out.in)
        timer = Timer(_ -> process_running(proc) && kill(proc), 2)
        text = read(out, String)
        wait(proc)
        close(timer)
        success(proc) || return nothing
        lines = split(strip(text), '\n')
        length(lines) == 2 && realpath(lines[1]) == realpath(dir) || return nothing
        return occursin(r"^[0-9a-f]{40,64}$", lines[2]) ? String(lines[2]) : nothing
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
end

# A metadata value as a Python object `json.dumps` writes as finite JSON; `where` names it in the error.
function _json_value(x, where::AbstractString)::Py
    x === nothing && return pybuiltins.None
    x isa Union{Bool,AbstractString} && return Py(x)
    x isa Integer && return pyint(BigInt(x))
    if x isa AbstractFloat
        isfinite(x) || throw(ArgumentError("metadata: $where = $x is not a finite number"))
        return Py(Float64(x))
    end
    x isa Union{AbstractVector,Tuple} && return pylist([_json_value(v, "$where[$i]") for (i, v) in enumerate(x)])
    if x isa Union{NamedTuple,AbstractDict}
        out = pydict()
        for (k, v) in pairs(x)
            k isa Union{AbstractString,Symbol} || throw(ArgumentError("metadata: the key $(repr(k)) in $where is not a string"))
            out[string(k)] = _json_value(v, "$where.$k")
        end
        return out
    end
    throw(ArgumentError("metadata: $where = $(repr(x)) is not JSON-serializable " *
                        "(use nothing, Bool, integers, finite floats, strings, vectors, tuples, NamedTuples or Dicts)"))
end


# The values that the parameters of `framed`, a Result `res` in the frame `frame`, take when `res` is at the
# parameter values of `qc`, its quantization in its own frame, as the SymPy substitution `to_record` takes.
# The fresh sector label `n_g{j}` of a compact mode `j` takes that mode's sector offset in the new frame, the
# `n_g_sector` of circuit-quantization's NumPy frame change (`S_ccᵀ` times the base sector offsets); every other
# parameter takes its value in `qc.values`.
# A parameter of `res` that no exported expression of `framed` contains is left out.
function _framed_subs(framed::Py, qc::QuantizedCircuit{Py}, frame)::Py
    θ = qc.values
    values = Dict{String,Float64}(string(k) => v for (k, v) in pairs(θ))
    if frame !== nothing
        h = _transformed(qc.handle, frame)
        sector = pyconvert(Vector{Float64}, h.n_g_sector)
        for (j, t) in enumerate(pyconvert(Vector{String}, pylist(h.types)))
            t == "c" && (values["n_g$j"] = sector[j])
        end
    end
    symbols = _circuit_symbols(framed)
    subs = pydict()
    for k in keys(θ)                                   # in the circuit's parameter order, as `_values` lists them
        name = string(k)
        haskey(symbols, name) && (subs[symbols[name]] = _subs_value(values[name], symbols, symbols[name]))
    end
    return subs
end

const _RECORD_RTOL = 1e-9          # the check's relative tolerance

# The point at which `record_json`'s check evaluates a symbolic spec: the `k`th parameter takes `0.5 + 0.1 √(k + 1)`,
# positive, distinct and away from the simple values (0, ±1, integers) at which a coefficient could vanish by accident.
_sample_point(names::Vector{Symbol}) =
    NamedTuple{Tuple(names)}(Tuple(0.5 + 0.1 * sqrt(k + 1.0) for k in eachindex(names)))

_shown(x::Number) = isreal(x) ? real(x) : x

# Where the spec `a`, given to `record_json`, and the spec `b`, rebuilt from the record's provenance, first differ, as
# the text of an error, or `nothing` when they agree.
# The parameters, the mode names, the mode types, the number of cosines and the cosine vectors are compared first.
# Every coefficient is then evaluated (at `_sample_point` of the parameters when there are any) and compared with the
# relative tolerance `_RECORD_RTOL`: an entry of `E_C`, `E_L`, `flux_linear` or `n_g` relative to the largest
# magnitude of that matrix or vector in either spec, an omega or kappa entry relative to the largest magnitude of
# that vector, a cosine's energy relative to the larger of the two energies, its offset (a phase) relative to 1 or
# the larger offset, and the constant relative to the largest energy of either spec (`E_C`, `E_L` and the cosine
# energies).
function _record_mismatch(a::Spec, b::Spec)::Union{Nothing,String}
    differ(field, x, y) = "$field: $x in the spec, $y in the spec rebuilt from circuit, frame, values and the zero " *
                          "tolerances"
    a.parameters == b.parameters || return differ("parameters", a.parameters, b.parameters)
    length(a.modes) == length(b.modes) || return differ("the number of modes", length(a.modes), length(b.modes))
    for key in (:name, :type), (k, (m, n)) in enumerate(zip(a.modes, b.modes))
        getfield(m, key) == getfield(n, key) || return differ("modes[$k].$key", getfield(m, key), getfield(n, key))
    end
    for (field, x, y) in (("cosines", a.cosines, b.cosines), ("charge_cosines", a.charge_cosines, b.charge_cosines))
        length(x) == length(y) || return differ("the number of $field", length(x), length(y))
    end
    # numbers `x` and `y` of the field `field`, indexed alike, within `_RECORD_RTOL` of their largest magnitude (at
    # least `floor`)
    within(field, x::Number, y::Number; floor=0.0) = within(field, fill(x), fill(y); floor)   # 0-dimensional
    function within(field, x::AbstractArray, y::AbstractArray; floor=0.0)
        scale = max(floor, maximum(abs, x; init=0.0), maximum(abs, y; init=0.0))
        for i in CartesianIndices(x)
            (isequal(x[i], y[i]) || abs(x[i] - y[i]) <= _RECORD_RTOL * scale) && continue
            index = ndims(x) == 0 ? "" : "[$(join(Tuple(i), ", "))]"
            return differ(field * index, _shown(x[i]), _shown(y[i]))
        end
        return nothing
    end
    for (field, x, y, vector) in (("cosines", a.cosines, b.cosines, c -> c.omega),
                                  ("charge_cosines", a.charge_cosines, b.charge_cosines, c -> c.kappa))
        for k in eachindex(x)
            m = within("$field[$k].$(field == "cosines" ? "omega" : "kappa")", vector(x[k]), vector(y[k]))
            m === nothing || return m
        end
    end
    θ = _sample_point(a.parameters)
    at(c::Coefficient) = c(θ)
    ec_a, ec_b, el_a, el_b = at.(a.E_C), at.(b.E_C), at.(a.E_L), at.(b.E_L)
    for (field, x, y) in (("E_C", ec_a, ec_b), ("E_L", el_a, el_b),
                          ("flux_linear", at.(a.flux_linear), at.(b.flux_linear)), ("n_g", at.(a.n_g), at.(b.n_g)))
        m = within(field, x, y)
        m === nothing || return m
    end
    energy = maximum(abs, Iterators.flatten((ec_a, ec_b, el_a, el_b)); init=0.0)
    for (field, x, y) in (("cosines", a.cosines, b.cosines), ("charge_cosines", a.charge_cosines, b.charge_cosines))
        for k in eachindex(x)
            ea, eb = at(x[k].energy), at(y[k].energy)
            energy = max(energy, abs(ea), abs(eb))
            m = within("$field[$k].energy", ea, eb)
            m === nothing || return m
            m = within("$field[$k].offset", at(x[k].offset), at(y[k].offset); floor=1.0)
            m === nothing || return m
        end
    end
    return within("constant", at(a.constant), at(b.constant); floor=energy)
end

"""
    record_json(spec::Spec; circuit::Py, frame=nothing, values=nothing, metadata::NamedTuple=(;), zero_rtol=1e-12,
                zero_atol=0.0, check::Bool=true) -> String

The record of a run that solved `spec`, as JSON text: a kronfree-spec version 1 document whose spec fields are those of `spec` as [`spec_json`](@ref) writes them, so `parse_spec` of the text gives `spec` back coefficient for coefficient, and also a circuit-quantization record that `circuit_quantization.load_record` rebuilds.
Nothing in the spec fields is recomputed from `circuit`, `frame` or `values`.

`circuit` is the circuit-quantization `Result` the run started from, and `frame` a square real matrix `S` (or `nothing`) taking it to the coordinates `φ = S φ′` of the spec, checked exactly as `Spec(res; frame)` checks it.
The blocks `_circuit`, `_frame`, `_values` and `_generator` are those of `circuit_quantization.to_record` of `circuit` in the frame `frame`; the spec fields that `to_record` also writes are not kept.
`values` gives the parameter values in the frame of `circuit`, exactly as `QuantizedCircuit(circuit; values)` takes them and checked by the same rules: a `NamedTuple`, or an `AbstractDict` keyed by `Symbol`s or strings, with a finite real number for every parameter of `Spec(circuit)`.
In the new frame a compact mode's sector label `n_g{j}` is a fresh parameter (see `Spec(res; frame)`), so `_values` gives it `S_ccᵀ n_g`, the base sector offsets transformed as circuit-quantization's numeric frame change transforms them, and every other parameter its value in `values`.
`zero_rtol` and `zero_atol` are the round-off zeroing tolerances `spec` was built with, recorded so that the run can be rebuilt; they are checked as `Spec(res)` checks them and need circuit-quantization at commit c8c8714 or later.
`_generator.kronfree` holds this package's version and git commit (`null` when not available), and `metadata` is written under `_metadata` and must be JSON-serializable.
The Julia-only key `_kronfree` records the frame of `circuit` itself, `frame`, the base-frame `values` in the circuit's parameter order, and `zero_rtol` and `zero_atol`, so that [`load_record`](@ref) gives back `circuit`, `frame`, `values` and the tolerances separately.

With `check = true` (the default), the spec is rebuilt from the provenance before anything is returned and compared with `spec`.
With `values` it is rebuilt by the loop route, `Spec(QuantizedCircuit(circuit; values), frame; zero_rtol, zero_atol)` (`Spec(QuantizedCircuit(circuit; values); zero_rtol, zero_atol)` when `frame === nothing`); without `values` it is `Spec(circuit; frame, zero_rtol, zero_atol)`, read from the spec fields of `to_record`'s output, which are that spec's JSON.
The two specs must have the same parameters, mode names and mode types, the same number of cosines, and the same cosine vectors and coefficients, the coefficients evaluated at a fixed sample point of the parameters when there are any.
Numbers are compared with relative tolerance `1e-9`: an entry of `E_C`, `E_L`, `flux_linear` or `n_g` relative to the largest magnitude of that matrix or vector in either spec, an omega entry relative to the largest magnitude of its vector, a cosine's energy relative to the larger of the two energies, its offset relative to 1 or the larger offset, and the constant relative to the largest energy in either spec.
So a spec zeroed with a different tolerance than the one recorded still passes when every entry zeroed by one and not the other is below `1e-9` of its scale.
A mismatch is an `ArgumentError` that names the first field that differs, with both values; mode names other than the circuit's defaults (from `mode_names`) are a mismatch.
`check = false` skips the comparison, for a spec edited by hand or a loop that writes many records; the provenance blocks are still written from `circuit`, `frame` and `values`.
"""
function KronFreeCircuits.record_json(spec::Spec; circuit::Py, frame::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                      values=nothing, metadata::NamedTuple=(;), zero_rtol::Real=_ZERO_RTOL,
                                      zero_atol::Real=_ZERO_ATOL, check::Bool=true)
    res = circuit
    zero_rtol = _check_zero_tolerance(zero_rtol, :zero_rtol)
    zero_atol = _check_zero_tolerance(zero_atol, :zero_atol)
    meta = _json_value(metadata, "metadata")
    _require_zero_keywords()
    cq, json = circuit_quantization(), pyimport("json")
    framed = frame === nothing ? res : _in_frame(res, frame)
    qc, subs = nothing, nothing
    if values !== nothing
        qc = QuantizedCircuit(res; values)
        subs = _framed_subs(framed, qc, frame)
        if check
            rebuilt = frame === nothing ? Spec(qc; zero_rtol, zero_atol) : Spec(qc, frame; zero_rtol, zero_atol)
            _check_record(spec, rebuilt)
        end
    end
    data = cq.to_record(framed; subs, zero_rtol, zero_atol)
    # Without values, `to_record`'s spec fields are `to_spec(framed; zero_rtol, zero_atol)`: the JSON of
    # `Spec(res; frame, zero_rtol, zero_atol)`.
    check && qc === nothing && _check_record(spec, parse_spec(pyconvert(String, json.dumps(data; allow_nan=false))))

    base = pybuiltins.None
    if frame !== nothing
        # The frame of `res` in the encoding of `_frame`: its matrices name the symbols of `_circuit.symbols` by id, so
        # every symbol they use must have the same id in both records.
        plain = cq.to_record(res; simplify=false)
        ids = Set(pyconvert(String, pystr(s)) for s in data["_circuit"]["symbols"])
        all(pyconvert(String, pystr(s)) in ids for s in plain["_circuit"]["symbols"]) ||
            throw(ArgumentError("record_json: the coordinates of circuit use symbols that the framed record names " *
                                "differently"))
        base = plain["_frame"]
    end
    # The spec fields as `spec_json` writes them, then the provenance blocks of `to_record` (every key that starts with
    # `_`), then this package's own.
    doc = json.loads(spec_json(spec))
    for key in data.keys()
        startswith(pyconvert(String, key), "_") && (doc[key] = data[key])
    end
    doc["_generator"]["kronfree"] = pydict(Dict("version" => string(pkgversion(KronFreeCircuits)),
                                                "git_commit" => something(_git_commit(), pybuiltins.None)))
    θ = qc === nothing ? nothing : qc.values
    doc["_kronfree"] = pydict(Dict(
        "base_frame" => base,
        "frame" => frame === nothing ? pybuiltins.None : pylist([pylist(Float64.(r)) for r in eachrow(frame)]),
        "values" => θ === nothing ? pybuiltins.None :
                    pylist([pydict(Dict("name" => string(k), "value" => v)) for (k, v) in pairs(θ)]),
        "zero_rtol" => zero_rtol, "zero_atol" => zero_atol))
    doc["_metadata"] = meta
    return pyconvert(String, json.dumps(doc; indent=2, ensure_ascii=false, allow_nan=false))
end

# An `ArgumentError` unless `_record_mismatch` finds the given spec and the rebuilt one to agree.
function _check_record(spec::Spec, rebuilt::Spec)
    m = _record_mismatch(spec, rebuilt)
    m === nothing || throw(ArgumentError("record_json: the record's circuit, frame and values do not describe the " *
                                         "spec: $m; check = false writes the record anyway"))
    return nothing
end

"""
    save_record(path::AbstractString, spec::Spec; circuit::Py, frame=nothing, values=nothing,
                metadata::NamedTuple=(;), zero_rtol=1e-12, zero_atol=0.0, check::Bool=true) -> path

Write [`record_json`](@ref) of `spec`, with the same keywords, to `path`, followed by a newline, and return `path`, as [`save_spec`](@ref) does for a spec.
[`load_record`](@ref) reads the file back.
"""
function KronFreeCircuits.save_record(path::AbstractString, spec::Spec; circuit::Py, kwargs...)
    write(path, KronFreeCircuits.record_json(spec; circuit, kwargs...), "\n")
    return path
end

# The JSON text of `source`: the text itself when it starts with `{` (after whitespace), otherwise the file it names.
function _record_text(source::AbstractString)::String
    startswith(lstrip(source), '{') && return String(source)
    isfile(source) || throw(ArgumentError("load_record: no file $(repr(source)) (JSON text must start with \"{\")"))
    return read(source, String)
end

_record_error(what) = throw(ArgumentError("load_record: malformed _kronfree block: $what"))

"""
    load_record(source::AbstractString) -> (; result::Py, spec::Spec, frame, values, zero_rtol, zero_atol)

Read a record written by [`save_record`](@ref) or [`record_json`](@ref), from JSON text or from a file.
`source` is JSON text when its first non-blank character is `{`, and otherwise the path of a file (write a path that begins with `{` as `./{...`); a missing file is an `ArgumentError`.
`spec` is `parse_spec` of the document: the spec the run solved, as it was given to `record_json`.
`result` is the symbolic circuit-quantization `Result` that was passed as `circuit`, rebuilt in its own frame (with its own residual frame change, if it had one) by `circuit_quantization.load_record`.
`frame` is the `frame` that was passed (a `Matrix{Float64}`, or `nothing`), so `Spec(result; frame, zero_rtol, zero_atol)` is the symbolic spec of the run.
`values` is the `NamedTuple` of `Float64` parameter values that was passed, in the frame of `result` and in the circuit's parameter order, or `nothing`; so `Spec(QuantizedCircuit(result; values), frame; zero_rtol, zero_atol)` rebuilds the spec of the run, which the check of `record_json` compared with `spec` unless the record was written with `check = false`.
`zero_rtol` and `zero_atol` are the round-off zeroing tolerances of the run, as `Float64`.
Nothing in the document is evaluated: numbers are read as JSON numbers and expressions by circuit-quantization's parser.
"""
function KronFreeCircuits.load_record(source::AbstractString)
    text = _record_text(source)
    spec = parse_spec(text)
    data = pyimport("json").loads(text)
    pyin("_kronfree", data) || _record_error("missing; the document was not written by record_json or save_record")
    own = data["_kronfree"]
    pyisinstance(own, pybuiltins.dict) || _record_error("not an object")
    field(key) = pyin(key, own) ? own[key] : _record_error("missing \"$key\"")
    isnull(x) = pyis(x, pybuiltins.None)
    function number(x)
        (pyisinstance(x, pybuiltins.bool) || !pyisinstance(x, pytuple((pybuiltins.int, pybuiltins.float)))) &&
            _record_error("$(pystr(x)) is not a number")
        y = pyconvert(Float64, x)
        return isfinite(y) ? y : _record_error("$y is not finite")
    end

    frame = field("frame")
    if !isnull(frame)
        rows = [[number(x) for x in r] for r in frame]
        n = length(rows)
        all(length(r) == n for r in rows) || _record_error("frame is not a square matrix")
        frame = [rows[i][j] for i in 1:n, j in 1:n]
    else
        frame = nothing
    end
    values = field("values")
    values = isnull(values) ? nothing :
             NamedTuple(Symbol(pyconvert(String, item["name"])) => number(item["value"]) for item in values)
    zero_rtol, zero_atol = number(field("zero_rtol")), number(field("zero_atol"))
    min(zero_rtol, zero_atol) >= 0 || _record_error("a zero tolerance is negative")

    # The Result as it was passed to record_json: the record with the frame of `res` in place of `_frame`, and without
    # `_values`, so that circuit-quantization rebuilds it symbolic.
    base = field("base_frame")
    isnull(base) == (frame === nothing) || _record_error("base_frame and frame must be given together")
    rebuilt = pydict(data)
    isnull(base) || (rebuilt["_frame"] = base)
    rebuilt["_values"] = pylist()
    result = circuit_quantization().load_record(rebuilt)
    return (; result, spec, frame, values, zero_rtol, zero_atol)
end

end
