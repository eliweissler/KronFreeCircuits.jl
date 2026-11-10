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

# --- Run records -----------------------------------------------------------------------------------------------------
# The document is `circuit_quantization.to_record` of `res` in the frame `frame`, so `_circuit`, `_frame` and `_values`
# mean to a Python reader what they mean here.
# The Julia-owned key `_kronfree` adds what that record cannot say: the frame of `res` itself (`base_frame`, in the
# matrix encoding of `_frame`, `null` when no `frame` was applied), the `frame` argument and the `values` as JSON numbers.

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
    catch
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

# `values` (name ⇒ number) as `Float64` by name in the given order, checked against the parameters of `res`: an
# unknown name, a value that is not a real number, or a parameter left without a value is an `ArgumentError`.
function _record_values(res::Py, values)::NamedTuple
    symbols = _circuit_symbols(res)
    known = join(sort!(collect(keys(symbols))), ", ")
    out = Pair{Symbol,Float64}[]
    for (k, v) in pairs(values)
        name = string(k)
        haskey(symbols, name) ||
            throw(ArgumentError("values: $name is not a parameter of the circuit; its parameters are $known"))
        v isa Real && !(v isa Bool) && isfinite(v) ||
            throw(ArgumentError("values: $name must be a finite real number, got $(repr(v))"))
        push!(out, Symbol(name) => Float64(v))
    end
    left = setdiff(keys(symbols), string.(first.(out)))
    isempty(left) || throw(ArgumentError("values: no value for " * join(sort!(collect(left)), ", ") *
                                         "; a record with values needs every parameter of the circuit: $known"))
    return NamedTuple(out)
end

"""
    run_record(res::Py; frame=nothing, values=nothing, metadata::NamedTuple=(;)) -> String

The run record of the circuit-quantization `Result` `res`, as JSON text: a kronfree-spec version 1 document that
`parse_spec` reads, and also a circuit-quantization record that `circuit_quantization.load_record` rebuilds.
It is `circuit_quantization.to_record` of `res` in the frame `φ = S φ′` given by `frame` (checked exactly as
`Spec(res; frame)` checks it), so its spec fields, `_circuit`, `_frame` and `_values` are those of
that record.
`values`, a `NamedTuple` (or other name ⇒ number collection) giving a finite real number to every parameter of the
circuit in the new frame, makes the spec fields numeric and is recorded in `_values`; an unknown name or a parameter left
without a value is an `ArgumentError` listing the circuit's parameters.
Without `values` the spec stays symbolic.
`_generator.kronfree` holds this package's version and git commit (`null` when not available), `metadata` is written
under `_metadata` and must be JSON-serializable, and the Julia-only key `_kronfree` records the frame of `res` itself,
`frame` and `values`, so that [`load_record`](@ref) gives back `res` and `frame` separately.
"""
function KronFreeCircuits.run_record(res::Py; frame::Union{Nothing,AbstractMatrix{<:Real}}=nothing, values=nothing,
                                     metadata::NamedTuple=(;))
    cq, json = circuit_quantization(), pyimport("json")
    meta = _json_value(metadata, "metadata")
    framed = frame === nothing ? res : _in_frame(res, frame)
    θ = values === nothing ? nothing : _record_values(framed, values)
    data = cq.to_record(framed; subs=_pysubs(framed, θ))
    base = pybuiltins.None
    if frame !== nothing
        # The frame of `res` in the encoding of `_frame`: its matrices name the symbols of `_circuit.symbols` by id, so
        # every symbol they use must have the same id in both records.
        plain = cq.to_record(res; simplify=false)
        ids = Set(pyconvert(String, pystr(s)) for s in data["_circuit"]["symbols"])
        all(pyconvert(String, pystr(s)) in ids for s in plain["_circuit"]["symbols"]) ||
            throw(ArgumentError("run_record: the coordinates of res use symbols that the framed record names differently"))
        base = plain["_frame"]
    end
    data["_generator"]["kronfree"] = pydict(Dict("version" => string(pkgversion(KronFreeCircuits)),
                                                 "git_commit" => something(_git_commit(), pybuiltins.None)))
    data["_kronfree"] = pydict(Dict(
        "base_frame" => base,
        "frame" => frame === nothing ? pybuiltins.None : pylist([pylist(Float64.(r)) for r in eachrow(frame)]),
        "values" => θ === nothing ? pybuiltins.None :
                    pylist([pydict(Dict("name" => string(k), "value" => v)) for (k, v) in pairs(θ)])))
    data["_metadata"] = meta
    return pyconvert(String, json.dumps(data; indent=2, ensure_ascii=false, allow_nan=false)) * "\n"
end

# The JSON text of `source`: the text itself when it starts with `{` (after whitespace), otherwise the file it names.
function _record_text(source::AbstractString)::String
    startswith(lstrip(source), '{') && return String(source)
    isfile(source) || throw(ArgumentError("load_record: no file $(repr(source)) (JSON text must start with \"{\")"))
    return read(source, String)
end

_record_error(what) = throw(ArgumentError("load_record: malformed _kronfree block: $what"))

"""
    load_record(source::AbstractString) -> (; result::Py, spec::Spec, frame, values)

Read a record written by [`run_record`](@ref), from JSON text or from a file.
`source` is JSON text when its first non-blank character is `{`, and otherwise the path of a file (write a path that
begins with `{` as `./{...`); a missing file is an `ArgumentError`.
`spec` is `parse_spec` of the document: the spec of the run, in the recorded frame, numeric when values were recorded.
`result` is the symbolic circuit-quantization `Result` that was passed to `run_record`, rebuilt in its own frame (with
its own residual frame change, if it had one) by `circuit_quantization.load_record`, and `frame` is the `frame` that was
passed (a `Matrix{Float64}`, or `nothing`), so `Spec(result; frame)` is the symbolic spec of the run.
`values` is the `NamedTuple` of `Float64` parameter values in the order they were given, or `nothing`.
Nothing in the document is evaluated: numbers are read as JSON numbers and expressions by circuit-quantization's parser.
"""
function KronFreeCircuits.load_record(source::AbstractString)
    text = _record_text(source)
    spec = parse_spec(text)
    data = pyimport("json").loads(text)
    pyin("_kronfree", data) || _record_error("missing; the document was not written by run_record")
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

    # The Result as it was passed to run_record: the record with the frame of `res` in place of `_frame`, and without
    # `_values`, so that circuit-quantization rebuilds it symbolic.
    base = field("base_frame")
    isnull(base) == (frame === nothing) || _record_error("base_frame and frame must be given together")
    rebuilt = pydict(data)
    isnull(base) || (rebuilt["_frame"] = base)
    rebuilt["_values"] = pylist()
    result = circuit_quantization().load_record(rebuilt)
    return (; result, spec, frame, values)
end

end
