# Low-level encoding of the save file: tagged values and settings structs.
# The file holds only numbers, Bool, String and arrays of Int64/Float64/ComplexF64/String (never a Symbol or an auto-encoded
# struct), so it stays readable without KronFreeCircuits. A value is `name/kind` plus `name/value` (or `name/k` etc.).

const _PLAIN_KINDS = Dict(Int => "int", Float64 => "float", Bool => "bool", String => "string", Vector{Int} => "ints",
                          Vector{Float64} => "floats", Vector{String} => "strings",
                          Matrix{Float64} => "matrix_real", Matrix{ComplexF64} => "matrix_complex")
# A custom solver's setting that is not a supported value: only its type is saved.
struct _Unsaved
    type::String
end

# The kind tag of a supported value, or `nothing`.
function _value_kind(v)
    kind = get(_PLAIN_KINDS, typeof(v), nothing)
    kind === nothing || return kind
    return v isa Symbol ? "symbol" : v === nothing ? "nothing" : v isa Port ? "port" : v isa Tuple{Symbol,Port} ? "modeport" :
           v isa Type ? "arraytype" : v isa _Unsaved ? "unsaved" : nothing
end

function _write_value(g, name, v)
    kind = _value_kind(v)
    kind === nothing && throw(ArgumentError("cannot save $name: a $(typeof(v)) is not a supported value"))
    g["$name/kind"] = kind
    if kind in values(_PLAIN_KINDS); g["$name/value"] = v
    elseif kind == "symbol"; g["$name/value"] = String(v)
    elseif kind == "port"; g["$name/value"] = String(v.kind); g["$name/k"] = v.k
    elseif kind == "modeport"; g["$name/mode"] = String(v[1]); g["$name/port_kind"] = String(v[2].kind); g["$name/k"] = v[2].k
    elseif kind == "arraytype"; g["$name/value"] = string(nameof(v))
    elseif kind == "unsaved"; g["$name/value"] = v.type
    end
    return nothing
end

function _read_value(g, name)
    kind = g["$name/kind"]
    kind == "nothing" && return nothing
    kind == "modeport" && return (Symbol(g["$name/mode"]), Port(Symbol(g["$name/port_kind"]), g["$name/k"]))
    kind == "symbol" && return Symbol(g["$name/value"])
    kind == "port" && return Port(Symbol(g["$name/value"]), g["$name/k"])
    kind == "unsaved" && return _Unsaved(g["$name/value"])
    kind in values(_PLAIN_KINDS) || kind == "arraytype" || throw(ArgumentError("unknown value kind \"$kind\" at $name"))
    return g["$name/value"]
end

# every concrete solver, selection rule and basis (the test suite checks this list is complete)
const _SAVE_TYPES = Dict{String,DataType}(string(nameof(T)) => T for T in
    (Dense, Lanczos, LOBPCG, ShiftInvert, ChebyshevFilter, Auto, Lowest, Indices, Connected, Follow, Within,
     ChargeBasis, PhaseGrid, PeriodicGrid, FockBasis))

function _write_struct(g, name, x)
    g["$name/type"] = string(nameof(typeof(x)))
    for f in fieldnames(typeof(x))
        _write_value(g, "$name/fields/$f", f === :callback ? nothing : getfield(x, f))   # a callback is not saved
    end
    return nothing
end

"""
    SavedSolver(name, fields)

Stands in, in a system read by [`load_system`](@ref), for a custom solver whose type was not passed in `solvers`:
`name` is the solver's type name and `fields` its saved settings (`field => value`; a setting that could not be saved
holds only its type). The node's results, ports and derivatives are those saved; solving with it throws.
"""
struct SavedSolver <: Solver
    name::String
    fields::Vector{Pair{Symbol,Any}}
end

solve(s::SavedSolver, H, m; kwargs...) = throw(ArgumentError(
    "this node was solved with the custom solver $(s.name), which is not available here: load the file with " *
    "load_system(path; solvers=[$(s.name)]) where $(s.name) is defined, or give the node another solver"))

# The node's solver. A custom one (not in `_SAVE_TYPES`) is marked `custom`, and a setting that is not a supported value
# is saved as its type only; a `SavedSolver` is written back under its original name.
function _write_solver(g, name, s::Solver)
    builtin = haskey(_SAVE_TYPES, string(nameof(typeof(s))))
    g["$name/custom"] = !builtin
    builtin && return _write_struct(g, name, s)
    fields = s isa SavedSolver ? s.fields : [f => getfield(s, f) for f in fieldnames(typeof(s))]
    g["$name/type"] = s isa SavedSolver ? s.name : string(nameof(typeof(s)))
    for (f, v) in fields
        _write_value(g, "$name/fields/$f", _value_kind(v) === nothing ? _Unsaved(string(typeof(v))) : v)
    end
    return nothing
end

# The node's solver: a built-in one, a custom one rebuilt from `solvers` (type name => type) when all its settings
# were saved, or a `SavedSolver`.
function _read_solver(g, name; on, solvers)
    g["$name/custom"] || return _read_struct(g, name; on)
    type = g["$name/type"]
    names = haskey(g, "$name/fields") ? keys(g["$name/fields"]) : String[]   # a solver with no settings has no group
    fields = Pair{Symbol,Any}[Symbol(f) => _read_value(g, "$name/fields/$f") for f in names]
    T = get(solvers, type, nothing)
    T === nothing && return SavedSolver(type, fields)
    unsaved = [f for (f, v) in fields if v isa _Unsaved]
    if !isempty(unsaved)
        @warn "$type: settings $(join(unsaved, ", ")) were not saved, so the node keeps a SavedSolver"
        return SavedSolver(type, fields)
    end
    vals = Dict(fields)
    missing = [f for f in fieldnames(T) if f !== :on && !haskey(vals, f)]
    isempty(missing) || throw(ArgumentError("$type in the file has no saved $(join(missing, ", ")): the solvers entry " *
                                            "has different fields from the solver that was saved"))
    return T((f === :on ? on : vals[f] for f in fieldnames(T))...)
end

function _read_struct(g, name; on=Array)
    type = g["$name/type"]
    haskey(_SAVE_TYPES, type) || throw(ArgumentError("unknown type \"$type\" at $name"))
    T = _SAVE_TYPES[type]
    fields = map(fieldnames(T)) do f
        f === :on ? on : f === :callback ? nothing : _read_value(g, "$name/fields/$f")
    end
    return T(fields...)
end
