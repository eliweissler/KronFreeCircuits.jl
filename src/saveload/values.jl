# Low-level encoding of the save file: tagged values and settings structs.
# The file holds only numbers, Bool, String and arrays of Int64/Float64/ComplexF64/String (never a Symbol or an auto-encoded
# struct), so it stays readable without KronFreeCircuits. A value is `name/kind` plus `name/value` (or `name/k` etc.).

const _PLAIN_KINDS = Dict(Int => "int", Float64 => "float", Bool => "bool", String => "string", Vector{Int} => "ints",
                          Vector{Float64} => "floats", Vector{String} => "strings",
                          Matrix{Float64} => "matrix_real", Matrix{ComplexF64} => "matrix_complex")
function _write_value(g, name, v)
    kind = get(_PLAIN_KINDS, typeof(v), nothing)
    kind === nothing && (kind = v isa Symbol ? "symbol" : v === nothing ? "nothing" : v isa Port ? "port" :
                                v isa Tuple{Symbol,Port} ? "modeport" : v isa Type ? "arraytype" : nothing)
    kind === nothing && throw(ArgumentError("cannot save $name: a $(typeof(v)) is not a supported value"))
    g["$name/kind"] = kind
    if kind in values(_PLAIN_KINDS); g["$name/value"] = v
    elseif kind == "symbol"; g["$name/value"] = String(v)
    elseif kind == "port"; g["$name/value"] = String(v.kind); g["$name/k"] = v.k
    elseif kind == "modeport"; g["$name/mode"] = String(v[1]); g["$name/port_kind"] = String(v[2].kind); g["$name/k"] = v[2].k
    elseif kind == "arraytype"; g["$name/value"] = string(nameof(v))
    end
    return nothing
end

function _read_value(g, name)
    kind = g["$name/kind"]
    kind == "nothing" && return nothing
    kind == "modeport" && return (Symbol(g["$name/mode"]), Port(Symbol(g["$name/port_kind"]), g["$name/k"]))
    kind == "symbol" && return Symbol(g["$name/value"])
    kind == "port" && return Port(Symbol(g["$name/value"]), g["$name/k"])
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

function _read_struct(g, name; on=Array)
    type = g["$name/type"]
    haskey(_SAVE_TYPES, type) || throw(ArgumentError("unknown type \"$type\" at $name"))
    T = _SAVE_TYPES[type]
    fields = map(fieldnames(T)) do f
        f === :on ? on : f === :callback ? nothing : _read_value(g, "$name/fields/$f")
    end
    return T(fields...)
end
