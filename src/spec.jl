"""
    ModeSpec(name, type)

A circuit mode: its name and type, `:c` (compact), `:e` (extended) or `:h` (harmonic).
"""
struct ModeSpec
    name::Symbol
    type::Symbol
end

"""
    CosineSpec(energy, omega, offset)

The term `-energy * cos(omegaᵀφ + offset)`.
"""
struct CosineSpec
    energy::Coefficient
    omega::Vector{Float64}
    offset::Coefficient
end

"""
    ChargeCosineSpec(energy, kappa, offset)

The term `-energy * cos(2π kappaᵀ(n - n_g) + offset)`.
"""
struct ChargeCosineSpec
    energy::Coefficient
    kappa::Vector{Float64}
    offset::Coefficient
end

"""
    Spec

A circuit Hamiltonian in the canonical form
`4 (n-n_g)ᵀ E_C (n-n_g) + ½ φᵀ E_L φ + φᵀ flux_linear - Σ energy cos(omegaᵀφ + offset)
 - Σ energy cos(2π kappaᵀ(n-n_g) + offset) + constant`,
with every coefficient a compiled expression of the named `parameters`.
Matrices and vectors are indexed in the order of `modes`.
"""
struct Spec
    parameters::Vector{Symbol}
    modes::Vector{ModeSpec}
    E_C::Matrix{Coefficient}
    E_L::Matrix{Coefficient}
    flux_linear::Vector{Coefficient}
    n_g::Vector{Coefficient}
    cosines::Vector{CosineSpec}
    charge_cosines::Vector{ChargeCosineSpec}
    constant::Coefficient
end

const _SPEC_KEYS = (:format, :version, :units, :parameters, :modes, :E_C, :E_L,
                    :flux_linear, :n_g, :cosines, :charge_cosines, :constant)
const _REQUIRED_KEYS = (:format, :version, :parameters, :modes, :E_C, :E_L,
                        :flux_linear, :n_g, :cosines)
const _MODE_TYPES = (:c, :e, :h)
const _INTEGER_TOL = 1e-12

_spec_error(msg) = throw(ArgumentError("spec: " * msg))

# The first complex construct (`im`, or a call of `cis` or `cispi`) in a spec expression, or `nothing`.
function _complex_construct(ex)
    ex === :im && return "the symbol `im`"
    ex isa Expr || return nothing
    ex.head === :call && ex.args[1] in (:cis, :cispi) && return "a call of `$(ex.args[1])`"
    for a in ex.args
        r = _complex_construct(a)
        r === nothing || return r
    end
    return nothing
end

# Every spec expression is real-valued (spec Section 20f, "Real inputs"): complex structure enters only through the
# package's own phase factors and the bases' operators.
function _coefficient(x, params::Vector{Symbol}, where_::AbstractString)
    x isa Real && !(x isa Bool) && return Coefficient(x)
    x isa AbstractString || _spec_error("$where_ must be an expression string, got $(repr(x))")
    c = try
        Coefficient(x, params)
    catch e
        e isa ArgumentError && _spec_error("$where_: $(e.msg)")
        rethrow()
    end
    bad = _complex_construct(c.ex)
    bad === nothing || _spec_error("$where_: the expression \"$x\" uses $bad; spec expressions must be real-valued")
    return c
end

function _vector(d, key::Symbol, N::Int, params)
    v = d[key]
    (v isa AbstractVector && length(v) == N) ||
        _spec_error("$key must be an array of length $N (the number of modes)")
    return [_coefficient(v[i], params, "$key[$i]") for i in 1:N]
end

function _matrix(d, key::Symbol, N::Int, params)
    rows = d[key]
    (rows isa AbstractVector && length(rows) == N &&
     all(r -> r isa AbstractVector && length(r) == N, rows)) ||
        _spec_error("$key must be a $N×$N array of arrays (the number of modes)")
    return [_coefficient(rows[i][j], params, "$key[$i,$j]") for i in 1:N, j in 1:N]
end

function _numbers(v, key::AbstractString, N::Int)
    (v isa AbstractVector && length(v) == N && all(x -> x isa Real && !(x isa Bool), v)) ||
        _spec_error("$key must be an array of $N numbers (the number of modes)")
    return Float64.(v)
end

"""
    parse_spec(json::AbstractString) -> Spec

Parse and validate a `kronfree-spec` version 1 JSON document.
Keys starting with `_` are ignored; any other unknown key is an error.
Every expression (`E_C`, `E_L`, `flux_linear`, `n_g`, cosine and charge-cosine `energy` and `offset`, `constant`) must be
real-valued: one that uses `im` or calls `cis` or `cispi` is rejected.
Throws `ArgumentError` on any invalid content.
"""
function parse_spec(json::AbstractString)
    d = JSON3.read(json)
    d isa JSON3.Object || _spec_error("top level must be a JSON object")
    for k in keys(d)
        startswith(String(k), "_") || k in _SPEC_KEYS || _spec_error("unknown key \"$k\"")
    end
    for k in _REQUIRED_KEYS
        haskey(d, k) || _spec_error("missing required key \"$k\"")
    end
    d[:format] == "kronfree-spec" || _spec_error("format must be \"kronfree-spec\", got $(repr(d[:format]))")
    d[:version] == 1 || _spec_error("version must be 1, got $(repr(d[:version]))")
    !haskey(d, :units) || d[:units] == "GHz" || _spec_error("units must be \"GHz\", got $(repr(d[:units]))")

    (d[:parameters] isa AbstractVector && all(p -> p isa AbstractString, d[:parameters])) ||
        _spec_error("parameters must be an array of strings")
    params = Symbol[Symbol(p) for p in d[:parameters]]
    allunique(params) || _spec_error("parameters must be unique")
    try
        check_parameter_names(params)
    catch e
        e isa ArgumentError && _spec_error(e.msg)
        rethrow()
    end

    for k in (:modes, :cosines)
        d[k] isa AbstractVector || _spec_error("$k must be an array")
    end
    get(d, :charge_cosines, []) isa AbstractVector || _spec_error("charge_cosines must be an array")
    modes = ModeSpec[]
    for (i, m) in enumerate(d[:modes])
        (m isa JSON3.Object && haskey(m, :name) && haskey(m, :type)) ||
            _spec_error("modes[$i] must be an object with \"name\" and \"type\"")
        t = Symbol(m[:type])
        t in _MODE_TYPES || _spec_error("modes[$i] (\"$(m[:name])\") has type \"$t\"; expected c, e or h")
        push!(modes, ModeSpec(Symbol(m[:name]), t))
    end
    allunique(m.name for m in modes) || _spec_error("mode names must be unique")
    N = length(modes)
    N > 0 || _spec_error("modes must be nonempty")

    E_C = _matrix(d, :E_C, N, params)
    E_L = _matrix(d, :E_L, N, params)
    flux_linear = _vector(d, :flux_linear, N, params)
    n_g = _vector(d, :n_g, N, params)

    cosines = CosineSpec[]
    for (k, c) in enumerate(d[:cosines])
        where_ = "cosines[$k]"
        (c isa JSON3.Object && haskey(c, :energy) && haskey(c, :omega) && haskey(c, :offset)) ||
            _spec_error("$where_ must have \"energy\", \"omega\" and \"offset\"")
        omega = _numbers(c[:omega], "$where_.omega", N)
        for (j, m) in enumerate(modes)
            m.type === :c && abs(omega[j] - round(omega[j])) > _INTEGER_TOL &&
                _spec_error("$where_.omega[$j] = $(omega[j]) on compact mode $(m.name) is not an integer")
        end
        push!(cosines, CosineSpec(_coefficient(c[:energy], params, "$where_.energy"), omega,
                                  _coefficient(c[:offset], params, "$where_.offset")))
    end

    charge_cosines = ChargeCosineSpec[]
    for (k, c) in enumerate(get(d, :charge_cosines, ()))
        where_ = "charge_cosines[$k]"
        (c isa JSON3.Object && haskey(c, :energy) && haskey(c, :kappa) && haskey(c, :offset)) ||
            _spec_error("$where_ must have \"energy\", \"kappa\" and \"offset\"")
        kappa = _numbers(c[:kappa], "$where_.kappa", N)
        for (j, m) in enumerate(modes)
            m.type === :c && kappa[j] != 0 &&
                _spec_error("$where_.kappa[$j] = $(kappa[j]) on compact mode $(m.name) must be zero")
        end
        push!(charge_cosines, ChargeCosineSpec(_coefficient(c[:energy], params, "$where_.energy"), kappa,
                                               _coefficient(c[:offset], params, "$where_.offset")))
    end

    constant = _coefficient(get(d, :constant, "0"), params, "constant")
    return Spec(params, modes, E_C, E_L, flux_linear, n_g, cosines, charge_cosines, constant)
end

"""
    load_spec(path::AbstractString) -> Spec

Read and validate the spec JSON file at `path` (see [`parse_spec`](@ref)).
"""
load_spec(path::AbstractString) = parse_spec(read(path, String))
