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
    M = [_coefficient(rows[i][j], params, "$key[$i,$j]") for i in 1:N, j in 1:N]
    _check_symmetric(M, rows, key)
    return M
end

# E_C and E_L are symmetric by construction (a scaled inverse capacitance and an inverse inductance matrix), and only
# their symmetric part enters the Hamiltonian, so an asymmetric pair is a typo or an exporter bug: it is rejected, not
# symmetrized (Eli, 2026-10-19).
# Identical expressions pass; otherwise a pair with a non-analytic entry (`_nonanalytic`) is rejected, since agreement at
# test points cannot prove it equal, and an analytic pair is compared at the three test points of `_test_points`.
function _check_symmetric(M::Matrix{Coefficient}, rows, key::Symbol)
    shown(x) = x isa AbstractString ? "\"$x\"" : repr(x)
    for i in axes(M, 1), j in i+1:size(M, 2)
        a, b = M[i, j], M[j, i]
        a.ex == b.ex && continue
        pair = "$key[$i,$j] = $(shown(rows[i][j])) and $key[$j,$i] = $(shown(rows[j][i]))"
        for (c, raw) in ((a, rows[i][j]), (b, rows[j][i]))
            bad = _nonanalytic(c.ex)
            bad === nothing ||
                _spec_error("$key must be symmetric, but $pair are not identical and $(shown(raw)) $bad, so a " *
                            "comparison at test points cannot prove them equal (the coefficient is not real-analytic)")
        end
        points = _test_points((a, b), (;))
        va, vb = _at_points(a, points), _at_points(b, points)
        (va === nothing || vb === nothing) &&
            _spec_error("$key must be symmetric, but $pair cannot be evaluated at a test point to compare them")
        all(_agree.(va, vb)) || _spec_error("$key must be symmetric, but $pair differ")
    end
    return nothing
end

function _numbers(v, key::AbstractString, N::Int)
    (v isa AbstractVector && length(v) == N && all(x -> x isa Real && !(x isa Bool), v)) ||
        _spec_error("$key must be an array of $N numbers (the number of modes)")
    return Float64.(v)
end

# Mode `i` of a spec, with its type checked.
function _mode(i::Int, name, type)
    t = Symbol(type)
    t in _MODE_TYPES || _spec_error("modes[$i] (\"$name\") has type \"$t\"; expected c, e or h")
    return ModeSpec(Symbol(name), t)
end

function _check_mode_names(modes::Vector{ModeSpec})
    allunique(m.name for m in modes) || _spec_error("mode names must be unique")
    isempty(modes) && _spec_error("modes must be nonempty")
    return nothing
end

# A cosine's `omega` must be an integer (to `_INTEGER_TOL`) on every compact mode.
function _check_compact_omega(omega::Vector{Float64}, modes::Vector{ModeSpec}, where_::AbstractString)
    for (j, m) in enumerate(modes)
        m.type === :c && abs(omega[j] - round(omega[j])) > _INTEGER_TOL &&
            _spec_error("$where_.omega[$j] = $(omega[j]) on compact mode $(m.name) is not an integer")
    end
    return nothing
end

# A declared parameter name as the Julia identifier an expression's use of it parses to (e.g. µ U+00B5 is μ U+03BC).
function _identifier(name::AbstractString)
    ex = Meta.parse(name; raise=false)
    return ex isa Symbol ? ex : Symbol(name)
end

"""
    parse_spec(json::AbstractString) -> Spec

Parse and validate a `kronfree-spec` version 1 JSON document.
Keys starting with `_` are ignored; any other unknown key is an error.
Every expression (`E_C`, `E_L`, `flux_linear`, `n_g`, cosine and charge-cosine `energy` and `offset`, `constant`) must be
real-valued: one that uses `im` or calls `cis` or `cispi` is rejected.
`E_C` and `E_L` must be symmetric: identical entries `[i,j]` and `[j,i]` pass; otherwise the pair is rejected if either
entry is not real-analytic (the syntactic gate of `classify`: `abs`, `sign`, or a branch cut), and an analytic pair must
agree at three fixed pseudo-random parameter points (to `1e-12` relative).
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
    params = Symbol[_identifier(p) for p in d[:parameters]]
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
        push!(modes, _mode(i, m[:name], m[:type]))
    end
    _check_mode_names(modes)
    N = length(modes)

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
        _check_compact_omega(omega, modes, where_)
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

# A pair of `numeric_spec` matrix entries both this small relative to the largest entry is round-off of a zero.
const _ROUNDOFF_ZERO = 1e-15

function _numeric_matrix(M::AbstractMatrix{<:Real}, key::Symbol, N::Int)
    (size(M) == (N, N) && !(eltype(M) <: Bool)) ||
        _spec_error("$key must be a $N×$N matrix of numbers (the number of modes)")
    return Matrix{Float64}(M)
end

function _check_finite(x::Real, where_::AbstractString)
    isfinite(x) || _spec_error("$where_ = $x is not finite")
    return nothing
end

# Symmetrize a numeric `E_C` or `E_L` at round-off level, as the circuit-quantization exporter does for a numeric
# matrix: with `scale = max|M|`, a pair differing by at most `tol * scale` becomes its mean, a pair whose entries are
# both at most `_ROUNDOFF_ZERO * scale` in magnitude becomes exactly zero, and a larger asymmetry is rejected.
function _symmetrize!(M::Matrix{Float64}, key::Symbol, tol::Real)
    scale = maximum(abs, M; init=0.0)
    for i in axes(M, 1), j in i+1:size(M, 2)
        a, b = M[i, j], M[j, i]
        abs(a - b) <= tol * scale ||
            _spec_error("$key must be symmetric, but $key[$i,$j] = $a and $key[$j,$i] = $b differ by more than " *
                        "symmetry_tol = $tol relative to max|$key| = $scale")
        M[i, j] = M[j, i] = max(abs(a), abs(b)) <= _ROUNDOFF_ZERO * scale ? 0.0 : (a + b) / 2
    end
    return M
end

"""
    numeric_spec(; modes, E_C, E_L, flux_linear, n_g, cosines=Tuple[], constant=0.0, symmetry_tol=1e-10) -> Spec

A `Spec` with no parameters from plain numeric arrays, the same `Spec` that [`parse_spec`](@ref) builds from a JSON
document of the same numbers.
`modes` is a vector of `name => type` pairs with `type` one of `:c`, `:e` or `:h`.
`E_C` and `E_L` are `N×N` real matrices, `flux_linear` and `n_g` real vectors of length `N = length(modes)`.
Each cosine is a tuple `(energy, omega, offset)` with `omega` a real vector of length `N`; an `omega` entry on a compact
mode within `1e-12` of an integer is rounded to it, any other is rejected.
Every coefficient is a constant (nothing is compiled), and an exact zero is a literal zero, as `"0"` is in a JSON spec.
`E_C` and `E_L` are symmetrized at round-off level: with `scale` the largest magnitude in the matrix, a pair
`[i,j]`, `[j,i]` differing by at most `symmetry_tol * scale` is replaced by its mean, and a pair whose entries are both
at most `1e-15 * scale` in magnitude by exactly zero.
The caller's arrays are not modified.
Throws `ArgumentError` on a wrong shape, an unknown mode type, a repeated or missing mode name, a non-finite number, a
larger asymmetry (naming the pair) or a non-integer `omega` on a compact mode.
"""
function numeric_spec(; modes::AbstractVector{<:Pair{Symbol,Symbol}},
                      E_C::AbstractMatrix{<:Real}, E_L::AbstractMatrix{<:Real},
                      flux_linear::AbstractVector{<:Real}, n_g::AbstractVector{<:Real},
                      cosines::AbstractVector = Tuple[], constant::Real = 0.0, symmetry_tol::Real = 1e-10)
    N = length(modes)
    EC, EL = _numeric_matrix(E_C, :E_C, N), _numeric_matrix(E_L, :E_L, N)
    fl, ng = _numbers(flux_linear, "flux_linear", N), _numbers(n_g, "n_g", N)
    entries = map(enumerate(cosines)) do (k, c)
        where_ = "cosines[$k]"
        (c isa Union{Tuple,AbstractVector} && length(c) == 3) ||
            _spec_error("$where_ must be a tuple (energy, omega, offset)")
        energy, offset = c[1], c[3]
        (energy isa Real && !(energy isa Bool) && offset isa Real && !(offset isa Bool)) ||
            _spec_error("$where_: energy and offset must be real numbers")
        (where_, Float64(energy), _numbers(c[2], "$where_.omega", N), Float64(offset))
    end
    (constant isa Bool) && _spec_error("constant must be a real number")

    specs = [_mode(i, name, type) for (i, (name, type)) in enumerate(modes)]
    _check_mode_names(specs)

    for (key, A) in ((:E_C, EC), (:E_L, EL)), I in CartesianIndices(A)
        _check_finite(A[I], "$key[$(I[1]),$(I[2])]")
    end
    for (key, v) in (("flux_linear", fl), ("n_g", ng)), i in eachindex(v)
        _check_finite(v[i], "$key[$i]")
    end
    for (where_, energy, omega, offset) in entries
        _check_finite(energy, "$where_.energy")
        _check_finite(offset, "$where_.offset")
        for j in eachindex(omega)
            _check_finite(omega[j], "$where_.omega[$j]")
        end
    end
    _check_finite(constant, "constant")

    _symmetrize!(EC, :E_C, symmetry_tol)
    _symmetrize!(EL, :E_L, symmetry_tol)

    cosine_specs = map(entries) do (where_, energy, omega, offset)
        _check_compact_omega(omega, specs, where_)
        for (j, m) in enumerate(specs)
            m.type === :c && (omega[j] = round(omega[j]))
        end
        CosineSpec(Coefficient(energy), omega, Coefficient(offset))
    end
    return Spec(Symbol[], specs, Coefficient.(EC), Coefficient.(EL), Coefficient.(fl), Coefficient.(ng),
                CosineSpec[cosine_specs...], ChargeCosineSpec[], Coefficient(Float64(constant)))
end
