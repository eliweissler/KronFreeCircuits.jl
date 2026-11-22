# Hamiltonian terms, the parameter point θ and the spec. Coefficients go through the spec grammar (`_coefficient_string`,
# read back by `_saved_coefficient`), so the file holds only strings and numbers.

# The text of `c`, refused unless it parses back to an identical coefficient.
function _checked_string(c::Coefficient, what)
    s = _coefficient_string(c)
    _identical(_saved_coefficient(s, c.deps), c) ||
        throw(ArgumentError("$what: coefficient $(c.ex) does not round-trip; written as \"$s\""))
    return s
end

function _write_terms(g, name, terms::Vector{Term})
    g["$name/coefficients"] = [_checked_string(t.coeff, "term $i") for (i, t) in enumerate(terms)]
    for (i, t) in enumerate(terms)
        g["$name/deps/$i"] = String.(t.coeff.deps)
    end
    fs = [(i, f) for (i, t) in enumerate(terms) for f in t.factors]
    g["$name/factor_term"] = Int[i for (i, _) in fs]
    _write_portlist(g, "$name/factors", [(f.mode, f.port) for (_, f) in fs])
    return nothing
end

function _read_terms(g, name)
    coefficients = g["$name/coefficients"]
    factors = [Factor[] for _ in coefficients]
    for (i, (mode, port)) in zip(g["$name/factor_term"], _read_portlist(g, "$name/factors"))
        push!(factors[i], Factor(mode, port))
    end
    return Term[Term(_saved_coefficient(s, Symbol.(g["$name/deps/$i"])), factors[i]) for (i, s) in enumerate(coefficients)]
end

function _write_theta(g, name, θ::NamedTuple)
    g["$name/names"] = String.(collect(keys(θ)))
    for (i, v) in enumerate(values(θ))
        _write_value(g, "$name/values/$i", v)
    end
    return nothing
end

function _read_theta(g, name)
    names = g["$name/names"]
    return NamedTuple{Tuple(Symbol.(names))}(Tuple(_read_value(g, "$name/values/$i") for i in eachindex(names)))
end

_same_term(a::Term, b::Term) = _identical(a.coeff, b.coeff) && length(a.factors) == length(b.factors) &&
    all(p -> p[1].mode === p[2].mode && p[1].port == p[2].port, zip(a.factors, b.factors))
_terms_identical(a::Vector{Term}, b::Vector{Term}) = length(a) == length(b) && all(_same_term.(a, b))

function _write_system_terms(f, sys::System)
    from_spec = false
    n = 0
    if sys.spec !== nothing
        json = spec_json(sys.spec)
        f["spec"] = json
        rebuilt = parse_spec(json)
        built = circuit_terms(rebuilt)
        n = length(built)
        from_spec = n <= length(sys.terms) && _terms_identical(built, sys.terms[1:n]) &&
                    _identical(constant_term(rebuilt), sys.constant)
    end
    f["terms/from_spec"] = from_spec
    if !from_spec
        n = 0
        f["constant"] = _checked_string(sys.constant, "constant")
        f["constant_deps"] = String.(sys.constant.deps)
    end
    _write_terms(f, "terms", sys.terms[n+1:end])
    return nothing
end

function _read_system_terms(f)
    spec = haskey(f, "spec") ? parse_spec(f["spec"]) : nothing
    extra = _read_terms(f, "terms")
    f["terms/from_spec"] || return spec, extra, _saved_coefficient(f["constant"], Symbol.(f["constant_deps"]))
    return spec, vcat(circuit_terms(spec), extra), constant_term(spec)
end
