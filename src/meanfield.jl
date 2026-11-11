# Mean-field terms: for a term that straddles the children of a node, one child keeps its own factors and every other
# group of factors is replaced by its expectation value in a state of the child that owns it.

"""
    Expectations

One child's expectation value of each of its factor keys (sorted lists of `(mode, port)` pairs).
Two factors of one term that sit in the same child form one key, whose value is the child's joint expectation, not a product of single ones.
"""
const Expectations = Dict{_FactorKey,ComplexF64}

# Every child of `parent` must be a node: a bare `mode => basis` child has no kept states to take an expectation in.
function _check_node_children(parent::Node)
    for ch in parent.children
        ch isa Node && continue
        m = ch isa Pair ? ch.first : ch.mode
        throw(ArgumentError("mean field needs every child of the parent to be a Node, but the child on mode :$m is not; " *
                            "wrap the mode in a Node"))
    end
end

function _child_index(parent::Node, child::Node)
    _check_node_children(parent)
    i = findfirst(ch -> ch === child, parent.children)
    i === nothing && throw(ArgumentError("`child` (on modes $(modes(child))) is not a child of `parent`"))
    return i
end

"""
    straddling_terms(parent, terms) -> Vector{Term}

The terms that `route` places at `parent` when it is the root of its own tree, in their order in `terms`.
These are the terms that straddle two or more of its children.
A term with a factor on a mode outside `parent` is not among them.
"""
function straddling_terms(parent::Node, terms::Vector{Term})
    _check_node_children(parent)
    return _terms_at(parent, terms)
end

# The terms that mean field reduces for `parent`, in order: those that straddle its children, and those that reach
# outside it (a factor on a mode outside `parent` and one inside).
function _reduced_terms(parent::Node, terms::Vector{Term})
    straddling = Base.IdSet{Term}(straddling_terms(parent, terms))
    inside = Set(modes(parent))
    reaches(t) = any(in(inside), modes(t)) && !all(in(inside), modes(t))
    return Term[t for t in terms if t in straddling || reaches(t)]
end

# The sorted key of the factors of `t` on the modes in `inside`.
_key_in(t::Term, inside) = filter(((m, p),) -> m in inside, _factor_key(t))

"""
    child_keys(parent, terms, child) -> Vector{_FactorKey}

The factor keys that the straddling terms of `parent` need inside `child`, each distinct key once, sorted, in order of first appearance.
A term that reaches outside `parent` contributes its key inside `child` too.
"""
function child_keys(parent::Node, terms::Vector{Term}, child::Node)
    _child_index(parent, child)
    inside = Set(modes(child))
    keys = _FactorKey[]
    for t in _reduced_terms(parent, terms)
        K = _key_in(t, inside)
        isempty(K) || K in keys || push!(keys, K)
    end
    return keys
end

# A key's operator on the kept states of `solved`: its stored port or joint port, else computed from the kept vectors.
function _key_matrix(solved::MaterializedNode, leaves, K::_FactorKey)
    if length(K) == 1
        haskey(solved.ports, only(K)) && return solved.ports[only(K)]
        return _node_port(solved.node, leaves, solved.children, solved.dims, solved.vectors, only(K)...)
    end
    haskey(solved.joint_ports, K) && return solved.joint_ports[K]
    return _joint_port(solved.node, leaves, solved.children, solved.dims, solved.vectors, K)
end

"""
    expectations(solved, keys; state=1, weights=nothing) -> Expectations

The expectation value of each key in the `state`-th kept state of the solved child `solved`.
With `weights` it is the average over the kept states instead.
`state` and `weights` are alternatives, and `weights` must be nonnegative, sum to one within `1e-12` and have one entry per kept state.
A one-factor key reads the port of `solved` and a longer one its joint port, computed from the kept vectors when `solved` did not form it.
"""
function expectations(solved::MaterializedNode, keys::Vector{_FactorKey}; state::Int=1,
                      weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    solved.node isa Node || throw(ArgumentError("`solved` must be a materialized Node"))
    nkept = size(solved.vectors, 2)
    if weights === nothing
        1 <= state <= nkept || throw(ArgumentError("state $state is not a kept state: the node keeps $nkept states"))
        w = [i == state ? 1.0 : 0.0 for i in 1:nkept]
    else
        state == 1 || throw(ArgumentError("give either `state` or `weights`, not both"))
        length(weights) == nkept ||
            throw(ArgumentError("`weights` has $(length(weights)) entries, but the node keeps $nkept states"))
        all(>=(0), weights) || throw(ArgumentError("`weights` must be nonnegative"))
        abs(sum(weights) - 1) <= 1e-12 || throw(ArgumentError("`weights` must sum to one, but they sum to $(sum(weights))"))
        w = Float64.(weights)
    end
    leaves = _mode_leaves(solved.node)
    out = Expectations()
    for K in keys
        P = _key_matrix(solved, leaves, _sortkey!(copy(K)))
        out[K] = sum(w[i] * P[i, i] for i in 1:nkept)
    end
    return out
end

# The expectation value of one port at the fixed phase `φ0`.
function _pinned(p::Port, φ0::Float64)
    p.kind === :expiphi && return cis(p.k * φ0)
    p.kind === :phi && return ComplexF64(φ0)
    p.kind === :phi2 && return ComplexF64(φ0^2)
    p.kind === :exp2pin && return 1.0 + 0.0im
    return 0.0 + 0.0im                                    # :n, :n2
end

"""
    pinned_expectations(keys, displacement) -> Expectations

A starting table for a child that sits at the phases `displacement` (a `mode => φ⁰` dictionary; an absent mode has `φ⁰ = 0`).
A key's value is the product over its factors of `exp(i k φ⁰)` for `expiphi(k)`, `φ⁰` for `phi`, `φ⁰^2` for `phi2`, `0` for `n` and `n2`, and `1` for `exp2pin`.
"""
function pinned_expectations(keys::Vector{_FactorKey}, displacement::AbstractDict{Symbol,<:Real})
    return Expectations(K => prod(_pinned(p, Float64(get(displacement, m, 0.0))) for (m, p) in K; init=1.0 + 0.0im) for K in keys)
end

"""
    delocalized_expectations(keys) -> Expectations

A starting table for a fully delocalized child: every key has expectation value zero.
"""
delocalized_expectations(keys::Vector{_FactorKey}) = Expectations(K => 0.0 + 0.0im for K in keys)

# The table entry for `key`, or an `ArgumentError` naming what is missing.
function _lookup(table::Expectations, key::_FactorKey, what)
    haskey(table, key) || throw(ArgumentError("the expectation table of $what lacks the key $(_key_string(key))"))
    return table[key]
end

"""
    mean_field_terms(parent, terms, θ, child, others; outside=nothing) -> Vector{Term}

The extra terms for `child`: each straddling term of `parent` reduced to the factors on `child`'s modes.
Its coefficient at `θ` is multiplied by the expectation value, from `others`, of each other child's group of its factors (one key per child).
`others` maps each other child that a term touches to its table.
Terms with the same factors are merged by adding their coefficients, and a merged term whose coefficient is zero is kept.
A straddling term that does not touch `child` contributes nothing.
The returned terms have constant coefficients.

A term with a factor on a mode outside `parent` is absent when `outside` is `nothing`.
When `outside` is given it must hold the expectation value of each such term's outside factors (as one key), which is one more factor of the coefficient.
"""
function mean_field_terms(parent::Node, terms::Vector{Term}, θ::NamedTuple, child::Node,
                          others::AbstractDict{Node,Expectations}; outside::Union{Nothing,Expectations}=nothing)
    _child_index(parent, child)
    inside_parent = Set(modes(parent))
    own = Set(modes(child))
    out = Term[]
    index = Dict{_FactorKey,Int}()
    for t in _reduced_terms(parent, terms)
        outer = Factor[f for f in t.factors if f.mode ∉ inside_parent]
        isempty(outer) || outside !== nothing || continue
        mine = Factor[f for f in t.factors if f.mode in own]
        isempty(mine) && continue
        z = ComplexF64(t.coeff(θ))
        isempty(outer) || (z *= _lookup(outside, _sortkey!(Tuple{Symbol,Port}[(f.mode, f.port) for f in outer]), "the outside factors"))
        for other in parent.children
            other === child && continue
            key = _key_in(t, Set(modes(other)))
            isempty(key) && continue
            haskey(others, other) || throw(ArgumentError("`others` has no table for the child on modes $(modes(other))"))
            z *= _lookup(others[other], key, "the child on modes $(modes(other))")
        end
        k = _factor_key(Term(t.coeff, mine))
        if haskey(index, k)
            i = index[k]
            out[i] = Term(Coefficient(out[i].coeff((;)) + z), out[i].factors)
        else
            index[k] = length(out) + 1
            push!(out, Term(Coefficient(z), mine))
        end
    end
    return out
end
