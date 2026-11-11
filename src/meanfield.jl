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

"""
    MeanFieldChild

One child of a node solved with its bare terms plus the mean-field terms of its siblings.
`node` is the child and `solved` holds the kept states of its mean-field operator, with the ports and joint ports of `child_keys`.
The energies of `solved` are mean-field energies: eigenvalues of the bare operator plus the mean-field terms, not energies of the child's own operator.
`terms` are the mean-field terms that were added, with constant coefficients.
`bare_energy` is the expectation value of the child's bare operator in the state that `expectations` were taken in (or their weighted average).
`expectations` holds the value of every key of `child_keys` in that state.
"""
struct MeanFieldChild
    node::Node
    solved::MaterializedNode
    terms::Vector{Term}
    bare_energy::Float64
    expectations::Expectations
end

# The keys of `child_keys` split into the ports (one factor) and the joint ports (more) that `_solve_node` forms.
function _split_keys(keys::Vector{_FactorKey})
    ports = Tuple{Symbol,Port}[only(K) for K in keys if length(K) == 1]
    joint = _FactorKey[K for K in keys if length(K) > 1]
    return ports, joint
end

# The real part of a sum `z` that should be real, or an error when its imaginary part is not negligible: `what` is the
# sum, and `terms` the terms that are then not Hermitian.
function _real_sum(z::Number, what, terms)
    abs(imag(z)) <= 1e-8 * max(1.0, abs(real(z))) ||
        error("$what has imaginary part $(imag(z)) (real part $(real(z))): $terms are not Hermitian")
    return real(z)
end

"""
    bare_child(parent, terms, θ, child) -> MaterializedNode

The bare solve of `child`: the node as the root of its own tree, on its own terms only, with every node complex.
It forms the ports and joint ports of `child_keys(parent, terms, child)`, so that its kept states can serve as a table.
A branch child's own children are solved here once, and every `mean_field_child` of it reuses them.
"""
function bare_child(parent::Node, terms::Vector{Term}, θ::NamedTuple, child::Node)
    return _solve_subtree(child, terms, θ; keys=child_keys(parent, terms, child), complex=true)
end

"""
    mean_field_child(parent, terms, θ, child, bare, others; state=1, weights=nothing, outside=nothing) -> MeanFieldChild

One mean-field solve of `child`: its own node on its bare terms plus `mean_field_terms(parent, terms, θ, child, others; outside)`, in complex arithmetic.
The children of `child` are taken from `bare` (from `bare_child`) and are not solved again.
The expectations and the bare energy are taken in the `state`-th kept state, or averaged with `weights`, as in `expectations`.
"""
function mean_field_child(parent::Node, terms::Vector{Term}, θ::NamedTuple, child::Node, bare::MaterializedNode,
                          others::AbstractDict{Node,Expectations}; state::Int=1,
                          weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
                          outside::Union{Nothing,Expectations}=nothing)
    bare.node === child || throw(ArgumentError("`bare` is not a solution of `child`"))
    keys = child_keys(parent, terms, child)
    mf = mean_field_terms(parent, terms, θ, child, others; outside)
    ports, joint_ports = _split_keys(keys)
    solved = _solve_node(child, bare.children, vcat(_terms_at(child, terms), mf), θ; ports, joint_ports, real=false)
    table = expectations(solved, keys; state, weights)
    E = weights === nothing ? solved.energies[state] : sum(weights[i] * solved.energies[i] for i in eachindex(weights))
    shift = sum((t.coeff((;)) * table[_factor_key(t)] for t in mf); init=0.0 + 0.0im)
    s = _real_sum(shift, "the expectation of the mean-field terms of the child on modes $(modes(child))",
                  "the mean-field terms (the tables in `others` are inconsistent)")
    return MeanFieldChild(child, solved, mf, E - s, table)
end

"""
    product_energy(parent, terms, θ, bare_energies, tables; constant=0.0) -> Float64

The expectation value of the operator of `parent` in the product of its children's states, with no double counting.
It is `constant` plus the bare energy of each child plus, for each straddling term, its coefficient at `θ` times each touched child's expectation of its key.
`bare_energies` and `tables` must have an entry for every child of `parent`, and each table every key the straddling terms need.
A term with a factor outside `parent` is not part of its operator and is not counted.
"""
function product_energy(parent::Node, terms::Vector{Term}, θ::NamedTuple, bare_energies::AbstractDict{Node,Float64},
                        tables::AbstractDict{Node,Expectations}; constant::Number=0.0)
    _check_node_children(parent)
    for ch in parent.children
        haskey(bare_energies, ch) || throw(ArgumentError("`bare_energies` has no entry for the child on modes $(modes(ch))"))
        haskey(tables, ch) || throw(ArgumentError("`tables` has no table for the child on modes $(modes(ch))"))
    end
    z = 0.0 + 0.0im
    for t in straddling_terms(parent, terms)
        w = ComplexF64(t.coeff(θ))
        for ch in parent.children
            key = _key_in(t, Set(modes(ch)))
            isempty(key) || (w *= _lookup(tables[ch], key, "the child on modes $(modes(ch))"))
        end
        z += w
    end
    coupling = _real_sum(z, "the straddling terms' expectation at the node on modes $(modes(parent))",
                         "the straddling terms with these tables")
    return Float64(real(constant)) + sum(bare_energies[ch] for ch in parent.children) + coupling
end

"""
    MeanFieldSolution

A self-consistent mean-field solution of the children of a node: each child solved in the mean field of the others.
`energy` is the product energy, and `children` holds every child's `MeanFieldChild` of the last iteration.
`start` is where the run began: `nothing` (delocalized), a displacement `Dict`, or `:warm` when the run was started from another solution.
A warm-started solution keeps no reference to the solution it came from.
`converged` is `false` when the run stopped at its iteration limit, and `iterations` is the number of iterations run.
The `solved` and `terms` of each child are those of a final solve in the mean field of the final expectation values, made after the last iteration and not counted in `iterations`.
The `expectations` and `bare_energy` of each child are the values `energy` was computed from: those of the last iteration's solve, the one before the final solve (under mixing, the mixed values).
"""
struct MeanFieldSolution
    energy::Float64
    children::IdDict{Node,MeanFieldChild}
    iterations::Int
    converged::Bool
    start                    # `nothing` (delocalized), a displacement Dict, or `:warm` (started from another solution)
end

"""
    default_starts(parent; displacements=(0.0, π, -π), max_starts=81) -> Vector{Any}

The starts of the self-consistent loop: the delocalized start `nothing`, then every combination of `displacements` over the modes under `parent`.
Each combination is a `Dict{Symbol,Float64}`, in the order of `modes(parent)` with the first mode varying slowest.
More than `max_starts` combinations is an `ArgumentError`.
"""
function default_starts(parent::Node; displacements=(0.0, π, -π), max_starts::Int=81)
    ms = modes(parent)
    n = length(displacements)^length(ms)
    n > max_starts && throw(ArgumentError("default_starts would make $n displaced starts " *
                                          "($(length(displacements)) displacements over $(length(ms)) modes), " *
                                          "more than max_starts = $max_starts; pass `starts` or raise `max_starts`"))
    starts = Any[nothing]
    for c in Iterators.product(fill(displacements, length(ms))...)          # the first factor varies fastest
        push!(starts, Dict{Symbol,Float64}(m => Float64(c[end+1-i]) for (i, m) in enumerate(ms)))
    end
    return starts
end

# The starting table of every child of `parent` for the start `start`.
function _initial_tables(parent::Node, terms::Vector{Term}, start)
    if start isa AbstractDict
        under = modes(parent)
        for m in keys(start)
            m in under || throw(ArgumentError("the start displaces mode :$m, which is not under `parent` (its modes are $under)"))
        end
    end
    start isa MeanFieldSolution || return IdDict{Node,Expectations}(
        ch => (start === nothing ? delocalized_expectations(child_keys(parent, terms, ch)) :
                                   pinned_expectations(child_keys(parent, terms, ch), start)) for ch in parent.children)
    for ch in parent.children
        haskey(start.children, ch) ||
            throw(ArgumentError("the starting solution has no child on modes $(modes(ch)): it is not a solution of `parent`"))
    end
    return IdDict{Node,Expectations}(ch => copy(start.children[ch].expectations) for ch in parent.children)
end

# The largest absolute difference of any table entry of any child between two sets of tables.
_table_change(a, b) = maximum(abs(a[ch][k] - b[ch][k]) for ch in keys(a) for k in keys(a[ch]); init=0.0)

"""
    self_consistent(parent, terms, θ, bare; start=nothing, tol=1e-11, maxiter=300, mixing=0.5, constant=0.0,
                    state=1, weights=nothing, outside=nothing) -> MeanFieldSolution

The self-consistent mean-field loop over the children of `parent`, with `bare` mapping each child to its `bare_child`.
Each iteration solves every child with `mean_field_child` from the tables of the previous iteration, then computes the product energy from the new tables and bare energies.
After the energy has risen (by more than `1e-12`) twice, every later iteration replaces the new tables and bare energies by `mixing` times the new plus `1 - mixing` times the previous ones.
The run converges when `|E - E_prev| < tol * max(1, |E|)` and no table entry changed by `sqrt(tol)` or more.
A run that reaches `maxiter` iterations is returned with `converged = false`.
When the loop ends, converged or not, every child is solved once more with `mean_field_child` from the final tables, the ones the last energy was computed from; this solve is not counted in `iterations`.
The solution holds the energy of the last iteration and, for each child, a `MeanFieldChild` whose `solved` and `terms` are those of that final solve, so `solved` holds the child's states in the mean field of the final expectation values.
Its `expectations` and `bare_energy` are the ones the energy was computed from: those of the last iteration's solve (under mixing, the mixed values), so `product_energy` of them reproduces `energy`.
`start` is `nothing` (delocalized), a `Dict{Symbol,Float64}` of displacements (see `pinned_expectations`), or a `MeanFieldSolution` whose tables it continues from.
A displacement that names a mode not under `parent` is an `ArgumentError`, and a mode it leaves out has displacement zero.
The returned solution's `start` is the given start, except that a warm start is recorded as `:warm`, so that no chain of earlier solutions is kept alive.
`constant` is added to the product energy, and `state`, `weights` and `outside` are passed to `mean_field_child`.
"""
function self_consistent(parent::Node, terms::Vector{Term}, θ::NamedTuple, bare::AbstractDict{Node,<:MaterializedNode};
                         start=nothing, tol::Real=1e-11, maxiter::Int=300, mixing::Real=0.5, constant::Number=0.0,
                         state::Int=1, weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         outside::Union{Nothing,Expectations}=nothing)
    _check_node_children(parent)
    maxiter >= 1 || throw(ArgumentError("maxiter must be at least 1, got $maxiter"))
    0 < mixing <= 1 || throw(ArgumentError("mixing must lie in (0, 1], got $mixing"))
    for ch in parent.children
        haskey(bare, ch) || throw(ArgumentError("`bare` has no solve of the child on modes $(modes(ch))"))
    end
    tables = _initial_tables(parent, terms, start)
    bare_energies = IdDict{Node,Float64}()
    E_prev, rises, mixing_on = nothing, 0, false
    for it in 1:maxiter
        solved = IdDict{Node,MeanFieldChild}(ch => mean_field_child(parent, terms, θ, ch, bare[ch], tables; state, weights, outside)
                                             for ch in parent.children)
        new_tables = IdDict{Node,Expectations}(ch => c.expectations for (ch, c) in solved)
        new_bare = IdDict{Node,Float64}(ch => c.bare_energy for (ch, c) in solved)
        if mixing_on
            new_tables = IdDict{Node,Expectations}(ch => Expectations(k => mixing * v + (1 - mixing) * tables[ch][k] for (k, v) in t)
                                                   for (ch, t) in new_tables)
            new_bare = IdDict{Node,Float64}(ch => mixing * e + (1 - mixing) * bare_energies[ch] for (ch, e) in new_bare)
        end
        E = product_energy(parent, terms, θ, new_bare, new_tables; constant)
        E_prev !== nothing && E > E_prev + 1e-12 && (rises += 1)
        converged = E_prev !== nothing && abs(E - E_prev) < tol * max(1, abs(E)) && _table_change(new_tables, tables) < sqrt(tol)
        tables, bare_energies, E_prev = new_tables, new_bare, E
        rises >= 2 && (mixing_on = true)                                     # from the next iteration on
        if converged || it == maxiter
            # one more solve of every child, in the mean field of the final tables (not counted as an iteration)
            final = IdDict{Node,MeanFieldChild}(ch => mean_field_child(parent, terms, θ, ch, bare[ch], tables; state, weights, outside)
                                                for ch in parent.children)
            children = IdDict{Node,MeanFieldChild}(ch => MeanFieldChild(ch, c.solved, c.terms, bare_energies[ch], tables[ch])
                                                   for (ch, c) in final)
            return MeanFieldSolution(E, children, it, converged, start isa MeanFieldSolution ? :warm : start)
        end
    end
end

"""
    mean_field_solutions(parent, terms, θ; starts=default_starts(parent), distinct=1e-3, kwargs...) -> Vector{MeanFieldSolution}
    mean_field_solutions(parent, spec::Spec, θ; kwargs...) -> Vector{MeanFieldSolution}

Run `self_consistent` from every start in `starts`, with the bare solve of each child formed once and shared, and return the distinct solutions, lowest energy first.
A solution is kept when, against every solution already kept, some table entry of some child differs by more than `distinct`.
Ties in energy keep the order of `starts`, and runs that did not converge take part like any other.
The remaining keywords are passed to `self_consistent`.
The second form takes the terms from `circuit_terms(spec)` and `constant` from `constant_term(spec)` at `θ`.
"""
function mean_field_solutions(parent::Node, terms::Vector{Term}, θ::NamedTuple; starts=default_starts(parent),
                              distinct::Real=1e-3, kwargs...)
    isempty(starts) && throw(ArgumentError("`starts` is empty: give at least one start"))
    _check_node_children(parent)
    bare = IdDict{Node,MaterializedNode}(ch => bare_child(parent, terms, θ, ch) for ch in parent.children)
    runs = MeanFieldSolution[self_consistent(parent, terms, θ, bare; start, kwargs...) for start in starts]
    sort!(runs; by=s -> s.energy, alg=MergeSort)                             # stable: ties keep the order of `starts`
    tables(s) = IdDict{Node,Expectations}(ch => c.expectations for (ch, c) in s.children)
    kept = MeanFieldSolution[]
    for s in runs
        all(k -> _table_change(tables(s), tables(k)) > distinct, kept) && push!(kept, s)
    end
    return kept
end

function mean_field_solutions(parent::Node, spec::Spec, θ::NamedTuple; kwargs...)
    c = constant_term(spec)(θ)
    abs(imag(c)) <= 1e-12 || throw(ArgumentError("the constant of the spec at θ has imaginary part $(imag(c))"))
    return mean_field_solutions(parent, circuit_terms(spec), θ; constant=real(c), kwargs...)
end
