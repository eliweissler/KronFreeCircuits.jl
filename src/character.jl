# How complex conjugation acts on each port's operator (spec Section 20f):
# `conj(O_p) = σ_p O_{π(p)}` with a sign σ_p and an involution π on ports.

"""
    PortCharacter(kind, partner)

How conjugation acts on a port operator: `kind` is `:real` (`conj(O) = O`), `:imaginary` (`conj(O) = -O`),
`:paired` (`conj(O_p) = O_partner`, `partner ≠ p`) or `:general` (no relation).
`partner` is the port itself except for `:paired`.
"""
struct PortCharacter
    kind::Symbol
    partner::Port
end

"The conjugation sign of a character: `-1` for `:imaginary`, else `+1`."
σ(c::PortCharacter)::Int = c.kind === :imaginary ? -1 : 1

_real_char(p::Port) = PortCharacter(:real, p)
_imag_char(p::Port) = PortCharacter(:imaginary, p)
# a nonzero exponent pairs with its negative; `k = 0` is the identity, which is real
_paired_char(p::Port) = iszero(p.k) ? _real_char(p) : PortCharacter(:paired, Port(p.kind, -p.k))

"""
    character(b::Basis, p::Port) -> PortCharacter
    character(g::GivenSystem, p::Port) -> PortCharacter

The conjugation character of port `p`, from the table of the architecture spec (Section 20f).
A port the basis does not serve throws the same `ArgumentError` as `localop`.
A `GivenSystem`'s character is read exactly from its stored operator.
"""
function character(b::ChargeBasis, p::Port)
    p.kind in (:n, :n2, :expiphi) && (localop(b, p); return _real_char(p))
    p.kind === :exp2pin && return _paired_char(p)
    throw(_unavailable(b, p))
end

function character(b::PhaseGrid, p::Port)
    p.kind in (:n2, :phi, :phi2) && return _real_char(p)
    p.kind === :n && return _imag_char(p)
    p.kind === :exp2pin && (localop(b, p); return _real_char(p))
    p.kind === :expiphi && return _paired_char(p)
    throw(_unavailable(b, p))
end

function character(b::PeriodicGrid, p::Port)
    localop(b, p)                                           # validates the port exactly as `localop` does
    p.kind in (:d2, :d2wrap) && return _real_char(p)
    p.kind in (:d1, :d1wrap) && return _imag_char(p)
    return _paired_char(p)                                  # :expiphi, the only other served kind
end

function character(b::FockBasis, p::Port)
    p.kind in (:phi, :phi2, :n2, :exp2pin) && return _real_char(p)
    p.kind === :n && return _imag_char(p)
    p.kind === :expiphi && return _paired_char(p)
    throw(_unavailable(b, p))
end

function character(g::GivenSystem, p::Port)
    M = get(g.operators, p.kind, nothing)
    M === nothing && throw(ArgumentError("GivenSystem :$(g.mode): port $(_portname(p)) is not available"))
    all(iszero, imag(M)) && return _real_char(p)
    all(iszero, real(M)) && return _imag_char(p)
    return PortCharacter(:general, p)
end

# ---- Node classification: the node test of spec Section 20f ----

const _TEST_SEED = 0x6b66      # the fixed seed of the three test points
const _TEST_RTOL = 1e-12       # agreement relative to the larger modulus

# Every mode's leaf, its `Basis` or `GivenSystem`, wherever it sits in the tree.
# A real node child's ports carry the characters of the same ports in its modes' leaves, so one table serves every node.
function _mode_leaves(root::Node)
    leaves = Dict{Symbol,Any}()
    for node in _all_nodes(root), ch in node.children
        ch isa Pair && (leaves[ch.first] = ch.second)
        ch isa GivenSystem && (leaves[ch.mode] = ch)
    end
    return leaves
end

# A factor's character from its mode's leaf.
# A port the leaf does not serve (or a given port with a nonzero exponent, which `materialize` rejects) is `:general`:
# the node stays complex and `materialize` raises its own error when it builds the operator.
function _factor_character(leaves::Dict{Symbol,Any}, f::Factor)
    leaf = leaves[f.mode]
    leaf isa GivenSystem && !iszero(f.port.k) && return PortCharacter(:general, f.port)
    try
        return character(leaf, f.port)
    catch e
        e isa ArgumentError || rethrow()
        return PortCharacter(:general, f.port)
    end
end

const _FactorKey = Vector{Tuple{Symbol,Port}}

_sortkey!(K::_FactorKey) = sort!(K; by=((m, p),) -> (m, p.kind, p.k))
_factor_key(t::Term) = _sortkey!(Tuple{Symbol,Port}[(f.mode, f.port) for f in t.factors])

_key_string(K::_FactorKey) = isempty(K) ? "no factors (a constant)" : join(("$(_portname(p)) on :$m" for (m, p) in K), " ⊗ ")
_term_string(t::Term) = "$(_key_string(_factor_key(t))) with coefficient `$(t.coeff.ex)`"

# The three test points: every parameter any coefficient uses, each drawn uniformly in [0.5, 2] from the fixed seed in
# sorted name order, so the values depend only on the set of parameters.
# A point is laid out like `θ` (its names in order, then any parameter `θ` lacks), with `1.0` for entries no coefficient
# reads: for a `Float64` θ its type is θ's, so the coefficients compile once for both.
# The classification itself does not specialize on θ (`@nospecialize`): it runs once per materialization.
function _test_points(coeffs, @nospecialize(θ::NamedTuple))
    deps = sort!(unique!(reduce(vcat, (c.deps for c in coeffs); init=Symbol[])))
    names = (keys(θ)..., (d for d in deps if !haskey(θ, d))...)
    rng = Random.MersenneTwister(_TEST_SEED)
    return map(1:3) do _
        draw = Dict(zip(deps, 0.5 .+ 1.5 .* rand(rng, length(deps))))
        NamedTuple{names}(Tuple(get(draw, n, 1.0) for n in names))
    end
end

# A coefficient at the test points, or `nothing` when it throws at one of them.
function _at_points(c::Coefficient, @nospecialize(points::Vector))
    try
        return ComplexF64[c(θr) for θr in points]
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

_agree(a::Number, b::Number) = abs(a - b) <= _TEST_RTOL * max(abs(a), abs(b))

# Why the terms of one node (and the root's constant) fail the node test, or `nothing` when they pass.
function _term_reason(ts::Vector{Term}, constant::Union{Nothing,Coefficient}, leaves, @nospecialize(points::Vector))
    entries = constant === nothing ? ts : vcat(ts, Term(constant, Factor[]))
    chars = Dict{Tuple{Symbol,Port},PortCharacter}()
    for t in entries, f in t.factors
        c = get!(() -> _factor_character(leaves, f), chars, (f.mode, f.port))
        c.kind === :general &&
            return "the term $(_term_string(t)) has the factor $(_portname(f.port)) on :$(f.mode), whose operator has no conjugation relation"
    end
    # C_K at the test points, summed over the terms of each factor key K (keys in order of first appearance).
    sums = Dict{_FactorKey,Vector{ComplexF64}}()
    first_term = Dict{_FactorKey,Term}()
    order = _FactorKey[]
    for t in entries
        K = _factor_key(t)
        v = _at_points(t.coeff, points)
        v === nothing && return "the coefficient of the term $(_term_string(t)) cannot be evaluated at a test point"
        if haskey(sums, K)
            sums[K] .+= v
        else
            sums[K] = v
            first_term[K] = t
            push!(order, K)
        end
    end
    # C_{π(K)} = σ_K conj(C_K): a key whose partner key has no terms compares against zero.
    for K in order
        σK = prod((σ(chars[mp]) for mp in K); init=1)
        πK = _sortkey!(Tuple{Symbol,Port}[(m, chars[(m, p)].partner) for (m, p) in K])
        image = get(sums, πK, nothing)
        for r in eachindex(points)
            a = image === nothing ? zero(ComplexF64) : image[r]
            _agree(a, σK * conj(sums[K][r])) && continue
            want = πK == K ? "its factors are their own conjugate image, so their summed coefficient must be " *
                             (σK == 1 ? "real" : "purely imaginary") :
                   "the terms on $(_key_string(πK)) must sum to $(σK == 1 ? "" : "minus ")the complex conjugate of its coefficient" *
                   (count(t -> _factor_key(t) == K, entries) > 1 ? " (summed with the other terms on the same factors)" : "")
            return "the term $(_term_string(first_term[K])) has no conjugate image: $want"
        end
    end
    return nothing
end

"""
    classify(root, routed, constant, θ) -> IdDict{Node,Bool}

Which nodes of the tree run in real arithmetic (`true`), decided bottom-up from the term list and the tree, never from
the values in `θ` (the node test of the architecture spec, Section 20f).
A node is real when its `arithmetic` is not `:complex`, every node below it is real, every factor of its `routed` terms
has a real, imaginary or paired character in its mode's leaf, and for every factor key `K` the summed coefficient
satisfies `C_{π(K)} = σ_K conj(C_K)` at three fixed pseudo-random parameter points; `constant` joins the root's empty key.
When any value of `θ` is not of a `Real` type, every node is complex.
A node with `arithmetic = :real` that tests complex throws an `ArgumentError` naming its modes and the reason (the first
term whose conjugate image is missing); on `:auto` the reason is logged at debug level.
"""
function classify(root::Node, routed::IdDict, constant::Coefficient, @nospecialize(θ::NamedTuple))
    leaves = _mode_leaves(root)
    nonreal = [k for (k, v) in pairs(θ) if !(v isa Real)]
    points = isempty(nonreal) ? _test_points(Iterators.flatten(((constant,), (t.coeff for ts in values(routed) for t in ts))), θ) : nothing
    out = IdDict{Node,Bool}()
    function visit(node::Node)
        foreach(visit, _subnodes(node))
        node.arithmetic === :complex && return (out[node] = false)
        complex_child = findfirst(ch -> !out[ch], collect(_subnodes(node)))
        reason = !isempty(nonreal) ? "the parameters $(join((":$k" for k in nonreal), ", ")) are not of a Real type" :
                 complex_child !== nothing ? "its child node on modes $(modes(collect(_subnodes(node))[complex_child])) is complex" :
                 _term_reason(get(routed, node, Term[]), node === root ? constant : nothing, leaves, points)
        reason === nothing && return (out[node] = true)
        node.arithmetic === :real &&
            throw(ArgumentError("node on modes $(modes(node)) has arithmetic = :real, but its Hamiltonian is not real by structure: $reason"))
        @debug "node on modes $(modes(node)) runs in complex arithmetic: $reason"
        return (out[node] = false)
    end
    visit(root)
    return out
end
