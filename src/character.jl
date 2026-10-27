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

# The conjugation character of the product `⊗ O` of a sorted factor key's operators, from its factors' characters:
# `:general` if any factor is; else `:paired` if any factor is, its partner the sorted key of the factors' partners;
# else `:real` or `:imaginary` by the product of the factors' signs `σ`, its partner the key itself.
# For a singleton key it agrees with `_factor_character` (`kind`, and `[partner]`).
function _key_character(leaves::Dict{Symbol,Any}, K::_FactorKey)
    chars = PortCharacter[_factor_character(leaves, Factor(m, p)) for (m, p) in K]
    any(c -> c.kind === :general, chars) && return (kind=:general, partner=K)
    any(c -> c.kind === :paired, chars) &&
        return (kind=:paired, partner=_sortkey!(Tuple{Symbol,Port}[(m, c.partner) for ((m, _), c) in zip(K, chars)]))
    return (kind=prod(σ, chars; init=1) == 1 ? :real : :imaginary, partner=K)
end

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

# The three-point test proves an identity for every θ only for coefficients real-analytic in θ, so a syntactic gate
# rejects the allowed calls that are not: `abs` and `sign` anywhere, and `sqrt`, `log` or `^` (with an exponent that is
# not an integer literal) where an argument can be complex-valued (branch cuts).
# On real-valued arguments `sqrt`, `log` and `^` are analytic where defined and throw a `DomainError` elsewhere.

# The calls the gate classifies: real-valued whatever their argument (`_REAL_VALUED_CALLS`), complex-valued when an argument
# is (`_ANALYTIC_CALLS`), and the calls it always treats as complex-valued (`_COMPLEX_CALLS`).
# Every allowed call (`_ALLOWED_CALLS`) must be listed, which the tests check.
const _REAL_VALUED_CALLS = (:real, :imag, :abs, :sign)
const _COMPLEX_CALLS = (:cis, :cispi)
const _ANALYTIC_CALLS = (:+, :-, :*, :/, ://, :^, :sqrt, :exp, :log, :cos, :sin, :tan, :cosh, :sinh, :tanh, :conj)

# Whether `ex` can be complex-valued with real parameters; anything unrecognized can (conservative).
function _maybe_complex(ex)
    ex isa Real && return false
    ex isa Symbol && return ex === :im          # parameters are real (a non-Real θ is rejected); pi, π, ℯ
    (ex isa Expr && ex.head === :call && ex.args[1] isa Symbol) || return true
    f, args = ex.args[1], @view ex.args[2:end]
    f in _REAL_VALUED_CALLS && return false
    f in _COMPLEX_CALLS && return true
    f in _ANALYTIC_CALLS &&
        return any(_maybe_complex, args)
    return true
end

# The first non-analytic call in `ex`, described, or `nothing`.
function _nonanalytic(ex)
    ex isa Expr || return nothing
    if ex.head === :call && ex.args[1] isa Symbol
        f, args = ex.args[1], @view ex.args[2:end]
        f in (:abs, :sign) && return "calls `$f`, which is not real-analytic"
        f in (:sqrt, :log) && any(_maybe_complex, args) && return "calls `$f` on a possibly complex argument (a branch cut)"
        f === :^ && !(length(args) == 2 && args[2] isa Integer) && any(_maybe_complex, args) &&
            return "raises a possibly complex value to a power that is not an integer literal (a branch cut)"
    end
    for a in ex.args
        r = _nonanalytic(a)
        r === nothing || return r
    end
    return nothing
end

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
        bad = _nonanalytic(t.coeff.ex)
        bad === nothing || return "the coefficient of the term $(_term_string(t)) $bad, so the test points cannot establish its conjugate image"
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

# All system parameters are real (spec Section 20f, "Real inputs"): a complex amplitude is passed as two real parameters.
function _check_real_parameters(@nospecialize(θ::NamedTuple))
    for (k, v) in pairs(θ)
        v isa Real || throw(ArgumentError("parameter :$k = $(repr(v)) is of type $(typeof(v)); all system parameters must be " *
                                          "of a Real type (pass a complex amplitude as two real parameters)"))
    end
    return nothing
end

"""
    classify(root, routed, constant, θ) -> IdDict{Node,Bool}

Which nodes of the tree run in real arithmetic (`true`), decided bottom-up from the term list and the tree, never from
the values in `θ` (the node test of the architecture spec, Section 20f).
A node is real when its `arithmetic` is not `:complex`, every node below it is real, every factor of its `routed` terms
has a real, imaginary or paired character in its mode's leaf, every coefficient passes a syntactic real-analyticity gate
(no `abs` or `sign`; no `sqrt`, `log` or non-integer-literal `^` of a possibly complex argument), and for every factor key
`K` the summed coefficient satisfies `C_{π(K)} = σ_K conj(C_K)` at three fixed pseudo-random parameter points;
`constant` joins the root's empty key.
A value of `θ` that is not of a `Real` type throws an `ArgumentError` naming the parameter.
A node with `arithmetic = :real` that tests complex throws an `ArgumentError` naming its modes and the reason (the first
term whose conjugate image is missing); on `:auto` the reason is logged at debug level.
"""
function classify(root::Node, routed::IdDict, constant::Coefficient, @nospecialize(θ::NamedTuple))
    _check_real_parameters(θ)
    leaves = _mode_leaves(root)
    points = _test_points(Iterators.flatten(((constant,), (t.coeff for ts in values(routed) for t in ts))), θ)
    out = IdDict{Node,Bool}()
    function visit(node::Node)
        foreach(visit, _subnodes(node))
        node.arithmetic === :complex && return (out[node] = false)
        subs = collect(_subnodes(node))
        complex_child = findfirst(ch -> !out[ch], subs)
        reason = complex_child !== nothing ? "its child node on modes $(modes(subs[complex_child])) is complex" :
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

# ---- The real operator of a real node (spec Section 20f, "The real operator") ----

# The canonical port of a factor's conjugation orbit and the sign `s` with `imagpart(O_p) = s · imagpart(O_canonical)`.
# A paired port's orbit is `{Port(kind, k), Port(kind, -k)}` and its canonical port the one with `k > 0`, since
# `realpart(O_{π(p)}) = realpart(O_p)` and `imagpart(O_{π(p)}) = −imagpart(O_p)`; any other port is its own canonical port.
function _canonical_port(c::PortCharacter)
    c.kind === :paired || return c.partner, 1
    return c.partner.k < 0 ? (Port(c.partner.kind, -c.partner.k), 1) : (c.partner, -1)
end

# `Re(c · i^m)`, exactly.
function _re_ipow(c::Number, m::Int)
    r = mod(m, 4)
    return Float64(r == 0 ? real(c) : r == 1 ? -imag(c) : r == 2 ? -real(c) : imag(c))
end

# The expansion of `Re(c ⊗_a O_a)` with `O_a = R_a + i I_a`: one real term `Re(c · i^{|S|}) ⊗_{a∈S} I_a ⊗_{a∉S} R_a` per
# subset `S` of the factors whose character is `:imaginary` (always in `S`) or `:paired`, a zero coefficient dropped.
# Each factor is `(axis, O_a, kind, s)`, with `kind` its character's kind and `s` the sign of its imaginary component
# relative to its canonical factor's.
# Each term is `(coeff, sign, S, factors)`: `factors` are this term's own components, `coeff` its coefficient on them,
# and `S[a]` whether factor `a` takes its imaginary component.
# On the canonical components the coefficient is `coeff * sign`, with `sign` the product of the factors' signs `s` over
# `S`, so expansions of different terms on the same modes, canonical ports and components add by their coefficients:
# a conjugate pair's real terms coincide.
# A caller that needs only `Re(c ⊗_a O_a)` on the factors' own components passes `s = 1` and ignores `sign`.
function _real_expansion(c::Number, factors::Vector{Tuple{Int,LocalOp,Symbol,Int}})
    comps = map(factors) do (a, op, kind, s)
        kind in (:real, :imaginary, :paired) ||
            error("internal error: the factor on axis $a of a real node has a :$kind character")
        R = kind === :imaginary ? nothing : realpart(op)
        I = kind === :real ? nothing : imagpart(op)
        choices = kind === :real ? (false,) : kind === :imaginary ? (true,) : (false, true)
        (a, s, R, I, choices)
    end
    out = NamedTuple{(:coeff, :sign, :S, :factors),Tuple{Float64,Int,Vector{Bool},Vector{Tuple{Int,LocalOp}}}}[]
    for S in Iterators.product((x[5] for x in comps)...)
        coeff = _re_ipow(c, count(S))
        iszero(coeff) && continue
        sign = prod((x[2] for (x, inS) in zip(comps, S) if inS); init=1)
        ops = Tuple{Int,LocalOp}[(x[1], inS ? x[4] : x[3]) for (x, inS) in zip(comps, S)]
        push!(out, (coeff=coeff, sign=sign, S=collect(Bool, S), factors=ops))
    end
    return out
end

# On factors `(axis, O_a, character)`: the signs `s` are those of the canonical ports (`_canonical_port`).
function _real_expansion(c::Number, factors::Vector{Tuple{Int,LocalOp,PortCharacter}})
    for (a, _, ch) in factors
        ch.kind === :general &&
            error("internal error: the factor $(_portname(ch.partner)) on axis $a of a real node has a :general character")
    end
    return _real_expansion(c, Tuple{Int,LocalOp,Symbol,Int}[(a, op, ch.kind, last(_canonical_port(ch))) for (a, op, ch) in factors])
end

"""
    _real_axisterms(c, factors) -> Vector{AxisTerm{Float64}}

The real terms of `Re(c ⊗_a O_a)` for `factors = [(axis, O_a, character), ...]`: `Re(c · i^{|S|}) ⊗_{a∈S} I_a ⊗_{a∉S} R_a`
over the subsets `S` of the `:imaginary` (always in `S`) and `:paired` factors, with `R_a`, `I_a` the real and
imaginary components (`realpart`, `imagpart`) of `O_a`; terms with a zero coefficient are dropped.
No terms are merged here: a node merges the real terms of all its routed terms by `(axis, mode, canonical port,
component)` keys, the mode included because a node child's axis carries several modes (`_real_node_terms`).
A `:general` factor is an internal error: classification admits none on a real node.
"""
_real_axisterms(c::Number, factors::Vector{Tuple{Int,LocalOp,PortCharacter}}) =
    AxisTerm{Float64}[AxisTerm(r.coeff, r.factors) for r in _real_expansion(c, factors)]
