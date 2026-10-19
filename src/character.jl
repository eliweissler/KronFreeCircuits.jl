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
