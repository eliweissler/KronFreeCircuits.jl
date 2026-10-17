"""
    GivenSystem(mode; energies, operators)

A pre-diagonalized subsystem, a tree child beside `Node` and `mode => basis` (a resonator or a qubit
from another tool, a subsystem exported from scqubits, a fixed model of a neighbor).
`energies` are its eigenvalues in ascending order; `operators` maps a port kind (any `Symbol`) to
its `k × k` matrix in that eigenbasis, `k = length(energies)`.

It materializes to a node with identity eigenvectors and all states kept: its energies are the
diagonal term on its parent's axis and its operators are the ports `(mode, Port(kind, 0))`.
Routing never descends into it, so a term whose factors all lie on its mode (a drive or Stark shift)
is applied at its parent through these ports. Coupling terms are ordinary `Term`s naming its mode
and a port kind it provides; pass them to the spec-form `materialize` as `extra_terms`.
"""
struct GivenSystem
    mode::Symbol
    energies::Vector{Float64}
    operators::Dict{Symbol,Matrix{ComplexF64}}

    function GivenSystem(mode::Symbol; energies::AbstractVector{<:Real}, operators::AbstractDict)
        k = length(energies)
        k >= 1 || throw(ArgumentError("GivenSystem :$mode needs at least one energy"))
        issorted(energies) || throw(ArgumentError("GivenSystem :$mode: energies must be in ascending order"))
        ops = Dict{Symbol,Matrix{ComplexF64}}()
        for (kind, M) in operators
            size(M) == (k, k) ||
                throw(ArgumentError("GivenSystem :$mode: operator :$kind has size $(size(M)), expected ($k, $k)"))
            ops[Symbol(kind)] = Matrix{ComplexF64}(M)
        end
        return new(mode, Vector{Float64}(energies), ops)
    end
end

dimension(g::GivenSystem) = length(g.energies)

# A given system is a tree child with one mode (tree.jl's `_child_modes`).
_child_modes(ch::GivenSystem) = Symbol[ch.mode]
