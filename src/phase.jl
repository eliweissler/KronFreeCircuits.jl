"""
    fix_phases!(V, convention, mask, reference)

Fix the global phase of each column of `V` in place and return `V`.
`:positive_sector` makes the largest-modulus entry among `mask` real and positive
(falling back to the global largest entry, with a warning, if the column has no weight on `mask`).
`:follow` makes the overlap with the matching `reference` column real and positive.
"""
function fix_phases!(V::AbstractMatrix, convention::Symbol, mask::AbstractVector{Bool}, reference)
    if convention === :positive_sector
        length(mask) == size(V, 1) ||
            throw(ArgumentError("mask has length $(length(mask)) but vectors have $(size(V, 1)) rows"))
        sector = findall(mask)
        for j in axes(V, 2)
            col = view(V, :, j)
            istar, fellback = _sector_peak(col, sector)
            fellback && @warn "fix_phases!: column $j has no weight on the phase mask; using its largest entry"
            col .*= exp(-im * angle(col[istar]))
        end
    elseif convention === :follow
        reference === nothing && throw(ArgumentError("phase convention :follow needs a reference"))
        for j in axes(V, 2)
            col = view(V, :, j)
            col .*= exp(-im * angle(dot(view(reference, :, j), col)))
        end
    else
        throw(ArgumentError("unknown phase convention :$convention"))
    end
    return V
end

# The index of the largest-modulus entry of `col` among `sector`, and whether it fell back to the
# global largest entry because `col` has no weight on `sector`.
function _sector_peak(col::AbstractVector, sector::AbstractVector{Int})
    istar = isempty(sector) ? 0 : sector[argmax(abs.(col[sector]))]
    (istar == 0 || abs(col[istar]) < 1e-14) && return argmax(abs.(col)), true
    return istar, false
end

# Σ_{i₁…iₙ} X[i₁, …, iₙ, :] rows[1][i₁] ⋯ rows[n][iₙ] (no conjugation), axis 1 fastest: one
# contraction per axis, never forming the product of the rows.
function _contract_rows(X::AbstractArray, rows)
    Y = vec(X)
    for r in rows
        Y = transpose(reshape(Y, length(r), :)) * r
    end
    return Y
end

"""
    branch_peak_phase(C, amps) -> (comp, α)

The branch phase rule of `:positive_sector`.
`C` is a kept vector of a branch node reshaped to the node's `dims`, and `amps[a]` is child `a`'s
`peak_amplitudes` matrix (`amps[a][p, i]` is the full-space amplitude of the child's kept state `i`
at the peak point of its kept state `p`).
Returns `comp`, the linear index of the dominant component `(a*, b*, …) = argmax |C|`, and `α`, the
exact full-space amplitude of the state at the joint peak of that component,
`α = Σ C[a, b, …] amps[1][a*, a] amps[2][b*, b] ⋯`.
A sign or phase change of a child state rescales its coefficients and its amplitudes inversely,
so `α` does not depend on the children's phase conventions.
"""
function branch_peak_phase(C::AbstractArray{<:Number}, amps::AbstractVector{<:AbstractMatrix})
    length(amps) == ndims(C) ||
        throw(ArgumentError("branch_peak_phase: $(length(amps)) amplitude matrices for a $(ndims(C))-axis tensor"))
    comp = argmax(abs.(vec(C)))
    ci = CartesianIndices(C)[comp]
    α = _contract_rows(C, [view(amps[a], ci[a], :) for a in 1:ndims(C)])[1]
    return comp, ComplexF64(α)
end
