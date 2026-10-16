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
            istar = isempty(sector) ? 0 : sector[argmax(abs.(col[sector]))]
            if istar == 0 || abs(col[istar]) < 1e-14
                @warn "fix_phases!: column $j has no weight on the phase mask; using its largest entry"
                istar = argmax(abs.(col))
            end
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
