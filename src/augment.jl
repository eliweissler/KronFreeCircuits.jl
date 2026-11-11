# Basis augmentation: several sources of vectors on a child's own space are merged into one orthonormal basis.

"""
    BasisSource(label, vectors, energies; cap = nothing, window = nothing)

One source of vectors on a child's own product space, for example the lowest mean-field states of the child in one self-consistent solution.
The columns of `vectors` are unit vectors, lowest state first, and `energies` holds one energy per column.
At most `cap` columns are offered, and only the columns whose energy is within `window` of the first energy are offered.
The arrays are copied and converted to `ComplexF64` and `Float64`.
"""
struct BasisSource
    label::String
    vectors::Matrix{ComplexF64}
    energies::Vector{Float64}
    cap::Union{Nothing,Int}
    window::Union{Nothing,Float64}
end

function BasisSource(label::AbstractString, vectors::AbstractMatrix, energies::AbstractVector{<:Real};
                     cap::Union{Nothing,Integer} = nothing, window::Union{Nothing,Real} = nothing)
    size(vectors, 2) == length(energies) ||
        throw(ArgumentError("source \"$label\" has $(size(vectors, 2)) columns but $(length(energies)) energies"))
    cap === nothing || cap >= 0 || throw(ArgumentError("source \"$label\": `cap` must be at least 0, got $cap"))
    window === nothing || window >= 0 || throw(ArgumentError("source \"$label\": `window` must be at least 0, got $window"))
    return BasisSource(String(label), Matrix{ComplexF64}(vectors), Vector{Float64}(energies),
                       cap === nothing ? nothing : Int(cap), window === nothing ? nothing : Float64(window))
end

"""
    UnionReport(labels, taken, dropped)

What `union_basis` did with each source, in the order the sources were given.
`taken[i]` vectors of source `i` were accepted, and `dropped[i]` were offered and rejected as dependent on the vectors already accepted.
A column that was never reached because the merge stopped at `m` is neither taken nor dropped.
"""
struct UnionReport
    labels::Vector{String}
    taken::Vector{Int}
    dropped::Vector{Int}
end

function Base.show(io::IO, ::MIME"text/plain", r::UnionReport)
    print(io, "UnionReport")
    for i in eachindex(r.labels)
        print(io, "\n  ", r.labels[i], ": taken ", r.taken[i], ", dropped ", r.dropped[i])
    end
end
Base.show(io::IO, r::UnionReport) = show(io, MIME"text/plain"(), r)

# The number of columns of `s` that are offered: limited by `cap`, and by `window` (the scan stops at the first column outside).
function _offered(s::BasisSource)
    n = size(s.vectors, 2)
    s.cap === nothing || (n = min(n, s.cap))
    if s.window !== nothing && n > 0
        k = 1
        while k < n && s.energies[k + 1] - s.energies[1] <= s.window
            k += 1
        end
        n = k
    end
    return n
end

"""
    union_basis(sources; m, drop = 1e-6, allow_fewer = false) -> (U, report)

Merge `sources`, a vector of `BasisSource`, into one orthonormal basis `U` of `m` columns.
The merge is level by level: the first offered column of each source in the order given, then the second of each, and so on, until `m` vectors are accepted.
A candidate is orthogonalized against the accepted vectors by classical Gram–Schmidt applied twice, and accepted when the norm of the remainder is above `drop`; it is then normalized.
A source that is capped or windowed stops offering at its limit, so an uncapped source fills the remainder.
`report` is a `UnionReport` counting, for each source, the vectors accepted and the vectors offered and rejected.
Ending with fewer than `m` vectors is an `ArgumentError` that prints the report, unless `allow_fewer = true`, which returns the vectors found.
"""
function union_basis(sources::Vector{BasisSource}; m::Integer, drop::Real = 1e-6, allow_fewer::Bool = false)
    isempty(sources) && throw(ArgumentError("`union_basis` needs at least one source"))
    m >= 1 || throw(ArgumentError("`m` must be at least 1, got $m"))
    drop >= 0 || throw(ArgumentError("`drop` must be at least 0, got $drop"))
    labels = [s.label for s in sources]
    nrow = size(sources[1].vectors, 1)
    for s in sources
        size(s.vectors, 1) == nrow ||
            throw(ArgumentError("source \"$(s.label)\" has vectors of length $(size(s.vectors, 1)), " *
                                "but source \"$(sources[1].label)\" has vectors of length $nrow"))
        size(s.vectors, 2) == length(s.energies) ||
            throw(ArgumentError("source \"$(s.label)\" has $(size(s.vectors, 2)) columns but $(length(s.energies)) energies"))
        for j in axes(s.vectors, 2)
            abs(norm(view(s.vectors, :, j)) - 1) <= 1e-8 ||
                throw(ArgumentError("column $j of source \"$(s.label)\" is not a unit vector (norm $(norm(view(s.vectors, :, j))))"))
        end
    end
    offered = [_offered(s) for s in sources]
    taken = zeros(Int, length(sources))
    dropped = zeros(Int, length(sources))
    U = Matrix{ComplexF64}(undef, nrow, 0)
    count = 0
    level = 1
    while count < m && any(offered .>= level)
        for (i, s) in enumerate(sources)
            level <= offered[i] || continue
            v = Vector{ComplexF64}(s.vectors[:, level])
            for _ in 1:2
                v -= U * (U' * v)
            end
            nv = norm(v)
            if nv > drop
                U = hcat(U, v / nv)
                taken[i] += 1
                count += 1
                count == m && break
            else
                dropped[i] += 1
            end
        end
        level += 1
    end
    report = UnionReport(labels, taken, dropped)
    if count < m && !allow_fewer
        throw(ArgumentError("`union_basis` found only $count of the $m requested independent vectors; the rest of the offered vectors were dropped as dependent or not offered:\n" *
                            sprint(show, report) * "\npass `allow_fewer = true` to return the $count vectors found"))
    end
    return U, report
end
