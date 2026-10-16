fixture(name::AbstractString) = joinpath(@__DIR__, "fixtures", name)

"""
    kron_reference(dims, terms)

Dense reference for `sum_t c_t A_{t,N} ⊗ ⋯ ⊗ A_{t,1}` with identities on axes a term
does not touch.  Axis 1 is fastest (column-major).
`terms` is a vector of `(c, [(axis, matrix), ...])`.
"""
function kron_reference(dims::Vector{Int},
                        terms::Vector{<:Tuple{ComplexF64,Vector{<:Tuple{Int,<:AbstractMatrix}}}})
    D = prod(dims)
    H = zeros(ComplexF64, D, D)
    for (c, factors) in terms
        ops = Dict{Int,AbstractMatrix}(a => M for (a, M) in factors)
        T = ones(ComplexF64, 1, 1)
        for a in 1:length(dims)          # builds kron(A_N, ..., A_1)
            M = get(ops, a, nothing)
            T = kron(M === nothing ? Matrix{ComplexF64}(I, dims[a], dims[a]) : M, T)
        end
        H .+= c .* T
    end
    return H
end
