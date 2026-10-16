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

# Three coupled truncated oscillators with cos-type couplings (shared by solver tests).
function _three_osc_pieces(d::Int)
    a = diagm(1 => sqrt.(1.0:d-1))
    X = a + a'
    cos_op = Matrix{ComplexF64}((exp(im * 0.5 * X) + exp(-im * 0.5 * X)) / 2)
    nvec = collect(0.0:d-1)
    ω = (1.0, 1.1, 1.2)
    return cos_op, nvec, ω
end

function three_oscillators(d::Int)
    cos_op, nvec, ω = _three_osc_pieces(d)
    terms = KronFreeCircuits.AxisTerm[]
    for i in 1:3
        push!(terms, KronFreeCircuits.AxisTerm(ω[i], [(i, KronFreeCircuits.DiagonalOp(nvec))]))
    end
    for (i, j) in ((1, 2), (2, 3), (1, 3))
        push!(terms, KronFreeCircuits.AxisTerm(0.2, [(i, KronFreeCircuits.DenseOp(cos_op)), (j, KronFreeCircuits.DenseOp(cos_op))]))
    end
    return KronFreeCircuits.KronFreeOperator([d, d, d], terms)
end

function kron_three_oscillators(d::Int)::Matrix{ComplexF64}
    cos_op, nvec, ω = _three_osc_pieces(d)
    N = Matrix{ComplexF64}(diagm(nvec))
    terms = Tuple{ComplexF64,Vector{Tuple{Int,Matrix{ComplexF64}}}}[]
    for i in 1:3
        push!(terms, (ComplexF64(ω[i]), [(i, N)]))
    end
    for (i, j) in ((1, 2), (2, 3), (1, 3))
        push!(terms, (0.2 + 0im, [(i, cos_op), (j, cos_op)]))
    end
    return kron_reference([d, d, d], terms)
end
