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

using JSON3

"""
    reference(name) -> (energies::Vector{Float64}, params::Dict{Symbol,Any})

The scqubits reference entry `name` of `scqubits_reference.json`.
"""
function reference(name::String)
    entry = JSON3.read(read(fixture("scqubits_reference.json"), String))[Symbol(name)]
    params = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(entry.params))
    return (energies=Vector{Float64}(entry.energies), params=params)
end

two_transmons_params(; g) = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.0, g=g)
three_transmons_params(; g) = (two_transmons_params(; g)..., E_J3=20.0, E_C3=0.8, n_g3=0.1)
zero_pi_params() = (E_J=10.0, E_L=0.04, E_CJ=20.0, E_CS=0.5, varphi_ext=2π * 0.23, n_g=0.1)
full_zero_pi_params() = (zero_pi_params()..., E_Cz=Float64(reference("full_zero_pi").params[:E_Cz]),
                         dE_J=0.05, dC_J=0.05, dC=0.08, dE_L=0.1)
function two_zero_pi_params(; g)
    p = zero_pi_params()
    suffixed(s) = NamedTuple{Tuple(Symbol(k, s) for k in keys(p))}(values(p))
    return (suffixed("1")..., suffixed("2")..., g=g)
end
