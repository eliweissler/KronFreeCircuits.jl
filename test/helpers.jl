fixture(name::AbstractString) = joinpath(@__DIR__, "fixtures", name)

"""
    kron_reference(dims, terms)

Dense reference for `sum_t c_t A_{t,N} ⊗ ⋯ ⊗ A_{t,1}` with identities on axes a term
does not touch.  Axis 1 is fastest (column-major).
`terms` is a vector of `(c, [(axis, matrix), ...])`.
"""
function kron_reference(dims::Vector{Int},
                        terms::Vector{<:Tuple{Number,AbstractVector}})
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

# Three coupled truncated oscillators whose cos couplings are real matrices: a Float64 operator.
function real_three_oscillators(d::Int)
    cos_op, nvec, ω = _three_osc_pieces(d)
    cos_r = real.(cos_op)
    terms = KronFreeCircuits.AxisTerm[]
    for i in 1:3
        push!(terms, KronFreeCircuits.AxisTerm(ω[i], [(i, KronFreeCircuits.DiagonalOp(nvec))]))
    end
    for (i, j) in ((1, 2), (2, 3), (1, 3))
        push!(terms, KronFreeCircuits.AxisTerm(0.2, [(i, KronFreeCircuits.DenseOp(cos_r)), (j, KronFreeCircuits.DenseOp(cos_r))]))
    end
    return KronFreeCircuits.KronFreeOperator([d, d, d], terms)
end

# Two identical uncoupled real oscillators: every level above the ground is degenerate.
function two_identical_real_oscillators(d::Int)
    nvec = collect(0.0:d-1)
    return KronFreeCircuits.KronFreeOperator([d, d], [KronFreeCircuits.AxisTerm(1.0, [(1, KronFreeCircuits.DiagonalOp(nvec))]),
                                                      KronFreeCircuits.AxisTerm(1.0, [(2, KronFreeCircuits.DiagonalOp(nvec))])])
end

"""
    complexified(H::KronFreeOperator)

`H` rebuilt with every coefficient (and the constant) converted to `ComplexF64`, with the same factors.
"""
function complexified(H::KronFreeCircuits.KronFreeOperator)
    terms = KronFreeCircuits.AxisTerm[]
    for t in H.diagterms
        push!(terms, KronFreeCircuits.AxisTerm(ComplexF64(t.coeff), t.factors))
    end
    for t in H.terms
        push!(terms, KronFreeCircuits.AxisTerm(ComplexF64(t.coeff), Tuple{Int,KronFreeCircuits.LocalOp}[(a, op) for (a, op, _) in t.factors]))
    end
    return KronFreeCircuits.KronFreeOperator(H.dims, terms)
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

"""
    check_eigenpairs(H, E, V; tol)

Independent validation of computed eigenpairs: every true residual `‖H v_j − E_j v_j‖ < tol`,
recomputed with `H * v`, and orthonormal columns, `‖V'V − I‖ < 1e-10`.
"""
function check_eigenpairs(H, E, V; tol)
    @test length(E) == size(V, 2)
    for j in eachindex(E)
        @test norm(H * V[:, j] - E[j] * V[:, j]) < tol
    end
    @test norm(V' * V - I) < 1e-10
    return nothing
end

"""
    primitive_hamiltonian(terms, θ, bases; constant=0) -> SparseMatrixCSC{ComplexF64}

Independent primitive-space reference for a term list: `Σ_t c_t(θ) A_{t,N} ⊗ ⋯ ⊗ A_{t,1}` on the
flat product of `bases` (a vector of `mode => basis` pairs, axis 1 fastest), each factor the dense
matrix of the basis's declared operator `localop(basis, port)`, plus `constant` times the identity.
Built with explicit sparse Kronecker products, never with the apply routines.
"""
function primitive_hamiltonian(terms, θ::NamedTuple, bases::AbstractVector; constant=0)
    names = [first(p) for p in bases]
    dims = [KronFreeCircuits.dimension(last(p)) for p in bases]
    D = prod(dims)
    H = spzeros(ComplexF64, D, D)
    for t in terms
        ops = Dict{Int,SparseMatrixCSC{ComplexF64,Int}}()
        for f in t.factors
            a = findfirst(==(f.mode), names)
            a === nothing && throw(ArgumentError("mode :$(f.mode) is not among the bases"))
            haskey(ops, a) && throw(ArgumentError("primitive_hamiltonian: two factors on mode :$(f.mode)"))
            ops[a] = sparse(KronFreeCircuits.dense(KronFreeCircuits.localop(last(bases[a]), f.port)))
        end
        T = sparse(ComplexF64[1;;])
        for a in eachindex(dims)                       # kron(A_N, ..., A_1): axis 1 fastest
            T = kron(get(() -> sparse(ComplexF64(1) * I, dims[a], dims[a]), ops, a), T)
        end
        H += t.coeff(θ) * T
    end
    return H + ComplexF64(constant) * sparse(ComplexF64(1) * I, D, D)
end

# The primitive-space eigenvector columns of a materialized node whose children are leaves or raw
# bases, axis 1 fastest.
_embed(c::KronFreeCircuits.MaterializedNode) = c.vectors
_embed(b::KronFreeCircuits.Basis) = Matrix{ComplexF64}(I, KronFreeCircuits.dimension(b), KronFreeCircuits.dimension(b))
_primitive(mn) = foldl((acc, c) -> kron(_embed(c), acc), mn.children; init=ones(ComplexF64, 1, 1)) * mn.vectors

# Per column, the true residual norm ‖H v − λ v‖ with an independently assembled H.
_residuals(H, E, V) = [norm(H * V[:, j] - E[j] * V[:, j]) for j in eachindex(E)]

using Random

"""
    spec_equal(a::Spec, b::Spec) -> Bool

`a` and `b` define the same functions: equal modes, parameter sets, `omega` and `kappa`, and every coefficient pair agreeing
to `1e-12` relative at three fixed pseudo-random points in `[0.5, 2]` (seeded, sorted parameter order).
"""
function spec_equal(a::KronFreeCircuits.Spec, b::KronFreeCircuits.Spec)
    a.modes == b.modes || return false
    sort(a.parameters) == sort(b.parameters) || return false
    length(a.cosines) == length(b.cosines) || return false
    length(a.charge_cosines) == length(b.charge_cosines) || return false
    all(p -> p[1].omega == p[2].omega, zip(a.cosines, b.cosines)) || return false
    all(p -> p[1].kappa == p[2].kappa, zip(a.charge_cosines, b.charge_cosines)) || return false
    pairs_ = Tuple{KronFreeCircuits.Coefficient,KronFreeCircuits.Coefficient}[]
    append!(pairs_, zip(a.E_C, b.E_C), zip(a.E_L, b.E_L), zip(a.flux_linear, b.flux_linear), zip(a.n_g, b.n_g))
    for (x, y) in zip(a.cosines, b.cosines)
        push!(pairs_, (x.energy, y.energy), (x.offset, y.offset))
    end
    for (x, y) in zip(a.charge_cosines, b.charge_cosines)
        push!(pairs_, (x.energy, y.energy), (x.offset, y.offset))
    end
    push!(pairs_, (a.constant, b.constant))
    for θ in _seeded_points(sort(a.parameters))
        for (x, y) in pairs_
            isapprox(x(θ), y(θ); rtol=1e-12, atol=0) || return false
        end
    end
    return true
end

# Three seeded pseudo-random points in [0.5, 2]^n over the parameter `names`, as NamedTuples.
function _seeded_points(names::AbstractVector{Symbol})
    rng = Random.MersenneTwister(20261020)
    return [NamedTuple{Tuple(names)}(Tuple(0.5 .+ 1.5 .* rand(rng, length(names)))) for _ in 1:3]
end

"""
    coefficients_agree(s::Spec, full::Spec, values::NamedTuple) -> Bool

`s` is `full` with `values` substituted: same modes, `omega`, `kappa`, and every coefficient of `s` at three seeded points in
`[0.5, 2]` equals the matching coefficient of `full` at the same points merged with `values`, to `1e-12` relative.
Cosines and charge cosines of `full` that are missing from `s` must vanish at every point; the rest match in order.
"""
function coefficients_agree(s::KronFreeCircuits.Spec, full::KronFreeCircuits.Spec, values::NamedTuple)
    s.modes == full.modes || return false
    all(p -> p ∉ keys(values), s.parameters) || return false
    all(p -> p in full.parameters, s.parameters) || return false
    pairs_ = Tuple{KronFreeCircuits.Coefficient,KronFreeCircuits.Coefficient}[]
    append!(pairs_, zip(s.E_C, full.E_C), zip(s.E_L, full.E_L), zip(s.flux_linear, full.flux_linear), zip(s.n_g, full.n_g))
    push!(pairs_, (s.constant, full.constant))
    points = _seeded_points(sort(full.parameters))
    agree(x, y) = all(θ -> isapprox(x(θ), y(merge(θ, values)); rtol=1e-12, atol=0), points)
    vanishes(y) = all(θ -> iszero(y(merge(θ, values))), points)
    # entries are (energy, offset, omega-or-kappa) triples; each `full` entry matches the next `s` entry or vanishes
    function match_entries(s_entries, full_entries, same_direction)
        j = 1
        for (energy, offset, direction) in full_entries
            if j <= length(s_entries) && same_direction(s_entries[j][3], direction) &&
               agree(s_entries[j][1], energy) && agree(s_entries[j][2], offset)
                j += 1
            elseif !vanishes(energy)
                return false
            end
        end
        return j == length(s_entries) + 1
    end
    entries(cs, key) = [(c.energy, c.offset, getfield(c, key)) for c in cs]
    match_entries(entries(s.cosines, :omega), entries(full.cosines, :omega), ==) || return false
    match_entries(entries(s.charge_cosines, :kappa), entries(full.charge_cosines, :kappa), ==) || return false
    return all(p -> agree(p[1], p[2]), pairs_)
end

"""
    minimal_spec_json(; constant) -> String

A spec of one compact mode and the parameter `E_J` (no cosines) whose `constant` is the expression string `constant`.
"""
minimal_spec_json(; constant::AbstractString) =
    """{"format": "kronfree-spec", "version": 1, "parameters": ["E_J"], "modes": [{"name": "varphi_1", "type": "c"}],
        "E_C": [["1"]], "E_L": [["0"]], "flux_linear": ["0"], "n_g": ["0"], "cosines": [],
        "constant": $(JSON3.write(constant))}"""

"""
    expr_text(json::AbstractString, key::AbstractString) -> String

The expression string stored under the top-level `key` of the spec JSON `json`.
"""
expr_text(json::AbstractString, key::AbstractString) = String(JSON3.read(json)[Symbol(key)])

# --- Born–Oppenheimer helpers (test_bo.jl) ---

# `promote_mode` with its `@info` silenced.
quiet_promote(spec, mode; kwargs...) =
    Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        KronFreeCircuits.promote_mode(spec, mode; kwargs...)
    end

"""
    crossing_spec() -> Spec

Two compact modes, fast `f` and slow `s`: `4 E_C (n_f - n_g)² + 4 E_Cs (n_s - n_gs)² - E_J cos(s) cos(φ_f)`,
written as two cosines of energy `E_J/2` with `omega` `[1, 1]` and `[1, -1]`.
"""
crossing_spec() = KronFreeCircuits.parse_spec("""{"format": "kronfree-spec", "version": 1,
    "parameters": ["E_C", "E_J", "n_g", "E_Cs", "n_gs"],
    "modes": [{"name": "f", "type": "c"}, {"name": "s", "type": "c"}],
    "E_C": [["E_C", "0"], ["0", "E_Cs"]], "E_L": [["0", "0"], ["0", "0"]], "flux_linear": ["0", "0"], "n_g": ["n_g", "n_gs"],
    "cosines": [{"energy": "E_J/2", "omega": [1, 1], "offset": "0"}, {"energy": "E_J/2", "omega": [1, -1], "offset": "0"}],
    "constant": "0"}""")

"""
    dense_crossing_reference(grid; E_C, E_J, n_g, ncut) -> Matrix{Float64}

All eigenvalues (rows) of `4 E_C (n - n_g)² - E_J cos(s) (S + S')/2` in the charge basis `n = -ncut..ncut`
(`S` the shift `|n⟩ → |n+1⟩`), at each `s` of `grid` (columns).
"""
function dense_crossing_reference(grid; E_C, E_J, n_g, ncut)
    ns = collect(-ncut:ncut)
    D = length(ns)
    S = diagm(-1 => ones(D - 1))
    out = zeros(D, length(grid))
    for (j, s) in enumerate(grid)
        H = Diagonal(4 * E_C .* (ns .- n_g) .^ 2) - E_J * cos(s) * (S + S') / 2
        out[:, j] = eigvals(Symmetric(Matrix(H)))
    end
    return out
end

"""
    synthetic_surface(grid, U; θ=NamedTuple()) -> BOSurface

A one-band `BOSurface` whose energies are `U.(grid)`, over a real `PromotedSpec` (from `promote_mode` of a small two-mode spec:
slow compact `s` with `E_C[s,s] = E_Cs`, `n_g[s] = n_gs`, and a dummy fast mode that does not couple to `s`).
"""
function synthetic_surface(grid, U; θ::NamedTuple=NamedTuple())
    spec = KronFreeCircuits.parse_spec("""{"format": "kronfree-spec", "version": 1, "parameters": ["E_Cs", "n_gs"],
        "modes": [{"name": "s", "type": "c"}, {"name": "f", "type": "c"}],
        "E_C": [["E_Cs", "0"], ["0", "1"]], "E_L": [["0", "0"], ["0", "0"]], "flux_linear": ["0", "0"], "n_g": ["n_gs", "0"],
        "cosines": [], "constant": "0"}""")
    p = quiet_promote(spec, :s)
    g = Vector{Float64}(collect(grid))
    return KronFreeCircuits.BOSurface(p, g, reshape(Float64.(U.(g)), 1, :), θ)
end

"""
    dense_periodic_reference(S; E_C, n_g, ncut=20) -> Vector{Float64}

Eigenvalues of `4 E_C (n - n_g)² + a0 + Σ_k a_k cos kθ + b_k sin kθ` in the charge basis `n = -ncut..ncut`, with `a0, a, b` the
Fourier coefficients of the surface `S` (band 1) computed here directly from `S.energies` by a plain sum, and `cos kθ`, `sin kθ`
the shifts `(S^k ± S^-k)/2`, `(S^k - S^-k)/2i`.
"""
function dense_periodic_reference(S; E_C, n_g, ncut=20)
    U, θ = S.energies[1, :], S.grid
    M = length(θ)
    ns = collect(-ncut:ncut)
    D = length(ns)
    shift(k) = diagm(-k => ones(D - abs(k)))
    H = Matrix{ComplexF64}(Diagonal(4 * E_C .* (ns .- n_g) .^ 2 .+ sum(U) / M))
    for k in 1:M÷2
        a = (k == M / 2 ? 1 : 2) / M * sum(U .* cos.(k .* θ))
        b = k == M / 2 ? 0.0 : 2 / M * sum(U .* sin.(k .* θ))
        H .+= a .* (shift(k) + shift(-k)) ./ 2 .+ b .* (shift(k) - shift(-k)) ./ 2im
    end
    return eigvals(Hermitian(H))
end

"""
    dense_zero_pi_bo_reference(θ, grid, basis) -> Matrix{Float64}

The two lowest eigenvalues (rows) of Appendix B's `H_φ(θ̃) = 4 E_Cφ n_φ² - 2 E_J cos θ̃ cos(φ - φ_ext/2) + E_L φ²` for each
`θ̃` of `grid` (columns), from plain matrices on the finite-difference grid of the `PhaseGrid` `basis` (no spec, term or
`materialize` code); `θ` has `E_Cphi, E_J, E_L, varphi_ext`.
"""
function dense_zero_pi_bo_reference(θ, grid, basis)
    n2 = Matrix(KronFreeCircuits.dense(KronFreeCircuits.localop(basis, KronFreeCircuits.n2port())))
    φ = KronFreeCircuits.coordinates(basis)
    out = zeros(2, length(grid))
    for (j, t) in enumerate(grid)
        H = 4 * θ.E_Cphi * n2 + Diagonal(θ.E_L .* φ .^ 2 .- 2 * θ.E_J * cos(t) .* cos.(φ .- θ.varphi_ext / 2))
        out[:, j] = eigvals(Hermitian(real(H)))[1:2]
    end
    return out
end
