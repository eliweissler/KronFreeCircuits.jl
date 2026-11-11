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

# The primitive-space eigenvector columns of a materialized node, through any number of levels, axis 1 fastest.
_embed(c::KronFreeCircuits.MaterializedNode) = _primitive(c)
_embed(b) = Matrix{ComplexF64}(I, KronFreeCircuits.dimension(b), KronFreeCircuits.dimension(b))
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
    synthetic_surface(grid, U; θ=NamedTuple(), fast_value=0.7) -> BOSurface

A one-band `BOSurface` whose energies are `U.(grid)`, over a real `PromotedSpec` (from `promote_mode` of a small two-mode spec:
slow compact `s` with `E_C[s,s] = E_Cs`, `n_g[s] = n_gs`, and a dummy fast mode with `E_C[f,f] = dummy`, which does not couple to `s`).
The spec also declares `varphi_ext` and `n_g1`, so a fit can vary over them.
The surface's `θ` is the given `θ` (the parameters the surface was computed at) plus `dummy = fast_value`, a fast parameter held fixed.
"""
function synthetic_surface(grid, U; θ::NamedTuple=NamedTuple(), fast_value::Real=0.7)
    p = _synthetic_promoted()
    g = Vector{Float64}(collect(grid))
    return KronFreeCircuits.BOSurface(p, g, reshape(Float64.(U.(g)), 1, :), merge(θ, (dummy=Float64(fast_value),)))
end

# The promoted spec of `synthetic_surface`, built once and shared (`fresh=true` gives a separate, equal one).
const _SYNTHETIC_PROMOTED = Ref{Any}(nothing)
function _synthetic_promoted(; fresh::Bool=false)
    fresh || _SYNTHETIC_PROMOTED[] === nothing || return _SYNTHETIC_PROMOTED[]
    spec = KronFreeCircuits.parse_spec("""{"format": "kronfree-spec", "version": 1,
        "parameters": ["E_Cs", "n_gs", "varphi_ext", "n_g1", "dummy"],
        "modes": [{"name": "s", "type": "c"}, {"name": "f", "type": "c"}],
        "E_C": [["E_Cs", "0"], ["0", "dummy"]], "E_L": [["0", "0"], ["0", "0"]], "flux_linear": ["0", "0"], "n_g": ["n_gs", "0"],
        "cosines": [], "constant": "0"}""")
    p = quiet_promote(spec, :s)
    fresh || (_SYNTHETIC_PROMOTED[] = p)
    return p
end

"""
    coupled_oscillators_spec() -> Spec

Two extended modes, slow `s` and fast `f`: `4 E_Cs n_s² + 4 E_Cf n_f² + ½ a φ_s² + ½ b φ_f² + c φ_s φ_f`.
For fixed `s` the fast ground energy is `ω/2 + ½ (a - c²/b) s²` with `ω = sqrt(8 E_Cf b)`.
"""
coupled_oscillators_spec() = KronFreeCircuits.parse_spec("""{"format": "kronfree-spec", "version": 1,
    "parameters": ["a", "b", "c", "E_Cf", "E_Cs"],
    "modes": [{"name": "s", "type": "e"}, {"name": "f", "type": "e"}],
    "E_C": [["E_Cs", "0"], ["0", "E_Cf"]], "E_L": [["a", "c"], ["c", "b"]], "flux_linear": ["0", "0"], "n_g": ["0", "0"],
    "cosines": [], "constant": "0"}""")

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

# Three compact modes coupled by one cosine on all three: grouping `a` and `b` in one child makes its terms joint.
# Every leaf of such a tree holds only charging terms, so its kept states are bare charge states, and those on negative
# charges have no weight on the phase mask (a warning, captured by the tests).
const abc_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_C", "E_J"],
 "modes": [{"name": "a", "type": "c"}, {"name": "b", "type": "c"}, {"name": "c", "type": "c"}],
 "E_C": [["E_C", "0", "0"], ["0", "E_C", "0"], ["0", "0", "E_C"]],
 "E_L": [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]],
 "flux_linear": ["0", "0", "0"], "n_g": ["0", "0", "0"],
 "cosines": [{"energy": "E_J", "omega": [1, 1, 1], "offset": "0"}]}
"""

# Runs `f()` and passes when every record it logs at Warn level or above is a warning matching `rx`, any number of them
# (none included); returns its value. The number of such warnings can depend on the path the self-consistent loop takes.
function only_warnings(rx, f)
    logs, val = Test.collect_test_logs(f; min_level=Base.CoreLogging.Warn)
    unexpected = [l.message for l in logs if !(l.level == Base.CoreLogging.Warn && occursin(rx, string(l.message)))]
    @test isempty(unexpected)
    isempty(unexpected) || foreach(m -> println(stderr, "unexpected log: ", first(string(m), 200)), unexpected)
    return val
end

# --- finite-difference helpers (derivative tests) ---

# Runs `f()` with logging silenced.
quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# `θ` with its entry `x` moved by `δ`.
shifted(θ::NamedTuple, x::Symbol, δ::Real) = merge(θ, NamedTuple{(x,)}((θ[x] + δ,)))

# The largest absolute element of a number or an array.
_maxabs(v) = maximum(abs, v)

# `(value, err)` from the plain difference `D(s)` at the steps `s`, `s/2`, `s/4`: with `R(s) = (4 D(s/2) − D(s)) / 3`
# its Richardson extrapolation, `value = R(s/2)` and `err = ‖R(s) − R(s/2)‖∞`.
function _richardson(D, s::Real)
    D1, D2, D4 = D(s), D(s / 2), D(s / 4)
    R1 = (4 .* D2 .- D1) ./ 3
    R2 = (4 .* D4 .- D2) ./ 3
    return (R2, _maxabs(R1 .- R2))
end

"""
    fd_derivative(f, θ, x; h) -> (value, err)

The first derivative of `f(θ)` (a number or an array) in `x`, from the central difference `(f(θ + s) − f(θ − s)) / 2s`.
It is Richardson-extrapolated over the steps `h`, `h/2`, `h/4`: `value = R(h/2)`, `err = ‖R(h) − R(h/2)‖∞`.
"""
fd_derivative(f, θ::NamedTuple, x::Symbol; h::Real) =
    _richardson(s -> (f(shifted(θ, x, s)) .- f(shifted(θ, x, -s))) ./ (2s), h)

"""
    fd_second(f, θ, x; h) -> (value, err)

The second derivative of `f(θ)` in `x`, from `(f(θ + s) − 2 f(θ) + f(θ − s)) / s²`, extrapolated as in `fd_derivative`.
"""
function fd_second(f, θ::NamedTuple, x::Symbol; h::Real)
    f0 = f(θ)
    return _richardson(s -> (f(shifted(θ, x, s)) .- 2 .* f0 .+ f(shifted(θ, x, -s))) ./ s^2, h)
end

"""
    fd_mixed(f, θ, x, λ; hx, hλ) -> (value, err)

The mixed derivative of `f(θ)` in `x` and `λ`, from `(f(+,+) − f(+,−) − f(−,+) + f(−,−)) / (4 s² hx hλ)` at the displacements `±s·hx` of `x` and `±s·hλ` of `λ`.
It is extrapolated over `s = 1, 1/2, 1/4`: `value = R(1/2)`, `err = ‖R(1) − R(1/2)‖∞`.
"""
function fd_mixed(f, θ::NamedTuple, x::Symbol, λ::Symbol; hx::Real, hλ::Real)
    at(a, b) = f(shifted(shifted(θ, x, a), λ, b))
    D(s) = (at(s * hx, s * hλ) .- at(s * hx, -s * hλ) .- at(-s * hx, s * hλ) .+ at(-s * hx, -s * hλ)) ./ (4 * s^2 * hx * hλ)
    return _richardson(D, 1)
end

# The second derivative in `x` and `λ`, diagonal or mixed.
fd_pair(f, θ, x::Symbol, λ::Symbol; hx::Real, hλ::Real = hx) =
    x === λ ? fd_second(f, θ, x; h=hx) : fd_mixed(f, θ, x, λ; hx, hλ)

"""
    check_fd(analytic, fd; rtol=1e-6, atol=0)

Two `@test`s against the finite difference `fd = (value, err)`: `err ≤ rtol·‖value‖∞ + atol` (the difference is trustworthy) and `‖analytic − value‖∞ ≤ rtol·‖value‖∞ + atol`, with `‖·‖∞` the largest absolute element.
"""
function check_fd(analytic, fd::Tuple; rtol::Real = 1e-6, atol::Real = 0.0)
    value, err = fd
    tol = rtol * _maxabs(value) + atol
    @test err <= tol
    @test _maxabs(analytic .- value) <= tol
    return nothing
end

# --- derivative test circuits (shared by every derivative test file) ---

const chain3_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_J1", "E_J2", "E_J3", "g12", "g23", "n_g2"],
 "modes": [{"name": "varphi_1", "type": "c"}, {"name": "varphi_2", "type": "c"}, {"name": "varphi_3", "type": "c"}],
 "E_C": [["1.2", "g12/8", "0"], ["g12/8", "1.0", "g23/8"], ["0", "g23/8", "0.8"]],
 "E_L": [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]],
 "flux_linear": ["0", "0", "0"], "n_g": ["0.21", "n_g2", "0.1"],
 "cosines": [{"energy": "E_J1", "omega": [1, 0, 0], "offset": "0"}, {"energy": "E_J2", "omega": [0, 1, 0], "offset": "0"},
             {"energy": "E_J3", "omega": [0, 0, 1], "offset": "0"}]}
"""
const abc_deriv_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_C", "E_J", "E_J3", "n_ga", "varphi_ext"],
 "modes": [{"name": "a", "type": "c"}, {"name": "b", "type": "c"}, {"name": "c", "type": "c"}],
 "E_C": [["E_C", "0", "0"], ["0", "1.3*E_C", "0"], ["0", "0", "0.8*E_C"]],
 "E_L": [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]],
 "flux_linear": ["0", "0", "0"], "n_g": ["n_ga", "0.13", "0.37"],
 "cosines": [{"energy": "E_J", "omega": [1, 0, 0], "offset": "0"}, {"energy": "0.7*E_J", "omega": [0, 1, 0], "offset": "0"},
             {"energy": "1.2*E_J", "omega": [0, 0, 1], "offset": "0"}, {"energy": "E_J3", "omega": [1, 1, 1], "offset": "varphi_ext"}]}
"""
const oscillator_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz", "parameters": ["E_C", "E_L"],
 "modes": [{"name": "phi", "type": "e"}], "E_C": [["E_C"]], "E_L": [["E_L"]], "flux_linear": ["0"], "n_g": ["0"],
 "cosines": [], "constant": "0"}
"""

# One transmon leaf: real on a charge basis, complex on a periodic grid (its `n` is imaginary there).
function dc_transmon(; periodic::Bool=false, keep=lowest(4), compute::Int=0, solver=Dense(), retain::Symbol=:auto,
                     θ=(E_J=20.0, E_C=1.0, n_g=0.2))
    b = periodic ? PeriodicGrid(64; order=8) : ChargeBasis(30)
    root = Node([:varphi_1 => b]; solver, compute, keep, retain, ports=[(:varphi_1, nport()), (:varphi_1, expiphi(1.0))])
    return (; root, spec=load_spec(fixture("transmon.json")), θ)
end
# One fluxonium leaf (real), 601 grid points, off its sweet spots.
function dc_fluxonium(; varphi_ext::Real=2.0, E_J::Real=8.9, solver=Dense(), compute::Int=8, keep=lowest(5), retain::Symbol=:auto)
    root = Node([:varphi_1 => PhaseGrid(-10π, 10π, 601; order=8)]; solver, compute, keep, retain,
                ports=[(:varphi_1, phiport()), (:varphi_1, nport())])
    return (; root, spec=load_spec(fixture("fluxonium.json")), θ=(E_J=E_J, E_C=2.5, E_L=0.5, varphi_ext=varphi_ext))
end
# One harmonic oscillator on a Fock basis matched to θ: H is diagonal there, so closed forms hold to round-off.
function dc_oscillator()
    θ = (E_C=0.5, E_L=1.3)
    root = Node([:phi => FockBasis(40; phi_zpf=(2 * θ.E_C / θ.E_L)^0.25)]; solver=Dense(), keep=lowest(6))
    return (; root, spec=KronFreeCircuits.parse_spec(oscillator_json), θ)
end
# Two coupled transmons, a two-level tree: real on charge bases, complex (leaves and root) on periodic grids.
# The offset charges 0.21 and 0.09 (here and in `dc_chain3`) are deliberate: a transmon leaf with E_J/E_C = 25 on a charge
# basis changes the peak index of one of its four lowest states at n_g ≈ 0.024, 0.16 and 0.258 (measured), and a finite
# difference across such a change is meaningless.
function dc_two_transmons(; periodic::Bool=false, solver=Dense(), compute::Int=0, keep=lowest(6), leafkeep::Int=4, g::Real=0.3,
                          retain::Symbol=:auto)
    b() = periodic ? PeriodicGrid(64; order=8) : ChargeBasis(30)
    leaves = [Node([Symbol("varphi_$i") => b()]; solver=Dense(), keep=lowest(leafkeep)) for i in 1:2]
    root = Node(leaves; solver, compute, keep, retain, ports=[(:varphi_1, nport()), (:varphi_1, expiphi(1.0))])
    θ = (E_J1=30.0, E_C1=1.2, n_g1=0.21, E_J2=25.0, E_C2=1.0, n_g2=0.09, g=g)
    return (; root, leaves, spec=load_spec(fixture("two_transmons.json")), θ)
end
# Three transmons in a chain, a three-level tree (real): `inner` holds modes 1 and 2, and `g23` couples it to the third.
function dc_chain3()
    leaves = [Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(4)) for i in 1:2]
    inner = Node(leaves; solver=Dense(), keep=lowest(10))
    third = Node([:varphi_3 => ChargeBasis(10)]; solver=Dense(), keep=lowest(4))
    root = Node([inner, third]; solver=Dense(), keep=lowest(6), ports=[(:varphi_1, nport())])
    θ = (E_J1=30.0, E_J2=25.0, E_J3=20.0, g12=0.3, g23=0.2, n_g2=0.09)
    return (; root, inner, third, leaves, spec=KronFreeCircuits.parse_spec(chain3_json), θ)
end
# The zero-pi as a two-level tree: every node is real, but the root's terms carry complex coefficients on paired ports.
# Leaf A holds bare charge states and warns about the phase mask; wrap its materializations in `quiet`.
function dc_zero_pi()
    A = Node([:theta => ChargeBasis(8)]; solver=Dense(), keep=lowest(8))
    B = Node([:phi => PhaseGrid(-12π, 12π, 201; order=6)]; solver=Dense(), compute=10, keep=lowest(10))
    root = Node([A, B]; solver=Dense(), keep=lowest(6), ports=[(:theta, nport()), (:phi, phiport())])
    return (; root, A, B, spec=load_spec(fixture("zero_pi.json")), θ=zero_pi_params())
end
# Three compact modes with one cosine on all three: `ab` is a two-mode leaf, so the root applies its joint ports.
# The root is complex (the cosine's offset), its children real.
function dc_abc()
    ab = Node([:a => ChargeBasis(5), :b => ChargeBasis(5)]; solver=Dense(), keep=lowest(8))
    c = Node([:c => ChargeBasis(5)]; solver=Dense(), keep=lowest(5))
    root = Node([ab, c]; solver=Dense(), keep=lowest(6))
    θ = (E_C=1.0, E_J=6.0, E_J3=0.6, n_ga=0.21, varphi_ext=0.7)
    return (; root, ab, c, spec=KronFreeCircuits.parse_spec(abc_deriv_json), θ)
end
# The cos 2φ element near its parity degeneracy (at n_g = 1/2, E_1 = 0): a doublet with detuning 0.0593 GHz at n_g = 0.49,
# coupled by the parameter E_1 with matrix element 0.7456.
function dc_cos2phi(; n_g::Real=0.49, keep=lowest(4), basis=ChargeBasis(20), E_2::Real=5.0)
    root = Node([:phi => basis]; solver=Dense(), keep, ports=[(:phi, expiphi(1.0))])
    return (; root, spec=load_spec(fixture("cos2phi.json")), θ=(E_C=1.0, n_g=n_g, E_1=0.0, E_2=E_2))
end

# A circuit solved at θ with logging silenced, and the same memoized on θ: the finite differences of many quantities of
# one circuit then share one set of solves, which is what keeps the tree tests fast.
dc_solve(c, θ=c.θ) = quiet(() -> materialize(c.root, c.spec, θ))
function dc_memo(c)
    cache = Dict{Any,Any}()
    return θ -> get!(() -> dc_solve(c, θ), cache, θ)
end

# The derivative data of the node `child` of circuit `c` by finite differences, holding only what the operator builder
# reads (energies, ports, joint ports; zero vectors, gauge and peak amplitudes; no children).  `at` is `dc_memo(c)`.
# Three arguments: the first derivative along `x`.  Four: the second derivative along `x` and `λ` (`fd_pair`).
function fd_child(at, c, child, x::Symbol; h)
    return _fd_child(at, c, child, f -> first(fd_derivative(f, c.θ, x; h)))
end
function fd_child(at, c, child, x::Symbol, λ::Symbol; hx, hλ=hx)
    return _fd_child(at, c, child, f -> first(fd_pair(f, c.θ, x, λ; hx, hλ)))
end
function _fd_child(at, c, child, fd)
    K = KronFreeCircuits
    m = K.materialized(at(c.θ), child)
    ports = Dict{Tuple{Symbol,K.Port},K.NodeMatrix}(k => fd(θ -> K.materialized(at(θ), child).ports[k]) for k in keys(m.ports))
    joint = Dict{Vector{Tuple{Symbol,K.Port}},K.NodeMatrix}(k => fd(θ -> K.materialized(at(θ), child).joint_ports[k]) for k in keys(m.joint_ports))
    return K.NodeDerivative{eltype(m.vectors)}(fd(θ -> K.materialized(at(θ), child).energies), zero(m.vectors), zeros(length(m.kept)),
                                               ports, joint, zero(m.peak_amplitudes), Any[nothing for _ in m.children])
end

# What the per-node rules need at the root of a one-node system: the node, "no child data", its terms, its resolvent.
function node_setup(sys)
    K = KronFreeCircuits
    mn = sys.root
    res = K._resolvent(mn, mn.retained.complete ? nothing : K.operator(sys))
    return (; mn, none=Any[nothing for _ in mn.children], terms=K.route(mn.node, sys.terms)[mn.node], res)
end
# The first-order rule on a one-node system.
function node_first_order(sys, x::Symbol; setup=node_setup(sys))
    K = KronFreeCircuits
    H1 = K._derivative_operator(setup.mn, setup.terms, sys.θ, x, setup.none; constant=sys.constant)
    return K._node_first_order(setup.mn, H1, setup.res, setup.none)
end
