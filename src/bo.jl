# Born–Oppenheimer surfaces and the compact slow stage (design Sections 4.2 and 4.3).

"""
    periodic_grid(M::Integer; start::Real=0.0) -> Vector{Float64}

The `M` points `start + 2π k/M`, `k = 0..M-1`: a uniform grid on a half-open `2π` interval.
"""
function periodic_grid(M::Integer; start::Real=0.0)
    M >= 1 || throw(ArgumentError("periodic_grid: M must be at least 1, got $M"))
    return start .+ 2π .* (0:M-1) ./ M
end

"""
    BOSurface

The Born–Oppenheimer surfaces of a [`PromotedSpec`](@ref) on a grid of the slow coordinate.
`energies` is `bands × length(grid)` and holds absolute energies (the fast spec's constant is included);
row `k` is the `k`-th lowest fast energy at each grid point (an envelope, not a branch followed through crossings).
`θ` holds the parameter values the surface was computed at, without the slow parameter.
"""
struct BOSurface
    promoted::PromotedSpec
    grid::Vector{Float64}
    energies::Matrix{Float64}
    θ::NamedTuple
end

"""
    bo_surface(fast_root::Node, p::PromotedSpec, θ::NamedTuple, grid::AbstractVector{<:Real}; bands::Integer=2) -> BOSurface

At each slow coordinate `x` of `grid`, materialize `fast_root` on `p.fast` at `θ` with `p.parameter = x` and record the lowest `bands`
energies.
Band 1 is the surface, and band 2 gives the gap to the first fast excitation.
`fast_root` must keep at least `bands` states, and `θ` must not contain `p.parameter`; otherwise an `ArgumentError` is thrown.
"""
function bo_surface(fast_root::Node, p::PromotedSpec, θ::NamedTuple, grid::AbstractVector{<:Real}; bands::Integer=2)
    bands >= 1 || throw(ArgumentError("bo_surface: bands must be at least 1, got $bands"))
    p.parameter in keys(θ) &&
        throw(ArgumentError("bo_surface: θ contains :$(p.parameter), the slow parameter; the grid supplies it"))
    g = Vector{Float64}(grid)
    E = zeros(Float64, bands, length(g))
    # Build the term list and constant once: every `Coefficient` compiles a fresh generated function, so rebuilding them
    # per grid point (as the spec method of `materialize` would) costs about 30 ms each.
    modetypes = Dict{Symbol,Symbol}(m.name => m.type for m in p.fast.modes)
    _check_bases(fast_root, modetypes)
    bases = _tree_bases(fast_root)
    terms = circuit_terms(p.fast, bases)
    c = constant_term(p.fast, bases)
    for (j, x) in enumerate(g)
        sys = materialize(fast_root, terms, merge(θ, NamedTuple{(p.parameter,)}((x,))); constant=c, modetypes=modetypes)
        e = energies(sys)
        length(e) >= bands ||
            throw(ArgumentError("bo_surface: bands = $bands, but the fast tree keeps only $(length(e)) states"))
        E[:, j] = real.(e[1:bands])
    end
    return BOSurface(p, g, E, θ)
end

# The parameters a spec's coefficients depend on: the union of `deps` over every coefficient.
function _spec_dependencies(spec::Spec)
    cs = Coefficient[spec.E_C...; spec.E_L...; spec.flux_linear; spec.n_g; spec.constant]
    for c in spec.cosines
        push!(cs, c.energy, c.offset)
    end
    for c in spec.charge_cosines
        push!(cs, c.energy, c.offset)
    end
    return unique!(reduce(vcat, (c.deps for c in cs); init=Symbol[]))
end

function _check_compact_periodic(s::BOSurface, band::Integer)
    s.promoted.mode.type === :c ||
        throw(ArgumentError("fourier_coefficients: the slow mode :$(s.promoted.mode.name) is not compact (type :$(s.promoted.mode.type)); " *
                            "an extended slow mode reaches the slow stage only through a fit"))
    1 <= band <= size(s.energies, 1) ||
        throw(ArgumentError("band must be in 1:$(size(s.energies, 1)), got $band"))
    M = length(s.grid)
    M >= 1 || throw(ArgumentError("fourier_coefficients: the grid is empty"))
    step = 2π / M
    all(j -> abs(s.grid[j+1] - s.grid[j] - step) <= 1e-12, 1:M-1) ||
        throw(ArgumentError("fourier_coefficients: the grid must be uniform and periodic (M points, spacing 2π/M, half-open), " *
                            "as periodic_grid(M) gives; an endpoint-repeating or unevenly spaced grid has no exact Fourier coefficients"))
    return nothing
end

"""
    fourier_coefficients(s::BOSurface; band::Integer=1) -> (a0, a, b)

The discrete Fourier coefficients of band `band` of a surface on a uniform periodic grid (`M` points, spacing `2π/M`):
`U(θ) = a0 + Σ_k a_k cos kθ + b_k sin kθ`, exact at the grid points, with `a0` the mean, `k = 1..⌊M/2⌋`,
`a_k = (2/M) Σ U_j cos kθ_j`, `b_k = (2/M) Σ U_j sin kθ_j`, and for even `M` the Nyquist harmonic with the factor `1/M` instead of `2/M`.
Since `cos(Mθ_j/2)` and `sin(Mθ_j/2)` are both `(-1)^j` times a constant, the Nyquist pair reproduces the data exactly for any grid start
(for the default start `b_{M/2}` is zero up to rounding).
Throws `ArgumentError` if the slow mode is not compact or the grid is not uniform and periodic.
"""
function fourier_coefficients(s::BOSurface; band::Integer=1)
    _check_compact_periodic(s, band)
    M = length(s.grid)
    U = s.energies[band, :]
    K = M ÷ 2
    a = zeros(K)
    b = zeros(K)
    for k in 1:K
        c = 2k == M ? 1 / M : 2 / M
        a[k] = c * sum(U[j] * cos(k * s.grid[j]) for j in 1:M)
        b[k] = c * sum(U[j] * sin(k * s.grid[j]) for j in 1:M)
    end
    return (sum(U) / M, a, b)
end

"""
    slow_spec(s::BOSurface; band::Integer=1, K=nothing, tol=nothing) -> (spec::Spec, dropped::Float64)

The one-mode spec of the slow mode of `s`: `E_C` and `n_g` from the promoted spec, no inductive terms, the mean of the surface as the constant, and
for each kept harmonic `k` the cosines `-a_k cos kθ` and `-b_k cos(kθ - π/2)` (entries with zero energy are omitted).
Harmonics `k ≤ K` are kept; with `tol` instead, the smallest `K` whose dropped amplitude `Σ_{k>K} hypot(a_k, b_k)` is at most `tol`;
with neither, all of them.
`dropped` is that amplitude sum for the harmonics left out.
`K` and `tol` are mutually exclusive, and `K` at or above the highest harmonic keeps all.

Every parameter in `s.θ` that `s.promoted.fast` depends on is substituted by its value in `E_C` and `n_g` (the surface assumed it);
the rest stay free parameters of the returned spec.
"""
function slow_spec(s::BOSurface; band::Integer=1, K::Union{Nothing,Integer}=nothing, tol::Union{Nothing,Real}=nothing)
    K === nothing || tol === nothing || throw(ArgumentError("slow_spec: give K or tol, not both"))
    K === nothing || K >= 0 || throw(ArgumentError("slow_spec: K must be nonnegative, got $K"))
    tol === nothing || tol >= 0 || throw(ArgumentError("slow_spec: tol must be nonnegative, got $tol"))
    a0, a, b = fourier_coefficients(s; band)
    amp = hypot.(a, b)
    Kmax = length(amp)
    tail(k) = sum(@view amp[k+1:end]; init=0.0)               # amplitude dropped by keeping k harmonics
    keep = K !== nothing ? min(K, Kmax) :
           tol !== nothing ? something(findfirst(k -> tail(k) <= tol, 0:Kmax), Kmax + 1) - 1 : Kmax
    cosines = CosineSpec[]
    for k in 1:keep
        iszero(a[k]) || push!(cosines, CosineSpec(Coefficient(-a[k]), [Float64(k)], Coefficient(0)))
        iszero(b[k]) || push!(cosines, CosineSpec(Coefficient(-b[k]), [Float64(k)], Coefficient(-π / 2)))
    end

    p = s.promoted
    deps = _spec_dependencies(p.fast)
    fixed = NamedTuple(k => s.θ[k] for k in keys(s.θ) if k in deps)
    E_C = substitute(p.E_C, fixed)
    n_g = substitute(p.n_g, fixed)
    used = union(E_C.deps, n_g.deps)
    params = filter(in(used), p.source.parameters)
    spec = Spec(params, [p.mode], reshape([E_C], 1, 1), reshape([Coefficient(0)], 1, 1), [Coefficient(0)], [n_g],
                cosines, ChargeCosineSpec[], Coefficient(a0))
    return (spec, tail(keep))
end
