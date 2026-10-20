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

# `E_C[s,s]` and `n_g[s]` of the slow stage, with every parameter of `θ` that the fast spec depends on (the surface assumed its value)
# substituted by that value, except those in `free`; the rest stay free parameters.
function _slow_stage_coefficients(p::PromotedSpec, θ::NamedTuple, free::AbstractVector{Symbol}=Symbol[])
    deps = _spec_dependencies(p.fast)
    fixed = NamedTuple(k => θ[k] for k in keys(θ) if k in deps && k ∉ free)
    return substitute(p.E_C, fixed), substitute(p.n_g, fixed)
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
    E_C, n_g = _slow_stage_coefficients(p, s.θ)
    used = union(E_C.deps, n_g.deps)
    params = filter(in(used), p.source.parameters)
    spec = Spec(params, [p.mode], reshape([E_C], 1, 1), reshape([Coefficient(0)], 1, 1), [Coefficient(0)], [n_g],
                cosines, ChargeCosineSpec[], Coefficient(a0))
    return (spec, tail(keep))
end

# Fits of surfaces to a reduced one-mode spec (design Section 4.4).

"""
    BOFit

The result of [`fit_surfaces`](@ref).
`terms` lists the model's (slow basis key, parameter expression) pairs in model order, and `constants` and `stderr` have one entry per term.
`stderr` is `NaN` for every term when the data points are exactly as many as the constants (no residual degrees of freedom).
`rms_residual` and `max_residual` are the root-mean-square and largest absolute residual over all data points.
`spec` is the reduced one-mode spec and `metadata` its `_bo` and `_generator` provenance, ready for [`save_spec`](@ref).
"""
struct BOFit
    terms::Vector{Tuple{Any,String}}
    constants::Vector{Float64}
    stderr::Vector{Float64}
    rms_residual::Float64
    max_residual::Float64
    spec::Spec
    metadata::Dict{String,Any}
end

# A slow basis key in canonical form (`:constant`, `:x`, `:x2`, or `(:cos, k)`, `(:sin, k)` with `k::Float64`), checked against the slow mode.
function _check_basis_key(key, mode::ModeSpec)
    known = "the keys are :constant, :x, :x2, (:cos, k) and (:sin, k)"
    if key isa Symbol
        key in (:constant, :x, :x2) || throw(ArgumentError("fit_surfaces: unknown slow basis key $(repr(key)); $known"))
        key === :constant && return key
        mode.type in (:e, :h) ||
            throw(ArgumentError("fit_surfaces: the key $(repr(key)) is a polynomial in the slow coordinate, which a compact slow mode " *
                                "(:$(mode.name)) does not have; use :constant, (:cos, k) and (:sin, k)"))
        return key
    end
    (key isa Tuple && length(key) == 2 && key[1] in (:cos, :sin) && key[2] isa Real) ||
        throw(ArgumentError("fit_surfaces: unknown slow basis key $(repr(key)); $known"))
    k = key[2]
    (isfinite(k) && k > 0) ||
        throw(ArgumentError("fit_surfaces: the harmonic of $(repr(key)) must be positive and finite"))
    mode.type === :c && !isinteger(k) &&
        throw(ArgumentError("fit_surfaces: the harmonic of $(repr(key)) must be an integer on the compact slow mode :$(mode.name)"))
    return (key[1], Float64(k))
end

# The value b(s) of a canonical slow basis key.
_basis_value(key::Symbol, s::Real) = key === :constant ? 1.0 : key === :x ? Float64(s) : Float64(s)^2
_basis_value(key::Tuple, s::Real) = key[1] === :cos ? -cos(key[2] * s) : -sin(key[2] * s)
# The largest value `|b|` can take at `s`: the scale against which a column that vanishes on the data is recognized.
_basis_bound(key::Symbol, s::Real) = abs(_basis_value(key, s))
_basis_bound(key::Tuple, s::Real) = 1.0

# Two promoted specs describe the same slow problem: the same source spec, slow mode and slow parameter.
_same_promoted(a::PromotedSpec, b::PromotedSpec) =
    a === b || (a.mode == b.mode && a.parameter == b.parameter && spec_json(a.source) == spec_json(b.source))

function _real_value(c::Coefficient, θ::NamedTuple, text::AbstractString)
    z = c(θ)
    iszero(imag(z)) || throw(ArgumentError("fit_surfaces: the expression \"$text\" is not real at θ = $θ"))
    return real(z)
end

# The coefficient `Σ scale * c_i * h_i` over the terms `idx`, as one expression (one compiled function per spec entry).
function _model_sum(constants, h::Vector{Coefficient}, idx, scale::Real=1)
    ex = Expr(:call, :+, (:($(Float64(scale * constants[i])) * $(h[i].ex)) for i in idx)...)
    return _make(length(idx) == 1 ? ex.args[2] : ex, union((h[i].deps for i in idx)...))
end

"""
    fit_surfaces(surfaces::AbstractVector{BOSurface}, model::AbstractVector{<:Pair}; band::Integer=1) -> BOFit

Fit band `band` of `surfaces` (Born–Oppenheimer surfaces of one slow mode at different parameter samples) to a linear model and return the
fitted constants with the reduced one-mode spec they define.

A model is `[key => ["expr", ...], ...]`.
Each expression is a scalar expression of the source spec's parameters (the `Coefficient` grammar), and each key names a slow basis function `b(s)`;
the fitted surface is `U(s; params) = Σ c_i h_i(params) b_i(s)` with unknown constants `c_i`, one per (key, expression) pair.
The keys are chosen so that each constant multiplies a spec entry directly:

| key | `b(s)` | reduced-spec entry that gains `c h(params)` | slow mode type |
| --- | --- | --- | --- |
| `:constant` | `1` | `constant` | any |
| `(:cos, k)` | `-cos(k s)` | cosine `energy`, `omega = [k]`, `offset = 0` | any (`k` an integer on a compact mode) |
| `(:sin, k)` | `-sin(k s)` | cosine `energy`, `omega = [k]`, `offset = -π/2` | any (`k` an integer on a compact mode) |
| `:x` | `s` | `flux_linear` | extended or harmonic |
| `:x2` | `s²` | `E_L`, as `2 c h` (the form is `½ E_L s²`) | extended or harmonic |

All data points (every surface, every grid point) enter one least-squares solve, `A \\ y` (QR).
`stderr[i] = sqrt(σ² (AᵀA)⁻¹ᵢᵢ)` with `σ² = RSS/(n - p)`; it is `NaN` when `n == p`.
Fewer points than constants, a rank-deficient design matrix (including a basis function that vanishes on every grid point), or an unknown key
is an `ArgumentError`.

The surfaces must share one source spec and slow mode, and may differ only in the parameters that occur in the model's expressions (the fitted-over parameters);
every other entry of their `θ` must agree exactly, else an `ArgumentError` names the parameter.
The reduced spec's parameters are the fitted-over ones and those of `E_C[s,s]` and `n_g[s]` that the fast spec does not depend on;
every other parameter of `E_C[s,s]` and `n_g[s]` is substituted by its shared fixed value, as in [`slow_spec`](@ref).
Parameters held fixed while computing the surfaces therefore never remain free in the reduced spec.
`metadata["_bo"]` records the slow mode, the fixed parameter values, each surface's grid size and fitted-over values, the model, constants,
standard errors and residuals; `metadata["_generator"]` records the package version.
"""
function fit_surfaces(surfaces::AbstractVector{BOSurface}, model::AbstractVector{<:Pair}; band::Integer=1)
    # one concrete method body for every model type (the body is long, and compiling it per key type costs seconds)
    m = Pair{Any,Vector{String}}[k => String.(v) for (k, v) in model]
    return _fit_surfaces(Vector{BOSurface}(surfaces), m, Int(band))
end

function _fit_surfaces(surfaces::Vector{BOSurface}, model::Vector{Pair{Any,Vector{String}}}, band::Int)
    isempty(surfaces) && throw(ArgumentError("fit_surfaces: no surfaces given"))
    isempty(model) && throw(ArgumentError("fit_surfaces: the model is empty"))
    first_s = first(surfaces)
    p = first_s.promoted
    for s in surfaces
        _same_promoted(s.promoted, p) ||
            throw(ArgumentError("fit_surfaces: the surfaces come from different source specs or slow modes"))
        1 <= band <= size(s.energies, 1) ||
            throw(ArgumentError("fit_surfaces: band must be in 1:$(size(s.energies, 1)), got $band"))
    end

    # model terms, in model order
    terms = Tuple{Any,String}[]
    keys_ = Any[]                                   # canonical key per term
    coefs = Coefficient[]
    for (key, exprs) in model
        ckey = _check_basis_key(key, p.mode)
        isempty(exprs) && throw(ArgumentError("fit_surfaces: the expression list of the key $(repr(key)) is empty"))
        for e in exprs
            push!(terms, (key, e))
            push!(keys_, ckey)
            push!(coefs, Coefficient(e, p.source.parameters))
        end
    end
    fitted = unique!(reduce(vcat, (c.deps for c in coefs); init=Symbol[]))

    # the surfaces vary only in the fitted-over parameters
    θ0 = first_s.θ
    for s in surfaces
        for k in fitted
            haskey(s.θ, k) ||
                throw(ArgumentError("fit_surfaces: the model is fitted over :$k, which is missing from a surface's θ $(s.θ)"))
        end
        Set(keys(s.θ)) == Set(keys(θ0)) ||
            throw(ArgumentError("fit_surfaces: the surfaces were computed with different parameters ($(join(keys(θ0), ", ")) vs $(join(keys(s.θ), ", ")))"))
        for k in keys(θ0)
            k in fitted || s.θ[k] == θ0[k] ||
                throw(ArgumentError("fit_surfaces: the surfaces disagree on the fixed parameter :$k ($(θ0[k]) vs $(s.θ[k])); " *
                                    "only parameters that occur in the model's expressions may vary between surfaces"))
        end
    end

    # design matrix and data
    nterm = length(terms)
    n = sum(s -> length(s.grid), surfaces)
    n >= nterm ||
        throw(ArgumentError("fit_surfaces: $n data points cannot determine $nterm constants"))
    A = Matrix{Float64}(undef, n, nterm)
    scale2 = zeros(nterm)                           # squared norm each column would have if its basis function were at its bound
    y = Vector{Float64}(undef, n)
    r = 0
    for s in surfaces
        h = [_real_value(c, s.θ, t[2]) for (c, t) in zip(coefs, terms)]
        for (j, x) in enumerate(s.grid)
            r += 1
            y[r] = s.energies[band, j]
            for i in 1:nterm
                A[r, i] = _basis_value(keys_[i], x) * h[i]
                scale2[i] += (_basis_bound(keys_[i], x) * h[i])^2
            end
        end
    end
    norms = [norm(@view A[:, i]) for i in 1:nterm]
    for i in 1:nterm
        norms[i] > 1e-10 * sqrt(scale2[i]) && continue          # a column of rounding noise (sin kx on the grid points) is zero
        throw(ArgumentError("fit_surfaces: the term $(repr(terms[i][1])) => \"$(terms[i][2])\" vanishes at every data point"))
    end
    sv = svd(A ./ norms'; full=false)
    nrank = count(>(max(n, nterm) * eps() * sv.S[1]), sv.S)
    nrank == nterm ||
        throw(ArgumentError("fit_surfaces: the design matrix is rank deficient (rank $nrank for $nterm constants): " *
                            "some model terms are linearly dependent on these surfaces"))

    c = A \ y
    resid = y - A * c
    rss = sum(abs2, resid)
    if n > nterm
        σ2 = rss / (n - nterm)
        # (AᵀA)⁻¹ᵢᵢ from the SVD of the column-normalized matrix: Σₖ V[i,k]² / S[k]², divided by the column norm squared
        cov_diag = [sum(abs2, sv.V[i, :] ./ sv.S) / norms[i]^2 for i in 1:nterm]
        stderr = sqrt.(σ2 .* cov_diag)
    else
        stderr = fill(NaN, nterm)
    end
    rms = sqrt(rss / n)
    maxres = maximum(abs, resid)

    # reduced spec: constants times the model's expressions, grouped by slow key
    groups = Dict{Any,Vector{Int}}()
    order = Any[]
    for (i, k) in enumerate(keys_)
        haskey(groups, k) || (push!(order, k); groups[k] = Int[])
        push!(groups[k], i)
    end
    zero_c = Coefficient(0)
    entry(k, scale=1) = haskey(groups, k) ? _model_sum(c, coefs, groups[k], scale) : zero_c
    cosines = CosineSpec[]
    for k in order
        k isa Tuple || continue
        push!(cosines, CosineSpec(entry(k), [k[2]], k[1] === :cos ? zero_c : Coefficient(-π / 2)))
    end
    E_C, n_g = _slow_stage_coefficients(p, θ0, fitted)
    used = union(E_C.deps, n_g.deps, fitted)
    spec = Spec(filter(in(used), p.source.parameters), [p.mode], reshape([E_C], 1, 1), reshape([entry(:x2, 2)], 1, 1),
                [entry(:x)], [n_g], cosines, ChargeCosineSpec[], entry(:constant))

    # provenance
    key_json(k) = k isa Symbol ? Dict{String,Any}("basis" => String(k)) : Dict{String,Any}("basis" => String(k[1]), "k" => k[2])
    nan_null(x) = isnan(x) ? nothing : x
    fixed = Dict{String,Any}(String(k) => θ0[k] for k in keys(θ0) if k ∉ fitted)
    bo = Dict{String,Any}(
        "slow_mode" => Dict{String,Any}("name" => String(p.mode.name), "type" => String(p.mode.type), "parameter" => String(p.parameter)),
        "band" => band,
        "fixed_parameters" => fixed,
        "surfaces" => [Dict{String,Any}("grid_size" => length(s.grid),
                                        "fitted_values" => Dict{String,Any}(String(k) => s.θ[k] for k in fitted)) for s in surfaces],
        "model" => [Dict{String,Any}("key" => key_json(k), "expression" => t[2]) for (k, t) in zip(keys_, terms)],
        "constants" => c,
        "stderr" => nan_null.(stderr),
        "rms_residual" => rms,
        "max_residual" => maxres,
    )
    metadata = Dict{String,Any}("_bo" => bo,
                                "_generator" => Dict{String,Any}("package" => "KronFreeCircuits", "version" => string(pkgversion(KronFreeCircuits))))
    return BOFit(terms, c, stderr, rms, maxres, spec, metadata)
end

# Compiled with the package: a first `fit_surfaces` call otherwise spends over a second in inference.
precompile(_fit_surfaces, (Vector{BOSurface}, Vector{Pair{Any,Vector{String}}}, Int))
