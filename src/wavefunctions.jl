# Phase and charge wavefunctions of a single mode, in the convention ⟨φ|n⟩ = e^{inφ}/√2π: the one the port operators
# already use (`e^{ikφ}|n⟩ = |n+k⟩`, `n = -i∂_φ`).  Function values, on the basis's own grid.

# The indices conjugate to an `N`-point grid, centred: `-(N-1)/2..(N-1)/2` for odd `N`, `-N/2..N/2-1` for even `N`
# (the order `fftshift` gives).
_conjugate_index(N::Int) = -(N ÷ 2):((N - 1) ÷ 2)

function _check_rows(b::Basis, V::AbstractVecOrMat, name::AbstractString)
    size(V, 1) == dimension(b) ||
        throw(DimensionMismatch("$name: V has $(size(V, 1)) rows, the basis has $(dimension(b)) states"))
    return nothing
end

# X[k] = Σ_j V[j] e^{-2πi jk/N} along the first axis, `j = 0..N-1`, in the order of `_conjugate_index(N)`.
_centred_dft(V::AbstractVecOrMat) = FFTW.fftshift(FFTW.fft(Array{ComplexF64}(V), 1), 1)

# `V` as a dense host array in floating point (a copy): a range, a view, a structured or a sparse `V` all give an `Array`.
_dense(V::AbstractVecOrMat) = Array{float(eltype(V))}(V)

_sign(k::Integer) = isodd(k) ? -1 : 1                       # e^{±iπk}, exactly

"""
    phase_wavefunction(basis, V) -> (φ, ψ)
    phase_wavefunction(basis::FockBasis, V, φ) -> (φ, ψ)
    phase_wavefunction(sys, [node], [φ]; charge=(), points=(;), states=:) -> (φ, ψ) or (axes, ψ)

The phase wavefunction `ψ(φ) = ⟨φ|ψ⟩` of the states in the columns of `V` (a vector, or a matrix with one state per
column, in `basis`), as function values at the points `φ`, with `⟨φ|n⟩ = e^{inφ}/√2π`.
`ψ` has the shape of `V` with one row per point, and `Δ Σ_j |ψ(φ_j)|² = ‖v‖²` for the spacing `Δ` of `φ`.
The points are the basis's own grid: nothing is interpolated.

| basis | `φ` | `ψ` |
|:------|:----|:----|
| `ChargeBasis(ncut)`, `N = 2ncut + 1` | `-π + 2πj/N`, the grid of `PeriodicGrid(N)` | `(1/√2π) Σ_n c_n e^{inφ_j}` |
| `PeriodicGrid`, `PhaseGrid` | the grid | `v_j/√Δ` |
| `FockBasis(N; phi_zpf)` | the points passed | `Σ_m c_m h_m(φ; φ_zpf)` |

`h_m(x; s) = (2π s²)^{-1/4} (2^m m!)^{-1/2} H_m(x/(√2 s)) e^{-x²/(4s²)}` is the oscillator function of level `m`.
A Fock basis has no grid, so its points are required, and its normalization is `∫|ψ(φ)|² dφ = ‖v‖²`.
The values of a real grid or Fock vector stay real; those of a charge basis are `ComplexF64`.
The result is a dense array in floating point, a new one, whatever holds `V` (a range, a view, a sparse or structured matrix).
See [`charge_wavefunction`](@ref) for the conjugate representation.

# Solved nodes

On a node of a solved system (the root by default) the states are the node's kept states, `states` selects some of them
(an index, a range or a vector; the state axis is kept), and the node must be a leaf: every child a basis.
A branch node throws.

On a leaf with several modes the result is `(axes, ψ)`: `axes` is a `NamedTuple` of the modes' grids keyed by mode name,
in the node's mode order, and `ψ[i₁, …, i_N, s]` is state `s` at the grid points, each mode transformed by its rule above
(the modes in `charge`, a mode name or a collection of them, to charge; the others to phase).
A `FockBasis` mode takes its points per mode, `points = (phi = φs,)`.
The normalization is the product of the one-mode rules, `(Π_k Δ_k) Σ |ψ[…, s]|² = 1`, where `Δ_k` is the spacing of axis
`k` and `1` for the charges of a `ChargeBasis` or a `PeriodicGrid`; a `FockBasis` axis is sampled at the points passed,
so along it the sum only approximates the integral `∫|ψ|² = 1`.
The result is real when every transform keeps the state real, `ComplexF64` otherwise.
A leaf with one mode keeps the one-mode form `(φ, ψ)`, and its points may also be passed positionally.
"""
function phase_wavefunction end

"""
    charge_wavefunction(basis, V) -> (n, ψ̃)
    charge_wavefunction(basis::FockBasis, V, n) -> (n, ψ̃)
    charge_wavefunction(sys, [node], [n]; phase=(), points=(;), states=:) -> (n, ψ̃) or (axes, ψ̃)

The charge wavefunction `⟨n|ψ⟩` of the states in the columns of `V` (a vector, or a matrix with one state per column, in
`basis`), with `⟨φ|n⟩ = e^{inφ}/√2π`; the result has the shape of `V` with one row per charge.

| basis | `n` | values |
|:------|:----|:-------|
| `ChargeBasis(ncut)` | `-ncut..ncut` | `c_n`, a dense floating-point copy of `V` (real stays real) |
| `PeriodicGrid(N)`, `Δ = 2π/N` | integers `ℓ` | `c_ℓ = (√2π/N) Σ_j ψ(φ_j) e^{-iℓφ_j}`, with `Σ_ℓ |c_ℓ|² = ‖v‖²` |
| `PhaseGrid(a, b, N)`, spacing `Δ` | `n_k = 2πk/(NΔ)` | `ψ̃(n_k) = (Δ/√2π) Σ_j ψ(φ_j) e^{-i n_k φ_j}`, with `Δn Σ_k |ψ̃(n_k)|² = ‖v‖²`, `Δn = 2π/(NΔ)` |
| `FockBasis(N; phi_zpf)` | the points passed | `ψ̃(n) = Σ_m c_m (-i)^m h_m(n; n_zpf)`, `n_zpf = 1/(2φ_zpf)`, with `∫|ψ̃(n)|² dn = ‖v‖²` |

Here `ψ(φ_j) = v_j/√Δ` are the values [`phase_wavefunction`](@ref) returns, and `ℓ` and `k` run over `-(N-1)/2..(N-1)/2`
for odd `N` and `-N/2..N/2-1` for even `N`.
`h_m` is the oscillator function given under [`phase_wavefunction`](@ref); a Fock basis has no grid, so its points are required.
A compact mode has a discrete charge and an extended one a continuous charge, sampled at the points conjugate to its grid.
The sums are one fast Fourier transform, and the grid's origin enters as a phase factor, so a real even `ψ(φ)` gives a
real `ψ̃(n)`.

On a solved leaf the keywords are those of [`phase_wavefunction`](@ref) with the roles exchanged: every mode goes to
charge except those named in `phase`, and a multi-mode leaf returns `(axes, ψ̃)`.

Limits:

- The charge resolution of a `PhaseGrid` is fixed by its window: `Δn = 2π/(NΔ)` over `|n| ≤ π/Δ`
  (`Δn ≈ 0.1` for `PhaseGrid(-10π, 10π, 601)`). A finer charge grid needs a wider phase window.
- A `PhaseGrid` holds both endpoints, so the transform's period is `NΔ`, one step more than `b - a`; this is harmless for a
  state that vanishes at the edges of the grid, and wraps the state around otherwise.
- The charges of an even-`N` `PeriodicGrid` are not those of a `ChargeBasis`, which always has an odd dimension.
- A grid basis discretizes `n` by finite differences, so the result is the state's representation in charge, not an
  eigenvector of the Hamiltonian written in a charge basis.
"""
function charge_wavefunction end

function phase_wavefunction(b::ChargeBasis, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "phase_wavefunction")
    N = dimension(b)
    # ψ(φ_j) = (1/√2π) Σ_n (c_n e^{-iπn}) e^{2πi jn/N} at φ_j = -π + 2πj/N
    C = Array{ComplexF64}(V) .* _sign.(-b.ncut:b.ncut)
    return collect(-π .+ 2π .* (0:N-1) ./ N), FFTW.bfft(FFTW.ifftshift(C, 1), 1) ./ sqrt(2π)
end

function phase_wavefunction(b::Union{PeriodicGrid,PhaseGrid}, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "phase_wavefunction")
    return coordinates(b), _dense(V) ./ sqrt(spacing(b))
end

function charge_wavefunction(b::ChargeBasis, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "charge_wavefunction")
    return coordinates(b), _dense(V)
end

function charge_wavefunction(b::PeriodicGrid, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "charge_wavefunction")
    ℓ = _conjugate_index(b.N)
    # c_ℓ = (1/√N) Σ_j v_j e^{-iℓφ_j} with φ_j = -π + 2πj/N
    return collect(Float64, ℓ), _centred_dft(V) .* (_sign.(ℓ) ./ sqrt(b.N))
end

function charge_wavefunction(b::PhaseGrid, V::AbstractVecOrMat{<:Number})
    _check_rows(b, V, "charge_wavefunction")
    Δ = spacing(b)
    n = 2π .* _conjugate_index(b.N) ./ (b.N * Δ)
    # ψ̃(n_k) = √(Δ/2π) e^{-i n_k a} Σ_j v_j e^{-2πi jk/N} with φ_j = a + jΔ
    return collect(n), _centred_dft(V) .* (sqrt(Δ / 2π) .* cis.(-b.a .* n))
end

# The oscillator functions `h_m(x; s) = (2π s²)^{-1/4} (2^m m!)^{-1/2} H_m(x/(√2 s)) e^{-x²/(4s²)}`, `m = 0..N-1`, of a
# coordinate with zero-point amplitude `s`, at the points `x`: a `length(x) × N` matrix, column `m + 1` for level `m`.
#
# `h_m(x; s) = (√2 s)^{-1/2} ψ_m(ξ)` with `ξ = x/(√2 s)` and `ψ_m` the orthonormal Hermite function, which GSL evaluates
# for every level at one point (`gsl_sf_hermite_func_array`) by a rescaled recurrence: a level `m ≳ 745` stays accurate
# where the Gaussian factor alone underflows (`|ξ| ≳ 38.6`), and a value below the `Float64` range comes out as zero.
# GSL returns `NaN` once `ξ²` overflows, so a point with `ξ²/2 ≥ 2^40`, where every level that fits in memory is below
# the range, is set to zero here (`±Inf` included); a `NaN` point gives `NaN`.
function _hermite_functions(x::AbstractVector{<:Real}, N::Int, s::Float64)
    H = zeros(length(x), N)
    c = 1 / sqrt(sqrt(2.0) * s)
    for (i, xi) in enumerate(x)
        ξ = Float64(xi) / (sqrt(2.0) * s)
        isnan(ξ) && (H[i, :] .= NaN; continue)
        ξ^2 / 2 < 0x1p40 || continue
        H[i, :] .= c .* GSL.sf_hermite_func_array(N - 1, ξ)
    end
    return H
end

_minus_i_power(m::Integer) = (ComplexF64(1), ComplexF64(0, -1), ComplexF64(-1), ComplexF64(0, 1))[mod(m, 4)+1]   # (-i)^m

function phase_wavefunction(b::FockBasis, V::AbstractVecOrMat{<:Number}, φ::AbstractVector{<:Real})
    _check_rows(b, V, "phase_wavefunction")
    return collect(Float64, φ), _hermite_functions(φ, b.N, b.phi_zpf) * _dense(V)
end

function charge_wavefunction(b::FockBasis, V::AbstractVecOrMat{<:Number}, n::AbstractVector{<:Real})
    _check_rows(b, V, "charge_wavefunction")
    return collect(Float64, n), _hermite_functions(n, b.N, 1 / (2 * b.phi_zpf)) * (_minus_i_power.(0:b.N-1) .* _dense(V))
end

phase_wavefunction(::FockBasis, ::AbstractVecOrMat{<:Number}) =
    throw(ArgumentError("FockBasis has no grid: pass the points, e.g. phase_wavefunction(b, V, φ)"))
charge_wavefunction(::FockBasis, ::AbstractVecOrMat{<:Number}) =
    throw(ArgumentError("FockBasis has no grid: pass the points, e.g. charge_wavefunction(b, V, n)"))

# One mode's transform of the `d × m` matrix `V` in basis `b`, to `rep` (`:phase` or `:charge`): `(grid, W)`.
_axis_transform(b::Basis, rep::Symbol, V::AbstractMatrix, pts) =
    rep === :phase ? (pts === nothing ? phase_wavefunction(b, V) : phase_wavefunction(b, V, pts)) :
                     (pts === nothing ? charge_wavefunction(b, V) : charge_wavefunction(b, V, pts))

# The states in the columns of `V`, over the product of `bases` (axis 1 fastest), with mode `k` (named `names[k]`) in
# the representation `reps[k]`, each axis transformed by its 1D method: `(axes, ψ)` with `ψ[i₁, …, i_N, state]`.
function _wavefunction(bases::Tuple, names::Tuple, reps::Tuple, V::AbstractMatrix, points)
    X = reshape(_dense(V), map(dimension, bases)..., size(V, 2))
    grids = Any[]
    for k in eachindex(bases)
        perm = (k, (i for i in 1:ndims(X) if i != k)...)
        Xp = permutedims(X, perm)
        g, W = _axis_transform(bases[k], reps[k], reshape(Xp, size(Xp, 1), :), get(points, names[k], nothing))
        push!(grids, g)
        X = permutedims(reshape(W, size(W, 1), size(Xp)[2:end]...), invperm(perm))
    end
    return NamedTuple{names}(Tuple(grids)), X
end

_mode_names(x::Symbol) = (x,)
_mode_names(x) = Tuple(x)

function _node_wavefunction(fname::String, default::Symbol, other, sys::System, node::Node, points, states, pts)
    mn = materialized(sys, node)
    all(ch -> ch isa Basis, mn.children) ||
        throw(ArgumentError("$fname: only leaf nodes (every child a basis) are supported; the node on modes " *
                            "$(modes(node)) has a child that is a node, so pass one of its leaves"))
    bases, names = Tuple(mn.children), Tuple(modes(node))
    if pts !== nothing
        if length(names) != 1
            k = findfirst(b -> b isa FockBasis, bases)
            throw(ArgumentError(k === nothing ?
                "$fname: this leaf has no FockBasis mode, and only a FockBasis takes points; its other modes have their own grids" :
                "$fname: a node with several modes takes its points per FockBasis mode, e.g. points = ($(names[k]) = φs,)"))
        end
        points = NamedTuple{names}((pts,))
    end
    flip = default === :phase ? :charge : :phase
    for nm in (_mode_names(other)..., keys(points)...)
        nm in names || throw(ArgumentError("$fname: :$nm is not a mode of this node (its modes are $(collect(names)))"))
    end
    for (b, nm) in zip(bases, names)
        b isa FockBasis && !haskey(points, nm) &&
            throw(ArgumentError("$fname: mode :$nm is a FockBasis, which has no grid: pass its points, e.g. points = ($nm = φs,)"))
        !(b isa FockBasis) && haskey(points, nm) &&
            throw(ArgumentError("$fname: mode :$nm is a $(nameof(typeof(b))), which has its own grid; points are only for a FockBasis"))
    end
    reps = map(nm -> nm in _mode_names(other) ? flip : default, names)
    V = states isa Colon ? mn.vectors : mn.vectors[:, states isa Integer ? (states:states) : states]
    length(names) == 1 && return _axis_transform(bases[1], reps[1], V, get(points, names[1], nothing))
    return _wavefunction(bases, names, reps, V, points)
end

for (f, default, other) in ((:phase_wavefunction, :phase, :charge), (:charge_wavefunction, :charge, :phase))
    @eval begin
        $f(b::Union{ChargeBasis,PeriodicGrid,PhaseGrid}, ::AbstractVecOrMat{<:Number}, ::AbstractVector{<:Real}) =
            throw(ArgumentError($(string(f)) * ": a $(nameof(typeof(b))) has its own grid; points are only for a FockBasis"))
        $f(sys::System, node::Node, pts::Union{Nothing,AbstractVector{<:Real}}=nothing; $other=(), points=(;), states=Colon()) =
            _node_wavefunction($(string(f)), $(QuoteNode(default)), $other, sys, node, points, states, pts)
        $f(sys::System, pts::Union{Nothing,AbstractVector{<:Real}}=nothing; kw...) = $f(sys, sys.root.node, pts; kw...)
    end
end
