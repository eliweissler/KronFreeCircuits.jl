# The two Born–Oppenheimer reduced specs committed in this directory, as `fit_surfaces` results.
# Shared by write_reduced.jl (which writes them) and test_bo.jl (which compares the committed files and Task 5's oscillator
# checks against them), so the construction cannot drift from the files.
# Needs the names of test/helpers.jl (`synthetic_surface`, `coupled_oscillators_spec`, `quiet_promote`) and KronFreeCircuits.

# L6 oscillators, extended slow mode through :x2: U(s) = ω/2 + ½ (a - c²/b) s², ω = sqrt(8 E_Cf b).
# Returns `(surface, fit)`; the sweep (21 slow points) is computed once per session.
const _BO_OSCILLATORS = Ref{Any}(nothing)
function bo_oscillators_construction()
    _BO_OSCILLATORS[] === nothing || return _BO_OSCILLATORS[]
    p = quiet_promote(coupled_oscillators_spec(), :s)
    leaf = Node([:f => PhaseGrid(-12, 12, 301)]; solver=Dense(), keep=lowest(1))
    θ = (a=1.0, b=2.0, c=0.6, E_Cf=0.5, E_Cs=0.1)
    S = bo_surface(leaf, p, θ, range(-1, 1; length=21); bands=1)
    return _BO_OSCILLATORS[] = (S, fit_surfaces([S], [:constant => ["1"], :x2 => ["1"]]))
end

# Compact slow mode, Di Paolo's form plus a sine harmonic, with `varphi_ext` fitted over (so it stays a free parameter):
# U(θ; φ) = 1.1 - (0.74 - 4e-4 cos φ) cos 2θ - 0.0011 cos(φ/2) cos θ + 0.0007 sin(φ/2) sin θ, on 32 points of θ at 5 values of φ.
# The surfaces are synthetic (`synthetic_surface`), so the fit recovers the constants to round-off.
function bo_compact_construction()
    U(φ) = θ -> 1.1 - (0.74 - 4e-4cos(φ))cos(2θ) - 0.0011cos(φ / 2)cos(θ) + 0.0007sin(φ / 2)sin(θ)
    surfs = [synthetic_surface(periodic_grid(32), U(φ); θ=(varphi_ext=φ,)) for φ in 2π .* (0:0.125:0.5)]
    return fit_surfaces(surfs, [:constant => ["1"], (:cos, 2) => ["1", "cos(varphi_ext)"],
                                (:cos, 1) => ["cos(varphi_ext/2)"], (:sin, 1) => ["sin(varphi_ext/2)"]])
end

"""
    bo_reduced_fits() -> Vector{Pair{String,BOFit}}

The committed files of this directory (name => fit); `save_spec(path, F.spec; metadata=F.metadata)` writes each.
"""
bo_reduced_fits() = ["oscillators_reduced.json" => bo_oscillators_construction()[2],
                     "compact_reduced.json" => bo_compact_construction()]
