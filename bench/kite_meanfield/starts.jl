# The 33 starts of the self-consistent loop of test/test_meanfield_kite.jl, shared by levels.jl and optimize.jl: the
# compact mode at 0 or π, each extended mode at 0, π/2, π or -π/2, and the delocalized start.
function kite_starts(spec)
    names = [m.name for m in spec.modes]
    ext = (0.0, π / 2, π, -π / 2)
    return Any[[Dict{Symbol,Float64}(zip(names, (a, b, c))) for a in (0.0, π) for b in ext for c in ext]; nothing]
end
