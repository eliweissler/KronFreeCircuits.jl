# `KronFreeCircuits.Experimental.dmrg_levels`: the lowest levels of an MPO by ITensorMPS's DMRG, ground state first, then
# each excited state with the earlier ones as overlap penalties. A thin wrapper: the MPO and its sites are the caller's.
module KronFreeCircuitsITensorMPSExt

using KronFreeCircuits
using ITensorMPS: ITensorMPS, MPO, MPS, Sweeps, dmrg, inner, random_mps, siteinds
import KronFreeCircuits.Experimental: dmrg_levels

# The start of level `k` from `init`: nothing (random), an MPS, state names, or a function of `k` returning one of these.
function _start(init, k::Int, H::MPO)
    s = init isa Function ? init(k) : init
    s === nothing && return random_mps(siteinds(first, H; plev=0); linkdims=10)
    s isa MPS && return s
    return MPS(siteinds(first, H; plev=0), s)
end

function dmrg_levels(H::MPO; nlevels::Integer=1, weight::Union{Nothing,Real}=nothing, init=nothing,
                     sweeps::Union{Nothing,Sweeps}=nothing, observer=nothing, kwargs...)
    observer === nothing || observer isa Function || throw(ArgumentError(
        "observer must be a function k -> observer (got a $(typeof(observer))): an observer keeps the energies it has " *
        "seen, and one shared by the levels would stop the excited solves early"))
    nlevels >= 1 || throw(ArgumentError("nlevels must be at least 1, got $nlevels"))
    if nlevels > 1
        weight === nothing && throw(ArgumentError(
            "dmrg_levels needs `weight` for nlevels > 1: the penalty on overlapping the states already found, larger " *
            "than the gaps to the states it should not fall back to and on the scale of H (see its docstring)"))
        weight > 0 && isfinite(weight) || throw(ArgumentError("weight must be positive and finite, got $weight"))
    end
    states, observers = MPS[], Any[]
    for k in 1:nlevels
        ψ0 = _start(init, k, H)
        obs = observer === nothing ? nothing : observer(k)
        push!(observers, obs)
        kw = obs === nothing ? kwargs : (; kwargs..., observer=obs)
        sw = sweeps === nothing ? () : (sweeps,)
        _, ψ = k == 1 ? dmrg(H, ψ0, sw...; kw...) : dmrg(H, states, ψ0, sw...; weight, kw...)
        push!(states, ψ)
    end
    # energies and variances of the states themselves (dmrg's returned energy at k > 1 includes the penalty)
    norms = [real(inner(ψ, ψ)) for ψ in states]
    energies = [real(inner(ψ', H, ψ)) / n for (ψ, n) in zip(states, norms)]
    variances = [real(inner(H, ψ, H, ψ)) / n - E^2 for (ψ, n, E) in zip(states, norms, energies)]
    overlaps = [i == j ? 0.0 : abs(inner(states[i], states[j])) / sqrt(norms[i] * norms[j]) for i in 1:nlevels, j in 1:nlevels]
    return (; energies, states, variances, overlaps, observers)
end

end # module
