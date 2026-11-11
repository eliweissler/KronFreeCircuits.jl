# How fast the sum over states of the first-order rule converges (design §4.5, Decision 2).
# For each circuit and parameter, the state derivatives are recomputed with the resolvent's other states cut to the nearest
# `M` above the kept ones, and the largest relative error (to the largest element of the full result) of the state
# derivatives and of each port derivative against the full sum is printed.
# Run: $HOME/.juliaup/bin/julia --project bench/derivative_sums.jl
using Test, LinearAlgebra, KronFreeCircuits
include(joinpath(@__DIR__, "..", "test", "helpers.jl"))

const K = KronFreeCircuits

# `res` with its other states cut to the `M` lowest (all of them when there are fewer).
function cut(res::K.Resolvent, M::Int)
    order = sortperm(res.other_energies)[1:min(M, length(res.other_energies))]
    return K.Resolvent(res.energies, res.vectors, res.other_energies[order], res.other_vectors[:, order], res.complement,
                       res.degenerate_tol)
end

relerr(a, b) = maximum(abs, a .- b) / maximum(abs, b)

for (name, c, x) in (("fluxonium", dc_fluxonium(), :varphi_ext), ("fluxonium", dc_fluxonium(), :E_J),
                     ("transmon", dc_transmon(), :n_g))
    sys = dc_solve(c)
    setup = node_setup(sys)
    full = node_first_order(sys, x; setup)
    H1 = K._derivative_operator(setup.mn, setup.terms, sys.θ, x, setup.none; constant=sys.constant)
    portkeys = sort!(collect(keys(full.ports)); by=string)
    println("\n$name (dimension $(prod(setup.mn.dims)), $(length(setup.mn.energies)) kept, ",
            "$(length(setup.res.other_energies)) other states) wrt $x")
    println(rpad("M", 6), rpad("states", 14), join((rpad(K._portname(p), 14) for (_, p) in portkeys)))
    for M in (0, 5, 10, 20, 40, 80, 160, typemax(Int))
        d = K._node_first_order(setup.mn, H1, cut(setup.res, M), setup.none)
        errs = [relerr(d.vectors, full.vectors); [relerr(d.ports[k], full.ports[k]) for k in portkeys]]
        println(rpad(M == typemax(Int) ? "all" : string(M), 6), join((rpad(string(round(e; sigdigits=2)), 14) for e in errs)))
    end
end
