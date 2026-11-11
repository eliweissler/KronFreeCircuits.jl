# The cost of the default retention (`Node(...; retain = :auto)`) against none (`retain = :never`): the forward time of
# `materialize` and the memory of the materialized `System`, on three trees.
#
#   julia --project bench/retention_cost.jl
#
# (a) a transmon leaf on ChargeBasis(400) (801 states), Dense, compute = 10, keep = lowest(10): retaining computes the
#     full decomposition instead of the lowest 10 pairs, and stores the 791 unkept vectors;
# (b) a fluxonium leaf on PhaseGrid(-10π, 10π, 901; order = 8), Dense, compute = 10, keep = lowest(10): the same;
# (c) the KITE tree of test/test_meanfield_kite.jl (`kite_tree`) on test/fixtures/kite/optimum.json at θ = (;).
# Each line prints the minimum of five timings of `materialize` (after one untimed warm-up) and `Base.summarysize` of
# the result.
# Warnings (the KITE's phase-mask warning of its compact leaf) are silenced: they are the same with and without retention.

using KronFreeCircuits, Logging, Printf

const FIXTURES = joinpath(@__DIR__, "..", "test", "fixtures")
const REPEATS = 5

transmon(retain) = (Node([:varphi_1 => ChargeBasis(400)]; solver=Dense(), compute=10, keep=lowest(10), retain),
                    load_spec(joinpath(FIXTURES, "transmon.json")), (E_J=30.0, E_C=1.2, n_g=0.25))
fluxonium(retain) = (Node([:varphi_1 => PhaseGrid(-10π, 10π, 901; order=8)]; solver=Dense(), compute=10, keep=lowest(10),
                          retain),
                     load_spec(joinpath(FIXTURES, "fluxonium.json")), (E_J=8.9, E_C=2.5, E_L=0.5, varphi_ext=2.0))
function kite(retain)
    spec = load_spec(joinpath(FIXTURES, "kite", "optimum.json"))
    leaves = [Node([m => b]; solver=Dense(), compute=3, keep=lowest(3), arithmetic=:complex, retain)
              for (m, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5, when_capped=:allow)]
    return Node(leaves; compute=4, keep=lowest(4), retain), spec, (;)
end

function measure(build, retain)
    root, spec, θ = build(retain)
    sys = materialize(root, spec, θ)                                 # warm-up (compilation)
    best = Inf
    for _ in 1:REPEATS
        GC.gc()
        best = min(best, @elapsed(sys = materialize(root, spec, θ)))
    end
    return best, Base.summarysize(sys)
end

with_logger(NullLogger()) do
    @printf("%-10s %-7s %12s %14s\n", "tree", "retain", "min time (s)", "size (bytes)")
    for (name, build) in (("transmon", transmon), ("fluxonium", fluxonium), ("kite", kite)), retain in (:never, :auto)
        t, sz = measure(build, retain)
        @printf("%-10s %-7s %12.4f %14d\n", name, retain, t, sz)
    end
end
