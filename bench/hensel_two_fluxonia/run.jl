# Benchmark: two fluxonia with a tunable coupler (Hensel et al., arXiv:2611.05627v1 Sec. IV), the suite's first case.
#
#   julia --project=bench -t 8 bench/hensel_two_fluxonia/run.jl [stage ...]
#
# Stages (default: ladder reference):
#   ladder     the headline: the pairwise binary tree in local coordinates with the stiff frame modes f2, f3 sheared
#              onto the junctions (bench_archive/hensel_two_fluxonia, "binary"), on a ladder of kept counts; the
#              accuracy rule of bench/README.md picks the accurate rung at 1 MHz (the paper's class) and at 10 kHz
#   reference  the converged answer of the 2026-11-06 study: the flat local tree at kc 80 with 40-level coupler bases
#              (5 kHz from its neighbours there); the ladder's rungs are compared with it
#   paper      the paper's own truncated space (14 Fock levels on each of the six normal modes, 7.5 M states, their
#              junction construction), solved flat by LOBPCG: the same-machine counterpart of their exact-HD Lanczos.
#              Expensive: 2 to 6 min per point on the A4500, 24 to 43 min on 8 CPU cores.
#
# Knobs: those of ../common.jl, and HENSEL_TOL (LOBPCG tolerance, GHz, default 1e-6), HENSEL_FLUXES (comma list of
# Φ_c/Φ_0, default the paper's grid 0, 0.1, …, 0.5), HENSEL_RUNGS (comma list of ladder rungs to run, 1-based: one
# rung alone in a fresh process gives its own peak RSS).

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "circuit.jl"))

const STAGES = isempty(ARGS) ? ["ladder", "reference"] : ARGS
const FLUXES = haskey(ENV, "HENSEL_FLUXES") ? parse.(Float64, split(ENV["HENSEL_FLUXES"], ",")) :
               SMOKE ? [0.3, 0.4] : collect(0.0:0.1:0.5)
const REC = Recorder("hensel_two_fluxonia", @__DIR__)
REC.record["environment"]["fluxes"] = FLUXES
REC.record["environment"]["tol"] = TOL

# Published costs, arXiv:2611.05627v1 Table tab:cost (sec04 ll. 118-135, numbers.tex ll. 79-89): wall times summed over
# grid points 2 to 6 (point 1 carries their compilation), on one 16-core Xeon 6517P node, peak memory net of baseline.
const PUBLISHED = [("exact HD, Lanczos, m = 14 (7.5 M states)", 68 * 60 / 5, 16.0),
                   ("TN, bond cap 8", 12 * 60 / 5, 0.66),
                   ("TN, bond cap 64", 87 * 60 / 5, 2.1)]
REC.record["published"] = [Dict("solver" => s, "seconds_per_point" => t, "peak_GB" => m,
                                "source" => "arXiv:2611.05627v1 Table tab:cost, sum over grid points 2-6 / 5") for (s, t, m) in PUBLISHED]

const GRID = SMOKE ? PhaseGrid(-3π, 3π, 121; order=8) : PhaseGrid(-3π, 3π, 181; order=8)
gridname(g) = "PhaseGrid($(g.a / π)π, $(g.b / π)π, $(g.N); order=$(g.order))"

runs = Dict{String,Any}()

if "reference" in STAGES
    s = SMOKE ? (kq=8, nc=12, nf1=12, kc=10, kf2=2, kf3=3) : (kq=24, nc=40, nf1=40, kc=80, kf2=4, kf3=8)
    root = local_root(; grid=GRID, s..., compute=NLEVELS)
    runs["reference"] = sweep!(REC, "reference: flat local, kc $(s.kc)", root, local_terms(), FLUXES, θ_at;
                               settings=Dict("tree" => "flat local", "grid" => gridname(GRID), (string(k) => v for (k, v) in pairs(s))...))
end

if "ladder" in STAGES
    B = sheared_coordinates((:f2, :f3))
    terms = local_terms(B)
    nfock = SMOKE ? Dict(:c => 12, :f1 => 12, :f2 => 8, :f3 => 8) : Dict(:c => 30, :f1 => 30, :f2 => 20, :f3 => 20)
    rungs = SMOKE ? [(8, 6, 10, 20), (8, 6, 10, 30), (10, 8, 12, 30)] :
            [(40, 20, 60, 120), (40, 20, 60, 240), (60, 30, 100, 240), (60, 30, 100, 480)]
    haskey(ENV, "HENSEL_RUNGS") && (rungs = rungs[parse.(Int, split(ENV["HENSEL_RUNGS"], ","))])   # e.g. "1" for one rung alone
    ladder = Any[]
    for (k1q, k1f, k2, kM) in rungs
        root = binary_root(; grid=GRID, nfock, k1q, k1f, k2, kM, compute=NLEVELS, B)
        push!(ladder, sweep!(REC, "binary sheared k1 $k1q/$k1f k2 $k2 kM $kM", root, terms, FLUXES, θ_at;
                             settings=Dict("tree" => "binary, f2 f3 sheared", "grid" => gridname(GRID),
                                           "nfock" => Dict(string(a) => b for (a, b) in nfock),
                                           "k1q" => k1q, "k1f" => k1f, "k2" => k2, "kM" => kM)))
    end
    ref = get(runs, "reference", nothing)
    for target in (1e-3, 1e-5)
        ladder_summary!(REC, "binary sheared", ladder; target, reference=ref)
    end
end

if "paper" in STAGES
    m = SMOKE ? 4 : 14
    # The space is complex (the GivenSystem exponentials have general character): 120 MB per vector at m = 14; on the
    # 20 GB A4500 a block of six columns with no guard fits (2026-11-06), on the host the default guard is kept.
    guard = DEVICE == "gpu" ? 0 : 4
    compute = DEVICE == "gpu" ? 6 : NLEVELS
    root = Node(paper_children(m); solver=root_solver(; guard), compute, keep=lowest(compute))
    runs["paper"] = sweep!(REC, "paper's space, m = $m", root, paper_terms(), FLUXES, θ_at;
                           settings=Dict("tree" => "flat, normal-mode Fock (paper's construction)", "m" => m,
                                         "compute" => compute, "guard" => guard))
end

println("\npublished (arXiv:2611.05627v1 Table tab:cost, 16-core Xeon 6517P), per grid point:")
for (s, t, m) in PUBLISHED
    @printf("   %-45s %7.1f s  %5.2f GB\n", s, t, m)
end
println("results: $(REC.out)/results.json")
