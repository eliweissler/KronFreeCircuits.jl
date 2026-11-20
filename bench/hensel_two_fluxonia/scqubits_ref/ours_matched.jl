# KronFreeCircuits' side of the matched scqubits comparison: the binary tree in HENSEL_FRAME at the settings of
# hensel_scq.binary_tree, every point cold (no `previous`), with HENSEL_PRECOND = jacobi (scqubits' scipy LOBPCG gets the
# same shifted Jacobi) or ritz (the package default). One untimed call compiles; then the median of three per point.
#
#   HENSEL_PRECOND=jacobi HENSEL_FRAME=shear julia --project=bench -t 8 bench/hensel_two_fluxonia/scqubits_ref/ours_matched.jl [kM]
#
# HENSEL_FRAME: shear (default; f2, f3 sheared onto the junctions, the benchmark's frame) or local.

include(joinpath(@__DIR__, "..", "..", "common.jl"))
include(joinpath(@__DIR__, "..", "circuit.jl"))
using Logging
global_logger(ConsoleLogger(stderr, Logging.Error))

const KM = isempty(ARGS) ? 240 : parse(Int, ARGS[1])
const FRAME = get(ENV, "HENSEL_FRAME", "shear")
const B = FRAME == "shear" ? sheared_coordinates((:f2, :f3)) : LC
grid = PhaseGrid(-3π, 3π, 181; order=8)
nfock = Dict(:c => 30, :f1 => 30, :f2 => 20, :f3 => 20)
root = binary_root(; grid, nfock, k1q=40, k1f=20, k2=60, kM=KM, compute=8, B)
terms = local_terms(B)
materialize(root, terms, θ_at(0.3))                     # compile
points = Any[]
for fc in 0.0:0.1:0.5
    _, r = timed_point(() -> materialize(root, terms, θ_at(fc)); repeats=3)
    E = energies(materialize(root, terms, θ_at(fc)))[1:8]
    bynode = Dict(join(string.(n["modes"]), ",") => n["operator_seconds"] + n["solve_seconds"] + n["assemble_seconds"] for n in r["nodes"])
    @printf("Φc = %.1f: %6.3f s (first %.3f)  E1-E0 = %.6f  M %.3f s, root %.3f s\n", fc, r["seconds"], r["first_seconds"],
            E[2] - E[1], bynode["c,f1,q1,f3"], bynode["c,f1,q1,f3,q2,f2"])
    r["flux"] = fc; r["energies"] = E
    push!(points, r)
end
open(joinpath(@__DIR__, "matched-ours-$(FRAME)-kM$(KM)-$(PRECOND).json"), "w") do io
    JSON3.write(io, Dict("frame" => FRAME, "kM" => KM, "preconditioner" => string(PRECOND), "environment" => environment("scqubits_ref"),
                         "points" => points))
end
