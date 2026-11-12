# Benchmark: two capacitively coupled zero-pi qubits (Eli's two-qubit gate study, arXiv:2609.16630), the same
# hierarchy as its Python pipeline (two_qubit_gate_for_zero_pi_qubit/ham_data.py, scqubits 4.3.1):
#   each qubit's (theta, phi) subsystem in the charge basis (n_cut 90) x a 300-point phi grid over ±6π (scqubits'
#   default 7-point stencil = PhaseGrid order 6), its lowest 300 levels, of which the pipeline keeps the ones
#   connected to levels 0 and 2 through n_theta above 1e-5 (`trunc_by_thresh` = the package's `connected`);
#   the zeta modes are exactly decoupled and left out of the coupled problem (here: one Fock level each);
#   the root over the two qubits for the lowest NTOTAL levels (the pipeline: 3000; here 1000 by default).
#
#   julia --project=bench -t 8 --gcthreads=1 bench/coupled_zero_pi/run.jl
#
# The circuit comes from circuit-quantization (quantize.py -> specs/coupled_zero_pi.json). The reference is the
# pipeline's npz, reduced by reference.py to reference/<name>.json.
#
# Knobs: those of ../common.jl, and
#   ZP_REFERENCE   reference JSON (default reference/two_qubit_data_default.json); absent: no comparison
#   ZP_NTOTAL      levels of the coupled system (default 1000)
#   ZP_TRUNC       levels computed per qubit (default 300)
#   ZP_THRESH      the connected rule's threshold on |n_theta| (default 1e-5)
#   ZP_ROOT        root solver: dense (default) or lobpcg

include(joinpath(@__DIR__, "..", "common.jl"))
import JSON3

const SPEC = load_spec(joinpath(@__DIR__, "specs", "coupled_zero_pi.json"))
const NTOTAL = parse(Int, get(ENV, "ZP_NTOTAL", SMOKE ? "20" : "1000"))
const TRUNC = parse(Int, get(ENV, "ZP_TRUNC", SMOKE ? "40" : "300"))
const THRESH = parse(Float64, get(ENV, "ZP_THRESH", "1e-5"))
const NCUT, NPHI = SMOKE ? (20, 80) : (90, 300)
θ_zero(_=nothing) = (n_g1=0.0, n_g4=0.0, varphi_ext1=0.0, varphi_ext2=0.0)    # the default file: zero flux, n_g = 0
const REC = Recorder("coupled_zero_pi", @__DIR__)
REC.record["environment"]["settings"] = Dict("ntotal" => NTOTAL, "trunc" => TRUNC, "thresh" => THRESH, "ncut" => NCUT, "nphi" => NPHI)

qubit(q; selection=true) = Node([Symbol("theta$q") => ChargeBasis(NCUT), Symbol("phi$q") => PhaseGrid(-6π, 6π, NPHI; order=6)];
                                compute=TRUNC,
                                keep=selection ? connected([1, 3], (Symbol("theta$q"), nport()), THRESH) : lowest(TRUNC))
function root_node(; selection=true)
    rootsolver = get(ENV, "ZP_ROOT", "dense") == "dense" ? Dense() : LOBPCG(; tol=1e-8, maxiter=3000, guard=8, on=ON)
    return Node([qubit(1; selection), qubit(2; selection), :zeta1 => FockBasis(1; phi_zpf=1.0), :zeta2 => FockBasis(1; phi_zpf=1.0)];
                solver=rootsolver, compute=NTOTAL, keep=lowest(NTOTAL))
end

root = root_node()
run = sweep!(REC, "two-level tree, connected leaves, $NTOTAL levels", root, SPEC, [0.0], θ_zero;
             settings=Dict("tree" => "root over two (theta, phi) leaves", "ntotal" => NTOTAL, "trunc" => TRUNC, "thresh" => THRESH))
p = run["points"][end]
# the full spectrum and the leaves, for the comparison (sweep! keeps the lowest NLEVELS)
sys = materialize(root, SPEC, θ_zero())
E = energies(sys)
q1, q2 = materialized(sys, root.children[1]), materialized(sys, root.children[2])
leaf1 = q1.retained.energies; leaf2 = q2.retained.energies                    # every computed level, ascending
kept1 = q1.kept .- 1; kept2 = q2.kept .- 1                                    # 0-based, as the pipeline's hspace_*_charge
@printf("qubit leaves: %d and %d of %d computed levels kept (connected to 0 and 2 through n_theta > %.0e)\n",
        length(kept1), length(kept2), TRUNC, THRESH)
@printf("coupled: %d levels, E1 - E0 = %.9f GHz, E_%d - E0 = %.6f GHz\n", length(E), E[2] - E[1], length(E) - 1, E[end] - E[1])
out = Dict("levels" => E .- E[1], "leaf1" => leaf1, "leaf2" => leaf2, "kept1" => kept1, "kept2" => kept2)
open(io -> JSON3.write(io, out), joinpath(REC.out, "levels.json"), "w")

reffile = get(ENV, "ZP_REFERENCE", joinpath(@__DIR__, "reference", "two_qubit_data_default.json"))
if !SMOKE && isfile(reffile)
    ref = JSON3.read(read(reffile, String))
    eval1, eval2 = Vector{Float64}(ref.eval1), Vector{Float64}(ref.eval2)
    rel(x) = x .- x[1]
    n1 = min(length(eval1), length(leaf1)); n2 = min(length(eval2), length(leaf2))
    d1 = maximum(abs, rel(leaf1)[1:n1] .- rel(eval1)[1:n1]); d2 = maximum(abs, rel(leaf2)[1:n2] .- rel(eval2)[1:n2])
    @printf("qubit 1: %d levels, worst |Δ(E_k - E_0)| = %.2e GHz;  qubit 2: %d levels, %.2e GHz\n", n1, d1, n2, d2)
    same1 = kept1 == Vector{Int}(ref.hspace_1_charge); same2 = kept2 == Vector{Int}(ref.hspace_2_charge)
    @printf("kept sets: qubit 1 %s (%d vs %d), qubit 2 %s (%d vs %d)\n", same1 ? "identical" : "DIFFER", length(kept1),
            length(ref.hspace_1_charge), same2 ? "identical" : "DIFFER", length(kept2), length(ref.hspace_2_charge))
    evt = Vector{Float64}(ref.evals_tot)
    n = min(length(evt), length(E))
    dt = abs.((E[1:n] .- E[1]) .- (evt[1:n] .- evt[1]))
    @printf("coupled: %d levels, worst |Δ(E_k - E_0)| = %.2e GHz (median %.2e); first ten: %s\n", n, maximum(dt),
            sort(dt)[cld(n, 2)], join((@sprintf("%.1e", x) for x in dt[1:10]), " "))
    REC.record["comparison"] = Dict("reference" => reffile, "leaf1_worst" => d1, "leaf2_worst" => d2,
                                    "kept_identical" => [same1, same2], "coupled_levels" => n, "coupled_worst" => maximum(dt),
                                    "coupled_deviation" => dt)
    save(REC)
end
println("results: $(REC.out)/results.json")
