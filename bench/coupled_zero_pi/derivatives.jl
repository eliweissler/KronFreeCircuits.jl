# Derivatives of the coupled zero-pi levels along an offset charge and a loop flux: a benchmark of the linear solves on
# the complement (Sternheimer) at large iterative nodes, the 54 300-state (θ, φ) leaves and the 23 870-state root.
# Same tree as run.jl (root by LOBPCG); ZP_NTOTAL levels at the root (default 200).
#
#   BENCH_DEVICE=cpu|gpu julia --project=bench -t 8 --gcthreads=1 bench/coupled_zero_pi/derivatives.jl
#
# Writes ZP_DERIV_OUT (default results/derivatives-<device>.json): times, the per-node solve records and the root's
# energy derivatives.

include(joinpath(@__DIR__, "..", "common.jl"))
import JSON3

const SPEC = load_spec(joinpath(@__DIR__, "specs", "coupled_zero_pi.json"))
const NTOTAL = parse(Int, get(ENV, "ZP_NTOTAL", "200"))
const TRUNC, THRESH, NCUT, NPHI = 300, 1e-5, 90, 300
const WRT = (:n_g1, :varphi_ext1)
θ = (n_g1=0.0, n_g4=0.0, varphi_ext1=0.0, varphi_ext2=0.0)
leafsolver = DEVICE == "gpu" ? Auto(on=ON) : Auto()
qubit(q) = Node([Symbol("theta$q") => ChargeBasis(NCUT), Symbol("phi$q") => PhaseGrid(-6π, 6π, NPHI; order=6)];
                compute=TRUNC, solver=leafsolver, keep=connected([1, 3], (Symbol("theta$q"), nport()), THRESH), retain=:always)
root = Node([qubit(1), qubit(2), :zeta1 => FockBasis(1; phi_zpf=1.0), :zeta2 => FockBasis(1; phi_zpf=1.0)];
            solver=LOBPCG(; tol=1e-8, maxiter=3000, guard=8, on=ON), compute=NTOTAL, keep=lowest(NTOTAL), retain=:always)

println("device $DEVICE, levels $NTOTAL, threads $(Threads.nthreads())")
t_mat = @elapsed sys = materialize(root, SPEC, θ)
t_first = @elapsed d = derivatives(sys; wrt=WRT)                                  # includes compilation
times = [@elapsed(d = derivatives(sys; wrt=WRT)) for _ in 1:2]
solves = [(; modes=string.(r.modes), r.columns, r.iterations, r.residual, n_from_start=length(r.from_start),
           n_at_floor=length(r.at_floor)) for r in d.linear_solves]
@printf("materialize %.1f s; derivatives first call %.1f s, then %s s\n", t_mat, t_first, join((@sprintf("%.1f", t) for t in times), ", "))
for s in solves
    @printf("  solve on %-28s %5d columns, %5d iterations, residual %.1e, from start %d, at floor %d\n",
            join(s.modes, ","), s.columns, s.iterations, s.residual, s.n_from_start, s.n_at_floor)
end
dE = Dict(string(x) => energies(d, x) for x in WRT)
out = get(ENV, "ZP_DERIV_OUT", joinpath(@__DIR__, "results", "derivatives-$DEVICE.json"))
mkpath(dirname(out))
open(io -> JSON3.write(io, Dict("device" => DEVICE, "levels" => NTOTAL, "materialize_s" => t_mat,
                                 "derivatives_first_s" => t_first, "derivatives_s" => times, "solves" => solves, "dE" => dE)), out, "w")
println("wrote $out")
