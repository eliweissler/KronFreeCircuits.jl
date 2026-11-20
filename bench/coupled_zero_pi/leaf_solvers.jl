# Zero-pi leaves on the CPU: ShiftInvert (sparse factor) vs ChebyshevFilter (matrix-free), same tree as run.jl.
#   BENCH_DEVICE=cpu julia --project=bench -t 8 --gcthreads=1 bench/coupled_zero_pi/leaf_solvers.jl
include(joinpath(@__DIR__, "..", "common.jl"))
const SPEC = load_spec(joinpath(@__DIR__, "specs", "coupled_zero_pi.json"))
const NCUT, NPHI, TRUNC, NTOTAL, THRESH = 90, 300, 300, 1000, 1e-5
θ = (n_g1=0.0, n_g4=0.0, varphi_ext1=0.0, varphi_ext2=0.0)
qubit(q, s) = Node([Symbol("theta$q") => ChargeBasis(NCUT), Symbol("phi$q") => PhaseGrid(-6π, 6π, NPHI; order=6)];
                   compute=TRUNC, solver=s, keep=connected([1, 3], (Symbol("theta$q"), nport()), THRESH))
tree(s) = Node([qubit(1, s), qubit(2, s), :zeta1 => FockBasis(1; phi_zpf=1.0), :zeta2 => FockBasis(1; phi_zpf=1.0)];
               solver=LOBPCG(; tol=1e-8, maxiter=3000, guard=8), compute=NTOTAL, keep=lowest(NTOTAL))
println("BLAS threads ", BLAS.get_num_threads(), ", Julia threads ", Threads.nthreads())
ref = nothing
for (name, s) in (("ShiftInvert", ShiftInvert()), ("ChebyshevFilter", ChebyshevFilter()))
    r = tree(s); node_timings(() -> materialize(r, SPEC, θ))     # compile
    for rep in 1:2
        t = @elapsed ((sys, log) = node_timings(() -> materialize(r, SPEC, θ)))
        lv = [l for l in log if length(l.modes) == 2]
        E = energies(sys); global ref = ref === nothing ? E : ref
        @printf("%-15s rep %d: leaves %s s (dim %d, %d pairs), root %.1f s, total %.1f s, max|ΔE| vs ShiftInvert %.1e GHz\n",
                name, rep, join((@sprintf("%.1f", l.solve_seconds) for l in lv), ", "), lv[1].dim, lv[1].compute, log[end].solve_seconds, t,
                maximum(abs, (E .- E[1]) .- (ref .- ref[1])))
    end
end
