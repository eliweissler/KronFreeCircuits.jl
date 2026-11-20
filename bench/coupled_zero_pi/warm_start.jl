# Zero-pi root LOBPCG: cold (product-state seed) vs warm-started from its own converged eigenvectors (an upper bound
# on what any better seed, e.g. perturbation theory, could save). Same tree and settings as coupled_zero_pi/run.jl.
#   BENCH_DEVICE=gpu julia --project=bench -t 8 --gcthreads=1 bench/coupled_zero_pi/warm_start.jl
include(joinpath(@__DIR__, "..", "common.jl"))
const SPEC = load_spec(joinpath(@__DIR__, "specs", "coupled_zero_pi.json"))
const NCUT, NPHI, TRUNC, NTOTAL, THRESH = 90, 300, 300, 1000, 1e-5
θ = (n_g1=0.0, n_g4=0.0, varphi_ext1=0.0, varphi_ext2=0.0)
qubit(q) = Node([Symbol("theta$q") => ChargeBasis(NCUT), Symbol("phi$q") => PhaseGrid(-6π, 6π, NPHI; order=6)];
                compute=TRUNC, solver=Auto(on=ON), keep=connected([1, 3], (Symbol("theta$q"), nport()), THRESH))
root = Node([qubit(1), qubit(2), :zeta1 => FockBasis(1; phi_zpf=1.0), :zeta2 => FockBasis(1; phi_zpf=1.0)];
            solver=LOBPCG(; tol=1e-8, maxiter=3000, guard=8, on=ON), compute=NTOTAL, keep=lowest(NTOTAL))
rootrec(log) = log[end]
sys, _ = node_timings(() -> materialize(root, SPEC, θ))          # compile
for rep in 1:3
    c, lc = node_timings(() -> materialize(root, SPEC, θ)); r = rootrec(lc)
    @printf("cold  rep %d: dim %d, %d iterations, %d matvecs, solve %.1f s, warm=%s\n", rep, r.dim, r.iterations, r.n_matvec, r.solve_seconds, r.warm)
    w, lw = node_timings(() -> materialize(root, SPEC, θ; previous=c)); s = rootrec(lw)
    @printf("warm  rep %d: dim %d, %d iterations, %d matvecs, solve %.1f s, warm=%s, max|ΔE| %.1e GHz\n", rep, s.dim, s.iterations, s.n_matvec, s.solve_seconds, s.warm, maximum(abs, energies(w) .- energies(c)))
end
