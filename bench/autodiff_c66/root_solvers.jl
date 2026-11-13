# The c66 tree of kronfree.jl with its root (2400 states, 6 kept) solved by each root solver, CPU and GPU: time to
# materialize, to the Jacobian (first derivatives, 5 parameters) and to the full Hessian (order 2, 10 mixed pairs), and
# the largest difference of every result from the Dense root's, relative to the largest entry (second derivatives of
# the ports of the near-degenerate top pair, 1.987 and 1.998 GHz, reach ~3e4).
#
#   julia --project=bench -t 8 --gcthreads=1 bench/autodiff_c66/root_solvers.jl

using KronFreeCircuits, LinearAlgebra, Printf, Statistics, CUDA, cuTENSOR

const SPEC = load_spec(joinpath(@__DIR__, "specs", "c66_sqcircuit.json"))
const θ = (E_J=28.83436338, E_L13=0.10343485, E_C23=1.36334654, E_L23=0.05167763, varphi_ext=0.0)
const WRT = (:E_J, :E_L13, :E_C23, :E_L23, :varphi_ext)
const MIXED = Tuple((WRT[i], WRT[j]) for i in 1:5 for j in i+1:5)
const PORTS = [(:varphi1, nport()), (:varphi2, nport()), (:varphi1, phiport()), (:varphi2, phiport())]

function tree(rootsolver)
    EC = real(SPEC.E_C[1, 1](θ)); EL = real(SPEC.E_L[1, 1](θ))
    l1 = Node([:varphi1 => FockBasis(60; phi_zpf=(8EC / EL)^(1 / 4) / sqrt(2))]; solver=Dense(), compute=40, keep=lowest(40), retain=:always)
    l2 = Node([:varphi2 => PhaseGrid(-24π, 24π, 1153; order=8)]; solver=Dense(), compute=60, keep=lowest(60), retain=:always)
    return Node([l1, l2]; solver=rootsolver, compute=6, keep=lowest(6), ports=PORTS, retain=:always)
end

# every number the benchmark reads: energies, ports, and their first and second derivatives
function results(sys, d1, d2)
    out = [energies(sys); (vec(port(sys, m, q)) for (m, q) in PORTS)...]
    for x in WRT
        append!(out, energies(d1, x)); foreach(((m, q),) -> append!(out, vec(port(d1, x, m, q))), PORTS)
        append!(out, energies(d2, x, x)); foreach(((m, q),) -> append!(out, vec(port(d2, x, x, m, q))), PORTS)
    end
    for (x, y) in MIXED
        append!(out, energies(d2, x, y)); foreach(((m, q),) -> append!(out, vec(port(d2, x, y, m, q))), PORTS)
    end
    return out
end
# ports carry an arbitrary phase per state: compare magnitudes
absdiff(a, b) = maximum(abs.(abs.(a) .- abs.(b))) / maximum(abs, b)   # relative to the largest entry

# mean and standard deviation of n trials after a compiling run
trials(f, n=10) = (f(); ts = [@elapsed(f()) for _ in 1:n]; (mean(ts), std(ts)))
function run(name, s, ref)
    root = tree(s)
    sys = materialize(root, SPEC, θ)
    t_mat = trials(() -> materialize(root, SPEC, θ))
    t_jac = trials(() -> derivatives(sys; wrt=WRT))
    t_hes = trials(() -> derivatives(sys; wrt=WRT, order=2, mixed=MIXED))
    r = results(sys, derivatives(sys; wrt=WRT), derivatives(sys; wrt=WRT, order=2, mixed=MIXED))
    @printf("%-15s materialize %6.3f ± %.3f s   Jacobian %6.3f ± %.3f s   Hessian %6.3f ± %.3f s   max rel. diff vs Dense %.1e\n",
            name, t_mat..., t_jac..., t_hes..., ref === nothing ? 0.0 : absdiff(r, ref))
    return r
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("threads: ", Threads.nthreads(), "; GPU: ", CUDA.name(CUDA.device()))
    ref = run("Dense (CPU)", Dense(), nothing)
    run("Lanczos (CPU)", Lanczos(tol=1e-11), ref)
    run("Lanczos (GPU)", Lanczos(tol=1e-11, on=CuArray), ref)
    run("LOBPCG (CPU)", LOBPCG(tol=1e-9), ref)
    run("LOBPCG (GPU)", LOBPCG(tol=1e-9, on=CuArray), ref)
end
