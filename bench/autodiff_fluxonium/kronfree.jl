# Jacobian and Hessian of the enumeration's optimized fluxonium's quantities with KronFreeCircuits' perturbative
# derivatives, on the circuit-quantization spec specs/fluxonium_sqcircuit.json (quantize.py), at the optimum.
#
#   julia --project=bench -t 8 --gcthreads=1 bench/autodiff_fluxonium/kronfree.jl     (writes results/kronfree.json)
#
# Parameters: E_C, E_J, E_L (GHz) and the loop flux f (units of Φ0; the spec's varphi_ext = 2π f).
# Quantities: ω01 = E1 − E0, α = E2 − 2E1 + E0, and |⟨0|O|1⟩|² for O = n, φ of the mode varphi = φ2 − φ1.
# One node: varphi on PhaseGrid(GRID), Dense, 6 levels.

using KronFreeCircuits, LinearAlgebra, Printf, Logging, Statistics
import JSON3

const W = Ref(0)
struct CountEdge <: AbstractLogger; inner::AbstractLogger; end
Logging.min_enabled_level(l::CountEdge) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::CountEdge, a...) = Logging.shouldlog(l.inner, a...)
Logging.handle_message(l::CountEdge, lv, msg, a...; kw...) =
    (startswith(string(msg), "PhaseGrid on mode") && (W[] += 1); Logging.handle_message(l.inner, lv, msg, a...; kw...))
global_logger(CountEdge(ConsoleLogger(stderr)))

const SPEC = load_spec(joinpath(@__DIR__, "specs", "fluxonium_sqcircuit.json"))
const P0 = (E_C=2.63, E_J=2.85, E_L=0.051, f=0.0)
const PARAMS = (:E_C, :E_J, :E_L, :f)
θof(p) = (E_C=p.E_C, E_J=p.E_J, E_L=p.E_L, varphi_ext=2π * p.f)
const GRID = (parse(Float64, get(ENV, "HALFWIDTH_PI", "16")), parse(Int, get(ENV, "NPTS", "513")))
specname(x) = x === :f ? :varphi_ext : x
fac(x) = x === :f ? 2π : 1.0                                      # d/df = 2π d/dvarphi_ext
const PORTS = [(:varphi, nport()), (:varphi, phiport())]
const QNAMES = ["omega01", "alpha", "|n_01|^2", "|phi_01|^2"]
const NQ, NP = length(QNAMES), length(PARAMS)

function tree()
    w, n = GRID
    return Node([:varphi => PhaseGrid(-w * π, w * π, n; order=8)]; solver=Dense(), compute=6, keep=lowest(6), ports=PORTS, retain=:always)
end

quantities(E, P) = [E[2] - E[1], E[3] - 2E[2] + E[1], (abs2(M[1, 2]) for M in P)...]

function solve(root, p)
    sys = materialize(root, SPEC, θof(p))
    return sys, quantities(energies(sys), [port(sys, m, q) for (m, q) in PORTS])
end

"Jacobian (NQ × NP) and Hessians (NQ × NP × NP) from `derivatives(sys; order = 2)` with every mixed pair."
function jac_hess(sys)
    wrt = Tuple(specname.(PARAMS))
    mixed = Tuple((wrt[i], wrt[j]) for i in 1:NP for j in i+1:NP)
    d = derivatives(sys; wrt, order=2, mixed)
    E = energies(sys)
    P = [port(sys, m, q) for (m, q) in PORTS]
    dE(x) = energies(d, specname(x)) .* fac(x)
    d2E(x, y) = energies(d, specname(x), specname(y)) .* (fac(x) * fac(y))
    dP(x, k) = port(d, specname(x), PORTS[k]...) .* fac(x)
    d2P(x, y, k) = port(d, specname(x), specname(y), PORTS[k]...) .* (fac(x) * fac(y))
    J = zeros(NQ, NP); H = zeros(NQ, NP, NP)
    lin(e) = [e[2] - e[1], e[3] - 2e[2] + e[1]]
    for (a, x) in enumerate(PARAMS)
        J[1:2, a] = lin(dE(x))
        for k in 1:length(PORTS)
            J[2+k, a] = 2real(conj(P[k][1, 2]) * dP(x, k)[1, 2])
        end
        for (b, y) in enumerate(PARAMS)
            b < a && continue
            H[1:2, a, b] = lin(d2E(x, y))
            for k in 1:length(PORTS)
                H[2+k, a, b] = 2real(conj(dP(x, k)[1, 2]) * dP(y, k)[1, 2] + conj(P[k][1, 2]) * d2P(x, y, k)[1, 2])
            end
            H[:, b, a] = H[:, a, b]
        end
    end
    return J, H
end

root = tree()
sys, q0 = solve(root, P0)                                          # compile
# timings: mean and standard deviation of 10 trials after a compiling run
trials(f, n=10) = (ts = [@elapsed(f()) for _ in 1:n]; (mean(ts), std(ts)))
t_mat = trials(() -> solve(root, P0))
J, H = jac_hess(sys)                                               # compile
t_der = trials(() -> jac_hess(sys))
derivatives(sys; wrt=Tuple(specname.(PARAMS)))                     # compile first order alone
t_jac = trials(() -> derivatives(sys; wrt=Tuple(specname.(PARAMS))))
@printf("range warnings: %d\nmaterialize %.4f ± %.4f s, first derivatives alone %.4f ± %.4f s, derivatives (first and second, all mixed pairs) %.4f ± %.4f s (10 trials)\n", W[], t_mat..., t_jac..., t_der...)
for (i, n) in enumerate(QNAMES)
    @printf("%-12s = %+.9e   J: %s\n", n, q0[i], join((@sprintf("%+.4e", x) for x in J[i, :]), " "))
end

# central finite differences of materialize for the Jacobian (step 1e-4 relative, 1e-4 Φ0 for the flux)
step(x) = x === :f ? 1e-4 : 1e-4 * getfield(P0, x)
Jfd = zeros(NQ, NP)
for (a, x) in enumerate(PARAMS)
    h = step(x)
    qp = solve(root, merge(P0, NamedTuple{(x,)}((getfield(P0, x) + h,))))[2]
    qm = solve(root, merge(P0, NamedTuple{(x,)}((getfield(P0, x) - h,))))[2]
    Jfd[:, a] = (qp .- qm) ./ 2h
end
# each entry against the largest entry of its row (the flux column vanishes at f = 0 by symmetry: levels are even in f)
rel = maximum(abs.(J .- Jfd) ./ maximum(abs, Jfd; dims=2))
@printf("Jacobian vs central differences: max deviation %.1e of each row's largest entry\n", rel)
# the Hessian against central differences of the derivative Jacobian
Hfd = zeros(NQ, NP, NP)
for (a, x) in enumerate(PARAMS)
    h = step(x)
    Jp = jac_hess(solve(root, merge(P0, NamedTuple{(x,)}((getfield(P0, x) + h,))))[1])[1]
    Jm = jac_hess(solve(root, merge(P0, NamedTuple{(x,)}((getfield(P0, x) - h,))))[1])[1]
    Hfd[:, :, a] = (Jp .- Jm) ./ 2h
end
hdev = maximum(maximum(abs, H[i, :, :] .- Hfd[i, :, :]) / maximum(abs, Hfd[i, :, :]) for i in 1:NQ)
@printf("Hessian vs central differences of the Jacobian: max deviation %.1e of each Hessian's largest entry\n", hdev)
mkpath(joinpath(@__DIR__, "results"))
open(io -> JSON3.pretty(io, JSON3.write(Dict("params" => string.(PARAMS), "quantities" => QNAMES, "values" => q0,
                                             "jacobian" => J, "hessian" => [H[i, :, :] for i in 1:NQ], "jacobian_fd" => Jfd, "hessian_fd" => [Hfd[i, :, :] for i in 1:NQ],
                                             "seconds_materialize_mean_std" => collect(t_mat), "seconds_derivatives_mean_std" => collect(t_der), "seconds_first_derivatives_mean_std" => collect(t_jac), "timing_trials" => 10, "threads" => Threads.nthreads(),
                                             "range_warnings" => W[], "grid" => collect(GRID)))),
     joinpath(@__DIR__, "results", "kronfree.json"), "w")
