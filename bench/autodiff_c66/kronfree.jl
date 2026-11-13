# Jacobian and Hessian of the n3_g1_c66 qubit's quantities with KronFreeCircuits' perturbative derivatives, on the
# circuit-quantization spec specs/c66_sqcircuit.json (quantize.py), at the enumeration optimum.
#
#   julia --project=bench -t 8 --gcthreads=1 bench/autodiff_c66/kronfree.jl     (writes results/kronfree.json)
#
# Parameters: E_J, E_L13, E_C23, E_L23 (GHz) and the loop flux f (units of Φ0; the spec's varphi_ext = 2π f).
# Quantities: ω01 = E1 − E0, α = E2 − 2E1 + E0, and |⟨0|O|1⟩|² for O = n1, n2, φ1, φ2, the charge and phase of the
# appendix's modes varphi1 = (φ3 − φ2)/2 (harmonic) and varphi2 = φ2 − φ1 (the junction's).
# Tree (no range warning, converged to ~1e-9 GHz in ω01 and α, 2026-11-12): a leaf per mode, varphi1 on 60 Fock levels
# keeping 40, varphi2 on PhaseGrid(±24π, 1153 points, order 8) keeping 60, the root by Lanczos.

using KronFreeCircuits, LinearAlgebra, Printf, Logging, Statistics
import JSON3

const W = Ref(0)
struct CountEdge <: AbstractLogger; inner::AbstractLogger; end
Logging.min_enabled_level(l::CountEdge) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::CountEdge, a...) = Logging.shouldlog(l.inner, a...)
Logging.handle_message(l::CountEdge, lv, msg, a...; kw...) =
    (startswith(string(msg), "PhaseGrid on mode") && (W[] += 1); Logging.handle_message(l.inner, lv, msg, a...; kw...))
global_logger(CountEdge(ConsoleLogger(stderr)))

const SPEC = load_spec(joinpath(@__DIR__, "specs", "c66_sqcircuit.json"))
const P0 = (E_J=28.83436338, E_L13=0.10343485, E_C23=1.36334654, E_L23=0.05167763, f=0.0)
const PARAMS = (:E_J, :E_L13, :E_C23, :E_L23, :f)
θof(p) = (E_J=p.E_J, E_L13=p.E_L13, E_C23=p.E_C23, E_L23=p.E_L23, varphi_ext=2π * p.f)
specname(x) = x === :f ? :varphi_ext : x
fac(x) = x === :f ? 2π : 1.0                                      # d/df = 2π d/dvarphi_ext
const PORTS = [(:varphi1, nport()), (:varphi2, nport()), (:varphi1, phiport()), (:varphi2, phiport())]
const QNAMES = ["omega01", "alpha", "|n1_01|^2", "|n2_01|^2", "|phi1_01|^2", "|phi2_01|^2"]

function tree()
    EC = [real(SPEC.E_C[i, j](θof(P0))) for i in 1:2, j in 1:2]
    EL = [real(SPEC.E_L[i, j](θof(P0))) for i in 1:2, j in 1:2]
    σ1 = (8EC[1, 1] / EL[1, 1])^(1 / 4) / sqrt(2)
    l1 = Node([:varphi1 => FockBasis(60; phi_zpf=σ1)]; solver=Dense(), compute=40, keep=lowest(40), retain=:always)
    l2 = Node([:varphi2 => PhaseGrid(-24π, 24π, 1153; order=8)]; solver=Dense(), compute=60, keep=lowest(60), retain=:always)
    # the root by Lanczos for its 6 levels (2026-11-12, CPU: solve 0.32 s, Jacobian 0.21 s, full Hessian 0.73 s, against
    # 2.21 / 0.25 / 1.23 s with a Dense root's full decomposition; values and derivatives the same to 1e-13)
    return Node([l1, l2]; solver=Lanczos(tol=1e-11), compute=6, keep=lowest(6), ports=PORTS, retain=:always)
end

quantities(E, P) = [E[2] - E[1], E[3] - 2E[2] + E[1], (abs2(M[1, 2]) for M in P)...]

function solve(root, p)
    sys = materialize(root, SPEC, θof(p))
    return sys, quantities(energies(sys), [port(sys, m, q) for (m, q) in PORTS])
end

"Jacobian (6 × 5) and Hessians (6 × 5 × 5) from `derivatives(sys; order = 2)` with every mixed pair."
function jac_hess(sys)
    wrt = Tuple(specname.(PARAMS))
    mixed = Tuple((wrt[i], wrt[j]) for i in 1:5 for j in i+1:5)
    d = derivatives(sys; wrt, order=2, mixed)
    E = energies(sys)
    P = [port(sys, m, q) for (m, q) in PORTS]
    dE(x) = energies(d, specname(x)) .* fac(x)
    d2E(x, y) = energies(d, specname(x), specname(y)) .* (fac(x) * fac(y))
    dP(x, k) = port(d, specname(x), PORTS[k]...) .* fac(x)
    d2P(x, y, k) = port(d, specname(x), specname(y), PORTS[k]...) .* (fac(x) * fac(y))
    J = zeros(6, 5); H = zeros(6, 5, 5)
    lin(e) = [e[2] - e[1], e[3] - 2e[2] + e[1]]
    for (a, x) in enumerate(PARAMS)
        J[1:2, a] = lin(dE(x))
        for k in 1:4
            J[2+k, a] = 2real(conj(P[k][1, 2]) * dP(x, k)[1, 2])
        end
        for (b, y) in enumerate(PARAMS)
            b < a && continue
            H[1:2, a, b] = lin(d2E(x, y))
            for k in 1:4
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
@printf("range warnings: %d\nmaterialize %.4f ± %.4f s, first derivatives alone %.4f ± %.4f s, derivatives (first and second, 5 params, 10 mixed pairs) %.4f ± %.4f s (10 trials)\n", W[], t_mat..., t_jac..., t_der...)
for (i, n) in enumerate(QNAMES)
    @printf("%-12s = %+.9e   J: %s\n", n, q0[i], join((@sprintf("%+.4e", x) for x in J[i, :]), " "))
end

# central finite differences of materialize for the Jacobian (step 1e-4 relative, 1e-4 Φ0 for the flux)
step(x) = x === :f ? 1e-4 : 1e-4 * getfield(P0, x)
Jfd = zeros(6, 5)
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
Hfd = zeros(6, 5, 5)
for (a, x) in enumerate(PARAMS)
    h = step(x)
    Jp = jac_hess(solve(root, merge(P0, NamedTuple{(x,)}((getfield(P0, x) + h,))))[1])[1]
    Jm = jac_hess(solve(root, merge(P0, NamedTuple{(x,)}((getfield(P0, x) - h,))))[1])[1]
    Hfd[:, :, a] = (Jp .- Jm) ./ 2h
end
hdev = maximum(maximum(abs, H[i, :, :] .- Hfd[i, :, :]) / maximum(abs, Hfd[i, :, :]) for i in 1:6)
@printf("Hessian vs central differences of the Jacobian: max deviation %.1e of each Hessian's largest entry\n", hdev)
mkpath(joinpath(@__DIR__, "results"))
open(io -> JSON3.pretty(io, JSON3.write(Dict("params" => string.(PARAMS), "quantities" => QNAMES, "values" => q0,
                                             "jacobian" => J, "hessian" => [H[i, :, :] for i in 1:6], "jacobian_fd" => Jfd, "hessian_fd" => [Hfd[i, :, :] for i in 1:6],
                                             "seconds_materialize_mean_std" => collect(t_mat), "seconds_derivatives_mean_std" => collect(t_der), "seconds_first_derivatives_mean_std" => collect(t_jac), "timing_trials" => 10, "threads" => Threads.nthreads(),
                                             "range_warnings" => W[]))),
     joinpath(@__DIR__, "results", "kronfree.json"), "w")
