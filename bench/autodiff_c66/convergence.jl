# Convergence of kronfree.jl's tree: values, Jacobian and Hessian of the six quantities under one refinement of each
# truncation in turn, and of all at once, against the production tree (results/convergence.json).
#
#   julia --project=bench -t 8 --gcthreads=1 bench/autodiff_c66/convergence.jl
#
# Production tree: varphi1 on FockBasis(60) keeping 40, varphi2 on PhaseGrid(±24π, 1153, order 8) keeping 60, the root by
# Lanczos keeping 6. Changes are reported as the largest |change| of each quantity (values: absolute, GHz or
# dimensionless; Jacobian and Hessian: over each row's largest entry, as kronfree.jl compares against differences).

using KronFreeCircuits, LinearAlgebra, Printf, Logging
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
fac(x) = x === :f ? 2π : 1.0
const PORTS = [(:varphi1, nport()), (:varphi2, nport()), (:varphi1, phiport()), (:varphi2, phiport())]
const QNAMES = ["omega01", "alpha", "|n1_01|^2", "|n2_01|^2", "|phi1_01|^2", "|phi2_01|^2"]

function tree(; nfock=60, keep1=40, a=24, npts=1153, keep2=60)
    EC = [real(SPEC.E_C[i, j](θof(P0))) for i in 1:2, j in 1:2]
    EL = [real(SPEC.E_L[i, j](θof(P0))) for i in 1:2, j in 1:2]
    σ1 = (8EC[1, 1] / EL[1, 1])^(1 / 4) / sqrt(2)
    l1 = Node([:varphi1 => FockBasis(nfock; phi_zpf=σ1)]; solver=Dense(), compute=keep1, keep=lowest(keep1), retain=:always)
    l2 = Node([:varphi2 => PhaseGrid(-a * π, a * π, npts; order=8)]; solver=Dense(), compute=keep2, keep=lowest(keep2), retain=:always)
    return Node([l1, l2]; solver=Lanczos(tol=1e-11), compute=6, keep=lowest(6), ports=PORTS, retain=:always)
end

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
    lin(e) = [e[2] - e[1], e[3] - 2e[2] + e[1]]
    q = [E[2] - E[1], E[3] - 2E[2] + E[1], (abs2(M[1, 2]) for M in P)...]
    J = zeros(6, 5); H = zeros(6, 5, 5)
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
    return q, J, H
end

const VARIANTS = [("production: Fock 60/40, grid ±24π 1153/60", (;)),
                  ("Fock 80, keep 60", (nfock=80, keep1=60)),
                  ("grid leaf keep 80", (keep2=80,)),
                  ("grid 1537 points on ±24π (finer)", (npts=1537,)),
                  ("grid ±32π, 1537 points (wider)", (a=32, npts=1537)),
                  ("all: Fock 80/60, grid ±32π 2049/80", (nfock=80, keep1=60, a=32, npts=2049, keep2=80))]

rows = Any[]
q0 = J0 = H0 = nothing
for (name, kw) in VARIANTS
    W[] = 0
    sys = materialize(tree(; kw...), SPEC, θof(P0))
    q, J, H = jac_hess(sys)
    global q0, J0, H0
    q0 === nothing && ((q0, J0, H0) = (q, J, H))
    dq = abs.(q .- q0)
    dJ = [maximum(abs, J[i, :] .- J0[i, :]) / maximum(abs, J0[i, :]) for i in 1:6]
    dH = [maximum(abs, H[i, :, :] .- H0[i, :, :]) / maximum(abs, H0[i, :, :]) for i in 1:6]
    @printf("%-42s edge warnings %d\n", name, W[])
    @printf("   values  Δ: %s\n   Jacobian Δ/row max: %s\n   Hessian  Δ/max:     %s\n",
            join((@sprintf("%s %.1e", n, x) for (n, x) in zip(QNAMES, dq)), ", "),
            join((@sprintf("%.1e", x) for x in dJ), " "), join((@sprintf("%.1e", x) for x in dH), " "))
    push!(rows, Dict("variant" => name, "settings" => Dict(string(k) => v for (k, v) in pairs(kw)), "edge_warnings" => W[],
                     "values" => q, "jacobian" => J, "hessian" => [H[i, :, :] for i in 1:6],
                     "value_change" => dq, "jacobian_change_rel" => dJ, "hessian_change_rel" => dH))
end
mkpath(joinpath(@__DIR__, "results"))
open(io -> JSON3.pretty(io, JSON3.write(Dict("quantities" => QNAMES, "params" => string.(PARAMS), "variants" => rows))),
     joinpath(@__DIR__, "results", "convergence.json"), "w")
