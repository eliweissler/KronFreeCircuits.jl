# A design problem solved with the outer rule: recover the planted parameters of two coupled transmons from three of
# their observables, minimizing a least-squares loss with BFGS on the gradient that Zygote takes through the `rrule` of
# `spectrum` (the package's ChainRulesCore extension).
#
#   julia --project=bench/derivative_design bench/derivative_design/design.jl
#
# The script writes no files: everything it reports goes to standard output.
#
# Exits nonzero unless
#   1. the gradient at the start agrees with central finite differences of the loss to 1e-6 relative (to the largest
#      component of the finite difference);
#   2. the final loss is below 1e-12;
#   3. the final x is within 1e-4 of the planted x* in each component;
#   4. at the final point the largest component of the gradient and of its finite difference are both below 1e-5.
# How the optimizer stopped (its gradient tolerance, or another reason) is printed and is not one of the criteria.

using KronFreeCircuits, Optim, Zygote, LinearAlgebra, Printf

BLAS.set_num_threads(1)
const T0 = time()
println("derivative_design: two coupled transmons, three parameters, BFGS on the gradient of the outer rule")
println("Julia $(VERSION), $(Threads.nthreads()) thread(s), BLAS threads $(BLAS.get_num_threads())")

# --- The problem ------------------------------------------------------------------------------------------------------
# Two transmons, sum_i [4 E_Ci (n_i - n_gi)^2 - E_Ji cos(varphi_i)] + g (n_1 - n_g1)(n_2 - n_g2), as a two-level tree:
# one leaf per transmon on ChargeBasis(30) keeping its 4 lowest states, and a Dense root keeping 6, with the port n_1.
const SPEC = load_spec(joinpath(@__DIR__, "..", "..", "test", "fixtures", "two_transmons.json"))
const LEAVES = [Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(4)) for i in 1:2]
const ROOT = Node(LEAVES; solver=Dense(), keep=lowest(6), ports=[(:varphi_1, nport())])
const PORTS = [(:varphi_1, nport())]

# The free parameters x = (E_J1, E_J2, g); E_C1, E_C2, n_g1, n_g2 are fixed.
theta(x) = (E_J1=x[1], E_C1=1.2, n_g1=0.21, E_J2=x[2], E_C2=1.0, n_g2=0.09, g=x[3])

# The features of a point: the two lowest transition energies and the magnitude of the charge matrix element n_1[1, 2].
function features(x)
    y = spectrum(ROOT, SPEC, theta(x); ports=PORTS)
    E, n = y.energies, y.ports[1]
    return [E[2] - E[1], E[3] - E[1], abs(n[1, 2])]
end

const XSTAR = [28.0, 24.0, 0.25]          # the planted design
const X0 = [30.0, 25.0, 0.3]              # the start
const TARGET = features(XSTAR)
const WEIGHT = [1.0, 1.0, 0.1]

const NLOSS = Ref(0)                      # every evaluation of the loss: the finite differences' and the forward pass
                                          # inside each gradient included
const NGRAD = Ref(0)                      # every gradient through Zygote
loss(x) = (NLOSS[] += 1; sum(((features(x) .- TARGET) ./ WEIGHT) .^ 2))
loss_gradient(x) = (NGRAD[] += 1; Zygote.gradient(loss, x)[1])

println("planted x* = ", XSTAR, "  start x0 = ", X0)
@printf("targets φ(x*) = [%.12f, %.12f, %.12f]\n", TARGET...)

# --- The gradient check -------------------------------------------------------------------------------------------------
# Central differences of the loss with step h = 1e-4 and h/2, extrapolated once: (4 D(h/2) - D(h)) / 3.
const H = 1e-4
function fd_gradient(f, x; h=H)
    D(i, s) = (e = zeros(length(x)); e[i] = s; (f(x .+ e) - f(x .- e)) / (2s))
    return [(4 * D(i, h / 2) - D(i, h)) / 3 for i in eachindex(x)]
end

function report_check(label, x)
    g = loss_gradient(x)
    fd = fd_gradient(loss, x)
    rel = maximum(abs.(g .- fd)) / maximum(abs.(fd))
    println("gradient check at $label, x = ", x)
    @printf("  gradient (Zygote)   = [%+.10e, %+.10e, %+.10e]\n", g...)
    @printf("  finite difference   = [%+.10e, %+.10e, %+.10e]\n", fd...)
    @printf("  max |difference|    = %.3e, relative to max |fd| = %.3e\n", maximum(abs.(g .- fd)), rel)
    return (; g, fd, rel)
end

failures = String[]
check(ok, what) = (println(ok ? "PASS  " : "FAIL  ", what); ok || push!(failures, what))

start = report_check("x0", X0)
@printf("loss at x0 = %.10e\n", loss(X0))

# --- The optimization ---------------------------------------------------------------------------------------------------
g!(G, x) = (G .= loss_gradient(x); G)
const OPTIONS = Optim.Options(g_tol=1e-10, iterations=200, store_trace=true, extended_trace=true)
const NLOSS_BEFORE, NGRAD_BEFORE = NLOSS[], NGRAD[]
const T_OPT = time()
res = optimize(loss, g!, copy(X0), BFGS(), OPTIONS)
const WALL_OPT = time() - T_OPT

println("\niterations (loss, x = (E_J1, E_J2, g), ‖gradient‖∞):")
for st in Optim.trace(res)
    x = st.metadata["x"]
    @printf("  %4d  %.6e  (%.10f, %.10f, %.10f)  %.3e\n", st.iteration, st.value, x..., st.g_norm)
end

xf = Optim.minimizer(res)
println("\nhow the optimizer stopped: termination code ", Optim.termination_code(res),
        "; gradient tolerance met: ", Optim.g_converged(res),
        "; x change zero: ", Optim.x_converged(res),
        "; f change zero: ", Optim.f_converged(res),
        "; iteration limit reached: ", Optim.iteration_limit_reached(res))
println(Optim.g_converged(res) ? "the optimizer CONVERGED: ‖gradient‖∞ ≤ g_tol = 1e-10" :
        "the optimizer did NOT meet its gradient tolerance g_tol = 1e-10; it stopped for the reason above")
@printf("iterations %d; Optim's evaluations: %d of the loss, %d of the gradient; optimization wall time %.1f s\n",
        Optim.iterations(res), Optim.f_calls(res), Optim.g_calls(res), WALL_OPT)
@printf("evaluations counted by the script during the optimization: %d of the loss (one inside each gradient), %d of the gradient\n",
        NLOSS[] - NLOSS_BEFORE, NGRAD[] - NGRAD_BEFORE)
println("final x = ", xf)
@printf("final x - x* = [%+.3e, %+.3e, %+.3e]\n", (xf .- XSTAR)...)
@printf("final loss = %.6e\n", Optim.minimum(res))
@printf("features at the final x = [%.12f, %.12f, %.12f]\n", features(xf)...)

final = report_check("the final x", xf)

# --- The pass criteria --------------------------------------------------------------------------------------------------
println()
check(start.rel ≤ 1e-6, @sprintf("gradient at x0 agrees with finite differences to 1e-6 relative (%.3e)", start.rel))
check(Optim.minimum(res) < 1e-12, @sprintf("final loss below 1e-12 (%.3e)", Optim.minimum(res)))
check(all(abs.(xf .- XSTAR) .≤ 1e-4), @sprintf("final x within 1e-4 of x* in each component (max %.3e)",
                                               maximum(abs.(xf .- XSTAR))))
check(maximum(abs.(final.g)) < 1e-5 && maximum(abs.(final.fd)) < 1e-5,
      @sprintf("at the final x, max |gradient| (%.3e) and max |finite difference| (%.3e) below 1e-5",
               maximum(abs.(final.g)), maximum(abs.(final.fd))))
@printf("\ntotal evaluations: %d of the loss (one inside each gradient), %d of the gradient; total wall time %.1f s\n",
        NLOSS[], NGRAD[],
        time() - T0)
if isempty(failures)
    println("all checks passed")
else
    println("$(length(failures)) check(s) failed")
    exit(1)
end
