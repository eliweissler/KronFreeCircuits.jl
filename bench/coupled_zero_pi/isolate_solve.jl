# Isolate the linear solves on the complement (the Sternheimer step of `derivatives`) of the coupled zero-pi tree and
# replay them with each back end: the 54 300-state (θ, φ) leaves and the 23 870-state root (ZP_NTOTAL levels, default
# 200). Same tree as derivatives.jl. Runs `derivatives` once along ISO_WRT (default n_g1) with the capture hook on, lists
# every solve, and replays those in ISO_SOLVES (default: every one) with each back end in ISO_BACKENDS.
#
#   BENCH_DEVICE=cpu|gpu [ISO_WRT=n_g1] [ISO_SOLVES=1,3] [ISO_BACKENDS=builtin,krylov_stacked,krylov] \
#       julia --project=bench -t 8 --gcthreads=1 bench/coupled_zero_pi/isolate_solve.jl

include(joinpath(@__DIR__, "..", "common.jl"))
const KF = KronFreeCircuits

const SPEC = load_spec(joinpath(@__DIR__, "specs", "coupled_zero_pi.json"))
const NTOTAL = parse(Int, get(ENV, "ZP_NTOTAL", "200"))
const TRUNC, THRESH, NCUT, NPHI = 300, 1e-5, 90, 300
θ = (n_g1=0.0, n_g4=0.0, varphi_ext1=0.0, varphi_ext2=0.0)
leafsolver = DEVICE == "gpu" ? Auto(on=ON) : Auto()
qubit(q) = Node([Symbol("theta$q") => ChargeBasis(NCUT), Symbol("phi$q") => PhaseGrid(-6π, 6π, NPHI; order=6)];
                compute=TRUNC, solver=leafsolver, keep=connected([1, 3], (Symbol("theta$q"), nport()), THRESH), retain=:always)
root = Node([qubit(1), qubit(2), :zeta1 => FockBasis(1; phi_zpf=1.0), :zeta2 => FockBasis(1; phi_zpf=1.0)];
            solver=LOBPCG(; tol=1e-8, maxiter=3000, guard=8, on=ON), compute=NTOTAL, keep=lowest(NTOTAL), retain=:always)

wrt = Symbol(get(ENV, "ISO_WRT", "n_g1"))
sys = materialize(root, SPEC, θ)
KF._CAPTURE[] = Any[]
withenv(() -> derivatives(sys; wrt), "KRONFREE_LINSOLVE" => "builtin")
caps = KF._CAPTURE[]; KF._CAPTURE[] = nothing
println("device $DEVICE, $NTOTAL root levels, derivatives along :$wrt: $(length(caps)) solve(s) on the complement")
for (k, c) in enumerate(caps)
    @printf("  [%d] dimension %d, %d known states, %d columns, shifts %.3f to %.3f, tol %.0e, operator on %s\n", k,
            size(c.B, 1), size(c.basis, 2), size(c.B, 2), minimum(c.shifts), maximum(c.shifts), c.tol, c.H.on)
end
sync() = DEVICE == "gpu" && CUDA.synchronize()
pick = haskey(ENV, "ISO_SOLVES") ? parse.(Int, split(ENV["ISO_SOLVES"], ",")) : collect(eachindex(caps))
for k in pick
    c = caps[k]
    println("solve [$k]: $(size(c.B, 1)) states, $(size(c.B, 2)) columns")
    sols = Dict{String,Any}()
    for be in split(get(ENV, "ISO_BACKENDS", "builtin,krylov_stacked,krylov"), ",")
        run() = withenv("KRONFREE_LINSOLVE" => be) do
            KF._complement_solve(c.H, c.D, c.basis, c.B, c.shifts; tol=c.tol, maxiter=c.maxiter, eta=c.eta, scale=c.scale)
        end
        run(); sync()                                                                # compile and warm
        t = @elapsed (r = run(); sync())
        X, iterations, residuals, from_start, at_floor = r
        sols[be] = Array(X)
        @printf("  %-15s %8.2f s, iterations %5d, worst residual %.1e, from start %d, at floor %d\n",
                be, t, iterations, maximum(residuals; init=0.0), length(from_start), length(at_floor))
    end
    for (be, X) in sols
        be == "builtin" || !haskey(sols, "builtin") ||
            @printf("  max |X_%s − X_builtin| / max |X_builtin| = %.1e\n", be, maximum(abs, X .- sols["builtin"]) / maximum(abs, sols["builtin"]))
    end
end
