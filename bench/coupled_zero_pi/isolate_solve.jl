# Isolate the linear solves on the complement (the Sternheimer step of `derivatives`) of the coupled zero-pi tree and
# replay them, for debugging: the 54 300-state (θ, φ) leaves and the 23 870-state root (ZP_NTOTAL levels, default 200).
# Same tree as derivatives.jl. Runs `derivatives` once along ISO_WRT (default n_g1) with the capture hook on, lists every
# solve, and replays those in ISO_SOLVES (default: every one). (The library back ends compared with the built-in solver
# on 2026-11-17, README "The derivatives' linear solve", are in commit fcc0812.)
#
#   BENCH_DEVICE=cpu|gpu [ISO_WRT=n_g1] [ISO_SOLVES=1,3] \
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
derivatives(sys; wrt)
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
    run() = KF._complement_solve(c.H, c.D, c.basis, c.B, c.shifts; tol=c.tol, maxiter=c.maxiter, eta=c.eta, scale=c.scale)
    run(); sync()                                                                    # compile and warm
    t = @elapsed (r = run(); sync())
    X, iterations, residuals, from_start, at_floor = r
    @printf("  %.2f s, %d iterations, worst residual %.1e, from start %d, at floor %d\n",
            t, iterations, maximum(residuals; init=0.0), length(from_start), length(at_floor))
end
