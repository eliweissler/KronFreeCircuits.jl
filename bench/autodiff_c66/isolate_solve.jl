# Isolate one linear solve on the complement (the Sternheimer step of `derivatives`) of the c66 tree and replay it, for
# debugging. Runs `derivatives` once along ISO_WRT with the capture hook on, lists every solve it made, then replays solve
# ISO_SOLVE (default: the largest) and reports its time, iterations and residuals. (The library back ends compared
# with it on 2026-11-17 are in commit fcc0812.)
#
#   KF_DEVICE=cpu|gpu ISO_WRT=E_J ISO_SOLVE=<k> julia --project=bench -t 8 --gcthreads=1 bench/autodiff_c66/isolate_solve.jl
#
# ISO_SAVE=<file>: also write the chosen solve's inputs (on the host; the operator as a dense matrix when it has at most
# 10 000 rows) with Serialization, to load anywhere.

using KronFreeCircuits, LinearAlgebra, Printf, Serialization
const DEVICE = get(ENV, "KF_DEVICE", "cpu")
DEVICE == "gpu" && (using CUDA, cuTENSOR)
const GPU = DEVICE == "gpu"
const KF = KronFreeCircuits

const SPEC = load_spec(joinpath(@__DIR__, "specs", "c66_sqcircuit.json"))
const P0 = (E_J=28.83436338, E_L13=0.10343485, E_C23=1.36334654, E_L23=0.05167763, f=0.0)
θof(p) = (E_J=p.E_J, E_L13=p.E_L13, E_C23=p.E_C23, E_L23=p.E_L23, varphi_ext=2π * p.f)
const PORTS = [(:varphi1, nport()), (:varphi2, nport()), (:varphi1, phiport()), (:varphi2, phiport())]

# kronfree.jl's tree
function tree()
    EC = [real(SPEC.E_C[i, j](θof(P0))) for i in 1:2, j in 1:2]
    EL = [real(SPEC.E_L[i, j](θof(P0))) for i in 1:2, j in 1:2]
    σ1 = (8EC[1, 1] / EL[1, 1])^(1 / 4) / sqrt(2)
    l1 = Node([:varphi1 => FockBasis(60; phi_zpf=σ1)]; solver=Dense(), compute=40, keep=lowest(40), retain=:always)
    l2 = Node([:varphi2 => PhaseGrid(-24π, 24π, 1153; order=8)];
              solver=GPU ? LOBPCG(tol=1e-10, on=CuArray) : Dense(), compute=60, keep=lowest(60), retain=:always)
    return Node([l1, l2]; solver=Lanczos(tol=1e-11, on=GPU ? CuArray : Array), compute=6, keep=lowest(6), ports=PORTS, retain=:always)
end

wrt = Symbol(get(ENV, "ISO_WRT", "E_J"))
sys = materialize(tree(), SPEC, θof(P0))
KF._CAPTURE[] = Any[]
derivatives(sys; wrt)
caps = KF._CAPTURE[]; KF._CAPTURE[] = nothing
println("device $DEVICE, derivatives along :$wrt: $(length(caps)) solve(s) on the complement")
for (k, c) in enumerate(caps)
    @printf("  [%d] dimension %d, %d known states, %d columns, shifts %s, tol %.0e, maxiter %d, operator on %s\n", k,
            size(c.B, 1), size(c.basis, 2), size(c.B, 2), join((@sprintf("%.4f", s) for s in c.shifts), " "), c.tol,
            c.maxiter, c.H.on)
end
isempty(caps) && exit()
k = haskey(ENV, "ISO_SOLVE") ? parse(Int, ENV["ISO_SOLVE"]) : argmax([size(c.B, 1) * size(c.B, 2) for c in caps])
c = caps[k]
println("replaying solve [$k]")
if haskey(ENV, "ISO_SAVE")
    serialize(ENV["ISO_SAVE"], (; H=size(c.B, 1) <= 10_000 ? Matrix(c.H) : nothing, D=Array(c.D), basis=Array(c.basis),
                                B=Array(c.B), shifts=c.shifts, c.tol, c.maxiter, c.eta, c.scale))
    println("saved its inputs to $(ENV["ISO_SAVE"])")
end
replay() = KF._complement_solve(c.H, c.D, c.basis, c.B, c.shifts; tol=c.tol, maxiter=c.maxiter, eta=c.eta, scale=c.scale)
replay()                                                                             # compile
GPU && CUDA.synchronize()
t = @elapsed (r = replay(); GPU && CUDA.synchronize())
X, iterations, residuals, from_start, at_floor = r
@printf("%.3f s, %d iterations (all columns in lockstep), residuals %s, from start %s, at floor %s\n",
        t, iterations, join((@sprintf("%.1e", x) for x in residuals), " "), from_start, at_floor)
