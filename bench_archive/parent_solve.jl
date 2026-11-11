# Part A: the audit's stored 300 × 300 retained parent, as a resident regression.
#
#   BENCH_FIXTURE=path/to/larger-real300-operators.bin BENCH_DEVICE=cpu|gpu julia --project=... bench/parent_solve.jl
#
# Rebuilds the stored operator (checking H * x ≈ y), solves the stored coupling point cold for
# BENCH_NEV states (:ritz, default guard), then the coupling scaled by 1.01 warm-started from those
# vectors, with :jacobi and :ritz and each guard count in BENCH_GUARDS (default "4,0"; the audit had
# no guard columns). The operator is built on the device before timing; each timed sample is one
# `lobpcg_solve` call (its host-side seed perturbation, block upload and result download included;
# measured separately as the entry overhead). One untimed warm-up precedes BENCH_REPEATS samples.
# Each sample also records the iteration-loop time inside the solve: wall-clock stamps taken in the
# solver callback (after a device synchronization on the GPU), first to last callback, and the
# time from the call to the first callback (lobpcg_solve's entry work plus the eigensolver's setup
# through its first residuals). BENCH_PRECONDITIONERS selects the warm preconditioners (default
# "jacobi,ritz"). The script exits nonzero when any sample failed or did not validate.
# BENCH_COMPARE=path/to/cpu.json compares the energies with an earlier run's, configuration by
# configuration (host/device agreement).

include(joinpath(@__DIR__, "common.jl"))
using Serialization

const FIXTURE = get(ENV, "BENCH_FIXTURE", "")
if isempty(FIXTURE)
    println("BENCH_FIXTURE is not set: part A skipped")
    exit(0)
end
const NEV = parse(Int, get(ENV, "BENCH_NEV", SMOKE ? "6" : "120"))
const GUARDS = parse.(Int, split(get(ENV, "BENCH_GUARDS", "4,0"), ","))
const PRECONDITIONERS = Symbol.(split(get(ENV, "BENCH_PRECONDITIONERS", "jacobi,ritz"), ","))

# Wall-clock stamps of the current solve: the call, then one per solver callback (synchronized).
const STAMPS = Float64[]
stamp!() = (devsync(); push!(STAMPS, time_ns() / 1e9); nothing)
stamp_callback(info) = stamp!()
const TOL = 1e-8
const STEP = 1.01

entry = only(deserialize(FIXTURE))
dims = Vector{Int}(entry["dims"])
coupling = [i for (i, (c, fs)) in enumerate(entry["terms"]) if any(f -> f[2] == "dense", fs)]
length(coupling) == 1 || error("expected exactly one dense (coupling) term, found $(length(coupling))")
function fixture_terms(scale)
    return AxisTerm[AxisTerm(i in coupling ? scale * c : c,
                             Tuple{Int,KronFreeCircuits.LocalOp}[(a, k == "diag" ? DiagonalOp(ComplexF64.(M)) : DenseOp(M))
                                                                  for (a, k, M) in fs])
                    for (i, (c, fs)) in enumerate(entry["terms"])]
end
cap = NEV + maximum(GUARDS)
H0h = KronFreeOperator(dims, fixture_terms(1.0); constant=entry["constant"], block_capacity=1)
fit = norm(H0h * entry["x"] - entry["y"]) / norm(entry["y"])
fit < 1e-12 || error("the stored operator does not rebuild: ‖H x − y‖ / ‖y‖ = $fit")
@printf("fixture %s: dims %s, coupling %s, ‖Hx − y‖/‖y‖ = %.2e\n", entry["name"], dims,
        entry["terms"][only(coupling)][1], fit)

t_build0, H0 = timed(() -> KronFreeOperator(dims, fixture_terms(1.0); constant=entry["constant"], block_capacity=cap, on=ON))
t_build1, H1 = timed(() -> KronFreeOperator(dims, fixture_terms(STEP); constant=entry["constant"], block_capacity=cap, on=ON))
@printf("setup: operator construction on %s %.3f s (stored point), %.3f s (stepped point)\n", DEVICE, t_build0, t_build1)

record = Dict{String,Any}("part" => "A", "environment" => environment(), "fixture" => FIXTURE, "nev" => NEV,
                          "tol" => TOL, "step" => STEP, "guards" => GUARDS, "fixture_fit" => fit,
                          "setup_seconds" => Dict("build_stored" => t_build0, "build_stepped" => t_build1),
                          "configs" => Any[])

# Compilation warm-up on the same types at a small width (not a timed configuration).
lobpcg_solve(LOBPCG(; on=ON), H0, 4)

# One timed `lobpcg_solve`, validated; a solver error is recorded as a failed sample (with its
# time to failure) and returns `nothing`.
function sample!(cfg, s, H, X0)
    t0 = time_ns()
    empty!(STAMPS)
    t, (E, V, res) = try
        timed(() -> (stamp!(); lobpcg_solve(s, H, NEV; X0=X0)))
    catch err
        err isa InterruptException && rethrow()
        msg = sprint(showerror, err)
        push!(cfg["samples"], Dict("seconds" => (time_ns() - t0) / 1e9, "failed" => true, "error" => msg, "ok" => false))
        println("FAILED ", cfg["label"], ": ", first(msg, 300))
        return nothing
    end
    v = validate(H, E, V; tol=TOL)
    push!(cfg["samples"], Dict("seconds" => t, "n_matvec" => res.n_matvec, "iterations" => iterations(res),
                               "loop_seconds" => STAMPS[end] - STAMPS[2], "to_first_callback_seconds" => STAMPS[2] - STAMPS[1],
                               "callbacks" => length(STAMPS) - 1,
                               "max_residual" => v.max_residual, "ortho" => v.ortho, "ok" => v.ok))
    v.ok || @warn "validation failed" cfg["label"] v
    return E, V
end

# Cold solve of the stored point: its vectors seed every warm configuration.
cold = Dict{String,Any}("label" => "cold ritz guard=4", "kind" => "cold", "preconditioner" => "ritz", "guard" => 4,
                        "samples" => Any[])
cold_result = sample!(cold, LOBPCG(; on=ON, callback=stamp_callback), H0, nothing)
push!(record["configs"], cold)
if cold_result === nothing
    write_json("parent_solve-$DEVICE$(SMOKE ? "-smoke" : "").json", record)
    error("the cold solve failed; no warm configuration can run")
end
E0, V0 = cold_result
cold["energies"] = E0
s0 = only(cold["samples"])
@printf("%-28s %8.3f s (loop %.3f s)  H columns %5d  iters %4d  max residual %.2e  ortho %.1e  %s\n", cold["label"],
        s0["seconds"], s0["loop_seconds"], s0["n_matvec"], s0["iterations"], s0["max_residual"], s0["ortho"], s0["ok"] ? "ok" : "FAILED")

# Entry overhead inside each timed lobpcg_solve call: seed perturbation, block upload, result download.
let T = eltype(H1)
    KronFreeCircuits._perturbed_seed(T, V0)            # compile outside the timing
    t_seed, X = timed(() -> KronFreeCircuits._perturbed_seed(T, V0))
    Xw = hcat(X, randn(T, size(X, 1), 4))
    KronFreeCircuits.adapt(ON, Xw[:, 1:1])
    t_up, Xd = timed(() -> KronFreeCircuits.adapt(ON, Xw))
    t_down, _ = timed(() -> Matrix{T}(Array(view(Xd, :, 1:NEV))))
    record["entry_overhead_seconds"] = Dict("seed_perturbation" => t_seed, "upload" => t_up, "download" => t_down)
    @printf("entry overhead per warm solve: seed perturbation %.3f s, upload %.3f s, download %.3f s\n", t_seed, t_up, t_down)
end

Eref = nothing
Vport = nothing
for pc in PRECONDITIONERS, g in GUARDS
    global Eref, Vport
    s = LOBPCG(; preconditioner=pc, guard=g, on=ON, callback=stamp_callback)
    cfg = Dict{String,Any}("label" => "warm $pc guard=$g", "kind" => "warm", "preconditioner" => string(pc),
                           "guard" => g, "samples" => Any[])
    push!(record["configs"], cfg)
    try
        lobpcg_solve(s, H1, NEV; X0=V0)                 # untimed warm-up
    catch err
        err isa InterruptException && rethrow()
        println("warm-up of ", cfg["label"], " failed: ", first(sprint(showerror, err), 300))
    end
    results = [sample!(cfg, s, H1, V0) for _ in 1:REPEATS]
    if any(isnothing, results)
        println("FAILED ", cfg["label"], ": ", count(isnothing, results), " of ", REPEATS, " samples")
        continue
    end
    Es = first.(results)
    pc === :ritz && g == GUARDS[1] && (Vport = last(results[end]))
    cfg["energies"] = Es[1]
    cfg["energy_spread_between_repeats"] = maximum(maximum(abs.(E .- Es[1])) for E in Es)
    Eref === nothing && (Eref = Es[1])
    cfg["max_energy_difference_to_first_config"] = maximum(abs.(Es[1] .- Eref))
    ts = [x["seconds"] for x in cfg["samples"]]
    smp = cfg["samples"]
    @printf("%-28s %s  loop %s  H columns %s  iters %s  max residual %.2e  ortho %.1e  ΔE vs first %.1e  %s\n", cfg["label"],
            summary_line(ts), summary_line([x["loop_seconds"] for x in smp]), join(unique(x["n_matvec"] for x in smp), "/"), join(unique(x["iterations"] for x in smp), "/"),
            maximum(x["max_residual"] for x in smp), maximum(x["ortho"] for x in smp),
            cfg["max_energy_difference_to_first_config"], all(x["ok"] for x in smp) ? "ok" : "FAILED")
end

# Port formation after the solve, on the host as in `materialize`: the coupling's two leaf charge
# ports (the stored dense factors) in the kept basis of the warm :ritz solution, the ports an
# ancestor of this parent would need.
if Vport !== nothing
    ports = [(a, DenseOp(M)) for (a, k, M) in entry["terms"][only(coupling)][2]]
    form() = [KronFreeCircuits._port_matrix(Vport, dims, a, op) for (a, op) in ports]
    form()
    tp = [first(timed(form)) for _ in 1:REPEATS]
    record["port_formation_seconds"] = tp
    @printf("%-28s %s  (%d ports, %d states, host)\n", "port formation", summary_line(tp), length(ports), NEV)
end
if DEVICE == "gpu"
    tu = [first(timed(() -> KronFreeCircuits._on_storage(H0h, ON; capacity=cap))) for _ in 1:REPEATS + 1][2:end]
    record["operator_upload_seconds"] = tu
    @printf("%-28s %s\n", "operator upload alone", summary_line(tu))
end

compare = get(ENV, "BENCH_COMPARE", "")
if !isempty(compare)
    other = JSON3.read(read(compare, String))
    agreement = Dict{String,Any}()
    for cfg in record["configs"], o in other.configs
        String(o.label) == cfg["label"] && haskey(cfg, "energies") && haskey(o, :energies) || continue
        d = maximum(abs.(Vector{Float64}(o.energies) .- cfg["energies"]))
        agreement[cfg["label"]] = d
        @printf("agreement with %s, %-22s max |ΔE| = %.2e  %s\n", basename(compare), cfg["label"], d, d < 1e-9 ? "ok" : "ABOVE 1e-9")
    end
    record["agreement_with"] = Dict("file" => compare, "max_energy_difference" => agreement)
end

write_json("parent_solve-$DEVICE$(SMOKE ? "-smoke" : "").json", record)

nfailed = count(x -> !x["ok"], (x for cfg in record["configs"] for x in cfg["samples"]))
nsamples = sum(length(cfg["samples"]) for cfg in record["configs"])
println("summary: $nfailed of $nsamples samples failed or did not validate")
nfailed == 0 || exit(1)
