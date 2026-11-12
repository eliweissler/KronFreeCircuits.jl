# Shared harness of the benchmark suite (bench/README.md): environment knobs, the run record, per-point timing with
# `node_timings`, sweeps warm-started along their parameter points, and the accuracy ladder.
#
#   BENCH_DEVICE        cpu | gpu      (gpu loads CUDA and cuTENSOR)                    default cpu
#   BENCH_REPEATS       timed repeats per point after the first call                    default 3
#   BENCH_BLAS_THREADS  BLAS threads on the CPU                                          default 8
#   BENCH_SMOKE         1: tiny sizes (checks the script, not the physics)               default 0
#   BENCH_OUT           results directory                     default <case>/results/<host>-<yyyymmdd-HHMM>/

using LinearAlgebra, Printf, Dates, Random, Statistics
using KronFreeCircuits
using KronFreeCircuits: dimension
import JSON3

const DEVICE = get(ENV, "BENCH_DEVICE", "cpu")
DEVICE in ("cpu", "gpu") || error("BENCH_DEVICE must be cpu or gpu, got $DEVICE")
const SMOKE = get(ENV, "BENCH_SMOKE", "0") == "1"
const REPEATS = parse(Int, get(ENV, "BENCH_REPEATS", "3"))
if DEVICE == "gpu"
    using CUDA, cuTENSOR
    const ON = CuArray
    BLAS.set_num_threads(1)
else
    const ON = Array
    BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_BLAS_THREADS", "8")))
end
const NLEVELS = 8                                          # levels compared by the accuracy rule

# ---- run identity -----------------------------------------------------------------------------------------------

function _git(args...)
    dir = pkgdir(KronFreeCircuits)
    try
        return strip(read(Cmd(`git -C $dir $args`), String))
    catch
        return ""
    end
end

"The run's identity: host, hardware, threads, versions, commit (and a dirty working tree), date."
function environment(case::AbstractString)
    env = Dict{String,Any}(
        "case" => case, "host" => gethostname(), "cpu" => Sys.cpu_info()[1].model, "cores" => Sys.CPU_THREADS,
        "device" => DEVICE, "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(), "package_version" => string(pkgversion(KronFreeCircuits)),
        "commit" => _git("rev-parse", "HEAD"), "dirty" => !isempty(_git("status", "--porcelain", "--", "src")),
        "date" => string(now()), "smoke" => SMOKE, "repeats" => REPEATS)
    DEVICE == "gpu" && (env["gpu"] = CUDA.name(CUDA.device()))
    return env
end

"""
    Recorder(case, dir) -> Recorder

Collects the runs of one case and writes them to `<out>/results.json` after every run, `out` being `BENCH_OUT` or
`dir/results/<host>-<date>/`.
"""
struct Recorder
    out::String
    record::Dict{String,Any}
end
function Recorder(case::AbstractString, dir::AbstractString)
    out = get(ENV, "BENCH_OUT", joinpath(dir, "results", "$(gethostname())-$(Dates.format(now(), "yyyymmdd-HHMM"))"))
    mkpath(out)
    env = environment(case)
    println("case $case: device $DEVICE, threads $(env["threads"]), BLAS $(env["blas_threads"]), commit ",
            first(env["commit"], 8), env["dirty"] ? " (src dirty)" : "", ", smoke $SMOKE, out $out")
    return Recorder(out, Dict{String,Any}("environment" => env, "runs" => Any[], "summaries" => Any[]))
end
save(r::Recorder) = open(io -> JSON3.pretty(io, JSON3.write(r.record)), joinpath(r.out, "results.json"), "w")

# ---- sizes ------------------------------------------------------------------------------------------------------

"The dimension of one product space over every leaf basis (and given system) of the tree: what a flat solve holds."
flat_dim(n::Node) = prod(BigInt(_flat(c)) for c in n.children)
_flat(c::Node) = flat_dim(c)
_flat(c::GivenSystem) = length(c.energies)
_flat(c::Pair) = dimension(c.second)

# ---- one point ----------------------------------------------------------------------------------------------------

function _call(f)
    GC.gc()
    s = @timed node_timings(f)
    sys, nodes = s.value
    return (; sys, nodes, seconds=s.time, bytes=s.bytes, gc_seconds=s.gctime)
end

function _memory(sys)
    m = Dict{String,Any}("maxrss_bytes" => Sys.maxrss(), "system_bytes" => Base.summarysize(sys))
    if DEVICE == "gpu"
        m["gpu_used_bytes"] = CUDA.used_memory()
        m["gpu_cached_bytes"] = CUDA.cached_memory()
    end
    return m
end

_node_dict(r) = Dict(string(k) => (v isa Symbol ? string(v) : v) for (k, v) in pairs(r))

"""
    timed_point(f; repeats=REPEATS) -> (sys, record)

Call `f()` (a `materialize`) once, then `repeats` more times, each inside `node_timings`.
The record holds the first call's time (`first_seconds`: with compilation on a process's first point), every repeat's
time and their median (`seconds`: the headline), the last repeat's per-node records, its allocations, and memory.
"""
function timed_point(f; repeats::Int=REPEATS)
    first_call = _call(f)
    reps = [_call(f) for _ in 1:repeats]
    last_call = isempty(reps) ? first_call : reps[end]
    times = [r.seconds for r in reps]
    root = last_call.nodes[end]
    rec = Dict{String,Any}(
        "first_seconds" => first_call.seconds, "repeat_seconds" => times,
        "seconds" => isempty(times) ? first_call.seconds : median(times),
        "bytes_allocated" => last_call.bytes, "gc_seconds" => last_call.gc_seconds,
        "nodes" => [_node_dict(r) for r in last_call.nodes],
        "root_dim" => root.dim,
        "root_seconds_per_matvec" => root.n_matvec === nothing ? nothing : root.solve_seconds / root.n_matvec,
        "memory" => _memory(last_call.sys))
    return last_call.sys, rec
end

# ---- sweeps -------------------------------------------------------------------------------------------------------

"""
    sweep!(rec, label, root, model, points, θ; settings, reference=nothing) -> Dict

Materialize `root` on `model` (a spec or a term list) at every point `p` of `points` with parameters `θ(p)`, each
point warm-started from the previous one (`previous`), timing each with [`timed_point`](@ref); records the lowest
`NLEVELS` energies and the transitions ω₀ⱼ, and appends the run to `rec`.
A failure at a point is recorded and ends the sweep.
"""
function sweep!(rec::Recorder, label::AbstractString, root::Node, model, points, θ; settings=Dict(), terms_kw=(;))
    run = Dict{String,Any}("label" => label, "settings" => settings, "flat_dim" => string(flat_dim(root)), "points" => Any[])
    prev = nothing
    for p in points
        local sys, r
        try
            sys, r = timed_point(() -> materialize(root, model, θ(p); previous=prev, terms_kw...))
        catch err
            msg = sprint(showerror, err)
            println("$label  p = $p  FAILED: ", first(msg, 400))
            push!(run["points"], Dict("point" => p, "error" => first(msg, 4000)))
            break
        end
        E = energies(sys)[1:min(NLEVELS, end)]
        r["point"] = p
        r["energies"] = E
        r["transitions"] = E[2:end] .- E[1]
        push!(run["points"], r)
        rt = r["nodes"][end]
        @printf("%-40s p = %-6s E1-E0 = %.6f GHz  %7.2f s (first %.1f)  root %d%s\n", label, string(p), E[2] - E[1],
                r["seconds"], r["first_seconds"], r["root_dim"],
                rt["n_matvec"] === nothing ? "" : @sprintf(", %d it, %d matvec", rt["iterations"], rt["n_matvec"]))
        flush(stdout)
        prev = sys
    end
    push!(rec.record["runs"], run)
    save(rec)
    return run
end

# ---- accuracy -----------------------------------------------------------------------------------------------------

_ok(p) = haskey(p, "energies")
_at(run, pt) = (i = findfirst(q -> q["point"] == pt && _ok(q), run["points"]); i === nothing ? nothing : run["points"][i])

"""
    deviation(a, b; k=NLEVELS) -> Float64

The largest change of a transition Eⱼ − Eᵢ (i < j ≤ k) between runs `a` and `b`, over the points both completed,
in GHz; `Inf` when they share no point.
"""
function deviation(a, b; k::Int=NLEVELS)
    worst, n = 0.0, 0
    for p in a["points"]
        _ok(p) || continue
        q = _at(b, p["point"])
        q === nothing && continue
        Ea, Eb = p["energies"], q["energies"]
        m = min(k, length(Ea), length(Eb))
        for i in 1:m, j in i+1:m
            worst = max(worst, abs((Ea[j] - Ea[i]) - (Eb[j] - Eb[i])))
        end
        n += 1
    end
    return n == 0 ? Inf : worst
end

"""
    ladder_summary!(rec, name, runs; target, reference=nothing) -> Dict

The accuracy rule of bench/README.md on a ladder of `runs` (each refining the last): the change between consecutive
rungs, the accurate rung (the first whose change to the next is below `target`, GHz), its time per point and its
check's, and, with a `reference` run, every rung's deviation from it (the accurate rung must lie within `target`).
"""
function ladder_summary!(rec::Recorder, name::AbstractString, runs; target::Real, reference=nothing)
    complete(r) = all(_ok, r["points"]) && !isempty(r["points"])
    per_point(r) = median([p["seconds"] for p in r["points"] if _ok(p)][min(2, end):end])   # warm points
    rungs = Any[]
    for (i, r) in enumerate(runs)
        d = Dict{String,Any}("label" => r["label"], "complete" => complete(r), "seconds_per_point" => per_point(r),
                             "change_to_next" => i < length(runs) ? deviation(r, runs[i+1]) : nothing)
        reference === nothing || (d["deviation_from_reference"] = deviation(r, reference))
        push!(rungs, d)
    end
    i = findfirst(j -> j < length(runs) && rungs[j]["complete"] && rungs[j+1]["complete"] &&
                       rungs[j]["change_to_next"] < target, eachindex(rungs))
    s = Dict{String,Any}("ladder" => name, "target_GHz" => target, "rungs" => rungs,
                         "reference" => reference === nothing ? nothing : reference["label"])
    if i === nothing
        s["accurate"] = nothing
        println("ladder $name: no rung reaches $(target * 1e3) MHz")
    else
        s["accurate"] = rungs[i]["label"]
        s["seconds_per_point"] = rungs[i]["seconds_per_point"]
        s["check_seconds_per_point"] = rungs[i+1]["seconds_per_point"]
        ok_ref = reference === nothing || rungs[i]["deviation_from_reference"] < target
        s["within_reference"] = reference === nothing ? nothing : ok_ref
        @printf("ladder %s: accurate rung %s (change %.2e GHz < %.0e), %.2f s per point (check %.2f s)%s\n", name,
                rungs[i]["label"], rungs[i]["change_to_next"], target, s["seconds_per_point"], s["check_seconds_per_point"],
                reference === nothing ? "" : @sprintf(", %.2e GHz from %s%s", rungs[i]["deviation_from_reference"],
                                                      reference["label"], ok_ref ? "" : "  ** OUTSIDE TARGET **"))
    end
    for d in rungs
        @printf("   %-40s change to next %-10s  %8.2f s/pt%s\n", d["label"],
                d["change_to_next"] === nothing ? "-" : @sprintf("%.2e", d["change_to_next"]), d["seconds_per_point"],
                haskey(d, "deviation_from_reference") ? @sprintf("   vs ref %.2e", d["deviation_from_reference"]) : "")
    end
    push!(rec.record["summaries"], s)
    save(rec)
    return s
end
