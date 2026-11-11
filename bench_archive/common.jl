# Shared setup for the benchmark scripts: environment knobs, device selection, synchronized
# timing, independent eigenpair validation, and the JSON record.
#
#   BENCH_DEVICE   cpu | gpu        (default cpu; gpu loads CUDA and cuTENSOR)
#   BENCH_NEV      parent states    (default 120; 12 with BENCH_SMOKE=1 in part B, 6 in part A)
#   BENCH_REPEATS  timed samples    (default 3), each after one untimed warm-up
#   BENCH_SMOKE    1 for reduced sizes
#   BENCH_OUT      JSON output path (default: next to results.md)
#
# CPU runs use BLAS.set_num_threads(BENCH_BLAS_THREADS) (default 8); GPU runs use one BLAS thread.

using LinearAlgebra, Printf, Dates, Random, Logging
using JSON3
using KronFreeCircuits
using KronFreeCircuits: AxisTerm, LOBPCG, lobpcg_solve

const DEVICE = get(ENV, "BENCH_DEVICE", "cpu")
DEVICE in ("cpu", "gpu") || error("BENCH_DEVICE must be cpu or gpu, got $DEVICE")
const SMOKE = get(ENV, "BENCH_SMOKE", "0") == "1"
const REPEATS = parse(Int, get(ENV, "BENCH_REPEATS", "3"))
const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "docs", "spikes", "2026-10-17-plan2-benchmark"))

if DEVICE == "gpu"
    using CUDA, cuTENSOR
    CUDA.functional() || error("BENCH_DEVICE=gpu but CUDA is not functional on this machine")
    BLAS.set_num_threads(1)
else
    BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_BLAS_THREADS", "8")))
end
const ON = DEVICE == "gpu" ? CuArray : Array

devsync() = DEVICE == "gpu" ? CUDA.synchronize() : nothing

"""
    timed(f) -> (seconds, value)

Wall time of `f()` with a device synchronization before and after (GC run outside the timing).
"""
function timed(f)
    GC.gc()
    devsync()
    t0 = time_ns()
    v = f()
    devsync()
    return (time_ns() - t0) / 1e9, v
end

# The host operator of `H` (validation applies H on the host, independently of the device path).
host(H::KronFreeOperator) = H.on === Array ? H : KronFreeCircuits._on_storage(H, Array; capacity=1)

"""
    validate(H, E, V; tol) -> NamedTuple

`check_eigenpairs` semantics without Test: true residuals `‖H v_j − E_j v_j‖` recomputed with the
host operator, and orthonormality `‖V'V − I‖`; `ok` requires every residual `< tol` and
orthonormality `< 1e-10`.
"""
function validate(H::KronFreeOperator, E::AbstractVector, V::AbstractMatrix; tol)
    Hh = host(H)
    V = Array(V)
    res = zeros(length(E))
    for c0 in 1:32:length(E)                 # blocks of 32 columns bound the host scratch
        cols = c0:min(c0 + 31, length(E))
        HV = Hh * V[:, cols]
        for (k, j) in enumerate(cols)
            res[j] = norm(view(HV, :, k) .- E[j] .* view(V, :, j))
        end
    end
    ortho = norm(V' * V - I)
    return (max_residual=maximum(res), ortho=ortho, ok=all(<(tol), res) && ortho < 1e-10)
end

iterations(res) = size(res.residual_history, 2) - 1

function median_(x)
    s = sort(x)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

summary_line(t) = isempty(t) ? "no samples" :
    @sprintf("median %.3f s (min %.3f, max %.3f, n=%d)", median_(t), minimum(t), maximum(t), length(t))

function environment()
    pkgs = Dict{String,Any}()
    for (id, m) in Base.loaded_modules
        id.name in ("KronFreeCircuits", "ITensors", "NDTensors", "LOBPCGEigensolver", "KernelAbstractions",
                    "KrylovKit", "CUDA", "cuTENSOR", "JSON3", "Adapt") || continue
        pkgs[id.name] = string(pkgversion(m))
    end
    env = Dict{String,Any}("host" => gethostname(), "date" => string(now()), "julia" => string(VERSION),
                           "cpu" => Sys.cpu_info()[1].model, "julia_threads" => Threads.nthreads(),
                           "blas_threads" => BLAS.get_num_threads(), "blas" => string(BLAS.get_config()),
                           "device" => DEVICE, "smoke" => SMOKE, "repeats" => REPEATS, "packages" => pkgs,
                           "commit" => get(ENV, "BENCH_COMMIT", ""))
    if DEVICE == "gpu"
        env["gpu"] = CUDA.name(CUDA.device())
        env["cuda_runtime"] = string(CUDA.runtime_version())
        env["cuda_driver"] = string(CUDA.driver_version())
    end
    return env
end

# Forwards every message to the console logger and counts warnings by the first 60 characters of
# their message (a message with `maxlog` is counted only while the console logger still shows it).
struct CountingLogger <: AbstractLogger
    inner::AbstractLogger
    counts::Dict{String,Int}
end
Logging.min_enabled_level(l::CountingLogger) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::CountingLogger, args...) = Logging.shouldlog(l.inner, args...)
Logging.catch_exceptions(l::CountingLogger) = Logging.catch_exceptions(l.inner)
function Logging.handle_message(l::CountingLogger, level, message, args...; kwargs...)
    if level >= Logging.Warn
        key = first(string(message), 60)
        l.counts[key] = get(l.counts, key, 0) + 1
    end
    return Logging.handle_message(l.inner, level, message, args...; kwargs...)
end
const WARNINGS = Dict{String,Int}()
global_logger(CountingLogger(ConsoleLogger(stderr), WARNINGS))

function write_json(default_name, record)
    record["warning_counts"] = WARNINGS
    path = get(ENV, "BENCH_OUT", joinpath(RESULTS_DIR, default_name))
    mkpath(dirname(path))
    open(io -> JSON3.pretty(io, record), path, "w")
    println("wrote ", path)
    return path
end
