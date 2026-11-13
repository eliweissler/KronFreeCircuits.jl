# The single-mode fluxonium against the array DMRG (Di Paolo et al., arXiv:1912.01018, Fig. 3): for each z, the
# circuit-quantization spec of quantize_single.py solved at φ_ext = π on a phase grid, ω01, and the phase-slip charge
# dispersion of the paper's Eq. 6 (superfluxo.tex l. 180, `eq: charge dispersion total`):
#   |Δω01| = 2 N ε0 |⟨T⟩_1 − ⟨T⟩_0|,   T = exp(−2πi n) (a 2π displacement of φ),
#   ε0 = 8√2 ħω_p exp(−4/(π z)) / √(π z)   (l. 173, the transmon band width of one array junction, ω_p/2π = 12.5 GHz).
# The bare single-mode model has no offset charge; Eq. 6 is how it acquires the dispersion.
#
#   julia --project=bench bench/array_fluxonium/single_mode.jl        (writes results/single_mode_N40.json)

using KronFreeCircuits, Printf, LinearAlgebra
import JSON3

const NJ = 40
const FP = 12.5                                     # GHz
const ZS = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
# spacing 2π/40 (so exp2pin(-1) is an exact 40-point shift), range ±10π: no kept state near the edge
const GRID = PhaseGrid(-10π, 10π, 401; order=8)
ε0(z) = 8sqrt(2) * FP * exp(-4 / (π * z)) / sqrt(π * z)

warnings = Ref(0)
struct CountEdge <: Base.CoreLogging.AbstractLogger
    inner::Base.CoreLogging.AbstractLogger
end
using Logging
Logging.min_enabled_level(l::CountEdge) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::CountEdge, a...) = Logging.shouldlog(l.inner, a...)
function Logging.handle_message(l::CountEdge, level, msg, a...; kw...)
    startswith(string(msg), "PhaseGrid on mode") && (warnings[] += 1)
    Logging.handle_message(l.inner, level, msg, a...; kw...)
end
global_logger(CountEdge(ConsoleLogger(stderr)))

# ε0 exactly: the ground-band width E0(n_g = ½) − E0(n_g = 0) of one array junction's transmon
# 4 E_C,ii (n − n_g)² − E_J cos θ, with E_C,ii and E_J from the array's circuit-quantization spec (quantize.py)
function eps0_exact(z)
    aspec = load_spec(joinpath(@__DIR__, "specs", @sprintf("array_N%d_z%.3f.json", NJ, z)))
    ec = real(aspec.E_C[1, 1]((n_g=0.0, varphi_ext=π)))
    ej = real(only(c for c in aspec.cosines if count(!=(0), c.omega) == 1 && c.omega[1] == 1).energy((n_g=0.0, varphi_ext=π)))
    e0(ng) = (ns = -30:30; eigmin(SymTridiagonal(4ec .* (ns .- ng) .^ 2, fill(-ej / 2, length(ns) - 1))))
    return e0(0.5) - e0(0.0), ec, ej
end
koch(z) = ε0(z) / sqrt(π)                           # Koch et al.'s asymptotic band width: the printed Eq. 5 over √π

rows = Any[]
@printf("%6s %9s %9s %10s %10s %11s %11s %11s %11s %11s\n", "z", "E_L", "ω01", "<T>_0", "<T>_1", "ε0 printed", "ε0 Koch",
        "ε0 exact", "Eq.6 print", "Eq.6 exact")
for z in ZS
    spec = load_spec(joinpath(@__DIR__, "specs", @sprintf("single_N%d_z%.3f.json", NJ, z)))
    node = Node([:phi => GRID]; solver=Dense(), compute=6, keep=lowest(6), ports=[(:phi, exp2pin(-1))])
    sys = materialize(node, spec, (varphi_ext=π,))
    E = energies(sys)
    T = port(sys, :phi, exp2pin(-1))
    t0, t1 = T[1, 1], T[2, 2]
    dω = 2NJ * ε0(z) * abs(t1 - t0)
    ex, ec, ej = eps0_exact(z)
    dωx = 2NJ * ex * abs(t1 - t0)
    EL = real(spec.E_L[1, 1]((varphi_ext=π,)))
    @printf("%6.3f %9.5f %9.6f %10.3e %10.3e %11.4e %11.4e %11.4e %11.4e %11.4e\n", z, EL, E[2] - E[1], real(t0), real(t1),
            ε0(z), koch(z), ex, dω, dωx)
    push!(rows, Dict("z" => z, "E_L" => EL, "omega01" => E[2] - E[1], "T0" => [real(t0), imag(t0)],
                     "T1" => [real(t1), imag(t1)], "eps0_printed" => ε0(z), "eps0_koch" => koch(z), "eps0_exact" => ex,
                     "site_EC" => ec, "site_EJ" => ej,
                     "delta_omega01_eq6_printed" => dω, "delta_omega01_eq6_exact" => dωx))
end
println("range warnings: ", warnings[])
mkpath(joinpath(@__DIR__, "results"))
open(io -> JSON3.pretty(io, JSON3.write(Dict("NJ" => NJ, "grid" => "PhaseGrid(-10π, 10π, 401; order=8)",
                                             "range_warnings" => warnings[], "rows" => rows))),
     joinpath(@__DIR__, "results", "single_mode_N$NJ.json"), "w")
