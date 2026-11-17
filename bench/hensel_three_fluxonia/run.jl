# Benchmark: three fluxonia with two tunable couplers (Hensel et al., arXiv:2611.05627v1 Sec. V), at the paper's working
# point (qubits at Φ_0/2, couplers unbiased), lowest eight levels. No flat converged reference is possible (the paper's
# own estimate for exact HD: 2.2 TB, 66 h), so accuracy is judged by the ladder's own convergence and against the paper's
# Table tab:q3levels (printed to 1 MHz).
#
#   julia --project=bench -t 8 --gcthreads=1 bench/hensel_three_fluxonia/run.jl
#
# The circuit comes from circuit-quantization: quantize.py writes specs/three_fluxonia_<frame>.json, which this script
# loads (crosscheck/ holds an independent hand-built construction that agrees with the specs to 1e-10 GHz).
#
# Knobs: those of ../common.jl, and
#   HENSEL3_FRAME   the spec to load: rule (default: the shear of shear_rule.md), full, none, or archive (the
#                   hand-picked fractions of bench_archive/hensel_three_fluxonia: f1 0.35, f2 0.40, f3 0.60, f4 0.80,
#                   f5 0.85, f6 0.95 of the full shear on every junction)
#   HENSEL3_RUNGS   "kM,kM2;kM,kM2;..." of the four-level tree (default 240,240;480,480;480,960)
#   HENSEL3_K       "k1j,k1f,k2,kf" (default 40,20,60,4)
#   HENSEL_TOL      LOBPCG tolerance, GHz (default 1e-6)
#   HENSEL3_QGRID   qubit grid "a,N": ±aπ with N points (default 5,301)
#   HENSEL3_CGRID   coupler grid "a,N": ±aπ with N points (default 4,241)
#   HENSEL3_STAGE   ladder (default: the four-level tree) or flat (the flat reference, trees.jl `flat_root`)
#   HENSEL3_FLAT    "kq,kc,kf;..." rungs of the flat reference (qubit-pair, coupler-pair kept states, loose Fock levels;
#                   default 16,32,4;24,48,4)
#   HENSEL3_GUARD   LOBPCG guard columns at the flat root (default 4; 0 saves memory)
#   HENSEL3_DRYRUN  1: print each flat rung's dimension and estimated memory, solve nothing

include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "trees.jl"))

const FRAME = get(ENV, "HENSEL3_FRAME", "rule")
const REC = Recorder("hensel_three_fluxonia", @__DIR__)

const SPEC, B = load_frame(FRAME)
const PARTNER, LOOSE = best_pairing(B)
println("frame $FRAME: pairs ", join(["($J, $(PARTNER[J]))" for J in JNAMES], " "), ", loose $LOOSE")
REC.record["environment"]["frame"] = FRAME
REC.record["environment"]["spec"] = "specs/three_fluxonia_$FRAME.json"
REC.record["environment"]["shear"] = read(joinpath(@__DIR__, "specs", "three_fluxonia_$(FRAME)_shear.json"), String)
REC.record["environment"]["pairing"] = Dict(string(k) => string(v) for (k, v) in PARTNER)
REC.record["environment"]["tol"] = TOL
REC.record["published"] = Dict(
    "transitions_GHz" => PAPER, "stiff_shift_GHz" => PAPER_STIFF_SHIFT,
    "source" => "arXiv:2611.05627v1 Table tab:q3levels (appC), cap-128 DMRG at 12 Fock levels on the low modes, 4 on the stiff",
    "wall" => Dict("cap 16" => "27 min", "cap 32" => "81 min", "cap 64" => "4.4 h", "cap 128" => "13 h", "study" => "19 h",
                   "exact HD (estimate, not run)" => "66 h, 2.2 TB"),
    "memory" => Dict("cap 16" => "0.97 GB", "cap 32" => "1.3 GB", "cap 64" => "2.4 GB", "cap 128" => "5.8 GB"),
    "hardware" => "one node, Intel Xeon 6517P, 16 physical cores, 230 GB (numbers.tex, appD)")

const STAGE = get(ENV, "HENSEL3_STAGE", "ladder")
if STAGE == "flat"
    frungs = SMOKE ? [(6, 8, 2), (8, 10, 2)] :
             [Tuple(parse.(Int, split(r, ","))) for r in split(get(ENV, "HENSEL3_FLAT", "16,32,4;24,48,4"), ";")]
    guard = parse(Int, get(ENV, "HENSEL3_GUARD", "4"))
    for (kq, kc, kf) in frungs
        @printf("flat rung kq %d kc %d kf %d: dimension %.3e, estimated eigensolver memory %.1f GB (guard %d)\n", kq, kc, kf,
                Float64(flat_dim(kq, kc, kf)), Float64(flat_bytes(kq, kc, kf; compute=NLEVELS, guard)) / 1e9, guard)
    end
    flush(stdout)
    get(ENV, "HENSEL3_DRYRUN", "0") == "1" && exit(0)
    flat = Any[]
    for (kq, kc, kf) in frungs
        root = flat_root(B, PARTNER, LOOSE; kq, kc, kf, compute=NLEVELS, guard, smoke=SMOKE)
        run = sweep!(REC, "flat $FRAME kq $kq kc $kc kf $kf", root, SPEC, [0.0], θ_work;
                     settings=Dict("tree" => "flat", "frame" => FRAME, "kq" => kq, "kc" => kc, "kf" => kf, "guard" => guard,
                                   "dim" => string(flat_dim(kq, kc, kf))))
        p = run["points"][end]
        if haskey(p, "transitions")
            tr = p["transitions"]
            p["deviation_from_paper_MHz"] = 1e3 .* (tr[1:7] .- PAPER)
            @printf("   transitions (GHz) %s\n   vs paper (MHz)    %s\n", join((@sprintf("%.5f", x) for x in tr[1:7]), " "),
                    join((@sprintf("%+.2f", x) for x in p["deviation_from_paper_MHz"]), " "))
            flush(stdout)
        end
        push!(flat, run)
    end
    length(flat) >= 2 && ladder_summary!(REC, "flat, $FRAME frame", flat; target=1e-4)
    println("results: $(REC.out)/results.json")
    exit(0)
end

k1j, k1f, k2, kf = SMOKE ? (8, 6, 10, 3) : parse.(Int, split(get(ENV, "HENSEL3_K", "40,20,60,4"), ","))
rungs = SMOKE ? [(20, 20), (20, 30)] :
        [Tuple(parse.(Int, split(r, ","))) for r in split(get(ENV, "HENSEL3_RUNGS", "240,240;480,480;480,960"), ";")]
ladder = Any[]
for (kM, kM2) in rungs
    root = fourlevel_root(B, PARTNER, LOOSE; k1j, k1f, k2, kM, kM2, kf, compute=NLEVELS, smoke=SMOKE)
    run = sweep!(REC, "fourlevel $FRAME kM $kM kM2 $kM2", root, SPEC, [0.0], θ_work;
                 settings=Dict("tree" => "four-level", "frame" => FRAME, "k1j" => k1j, "k1f" => k1f, "k2" => k2,
                               "kM" => kM, "kM2" => kM2, "kf" => kf))
    p = run["points"][end]
    if haskey(p, "transitions")
        tr = p["transitions"]
        dev = 1e3 .* (tr[1:7] .- PAPER)
        p["deviation_from_paper_MHz"] = dev
        @printf("   transitions (GHz) %s\n   vs paper (MHz)    %s\n", join((@sprintf("%.4f", x) for x in tr[1:7]), " "),
                join((@sprintf("%+.2f", x) for x in dev), " "))
        flush(stdout)
    end
    push!(ladder, run)
end
save(REC)
length(ladder) >= 2 && ladder_summary!(REC, "four-level, $FRAME frame", ladder; target=1e-3)
println("results: $(REC.out)/results.json")
