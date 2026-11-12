const ON = Array
cd(joinpath(ENV["HOME"], "src/KronFreeCircuits-bench"))
include(joinpath(ENV["HOME"], "src/KronFreeCircuits-bench/bench/hensel_three_fluxonia/crosscheck/julia_construction.jl"))
using Logging; global_logger(ConsoleLogger(stderr, Logging.Error))
θs = (varphi_q1=π, varphi_q2=π, varphi_q3=π, varphi_c1=0.0, varphi_c2=0.0)
for frame in ("none", "full", "rule", "archive")
    spec = load_spec("bench/hensel_three_fluxonia/specs/three_fluxonia_$frame.json")
    names = [m.name for m in spec.modes]
    names == LC.names || error("mode order differs: $names")
    ECs = [real(spec.E_C[i, j](θs)) for i in 1:11, j in 1:11]
    ELs = [real(spec.E_L[i, j](θs)) for i in 1:11, j in 1:11]
    A = frame == "none" ? zeros(6, 5) : frame == "full" ? (r = rule_shear(full=true); r[1]) :
        frame == "rule" ? rule_shear()[1] : rule_shear(full=true)[1] .* [0.35, 0.40, 0.60, 0.80, 0.85, 0.95]
    B = frame_of(A)
    # align each frame mode's scale and sign (a convention): EL_spec = D EL_jul D, EC_spec = D⁻¹ EC_jul D⁻¹
    d = ones(11)
    for k in 6:11
        d[k] = sqrt(ELs[k, k] / B.EL[k, k])
        J = argmax(abs.(B.EL[k, 1:5]) .+ abs.(B.EC[k, 1:5]))
        ref = abs(B.EL[k, J]) > abs(B.EC[k, J]) ? (ELs[k, J], B.EL[k, J]) : (ECs[k, J], B.EC[k, J])
        sign(ref[1]) != sign(ref[2]) && (d[k] = -d[k])
    end
    D = Diagonal(d)
    dEL = maximum(abs, ELs - D * B.EL * D); dEC = maximum(abs, ECs - inv(D) * B.EC * inv(D))
    @printf("%-8s frame-mode scale factors |d| %s;  max |ΔE_L| %.1e GHz (of %.1f), max |ΔE_C| %.1e GHz (of %.2f)\n", frame,
            join(round.(abs.(d[6:11]); digits=3), " "), dEL, maximum(abs, ELs), dEC, maximum(abs, ECs))
    if frame == "rule"
        println("   cosines in the spec: ", join([string(c) for c in spec.cosines], " | ")[1:min(end, 600)])
        # same small tree from the spec and from the Julia terms: compare levels
        for (lbl, Bx, model, θ) in (("spec", nothing, spec, θs), ("julia", B, terms_of(B), θ_work()))
        end
        partner, loose = best_pairing(B)
        Bs = (; B..., σ=(8 .* diag(ECs) ./ diag(ELs)) .^ (1 / 4) ./ sqrt(2))      # spec frame: its own zero-point spreads
        rs = fourlevel_root(Bs, partner, loose; k1j=24, k1f=12, k2=30, kM=60, kM2=60, kf=3)
        rj = fourlevel_root(B, partner, loose; k1j=24, k1f=12, k2=30, kM=60, kM2=60, kf=3)
        Es = energies(materialize(rs, spec, θs))[1:8]
        Ej = energies(materialize(rj, terms_of(B), θ_work()))[1:8]
        @printf("   small tree from the spec:  %s\n   small tree from Julia:     %s\n   max |Δ transition| %.2e GHz, |ΔE0| %.2e GHz\n",
                join((@sprintf("%.6f", x) for x in Es[2:8] .- Es[1]), " "), join((@sprintf("%.6f", x) for x in Ej[2:8] .- Ej[1]), " "),
                maximum(abs, (Es[2:8] .- Es[1]) .- (Ej[2:8] .- Ej[1])), abs(Es[1] - Ej[1]))
    end
end
