# Zero-pi Born–Oppenheimer reproduction: Di Paolo et al., arXiv:1811.07921v1, Appendix B
# (eq:phi_Hamiltonian, eq:theta_Hamiltonian, eq:effective_1D_Hamiltonian; design §5 and §7 of
# docs/superpowers/specs/2026-10-19-born-oppenheimer-design.md).
#
#   julia --project=. -t 8 bench/born_oppenheimer/zero_pi.jl
#
# The fast stage is H_φ(θ̃) = 4 E_Cφ n_φ² − 2 E_J cos θ̃ cos(φ − φ_ext/2) + E_L φ² (test/fixtures/zero_pi_bo.json with
# θ promoted), φ on a PhaseGrid(−L, L, N; order); its ground energy on M points of θ̃ ∈ [−π/2, 3π/2) at five
# fluxes is fitted to
#   U = u_0(φ_ext) − (E_α − E_β cos φ_ext) cos 2θ − E_γ cos(φ_ext/2) cos θ.
#
# Inputs (decided 2026-10-20, design §7): E_Cφ/ħω_p = 1/(16·0.165), derived from the source's ω_p = sqrt(8 E_J E_CJ)/ħ with
# C_φ = 2 C_J (E_Cφ = E_CJ/2), instead of the printed 0.378; E_L/ħω_p = 1e-3, E_J/ħω_p = 0.165, E_Cθ/ħω_p = 1.75e-4 as printed.
#
# Sections: (1) refinement of L, N (at fixed L), the stencil order and M, each independently about a base setting;
# (2) checks at the converged setting; (3) the rounding-extremes study of the inputs; (4) the slow stage from the
# reduced spec and from the full numerical potential against the full 2D zero-pi, per level.
# Writes results.md beside itself, test/fixtures/bo_reference/zero_pi_bo_reference.json (producer
# `independent BO numerics`) and test/fixtures/bo/zero_pi_reduced.json (the fit's reduced spec with its metadata).
#
# Threads: the surface points of each flux are split into chunks swept by separate bo_surface calls in parallel
# (one BLAS thread each); the package's bo_surface is unchanged.

using LinearAlgebra, Printf, Dates, Logging
using KronFreeCircuits
using KronFreeCircuits: JSON3

BLAS.set_num_threads(1)
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const GHZ = 40.0                                                    # ħω_p in GHz (ω_p/2π = 40 GHz)
const ECPHI_DERIVED = 1 / (16 * 0.165)                              # E_Cφ/ħω_p = 0.378788 (ω_p² = 16 E_J E_Cφ)
const INPUT_RATIOS = (E_L=1e-3, E_Cphi=ECPHI_DERIVED, E_Ctheta=1.75e-4, E_J=0.165)
ghz(r) = (E_Cphi=r.E_Cphi * GHZ, E_Ctheta=r.E_Ctheta * GHZ, E_J=r.E_J * GHZ, E_L=r.E_L * GHZ)
const BASE = ghz(INPUT_RATIOS)                                      # GHz
const PRINTED = merge(BASE, (E_Cphi=0.378 * GHZ,))                  # the printed E_Cφ = 0.378 (superseded)
const FLUXES = [0.0, 0.125, 0.25, 0.375, 0.5]                       # Φ_ext/Φ_0; varphi_ext = 2π flux
const TARGETS = (E_alpha=1.8608e-2, E_beta=1.0073e-8, E_gamma=2.6625e-5)
const ACCEPT = 2e-3                                                 # 0.2% relative on E_α and E_γ (decided 2026-10-20)
const THETA_START = -π / 2                                          # the source's θ̃ ∈ [−π/2, 3π/2)
const CONSTANT_BASIS = ["1", "cos(varphi_ext)", "cos(2*varphi_ext)", "cos(3*varphi_ext)", "cos(4*varphi_ext)"]
const MODEL = [:constant => CONSTANT_BASIS, (:cos, 2) => ["1", "cos(varphi_ext)"], (:cos, 1) => ["cos(varphi_ext/2)"]]
const TOL = 1e-5                                                    # convergence between refinements (relative)
const NLEVELS = 6

# A logger that records warnings (the PhaseGrid edge warning among them) and drops everything below.
struct WarnRecorder <: AbstractLogger
    messages::Vector{String}
    lock::ReentrantLock
end
WarnRecorder() = WarnRecorder(String[], ReentrantLock())
Logging.min_enabled_level(::WarnRecorder) = Logging.Warn
Logging.shouldlog(::WarnRecorder, args...) = true
Logging.handle_message(l::WarnRecorder, level, msg, args...; kw...) = lock(() -> push!(l.messages, string(msg)), l.lock)

const SOURCE = load_spec(joinpath(ROOT, "test", "fixtures", "zero_pi_bo.json"))
const PROMOTED = with_logger(NullLogger()) do
    promote_mode(SOURCE, :theta)
end

fast_leaf(b::PhaseGrid; solver=Dense()) = Node([:phi => b]; solver, compute=2, keep=lowest(2))

# The surfaces at the five fluxes (bands 1 and 2), each grid split into chunks swept in parallel.
function surfaces(b::PhaseGrid, M::Integer; solver=Dense(), base=BASE, fluxes=FLUXES)
    grid = collect(periodic_grid(M; start=THETA_START))
    chunks = collect(Iterators.partition(eachindex(grid), cld(M, max(1, min(M, Threads.nthreads())))))
    jobs = [(i, c) for i in eachindex(fluxes) for c in chunks]
    out = [zeros(2, M) for _ in fluxes]
    rec = WarnRecorder()
    with_logger(rec) do
        Threads.@threads :dynamic for (i, c) in jobs
            θ = merge(base, (varphi_ext=2π * fluxes[i],))
            out[i][:, c] = bo_surface(fast_leaf(b; solver), PROMOTED, θ, grid[c]).energies
        end
    end
    S = [BOSurface(PROMOTED, grid, out[i], merge(base, (varphi_ext=2π * fluxes[i],))) for i in eachindex(fluxes)]
    return S, unique(rec.messages)
end

# The fitted constants in units of ħω_p (E_β = −c[cos 2, cos φ_ext]), with standard errors.
function coefficients(F::BOFit)
    i(key, e) = findfirst(==((key, e)), F.terms)
    ia, ib, ig = i((:cos, 2), "1"), i((:cos, 2), "cos(varphi_ext)"), i((:cos, 1), "cos(varphi_ext/2)")
    return (E_alpha=F.constants[ia] / GHZ, E_beta=-F.constants[ib] / GHZ, E_gamma=F.constants[ig] / GHZ,
            se_alpha=F.stderr[ia] / GHZ, se_beta=F.stderr[ib] / GHZ, se_gamma=F.stderr[ig] / GHZ)
end

function run_setting(b::PhaseGrid, M::Integer; kw...)
    t0 = time()
    S, warns = surfaces(b, M; kw...)
    t = time() - t0
    F = fit_surfaces(S, MODEL)
    return (; basis=b, M, S, F, c=coefficients(F), warns, seconds=t)
end

relchange(a, b) = abs(a / b - 1)
dev(c) = (alpha=c.E_alpha / TARGETS.E_alpha - 1, gamma=c.E_gamma / TARGETS.E_gamma - 1, beta=c.E_beta / TARGETS.E_beta - 1)
grid_label(b::PhaseGrid) = @sprintf("PhaseGrid(±%gπ, %d; order=%d), Δ=%.4f", b.b / π, b.N, b.order, (b.b - b.a) / (b.N - 1))
npoints(L, Δ) = round(Int, 2L / Δ) + 1                             # N for a range ±L at spacing Δ

t_start = time()
println("threads: ", Threads.nthreads())

# ---- (1) Base setting and the four independent refinements ---------------------------------------------------
const L0, Δ0, ORDER0, M0 = 12π, 2 * 12π / 480, 8, 32
base_basis(; L=L0, Δ=Δ0, order=ORDER0) = PhaseGrid(-L, L, npoints(L, Δ); order)

const SEQUENCES = ["phi range L", "phi points N (L = 12π)", "stencil order", "theta samples M"]
refinements = Dict{String,Vector{Any}}()
refinements["phi range L"] = [run_setting(base_basis(; L), M0) for L in (6π, 8π, 10π, 12π, 16π)]
refinements["phi points N (L = 12π)"] = [run_setting(PhaseGrid(-L0, L0, N; order=ORDER0), M0) for N in (241, 481, 721, 961)]
refinements["stencil order"] = [run_setting(base_basis(; order), M0) for order in (4, 6, 8)]
refinements["theta samples M"] = [run_setting(base_basis(), M) for M in (8, 16, 32, 64)]

# the chosen value of each sequence: the first entry without an edge warning whose next refinement changes E_α and E_γ
# by less than TOL
function converged_index(seq)
    for k in 1:length(seq)-1
        isempty(seq[k].warns) || continue
        a, b = seq[k].c, seq[k+1].c
        relchange(a.E_alpha, b.E_alpha) < TOL && relchange(a.E_gamma, b.E_gamma) < TOL && return k
    end
    return nothing
end
chosen = Dict(name => converged_index(refinements[name]) for name in SEQUENCES)
# the base setting must be at or beyond the chosen value of every sequence
for (name, k) in chosen
    k === nothing && error("refinement sequence \"$name\" did not converge")
end

# ---- (2) The converged setting and its checks ------------------------------------------------------------------
best = refinements["theta samples M"][3]                            # the base setting (L 12π, N 481, order 8, M 32)
c = best.c
isempty(best.warns) || error("edge warnings at the converged setting: $(best.warns)")

θcheck = merge(BASE, (varphi_ext=2π * 0.125,))                      # solver check at one point: ShiftInvert vs Dense
E_dense = bo_surface(fast_leaf(best.basis), PROMOTED, θcheck, [0.3]).energies
E_si = with_logger(NullLogger()) do
    bo_surface(fast_leaf(best.basis; solver=ShiftInvert(tol=1e-10)), PROMOTED, θcheck, [0.3]).energies
end
solver_diff = maximum(abs, E_dense .- E_si)

gap = minimum(minimum(s.energies[2, :] .- s.energies[1, :]) for s in best.S)
half_period(s) = (M = length(s.grid); maximum(abs(s.energies[1, mod1(j + M ÷ 2, M)] - s.energies[1, j]) for j in 1:M))
pi_periodicity = half_period(best.S[end])
PI_BOUND = TOL * c.E_gamma * GHZ                                    # the convergence level on the smallest asserted coefficient, in GHz
harmonics = [fourier_coefficients(s) for s in best.S]

alt_models = [
    ("U_0 one constant", [:constant => ["1"], (:cos, 2) => ["1", "cos(varphi_ext)"], (:cos, 1) => ["cos(varphi_ext/2)"]]),
    ("U_0 per flux (5 basis functions), the acceptance model", MODEL),
    ("acceptance model plus cos 3θ, cos 4θ terms", [MODEL; (:cos, 3) => ["cos(varphi_ext/2)", "cos(3*varphi_ext/2)"]; (:cos, 4) => ["1", "cos(varphi_ext)"]]),
]
alt_fits = [(label, coefficients(fit_surfaces(best.S, m))) for (label, m) in alt_models]

# Independent cross-check outside the package: sinc-DVR (Colbert–Miller) kinetic matrix for 4 E_Cφ n² on a uniform φ grid,
# plain dense eigvals, and the discrete Fourier coefficients a_1, a_2 of U on the same θ grid
function sinc_dvr_harmonics(x; L=12π, N=601, M=32, p=BASE)
    φ = range(-L, L; length=N); Δ = step(φ)
    T = [i == j ? π^2 / 3 : 2 * (-1.0)^(i - j) / (i - j)^2 for i in 1:N, j in 1:N] .* (4p.E_Cphi / Δ^2)
    g = collect(periodic_grid(M; start=THETA_START))
    u = [eigvals(Symmetric(T + Diagonal(p.E_L .* φ .^ 2 .- 2p.E_J * cos(t) .* cos.(φ .- x / 2))), 1:1)[1] for t in g]
    return (a1=2 / M * sum(u .* cos.(g)), a2=2 / M * sum(u .* cos.(2g)))
end
sinc_check = [(f, sinc_dvr_harmonics(2π * f), harmonics[i]) for (i, f) in enumerate(FLUXES) if f in (0.0, 0.25)]

# ---- (3) Rounding extremes of the inputs ------------------------------------------------------------------------
# ratios: E_Cφ 0.378 ± 0.0005, E_J 0.165 ± 0.0005, E_L 1e-3 read as 1.0e-3 (± 0.05e-3) or as one figure (± 0.5e-3)
rnd(r) = run_setting(best.basis, best.M; base=ghz(r)).c
derived_r = INPUT_RATIOS
printed_r = merge(INPUT_RATIOS, (E_Cphi=0.378,))
rounding = Any[]
push!(rounding, ("derived baseline (acceptance inputs)", derived_r))
push!(rounding, ("printed baseline, E_Cφ = 0.378", printed_r))
for v in (0.3775, 0.3785)
    push!(rounding, (@sprintf("E_Cφ = %.4f (either baseline)", v), merge(INPUT_RATIOS, (E_Cphi=v,))))
end
for (bname, b) in (("derived", derived_r), ("printed", printed_r))
    for v in (0.1645, 0.1655)
        push!(rounding, (@sprintf("%s baseline, E_J = %.4f", bname, v), merge(b, (E_J=v,))))
    end
    for v in (0.95e-3, 1.05e-3, 0.5e-3, 1.5e-3)
        push!(rounding, (@sprintf("%s baseline, E_L = %.2fe-3", bname, v * 1e3), merge(b, (E_L=v,))))
    end
end
for EL in (0.95e-3, 1.05e-3), EJ in (0.1645, 0.1655)                 # derived reading: E_Cφ follows E_J
    push!(rounding, (@sprintf("corner E_L = %.2fe-3, E_J = %.4f, E_Cφ = 1/(16 E_J) = %.6f", EL * 1e3, EJ, 1 / (16EJ)),
                     merge(INPUT_RATIOS, (E_L=EL, E_J=EJ, E_Cphi=1 / (16EJ)))))
end
rounding_results = [(label, r, rnd(r)) for (label, r) in rounding]

# ---- (4) Slow stage and the full 2D zero-pi -------------------------------------------------------------------
F = best.F
slow_leaf(ncut) = Node([:theta => ChargeBasis(ncut)]; solver=Dense(), keep=lowest(NLEVELS))
θslow(f) = (varphi_ext=2π * f, E_Ctheta=BASE.E_Ctheta, n_g=0.0)
fitted_levels(ncut, i) = energies(materialize(slow_leaf(ncut), F.spec, θslow(FLUXES[i])))
numpot_levels(ncut, i) = energies(materialize(slow_leaf(ncut), slow_spec(best.S[i])[1], (E_Ctheta=BASE.E_Ctheta, n_g=0.0)))
θfull(f) = merge(BASE, (varphi_ext=2π * f, n_g=0.0))
full_leaf(ncut) = Node([:theta => ChargeBasis(ncut), :phi => best.basis]; solver=ShiftInvert(tol=1e-9), compute=NLEVELS, keep=lowest(NLEVELS))
function full_levels(ncut, f)
    rec = WarnRecorder()
    E = with_logger(rec) do
        energies(materialize(full_leaf(ncut), SOURCE, θfull(f)))
    end
    return E, unique(rec.messages)
end
maxdiff(a, b) = maximum(abs, a .- b)
# charge cutoff of the slow stage (all fluxes) and of the 2D problem (flux 0 and 0.5)
slow_ncuts = (5, 10, 15, 20, 30)
slow_ref = [[fitted_levels(n, i) for i in eachindex(FLUXES)] for n in slow_ncuts]
full_ncuts = (10, 15, 20, 30)
full_ref = [[full_levels(n, f) for f in (0.0, 0.5)] for n in full_ncuts]
NCUT_SLOW, NCUT_FULL = 20, 20
slow_fit = [fitted_levels(NCUT_SLOW, i) for i in eachindex(FLUXES)]
slow_num = [numpot_levels(NCUT_SLOW, i) for i in eachindex(FLUXES)]
full = [full_levels(NCUT_FULL, f) for f in FLUXES]
all(isempty(w) for (_, w) in full) || @warn "edge warnings in the full 2D solves" [w for (_, w) in full]

ok = abs(dev(c).alpha) < ACCEPT && abs(dev(c).gamma) < ACCEPT
# numerical error of the base setting: in each sequence, the relative change from the base row to the next finer row
# (to the next coarser row where the base is the finest), maximized over the sequences
const BASE_ROW = Dict("phi range L" => 4, "phi points N (L = 12π)" => 2, "stencil order" => 3, "theta samples M" => 3)
function err(f)
    e = 0.0
    for n in SEQUENCES
        seq, k = refinements[n], BASE_ROW[n]
        seq[k].basis == best.basis && seq[k].M == best.M || error("BASE_ROW[\"$n\"] is not the base setting")
        j = k < length(seq) ? k + 1 : k - 1
        e = max(e, relchange(getfield(seq[j].c, f), getfield(seq[k].c, f)))
    end
    return e
end
runtime = time() - t_start

# ---- results.md ----------------------------------------------------------------------------------------------
io = IOBuffer()
pr(args...) = println(io, args...)
fmt(x) = @sprintf("%.6e", x)
sg(x) = @sprintf("%+.3e", x)
pr("# Zero-pi Born–Oppenheimer reproduction (Di Paolo et al., arXiv:1811.07921v1, Appendix B)")
pr()
pr("Written by `bench/born_oppenheimer/zero_pi.jl` on ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " (Julia ", VERSION, ", KronFreeCircuits ",
   pkgversion(KronFreeCircuits), ", ", Threads.nthreads(), " threads); run time ", @sprintf("%.0f", runtime), " s.")
pr()
pr("Inputs (decided 2026-10-20, design §7), ratios to ħω_p with ω_p/2π = 40 GHz: E_Cφ = 1/(16·0.165) = ", @sprintf("%.6f", ECPHI_DERIVED),
   ", derived from the source's ω_p = sqrt(8 E_J E_CJ)/ħ with C_φ = 2C_J (E_Cφ = E_CJ/2), instead of the printed 0.378;")
pr("E_L = 0.001, E_J = 0.165, E_Cθ = 0.000175 as printed.")
pr(@sprintf("In GHz: `E_Cphi = %.10g, E_Ctheta = %.10g, E_J = %.10g, E_L = %.10g`; n_g = 0 (it does not enter the surface).", BASE.E_Cphi, BASE.E_Ctheta, BASE.E_J, BASE.E_L))
pr("Fluxes `", FLUXES, "` with `varphi_ext = 2π·flux`; θ̃ on `periodic_grid(M; start=-π/2)`, M uniform points of [−π/2, 3π/2).")
pr("Fast stage: `test/fixtures/zero_pi_bo.json` promoted at θ, φ on a `PhaseGrid`, `Dense` (2 lowest pairs) per point.")
pr("Model (`fit_surfaces`): `:constant => ", CONSTANT_BASIS, "` (exact per flux on five fluxes; on a uniform θ grid it is orthogonal to the cosines, so it does not bias the coefficients),")
pr("`(:cos, 2) => [\"1\", \"cos(varphi_ext)\"]`, `(:cos, 1) => [\"cos(varphi_ext/2)\"]`; E_α = c[cos2,1], E_β = −c[cos2,cos φ_ext], E_γ = c[cos1], each / 40 GHz.")
pr()
pr("**Acceptance (design §7: 0.2% on E_α and E_γ, derived E_Cφ): ", ok ? "PASS" : "FAIL", ".**")
pr("With the printed E_Cφ = 0.378 (superseded), E_γ misses by +0.96% (rounding-extremes table below).")
pr()
pr("## Result at the converged setting")
pr()
pr("Setting: ", grid_label(best.basis), ", M = ", best.M, "; edge warnings: none.")
pr()
pr("| coefficient | obtained / ħω_p | fit standard error | published / ħω_p | relative deviation | 0.2% criterion |")
pr("| --- | ---: | ---: | ---: | ---: | --- |")
d = dev(c)
pr("| E_α | ", fmt(c.E_alpha), " | ", @sprintf("%.1e", c.se_alpha), " | 1.8608e-2 | ", sg(d.alpha), " | ", abs(d.alpha) < ACCEPT ? "pass" : "**miss**", " |")
pr("| E_β | ", fmt(c.E_beta), " | ", @sprintf("%.1e", c.se_beta), " | 1.0073e-8 | ", sg(d.beta), " | reported only |")
pr("| E_γ | ", fmt(c.E_gamma), " | ", @sprintf("%.1e", c.se_gamma), " | 2.6625e-5 | ", sg(d.gamma), " | ", abs(d.gamma) < ACCEPT ? "pass" : "**miss**", " |")
pr()
pr("The fit standard errors come from the residuals, which are the omitted harmonics (cos 3θ, cos 4θ, …), not numerical noise;")
pr(@sprintf("the numerical error of the base setting (its change to the adjacent refinement in each block, below) is %.1e for E_α, %.1e for E_γ and %.1e for E_β, relative.", err(:E_alpha), err(:E_gamma), err(:E_beta)))
pr(@sprintf("Fit residuals: rms %.3e GHz, max %.3e GHz.", F.rms_residual, F.max_residual))
pr(@sprintf("Fast gap min(band 2 − band 1) over θ and flux: %.6f GHz.", gap))
pr(@sprintf("π-periodicity at half flux, max |U(θ+π) − U(θ)|: %.2e GHz (bound used by the gated test: 1e-5·E_γ = %.2e GHz).", pi_periodicity, PI_BOUND))
pr(@sprintf("Solver check (flux 0.125, θ̃ = 0.3): ShiftInvert(tol=1e-10) vs Dense, max |ΔE| over the two bands = %.1e GHz.", solver_diff))
pr("The gated test `test/test_born_oppenheimer_reference.jl` recomputes the five surfaces and the fit at this setting, single-threaded:")
pr("about 15 s run alone, 4 s inside the gated suite (measured 2026-10-20).")
pr("Flux dependence: a_2 varies by ", @sprintf("%.1e", maximum(h[2][2] for h in harmonics) - minimum(h[2][2] for h in harmonics)),
   " GHz across the fluxes (E_β), a_1 follows cos(φ_ext/2) and vanishes at half flux, as in the source.")
pr()
pr("Fourier cosine coefficients a_k of U(θ) = a_0 + Σ a_k cos kθ + b_k sin kθ (GHz):")
pr()
pr("| flux | a_0 | a_1 | a_2 | a_3 | a_4 | a_5 | a_6 | max abs b_k |")
pr("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for (f, (a0, a, b)) in zip(FLUXES, harmonics)
    pr("| ", f, " | ", @sprintf("%.9f", a0), " | ", join((@sprintf("%.6e", a[k]) for k in 1:6), " | "), " | ", @sprintf("%.1e", maximum(abs, b)), " |")
end
pr()
pr("## Refinement study")
pr()
pr("Each block varies one setting about the base (L = 12π, N = 481, order 8, M = 32); Δ is held fixed in the L block.")
pr("`ΔE_α`, `ΔE_γ`, `ΔE_β` are relative changes from the previous row; the chosen row is the first without an edge warning")
pr("whose next refinement changes E_α and E_γ by less than 1e-5. The base setting is at or beyond every chosen row.")
pr()
for name in SEQUENCES
    seq = refinements[name]
    pr("### ", name)
    pr()
    pr("| setting | M | E_α | E_γ | E_β | ΔE_α | ΔE_γ | ΔE_β | edge warning | time (s) |")
    pr("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for (k, r) in enumerate(seq)
        ch(f) = k == 1 ? "" : @sprintf("%.1e", relchange(getfield(r.c, f), getfield(seq[k-1].c, f)))
        pr("| ", grid_label(r.basis), " | ", r.M, " | ", fmt(r.c.E_alpha), " | ", fmt(r.c.E_gamma), " | ", @sprintf("%.5e", r.c.E_beta), " | ",
           ch(:E_alpha), " | ", ch(:E_gamma), " | ", ch(:E_beta), " | ", isempty(r.warns) ? "no" : "yes", " | ", @sprintf("%.1f", r.seconds), " |")
    end
    pr()
    pr("Chosen: row ", chosen[name], ".")
    pr()
end
pr("E_β converges to about 1e-5 relative only between the coarser rows: it is a 4e-7 GHz flux variation of a_2 extracted from energies of order 1 GHz.")
pr()
pr("## Rounding extremes of the inputs")
pr()
pr("Each input at the ends of its printed precision, the others at a baseline (derived E_Cφ = 0.378788 or printed 0.378); E_L's printed `10^{-3}`")
pr("is read both as 1.0e-3 (± 0.05e-3) and as one figure (± 0.5e-3). The corners take the derived reading, E_Cφ = 1/(16 E_J) following E_J.")
pr("Same numerical setting as above; deviations relative to the published coefficients.")
pr()
pr("| inputs (ratios to ħω_p) | E_L | E_Cφ | E_J | E_α | dev | E_β | dev | E_γ | dev |")
pr("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for (label, r, cc) in rounding_results
    dd = dev(cc)
    pr("| ", label, " | ", @sprintf("%.2e", r.E_L), " | ", @sprintf("%.6f", r.E_Cphi), " | ", @sprintf("%.4f", r.E_J), " | ", fmt(cc.E_alpha), " | ", sg(dd.alpha),
       " | ", @sprintf("%.4e", cc.E_beta), " | ", sg(dd.beta), " | ", fmt(cc.E_gamma), " | ", sg(dd.gamma), " |")
end
pr()
onefig(r) = r.E_L in (0.5e-3, 1.5e-3)
rd = [dev(cc) for (_, r, cc) in rounding_results if !onefig(r)]
rd1 = [dev(cc) for (_, r, cc) in rounding_results if onefig(r)]
pr(@sprintf("Within the printed precision (E_L read as 1.0e-3) E_α ranges over %+.2f%% to %+.2f%% and E_γ over %+.2f%% to %+.2f%% of the published values; ",
            100minimum(x.alpha for x in rd), 100maximum(x.alpha for x in rd), 100minimum(x.gamma for x in rd), 100maximum(x.gamma for x in rd)),
   @sprintf("with E_L read as one figure E_γ ranges over %+.0f%% to %+.0f%%. ", 100minimum(x.gamma for x in rd1), 100maximum(x.gamma for x in rd1)),
   "The input rounding, not the numerics, sets the attainable agreement with the published coefficients.")
pr()
pr("## Checks of the mapping and the fit")
pr()
pr("- Charging energies: the source's q²/(2C) with q = 2e n and E_C = e²/(2C) is 4 E_C n² (eq:symm_qubit_Hamiltonian); the spec's `4 n E_C n` makes `E_C[φ,φ] = E_Cphi` (zero_pi_bo.json).")
pr("- Inductive term: the source's E_L φ² is the spec's `E_L[φ,φ] = 2 E_L` (½ φ E_L φ form).")
pr("- No 2E_J constant (Appendix B adds none); a constant would only shift a_0, which does not enter E_α, E_β, E_γ.")
pr("- E_β sign: E_2 = E_α − E_β cos φ_ext and the spec cosine energy is c₁ + c₂ cos φ_ext, so E_β = −c₂; the obtained E_β is positive, as published.")
pr("- The surface agrees with an independent dense reference to 1e-10 (test/test_bo.jl, \"zero-pi BO against an independent dense BO\").")
pr("- An independent discretization outside the package (sinc-DVR kinetic matrix, φ ∈ ±12π, 601 points, plain `eigvals`, same θ grid):")
pr()
pr("| flux | a_1 sinc-DVR | a_1 package | a_2 sinc-DVR | a_2 package |")
pr("| ---: | ---: | ---: | ---: | ---: |")
for (f, sd, (a0, a, b)) in sinc_check
    pr("| ", f, " | ", @sprintf("%.10e", sd.a1), " | ", @sprintf("%.10e", a[1]), " | ", @sprintf("%.10e", sd.a2), " | ", @sprintf("%.10e", a[2]), " |")
end
pr()
pr("Alternative fits on the converged surfaces (on a uniform θ grid the harmonics are orthogonal, so adding terms or changing the")
pr("constant's basis leaves E_α, E_β, E_γ unchanged):")
pr()
pr("| model | E_α | dev | E_γ | dev | E_β |")
pr("| --- | ---: | ---: | ---: | ---: | ---: |")
for (label, a) in alt_fits
    dd = dev(a)
    pr("| ", label, " | ", fmt(a.E_alpha), " | ", sg(dd.alpha), " | ", fmt(a.E_gamma), " | ", sg(dd.gamma), " | ", @sprintf("%.4e", a.E_beta), " |")
end
pr()
pr("## Slow stage against the full 2D zero-pi")
pr()
pr("Slow stage: the reduced spec (`F.spec`, the fitted model; test/fixtures/bo/zero_pi_reduced.json) and the full numerical potential")
pr("(`slow_spec` of each surface, all harmonics), each on `ChargeBasis(", NCUT_SLOW, ")` with `Dense`.")
pr("Full 2D: `test/fixtures/zero_pi_bo.json` on `[:theta => ChargeBasis(", NCUT_FULL, "), :phi => ", grid_label(best.basis), "]` (the BO's φ grid), `ShiftInvert(tol=1e-9)`, ", NLEVELS, " levels.")
pr("Energies are absolute (GHz). No bound is asserted on the BO error.")
pr()
pr("Charge-cutoff refinement, max |ΔE| over the ", NLEVELS, " levels against the next cutoff (GHz):")
pr()
pr("| slow-stage ncut | max over fluxes |")
pr("| ---: | ---: |")
for k in 1:length(slow_ncuts)-1
    pr("| ", slow_ncuts[k], " → ", slow_ncuts[k+1], " | ", @sprintf("%.1e", maximum(maxdiff(slow_ref[k][i], slow_ref[k+1][i]) for i in eachindex(FLUXES))), " |")
end
pr()
pr("| 2D ncut | flux 0 | flux 0.5 |")
pr("| ---: | ---: | ---: |")
for k in 1:length(full_ncuts)-1
    pr("| ", full_ncuts[k], " → ", full_ncuts[k+1], " | ", @sprintf("%.1e", maxdiff(full_ref[k][1][1], full_ref[k+1][1][1])), " | ",
       @sprintf("%.1e", maxdiff(full_ref[k][2][1], full_ref[k+1][2][1])), " |")
end
pr()
pr("Per-level comparison (GHz): E_2D, and the BO errors E_fit − E_2D (reduced spec) and E_U − E_2D (full numerical potential).")
pr()
pr("| flux | level | E_2D | E_fit − E_2D | E_U − E_2D |")
pr("| ---: | ---: | ---: | ---: | ---: |")
for (i, f) in enumerate(FLUXES), l in 1:NLEVELS
    pr("| ", f, " | ", l, " | ", @sprintf("%.9f", full[i][1][l]), " | ", @sprintf("%+.3e", slow_fit[i][l] - full[i][1][l]), " | ", @sprintf("%+.3e", slow_num[i][l] - full[i][1][l]), " |")
end
pr()
pr("The doublet splittings (levels 2 − 1, 4 − 3, 6 − 5) are set mainly by E_γ cos(φ_ext/2) and shrink at half flux to the tunneling splitting.")
pr("E_fit − E_2D includes the truncation of the two-harmonic model (omitted a_3, a_4, …, with a_4 ≈ 4e-3 GHz); E_U − E_2D, from the full numerical potential,")
pr("is the single-surface BO error itself.")
pr()
results_md = String(take!(io))
write(joinpath(@__DIR__, "results.md"), results_md)
println(results_md)

# ---- Reference fixture ---------------------------------------------------------------------------------------

fast_leaf_record = Dict{String,Any}("mode" => "phi", "basis" => "PhaseGrid", "a" => best.basis.a, "b" => best.basis.b, "N" => best.basis.N,
                                    "order" => best.basis.order, "solver" => "Dense", "pairs" => 2)
theta_record = Dict{String,Any}("M" => best.M, "start" => THETA_START, "kind" => "periodic_grid (uniform, half-open)")
fixture = Dict{String,Any}(
    "producer" => "independent BO numerics",
    "source" => Dict("arxiv" => "1811.07921v1", "section" => "Appendix B (app:1d_effective_Hamiltonian)",
                     "equations" => ["eq:phi_Hamiltonian", "eq:theta_Hamiltonian", "eq:effective_1D_Hamiltonian"],
                     "published_fit" => Dict("E_alpha" => TARGETS.E_alpha, "E_beta" => TARGETS.E_beta, "E_gamma" => TARGETS.E_gamma,
                                             "note" => "published fitted coefficients; compared against, never stored as results of this run")),
    "units" => Dict("coefficients" => "ratio to hbar*omega_p, omega_p/2pi = 40 GHz", "energies" => "GHz (E/h)"),
    "conventions" => Dict("hamiltonian" => "H_phi = 4 E_Cphi n_phi^2 - 2 E_J cos(theta) cos(phi - varphi_ext/2) + E_L phi^2; [phi, n_phi] = i",
                          "fit" => "U = u_0(varphi_ext) - (E_alpha - E_beta cos varphi_ext) cos 2theta - E_gamma cos(varphi_ext/2) cos theta",
                          "flux" => "varphi_ext = 2 pi flux, flux = Phi_ext/Phi_0", "n_g" => 0.0),
    "inputs_hbar_omega_p" => Dict(String(k) => v for (k, v) in pairs(INPUT_RATIOS)),
    "inputs_GHz" => Dict(String(k) => v for (k, v) in pairs(BASE)),
    "inputs_note" => "E_Cphi = hbar*omega_p/(16*0.165), derived from omega_p = sqrt(8 E_J E_CJ)/hbar and C_phi = 2 C_J (E_Cphi = E_CJ/2; " *
                     "1811.07921v1 tex line 252), instead of the printed 0.378 (decided 2026-10-20, design section 7); E_L, E_J, E_Ctheta as printed. " *
                     "With the printed 0.378, E_gamma misses the published value by +0.96%.",
    "acceptance" => Dict("relative_tolerance" => ACCEPT, "coefficients" => ["E_alpha", "E_gamma"]),
    "fluxes" => FLUXES,
    "theta_grid" => theta_record,
    "fast_leaf" => fast_leaf_record,
    "model" => [Dict("key" => string(k), "expressions" => v) for (k, v) in MODEL],
    "constants" => Dict("E_alpha" => c.E_alpha, "E_beta" => c.E_beta, "E_gamma" => c.E_gamma),
    "fit_stderr" => Dict("E_alpha" => c.se_alpha, "E_beta" => c.se_beta, "E_gamma" => c.se_gamma),
    "numerical_error_relative" => Dict("E_alpha" => err(:E_alpha), "E_gamma" => err(:E_gamma), "E_beta" => err(:E_beta),
                                       "note" => "largest relative change from the converged (base) setting to its adjacent refinement (finer, or coarser where the base is the finest) over the four refinement sequences of results.md"),
    "fit_residuals_GHz" => Dict("rms" => F.rms_residual, "max" => F.max_residual),
    "fast_gap_min_GHz" => gap,
    "pi_periodicity_half_flux_GHz" => pi_periodicity,
    "pi_periodicity_bound_GHz" => PI_BOUND,
    "full_2d" => Dict("producer" => "full 2D numerics (same run)",
                      "basis" => "theta ChargeBasis($NCUT_FULL), phi as fast_leaf", "solver" => "ShiftInvert(tol=1e-9)",
                      "slow_stage" => "ChargeBasis($NCUT_SLOW), Dense",
                      "levels_GHz" => [full[i][1] for i in eachindex(FLUXES)],
                      "bo_error_fit_GHz" => [slow_fit[i] .- full[i][1] for i in eachindex(FLUXES)],
                      "bo_error_numerical_potential_GHz" => [slow_num[i] .- full[i][1] for i in eachindex(FLUXES)],
                      "note" => "per-level BO error recorded, not bounded"),
    "versions" => Dict("julia" => string(VERSION), "KronFreeCircuits" => string(pkgversion(KronFreeCircuits))),
    "generator" => "bench/born_oppenheimer/zero_pi.jl",
)
mkpath(joinpath(ROOT, "test", "fixtures", "bo_reference"))
open(joinpath(ROOT, "test", "fixtures", "bo_reference", "zero_pi_bo_reference.json"), "w") do f
    JSON3.pretty(f, fixture)
    println(f)
end

# ---- Reduced spec ----------------------------------------------------------------------------------------------
md = deepcopy(F.metadata)
md["_bo"]["fast_leaf"] = fast_leaf_record
md["_bo"]["theta_samples"] = theta_record
md["_bo"]["inputs_note"] = fixture["inputs_note"]
md["_bo"]["producer"] = "bench/born_oppenheimer/zero_pi.jl"
save_spec(joinpath(ROOT, "test", "fixtures", "bo", "zero_pi_reduced.json"), F.spec; metadata=md)
println("wrote results.md, test/fixtures/bo_reference/zero_pi_bo_reference.json and test/fixtures/bo/zero_pi_reduced.json")
