# DMRG (ITensorMPS) on an array-fluxonium kronfree-spec written by ../quantize.py (circuit-quantization). Everything
# comes from the spec: H = 4 (n − g)ᵀ E_C (n − g) − Σ_i E_J,i cos(θ_i + o_i) − E_Jb cos(ωᵀθ + o_b) + constant, with
# g = the spec's offsets at the given n_g (full form). It is split as
#   site i:     4 E_C,ii (n_i − w_i)² − E_J,i cos θ_i,     w_i = g_i + Σ_{j≠i} E_C,ij g_j / E_C,ii
#   couplings:  8 E_C,ij n_i n_j  (i < j)
#   black sheep: −(E_Jb/2) (e^{i o_b} Π_i U_i^{ω_i} + h.c.),  U = e^{iθ}
#   constant:   Σ_i 4 E_C,ii (g_i² − w_i²) + Σ_{i<j} 8 E_C,ij g_i g_j + the spec's constant.
# Site basis (BASIS): "eigen" = the D lowest states of each site Hamiltonian (solved by KronFreeCircuits on a one-mode
# spec), or "charge" = ChargeBasis(NCUT). Excited state by penalty (`dmrg(H, [ψ0], …; weight)`).
#
#   julia --project=bench/array_fluxonium/dmrg -t 1 bench/array_fluxonium/dmrg/dmrg_spec.jl
#
# Env: SPEC (path), NG (comma list, default 0,0.5), PHIEXT (default π), NLEV (2 or 3), BASIS, D, NCUT, MAXDIM, SWEEPS, CUTOFF, ETOL,
# WEIGHT, NOISESWEEPS, BLASTHREADS, OUTLEVEL.

using ITensors, ITensorMPS, KronFreeCircuits, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int, get(ENV, "BLASTHREADS", "1")))
const OUTLEVEL = Ref(parse(Int, get(ENV, "OUTLEVEL", "0")))

# per-site matrices, looked up by the site index's "n=<i>" tag
const SITEMATS = Dict{Int,NamedTuple}()
siteno(s::Index) = parse(Int, match(r"n=(\d+)", string(tags(s))).captures[1])
ITensors.op(::OpName"N", ::SiteType"Junc", s::Index) = itensor(SITEMATS[siteno(s)].n, s', dag(s))
ITensors.op(::OpName"U", ::SiteType"Junc", s::Index) = itensor(SITEMATS[siteno(s)].U, s', dag(s))
ITensors.op(::OpName"Ud", ::SiteType"Junc", s::Index) = itensor(Matrix(SITEMATS[siteno(s)].U'), s', dag(s))
ITensors.op(::OpName"H0", ::SiteType"Junc", s::Index) = itensor(SITEMATS[siteno(s)].h0, s', dag(s))
ITensors.state(::StateName"g", ::SiteType"Junc", s::Index) =
    (h = SITEMATS[siteno(s)].h0; v = zeros(dim(s)); v[argmin(diag(h))] = 1; itensor(v, s))

fmt(x) = @sprintf("%.17g", x)

"Site matrices (n, e^{iθ}, h0) of 4 d (n − w)² − E_J cos θ in the charge basis."
function charge_site(ncut, d, w, EJ)
    ns = collect(-ncut:ncut)
    U = zeros(2ncut + 1, 2ncut + 1)
    for k in 1:2ncut
        U[k+1, k] = 1                                  # e^{iθ}|n⟩ = |n+1⟩
    end
    return (n=Matrix(Diagonal(Float64.(ns))), U, h0=Matrix(Diagonal(4d .* (ns .- w) .^ 2)) - EJ / 2 * (U + U'))
end

"The D lowest states of 4 d (n − w)² − E_J cos θ, solved by KronFreeCircuits; (n, e^{iθ}, diagonal energies) in them."
function eigen_site(ncut, D, d, w, EJ)
    json = """
    {"format": "kronfree-spec", "version": 1, "units": "GHz", "parameters": [],
     "modes": [{"name": "t", "type": "c"}], "E_C": [[$(fmt(d))]], "E_L": [[0]], "flux_linear": ["0"],
     "n_g": ["$(fmt(w))"], "cosines": [{"energy": "$(fmt(EJ))", "omega": [1], "offset": "0"}], "constant": "0"}
    """
    node = Node([:t => ChargeBasis(ncut)]; solver=Dense(), keep=lowest(D), ports=[(:t, nport()), (:t, expiphi(1))])
    sys = materialize(node, parse_spec(json), NamedTuple())
    n, U = port(sys, :t, nport()), port(sys, :t, expiphi(1))
    maximum(abs, imag(n)) + maximum(abs, imag(U)) < 1e-10 || error("complex site ports")
    return (n=Matrix(real(n)), U=Matrix(real(U)), h0=Matrix(Diagonal(KronFreeCircuits.energies(sys))))
end

realof(c, θ) = (v = c(θ); abs(imag(v)) < 1e-12 || error("complex coefficient $v"); real(v))

function run(spec, θ; basis, D, ncut, maxdim, sweeps, cutoff, weight)
    N = length(spec.modes)
    EC = [realof(spec.E_C[i, j], θ) for i in 1:N, j in 1:N]
    all(iszero, (realof(spec.E_L[i, j], θ) for i in 1:N, j in 1:N)) || error("the array spec has no inductive terms")
    g = [realof(spec.n_g[i], θ) for i in 1:N]
    site_cos = Dict{Int,Float64}(); bs = nothing
    for c in spec.cosines
        nz = findall(!=(0), c.omega)
        o = realof(c.offset, θ); E = realof(c.energy, θ)
        if length(nz) == 1
            i = only(nz)
            c.omega[i] == 1 && abs(sin(o)) < 1e-12 && cos(o) > 0 || error("site cosine with offset $o or omega $(c.omega)")
            site_cos[i] = E
        else
            abs(sin(o)) < 1e-12 || error("black-sheep offset $o: the Hamiltonian is complex (only varphi_ext = 0 or π here)")
            all(x -> abs(x) == 1, c.omega) || error("black-sheep omega $(c.omega)")
            bs = (; E, sign=cos(o), omega=Int.(c.omega))
        end
    end
    w = [g[i] + sum(EC[i, j] * g[j] for j in 1:N if j != i; init=0.0) / EC[i, i] for i in 1:N]
    const_ = sum(4EC[i, i] * (g[i]^2 - w[i]^2) for i in 1:N) +
             sum(8EC[i, j] * g[i] * g[j] for i in 1:N for j in i+1:N; init=0.0) + realof(spec.constant, θ)
    for i in 1:N
        SITEMATS[i] = basis == "charge" ? charge_site(ncut, EC[i, i], w[i], site_cos[i]) :
                      eigen_site(ncut, D, EC[i, i], w[i], site_cos[i])
    end
    sites = [Index(size(SITEMATS[i].n, 1), "Junc,Site,n=$i") for i in 1:N]
    os = OpSum()
    for i in 1:N
        os += "H0", i
    end
    for i in 1:N, j in i+1:N
        if abs(EC[i, j]) > 0
            os += (8EC[i, j], "N", i, "N", j)
        end
    end
    # −(E_Jb/2)(e^{i o} Π U^{ω} + h.c.), e^{i o} = ±1
    c = -bs.E / 2 * bs.sign
    os += (c, Iterators.flatten((bs.omega[i] == 1 ? "U" : "Ud", i) for i in 1:N)...)
    os += (c, Iterators.flatten((bs.omega[i] == 1 ? "Ud" : "U", i) for i in 1:N)...)
    t0 = time(); H = MPO(os, sites); tmpo = time() - t0

    sw = Sweeps(sweeps)
    setmaxdim!(sw, maxdim...)
    setcutoff!(sw, cutoff)
    nn = parse(Int, get(ENV, "NOISESWEEPS", "4"))
    setnoise!(sw, [10.0^(-5 - 4 * (k - 1) / max(nn - 1, 1)) for k in 1:nn]..., 0.0)
    etol = parse(Float64, get(ENV, "ETOL", "1e-10"))
    obs0 = DMRGObserver(; energy_tol=etol, minsweeps=nn + 2)
    obs1 = DMRGObserver(; energy_tol=etol, minsweeps=nn + 2)
    t0 = time()
    E0, ψ0 = dmrg(H, MPS(sites, fill("g", N)), sw; outputlevel=OUTLEVEL[], observer=obs0)
    E1, ψ1 = dmrg(H, [ψ0], random_mps(sites; linkdims=10), sw; weight, outputlevel=OUTLEVEL[], observer=obs1)
    E2 = NaN; var2 = NaN; ov2 = NaN; nsw2 = 0
    if parse(Int, get(ENV, "NLEV", "2")) >= 3          # a third level, penalized against both lower states: a missed
        obs2 = DMRGObserver(; energy_tol=etol, minsweeps=nn + 2)   # or crossing state near E1 shows up here
        E2, ψ2 = dmrg(H, [ψ0, ψ1], random_mps(sites; linkdims=10), sw; weight, outputlevel=OUTLEVEL[], observer=obs2)
        var2 = inner(H, ψ2, H, ψ2) - E2^2
        ov2 = max(abs(inner(ψ0, ψ2)), abs(inner(ψ1, ψ2)))
        nsw2 = length(ITensorMPS.energies(obs2))
        E2 += const_
    end
    tdmrg = time() - t0
    var0 = inner(H, ψ0, H, ψ0) - E0^2
    var1 = inner(H, ψ1, H, ψ1) - E1^2
    E0 += const_; E1 += const_
    isnan(E2) || @printf("   third level: E2 = %.9f ω02 = %.9f ω12 = %.9f  var %.1e  max overlap with ψ0, ψ1 %.1e  sweeps %d\n",
                         E2, E2 - E0, E2 - E1, var2, ov2, nsw2)
    @printf("DMRG %s N=%d%s θ=%s: E0 = %.9f E1 = %.9f ω01 = %.9f  maxlink %d/%d  MPO bond %d  ⟨ψ0|ψ1⟩ = %.1e  var %.1e/%.1e  sweeps %d/%d  (MPO %.1f s, DMRG %.1f s)\n",
            basis, N, basis == "eigen" ? " D=$D" : " ncut=$ncut", θ, E0, E1, E1 - E0, maxlinkdim(ψ0), maxlinkdim(ψ1),
            maxlinkdim(H), abs(inner(ψ0, ψ1)), var0, var1, length(ITensorMPS.energies(obs0)), length(ITensorMPS.energies(obs1)),
            tmpo, tdmrg)
    flush(stdout)
    return E0, E1
end

if abspath(PROGRAM_FILE) == @__FILE__
    spec = load_spec(ENV["SPEC"])
    basis = get(ENV, "BASIS", "eigen")
    D = parse(Int, get(ENV, "D", "10"))
    ncut = parse(Int, get(ENV, "NCUT", "8"))
    md = parse(Int, get(ENV, "MAXDIM", "100"))
    maxdim = [min(md, m) for m in (20, 40, 80, 160, md)]
    sweeps = parse(Int, get(ENV, "SWEEPS", "300"))
    cutoff = parse(Float64, get(ENV, "CUTOFF", "1e-10"))
    weight = parse(Float64, get(ENV, "WEIGHT", "100"))
    φ = haskey(ENV, "PHIEXT") ? parse(Float64, ENV["PHIEXT"]) : Float64(π)
    ngs = parse.(Float64, split(get(ENV, "NG", "0,0.5"), ","))
    res = Dict(ng => run(spec, (n_g=ng, varphi_ext=φ); basis, D, ncut, maxdim, sweeps, cutoff, weight) for ng in ngs)
    length(ngs) == 2 && @printf("Δω01 = %.6e\n", (res[ngs[1]][2] - res[ngs[1]][1]) - (res[ngs[2]][2] - res[ngs[2]][1]))
end
