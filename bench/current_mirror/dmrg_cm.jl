# DMRG (ITensorMPS) on the full current-mirror spec (specs/cm_N<N>[_rung].json, circuit-quantization): every term from
# the spec, H = 4 nᵀ E_C n − Σ_c E_c cos(ω_cᵀθ + o_c) + constant (zero offsets n_g here), as
#   site i:     4 E_C,ii n_i²
#   couplings:  8 E_C,ij n_i n_j  (i < j; all-to-all, long-range)
#   cosines:    −(E_c/2)(e^{i o_c} Π_i U_i^{ω_ci} + h.c.),  U = e^{iθ}, ω_ci ∈ {−1, 0, 1}
# in the raw charge basis (2 NCUT + 1 states per site), the MPS sites ordered rung by rung (the spec's rung modes, so
# the junctions are between neighbouring rungs; the Moebius pair N ↔ 1 closes the chain). The lowest NLEV levels by the
# penalty method: level k penalized against levels 1 .. k−1 (a degenerate partner is a separate level).
#
#   CM_N=3 julia --project=bench/array_fluxonium/dmrg -t 1 bench/current_mirror/dmrg_cm.jl
#
# Env: CM_N, CM_COORDS (rung, default; or theta), NCUT (10), NLEV (6), MAXDIM (200), SWEEPS (200), CUTOFF (1e-10),
#      ETOL (1e-10), WEIGHT (100), NOISESWEEPS (4), BLASTHREADS (1), KRYLOVDIM (3), EIGMAXITER (1).

using ITensors, ITensorMPS, KronFreeCircuits, LinearAlgebra, Printf
import JSON3
BLAS.set_num_threads(parse(Int, get(ENV, "BLASTHREADS", "1")))

const N = parse(Int, get(ENV, "CM_N", "3"))
const COORDS = get(ENV, "CM_COORDS", "rung")
const NCUT = parse(Int, get(ENV, "NCUT", "10"))
const NLEV = parse(Int, get(ENV, "NLEV", "6"))
const SPEC = load_spec(joinpath(@__DIR__, "specs", COORDS == "theta" ? "cm_N$N.json" : "cm_N$(N)_$COORDS.json"))
const θ = merge((E_J=19.0, varphi_ext=0.0), NamedTuple(Symbol("n_g$i") => 0.0 for i in 1:2N-1))
realof(c) = (v = c(θ); abs(imag(v)) < 1e-12 || error("complex coefficient $v"); real(v))

names = [Symbol(m.name) for m in SPEC.modes]
# rung order: rung j's modes, j = 1 .. N
rung = COORDS == "rung" ? [j < N ? [Symbol("a$j"), Symbol("b$j")] : [Symbol("a$N")] for j in 1:N] :
       [j < N ? [Symbol("theta$j"), Symbol("theta$(j + N)")] : [Symbol("theta$N")] for j in 1:N]
order = [findfirst(==(m), names) for m in reduce(vcat, rung)]          # MPS position -> spec mode index
pos = invperm(order)                                                   # spec mode index -> MPS position
M = length(names)

const D = 2NCUT + 1
nmat = Matrix(Diagonal(Float64.(-NCUT:NCUT)))
Umat = zeros(D, D); for k in 1:D-1; Umat[k+1, k] = 1; end            # e^{iθ}|n⟩ = |n+1⟩
ITensors.op(::OpName"N", ::SiteType"Q", s::Index) = itensor(nmat, s', dag(s))
ITensors.op(::OpName"N2", ::SiteType"Q", s::Index) = itensor(nmat^2, s', dag(s))
ITensors.op(::OpName"U", ::SiteType"Q", s::Index) = itensor(Umat, s', dag(s))
ITensors.op(::OpName"Ud", ::SiteType"Q", s::Index) = itensor(Matrix(Umat'), s', dag(s))
ITensors.state(::StateName"0", ::SiteType"Q", s::Index) = (v = zeros(D); v[NCUT+1] = 1; itensor(v, s))

EC = [realof(SPEC.E_C[i, j]) for i in 1:M, j in 1:M]
all(iszero, (realof(SPEC.E_L[i, j]) for i in 1:M, j in 1:M)) || error("inductive terms")
all(iszero, (realof(g) for g in SPEC.n_g)) || error("nonzero offsets")
sites = [Index(D, "Q,Site,n=$p") for p in 1:M]
os = OpSum()
for i in 1:M
    add!(os, 4EC[i, i], "N2", pos[i])
    for j in i+1:M
        abs(EC[i, j]) > 1e-14 && add!(os, 8EC[i, j], "N", pos[i], "N", pos[j])
    end
end
for c in SPEC.cosines
    E, o = realof(c.energy), realof(c.offset)
    abs(sin(o)) < 1e-12 || error("complex cosine offset $o")
    nz = findall(!=(0), c.omega)
    all(i -> abs(c.omega[i]) == 1, nz) || error("cosine omega $(c.omega)")
    coef = -E / 2 * cos(o)
    add!(os, coef, Iterators.flatten((c.omega[i] == 1 ? "U" : "Ud", pos[i]) for i in nz)...)
    add!(os, coef, Iterators.flatten((c.omega[i] == 1 ? "Ud" : "U", pos[i]) for i in nz)...)
end
const0 = realof(SPEC.constant)
t0 = time(); H = MPO(os, sites); tmpo = time() - t0

md = parse(Int, get(ENV, "MAXDIM", "200"))
sw = Sweeps(parse(Int, get(ENV, "SWEEPS", "200")))
setmaxdim!(sw, [min(md, m) for m in (20, 40, 80, 160, md)]...)
setcutoff!(sw, parse(Float64, get(ENV, "CUTOFF", "1e-10")))
nn = parse(Int, get(ENV, "NOISESWEEPS", "4"))
setnoise!(sw, [10.0^(-5 - 4 * (k - 1) / max(nn - 1, 1)) for k in 1:nn]..., 0.0)
etol = parse(Float64, get(ENV, "ETOL", "1e-10"))
weight = parse(Float64, get(ENV, "WEIGHT", "100"))
# the local eigensolver of each DMRG step (ITensorMPS defaults: Krylov dimension 3, one iteration)
const KRYLOV = (eigsolve_krylovdim=parse(Int, get(ENV, "KRYLOVDIM", "3")), eigsolve_maxiter=parse(Int, get(ENV, "EIGMAXITER", "1")))
println("current mirror N = $N, DMRG on $M sites ($COORDS coordinates, charge basis ncut $NCUT), MPO bond $(maxlinkdim(H)) ",
        "[$(round(tmpo; digits=1)) s]; $NLEV levels by penalty, maxdim $md, weight $weight")
ψs = MPS[]; Es = Float64[]; info = []
tall = time()
for k in 1:NLEV
    obs = DMRGObserver(; energy_tol=etol, minsweeps=nn + 2)
    t = time()
    E, ψ = k == 1 ? dmrg(H, MPS(sites, fill("0", M)), sw; outputlevel=0, observer=obs, KRYLOV...) :
                    dmrg(H, ψs, random_mps(sites; linkdims=10), sw; weight, outputlevel=0, observer=obs, KRYLOV...)
    var = inner(H, ψ, H, ψ) - E^2
    ov = isempty(ψs) ? 0.0 : maximum(abs(inner(p, ψ)) for p in ψs)
    push!(ψs, ψ); push!(Es, E + const0)
    push!(info, (sweeps=length(ITensorMPS.energies(obs)), var=var, overlap=ov, maxlink=maxlinkdim(ψ), seconds=time() - t))
    @printf("  level %d: E = %.9f GHz (from E0 %.6f)  var %.1e  max overlap %.1e  maxlink %d  sweeps %d  [%.1f s]\n",
            k, Es[end], Es[end] - Es[1], var, ov, maxlinkdim(ψ), info[end].sweeps, info[end].seconds)
    flush(stdout)
end
tdmrg = time() - tall
exact = let f = joinpath(@__DIR__, "results", "lobpcg_flat_N$(N)_nc10.json")
    isfile(f) ? Vector{Float64}(JSON3.read(read(f, String)).energies) : nothing
end
if exact !== nothing
    n = min(length(exact), length(Es))
    d = (Es[1:n] .- Es[1]) .- (exact[1:n] .- exact[1])
    @printf("against the exact levels: E0 - exact %.2e GHz; worst |Δ(E_k - E_0)| %.2e GHz\n", Es[1] - exact[1], maximum(abs, d))
end
@printf("DMRG total %.1f s (MPO %.1f s)\n", tdmrg, tmpo)
mkpath(joinpath(@__DIR__, "results"))
open(io -> JSON3.pretty(io, JSON3.write(Dict("N" => N, "coords" => COORDS, "ncut" => NCUT, "maxdim" => md, "energies" => Es,
                                             "info" => info, "seconds" => tdmrg, "seconds_mpo" => tmpo))),
     joinpath(@__DIR__, "results", "dmrg_N$(N)_$(COORDS)_nc$(NCUT)_md$md.json"), "w")
