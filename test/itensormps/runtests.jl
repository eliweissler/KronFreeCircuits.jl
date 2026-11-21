# The ITensorMPS tier: KronFreeCircuits.Experimental.dmrg_levels (the ITensorMPS extension) against exact diagonalization.
#   julia --project=test/itensormps -e 'using Pkg; Pkg.instantiate()'; julia --project=test/itensormps test/itensormps/runtests.jl
using Test, LinearAlgebra, Random, ITensors, ITensorMPS, KronFreeCircuits
using KronFreeCircuits.Experimental: dmrg_levels

# Transverse-field Ising chain, H = −J Σ σᶻσᶻ − h Σ σˣ, open ends, in its gapped paramagnetic phase (h > J).
const N, J, h = 8, 1.0, 1.6

function ising_mpo(sites)
    os = OpSum()
    for j in 1:N-1
        os += -4J, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
        os += -2h, "Sx", j
    end
    return MPO(os, sites)
end

function ising_dense()
    σz, σx, I2 = [1.0 0; 0 -1], [0.0 1; 1 0], Matrix(1.0I, 2, 2)
    site(op, j) = foldl(kron, [k == j ? op : I2 for k in 1:N])
    return -J * sum(site(σz, j) * site(σz, j + 1) for j in 1:N-1) - h * sum(site(σx, j) for j in 1:N)
end

const EXACT = eigvals(Symmetric(ising_dense()))[1:3]
const KW = (nsweeps=12, maxdim=[10, 20, 40, 80], cutoff=1e-12, outputlevel=0)

@testset "dmrg_levels" begin
    Random.seed!(1)
    sites = siteinds("S=1/2", N)
    H = ising_mpo(sites)

    r = dmrg_levels(H; nlevels=3, weight=20.0, KW...)
    @test r.energies ≈ EXACT atol = 1e-8
    @test length(r.states) == 3 && all(v -> abs(v) < 1e-7, r.variances)
    @test all(iszero, diag(r.overlaps)) && maximum(r.overlaps) < 1e-6
    @test size(r.overlaps) == (3, 3) && isapprox(r.overlaps, r.overlaps'; atol=1e-12)

    # one level needs no weight; a named product start and an MPS start give the same ground state
    g = dmrg_levels(H; init=fill("Up", N), KW...)
    @test only(g.energies) ≈ EXACT[1] atol = 1e-8
    @test only(dmrg_levels(H; init=random_mps(sites; linkdims=4), KW...).energies) ≈ EXACT[1] atol = 1e-8

    # a start per level and a fresh observer per level, with a Sweeps object
    sw = Sweeps(12); setmaxdim!(sw, 10, 20, 40, 80); setcutoff!(sw, 1e-12)
    q = dmrg_levels(H; nlevels=2, weight=20.0, sweeps=sw, outputlevel=0,
                    init=k -> k == 1 ? fill("Up", N) : random_mps(sites; linkdims=4),
                    observer=k -> DMRGObserver(; energy_tol=1e-12, minsweeps=3))
    @test q.energies ≈ EXACT[1:2] atol = 1e-8
    @test length(q.observers) == 2 && q.observers[1] !== q.observers[2]
    @test all(o -> 3 <= length(ITensorMPS.energies(o)) <= 12, q.observers)
    @test_throws "must be a function" dmrg_levels(H; nlevels=2, weight=20.0, observer=DMRGObserver(), KW...)

    # the weight is the caller's: required for excited states, and positive
    @test_throws "needs `weight`" dmrg_levels(H; nlevels=2, KW...)
    @test_throws ArgumentError dmrg_levels(H; nlevels=2, weight=-1.0, KW...)
    @test_throws ArgumentError dmrg_levels(H; nlevels=0, KW...)
end
