# Included by the optional ChainRules tier. Also runnable from the main project:
# julia --compiled-modules=existing --project -e 'using KronFreeCircuits; include("test/chainrules/test_pathological.jl")'
module PathologicalChainRulesReview
using Test, LinearAlgebra, KronFreeCircuits
const CRC = Base.get_extension(KronFreeCircuits, :KronFreeCircuitsChainRulesCoreExt).ChainRulesCore

@testset "AD zero weights must not propagate unrelated degenerate derivatives" begin
    key = (:phi, nport())
    root = Node([:phi => ChargeBasis(1)]; solver=Dense(), ports=[key])
    spec = load_spec(joinpath(@__DIR__, "..", "fixtures", "cos2phi.json"))
    θ = (E_C=1.0, E_1=0.0, E_2=0.0, n_g=0.0)
    # H = diag(4,0,4); E_2 couples only the excited pair. Ground energy is
    # identically zero for small E_2, although individual excited derivatives
    # are deliberately undefined by the derivative API at the crossing.
    sys = materialize(root, spec, θ)
    d = derivatives(sys; wrt=:E_2)
    @test energies(d, :E_2)[1] == 0
    @test all(isnan, energies(d, :E_2)[2:3])
    for e2 in (-0.01, 0.01)
        @test energies(materialize(root, spec, merge(θ, (E_2=e2,))))[1] == 0
    end
    y, back = CRC.rrule(spectrum, root, spec, θ)
    grad = back((energies=[1.0, 0.0, 0.0],))[4]
    @test_broken grad[:E_2] == 0.0
    # A nonzero cotangent on the undefined pair must still propagate NaN.
    @test isnan(back((energies=[0.0, 1.0, 0.0],))[4][:E_2])
    no = CRC.NoTangent()
    _, sparse = CRC.frule((no, no, no, (E_C=1.0,)), spectrum, root, spec, θ)
    _, explicit = CRC.frule((no, no, no, (E_C=1.0, E_1=0.0, E_2=0.0, n_g=0.0)), spectrum, root, spec, θ)
    @test sparse.energies ≈ [0.0, 4.0, 4.0] atol=1e-12
    @test_broken explicit.energies ≈ sparse.energies

    # Elementwise-zero cotangents need the same treatment for ports. Only the
    # ground-state diagonal is used here; undefined excited-state rows and
    # columns must not contaminate that scalar contraction.
    yp, backp = CRC.rrule(spectrum, root, spec, θ; ports=[key])
    Pbar = zeros(ComplexF64, size(only(yp.ports)))
    Pbar[1, 1] = 1
    portgrad = backp((energies=zeros(3), ports=[Pbar]))[4]
    @test_broken portgrad[:E_2] == 0.0
end
end
