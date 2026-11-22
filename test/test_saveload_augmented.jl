using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: augment, UnionReport, bare_child, _terms_identical

@testset "augmented child" begin
    spec = load_spec(fixture("zero_pi.json")); θ = zero_pi_params(); terms = circuit_terms(spec)
    grid = PhaseGrid(-12π, 12π, 201; order=6)
    A = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=17, keep=lowest(17))
    B = Node([:phi => grid]; solver=Dense(), compute=6, keep=lowest(6))
    root = Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))
    w8 = ntuple(_ -> (:warn, r"no weight on the phase mask"), 8)    # as in test_augment.jl
    bareB = bare_child(root, terms, θ, B)
    augB = augment(root, terms, θ, B, bareB, Matrix{ComplexF64}(bareB.vectors), UnionReport(["bare"], [6], [0]))
    sysB = @test_logs w8... materialize(root, spec, θ; augmented=[augB])

    p = tempname() * ".jld2"
    save_system(p, sysB)
    loaded = load_system(p)
    assert_same_system(loaded, sysB)
    @test length(loaded.augmented) == 1
    a = loaded.augmented[1]
    @test a.node === loaded.root.node.children[2]
    @test isequal(a.basis, augB.basis)
    @test (a.report.labels, a.report.taken, a.report.dropped) == (["bare"], [6], [0])
    @test _terms_identical(a.terms, augB.terms)
end
