using Test, LinearAlgebra, KronFreeCircuits

@testset "union basis" begin
    e(i, n=8) = (v = zeros(ComplexF64, n); v[i] = 1; v)
    src(label, cols; kw...) = BasisSource(label, reduce(hcat, cols), collect(1.0:length(cols)); kw...)
    a = src("well 1", [e(1), e(2), e(3)]); b = src("well 2", [e(4), e(1), e(5)])
    U, r = union_basis([a, b]; m=4)
    @test U ≈ hcat(e(1), e(4), e(2), e(3)) && r.taken == [3, 1] && r.dropped == [0, 1]     # level by level; e(1) twice
    @test U' * U ≈ I
    # a cap leaves the remainder to the uncapped source
    U2, r2 = union_basis([src("well 1", [e(1), e(2), e(3)]; cap=1), src("bare", [e(6), e(7), e(8)])]; m=4)
    @test r2.taken == [1, 3]
    # a window in energy
    @test union_basis([src("w", [e(1), e(2), e(3)]; window=1.5), src("bare", [e(6), e(7)])]; m=4)[2].taken == [2, 2]
    # coinciding sources: reported, and an error when m cannot be reached (Review Focus 4)
    @test_throws "dropped" union_basis([a, a]; m=4)
    U3, r3 = union_basis([a, a]; m=4, allow_fewer=true)
    @test size(U3, 2) == 3 && r3.dropped == [0, 3]
    # nearly dependent: a remainder below the threshold is dropped, above it is kept
    near(δ) = normalize(e(1) + δ * e(2))
    @test union_basis([src("x", [e(1)]), src("y", [near(1e-8)])]; m=2, allow_fewer=true)[2].dropped == [0, 1]
    @test size(union_basis([src("x", [e(1)]), src("y", [near(1e-3)])]; m=2)[1], 2) == 2

    # the error message carries the report and the escape hatch
    err = try union_basis([a, a]; m=4) catch ex; ex end
    @test err isa ArgumentError && occursin("allow_fewer = true", err.msg) && occursin("well 1", err.msg) && occursin("taken 0, dropped 3", err.msg)

    # stopping at m in the middle of a level leaves later columns neither taken nor dropped
    _, r4 = union_basis([a, b]; m=2)
    @test r4.taken == [1, 1] && r4.dropped == [0, 0]
    # a source that offers nothing because of cap = 0
    U5, r5 = union_basis([src("none", [e(1), e(2)]; cap=0), src("bare", [e(3), e(4)])]; m=2)
    @test r5.taken == [0, 2] && r5.dropped == [0, 0] && U5 ≈ hcat(e(3), e(4))
    # real input is converted, and the caller's arrays are not mutated
    V = Matrix{Float64}(I, 4, 2); V0 = copy(V); en = [0.0, 1.0]
    s = BasisSource("real", V, en)
    union_basis([s, s |> x -> BasisSource("other", V, en)]; m=2)
    @test V == V0 && s.vectors isa Matrix{ComplexF64}

    # validation
    @test_throws ArgumentError union_basis(BasisSource[]; m=1)
    @test_throws ArgumentError union_basis([a]; m=0)
    @test_throws ArgumentError union_basis([a, src("small", [e(1, 4)])]; m=2)
    @test_throws ArgumentError BasisSource("x", reduce(hcat, [e(1), e(2)]), [1.0])
    @test_throws ArgumentError src("x", [e(1)]; cap=-1)
    @test_throws ArgumentError src("x", [e(1)]; window=-1.0)
    @test_throws ArgumentError union_basis([a]; m=1, drop=-1.0)
    @test_throws ArgumentError union_basis([src("big", [2 * e(1)])]; m=1)
end

@testset "augmented child" begin
    spec = load_spec(fixture("zero_pi.json")); θ = zero_pi_params(); terms = circuit_terms(spec)
    grid = PhaseGrid(-12π, 12π, 201; order=6)                             # the zero-pi grid of test_materialize.jl
    mk() = (local A = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=17, keep=lowest(17));
            local B = Node([:phi => grid]; solver=Dense(), compute=6, keep=lowest(6));
            (A, B, Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))))
    # every solve of A, which keeps all 17 states of its charge basis, notes a phase-convention fallback for 8 of them
    w8 = ntuple(_ -> (:warn, r"no weight on the phase mask"), 8)
    A, B, root = mk()
    plain = energies(@test_logs w8... materialize(root, spec, θ))
    bareA, bareB = (@test_logs w8... bare_child(root, terms, θ, A)), bare_child(root, terms, θ, B)
    # the bare vectors as the union basis reproduce the bare child exactly
    rep(n) = UnionReport(["bare"], [n], [0])
    augB = augment(root, terms, θ, B, bareB, Matrix{ComplexF64}(bareB.vectors), rep(6))
    @test augB.solved.energies ≈ bareB.energies rtol=1e-10
    sysB = @test_logs w8... materialize(root, spec, θ; augmented=[augB])
    @test energies(sysB) ≈ plain rtol=1e-9
    # a rotated basis of the same span gives the same parent spectrum
    Q = Matrix(qr(randn(ComplexF64, 6, 6)).Q)
    augQ = augment(root, terms, θ, B, bareB, bareB.vectors * Q, rep(6))
    @test energies(@test_logs w8... materialize(root, spec, θ; augmented=[augQ])) ≈ plain rtol=1e-9
    # the whole space of A: the parent no longer depends on A's truncation
    full = @test_logs w8... augment(root, terms, θ, A, bareA, Matrix{ComplexF64}(I, 17, 17), rep(17))
    @test energies(materialize(root, spec, θ; augmented=[full])) ≈ plain rtol=1e-9
    @test augB.basis ≈ bareB.vectors && augB.terms == KronFreeCircuits._terms_at(B, terms)
    @test arithmetic(sysB, root) == :complex
    # the parent takes the augmented child as it is, not solving it again
    @test materialized(sysB, B).vectors == augB.solved.vectors
    # the term-list form takes the keyword too
    @test energies(@test_logs w8... materialize(root, terms, θ; constant=constant_term(spec), augmented=[augB])) ≈ plain rtol=1e-9
    # a port the tree needs that the augmented child lacks is formed from its vectors, on a copy
    s = augB.solved
    bareonly = AugmentedChild(B, typeof(s)(s.node, s.dims, s.energies, s.vectors, s.kept, empty(s.ports), empty(s.joint_ports),
                                           s.children, s.peak_component, s.peak_amplitudes), augB.basis, augB.terms, augB.report)
    @test energies(@test_logs w8... materialize(root, spec, θ; augmented=[bareonly])) ≈ plain rtol=1e-9
    @test isempty(bareonly.solved.ports)
    # Review Focus 5 (`mk` declares its nodes `local`, so a second tree does not rebind A and B)
    A2, B2, root2 = mk()
    @test_throws ArgumentError materialize(root2, spec, θ; augmented=[augB])              # node of another tree
    @test_throws ArgumentError materialize(root, spec, θ; augmented=[augB, augB])
    @test_throws ArgumentError materialize(root, spec, θ; augmented=[augB], previous=sysB)
    Ar = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=4, keep=lowest(4))
    Br = Node([:phi => grid]; solver=Dense(), compute=4, keep=lowest(4))
    rootr = Node([Ar, Br]; solver=Dense(), compute=4, keep=lowest(4), arithmetic=:real)
    bBr = bare_child(rootr, terms, θ, Br)
    augr = augment(rootr, terms, θ, Br, bBr, Matrix{ComplexF64}(bBr.vectors), rep(4))
    @test_throws "real" materialize(rootr, spec, θ; augmented=[augr])
    # augment's own checks
    @test_throws ArgumentError augment(root, terms, θ, A, bareB, Matrix{ComplexF64}(bareB.vectors), rep(6))   # `bare` of another node
    @test_throws ArgumentError augment(root, terms, θ, B, bareB, Matrix{ComplexF64}(bareB.vectors[1:end-1, :]), rep(6))
    @test_throws ArgumentError augment(root, terms, θ, B, bareB, zeros(ComplexF64, size(bareB.vectors, 1), 0), rep(0))
    @test_throws "deviation" augment(root, terms, θ, B, bareB, 2 * bareB.vectors, rep(6))
end

@testset "augmented branch child, and an augmented child below a non-root parent" begin
    spec = load_spec(fixture("three_transmons.json")); θ = three_transmons_params(g=0.4); terms = circuit_terms(spec)
    mk() = (local T1 = Node([:varphi_1 => ChargeBasis(6)]; solver=Dense(), compute=6, keep=lowest(6));
            local T2 = Node([:varphi_2 => ChargeBasis(6)]; solver=Dense(), compute=6, keep=lowest(6));
            local T3 = Node([:varphi_3 => ChargeBasis(6)]; solver=Dense(), compute=6, keep=lowest(6));
            local C = Node([T1, T2]; solver=Dense(), compute=10, keep=lowest(10));
            (T1, T2, T3, C, Node([C, T3]; solver=Dense(), compute=5, keep=lowest(5))))
    T1, T2, T3, C, root = mk()
    plain = energies(materialize(root, spec, θ))
    rep(n) = UnionReport(["bare"], [n], [0])
    # a branch child (two leaves under it) built from its bare vectors
    bC = bare_child(root, terms, θ, C)
    augC = augment(root, terms, θ, C, bC, bC.vectors, rep(10))
    @test augC.solved.energies ≈ bC.energies rtol=1e-10
    @test size(augC.basis) == (36, 10)
    sysC = materialize(root, spec, θ; augmented=[augC])
    @test energies(sysC) ≈ plain rtol=1e-9
    @test arithmetic(sysC, root) == :complex
    # an augmented child one level below the root, its parent the branch C
    b1 = bare_child(C, terms, θ, T1)
    aug1 = augment(C, terms, θ, T1, b1, b1.vectors * Matrix(qr(randn(ComplexF64, 6, 6)).Q), rep(6))
    sys1 = materialize(root, spec, θ; augmented=[aug1])
    @test energies(sys1) ≈ plain rtol=1e-9
    @test arithmetic(sys1, C) == :complex && arithmetic(sys1, root) == :complex && arithmetic(sys1, T2) == :real
    # the system records its augmented children, and operator() refuses an augmented node but serves its ancestors
    @test augmented(sysC) == [augC] && augmented(sysC)[1] === augC && isempty(augmented(materialize(root, spec, θ)))
    @test_throws "augmented(sys)" KronFreeCircuits.operator(sysC, C)
    H = KronFreeCircuits.operator(sysC, root); V = eigenvectors(sysC)
    @test norm(H * V - V * Diagonal(energies(sysC))) < 1e-8
    # two augmented siblings, both from their bare vectors
    b3 = bare_child(root, terms, θ, T3)
    aug3 = augment(root, terms, θ, T3, b3, b3.vectors, rep(6))
    @test energies(materialize(root, spec, θ; augmented=[augC, aug3])) ≈ plain rtol=1e-9
    # a single-column union basis
    one = augment(C, terms, θ, T1, b1, b1.vectors[:, 1:1], rep(1))
    sys_one = materialize(root, spec, θ; augmented=[one])
    @test energies(sys_one, T1) ≈ b1.energies[1:1] && length(materialized(sys_one, C).dims) == 2 && materialized(sys_one, C).dims[1] == 1
    # an augmented node inside another augmented node is refused, naming both, in either order
    @test_throws r"varphi_1.*inside.*varphi_1, :varphi_2" materialize(root, spec, θ; augmented=[augC, aug1])
    @test_throws r"varphi_1.*inside.*varphi_1, :varphi_2" materialize(root, spec, θ; augmented=[aug1, augC])
    # the root itself may not be augmented, and a real ancestor below the root is named
    broot = bare_child(Node([root]), terms, θ, root)
    @test_throws "root" materialize(root, spec, θ; augmented=[augment(Node([root]), terms, θ, root, broot, broot.vectors, rep(5))])
    T1r, T2r, T3r, _, _ = mk()
    Cr = Node([T1r, T2r]; solver=Dense(), compute=10, keep=lowest(10), arithmetic=:real)
    rootr = Node([Cr, T3r]; solver=Dense(), compute=5, keep=lowest(5))
    b1r = bare_child(Cr, terms, θ, T1r)
    @test_throws "varphi_2" materialize(rootr, spec, θ; augmented=[augment(Cr, terms, θ, T1r, b1r, b1r.vectors, rep(6))])
    # an augmented node whose own arithmetic is :real is refused, naming it
    T1s = Node([:varphi_1 => ChargeBasis(6)]; solver=Dense(), compute=6, keep=lowest(6), arithmetic=:real)
    Cs = Node([T1s, Node([:varphi_2 => ChargeBasis(6)]; solver=Dense(), compute=6, keep=lowest(6))]; solver=Dense(), compute=10, keep=lowest(10))
    roots = Node([Cs, Node([:varphi_3 => ChargeBasis(6)]; solver=Dense(), compute=6, keep=lowest(6))]; solver=Dense(), compute=5, keep=lowest(5))
    b1s = bare_child(Cs, terms, θ, T1s)
    @test_throws "augmented node on modes [:varphi_1] has arithmetic = :real" materialize(roots, spec, θ; augmented=[augment(Cs, terms, θ, T1s, b1s, b1s.vectors, rep(6))])
end
