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
                                           s.children, s.peak_component, s.peak_amplitudes), augB.basis, augB.terms, augB.θ, augB.report)
    @test energies(@test_logs w8... materialize(root, spec, θ; augmented=[bareonly])) ≈ plain rtol=1e-9
    @test isempty(bareonly.solved.ports)
    # the augmented child records its θ: `materialize` refuses it at a θ that differs in a parameter its terms depend on,
    # and takes it at a θ that differs only in a parameter they do not depend on
    @test augB.θ == θ
    deps = union((t.coeff.deps for t in augB.terms)...)
    @test :E_L in deps && :n_g ∉ deps
    @test_throws r"modes \[:phi\].*E_L = 0.04.*E_L = 0.05.*rebuild" materialize(root, spec, merge(θ, (E_L=0.05,)); augmented=[augB])
    @test_throws "E_L" materialize(root, terms, merge(θ, (E_L=0.05,)); constant=constant_term(spec), augmented=[augB])
    θg = merge(θ, (n_g=0.3,))
    phase = r"no weight on the phase mask"
    @test energies(only_warnings(phase, () -> materialize(root, spec, θg; augmented=[augB]))) ≈
          energies(only_warnings(phase, () -> materialize(root, spec, θg))) rtol=1e-9
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

@testset "mean_field_augment" begin
    spec = load_spec(fixture("zero_pi.json")); θ = zero_pi_params(); terms = circuit_terms(spec)
    # wider than the zero-pi grid of test_materialize.jl, so that ten kept states of :phi stay clear of its edges
    grid = PhaseGrid(-20π, 20π, 201; order=6)
    A = Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=10, keep=lowest(10))
    B = Node([:phi => grid]; solver=Dense(), compute=10, keep=lowest(10))
    root = Node([A, B]; solver=Dense(), compute=4, keep=lowest(4))
    exact = energies(materialize(Node([:theta => ChargeBasis(8), :phi => grid];
                                      solver=Dense(), compute=4, keep=lowest(4)), spec, θ))
    # the solves of the charge leaf, keeping ten states, note a phase-convention fallback for some of them; how many
    # depends on the path of the loop, which round-off decides in this symmetric problem, so any number is admitted
    phase = r"no weight on the phase mask"
    quiet(f) = only_warnings(phase, f)
    err(m; kw...) = (r = quiet(() -> mean_field_augment(root, spec, θ; m, kw...));
                     maximum(abs, energies(materialize(root, spec, θ; augmented=r.children)) .- exact))
    e6, e10 = err(6; bare=true), err(10; bare=true)
    @test e10 <= e6 + 1e-12                                                # converges as m grows (nested unions)
    # the sizes themselves, so that a wrong basis is caught too: measured 0.01511 (m = 6) and 0.002762 (m = 10) GHz, the
    # largest error of the four levels (bare children at equal m are off by 3.6 and 3.0 GHz); asserted to 10% relative
    @test e6 ≈ 0.01511 rtol=0.1
    @test e10 ≈ 0.002762 rtol=0.1
    r = quiet(() -> mean_field_augment(root, spec, θ; m=8, bare=true, cap=2))
    @test all(c.report.taken[end] >= 8 - 2 * length(r.used) for c in r.children)   # the bare ladder fills the rest
    @test r.energy == r.solutions[1].energy && r.unconverged == 0
    @test all(size(c.basis, 2) == 8 for c in r.children)
    @test [c.node for c in r.children] == [A, B]
    @test all(c.report.labels == [["solution $i" for i in r.used]; "bare"] for c in r.children)
    @test r.used == [i for i in eachindex(r.solutions) if r.solutions[i].energy - r.energy <= 0.5][1:min(end, 4)]
    @test quiet(() -> mean_field_augment(root, spec, θ; m=8, max_solutions=1)).used == [1]
    small = Node([Node([:theta => ChargeBasis(8)]; solver=Dense(), compute=3, keep=lowest(3)), B];
                 solver=Dense(), compute=4, keep=lowest(4))
    quiet(() -> @test_throws "kept" mean_field_augment(small, spec, θ; m=8))
    @test_throws ArgumentError mean_field_augment(root, spec, θ; m=8, cap=0)
    @test_throws ArgumentError mean_field_augment(root, spec, θ; m=8, window=-1)
    @test_throws ArgumentError mean_field_augment(root, spec, θ; m=8, max_solutions=0)
    @test_throws ArgumentError mean_field_augment(root, spec, θ; m=0)
    # the edge weight of a grid leaf is the measure of the PhaseGrid edge warning; a charge leaf has none
    sB = r.solutions[1].children[B].solved
    @test edge_weight(sB) == KronFreeCircuits._edge_support(sB.vectors[:, 1:1], sB.dims, 1, 3)[1] < 1e-6
    @test edge_weight(r.solutions[1].children[A].solved) == 0.0
    @test_throws ArgumentError edge_weight(materialized(materialize(root, spec, θ; augmented=r.children), root))
    narrow = Node([:phi => PhaseGrid(-2.0, 2.0, 41)]; solver=Dense(), compute=1, keep=lowest(1))
    wide = Node([A, narrow]; solver=Dense(), compute=1, keep=lowest(1))
    @test edge_weight(@test_logs (:warn, r"too small") bare_child(wide, terms, θ, narrow)) > 1e-6
end

# The wrapper on multi-mode children: (i) a parent whose first child is a branch of two leaves, (ii) a parent whose first
# child is one leaf on two modes; the second child is a one-mode leaf. Three coupled transmons on small charge bases, so
# that the exact ground energy (729 states) is a dense solve.
@testset "mean_field_augment on a branch child and on a two-mode leaf" begin
    spec = numeric_spec(; modes=[:a => :c, :b => :c, :c => :c], E_C=[1.0 0.3 0.2; 0.3 0.8 0.3; 0.2 0.3 1.2], E_L=zeros(3, 3),
                        flux_linear=zeros(3), n_g=[0.2, 0.1, 0.3],
                        cosines=[(4.0, [1.0, 0, 0], 0.0), (3.0, [0, 1.0, 0], 0.0), (5.0, [0, 0, 1.0], 0.0),
                                 (1.5, [1.0, 0, -1.0], 0.3), (1.0, [0, 1.0, -1.0], 0.0)])
    θ = (;)
    cb = ChargeBasis(4)                                                     # 9 states per mode
    exact = energies(materialize(Node([:a => cb, :b => cb, :c => cb]; solver=Dense(), compute=1, keep=lowest(1)), spec, θ))[1]
    leaf(m, k) = Node([m => cb]; solver=Dense(), compute=k, keep=lowest(k))
    # `k` states kept by the first child and by the second (at most its 9)
    branch(k) = (local C = Node([leaf(:a, 9), leaf(:b, 9)]; solver=Dense(), compute=k, keep=lowest(k));
                 (C, Node([C, leaf(:c, min(k, 9))]; solver=Dense(), compute=4, keep=lowest(4))))
    twomode(k) = (local L = Node([:a => cb, :b => cb]; solver=Dense(), compute=k, keep=lowest(k));
                  (L, Node([L, leaf(:c, min(k, 9))]; solver=Dense(), compute=4, keep=lowest(4))))
    # keeping every state of a charge leaf, or all 81 of the first child, leaves states the phase fixing cannot anchor
    quiet(f) = only_warnings(r"no weight on the phase mask|heavy hybridization", f)
    starts = [nothing, Dict(:a => 0.0, :b => 0.0, :c => 0.0), Dict(:a => π, :b => π, :c => π)]   # fewer than 28, for time
    for (name, mk) in (("branch child", branch), ("two-mode leaf", twomode))
        @testset "$name" begin
            m = 4
            C, root = mk(m)
            r = quiet(() -> mean_field_augment(root, spec, θ; m, starts))
            @test r.energy >= exact                                         # the product energy is an upper bound
            aug = energies(quiet(() -> materialize(root, spec, θ; augmented=r.children)))[1]
            plain = energies(quiet(() -> materialize(root, spec, θ)))[1]   # bare children, m states each
            @test aug >= exact - 1e-10                                      # variational
            @test aug <= plain + 1e-10                                      # no worse than bare children at equal m
            # the bare solve of each child is formed once: the solutions' states and the union basis are expressed on
            # the same grandchildren (the same objects)
            i = findfirst(c -> c.node === C, r.children)
            for j in r.used
                @test all(r.solutions[j].children[C].solved.children .=== r.children[i].solved.children)
            end
            # with the bare states and m the first child's full dimension, the union spans its whole product space, and
            # the second child (9 states) its own: the parent reproduces the plain spectrum
            Cf, rootf = mk(81)
            rf = quiet(() -> mean_field_augment(rootf, spec, θ; m=81, bare=true, cap=2, allow_fewer=true, starts))
            @test [size(c.basis) for c in rf.children] == [(81, 81), (9, 9)]
            @test energies(quiet(() -> materialize(rootf, spec, θ; augmented=rf.children))) ≈
                  energies(quiet(() -> materialize(rootf, spec, θ))) atol=1e-9
        end
    end
    # a run stopped at its iteration limit is used like the others and counted
    C, root = branch(4)
    ru = quiet(() -> mean_field_augment(root, spec, θ; m=4, maxiter=1, starts))
    @test ru.unconverged > 0 && ru.unconverged == length(ru.used)
    @test all(!ru.solutions[i].converged for i in ru.used)
    @test ru.used == [i for i in eachindex(ru.solutions) if ru.solutions[i].energy - ru.energy <= 0.5][1:min(end, 4)]
    @test all(size(c.basis, 2) == 4 for c in ru.children)
end
