using Random
using KronFreeCircuits: SelectionContext, select, lowest, indices, connected, follow, ports_needed,
                        validate_selection, fix_phases!, nport, Port

@testset "selection rules" begin
    P = zeros(ComplexF64, 6, 6); P[1,4] = P[4,1] = 0.5; P[4,6] = P[6,4] = 0.2; P[2,3] = P[3,2] = 0.9
    ctx = SelectionContext(collect(1.0:6), Matrix{ComplexF64}(I, 6, 6), Dict((:m, nport()) => P), nothing)
    @test select(connected([1, 2], (:m, nport()), 0.1), ctx) == [1, 2, 3, 4, 6]
    @test ports_needed(connected([1], (:m, nport()), 0.1)) == [(:m, nport())]
    @test select(connected([2, 1, 2], (:m, nport()), 0.1), ctx) == [1, 2, 3, 4, 6]     # core deduplicated
    @test connected([2, 1, 2], (:m, nport()), 0.1).core == [2, 1]
    err = try select(connected([1, 7], (:m, nport()), 0.1), ctx); nothing catch e e end
    @test err isa ArgumentError && occursin("connected", err.msg) && occursin("[7]", err.msg)
    @test_throws ArgumentError select(connected([0], (:m, nport()), 0.1), ctx)
    # keyword (spec) form: port as (mode, Port) or (kind, mode)
    fields(r) = (r.core, r.port, r.thresh)
    @test fields(connected(; core=[1, 2], port=(:m, nport()), thresh=0.1)) == fields(connected([1, 2], (:m, nport()), 0.1))
    @test fields(connected(; core=[1, 2], port=(:n, :m), thresh=0.1)) == fields(connected([1, 2], (:m, nport()), 0.1))
    @test connected(; core=[1], port=(:phi2, :m), thresh=0.1).port == (:m, KronFreeCircuits.phi2port())
    @test_throws ArgumentError connected(; core=[1], port=(:bogus, :m), thresh=0.1)
    @test select(lowest(3), ctx) == [1, 2, 3] && select(lowest(0), ctx) == 1:6
    @test_throws ArgumentError validate_selection([1, 1], 6, lowest(2))     # Review Focus 5
    @test_throws ArgumentError validate_selection(Int[], 6, lowest(0))
    @test_throws ArgumentError validate_selection([7], 6, indices([7]))
    V = Matrix{ComplexF64}(I, 6, 6)[:, [3, 1, 2, 4, 5, 6]]                   # permuted eigenvectors
    @test select(follow(Matrix{ComplexF64}(I, 6, 6)[:, 1:3]), SelectionContext(ctx.energies, V, ctx.ports, nothing)) == [2, 3, 1]
    @test_logs (:warn,) select(follow(Matrix{ComplexF64}(I, 6, 6)[:, 1:1]), SelectionContext(ctx.energies, V .* 0.1 .+ 0.05, ctx.ports, nothing))
end
@testset "follow: global greedy by overlap, previous before reference" begin
    # ref1 overlaps (0.63, 0.77) and ref2 (0, 1): per-reference greedy would give ref1 the 0.77 state
    # and ref2 the 0 state; global greedy by overlap assigns ref2 first.
    Vc = Matrix{ComplexF64}(I, 2, 2)
    R = ComplexF64[0.63 0; 0.77 1]
    cx = SelectionContext([0.0, 1.0], Vc, Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}(), nothing)
    @test (@test_logs (:warn, r"reference states \[1\]") select(follow(R; minoverlap=0.7), cx)) == [1, 2]
    @test_throws ArgumentError select(follow(ComplexF64[1 0 0; 0 1 0]), cx)        # 3 references, 2 states
    @test_throws ArgumentError select(follow(), cx)                               # no reference, no previous
    # the previous materialization's vectors take precedence over an explicit reference
    prev = (vectors=Vc[:, [2, 1]],)
    cxp = SelectionContext([0.0, 1.0], Vc, cx.ports, prev)
    @test select(follow(Vc), cxp) == [2, 1] && select(follow(), cxp) == [2, 1]
end
@testset "fix_phases!" begin
    v = [0.3, -0.8im, 0.5]; mask = [false, true, true]
    W = fix_phases!(reshape(copy(v), :, 1), :positive_sector, mask, nothing)
    @test W[2, 1] ≈ 0.8 && abs(W[1, 1]) ≈ 0.3                                # largest in positive sector real positive
    r = reshape(exp(0.4im) .* v, :, 1)
    @test fix_phases!(reshape(copy(v), :, 1), :follow, mask, r) ≈ r
end
@testset "selection exports" begin
    @test all(s -> Base.isexported(KronFreeCircuits, s), (:constant_term, :SelectionRule, :SelectionContext))
    @test !Base.isexported(KronFreeCircuits, :select)          # custom rules extend KronFreeCircuits.select
end
@testset "physical overlap through a changed child basis" begin
    using KronFreeCircuits: MaterializedNode, _child_overlap, _transport_previous, Node, ChargeBasis, lowest, Port
    rng = MersenneTwister(51)
    orth(n, k) = Matrix(qr(randn(rng, ComplexF64, n, k)).Q)[:, 1:k]
    leafnode = Node([:a => ChargeBasis(2)]; keep=lowest(3))          # basis dim 5
    mk(node, dims, V, children) = MaterializedNode(node, dims, zeros(size(V, 2)), V, collect(1:size(V, 2)),
                                                  Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}(), children, ones(Int, size(V, 2)), Matrix{ComplexF64}(I, size(V, 2), size(V, 2)))
    Vold = orth(5, 3); Vnew = Vold[:, [2, 1, 3]] .* [1, -1, im]'       # permuted and rephased leaf basis
    lo, ln = mk(leafnode, [5], Vold, Any[ChargeBasis(2)]), mk(leafnode, [5], Vnew, Any[ChargeBasis(2)])
    S = _child_overlap(lo, ln)
    @test S ≈ Vold' * Vnew
    bnode = Node([leafnode, :b => ChargeBasis(1)]; keep=lowest(4))     # dims [3, 3]
    Cold = orth(9, 4)
    bo = mk(bnode, [3, 3], Cold, Any[lo, ChargeBasis(1)])
    R̃ = _transport_previous(bo, Any[ln, ChargeBasis(1)], [3, 3])
    @test R̃ ≈ kron(Matrix{ComplexF64}(I, 3, 3), S') * Cold            # axis 1 fastest: kron(I_b, S') acts on axis 1
    Cnew = kron(Matrix{ComplexF64}(I, 3, 3), S') * Cold               # the same physical states in the new child basis
    bn = mk(bnode, [3, 3], Cnew, Any[ln, ChargeBasis(1)])
    @test _child_overlap(bo, bn) ≈ Matrix{ComplexF64}(I, 4, 4) atol=1e-12
    @test_throws ArgumentError _child_overlap(bo, mk(bnode, [3, 2], Cold[1:6, :], Any[ln, ChargeBasis(1)]))
end
@testset "transport through two changed children; input untouched; basis change rejected" begin
    using KronFreeCircuits: MaterializedNode, _child_overlap, _transport_previous, Node, ChargeBasis, PhaseGrid, FockBasis, lowest, Port
    rng = MersenneTwister(52)
    orth(n, k) = Matrix(qr(randn(rng, ComplexF64, n, k)).Q)[:, 1:k]
    mk(node, dims, V, children) = MaterializedNode(node, dims, zeros(size(V, 2)), V, collect(1:size(V, 2)),
                                                  Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}(), children, ones(Int, size(V, 2)), Matrix{ComplexF64}(I, size(V, 2), size(V, 2)))
    la = Node([:a => ChargeBasis(2)]; keep=lowest(3)); lb = Node([:b => ChargeBasis(3)]; keep=lowest(2))   # dims 5 and 7
    Ua, Ub = orth(5, 3), orth(7, 2)
    Wa = Ua * Matrix(qr(randn(rng, ComplexF64, 3, 3)).Q); Wb = Ub * Matrix(qr(randn(rng, ComplexF64, 2, 2)).Q)   # same spans, rotated
    ao, an = mk(la, [5], Ua, Any[ChargeBasis(2)]), mk(la, [5], Wa, Any[ChargeBasis(2)])
    bo, bn = mk(lb, [7], Ub, Any[ChargeBasis(3)]), mk(lb, [7], Wb, Any[ChargeBasis(3)])
    Sa, Sb = _child_overlap(ao, an), _child_overlap(bo, bn)
    top = Node([la, lb, :c => ChargeBasis(1)]; keep=lowest(5))        # dims [3, 2, 3]
    C = orth(18, 5); Csaved = copy(C)
    po = mk(top, [3, 2, 3], C, Any[ao, bo, ChargeBasis(1)])
    R̃ = _transport_previous(po, Any[an, bn, ChargeBasis(1)], [3, 2, 3])
    @test R̃ ≈ kron(Matrix{ComplexF64}(I, 3, 3), Sb', Sa') * C         # axis 1 fastest
    @test po.vectors == Csaved                                        # input untouched
    @test_throws ArgumentError _transport_previous(po, Any[an, bn, FockBasis(3; phi_zpf=1.0)], [3, 2, 3])   # same dimension, different basis on axis 3
    lg = Node([:g => PhaseGrid(-2π, 2π, 9; order=2)]; keep=lowest(2)); lg2 = Node([:g => PhaseGrid(-3π, 3π, 9; order=2)]; keep=lowest(2))
    V9 = orth(9, 2)
    @test_throws ArgumentError _child_overlap(mk(lg, [9], V9, Any[PhaseGrid(-2π, 2π, 9; order=2)]), mk(lg2, [9], V9, Any[PhaseGrid(-3π, 3π, 9; order=2)]))
end

# Shared by the T1-T3 unit cases: a MaterializedNode with placeholder energies, ports and peaks, and the
# explicit primitive-space vectors of a materialized node (axis 1 fastest), built without the transport.
_mkmn(node, dims, V, children) = KronFreeCircuits.MaterializedNode(node, dims, zeros(size(V, 2)), V, collect(1:size(V, 2)),
    Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}(), children, ones(Int, size(V, 2)), Matrix{ComplexF64}(I, size(V, 2), size(V, 2)))
function _primitive(mn)
    W = ones(ComplexF64, 1, 1)
    for c in mn.children
        d = c isa KronFreeCircuits.MaterializedNode ? _primitive(c) : Matrix{ComplexF64}(I, KronFreeCircuits.dimension(c), KronFreeCircuits.dimension(c))
        W = kron(d, W)
    end
    return W * mn.vectors
end

@testset "T1: overlaps through several levels against explicit primitive vectors; gauge covariance" begin
    using KronFreeCircuits: _child_overlap, _transport_previous, Node, ChargeBasis, lowest
    rng = MersenneTwister(61)
    orth(n, k) = Matrix(qr(randn(rng, ComplexF64, n, k)).Q)[:, 1:k]
    unitary(k) = Matrix(qr(randn(rng, ComplexF64, k, k)).Q)
    phases(k) = Diagonal(cis.(2π .* rand(rng, k)))
    perm(p) = Matrix{ComplexF64}(I, length(p), length(p))[:, p]
    Id(k) = Matrix{ComplexF64}(I, k, k)
    # top = [lc, :d => basis, mid], mid = [la, :b => basis]: a depth-2 child on the last axis beside a depth-1
    # child and a basis axis; dims top [3, 3, 4], mid [3, 3].
    la = Node([:a => ChargeBasis(2)]; keep=lowest(3)); lc = Node([:c => ChargeBasis(2)]; keep=lowest(3))
    mid = Node([la, :b => ChargeBasis(1)]; keep=lowest(4))
    top = Node([lc, :d => ChargeBasis(1), mid]; keep=lowest(5))
    build(Va, Vm, Vc, Vt) = (a = _mkmn(la, [5], Va, Any[ChargeBasis(2)]); m = _mkmn(mid, [3, 3], Vm, Any[a, ChargeBasis(1)]);
                             c = _mkmn(lc, [5], Vc, Any[ChargeBasis(2)]); _mkmn(top, [3, 3, 4], Vt, Any[c, ChargeBasis(1), m]))
    old = build(orth(5, 3), orth(9, 4), orth(5, 3), orth(36, 5))
    # physically different states at every level, two children of the top changed at once
    new = build(orth(5, 3), orth(9, 4), orth(5, 3), orth(36, 5))
    @test _child_overlap(old, new) ≈ _primitive(old)' * _primitive(new) atol=1e-12
    @test _child_overlap(old.children[3], new.children[3]) ≈ _primitive(old.children[3])' * _primitive(new.children[3]) atol=1e-12
    # a pure gauge change: child columns permuted and rephased, an exactly degenerate cluster (states 1-2
    # of la, taken degenerate) rotated by a complex unitary first; parent coefficients transformed to match
    Ua = [unitary(2) zeros(2, 1); zeros(1, 2) cis(0.7)] * perm([3, 1, 2]) * phases(3)
    Uc = phases(3) * perm([2, 3, 1]); Um = perm([4, 2, 1, 3]) * phases(4); Ut = phases(5) * perm([5, 1, 4, 2, 3])
    Va, Vc = old.children[3].children[1].vectors, old.children[1].vectors
    Vm, Vt = old.children[3].vectors, old.vectors
    gauge = build(Va * Ua, kron(Id(3), Ua') * Vm * Um, Vc * Uc, kron(Um', Id(3), Uc') * Vt * Ut)
    @test _primitive(gauge) ≈ _primitive(old) * Ut                       # the same physical states
    @test _child_overlap(old, gauge) ≈ Ut atol=1e-12                     # overlap sees only the top's own gauge
    @test _transport_previous(old, gauge.children, [3, 3, 4]) ≈ gauge.vectors * Ut' atol=1e-12
    @test old.vectors == Vt                                              # previous vectors untouched
end
@testset "T2: basis compatibility" begin
    using KronFreeCircuits: _child_overlap, _transport_previous, Node, ChargeBasis, PhaseGrid, FockBasis, lowest
    rng = MersenneTwister(62)
    orth(n, k) = Matrix(qr(randn(rng, ComplexF64, n, k)).Q)[:, 1:k]
    leaf(b) = _mkmn(Node([:x => b]; keep=lowest(2)), [KronFreeCircuits.dimension(b)], orth(KronFreeCircuits.dimension(b), 2), Any[b])
    msg(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError" end
    # equal dimensions, different bases: Fock scale, grid interval, stencil order, basis kind
    for (b1, b2) in ((FockBasis(6; phi_zpf=1.0), FockBasis(6; phi_zpf=1.2)),
                     (PhaseGrid(-2π, 2π, 7; order=2), PhaseGrid(-2π, 2.5π, 7; order=2)),
                     (PhaseGrid(-2π, 2π, 7; order=2), PhaseGrid(-2π, 2π, 7; order=4)),
                     (ChargeBasis(3), FockBasis(7; phi_zpf=1.0)))
        @test occursin("not implemented", msg(() -> _child_overlap(leaf(b1), leaf(b2))))
        @test _child_overlap(leaf(b1), leaf(b1)) isa Matrix                 # the same basis is accepted
    end
    # a node child on one side and a basis child on the other
    la = Node([:a => ChargeBasis(1)]; keep=lowest(3)); top = Node([la]; keep=lowest(2))
    a = _mkmn(la, [3], orth(3, 3), Any[ChargeBasis(1)])
    t = _mkmn(top, [3], orth(3, 2), Any[a])
    @test_throws ArgumentError _transport_previous(t, Any[ChargeBasis(1)], [3])
    # a changed child kept count is outside the (restricted) transport contract
    a4 = _mkmn(Node([:a => ChargeBasis(2)]; keep=lowest(4)), [5], orth(5, 4), Any[ChargeBasis(2)])
    @test_throws ArgumentError _transport_previous(t, Any[a4], [4])
end
@testset "T3: following collisions, ties and vanishing overlap" begin
    noports = Dict{Tuple{Symbol,Port},Matrix{ComplexF64}}()
    I3 = Matrix{ComplexF64}(I, 3, 3)
    # competing assignments with orthonormal references: both prefer current state 1, ref 2 more strongly
    r1 = normalize(ComplexF64[0.7, 0.6, 0.39]); r2 = normalize(ComplexF64[0.7, -0.6, -0.13 / 0.39])
    R = [r1 r2]; O = abs.(R' * I3)
    @test R' * R ≈ I && argmax(O[1, :]) == argmax(O[2, :]) == 1 && O[2, 1] > O[1, 1]
    ctx = SelectionContext(zeros(3), I3, noports, nothing, R)
    @test (@test_logs (:warn, r"reference states \[1\]") select(follow(; minoverlap=0.65), ctx)) == [2, 1]
    # exact ties (equal two-state mixing, overlap 1/sqrt(2)): a unique assignment, warned above 1/sqrt(2)
    H2 = ComplexF64[1 1; 1 -1] / sqrt(2)
    ctx2 = SelectionContext(zeros(2), H2, noports, nothing, Matrix{ComplexF64}(I, 2, 2))
    idx = @test_logs (:warn, r"reference states \[1, 2\]") select(follow(; minoverlap=0.8), ctx2)
    @test sort(idx) == [1, 2]
    @test sort(@test_logs select(follow(; minoverlap=0.7), ctx2)) == [1, 2]
    # equal three-state mixing has overlap 1/sqrt(3) > 0.5: the default threshold is silent, 0.6 warns
    F3 = [cis(2π * j * k / 3) for j in 0:2, k in 0:2] / sqrt(3)
    ctx3 = SelectionContext(zeros(3), F3, noports, nothing, I3)
    @test sort(@test_logs select(follow(), ctx3)) == [1, 2, 3]
    @test sort(@test_logs (:warn, r"reference states \[1, 2, 3\]") select(follow(; minoverlap=0.6), ctx3)) == [1, 2, 3]
    # a reference orthogonal to every computed state: assigned the free state, warned; the :follow phase
    # it would define is undefined, and fix_phases! leaves the column unchanged and finite
    Vc = I3[:, 1:2]; Rz = I3[:, [1, 3]]
    ctxz = SelectionContext(zeros(2), Vc, noports, nothing, Rz)
    @test (@test_logs (:warn, r"reference states \[2\]") select(follow(), ctxz)) == [1, 2]
    W = fix_phases!(copy(Vc), :follow, trues(3), Rz)
    @test all(isfinite, W) && W == Vc
    # more references than computed states, and a row mismatch
    @test_throws ArgumentError select(follow(), SelectionContext(zeros(2), Vc, noports, nothing, I3))
    @test_throws ArgumentError select(follow(), SelectionContext(zeros(2), Vc, noports, nothing, I3[1:2, 1:2]))
    # the context's reference (the transported previous vectors) is what follow reads; the four-argument
    # constructor keeps the previous solution's raw vectors as the reference
    prev = (vectors=Vc[:, [2, 1]],)
    @test SelectionContext(zeros(2), Vc, noports, prev).reference == prev.vectors
    @test SelectionContext(zeros(2), Vc, noports, nothing).reference === nothing
    @test select(follow(), SelectionContext(zeros(2), Vc, noports, prev, Vc)) == [1, 2]
end
