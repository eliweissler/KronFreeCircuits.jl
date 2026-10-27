using KronFreeCircuits: _port_matrix, DenseOp

# Two modes, `c` compact and `e` extended, coupled by one cosine on both: its factors `expiphi(±1)` on `c` and `e`.
const ce_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_C", "E_J", "E_L"],
 "modes": [{"name": "c", "type": "c"}, {"name": "e", "type": "e"}],
 "E_C": [["E_C", "0"], ["0", "E_C"]],
 "E_L": [["0", "0"], ["0", "E_L"]],
 "flux_linear": ["0", "0"], "n_g": ["0", "0"],
 "cosines": [{"energy": "E_J", "omega": [1, 1], "offset": "0"}]}
"""

# Three compact modes coupled by one cosine on all three: grouping `a` and `b` in one child makes its terms joint.
# Every leaf of such a tree holds only charging terms, so its kept states are bare charge states, and those on negative
# charges have no weight on the phase mask (a warning, captured below).
const abc_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_C", "E_J"],
 "modes": [{"name": "a", "type": "c"}, {"name": "b", "type": "c"}, {"name": "c", "type": "c"}],
 "E_C": [["E_C", "0", "0"], ["0", "E_C", "0"], ["0", "0", "E_C"]],
 "E_L": [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]],
 "flux_linear": ["0", "0", "0"], "n_g": ["0", "0", "0"],
 "cosines": [{"energy": "E_J", "omega": [1, 1, 1], "offset": "0"}]}
"""

@testset "_port_matrix over several axes equals the Kronecker sandwich" begin
    rng = MersenneTwister(11); dims = [3, 4, 2]
    V = randn(rng, ComplexF64, prod(dims), 5)
    A1 = randn(rng, ComplexF64, 3, 3); A3 = randn(rng, ComplexF64, 2, 2)
    ops = [(1, DenseOp(A1)), (3, DenseOp(A3))]
    P = _port_matrix(V, dims, ops)
    @test P ≈ V' * kron(A3, Matrix(I, 4, 4), A1) * V atol=1e-12 * norm(V)^2 * norm(A1) * norm(A3)
    @test _port_matrix(V, dims, 1, DenseOp(A1)) == _port_matrix(V, dims, [(1, DenseOp(A1))])
    Vr = randn(rng, prod(dims), 5)
    @test _port_matrix(Vr, dims, [(1, DenseOp(real(A1))), (3, DenseOp(real(A3)))]) isa Matrix{Float64}
    @test_throws ArgumentError _port_matrix(V, dims, [(1, DenseOp(A1)), (1, DenseOp(A1))])
end

# With `c` on a `ChargeBasis` (real `expiphi`) and `e` on a `PhaseGrid` (paired `expiphi`), the cosine's terms have no
# conjugate image and the leaf is complex; on a `PeriodicGrid` (paired) it is real.
# The small grid on `e` warns on edge support, which does not affect the sandwich identities.
@testset "_joint_port on a leaf equals the primitive sandwich, real and complex" begin
    using KronFreeCircuits: _joint_port, _mode_leaves, dense, localop
    bc, be, bp = ChargeBasis(3), PhaseGrid(-3.0, 3.0, 7; order=2), PeriodicGrid(6; order=2)
    spec = KronFreeCircuits.parse_spec(ce_json); θ = (E_C=1.0, E_J=2.0, E_L=0.5)
    sandwich(V, b, sgn) = V' * kron(dense(localop(be, expiphi(-sgn))), dense(localop(b, expiphi(sgn)))) * V
    leaf = Node([:c => bp, :e => be]; solver=Dense(), keep=lowest(5))
    sys = @test_logs (:warn, r"PhaseGrid on mode :e is too small") match_mode=:any materialize(leaf, spec, θ)
    @test arithmetic(sys) === :real
    mn = sys.root; V = eigenvectors(sys)
    for sgn in (1, -1)
        P = _joint_port(leaf, _mode_leaves(leaf), mn.children, mn.dims, V, [(:c, expiphi(sgn)), (:e, expiphi(-sgn))])
        @test P isa Matrix{ComplexF64}                 # grid expiphi (paired) ⊗ grid expiphi (paired): paired
        @test P ≈ sandwich(V, bp, sgn) atol=1e-12
    end
    leafc = Node([:c => bc, :e => be]; solver=Dense(), keep=lowest(5))
    sysc = @test_logs (:warn, r"PhaseGrid on mode :e is too small") match_mode=:any materialize(leafc, spec, θ); mc = sysc.root; Vc = eigenvectors(sysc)
    @test arithmetic(sysc) === :complex
    Pc = _joint_port(leafc, _mode_leaves(leafc), mc.children, mc.dims, Vc, [(:c, expiphi(1)), (:e, expiphi(-1))])
    @test Pc isa Matrix{ComplexF64}
    @test Pc ≈ sandwich(Vc, bc, 1) atol=1e-12
end

# The ce_json leaf on a charge basis with its cosine replaced by a real coupling `n_c φ_e`, so that it is real.
@testset "_joint_port on a real leaf: the natural type follows the key character" begin
    using KronFreeCircuits: _joint_port, _mode_leaves, dense, localop, Term, Factor, Coefficient
    bc, be = ChargeBasis(3), PhaseGrid(-3.0, 3.0, 7; order=2)
    spec = KronFreeCircuits.parse_spec(ce_json); θ = (E_C=1.0, E_J=2.0, E_L=0.5)
    terms = filter(t -> length(t.factors) == 1, circuit_terms(spec, Dict(:c => bc, :e => be)))
    push!(terms, Term(Coefficient(0.3), [Factor(:c, nport()), Factor(:e, phiport())]))
    leaf = Node([:c => bc, :e => be]; solver=Dense(), keep=lowest(5))
    sys = @test_logs (:warn, r"PhaseGrid on mode :e is too small") match_mode=:any materialize(leaf, terms, θ)
    @test arithmetic(sys) === :real
    mn = sys.root; V = eigenvectors(sys)
    sandwich(pc, pe) = V' * kron(dense(localop(be, pe)), dense(localop(bc, pc))) * V
    for (pc, pe, T, re0) in ((expiphi(1), expiphi(-1), ComplexF64, false),     # real ⊗ paired: paired
                             (nport(), phiport(), Float64, false),             # real ⊗ real: real
                             (nport(), nport(), ComplexF64, true))             # real ⊗ imaginary: imaginary
        P = _joint_port(leaf, _mode_leaves(leaf), mn.children, mn.dims, V, [(:c, pc), (:e, pe)])
        @test P isa Matrix{T}
        @test P ≈ sandwich(pc, pe) atol=1e-12
        re0 && @test iszero(real(P))
    end
end

@testset "truncated child: joint port is V† A_joint V, separate projection is not" begin
    spec = KronFreeCircuits.parse_spec(abc_json); θ = (E_C=1.0, E_J=1.0)
    ba = ChargeBasis(4); ab = Node([:a => ba, :b => ba]; solver=Dense(), keep=lowest(6), ports=[(:a, expiphi(1))]); c = Node([:c => ba]; solver=Dense())
    root = Node([ab, c]; solver=Dense())
    sys = @test_logs (:warn, r"no weight on the phase mask") match_mode=:any materialize(root, spec, θ)
    Vab = eigenvectors(sys, ab)                        # 81 × 6, phase-fixed kept vectors
    Ea = KronFreeCircuits.dense(KronFreeCircuits.localop(ba, expiphi(1)))
    Ajoint = kron(Ea, Ea)                              # kron(B_b, A_a): axis 1 (a) fastest
    Pj = port(sys, [(:a, expiphi(1)), (:b, expiphi(1))], ab)
    @test Pj ≈ Vab' * Ajoint * Vab atol=1e-12
    @test port(sys, [(:a, expiphi(-1)), (:b, expiphi(-1))], ab) ≈ Pj' atol=1e-12          # adjoint pair
    Psep = (Vab' * kron(I(9), Ea) * Vab) * (Vab' * kron(Ea, I(9)) * Vab)   # ChargeBasis(4) has 9 states
    @test norm(Psep - Pj) > 1e-3 * norm(Pj)            # the separate projection demonstrably differs
    # the root's spectrum is that of the independently projected flat Hamiltonian
    Hflat = primitive_hamiltonian(circuit_terms(spec), θ, [:a => ba, :b => ba, :c => ba])
    P = kron(_primitive(KronFreeCircuits.materialized(sys, c)), _primitive(KronFreeCircuits.materialized(sys, ab)))
    Hred = Hermitian(P' * Hflat * P)
    M = Matrix(KronFreeCircuits.operator(sys))         # the root operator as the package assembles it
    @test norm(M - M') <= 1e-12 * norm(M)
    @test energies(sys) ≈ eigvals(Hred) atol=1e-10
    @test_throws ArgumentError port(sys, [(:a, expiphi(2)), (:b, expiphi(2))], ab)
    @test port(sys, [(:a, expiphi(1))], ab) == port(sys, :a, expiphi(1), ab)             # a singleton key delegates to the (mode, Port) method
end

@testset "real and complex arithmetic agree on a joint-port tree" begin
    spec = KronFreeCircuits.parse_spec(abc_json); θ = (E_C=1.0, E_J=1.0); ba = ChargeBasis(4)
    mk(arith) = Node([Node([:a => ba, :b => ba]; solver=Dense(), keep=lowest(6), arithmetic=arith), Node([:c => ba]; solver=Dense())]; solver=Dense())
    sr = @test_logs (:warn, r"no weight on the phase mask") match_mode=:any materialize(mk(:auto), spec, θ)
    sc = @test_logs (:warn, r"no weight on the phase mask") match_mode=:any materialize(mk(:complex), spec, θ)
    @test arithmetic(sr, sr.root.node.children[1]) === :real && arithmetic(sc, sc.root.node.children[1]) === :complex
    @test port(sr, [(:a, expiphi(1)), (:b, expiphi(1))], sr.root.node.children[1]) isa Matrix{Float64}     # charge-basis expiphi: real character
    @test energies(sr) ≈ energies(sc) atol=1e-10
end
