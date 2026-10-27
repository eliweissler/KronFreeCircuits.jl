using KronFreeCircuits: _port_matrix, DenseOp, Term, Factor, Coefficient, materialized, operator, _mode_leaves, _key_character

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

# The spectrum of the flat Hamiltonian `Hflat` projected onto the span of the root's children's kept states.
projected_spectrum(sys, root, Hflat) = begin
    P = foldl((acc, c) -> kron(_primitive(materialized(sys, c)), acc), root.children; init=ones(ComplexF64, 1, 1))
    eigvals(Hermitian(P' * Hflat * P))
end

# `g n_b φ_c n_d` makes the key `(n_b, φ_c)` imaginary: the real-arithmetic child `bc` forms it as a purely imaginary matrix, which the real root consumes.
@testset "imaginary joint key consumed at a real root, against the complex twin and the projected flat Hamiltonian" begin
    using KronFreeCircuits: dense, localop
    bg = PhaseGrid(-12.0, 12.0, 25; order=2)
    bases = [:b => bg, :c => bg, :d => bg]
    terms = Term[]
    for (m, ec, el) in ((:b, 4.0, 0.5), (:c, 3.1, 0.7), (:d, 2.3, 0.9))   # distinct, so no degenerate truncation edge
        push!(terms, Term(Coefficient(ec), [Factor(m, n2port())]))
        push!(terms, Term(Coefficient(el), [Factor(m, phi2port())]))
    end
    push!(terms, Term(Coefficient(0.7), [Factor(:b, nport()), Factor(:c, phiport()), Factor(:d, nport())]))
    θ = (;)
    mk(arith) = Node([Node([:b => bg, :c => bg]; solver=Dense(), keep=lowest(9), arithmetic=arith),
                      Node([:d => bg]; solver=Dense(), keep=lowest(5))]; solver=Dense(), arithmetic=arith)
    rootr, rootc = mk(:auto), mk(:complex)
    sr = materialize(rootr, terms, θ); sc = materialize(rootc, terms, θ)
    bc = rootr.children[1]
    @test arithmetic(sr) === :real && arithmetic(sr, bc) === :real
    @test arithmetic(sc) === :complex
    K = [(:b, nport()), (:c, phiport())]
    @test _key_character(_mode_leaves(rootr), K).kind === :imaginary
    Pj = port(sr, K, bc)
    @test Pj isa Matrix{ComplexF64} && iszero(real(Pj))
    Vbc = eigenvectors(sr, bc)
    @test Pj ≈ Vbc' * kron(dense(localop(bg, phiport())), dense(localop(bg, nport()))) * Vbc atol=1e-11
    M = Matrix(operator(sr))
    @test eltype(M) === Float64
    @test norm(M - M') <= 1e-12 * norm(M)
    @test energies(sr) ≈ energies(sc) atol=1e-9
    Hflat = primitive_hamiltonian(terms, θ, bases)
    @test energies(sr) ≈ projected_spectrum(sr, rootr, Hflat) atol=1e-9
end

# Modes named so that the sorted order (ww, xx, yy, zz) differs from the tree order (zz, yy, xx, ww).
@testset "a term hosted at an intermediate node with a joint group in its grandchild, non-alphabetical modes" begin
    b = ChargeBasis(3)
    zz, yy, xx, ww = :zz, :yy, :xx, :ww
    bases = [zz => b, yy => b, xx => b, ww => b]
    terms = Term[]
    for (m, ec) in ((zz, 1.0), (yy, 1.3), (xx, 0.8), (ww, 1.1))
        push!(terms, Term(Coefficient(ec), [Factor(m, n2port())]))
    end
    # hosted at the intermediate node [[zz, yy], xx]: factors on zz and yy (the grandchild's group) and on xx
    push!(terms, Term(Coefficient(0.3), [Factor(xx, nport()), Factor(zz, nport()), Factor(yy, nport())]))
    # hosted at the root: a cosine over all four modes, both signs
    for s in (1, -1)
        push!(terms, Term(Coefficient(0.5), [Factor(ww, expiphi(-s)), Factor(zz, expiphi(s)), Factor(xx, expiphi(s)), Factor(yy, expiphi(-s))]))
    end
    θ = (;)
    zy = Node([zz => b, yy => b]; solver=Dense(), keep=lowest(6))
    zyx = Node([zy, xx => b]; solver=Dense(), keep=lowest(10))
    root = Node([zyx, Node([ww => b]; solver=Dense())]; solver=Dense())
    sys = @test_logs (:warn, r"no weight on the phase mask") match_mode=:any materialize(root, terms, θ)
    @test arithmetic(sys) === :real
    @test haskey(materialized(sys, zy).joint_ports, [(yy, nport()), (zz, nport())])
    @test haskey(materialized(sys, zy).joint_ports, [(yy, expiphi(-1)), (zz, expiphi(1))])
    @test haskey(materialized(sys, zyx).joint_ports, [(xx, expiphi(1)), (yy, expiphi(-1)), (zz, expiphi(1))])
    Hflat = primitive_hamiltonian(terms, θ, bases)
    @test energies(sys) ≈ projected_spectrum(sys, root, Hflat) atol=1e-10
    M = Matrix(operator(sys))
    @test norm(M - M') <= 1e-12 * norm(M)
    zyc = Node([zz => b, yy => b]; solver=Dense(), keep=lowest(6), arithmetic=:complex)
    zyxc = Node([zyc, xx => b]; solver=Dense(), keep=lowest(10), arithmetic=:complex)
    rootc = Node([zyxc, Node([ww => b]; solver=Dense())]; solver=Dense(), arithmetic=:complex)
    sysc = @test_logs (:warn, r"no weight on the phase mask") match_mode=:any materialize(rootc, terms, θ)
    @test energies(sys) ≈ energies(sysc) atol=1e-10
end
