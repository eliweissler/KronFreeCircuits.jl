using KronFreeCircuits: Dense

# Modes `c` (compact), `e` (extended) and `h` (harmonic) with `E_L` on `e` and `h`, and one cosine whose `omega` and `offset` are substituted.
ceh_json(omega, offset) = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_C", "E_J", "E_L"],
 "modes": [{"name": "c", "type": "c"}, {"name": "e", "type": "e"}, {"name": "h", "type": "h"}],
 "E_C": [["E_C", "0", "0"], ["0", "E_C", "0"], ["0", "0", "E_C"]],
 "E_L": [["0", "0", "0"], ["0", "E_L", "0"], ["0", "0", "E_L"]],
 "flux_linear": ["0", "0", "0"], "n_g": ["0", "0", "0"],
 "cosines": [{"energy": "E_J", "omega": $(omega), "offset": "$(offset)"}]}
"""

# Four compact modes coupled by one cosine with alternating signs.
const abcd_json = """
{"format": "kronfree-spec", "version": 1, "units": "GHz",
 "parameters": ["E_C", "E_J"],
 "modes": [{"name": "a", "type": "c"}, {"name": "b", "type": "c"}, {"name": "c", "type": "c"}, {"name": "d", "type": "c"}],
 "E_C": [["E_C", "0", "0", "0"], ["0", "E_C", "0", "0"], ["0", "0", "E_C", "0"], ["0", "0", "0", "E_C"]],
 "E_L": [["0", "0", "0", "0"], ["0", "0", "0", "0"], ["0", "0", "0", "0"], ["0", "0", "0", "0"]],
 "flux_linear": ["0", "0", "0", "0"], "n_g": ["0", "0", "0", "0"],
 "cosines": [{"energy": "E_J", "omega": [1, -1, 1, -1], "offset": "pi"}]}
"""

# Runs `f()`, passing if every record it logs matches `rx` (an error-level record never matches); returns its value.
function allowing(rx, f)
    logs, val = Test.collect_test_logs(f; min_level=Base.CoreLogging.Warn)
    unexpected = [l.message for l in logs if !(l.level == Base.CoreLogging.Warn && occursin(rx, l.message))]
    @test isempty(unexpected)
    isempty(unexpected) || foreach(m -> println(stderr, "unexpected log: ", first(m, 200)), unexpected)
    return val
end

# With `c` on a `ChargeBasis` and `e` on a `PhaseGrid` (and `h` on a Fock basis), the cosine's terms have no conjugate image
# in the term list, so the tree classifies complex (architecture spec §20f).
@testset "mixed bases, both KITE signs, π and generic offsets" begin
    using KronFreeCircuits: dense, localop
    θ = (E_C=1.0, E_J=2.0, E_L=0.5)
    # keeping every state of the grid puts weight on its edge and on states the phase fixing cannot anchor: those warnings are expected there
    rx_all = r"PhaseGrid on mode :e is too small|heavy hybridization|no weight on the phase mask|peak amplitude vanishes"
    bc, be, bh = ChargeBasis(3), PhaseGrid(-16.0, 16.0, 33; order=2), FockBasis(4; phi_zpf=0.6)   # wide enough for the 8 kept states
    for (omega, offset) in (([1, 1, -1], "pi"), ([1, -1, -1], "pi"), ([1, 1, -1], "0.37*pi"), ([1, -1, 1], "0"))
        spec = KronFreeCircuits.parse_spec(ceh_json(omega, offset))
        for keepn in (0, 8)                                   # untruncated, then genuinely truncated
            ce = Node([:c => bc, :e => be]; solver=Dense(), keep=lowest(keepn))
            root = Node([ce, Node([:h => bh]; solver=Dense())]; solver=Dense())
            # E_J = 2 hybridizes the kept states of `ce` and `h` strongly, which the root reports
            rx = keepn == 0 ? rx_all : r"heavy hybridization"
            sys = allowing(rx, () -> materialize(root, spec, θ))
            @test arithmetic(sys) === :complex
            Hflat = primitive_hamiltonian(circuit_terms(spec), θ, [:c => bc, :e => be, :h => bh])
            P = kron(_primitive(KronFreeCircuits.materialized(sys, root.children[2])), _primitive(KronFreeCircuits.materialized(sys, ce)))
            @test energies(sys) ≈ eigvals(Hermitian(P' * Hflat * P)) atol=1e-9
            K = [(:c, expiphi(omega[1])), (:e, expiphi(omega[2]))]
            Vce = eigenvectors(sys, ce)
            @test port(sys, K, ce) ≈ Vce' * kron(dense(localop(be, expiphi(omega[2]))), dense(localop(bc, expiphi(omega[1])))) * Vce atol=1e-11
            @test port(sys, [(:c, expiphi(-omega[1])), (:e, expiphi(-omega[2]))], ce) ≈ port(sys, K, ce)' atol=1e-11
        end
    end
    # a PeriodicGrid compact mode inside the joint group: untruncated hierarchy equals the flat leaf,
    # and the truncated joint port equals the sandwich of the grid's own expiphi operator
    spec = KronFreeCircuits.parse_spec(ceh_json([1, 1, -1], "pi"))
    bp = PeriodicGrid(8; order=2)
    flat = allowing(rx_all, () -> materialize(Node([:c => bp, :e => be, :h => bh]; solver=Dense()), spec, θ))
    full = allowing(rx_all, () -> materialize(Node([Node([:c => bp, :e => be]; solver=Dense()), Node([:h => bh]; solver=Dense())]; solver=Dense()), spec, θ))
    @test energies(full) ≈ energies(flat) atol=1e-9
    ce = Node([:c => bp, :e => be]; solver=Dense(), keep=lowest(8))
    sys = allowing(r"heavy hybridization", () -> materialize(Node([ce, Node([:h => bh]; solver=Dense())]; solver=Dense()), spec, θ))
    @test arithmetic(sys, ce) === :real                       # every factor is paired: the real-arithmetic joint-port path
    @test arithmetic(sys) === :real                           # and the root is real too
    Vce = eigenvectors(sys, ce)
    @test port(sys, [(:c, expiphi(1)), (:e, expiphi(1))], ce) ≈ Vce' * kron(dense(localop(be, expiphi(1))), dense(localop(bp, expiphi(1)))) * Vce atol=1e-11
    # the real path is pinned against the complex one on the same tree
    ce_c = Node([:c => bp, :e => be]; solver=Dense(), keep=lowest(8), arithmetic=:complex)
    sys_c = allowing(r"heavy hybridization", () -> materialize(Node([ce_c, Node([:h => bh]; solver=Dense())]; solver=Dense()), spec, θ))
    @test arithmetic(sys_c, ce_c) === :complex
    @test energies(sys) ≈ energies(sys_c) atol=1e-9
end

@testset "nested joint ports: a group inside a grandchild" begin
    using KronFreeCircuits: dense, localop
    spec = KronFreeCircuits.parse_spec(abcd_json); θ = (E_C=1.0, E_J=1.0); b = ChargeBasis(3)
    ab = Node([:a => b, :b => b]; solver=Dense(), keep=lowest(5)); abc = Node([ab, :c => b]; solver=Dense(), keep=lowest(7))
    root = Node([abc, Node([:d => b]; solver=Dense())]; solver=Dense())
    sys = @test_logs (:warn, r"no weight on the phase mask") match_mode=:any materialize(root, spec, θ)
    @test arithmetic(sys) === :real
    @test haskey(KronFreeCircuits.materialized(sys, ab).joint_ports, [(:a, expiphi(1)), (:b, expiphi(-1))])
    @test haskey(KronFreeCircuits.materialized(sys, abc).joint_ports, [(:a, expiphi(1)), (:b, expiphi(-1)), (:c, expiphi(1))])
    E1 = dense(localop(b, expiphi(1))); Em = dense(localop(b, expiphi(-1)))
    Pabc = _primitive(KronFreeCircuits.materialized(sys, abc))          # 343 × 7 primitive-space vectors
    @test port(sys, [(:a, expiphi(1)), (:b, expiphi(-1)), (:c, expiphi(1))], abc) ≈ Pabc' * kron(E1, Em, E1) * Pabc atol=1e-11
    Hflat = primitive_hamiltonian(circuit_terms(spec), θ, [:a => b, :b => b, :c => b, :d => b])
    P = kron(_primitive(KronFreeCircuits.materialized(sys, root.children[2])), Pabc)
    @test energies(sys) ≈ eigvals(Hermitian(P' * Hflat * P)) atol=1e-10
    # untruncated nesting equals the flat spectrum
    full = Node([Node([Node([:a => b, :b => b]; solver=Dense()), :c => b]; solver=Dense()), Node([:d => b]; solver=Dense())]; solver=Dense())
    sysf = @test_logs (:warn, r"no weight on the phase mask") match_mode=:any materialize(full, spec, θ)
    @test energies(sysf) ≈ eigvals(Hermitian(Matrix(Hflat))) atol=1e-10
end
