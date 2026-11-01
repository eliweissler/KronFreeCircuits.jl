# Plane-wave symbols of the order-`order` cyclic stencils (design §4.5): n e^{ikφ} = σ₁(k) e^{ikφ}, n² e^{ikφ} = σ₂(k) e^{ikφ}
function periodic_symbols(order)
    w1 = KronFreeCircuits.fd_weights(order, 1); w2 = KronFreeCircuits.fd_weights(order, 2); h = order ÷ 2
    σ1(k, Δ) = sum(w1[j+h+1] * sin(j * k * Δ) for j in -h:h) / Δ
    σ2(k, Δ) = -sum(w2[j+h+1] * cos(j * k * Δ) for j in -h:h) / Δ^2
    return σ1, σ2
end

@testset "free rotor on the periodic grid: continuum spectrum, discrete spectrum, imaginary part" begin
    spec = load_spec(fixture("transmon.json"))
    σ1, σ2 = periodic_symbols(8)
    for ng in (0.0, 0.3, 0.5), N in (32, 64)
        sys = materialize(Node([:varphi_1 => PeriodicGrid(N; order=8)]; solver=Dense(), keep=lowest(6)), spec, (E_J=0.0, E_C=1.0, n_g=ng))
        exact = sort([4 * (n - ng)^2 for n in -8:8])[1:6]
        # the order-8 stencil's own error on the sixth level (|n - n_g| = 3 at n_g = 0) is 1.53e-4 at N = 32
        @test energies(sys) ≈ exact atol=(N == 32 ? 2e-4 : 1e-6)
        # exact for the discretization: the periodic-gauge symbols, 4 (σ₂(k) - 2 n_g σ₁(k) + n_g²)
        Δ = 2π / N
        disc = sort([4 * (σ2(n, Δ) - 2ng * σ1(n, Δ) + ng^2) for n in -8:8])[1:6]
        @test energies(sys) ≈ disc atol=1e-10
        A = Matrix(KronFreeCircuits.operator(sys))
        # the linear term -8 E_C n_g n is imaginary on the grid
        ng == 0.0 ? (@test maximum(abs, imag(A)) == 0) : (@test maximum(abs, imag(A)) > 0)
    end
end

@testset "n_g = ½: the sweet-spot pairs split by the discretization, at order 8 (design §4.5)" begin
    spec = load_spec(fixture("transmon.json")); σ1, σ2 = periodic_symbols(8)
    split = Float64[]; predicted = Float64[]
    for N in (32, 64, 128)
        E = energies(materialize(Node([:varphi_1 => PeriodicGrid(N; order=8)]; solver=Dense(), keep=lowest(4)), spec, (E_J=0.0, E_C=1.0, n_g=0.5)))
        Δ = 2π / N
        pred(k) = 4 * (σ2(k, Δ) - σ2(1 - k, Δ) - σ1(k, Δ) + σ1(1 - k, Δ))     # E(k) - E(1-k) of the pair (k, 1-k)
        if N < 128      # at N = 128 the predicted 9.1e-14 is below the dense solver's roundoff (eigenvalues up to 4 (N/2)² ≈ 1.6e4)
            @test abs(E[2] - E[1]) ≈ abs(pred(0)) atol=1e-12                 # pair (0, 1)
            @test abs(E[4] - E[3]) ≈ abs(pred(-1)) atol=1e-12                # pair (-1, 2)
        end
        push!(split, abs(E[2] - E[1])); push!(predicted, abs(pred(0)))
    end
    @test split[1] ≈ 1.1e-8 rtol=0.1                 # builder check, 2026-10-30: 1.1e-8, 4.4e-11, 9e-14 (formula)
    @test split[1] / split[2] > 100 && predicted[2] / predicted[3] > 100     # about 2^8 per doubling
    @test split[3] < 5e-12                           # measured 1.8e-12: roundoff of the dense solve, not the discretization
end

@testset "transmon: periodic grid vs charge basis; charge port; literals 0 and 0.5" begin
    spec = load_spec(fixture("transmon.json")); θ = (E_J=20.0, E_C=1.0, n_g=0.3)
    sc = materialize(Node([:varphi_1 => ChargeBasis(20)]; solver=Dense(), keep=lowest(6), ports=[(:varphi_1, nport())]), spec, θ)
    sp = materialize(Node([:varphi_1 => PeriodicGrid(128; order=8)]; solver=Dense(), keep=lowest(6)), spec, θ)
    @test energies(sp) ≈ energies(sc) atol=1e-7
    @test abs.(KronFreeCircuits.charge_port(sp, :varphi_1)) ≈ abs.(port(sc, :varphi_1, nport())) atol=1e-6
    # the diagonal ⟨k|n|k⟩ is gauge invariant
    @test real(diag(KronFreeCircuits.charge_port(sp, :varphi_1))) ≈ real(diag(port(sc, :varphi_1, nport()))) atol=1e-6
    # on a charge-basis mode the on-demand charge port is the formed port
    @test KronFreeCircuits.charge_port(sc, :varphi_1) ≈ port(sc, :varphi_1, nport()) atol=1e-12
    # the cyclic BandedOps in the sparse assembly used by ShiftInvert
    @test sparse(KronFreeCircuits.operator(sp)) ≈ Matrix(KronFreeCircuits.operator(sp))
    @test_throws ArgumentError KronFreeCircuits.charge_port(sp, :nope)
    # structure decides: a parameter n_g is complex; the literal "0" is real; the literal "0.5" is complex (design §4.4)
    leaf(b; kw...) = Node([:varphi_1 => b]; solver=Dense(), keep=lowest(6), kw...)
    lit(v) = parse_spec(replace(read(fixture("transmon.json"), String), "\"n_g\": [\"n_g\"]" => "\"n_g\": [\"$v\"]"))
    s0 = materialize(leaf(PeriodicGrid(128); ports=[(:varphi_1, nport())]), lit("0"), (E_J=20.0, E_C=1.0))
    @test arithmetic(s0) === :real && s0.root isa KronFreeCircuits.MaterializedNode{Float64}
    @test eltype(KronFreeCircuits.operator(s0)) === Float64
    @test port(s0, :varphi_1, nport()) isa Matrix{ComplexF64}                 # an imaginary port of a real node
    @test KronFreeCircuits.charge_port(s0, :varphi_1) ≈ port(s0, :varphi_1, nport()) atol=1e-12
    @test energies(s0) ≈ energies(materialize(leaf(ChargeBasis(20)), lit("0"), (E_J=20.0, E_C=1.0))) atol=1e-7
    sh = materialize(leaf(PeriodicGrid(128); ports=[(:varphi_1, nport())]), lit("0.5"), (E_J=20.0, E_C=1.0))
    @test arithmetic(sh) === :complex
    @test maximum(abs, imag(Matrix(KronFreeCircuits.operator(sh)))) > 0
    shc = materialize(leaf(ChargeBasis(20); ports=[(:varphi_1, nport())]), lit("0.5"), (E_J=20.0, E_C=1.0))
    @test energies(sh) ≈ energies(shc) atol=1e-7
    @test abs.(port(sh, :varphi_1, nport())) ≈ abs.(port(shc, :varphi_1, nport())) atol=1e-6
    @test arithmetic(materialize(leaf(PeriodicGrid(64)), spec, θ)) === :complex
end

@testset "cos 2φ: both parity sectors, exact half-period translation, sweet-spot degeneracy, divisibility" begin
    spec = load_spec(fixture("cos2phi.json"))
    θ2 = (E_C=1.0, n_g=0.2, E_1=0.0, E_2=15.0)
    sc = materialize(Node([:phi => ChargeBasis(20)]; solver=Dense(), keep=lowest(8)), spec, θ2)
    N = 128; b = PeriodicGrid(N; order=8)      # N = 64 is 1.1e-5 off on the eighth level (eighth-order decay, ×256 per doubling)
    sp = materialize(Node([:phi => b]; solver=Dense(), keep=lowest(8)), spec, θ2)
    @test energies(sp) ≈ energies(sc) atol=1e-7
    # the half-period translation T: φ -> φ + π is an exact grid translation on the periodic function space (a plain
    # circshift, T² = I); it commutes with n, n² and cos 2φ at every n_g, so the kept-span projector commutes with T
    V = eigenvectors(sp)
    T = circshift(Matrix{Float64}(I, N, N), (N ÷ 2, 0))
    @test T^2 == I
    @test norm(T * (V * V') - (V * V') * T) < 1e-9
    # the sectors are the two eigenvalues ±1 of T; both are present among the kept states
    parity = [real(dot(V[:, j], T * V[:, j])) for j in 1:8]
    @test all(abs.(abs.(parity) .- 1) .< 1e-6) && any(>(0.5), parity) && any(<(-0.5), parity)
    # at n_g = ½ the two sectors are degenerate (k ↦ 1 - k): exactly in the charge basis (6e-14), on the grid only to
    # O(Δ⁸) (design §1, "What it costs"); measured pairs (1,2), (3,4), (5,6): N = 64: 4.5e-9, 1.1e-7, 8.0e-7;
    # N = 128: 1.8e-11, 4.3e-10, 3.3e-9; N = 256: 1.3e-12, 2.1e-12, 1.4e-11 (about 2⁸ per doubling, then roundoff)
    θ½ = (E_C=1.0, n_g=0.5, E_1=0.0, E_2=15.0)
    E½ = energies(materialize(Node([:phi => b]; solver=Dense(), keep=lowest(8)), spec, θ½))
    @test abs(E½[2] - E½[1]) < 4e-11 && abs(E½[4] - E½[3]) < 1e-9
    E½64 = energies(materialize(Node([:phi => PeriodicGrid(64; order=8)]; solver=Dense(), keep=lowest(8)), spec, θ½))
    @test abs(E½64[2] - E½64[1]) / abs(E½[2] - E½[1]) > 100
    Ec½ = energies(materialize(Node([:phi => ChargeBasis(30)]; solver=Dense(), keep=lowest(8)), spec, θ½))
    @test abs(Ec½[2] - Ec½[1]) < 1e-12 && abs(Ec½[4] - Ec½[3]) < 1e-12
    @test_throws ArgumentError materialize(Node([:phi => PeriodicGrid(63; order=8)]; solver=Dense(), keep=lowest(8)), spec, θ2)   # 63 not divisible by 2
    # a weak cos φ mixes the sectors: T no longer commutes with the kept span
    sm = materialize(Node([:phi => b]; solver=Dense(), keep=lowest(8)), spec, (E_C=1.0, n_g=0.2, E_1=0.5, E_2=15.0))
    Vm = eigenvectors(sm)
    @test norm(T * (Vm * Vm') - (Vm * Vm') * T) > 1e-3
end

@testset "every expiphi harmonic on a PeriodicGrid is checked: extra_terms and the term-list form" begin
    spec = load_spec(fixture("transmon.json")); θ = (E_J=0.0, E_C=1.0, n_g=0.2)
    cos2 = [Term(Coefficient(-7.5), [Factor(:varphi_1, expiphi(s))]) for s in (2, -2)]     # −15 cos 2φ, not in the spec
    leaf(N) = Node([:varphi_1 => PeriodicGrid(N; order=8)]; solver=Dense(), keep=lowest(6))
    err = try materialize(leaf(63), spec, θ; extra_terms=cos2); nothing catch e; e end
    @test err isa ArgumentError && occursin("divisible", err.msg) && occursin("ω = 2", err.msg)
    sp = materialize(leaf(64), spec, θ; extra_terms=cos2)
    sc = materialize(Node([:varphi_1 => ChargeBasis(20)]; solver=Dense(), keep=lowest(6)), spec, θ; extra_terms=cos2)
    @test energies(sp) ≈ energies(sc) atol=1e-4                       # N = 64, order 8: the stencil error, not the check
    # the term-list form checks too
    terms = vcat(circuit_terms(spec), cos2)
    @test_throws ArgumentError materialize(leaf(63), terms, θ; modetypes=Dict(:varphi_1 => :c))
    @test_throws ArgumentError materialize(leaf(63), terms, θ)                              # with no mode types at all
    # a non-integer harmonic on a periodic grid is refused before assembly, naming the mode and k
    half = [Term(Coefficient(-1.0), [Factor(:varphi_1, expiphi(s))]) for s in (0.5, -0.5)]
    err = try materialize(leaf(64), spec, θ; extra_terms=half); nothing catch e; e end
    @test err isa ArgumentError && occursin(":varphi_1", err.msg) && occursin("0.5", err.msg)
end

@testset "the term-list form: n2port() and nport() on a PeriodicGrid mode need no spec" begin
    spec = load_spec(fixture("transmon.json")); θ = (E_J=20.0, E_C=1.0, n_g=0.3)
    leaf(b; kw...) = Node([:varphi_1 => b]; solver=Dense(), keep=lowest(6), kw...)
    terms = circuit_terms(spec)
    @test any(t -> any(f -> f.port == n2port(), t.factors), terms) && any(t -> any(f -> f.port == nport(), t.factors), terms)
    @test all(f.port.kind in (:n, :n2, :expiphi) for t in terms for f in t.factors)
    sp = materialize(leaf(PeriodicGrid(128); ports=[(:varphi_1, nport()), (:varphi_1, n2port())]), terms, θ; constant=constant_term(spec), modetypes=Dict(:varphi_1 => :c))
    sc = materialize(leaf(ChargeBasis(20); ports=[(:varphi_1, nport()), (:varphi_1, n2port())]), spec, θ)
    @test energies(sp) ≈ energies(sc) atol=1e-7
    @test abs.(port(sp, :varphi_1, nport())) ≈ abs.(port(sc, :varphi_1, nport())) atol=1e-6
    @test port(sp, :varphi_1, n2port()) isa Matrix{ComplexF64}      # a real port of a complex node
    @test abs.(port(sp, :varphi_1, n2port())) ≈ abs.(port(sc, :varphi_1, n2port())) atol=1e-6
    # a selection rule on the charge selects the charge basis's states (closure from the ground state at 0.5: states 1-6)
    rule = connected([1], (:varphi_1, nport()), 0.5)
    skp = materialize(Node([:varphi_1 => PeriodicGrid(128)]; solver=Dense(), compute=10, keep=rule), spec, θ)
    skc = materialize(Node([:varphi_1 => ChargeBasis(20)]; solver=Dense(), compute=10, keep=rule), spec, θ)
    @test skp.root.kept == skc.root.kept == 1:6
    @test energies(skp) ≈ energies(skc) atol=1e-7
    # a linear charge term c n (a static gate drive)
    lin = [Term(Coefficient(0.7), [Factor(:varphi_1, nport())])]
    @test energies(materialize(leaf(PeriodicGrid(128)), spec, θ; extra_terms=lin)) ≈ energies(materialize(leaf(ChargeBasis(20)), spec, θ; extra_terms=lin)) atol=1e-7
end

@testset "two transmons: a PeriodicGrid mode coupled through its charge; n_g only in coefficients" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(; g=0.1)
    tc = Node([Node([:varphi_1 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8)), Node([:varphi_2 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8))]; solver=Dense(), keep=lowest(10))
    tp = Node([Node([:varphi_1 => PeriodicGrid(96; order=8)]; solver=Dense(), keep=lowest(8)), Node([:varphi_2 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8))]; solver=Dense(), keep=lowest(10))
    @test energies(materialize(tp, spec, θ)) ≈ energies(materialize(tc, spec, θ)) atol=1e-6
    # the offset charge is a coefficient of the linear charge terms and the constant only; no operator data depends on it
    terms = circuit_terms(spec)
    with_ng1 = filter(t -> :n_g1 in KronFreeCircuits.free_symbols(t.coeff), terms)
    @test !isempty(with_ng1) && all(t -> length(t.factors) == 1 && t.factors[1].port == nport(), with_ng1)   # only the linear charge terms
    @test :n_g1 in KronFreeCircuits.free_symbols(constant_term(spec))
end

@testset "nport() on a PeriodicGrid mode above its leaf: ancestor ports, selection rules, coupling terms" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(; g=0.1)
    leaves(b1, b2=ChargeBasis(10)) = [Node([:varphi_1 => b1]; solver=Dense(), keep=lowest(8)), Node([:varphi_2 => b2]; solver=Dense(), keep=lowest(8))]
    top(b1, b2=ChargeBasis(10); kw...) = Node(leaves(b1, b2); solver=Dense(), keep=lowest(10), kw...)
    # the root's nport() on the grid mode is an ordinary routed port
    sp = materialize(top(PeriodicGrid(96); ports=[(:varphi_1, nport())]), spec, θ)
    sc = materialize(top(ChargeBasis(10); ports=[(:varphi_1, nport())]), spec, θ)
    @test abs.(port(sp, :varphi_1, nport())) ≈ abs.(port(sc, :varphi_1, nport())) atol=1e-5
    # a selection rule on the charge at the root: at 0.7 the closure from the ground state is states 1, 3, 6, 10, 13
    # of the lowest 20 (every modulus from them is at least 0.029 from the threshold)
    rule = connected([1], (:varphi_1, nport()), 0.7)
    rp = materialize(top(PeriodicGrid(96); compute=20, keep=rule), spec, θ)
    rc = materialize(top(ChargeBasis(10); compute=20, keep=rule), spec, θ)
    @test rp.root.kept == rc.root.kept == [1, 3, 6, 10, 13]
    @test energies(rp) ≈ energies(rc) atol=1e-5
    # a coupling 0.05 n_1 n_2 written with nport(), one or both modes on the grid
    plain = [Term(Coefficient(0.05), [Factor(:varphi_1, nport()), Factor(:varphi_2, nport())])]
    Ep = energies(materialize(top(PeriodicGrid(96)), spec, θ; extra_terms=plain))
    @test Ep ≈ energies(materialize(top(ChargeBasis(10)), spec, θ; extra_terms=plain)) atol=1e-5
    θ2 = (θ..., n_g2=0.15)
    @test energies(materialize(top(PeriodicGrid(48), PeriodicGrid(48)), spec, θ2; extra_terms=plain)) ≈
          energies(materialize(top(ChargeBasis(10), ChargeBasis(10)), spec, θ2; extra_terms=plain)) atol=1e-5
    # the term-list form at the root needs no spec: the same terms and constant give the same energies
    Et = energies(materialize(top(PeriodicGrid(96); ports=[(:varphi_1, nport())]), vcat(circuit_terms(spec), plain), θ; constant=constant_term(spec)))
    @test Et ≈ Ep atol=1e-12
end

@testset "two transmons with both offset charges nonzero: the sign of the offset, one or both modes on the grid" begin
    # with n_g2 = 0 the spectrum is invariant under n_g1 -> -n_g1, so the sign of the offset is fixed only when both
    # offset charges are nonzero and the coupling is strong enough to see it
    spec = load_spec(fixture("two_transmons.json")); θ = (two_transmons_params(; g=0.5)..., E_J1=3.0, E_J2=2.5, n_g2=0.15)
    leaf(m, b) = Node([m => b]; solver=Dense(), keep=lowest(8))
    top(b1, b2) = Node([leaf(:varphi_1, b1), leaf(:varphi_2, b2)]; solver=Dense(), keep=lowest(10))
    Ec = energies(materialize(top(ChargeBasis(12), ChargeBasis(12)), spec, θ))
    Ef = energies(materialize(top(ChargeBasis(12), ChargeBasis(12)), spec, (θ..., n_g2=-0.15)))
    @test maximum(abs, Ec .- Ef) > 1e-3                      # the relative sign of the offset charges is visible
    @test energies(materialize(top(PeriodicGrid(96; order=8), ChargeBasis(12)), spec, θ)) ≈ Ec atol=1e-6
    @test energies(materialize(top(ChargeBasis(12), PeriodicGrid(96; order=8)), spec, θ)) ≈ Ec atol=1e-6
    @test energies(materialize(top(PeriodicGrid(96; order=8), PeriodicGrid(96; order=8)), spec, θ)) ≈ Ec atol=1e-6
    # the same Hamiltonian in one leaf: both factors of the coupling inside one node
    one = Node([:varphi_1 => PeriodicGrid(48; order=8), :varphi_2 => ChargeBasis(8)]; solver=Dense(), keep=lowest(10))
    ref = Node([:varphi_1 => ChargeBasis(12), :varphi_2 => ChargeBasis(8)]; solver=Dense(), keep=lowest(10))
    @test energies(materialize(one, spec, θ)) ≈ energies(materialize(ref, spec, θ)) atol=1e-5
    # the constant is the full quadratic form of the offset charges, whatever the bases
    @test constant_term(spec)(θ) ≈ 4 * (θ.E_C1 * θ.n_g1^2 + θ.E_C2 * θ.n_g2^2 + 2 * θ.g / 8 * θ.n_g1 * θ.n_g2)
end

@testset "non-integer harmonic on a compact mode is rejected at load" begin
    txt = replace(read(fixture("cos2phi.json"), String), "\"omega\": [2]" => "\"omega\": [1.5]")
    @test_throws ArgumentError parse_spec(txt)
end
