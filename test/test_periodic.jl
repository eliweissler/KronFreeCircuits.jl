@testset "free rotor on the periodic grid vs the exact (n - n_g)^2 spectrum" begin
    spec = load_spec(fixture("transmon.json"))
    for ng in (0.0, 0.3, 0.5), N in (32, 64)
        sys = materialize(Node([:varphi_1 => PeriodicGrid(N; order=8)]; solver=Dense(), keep=lowest(6)), spec, (E_J=0.0, E_C=1.0, n_g=ng))
        exact = sort([4 * (n - ng)^2 for n in -8:8])[1:6]
        # the order-8 stencil's own error on the sixth level (|n - n_g| = 3 at n_g = 0) is 1.53e-4 at N = 32
        @test energies(sys) ≈ exact atol=(N == 32 ? 2e-4 : 1e-6)
        # exact for the discretization: the twisted stencil's symbol at q = n - n_g
        w = KronFreeCircuits.fd_weights(8, 2); Δ = 2π / N
        disc = sort([4 * real(-sum(w[j+5] * cis(j * (n - ng) * Δ) for j in -4:4) / Δ^2) for n in -8:8])[1:6]
        @test energies(sys) ≈ disc atol=1e-10
        H = KronFreeCircuits.operator(sys); A = Matrix(H)
        ng == 0.5 || ng == 0.0 ? (@test maximum(abs, imag(A)) == 0) : (@test maximum(abs, imag(A)) > 0)   # twist phase ±1 is real
    end
end

@testset "transmon: periodic grid vs charge basis; physical charge port" begin
    spec = load_spec(fixture("transmon.json")); θ = (E_J=20.0, E_C=1.0, n_g=0.3)
    sc = materialize(Node([:varphi_1 => ChargeBasis(20)]; solver=Dense(), keep=lowest(6), ports=[(:varphi_1, nport())]), spec, θ)
    sp = materialize(Node([:varphi_1 => PeriodicGrid(128; order=8)]; solver=Dense(), keep=lowest(6)), spec, θ)
    @test energies(sp) ≈ energies(sc) atol=1e-7
    @test abs.(KronFreeCircuits.charge_port(sp, :varphi_1)) ≈ abs.(port(sc, :varphi_1, nport())) atol=1e-6
    # the diagonal ⟨k|n|k⟩ is gauge invariant: it fixes the sign of the offset-charge identity piece
    @test real(diag(KronFreeCircuits.charge_port(sp, :varphi_1))) ≈ real(diag(port(sc, :varphi_1, nport()))) atol=1e-6
    # on a charge-basis mode the on-demand charge port is the formed port
    @test KronFreeCircuits.charge_port(sc, :varphi_1) ≈ port(sc, :varphi_1, nport()) atol=1e-12
    # the wrap-corner BandedOps in the sparse assembly used by ShiftInvert
    @test sparse(KronFreeCircuits.operator(sp)) ≈ Matrix(KronFreeCircuits.operator(sp))
    @test_throws ArgumentError KronFreeCircuits.charge_port(sp, :nope)
    # with E_J ≠ 0 the assembled operator is exactly real at n_g ∈ {0, ½} (twist phase ±1) and complex otherwise
    for (ng, isreal_) in ((0.0, true), (0.5, true), (0.3, false))
        s = materialize(Node([:varphi_1 => PeriodicGrid(64; order=8)]; solver=Dense(), keep=lowest(4)), spec, (θ..., n_g=ng))
        @test isreal_assembled(KronFreeCircuits.operator(s)) == isreal_
    end
end

@testset "cos 2φ: both parity sectors, exact half-period translation, divisibility" begin
    spec = load_spec(fixture("cos2phi.json"))
    θ2 = (E_C=1.0, n_g=0.2, E_1=0.0, E_2=15.0)
    sc = materialize(Node([:phi => ChargeBasis(20)]; solver=Dense(), keep=lowest(8)), spec, θ2)
    N = 128; b = PeriodicGrid(N; order=8)      # N = 64 is 1.1e-5 off on the eighth level (eighth-order decay, ×256 per doubling)
    sp = materialize(Node([:phi => b]; solver=Dense(), keep=lowest(8)), spec, θ2)
    @test energies(sp) ≈ energies(sc) atol=1e-7
    # the half-period translation T: φ -> φ + π is an exact grid translation on the twisted function space, so it must
    # carry the twist phase on the half that wraps (the listing's shift(N ÷ 2) construction, same convention as Task 7's test);
    # then T² = cis(±2π n_g) I, T commutes with H, and the kept-span projector commutes with T
    V = eigenvectors(sp)
    T = circshift(Matrix{ComplexF64}(I, N, N), (N ÷ 2, 0)); T[1:N÷2, :] .*= cis(2π * θ2.n_g)
    phase = (T^2)[1, 1]; @test T^2 ≈ phase * I
    @test norm(T * (V * V') - (V * V') * T) < 1e-9
    # the sectors are the two eigenvalues ±√phase of T; both are present among the kept states
    parity = [real(dot(V[:, j], T * V[:, j]) / cis(angle(phase) / 2)) for j in 1:8]
    @test all(abs.(abs.(parity) .- 1) .< 1e-6) && any(>(0.5), parity) && any(<(-0.5), parity)
    @test_throws ArgumentError materialize(Node([:phi => PeriodicGrid(63; order=8)]; solver=Dense(), keep=lowest(8)), spec, θ2)   # 63 not divisible by 2
    # a weak cos φ mixes the sectors: T no longer commutes with the kept span
    sm = materialize(Node([:phi => b]; solver=Dense(), keep=lowest(8)), spec, (E_C=1.0, n_g=0.2, E_1=0.5, E_2=15.0))
    Vm = eigenvectors(sm)
    @test norm(T * (Vm * Vm') - (Vm * Vm') * T) > 1e-3
end

@testset "a twisted mode coupled through its charge: two transmons" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(; g=0.1)
    tc = Node([Node([:varphi_1 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8)), Node([:varphi_2 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8))]; solver=Dense(), keep=lowest(10))
    tp = Node([Node([:varphi_1 => PeriodicGrid(96; order=8)]; solver=Dense(), keep=lowest(8)), Node([:varphi_2 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8))]; solver=Dense(), keep=lowest(10))
    @test energies(materialize(tp, spec, θ)) ≈ energies(materialize(tc, spec, θ)) atol=1e-6
    # the twist phases are coefficients: n_g1 is a free symbol of the wrap terms, and no operator data depends on it
    terms = circuit_terms(spec, Dict(:varphi_1 => PeriodicGrid(96; order=8), :varphi_2 => ChargeBasis(10)))
    wrap = filter(t -> any(f -> f.port.kind in (:d1wrap, :d2wrap), t.factors), terms)
    @test !isempty(wrap) && all(t -> :n_g1 in KronFreeCircuits.free_symbols(t.coeff), wrap)
end

@testset "a twisted mode's charge above the leaf: the documented recipe for ports and coupling terms" begin
    spec = load_spec(fixture("two_transmons.json")); θ = two_transmons_params(; g=0.1); ng = θ.n_g1
    leaves(b1) = [Node([:varphi_1 => b1]; solver=Dense(), keep=lowest(8)), Node([:varphi_2 => ChargeBasis(10)]; solver=Dense(), keep=lowest(8))]
    # nport() on the twisted mode is refused, naming both routes
    err = try materialize(Node(leaves(PeriodicGrid(96)); solver=Dense(), keep=lowest(10), ports=[(:varphi_1, nport())]), spec, θ); nothing catch e; e end
    @test err isa ArgumentError && occursin("charge_port", err.msg) && occursin("Port(:d1wrap, -1)", err.msg)
    # the root port from its three pieces equals the charge-basis root port
    pieces = [(:varphi_1, Port(:d1, 0)), (:varphi_1, Port(:d1wrap, 1)), (:varphi_1, Port(:d1wrap, -1))]
    sp = materialize(Node(leaves(PeriodicGrid(96)); solver=Dense(), keep=lowest(10), ports=pieces), spec, θ)
    sc = materialize(Node(leaves(ChargeBasis(10)); solver=Dense(), keep=lowest(10), ports=[(:varphi_1, nport())]), spec, θ)
    n = port(sp, :varphi_1, Port(:d1, 0)) + cispi(2ng) * port(sp, :varphi_1, Port(:d1wrap, 1)) +
        cispi(-2ng) * port(sp, :varphi_1, Port(:d1wrap, -1)) + ng * I
    @test abs.(n) ≈ abs.(port(sc, :varphi_1, nport())) atol=1e-5
    # a coupling 0.05 n_1 n_2 as three twisted terms plus the n_g share on the other factor
    c = Coefficient(0.05); ngc = Coefficient("n_g1", [:n_g1])
    twisted = [Term(c, [Factor(:varphi_1, Port(:d1, 0)), Factor(:varphi_2, nport())]),
               Term(c * twist_factor(ngc, 1), [Factor(:varphi_1, Port(:d1wrap, 1)), Factor(:varphi_2, nport())]),
               Term(c * twist_factor(ngc, -1), [Factor(:varphi_1, Port(:d1wrap, -1)), Factor(:varphi_2, nport())]),
               Term(c * ngc, [Factor(:varphi_2, nport())])]
    plain = [Term(c, [Factor(:varphi_1, nport()), Factor(:varphi_2, nport())])]
    @test energies(materialize(Node(leaves(PeriodicGrid(96)); solver=Dense(), keep=lowest(10)), spec, θ; extra_terms=twisted)) ≈
          energies(materialize(Node(leaves(ChargeBasis(10)); solver=Dense(), keep=lowest(10)), spec, θ; extra_terms=plain)) atol=1e-5
end

@testset "two transmons with both offset charges nonzero: the twist sign, one or both modes twisted" begin
    # with n_g2 = 0 the spectrum is invariant under n_g1 -> -n_g1 (flip both phases), so the sign of the
    # twist is fixed only when both offset charges are nonzero and the coupling is strong enough to see it
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
    # bookkeeping: the twisted mode has no linear charge term and no share of the constant
    bases = Dict{Symbol,Basis}(:varphi_1 => PeriodicGrid(48; order=8), :varphi_2 => ChargeBasis(8))
    ts = circuit_terms(spec, bases)
    @test !any(t -> any(f -> f.mode === :varphi_1 && f.port in (nport(), n2port()), t.factors), ts)
    @test constant_term(spec, bases)(θ) ≈ 4 * θ.E_C2 * θ.n_g2^2
    @test constant_term(spec)(θ) ≈ 4 * (θ.E_C1 * θ.n_g1^2 + θ.E_C2 * θ.n_g2^2 + 2 * θ.g / 8 * θ.n_g1 * θ.n_g2)
end

@testset "circuit_terms(spec) is the charge-basis expansion: identical to an empty basis dictionary" begin
    for f in ("transmon.json", "two_transmons.json", "three_transmons.json", "zero_pi.json", "full_zero_pi.json", "fluxonium.json", "cos2phi.json")
        spec = load_spec(fixture(f))
        a, b = circuit_terms(spec), circuit_terms(spec, Dict{Symbol,Basis}())
        @test length(a) == length(b) && all(t.coeff.ex == u.coeff.ex && t.factors == u.factors for (t, u) in zip(a, b))
        @test constant_term(spec).ex == constant_term(spec, Dict{Symbol,Basis}()).ex
    end
end

@testset "non-integer harmonic on a compact mode is rejected at load" begin
    txt = replace(read(fixture("cos2phi.json"), String), "\"omega\": [2]" => "\"omega\": [1.5]")
    @test_throws ArgumentError parse_spec(txt)
end
