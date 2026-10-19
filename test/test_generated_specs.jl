# Provenance of the hand-written fixtures: each test/fixtures/generated/<name>.json is the same Hamiltonian built as a
# lumped circuit and exported by circuit-quantization (test/fixtures/generate_specs.py, which records the circuit,
# coordinates, flux gauge and parameter mapping in each file's `_comment`).
# For each pair the two specs must define the same Hamiltonian: the same modes, the same E_C, E_L, flux_linear, n_g and
# constant as functions of the parameters, the same cosines up to cos(-x) = cos(x), and the same energies on one tree.
# Every difference that is a convention is asserted explicitly, with its reason.
#
# cos2phi.json has no generated counterpart: no lumped circuit has one compact mode with junction multipliers 2 and 1
# (two such junctions have non-parallel incidence rows, so the circuit has at least two dynamical modes); see
# `cos2phi_not_expressible` in generate_specs.py.

using Random
using KronFreeCircuits: Spec, CosineSpec, parse_spec, load_spec, Node, Dense, lowest, ChargeBasis, PhaseGrid, FockBasis,
                        materialize, energies

generated(name) = fixture(joinpath("generated", name * ".json"))

# |a - b| within 1e-12 relative (absolute below 1).
_close(a, b; tol=1e-12) = abs(a - b) <= tol * max(1.0, abs(a), abs(b))

# The spec restricted to the modes `keep`, after asserting that every other mode is exactly decoupled at θ:
# no E_C, E_L cross entries, no flux_linear or n_g, and no cosine acting on it.
function decoupled_restriction(s::Spec, keep::Vector{Int}, θ)
    drop = setdiff(1:length(s.modes), keep)
    for d in drop, k in keep
        @test s.E_C[d, k](θ) == 0 && s.E_C[k, d](θ) == 0
        @test s.E_L[d, k](θ) == 0 && s.E_L[k, d](θ) == 0
    end
    for d in drop
        @test s.flux_linear[d](θ) == 0 && s.n_g[d](θ) == 0
        @test all(c -> c.omega[d] == 0, s.cosines)
    end
    return Spec(s.parameters, s.modes[keep], s.E_C[keep, keep], s.E_L[keep, keep], s.flux_linear[keep], s.n_g[keep],
                [CosineSpec(c.energy, c.omega[keep], c.offset) for c in s.cosines], s.charge_cosines, s.constant)
end

# The cosines evaluated at θ as (omega, energy, offset), each with the sign of (omega, offset) fixed so that the first
# nonzero omega entry is positive (cos(-x) = cos(x)); with `merge`, terms with equal (omega, offset) are summed.
function canonical_cosines(s::Spec, θ; merge=false)
    out = Tuple{Vector{Float64},Float64,Float64}[]
    for c in s.cosines
        w, E, δ = copy(c.omega), real(c.energy(θ)), real(c.offset(θ))
        if w[findfirst(!iszero, w)] < 0
            w, δ = -w, -δ
        end
        i = merge ? findfirst(t -> t[1] == w && _close(t[3], δ), out) : nothing
        i === nothing ? push!(out, (w, E, δ)) : (out[i] = (w, out[i][2] + E, out[i][3]))
    end
    return sort(out; by=t -> (t[1], t[3]))
end

function same_cosines(a::Spec, b::Spec, θ; merge=false)
    ca, cb = canonical_cosines(a, θ; merge), canonical_cosines(b, θ; merge)
    length(ca) == length(cb) || return false
    return all(x[1] == y[1] && _close(x[2], y[2]) && _close(x[3], y[3]) for (x, y) in zip(ca, cb))
end

# The coefficient comparison at θ; `Δconstant(θ)` is the documented hand-minus-generated constant.
function same_hamiltonian(gen::Spec, hand::Spec, θ; Δconstant=θ -> 0.0, merge=false)
    @test [(m.name, m.type) for m in gen.modes] == [(m.name, m.type) for m in hand.modes]
    N = length(hand.modes)
    @test all(_close(gen.E_C[i, j](θ), hand.E_C[i, j](θ)) for i in 1:N, j in 1:N)
    @test all(_close(gen.E_L[i, j](θ), hand.E_L[i, j](θ)) for i in 1:N, j in 1:N)
    @test all(_close(gen.flux_linear[i](θ), hand.flux_linear[i](θ)) for i in 1:N)
    @test all(_close(gen.n_g[i](θ), hand.n_g[i](θ)) for i in 1:N)
    @test same_cosines(gen, hand, θ; merge)
    @test _close(hand.constant(θ) - gen.constant(θ), Δconstant(θ))
    @test isempty(gen.charge_cosines) && isempty(hand.charge_cosines)
end

# Energies of both specs on the same Dense tree at θ, the hand fixture's shifted by -Δconstant.
# The trees are deliberately small (only the two specs on the same discretization are compared), so a PhaseGrid edge
# warning is expected and is the only log allowed.
function same_energies(gen::Spec, hand::Spec, tree, θ; Δconstant=θ -> 0.0)
    quiet(s) = begin
        logs, sys = Test.collect_test_logs(() -> materialize(tree, s, θ))
        @test all(l -> occursin(r"PhaseGrid on mode :\w+ is too small", l.message), logs)
        energies(sys)
    end
    Eg, Eh = quiet(gen), quiet(hand)
    @test maximum(abs.(Eh .- Δconstant(θ) .- Eg)) <= 1e-12 * max(1.0, maximum(abs, Eh))
end

u(rng, a, b) = a + (b - a) * rand(rng)

@testset "generated specs: every file loads" begin
    for f in readdir(fixture("generated"))
        endswith(f, ".json") || continue
        s = load_spec(fixture(joinpath("generated", f)))
        @test s isa Spec
        @test all(p -> !(p in (:im, :cis, :cispi)), s.parameters)
    end
    @test sort(filter(f -> endswith(f, ".json"), readdir(fixture("generated")))) ==
          sort(["transmon.json", "two_transmons.json", "three_transmons.json", "fluxonium.json", "zero_pi.json",
                "full_zero_pi.json", "two_zero_pi.json"])
end

@testset "generated vs hand: transmon" begin
    gen, hand = load_spec(generated("transmon")), load_spec(fixture("transmon.json"))
    @test Set(gen.parameters) == Set(hand.parameters)
    rng = MersenneTwister(61)
    pt() = (E_J=u(rng, 5, 30), E_C=u(rng, 0.5, 2), n_g=u(rng, -0.5, 0.5))
    for _ in 1:3
        same_hamiltonian(gen, hand, pt())
    end
    same_energies(gen, hand, Node([:varphi_1 => ChargeBasis(10)]; solver=Dense(), keep=lowest(6)), pt())
end

@testset "generated vs hand: $name" for (name, n) in (("two_transmons", 2), ("three_transmons", 3))
    # g enters through the coupling capacitor (generate_specs.py, transmons()); the generated E_C reproduces the hand
    # fixture's [[E_C1, g/8], [g/8, E_C2]] exactly.
    gen, hand = load_spec(generated(name)), load_spec(fixture(name * ".json"))
    @test Set(gen.parameters) == Set(hand.parameters)
    rng = MersenneTwister(62 + n)
    function pt()
        p = (E_C1=u(rng, 0.5, 2), E_C2=u(rng, 0.5, 2), g=u(rng, 0.05, 0.5),
             E_J1=u(rng, 5, 30), E_J2=u(rng, 5, 30), n_g1=u(rng, -0.5, 0.5), n_g2=u(rng, -0.5, 0.5))
        return n == 3 ? merge(p, (E_C3=u(rng, 0.5, 2), E_J3=u(rng, 5, 30), n_g3=u(rng, -0.5, 0.5))) : p
    end
    for _ in 1:3
        same_hamiltonian(gen, hand, pt())
    end
    leaf(i) = Node([Symbol("varphi_$i") => ChargeBasis(5)]; solver=Dense(), keep=lowest(4))
    same_energies(gen, hand, Node([leaf(i) for i in 1:n]; solver=Dense(), keep=lowest(8)), pt())
end

@testset "generated vs hand: fluxonium" begin
    gen, hand = load_spec(generated("fluxonium")), load_spec(fixture("fluxonium.json"))
    @test Set(gen.parameters) == Set(hand.parameters)
    # Convention: circuit-quantization omits the flux-only inductive energy E_L varphi_ext^2/2 of
    # (1/2) E_L (varphi + varphi_ext)^2 (spec-format.md, "Complete example"); the hand fixture keeps it.
    Δ(θ) = θ.E_L * θ.varphi_ext^2 / 2
    rng = MersenneTwister(64)
    pt() = (E_J=u(rng, 2, 10), E_C=u(rng, 0.5, 2), E_L=u(rng, 0.5, 1.5), varphi_ext=u(rng, -π, π))
    for _ in 1:3
        same_hamiltonian(gen, hand, pt(); Δconstant=Δ)
    end
    same_energies(gen, hand, Node([:varphi_1 => PhaseGrid(-6π, 6π, 121)]; solver=Dense(), keep=lowest(6)), pt();
                  Δconstant=Δ)
end

@testset "generated vs hand: zero_pi" begin
    gen3, hand = load_spec(generated("zero_pi")), load_spec(fixture("zero_pi.json"))
    @test Set(gen3.parameters) == Set(hand.parameters)
    # Convention: scqubits' ZeroPi.sparse_potential_mat adds 2 E_J to the potential (zero_pi.json's comment); the
    # circuit has no such constant.
    # zero_pi.json already uses circuit-quantization's (n - n_g) convention (not scqubits' (n + n_g)), so n_g must be equal.
    Δ(θ) = 2θ.E_J
    rng = MersenneTwister(65)
    pt() = (E_J=u(rng, 5, 15), E_L=u(rng, 0.1, 1), E_CJ=u(rng, 10, 30), E_CS=u(rng, 0.2, 1),
            varphi_ext=u(rng, -π, π), n_g=u(rng, -0.5, 0.5))
    for _ in 1:3
        θ = pt()
        # The circuit's zeta is an exactly decoupled harmonic mode (scqubits' ZeroPi drops it); here it is
        # scqubits' FullZeroPi zeta: E_C = E_Cz = E_CS E_CJ/(E_CJ - E_CS), E_L = E_L.
        @test [(m.name, m.type) for m in gen3.modes] == [(:theta, :c), (:phi, :e), (:zeta, :h)]
        @test _close(gen3.E_C[3, 3](θ), θ.E_CS * θ.E_CJ / (θ.E_CJ - θ.E_CS)) && _close(gen3.E_L[3, 3](θ), θ.E_L)
        same_hamiltonian(decoupled_restriction(gen3, [1, 2], θ), hand, θ; Δconstant=Δ)
    end
    θ = pt()
    gen = decoupled_restriction(gen3, [1, 2], θ)
    tree = Node([:theta => ChargeBasis(4), :phi => PhaseGrid(-4π, 4π, 41)]; solver=Dense(), keep=lowest(6))
    same_energies(gen, hand, tree, θ; Δconstant=Δ)
end

@testset "generated vs hand: full_zero_pi" begin
    gen, hand = load_spec(generated("full_zero_pi")), load_spec(fixture("full_zero_pi.json"))
    # E_Cz is not a circuit parameter: the large capacitors fix it as E_CS E_CJ/(E_CJ - E_CS) (scqubits derives EC the
    # same way when ECS is given).
    @test Set(gen.parameters) == setdiff(Set(hand.parameters), [:E_Cz])
    Ecz(θ) = θ.E_CS * θ.E_CJ / (θ.E_CJ - θ.E_CS)
    # Conventions: scqubits' potential adds 2 E_J, and FullZeroPi writes the zeta oscillator as E_zeta a^dag a without
    # its zero-point energy E_zeta/2 = sqrt(2 E_Cz E_L) (full_zero_pi.json's comment).
    # The fixture's offset-charge correction 2 E_CS n_g^2 (1 - 1/D) vanishes at dC_J = dC = 0 (D = 1).
    Δ(θ) = 2θ.E_J - sqrt(2θ.E_Cz * θ.E_L)
    rng = MersenneTwister(66)
    function pt(; dC_J=0.0, dC=0.0)
        p = (E_J=u(rng, 5, 15), E_L=u(rng, 0.1, 1), E_CJ=u(rng, 10, 30), E_CS=u(rng, 0.2, 1), varphi_ext=u(rng, -π, π),
             n_g=u(rng, -0.5, 0.5), dE_J=u(rng, -0.1, 0.1), dE_L=u(rng, -0.1, 0.1), dC_J=dC_J, dC=dC)
        return merge(p, (E_Cz=Ecz(p),))
    end
    # scqubits' FullZeroPi is first order in the capacitive asymmetries dC_J, dC; the circuit is exact.  They agree
    # exactly at dC_J = dC = 0 (dE_J, dE_L enter both linearly).  The fixture splits each junction's energy into two
    # cosines, E_J and +-E_J dE_J/2 (scqubits' cos cos and sin sin terms), which merge into the circuit's
    # E_J (1 +- dE_J/2).
    @test length(hand.cosines) == 2length(gen.cosines)
    for _ in 1:3
        same_hamiltonian(gen, hand, pt(); Δconstant=Δ, merge=true)
    end
    # Away from dC_J = dC = 0 the charging matrices differ at second order, with the coefficients of (generated - hand)
    # derived by series in sympy from the two files: checked as the limit of the gap over ε² along (dC_J, dC) = ε (a, b).
    θ = pt()
    a, b = 0.7, -0.9
    at(ε) = merge(θ, (dC_J=a * ε, dC=b * ε))
    gapC(i, j, ε) = real(gen.E_C[i, j](at(ε)) - hand.E_C[i, j](at(ε)))
    S, J = θ.E_CS, θ.E_CJ
    second = Dict((1, 1) => S * ((J - S) * b^2 + S * a^2) / (8J), (2, 2) => S * a^2 / 8, (3, 3) => S * b^2 / 4,
                  (2, 3) => -sqrt(2) * S * a * b / 8, (1, 2) => 0.0, (1, 3) => 0.0)
    ε = 1e-4
    for ((i, j), c) in second
        @test abs(gapC(i, j, ε) / ε^2 - c) <= 1e-4 * S   # observed residual (third order) up to 3e-6
    end
    # The fixture's θ offset charge n_g/D, D = 1 - E_CS dC_J²/(4E_CJ) - E_CS dC²/(4E_Cz), differs from the circuit's
    # island offset n_g at second order: halving the asymmetries quarters the gap, with coefficient
    # -n_g (E_CS a² + (E_CJ - E_CS) b²)/(4E_CJ).
    gapn(ε) = real(gen.n_g[1](at(ε)) - hand.n_g[1](at(ε)))
    @test 3.99 < gapn(1e-3) / gapn(5e-4) < 4.01
    @test abs(gapn(ε) / ε^2 + θ.n_g * (S * a^2 + (J - S) * b^2) / (4J)) <= 1e-4 * abs(θ.n_g) * S
    θ = pt()
    zp = Node([:theta => ChargeBasis(3), :phi => PhaseGrid(-4π, 4π, 31)]; solver=Dense(), keep=lowest(8))
    zeta = Node([:zeta => FockBasis(5; phi_zpf=(2θ.E_Cz / θ.E_L)^(1 / 4))]; solver=Dense(), keep=lowest(5))
    same_energies(gen, hand, Node([zp, zeta]; solver=Dense(), keep=lowest(6)), θ; Δconstant=Δ)
end

@testset "generated vs hand: two_zero_pi" begin
    gen6, hand = load_spec(generated("two_zero_pi")), load_spec(fixture("two_zero_pi.json"))
    @test Set(gen6.parameters) == Set(hand.parameters)
    # Convention: scqubits' potential adds 2 E_J per zero-pi (two_zero_pi.json's comment).
    Δ(θ) = 2θ.E_J1 + 2θ.E_J2
    rng = MersenneTwister(67)
    function pt()
        q(i) = (Symbol("E_J$i") => u(rng, 5, 15), Symbol("E_L$i") => u(rng, 0.1, 1), Symbol("E_CJ$i") => u(rng, 10, 30),
                Symbol("E_CS$i") => u(rng, 0.2, 1), Symbol("varphi_ext$i") => u(rng, -π, π), Symbol("n_g$i") => u(rng, -0.5, 0.5))
        return (; q(1)..., q(2)..., g=u(rng, 0.005, 0.02))
    end
    for _ in 1:3
        θ = pt()
        # zeta1, zeta2 are exactly decoupled harmonic modes (two_zero_pi.json, like zero_pi.json, has none).
        @test [m.name for m in gen6.modes] == [:theta1, :phi1, :theta2, :phi2, :zeta1, :zeta2]
        same_hamiltonian(decoupled_restriction(gen6, [1, 2, 3, 4], θ), hand, θ; Δconstant=Δ)
    end
    θ = pt()
    gen = decoupled_restriction(gen6, [1, 2, 3, 4], θ)
    leaf(i) = Node([Symbol("theta$i") => ChargeBasis(3), Symbol("phi$i") => PhaseGrid(-4π, 4π, 31)]; solver=Dense(), keep=lowest(5))
    same_energies(gen, hand, Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(6)), θ; Δconstant=Δ)
end

@testset "n_g on a noncompact mode is accepted and expanded" begin
    # The format keeps voltage-source offsets in n_g on any mode (spec-format.md); none of the generated circuits has a
    # source, so check the loader on the generated fluxonium with n_g = a on its extended mode: parse_spec accepts it,
    # and 4 E_C (n - a)^2 is unitarily equivalent to 4 E_C n^2 (n -> n + a is e^{i a varphi} on a noncompact mode).
    d = JSON3.read(read(generated("fluxonium"), String), Dict{String,Any})
    d["n_g"] = ["0.3"]
    s = parse_spec(JSON3.write(d))
    @test real(s.n_g[1]((;))) == 0.3
    θ = (E_J=4.0, E_C=1.0, E_L=1.0, varphi_ext=0.4)
    tree = Node([:varphi_1 => PhaseGrid(-8π, 8π, 401)]; solver=Dense(), keep=lowest(4))
    @test energies(materialize(tree, s, θ)) ≈ energies(materialize(tree, load_spec(generated("fluxonium")), θ)) atol=1e-6
end
