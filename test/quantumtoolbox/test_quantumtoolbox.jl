using KronFreeCircuits: operator

# Two coupled transmons, every product state kept at the top, so the dressed basis (n = 36) and the matrix-free basis
# span the same space. Real: charge-basis leaves (n_g only shifts their diagonal). Complex: periodic-grid leaves.
spec = load_spec(fixture("two_transmons.json"))
_extra(b) = b isa ChargeBasis ? exp2pin(0.1) : expiphi(1)      # exp2pin is a complex-character port on a real node
leaf(i, b) = Node([Symbol("varphi_$i") => b]; keep=lowest(6), ports=[(Symbol("varphi_$i"), q) for q in (nport(), _extra(b))])
top(b) = Node([leaf(1, b), leaf(2, b)]; solver=Dense(), keep=lowest(36), ports=[(:varphi_1, nport()), (:varphi_1, _extra(b))])
const SYSTEMS = [(:real, materialize(top(ChargeBasis(15)), spec, two_transmons_params(g=0.3))),
                 (:complex, materialize(top(PeriodicGrid(32)), spec, two_transmons_params(g=0.3)))]
@test arithmetic(SYSTEMS[1][2]) === :real && arithmetic(SYSTEMS[2][2]) === :complex

dense_operator(O) = O * Matrix{ComplexF64}(I, size(O)...)
errmsg(f) = try f(); nothing catch e; e isa ArgumentError ? e.msg : e end

@testset "static objects ($kind)" for (kind, sys) in SYSTEMS
    E = energies(sys); V = eigenvectors(sys); N = 36
    @test qt_hamiltonian(sys).data ≈ 2π * Diagonal(E .- E[1])
    @test size(qt_hamiltonian(sys; n=10)) == (10, 10)

    Href = 2π * (dense_operator(operator(sys)) - E[1] * I)
    x = randn(ComplexF64, N)
    y = similar(x); mul!(y, qt_hamiltonian(sys; basis=:free).data, x)
    @test y ≈ Href * x
    X = randn(ComplexF64, N, 3)
    Y = similar(X); mul!(Y, qt_hamiltonian(sys; basis=:free, batch=true).data, X)
    @test Y ≈ Href * X

    @test qt_operator(sys, (:varphi_1, nport())).data ≈ port(sys, :varphi_1, nport())
    @test size(qt_operator(sys, (:varphi_1, nport()); n=10)) == (10, 10)
    Pref = dense_operator(port_operator(sys, (:varphi_1, nport())))
    mul!(y, qt_operator(sys, (:varphi_1, nport()); basis=:free).data, x)
    @test y ≈ Pref * x

    p11 = LinearIndices((6, 6))[1, 1]
    kf = qt_ket(sys, (1, 1); basis=:free)
    @test isket(kf) && kf.data == Float64.(1:N .== p11)
    @test to_dressed(sys, kf).data ≈ conj(V[p11, :])
    @test size(to_dressed(sys, kf; n=10)) == (10,)
    kd = qt_ket(sys, (1, 1))
    @test isket(kd) && kd.data == Float64.(1:N .== dressed_index(sys, (1, 1)))

    B = qt_ket(sys, [(1, 1), (2, 1)])
    @test isoper(B) && size(B) == (N, 2)
    @test B.data[:, 1] == qt_ket(sys, (1, 1)).data && B.data[:, 2] == qt_ket(sys, (2, 1)).data
    Bf = qt_ket(sys, [(1, 1), (2, 1)]; basis=:free)
    @test to_dressed(sys, Bf).data ≈ V' * Bf.data && size(to_dressed(sys, Bf)) == (N, 2)

    @test occursin("37", errmsg(() -> qt_hamiltonian(sys; n=37)))
    @test occursin("bogus", errmsg(() -> qt_hamiltonian(sys; basis=:bogus)))
    @test occursin("(7, 1)", errmsg(() -> qt_ket(sys, (7, 1))))
    @test occursin("(7, 1)", errmsg(() -> qt_ket(sys, (7, 1); basis=:free)))
    @test occursin("37", errmsg(() -> qt_operator(sys, (:varphi_1, nport()); n=37)))
    if CUDA.functional()
        @test occursin("CuArray", errmsg(() -> qt_hamiltonian(sys; on=CuArray)))
    end
end

const TLIST = collect(range(0, 20; length=21))
evolve(H, ψ0) = sesolve(H, ψ0, TLIST; reltol=1e-10, abstol=1e-12, progress_bar=Val(false)).states[end]
function drive_of(sys)
    E = energies(sys); ωd = 2π * (E[2] - E[1])
    return (p, t) -> 0.05 * cos(ωd * t)
end
const LABELS = [(1, 1), (2, 1), (1, 2)]

@testset "lab drives ($kind)" for (kind, sys) in SYSTEMS
    f = drive_of(sys)
    drives = [(:varphi_1, nport()) => f]
    Hd, frd = drive_hamiltonian(sys, drives; n=36)
    Hf, frf = drive_hamiltonian(sys, drives; basis=:free)
    @test frd.ω === nothing && frf.ω === nothing

    # the same physical initial state in both bases: the product state (2, 1), and its dressed amplitudes
    ψf0 = qt_ket(sys, (2, 1); basis=:free)
    ψd = evolve(Hd, to_dressed(sys, ψf0))
    ψf = evolve(Hf, ψf0)
    @test isapprox(to_dressed(sys, ψf).data, ψd.data; atol=1e-7)

    # partial propagator: an N×3 block evolves as three single kets
    Hfb, _ = drive_hamiltonian(sys, drives; basis=:free, batch=true)
    Bf = evolve(Hfb, qt_ket(sys, LABELS; basis=:free))
    for (j, l) in enumerate(LABELS)
        @test isapprox(Bf.data[:, j], evolve(Hf, qt_ket(sys, l; basis=:free)).data; atol=1e-7)
    end
    Bd = evolve(Hd, qt_ket(sys, LABELS))
    for (j, l) in enumerate(LABELS)
        @test isapprox(Bd.data[:, j], evolve(Hd, qt_ket(sys, l)).data; atol=1e-7)
    end

    # the other formed port: complex-character on the real system (exp2pin), expiphi on the complex one
    drives2 = [(:varphi_1, kind === :real ? exp2pin(0.1) : expiphi(1)) => f]
    ψd2 = evolve(first(drive_hamiltonian(sys, drives2; n=36)), to_dressed(sys, ψf0))
    ψf2 = evolve(first(drive_hamiltonian(sys, drives2; basis=:free)), ψf0)
    @test isapprox(to_dressed(sys, ψf2).data, ψd2.data; atol=1e-7)

    @test errmsg(() -> drive_hamiltonian(sys, [(:varphi_2, n2port()) => f])) isa String
    @test errmsg(() -> drive_hamiltonian(sys, [(:varphi_2, n2port()) => f]; basis=:free)) isa String
    @test occursin("bogus", errmsg(() -> drive_hamiltonian(sys, drives; frame=:bogus)))
end

@testset "to_lab on arrays" begin
    ψ = [1.0, 2.0]
    @test to_lab(Frame(nothing), ψ, 3.0) === ψ
    @test to_lab(Frame([0.0, 1.0]), [1.0, 1.0], π) ≈ [1, -1]
    @test to_lab(Frame([0.0, 1.0]), [1.0 2.0; 1.0 2.0], π) ≈ [1 2; -1 -2]
end

@testset "interaction picture ($kind)" for (kind, sys) in SYSTEMS
    f = drive_of(sys)
    drives = [(:varphi_1, nport()) => f]
    T = TLIST[end]
    for basis in (:dressed, :free)
        kw = basis === :dressed ? (; n=36) : (; basis=:free)
        ket(l) = basis === :dressed ? qt_ket(sys, l) : qt_ket(sys, l; basis=:free)
        Hl, _ = drive_hamiltonian(sys, drives; kw...)
        Hi, fr = drive_hamiltonian(sys, drives; frame=:interaction, kw...)
        @test fr.ω isa Vector{Float64} && length(fr.ω) == 36
        ψ0 = ket((2, 1))
        @test isapprox(to_lab(fr, evolve(Hi, ψ0), T).data, evolve(Hl, ψ0).data; atol=1e-7)

        Hlb, _ = drive_hamiltonian(sys, drives; batch=true, kw...)
        Hib, frb = drive_hamiltonian(sys, drives; frame=:interaction, batch=true, kw...)
        B0 = ket(LABELS)
        Bi = evolve(Hib, B0)
        @test size(to_lab(frb, Bi, T)) == (36, 3)
        @test isapprox(to_lab(frb, Bi, T).data, evolve(Hlb, B0).data; atol=1e-7)
    end
    @test drive_hamiltonian(sys, drives; n=36, frame=:interaction)[2].ω ≈ 2π * (energies(sys) .- energies(sys)[1])

    # no drives: the dressed interaction-picture Hamiltonian vanishes and the state stays put
    Hi0, _ = drive_hamiltonian(sys, Pair[]; frame=:interaction, n=36)
    ψ0 = qt_ket(sys, (2, 1))
    @test isapprox(evolve(Hi0, ψ0).data, ψ0.data; atol=1e-12)
end

@testset "no drives, every basis and frame ($kind)" for (kind, sys) in SYSTEMS
    Ω = 2π * (energies(sys) .- energies(sys)[1])
    T = TLIST[end]
    ψd0 = qt_ket(sys, (2, 1))
    i = dressed_index(sys, (2, 1))
    ψd = evolve(first(drive_hamiltonian(sys, Pair[])), ψd0)                         # dressed lab: H0 alone
    @test isapprox(ψd.data, cis(-Ω[i] * T) .* ψd0.data; atol=1e-7)
    ψf0 = qt_ket(sys, (2, 1); basis=:free)
    ψfl = evolve(first(drive_hamiltonian(sys, Pair[]; basis=:free)), ψf0)
    @test isapprox(to_dressed(sys, ψfl).data, cis.(-Ω .* T) .* to_dressed(sys, ψf0).data; atol=1e-7)
    Hfi, fr = drive_hamiltonian(sys, Pair[]; basis=:free, frame=:interaction)
    @test isapprox(to_lab(fr, evolve(Hfi, ψf0), T).data, ψfl.data; atol=1e-7)
end

@testset "GPU ($kind)" for (kind, sys) in SYSTEMS
    if !CUDA.functional()
        @test_skip "no functional CUDA device"
    else
        f = drive_of(sys)
        drives = [(:varphi_1, nport()) => f]
        # 2 ns, not 20: at 36 states the device is kernel-launch bound (20 ns took ~10 min per system)
        tg = [0.0, 1.0, 2.0]
        evolve_g(H, ψ0) = sesolve(H, ψ0, tg; reltol=1e-10, abstol=1e-12, progress_bar=Val(false)).states[end]
        T = tg[end]
        for frame in (:lab, :interaction), batch in (false, true)
            ket(on) = batch ? qt_ket(sys, LABELS; basis=:free, on) : qt_ket(sys, (2, 1); basis=:free, on)
            ψ0g = ket(CuArray)
            @test ψ0g.data isa CuArray
            Hc, frc = drive_hamiltonian(sys, drives; basis=:free, frame, batch)
            Hg, frg = drive_hamiltonian(sys, drives; basis=:free, frame, batch, on=CuArray)
            ψc = evolve_g(Hc, ket(Array))
            ψg = evolve_g(Hg, ψ0g)
            @test ψg.data isa CuArray
            @test isapprox(Array(ψg.data), ψc.data; atol=1e-7)
            lg = to_lab(frg, ψg, T)
            @test lg.data isa CuArray
            @test isapprox(Array(lg.data), to_lab(frc, ψc, T).data; atol=1e-7)
        end
        @test to_dressed(sys, qt_ket(sys, (2, 1); basis=:free, on=CuArray)).data ≈
              to_dressed(sys, qt_ket(sys, (2, 1); basis=:free)).data
    end
end

@testset "dressed interaction picture takes a ket or a block ($kind)" for (kind, sys) in SYSTEMS
    drives = [(:varphi_1, nport()) => drive_of(sys)]
    Hi, fr = drive_hamiltonian(sys, drives; frame=:interaction, n=36)              # default batch = false
    Hl, _ = drive_hamiltonian(sys, drives; n=36)
    B0 = qt_ket(sys, LABELS)
    @test isapprox(to_lab(fr, evolve(Hi, B0), TLIST[end]).data, evolve(Hl, B0).data; atol=1e-7)
    ψ0 = qt_ket(sys, (2, 1))
    Hib, frb = drive_hamiltonian(sys, drives; frame=:interaction, n=36, batch=true)
    @test isapprox(to_lab(frb, evolve(Hib, ψ0), TLIST[end]).data, evolve(Hl, ψ0).data; atol=1e-7)
end

const P0 = QuantumToolbox.NullParameters()
const F0 = (p, t) -> 0.05 * cos(t)                                         # 0.05 GHz at t = 0
# H(t) x of a matrix-free QobjEvo, in place, as sesolve applies it
apply_at(H, x, t) = (w = similar(x); H.data(w, x, x, P0, t); w)
phase(Ω, t, M) = cis.(Ω .* t) .* M .* transpose(cis.(-Ω .* t))           # e^{iΩt} M e^{−iΩt}

@testset "drive terms are 2π f(t) O, on n < N kept states ($kind)" for (kind, sys) in SYSTEMS
    E = energies(sys); n = 10; t = 0.7
    key = (:varphi_1, nport())
    P = port(sys, key...)[1:n, 1:n]
    Hd, fr = drive_hamiltonian(sys, [key => F0]; n)
    @test fr.ω === nothing && size(Hd) == (n, n)
    @test Hd(P0, t).data ≈ 2π * (Diagonal(E[1:n] .- E[1]) + F0(P0, t) * P)
    Hi, fri = drive_hamiltonian(sys, [key => F0]; n, frame=:interaction)
    @test fri.ω ≈ 2π * (E[1:n] .- E[1])
    @test Hi(P0, t).data ≈ phase(fri.ω, t, 2π * F0(P0, t) * P)
    @test size(qt_ket(sys, LABELS; n)) == (n, 3)

    # matrix-free, lab and interaction picture: the same terms, applied
    H0 = dense_operator(operator(sys)) - E[1] * I
    Pf = dense_operator(port_operator(sys, key))
    x = randn(ComplexF64, 36)
    Hf, _ = drive_hamiltonian(sys, [key => F0]; basis=:free)
    @test apply_at(Hf, x, t) ≈ 2π * (H0 + F0(P0, t) * Pf) * x
    Hfi, frf = drive_hamiltonian(sys, [key => F0]; basis=:free, frame=:interaction)
    @test frf.ω ≈ 2π * real.(diag(H0))
    @test apply_at(Hfi, x, t) ≈ phase(frf.ω, t, 2π * (H0 + F0(P0, t) * Pf) - Diagonal(frf.ω)) * x
end

@testset "drives on a child node ($kind)" for (kind, sys) in SYSTEMS
    node = sys.root.node.children[1]                                       # leaf varphi_1: 6 kept of its 31 or 32
    key = (:varphi_1, nport())
    El = energies(sys, node); t = 0.7
    Hd, _ = drive_hamiltonian(sys, [key => F0], node)
    @test size(Hd) == (6, 6)
    @test Hd(P0, t).data ≈ 2π * (Diagonal(El .- El[1]) + F0(P0, t) * port(sys, key..., node))
    H0 = dense_operator(operator(sys, node)) - El[1] * I
    N = size(H0, 1)
    x = randn(ComplexF64, N)
    Hf, _ = drive_hamiltonian(sys, [key => F0], node; basis=:free)
    @test apply_at(Hf, x, t) ≈ 2π * (H0 + F0(P0, t) * dense_operator(port_operator(sys, key, node))) * x
    Hfi, fr = drive_hamiltonian(sys, [key => drive_of(sys)], node; basis=:free, frame=:interaction)
    Hfl, _ = drive_hamiltonian(sys, [key => drive_of(sys)], node; basis=:free)
    ψ0 = qt_ket(sys, (2,), node; basis=:free)
    @test size(ψ0) == (N,)
    short(H) = sesolve(H, ψ0, [0.0, 2.0]; reltol=1e-10, abstol=1e-12, progress_bar=Val(false)).states[end]  # stiff leaf
    @test isapprox(to_lab(fr, short(Hfi), 2.0).data, short(Hfl).data; atol=1e-7)
end

@testset "complex-character port in the interaction picture ($kind)" for (kind, sys) in SYSTEMS
    drives = [(:varphi_1, _extra(first(sys.root.node.children[1].children).second)) => drive_of(sys)]
    for kw in ((; n=36), (; basis=:free))
        ψ0 = haskey(kw, :n) ? qt_ket(sys, (2, 1)) : qt_ket(sys, (2, 1); basis=:free)
        Hi, fr = drive_hamiltonian(sys, drives; frame=:interaction, kw...)
        Hl, _ = drive_hamiltonian(sys, drives; kw...)
        @test isapprox(to_lab(fr, evolve(Hi, ψ0), TLIST[end]).data, evolve(Hl, ψ0).data; atol=1e-7)
    end
end

@testset "joint-port drives ($kind)" for (kind, sys) in SYSTEMS
    # n1 n2 at the root: no node stores it (joint ports are formed for ancestors), so the dressed basis projects the
    # product-basis operator onto the kept states
    J = [(:varphi_1, nport()), (:varphi_2, nport())]
    E = energies(sys); V = eigenvectors(sys); t = 0.7
    PJ = V' * dense_operator(port_operator(sys, J)) * V
    Hd, _ = drive_hamiltonian(sys, [J => F0]; n=36)
    @test Hd(P0, t).data ≈ 2π * (Diagonal(E .- E[1]) + F0(P0, t) * PJ)
    @test qt_operator(sys, J; n=10).data ≈ PJ[1:10, 1:10]
    ψf0 = qt_ket(sys, (2, 1); basis=:free)
    ψd = evolve(first(drive_hamiltonian(sys, [J => drive_of(sys)]; n=36)), to_dressed(sys, ψf0))
    ψf = evolve(first(drive_hamiltonian(sys, [J => drive_of(sys)]; basis=:free)), ψf0)
    @test isapprox(to_dressed(sys, ψf).data, ψd.data; atol=1e-7)
end

@testset "matrix-free callables on a real vector ($kind)" for (kind, sys) in SYSTEMS
    E = energies(sys)
    xr = randn(36)
    Href = 2π * (dense_operator(operator(sys)) - E[1] * I)
    F = qt_hamiltonian(sys; basis=:free).data
    @test F.op(xr, xr, P0, 0.0) ≈ Href * xr                                # out of place: a complex result
    Fi = first(drive_hamiltonian(sys, Pair[]; basis=:free, frame=:interaction)).data
    @test Fi.op(xr, xr, P0, 0.0) ≈ (Href - Diagonal(2π * real.(diag(Href / 2π)))) * xr
    @test qt_operator(sys, (:varphi_1, nport()); basis=:free).data * xr ≈ dense_operator(port_operator(sys, (:varphi_1, nport()))) * xr
end
