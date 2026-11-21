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
