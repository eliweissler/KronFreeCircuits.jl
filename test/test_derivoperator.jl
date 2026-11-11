using Test, LinearAlgebra, Random, KronFreeCircuits
using KronFreeCircuits: _derivative_operator, _sandwich, _axis_ops, operator, route, materialized

@testset "_sandwich reproduces every stored port and joint port, and is two-sided" begin
    rng = MersenneTwister(3)
    for c in (dc_two_transmons(periodic=true), dc_zero_pi(), dc_abc(), dc_chain3())
        sys = dc_solve(c)
        for node in KronFreeCircuits._all_nodes(c.root)
            mn = materialized(sys, node); V = mn.vectors; k = size(V, 2)
            Q = Matrix(qr(randn(rng, eltype(V), k, k)).Q)
            for (K, P) in vcat([[key] => P for (key, P) in mn.ports], collect(mn.joint_ports))
                ops = _axis_ops(node, mn.children, K)
                S = _sandwich(mn, V, V, K, ops)
                @test typeof(S) === typeof(P)
                @test S ≈ P atol=1e-12
                @test _sandwich(mn, V * Q, V, K, ops) ≈ Q' * P atol=1e-11          # a different left argument
                @test _sandwich(mn, V, V * Q, K, ops) ≈ P * Q atol=1e-11           # a different right argument
            end
        end
    end
end

@testset "_sandwich on a joint key of a complex node, against a dense product" begin
    # `dc_abc` with the two-mode child on periodic grids: the child is complex (its `n` is imaginary), with joint ports.
    ab = Node([:a => PeriodicGrid(24; order=8), :b => PeriodicGrid(24; order=8)]; solver=Dense(), keep=lowest(8))
    root = Node([ab, Node([:c => ChargeBasis(5)]; solver=Dense(), keep=lowest(5))]; solver=Dense(), keep=lowest(6))
    sys = quiet(() -> materialize(root, KronFreeCircuits.parse_spec(abc_deriv_json), dc_abc().θ))
    mn = materialized(sys, ab); V = mn.vectors
    @test mn isa KronFreeCircuits.MaterializedNode{ComplexF64} && !isempty(mn.joint_ports)
    W = randn(MersenneTwister(5), ComplexF64, size(V))
    for (K, P) in mn.joint_ports
        ops = _axis_ops(ab, mn.children, K)
        S = _sandwich(mn, V, V, K, ops)
        @test typeof(S) === typeof(P)
        @test S ≈ P atol=1e-12
        mats = Matrix{ComplexF64}[Matrix{ComplexF64}(I, d, d) for d in mn.dims]
        for (a, op, _) in ops
            mats[a] = Matrix(KronFreeCircuits.dense(op))
        end
        A = foldl((acc, M) -> kron(M, acc), mats; init=ones(ComplexF64, 1, 1))     # axis 1 fastest
        @test _sandwich(mn, W, V, K, ops) ≈ W' * A * V atol=1e-10
        @test _sandwich(mn, V, W, K, ops) ≈ V' * A * W atol=1e-10
    end
end

@testset "∂H/∂x at a leaf equals the finite difference of the forward operator" begin
    for (c, xs) in ((dc_transmon(), (:n_g, :E_J, :E_C)), (dc_transmon(periodic=true), (:n_g, :E_J)),
                    (dc_fluxonium(), (:varphi_ext, :E_L, :E_J)))
        at = dc_memo(c); sys = at(c.θ)
        for x in xs
            H1 = _derivative_operator(sys.root, route(c.root, sys.terms)[c.root], c.θ, x, Any[nothing]; constant=sys.constant)
            @test eltype(H1) === eltype(operator(sys))
            check_fd(Matrix(H1), fd_derivative(θ -> Matrix(operator(at(θ))), c.θ, x; h=1e-3); rtol=1e-8, atol=1e-8)
        end
    end
    # Review Focus 1: at varphi_ext = 0 the flux term's coefficient E_L*varphi_ext vanishes and the forward operator has
    # no such term; its derivative is E_L φ.
    c = dc_fluxonium(varphi_ext=0.0); sys = dc_solve(c)
    H1 = Matrix(_derivative_operator(sys.root, route(c.root, sys.terms)[c.root], c.θ, :varphi_ext, Any[nothing]; constant=sys.constant))
    @test H1 ≈ c.θ.E_L * Diagonal(KronFreeCircuits.coordinates(PhaseGrid(-10π, 10π, 601; order=8))) atol=1e-12
    @test norm(H1) > 1
end

@testset "∂H/∂x at a branch: coefficients, the children's ports and their energies" begin
    for (c, xs) in ((dc_two_transmons(), (:g, :n_g1, :E_J1, :E_J2)), (dc_two_transmons(periodic=true), (:g, :n_g1)),
                    (dc_zero_pi(), (:varphi_ext, :n_g, :E_J)), (dc_abc(), (:varphi_ext, :E_J3, :n_ga, :E_J)))
        at = dc_memo(c); sys = at(c.θ)
        terms = route(c.root, sys.terms)[c.root]
        for x in xs
            dx = Any[fd_child(at, c, ch, x; h=1e-3) for ch in c.root.children]
            H1 = _derivative_operator(sys.root, terms, c.θ, x, dx; constant=sys.constant)
            @test eltype(H1) === eltype(operator(sys))
            check_fd(Matrix(H1), fd_derivative(θ -> Matrix(operator(at(θ))), c.θ, x; h=1e-3); rtol=1e-6, atol=1e-7)
        end
    end
    # each contribution matters on this circuit
    c = dc_two_transmons(); at = dc_memo(c); sys = at(c.θ); terms = route(c.root, sys.terms)[c.root]
    op(x, parts) = Matrix(_derivative_operator(sys.root, terms, c.θ, x, Any[fd_child(at, c, ch, x; h=1e-3) for ch in c.root.children];
                                               constant=sys.constant, parts))
    all3 = (:coefficients, :ports, :energies)
    @test norm(op(:E_J2, all3) - op(:E_J2, (:coefficients, :ports))) > 0.1          # the children's energy derivatives
    @test norm(op(:E_J1, all3) - op(:E_J1, (:coefficients, :energies))) > 1e-3      # the children's port derivatives
    @test norm(op(:g, all3) - op(:g, (:ports, :energies))) > 0.1                    # the coefficients' derivatives
end
