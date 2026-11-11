using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: _derivative_operator, operator, route

function check_node2(c, x, λ; hx, hλ=hx, rtol=1e-5, atol=1e-8)
    at = dc_memo(c); sys = at(c.θ); r = node_second_order(sys, x, λ); d = r.d
    fd(f) = fd_pair(f, c.θ, x, λ; hx, hλ)
    @test d.children == Any[nothing]
    check_fd(d.energies, fd(θ -> energies(at(θ))); rtol, atol)                              # absolute: Review Focus 2
    check_fd(d.vectors, fd(θ -> eigenvectors(at(θ))); rtol, atol)
    check_fd(d.peak_amplitudes, fd(θ -> at(θ).root.peak_amplitudes); rtol, atol)
    for key in keys(sys.root.ports)                                                         # Review Focus 3
        check_fd(d.ports[key], fd(θ -> at(θ).root.ports[key]); rtol, atol)
    end
    V = eigenvectors(sys)                                                                   # the normalization condition
    @test [real(dot(V[:, j], d.vectors[:, j])) for j in axes(V, 2)] ≈
          [-real(dot(r.dX.vectors[:, j], r.dΛ.vectors[:, j])) for j in axes(V, 2)] atol=1e-10
    # eq:eig_val_secondgrad of arXiv:2408.12704v3: ∂²f_i = ⟨f_i|∂²H|f_i⟩ + 2 Re⟨∂_x f_i|∂_λH|f_i⟩
    Hλ, Hxλ = Matrix(r.Hλ), Matrix(r.Hxλ)
    @test d.energies ≈ [real(V[:, j]' * Hxλ * V[:, j]) + 2 * real(r.dX.vectors[:, j]' * Hλ * V[:, j]) for j in axes(V, 2)] rtol=1e-9 atol=1e-10
    return d
end

@testset "second order at a dense leaf against finite differences" begin
    check_node2(dc_transmon(), :E_J, :E_J; hx=2e-2)
    check_node2(dc_transmon(), :n_g, :n_g; hx=5e-3)
    check_node2(dc_transmon(), :E_J, :n_g; hx=2e-2, hλ=5e-3)
    d = check_node2(dc_transmon(periodic=true), :n_g, :n_g; hx=5e-3, rtol=1e-6)
    @test maximum(abs, d.gauge) > 1e-3
    check_node2(dc_transmon(periodic=true), :E_J, :E_J; hx=2e-2)
    check_node2(dc_transmon(periodic=true), :E_J, :n_g; hx=2e-2, hλ=5e-3)
    check_node2(dc_fluxonium(), :varphi_ext, :varphi_ext; hx=1e-2, rtol=1e-6, atol=1e-6)
    check_node2(dc_fluxonium(), :E_J, :E_J; hx=2e-2)
    check_node2(dc_fluxonium(), :E_J, :varphi_ext; hx=1e-2, atol=1e-6)
    check_node2(dc_transmon(compute=8, keep=indices([1, 3, 4])), :E_J, :n_g; hx=2e-2, hλ=5e-3)   # Review Focus 4
end

@testset "analytic second derivatives" begin
    # oscillator, E_k = (k + 1/2) sqrt(8 E_C E_L): ∂²E_k/∂E_L² = −(k + 1/2) sqrt(2 E_C) / (2 E_L^(3/2)), ∂²E_k/∂E_C∂E_L = (k + 1/2) / sqrt(2 E_C E_L)
    c = dc_oscillator(); k = 0:5; (; E_C, E_L) = c.θ
    @test node_second_order(dc_solve(c), :E_L, :E_L).d.energies ≈ -(k .+ 0.5) .* sqrt(2E_C) ./ (2 * E_L^1.5) rtol=1e-9
    @test node_second_order(dc_solve(c), :E_C, :E_L).d.energies ≈ (k .+ 0.5) ./ sqrt(2 * E_C * E_L) rtol=1e-9
    # transmon at n_g = 0: the cosine law of the charge dispersion gives ∂²E_0/∂n_g² = 2π² ε_0
    tm(ng) = dc_transmon(θ=(E_J=30.0, E_C=1.2, n_g=ng))
    ε = energies(dc_solve(tm(0.5))) - energies(dc_solve(tm(0.0)))
    @test node_second_order(dc_solve(tm(0.0)), :n_g, :n_g).d.energies[1] ≈ 2π^2 * ε[1] rtol=1e-4
    # fluxonium with E_J = 0: the flux displaces an oscillator, so every level's curvature vanishes (E_L from the constant
    # cancels the sum over states)
    @test maximum(abs, node_second_order(dc_solve(dc_fluxonium(varphi_ext=1.0, E_J=0.0)), :varphi_ext, :varphi_ext).d.energies) < 1e-6
    # exact degeneracy at second order: on the charges −1, 0, 1, −cos φ couples |−1⟩ and |+1⟩ through |0⟩
    c1 = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(0), E_2=0.0)
    d = quiet(() -> node_second_order(dc_solve(c1), :E_1, :E_1).d)
    @test isapprox(d.energies[1], -0.25; atol=1e-12) && all(isnan, d.energies[2:3]) && all(isnan, d.vectors[:, 2:3])
end

@testset "∂²H/∂x∂λ at a branch equals the second difference of the forward operator" begin
    for (c, pairs) in ((dc_two_transmons(), ((:g, :g), (:g, :E_J1), (:E_J1, :E_J2), (:E_J1, :E_J1))),
                       (dc_abc(), ((:varphi_ext, :varphi_ext), (:E_J, :varphi_ext), (:E_J3, :E_J))))
        at = dc_memo(c); sys = at(c.θ); terms = route(c.root, sys.terms)[c.root]
        for (x, λ) in pairs
            first_of(p) = Any[fd_child(at, c, ch, p; h=1e-3) for ch in c.root.children]
            second = Any[fd_child(at, c, ch, x, λ; hx=2e-2) for ch in c.root.children]
            H2 = _derivative_operator(sys.root, terms, c.θ, x, λ, first_of(x), first_of(λ), second; constant=sys.constant)
            check_fd(Matrix(H2), fd_pair(θ -> Matrix(operator(at(θ))), c.θ, x, λ; hx=2e-2); rtol=1e-5, atol=1e-6)
        end
    end
end
