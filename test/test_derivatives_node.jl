using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: _derivative_operator, _peak_map, _peak_amps, route, materialized

# energies, vectors, both ports and the peak amplitudes against finite differences
function check_node(c, x; h=1e-3, rtol=1e-6)
    at = dc_memo(c); sys = at(c.θ); d = node_first_order(sys, x)
    @test eltype(d.vectors) === eltype(eigenvectors(sys)) && eltype(d.peak_amplitudes) === eltype(d.vectors)
    @test d.children == Any[nothing]
    check_fd(d.energies, fd_derivative(θ -> energies(at(θ)), c.θ, x; h); rtol)                   # absolute energies: Review Focus 2
    check_fd(d.vectors, fd_derivative(θ -> eigenvectors(at(θ)), c.θ, x; h); rtol)
    check_fd(d.peak_amplitudes, fd_derivative(θ -> at(θ).root.peak_amplitudes, c.θ, x; h); rtol)
    for (key, P) in sys.root.ports                                                               # off-diagonals of expiphi: Review Focus 3
        @test eltype(d.ports[key]) === eltype(P)
        check_fd(d.ports[key], fd_derivative(θ -> at(θ).root.ports[key], c.θ, x; h); rtol)
    end
    return d
end

@testset "first order at a dense leaf against finite differences" begin
    for x in (:n_g, :E_J, :E_C); check_node(dc_transmon(), x); end                               # real
    for x in (:n_g, :E_J); d = check_node(dc_transmon(periodic=true), x)                         # complex
        x === :n_g && @test maximum(abs, d.gauge) > 0.1                                          # the gauge term is really there
    end
    for x in (:varphi_ext, :E_J, :E_L); check_node(dc_fluxonium(), x); end
    @test all(iszero, check_node(dc_transmon(), :E_J).gauge)
    # Review Focus 4: the kept states are not the lowest computed ones
    for x in (:n_g, :E_J); check_node(dc_transmon(compute=8, keep=indices([1, 3, 4])), x); end
    for x in (:n_g, :E_J); check_node(dc_transmon(periodic=true, compute=8, keep=indices([1, 3, 4])), x); end
    # a dense node that did not retain is re-diagonalized, with a notice, to the same result
    a = node_first_order(dc_solve(dc_transmon(compute=8)), :n_g)
    sn = dc_solve(dc_transmon(compute=8, retain=:never))
    b = @test_logs (:info, r"re-diagonalized") match_mode=:any node_first_order(sn, :n_g)
    @test b.energies ≈ a.energies rtol=1e-10
    @test b.vectors ≈ a.vectors rtol=1e-9
end

@testset "the peak map reproduces the stored peak amplitudes" begin
    for c in (dc_transmon(periodic=true), dc_chain3(), dc_abc(), dc_two_transmons(periodic=true))
        sys = dc_solve(c)
        for node in KronFreeCircuits._all_nodes(c.root)
            mn = materialized(sys, node)
            @test _peak_map(mn, mn.vectors, _peak_amps(mn)) ≈ mn.peak_amplitudes atol=1e-12
        end
    end
end

@testset "first order reduces to eq:eig_val_grad and eq:eig_vec_grad of arXiv:2408.12704v3" begin
    # ∂f_i/∂x = ⟨f_i|∂H/∂x|f_i⟩ and ∂|f_i⟩/∂x = Σ_{m≠i} ⟨f_m|∂H/∂x|f_i⟩ / (f_i − f_m) |f_m⟩, the sum over all states of a dense
    # node.  On a complex node the rule adds i a_i |f_i⟩, the phase-convention term the paper's formula sets to zero.
    spec = load_spec(fixture("transmon.json")); θ = (E_J=20.0, E_C=1.0, n_g=0.2)
    for (basis, x, iscomplex) in ((ChargeBasis(8), :E_J, false), (PeriodicGrid(16; order=8), :n_g, true))
        leaf = Node([:varphi_1 => basis]; solver=Dense())                     # every state kept
        sys = quiet(() -> materialize(leaf, spec, θ)); d = node_first_order(sys, x)
        E, V = energies(sys), eigenvectors(sys)
        H1 = Matrix(_derivative_operator(sys.root, route(leaf, sys.terms)[leaf], θ, x, Any[nothing]; constant=sys.constant))
        @test d.energies ≈ [real(V[:, i]' * H1 * V[:, i]) for i in eachindex(E)] rtol=1e-12
        paper = reduce(hcat, [sum(V[:, m] * (V[:, m]' * H1 * V[:, i]) / (E[i] - E[m]) for m in eachindex(E) if m != i) for i in eachindex(E)])
        along = [V[:, i]' * d.vectors[:, i] for i in eachindex(E)]
        @test d.vectors - V * Diagonal(along) ≈ paper rtol=1e-9
        @test maximum(abs, real.(along)) < 1e-12
        @test imag.(along) ≈ d.gauge atol=1e-12
        @test iscomplex ? maximum(abs, d.gauge) > 0.1 : all(iszero, d.gauge)
    end
end

@testset "analytic checks at single nodes" begin
    # oscillator: E_k = (k + 1/2) sqrt(8 E_C E_L), so ∂E_k/∂E_L = (k + 1/2) sqrt(2 E_C / E_L) and ∂E_k/∂E_C = (k + 1/2) sqrt(2 E_L / E_C)
    c = dc_oscillator(); k = 0:5
    @test node_first_order(dc_solve(c), :E_L).energies ≈ (k .+ 0.5) .* sqrt(2 * c.θ.E_C / c.θ.E_L) rtol=1e-10
    @test node_first_order(dc_solve(c), :E_C).energies ≈ (k .+ 0.5) .* sqrt(2 * c.θ.E_L / c.θ.E_C) rtol=1e-10
    # transmon charge dispersion: E_m(n_g) = Ē_m − (ε_m/2) cos(2π n_g) to tight-binding accuracy, with ε_m = E_m(1/2) − E_m(0),
    # so ∂E_m/∂n_g at n_g = 1/4 is π ε_m; at n_g = 0 it vanishes by symmetry
    tm(ng) = dc_transmon(θ=(E_J=30.0, E_C=1.2, n_g=ng))
    ε = energies(dc_solve(tm(0.5))) - energies(dc_solve(tm(0.0)))
    dq = node_first_order(dc_solve(tm(0.25)), :n_g).energies
    @test dq[1] ≈ π * ε[1] rtol=1e-6
    @test dq[2] ≈ π * ε[2] rtol=1e-5
    @test maximum(abs, node_first_order(dc_solve(tm(0.0)), :n_g).energies[1:2]) < 1e-9
    # fluxonium: at varphi_ext = π every level is stationary by symmetry; with E_J = 0 the flux only displaces an oscillator
    @test maximum(abs, node_first_order(dc_solve(dc_fluxonium(varphi_ext=π)), :varphi_ext).energies) < 1e-7
    @test maximum(abs, node_first_order(dc_solve(dc_fluxonium(varphi_ext=1.0, E_J=0.0)), :varphi_ext).energies) < 1e-7
end

@testset "exact degeneracy: undefined with a coupling, skipped without one" begin
    # 4 E_C n² on the charges −1, 0, 1 with every cosine off: energies 0, 4, 4, the pair |−1⟩, |+1⟩ exactly degenerate
    c = dc_cos2phi(n_g=0.0, basis=ChargeBasis(1), keep=lowest(0), E_2=0.0)
    sys = dc_solve(c)
    @test energies(sys) ≈ [0.0, 4.0, 4.0] atol=1e-14
    d2 = @test_logs (:warn, r"degenerate") match_mode=:any node_first_order(sys, :E_2)    # −cos 2φ couples the pair directly
    @test isnan.(d2.energies) == [false, true, true] && all(isnan, d2.vectors[:, 2:3]) && all(isfinite, d2.vectors[:, 1])
    dn = node_first_order(sys, :n_g)                                                      # −8 E_C n is diagonal: no coupling
    @test all(isfinite, dn.vectors)
    @test sort(dn.energies) ≈ [-8.0, 0.0, 8.0] atol=1e-12
    d1 = node_first_order(sys, :E_1)                                                      # −cos φ couples the pair only through |0⟩
    @test all(isfinite, d1.vectors) && maximum(abs, d1.energies) < 1e-12
end
