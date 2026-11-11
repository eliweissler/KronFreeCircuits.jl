using Test, LinearAlgebra, KronFreeCircuits

# The KITE at the paper's optimum frame with the flux offset `delta`, the offset charge `n_g1` and `E_J` symbolic
# (test/python/generate_kite_symbolic.jl).  The tree: three leaves on the reference calculation's bases, twelve states
# each, and a root of dimension 1728.  Twelve states per leaf leave the levels 13 to 33 MHz above the paper's, which does
# not matter here: the derivatives are those of this calculation.
const KITE_SYM = load_spec(fixture("kite/optimum_symbolic.json"))
function kite_deriv_tree(solver; retain=:auto)
    bases = [:varphi_1 => ChargeBasis(16), :varphi_2 => PhaseGrid(-18.915907192616967, 18.915907192616967, 120; order=8),
             :varphi_3 => PhaseGrid(-17.714011073078225, 17.714011073078225, 120; order=8)]
    return Node([Node([m => b]; solver=Dense(), keep=lowest(12)) for (m, b) in bases]; solver, compute=6, keep=lowest(4), retain)
end
ksolve(root, θ) = quiet(() -> materialize(root, KITE_SYM, θ))

# The four lowest levels at (E_J = 16.83, delta = 0.0, n_g1 = 0.1) on this tree, as printed by
#     JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=test/python test/python/generate_kite_symbolic.jl
# when the fixture was generated; regenerate both together.
const KITE_SYM_LEVELS = [-9.510804644179908, -9.497205434969919, -8.425747834344323, -8.308985980043891]

@testset "the symbolic KITE fixture" begin
    @test KITE_SYM.parameters == [:delta, :n_g1, :E_J]
    E = energies(ksolve(kite_deriv_tree(Dense()), (E_J=16.83, delta=0.0, n_g1=0.1)))
    @test length(KITE_SYM_LEVELS) == 4
    @test E ≈ KITE_SYM_LEVELS atol=1e-7                                          # the fixture is the one that was generated
    @test maximum(abs, E - [-9.523995, -9.510525, -8.453790, -8.337862]) < 0.04   # evidence: the paper's levels, to this tree's truncation
end

@testset "the KITE doublet: block, point derivatives, effective Hamiltonian (design acceptance 7)" begin
    θ = (E_J=16.83, delta=0.0, n_g1=0.1)            # n_g1 = 0 would make the compact leaf's kept charge states degenerate pairs
    root = kite_deriv_tree(LOBPCG(tol=1e-11)); sys = ksolve(root, θ); E0 = energies(sys)
    Δ = E0[2] - E0[1]
    @test 0.010 < Δ < 0.020                         # the 13.5 MHz doublet, on this tree
    d = derivatives(sys; wrt=(:delta, :E_J), order=2, mixed=((:E_J, :delta),), scale=(delta=0.01,))
    b = only(b for b in blocks(d) if b.node === root && b.parameters == (:delta,) && 1 in b.states)
    @test b.states == [1, 2]
    @test !b.exact
    @test b.detuning ≈ Δ atol=1e-9
    @test b.coupling > 1.0
    # point derivatives against finite differences at a step small against Δ/c ≈ 8e-3; with so small a step the finite
    # difference itself is good to about 1e-6 at first order and 1e-5 at second (measured), hence rtol = 1e-5 and 1e-4
    cache = Dict{Any,Any}(); f(t) = get!(() -> energies(ksolve(root, t)), cache, t)
    check_fd(energies(d, :delta), fd_derivative(f, θ, :delta; h=2e-4); rtol=1e-5)
    E2 = energies(d, :delta, :delta)
    check_fd(E2, fd_second(f, θ, :delta; h=2e-4); rtol=1e-4)
    @test E2[2] - E2[1] ≈ 4 * b.coupling^2 / b.detuning rtol=1e-2          # the curvature 4c²/Δ of the transition frequency
    # the effective Hamiltonian at displacements up to δx against a full re-solve; the Taylor expansion fails there
    for δ in (0.01, 0.005)
        @test block_levels(b, (delta=δ,)).energies ≈ f(shifted(θ, :delta, δ))[1:2] atol=1e-4
    end
    taylor = E0 .+ 0.01 .* energies(d, :delta) .+ 0.5 * 0.01^2 .* E2
    @test maximum(abs, (taylor - f(shifted(θ, :delta, 0.01)))[1:2]) > 1e-3
    # the full-spectrum route gives the same derivatives
    dd = derivatives(ksolve(kite_deriv_tree(Dense(); retain=:always), θ); wrt=:delta, order=2)
    @test energies(dd, :delta) ≈ energies(d, :delta) rtol=1e-6
    @test energies(dd, :delta, :delta) ≈ E2 rtol=1e-5
    @info "KITE linear solves: largest iteration count" maximum(s.iterations for s in d.linear_solves)
end

@testset "a mixed derivative for dephasing off the sweet spot (design acceptance 8)" begin
    θ = (E_J=16.83, delta=0.05, n_g1=0.1)
    root = kite_deriv_tree(Dense(); retain=:always); sys = ksolve(root, θ)
    d = derivatives(sys; wrt=(:E_J, :delta), order=2, mixed=((:E_J, :delta),))
    f01(E) = E[2] - E[1]
    m = f01(energies(d, :E_J, :delta))
    @test m ≈ 0.04277521 rtol=1e-5                                       # measured by finite differences when the plan was written
    check_fd(m, fd_mixed(t -> f01(energies(ksolve(root, t))), θ, :E_J, :delta; hx=1e-2, hλ=1e-3); rtol=1e-5)
end
