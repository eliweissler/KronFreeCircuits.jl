using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: materialized

# the root's derivatives of two trees that differ only in the root's solver
function same_derivatives(ci, cd, x; kwargs...)
    di, dd = derivatives(dc_solve(ci); wrt=x, kwargs...), derivatives(dc_solve(cd); wrt=x)
    @test energies(di, x) ≈ energies(dd, x) rtol=1e-8 atol=1e-9
    @test norm(eigenvectors(di, x) - eigenvectors(dd, x), Inf) <= 1e-6 * norm(eigenvectors(dd, x), Inf)
    for (mode, p) in keys(materialized(dc_solve(cd), cd.root).ports)
        @test norm(port(di, x, mode, p) - port(dd, x, mode, p), Inf) <= 1e-6 * norm(port(dd, x, mode, p), Inf)
    end
    return di
end
tree(solver; kw...) = dc_two_transmons(; leafkeep=12, solver, compute=8, keep=lowest(6), kw...)     # root dimension 144

@testset "the linear-solve route equals the full-spectrum route" begin
    worst = 0
    for x in (:g, :n_g1, :E_J2)
        di = same_derivatives(tree(LOBPCG(tol=1e-11)), tree(Dense()), x)
        @test !isempty(di.linear_solves)
        @test all(s -> s.iterations < 5000 && s.residual <= 1e-9, di.linear_solves)       # 10 · lintol
        worst = max(worst, maximum(s.iterations for s in di.linear_solves))
    end
    # a complex node (design §5.1): periodic grids make the leaves and the root complex; the ports include expiphi
    for x in (:g, :n_g1)
        di = same_derivatives(tree(LOBPCG(tol=1e-11); periodic=true), tree(Dense(); periodic=true), x)
        worst = max(worst, maximum(s.iterations for s in di.linear_solves))
    end
    same_derivatives(tree(Lanczos(tol=1e-12)), tree(Dense()), :g)
    # a leaf solved by shift-invert
    fl(solver) = dc_fluxonium(; solver, compute=6, keep=lowest(4))
    for x in (:varphi_ext, :E_J)
        di = same_derivatives(fl(ShiftInvert(tol=1e-11)), fl(Dense()), x)
        worst = max(worst, maximum(s.iterations for s in di.linear_solves))
    end
    @info "largest iteration count of the linear solves in this testset" worst
    # and against finite differences of the iterative calculation itself
    c = tree(LOBPCG(tol=1e-11)); at = dc_memo(c); d = derivatives(at(c.θ); wrt=:g)
    check_fd(energies(d, :g), fd_derivative(θ -> energies(at(θ)), c.θ, :g; h=1e-3); rtol=1e-6)
    check_fd(eigenvectors(d, :g), fd_derivative(θ -> eigenvectors(at(θ)), c.θ, :g; h=1e-3); rtol=1e-6)
    # Review Focus 4: kept states that are not the lowest computed ones
    sel(solver) = dc_two_transmons(; leafkeep=12, solver, compute=4, keep=indices([1, 3]))
    same_derivatives(sel(LOBPCG(tol=1e-11)), sel(Dense()), :g)
    # without retained vectors the kept states alone are projected out, which needs them to be the lowest
    same_derivatives(tree(LOBPCG(tol=1e-11); retain=:never), tree(Dense()), :g)
    never = dc_two_transmons(; leafkeep=12, solver=LOBPCG(tol=1e-11), compute=4, keep=indices([1, 3]), retain=:never)
    @test_throws "retain" derivatives(dc_solve(never); wrt=:g)
end

@testset "the sum over the computed states alone is not the derivative; a solve that fails is an error" begin
    sys = dc_solve(tree(LOBPCG(tol=1e-11)))
    full = eigenvectors(derivatives(sys; wrt=:g), :g)
    cut = eigenvectors(derivatives(sys; wrt=:g, _truncate=true), :g)
    @test norm(cut - full, Inf) > 1e-3 * norm(full, Inf)
    @test_throws "did not converge" derivatives(sys; wrt=:g, linmaxiter=2)
end
