using KronFreeCircuits: _port_matrix, DenseOp

@testset "_port_matrix over several axes equals the Kronecker sandwich" begin
    rng = MersenneTwister(11); dims = [3, 4, 2]
    V = randn(rng, ComplexF64, prod(dims), 5)
    A1 = randn(rng, ComplexF64, 3, 3); A3 = randn(rng, ComplexF64, 2, 2)
    ops = [(1, DenseOp(A1)), (3, DenseOp(A3))]
    P = _port_matrix(V, dims, ops)
    @test P ≈ V' * kron(A3, Matrix(I, 4, 4), A1) * V atol=1e-12 * norm(V)^2 * norm(A1) * norm(A3)
    @test _port_matrix(V, dims, 1, DenseOp(A1)) == _port_matrix(V, dims, [(1, DenseOp(A1))])
    Vr = randn(rng, prod(dims), 5)
    @test _port_matrix(Vr, dims, [(1, DenseOp(real(A1))), (3, DenseOp(real(A3)))]) isa Matrix{Float64}
    @test_throws ArgumentError _port_matrix(V, dims, [(1, DenseOp(A1)), (1, DenseOp(A1))])
end
