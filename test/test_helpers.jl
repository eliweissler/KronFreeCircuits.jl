@testset "kron_reference" begin
    A = ComplexF64[1 2; 3 4]; B = ComplexF64[0 1; 1 0]
    H = kron_reference([2, 2], [(1.0 + 0im, [(1, A)]), (2.0 + 0im, [(2, B)])])
    @test H ≈ kron(I(2), A) + 2 * kron(B, I(2))   # axis 1 fastest
end
