using KronFreeCircuits: branch_peak_phase

@testset "branch_peak_phase" begin
    # two children with 3 and 2 kept states; synthetic peak-amplitude matrices and a coefficient tensor
    A1 = ComplexF64[0.9 0.1 0.2; 0.3 0.8 0.1; 0.2 0.2 0.7]; A2 = ComplexF64[0.95 0.2; 0.1 0.9]
    C = zeros(ComplexF64, 3, 2); C[2, 1] = 0.8im; C[1, 2] = 0.5; C[3, 2] = 0.2
    comp, α = branch_peak_phase(C, [A1, A2])
    @test comp == LinearIndices((3, 2))[2, 1]                       # dominant component
    # exact amplitude at the joint peak (row 2 of A1, row 1 of A2): Σ_ab C[a,b] A1[2,a] A2[1,b]
    @test α ≈ sum(C[a, b] * A1[2, a] * A2[1, b] for a in 1:3, b in 1:2)
    # child sign flip invariance: flipping child-1 state 2 flips C[2,:] and A1[:,2] together; α unchanged
    C2 = copy(C); C2[2, :] .*= -1; A1f = copy(A1); A1f[:, 2] .*= -1
    @test branch_peak_phase(C2, [A1f, A2])[2] ≈ α
    # a mask restricts the dominant component (axis 1 fastest); with no weight on it, the global argmax
    m = trues(3, 2); m[2, 1] = false
    @test branch_peak_phase(C, [A1, A2]; mask=vec(m))[1] == LinearIndices((3, 2))[1, 2]
    @test branch_peak_phase(C, [A1, A2]; mask=falses(6))[1] == comp
    # three children recursion shape
    @test branch_peak_phase(randn(ComplexF64, 2, 3, 2), [randn(ComplexF64, 2, 2), randn(ComplexF64, 3, 3), randn(ComplexF64, 2, 2)]) isa Tuple{Int,ComplexF64}
end
