using Test, LinearAlgebra, KronFreeCircuits

@testset "union basis" begin
    e(i, n=8) = (v = zeros(ComplexF64, n); v[i] = 1; v)
    src(label, cols; kw...) = BasisSource(label, reduce(hcat, cols), collect(1.0:length(cols)); kw...)
    a = src("well 1", [e(1), e(2), e(3)]); b = src("well 2", [e(4), e(1), e(5)])
    U, r = union_basis([a, b]; m=4)
    @test U ≈ hcat(e(1), e(4), e(2), e(3)) && r.taken == [3, 1] && r.dropped == [0, 1]     # level by level; e(1) twice
    @test U' * U ≈ I
    # a cap leaves the remainder to the uncapped source
    U2, r2 = union_basis([src("well 1", [e(1), e(2), e(3)]; cap=1), src("bare", [e(6), e(7), e(8)])]; m=4)
    @test r2.taken == [1, 3]
    # a window in energy
    @test union_basis([src("w", [e(1), e(2), e(3)]; window=1.5), src("bare", [e(6), e(7)])]; m=4)[2].taken == [2, 2]
    # coinciding sources: reported, and an error when m cannot be reached (Review Focus 4)
    @test_throws "dropped" union_basis([a, a]; m=4)
    U3, r3 = union_basis([a, a]; m=4, allow_fewer=true)
    @test size(U3, 2) == 3 && r3.dropped == [0, 3]
    # nearly dependent: a remainder below the threshold is dropped, above it is kept
    near(δ) = normalize(e(1) + δ * e(2))
    @test union_basis([src("x", [e(1)]), src("y", [near(1e-8)])]; m=2, allow_fewer=true)[2].dropped == [0, 1]
    @test size(union_basis([src("x", [e(1)]), src("y", [near(1e-3)])]; m=2)[1], 2) == 2

    # the error message carries the report and the escape hatch
    err = try union_basis([a, a]; m=4) catch ex; ex end
    @test err isa ArgumentError && occursin("allow_fewer = true", err.msg) && occursin("well 1", err.msg) && occursin("taken 0, dropped 3", err.msg)

    # stopping at m in the middle of a level leaves later columns neither taken nor dropped
    _, r4 = union_basis([a, b]; m=2)
    @test r4.taken == [1, 1] && r4.dropped == [0, 0]
    # a source that offers nothing because of cap = 0
    U5, r5 = union_basis([src("none", [e(1), e(2)]; cap=0), src("bare", [e(3), e(4)])]; m=2)
    @test r5.taken == [0, 2] && r5.dropped == [0, 0] && U5 ≈ hcat(e(3), e(4))
    # real input is converted, and the caller's arrays are not mutated
    V = Matrix{Float64}(I, 4, 2); V0 = copy(V); en = [0.0, 1.0]
    s = BasisSource("real", V, en)
    union_basis([s, s |> x -> BasisSource("other", V, en)]; m=2)
    @test V == V0 && s.vectors isa Matrix{ComplexF64}

    # validation
    @test_throws ArgumentError union_basis(BasisSource[]; m=1)
    @test_throws ArgumentError union_basis([a]; m=0)
    @test_throws ArgumentError union_basis([a, src("small", [e(1, 4)])]; m=2)
    @test_throws ArgumentError BasisSource("x", reduce(hcat, [e(1), e(2)]), [1.0])
    @test_throws ArgumentError src("x", [e(1)]; cap=-1)
    @test_throws ArgumentError src("x", [e(1)]; window=-1.0)
    @test_throws ArgumentError union_basis([a]; m=1, drop=-1.0)
    @test_throws ArgumentError union_basis([src("big", [2 * e(1)])]; m=1)
end
