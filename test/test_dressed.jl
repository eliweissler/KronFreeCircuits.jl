using KronFreeCircuits: materialized, Node, Dense, lowest, nport, ChargeBasis, load_spec, materialize,
                        bare_labels, bare_overlaps, decompose, dressed_index

@testset "dressed/bare" begin
    spec = load_spec(fixture("two_transmons.json"))
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20), ports=[(:varphi_1, nport())])
    sys0 = materialize(top, spec, two_transmons_params(g=0.0))
    @test length(bare_labels(sys0, top)) == 64 && bare_labels(sys0, top)[2] == (2, 1)
    @test all(maximum(bare_overlaps(sys0, top); dims=2) .≈ 1)           # uncoupled: exact product states
    @test dressed_index(sys0, top, (1, 1)) == 1
    sys = materialize(top, spec, two_transmons_params(g=0.3))
    d = decompose(sys, top, 1)
    @test d[1].first == (1, 1)
    @test sum(abs2, last.(d)) ≈ 1 atol=1e-3
    @test dressed_index(sys, top, d[1].first) == 1
    @test dressed_index(sys0, top, (8, 8)) === nothing            # uncoupled: product state (8,8) ranks 64th, not among the 20 kept
    @test length(bare_labels(sys, top; depth=2)[1]) == 2 && bare_labels(sys, top; depth=2)[1][1] isa Tuple
    @test size(bare_overlaps(sys, top)) == (20, 64)
    @test_throws ArgumentError dressed_index(sys, top, (9, 1))
    @test_throws ArgumentError dressed_index(sys, top, (1,))
    @testset "depth=2 labels" begin
        q1, q2 = materialized(sys, top.children[1]), materialized(sys, top.children[2])
        dom(q, i) = (argmax(abs2.(view(q.vectors, :, i))),)
        L = bare_labels(sys, top; depth=2)
        for (i, j) in [(1, 1), (2, 1), (1, 3), (5, 7), (8, 8)]
            @test L[LinearIndices((8, 8))[i, j]] == (dom(q1, i), dom(q2, j))
        end
        @test length(unique(dom(q1, i) for i in 1:8)) >= 4   # charge-basis dominants are not all distinct (several states peak at the same n)
        @test issorted(abs2.(last.(d)); rev=true) && all(abs2.(last.(d)) .> 1e-3)
        @test_throws ArgumentError decompose(sys, top, 21)
    end
    @testset "dressed_index warning" begin
        @test_logs dressed_index(sys0, top, (1, 1))
        # strong coupling hybridizes some kept states (diagnostic ratio down to 0.14)
        sysg = @test_logs (:warn, r"Branch node") match_mode=:any materialize(top, spec, two_transmons_params(g=1.5))
        B = bare_overlaps(sysg, top)
        ps = [p for p in 1:64 if 1e-12 < sum(B[:, p]) && maximum(B[:, p]) < 0.5]
        @test !isempty(ps)
        label = Tuple(CartesianIndices((8, 8))[first(ps)])
        @test_logs (:warn, r"overlap") dressed_index(sysg, top, label)
    end
end

@testset "dressed_state_string" begin
    using KronFreeCircuits: dressed_state_string, GivenSystem, Term, Factor, Coefficient, Port
    spec = load_spec(fixture("two_transmons.json"))
    mk(i) = Node([Symbol("varphi_$i") => ChargeBasis(30)]; keep=lowest(8))
    top = Node([mk(1), mk(2)]; solver=Dense(), keep=lowest(20), ports=[(:varphi_1, nport())])
    sys = materialize(top, spec, two_transmons_params(g=0.3))
    # physics indices: dressed state n is decompose's n + 1, and every bare label is decompose's minus one
    for n in (0, 4)
        d = decompose(sys, top, n + 1; thresh=1e-2)
        s = dressed_state_string(sys, top, n; thresh=1e-2, digits=4)
        @test startswith(s, "|ψ$(n == 0 ? "₀" : "₄")⟩ = ")
        kets = [m.captures[1] for m in eachmatch(r"\|([0-9,]+)⟩", s)]
        @test kets == [join(l .- 1, ",") for l in first.(d)]
    end
    @test dressed_state_string(sys, 0) == dressed_state_string(sys, top, 0)
    @test_throws "0:19" dressed_state_string(sys, top, 20)
    @test_throws ArgumentError dressed_state_string(sys, top, -1)
    @test endswith(dressed_state_string(sys, top, 0; thresh=2.0), "(no bare state with |c|² ≥ 2.0)")

    # exact formatting on a hand-made two-level system: amplitudes ±1/√2 after a known rotation
    E = [0.0, 1.0]; X = [0.0 1.0; 1.0 0.0]
    ga, gb = GivenSystem(:a; energies=E, operators=Dict(:x => X)), GivenSystem(:b; energies=E, operators=Dict(:x => X))
    root = Node([ga, gb]; solver=Dense(), keep=lowest(4), ports=[(:a, Port(:x, 0))])
    s2 = materialize(root, [Term(Coefficient(0.5), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])], (;))
    v = s2.root.vectors[:, 1]
    sgn = sign(v[4] / v[1]) < 0 ? "−" : "+"
    t = dressed_state_string(s2, 0; digits=2)
    @test occursin(r"^\|ψ₀⟩ = −?0\.\d\d\|(0,0|1,1)⟩ [−+] 0\.\d\d\|(0,0|1,1)⟩$", t) && occursin(" $sgn ", t)
    @test dressed_state_string(s2, 0; digits=2, label="00") == replace(t, "ψ₀" => "00")
    l = dressed_state_string(s2, 0; digits=2, latex=true)
    @test startswith(l, "\\widetilde{|\\psi_{0}\\rangle} = ") && occursin("\\,|0,0\\rangle", l) && !occursin("−", l)
    @test startswith(dressed_state_string(s2, 0; latex=true, label="00"), "\\widetilde{|00\\rangle} = ")
    # a complex amplitude prints in parentheses: X⊗Y couples |0,0⟩ to |1,1⟩ with a phase of i
    Y = ComplexF64[0 -im; im 0]
    gy = GivenSystem(:b; energies=E, operators=Dict(:y => Y))
    rooty = Node([ga, gy]; solver=Dense(), keep=lowest(4), ports=[(:a, Port(:x, 0))])
    s3 = materialize(rooty, [Term(Coefficient(0.5), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:y, 0))])], (;))
    @test eltype(s3.root.vectors) <: Complex
    @test occursin(r"\(-?\d\.\d+[+-]\d\.\d+i\)\|(0,0|1,1)⟩", dressed_state_string(s3, 0; digits=2))
end
