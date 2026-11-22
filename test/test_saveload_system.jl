using KronFreeCircuits: GivenSystem, Term, Factor, Coefficient, System, MaterializedNode, Retained, JLD2

round_trip(sys; kwargs...) = (p = tempname() * ".jld2"; save_system(p, sys; kwargs...); (p, load_system(p)))

@testset "real leaf" begin
    c = dc_transmon(); sys = materialize(c.root, c.spec, c.θ)
    _, loaded = round_trip(sys)
    assert_same_system(loaded, sys)
end

@testset "complex leaf" begin
    c = dc_transmon(periodic=true); sys = materialize(c.root, c.spec, c.θ)
    _, loaded = round_trip(sys)
    assert_same_system(loaded, sys)
    @test eltype(loaded.root.vectors) == ComplexF64
end

@testset "two-level and three-level trees" begin
    function check_identity(mn::MaterializedNode)
        mn.node isa Node || return
        for (i, ch) in enumerate(mn.children)
            ch isa MaterializedNode || continue
            @test ch.node === mn.node.children[i]
            check_identity(ch)
        end
    end
    for c in (dc_two_transmons(), dc_chain3())
        sys = materialize(c.root, c.spec, c.θ)
        _, loaded = round_trip(sys)
        assert_same_system(loaded, sys)
        check_identity(loaded.root)
    end
end

@testset "given system" begin
    E1 = [0.0, 1.0, 2.5]; X1 = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
    E2 = [0.0, 0.8];     X2 = ComplexF64[0 1; 1 0]
    g1 = GivenSystem(:a; energies=E1, operators=Dict(:x => X1))
    g2 = GivenSystem(:b; energies=E2, operators=Dict(:x => X2))
    top = Node([g1, g2]; solver=Dense(), keep=lowest(6), ports=[(:a, Port(:x, 0))])
    terms = [Term(Coefficient(0.1), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])]
    sys = materialize(top, terms, (;))
    _, loaded = round_trip(sys)
    assert_same_system(loaded, sys)
    for (i, ch) in enumerate(loaded.root.children)
        @test ch.node === loaded.root.node.children[i] === ch.children[1]
    end
end

@testset "retained = false" begin
    c = dc_fluxonium(compute=8, keep=lowest(5), retain=:always); sys = materialize(c.root, c.spec, c.θ)
    full, _ = round_trip(sys)
    p, loaded = round_trip(sys; retained=false)
    @test loaded.root.retained.vectors === nothing
    @test loaded.root.retained.complete == false
    @test loaded.root.node.retain === :always
    r, mn = sys.root.retained, sys.root
    expected_root = typeof(mn)(mn.node, mn.dims, mn.energies, mn.vectors, mn.kept, mn.ports, mn.joint_ports, mn.children,
                               mn.peak_component, mn.peak_amplitudes,
                               typeof(r)(r.energies, r.unkept, nothing, false, r.followed))
    assert_same_system(loaded, System(expected_root, sys.spec, sys.terms, sys.θ, sys.constant))
    @test filesize(p) < filesize(full)
end

@testset "device chosen at load" begin
    c = dc_two_transmons(solver=LOBPCG(), compute=6, leafkeep=6); sys = materialize(c.root, c.spec, c.θ)
    p, loaded = round_trip(sys)
    assert_same_system(loaded, sys)
    @test loaded.root.node.solver.on === Array
    @test load_system(p; on=Vector).root.node.solver.on === Vector
end

@testset "failed save leaves the old file" begin
    c = dc_transmon(); sys = materialize(c.root, c.spec, c.θ)
    p, _ = round_trip(sys)
    @test_throws ArgumentError save_system(p, sys; metadata=(f=sin,))
    assert_same_system(load_system(p), sys)
    @test !isfile(p * ".partial")
end

@testset "not a system file" begin
    txt = tempname() * ".jld2"; write(txt, "hello")
    @test_throws "not a KronFreeCircuits system file" load_system(txt)
    other = tempname() * ".jld2"; JLD2.jldsave(other; x=1)
    @test_throws "not a KronFreeCircuits system file" load_system(other)
    noversion = tempname() * ".jld2"; JLD2.jldsave(noversion; format="KronFreeCircuits.System")
    @test_throws "not a KronFreeCircuits system file" load_system(noversion)
end

@testset "unknown format version" begin
    c = dc_transmon(); sys = materialize(c.root, c.spec, c.θ)
    p0, _ = round_trip(sys)
    p = tempname() * ".jld2"; cp(p0, p)
    JLD2.jldopen(p, "r+") do f
        delete!(f, "format_version")
        f["format_version"] = 2
    end
    @test_throws "version 2" load_system(p)
end
