@testset "a loaded system works" begin
    roundtrip(sys; kw...) = (p = tempname() * ".jld2"; save_system(p, sys; kw...); load_system(p))

    @testset "port, apply, decompose" begin
        c = dc_two_transmons(); S = materialize(c.root, c.spec, c.θ); L = roundtrip(S)
        @test port(L, :varphi_1, nport()) == port(S, :varphi_1, nport())
        @test decompose(L, L.root.node, 2) == decompose(S, S.root.node, 2)
        @test bare_labels(L) == bare_labels(S)
    end

    @testset "derivatives" begin
        c = dc_two_transmons(retain=:always); S = materialize(c.root, c.spec, c.θ)
        dS = energies(derivatives(S; wrt=:E_J1), :E_J1)
        @test energies(derivatives(roundtrip(S); wrt=:E_J1), :E_J1) == dS
        L = roundtrip(S; retained=false)
        dL = @test_logs (:info, r"re-diagonalized") match_mode=:any energies(derivatives(L; wrt=:E_J1), :E_J1)
        @test isapprox(dL, dS; rtol=1e-8)
    end

    @testset "previous" begin
        # (a root with keep=follow() needs a reference on the first solve, so the root follows phases instead)
        function tree()
            c = dc_two_transmons()
            leaves = [Node([Symbol("varphi_$i") => ChargeBasis(30)]; solver=Dense(), keep=lowest(4)) for i in 1:2]
            (; c..., root=Node(leaves; solver=Dense(), keep=lowest(6), phase=:follow))
        end
        c = tree(); S = materialize(c.root, c.spec, c.θ); L = roundtrip(S)
        θ2 = merge(c.θ, (g=0.31,))
        c1 = tree(); s1 = materialize(c1.root, c1.spec, θ2; previous=S)
        c2 = tree(); s2 = materialize(c2.root, c2.spec, θ2; previous=L)
        @test energies(s2) == energies(s1)
        @test s2.root.kept == s1.root.kept
        @test eigenvectors(s2, c2.root) == eigenvectors(s1, c1.root)
    end
end
