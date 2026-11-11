using Test, KronFreeCircuits
@testset "spectrum" begin
    c = dc_two_transmons()
    y = spectrum(c.root, c.spec, c.θ; ports=[(:varphi_1, nport())])
    sys = materialize(c.root, c.spec, c.θ)
    @test y.energies == energies(sys) && y.ports[1] == port(sys, :varphi_1, nport()) && y.ports[1] !== port(sys, :varphi_1, nport())
    @test_throws ArgumentError spectrum(c.root, c.spec, c.θ; ports=[(:varphi_2, n2port())])     # not formed at the root
end

@testset "the core depends on no autodiff package" begin
    text = read(joinpath(pkgdir(KronFreeCircuits), "Project.toml"), String)
    deps = match(r"\[deps\]\n(.*?)\n\["s, text * "\n[").captures[1]           # the [deps] table alone
    weak = match(r"\[weakdeps\]\n(.*?)\n\["s, text * "\n[").captures[1]
    @test !occursin("ChainRulesCore", deps) && !occursin("Zygote", text) && !occursin("ForwardDiff", text)
    @test occursin("ChainRulesCore", weak)
end
