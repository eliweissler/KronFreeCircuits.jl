using KronFreeCircuits: GivenSystem, Term, Factor, Coefficient, SavedSolver

# A custom solver as in the README, with one setting that can be saved and one that cannot.
struct _ExactSolver <: Solver
    tol::Float64
    note::Any
end
function KronFreeCircuits.solve(::_ExactSolver, H, m)
    F = eigen(Hermitian(Matrix(H)), 1:m)
    return F.values, F.vectors
end

@testset "positions and one node" begin
    c = dc_chain3(); sys = materialize(c.root, c.spec, c.θ)
    p = tempname() * ".jld2"; save_system(p, sys)
    @test system_nodes(p) == [Int[], [1], [1, 1], [1, 2], [2]]
    mn = sys.root.children[1].children[2]
    n = load_node(p, [1, 2])
    @test n.energies == mn.energies && n.vectors == mn.vectors && n.kept == mn.kept && n.dims == mn.dims
    @test n.ports == mn.ports
    @test load_node(p).solver == "Dense" && load_node(p).on == "Array"   # Dense runs on the CPU
    @test_throws "valid positions" load_node(p, [3])
    @test_throws "valid positions" load_node(p, [1, 1, 1])
end

@testset "metadata" begin
    c = dc_chain3(); sys = materialize(c.root, c.spec, c.θ)
    p = tempname() * ".jld2"
    save_system(p, sys; metadata=(note="fig 3", grid=[1, 2, 3], flux=0.25))
    @test load_metadata(p) == Dict("note" => "fig 3", "grid" => [1, 2, 3], "flux" => 0.25)
    save_system(p, sys; metadata=Dict("r" => 1:3))
    @test load_metadata(p) == Dict("r" => [1, 2, 3])
    save_system(p, sys)
    @test isempty(load_metadata(p))
end

@testset "unstorable metadata" begin
    c = dc_chain3(); sys = materialize(c.root, c.spec, c.θ)
    p2 = tempname() * ".jld2"
    @test_throws "metadata" save_system(p2, sys; metadata=(f=sin,))
    @test !isfile(p2) && !isfile(p2 * ".partial")
end

@testset "given node and bad metadata keys" begin
    E1 = [0.0, 1.0, 2.5]; X1 = ComplexF64[0 1 0; 1 0 √2; 0 √2 0]
    g1 = GivenSystem(:a; energies=E1, operators=Dict(:x => X1))
    g2 = GivenSystem(:b; energies=[0.0, 0.8], operators=Dict(:x => ComplexF64[0 1; 1 0]))
    sys = materialize(Node([g1, g2]; solver=Dense(), keep=lowest(6), ports=[(:a, Port(:x, 0))]),
                      [Term(Coefficient(0.1), [Factor(:a, Port(:x, 0)), Factor(:b, Port(:x, 0))])], (;))
    p = tempname() * ".jld2"; save_system(p, sys)
    n = load_node(p, [1])
    @test n.energies == sys.root.children[1].energies && n.solver === nothing && n.on === nothing
    c = dc_two_transmons(solver=Lanczos(), compute=10); lan = materialize(c.root, c.spec, c.θ)
    save_system(p, lan)
    @test load_node(p).solver == "Lanczos" && load_node(p).on == "Array"
    for k in ("", "a/b")
        @test_throws "metadata" save_system(p, sys; metadata=Dict(k => 1))
    end
end

@testset "custom solver" begin
    c = dc_chain3()
    custom(note) = Node(c.root.children; solver=_ExactSolver(1e-10, note), compute=6, keep=lowest(6), ports=c.root.ports)
    sys = materialize(custom(:plain), c.spec, c.θ)
    p = tempname() * ".jld2"; save_system(p, sys)
    n = load_node(p)
    @test (n.solver, n.custom, n.on) == ("_ExactSolver", true, nothing)
    @test load_node(p, [1, 1]).custom == false
    # without the solver's code: the saved results and derivatives, and a clear error on a new solve
    a = load_system(p)
    s = a.root.node.solver
    @test s isa SavedSolver && s.name == "_ExactSolver" && Dict(s.fields) == Dict(:tol => 1e-10, :note => :plain)
    @test a.root.energies == sys.root.energies && a.root.vectors == sys.root.vectors && a.root.ports == sys.root.ports
    @test energies(derivatives(a; wrt=:g12), :g12) ≈ energies(derivatives(sys; wrt=:g12), :g12) rtol = 1e-6
    @test_throws "load_system(path; solvers=[_ExactSolver])" materialize(a.root.node, c.spec, c.θ)
    # saving the loaded system keeps the original name and settings
    p2 = tempname() * ".jld2"; save_system(p2, a)
    @test load_node(p2).solver == "_ExactSolver" && load_system(p2).root.node.solver.fields == s.fields
    # with the solver's code the real solver comes back and solves
    b = load_system(p; solvers=[_ExactSolver])
    @test b.root.node.solver == _ExactSolver(1e-10, :plain)
    assert_same_system(b, sys)
    @test materialize(b.root.node, c.spec, c.θ).root.energies ≈ sys.root.energies
    # a setting that cannot be saved keeps only its type, and the node keeps a SavedSolver even with the code
    save_system(p, materialize(custom(sin), c.spec, c.θ))
    b = @test_logs (:warn, r"note were not saved") load_system(p; solvers=[_ExactSolver])
    @test b.root.node.solver isa SavedSolver && Dict(b.root.node.solver.fields)[:note] == KronFreeCircuits._Unsaved("typeof(sin)")
end
