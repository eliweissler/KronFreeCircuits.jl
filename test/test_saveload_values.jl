using KronFreeCircuits: _write_struct, _read_struct, _write_value, _read_value, _SAVE_TYPES, JLD2
using InteractiveUtils: subtypes
allsub(T) = isabstracttype(T) ? reduce(vcat, map(allsub, subtypes(T)); init=Any[]) :
            (parentmodule(T) === KronFreeCircuits ? Any[T] : Any[])
@testset "settings round trip" begin
    xs = Any[Dense(), Lanczos(tol=1e-9, krylovdim=30), LOBPCG(tol=1e-7, preconditioner=:none, callback=x -> nothing),
             ShiftInvert(), ChebyshevFilter(degree=20), Auto(dense_cutoff=500),
             lowest(4), indices([3, 1, 2]), connected([1, 2], (:phi, nport()), 1e-3), follow(), follow([1.0 0; 0 1]; minoverlap=0.3),
             follow(ComplexF64[1 0; 0 1]), within(2.5; max=8),
             ChargeBasis(10), PhaseGrid(-3π, 3π, 201; order=6), PeriodicGrid(64; sectors=2), FockBasis(30; phi_zpf=0.7)]
    path = tempname() * ".jld2"
    JLD2.jldopen(path, "w") do f
        for (i, x) in enumerate(xs); _write_struct(f, "x$i", x); end
    end
    JLD2.jldopen(path, "r") do f
        for (i, x) in enumerate(xs)
            y = _read_struct(f, "x$i")
            @test typeof(y) == typeof(x)
            for n in fieldnames(typeof(x))
                n === :callback ? (@test getfield(y, n) === nothing) : (@test isequal(getfield(y, n), getfield(x, n)))
            end
        end
    end
    # every concrete solver, rule and basis type is in the allowlist (a new type must be added to it); SavedSolver is
    # the stand-in for a custom solver and is saved under that solver's name
    @test Set(keys(_SAVE_TYPES)) == Set(string(nameof(T)) for T in vcat(allsub(Solver), allsub(SelectionRule), allsub(Basis))
                                        if T !== KronFreeCircuits.SavedSolver)
    @test_throws ArgumentError JLD2.jldopen(f -> _write_value(f, "bad", x -> x), tempname() * ".jld2", "w")
end
@testset "the on keyword replaces the saved device" begin
    path = tempname() * ".jld2"
    JLD2.jldopen(f -> _write_struct(f, "s", LOBPCG(on=Array)), path, "w")
    JLD2.jldopen(path, "r") do f
        @test _read_struct(f, "s").on === Array
        @test _read_struct(f, "s"; on=Vector).on === Vector       # any array type passes through unchanged
        @test f["s/fields/on/value"] == "Array"                    # the saved name, for the record
    end
end
@testset "unknown types and kinds are refused" begin
    path = tempname() * ".jld2"
    JLD2.jldopen(path, "w") do f
        f["x/type"] = "Evil"
        f["y/kind"] = "evil"
    end
    JLD2.jldopen(path, "r") do f
        @test_throws ArgumentError _read_struct(f, "x")
        @test_throws ArgumentError _read_value(f, "y")
    end
end
