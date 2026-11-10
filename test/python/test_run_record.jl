using JSON3
include("kite.jl")

res, θ, frames = kite()
@testset "a run record rebuilds the circuit and frame" begin     # acceptance 4
    text = run_record(res; frame=frames.optimum, values=θ, metadata=(note="kite test",))
    @test parse_spec(text) isa Spec                               # still a valid spec
    back = load_record(text)
    @test back.frame ≈ frames.optimum && back.values == θ
    a, b = Spec(res; frame=frames.optimum), Spec(back.result; frame=back.frame)
    @test a.parameters == b.parameters
    @test all(isapprox(a.E_C[i,j](θ), b.E_C[i,j](θ); rtol=1e-12) for i in 1:3, j in 1:3)
    path = joinpath(mktempdir(), "run.json"); write(path, text)
    @test load_record(path).values == θ
    @test occursin("\"_generator\"", text) && occursin("kronfree", text)
end

# Two specs with the same parameters whose coefficients agree at v.
function same_spec(a::Spec, b::Spec, v::NamedTuple)
    ≈(x, y) = isapprox(x(v), y(v); rtol=1e-12, atol=1e-14)
    return a.parameters == b.parameters && [m.type for m in a.modes] == [m.type for m in b.modes] &&
           all(a.E_C .≈ b.E_C) && all(a.E_L .≈ b.E_L) && all(a.flux_linear .≈ b.flux_linear) &&
           all(a.n_g .≈ b.n_g) && a.constant ≈ b.constant && length(a.cosines) == length(b.cosines) &&
           all(c.omega == d.omega && c.energy ≈ d.energy && c.offset ≈ d.offset for (c, d) in zip(a.cosines, b.cosines))
end

# The spec of the Result that Python's own `load_record` rebuilds from `text`.
python_spec(text::AbstractString) = Spec(circuit_quantization().load_record(text))

residual(r::Py) = pyis(r.residual_S, pybuiltins.None) ? nothing : pyconvert(Matrix{Float64}, pyimport("numpy").array(r.residual_S.evalf(), dtype="float"))

@testset "the record round trip: frame $(f !== nothing), values $(v !== nothing)" for
        f in (nothing, frames.optimum), v in (nothing, θ)
    text = run_record(res; frame=f, values=v)
    back = load_record(text)
    @test back.frame == f && back.frame isa Union{Nothing,Matrix{Float64}}
    @test back.values === v
    @test isempty(back.spec.parameters) == (v !== nothing)          # numeric exactly when values are given
    @test same_spec(back.spec, parse_spec(text), θ)                 # (a) the spec of the document
    @test same_spec(back.spec, Spec(res; frame=f, subs=v), θ)       # ... which is the spec of the run
    @test same_spec(Spec(back.result; frame=back.frame), Spec(res; frame=f), θ)   # (c) result is symbolic, in res's frame
    @test residual(back.result) === nothing
    @test same_spec(python_spec(text), parse_spec(text), θ)        # (b) a faithful circuit-quantization record
    d = JSON3.read(text)
    @test !occursin("NaN", text) && !occursin("Infinity", text)
    @test d._generator.kronfree.version == string(pkgversion(KronFreeCircuits))
    @test haskey(d._generator.kronfree, :git_commit)
    @test (d._frame.S === nothing) == (f === nothing)
end

@testset "a Result that already carries a residual frame" begin
    sympy = pyimport("sympy")
    S1 = [1 0 0; 0 1 1; 0 0 1]
    res1 = res.transform(sympy.Matrix(pylist(pylist.(eachrow(S1)))))
    text = run_record(res1; frame=frames.optimum, values=θ)
    back = load_record(text)
    @test back.frame == frames.optimum
    @test residual(back.result) == S1                                # result is res1, in res1's own frame
    @test same_spec(Spec(back.result; frame=back.frame), Spec(res1; frame=frames.optimum), θ)
    @test same_spec(back.spec, Spec(res1; frame=frames.optimum, subs=θ), θ)
    @test same_spec(python_spec(text), parse_spec(text), θ)
    @test JSON3.read(text)._frame.S.entries != JSON3.read(run_record(res; frame=frames.optimum))._frame.S.entries
end

@testset "record arguments" begin
    text = run_record(res; metadata=(note="kite", tags=["a", "b"], n=3, x=0.5, ok=true, none=nothing, sub=(k=1,)))
    d = JSON3.read(text)
    @test d._metadata.note == "kite" && d._metadata.tags == ["a", "b"] && d._metadata.n == 3
    @test d._metadata.x == 0.5 && d._metadata.ok === true && d._metadata.none === nothing && d._metadata.sub.k == 1
    @test_throws ArgumentError run_record(res; metadata=(bad=1 + 2im,))
    @test_throws ArgumentError run_record(res; metadata=(bad=NaN,))
    @test_throws ArgumentError run_record(res; metadata=(bad=[Inf],))

    error_of(f) = try f(); nothing catch e; e end
    err = error_of(() -> run_record(res; values=merge(θ, (E_K=1.0,))))
    @test err isa ArgumentError && occursin("E_K", err.msg) && occursin("E_CJ, E_CS, E_J, E_L", err.msg)
    err = error_of(() -> run_record(res; values=Base.structdiff(θ, (E_J=0,))))
    @test err isa ArgumentError && occursin("E_J", err.msg) && occursin("E_CJ, E_CS, E_J, E_L", err.msg)
    @test_throws ArgumentError run_record(res; values=merge(θ, (E_J="E_L",)))       # numbers only
    @test_throws ArgumentError run_record(res; frame=ones(2, 2))                     # Spec(res; frame)'s checks
    @test_throws ArgumentError run_record(res; frame=[1 0 0; 0 1 0; 0 0 NaN])
    @test_throws ArgumentError run_record(res; frame=[2 0 0; 0 1 0; 0 0 1])

    missing_path = joinpath(mktempdir(), "absent.json")
    err = error_of(() -> load_record(missing_path))
    @test err isa ArgumentError && occursin(missing_path, err.msg)
    @test load_record("  \n" * run_record(res)).values === nothing          # leading whitespace: still JSON text
    plain = pyconvert(String, pyimport("json").dumps(circuit_quantization().to_record(res)))
    err = error_of(() -> load_record(plain))                                # not written by run_record
    @test err isa ArgumentError && occursin("_kronfree", err.msg)
end

@testset "a record loads in a fresh Julia session" begin                     # acceptance 4
    text = run_record(res; frame=frames.optimum, values=θ)
    path = joinpath(mktempdir(), "run.json"); write(path, text)
    s = Spec(res; frame=frames.optimum)
    code = """
        using KronFreeCircuits, PythonCall
        back = load_record(ARGS[1])
        s = Spec(back.result; frame=back.frame)
        at(c) = real(c(back.values))
        print(join(string.(s.parameters), ","), ";", at(s.E_C[1, 1]), ";", at(s.E_C[2, 3]), ";", at(s.cosines[1].energy))
        """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code $path`
    out = read(cmd, String)
    params, x11, x23, e1 = split(out, ";")
    @test params == join(string.(s.parameters), ",")
    @test parse(Float64, x11) ≈ s.E_C[1, 1](θ) rtol=1e-12
    @test parse(Float64, x23) ≈ s.E_C[2, 3](θ) rtol=1e-12
    @test parse(Float64, e1) ≈ s.cosines[1].energy(θ) rtol=1e-12
end
