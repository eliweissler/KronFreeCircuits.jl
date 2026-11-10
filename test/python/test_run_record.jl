using JSON3
include("kite.jl")

res, θ, frames = kite()
θc = NamedTuple{Tuple(Spec(res).parameters)}(θ)                  # θ in the circuit's parameter order
@testset "a run record rebuilds the circuit and frame" begin     # acceptance 4
    text = run_record(res; frame=frames.optimum, values=θ, metadata=(note="kite test",))
    @test parse_spec(text) isa Spec                               # still a valid spec
    back = load_record(text)
    @test back.frame ≈ frames.optimum && back.values === θc
    a, b = Spec(res; frame=frames.optimum), Spec(back.result; frame=back.frame)
    @test a.parameters == b.parameters
    @test all(isapprox(a.E_C[i,j](θ), b.E_C[i,j](θ); rtol=1e-12) for i in 1:3, j in 1:3)
    path = joinpath(mktempdir(), "run.json"); write(path, text)
    @test load_record(path).values === θc
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
    @test back.values === (v === nothing ? nothing : θc)            # in the circuit's parameter order
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
    back = load_record(run_record(res1))                              # no frame applied
    @test back.frame === nothing && residual(back.result) == S1
    @test same_spec(back.spec, Spec(res1), θ) && same_spec(Spec(back.result), Spec(res1), θ)
    text = run_record(res1; frame=frames.optimum, values=θ)
    back = load_record(text)
    @test back.frame == frames.optimum
    @test residual(back.result) == S1                                # result is res1, in res1's own frame
    @test same_spec(Spec(back.result; frame=back.frame), Spec(res1; frame=frames.optimum), θ)
    @test same_spec(back.spec, Spec(res1; frame=frames.optimum, subs=θ), θ)
    @test same_spec(python_spec(text), parse_spec(text), θ)
    @test JSON3.read(text)._frame.S.entries != JSON3.read(run_record(res; frame=frames.optimum))._frame.S.entries
end

@testset "the path route, without frame or values" begin
    path = joinpath(mktempdir(), "plain.json"); write(path, run_record(res))
    back = load_record(path)
    @test back.frame === nothing && back.values === nothing && same_spec(back.spec, Spec(res), θ)
    @test same_spec(Spec(back.result), Spec(res), θ) && residual(back.result) === nothing
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
    listing = "the circuit's parameters are " * join(Spec(res).parameters, ", ")   # in the circuit's order
    err = error_of(() -> run_record(res; values=merge(θ, (E_K=1.0,))))
    @test err isa ArgumentError && occursin("E_K", err.msg) && occursin(listing, err.msg)
    err = error_of(() -> run_record(res; values=Base.structdiff(θ, (E_J=0,))))
    @test err isa ArgumentError && occursin("no value for E_J", err.msg) && occursin(listing, err.msg)
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

# --- Offset charges, tolerances and parameter values in a record ------------------------------------------------------
include("circuits.jl")

# Two specs of constants, equal field by field: modes, every coefficient, every cosine.
function same_constants(a::Spec, b::Spec; rtol=1e-12, atol=1e-13)
    v(c) = c((;))
    ≈(x, y) = isapprox(v(x), v(y); rtol, atol)
    return isempty(a.parameters) && isempty(b.parameters) && a.modes == b.modes &&
           all(a.E_C .≈ b.E_C) && all(a.E_L .≈ b.E_L) && all(a.flux_linear .≈ b.flux_linear) &&
           all(a.n_g .≈ b.n_g) && a.constant ≈ b.constant && length(a.cosines) == length(b.cosines) &&
           all(c.omega == d.omega && c.energy ≈ d.energy && c.offset ≈ d.offset for (c, d) in zip(a.cosines, b.cosines))
end

# The fixture `two_transmons` (two compact modes) at values in its base frame: element values for E_C1 = 0.3,
# E_C2 = 0.25 and g = 0.08 GHz (see its comment in test/fixtures/generated), and nonzero sector offsets.
two_transmons_values() = (x_1=0.0749 / 0.24, x_2=0.0749 / 0.29, x_c=8 * 0.0749 / 0.08, n_g1=0.3, n_g2=0.1,
                          E_J1=12.0, E_J2=10.0)

@testset "a record with two compact modes carries the transformed sector offsets" begin    # Critical 1
    res2, _ = fixture_result("two_transmons")
    θ2 = two_transmons_values()
    S = [1.0 1.0; 0.0 1.0]                                              # S_cc ≠ ±1: S_ccᵀ, S_cc⁻¹ and S_cc differ
    ref = Spec(QuantizedCircuit(res2; values=θ2), S)
    @test [ref.n_g[j]((;)) for j in 1:2] ≈ [0.3, 0.4]                   # S_ccᵀ (0.3, 0.1)
    text = run_record(res2; frame=S, values=θ2)
    rec = load_record(text)
    @test rec.values == θ2 && rec.frame == S                            # base-frame values, as given
    @test same_constants(rec.spec, ref)                                 # the spec of the run is the loop route's
    @test same_constants(Spec(QuantizedCircuit(rec.result; values=rec.values), rec.frame), ref)
    @test same_constants(Spec(circuit_quantization().load_record(text)), ref)   # Python's own reading
end

@testset "a record carries the zero tolerance of its run" begin               # Important 2
    iz = KronFreeCircuits.iszero_expr
    qc = QuantizedCircuit(res; values=θ)
    # The normal frame turned by 1e-10 rad: E_L[2,3] ≈ 6e-12 of max|E_L|, zeroed at zero_rtol = 1e-9 only.
    ε = 1e-10
    S = frames.normal * [1 0 0; 0 cos(ε) -sin(ε); 0 sin(ε) cos(ε)]
    for zr in (1e-12, 1e-9)
        rec = load_record(run_record(res; frame=S, values=θ, zero_rtol=zr))
        l = Spec(qc, S; zero_rtol=zr)
        @test iz.(rec.spec.E_C) == iz.(l.E_C) && iz.(rec.spec.E_L) == iz.(l.E_L)
        @test rec.zero_rtol == zr && rec.zero_atol == 0.0
    end
    @test !iz(load_record(run_record(res; frame=S, values=θ)).spec.E_L[2, 3])
    @test iz(load_record(run_record(res; frame=S, values=θ, zero_rtol=1e-9)).spec.E_L[2, 3])
    rec = load_record(run_record(res; frame=S, values=θ, zero_rtol=0, zero_atol=1e-10))
    @test rec.zero_rtol == 0.0 && rec.zero_atol == 1e-10 && iz(rec.spec.E_L[2, 3])
    @test_throws "zero_rtol" run_record(PythonCall.pybuiltins.None; zero_rtol=-1.0)  # before any Python call
    @test_throws "zero_atol" run_record(PythonCall.pybuiltins.None; zero_atol=NaN)
end

@testset "one validator for values" begin                                     # Important 3
    sympy = pyimport("sympy")
    r, θk, _ = kite(; cb_flux=sympy.Symbol("phi_cb"; real=true))   # a flux symbol the exported spec does not contain
    order = Spec(r).parameters
    error_of(f) = try f(); nothing catch e; e end
    # θk serves both; phi_cb is a parameter of neither
    @test collect(keys(QuantizedCircuit(r; values=θk).values)) == order
    rec = load_record(run_record(r; values=θk))
    @test collect(keys(rec.values)) == order && rec.values == NamedTuple{Tuple(order)}(θk)   # the circuit's order
    for f in (() -> QuantizedCircuit(r; values=merge(θk, (phi_cb=0.1,))),
              () -> run_record(r; values=merge(θk, (phi_cb=0.1,))))
        e = error_of(f)
        @test e isa ArgumentError && occursin("phi_cb is not a parameter of the circuit", e.msg) &&
              all(occursin(string(k), e.msg) for k in order)
    end
    # the same message from both, for each kind of bad input
    d = Dict{Any,Any}(pairs(θk)); d["E_J"] = 1.0                         # one name given twice
    for bad in (d, [:E_J => 1.0], Base.structdiff(θk, (E_L=0,)), merge(θk, (E_J=true,)))
        a, b = error_of(() -> QuantizedCircuit(r; values=bad)), error_of(() -> run_record(r; values=bad))
        @test a isa ArgumentError && b isa ArgumentError && a.msg == b.msg
    end
    # subs accepts only the symbols the export can contain
    e = error_of(() -> Spec(r; subs=(phi_cb=0.1,)))
    @test e isa ArgumentError && occursin("phi_cb is not a symbol", e.msg) &&
          !occursin("phi_cb", split(e.msg, "symbols are")[end])
end
