using JSON3
include("kite.jl")

res, θ, frames = kite()
θc = NamedTuple{Tuple(Spec(res).parameters)}(θ)                  # θ in the circuit's parameter order
qc = QuantizedCircuit(res; values=θ)

# The spec a run solves: without values the symbolic spec `Spec(r; frame)`, with values the loop route's spec of
# constants.
function solved(r, f, v; zero...)
    v === nothing && return Spec(r; frame=f, zero...)
    q = QuantizedCircuit(r; values=v)
    return f === nothing ? Spec(q; zero...) : Spec(q, f; zero...)
end

# Two specs equal coefficient for coefficient: the same parameters and modes, every coefficient the same expression
# (a number of the same type and value, so an exact zero stays an exact zero), every cosine vector the same.
function identical(a::Spec, b::Spec)
    same(x, y) = typeof(x.ex) == typeof(y.ex) && isequal(x.ex, y.ex)
    cos_a = [(c.energy, c.offset) for c in a.cosines]
    cos_b = [(c.energy, c.offset) for c in b.cosines]
    return a.parameters == b.parameters && a.modes == b.modes &&
           all(same.(a.E_C, b.E_C)) && all(same.(a.E_L, b.E_L)) && all(same.(a.flux_linear, b.flux_linear)) &&
           all(same.(a.n_g, b.n_g)) && same(a.constant, b.constant) && length(a.cosines) == length(b.cosines) &&
           all(c.omega == d.omega for (c, d) in zip(a.cosines, b.cosines)) &&
           all(same(x[1], y[1]) && same(x[2], y[2]) for (x, y) in zip(cos_a, cos_b)) &&
           isempty(a.charge_cosines) && isempty(b.charge_cosines)
end

error_of(f) = try f(); nothing catch e; e end

@testset "a record rebuilds the circuit and frame" begin                  # acceptance 4
    spec = solved(res, frames.optimum, θ)
    text = record_json(spec; circuit=res, frame=frames.optimum, values=θ, metadata=(note="kite test",))
    @test identical(parse_spec(text), spec)                       # still a valid spec: the solved one
    back = load_record(text)
    @test back.frame ≈ frames.optimum && back.values === θc
    a, b = Spec(res; frame=frames.optimum), Spec(back.result; frame=back.frame)
    @test a.parameters == b.parameters
    @test all(isapprox(a.E_C[i,j](θ), b.E_C[i,j](θ); rtol=1e-12) for i in 1:3, j in 1:3)
    path = joinpath(mktempdir(), "run.json"); write(path, text)
    @test load_record(path).values === θc
    @test occursin("\"_generator\"", text) && occursin("kronfree", text)
end

@testset "the document's spec is the given spec, at zero_rtol = 1e-9" begin
    # The KITE at the optimum frame by the loop route, zeroed at 1e-9: rebuilding the spec by another route without
    # the tolerance would give it another zero pattern.
    spec = solved(res, frames.optimum, θ; zero_rtol=1e-9)
    text = record_json(spec; circuit=res, frame=frames.optimum, values=θ, zero_rtol=1e-9)
    iz = KronFreeCircuits.iszero_expr
    doc = parse_spec(text)
    @test identical(doc, spec)
    @test iz.(doc.E_C) == iz.(spec.E_C) && iz.(doc.E_L) == iz.(spec.E_L) && iz.(doc.n_g) == iz.(spec.n_g)
    @test load_record(text).zero_rtol == 1e-9
end

# Two specs with the same parameters whose coefficients agree at v, and whose cosine vectors agree to round-off (the
# loop route and the symbolic route compute a non-integer omega entry by different arithmetic).
function same_spec(a::Spec, b::Spec, v::NamedTuple)
    ≈(x, y) = isapprox(x(v), y(v); rtol=1e-12, atol=1e-14)
    return a.parameters == b.parameters && [m.type for m in a.modes] == [m.type for m in b.modes] &&
           all(a.E_C .≈ b.E_C) && all(a.E_L .≈ b.E_L) && all(a.flux_linear .≈ b.flux_linear) &&
           all(a.n_g .≈ b.n_g) && a.constant ≈ b.constant && length(a.cosines) == length(b.cosines) &&
           all(isapprox(c.omega, d.omega; rtol=1e-12) && c.energy ≈ d.energy && c.offset ≈ d.offset
               for (c, d) in zip(a.cosines, b.cosines))
end

# The spec of the Result that Python's own `load_record` rebuilds from `text`.
python_spec(text::AbstractString) = Spec(circuit_quantization().load_record(text))

residual(r::Py) = pyis(r.residual_S, pybuiltins.None) ? nothing : pyconvert(Matrix{Float64}, pyimport("numpy").array(r.residual_S.evalf(), dtype="float"))

@testset "the record round trip: frame $(f !== nothing), values $(v !== nothing)" for
        f in (nothing, frames.optimum), v in (nothing, θ)
    spec = solved(res, f, v)
    text = record_json(spec; circuit=res, frame=f, values=v)
    back = load_record(text)
    @test back.frame == f && back.frame isa Union{Nothing,Matrix{Float64}}
    @test back.values === (v === nothing ? nothing : θc)            # in the circuit's parameter order
    @test isempty(back.spec.parameters) == (v !== nothing)          # numeric exactly when values are given
    @test identical(back.spec, spec)                                # (a) the spec of the document is the solved spec
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
    spec1 = Spec(res1)
    back = load_record(record_json(spec1; circuit=res1))                 # no frame applied
    @test back.frame === nothing && residual(back.result) == S1
    @test identical(back.spec, spec1) && same_spec(Spec(back.result), Spec(res1), θ)
    text = record_json(solved(res1, frames.optimum, θ); circuit=res1, frame=frames.optimum, values=θ)
    back = load_record(text)
    @test back.frame == frames.optimum
    @test residual(back.result) == S1                                # result is res1, in res1's own frame
    @test same_spec(Spec(back.result; frame=back.frame), Spec(res1; frame=frames.optimum), θ)
    @test same_spec(back.spec, Spec(res1; frame=frames.optimum, subs=θ), θ)
    @test same_spec(python_spec(text), parse_spec(text), θ)
    other = record_json(solved(res, frames.optimum, nothing); circuit=res, frame=frames.optimum)
    @test JSON3.read(text)._frame.S.entries != JSON3.read(other)._frame.S.entries
end

sres = Spec(res)                                                    # the symbolic spec of res in its own frame

@testset "save_record writes the file; the path route, without frame or values" begin
    dir = mktempdir()
    path = joinpath(dir, "plain.json")
    @test save_record(path, sres; circuit=res) === path
    back = load_record(path)
    @test back.frame === nothing && back.values === nothing && identical(back.spec, sres)
    @test same_spec(Spec(back.result), Spec(res), θ) && residual(back.result) === nothing

    spec = solved(res, frames.optimum, θ)
    path = joinpath(dir, "run.json")
    @test save_record(path, spec; circuit=res, frame=frames.optimum, values=θ, metadata=(note="saved",)) === path
    text = read(path, String)
    @test text == record_json(spec; circuit=res, frame=frames.optimum, values=θ, metadata=(note="saved",)) * "\n"
    a, b = load_record(path), load_record(text)
    @test a.frame == b.frame && a.values === b.values && a.zero_rtol == b.zero_rtol && a.zero_atol == b.zero_atol
    @test identical(a.spec, b.spec) && identical(a.spec, spec)
    @test same_spec(Spec(a.result), Spec(b.result), θ)
end

@testset "record arguments" begin
    text = record_json(sres; circuit=res,
                       metadata=(note="kite", tags=["a", "b"], n=3, x=0.5, ok=true, none=nothing, sub=(k=1,)))
    d = JSON3.read(text)
    @test d._metadata.note == "kite" && d._metadata.tags == ["a", "b"] && d._metadata.n == 3
    @test d._metadata.x == 0.5 && d._metadata.ok === true && d._metadata.none === nothing && d._metadata.sub.k == 1
    @test_throws ArgumentError record_json(sres; circuit=res, metadata=(bad=1 + 2im,))
    @test_throws ArgumentError record_json(sres; circuit=res, metadata=(bad=NaN,))
    @test_throws ArgumentError record_json(sres; circuit=res, metadata=(bad=[Inf],))

    spec = solved(res, nothing, θ)
    listing = "the circuit's parameters are " * join(Spec(res).parameters, ", ")   # in the circuit's order
    err = error_of(() -> record_json(spec; circuit=res, values=merge(θ, (E_K=1.0,))))
    @test err isa ArgumentError && occursin("E_K", err.msg) && occursin(listing, err.msg)
    err = error_of(() -> record_json(spec; circuit=res, values=Base.structdiff(θ, (E_J=0,))))
    @test err isa ArgumentError && occursin("no value for E_J", err.msg) && occursin(listing, err.msg)
    @test_throws ArgumentError record_json(spec; circuit=res, values=merge(θ, (E_J="E_L",)))       # numbers only
    @test_throws ArgumentError record_json(sres; circuit=res, frame=ones(2, 2))          # Spec(res; frame)'s checks
    @test_throws ArgumentError record_json(sres; circuit=res, frame=[1 0 0; 0 1 0; 0 0 NaN])
    @test_throws ArgumentError record_json(sres; circuit=res, frame=[2 0 0; 0 1 0; 0 0 1])
    @test_throws UndefKeywordError record_json(sres)                                      # circuit is required

    missing_path = joinpath(mktempdir(), "absent.json")
    err = error_of(() -> load_record(missing_path))
    @test err isa ArgumentError && occursin(missing_path, err.msg)
    @test load_record("  \n" * record_json(sres; circuit=res)).values === nothing     # leading whitespace: JSON text
    plain = pyconvert(String, pyimport("json").dumps(circuit_quantization().to_record(res)))
    err = error_of(() -> load_record(plain))                                # not written by save_record
    @test err isa ArgumentError && occursin("_kronfree", err.msg)
end

@testset "the check: the provenance must describe the spec" begin
    spec = solved(res, frames.optimum, θ)
    caught(f) = (e = error_of(f); e isa ArgumentError && occursin("check = false", e.msg) ? e.msg : nothing)
    names_field(msg) = msg !== nothing && occursin(r"(parameters|modes|E_C|E_L|flux_linear|n_g|cosines|constant)", msg)

    # the spec of one frame recorded with another frame
    msg = caught(() -> record_json(spec; circuit=res, frame=frames.normal, values=θ))
    @test names_field(msg)
    text = record_json(spec; circuit=res, frame=frames.normal, values=θ, check=false)
    @test identical(parse_spec(text), spec) && load_record(text).frame == frames.normal   # written as given
    # a spec recorded with other values
    θ′ = merge(θ, (E_J=θ.E_J * 1.01,))
    msg = caught(() -> record_json(spec; circuit=res, frame=frames.optimum, values=θ′))
    @test names_field(msg) && occursin("cosines", msg)
    text = record_json(spec; circuit=res, frame=frames.optimum, values=θ′, check=false)
    @test identical(parse_spec(text), spec) && load_record(text).values.E_J == θ′.E_J
    # a numeric spec recorded without values, and a symbolic spec recorded with values
    @test occursin("parameters", something(caught(() -> record_json(spec; circuit=res, frame=frames.optimum)), ""))
    @test occursin("parameters", something(caught(() -> record_json(sres; circuit=res, values=θ)), ""))
    # mode names other than the circuit's defaults
    named = Spec(res; mode_names=["a", "b", "c"])
    @test occursin("modes", something(caught(() -> record_json(named; circuit=res)), ""))
    @test parse_spec(record_json(named; circuit=res, check=false)).modes == named.modes
    # a symbolic spec with and without a frame passes the check and round-trips
    @test identical(parse_spec(record_json(sres; circuit=res)), sres)
    framed = Spec(res; frame=frames.optimum)
    @test identical(parse_spec(record_json(framed; circuit=res, frame=frames.optimum)), framed)
end

@testset "a record loads in a fresh Julia session" begin                     # acceptance 4
    text = record_json(solved(res, frames.optimum, θ); circuit=res, frame=frames.optimum, values=θ)
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
    text = record_json(ref; circuit=res2, frame=S, values=θ2)
    rec = load_record(text)
    @test rec.values == θ2 && rec.frame == S                            # base-frame values, as given
    @test identical(rec.spec, ref)                                      # the record's spec is the solved spec
    @test same_constants(Spec(QuantizedCircuit(rec.result; values=rec.values), rec.frame), ref)
    @test same_constants(Spec(circuit_quantization().load_record(text)), ref)   # Python's own reading
end

@testset "a record carries the zero tolerance of its run" begin               # Important 2
    iz = KronFreeCircuits.iszero_expr
    # The normal frame turned by 1e-10 rad: E_L[2,3] ≈ 6e-12 of max|E_L|, zeroed at zero_rtol = 1e-9 only.
    ε = 1e-10
    S = frames.normal * [1 0 0; 0 cos(ε) -sin(ε); 0 sin(ε) cos(ε)]
    for zr in (1e-12, 1e-9)
        l = Spec(qc, S; zero_rtol=zr)
        rec = load_record(record_json(l; circuit=res, frame=S, values=θ, zero_rtol=zr))
        @test identical(rec.spec, l)
        @test rec.zero_rtol == zr && rec.zero_atol == 0.0
        again = Spec(QuantizedCircuit(rec.result; values=rec.values), rec.frame; zero_rtol=rec.zero_rtol,
                     zero_atol=rec.zero_atol)                           # the record rebuilds the run
        @test iz.(again.E_C) == iz.(l.E_C) && iz.(again.E_L) == iz.(l.E_L)
    end
    @test !iz(Spec(qc, S).E_L[2, 3]) && iz(Spec(qc, S; zero_rtol=1e-9).E_L[2, 3])
    # A spec zeroed at a looser tolerance than the one recorded passes the check: the entry is below 1e-9 of E_L's
    # scale.
    rec = load_record(record_json(Spec(qc, S; zero_rtol=1e-9); circuit=res, frame=S, values=θ))
    @test rec.zero_rtol == 1e-12 && iz(rec.spec.E_L[2, 3])
    l = Spec(qc, S; zero_rtol=0, zero_atol=1e-10)
    rec = load_record(record_json(l; circuit=res, frame=S, values=θ, zero_rtol=0, zero_atol=1e-10))
    @test rec.zero_rtol == 0.0 && rec.zero_atol == 1e-10 && iz(rec.spec.E_L[2, 3])
    none = PythonCall.pybuiltins.None
    @test_throws "zero_rtol" record_json(l; circuit=none, zero_rtol=-1.0)            # before any Python call
    @test_throws "zero_atol" record_json(l; circuit=none, zero_atol=NaN)
    @test_throws "zero_rtol" save_record(joinpath(mktempdir(), "x.json"), l; circuit=none, zero_rtol=-1.0)
end

@testset "one validator for values" begin                                     # Important 3
    sympy = pyimport("sympy")
    r, θk, _ = kite(; cb_flux=sympy.Symbol("phi_cb"; real=true))   # a flux symbol the exported spec does not contain
    order = Spec(r).parameters
    # θk serves both; phi_cb is a parameter of neither
    @test collect(keys(QuantizedCircuit(r; values=θk).values)) == order
    spec = solved(r, nothing, θk)
    rec = load_record(record_json(spec; circuit=r, values=θk))
    @test collect(keys(rec.values)) == order && rec.values == NamedTuple{Tuple(order)}(θk)   # the circuit's order
    for f in (() -> QuantizedCircuit(r; values=merge(θk, (phi_cb=0.1,))),
              () -> record_json(spec; circuit=r, values=merge(θk, (phi_cb=0.1,))))
        e = error_of(f)
        @test e isa ArgumentError && occursin("phi_cb is not a parameter of the circuit", e.msg) &&
              all(occursin(string(k), e.msg) for k in order)
    end
    # the same message from both, for each kind of bad input
    d = Dict{Any,Any}(pairs(θk)); d["E_J"] = 1.0                         # one name given twice
    for bad in (d, [:E_J => 1.0], Base.structdiff(θk, (E_L=0,)), merge(θk, (E_J=true,)))
        a, b = error_of(() -> QuantizedCircuit(r; values=bad)), error_of(() -> record_json(spec; circuit=r, values=bad))
        @test a isa ArgumentError && b isa ArgumentError && a.msg == b.msg
    end
    # subs accepts only the symbols the export can contain
    e = error_of(() -> Spec(r; subs=(phi_cb=0.1,)))
    @test e isa ArgumentError && occursin("phi_cb is not a symbol", e.msg) &&
          !occursin("phi_cb", split(e.msg, "symbols are")[end])
end
