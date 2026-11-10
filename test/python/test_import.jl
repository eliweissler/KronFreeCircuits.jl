const _EXT = Base.get_extension(KronFreeCircuits, :KronFreeCircuitsPythonCallExt)

@testset "module import" begin
    cq = circuit_quantization()
    @test pyconvert(Bool, pyhasattr(cq, "quantize"))
    @test circuit_quantization() === cq                         # cached
    msg = _EXT._import_help("/some/python")
    @test occursin("/some/python", msg) && occursin("JULIA_PYTHONCALL_EXE", msg) && occursin("JULIA_CONDAPKG_BACKEND", msg)
end

@testset "import failures" begin
    err(f) = try f(); nothing catch e; e end
    # the module itself is missing: the help on choosing the interpreter
    e = err(() -> _EXT._import_module("kronfree_no_such_module"))
    @test e isa ErrorException && occursin("kronfree_no_such_module", e.msg) && occursin("JULIA_PYTHONCALL_EXE", e.msg)
    # the module is found but fails to import: Python's own message, not the help
    dir = mktempdir()
    write(joinpath(dir, "kronfree_missing_dependency_module.py"), "import kronfree_no_such_dependency\n")
    write(joinpath(dir, "kronfree_failing_module.py"), "raise ValueError('failed while importing')\n")
    sys = pyimport("sys")
    sys.path.insert(0, dir)
    try
        e = err(() -> _EXT._import_module("kronfree_missing_dependency_module"))
        @test e isa ErrorException && occursin("ModuleNotFoundError", e.msg) &&
              occursin("kronfree_no_such_dependency", e.msg) && !occursin("JULIA_PYTHONCALL_EXE", e.msg)
        e = err(() -> _EXT._import_module("kronfree_failing_module"))
        @test e isa ErrorException && occursin("ValueError: failed while importing", e.msg) &&
              !occursin("JULIA_PYTHONCALL_EXE", e.msg)
    finally
        sys.path.remove(dir)
    end
end

@testset "circuit-quantization without the zero-tolerance keywords" begin       # Important 1
    builtins = pyimport("builtins")
    origin = "/old/site-packages/circuit_quantization/__init__.py"
    old = builtins.eval("lambda res, *, subs=None, simplify=True, mode_names=None: None", pydict())   # to_spec before c8c8714
    msg = _EXT._zero_keywords_missing(old, origin)
    @test msg isa String && occursin("c8c8714", msg) && occursin("to-spec-zero-tolerance", msg) && occursin(origin, msg)
    @test _EXT._zero_keywords_missing(builtins.eval("lambda res, *, zero_rtol=0: None", pydict()), origin) isa String
    @test _EXT._zero_keywords_missing(circuit_quantization().to_spec, origin) === nothing   # the installed one
    # the cached answer is what Spec(res) and record_json consult, before calling to_spec
    saved = _EXT._ZERO_KEYWORDS[]
    try
        _EXT._ZERO_KEYWORDS[] = msg
        @test_throws ArgumentError(msg) Spec(PythonCall.pybuiltins.None)
        spec = load_spec(joinpath(@__DIR__, "..", "fixtures", "transmon.json"))
        @test_throws ArgumentError(msg) record_json(spec; circuit=PythonCall.pybuiltins.None)
    finally
        _EXT._ZERO_KEYWORDS[] = saved
    end
end
