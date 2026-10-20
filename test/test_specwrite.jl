@testset "spec writer" begin
    # every hand-written and generated fixture round-trips to the same functions
    jsons(dir) = filter(f -> endswith(f, ".json") && f != "scqubits_reference.json", readdir(joinpath(@__DIR__, "fixtures", dir)))
    for f in vcat(fixture.(jsons("")), fixture.("generated/" .* jsons("generated")))
        s = load_spec(f)
        @test spec_equal(parse_spec(spec_json(s)), s)
    end
    # hard expressions (Review Focus 4)
    for src in ("-2*E_J", "E_J^2/(-3.5)", "-(-(E_J))", "pi*E_J/2", "2*E_J", "1e-300*E_J", "1 // 3")
        s = parse_spec(minimal_spec_json(constant=src))      # one compact mode, parameter E_J
        @test !occursin(r"\d(?![eE][-+]?\d)[A-Za-z_]", expr_text(spec_json(s), "constant"))   # no implicit multiplication
        @test spec_equal(parse_spec(spec_json(s)), s)
    end
    @test_throws ArgumentError spec_json(load_spec(fixture("transmon.json")); metadata=Dict("comment" => 1))
    md = Dict{String,Any}("_generator" => Dict("package" => "KronFreeCircuits"))
    @test KronFreeCircuits.JSON3.read(spec_json(load_spec(fixture("transmon.json")); metadata=md))._generator.package == "KronFreeCircuits"
    # exact text, big integers, charge cosines, and the file round trip
    @test KronFreeCircuits.expr_string(:(-2 * E_J ^ 2 / pi)) == "(((-2) * (E_J ^ 2)) / pi)"
    big = parse_spec(minimal_spec_json(constant="-44173801000000000000000000000000*E_J"))
    @test expr_text(spec_json(big), "constant") == "((-44173801000000000000000000000000) * E_J)"
    @test spec_equal(parse_spec(spec_json(big)), big)
    cc = parse_spec("""{"format": "kronfree-spec", "version": 1, "parameters": ["E_J"], "modes": [{"name": "a", "type": "e"}],
                        "E_C": [["1"]], "E_L": [["1"]], "flux_linear": ["0"], "n_g": ["0"],
                        "cosines": [{"energy": "E_J", "omega": [0.5], "offset": "0"}],
                        "charge_cosines": [{"energy": "E_J", "kappa": [1], "offset": "pi/2"}]}""")
    @test spec_equal(parse_spec(spec_json(cc)), cc)
    @test !haskey(JSON3.read(spec_json(load_spec(fixture("transmon.json")))), :charge_cosines)
    mktempdir() do dir
        path = joinpath(dir, "spec.json")
        @test save_spec(path, cc; metadata=Dict("_note" => "x")) == path
        @test spec_equal(load_spec(path), cc)
    end
end
