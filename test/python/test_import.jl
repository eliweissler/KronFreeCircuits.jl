@testset "module import" begin
    cq = circuit_quantization()
    @test pyconvert(Bool, pyhasattr(cq, "quantize"))
    @test circuit_quantization() === cq                         # cached
    msg = Base.get_extension(KronFreeCircuits, :KronFreeCircuitsPythonCallExt)._import_help("/some/python")
    @test occursin("/some/python", msg) && occursin("JULIA_PYTHONCALL_EXE", msg) && occursin("JULIA_CONDAPKG_BACKEND", msg)
end
