@testset "kron_reference" begin
    A = ComplexF64[1 2; 3 4]; B = ComplexF64[0 1; 1 0]
    H = kron_reference([2, 2], [(1.0 + 0im, [(1, A)]), (2.0 + 0im, [(2, B)])])
    @test H ≈ kron(I(2), A) + 2 * kron(B, I(2))   # axis 1 fastest
end

@testset "circuit-quantization interface without PythonCall" begin
    @test_throws "load PythonCall" circuit_quantization()
end

# The core may name PythonCall in prose (the hint and the docstrings of src/pyinterface.jl), but no code under src/ may
# load or call it: only the extension does.
const PYTHON_CODE_USE = r"using\s+PythonCall|import\s+PythonCall|PythonCall\.|pyimport"

@testset "the core never mentions Python" begin
    src = joinpath(pkgdir(KronFreeCircuits), "src")
    files = [joinpath(root, f) for (root, _, fs) in walkdir(src) for f in fs if endswith(f, ".jl")]
    @test !isempty(files)
    for f in files
        m = match(PYTHON_CODE_USE, read(f, String))
        name = relpath(f, src)
        @test (name => (m === nothing ? nothing : m.match)) == (name => nothing)   # a failure names the file and the match
    end
    @test Base.get_extension(KronFreeCircuits, :KronFreeCircuitsPythonCallExt) === nothing
end
