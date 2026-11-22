# A saved system is plain HDF5: read it with h5py, no KronFreeCircuits needed (docs/save-format.md).
using KronFreeCircuits: nport, expiphi

@testset "save_system file read with h5py" begin
    h5py = try
        pyimport("h5py")
    catch
        nothing
    end
    if h5py === nothing
        @test_skip "h5py is not installed in this Python; pip install h5py to run this test"
    else
        root = Node([:varphi_1 => PeriodicGrid(64; order=8)]; solver=Dense(), keep=lowest(4),
                    ports=[(:varphi_1, nport()), (:varphi_1, expiphi(1.0))])
        sys = materialize(root, load_spec(joinpath(@__DIR__, "..", "fixtures", "transmon.json")), (E_J=20.0, E_C=1.0, n_g=0.2))
        path = tempname() * ".jld2"
        save_system(path, sys)
        h5py.get_config().complex_names = ("re", "im")
        f = h5py.File(path, "r")
        try
            @test pyconvert(Int, f["format_version"][()]) == 1
            E = f["root/energies"][()]
            V = f["root/vectors"][()]
            @test pyconvert(Vector{Float64}, E) == sys.root.energies
            @test permutedims(pyconvert(Matrix{ComplexF64}, V)) == sys.root.vectors
        finally
            f.close()
        end
    end
end
