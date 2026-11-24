# Build the documentation site: julia --project=docs docs/make.jl
# The examples run during the build, with the working directory set to the circuit-quantization-generated specs in
# test/fixtures/generated, so a page writes `load_spec("transmon.json")` as a user would.
using Documenter, KronFreeCircuits

DocMeta.setdocmeta!(KronFreeCircuits, :DocTestSetup, :(using KronFreeCircuits); recursive=true)

makedocs(;
    sitename="KronFreeCircuits.jl",
    modules=[KronFreeCircuits],
    authors="Eli Weissler",
    workdir=joinpath(pkgdir(KronFreeCircuits), "test", "fixtures", "generated"),
    checkdocs=:exports,
    doctest=false,
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true", edit_link="main",
                           size_threshold=400 * 1024, size_threshold_warn=200 * 1024),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Manual" => [
            "Building circuits" => "manual/circuits.md",
            "Trees and truncation" => "manual/trees.md",
            "Solvers" => "manual/solvers.md",
            "Real arithmetic" => "manual/real_arithmetic.md",
            "Derivatives" => "manual/derivatives.md",
            "Wavefunctions" => "manual/wavefunctions.md",
            "Saving and loading" => "manual/saveload.md",
            "Time evolution with QuantumToolbox" => "manual/quantumtoolbox.md",
        ],
        "Tutorial: two coupled zero-pi qubits" => "tutorials/two_zero_pi.md",
        "Extending" => "extending.md",
        "Experimental" => "experimental.md",
        "API reference" => ["Exported" => "api/exported.md", "Public and experimental" => "api/public.md"],
        "Development" => "development.md",
    ],
)

deploydocs(; repo="github.com/eliweissler/KronFreeCircuits.jl.git", devbranch="main", push_preview=false)
