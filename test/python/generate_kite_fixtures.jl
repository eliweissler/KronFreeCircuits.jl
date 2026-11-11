# Generator of the KITE fixtures under test/fixtures/kite/: the numeric specs of the three frames and the reference
# calculation's intermediate numbers of the mean-field loop (Task 8 of the mean-field augmentation).
# A generator, not a test file: test/python/runtests.jl does not include it.
#
# Run it with the interpreter that has circuit-quantization installed, from the repository root:
#     JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$CQ_PYTHON julia --project=test/python test/python/generate_kite_fixtures.jl
# The fixtures are committed and regenerated only on purpose (a change of circuit-quantization or of the settings below),
# never as part of a test run.
#
# It writes four files:
#   published.json, normal.json, optimum.json   the numeric spec `Spec(QuantizedCircuit(res; values=θ), S)` of each frame
#   meanfield_reference.json                    per frame, the reference's grid, its 33 mean-field runs and its wells
# The reference is the Python prototype in circuit-quantization's `product_state` module, run with the settings that
# produced Figure 8(b): `default_bases(h, charge_cutoff=16, max_points=120, extent_scale=1.5)`, no adaptive growth, and
# `_scf(h, bases, 1e-11, 300, _initial(h, start))` from 33 starts.
# The distinct solutions are the runs sorted by energy (stable), each kept when its ⟨φ⟩ or its cosine expectations `X`
# differ by more than 1e-3 from every solution kept before it; the wells are the distinct solutions below the lowest
# plus 0.5 GHz, at most four.
# The levels of a mode in a well are the three lowest eigenvalues of `_mode_hamiltonian` at the well's expectations.
# Each well also records those expectations, `ex` of `_scf`: ⟨n⟩ and ⟨φ⟩ of every mode, and for each cosine the
# expectation `X` of its factor exp(i omega[j] φ_j) on every mode j, as real and imaginary parts (1 where omega[j] = 0).
using KronFreeCircuits, PythonCall, LinearAlgebra
const JSON3 = KronFreeCircuits.JSON3
include("kite.jl")

const OUT = normpath(joinpath(@__DIR__, "..", "fixtures", "kite"))
const GRID = (charge_cutoff=16, max_points=120, extent_scale=1.5)
const TOL, MAXITER = 1e-11, 300
const DISTINCT, WINDOW, MAX_WELLS, NLEVELS = 1e-3, 0.5, 4, 3
# The compact mode (the first) at 0 or π, each extended mode at 0, π/2, π, -π/2 (the last mode varying fastest), then
# the delocalized start.
const EXT = (0.0, π / 2, Float64(π), -π / 2)
const STARTS = Any[[(a, b, c) for a in (0.0, Float64(π)) for b in EXT for c in EXT]; nothing]

rows(M) = [collect(M[i, :]) for i in 1:size(M, 1)]
pyrows(M) = pylist([pylist(r) for r in rows(M)])

# One frame: the reference's grid, its runs from every start, and its wells.
function reference(ps, h0, S)
    h = h0.transformed(pyrows(S))
    bases = ps.default_bases(h; GRID...)
    ops = ps._mode_operators(h, bases)
    N = pyconvert(Int, h.size)
    runs = map(STARTS) do start
        r = ps._scf(h, bases, TOL, MAXITER, ps._initial(h, start === nothing ? nothing : pytuple(start)))
        ex = r[2]
        (energy=pyconvert(Float64, r[0]), iterations=pyconvert(Int, r[3]), converged=pyconvert(Bool, r[4]),
         start=start === nothing ? nothing : collect(start), ex=ex,
         phi=pyconvert(Vector{Float64}, ex["phi"]), X=[pyconvert(Vector{ComplexF64}, x) for x in ex["X"]])
    end
    sorted = sort(runs; by=r -> r.energy, alg=MergeSort)
    far(a, b) = max(maximum(abs.(a.phi .- b.phi)), maximum(maximum(abs.(x .- y)) for (x, y) in zip(a.X, b.X))) > DISTINCT
    distinct = []
    for r in sorted
        all(o -> far(r, o), distinct) && push!(distinct, r)
    end
    wells = first(filter(r -> r.energy < distinct[1].energy + WINDOW, distinct), MAX_WELLS)
    eigh = pyimport("scipy.linalg").eigh
    levels(r) = [pyconvert(Vector{Float64}, eigh(ps._mode_hamiltonian(h, ops, r.ex, i - 1); eigvals_only=true,
                                                 subset_by_index=pytuple((0, NLEVELS - 1)))) for i in 1:N]
    extended = [b for b in bases if !pyconvert(Bool, b.compact)]
    return (frame=rows(S),
            grid=(charge_cutoff=GRID.charge_cutoff, extents=[pyconvert(Float64, b.extent) for b in extended],
                  points=[pyconvert(Int, b.points) for b in extended]),
            starts=length(runs), converged=count(r -> r.converged, runs), distinct=length(distinct),
            solutions=[(energy=r.energy, iterations=r.iterations, converged=r.converged, start=r.start, levels=levels(r),
                        phi=r.phi, expectations=(n=pyconvert(Vector{Float64}, r.ex["n"]), phi=r.phi,
                                                 X=[(re=real.(x), im=imag.(x)) for x in r.X])) for r in wells])
end

function main()
    mkpath(OUT)
    res, θ, frames = kite()
    cq = circuit_quantization()
    ps = pyimport("circuit_quantization.product_state")
    root = dirname(dirname(dirname(pyconvert(String, cq.__file__))))
    commit = readchomp(`git -C $root rev-parse HEAD`)
    qc = QuantizedCircuit(res; values=θ)
    h0 = cq.numeric_hamiltonian(res, pyvalues(θ))
    out = Pair{Symbol,Any}[:_generator => "test/python/generate_kite_fixtures.jl",
                           :_date => Libc.strftime("%Y-%m-%d", time()),
                           :_circuit_quantization => commit]
    for name in (:published, :normal, :optimum)
        S = frames[name]
        save_spec(joinpath(OUT, "$name.json"), Spec(qc, S))
        t = @elapsed r = reference(ps, h0, S)
        println("--- $name: grid $(r.grid.points) extents $(r.grid.extents), $(r.converged)/$(r.starts) converged, ",
                "$(r.distinct) distinct, $(length(r.solutions)) wells [$(round(t; digits=1)) s]")
        for s in r.solutions
            println("    E = ", round(s.energy; digits=12), "  iterations $(s.iterations)  converged $(s.converged)  ",
                    "start $(s.start)  ⟨φ⟩ = $(round.(s.phi; digits=6))")
        end
        push!(out, name => r)
    end
    open(joinpath(OUT, "meanfield_reference.json"), "w") do io
        JSON3.pretty(io, JSON3.write((; out...)))
        println(io)
    end
    println("wrote ", OUT)
end

main()
