include("kite.jl")

# The numeric route in the frame S at θ, as Julia arrays: circuit-quantization's checked NumPy frame change.
# The reference the symbolic route is compared with; `Spec(QuantizedCircuit(res; values=θ), S)` takes its place
# once it exists.
function numeric_route(res::Py, θ::NamedTuple, S::AbstractMatrix{<:Real})
    h = circuit_quantization().numeric_hamiltonian(res, pyvalues(θ)).transformed(pylist(pylist.(eachrow(S))))
    return (types=Symbol.(pyconvert(Vector{String}, pylist(h.types))),
            E_C=pyconvert(Matrix{Float64}, h.E_C), E_L=pyconvert(Matrix{Float64}, h.E_L),
            flux_linear=pyconvert(Vector{Float64}, h.flux_linear), n_g=pyconvert(Vector{Float64}, h.n_g),
            n_g_sector=pyconvert(Vector{Float64}, h.n_g_sector),
            cosines=[(energy=pyconvert(Float64, E), omega=pyconvert(Vector{Float64}, w), offset=pyconvert(Float64, d))
                     for (E, w, d) in h.cosines])
end

# The symbolic spec `s` at θ equals the numeric route `n`, with the fresh compact sector labels `n_g{j}` of `s` given
# the numeric route's transformed sector values.
function agrees(s::Spec, θ::NamedTuple, n)
    labels = (Symbol("n_g$j") => n.n_g_sector[j] for (j, m) in enumerate(s.modes) if m.type === :c)
    v = merge(θ, NamedTuple(labels))
    ≈(x, y) = isapprox(x, y; rtol=1e-10, atol=1e-12)
    ev(c) = c(v)
    return [m.type for m in s.modes] == n.types &&
           all(ev.(s.E_C) .≈ n.E_C) && all(ev.(s.E_L) .≈ n.E_L) &&
           all(ev.(s.flux_linear) .≈ n.flux_linear) && all(ev.(s.n_g) .≈ n.n_g) &&
           length(s.cosines) == length(n.cosines) &&
           all(ev(c.energy) ≈ d.energy && all(c.omega .≈ d.omega) && ev(c.offset) ≈ d.offset
               for (c, d) in zip(s.cosines, n.cosines))
end

res, θ, frames = kite()

@testset "final route keeps parameters symbolic in a fixed frame" begin
    t = @elapsed s = Spec(res; frame=frames.optimum)
    @test !isempty(s.parameters)
    @test agrees(s, θ, numeric_route(res, θ, frames.optimum))
    @test t < 60                                              # a regression guard, not a target
    @test_throws ArgumentError Spec(res; frame=ones(2, 2))
end

@testset "final route equals the numeric route: $name frame" for name in keys(frames)
    S = frames[name]
    s = Spec(res; frame=S)
    @test sort(s.parameters) == sort(collect(keys(θ)))
    @test agrees(s, θ, numeric_route(res, θ, S))
end

@testset "a nonzero compact offset charge stays on the compact mode" begin
    θg = merge(θ, (n_g1=0.3,))
    s, n = Spec(res; frame=frames.optimum), numeric_route(res, θg, frames.optimum)
    @test agrees(s, θg, n)
    @test [m.type for m in s.modes] == [:c, :e, :e]
    @test n.n_g_sector == [0.3, 0.0, 0.0] && n.n_g ≈ [0.3, 0.0, 0.0]
    @test s.n_g[1](θg) == 0.3 && all(iszero(g(θg)) for g in s.n_g[2:3])
end

@testset "frame arguments" begin
    s = Spec(res; frame=frames.optimum)
    same(a, b) = a.parameters == b.parameters && all(x(θ) == y(θ) for (x, y) in zip(a.E_C, b.E_C))
    @test same(Spec(res; frame=[1 0 0; 0 1 0; 0 0 1]), Spec(res))        # an integer matrix; the identity frame
    @test same(Spec(res; frame=Matrix{Bool}(I, 3, 3)), Spec(res))
    @test same(Spec(res; frame=permutedims(frames.optimum)'), s)          # an adjoint
    @test same(Spec(res; frame=view([frames.optimum zeros(3)], :, 1:3)), s)  # a view
    @test Spec(res; frame=frames.optimum, subs=(n_g1=0,)).parameters == setdiff(s.parameters, [:n_g1])

    err(f) = try f(); nothing catch e; e end
    e = err(() -> Spec(res; frame=ones(2, 2)))
    @test e isa ArgumentError && occursin("3×3", e.msg) && occursin("2×2", e.msg)
    e = err(() -> Spec(res; frame=[1.0 0 0; 0 NaN 0; 0 0 1]))
    @test e isa ArgumentError && occursin("finite", e.msg)
    e = err(() -> Spec(res; frame=[1.5 0 0; 0 1 0; 0 0 1]))                 # a compact block entry off the integers
    @test e isa ArgumentError && occursin("integer", e.msg)
    e = err(() -> Spec(res; frame=[2 0 0; 0 1 0; 0 0 1]))                   # an integer compact block, not unimodular
    @test e isa ArgumentError && occursin("GL(N_c, Z)", e.msg)
    e = err(() -> Spec(res; frame=[1 0 0; 0.5 1 0; 0 0 1]))                 # an extended mode mixed into the compact one
    @test e isa ArgumentError && occursin("mixes", e.msg)
    e = err(() -> Spec(res; frame=[1 0 0; 0 1 1; 0 1 1]))                   # singular
    @test e isa ArgumentError && occursin("invertible", e.msg)
end
