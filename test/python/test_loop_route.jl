# The loop route: a QuantizedCircuit made once at fixed values, and Spec(qc, S) from circuit-quantization's NumPy
# frame change, compared with the symbolic route and with the arrays Python returns.
include("kite.jl")
include("circuits.jl")

evalc(c) = c((;))                                              # a constant coefficient's value
≊(x, y) = isapprox(x, y; rtol=1e-10, atol=1e-12)

# The loop-route spec `a` (no parameters) equals the spec `b` evaluated at `v`: mode names and types, every coefficient.
function same_spec(a::Spec, b::Spec, v::NamedTuple)
    ev(c) = c(v)
    return isempty(a.parameters) && a.modes == b.modes &&
           all(evalc.(a.E_C) .≊ ev.(b.E_C)) && all(evalc.(a.E_L) .≊ ev.(b.E_L)) &&
           all(evalc.(a.flux_linear) .≊ ev.(b.flux_linear)) && all(evalc.(a.n_g) .≊ ev.(b.n_g)) &&
           length(a.cosines) == length(b.cosines) &&
           all(evalc(c.energy) ≊ ev(d.energy) && all(c.omega .≊ d.omega) && evalc(c.offset) ≊ ev(d.offset)
               for (c, d) in zip(a.cosines, b.cosines)) &&
           evalc(a.constant) ≊ ev(b.constant)
end

# The values of the symbolic route's fresh compact sector labels `n_g{j}` in the frame S: S_ccᵀ applied to the base
# sector values in θ (the KITE's compact modes keep their positions).
function sector_labels(θ::NamedTuple, S::AbstractMatrix, types)
    c = findall(==(:c), types)
    labels = Symbol.("n_g" .* string.(c))
    return NamedTuple{Tuple(labels)}(Tuple(S[c, c]' * [Float64(θ[k]) for k in labels]))
end

# The arrays circuit-quantization's NumPy frame change returns, converted to Julia arrays.
function python_arrays(res::Py, θ::NamedTuple, S::AbstractMatrix{<:Real})
    h = circuit_quantization().numeric_hamiltonian(res, pyvalues(θ)).transformed(pylist(pylist.(eachrow(S))))
    return (types=Symbol.(pyconvert(Vector{String}, pylist(h.types))),
            E_C=pyconvert(Matrix{Float64}, h.E_C), E_L=pyconvert(Matrix{Float64}, h.E_L),
            flux_linear=pyconvert(Vector{Float64}, h.flux_linear), n_g=pyconvert(Vector{Float64}, h.n_g),
            n_g_sector=pyconvert(Vector{Float64}, h.n_g_sector), constant=pyconvert(Float64, h.constant),
            cosines=[(energy=pyconvert(Float64, E), omega=pyconvert(Vector{Float64}, w), offset=pyconvert(Float64, d))
                     for (E, w, d) in h.cosines])
end

# A frame as the NumPy array circuit-quantization's numeric frame change takes.
_np_frame(S::AbstractMatrix) = pyimport("numpy").array(Matrix{Float64}(S))

# A static-array-like matrix: its size is in its type and it is not a Matrix.
struct Fixed3 <: AbstractMatrix{Float64}
    data::NTuple{9,Float64}
end
Fixed3(M::AbstractMatrix) = Fixed3(Tuple(Float64.(vec(M))))
Base.size(::Fixed3) = (3, 3)
Base.getindex(M::Fixed3, i::Int, j::Int) = M.data[i + 3(j - 1)]

res, θ, frames = kite()
qc = QuantizedCircuit(res; values=θ)
base_types = [m.type for m in Spec(qc).modes]
# The optimum frame with the compact mode's sign flipped (φ′₁ → -φ′₁): S_cc = -1, still admissible.
flipped = copy(frames.optimum); flipped[:, 1] .*= -1

@testset "QuantizedCircuit" begin
    @test qc isa QuantizedCircuit
    @test qc.values isa NamedTuple && all(v isa Float64 for v in qc.values)
    @test Set(keys(qc.values)) == Set(keys(θ)) && all(qc.values[k] == θ[k] for k in keys(θ))
    @test collect(keys(qc.values)) == Spec(res).parameters                 # the circuit's parameter order
    shuffled = NamedTuple{reverse(keys(θ))}(reverse(values(θ)))
    @test keys(QuantizedCircuit(res; values=shuffled).values) == keys(qc.values)
    d = Dict{Any,Any}(string(k) => v for (k, v) in pairs(θ)); d[:E_J] = 16.83; delete!(d, "E_J")
    @test QuantizedCircuit(res; values=d).values == qc.values                # strings and symbols as keys
    @test QuantizedCircuit(res; values=Dict(pairs(θ))).values == qc.values
end

@testset "loop route equals the symbolic route: $name frame" for (name, S) in
        (pairs(frames)..., :flipped => flipped)
    a = Spec(qc, S)
    s = Spec(res; frame=S)
    v = merge(θ, sector_labels(θ, S, base_types))
    @test same_spec(a, substitute(s, v), (;))
    @test same_spec(a, s, v)
    @test [m.name for m in a.modes] == [m.name for m in s.modes]
end

@testset "loop route equals Python's arrays: $name frame" for (name, S) in (pairs(frames)..., :flipped => flipped)
    θg = merge(θ, (n_g1=0.3,))
    a, n = Spec(QuantizedCircuit(res; values=θg), S), python_arrays(res, θg, S)
    @test [m.type for m in a.modes] == n.types
    @test evalc.(a.E_C) ≈ n.E_C && evalc.(a.E_L) ≈ n.E_L                # round-off symmetrization only
    @test evalc.(a.flux_linear) == n.flux_linear && evalc.(a.n_g) == n.n_g
    @test length(a.cosines) == length(n.cosines)
    @test all(evalc(c.energy) == d.energy && c.omega ≈ d.omega && evalc(c.offset) == d.offset
              for (c, d) in zip(a.cosines, n.cosines))
    @test evalc(a.constant) == n.constant
end

@testset "base frame" begin
    a, b = Spec(qc), Spec(qc, frames.published)
    @test a.modes == b.modes && evalc.(a.E_C) == evalc.(b.E_C) && evalc.(a.E_L) == evalc.(b.E_L)
    @test same_spec(a, Spec(res), θ)
    @test [m.name for m in a.modes] == [:varphi_1, :varphi_2, :varphi_3]
end

@testset "offset charges follow the sector convention" begin    # acceptance 8
    θg = merge(θ, (n_g1=0.3,))
    qg = QuantizedCircuit(res; values=θg)
    for S in (frames.optimum, frames.normal, flipped)
        a = Spec(qg, S)
        @test all(iszero(evalc(a.n_g[j])) for j in eachindex(a.modes) if a.modes[j].type != :c)
    end
    a = Spec(qg, frames.optimum)
    @test evalc(a.n_g[1]) ≊ 0.3
    a = Spec(qg, flipped)                                        # S_cc = -1 flips the compact sector offset
    @test [m.type for m in a.modes] == [:c, :e, :e]
    @test evalc(a.n_g[1]) ≊ -0.3
    @test all(iszero(evalc(a.n_g[j])) for j in 2:3)
    s = Spec(res; frame=flipped)
    @test same_spec(a, s, merge(θg, (n_g1=-0.3,)))               # the fresh label at the transformed value
    @test !same_spec(a, s, θg)                                   # and not at the base value
end

@testset "S inputs" begin                                        # acceptance 3, Review Focus 3
    I3 = [1.0 0 0; 0 1 0; 0 0 1]
    @test Spec(qc, I3).E_C[1, 1]((;)) == Spec(qc).E_C[1, 1]((;))
    @test Spec(qc, Int[1 0 0; 0 1 0; 0 0 1]) isa Spec
    @test Spec(qc, Matrix{Bool}(I, 3, 3)) isa Spec
    a = Spec(qc, frames.optimum)
    for S in (permutedims(frames.optimum)', view([frames.optimum zeros(3)], :, 1:3), Fixed3(frames.optimum))
        @test same_spec(Spec(qc, S), a, (;))
    end
    err(f) = try f(); nothing catch e; e end
    e = err(() -> Spec(qc, ones(2, 2)))
    @test e isa ArgumentError && occursin("3×3", e.msg) && occursin("2×2", e.msg)
    e = err(() -> Spec(qc, [NaN 0 0; 0 1 0; 0 0 1]))
    @test e isa ArgumentError && occursin("finite", e.msg)
    bad = copy(frames.optimum); bad[2, 1] = 0.1                  # a noncompact row feeding the compact column
    e = err(() -> Spec(qc, bad))
    @test e isa ArgumentError && occursin("S1[1, 0]", e.msg) && occursin("mixes", e.msg)
    @test occursin("count from 0", e.msg)                        # Python's indices, said to be Python's
    half = copy(I3); half[1, 1] = 0.5
    e = err(() -> Spec(qc, half))                                # compact block not an integer
    @test e isa ArgumentError && occursin("integer", e.msg)
    e = err(() -> Spec(qc, [2 0 0; 0 1 0; 0 0 1]))               # an integer compact block, not unimodular
    @test e isa ArgumentError && occursin("GL(N_c, Z)", e.msg)
    e = err(() -> Spec(qc, [1 0 0; 0 1 1; 0 1 1]))               # singular
    @test e isa ArgumentError && occursin("invertible", e.msg)
end

@testset "size and finiteness are checked before any Python call" begin
    # A handle that is not a numeric Hamiltonian: any Python call on it would fail with a Python exception.
    broken = QuantizedCircuit(PythonCall.pybuiltins.None, qc.values, qc.mode_names)
    @test_throws "3×3" Spec(broken, ones(2, 2))
    @test_throws "finite" Spec(broken, [Inf 0 0; 0 1 0; 0 0 1])
end

@testset "parameter names" begin                                 # Review Focus 2
    err(f) = try f(); nothing catch e; e end
    listed(msg) = all(occursin(string(k), msg) for k in keys(θ))
    e = err(() -> QuantizedCircuit(res; values=merge(θ, (E_JJ=1.0,))))
    @test e isa ArgumentError && occursin("E_JJ", e.msg) && listed(e.msg)
    e = err(() -> QuantizedCircuit(res; values=Base.structdiff(θ, (E_L=0,))))
    @test e isa ArgumentError && occursin("E_L", e.msg) && listed(e.msg)
    e = err(() -> QuantizedCircuit(res; values=Dict("E_J" => 1.0)))
    @test e isa ArgumentError && occursin("E_CJ", e.msg) && listed(e.msg)
    e = err(() -> QuantizedCircuit(res; values=merge(θ, (E_J=NaN,))))
    @test e isa ArgumentError && occursin("E_J", e.msg)
    e = err(() -> QuantizedCircuit(res; values=merge(θ, (E_J=1.0 + 2im,))))
    @test e isa ArgumentError && occursin("E_J", e.msg)
    e = err(() -> QuantizedCircuit(res; values=merge(θ, (E_J=true,))))
    @test e isa ArgumentError && occursin("E_J", e.msg)
    d = Dict{Any,Any}(pairs(θ)); d["E_J"] = 1.0                    # one name given twice
    e = err(() -> QuantizedCircuit(res; values=d))
    @test e isa ArgumentError && occursin("E_J", e.msg)
    e = err(() -> QuantizedCircuit(res; values=Dict(1 => 1.0)))
    @test e isa ArgumentError
end

@testset "round-off zeroing reaches numeric_spec" begin
    iz = KronFreeCircuits.iszero_expr
    # The normal frame turned by 1e-10 rad in its noncompact plane: E_L[2,3] ≈ 4e-11 GHz, about 6e-12 of max|E_L|.
    ε = 1e-10
    S = frames.normal * [1 0 0; 0 cos(ε) -sin(ε); 0 sin(ε) cos(ε)]
    a = Spec(qc, S)                                              # the default zero_rtol = 1e-12 keeps it
    @test !iz(a.E_L[2, 3]) && !iz(a.E_L[3, 2]) && 1e-12 < abs(evalc(a.E_L[2, 3])) / 6.42 < 1e-11
    b = Spec(qc, S; zero_rtol=1e-9)
    @test iz(b.E_L[2, 3]) && iz(b.E_L[3, 2]) && evalc(b.E_L[2, 2]) == evalc(a.E_L[2, 2])
    @test iz(Spec(qc, S; zero_rtol=0, zero_atol=1e-10).E_L[2, 3])
    @test !iz(Spec(qc, frames.normal; zero_rtol=0, zero_atol=0).E_C[1, 2])  # zeroing off: the normal frame's round-off
    @test iz(Spec(qc, frames.normal).E_C[1, 2])                              # survives, and the default zeroes it
    # the base-frame method forwards the tolerances too: a QuantizedCircuit whose base frame is the normal frame
    qn = QuantizedCircuit(qc.handle.transformed(_np_frame(frames.normal)), qc.values, qc.mode_names)
    @test iz(Spec(qn).E_C[1, 2]) && !iz(Spec(qn; zero_rtol=0).E_C[1, 2])
    # the loop route and the symbolic route zero the same entries, at either tolerance
    for zr in (1e-12, 1e-9)
        l, s = Spec(qc, S; zero_rtol=zr), Spec(res; frame=S, subs=θ, zero_rtol=zr)
        @test iz.(l.E_C) == iz.(s.E_C) && iz.(l.E_L) == iz.(s.E_L) && same_spec(l, s, (;))
    end
    # a bad tolerance names the keyword, before any Python call
    broken = QuantizedCircuit(PythonCall.pybuiltins.None, qc.values, qc.mode_names)
    @test_throws "zero_rtol" Spec(broken, S; zero_rtol=-1.0)
    @test_throws "zero_atol" Spec(broken, S; zero_atol=NaN)
    @test_throws "zero_atol" Spec(broken; zero_atol=Inf)
end

@testset "two compact modes: the offset-charge convention with S_cc ≠ ±1" begin             # Important 4
    res2, _ = fixture_result("two_transmons")
    # element values for E_C1 = 0.3, E_C2 = 0.25 and g = 0.08 GHz (see the fixture's comment), nonzero sector offsets
    θ2 = (x_1=0.0749 / 0.24, x_2=0.0749 / 0.29, x_c=8 * 0.0749 / 0.08, n_g1=0.3, n_g2=0.1, E_J1=12.0, E_J2=10.0)
    q2 = QuantizedCircuit(res2; values=θ2)
    S = [1.0 1.0; 0.0 1.0]                                       # S_cc = S: Sᵀ, S⁻¹, S and S⁻ᵀ all differ
    base, a = Spec(q2), Spec(q2, S)
    @test [m.type for m in a.modes] == [:c, :c]
    n0, n1 = evalc.(base.n_g), evalc.(a.n_g)
    @test n0 ≈ [0.3, 0.1] && n1 ≈ S' * n0                        # the loop route: Sᵀ n_g = (0.3, 0.4)
    @test !(n1 ≈ S \ n0) && !(n1 ≈ S * n0) && !(n1 ≈ S' \ n0)
    s = Spec(res2; frame=S)                                      # the symbolic route, fresh labels n_g1, n_g2
    @test same_spec(a, s, merge(θ2, sector_labels(θ2, S, [m.type for m in base.modes])))
    @test !same_spec(a, s, θ2)                                   # and not at the base labels' values
    # the spectrum, independent of both routes' bookkeeping: the six lowest levels of the framed spec equal the base
    # frame's on a full charge basis of 31 states per mode (they agree to 5e-14 GHz at this cutoff, 2e-11 at 12)
    tree = Node([:varphi_1 => ChargeBasis(15), :varphi_2 => ChargeBasis(15)]; solver=Dense(), keep=lowest(6))
    E0, E1 = energies(materialize(tree, base, (;))), energies(materialize(tree, a, (;)))
    @test maximum(abs.(E1 - E0)) < 1e-11
end
