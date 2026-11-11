using Test, LinearAlgebra, KronFreeCircuits
using KronFreeCircuits: Coefficient, differentiate, phase_factor, iszero_expr, circuit_terms, constant_term

@testset "differentiate: every grammar element against finite differences" begin
    P = [:x, :y]
    cases = ["3.5", "y", "x", "x + y", "x - y", "-x", "x * y * x", "x / y", "y / x", "x^3", "x^(-2)", "x^2.5", "x^y", "y^x",
             "(x*y)^(x+1)", "1//2 * x", "sqrt(x)", "exp(x*y)", "log(x)", "cos(x)", "sin(2*x)", "tan(x/2)", "cosh(x)",
             "sinh(x)", "tanh(x)", "abs(x - 1)", "abs(x - 3)", "cis(x)", "cispi(x)", "conj(cis(x) * y)", "real(cis(x)*x)",
             "imag(cis(x)*x)", "abs(cis(x) * (x + 2))", "sign(y) * x", "sign(x - 3) * x", "exp(im * x) * pi + ℯ^x",
             "sqrt(cis(x))", "log(cis(x) + 2)", "x .* y .+ x .^ 2", "cos(π * x) * y", "x + y + x"]
    for src in cases, θ in [(x=0.7, y=1.3), (x=1.9, y=0.6), (x=1.2, y=2.4)], v in P
        c = Coefficient(src, P)
        d1 = differentiate(c, v)
        check_fd(d1(θ), fd_derivative(t -> c(t), θ, v; h=2e-3); rtol=1e-8, atol=1e-10)
        check_fd(differentiate(d1, v)(θ), fd_second(t -> c(t), θ, v; h=1e-2); rtol=1e-6, atol=1e-7)
        w = v === :x ? :y : :x
        check_fd(differentiate(d1, w)(θ), fd_mixed(t -> c(t), θ, v, w; hx=1e-2, hλ=1e-2); rtol=1e-6, atol=1e-7)
    end
end

@testset "differentiate: constants, folding, literals of any type, errors" begin
    c = Coefficient("x^2", [:x])
    @test differentiate(c, :x)((x=1.7,)) ≈ 3.4
    d2 = differentiate(differentiate(c, :x), :x)
    @test d2.ex isa Number && d2.ex == 2 && isempty(d2.deps)
    @test iszero_expr(differentiate(d2, :x))
    @test iszero_expr(differentiate(Coefficient("cos(y)", [:x, :y]), :x))             # no dependence: a structural zero
    # coefficients built by the package's algebra carry ComplexF64 literals and exp(±im*offset)
    E, φ = Coefficient("E_J", [:E_J, :phi]), Coefficient("phi + 0.3", [:E_J, :phi])
    k = (0.5 - 0.25im) * (E * phase_factor(φ, -1)) + Coefficient(2.0im)
    θ = (E_J=3.1, phi=0.9)
    @test differentiate(k, :E_J)(θ) ≈ (0.5 - 0.25im) * exp(-im * 1.2)
    @test differentiate(k, :phi)(θ) ≈ (0.5 - 0.25im) * 3.1 * (-im) * exp(-im * 1.2)
    @test differentiate(differentiate(k, :phi), :phi)(θ) ≈ -(0.5 - 0.25im) * 3.1 * exp(-im * 1.2)
    @test differentiate(k, :E_J).deps == [:phi]
    nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args[2:end]; init=0) : 1
    @test nodes(differentiate(differentiate(k, :phi), :phi).ex) <= 40                 # second derivatives stay small
    # sign of a possibly complex argument that depends on x
    @test_throws ArgumentError differentiate(Coefficient("sign(cis(x)) * x", [:x, :y]), :x)
    @test differentiate(Coefficient("sign(cis(y)) * x", [:x, :y]), :x)((x=1.0, y=0.2)) ≈ sign(cis(0.2))
end

@testset "differentiate: constant folding never changes a value" begin
    θ = (x=1.3, y=0.2)
    tiny = Coefficient("x / 1.0e-170", [:x, :y])                                   # 1.0e-170^2 underflows to 0.0
    @test iszero_expr(differentiate(tiny, :y))
    @test differentiate(tiny, :x)(θ) ≈ 1e170
    wrap = Coefficient("x / 4294967296", [:x, :y])                                 # 4294967296^2 wraps to 0 in Int64
    @test differentiate(wrap, :x)(θ) ≈ 1 / 4294967296
    @test iszero_expr(differentiate(wrap, :y))
    big_product = Coefficient("x * 4294967296 * 4294967296", [:x, :y])             # evaluates in Float64 as written
    @test big_product(θ) ≈ 1.3 * 2.0^64
    d = differentiate(big_product, :x)
    @test d(θ) ≈ 1.8446744073709552e19
    check_fd(d(θ), fd_derivative(t -> big_product(t), θ, :x; h=2e-3); rtol=1e-8, atol=1e-10)
    wide = Coefficient("x * 4294967296 * 4294967296 * y", [:x, :y])                # number factors are folded together, exactly
    @test differentiate(wide, :x)((x=0.3, y=0.5)) ≈ 9.223372036854776e18
    @test differentiate(wide, :y)((x=0.5, y=0.3)) ≈ 9.223372036854776e18
    @test differentiate(Coefficient("2.5 * x * 4.0", [:x, :y]), :x)(θ) === 10.0 + 0.0im
    wide_sum = Coefficient("4611686018427387904 * x + 4611686018427387904 * x + x * y", [:x, :y])  # 2^62 + 2^62 wraps in Int64
    @test differentiate(wide_sum, :x)((x=0.3, y=0.5)) ≈ 2.0^63 + 0.5
    big_power = Coefficient("x * 4294967296^2", [:x, :y])
    check_fd(differentiate(big_power, :x)(θ), fd_derivative(t -> big_power(t), θ, :x; h=2e-3); rtol=1e-8, atol=1e-10)
    @test_throws ArgumentError KronFreeCircuits._diff(Expr(:ref, :x, 1), :x)        # an unexpected node names itself
    @test_throws ArgumentError KronFreeCircuits._diff("x", :x)
end

@testset "differentiate: every coefficient of the JSON fixtures" begin
    JSON3 = KronFreeCircuits.JSON3
    files = [joinpath(dir, f) for (dir, _, fs) in walkdir(fixture("")) for f in fs if endswith(f, ".json")]
    ntested = 0
    failed = String[]                 # coefficients that cannot be evaluated at the test point: reported, never skipped silently
    for file in files
        doc = JSON3.read(read(file, String))
        doc isa JSON3.Object && get(doc, :format, nothing) == "kronfree-spec" || continue
        spec = load_spec(file)
        ps = spec.parameters
        isempty(ps) && continue
        θ = NamedTuple{Tuple(ps)}(Tuple(0.6 .+ 0.37 .* (1:length(ps)) ./ length(ps)))
        for c in vcat([t.coeff for t in circuit_terms(spec)], constant_term(spec)), x in c.deps
            try c(θ) catch; push!(failed, "$file: $(c.ex)"); continue end
            d1 = differentiate(c, x)
            # steps 4x below 1e-3 and 1e-2: the pole of 4*(E_CJ*E_CS)/(E_CJ-E_CS) is 0.06 from the test point
            check_fd(d1(θ), fd_derivative(t -> c(t), θ, x; h=2.5e-4); rtol=1e-7, atol=1e-9)
            check_fd(differentiate(d1, x)(θ), fd_second(t -> c(t), θ, x; h=2.5e-3); rtol=1e-5, atol=1e-6)
            ntested += 1
        end
    end
    @test ntested > 50
    @test isempty(failed)             # a failure here prints the files and expressions
end
