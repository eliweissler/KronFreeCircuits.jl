# Throwaway benchmark: ways to apply a sum-of-products Hamiltonian to a state
# without building the Kronecker product. Compares
#   sparse   : explicit sparse Kronecker matrix (QuantumToolbox / SparseArrays)
#   blas     : reshape + loop of dense gemm (mode-k product via BLAS)
#   ka       : KernelAbstractions kernel, one work-item per output element
#   itensor  : ITensors.jl contraction
# on (A) 3 oscillator-like modes with dense cosine couplings, local dim d,
# and (B) a two-leaf composite with dense 300x300 leaf operators.
# Generic over array type: pass ArrayType=CuArray to run on GPU.

using LinearAlgebra, SparseArrays, Printf, Random
using KernelAbstractions
using KrylovKit
using QuantumToolbox
using ITensors

BLAS.set_num_threads(Threads.nthreads())
const T = ComplexF64

ARRAY = get(ENV, "BENCH_ARRAY", "Array")
if ARRAY == "CuArray"
    using CUDA
    const AT = CuArray
    device_sync() = CUDA.synchronize()
else
    const AT = Array
    device_sync() = nothing
end
to_dev(x) = AT(x)

# ---------------------------------------------------------------- physics
"Dense cosine-like local operator cos(alpha (a + a^dag)) on d levels."
function cos_op(d, alpha)
    a = diagm(1 => sqrt.(1:d-1))
    x = Matrix{T}(a + a')
    return Matrix{T}(Hermitian((exp(im*alpha*x) + exp(-im*alpha*x))/2))
end
num_op(d) = Matrix{T}(diagm(0 => collect(0:d-1)))

"A term is (coefficient, Dict mode=>dense matrix)."
struct Term
    c::T
    factors::Vector{Tuple{Int,Matrix{T}}}
end

function model_A(d; w=(1.0,1.1,1.2), g=0.2, alpha=0.5)
    dims = (d,d,d)
    n = num_op(d); x = cos_op(d, alpha)
    terms = Term[]
    for i in 1:3; push!(terms, Term(T(w[i]), [(i,n)])); end
    for (i,j) in ((1,2),(2,3),(1,3)); push!(terms, Term(T(g), [(i,x),(j,x)])); end
    return dims, terms
end

function model_B(k; g=0.05, seed=1)
    rng = MersenneTwister(seed)
    E1 = Matrix{T}(diagm(0 => sort(5 .* rand(rng, k))))
    E2 = Matrix{T}(diagm(0 => sort(5 .* rand(rng, k))))
    R = randn(rng, T, k, k); n1 = Matrix{T}(Hermitian(R + R'))/sqrt(k)
    R = randn(rng, T, k, k); n2 = Matrix{T}(Hermitian(R + R'))/sqrt(k)
    terms = [Term(one(T), [(1,E1)]), Term(one(T), [(2,E2)]), Term(T(g), [(1,n1),(2,n2)])]
    return (k,k), terms
end

# ---------------------------------------------------------------- sparse
function build_sparse(dims, terms)
    N = length(dims)
    H = spzeros(T, prod(dims), prod(dims))
    for t in terms
        ops = SparseMatrixCSC{T,Int}[sparse(one(T)*I, d, d) for d in dims]
        for (k,A) in t.factors; ops[k] = sparse(A); end
        # QuantumToolbox convention: tensor(A1, A2, ...) = kron(A1, A2, ...), first mode slowest.
        # We use first mode FASTEST (column-major reshape), so kron in reverse order.
        H += t.c * kron(reverse(ops)...)
    end
    return H
end

# ---------------------------------------------------------------- blas mode-k
"Y[:, i_k, :] = sum_j A[i_k, j] X[:, j, :] with X reshaped (left, d_k, right)."
function modek_blas!(Y, A, X, dims, k)
    left = prod(dims[1:k-1]); dk = dims[k]; right = prod(dims[k+1:end])
    Xr = reshape(X, left, dk, right); Yr = reshape(Y, left, dk, right)
    At = transpose(A)
    if left == 1
        mul!(reshape(Yr, dk, right), A, reshape(Xr, dk, right))
    elseif right == 1
        mul!(reshape(Yr, left, dk), reshape(Xr, left, dk), At)
    else
        for r in 1:right
            @views mul!(Yr[:, :, r], Xr[:, :, r], At)
        end
    end
    return Y
end

# ---------------------------------------------------------------- KA mode-k
@kernel function modek_kernel!(Y, @Const(X), @Const(A), nk, stride_k)
    t = @index(Global, Linear)
    t0 = t - 1
    ik = (t0 ÷ stride_k) % nk
    base = t0 - ik*stride_k
    s = zero(eltype(Y))
    @inbounds for jk in 0:nk-1
        s += A[ik+1, jk+1] * X[base + jk*stride_k + 1]
    end
    @inbounds Y[t] = s
end

function modek_ka!(Y, A, X, dims, k)
    stride_k = prod(dims[1:k-1])
    backend = get_backend(Y)
    kern = modek_kernel!(backend, 256)
    kern(Y, X, A, dims[k], stride_k; ndrange=length(Y))
    KernelAbstractions.synchronize(backend)
    return Y
end


# ---------------------------------------------------------------- permute + one gemm (what ITensor does)
function modek_permute!(Y, A, X, dims, k)
    N = length(dims)
    others = Tuple(i for i in 1:N if i != k)
    perm = (k, others...)
    Xp = permutedims(reshape(X, dims...), perm)
    Yp = reshape(A * reshape(Xp, dims[k], :), dims[k], dims[collect(others)]...)
    permutedims!(reshape(Y, dims...), Yp, invperm(perm))
    return Y
end

# ---------------------------------------------------------------- strided-batched gemm (GPU), loop fallback
function modek_batched!(Y, A, X, dims, k)
    left = prod(dims[1:k-1]); dk = dims[k]; right = prod(dims[k+1:end])
    Xr = reshape(X, left, dk, right); Yr = reshape(Y, left, dk, right)
    if left == 1
        mul!(reshape(Yr, dk, right), A, reshape(Xr, dk, right))
    elseif right == 1
        mul!(reshape(Yr, left, dk), reshape(Xr, left, dk), transpose(A))
    elseif AT !== Array
        # Y[:, :, r] = X[:, :, r] * A^T for every r, batch index broadcast on A
        CUDA.CUBLAS.gemm_strided_batched!('N', 'T', one(T), Xr, reshape(A, dk, dk, 1), zero(T), Yr)
    else
        for r in 1:right
            @views mul!(Yr[:, :, r], Xr[:, :, r], transpose(A))
        end
    end
    return Y
end

# ---------------------------------------------------------------- generic apply
struct KronFreeOp{F}
    dims::NTuple
    terms::Vector  # device matrices
    buf1::Any
    buf2::Any
    modek!::F
end
function KronFreeOp(dims, terms, modek!)
    dterms = [(t.c, [(k, to_dev(A)) for (k,A) in t.factors]) for t in terms]
    n = prod(dims)
    KronFreeOp(dims, dterms, to_dev(zeros(T, n)), to_dev(zeros(T, n)), modek!)
end
function LinearAlgebra.mul!(y, H::KronFreeOp, x)
    fill!(y, zero(T))
    for (c, factors) in H.terms
        src = x
        for (idx,(k,A)) in enumerate(factors)
            dst = isodd(idx) ? H.buf1 : H.buf2
            H.modek!(dst, A, src, H.dims, k)
            src = dst
        end
        axpy!(c, src, y)
    end
    return y
end
Base.:*(H::KronFreeOp, x) = mul!(similar(x), H, x)

# ---------------------------------------------------------------- ITensor
struct ITensorOp
    dims::NTuple
    inds::Vector{Index{Int}}
    terms::Vector
end
function ITensorOp(dims, terms)
    inds = [Index(d, "m$i") for (i,d) in enumerate(dims)]
    its = [(t.c, [(k, ITensor(to_dev(A), inds[k]', inds[k])) for (k,A) in t.factors]) for t in terms]
    ITensorOp(dims, inds, its)
end
function apply_it(H::ITensorOp, psi::ITensor)
    out = nothing
    for (c, factors) in H.terms
        phi = psi
        for (k, A) in factors
            phi = noprime(A * phi)
        end
        out = out === nothing ? c*phi : out + c*phi
    end
    return out
end

# ---------------------------------------------------------------- timing
function timeit(f; min_trials=5, min_time=0.5)
    f(); device_sync()
    ts = Float64[]
    t_start = time()
    while length(ts) < min_trials || time() - t_start < min_time
        t = @elapsed (f(); device_sync())
        push!(ts, t)
        length(ts) >= 50 && break
    end
    return minimum(ts)*1e3  # ms
end

function bench_model(name, dims, terms; sparse_max_nnz=3e7, do_eig=false, neig=10)
    n = prod(dims)
    println("\n=== $name  dims=$dims  n=$n  nterms=$(length(terms))")
    rng = MersenneTwister(0)
    x0 = randn(rng, T, n); x0 ./= norm(x0)
    x = to_dev(x0); y = similar(x)
    results = Dict{String,Float64}()
    ref = nothing

    # sparse
    est_nnz = sum(prod(k in first.(t.factors) ? dims[k]^2 : dims[k] for k in 1:length(dims)) for t in terms)
    if AT === Array && est_nnz <= sparse_max_nnz
        Hs = build_sparse(dims, terms)
        results["sparse"] = timeit(() -> mul!(y, Hs, x))
        ref = Hs * x
        @printf("  sparse   nnz=%.2e  %10.3f ms\n", nnz(Hs), results["sparse"])
    else
        @printf("  sparse   skipped (est nnz %.1e)\n", est_nnz)
    end

    # blas
    Hb = KronFreeOp(dims, terms, modek_blas!)
    results["blas"] = timeit(() -> mul!(y, Hb, x))
    yb = Array(Hb * x)
    ref === nothing && (ref = yb)
    @printf("  blas     err=%.1e  %10.3f ms\n", norm(yb - Array(ref)), results["blas"])

    # ka
    Hk = KronFreeOp(dims, terms, modek_ka!)
    results["ka"] = timeit(() -> mul!(y, Hk, x))
    yk = Array(Hk * x)
    @printf("  ka       err=%.1e  %10.3f ms\n", norm(yk - Array(ref)), results["ka"])


    for (label, fn) in (("permute", modek_permute!), ("batched", modek_batched!))
        try
            Hx = KronFreeOp(dims, terms, fn)
            results[label] = timeit(() -> mul!(y, Hx, x))
            yx = Array(Hx * x)
            @printf("  %-8s err=%.1e  %10.3f ms\n", label, norm(yx - Array(ref)), results[label])
        catch e
            println("  $label  FAILED: ", sprint(showerror, e)[1:min(end,200)])
        end
    end

    # itensor
    try
        Hi = ITensorOp(dims, terms)
        psi = ITensor(reshape(x, dims...), Hi.inds...)
        results["itensor"] = timeit(() -> apply_it(Hi, psi))
        yi = vec(Array(array(apply_it(Hi, psi), Hi.inds...)))
        @printf("  itensor  err=%.1e  %10.3f ms\n", norm(yi - Array(ref)), results["itensor"])
    catch e
        println("  itensor  FAILED: ", sprint(showerror, e)[1:min(end,300)])
    end

    if do_eig
        println("  -- KrylovKit eigsolve, lowest $neig, tol 1e-8")
        for (label, H) in (("blas", Hb), ("ka", Hk))
            t = @elapsed begin
                vals, vecs, info = KrylovKit.eigsolve(v -> H*v, x, neig, :SR; ishermitian=true, tol=1e-8, krylovdim=max(40, 3neig), maxiter=500)
            end
            @printf("     %-6s  %8.2f s  converged=%d  matvecs=%d  E0=%.6f\n", label, t, info.converged, info.numops, real(vals[1]))
        end
        if haskey(results, "sparse")
            Hs = build_sparse(dims, terms)
            t = @elapsed begin
                vals, vecs, info = KrylovKit.eigsolve(v -> Hs*v, x, neig, :SR; ishermitian=true, tol=1e-8, krylovdim=max(40, 3neig), maxiter=500)
            end
            @printf("     %-6s  %8.2f s  converged=%d  matvecs=%d  E0=%.6f\n", "sparse", t, info.converged, info.numops, real(vals[1]))
        end
    end
    return results
end

println("Julia $(VERSION), threads=$(Threads.nthreads()), BLAS threads=$(BLAS.get_num_threads()), array=$ARRAY")
AT === Array || println(CUDA.name(CUDA.device()))

ds = parse.(Int, split(get(ENV, "BENCH_DS", "20,40,60,80"), ","))
for d in ds
    dims, terms = model_A(d)
    bench_model("A: 3 modes cosine-coupled", dims, terms; do_eig=(d == ds[1] && get(ENV,"BENCH_EIG","1")=="1"))
end
for k in parse.(Int, split(get(ENV, "BENCH_KS", "100,300"), ","))
    dims, terms = model_B(k)
    bench_model("B: two leaves, dense $(k)x$(k) ports", dims, terms; do_eig=(k == 100 && get(ENV,"BENCH_EIG","1")=="1"))
end
println("\nDONE")
