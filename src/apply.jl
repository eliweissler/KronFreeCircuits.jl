# Apply layer: one structured local operator along one axis of a state tensor, and a
# Hamiltonian composed as a sum of products of such applications.
#
# Dispatch is on operator structure only, never on device.  BandedOp and ShiftOp run as
# KernelAbstractions kernels on `get_backend(Y)`; DenseOp and DiagonalOp use `mul!` and
# broadcasting.

function _check_axis(Y::AbstractArray, op::LocalOp, X::AbstractArray, k::Int)
    size(Y) == size(X) || throw(DimensionMismatch("apply_axis!: size(Y) = $(size(Y)) differs from size(X) = $(size(X))"))
    1 <= k <= ndims(X) || throw(ArgumentError("apply_axis!: axis $k out of range for a $(ndims(X))-dimensional array"))
    size(X, k) == dimension(op) ||
        throw(DimensionMismatch("apply_axis!: size(X, $k) = $(size(X, k)) but dimension(op) = $(dimension(op))"))
    Base.mightalias(Y, X) && throw(ArgumentError("apply_axis!: Y must not alias X"))
    return nothing
end

# (stride_k, d_k): stride of axis k in the flat column-major layout, and its length.
_axis_layout(X::AbstractArray, k::Int) = (prod(size(X)[1:k-1]), size(X, k))

"""
    apply_axis!(Y, op::LocalOp, X, k) -> Y

Overwrite `Y` with `op` applied along axis `k` of `X`:
`Y[…, i, …] = Σ_j A[i, j] X[…, j, …]` with `A = dense(op)` acting on axis `k`.
`X` and `Y` have identical size, `size(X, k) == dimension(op)`, and may carry any number of
further axes (e.g. a trailing block axis).  `Y` must not alias `X`.
"""
function apply_axis! end

function apply_axis!(Y::AbstractArray, op::DenseOp, X::AbstractArray, k::Int)
    _check_axis(Y, op, X, k)
    isempty(Y) && return Y
    left, d = _axis_layout(X, k)
    right = length(X) ÷ (left * d)
    M = op.M
    if left == 1
        mul!(reshape(Y, d, right), M, reshape(X, d, right))
    elseif right == 1
        mul!(reshape(Y, left, d), reshape(X, left, d), transpose(M))
    else
        Xr = reshape(X, left, d, right)
        Yr = reshape(Y, left, d, right)
        for r in 1:right
            @views mul!(Yr[:, :, r], Xr[:, :, r], transpose(M))   # Y[l,i,r] = Σ_j M[i,j] X[l,j,r]
        end
    end
    return Y
end

function apply_axis!(Y::AbstractArray, op::DiagonalOp, X::AbstractArray, k::Int)
    _check_axis(Y, op, X, k)
    d = size(X, k)
    Y .= reshape(op.d, ntuple(i -> i == k ? d : 1, ndims(X))) .* X
    return Y
end

# Y[t] = Σ_b W[i_k+1, b] X[t with i_k -> i_k + off_b], all indices 0-based except array access.
@kernel function _banded_kernel!(Y, @Const(X), @Const(W), @Const(offsets), stride, d)
    t = @index(Global, Linear)
    t0 = t - 1
    i = (t0 ÷ stride) % d
    base = t0 - i * stride
    acc = zero(eltype(Y))
    for b in 1:length(offsets)
        j = i + offsets[b]
        if 0 <= j < d
            acc += W[i + 1, b] * X[base + j * stride + 1]
        end
    end
    Y[t] = acc
end

function apply_axis!(Y::AbstractArray, op::BandedOp{T}, X::AbstractArray, k::Int) where {T}
    _check_axis(Y, op, X, k)
    isempty(Y) && return Y
    stride, d = _axis_layout(X, k)
    nb = length(op.offsets)
    # Kernel-ready weights (d_k × nbands) and offsets, allocated like Y so they live on its device.
    Wh = Matrix{T}(undef, d, nb)
    for b in 1:nb
        Wh[:, b] .= op.weights[b]
    end
    W = copyto!(similar(Y, T, d, nb), Wh)
    offs = copyto!(similar(Y, Int, nb), op.offsets)
    backend = get_backend(Y)
    _banded_kernel!(backend, 256)(vec(Y), vec(X), W, offs, stride, d; ndrange=length(Y))
    KernelAbstractions.synchronize(backend)
    return Y
end

# Y[t] = factor * X[t with i_k -> i_k - s] when 0 <= i_k - s < d_k, else 0.
@kernel function _shift_kernel!(Y, @Const(X), factor, s, stride, d)
    t = @index(Global, Linear)
    t0 = t - 1
    i = (t0 ÷ stride) % d
    base = t0 - i * stride
    j = i - s
    Y[t] = (0 <= j < d) ? factor * X[base + j * stride + 1] : zero(eltype(Y))
end

function apply_axis!(Y::AbstractArray, op::ShiftOp, X::AbstractArray, k::Int)
    _check_axis(Y, op, X, k)
    isempty(Y) && return Y
    stride, d = _axis_layout(X, k)
    backend = get_backend(Y)
    _shift_kernel!(backend, 256)(vec(Y), vec(X), op.factor, op.s, stride, d; ndrange=length(Y))
    KernelAbstractions.synchronize(backend)
    return Y
end

"""
    AxisTerm(coeff, factors)

One product term `coeff * ⊗_k A_k`, with `factors = [(axis, op), ...]` listing the axes it
touches in strictly increasing order (at most one factor per axis); identity elsewhere.
"""
struct AxisTerm
    coeff::ComplexF64
    factors::Vector{Tuple{Int,LocalOp}}
    function AxisTerm(coeff::Number, factors::AbstractVector{<:Tuple{Int,LocalOp}})
        axes = first.(factors)
        all(diff(axes) .> 0) || throw(ArgumentError("AxisTerm: factor axes must be strictly increasing, got $axes"))
        return new(ComplexF64(coeff), Vector{Tuple{Int,LocalOp}}(factors))
    end
end

"""
    KronFreeOperator(dims, terms; constant=0.0)

`H = Σ_t c_t ⊗_k A_{t,k} + constant * I` on the tensor-product space with axis sizes `dims`
(axis 1 fastest), applied without forming Kronecker products.  Two work buffers of shape
`(dims..., M)` are reallocated whenever a call's block width `M` changes.

`ishermitian(H)` returns `true` unconditionally: callers are responsible for building `terms`
that sum to a Hermitian operator.
"""
mutable struct KronFreeOperator
    dims::Vector{Int}
    terms::Vector{AxisTerm}
    constant::ComplexF64
    buf1::Array{ComplexF64}
    buf2::Array{ComplexF64}
end

function KronFreeOperator(dims::AbstractVector{<:Integer}, terms::AbstractVector{AxisTerm}; constant::Number=0.0)
    dims = Vector{Int}(dims)
    all(>=(1), dims) || throw(ArgumentError("KronFreeOperator: dims must be positive, got $dims"))
    for term in terms, (a, op) in term.factors
        1 <= a <= length(dims) || throw(ArgumentError("KronFreeOperator: factor axis $a outside 1:$(length(dims))"))
        dimension(op) == dims[a] ||
            throw(DimensionMismatch("KronFreeOperator: factor on axis $a has dimension $(dimension(op)), expected $(dims[a])"))
    end
    shape = (dims..., 1)
    return KronFreeOperator(dims, Vector{AxisTerm}(terms), ComplexF64(constant),
                            Array{ComplexF64}(undef, shape), Array{ComplexF64}(undef, shape))
end

Base.size(H::KronFreeOperator) = (n = prod(H.dims); (n, n))
Base.size(H::KronFreeOperator, i::Integer) = size(H)[i]
Base.eltype(::KronFreeOperator) = ComplexF64
LinearAlgebra.ishermitian(::KronFreeOperator) = true

function _ensure_buffers!(H::KronFreeOperator, shape::Tuple)
    if size(H.buf1) != shape
        H.buf1 = Array{ComplexF64}(undef, shape)
        H.buf2 = Array{ComplexF64}(undef, shape)
    end
    return nothing
end

# Y .+= coeff * (⊗ factors) X, chaining factor applications through the two buffers.
function _accumulate_term!(Y::AbstractArray, term::AxisTerm, X::AbstractArray, b1::AbstractArray, b2::AbstractArray)
    src = X
    dst, other = b1, b2
    for (a, op) in term.factors
        apply_axis!(dst, op, src, a)
        src = dst
        dst, other = other, dst
    end
    Y .+= term.coeff .* src
    return Y
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, H::KronFreeOperator, x::AbstractVecOrMat)
    n = prod(H.dims)
    size(x, 1) == n || throw(DimensionMismatch("KronFreeOperator of size $(size(H)) applied to x with $(size(x, 1)) rows"))
    size(y) == size(x) || throw(DimensionMismatch("mul!: size(y) = $(size(y)) differs from size(x) = $(size(x))"))
    Base.mightalias(y, x) && throw(ArgumentError("mul!: y must not alias x"))
    shape = (H.dims..., size(x, 2))
    _ensure_buffers!(H, shape)
    X = reshape(x, shape)
    Y = reshape(y, shape)
    fill!(Y, zero(eltype(Y)))
    for term in H.terms
        _accumulate_term!(Y, term, X, H.buf1, H.buf2)
    end
    y .+= H.constant .* x
    return y
end

Base.:*(H::KronFreeOperator, x::AbstractVecOrMat) = mul!(similar(x, ComplexF64), H, x)

"""
    diagonal(H::KronFreeOperator) -> Vector{ComplexF64}

`constant + Σ_t c_t ⊗_k diag(A_{t,k})` (ones on axes a term does not touch), axis 1 fastest.
"""
function diagonal(H::KronFreeOperator)
    N = length(H.dims)
    D = fill(H.constant, H.dims...)
    for term in H.terms
        acc = fill(term.coeff, ntuple(_ -> 1, N))
        for (a, op) in term.factors
            acc = acc .* reshape(diagonal(op), ntuple(i -> i == a ? H.dims[a] : 1, N))
        end
        D .+= acc
    end
    return vec(D)
end

const _MATRIX_BLOCK = 256

"""
    Matrix(H::KronFreeOperator) -> Matrix{ComplexF64}

The explicit `n × n` matrix, built by applying `H` to blocks of at most 256 identity columns,
so the peak memory beyond the result is a few `n × 256` blocks.
"""
function Base.Matrix(H::KronFreeOperator)
    n = prod(H.dims)
    M = Matrix{ComplexF64}(undef, n, n)
    B = min(n, _MATRIX_BLOCK)
    E = zeros(ComplexF64, n, B)          # identity block, reused
    Y = Matrix{ComplexF64}(undef, n, B)
    for c0 in 1:B:n
        w = min(B, n - c0 + 1)
        if w < B                         # last, narrower block
            E = zeros(ComplexF64, n, w)
            Y = Matrix{ComplexF64}(undef, n, w)
        end
        for j in 1:w
            E[c0 + j - 1, j] = 1
        end
        mul!(Y, H, E)
        copyto!(view(M, :, c0:c0+w-1), Y)
        for j in 1:w
            E[c0 + j - 1, j] = 0
        end
    end
    return M
end
