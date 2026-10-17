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
    return _dense_axis!(Y, op.M, X, k)
end

# Y = M applied along axis k of X, with M already stored like Y.
function _dense_axis!(Y::AbstractArray, M::AbstractMatrix, X::AbstractArray, k::Int)
    isempty(Y) && return Y
    left, d = _axis_layout(X, k)
    right = length(X) ÷ (left * d)
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
    return _diagonal_axis!(Y, op.d, X, k)
end

function _diagonal_axis!(Y::AbstractArray, dvec::AbstractVector, X::AbstractArray, k::Int)
    Y .= reshape(dvec, ntuple(i -> i == k ? size(X, k) : 1, ndims(X))) .* X
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

# Kernel-ready host data of a BandedOp: weights as a (d_k × nbands) matrix, and the offsets.
function _banded_data(op::BandedOp{T}) where {T}
    Wh = Matrix{T}(undef, op.n, length(op.offsets))
    for b in eachindex(op.weights)
        Wh[:, b] .= op.weights[b]
    end
    return Wh, copy(op.offsets)
end

function apply_axis!(Y::AbstractArray, op::BandedOp{T}, X::AbstractArray, k::Int) where {T}
    _check_axis(Y, op, X, k)
    isempty(Y) && return Y
    # Kernel-ready weights and offsets, allocated like Y so they live on its device.
    Wh, offsh = _banded_data(op)
    W = copyto!(similar(Y, T, size(Wh)), Wh)
    offs = copyto!(similar(Y, Int, length(offsh)), offsh)
    return _banded_axis!(Y, W, offs, X, k)
end

# W and offs already stored like Y.
function _banded_axis!(Y::AbstractArray, W::AbstractMatrix, offs::AbstractVector, X::AbstractArray, k::Int)
    isempty(Y) && return Y
    stride, d = _axis_layout(X, k)
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

# The array data an application of `op` reads, stored as `on`: the matrix of a DenseOp, the
# diagonal of a DiagonalOp, (weights, offsets) of a BandedOp; a ShiftOp needs only scalars.
_factor_data(on, op::DenseOp) = adapt(on, op.M)
_factor_data(on, op::DiagonalOp) = adapt(on, op.d)
_factor_data(on, op::BandedOp) = adapt(on, _banded_data(op))
_factor_data(on, ::ShiftOp) = nothing

# apply_axis! reading the factor's data from `data` (see `_factor_data`) instead of from `op`.
function _apply_factor!(Y::AbstractArray, op::DenseOp, data, X::AbstractArray, k::Int)
    _check_axis(Y, op, X, k)
    return _dense_axis!(Y, data, X, k)
end
function _apply_factor!(Y::AbstractArray, op::DiagonalOp, data, X::AbstractArray, k::Int)
    _check_axis(Y, op, X, k)
    return _diagonal_axis!(Y, data, X, k)
end
function _apply_factor!(Y::AbstractArray, op::BandedOp, data, X::AbstractArray, k::Int)
    _check_axis(Y, op, X, k)
    return _banded_axis!(Y, data[1], data[2], X, k)
end
_apply_factor!(Y::AbstractArray, op::ShiftOp, ::Nothing, X::AbstractArray, k::Int) = apply_axis!(Y, op, X, k)

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

# A term kept in the contraction sum: its coefficient and, per factor, `(axis, op, data)` with
# `op` for dispatch and `data = _factor_data(on, op)` the stored array the application reads.
struct _OperatorTerm
    coeff::ComplexF64
    factors::Vector{Tuple{Int,LocalOp,Any}}
end

"""
    KronFreeOperator(dims, terms; constant=0.0, block_capacity=1, on=Array)

`H = Σ_t c_t ⊗_k A_{t,k} + constant * I` on the tensor-product space with axis sizes `dims`
(axis 1 fastest), applied without forming Kronecker products.
Terms whose factors are all `DiagonalOp` are fused with `constant` into one state-shaped vector
`H.diag`, applied by a single broadcast; the remaining terms are kept in `H.terms` as records
holding, per factor, `(axis, op, data)` with `data` the array the application reads.
Every array `H` owns (`diag`, factor data, scratch) is `adapt(on, …)` at construction.
Two work buffers reserve `block_capacity` columns at construction. Calls reuse their active
prefix, growing storage only when the requested block exceeds capacity. Reserve the solver's
maximum block width before iteration to avoid state-sized scratch reallocations.
`mul!(y, H, x)` writes into caller-owned output; `H * x` allocates its output on `on`.
The scratch belongs to `H`, so simultaneous applications require separate operators/workspaces.

`ishermitian(H)` returns `true` unconditionally: callers are responsible for building `terms`
that sum to a Hermitian operator.
"""
mutable struct KronFreeOperator{V<:AbstractVector,B<:AbstractArray}
    dims::Vector{Int}
    diag::V
    terms::Vector{_OperatorTerm}
    buf1::B
    buf2::B
    on::Any
end

function KronFreeOperator(dims::AbstractVector{<:Integer}, terms::AbstractVector{AxisTerm};
                          constant::Number=0.0, block_capacity::Integer=1, on=Array)
    dims = Vector{Int}(dims)
    all(>=(1), dims) || throw(ArgumentError("KronFreeOperator: dims must be positive, got $dims"))
    for term in terms, (a, op) in term.factors
        1 <= a <= length(dims) || throw(ArgumentError("KronFreeOperator: factor axis $a outside 1:$(length(dims))"))
        dimension(op) == dims[a] ||
            throw(DimensionMismatch("KronFreeOperator: factor on axis $a has dimension $(dimension(op)), expected $(dims[a])"))
    end
    block_capacity >= 1 || throw(ArgumentError("KronFreeOperator: block_capacity must be positive"))
    T = promote_type(fieldtype(AxisTerm, :coeff), typeof(constant))
    D = fill(T(constant), dims...)
    kept = _OperatorTerm[]
    for term in terms
        if all(f -> f[2] isa DiagonalOp, term.factors)
            _add_term_diagonal!(D, term.coeff, term.factors)
        else
            push!(kept, _OperatorTerm(term.coeff,
                                      Tuple{Int,LocalOp,Any}[(a, op, _factor_data(on, op)) for (a, op) in term.factors]))
        end
    end
    shape = (dims..., Int(block_capacity))
    return KronFreeOperator(dims, adapt(on, vec(D)), kept,
                            adapt(on, Array{T}(undef, shape)), adapt(on, Array{T}(undef, shape)), on)
end

Base.size(H::KronFreeOperator) = (n = prod(H.dims); (n, n))
Base.size(H::KronFreeOperator, i::Integer) = size(H)[i]
Base.eltype(H::KronFreeOperator) = eltype(H.diag)
LinearAlgebra.ishermitian(::KronFreeOperator) = true

function _ensure_buffers!(H::KronFreeOperator, shape::Tuple)
    if length(H.buf1) < prod(shape)
        H.buf1 = similar(H.buf1, shape)
        H.buf2 = similar(H.buf2, shape)
    end
    return nothing
end

# A contiguous prefix preserves physical-axis layout as the active block shrinks.
_active_buffer(buf::AbstractArray, shape::Tuple) =
    size(buf) == shape ? buf : reshape(view(vec(buf), 1:prod(shape)), shape)

# Y .+= coeff * (⊗ factors) X, chaining factor applications through the two buffers.
function _accumulate_term!(Y::AbstractArray, term::_OperatorTerm, X::AbstractArray, b1::AbstractArray, b2::AbstractArray)
    src = X
    dst, other = b1, b2
    for (a, op, data) in term.factors
        _apply_factor!(dst, op, data, src, a)
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
    b1, b2 = _active_buffer(H.buf1, shape), _active_buffer(H.buf2, shape)
    # Initialize with the fused diagonal on the concretely typed caller array. Writing
    # the dynamically ranked reshape can box every element for active-column views.
    y .= H.diag .* x
    for term in H.terms
        _accumulate_term!(Y, term, X, b1, b2)
    end
    return y
end

Base.:*(H::KronFreeOperator, x::AbstractVecOrMat) = mul!(similar(H.diag, eltype(H), size(x)), H, x)

# D .+= coeff * ⊗_k diagonal(op_k) over a term's factors (ones on axes it does not touch).
function _add_term_diagonal!(D::Array, coeff::Number, factors)
    N = ndims(D)
    acc = fill(coeff, ntuple(_ -> 1, N))
    for f in factors
        a, op = f[1], f[2]
        acc = acc .* reshape(diagonal(op), ntuple(i -> i == a ? size(D, a) : 1, N))
    end
    D .+= acc
    return D
end

"""
    diagonal(H::KronFreeOperator) -> Vector

`H.diag + Σ_t c_t ⊗_k diag(A_{t,k})` over the terms not fused into `H.diag` (ones on axes a
term does not touch), axis 1 fastest, returned on the host.
"""
function diagonal(H::KronFreeOperator)
    D = copyto!(Array{eltype(H)}(undef, H.dims...), H.diag)
    for term in H.terms
        _add_term_diagonal!(D, term.coeff, term.factors)
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
