# Apply layer: one structured local operator along one axis of a state tensor, and a
# Hamiltonian composed as a sum of products of such applications.
#
# Dispatch is on operator structure only, never on device.  BandedOp and ShiftOp run as
# KernelAbstractions kernels on `get_backend(Y)`; DiagonalOp uses broadcasting.  A DenseOp
# uses `mul!` in the public `apply_axis!` (the reference) and an ITensor contraction inside
# `KronFreeOperator` (`_apply_dense!`).

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

# The array data an application of `op` reads, stored as `on`: the matrix of a DenseOp converted to the
# operator's element type `T` (so a gemm never mixes real and complex), the diagonal of a DiagonalOp,
# (weights, offsets) of a BandedOp; a ShiftOp needs only scalars.
_factor_data(on, op::DenseOp, ::Type{T}) where {T} = adapt(on, Matrix{T}(op.M))
_factor_data(on, op::DiagonalOp, ::Type) = adapt(on, op.d)
_factor_data(on, op::BandedOp, ::Type) = adapt(on, _banded_data(op))
_factor_data(on, ::ShiftOp, ::Type) = nothing

# apply_axis! reading the factor's data from `data` (see `_factor_data`) instead of from `op`.
# DenseOp factors inside an operator go through `_apply_dense!` instead.
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
The coefficient keeps its own type (`AxisTerm{typeof(coeff)}`).
"""
struct AxisTerm{T<:Number}
    coeff::T
    factors::Vector{Tuple{Int,LocalOp}}
    function AxisTerm{T}(coeff::T, factors::AbstractVector{<:Tuple{Int,LocalOp}}) where {T<:Number}
        axes = first.(factors)
        all(diff(axes) .> 0) || throw(ArgumentError("AxisTerm: factor axes must be strictly increasing, got $axes"))
        return new{T}(coeff, Vector{Tuple{Int,LocalOp}}(factors))
    end
end
AxisTerm(coeff::T, factors::AbstractVector{<:Tuple{Int,LocalOp}}) where {T<:Number} = AxisTerm{T}(coeff, factors)

# A term kept in the contraction sum: its coefficient and, per factor, `(axis, op, data)` with
# `op` for dispatch and `data = _factor_data(on, op)` the stored array the application reads.
struct _OperatorTerm{T<:Number}
    coeff::T
    factors::Vector{Tuple{Int,LocalOp,Any}}
end

"""
    KronFreeOperator(dims, terms; constant=0.0, block_capacity=1, on=Array)

`H = Σ_t c_t ⊗_k A_{t,k} + constant * I` on the tensor-product space with axis sizes `dims`
(axis 1 fastest), applied without forming Kronecker products.
Terms whose factors are all `DiagonalOp` are fused with `constant` into one state-shaped vector
`H.diag`, applied by a single broadcast; the remaining terms are kept in `H.terms` as records
holding, per factor, `(axis, op, data)` with `data` the array the application reads.
The fused terms are kept in `H.diagterms` (the constant as a zero-factor term) for introspection;
the application reads only `H.diag`.
Derived quantities (`sparse(H)`, `gershgorin_bound`, `separable_eigenbases`, `product_seed`) are
computed on request and memoized; nothing is assembled at construction.
Every array `H` owns (`diag`, factor data, scratch) is stored on `on` at construction (`adapt(on, …)`;
the scratch is allocated like the adapted `diag`).
`DenseOp` factors are applied as ITensor contractions over `H.inds` (one index per axis) and
`H.blockinds` (one block-axis index per width, cached on first use); the other factors keep
their kernels and broadcasts.
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
    inds::Vector{ITensors.Index}               # one distinct index per axis, even for equal sizes
    blockinds::Dict{Int,ITensors.Index}        # block-axis index per block width, made on demand
    diagterms::Vector{AxisTerm}                # the terms fused into `diag` (the constant as a zero-factor term); introspection only
    # Derived quantities, computed on first request and memoized (host data, valid for every storage);
    # shared by `with_block_capacity` and `_on_storage` copies.
    sparse::Base.RefValue{Any}                 # `sparse(H)`
    gershgorin::Base.RefValue{Any}             # `gershgorin_bound(H)`
    separable::Base.RefValue{Any}              # `separable_eigenbases(H)`
end

_empty_memos() = (Ref{Any}(nothing), Ref{Any}(nothing), Ref{Any}(nothing))

function KronFreeOperator(dims::AbstractVector{<:Integer}, terms::AbstractVector{<:AxisTerm};
                          constant::Number=0.0, block_capacity::Integer=1, on=Array)
    dims = Vector{Int}(dims)
    all(>=(1), dims) || throw(ArgumentError("KronFreeOperator: dims must be positive, got $dims"))
    for term in terms, (a, op) in term.factors
        1 <= a <= length(dims) || throw(ArgumentError("KronFreeOperator: factor axis $a outside 1:$(length(dims))"))
        dimension(op) == dims[a] ||
            throw(DimensionMismatch("KronFreeOperator: factor on axis $a has dimension $(dimension(op)), expected $(dims[a])"))
    end
    block_capacity >= 1 || throw(ArgumentError("KronFreeOperator: block_capacity must be positive"))
    T = promote_type(typeof(constant), (typeof(t.coeff) for t in terms)...,
                     (eltype(f[2]) for t in terms for f in t.factors)...)
    D = fill(T(constant), dims...)
    kept = _OperatorTerm[]
    fused = AxisTerm[]
    iszero(constant) || push!(fused, AxisTerm(constant, Tuple{Int,LocalOp}[]))
    for term in terms
        if all(f -> f[2] isa DiagonalOp, term.factors)
            _add_term_diagonal!(D, term.coeff, term.factors)
            push!(fused, term)
        else
            push!(kept, _OperatorTerm{T}(T(term.coeff),
                                         Tuple{Int,LocalOp,Any}[(a, op, _factor_data(on, op, T)) for (a, op) in term.factors]))
        end
    end
    shape = (dims..., Int(block_capacity))
    diag = adapt(on, vec(D))
    # Scratch allocated like the stored diagonal, so directly on `on` (no host staging).
    return KronFreeOperator(dims, diag, kept, similar(diag, T, shape), similar(diag, T, shape), on,
                            ITensors.Index[ITensors.Index(d, "axis$k") for (k, d) in enumerate(dims)],
                            Dict{Int,ITensors.Index}(), fused, _empty_memos()...)
end

# `H` with its fused diagonal, factor data and fresh scratch for `capacity` columns stored as `on`
# (`H` itself, unchanged, when it is already stored as `on`). Terms' operators, the fused terms, ITensor
# indices and the memos of derived quantities (host data, valid for both) are shared.
function _on_storage(H::KronFreeOperator, on; capacity::Integer=_block_capacity(H))
    H.on === on && return H
    diag = adapt(on, H.diag)
    T = eltype(H)
    terms = _OperatorTerm[_OperatorTerm{T}(t.coeff, Tuple{Int,LocalOp,Any}[(a, op, _factor_data(on, op, T)) for (a, op, _) in t.factors])
                          for t in H.terms]
    shape = (H.dims..., Int(capacity))
    return KronFreeOperator(H.dims, diag, terms, similar(diag, eltype(H), shape), similar(diag, eltype(H), shape),
                            on, H.inds, H.blockinds, H.diagterms, H.sparse, H.gershgorin, H.separable)
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

# The ITensor index of the block axis of width `w`, created on first use and reused after.
_block_index(H::KronFreeOperator, w::Int) = get!(() -> ITensors.Index(w, "block"), H.blockinds, w)

# dst = M applied along axis k of src, M = data (the factor matrix, stored like dst), as an
# ITensor contraction into wrappers of the arrays themselves (no copies, no result ITensor).
# Every wrapper lists its indices in storage order, and the operator is A = M[k', k], so
# NDTensors runs one gemm without hidden permutations or temporaries:
#   left == 1 (axis k leads in memory):   dst(k', rest) = A(k', k) src(k, rest)
#   right == 1 (axis k trails in memory): dst(rest, k') = src(rest, k) A(k', k)  (gemm, A transposed)
#   otherwise, axis k moved to the front through scratch:
#     dst <- permutedims(src), work(k', rest) = A(k', k) dst(k, rest), dst <- permutedims(work).
# `rest` is every other axis in order, then the block axis; size-1 axes may sit anywhere.
# `work` is state-shaped scratch distinct from dst that may be overwritten; it may be src
# itself when src is chain scratch (src is fully read before work is written). In the first two
# cases an operand the contraction backend would copy is staged in `work` (`_contraction_operand`).
function _apply_dense!(dst::AbstractArray, data::AbstractMatrix, src::AbstractArray, k::Int,
                       H::KronFreeOperator, work::AbstractArray)
    size(dst) == size(src) == size(work) || throw(DimensionMismatch("_apply_dense!: state shapes differ"))
    size(src, k) == size(data, 1) == size(data, 2) ||
        throw(DimensionMismatch("_apply_dense!: size(src, $k) = $(size(src, k)), factor is $(size(data))"))
    (Base.mightalias(dst, src) || Base.mightalias(dst, work)) &&
        throw(ArgumentError("_apply_dense!: dst must not alias src or work"))
    isempty(dst) && return dst
    T = eltype(dst)
    left, d = _axis_layout(src, k)
    right = length(src) ÷ (left * d)
    ik = H.inds[k]
    ik′ = ITensors.prime(ik)
    A = ITensors.itensor(data, (ik′, ik))
    rest = (H.inds[1:k-1]..., H.inds[k+1:end]..., _block_index(H, size(src, ndims(src))))
    if left == 1
        s = _contraction_operand(src, work)
        mul!(ITensors.itensor(dst, (ik′, rest...)), A, ITensors.itensor(s, (ik, rest...)), one(T), zero(T))
    elseif right == 1
        s = _contraction_operand(src, work)
        mul!(ITensors.itensor(dst, (rest..., ik′)), ITensors.itensor(s, (rest..., ik)), A, one(T), zero(T))
    else
        front = reshape(dst, d, left, right)
        permutedims!(front, reshape(src, left, d, right), (2, 1, 3))
        mul!(ITensors.itensor(work, (ik′, rest...)), A, ITensors.itensor(front, (ik, rest...)), one(T), zero(T))
        permutedims!(reshape(dst, left, d, right), reshape(work, d, left, right), (2, 1, 3))
    end
    return dst
end

# The array a contraction in `_apply_dense!` reads in place of `src`: `src` itself, or a copy of it
# staged in the free scratch `work` (same shape) for storage whose contraction backend would
# otherwise copy `src` into a fresh temporary. Extensions add methods for such storage.
_contraction_operand(src::AbstractArray, work::AbstractArray) = src

# Y .+= coeff * (⊗ factors) X, chaining factor applications through the two buffers.
# `other` is free scratch for a dense factor: b2 for the first factor, then the consumed src.
function _accumulate_term!(Y::AbstractArray, term::_OperatorTerm, X::AbstractArray, b1::AbstractArray, b2::AbstractArray,
                           H::KronFreeOperator)
    src = X
    dst, other = b1, b2
    for (a, op, data) in term.factors
        if op isa DenseOp
            _apply_dense!(dst, data, src, a, H, other)
        else
            _apply_factor!(dst, op, data, src, a)
        end
        src = dst
        dst, other = other, dst
    end
    Y .+= term.coeff .* src
    return Y
end

# `mul!` for a real operator and complex vectors: `H` applied to `real(x)` and `imag(x)` through real
# temporaries (allocating). A convenience path that solvers never take; they carry vectors of `eltype(H)`.
function _mul_real_operator!(y::AbstractVecOrMat, H::KronFreeOperator, x::AbstractVecOrMat)
    xr = real.(x)
    yr = similar(xr)
    mul!(yr, H, xr)
    xr .= imag.(x)
    yi = similar(xr)
    mul!(yi, H, xr)
    y .= complex.(yr, yi)
    return y
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, H::KronFreeOperator, x::AbstractVecOrMat)
    eltype(H) <: Real && eltype(x) <: Complex && return _mul_real_operator!(y, H, x)
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
        _accumulate_term!(Y, term, X, b1, b2, H)
    end
    return y
end

Base.:*(H::KronFreeOperator, x::AbstractVecOrMat) =
    mul!(similar(H.diag, promote_type(eltype(H), eltype(x)), size(x)), H, x)

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
term does not touch), axis 1 fastest, returned on the host. Its real part is `_full_diagonal(H)`.
"""
diagonal(H::KronFreeOperator) = Array(_device_diagonal(H))

# The diagonal of H on `H.on`: the fused `H.diag` plus, for each term kept in the contraction
# sum, `coeff * ⊗_k diagonal(op_k)`, built by broadcasts on the device.
function _device_diagonal(H::KronFreeOperator)
    N = length(H.dims)
    D = copy(reshape(H.diag, H.dims...))
    for term in H.terms
        acc = fill!(similar(H.diag, eltype(H), ntuple(_ -> 1, N)), term.coeff)
        for (a, op, _) in term.factors
            acc = acc .* reshape(adapt(H.on, diagonal(op)), ntuple(i -> i == a ? H.dims[a] : 1, N))
        end
        D .+= acc
    end
    return vec(D)
end

"""
    _full_diagonal(H::KronFreeOperator)

The real diagonal of the Hermitian operator `H` (fused part plus the diagonals of the terms
kept in the contraction sum), as an array on `H.on`.
"""
_full_diagonal(H::KronFreeOperator) = real.(_device_diagonal(H))

# The number of block columns the scratch buffers currently hold.
_block_capacity(H::KronFreeOperator) = length(H.buf1) ÷ prod(H.dims)

"""
    with_block_capacity(H::KronFreeOperator, m::Integer) -> KronFreeOperator

The same operator, sharing `H`'s factor data, `diag`, terms, indices and memoized derived
quantities, with its own fresh scratch buffers reserved for blocks of `m` columns (scratch is
never shared with `H`).
"""
function with_block_capacity(H::KronFreeOperator, m::Integer)
    m >= 1 || throw(ArgumentError("with_block_capacity: m must be positive, got $m"))
    shape = (H.dims..., Int(m))
    return KronFreeOperator(H.dims, H.diag, H.terms, similar(H.buf1, shape), similar(H.buf2, shape),
                            H.on, H.inds, H.blockinds, H.diagterms, H.sparse, H.gershgorin, H.separable)
end

const _MATRIX_BLOCK = 256

"""
    Matrix(H::KronFreeOperator) -> Matrix{eltype(H)}

The explicit `n × n` matrix, built by applying `H` to blocks of at most 256 identity columns,
so the peak memory beyond the result is a few `n × 256` blocks.
Host operators only: an operator stored on a device throws `ArgumentError`.
"""
function Base.Matrix(H::KronFreeOperator)
    H.diag isa Array || throw(ArgumentError("Matrix(H) is host-only, but H is stored as $(typeof(H.diag)); " *
                                            "build the operator with on=Array to form its matrix"))
    T = eltype(H)
    n = prod(H.dims)
    M = Matrix{T}(undef, n, n)
    B = min(n, _MATRIX_BLOCK)
    E = zeros(T, n, B)                   # identity block, reused
    Y = Matrix{T}(undef, n, B)
    for c0 in 1:B:n
        w = min(B, n - c0 + 1)
        if w < B                         # last, narrower block
            E = zeros(T, n, w)
            Y = Matrix{T}(undef, n, w)
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

# ---------------------------------------------------------------------------------------------
# Derived quantities (spec §20e decision 2): computed when asked for, memoized on the operator in
# `Ref` fields that start empty, never built at construction. All are host data.

"""
    sparse(H::KronFreeOperator; maxnnz=10^8) -> SparseMatrixCSC

The sparse matrix of `H`: the Kronecker sum `Σ_t c_t ⊗_k A_{t,k}` of the terms kept in the
contraction sum (each factor's sparse matrix built from its data, identity on absent axes, axis 1
fastest, i.e. `kron(A_N, …, A_1)`) plus the fused diagonal `H.diag`, with exact zeros dropped.
Before anything is assembled, the nonzeros are estimated as `prod(dims)` (the diagonal) plus
`Σ_t Π_k nnz(A_{t,k}) · Π_{absent} d` over the terms with nonzero coefficient; an estimate above
`maxnnz` throws `ArgumentError`, so a branch with dense port factors cannot ask for it.
Memoized in `H.sparse`; the memo is a host matrix whatever `H.on` is, and a later call returns it
regardless of `maxnnz`.
"""
function SparseArrays.sparse(H::KronFreeOperator; maxnnz::Integer=10^8)
    H.sparse[] === nothing || return H.sparse[]
    T = eltype(H)
    n = prod(H.dims)
    local_ = [Dict(a => sparse(op) for (a, op, _) in t.factors) for t in H.terms]
    estimate = Int128(n)
    for (t, ops) in zip(H.terms, local_)
        iszero(t.coeff) && continue
        estimate += prod((haskey(ops, a) ? Int128(nnz(ops[a])) : Int128(d) for (a, d) in enumerate(H.dims)); init=Int128(1))
    end
    estimate > maxnnz &&
        throw(ArgumentError("sparse(H): an estimated $estimate nonzeros exceed the cap maxnnz = $maxnnz; " *
                            "raise maxnnz to assemble it anyway"))
    rows, cols, vals = collect(1:n), collect(1:n), Vector{T}(Array(H.diag))
    for (t, ops) in zip(H.terms, local_)
        iszero(t.coeff) && continue
        M = sparse(fill(one(T), 1, 1))
        for (a, d) in enumerate(H.dims)                          # kron(A_N, …, A_1): axis 1 fastest
            M = kron(haskey(ops, a) ? ops[a] : sparse(one(T) * I, d, d), M)
        end
        i, j, v = findnz(M)
        append!(rows, i); append!(cols, j); append!(vals, t.coeff .* v)
    end
    S = dropzeros!(sparse(rows, cols, vals, n, n))
    H.sparse[] = S
    return S
end

"""
    gershgorin_bound(H::KronFreeOperator) -> Float64

A rigorous upper bound on the largest eigenvalue of the Hermitian operator `H`:
`maximum(real(H.diag)) + Σ_t |c_t| Π_k infnorm(A_{t,k})` over the terms kept in the contraction sum
(plus `maximum(abs ∘ imag, H.diag)`, zero for a Hermitian fused diagonal).
It holds because `H − Re(diag)` is Hermitian, so its 2-norm is at most its ∞-norm, and the ∞-norm
of a Kronecker product is the product of the factors' ∞-norms. Memoized in `H.gershgorin`.
"""
function gershgorin_bound(H::KronFreeOperator)
    H.gershgorin[] === nothing || return H.gershgorin[]
    R = real(eltype(H))
    off = sum((abs(t.coeff) * prod((infnorm(op) for (_, op, _) in t.factors); init=one(R)) for t in H.terms); init=zero(R))
    b = maximum(real, H.diag) + maximum(x -> abs(imag(x)), H.diag) + off
    H.gershgorin[] = b
    return b
end

"""
    separable_eigenbases(H::KronFreeOperator) -> (; U, λ)

Per axis `k`, the eigendecomposition `U[k] * Diagonal(λ[k]) * U[k]'` of the `d_k × d_k` sum of
every term with exactly one factor, on axis `k` (`c_t · dense(A)`, terms fused into the diagonal
included); the constant `H` was built with is added to axis 1. Terms on two or more axes, diagonal or
not, are coupling and are ignored. An axis whose single-axis sum is diagonal (in particular one with
no single-axis term) keeps the identity as `U[k]` and the diagonal, unsorted, as `λ[k]`; otherwise
`λ[k]` is ascending from `eigen!`. A single-axis sum that is not Hermitian to round-off means `H` is
not Hermitian and throws `ArgumentError`. Memoized in `H.separable`; host data.
"""
function separable_eigenbases(H::KronFreeOperator)
    H.separable[] === nothing || return H.separable[]
    T = eltype(H)
    A = [zeros(T, d, d) for d in H.dims]
    for t in H.terms
        length(t.factors) == 1 || continue
        a, op, _ = t.factors[1]
        A[a] .+= t.coeff .* dense(op)
    end
    for t in H.diagterms
        if length(t.factors) == 1
            a, op = t.factors[1]
            A[a] .+= t.coeff .* dense(op)
        elseif isempty(t.factors)                                # the constant
            A[1] .+= t.coeff .* Matrix{T}(I, H.dims[1], H.dims[1])
        end
    end
    U = Matrix{T}[]
    λ = Vector{real(T)}[]
    for (k, M) in enumerate(A)
        asym, nrm = norm(M - M'), norm(M)
        asym <= sqrt(eps(real(T))) * nrm ||
            throw(ArgumentError("separable_eigenbases: the single-axis terms on axis $k sum to a non-Hermitian " *
                                "matrix (‖M − M'‖ = $asym, ‖M‖ = $nrm), so the operator is not Hermitian"))
        if isdiag(M)
            push!(U, Matrix{T}(I, size(M)...))
            push!(λ, real(diag(M)))
        else
            F = eigen!(Hermitian((M + M') / 2))
            push!(U, F.vectors)
            push!(λ, F.values)
        end
    end
    sep = (U=U, λ=λ)
    H.separable[] = sep
    return sep
end

"""
    product_seed(H::KronFreeOperator, k::Integer) -> Matrix

`k` orthonormal product states on the host, of size `(prod(dims), k)`: the columns
`⊗_a U_a[:, i_a]` (axis 1 fastest) of `separable_eigenbases(H)` with the `k` smallest summed energies
`Σ_a λ_a[i_a]` (`partialsortperm` over the length-`prod(dims)` Kronecker sum of the `λ_a`, which is
smaller than the result), each column built by `kron` of the small per-axis vectors.
When every `U_a` is the identity (a branch, whose axes are already eigenbases) the columns are the
unit vectors of the `k` smallest entries of the full diagonal `_full_diagonal(H)` (the fused diagonal plus the
diagonals of the kept terms, which include diagonal matrices that are not `DiagonalOp`s and the port couplings).
"""
function product_seed(H::KronFreeOperator, k::Integer)
    n = prod(H.dims)
    1 <= k <= n || throw(ArgumentError("product_seed: k must lie in 1:$n (the operator's dimension), got $k"))
    sep = separable_eigenbases(H)
    X = zeros(eltype(H), n, k)
    if all(U -> U == I, sep.U)
        for (j, i) in enumerate(partialsortperm(Array(_full_diagonal(H)), 1:k))
            X[i, j] = 1
        end
        return X
    end
    sums = sep.λ[1]
    for a in 2:length(H.dims)
        sums = vec(sums .+ transpose(sep.λ[a]))                   # axis 1 fastest
    end
    cart = CartesianIndices(Tuple(H.dims))
    K = length(H.dims)
    for (j, flat) in enumerate(partialsortperm(sums, 1:k))
        idx = Tuple(cart[flat])
        if K == 1
            X[:, j] .= view(sep.U[1], :, idx[1])
        else
            v = sep.U[1][:, idx[1]]
            for a in 2:K-1
                v = kron(view(sep.U[a], :, idx[a]), v)
            end
            kron!(view(X, :, j), view(sep.U[K], :, idx[K]), v)
        end
    end
    return X
end
