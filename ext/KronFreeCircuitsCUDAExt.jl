module KronFreeCircuitsCUDAExt

using KronFreeCircuits, CUDA, cuTENSOR
import KronFreeCircuits: _diagonal_expectation!, _contraction_operand

# NDTensors' cuTENSOR contraction copies an operand stored at a nonzero offset (an offset column
# view of a larger block) into a fresh array before contracting; stage it in the free scratch
# `work` instead, so a dense factor applied to such a view allocates nothing on the device.
_contraction_operand(src::CuArray, work::CuArray) = iszero(src.offset) ? src : copyto!(work, src)

const _EXPECTATION_THREADS = 256      # threads per block: a power of two, for the tree reduction

# theta[col] = Σ_i D[i] |X[i,col]|² / Σ_i |X[i,col]|², one CUDA block per column. Each thread
# accumulates a strided share of both sums in `R`, then a shared-memory tree reduction combines
# them; weighting and normalization are fused, so nothing state-sized is allocated.
function _expectation_kernel!(theta, D, X, ::Type{R}, ::Val{NT}) where {R,NT}
    col = blockIdx().x
    tid = threadIdx().x
    weighted = zero(R)
    weight = zero(R)
    i = tid
    @inbounds while i <= size(X, 1)
        a = R(abs2(X[i, col]))
        weighted += R(D[i]) * a
        weight += a
        i += NT
    end
    ws = CuStaticSharedArray(R, NT)
    ns = CuStaticSharedArray(R, NT)
    @inbounds ws[tid] = weighted
    @inbounds ns[tid] = weight
    sync_threads()
    offset = NT ÷ 2
    while offset > 0
        if tid <= offset
            @inbounds ws[tid] += ws[tid+offset]
            @inbounds ns[tid] += ns[tid+offset]
        end
        sync_threads()
        offset ÷= 2
    end
    tid == 1 && @inbounds(theta[col] = ws[1] / ns[1])
    return nothing
end

# The device method of the Ritz initialization: the host formula, accumulated in
# `promote_type(eltype(D), real(eltype(X)))` (FP64 for this package's data), fused in one kernel.
function _diagonal_expectation!(theta::CuVector, D::CuVector, X::CuMatrix)
    size(X, 1) == length(D) || throw(DimensionMismatch("_diagonal_expectation!: X has $(size(X, 1)) rows, D has $(length(D))"))
    length(theta) == size(X, 2) || throw(DimensionMismatch("_diagonal_expectation!: theta has $(length(theta)) entries for $(size(X, 2)) columns"))
    isempty(theta) && return theta
    R = promote_type(eltype(D), real(eltype(X)))
    @cuda threads=_EXPECTATION_THREADS blocks=size(X, 2) _expectation_kernel!(theta, D, X, R, Val(_EXPECTATION_THREADS))
    return theta
end

end
