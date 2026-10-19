# Local operators acting on one axis of a state tensor; structure is carried in the type.

abstract type LocalOp end

struct DenseOp{T<:Number} <: LocalOp
    M::Matrix{T}
    function DenseOp{T}(M::Matrix{T}) where {T<:Number}
        size(M, 1) == size(M, 2) || throw(DimensionMismatch("DenseOp requires a square matrix, got $(size(M))"))
        return new{T}(M)
    end
end
DenseOp(M::Matrix{T}) where {T<:Number} = DenseOp{T}(M)
DenseOp(A::AbstractMatrix{T}) where {T<:Number} = DenseOp(Matrix{float(T)}(A))

struct DiagonalOp{T<:Number} <: LocalOp
    d::Vector{T}
end

# Y[i] = sum_b weights[b][i] * X[i + offsets[b]] for 1 <= i + offsets[b] <= n
struct BandedOp{T<:Number} <: LocalOp
    n::Int
    offsets::Vector{Int}
    weights::Vector{Vector{T}}
    function BandedOp{T}(n::Int, offsets::Vector{Int}, weights::Vector{Vector{T}}) where {T<:Number}
        n >= 0 || throw(ArgumentError("BandedOp requires n >= 0, got $n"))
        length(offsets) == length(weights) ||
            throw(DimensionMismatch("BandedOp: $(length(offsets)) offsets but $(length(weights)) weight vectors"))
        all(w -> length(w) == n, weights) ||
            throw(DimensionMismatch("BandedOp: every weights[b] must have length n = $n"))
        return new{T}(n, offsets, weights)
    end
end
BandedOp(n::Int, offsets::Vector{Int}, weights::Vector{Vector{T}}) where {T<:Number} = BandedOp{T}(n, offsets, weights)

# Y[i + s] = factor * X[i] for valid indices
struct ShiftOp{T<:Number} <: LocalOp
    n::Int
    s::Int
    factor::T
    function ShiftOp{T}(n::Int, s::Int, factor::T) where {T<:Number}
        n >= 0 || throw(ArgumentError("ShiftOp requires n >= 0, got $n"))
        return new{T}(n, s, factor)
    end
end
ShiftOp(n::Int, s::Int, factor::T) where {T<:Number} = ShiftOp{T}(n, s, factor)

# The element type of an operator's data (the type `diagonal`, `dense` and `sparse` return).
Base.eltype(::DenseOp{T}) where {T} = T
Base.eltype(::DiagonalOp{T}) where {T} = T
Base.eltype(::BandedOp{T}) where {T} = T
Base.eltype(::ShiftOp{T}) where {T} = T

dimension(op::DenseOp) = size(op.M, 1)
dimension(op::DiagonalOp) = length(op.d)
dimension(op::BandedOp) = op.n
dimension(op::ShiftOp) = op.n

Base.adjoint(op::DenseOp) = DenseOp(Matrix(op.M'))
Base.adjoint(op::DiagonalOp) = DiagonalOp(conj.(op.d))
function Base.adjoint(op::BandedOp{T}) where {T}
    n = op.n
    # A[i, i+o] = w[i]  =>  A'[j, j-o] = conj(w[j-o])
    ws = map(zip(op.offsets, op.weights)) do (o, w)
        w2 = zeros(T, n)
        for j in 1:n
            1 <= j - o <= n && (w2[j] = conj(w[j-o]))
        end
        w2
    end
    return BandedOp(n, -op.offsets, ws)
end
Base.adjoint(op::ShiftOp) = ShiftOp(op.n, -op.s, conj(op.factor))

function diagonal(op::BandedOp{T}) where {T}
    k = findfirst(==(0), op.offsets)
    return k === nothing ? zeros(T, op.n) : Vector{T}(op.weights[k])
end
diagonal(op::DenseOp) = diag(op.M)
diagonal(op::DiagonalOp{T}) where {T} = Vector{T}(op.d)
diagonal(op::ShiftOp{T}) where {T} = op.s == 0 ? fill(op.factor, op.n) : zeros(T, op.n)

dense(op::DenseOp) = copy(op.M)
dense(op::DiagonalOp{T}) where {T} = Matrix(Diagonal(Vector{T}(op.d)))
function dense(op::BandedOp{T}) where {T}
    A = zeros(T, op.n, op.n)
    for (o, w) in zip(op.offsets, op.weights), i in 1:op.n
        1 <= i + o <= op.n && (A[i, i+o] += w[i])
    end
    return A
end
function dense(op::ShiftOp{T}) where {T}
    A = zeros(T, op.n, op.n)
    for i in 1:op.n
        1 <= i + op.s <= op.n && (A[i+op.s, i] = op.factor)
    end
    return A
end

# Sparse matrices built from each operator's own data (never through `dense`), in the element
# type `dense` returns.
SparseArrays.sparse(op::DenseOp) = sparse(op.M)
SparseArrays.sparse(op::DiagonalOp{T}) where {T} = spdiagm(0 => Vector{T}(op.d))
function SparseArrays.sparse(op::BandedOp{T}) where {T}
    I, J, V = Int[], Int[], T[]
    for (o, w) in zip(op.offsets, op.weights), i in 1:op.n
        if 1 <= i + o <= op.n          # A[i, i+o] = w[i]
            push!(I, i); push!(J, i + o); push!(V, w[i])
        end
    end
    return sparse(I, J, V, op.n, op.n)
end
function SparseArrays.sparse(op::ShiftOp{T}) where {T}
    J = max(1, 1 - op.s):min(op.n, op.n - op.s)   # A[j+s, j] = factor
    return sparse(collect(J) .+ op.s, collect(J), fill(op.factor, length(J)), op.n, op.n)
end

"""
    realpart(op::LocalOp), imagpart(op::LocalOp)

The operator of the same structure type whose `Float64` data is the real (imaginary) part of `op`'s data
(of `factor` for a `ShiftOp`), so that `dense(realpart(op)) + im * dense(imagpart(op)) == dense(op)`.
For real data `imagpart` has zero data.
"""
function realpart end
function imagpart end
realpart(op::DenseOp) = DenseOp(Matrix{Float64}(real.(op.M)))
imagpart(op::DenseOp) = DenseOp(Matrix{Float64}(imag.(op.M)))
realpart(op::DiagonalOp) = DiagonalOp(Vector{Float64}(real.(op.d)))
imagpart(op::DiagonalOp) = DiagonalOp(Vector{Float64}(imag.(op.d)))
realpart(op::BandedOp) = BandedOp(op.n, copy(op.offsets), [Vector{Float64}(real.(w)) for w in op.weights])
imagpart(op::BandedOp) = BandedOp(op.n, copy(op.offsets), [Vector{Float64}(imag.(w)) for w in op.weights])
realpart(op::ShiftOp) = ShiftOp(op.n, op.s, Float64(real(op.factor)))
imagpart(op::ShiftOp) = ShiftOp(op.n, op.s, Float64(imag(op.factor)))

"""
    infnorm(op::LocalOp) -> Float64

The induced ∞-norm of `op`, the maximum absolute row sum of its matrix, from the operator's data.
"""
function infnorm end
infnorm(op::DenseOp) = float(opnorm(op.M, Inf))
infnorm(op::DiagonalOp) = float(maximum(abs, op.d; init=zero(real(eltype(op.d)))))
function infnorm(op::BandedOp)
    m = zero(real(eltype(eltype(op.weights))))
    for i in 1:op.n
        r = zero(m)
        for (o, w) in zip(op.offsets, op.weights)
            1 <= i + o <= op.n && (r += abs(w[i]))
        end
        m = max(m, r)
    end
    return float(m)
end
infnorm(op::ShiftOp) = abs(op.s) < op.n ? float(abs(op.factor)) : float(zero(abs(op.factor)))
