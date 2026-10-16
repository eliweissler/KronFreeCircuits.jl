# Local operators acting on one axis of a state tensor; structure is carried in the type.

abstract type LocalOp end

struct DenseOp <: LocalOp
    M::Matrix{ComplexF64}
end
DenseOp(A::AbstractMatrix{<:Number}) = DenseOp(Matrix{ComplexF64}(A))

struct DiagonalOp{T<:Number} <: LocalOp
    d::Vector{T}
end

# Y[i] = sum_b weights[b][i] * X[i + offsets[b]] for 1 <= i + offsets[b] <= n
struct BandedOp{T<:Number} <: LocalOp
    n::Int
    offsets::Vector{Int}
    weights::Vector{Vector{T}}
end

# Y[i + s] = factor * X[i] for valid indices
struct ShiftOp{T<:Number} <: LocalOp
    n::Int
    s::Int
    factor::T
end

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

function diagonal(op::BandedOp)
    k = findfirst(==(0), op.offsets)
    return k === nothing ? zeros(ComplexF64, op.n) : Vector{ComplexF64}(op.weights[k])
end
diagonal(op::DenseOp) = Vector{ComplexF64}(diag(op.M))
diagonal(op::DiagonalOp) = Vector{ComplexF64}(op.d)
diagonal(op::ShiftOp) = op.s == 0 ? fill(ComplexF64(op.factor), op.n) : zeros(ComplexF64, op.n)

dense(op::DenseOp) = copy(op.M)
dense(op::DiagonalOp) = Matrix(Diagonal(Vector{ComplexF64}(op.d)))
function dense(op::BandedOp)
    A = zeros(ComplexF64, op.n, op.n)
    for (o, w) in zip(op.offsets, op.weights), i in 1:op.n
        1 <= i + o <= op.n && (A[i, i+o] += w[i])
    end
    return A
end
function dense(op::ShiftOp)
    A = zeros(ComplexF64, op.n, op.n)
    for i in 1:op.n
        1 <= i + op.s <= op.n && (A[i+op.s, i] = op.factor)
    end
    return A
end
