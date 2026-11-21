module KronFreeCircuitsQuantumToolboxExt

# QuantumToolbox objects from a materialized System (conventions in src/quantumtoolbox.jl). Dressed objects are plain
# `Qobj`s; matrix-free ones are SciMLOperators `FunctionOperator`s over the `KronFreeOperator`, inside a `QobjEvo`.

using LinearAlgebra
using QuantumToolbox
using KronFreeCircuits
using KronFreeCircuits: System, Node, Port, KronFreeOperator, operator, materialized, _on_storage, adapt
import KronFreeCircuits: qt_hamiltonian, qt_operator, qt_ket, to_dressed, drive_hamiltonian

const SMO = QuantumToolbox.SciMLOperators

const _Key = Union{Tuple{Symbol,Port},AbstractVector{<:Tuple{Symbol,Port}}}
const _HERMITIAN_KINDS = (:n, :n2, :phi, :phi2)

# The validated retained count `n` of a request (all kept when `nothing`), after checking `basis` and `on`.
function _check_request(sys::System, node::Node, basis::Symbol, n, on)
    basis in (:dressed, :free) || throw(ArgumentError("basis must be :dressed or :free, got :$basis"))
    nk = length(energies(sys, node))
    n = n === nothing ? nk : n
    1 <= n <= nk || throw(ArgumentError("n = $n is outside 1:$nk, the kept states of the node"))
    basis === :dressed && on !== Array &&
        throw(ArgumentError("the dressed basis lives on the host; on=$on applies to basis=:free only"))
    return n
end

_keyvec(key::Tuple{Symbol,Port}) = Tuple{Symbol,Port}[key]
_keyvec(key::AbstractVector) = Tuple{Symbol,Port}[k for k in key]
_dressed_port(sys, key::Tuple{Symbol,Port}, node) = port(sys, key..., node)
_dressed_port(sys, key::AbstractVector, node) = port(sys, _keyvec(key), node)
_hermitian(key) = all(((_, p),) -> p.kind in _HERMITIAN_KINDS, _keyvec(key))

# ---- matrix-free apply: w = α scale (H − shift) v + β w, for a ket (vector) or an N×m block.
# A real H applied to complex v packs [Re V | Im V] into one real N×2m block and applies once; a complex H applies
# directly. Buffers live with H and are reallocated when the block width changes.
mutable struct KronFreeApply{O<:KronFreeOperator,R,C}
    H::O
    scale::Float64
    shift::Float64
    xr::R        # real input block (real H)
    yr::R        # real output block (real H)
    yc::C        # complex output block (complex H)
end

function KronFreeApply(H::KronFreeOperator, scale::Real, shift::Real)
    return KronFreeApply(H, Float64(scale), Float64(shift), similar(H.diag, Float64, 0, 0), similar(H.diag, Float64, 0, 0),
                         similar(H.diag, ComplexF64, 0, 0))
end

# H v into a complex N×m block (owned by A; valid until the next call), v reshaped to N×m.
function _apply_block(A::KronFreeApply, V::AbstractMatrix)
    N, m = size(V)
    if eltype(A.H) <: Real
        if size(A.xr) != (N, 2m)
            A.xr = similar(A.H.diag, Float64, N, 2m); A.yr = similar(A.xr)
        end
        A.xr[:, 1:m] .= real.(V); A.xr[:, m+1:2m] .= imag.(V)
        mul!(A.yr, A.H, A.xr)
        size(A.yc) == (N, m) || (A.yc = similar(A.H.diag, ComplexF64, N, m))
        A.yc .= complex.(view(A.yr, :, 1:m), view(A.yr, :, m+1:2m))
    else
        size(A.yc) == (N, m) || (A.yc = similar(A.H.diag, ComplexF64, N, m))
        mul!(A.yc, A.H, V)
    end
    return A.yc
end

function _apply!(A::KronFreeApply, w, v, α, β)
    N = size(A.H, 1)
    V = reshape(v, N, :); W = reshape(w, N, :)
    Y = _apply_block(A, V)
    s = α * A.scale
    if iszero(β)
        W .= s .* (Y .- A.shift .* V)
    else
        W .= s .* (Y .- A.shift .* V) .+ β .* W
    end
    return w
end
(A::KronFreeApply)(w, v, u, p, t) = _apply!(A, w, v, true, false)
(A::KronFreeApply)(w, v, u, p, t, α, β) = _apply!(A, w, v, α, β)
(A::KronFreeApply)(v, u, p, t) = _apply!(A, similar(v), v, true, false)

# A FunctionOperator over the callable `f` on a product basis of dimension N: on an N-vector (`batch = false`) or on
# N×m blocks of any m (`batch = true`). `u = proto` is required: without it SciMLOperators fails in
# `update_coefficients!` ("cannot convert a value to nothing").
function _function_operator(f, N::Int, on, batch::Bool; hermitian::Bool, constant::Bool=true)
    proto = adapt(on, batch ? zeros(ComplexF64, N, 1) : zeros(ComplexF64, N))
    return SMO.FunctionOperator(f, proto, proto; u=proto, p=QuantumToolbox.NullParameters(), batch=batch, T=ComplexF64,
                                islinear=true, isconstant=constant, has_mul5=true, ishermitian=hermitian)
end

_free_qobjevo(F, N) = QobjEvo(F; type=Operator(), dims=N)

# ---- static objects

function qt_hamiltonian(sys::System, node::Node=sys.root.node; basis::Symbol=:dressed, n=nothing, on=Array,
                        batch::Bool=false)
    n = _check_request(sys, node, basis, n, on)
    E = energies(sys, node)
    basis === :dressed && return Qobj(2π * Diagonal(E[1:n] .- E[1]); type=Operator(), dims=n)
    H = _on_storage(operator(sys, node), on)
    N = size(H, 1)
    return _free_qobjevo(_function_operator(KronFreeApply(H, 2π, E[1]), N, on, batch; hermitian=true), N)
end

function qt_operator(sys::System, key::_Key, node::Node=sys.root.node; basis::Symbol=:dressed, n=nothing, on=Array,
                     batch::Bool=false)
    n = _check_request(sys, node, basis, n, on)
    basis === :dressed && return Qobj(Matrix(_dressed_port(sys, key, node)[1:n, 1:n]); type=Operator(), dims=n)
    O = port_operator(sys, key, node; on)
    N = size(O, 1)
    return _free_qobjevo(_function_operator(KronFreeApply(O, 1.0, 0.0), N, on, batch; hermitian=_hermitian(key)), N)
end

# The index of `label` in the requested basis (1-based), with errors naming the label.
function _label_index(sys::System, node::Node, label::Tuple, basis::Symbol, n::Int)
    if basis === :dressed
        i = dressed_index(sys, node, label)
        i === nothing && throw(ArgumentError("bare state $label has no weight on the kept dressed states"))
        i <= n || throw(ArgumentError("bare state $label is dressed state $i, outside the n = $n retained"))
        return i
    end
    dims = Tuple(materialized(sys, node).dims)
    length(label) == length(dims) && all(l -> l isa Integer, label) && all(1 .<= collect(label) .<= collect(dims)) ||
        throw(ArgumentError("label $label is not a product state of dims $dims"))
    return LinearIndices(dims)[label...]
end

_basis_dim(sys, node, basis, n) = basis === :dressed ? n : prod(materialized(sys, node).dims)

function qt_ket(sys::System, label::Tuple, node::Node=sys.root.node; basis::Symbol=:dressed, n=nothing, on=Array)
    n = _check_request(sys, node, basis, n, on)
    N = _basis_dim(sys, node, basis, n)
    v = zeros(ComplexF64, N); v[_label_index(sys, node, label, basis, n)] = 1
    return Qobj(adapt(on, v); type=Ket(), dims=N)
end

function qt_ket(sys::System, labels::AbstractVector, node::Node=sys.root.node; basis::Symbol=:dressed, n=nothing,
                on=Array)
    isempty(labels) && throw(ArgumentError("qt_ket: no labels"))
    n = _check_request(sys, node, basis, n, on)
    N = _basis_dim(sys, node, basis, n)
    m = length(labels)
    M = zeros(ComplexF64, N, m)
    for (j, l) in enumerate(labels)
        M[_label_index(sys, node, Tuple(l), basis, n), j] = 1
    end
    return Qobj(adapt(on, M); type=Operator(), dims=((N,), (m,)))
end

function to_dressed(sys::System, ψ::QuantumObject, node::Node=sys.root.node; n=nothing)
    n = _check_request(sys, node, :dressed, n, Array)
    V = eigenvectors(sys, node)
    data = Array(ψ.data)
    size(data, 1) == size(V, 1) ||
        throw(ArgumentError("to_dressed: the state has $(size(data, 1)) rows, the node's product basis $(size(V, 1))"))
    A = view(V, :, 1:n)' * data
    isket(ψ) && return Qobj(A; type=Ket(), dims=n)
    return Qobj(A; type=Operator(), dims=((n,), (size(A, 2),)))
end

end # module
