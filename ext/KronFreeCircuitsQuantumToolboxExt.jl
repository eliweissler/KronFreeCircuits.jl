module KronFreeCircuitsQuantumToolboxExt

# QuantumToolbox objects from a materialized System (conventions in src/quantumtoolbox.jl). Dressed objects are plain
# `Qobj`s; matrix-free ones are SciMLOperators `FunctionOperator`s over the `KronFreeOperator`, inside a `QobjEvo`.

using LinearAlgebra
using QuantumToolbox
using KronFreeCircuits
using KronFreeCircuits: System, Node, Port, KronFreeOperator, Frame, operator, materialized, _on_storage, adapt
import KronFreeCircuits: qt_hamiltonian, qt_operator, qt_ket, to_dressed, drive_hamiltonian, to_lab, _rotate_back

const SMO = QuantumToolbox.SciMLOperators

const _Key = Union{Tuple{Symbol,Port},AbstractVector{<:Tuple{Symbol,Port}}}

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

# The port `key` on the `n` lowest kept states of `node`: the stored port when `node` formed it, else the product-basis
# operator projected onto those states, V† O V (a joint port, which no node forms for itself, or a port missing from
# `node`'s `ports`; `port_operator` names what a child lacks).
function _dressed_port(sys::System, key, node::Node, n::Int)
    K = KronFreeCircuits._sortkey!(_keyvec(key))
    mn = materialized(sys, node)
    length(K) == 1 && haskey(mn.ports, only(K)) && return Matrix(port(sys, only(K)..., node)[1:n, 1:n])
    length(K) > 1 && haskey(mn.joint_ports, K) && return Matrix(port(sys, K, node)[1:n, 1:n])
    V = Matrix(view(eigenvectors(sys, node), :, 1:n))
    return V' * (port_operator(sys, K, node) * V)
end

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
(A::KronFreeApply)(v, u, p, t) = _apply!(A, similar(v, ComplexF64), v, true, false)    # complex even for a real v

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
    basis === :dressed && return Qobj(_dressed_port(sys, key, node, n); type=Operator(), dims=n)
    O = port_operator(sys, key, node; on)
    N = size(O, 1)
    return _free_qobjevo(_function_operator(KronFreeApply(O, 1.0, 0.0), N, on, batch; hermitian=ishermitian(O)), N)
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

# ---- driven Hamiltonians

function _check_drives(drives)
    for d in drives
        d isa Pair && d.first isa _Key ||
            throw(ArgumentError("each drive must be `key => f` with key a (mode, port) or [(mode, port), ...], got $d"))
    end
    return drives
end

# The coefficient of a matrix-free drive term: a ScalarOperator that QuantumToolbox's solver updates at each t.
_coefficient(f) = SMO.ScalarOperator(zero(ComplexF64); update_func=(a, u, p, t) -> ComplexF64(f(p, t)))

function drive_hamiltonian(sys::System, drives::AbstractVector, node::Node=sys.root.node; basis::Symbol=:dressed,
                           frame::Symbol=:lab, n=nothing, on=Array, batch::Bool=false)
    frame in (:lab, :interaction) || throw(ArgumentError("frame must be :lab or :interaction, got :$frame"))
    n = _check_request(sys, node, basis, n, on)
    _check_drives(drives)
    frame === :interaction && return _interaction_hamiltonian(sys, drives, node, basis, n, on, batch)
    if basis === :dressed
        H0 = qt_hamiltonian(sys, node; n)
        isempty(drives) && return QobjEvo(H0), Frame(nothing)
        terms = [(Qobj(2π * _dressed_port(sys, k, node, n); type=Operator(), dims=n), f) for (k, f) in drives]
        return QobjEvo((H0, terms...)), Frame(nothing)
    end
    E = energies(sys, node)
    H = _on_storage(operator(sys, node), on)
    N = size(H, 1)
    L = _function_operator(KronFreeApply(H, 2π, E[1]), N, on, batch; hermitian=true)
    for (k, f) in drives
        Ok = port_operator(sys, k, node; on)
        L = L + _coefficient(f) * _function_operator(KronFreeApply(Ok, 2π, 0.0), N, on, batch; hermitian=ishermitian(Ok))
    end
    return _free_qobjevo(L, N), Frame(nothing)
end

# ---- interaction picture, in the frame of the diagonal Ω of 2π(H0 − E₀) in the evolved basis:
#      H_I(t) = e^{iΩt} [2π(H0 − E₀) − diag Ω + Σ_k f_k(t) 2π O_k] e^{−iΩt}.
# Dressed: H0 is diagonal and drops out; H_I(t) is a dense matrix refilled at each t (a MatrixOperator, so it takes a
# ket or a block). Matrix-free: a FunctionOperator; the phase vector is formed once per call, buffers follow the block
# width.
mutable struct InteractionApply{V,C}
    Ω::V                 # rad/ns, where the states live
    ph::C                # e^{−iΩt}
    H0::KronFreeApply    # 2π(H0 − E₀)
    ops::Vector{KronFreeApply}   # 2π O_k
    fs::Vector{Any}
    X::Any               # ph .* V
    acc::Any             # the bracket applied to X
end

function _apply!(A::InteractionApply, w, v, p, t, α, β)
    N = length(A.Ω)
    V = reshape(v, N, :); W = reshape(w, N, :)
    m = size(V, 2)
    if !(A.X isa AbstractMatrix) || size(A.X) != (N, m)
        A.X = similar(A.ph, ComplexF64, N, m); A.acc = similar(A.X)
    end
    A.ph .= cis.(-A.Ω .* t)
    A.X .= A.ph .* V
    _apply!(A.H0, A.acc, A.X, true, false)
    A.acc .-= A.Ω .* A.X
    for (O, f) in zip(A.ops, A.fs)
        _apply!(O, A.acc, A.X, ComplexF64(f(p, t)), true)
    end
    if iszero(β)
        W .= α .* conj.(A.ph) .* A.acc
    else
        W .= α .* conj.(A.ph) .* A.acc .+ β .* W
    end
    return w
end
(A::InteractionApply)(w, v, u, p, t) = _apply!(A, w, v, p, t, true, false)
(A::InteractionApply)(w, v, u, p, t, α, β) = _apply!(A, w, v, p, t, α, β)
(A::InteractionApply)(v, u, p, t) = _apply!(A, similar(v, ComplexF64), v, p, t, true, false)

function _interaction_hamiltonian(sys::System, drives, node::Node, basis::Symbol, n::Int, on, batch::Bool)
    E = energies(sys, node)
    if basis === :dressed
        Ω = 2π .* (E[1:n] .- E[1])
        ops = [2π * Matrix{ComplexF64}(_dressed_port(sys, k, node, n)) for (k, _) in drives]
        fs = Any[f for (_, f) in drives]
        ph = zeros(ComplexF64, n)
        function fill_HI!(M, u, p, t)                  # M_ij = Σ_k f_k(t) e^{iΩ_i t} (2π P_k)_ij e^{−iΩ_j t}
            ph .= cis.(Ω .* t)
            fill!(M, 0)
            for (P, f) in zip(ops, fs)
                M .+= ComplexF64(f(p, t)) .* ph .* P .* ph'
            end
            return M
        end
        M = SMO.MatrixOperator(zeros(ComplexF64, n, n); update_func! = fill_HI!)
        return QobjEvo(M; type=Operator(), dims=n), Frame(Ω)
    end
    H = _on_storage(operator(sys, node), on)
    N = size(H, 1)
    Ω = 2π .* (real.(diagonal(H)) .- E[1])
    ops = KronFreeApply[KronFreeApply(port_operator(sys, k, node; on), 2π, 0.0) for (k, _) in drives]
    Ωd = adapt(on, Ω)
    A = InteractionApply(Ωd, similar(Ωd, ComplexF64), KronFreeApply(H, 2π, E[1]), ops, Any[f for (_, f) in drives],
                         nothing, nothing)
    F = _function_operator(A, N, on, batch; hermitian=all(O -> ishermitian(O.H), ops), constant=false)
    return _free_qobjevo(F, N), Frame(Ω)
end

to_lab(frame::Frame, ψ::QuantumObject, t::Real) =
    frame.ω === nothing ? ψ : Qobj(_rotate_back(frame.ω, ψ.data, t); type=ψ.type, dims=ψ.dimensions)

end # module
