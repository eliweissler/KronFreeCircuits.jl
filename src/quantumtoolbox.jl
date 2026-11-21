# Time evolution with QuantumToolbox: stubs whose methods live in the QuantumToolbox extension
# (ext/KronFreeCircuitsQuantumToolboxExt.jl), so the core loads without QuantumToolbox and its SciML stack.
#
# Conventions shared by every function here:
# - `basis = :dressed`: the `n` lowest kept eigenstates of `node` (default: all kept), dense, on the host.
#   `basis = :free`: `node`'s product basis of its children's kept states, applied matrix-free by the
#   `KronFreeOperator` (on `on`, e.g. `CuArray` with `using CUDA, cuTENSOR`), never formed.
# - Units: time in ns, Hamiltonians in rad/ns: `H/ħ = 2π [H0 − E₀ + Σ_k f_k(t) O_k]`, `f_k` in GHz, `E₀` the node's
#   lowest kept energy (so the ground state carries no phase in either basis). Operators from `qt_operator` carry no 2π.
# - `batch = true` (matrix-free only) builds operators that act on an N×m block, for several initial states evolved at
#   once from the `Operator` of `qt_ket(sys, labels)`; `batch = false` ones act on a single ket. Dressed objects take
#   either.

"""
    Frame(ω)

The frame a [`drive_hamiltonian`](@ref) is written in: `ω === nothing` for the lab frame, else the angular frequencies
Ω (rad/ns, one per basis state) of the interaction picture, `ψ_I(t) = e^{iΩt} ψ_lab(t)`.
"""
struct Frame
    ω::Union{Nothing,Vector{Float64}}
end

"""
    to_lab(frame, ψ, t)

The lab-frame state `e^{−iΩt} ψ` of an interaction-picture state `ψ` at time `t` (ns): a vector, an N×m block (each
row rotated by its Ω) or, with the QuantumToolbox extension, a `Ket`/`Operator` of the same type and dims; on the host
or the device, wherever `ψ` lives. The identity, returning `ψ` itself, for the lab frame (`frame.ω === nothing`).
"""
to_lab(frame::Frame, ψ, t::Real) = frame.ω === nothing ? ψ : _rotate_back(frame.ω, ψ, t)

function _rotate_back(ω::Vector{Float64}, ψ::AbstractVecOrMat, t::Real)
    size(ψ, 1) == length(ω) ||
        throw(DimensionMismatch("to_lab: the frame has $(length(ω)) frequencies, the state $(size(ψ, 1)) rows"))
    w = copyto!(similar(ψ, Float64, length(ω)), ω)                    # ω where ψ lives
    return cis.(-w .* t) .* ψ
end

const _QT_HINT = "load QuantumToolbox to use the time-evolution interface (`using QuantumToolbox`)"

"""
    qt_hamiltonian(sys, node=root; basis=:dressed, n=nothing, on=Array, batch=false)

`node`'s Hamiltonian `2π (H0 − E₀)` in rad/ns as a QuantumToolbox object: a diagonal `Qobj` of the `n` lowest kept
energies (`basis=:dressed`), or a `QobjEvo` around the matrix-free `KronFreeOperator` on `node`'s product basis
(`basis=:free`, on `on`; `batch` as for [`drive_hamiltonian`](@ref)).
Defined by the QuantumToolbox extension: $_QT_HINT.
"""
function qt_hamiltonian end

"""
    qt_operator(sys, key, node=root; basis=:dressed, n=nothing, on=Array, batch=false)

The port `key = (mode, p)` or joint port `key = [(mode, p), ...]` of `node`, with no 2π: a `Qobj` of
`port(sys, key, node)` on the `n` lowest kept states (`basis=:dressed`), or a `QobjEvo` around
[`port_operator`](@ref) (`basis=:free`). A dressed port `node` did not form (a joint port at the root, a port missing
from `node`'s `ports`) is `port_operator` projected onto the kept states, `V† O V`, which needs the children's ports.
Defined by the QuantumToolbox extension: $_QT_HINT.
"""
function qt_operator end

"""
    qt_ket(sys, label, node=root; basis=:dressed, n=nothing, on=Array)
    qt_ket(sys, labels::AbstractVector, node=root; basis=:dressed, n=nothing, on=Array)

The state of the bare label `label` (a tuple of child indices, as [`bare_labels`](@ref)): the dressed state that
[`dressed_index`](@ref) assigns to it, among the `n` retained (`basis=:dressed`), or the product state itself
(`basis=:free`, on `on`). With a vector of labels, the N×m `Operator` whose columns are those states: the initial state
of a partial propagator, evolved by `sesolve` as one block.
Defined by the QuantumToolbox extension: $_QT_HINT.
"""
function qt_ket end

"""
    to_dressed(sys, ψ, node=root; n=nothing)

The amplitudes `V† ψ` of a product-basis state `ψ` (a `Ket`, or an N×m `Operator` block) on the `n` lowest kept
dressed states of `node`, on the host, as a `Ket` or `n × m` `Operator`.
Defined by the QuantumToolbox extension: $_QT_HINT.
"""
function to_dressed end

"""
    drive_hamiltonian(sys, drives, node=root; basis=:dressed, frame=:lab, n=nothing, on=Array, batch=false)
        -> (H::QobjEvo, frame::Frame)

`H(t)/ħ = 2π [H0 − E₀ + Σ_k f_k(t) O_k]` for `drives = [key_k => f_k, ...]`, `key_k` a port or joint-port key of `node`
and `f_k(p, t)` its coefficient in GHz (QuantumToolbox's coefficient signature), for `sesolve`.

`frame = :lab` returns `H(t)` itself. `frame = :interaction` returns it in the frame of the diagonal Ω of
`2π (H0 − E₀)` in the evolved basis, `H_I(t) = e^{iΩt} (H(t) − diag Ω) e^{−iΩt}`: exact in the dressed basis (H0 drops
out), and in the matrix-free basis the off-diagonal part of H0 stays and is rotated with the drives.
The returned [`Frame`](@ref) takes a solution back to the lab frame with [`to_lab`](@ref).

`batch = true` (matrix-free only) evolves an N×m block of initial states at once (any m, from
`qt_ket(sys, labels; basis=:free)`); it then rejects a single ket, and `batch = false` rejects a block.
Defined by the QuantumToolbox extension: $_QT_HINT.
"""
function drive_hamiltonian end

# A MethodError on a time-evolution function with no method means the extension is not loaded; say how to load it.
function _qt_hint(io::IO, exc::MethodError, argtypes, kwargs)
    exc.f in (qt_hamiltonian, qt_operator, qt_ket, to_dressed, drive_hamiltonian) || return
    Base.get_extension(@__MODULE__, :KronFreeCircuitsQuantumToolboxExt) === nothing || return
    print(io, "\n", _QT_HINT)
end
