# Time evolution with QuantumToolbox

`using QuantumToolbox` beside this package loads an extension that turns a materialized `System` into QuantumToolbox objects for `sesolve`, in either of two bases:

- `basis=:dressed`: the `n` lowest kept eigenstates of a node (default: all kept), as dense `Qobj`s on the host;
- `basis=:free`: the node's product basis of its children's kept states, with the Hamiltonian applied matrix-free by the `KronFreeOperator`, on the host or, with `on=CuArray` (and `using CUDA, cuTENSOR`), on the GPU.

Time is in ns and Hamiltonians are in rad/ns: `H/ħ = 2π [H0 − E₀ + Σ_k f_k(t) O_k]`, with each drive coefficient `f_k(p, t)` in GHz and `E₀` the node's lowest kept energy.
`drive_hamiltonian(sys, [key => f, ...]; basis, frame)` builds `H(t)`, where each `key` is a port `(mode, port)` or a joint port `[(mode, port), ...]`.
A matrix-free drive needs its port formed by the child that owns its mode.
A dressed drive uses the node's stored port when it has one. Otherwise, for a joint port at the root or a port missing from the node's `ports`, it projects `port_operator` onto the kept states, `V† O V`, which also needs the children's ports.
`frame=:interaction` writes `H(t)` in the frame of the diagonal of `2π (H0 − E₀)` in the evolved basis:

- In the dressed basis this frame is exact, and only the rotated drives remain.
- In the matrix-free basis the off-diagonal part of H0 stays and is rotated along with the drives. No eigenstates of the node are needed.

The returned `Frame` takes a solution back to the lab frame with `to_lab(frame, ψ, t)`.

Several initial states evolve together as a partial propagator: `qt_ket(sys, labels)` returns the N×m `Operator` of those states, and `sesolve` evolves it as a single block.
Each right-hand-side evaluation is then one m-column apply of the operator, which is cheaper than m separate solves.
For the matrix-free basis, build the Hamiltonian with `batch=true`. It then takes a block of any width, but not a single ket.
`to_dressed(sys, ψ)` gives the dressed amplitudes of a product-basis state or block, and `qt_hamiltonian` and `qt_operator` give the static pieces (`qt_operator` without the 2π, for expectation values).


This page's examples are not run when the documentation is built, since QuantumToolbox is not a dependency of the documentation.

```julia
using KronFreeCircuits, QuantumToolbox

spec = load_spec("two_transmons.json")
leaf(i) = Node([Symbol("varphi_$i") => ChargeBasis(15)]; keep=lowest(6), ports=[(Symbol("varphi_$i"), nport())])
root = Node([leaf(1), leaf(2)]; solver=Dense(), keep=lowest(36), ports=[(:varphi_1, nport())])
sys = materialize(root, spec, (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.0, g=0.3))

E = energies(sys)
f(p, t) = 0.05 * cos(2π * (E[2] - E[1]) * t)                       # GHz
H, frame = drive_hamiltonian(sys, [(:varphi_1, nport()) => f]; basis=:free, frame=:interaction, batch=true)
ψ0 = qt_ket(sys, [(1, 1), (2, 1), (1, 2)]; basis=:free)              # three bare product states, one block
sol = sesolve(H, ψ0, range(0, 20; length=21); progress_bar=Val(false))
U = to_dressed(sys, to_lab(frame, sol.states[end], 20.0))            # dressed amplitudes at t = 20 ns, one column each
```

The package exports the everyday workflow; the operator machinery, the custom-solver interface (`Solver`, `solve`, `resolve`, ...), basis augmentation and a few rarely named types are public but not exported, so write `KronFreeCircuits.name` or `using KronFreeCircuits: name` for them.
Both packages export `Lanczos` and `spectrum`, so after `using KronFreeCircuits, QuantumToolbox` write `KronFreeCircuits.Lanczos()` and `KronFreeCircuits.spectrum` (or QuantumToolbox's) for either name.
