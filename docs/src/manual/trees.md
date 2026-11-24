# Trees and truncation

A tree says how to split a circuit into pieces, how to solve each piece, and how many of its states to pass up.

## Nodes

`Node(children; solver=Auto(), compute=0, keep=lowest(0), phase, ports=[], arithmetic=:auto, retain=:auto)` is one unit of the tree.
Each child is a `mode => basis` pair, another `Node`, or a [`GivenSystem`](@ref); the node's product basis has one axis per child.

- A **leaf** has only `mode => basis` children. Its product basis is the tensor product of its mode bases, so a leaf with two modes diagonalizes both modes together.
- A **branch** has at least one node child. Its product basis is the tensor product of its children's kept states.

Each mode can appear only once in the tree, and every mode that appears in a Hamiltonian term must be in it.
`materialize(root, spec, θ)` solves the tree bottom-up and returns a `System`.
Results are read at the root by default, or at any node by passing it:

```@example trees
using KronFreeCircuits

spec = load_spec("two_transmons.json")
θ = (E_J1=30.0, E_C1=1.2, n_g1=0.25, E_J2=25.0, E_C2=1.0, n_g2=0.0, g=0.3)
q1 = Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(6), ports=[(:varphi_1, nport())])
q2 = Node([:varphi_2 => ChargeBasis(30)]; keep=lowest(6))
root = Node([q1, q2]; keep=lowest(12))
sys = materialize(root, spec, θ)
energies(sys, q1)                       # q1's kept energies
```

```@example trees
size(eigenvectors(sys)), size(eigenvectors(sys, q1))   # root: 36 product states × 12 kept; q1: 61 charges × 6 kept
```

The spec's scalar constant is added at the root only, so the `energies` of child nodes exclude it.

## Computing and keeping

`compute` is how many of the lowest eigenpairs the solver finds (`0`: all, which only `Dense` can do), and `keep` is the rule that chooses the kept states among them:

| rule | keeps |
|:-----|:------|
| `lowest(k)` | the `k` lowest computed states |
| `indices(list)` | the states at the given indices, in the given order |
| `within(window; max=0)` | the states within `window` (energy units of the spec) of the node's lowest, at most `max` of them |
| `connected(core, (mode, port), thresh)` | `core` and every state reachable from it through matrix elements of `port` above `thresh`, to closure |
| `follow(reference=nothing; minoverlap=0.5)` | one state per reference state, by overlap: adiabatic following in a sweep |

`within` is an energy cutoff instead of a count: it spends kept states where a node's spectrum is dense and makes nodes of different size comparable.
`compute` must be large enough to close the window, and the rule warns when it is not.
Keeping the states of `q1` within 40 GHz of its ground state:

```@example trees
q1w = Node([:varphi_1 => ChargeBasis(30)]; compute=12, keep=within(40.0))
sysw = materialize(Node([q1w, q2]; keep=lowest(12)), spec, θ)
energies(sysw, q1w) .- energies(sysw, q1w)[1]
```

`connected` keeps the states a drive or a coupling reaches from a core, such as the states that `n` connects to the two lowest with matrix elements above `1e-3`:

```julia
Node([:varphi_1 => ChargeBasis(30)]; compute=12, keep=connected([1, 2], (:varphi_1, nport()), 1e-3))
```

## Ports

A port is an operator on one mode: `nport()` (charge), `n2port()`, `phiport()`, `phi2port()`, `expiphi(k)` (`e^{ikφ}`) and `exp2pin(k)` (`e^{2πikn}`).
A node forms the ports its parent needs to build its Hamiltonian, in its kept eigenbasis, and `ports=[(mode, port), ...]` asks for more.
`port(sys, mode, p)` reads one at the root and `port(sys, mode, p, node)` at a node:

```@example trees
round.(abs.(port(sys, :varphi_1, nport(), q1)); digits=3)
```

A term with several factors inside one child (a phase factor over several modes of a KITE, say) is applied through that child's joint port, formed in the child's product basis before its truncation; `port(sys, [(:varphi_1, expiphi(1)), (:varphi_4, expiphi(-1))], node)` reads one.
`port_operator(sys, key, node)` returns a port as a matrix-free operator on a node's product basis.

## Phases

By default each kept state's phase makes its full-space amplitude real and positive at one point: at a leaf, its largest entry with nonnegative coordinates; at a branch, the joint peak of its dominant product component.
`phase = :follow` instead aligns each state with the previous solution of a sweep.

## Bare-state bookkeeping

`decompose(sys, node, i)` lists dressed state `i`'s content in its children's kept states, and `dressed_index(sys, node, label)` finds the dressed state that best matches a bare label.
`dressed_state_string(sys, node, n)` prints a state with physics indices from 0.
`bare_labels` and `bare_overlaps` give the labels and overlaps for all states at once.

```@example trees
bare_labels(sys, root)[1:4]
```

## Given subsystems

A pre-diagonalized subsystem (a resonator, a qubit from another tool, a subsystem exported from scqubits) enters the tree as a `GivenSystem`, a child beside nodes and `mode => basis` pairs.
It takes its ascending energies and a dictionary from port kind to its matrix in that eigenbasis; it keeps all its states, with identity eigenvectors.
Coupling terms name its mode and a port kind it provides, and `materialize` takes them as `extra_terms`:

```@example trees
using LinearAlgebra
spec = load_spec("transmon.json")                             # mode varphi_1
θ = (E_J=30.0, E_C=1.2, n_g=0.25)
a = diagm(1 => sqrt.(1.0:4.0))                                # resonator annihilation operator, 5 levels
res = GivenSystem(:res; energies=6.0 .* (0:4), operators=Dict(:x => a + a'))
qubit = Node([:varphi_1 => ChargeBasis(30)]; keep=lowest(6))
top = Node([qubit, res]; keep=lowest(12))
coupling = [Term(Coefficient(0.05), [Factor(:varphi_1, nport()), Factor(:res, Port(:x, 0))])]   # 0.05 n (a + a†)
sys = materialize(top, spec, θ; extra_terms=coupling)
energies(sys)[1:4] .- energies(sys)[1]
```

A given system's `energies` must not include a constant that the parent also adds (for example a circuit spec's root constant), because `materialize` adds `constant_term(spec)` once.
If the given system is built from a materialized `System`, subtract it: `energies(sys) .- real(sys.constant(sys.θ))`.
A term whose factors all lie on the given mode (a drive or a Stark shift) is applied at its parent through the given operators.

## Sweeps and warm starts

`materialize(root, spec, θ; previous=sys)` reuses a previous solution at nearby parameters.
It warm-starts every `LOBPCG` node (including an `Auto()` branch) from the node at the same tree position in `sys`: its previous kept vectors become the initial block (a node whose product dimensions changed starts cold).
Only kept vectors are stored, so a node with `compute > keep` fills its other `compute − keep` columns randomly and is only partially warm.
`previous` also feeds the `follow` rule and the `:follow` phase convention.
