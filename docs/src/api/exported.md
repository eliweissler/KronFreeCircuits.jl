# Exported names

Everything `using KronFreeCircuits` brings into scope.

### Specs and terms

```@docs
Spec
load_spec
parse_spec
numeric_spec
spec_json
save_spec
substitute
promote_mode
Coefficient
Port
nport
n2port
phiport
phi2port
expiphi
exp2pin
Factor
Term
circuit_terms
constant_term
character
suggest_bases
```

### circuit-quantization interface

```@docs
circuit_quantization
QuantizedCircuit
record_json
save_record
load_record
```

### Bases and trees

```@docs
ChargeBasis
PhaseGrid
PeriodicGrid
FockBasis
Node
GivenSystem
```

### Solvers

```@docs
Auto
Dense
Lanczos
LOBPCG
ShiftInvert
ChebyshevFilter
```

### Selection rules

```@docs
lowest
indices
connected
follow
within
```

### Solving and reading results

```@docs
materialize
System
energies
port
eigenvectors
materialized
arithmetic
bare_labels
bare_overlaps
decompose
dressed_index
dressed_state_string
port_operator
spectrum
```

### Derivatives

```@docs
derivatives
blocks
block_levels
```

### Wavefunctions

```@docs
phase_wavefunction
charge_wavefunction
```

### Saving and loading

```@docs
save_system
load_system
system_nodes
load_node
load_metadata
```

### QuantumToolbox

```@docs
qt_hamiltonian
qt_operator
qt_ket
to_dressed
drive_hamiltonian
Frame
to_lab
```

