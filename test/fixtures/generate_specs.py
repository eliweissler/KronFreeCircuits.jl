"""Generate test/fixtures/generated/*.json from circuits, with circuit-quantization.

The hand-written fixtures in test/fixtures/*.json were transcribed from scqubits.
This script builds the same Hamiltonians as lumped circuits, quantizes them with
circuit-quantization (~/src/circuit-quantization), and exports each with
circuit_quantization.to_spec; test/test_generated_specs.jl
then checks that every generated spec defines the same Hamiltonian as its
hand-written counterpart.

Run from the repository root, only to regenerate the committed files:

    /opt/anaconda3/envs/circuit_quantization/bin/python test/fixtures/generate_specs.py

Python is never a test-time dependency of KronFreeCircuits.

What the script adds to circuit_quantization.to_spec's output:
- mode names: to_spec labels modes varphi_1..varphi_N; they are renamed to the
  hand fixture's names (labels only, the format calls them descriptive);
- `_generator.commit`, `.dirty`, `.script`: the circuit-quantization git commit,
  whether its tracked files had local changes, and this script, next to the
  `version` to_spec records;
- `_comment`: the circuit, coordinates, flux gauge and parameter mapping.
Every file is re-validated with circuit_quantization.load_spec before writing.

Conventions shared by all circuits:
- Element values use the GHz parametrization: C = ghz_to_farad(E), L = ghz_to_henry(E).
  Where a fixture parameter is a derived quantity rather than an element value
  (an E_C matrix entry such as the coupling g/8), each element is built from a
  positive auxiliary charging energy (x_...), and `subs` replaces it by the
  element value that produces the fixture's matrix; the relation is derived
  next to each builder.
- Each compact mode's offset charge is circuit-quantization's sector label
  n_g<k> (k the mode index); `subs` renames it to the fixture's name.
- Explicit Z_full fixes the fixture's coordinates; its column order is the mode order.

The cos2phi fixture is not generated: see `cos2phi_not_expressible` below.
"""

import json
import subprocess
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import sympy as sym

import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize

OUT = Path(__file__).resolve().parent / "generated"
CQ_DIR = Path(cq.__file__).resolve().parents[2]

pos = lambda names: sym.symbols(names, positive=True)  # noqa: E731
real = lambda names: sym.symbols(names, real=True)  # noqa: E731
S2 = sym.sqrt(2)


def cq_commit():
    git = ["git", "-C", str(CQ_DIR)]
    commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    # Tracked files only: circuit-quantization keeps untracked notebooks and notes, so an uncommitted *new*
    # module that the package imports would not set `dirty`.
    dirty = subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    return commit, bool(dirty)


def save(name, res, mode_names, comment, subs=None):
    """Export res like save_spec, rename the modes, record provenance, validate, write."""
    data = cq.to_spec(res, subs=subs)
    assert len(mode_names) == len(data["modes"])
    for mode, label in zip(data["modes"], mode_names):
        mode["name"] = label
    commit, dirty = cq_commit()
    data["_generator"]["commit"] = commit
    data["_generator"]["dirty"] = dirty
    data["_generator"]["script"] = "test/fixtures/generate_specs.py"
    data["_comment"] = comment
    cq.load_spec(data)
    OUT.mkdir(exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (OUT / f"{name}.json").write_text(text, encoding="utf-8")
    print(f"wrote {name}.json: modes {[(m['name'], m['type']) for m in data['modes']]}, parameters {data['parameters']}")
    return data


def substituted(obj, subs):
    """circuit_quantization.numerify(obj, subs), with every substituted expression put over one denominator.

    Used instead of to_spec(res, subs=subs) where that is impractically slow: sympy.cancel, which to_spec
    applies to each coefficient, does not finish in minutes on E_C[theta1,theta1] of two_zero_pi after
    substitution, while sympy.cancel(sympy.together(...)) of the same expression takes milliseconds.
    sympy.together is an identity, so the exported coefficients are those of to_spec(res, subs=subs).
    The Hamiltonian expression H, which to_spec does not read, is only substituted.
    """
    if isinstance(obj, (Circuit, Loops)):
        return obj
    if is_dataclass(obj):
        return replace(obj, **{f.name: (getattr(obj, f.name).subs(subs) if f.name == "H" else substituted(getattr(obj, f.name), subs))
                               for f in fields(obj) if f.init})
    if isinstance(obj, tuple):
        return tuple(substituted(x, subs) for x in obj)
    if isinstance(obj, sym.MatrixBase):
        return sym.ImmutableMatrix(obj.shape[0], obj.shape[1], [sym.together(v.subs(subs)) for v in obj])
    if isinstance(obj, sym.Basic):
        return sym.together(obj.subs(subs))
    return obj


def junction_flux_B(res, rings):
    """Branch-by-loop flux assignment B that puts each ring's flux on its two junctions.

    rings maps a loop name to {junction branch: share}; the shares are B's entries
    on those junctions, every inductor gets 0, and capacitor entries are solved so
    that G B = I.  Columns of the other loops keep res.B (the irrotational
    assignment), whose fluxes are zero here.  For a static flux any B with G B = I
    is a gauge choice (notes Sec V); this one reproduces scqubits' junction offsets.
    """
    G, names, branches = res.loops.G, list(res.loops.names), list(res.circuit.branch_names)
    B = sym.Matrix(res.B)
    caps = [i for i, b in enumerate(res.circuit.branches) if b.type == "C"]
    for loop, shares in rings.items():
        r = names.index(loop)
        b = sym.zeros(len(branches), 1)
        for junction, share in shares.items():
            b[branches.index(junction)] = share
        y = sym.symbols(f"y0:{len(caps)}")
        target = sym.Matrix([1 if k == r else 0 for k in range(G.rows)]) - G * b
        (solution,) = sym.linsolve((G[:, caps], target), y)
        solution = [s.subs({v: 0 for v in y}) for s in solution]
        for i, value in zip(caps, solution):
            b[i] = value
        assert G * b == sym.Matrix([1 if k == r else 0 for k in range(G.rows)])
        B[:, r] = b
    return B


# --- transmons ---------------------------------------------------------------


def transmon():
    EJ, EC = pos("E_J E_C")
    circuit = Circuit([Branch("J", 1, 0, C=cq.ghz_to_farad(EC), EJ=EJ)], ground=0)
    res = quantize(circuit, Loops.from_circuit(circuit), sym.Matrix([[1]]), ("c",))
    save("transmon", res, ["varphi_1"], [
        "Grounded junction J (node 1 to ground 0), C = ghz_to_farad(E_C), EJ = E_J; Z_full = [1], mode c.",
        "subs: sector label n_g1 -> n_g.",
    ], subs={real("n_g1"): real("n_g")})


def transmons(count):
    """count grounded transmons; transmons 1 and 2 coupled by a capacitor C_c between nodes 1 and 2.

    The fixture's parameters E_C1, E_C2, g are the entries of the 2x2 charging matrix
    T = [[E_C1, g/8], [g/8, E_C2]] of transmons 1 and 2, not element values.
    E_C = farad_to_ghz(1) * Cmat^-1, and with each capacitance written as
    C_k = ghz_to_farad(x_k), Cmat^-1 = T requires
    K = [[1/x_1 + 1/x_c, -1/x_c], [-1/x_c, 1/x_2 + 1/x_c]] = T^-1, i.e. with D = E_C1 E_C2 - g^2/64:
    x_c = 8 D/g, x_1 = D/(E_C2 - g/8), x_2 = D/(E_C1 - g/8).
    The circuit is physical (all capacitances positive) for 0 < g/8 < min(E_C1, E_C2).
    Transmon 3 (three_transmons) is uncoupled: its junction capacitance is ghz_to_farad(E_C3).
    """
    x1, x2, xc = pos("x_1 x_2 x_c")
    EJ = pos(" ".join(f"E_J{k}" for k in range(1, count + 1)))
    EC1, EC2, g = real("E_C1 E_C2 g")
    D = EC1 * EC2 - g**2 / 64
    branches = [Branch("J1", 1, 0, C=cq.ghz_to_farad(x1), EJ=EJ[0]),
                Branch("J2", 2, 0, C=cq.ghz_to_farad(x2), EJ=EJ[1]),
                Branch("Cc", 1, 2, C=cq.ghz_to_farad(xc))]
    if count == 3:
        branches.append(Branch("J3", 3, 0, C=cq.ghz_to_farad(pos("E_C3")), EJ=EJ[2]))
    circuit = Circuit(branches, ground=0)
    res = quantize(circuit, Loops.from_circuit(circuit, zero_capacitive=True), sym.eye(count), ("c",) * count)
    subs = {x1: D / (EC2 - g / 8), x2: D / (EC1 - g / 8), xc: 8 * D / g}
    name = {2: "two_transmons", 3: "three_transmons"}[count]
    comment = [
        "Grounded junctions J1 (node 1), J2 (node 2)" + (", J3 (node 3)" if count == 3 else "")
        + " with EJ = E_Jk, and a coupling capacitor Cc between nodes 1 and 2; Z_full = identity, all modes c.",
        "Element capacitances C = ghz_to_farad(x) with x_1 = D/(E_C2 - g/8), x_2 = D/(E_C1 - g/8), x_c = 8 D/g,"
        " D = E_C1 E_C2 - g^2/64: the values whose inverse capacitance matrix is [[E_C1, g/8], [g/8, E_C2]]"
        " (see transmons() in generate_specs.py).",
        "The capacitive loop J1-Cc-J2 carries zero flux (Loops.from_circuit with zero_capacitive=True).",
        "Sector labels n_g1, n_g2" + (", n_g3" if count == 3 else "") + " already carry the fixture's names.",
    ]
    if count == 3:
        comment.insert(2, "J3's junction capacitance is ghz_to_farad(E_C3).")
    save(name, res, [f"varphi_{k}" for k in range(1, count + 1)], comment, subs=subs)


# --- fluxonium ---------------------------------------------------------------


def fluxonium():
    EC, EL, EJ = pos("E_C E_L E_J")
    vx = real("varphi_ext")
    circuit = Circuit([Branch("J", 1, 0, C=cq.ghz_to_farad(EC), EJ=EJ),
                       Branch("L", 1, 0, L=cq.ghz_to_henry(EL))], ground=0)
    # One loop, closed by L: +L -J; its flux phase_to_weber(varphi_ext) sits on L (irrotational B).
    loops = Loops.from_tree(circuit, cq.default_tree(circuit), {"L": cq.phase_to_weber(vx)})
    res = quantize(circuit, loops, sym.Matrix([[1]]), ("e",))
    save("fluxonium", res, ["varphi_1"], [
        "README fluxonium: junction J and inductor L from node 1 to ground, C = ghz_to_farad(E_C), L = ghz_to_henry(E_L), EJ = E_J.",
        "Loop closed by L (+L -J) with flux phase_to_weber(varphi_ext), irrotational B (all flux on L); Z_full = [1], mode e.",
        "constant 0: circuit-quantization omits the flux-only inductive energy E_L varphi_ext^2/2 that the hand fixture keeps.",
    ])


# --- zero-pi -----------------------------------------------------------------

# Zero-pi as drawn in circuit-quantization's tests/circuits.py (notes fig:zeropi_kite):
# J1 (a->b), J2 (c->d), L1 (b->c), L2 (d->a), C1 (a->c), C2 (b->d), floating.
# Coordinates (theta, phi, zeta, Sigma), node rows a, b, c, d: the notes' eq:zp_Z with
# theta and phi negated, so that the junction phases are J1 = theta + phi and
# J2 = -theta + phi, and zeta divided by sqrt(2), so that zeta is scqubits'
# FullZeroPi coordinate (E_C[zeta,zeta] = E_Cz, E_L[zeta,zeta] = E_L at symmetry).
ZP_M = sym.Matrix([[1, 1, 1 / S2, 1], [-1, -1, 1 / S2, 1], [-1, 1, -1 / S2, 1], [1, -1, -1 / S2, 1]]) / 2


def zero_pi_branches(q, nodes, xJ, xB, yL, EJ):
    """One zero-pi's branches; xJ, xB charging energies of (J1, J2), (C1, C2); yL inductive energies of (L1, L2)."""
    a, b, c, d = nodes
    s = str(q)
    return [Branch(f"J{s}a", a, b, C=cq.ghz_to_farad(xJ[0]), EJ=EJ[0]),
            Branch(f"J{s}b", c, d, C=cq.ghz_to_farad(xJ[1]), EJ=EJ[1]),
            Branch(f"L{s}a", b, c, L=cq.ghz_to_henry(yL[0])),
            Branch(f"L{s}b", d, a, L=cq.ghz_to_henry(yL[1])),
            Branch(f"C{s}a", a, c, C=cq.ghz_to_farad(xB[0])),
            Branch(f"C{s}b", b, d, C=cq.ghz_to_farad(xB[1]))]


def ring_loops(circuit, fluxes):
    """Default fundamental loops, zero flux on every capacitive loop; fluxes maps qubit -> varphi_ext.

    A ring (+L<q>b +J<q>a +J<q>b +L<q>a) is closed by L<q>b and oriented along it;
    the fixture's varphi_ext threads it in the opposite sense, so the loop flux is
    phase_to_weber(-varphi_ext), and the shares +1/2 on J<q>a and J<q>b give both
    junction phases the offset -varphi_ext/2 (scqubits' cos(theta +- (phi - varphi_ext/2))).
    """
    default = Loops.from_circuit(circuit)
    flux = {name: 0 for name in default.names}
    rings = {}
    for q, vx in fluxes.items():
        name = f"L{q}b"
        assert default.branch_dict[name] == {f"L{q}b": 1, f"J{q}a": 1, f"J{q}b": 1, f"L{q}a": 1}, default.branch_dict[name]
        flux[name] = cq.phase_to_weber(-vx)
        rings[name] = {f"J{q}a": sym.Rational(1, 2), f"J{q}b": sym.Rational(1, 2)}
    return Loops.from_tree(circuit, cq.default_tree(circuit), flux), rings


def quantize_junction_gauge(circuit, loops, rings, Z, types):
    res = quantize(circuit, loops, Z, types)
    return quantize(circuit, loops, Z, types, B=junction_flux_B(res, rings))


ZP_GAUGE = ("Ring flux on the junctions: custom B with +1/2 of the ring flux on each junction and capacitor entries"
            " solving G B = I (static flux: a gauge choice; the irrotational B would put it on L<q>b instead,"
            " giving flux_linear on phi and zeta).")


def zero_pi():
    """Symmetric zero-pi, scqubits ZeroPi with dEJ = dCJ = 0.

    scqubits' E_CS is the charging energy of C + C_J (theta), so the large capacitors
    have charging energy E_B = 1/(1/E_CS - 1/E_CJ) = E_CS E_CJ/(E_CJ - E_CS)
    (FullZeroPi's E_Cz); physical for E_CS < E_CJ.
    The circuit also has the harmonic zeta mode, exactly decoupled at symmetry;
    scqubits' ZeroPi (and zero_pi.json) omits it.
    """
    EJ, EL, ECJ, ECS = pos("E_J E_L E_CJ E_CS")
    xB = pos("x_B")
    vx = real("varphi_ext")
    circuit = Circuit(zero_pi_branches(1, (1, 2, 3, 4), (ECJ, ECJ), (xB, xB), (EL, EL), (EJ, EJ)))
    loops, rings = ring_loops(circuit, {1: vx})
    res = quantize_junction_gauge(circuit, loops, rings, ZP_M, ("c", "e", "h", "cy"))
    save("zero_pi", res, ["theta", "phi", "zeta"], [
        "Symmetric zero-pi as in circuit-quantization tests/circuits.py zero_pi(): J1 (1->2), J2 (3->4), L1 (2->3), L2 (4->1),"
        " C1 (1->3), C2 (2->4), floating; named J1a, J1b, L1a, L1b, C1a, C1b here.",
        "Elements: junction C = ghz_to_farad(E_CJ), EJ = E_J; L = ghz_to_henry(E_L);"
        " large C = ghz_to_farad(E_CS*E_CJ/(E_CJ - E_CS)), so that C + C_J has charging energy E_CS (scqubits' ECS).",
        "Z_full columns (theta, phi, zeta, Sigma) = notes eq:zp_Z with theta, phi negated and zeta divided by sqrt(2);"
        " Sigma is cyclic (floating circuit).",
        "zeta is a decoupled harmonic mode (scqubits ZeroPi drops it; zero_pi.json has no zeta).",
        ZP_GAUGE,
        "subs: x_B -> E_CS*E_CJ/(E_CJ - E_CS), n_g1 -> n_g.",
        "constant 0: scqubits' potential adds 2 E_J (zero_pi.json keeps it).",
    ], subs={xB: ECS * ECJ / (ECJ - ECS), real("n_g1"): real("n_g")})


def full_zero_pi():
    """Asymmetric zero-pi with zeta, against scqubits FullZeroPi (full_zero_pi.json).

    Element mapping, chosen so that the circuit's Hamiltonian agrees with the
    fixture to first order in the asymmetries (verified symbolically by series):
      J1a (phase theta + phi): EJ = E_J (1 + dE_J/2), C = C_J (1 - dC_J/2);
      J1b (phase -theta + phi): EJ = E_J (1 - dE_J/2), C = C_J (1 + dC_J/2);
      C1a: C (1 + dC/2), C1b: C (1 - dC/2);
      L1a: inductive energy E_L (1 - dE_L/2), L1b: E_L (1 + dE_L/2);
    with C_J = ghz_to_farad(E_CJ) and C = ghz_to_farad(E_CS E_CJ/(E_CJ - E_CS)).
    E_L, the junction energies and the offsets are exact (linear in dE_L, dE_J);
    E_C is the exact inverse capacitance matrix, which differs from scqubits' model
    at second order in (dC_J, dC): no lumped circuit has the fixture's E_C for any
    (dC_J, dC) != 0 (an exact realization needs a negative capacitance in a
    b->c or d->a branch, and the diagonal entries differ even with one asymmetry).
    E_Cz is not a circuit parameter: the circuit fixes it as E_CS E_CJ/(E_CJ - E_CS).
    Sign of dC_J: the fixture follows scqubits 4.3.1's source (-2 ECS dCJ (i d/dphi) n_theta),
    which this mapping reproduces with J1a carrying E_J (1 + dE_J/2) but C_J (1 - dC_J/2);
    scqubits documents dEJ = (EJ1-EJ2)/EJavg and dCJ = (CJ1-CJ2)/CJavg on the same
    junction 1 (here J1a), so relative to dE_J this dC_J has the opposite sign to that
    definition (the docstring's +2 E_CS dC_J d_phi d_theta is the consistent sign).
    The sign is observable when dE_J != 0; keeping it is deliberate (Eli, 2026-10-19).
    """
    EJ, EL, ECJ, ECS = pos("E_J E_L E_CJ E_CS")
    dEJ, dCJ, dC, dEL = real("dE_J dC_J dC dE_L")
    xJ = pos("x_J1 x_J2")
    xB = pos("x_B1 x_B2")
    yL = pos("y_L1 y_L2")
    zJ = pos("z_J1 z_J2")
    vx = real("varphi_ext")
    circuit = Circuit(zero_pi_branches(1, (1, 2, 3, 4), xJ, xB, yL, zJ))
    loops, rings = ring_loops(circuit, {1: vx})
    res = quantize_junction_gauge(circuit, loops, rings, ZP_M, ("c", "e", "h", "cy"))
    EB = ECS * ECJ / (ECJ - ECS)
    subs = {xJ[0]: ECJ / (1 - dCJ / 2), xJ[1]: ECJ / (1 + dCJ / 2),
            xB[0]: EB / (1 + dC / 2), xB[1]: EB / (1 - dC / 2),
            yL[0]: EL * (1 - dEL / 2), yL[1]: EL * (1 + dEL / 2),
            zJ[0]: EJ * (1 + dEJ / 2), zJ[1]: EJ * (1 - dEJ / 2),
            real("n_g1"): real("n_g")}
    save("full_zero_pi", res, ["theta", "phi", "zeta"], [
        "Asymmetric zero-pi: the zero_pi.json circuit with distinct elements (see full_zero_pi() in generate_specs.py).",
        "J1a (theta + phi): EJ = E_J (1 + dE_J/2), C = C_J (1 - dC_J/2); J1b (-theta + phi): EJ = E_J (1 - dE_J/2), C = C_J (1 + dC_J/2);"
        " C1a = C (1 + dC/2), C1b = C (1 - dC/2); L1a energy E_L (1 - dE_L/2), L1b energy E_L (1 + dE_L/2);"
        " C_J = ghz_to_farad(E_CJ), C = ghz_to_farad(E_CS*E_CJ/(E_CJ - E_CS)).",
        "Z_full as zero_pi.json (zeta is scqubits' FullZeroPi coordinate). " + ZP_GAUGE,
        "E_C is the circuit's exact inverse capacitance: it matches scqubits' first-order FullZeroPi model at dC_J = dC = 0"
        " and differs from it at second order in (dC_J, dC). E_Cz is not independent: E_Cz = E_CS*E_CJ/(E_CJ - E_CS).",
        "dC_J sign (deliberate, Eli 2026-10-19): as in full_zero_pi.json, scqubits 4.3.1's source sign, opposite to Groszkowski et al.,"
        " arXiv:1708.02886v1, eq:Hzeropi1 (+2 E_CS dC_J d_phi d_theta, dX = (X1 - X2)/X): J1a carries E_J (1 + dE_J/2) but"
        " C_J (1 - dC_J/2), so dC_J here = -dC_J of the paper and of circuit-quantization; dE_J agrees.",
        "constant 0: no 2 E_J (scqubits' potential), no -E_zeta/2 (FullZeroPi's E_zeta a^dag a), no offset-charge correction.",
    ], subs=subs)


def two_zero_pi():
    """Two symmetric zero-pis coupled through the pure n_theta1 n_theta2 coupler.

    The coupler is scheme (a) of circuit-quantization's tests/circuits.py
    coupled_zero_pis: nodes a, d of each qubit couple to island 10 and nodes b, c
    to island 9, each through a capacitor of charging energy x_c (no C_0 to ground:
    the whole circuit floats).  With xJ_q, xB_q the charging energies of qubit q's
    junction and large capacitances, circuit-quantization gives (exactly)
      E_C[phi_q, phi_q] = 1/(1/x_c + 2/xJ_q),
      the theta block = (1/2) K^-1, K = [[1/xJ_1 + 1/xB_1 + k, -k], [-k, 1/xJ_2 + 1/xB_2 + k]], k = 1/(4 x_c),
    and zeta_q decoupled.  The fixture's theta block [[E_CS1/2, g/8], [g/8, E_CS2/2]]
    and E_C[phi_q, phi_q] = E_CJq/2 then fix, with D = E_CS1 E_CS2/4 - g^2/64,
      x_c = 4 D/g,
      1/xJ_1 = 1/E_CJ1 - g/(8 D),
      1/xB_1 = (E_CS2/4 + g/16)/D - 1/E_CJ1,
    and the same for qubit 2 with E_CS1.  At g -> 0 these reduce to zero_pi()'s values.
    Physical (positive capacitances) for small g, e.g. g < E_CS1 E_CS2/(2 max E_CJq).
    """
    vxs = {}
    branches = []
    aux = {}
    for q, nodes in ((1, (1, 2, 3, 4)), (2, (5, 6, 7, 8))):
        EJ, EL = pos(f"E_J{q} E_L{q}")
        xJ, xB = pos(f"x_J{q} x_B{q}")
        aux[q] = (xJ, xB)
        vxs[q] = real(f"varphi_ext{q}")
        branches += zero_pi_branches(q, nodes, (xJ, xJ), (xB, xB), (EL, EL), (EJ, EJ))
    xc = pos("x_c")
    for q, (a, b, c, d) in ((1, (1, 2, 3, 4)), (2, (5, 6, 7, 8))):
        branches += [Branch(f"Cc{q}_{n}", n, 10, C=cq.ghz_to_farad(xc)) for n in (a, d)]
        branches += [Branch(f"Cc{q}_{n}", n, 9, C=cq.ghz_to_farad(xc)) for n in (b, c)]
    circuit = Circuit(branches)
    # Columns theta1, phi1, theta2, phi2, zeta1, zeta2; then free Sigma1 - Sigma2, the two
    # coupler-island directions, and the cyclic all-ones column.
    Z = sym.zeros(10, 10)
    for q, rows in ((0, slice(0, 4)), (1, slice(4, 8))):
        Z[rows, 2 * q] = ZP_M[:, 0]
        Z[rows, 2 * q + 1] = ZP_M[:, 1]
        Z[rows, 4 + q] = ZP_M[:, 2]
    Z[:, 6] = sym.Matrix([1, 1, 1, 1, -1, -1, -1, -1, 0, 0])
    Z[:, 7] = sym.Matrix([0] * 8 + [1, -1])
    Z[:, 8] = sym.Matrix([0] * 8 + [1, 1])
    Z[:, 9] = sym.ones(10, 1)
    types = ("c", "e", "c", "e", "h", "h", "fr", "fr", "fr", "cy")
    loops, rings = ring_loops(circuit, vxs)
    res = quantize_junction_gauge(circuit, loops, rings, Z, types)
    ECJ1, ECS1, ECJ2, ECS2 = pos("E_CJ1 E_CS1 E_CJ2 E_CS2")
    g = real("g")
    D = ECS1 * ECS2 / 4 - g**2 / 64
    subs = {xc: 4 * D / g,
            aux[1][0]: 1 / (1 / ECJ1 - g / (8 * D)), aux[1][1]: 1 / ((ECS2 / 4 + g / 16) / D - 1 / ECJ1),
            aux[2][0]: 1 / (1 / ECJ2 - g / (8 * D)), aux[2][1]: 1 / ((ECS1 / 4 + g / 16) / D - 1 / ECJ2),
            real("n_g3"): real("n_g2")}
    save("two_zero_pi", substituted(res, subs), ["theta1", "phi1", "theta2", "phi2", "zeta1", "zeta2"], [
        "Two symmetric zero-pis (nodes 1-4 and 5-8, branches J<q>a, J<q>b, L<q>a, L<q>b, C<q>a, C<q>b as in zero_pi.json)"
        " coupled by scheme (a) of circuit-quantization tests/circuits.py coupled_zero_pis: Cc<q>_<n> from nodes 1, 4, 5, 8 to"
        " island 10 and from nodes 2, 3, 6, 7 to island 9; no capacitance to ground (floating).",
        "Element charging energies (see two_zero_pi() in generate_specs.py), D = E_CS1*E_CS2/4 - g^2/64: coupler x_c = 4 D/g;"
        " qubit 1 junction 1/(1/E_CJ1 - g/(8 D)), large C 1/((E_CS2/4 + g/16)/D - 1/E_CJ1); qubit 2 likewise with E_CS1.",
        "These give exactly E_C[theta block] = [[E_CS1/2, g/8], [g/8, E_CS2/2]] and E_C[phi_q,phi_q] = E_CJq/2.",
        "Z_full: each qubit's columns as zero_pi.json, ordered theta1, phi1, theta2, phi2, zeta1, zeta2;"
        " non-dynamical: Sigma1 - Sigma2, the coupler-island directions (free), all-ones (cyclic).",
        "zeta1, zeta2 are decoupled harmonic modes (two_zero_pi.json has none). " + ZP_GAUGE,
        "Sector labels n_g1 (theta1) and n_g3 (theta2) renamed n_g1, n_g2.",
        "The substitution is applied before to_spec (substituted() in generate_specs.py, numerify plus sympy.together),"
        " because to_spec's cancel of the substituted E_C does not finish; the coefficients are the same.",
        "constant 0: scqubits' potential adds 2 E_J1 + 2 E_J2 (two_zero_pi.json keeps them).",
    ])


def cos2phi_not_expressible():
    """Why cos2phi.json has no generated counterpart.

    Its Hamiltonian has one compact mode and the cosines -E_2 cos(2 phi) - E_1 cos(phi).
    In a lumped circuit a junction contributes -E_J cos(w^T varphi + offset), where w^T
    is its incidence row applied to the retained columns of Z.  Two junctions whose
    phases are 2 phi and phi have non-parallel incidence rows (parallel rows of a 0/+-1
    incidence matrix are equal up to sign), so they span two node-space directions;
    every direction a junction participates in is compact or extended (mode_types),
    so such a circuit has at least two dynamical modes.  A cos(2 phi) element (the kite,
    or the rhombus in its symmetric limit) arises only after eliminating other modes,
    which a kronfree-spec export does not do.
    """


if __name__ == "__main__":
    transmon()
    transmons(2)
    transmons(3)
    fluxonium()
    zero_pi()
    full_zero_pi()
    two_zero_pi()
