# The KITE (cos 2φ transmon) built through circuit-quantization, with its parameter values and three frames.
# A helper, not a test file: test files include it.
using PythonCall, LinearAlgebra

"""
    kite(; cb_flux=0) -> (res::Py, θ::NamedTuple, frames::NamedTuple)

The KITE of `kite` in circuit-quantization's `examples/product_state_demo.py`, quantized in the source's
`(varphi, phi_Sigma, phi_Delta)` frame with mode types `(c, e, e)`.
`θ` holds every parameter of the circuit by name at the values of `examples/hierarchical_kite.py`, at half flux
(`delta = 0`) and zero offset charge.
`cb_flux` is the phase threading the purely capacitive loop `cb` (a number, or a SymPy symbol to give that loop a flux parameter that the exported spec does not contain).
`frames` has `published` (the identity), `normal` (`normal_mode_transformation` at `θ`) and `optimum` (`S_OPT` of
`examples/hierarchical_kite.py`).
"""
function kite(; cb_flux=0)
    cq, sympy = circuit_quantization(), pyimport("sympy")
    E_J, E_CJ, E_CS, E_L = sympy.symbols("E_J E_CJ E_CS E_L"; positive=true)
    eps, delta = sympy.Symbol("epsilon"; real=true), sympy.Symbol("delta"; real=true)
    L(E) = cq.ghz_to_henry(E)
    C(E) = cq.ghz_to_farad(E)
    circuit = cq.Circuit(pylist([
        cq.Branch("L1", 1, 2; L=L(2 * E_L)),
        cq.Branch("L2", 1, 3; L=L(2 * E_L)),
        cq.Branch("CS", 1, 6; C=C(E_CS)),
        cq.Branch("J1", 2, 4; C=C(E_CJ) * (1 + eps), EJ=E_J * (1 + eps)),
        cq.Branch("J2", 3, 5; C=C(E_CJ) * (1 - eps), EJ=E_J * (1 - eps)),
        cq.Branch("L3", 5, 6; L=L(2 * E_L)),
        cq.Branch("L4", 4, 6; L=L(2 * E_L)),
    ]); ground=6)
    cycles = pydict(Dict("kite" => pydict(Dict("L1" => 1, "J1" => 1, "L4" => 1, "L2" => -1, "J2" => -1, "L3" => -1)),
                         "cb" => pydict(Dict("CS" => 1, "L1" => -1, "J1" => -1, "L4" => -1))))
    fluxes = pydict(Dict("kite" => cq.phase_to_weber(sympy.pi + delta), "cb" => cq.phase_to_weber(cb_flux)))
    loops = cq.Loops.from_branch_dict(circuit, cycles, fluxes)
    q, h = sympy.Rational(1, 4), sympy.Rational(1, 2)
    shares = pydict(Dict("kite" => pydict(Dict("L1" => q, "L4" => q, "L2" => -q, "L3" => -q, "CS" => h)),
                         "cb" => pydict(Dict("CS" => 1))))
    B = cq.B_from_shares(circuit, loops, shares)
    Z = sympy.Matrix(pylist([pylist(r) for r in ([1, 0, 0, 0, 0], [1, -h, h, 1, 0], [1, -h, -h, 0, 1],
                                                 [0, h, -h, 1, 0], [0, h, h, 0, 1])]))
    res = cq.quantize(circuit, loops, Z, pytuple(("c", "e", "e", "fz", "fz")); B)

    θ = (E_J=16.83, E_CJ=4.82, E_CS=0.072, E_L=1.27, epsilon=0.03, delta=0.0, n_g1=0.0)
    S_normal = pyconvert(Matrix{Float64}, cq.normal_mode_transformation(cq.numeric_hamiltonian(res, pyvalues(θ))))
    frames = (published=Matrix{Float64}(I, 3, 3),
              normal=S_normal,
              optimum=[1.0 0.2354 0.2306; 0.0 0.6962 0.7339; 0.0 -0.5719 0.629])
    return res, θ, frames
end

# Parameter values as the Python dict, keyed by name, that circuit-quantization's numeric functions take.
pyvalues(θ::NamedTuple) = pydict(Dict(string(k) => Float64(v) for (k, v) in pairs(θ)))
