# Experimental

Features that work and are tested but whose interface is not settled live in `KronFreeCircuits.Experimental`, kept out of the main export list:

```julia
using KronFreeCircuits, KronFreeCircuits.Experimental
```

The examples on this page are not run when the documentation is built: they take minutes, not seconds.
The fixtures they load are in the repository's `test/fixtures`.

## Born–Oppenheimer reduction

The Born–Oppenheimer pieces live in `KronFreeCircuits.Experimental` (not exported from the package; `using KronFreeCircuits.Experimental`).
A slow mode can be promoted to a parameter, the fast modes diagonalized on a grid of its coordinate, and the resulting surface fitted to a one-mode reduced spec.
The zero-pi of Di Paolo et al. (arXiv:1811.07921v1, Appendix B), with θ slow and φ fast:

```julia
using KronFreeCircuits, KronFreeCircuits.Experimental

p = promote_mode(load_spec("test/fixtures/zero_pi_bo.json"), :theta)        # fast spec over φ; θ is a parameter
leaf = Node([:phi => PhaseGrid(-12π, 12π, 481)]; solver=Dense(), compute=2, keep=lowest(2))
θ = (E_Cphi=40 / (16 * 0.165), E_Ctheta=0.007, E_J=6.6, E_L=0.04)           # GHz
S = [bo_surface(leaf, p, merge(θ, (varphi_ext=2π * f,)), periodic_grid(32; start=-π/2)) for f in 0:0.125:0.5]
F = fit_surfaces(S, [:constant => ["1", "cos(varphi_ext)", "cos(2*varphi_ext)", "cos(3*varphi_ext)", "cos(4*varphi_ext)"],
                     (:cos, 2) => ["1", "cos(varphi_ext)"], (:cos, 1) => ["cos(varphi_ext/2)"]])
F.constants, F.stderr, F.rms_residual                                        # E_α = c[6], E_β = −c[7], E_γ = c[8] in GHz
save_spec("zero_pi_reduced.json", F.spec; metadata=F.metadata)              # a kronfree-spec with its _bo provenance
```

`slow_spec(S[1])` gives the slow spec of one surface from its Fourier coefficients instead of a fit.
Promotion is a spec-to-spec transformation; a charge coupling between slow and fast modes is rejected, and only the single lowest surface is used (no derivative couplings).
The reproduction (`bench_archive/born_oppenheimer/zero_pi.jl`, results in `bench_archive/born_oppenheimer/results.md`) matches the published E_α and E_γ to 0.12% with E_Cφ derived from the plasma-frequency identity.

## Mean-field bases

In a frame that mixes modes, a Josephson cosine straddles several children, so a bare leaf never sees a junction and its eigenstates are a poor basis for the parent.
A child can instead be solved in the mean field of its siblings, and the mean-field states of several self-consistent solutions (one per well) can become the basis through which the child is presented to its parent.
The parent's operator is unchanged: it is projected on the new basis, so only the truncation changes.
This is the self-consistent product-state method of the circuit-quantization paper's Appendix C.

Mean field lives in `KronFreeCircuits.Experimental`; the basis augmentation it feeds (`augment`, `union_basis`, `BasisSource`, `AugmentedChild`, `augmented`) is core and public but not exported, so the example imports it by name.
The loop is a few lines.
The KITE in the paper's optimum frame (`test/fixtures/kite/optimum.json`, a spec of constants), with one leaf per mode keeping 20 states and two wells feeding each leaf's basis:

```julia
using KronFreeCircuits, KronFreeCircuits.Experimental
using KronFreeCircuits: augment, union_basis, BasisSource

spec = load_spec("test/fixtures/kite/optimum.json")
θ = (;)
terms = circuit_terms(spec)
leaves = [Node([m => b]; solver=Dense(), compute=20, keep=lowest(20), arithmetic=:complex)
          for (m, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5, when_capped=:allow)]
root = Node(leaves; solver=LOBPCG(tol=1e-10), compute=8, keep=lowest(8))

# the self-consistent loop from each start; the distinct solutions, lowest product energy first
names = [m.name for m in spec.modes]
starts = [Dict(zip(names, (0.0, 0.0, 0.0))), Dict(zip(names, (π, -π/2, π))), nothing]   # nothing: delocalized
bare = IdDict(leaf => bare_child(root, terms, θ, leaf) for leaf in leaves)    # each leaf's bare solve, formed once
sols = mean_field_solutions(root, spec, θ; starts, bare)
wells = filter(s -> s.energy - sols[1].energy <= 0.5, sols)                               # two, at -9.4294736 GHz

# each leaf's basis: the wells' m lowest mean-field states, merged level by level; its bare operator projected on it
m = 12
aug = map(leaves) do leaf
    sources = [BasisSource("well $i", s.children[leaf].solved.vectors[:, 1:m], s.children[leaf].solved.energies[1:m])
               for (i, s) in enumerate(wells)]
    U, report = union_basis(sources; m)                                                     # report: taken and dropped
    augment(root, terms, θ, leaf, bare[leaf], U, report)
end
sys = materialize(root, spec, θ; augmented=aug)
energies(sys)
```

The wrapper does the same in one call (the solutions within `window = 0.5` GHz of the lowest, at most four, each child's first `m` mean-field states per solution; `bare = true` adds the bare ladder and `cap` limits each solution's share):

```julia
r = mean_field_augment(root, spec, θ; m=12, starts)    # (; children, solutions, used, energy, unconverged)
sys = materialize(root, spec, θ; augmented=r.children)
```

The same bare solves go to `mean_field_solutions` and to `augment`: the solutions' states are coordinates on the bare solve's children, so `U` must be read on that solve, not on a new one.
Both give a largest error of the four lowest levels of 2.2679e-3 GHz against the paper's reference levels, the value of the table below (run on 2026-11-11).
The solves print `fix_phases!` notes for the compact leaf's degenerate charge pairs and PhaseGrid edge warnings for kept states 18 to 20, which no union of 12 states reaches.

`suggest_bases` sizes each grid from the mode's zero-point width and junction energies, and caps it at `max_points`.
A binding cap (the formula asks for more points) keeps the extent with a larger step, so the mode is under-resolved and an energy on that grid is no longer an upper bound of the resolved problem's: an optimizer can then go below the true ground level.
By default it warns; `when_capped = :error` refuses, and `when_capped = :allow` caps silently, as here, where 120 points are the figure's own choice.

The other pieces: `self_consistent` (one run of the loop, also from a previous solution as a warm start), `default_starts`, `mean_field_child`, `bare_child`, `mean_field_terms`, `product_energy`, `expectations`, `pinned_expectations` and `delocalized_expectations`, and `edge_weight` for a grid leaf's edge support.
An augmented child (`AugmentedChild`) keeps its union basis, its bare terms and the `θ` it was built at, and `augmented(sys)` returns those a system was built with; `materialize` refuses it at a `θ` that differs in a parameter its bare terms depend on.
Mean field acts one level deep, among the children of one node, and mean-field and augmented nodes run in complex arithmetic.
`KronFreeCircuits.operator(sys, node)` throws for an augmented node, and derivatives through the mean-field bases are not provided.

Figure 8(b) of the paper, reproduced by `bench_archive/kite_meanfield/levels.jl`: the largest error of the four lowest levels, in GHz, with two-well leaves, against the reference calculation's (ratio in brackets):

| frame | m = 8 | m = 12 | m = 16 | m = 20 |
|---|---|---|---|---|
| published | 6.9179e-01 (1.0000) | 2.2134e-01 (1.0000) | 3.7649e-02 (0.9986) | 6.5900e-04 (0.9726) |
| normal modes | 6.2176e-01 (1.0000) | 1.9890e-01 (1.0000) | 3.6005e-02 (1.0000) | 8.0124e-04 (1.0000) |
| optimum | 3.6677e-02 (1.0000) | 2.2679e-03 (1.0000) | 2.6093e-04 (1.0000) | 2.0717e-05 (1.0000) |

In the normal-mode and optimum frames the levels match the reference calculation's to 1.5e-13 GHz.
In the published frame the two wells' states coincide on one mode up to the loop's convergence residue, whose Gram–Schmidt remainders straddle the `1e-6` drop threshold, so the union depends on where the loop stopped and on round-off; that is the inferred cause of the ratios 0.9986 and 0.9726.
`bench_archive/kite_meanfield/optimize.jl` optimizes the frame from the identity on the product energy and ends at a frame as good as the paper's optimum (two wells at −9.429473654505 GHz against −9.429473599561 on the same bases), after 150 iterations, the cap, without meeting its gradient tolerance.
Both scripts, their runs and the measured details are in `bench_archive/kite_meanfield/README.md`.


## DMRG

`dmrg_levels(H::MPO; nlevels, weight, ...)` finds the lowest levels of an MPO by DMRG, a thin wrapper on ITensorMPS that is available once `using ITensorMPS` loads the package extension.
Building the MPO from a tree is the caller's; see its docstring in the [API reference](@ref "Experimental module").
