"""
    KronFreeCircuits.Experimental

Features that work and are tested but whose interface is not settled, kept out of the main export list:

- Born–Oppenheimer reduction (`bo_surface`, `fit_surfaces`, `slow_spec`, …; `experimental/bo.jl`): solve a fast subsystem
  on a grid of the slow coordinates and hand the slow coordinates an effective potential.
- Mean field (`self_consistent`, `mean_field_solutions`, `mean_field_terms`, …; `experimental/meanfield.jl`) and basis
  augmentation from its solutions (`mean_field_augment`; the augmentation itself, `KronFreeCircuits.augment`, is core).
- DMRG of the root (`dmrg_levels`), a thin wrapper on ITensorMPS, available once `ITensorMPS` is loaded.

`using KronFreeCircuits.Experimental` brings these names into scope.
"""
module Experimental

using ..KronFreeCircuits: periodic_grid, BOSurface, bo_surface, fourier_coefficients, slow_spec, BOFit, fit_surfaces,
                          Expectations, straddling_terms, child_keys, expectations, pinned_expectations,
                          delocalized_expectations, mean_field_terms, MeanFieldChild, bare_child, mean_field_child,
                          product_energy, MeanFieldSolution, default_starts, self_consistent, mean_field_solutions,
                          mean_field_augment

export periodic_grid, BOSurface, bo_surface, fourier_coefficients, slow_spec, BOFit, fit_surfaces,
       Expectations, straddling_terms, child_keys, expectations, pinned_expectations, delocalized_expectations,
       mean_field_terms, MeanFieldChild, bare_child, mean_field_child, product_energy, MeanFieldSolution, default_starts,
       self_consistent, mean_field_solutions, mean_field_augment, dmrg_levels

"""
    dmrg_levels(H::MPO; nlevels=1, weight, init=nothing, sweeps=nothing, observer=nothing, kwargs...)
        -> (; energies, states, variances, overlaps, observers)

The lowest `nlevels` eigenpairs of the MPO `H` by DMRG (ITensorMPS), a thin wrapper: the ground state by
`ITensorMPS.dmrg(H, ψ_init; kwargs...)`, then each excited state `k` by `ITensorMPS.dmrg(H, [ψ_0, …, ψ_{k-1}], ψ_init;
weight, kwargs...)`. Building `H` (and its site indices) from a tree is the caller's: any MPO works. Needs
`using ITensorMPS` (a package extension); `H` and the states may live on the GPU (`adapt(CuArray, H)`), as ITensorMPS
allows.

`weight` (required when `nlevels > 1`) is the penalty on overlapping the states already found: the excited solve
minimizes `⟨ψ|H|ψ⟩ + weight Σ_j |⟨ψ_j|ψ⟩|²`. It must exceed the gap between the state sought and the states it should not
fall back to, or a mixture with an earlier state can come out lower than the true next level; much larger than the
spectrum's scale only slows convergence and costs precision. A few times the largest gap of interest, in the units of
`H`, is the usual choice (for circuits in GHz with levels within a few GHz, 10 to 100). See the ITensorMPS documentation
of `dmrg` (its "excited states" section) for the method. The returned `overlaps` let you check the choice: a weight that
is too small shows as overlaps with earlier states that are not small.

- `init`: the start of every level: `nothing` for a random MPS on the MPO's site indices (`random_mps`, bond dimension
  10), an `MPS`, or a vector of state names that the sites' type defines (`MPS(sites, init)`); or a function `k -> start`
  returning one of these for level `k` (e.g. a product state for the ground state and random starts above it). On the
  GPU, return an `MPS` already on the device (the random and named starts are built on the CPU).
- `sweeps`: an ITensorMPS `Sweeps`, passed to every `dmrg` call (else give `nsweeps`, `maxdim`, … as keywords).
- `observer`: a function `k -> observer` (e.g. `k -> DMRGObserver(; energy_tol=1e-10)`), called once per level. A
  function, not an observer, because an observer keeps the energies it has seen: one shared by the levels would carry
  the ground state's history into the excited solves and stop them early. The observers are returned.
- `kwargs` go to every `dmrg` call unchanged (`nsweeps`, `maxdim`, `cutoff`, `noise`, `eigsolve_krylovdim`, `outputlevel`, …).

Returns `energies` (in the order found, real, of the states themselves: `dmrg`'s own value at `k > 1` includes the
penalty), `states` (the MPSs), `variances` (`⟨H²⟩ − ⟨H⟩²` per state, a convergence check in the units of `H` squared),
`overlaps` (`|⟨ψ_i|ψ_j⟩|` for `i ≠ j`, a matrix with zeros on the diagonal) and `observers` (`nothing` per level when
none was given).
"""
function dmrg_levels end

end # module Experimental
