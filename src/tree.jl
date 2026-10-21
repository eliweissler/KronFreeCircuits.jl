abstract type Solver end

"Full dense diagonalization."
struct Dense <: Solver end

"""
    Lanczos(; tol=1e-10, krylovdim=0, maxiter=1000, verbosity=0)

Iterative solver; `krylovdim == 0` lets the solver choose.
`maxiter` bounds the restarts (grid Hamiltonians are stiff: a 9k-dimensional zero-pi leaf needs several hundred) and `verbosity` is passed to KrylovKit (0 is silent).
A single-vector Krylov space does not resolve repeated eigenvalues (KrylovKit documents this); use `LOBPCG` or `Dense` where exact degeneracies are expected.
"""
struct Lanczos <: Solver
    tol::Float64
    krylovdim::Int
    maxiter::Int
    verbosity::Int
end
Lanczos(; tol=1e-10, krylovdim=0, maxiter=1000, verbosity=0) = Lanczos(tol, krylovdim, maxiter, verbosity)

"""
    LOBPCG(; tol=1e-8, maxiter=1000, preconditioner=:ritz, eta=0.01, guard=4, seed=:product, on=Array, callback=nothing)

Block iterative solver: the JuliaMolSim LOBPCG eigensolver (LOBPCGEigensolver.jl), used
through its public interface only.

- `tol`: convergence threshold on each residual norm `‖H v − λ v‖` of a unit vector, in the
  Hamiltonian's energy units; `maxiter` bounds the iterations. Pairs that have not converged
  within `maxiter` raise an error naming the unconverged count and the largest residual; they
  are never returned silently and never retried.
- `preconditioner`: `:ritz` (adaptive per-Ritz Jacobi: column `j` of the residual is divided by
  `hypot(D − θ_j, eta)`, with `D` the diagonal of `H` and `θ_j` the column's latest Ritz value),
  `:jacobi` (fixed shifted Jacobi, `1 / (D − σ)` with `σ = minimum(D) − 1`) or `:none`.
- `eta`: the regularization of `:ritz`, in the Hamiltonian's energy units. The default is taken
  from the audit fixture (GHz) and must be set for other energy scales.
- `guard`: extra columns appended to every block (cold or warm start; random with `seed=:random`, product states with `:product`). They are extra
  states: converged with the rest of the block (a guard still unconverged when `maxiter` runs
  out gives a warning), then discarded, and the lowest `m` pairs of the
  sorted block are returned. Guards make exact excited-state seeds and sector-confined seeds
  recoverable in practice; they are not a completeness guarantee. The guard shrinks so that the
  block fits `N ≥ 3(m + guard) + 5`. On a stiff grid leaf the guard columns can set the
  iteration count and emit the guard warning even on a warm start; `guard=0` avoids this for warm leaf sweeps at the documented completeness cost.
  A guard stopping rule is an open item (spec §20c execution notes).
- `seed`: `:product` starts the whole block of `m + guard` columns from
  [`product_seed`](@ref KronFreeCircuits.product_seed) (the lowest product states of the separable part, no
  random columns), `:random` from a random block. The product seed is perturbed like `X0` (below). A previous
  solution's vectors (`X0`) win over either for the first `m` columns; with `:product` a warm start's guard
  columns are the next `guard` product states by energy (random with `:random`). The cold retry after a failed
  warm start is always random.
- `on`: the array storage the block lives on (`Array`, or a device array type).
- `callback`: `nothing` or a function called with the solver's per-iteration `info` named tuple;
  its return value is ignored, and an exception it throws propagates unchanged.

The operator dimension must satisfy `N ≥ 3m + 5`; smaller problems call for `Dense()`.
With guards, a warm start reduces the work in `H` columns (`n_matvec`) rather than necessarily the
iteration count: the guard columns converge alongside the seeded columns, but once the
seeded columns lock, each remaining iteration applies `H` to only a few active columns.
Warm-start columns are perturbed by a vector of norm `sqrt(eps)` times the column's norm (a
fixed pseudo-random direction) so exact eigenvector seeds do not present an exactly zero residual; a warm-start block on which
the eigensolver still fails is replaced once by a random block, with a warning. A failure on the (perturbed) product-state seed is retried the same way, logged at debug level only.
"""
struct LOBPCG <: Solver
    tol::Float64
    maxiter::Int
    preconditioner::Symbol
    eta::Float64
    guard::Int
    seed::Symbol
    on::Any
    callback::Any
    function LOBPCG(tol::Real, maxiter::Integer, preconditioner::Symbol, eta::Real, guard::Integer, seed::Symbol, on, callback)
        tol > 0 && isfinite(tol) || throw(ArgumentError("LOBPCG: tol must be positive and finite, got $tol"))
        maxiter >= 1 || throw(ArgumentError("LOBPCG: maxiter must be at least 1, got $maxiter"))
        preconditioner in (:ritz, :jacobi, :none) ||
            throw(ArgumentError("LOBPCG: preconditioner must be :ritz, :jacobi or :none, got :$preconditioner"))
        eta > 0 && isfinite(eta) || throw(ArgumentError("LOBPCG: eta must be positive and finite, got $eta"))
        guard >= 0 || throw(ArgumentError("LOBPCG: guard must be nonnegative, got $guard"))
        seed in (:product, :random) || throw(ArgumentError("LOBPCG: seed must be :product or :random, got $(repr(seed))"))
        return new(tol, maxiter, preconditioner, eta, guard, seed, on, callback)
    end
end
LOBPCG(; tol=1e-8, maxiter=1000, preconditioner=:ritz, eta=0.01, guard=4, seed=:product, on=Array, callback=nothing) =
    LOBPCG(tol, maxiter, preconditioner, eta, guard, seed, on, callback)

"""
    ShiftInvert(; tol=1e-8, shift=:auto, shift_margin=5.0, inner_tol=1e-13, krylovdim=0, maxiter=100,
                block=1, slices=1, certify=:auto, maxnnz=10^8)

Leaf solver on the CPU (spec §20e decision 3): Lanczos on `(H − σI)⁻¹`, with `H − σI` factored
directly from the assembled sparse matrix `sparse(H; maxnnz)`.
Arithmetic follows the operator's element type: real when `eltype(H) <: Real`, complex otherwise.

- `tol`: the acceptance threshold on each true residual norm `‖H v − E v‖` of a unit vector, in the
  Hamiltonian's energy units, computed with the package operator. A pair above it raises an error
  naming the worst residual; pairs are never returned unvalidated. When `tol` lies below what double
  precision can deliver, the threshold is the roundoff floor `10 eps·‖H‖` instead (`‖H‖` taken as the
  larger of `gershgorin_bound(H)` and the lowest energy's magnitude; a debug message reports a pair
  accepted there), the rule `ChebyshevFilter` shares. A backward-stable
  solve, `Dense` included, leaves residuals of a few `eps·‖H‖`, and `‖H‖` grows like `1/h²` on a phase
  grid and like `ncut²` in a charge basis (7e5 on a 4000-point fluxonium grid, 1.6e7 at `ChargeBasis(2000)`
  with `E_C = 1`).
- `shift`: `:auto` puts `σ = E_min − shift_margin` below a short Lanczos estimate `E_min` of the
  lowest eigenvalue, so `H − σI` is positive definite and is factored by sparse Cholesky. If the
  estimate overshoots by more than `shift_margin` and the Cholesky fails, it is retried at
  `E_min − shift_margin·4^r` for `r = 1, 2, 3`, then at the Gershgorin lower bound of `H` minus
  `shift_margin` (a debug message names the shift that worked); only if every retry fails does the
  solve fall back to LU, with a warning. Slicing (`slices > 1`) does not retry.
  A number is used as `σ` itself; a shift at or above the estimate is interior, `H − σI` is factored
  by LU, and the `m` eigenvalues nearest `σ` are returned.
- `shift_margin`: the distance of the automatic shift below the estimate, in the Hamiltonian's
  energy units, and the base of the retry shifts above. The default is taken from the zero-pi
  fixtures (GHz) and must be set for other energy scales.
- `inner_tol`, `krylovdim`, `maxiter`: KrylovKit's tolerance on the inverted operator (not the
  acceptance criterion), Krylov dimension (`0` lets the solver choose `2m + 10`) and restart bound.
  The converged pairs are then polished: one step of block inverse iteration with the same
  factorization, and Rayleigh–Ritz with `H`. A Lanczos pair stops with Krylov residual `r` just under
  `inner_tol`, and `H v − E v = −(E − σ)(H − σ) r`, so components of `r` on high-lying states reach the
  true residual magnified by up to `|E − σ|·‖H‖` (up to 1e-5 on 8000-state leaves, with the vectors
  accurate to 1e-11); the step damps them to the roundoff floor, at about 3 % of the solve time.
- `block`: `1` (Lanczos) or `2` (block Lanczos, degeneracy insurance for exactly repeated
  eigenvalues). Block size 4 reported false convergence in KrylovKit 0.10.4 and is not offered.
  A block run that KrylovKit breaks off (the square root of a negative residual norm, on a nearly
  rank-deficient residual block) repeats as single-vector Lanczos.
- `slices`: spectrum slicing across threads. `1` is the single-shift solve above. An integer `k > 1`
  locates the range `[E_min, E_m]` by inertia counts (bisection on the negative pivots of
  `LDLᵀ(H − τI)`), cuts it into `k` contiguous intervals holding about `m / k` eigenvalues each, and
  solves each by Lanczos on `(H − σ_k I)⁻¹` with `σ_k` at its center (an `LU` factorization each),
  in parallel over `Threads.@threads` (serially on one thread). The inertia counts certify every
  interval: one returning a different number of states than its count raises naming it. `:auto` means
  `Threads.nthreads()` slices when there is more than one thread, and `1` otherwise. Slicing finds the
  lowest `m`, so `k > 1` requires `shift = :auto` (`:auto` with an explicit shift is one slice).
  BLAS and UMFPACK threads are not pinned inside the threaded loop, so many slices on a many-thread
  machine can oversubscribe the cores. Each inertia count at a cut costs two `LDLᵀ` factorizations
  (at `τ ± 10·tol`, to keep every cut clear of an eigenvalue).
- `certify`: count the eigenvalues below a cut `τ` in the first gap above the `m`-th computed energy
  (wider than `2·tol`, so a degenerate level is never cut) by the inertia of an `LDLᵀ` factorization of
  `H − τI`, and require it to equal the number of computed energies below `τ` (a completeness
  certificate). `:auto` (the default) certifies whenever the lowest `m` are sought: when `σ` lies below
  the spectrum (Cholesky), automatic or explicit, and with the automatic shift (the path `Auto()` takes)
  also on its LU fallback; an explicit interior shift returns the states nearest `σ` and is not certified.
  `true` certifies always (an interior shift then raises unless the states nearest `σ` are the lowest).
  Single-vector Lanczos resolves one copy of an exactly degenerate level per Krylov space, so it can
  return a 4-fold ground level as two copies, with passing residuals; the certificate detects the
  shortfall. With `σ` below the spectrum (Cholesky), the missing states (plus one) are then found by
  Lanczos on the inverse deflated against every computed vector, `x ↦ P (H − σI)⁻¹ P x` with
  `P = I − V V'`, merged with the computed ones by Rayleigh–Ritz, and the count repeats while each
  round recovers a state (a level with `c` copies takes about `c / block` rounds; a round that
  converges none raises); with `σ` not below the spectrum (LU) a short count raises at once. A
  certified solve also starts from fewer than `m + 1` converged pairs, when a few distinct levels
  exhaust the Krylov space, and recovers the rest the same way. The
  certificate costs one `LDLᵀ` factorization and one more Lanczos pair when it passes (two
  factorizations if the first breaks down; about +5 % on the 9331-state zero-pi leaf at m = 300), and
  one more Lanczos run and one or two factorizations per recovery round.
  `certify=false` runs no completeness check: a degenerate level may be undercounted, silently.
  A sliced solve is always certified (its count mismatch raises; it does not deflate).
- `maxnnz`: the nonzero cap of `sparse(H)`; a larger operator is refused before assembly.
"""
struct ShiftInvert <: Solver
    tol::Float64
    shift::Union{Symbol,Float64}
    shift_margin::Float64
    inner_tol::Float64
    krylovdim::Int
    maxiter::Int
    block::Int
    slices::Union{Int,Symbol}
    certify::Union{Bool,Symbol}
    maxnnz::Int
    function ShiftInvert(tol::Real, shift::Union{Symbol,Real}, shift_margin::Real, inner_tol::Real, krylovdim::Integer,
                         maxiter::Integer, block::Integer, slices::Union{Integer,Symbol}, certify::Union{Bool,Symbol}, maxnnz::Integer)
        tol > 0 && isfinite(tol) || throw(ArgumentError("ShiftInvert: tol must be positive and finite, got $tol"))
        shift === :auto || (shift isa Real && isfinite(shift)) ||
            throw(ArgumentError("ShiftInvert: shift must be :auto or a finite number, got $(repr(shift))"))
        shift_margin > 0 && isfinite(shift_margin) ||
            throw(ArgumentError("ShiftInvert: shift_margin must be positive and finite, got $shift_margin"))
        inner_tol > 0 && isfinite(inner_tol) ||
            throw(ArgumentError("ShiftInvert: inner_tol must be positive and finite, got $inner_tol"))
        krylovdim >= 0 || throw(ArgumentError("ShiftInvert: krylovdim must be nonnegative, got $krylovdim"))
        maxiter >= 1 || throw(ArgumentError("ShiftInvert: maxiter must be at least 1, got $maxiter"))
        block in (1, 2) || throw(ArgumentError("ShiftInvert: block must be 1 or 2 (block size 4 is not offered), got $block"))
        slices === :auto || (slices isa Integer && slices >= 1) ||
            throw(ArgumentError("ShiftInvert: slices must be :auto or an integer at least 1, got $(repr(slices))"))
        slices isa Integer && slices > 1 && shift isa Real &&
            throw(ArgumentError("ShiftInvert: slicing finds the lowest m and places its own shifts; " *
                                "slices = $slices requires shift = :auto, got shift = $shift"))
        certify isa Bool || certify === :auto ||
            throw(ArgumentError("ShiftInvert: certify must be :auto, true or false, got $(repr(certify))"))
        maxnnz >= 1 || throw(ArgumentError("ShiftInvert: maxnnz must be positive, got $maxnnz"))
        return new(tol, shift isa Real ? Float64(shift) : shift, shift_margin, inner_tol, krylovdim, maxiter, block, slices,
                   certify, maxnnz)
    end
end
ShiftInvert(; tol=1e-8, shift=:auto, shift_margin=5.0, inner_tol=1e-13, krylovdim=0, maxiter=100,
            block=1, slices=1, certify=:auto, maxnnz=10^8) =
    ShiftInvert(tol, shift, shift_margin, inner_tol, krylovdim, maxiter, block, slices, certify, maxnnz)

"""
    ChebyshevFilter(; tol=1e-8, degree=100, extra=40, maxiter=50, rescale_every=20, ortho=:svqb,
                    bound=:gershgorin, seed=:product, on=Array, callback=nothing)

Block leaf solver (spec §20e decision 3): Chebyshev-filtered subspace iteration (Zhou–Saad ChFSI) on
the package operator, on the host or a device. Each iteration applies to the unlocked columns of a
block of `nb = m + extra` columns (capped at the dimension) the degree-`degree` Chebyshev polynomial
of the first kind on the unwanted interval `[λ_cut, λ_max]` mapped to `[−1, 1]`, which damps that
interval and amplifies everything below it, then orthonormalizes and performs Rayleigh–Ritz.

- `tol`: the acceptance threshold on each true residual norm `‖H v − E v‖` of a unit vector, in the
  Hamiltonian's energy units, computed with the package operator. Columns lock (stop being filtered)
  in order from the bottom of the block once their Rayleigh–Ritz residual is below `tol / 2`; the
  iteration stops when the lowest `m + g` are locked (`g ≤ min(extra − 1, 4)` counted guard columns) and no
  unlocked Ritz value lies below the `m`-th locked energy. The returned pairs are the lowest `m` of all
  locked columns, accepted at `tol` with a fresh residual from a new application of the operator.
  Acceptance follows `ShiftInvert`'s rule, the alternative for the same node: when `tol` lies below
  the roundoff floor `10 eps·‖H‖`, the floor is the threshold instead and columns lock below half of it.
  Chebyshev's residuals stall higher than `ShiftInvert`'s polished ones on a fine phase grid (about 2.6,
  10 and 25 eps·‖H‖ at 1000, 4000 and 8000 points of the fluxonium grid, against 1 to 3), so at 8000
  points and `tol = 1e-8` it raises at `maxiter` where `ShiftInvert` succeeds (4000 points pass); use
  `ShiftInvert` on the CPU there, or loosen `tol`.
  No column locks, and the iteration does not stop, before the first filter application (unless the
  block spans the whole space, `nb = N`): the seed's own Rayleigh–Ritz is never accepted, since a seed
  spanning an invariant subspace has zero residuals there however far it lies from the lowest states.
- `degree ∈ 1:200`: the polynomial degree, i.e. block applies per iteration. An iteration lowers it
  so that the filter amplifies the lowest Ritz value of the block at most `1/√eps` over `λ_cut`
  (beyond that the upper columns of the block fall below the resolution of the Gram-based
  orthogonalization). On a narrow spectrum the cap binds hard (the three-oscillator test runs at
  degree 14 of 100); on a grid leaf with a large bound it binds moderately (the small zero-pi leaf
  runs at degree 65–81 of 100). The applied degree is reported to `callback` and in the `maxiter` error.
- `extra ≥ 2`: columns beyond `m` (they set the convergence rate and absorb near-degenerate states);
  at least two so at least one column serves as a convergence guard, since the top column sets the filter cut.
  Up to four of them double as guards that must converge before the iteration stops, as `LOBPCG`'s:
  `g = min(extra − 1, 4)`, one fewer than `extra` because the top column sets `λ_cut` and sits at the
  cut, where the filter does not converge it (hence `extra ≥ 2`: with `extra = 1` there is no guard, and
  a lower state outside the seed span can be missed with passing residuals). A guard is waived only
  while the last filter's log-gain at its Ritz value (`degree · acosh|x|`, `x` its image on the filter
  interval mapped to `[−1, 1]`) is below `0.1` times the log-gain at the `m`-th lowest Ritz value, the
  highest wanted state (so a deep ground state cannot waive the guards protecting the states above it):
  one (near-)degenerate with the top column gains about nothing and would never lock, while under a filter
  that is weak everywhere (a spectral bound far above the block) every guard keeps a ratio near 1 and
  is waited for. Guards near the cut converge slowly, so a small `extra` can take many more
  iterations; when guards are what remains unconverged at `maxiter`, the error advises raising
  `extra` or `maxiter`.
- `maxiter`: the bound on filter iterations; running out raises an error naming the converged count.
- `rescale_every`: every this many recurrence steps the pair `(T_{k−1}, T_k)` is divided column-wise
  by the norm of `T_k` (the recurrence is linear, so only column scales change), preventing overflow.
- `ortho`: `:svqb` (eigendecomposition of the unit-diagonal Gram matrix, two passes; directions
  with eigenvalue below `1e-12` times the largest are dropped and refilled with random vectors) or
  `:cholqr` (Cholesky-QR, two passes; a numerically rank-deficient block raises, advising `:svqb`).
- `bound`: the upper end `λ_max` of the filtered interval: `:gershgorin` uses
  [`gershgorin_bound`](@ref KronFreeCircuits.gershgorin_bound), a rigorous bound; a finite number is used as given. A
  number below the true largest eigenvalue amplifies the top of the spectrum; it is detected (a
  Ritz value above the bound, or a non-finite block) and raises an error naming the bound.
- `seed`: `:product` starts from [`product_seed`](@ref KronFreeCircuits.product_seed) (the lowest product states of the
  separable part) plus a fixed random admixture of relative size `1e-2` per column; `:random` from a
  random block. The admixture matters when the product states span an exactly invariant subspace that
  misses a lower state (a coupling acting only among higher product states): the filter's gain per
  iteration is capped near `1/√eps`, so a component of size `1e-2` along that state dominates a
  column within the first iteration or two, where a `√eps` one (`LOBPCG`'s perturbation) would not.
- `on`: the array storage the block lives on (`Array`, or a device array type).
- `callback`: `nothing` or a function called once per iteration, after locking, with the named
  tuple `(; iteration, λ, residuals, n_locked, λ_cut, λ_max, degree)` (host copies; `λ_cut` and
  `degree` are those of the last filter, `NaN` and `0` before the first); its return value is ignored.

`λ_cut` is the largest Ritz value `θ_nb` of the current block (initially of the seed's Rayleigh–Ritz):
every block direction is amplified, the lowest `m` the most, so the extra columns set the rate.
Results are host arrays.

There is no completeness certificate. A lower state outside the seed's span is found only through the
random admixture (or the random seed) and the guard columns: it is missed when its overlap with the
initial block is so small (below about `1e-2 · G^(−k)`, with `G ≈ 1/√eps` the capped gain per iteration
and `k` the iterations the lowest `m + g` columns take to lock) that it has not surfaced in a column
by then, and the stop rule then accepts the lowest states found. `ShiftInvert()` (inertia certificate,
on by default with the automatic shift) is the certified path.

The defaults (`seed=:product, extra=40, degree=100, tol=1e-8`) were tested against invariant-seed traps,
weak filters, crossing sweeps, degenerate blocks and deep references without a wrong result, but this is
tested, not proven. The stop is residual-based, so a state outside the product-seed span is found only if
the filter lifts its admixture of about `1e-2/√N` to `O(1)` before the `m` wanted and up to 4 guard
columns lock. This can fail silently with a weak filter (a low `degree`, or a spectral bound far above the
block) combined with a loose `tol`, or at a small `extra`. Use `tol ≤ 1e-6` and the default or a larger
`extra`, and `ShiftInvert()` (certified) or `Dense()` when completeness matters.
"""
struct ChebyshevFilter <: Solver
    tol::Float64
    degree::Int
    extra::Int
    maxiter::Int
    rescale_every::Int
    ortho::Symbol
    bound::Union{Symbol,Float64}
    seed::Symbol
    on::Any
    callback::Any
    function ChebyshevFilter(tol::Real, degree::Integer, extra::Integer, maxiter::Integer, rescale_every::Integer,
                             ortho::Symbol, bound::Union{Symbol,Real}, seed::Symbol, on, callback)
        tol > 0 && isfinite(tol) || throw(ArgumentError("ChebyshevFilter: tol must be positive and finite, got $tol"))
        1 <= degree <= 200 || throw(ArgumentError("ChebyshevFilter: degree must lie in 1:200, got $degree"))
        extra >= 2 || throw(ArgumentError("ChebyshevFilter: extra must be at least 2 so at least one column serves as a convergence " *
                                          "guard; the top column sets the filter cut and cannot be one, got $extra"))
        maxiter >= 1 || throw(ArgumentError("ChebyshevFilter: maxiter must be at least 1, got $maxiter"))
        rescale_every >= 1 || throw(ArgumentError("ChebyshevFilter: rescale_every must be at least 1, got $rescale_every"))
        ortho in (:svqb, :cholqr) || throw(ArgumentError("ChebyshevFilter: ortho must be :svqb or :cholqr, got $(repr(ortho))"))
        bound === :gershgorin || (bound isa Real && isfinite(bound)) ||
            throw(ArgumentError("ChebyshevFilter: bound must be :gershgorin or a finite number, got $(repr(bound))"))
        seed in (:product, :random) || throw(ArgumentError("ChebyshevFilter: seed must be :product or :random, got $(repr(seed))"))
        return new(tol, degree, extra, maxiter, rescale_every, ortho, bound isa Real ? Float64(bound) : bound, seed, on, callback)
    end
end
ChebyshevFilter(; tol=1e-8, degree=100, extra=40, maxiter=50, rescale_every=20, ortho=:svqb, bound=:gershgorin,
                seed=:product, on=Array, callback=nothing) =
    ChebyshevFilter(tol, degree, extra, maxiter, rescale_every, ortho, bound, seed, on, callback)

"""
    Auto(; dense_cutoff=1000, on=Array)

The default solver of every [`Node`](@ref): it picks a solver for the node at materialization, from
the node's kind, product dimension and storage (spec §20e decision 5).

- `Dense()` when the product dimension is at most `dense_cutoff`;
- otherwise, at a leaf (all children bases), `ShiftInvert()` when `on === Array` and
  `ChebyshevFilter(on=on)` on any other storage;
- otherwise, at a branch, `LOBPCG(preconditioner=:ritz, seed=:product, on=on)`.

The resolved solvers take their default settings; for other settings, pass the solver explicitly
(an explicit solver always wins). `compute == 0` (all eigenpairs) is valid only for a node that
resolves to `Dense`; otherwise `materialize` raises an `ArgumentError` naming the resolved solver.
`dense_cutoff = 1000` is the most conservative crossover measured on the benchmark machine: the iterative solvers win from about 700 to 1000 states on a transmon leaf and from 100 to 350 on the other shapes (`bench/dense_cutoff/results.md`). The warning for a large `Dense` node (above
10 000 states) applies to any `Dense` that runs, explicit or resolved (e.g. `Auto(dense_cutoff=20000)`
on a 15 000-state node).
"""
struct Auto <: Solver
    dense_cutoff::Int
    on::Any
    function Auto(dense_cutoff::Integer, on)
        dense_cutoff >= 0 || throw(ArgumentError("Auto: dense_cutoff must be nonnegative, got $dense_cutoff"))
        return new(dense_cutoff, on)
    end
end
Auto(; dense_cutoff=1000, on=Array) = Auto(dense_cutoff, on)

"""
    SelectionRule

A rule choosing which computed eigenstates a node keeps: `lowest`, `indices`, `connected` or `follow`.
A custom rule subtypes `SelectionRule` and implements `KronFreeCircuits.select(rule, ctx::SelectionContext)`,
returning the kept indices among the computed eigenpairs (in the order they are to be kept);
if it inspects ports it also implements `KronFreeCircuits.ports_needed(rule)`, returning the
`(mode, port)` pairs it needs in `ctx.ports`.
"""
abstract type SelectionRule end

"Keep the `k` lowest eigenstates (`k == 0`: keep all computed)."
struct Lowest <: SelectionRule
    k::Int
end
lowest(k::Int) = Lowest(k)

ports_needed(::SelectionRule) = Tuple{Symbol,Port}[]

"""
    Node(children; solver=Auto(), compute=0, keep=lowest(0), phase, ports=[], arithmetic=:auto)

A hierarchical unit.
Each child is a `mode => basis` pair, another `Node`, or a [`GivenSystem`](@ref) (a pre-diagonalized
subsystem); a node's product basis has one axis per child.

- `solver`: `Auto()` (the default), `Dense()`, `Lanczos()`, `LOBPCG()`, `ShiftInvert()` (a leaf on the CPU) or `ChebyshevFilter()`.
  [`Auto`](@ref) resolves at materialization: `Dense()` up to 2 500 product states; above that,
  `ShiftInvert()` at a leaf on `Array` storage, `ChebyshevFilter(on=on)` at a leaf on device storage,
  and `LOBPCG(preconditioner=:ritz, seed=:product, on=on)` at a branch. An `LOBPCG` node starts from the previous
  solution's kept vectors when `materialize` is given `previous` (see [`materialize`](@ref)).
- `compute`: how many of the lowest eigenpairs the solver returns; `0` means all (only valid with
  `Dense`, or with `Auto()` on a node that resolves to `Dense`). `Dense` builds the full matrix whatever `compute` is, and uses it only to return
  the lowest `compute` pairs.
- `keep`: the [`SelectionRule`](@ref) choosing the kept states among the computed ones.
- `phase`: the global-phase convention of the kept eigenvectors, `:positive_sector` or `:follow`.
  `:positive_sector` makes, at a leaf (all children bases), the largest entry with all coordinates
  nonnegative real and positive; at a branch, each state's exact full-space amplitude at the joint
  peak of its dominant product component, computed from the children's stored peak amplitudes,
  so the children's phase conventions do not leak into it.
  It defaults to `:follow` when `keep` is a `follow` rule and to `:positive_sector` otherwise; an
  explicit `phase` always wins. `:follow` needs the previous materialization (`previous`); at the
  first point of a sweep, with none, `:positive_sector` is used.
- `ports`: extra `(mode, port)` operators to form in this node's kept eigenbasis, beyond those
  its ancestors need (which are inferred).
- `arithmetic`: `:auto` (the default) runs the node in real arithmetic when its structure guarantees a
  real Hamiltonian at every θ (the node test of the architecture spec, Section 20f), else complex.
  `:complex` forces complex arithmetic for this node and so for every ancestor.
  `:real` asserts the node test: `materialize` throws an `ArgumentError` naming the first term whose
  conjugate image is missing when it fails.
  [`arithmetic`](@ref) reports the arithmetic a node was materialized in.
"""
mutable struct Node
    children::Vector{Any}
    solver::Solver
    compute::Int
    keep::SelectionRule
    phase::Symbol
    ports::Vector{Tuple{Symbol,Port}}
    arithmetic::Symbol

    function Node(children::AbstractVector; solver::Solver=Auto(), compute::Int=0,
                  keep::SelectionRule=lowest(0), phase::Union{Nothing,Symbol}=nothing,
                  ports=Tuple{Symbol,Port}[], arithmetic::Symbol=:auto)
        isempty(children) && throw(ArgumentError("a Node needs at least one child"))
        for ch in children
            ok = ch isa Node || ch isa GivenSystem || (ch isa Pair && ch.first isa Symbol && ch.second isa Basis)
            ok || throw(ArgumentError("each child must be a `mode => basis` pair, a Node or a GivenSystem, got $(repr(ch))"))
        end
        phase === nothing && (phase = keep isa Follow ? :follow : :positive_sector)
        phase in (:positive_sector, :follow) ||
            throw(ArgumentError("phase must be :positive_sector or :follow, got :$phase"))
        compute >= 0 || throw(ArgumentError("compute must be nonnegative"))
        arithmetic in (:auto, :complex, :real) ||
            throw(ArgumentError("arithmetic must be :auto, :complex or :real, got :$arithmetic"))
        # Auto() is checked at materialization, once it has resolved (see `_resolve_node_solver`).
        compute == 0 && !(solver isa Dense || solver isa Auto) &&
            throw(ArgumentError("compute == 0 (all eigenpairs) requires the Dense solver"))
        node = new(Any[ch for ch in children], solver, compute, keep, phase,
                   Tuple{Symbol,Port}[(m, p) for (m, p) in ports], arithmetic)
        ms = modes(node)
        for (i, m) in enumerate(ms)
            m in view(ms, 1:i-1) && throw(ArgumentError("mode :$m appears more than once in the tree"))
        end
        return node
    end
end

isleaf(node::Node) = all(ch -> ch isa Pair, node.children)

_child_modes(ch::Pair) = Symbol[ch.first]
_child_modes(ch::Node) = modes(ch)

"""
    modes(node::Node) -> Vector{Symbol}

All modes beneath `node`, in axis order.
"""
modes(node::Node) = reduce(vcat, (_child_modes(ch) for ch in node.children); init=Symbol[])

"""
    axis_of(node, mode) -> Int

Index of the child of `node` whose subtree contains `mode`.
"""
function axis_of(node::Node, mode::Symbol)
    for (i, ch) in enumerate(node.children)
        mode in _child_modes(ch) && return i
    end
    throw(ArgumentError("mode :$mode is not in this node"))
end

_subnodes(node::Node) = (ch for ch in node.children if ch isa Node)

function _all_nodes(root::Node)
    out = Node[]
    stack = Node[root]
    while !isempty(stack)
        n = pop!(stack)
        push!(out, n)
        append!(stack, reverse!(collect(_subnodes(n))))
    end
    return out
end

"""
    route(root, terms) -> IdDict{Node,Vector{Term}}

Assign each term to the lowest node containing all its modes (factor-free terms go to the root).
"""
function route(root::Node, terms::Vector{Term})
    treemodes = Set(modes(root))
    seen = Set{Symbol}()
    routed = IdDict{Node,Vector{Term}}(n => Term[] for n in _all_nodes(root))
    for t in terms
        for m in modes(t)
            m in treemodes ||
                throw(ArgumentError("mode :$m appears in a term but is not in the tree"))
            push!(seen, m)
        end
        node = root
        while true
            ms = modes(t)
            next = nothing
            for ch in _subnodes(node)
                if all(in(modes(ch)), ms)
                    next = ch
                    break
                end
            end
            (next === nothing || isempty(ms)) && break
            node = next
        end
        push!(routed[node], t)
    end
    for m in modes(root)
        m in seen || throw(ArgumentError("mode :$m is in the tree but in no term"))
    end
    return routed
end

"""
    required_ports(root, routed) -> IdDict{Node,Vector{Tuple{Symbol,Port}}}

The `(mode, port)` operators each node must expose to its ancestors.
In order, deduplicated: the factors inside the node of terms routed to strict ancestors; each
strict ancestor's `ports_needed(keep)` and `ports` on modes inside the node, outermost ancestor
first (an ancestor forms those ports from its children's, so every node on the path down to the
mode's leaf exposes them); the node's own `ports_needed(keep)` and `ports`.
An ancestor's `nport()` on a twisted ([`PeriodicGrid`](@ref)) mode is requested as its three derivative pieces
`Port(:d1, 0)`, `Port(:d1wrap, ±1)`, from which each node that needs the charge assembles it.
"""
function required_ports(root::Node, routed::IdDict)
    out = IdDict{Node,Vector{Tuple{Symbol,Port}}}()
    twisted = Set(m for (m, b) in _tree_bases(root) if _is_twisted(b))
    pieces = last.(_twisted_pieces(Coefficient(0), :d1))      # the charge's derivative ports (the twists are unused)
    function visit(node::Node, ancestors::Vector{Node})
        req = Tuple{Symbol,Port}[]
        if !isempty(ancestors)
            inside = Set(modes(node))
            for a in ancestors, t in get(routed, a, Term[]), f in t.factors
                f.mode in inside && push!(req, (f.mode, f.port))
            end
            for a in ancestors, (m, p) in Iterators.flatten((ports_needed(a.keep), a.ports))
                m in inside || continue
                if m in twisted && p == nport()
                    append!(req, ((m, q) for q in pieces))
                else
                    push!(req, (m, p))
                end
            end
        end
        append!(req, ports_needed(node.keep))
        append!(req, node.ports)
        out[node] = unique!(req)
        for ch in _subnodes(node)
            visit(ch, vcat(ancestors, node))
        end
    end
    visit(root, Node[])
    return out
end
