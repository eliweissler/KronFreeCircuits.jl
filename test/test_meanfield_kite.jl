using Test, KronFreeCircuits

# The KITE's mean-field loop against the reference calculation (circuit-quantization's product_state prototype), frame by
# frame, from the fixtures of test/python/generate_kite_fixtures.jl: the grids, the wells (the distinct solutions within
# 0.5 GHz of the lowest), their product energies and the three lowest mean-field levels of each mode, to 1e-6 GHz.
# The number of converged runs and the iteration counts are not compared: whether a run's mixing switches on is decided by
# round-off in the rise count, so they may differ between the two codes while the wells do not.
#
# The reference's levels are the eigenvalues of each mode's mean-field operator at the well's final expectations, the ones
# its energy was computed from.
# The `solved.energies` of a `MeanFieldChild` are one iteration older (the operator of the previous tables), and at
# convergence differ from those by up to 6.8e-5 GHz in these frames (the normal frame), so the levels compared are those of
# `mean_field_child` at the solution's own tables.
# At the same expectation values the two codes' mean-field operators give the same levels to 1e-12 GHz in every frame.
#
# The published frame's wells stop one iteration apart (220 here, 221 in the reference).
# The reference also tests ⟨φ⟩ of every extended mode for convergence, while this package's loop tests only the keys of
# `child_keys`, and in the published frame no term couples φ linearly, so ⟨φ⟩ is not among them.
# At iteration 220 the reference's ⟨φ₂⟩ changed by 3.17e-6, just above sqrt(1e-11) = 3.16e-6, while every cosine
# expectation changed by at most 3.07e-6: this package stops there and the reference runs one more iteration.
# The tables then differ by up to 3.05e-6, the product energies by 1.1e-11, and the levels by up to 6.2e-7 (mode 1),
# 6.9e-7 (mode 2) and 2.50e-6 GHz (mode 3, its third level), so mode 3 of the published frame misses 1e-6 and is marked
# broken until the two loops test convergence on the same quantities.
#
# Every call that solves the compact leaf with no cosine expectation warns once from `fix_phases!`: its operator is then
# 4 E_C n² alone, whose second kept state is the charge state |-1>, degenerate with |+1> at zero offset charge, with no
# weight on the positive-charge phase mask.
# That happens in the bare solve of the leaf, and in the first iteration of the delocalized start, whose cosine
# expectations are all zero; neither is a well state.

const KITE_REFERENCE = KronFreeCircuits.JSON3.read(read(fixture("kite/meanfield_reference.json"), String))
# The KITE's ground energy from the reference's direct 1,254,400-state solve in the published frame (the owner's
# reference scripts, and the design's §6.1); a product energy is an upper bound on it.
const KITE_GROUND = -9.523995
const KITE_PHASE_WARNING = (:warn, r"^fix_phases!: column 2 has no weight on the phase mask")

# The leaves and root of the comparison, on the bases the reference uses.
function kite_tree(spec)
    leaves = [Node([m => b]; solver=Dense(), compute=3, keep=lowest(3), arithmetic=:complex)
              for (m, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5)]
    return leaves, Node(leaves; compute=4, keep=lowest(4))
end

# The three lowest mean-field levels of each leaf at the tables of the solution `s` (the expectations its energy was
# computed from), with `bare` the bare solve of each leaf.
function kite_levels(root, spec, leaves, bare, s)
    tables = IdDict{Node,Expectations}(ch => c.expectations for (ch, c) in s.children)
    return [mean_field_child(root, circuit_terms(spec), (;), leaf, bare[leaf], tables).solved.energies[1:3] for leaf in leaves]
end

# ⟨φ⟩ of each extended mode in the lowest state of its child of the solution `s`.
function kite_phi(s, spec, leaves)
    out = Float64[]
    for (leaf, m) in zip(leaves, spec.modes)
        m.type === :c && continue
        key = [(m.name, phiport())]
        push!(out, real(expectations(s.children[leaf].solved, [key])[key]))
    end
    return out
end

# The index of the reference well whose ⟨φ⟩ on the extended modes is closest to that of `s`: the wells of a frame can be
# degenerate to 1e-11, so their order is not compared.
function kite_match(s, spec, leaves, wells)
    ext = [i for (i, m) in enumerate(spec.modes) if m.type !== :c]
    φ = kite_phi(s, spec, leaves)
    return argmin(i -> maximum(abs.(φ .- Float64.(wells[i].phi[ext]))), eachindex(wells))
end

@testset "KITE $name frame: the reference's grids, wells, product energies and mean-field levels" for name in
        (:published, :normal, :optimum)
    ref = KITE_REFERENCE[name]
    spec = load_spec(fixture("kite/$name.json"))
    leaves, root = kite_tree(spec)
    @test spec.modes[1].type === :c && all(m.type === :e for m in spec.modes[2:end])
    grids = [b for (_, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5) if b isa PhaseGrid]
    @test [g.N for g in grids] == collect(ref.grid.points)
    @test all(isapprox(g.b, e; rtol=1e-12) && g.a == -g.b for (g, e) in zip(grids, ref.grid.extents))
    @test only(b for (_, b) in suggest_bases(spec; charge_cutoff=16) if b isa ChargeBasis).ncut == ref.grid.charge_cutoff

    names = [m.name for m in spec.modes]
    ext = (0.0, π / 2, π, -π / 2)
    starts = Any[[Dict{Symbol,Float64}(zip(names, (a, b, c))) for a in (0.0, π) for b in ext for c in ext]; nothing]
    @test length(starts) == ref.starts
    # one warning from the bare solve of the compact leaf, one from the first iteration of the delocalized start
    sols = @test_logs KITE_PHASE_WARNING KITE_PHASE_WARNING mean_field_solutions(root, spec, (;); starts)
    bare = @test_logs KITE_PHASE_WARNING IdDict(ch => bare_child(root, circuit_terms(spec), (;), ch) for ch in leaves)
    @test all(s.energy >= KITE_GROUND for s in sols)
    wells = filter(s -> s.energy < sols[1].energy + 0.5, sols)
    @test length(wells) == length(ref.solutions) == 2
    matched = [kite_match(s, spec, leaves, ref.solutions) for s in wells]
    @test allunique(matched)                                 # each reference well is matched once
    for (s, r) in zip(wells, [ref.solutions[i] for i in matched])
        @test s.converged
        @test s.energy ≈ r.energy atol=1e-6
        for (i, (levels, reference)) in enumerate(zip(kite_levels(root, spec, leaves, bare, s), r.levels))
            if name === :published && i == 3     # the loops stop one iteration apart: 2.50e-6 GHz (see the top of the file)
                @test_broken levels ≈ Float64.(reference) atol=1e-6
            else
                @test levels ≈ Float64.(reference) atol=1e-6
            end
        end
    end
end
