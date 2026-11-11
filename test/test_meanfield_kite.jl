using Test, KronFreeCircuits

# The KITE's mean-field loop against the reference calculation (circuit-quantization's product_state prototype), frame by
# frame, from the fixtures of test/python/generate_kite_fixtures.jl: the grids, the wells (the distinct solutions within
# 0.5 GHz of the lowest), their product energies and the three lowest mean-field levels of each mode, to 1e-6 GHz.
# The number of converged runs and the iteration counts are not compared: whether a run's mixing switches on is decided by
# round-off in the rise count, so they may differ between the two codes while the wells do not.
# The mean-field operators themselves are compared at equal inputs: fed the reference's expectation values of a well,
# `mean_field_child` reproduces the reference's levels to 1e-9 GHz in every frame.
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
              for (m, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5, when_capped=:allow)]
    return leaves, Node(leaves; compute=4, keep=lowest(4))
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

# The table of every leaf built from the reference's expectation values `ex` of a well.
# A port exp(i k φ_j) takes the value X of a cosine whose omega[j] is k, or the conjugate of one whose omega[j] is -k;
# the value depends only on k and the state of mode j, so any such cosine gives it.
function kite_tables(spec, leaves, root, ex)
    X = [complex.(Float64.(x.re), Float64.(x.im)) for x in ex.X]
    tables = IdDict{Node,Expectations}()
    for (j, leaf) in enumerate(leaves)
        t = Expectations()
        for K in child_keys(root, circuit_terms(spec), leaf)
            (m, p) = only(K)
            if p.kind === :expiphi
                c = findfirst(c -> c.omega[j] == p.k, spec.cosines)
                t[K] = c === nothing ? conj(X[findfirst(c -> c.omega[j] == -p.k, spec.cosines)][j]) : X[c][j]
            elseif p.kind === :phi
                t[K] = Float64(ex.phi[j])
            elseif p.kind === :n
                t[K] = Float64(ex.n[j])
            else
                error("no reference value for the key $K")
            end
        end
        tables[leaf] = t
    end
    return tables
end

@testset "KITE $name frame: the reference's grids, wells, product energies and mean-field levels" for name in
        (:published, :normal, :optimum)
    ref = KITE_REFERENCE[name]
    spec = load_spec(fixture("kite/$name.json"))
    leaves, root = kite_tree(spec)
    @test spec.modes[1].type === :c && all(m.type === :e for m in spec.modes[2:end])
    grids = [b for (_, b) in suggest_bases(spec; charge_cutoff=16, max_points=120, extent_scale=1.5, when_capped=:allow) if b isa PhaseGrid]
    @test [g.N for g in grids] == collect(ref.grid.points)
    @test all(isapprox(g.b, e; rtol=1e-12) && g.a == -g.b for (g, e) in zip(grids, ref.grid.extents))
    @test only(b for (_, b) in suggest_bases(spec; charge_cutoff=16) if b isa ChargeBasis).ncut == ref.grid.charge_cutoff

    # the mean-field operators at equal inputs: the reference's expectation values of each well
    bare = @test_logs KITE_PHASE_WARNING IdDict(ch => bare_child(root, circuit_terms(spec), (;), ch) for ch in leaves)
    for r in ref.solutions
        tables = kite_tables(spec, leaves, root, r.expectations)
        for (leaf, levels) in zip(leaves, r.levels)
            mf = mean_field_child(root, circuit_terms(spec), (;), leaf, bare[leaf], tables)
            @test mf.solved.energies[1:3] ≈ Float64.(levels) atol=1e-9
        end
    end

    names = [m.name for m in spec.modes]
    ext = (0.0, π / 2, π, -π / 2)
    starts = Any[[Dict{Symbol,Float64}(zip(names, (a, b, c))) for a in (0.0, π) for b in ext for c in ext]; nothing]
    @test length(starts) == ref.starts
    # one warning from the bare solve of the compact leaf, one from the first iteration of the delocalized start
    sols = @test_logs KITE_PHASE_WARNING KITE_PHASE_WARNING mean_field_solutions(root, spec, (;); starts)
    @test all(s.energy >= KITE_GROUND for s in sols)
    wells = filter(s -> s.energy < sols[1].energy + 0.5, sols)
    @test length(wells) == length(ref.solutions) == 2
    matched = [kite_match(s, spec, leaves, ref.solutions) for s in wells]
    @test allunique(matched)                                 # each reference well is matched once
    # In the published frame the reference also requires ⟨φ⟩ and ⟨n⟩ of every mode to have converged, while this loop
    # requires its tables to have converged, and there ⟨φ⟩ of the extended modes is in no table (no term couples φ linearly).
    # So this loop stops one iteration earlier (220 against 221: the reference's ⟨φ₂⟩ changed by 3.17e-6 > sqrt(1e-11) at
    # iteration 220), and the levels differ by up to about 2.5e-6 GHz, the energies by 1.1e-11.
    # At equal expectation values the two mean-field operators agree to 5e-13 (the check above), so the levels of that
    # frame are compared to 1e-5.
    level_tol = name === :published ? 1e-5 : 1e-6
    for (s, r) in zip(wells, [ref.solutions[i] for i in matched])
        @test s.converged
        @test s.energy ≈ r.energy atol=1e-6
        for (leaf, levels) in zip(leaves, r.levels)
            @test s.children[leaf].solved.energies[1:3] ≈ Float64.(levels) atol=level_tol
        end
    end
end
