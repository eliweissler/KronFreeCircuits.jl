# Part B: full materialization of two coupled symmetric zero-pis (`two_zero_pi.json`), with the
# stages timed separately, plus larger robustness cases from the pathological-test catalogue.
#
#   BENCH_DEVICE=cpu|gpu BENCH_STAGES=leaf,tree,parts,f3,f9,f8 julia --project=... bench/end_to_end.jl
#
# Tree: leaves [theta_i => ChargeBasis(15), phi_i => PhaseGrid(-6π, 6π, 301; order=8)], each with
# the BENCH_LEAF_SOLVER solver computing and keeping 300 states; the parent (a counted LOBPCG(:ritz)
# with the default seed=:product, the solver Auto() resolves to there) keeps BENCH_NEV states and forms the
# charge ports of both leaves' theta modes in its kept basis (the ports an ancestor would need), so
# port formation after the parent solve is measured too. BENCH_SMOKE=1: ncut 6, 101 points, 30
# kept leaf states, 12 parent states. g = 0.05 (the stored audit parent's coupling).
#
# Stages (each a timed operation with one untimed warm-up and BENCH_REPEATS samples unless noted):
#   leaf   each leaf materialized alone (the term-list form on the leaf's own terms).
#   tree   the full-tree materialize (cold).
#   parts  leaf and parent operator construction on the device, the upload alone, the leaf and
#          parent solves alone (cold, on the resident operator), eigenvector download, and port
#          formation alone (`_port_matrix` for every port the tree forms).
#   f3     catalogue F3: g = 0, materialized warm from the g = 0.05 system (one sample), checked
#          against the pairwise sums of its leaf energies, then a cold parent solve for comparison.
#   f9     catalogue F9: one leaf on a 1.5× wider grid at the same spacing (one cold sample).
#   f8     catalogue F8: g = 0.5 materialized warm from g = 0.05 (a large step, one sample), then a
#          leaf-cutoff study of the parent (lowest 20 states, 6 with BENCH_SMOKE) with Dense on the same projected
#          operator where it fits, and a warm/cold comparison at the large step.
# Every solve is validated (true residuals below 1e-8 and orthonormality, recomputed with a host
# copy of the operator) after its sample, outside the timing.
# `parts`, `f3` and `f8` need a materialized g = 0.05 system: the last `tree` sample, or one
# untimed materialize when `tree` is not among the stages.
#
# Further knobs (defaults are the brief's configuration):
#   BENCH_LEAF_SOLVER   lobpcg (LOBPCG(:ritz) on BENCH_DEVICE) | lanczos (Lanczos() on the host,
#                       the audit's leaf solver) | dense (Dense() on the host) | shiftinvert
#                       (ShiftInvert() on the host) | chebyshev (ChebyshevFilter(on=BENCH_DEVICE's
#                       storage)) | auto (Auto(on=...), which resolves the 9331-state leaf to
#                       ShiftInvert on cpu and ChebyshevFilter on gpu)
#   BENCH_SLICES        with shiftinvert: ShiftInvert(slices=BENCH_SLICES) (spectrum slicing over
#                       Julia threads, so run julia -t N; unset: one slice)
#   BENCH_LEAF_MAXITER  LOBPCG maxiter at the leaves (default 1000, the solver default)
#   BENCH_ARITHMETIC    auto (default) | complex: the `arithmetic=` option of every node of the tree
#                       (leaves and parent). `auto` runs the nodes the structure guarantees real in
#                       Float64 (both leaves and the parent here); `complex` forces ComplexF64
#                       throughout. The JSON records the knob ("arithmetic") and the arithmetic each
#                       materialized node actually ran in (`arithmetic(sys, node)`, per sample in the
#                       leaf and tree stages and once as "node_arithmetic").
#   BENCH_LEAVES        which leaves the `leaf` stage times (default "1,2")
#   BENCH_WARMUP        1: one untimed full-size run before the samples of the leaf-dominated
#                       stages (leaf, tree, leaf solve alone; default); 0: none (the small-tree
#                       compilation warm-up always runs, and the other stages always warm up)
#   BENCH_LEAF_REPEATS  samples of the leaf-dominated stages (default BENCH_REPEATS; 0 skips the
#                       leaf solve alone in `parts`)
# A stage that throws (an unconverged solve) is recorded as failed, with its time to failure and,
# for LOBPCG nodes, the iterations and H columns reached; dependent stages are then skipped, and
# the script exits nonzero after writing the JSON.

include(joinpath(@__DIR__, "common.jl"))
using Test                                    # test/helpers.jl defines @test-based helpers
include(joinpath(@__DIR__, "..", "test", "helpers.jl"))
using KronFreeCircuits: operator, materialized, MaterializedNode, _port_matrix, _childop, route,
                        required_ports, axis_of, _node_operator, _edge_support, _on_storage, _resolve_node_solver

const G = 0.05
const NCUT, NPTS, KEEP = SMOKE ? (6, 101, 30) : (15, 301, 300)
const NEV = parse(Int, get(ENV, "BENCH_NEV", SMOKE ? "12" : "120"))
const STAGES = split(get(ENV, "BENCH_STAGES", "leaf,tree,parts,f3,f9,f8"), ",")
const TOL = 1e-8
const LEAF_SOLVER = get(ENV, "BENCH_LEAF_SOLVER", "lobpcg")
LEAF_SOLVER in ("lobpcg", "lanczos", "dense", "shiftinvert", "chebyshev", "auto") ||
    error("BENCH_LEAF_SOLVER must be lobpcg, lanczos, dense, shiftinvert, chebyshev or auto")
const SLICES = haskey(ENV, "BENCH_SLICES") ? parse(Int, ENV["BENCH_SLICES"]) : nothing
SLICES === nothing || LEAF_SOLVER == "shiftinvert" || error("BENCH_SLICES needs BENCH_LEAF_SOLVER=shiftinvert")
const LEAF_MAXITER = parse(Int, get(ENV, "BENCH_LEAF_MAXITER", "1000"))
const ARITH = Symbol(get(ENV, "BENCH_ARITHMETIC", "auto"))
ARITH in (:auto, :complex) || error("BENCH_ARITHMETIC must be auto or complex")
const LEAVES = parse.(Int, split(get(ENV, "BENCH_LEAVES", "1,2"), ","))
const WARMUP = get(ENV, "BENCH_WARMUP", "1") == "1"
const LEAF_REPEATS = parse(Int, get(ENV, "BENCH_LEAF_REPEATS", string(REPEATS)))
all(in(("leaf", "tree", "parts", "f3", "f9", "f8")), STAGES) || error("unknown stage in $STAGES")

spec = load_spec(fixture("two_zero_pi.json"))
θ = two_zero_pi_params(; g=G)
allterms = circuit_terms(spec)
modetypes = Dict{Symbol,Symbol}(m.name => m.type for m in spec.modes)

# Each solver records its work through its callback: LOBPCG its last iteration count and cumulative
# H columns, ChebyshevFilter its iteration count and last filter degree (H columns not counted, -1);
# solvers without a callback (Dense, Lanczos, ShiftInvert, Auto) record -1.
mutable struct Counter
    n_iter::Int
    n_matvec::Int
    degree::Int
    kind::Symbol                     # :lobpcg, :chebyshev or :none
end
Counter(kind::Symbol=:lobpcg) = reset!(Counter(0, 0, 0, kind))
reset!(c::Counter) = (c.n_iter = c.kind === :none ? -1 : 0; c.n_matvec = c.kind === :lobpcg ? 0 : -1;
                      c.degree = c.kind === :chebyshev ? 0 : -1; c)
counted(c::Counter; kw...) = LOBPCG(; on=ON, callback=info -> (c.n_iter = info.n_iter; c.n_matvec = info.n_matvec; nothing), kw...)
work(c::Counter) = c.kind === :chebyshev ? Dict("iterations" => c.n_iter, "n_matvec" => c.n_matvec, "degree" => c.degree) :
                   Dict("iterations" => c.n_iter, "n_matvec" => c.n_matvec)
const LEAF_KIND = LEAF_SOLVER == "lobpcg" ? :lobpcg : LEAF_SOLVER == "chebyshev" ? :chebyshev : :none
function leaf_solver(c::Counter)
    LEAF_SOLVER == "lanczos" && return Lanczos()
    LEAF_SOLVER == "dense" && return Dense()
    LEAF_SOLVER == "shiftinvert" && return SLICES === nothing ? ShiftInvert() : ShiftInvert(; slices=SLICES)
    LEAF_SOLVER == "chebyshev" && return ChebyshevFilter(; on=ON, callback=info -> (c.n_iter = info.iteration; c.degree = info.degree; nothing))
    LEAF_SOLVER == "auto" && return Auto(; on=ON)
    return counted(c; maxiter=LEAF_MAXITER)
end
# The solver `materialize` ran at `node` (Auto() resolved by the node's shape), and the arithmetic
# each node of the tree materialized in (recorded once per run, untimed).
resolved(s::System, node) = _resolve_node_solver(node, materialized(s, node).children)
node_arithmetics(s::System, T) = Dict(k => string(arithmetic(s, n)) for (k, n) in nodes(T))
function note_arithmetic!(s::System, T)
    haskey(record, "node_arithmetic") || (record["node_arithmetic"] = node_arithmetics(s, T))
end

function build_tree(; grid=(-6π, 6π, NPTS))
    cs = Dict("leaf1" => Counter(LEAF_KIND), "leaf2" => Counter(LEAF_KIND), "root" => Counter(:lobpcg))
    leaf(i) = Node([Symbol("theta$i") => ChargeBasis(NCUT), Symbol("phi$i") => PhaseGrid(grid...; order=8)];
                   solver=leaf_solver(cs["leaf$i"]), compute=KEEP, keep=lowest(KEEP), arithmetic=ARITH)
    l1, l2 = leaf(1), leaf(2)
    root = Node([l1, l2]; solver=counted(cs["root"]), compute=NEV, keep=lowest(NEV),
                ports=[(:theta1, nport()), (:theta2, nport())], arithmetic=ARITH)
    return (root=root, leaf1=l1, leaf2=l2, counters=cs)
end
nodes(T) = ("leaf1" => T.leaf1, "leaf2" => T.leaf2, "root" => T.root)
leafterms(node) = filter(t -> all(in(KronFreeCircuits.modes(node)), KronFreeCircuits.modes(t)), allterms)
materialize_leaf(node, θ) = materialize(node, leafterms(node), θ; modetypes=modetypes)

function check(H, E, V)
    v = validate(H, E, V; tol=TOL)
    return Dict("max_residual" => v.max_residual, "ortho" => v.ortho, "ok" => v.ok)
end
check(sys::System, node) = check(operator(sys, node), energies(sys, node), eigenvectors(sys, node))

ok(d::Dict) = haskey(d, "ok") ? d["ok"] : all(ok(v) for v in values(d) if v isa Union{Dict,AbstractVector})
ok(v::AbstractVector) = all(ok(x) for x in v if x isa Union{Dict,AbstractVector})
times(samples) = [s["seconds"] for s in samples]
function report(label, samples; extra="")
    smp = samples
    status = all(ok, smp) ? "ok" : "FAILED"
    @printf("%-34s %s %s %s\n", label, summary_line(times(smp)), extra, status)
end

"""
    repeated(f, post=v -> Dict("ok" => true); warmup=true, n=REPEATS) -> samples

One untimed `f()` (when `warmup`), then `n` samples: `v = f()` timed, then `post(v)`, untimed
(validation), returns the sample's `Dict`, merged with `"seconds"`.
"""
function repeated(f, post=v -> Dict{String,Any}("ok" => true); warmup=true, n=REPEATS)
    warmup && f()
    return [(t_v = timed(f); merge(post(t_v[2]), Dict{String,Any}("seconds" => t_v[1]))) for _ in 1:n]
end

"""
    attempt(f, name, counters) -> Bool

Run stage `f()`; if it throws, record `stages[name]` as failed (time to failure, the error, and
the LOBPCG counters reached) and return `false`.
"""
function attempt(f, name, counters=Dict{String,Counter}())
    t0 = time_ns()
    try
        f()
        return true
    catch err
        err isa InterruptException && rethrow()
        msg = sprint(showerror, err)
        stages[name] = Dict{String,Any}("failed" => true, "seconds_to_failure" => (time_ns() - t0) / 1e9, "error" => msg,
                                        "work_at_failure" => Dict(k => work(c) for (k, c) in counters))
        @printf("%-34s FAILED after %.1f s: %s\n", name, stages[name]["seconds_to_failure"], first(msg, 400))
        return false
    end
end

record = Dict{String,Any}("part" => "B", "arithmetic" => string(ARITH), "environment" => environment(), "g" => G, "ncut" => NCUT,
                          "points" => NPTS, "leaf_keep" => KEEP, "nev" => NEV, "tol" => TOL,
                          "leaf_solver" => LEAF_SOLVER, "leaf_slices" => something(SLICES, 1),
                          "leaf_maxiter" => LEAF_MAXITER, "warmup" => WARMUP,
                          "leaf_repeats" => LEAF_REPEATS, "stages" => Dict{String,Any}())
stages = record["stages"]

# Compilation warm-up on a small tree with the same types (not a timed stage).
let
    tsmall = @elapsed begin
        s1 = LOBPCG(; on=ON)
        # The small leaf (287 states) is below Auto's dense cutoff, so `auto` and sliced `shiftinvert`
        # compile the unsliced leaf solver they run at full size here (slicing compiles in the full-size warm-up).
        ls = LEAF_SOLVER == "lobpcg" ? s1 : LEAF_SOLVER == "auto" ? (ON === Array ? ShiftInvert() : ChebyshevFilter(; on=ON)) :
             LEAF_SOLVER == "shiftinvert" ? ShiftInvert() : leaf_solver(Counter(LEAF_KIND))
        lf(i) = Node([Symbol("theta$i") => ChargeBasis(3), Symbol("phi$i") => PhaseGrid(-6π, 6π, 41; order=8)];
                     solver=ls, compute=8, keep=lowest(8), arithmetic=ARITH)
        r = Node([lf(1), lf(2)]; solver=s1, compute=6, keep=lowest(6), ports=[(:theta1, nport()), (:theta2, nport())], arithmetic=ARITH)
        materialize(r, spec, θ)
    end
    @printf("compilation warm-up (small tree) %.1f s\n", tsmall)
end

T = build_tree()
sys = nothing

if "leaf" in STAGES
    for i in LEAVES
        name = "leaf$i"
        node, c = getfield(T, Symbol(name)), T.counters[name]
        attempt("leaf_materialize_$name", Dict(name => c)) do
            samples = repeated(() -> (reset!(c); materialize_leaf(node, θ)),
                               s -> Dict{String,Any}("node_arithmetic" => string(arithmetic(s, node)),
                                                     "work" => work(c), "validation" => check(s, node),
                                                     "resolved_solver" => string(nameof(typeof(resolved(s, node)))),
                                                     "energies" => energies(s, node));
                               warmup=WARMUP, n=LEAF_REPEATS)
            stages["leaf_materialize_$name"] = samples
            report("leaf materialize ($name)", samples; extra="H columns $(samples[end]["work"]["n_matvec"]) iters $(samples[end]["work"]["iterations"])")
        end
    end
end

if "tree" in STAGES
    attempt("tree_materialize", T.counters) do
        samples = repeated(() -> (foreach(reset!, values(T.counters)); materialize(T.root, spec, θ)),
                           function (s)
                               global sys = s
                               note_arithmetic!(s, T)
                               Dict{String,Any}("node_arithmetic" => node_arithmetics(s, T),
                                                "work" => Dict(k => work(c) for (k, c) in T.counters),
                                                "validation" => Dict(k => check(s, n) for (k, n) in nodes(T)),
                                                "resolved_solvers" => Dict(k => string(nameof(typeof(resolved(s, n)))) for (k, n) in nodes(T)),
                                                "energies" => energies(s))
                           end; warmup=WARMUP, n=LEAF_REPEATS)
        stages["tree_materialize"] = samples
        report("full-tree materialize", samples;
               extra="H columns leaf1/leaf2/root $(join((samples[end]["work"][k]["n_matvec"] for k in ("leaf1", "leaf2", "root")), "/"))")
    end
end

if sys === nothing && any(in(STAGES), ("parts", "f3", "f8")) && !("tree" in STAGES)
    attempt("setup_materialize", T.counters) do
        t = @elapsed (global sys = materialize(T.root, spec, θ))
        stages["setup_materialize_seconds"] = t
        @printf("untimed setup materialize %.1f s\n", t)
    end
end
sys === nothing && any(in(STAGES), ("parts", "f3", "f8")) &&
    println("no materialized system: stages parts, f3 and f8 skipped")

"parts" in STAGES && sys !== nothing && attempt("parts", T.counters) do
    leafsolver, rootsolver = resolved(sys, T.leaf1), T.root.solver     # Auto() as materialize resolved it
    note_arithmetic!(sys, T)
    # Operator construction on the solver's storage (includes the transfer of its data on the GPU).
    Hl = operator(sys, T.leaf1)
    s = repeated(() -> operator(sys, T.leaf1))
    stages["leaf_operator_construction"] = s; report("leaf operator construction", s)
    Hp = operator(sys)
    s = repeated(() -> operator(sys))
    stages["parent_operator_construction"] = s; report("parent operator construction", s)
    if DEVICE == "gpu"
        Hph = host(Hp)
        s = repeated(() -> _on_storage(Hph, ON; capacity=NEV + rootsolver.guard))
        stages["parent_operator_upload"] = s; report("parent operator upload alone", s)
        Vd = CuArray(eigenvectors(sys))
        s = repeated(() -> Array(Vd))
        stages["parent_vectors_download"] = s; report("parent eigenvector download", s)
        Vd = nothing
    end
    # Every leaf solver but LOBPCG (whose result carries its work) goes through the solver contract.
    # ChebyshevFilter reports its iterations through the leaf1 counter's callback.
    cl = T.counters["leaf1"]
    leafsolve() = leafsolver isa LOBPCG ? lobpcg_solve(leafsolver, Hl, KEEP) :
                  (reset!(cl); (KronFreeCircuits.solve(leafsolver, Hl, KEEP)..., nothing))
    s = LEAF_REPEATS == 0 ? Any[] :
        repeated(leafsolve, ((E, V, res),) -> Dict{String,Any}("n_matvec" => res === nothing ? -1 : res.n_matvec,
                                                              "iterations" => res === nothing ? cl.n_iter : iterations(res),
                                                              "solver" => string(nameof(typeof(leafsolver))),
                                                              "validation" => check(Hl, E, V));
                 warmup=WARMUP, n=LEAF_REPEATS)
    isempty(s) || (stages["leaf_solve"] = s; report("leaf solve alone (leaf1, cold)", s; extra="H columns $(s[end]["n_matvec"]) iters $(s[end]["iterations"])"))
    s = repeated(() -> lobpcg_solve(rootsolver, Hp, NEV),
                 ((E, V, res),) -> Dict{String,Any}("n_matvec" => res.n_matvec, "iterations" => iterations(res),
                                                    "validation" => check(Hp, E, V),
                                                    "max_energy_difference_to_tree" => maximum(abs.(E .- energies(sys)))))
    stages["parent_solve"] = s; report("parent solve alone (cold)", s; extra="H columns $(s[end]["n_matvec"]) iters $(s[end]["iterations"])")
    # Port formation, node by node, for every port the tree forms (on the host, as in materialize).
    needed = required_ports(T.root, route(T.root, sys.terms))
    for (name, node) in nodes(T)
        mn = materialized(sys, node)
        ps = needed[node]
        form() = [_port_matrix(mn.vectors, mn.dims, axis_of(node, mode), _childop(mn.children[axis_of(node, mode)], mode, p))
                  for (mode, p) in ps]
        s = repeated(form, function (Ps)
            err = maximum((norm(P - mn.ports[mp]) for (P, mp) in zip(Ps, ps)); init=0.0)
            Dict{String,Any}("ports" => [string(m, ":", p.kind, p.k == 0 ? "" : "($(p.k))") for (m, p) in ps],
                             "max_difference_to_stored" => err, "ok" => err < 1e-12 * max(1, length(mn.kept)))
        end)
        stages["port_formation_$name"] = s
        report("port formation ($name, $(length(ps)) ports)", s)
    end
end

"f3" in STAGES && sys !== nothing && attempt("f3", T.counters) do
    θ0 = two_zero_pi_params(; g=0.0)
    foreach(reset!, values(T.counters))
    t, sys0 = timed(() -> materialize(T.root, spec, θ0; previous=sys))
    warmwork = Dict(k => work(c) for (k, c) in T.counters)     # before the cold solve reuses the root counter
    E1, E2 = energies(sys0, T.leaf1), energies(sys0, T.leaf2)
    c0 = constant_term(spec)(θ0)
    sums = sort(vec([a + b for a in E1, b in E2]))
    E = energies(sys0)
    # The largest prefix of the lowest NEV that does not split a degenerate cluster.
    gaps = diff(sums[1:NEV+1])
    nc = findlast(k -> gaps[k] > 1e-7, 1:NEV)
    # At g = 0 the parent is diagonal in the leaves' kept bases: its lowest nc eigenvectors span
    # the product states of the lowest nc pairwise sums.
    order = sortperm(vec([a + b for a in E1, b in E2]))
    V = eigenvectors(sys0)
    lost = nc === nothing ? NaN : nc - norm(V[order[1:nc], 1:nc])^2
    Hp0 = operator(sys0)
    tc, (Ec, Vc, rc) = timed(() -> lobpcg_solve(T.root.solver, Hp0, NEV))
    stages["f3"] = Dict{String,Any}(
        "warm_materialize_seconds" => t, "work" => warmwork,
        "validation" => Dict(k => check(sys0, n) for (k, n) in nodes(T)),
        "max_energy_error_vs_pairwise_sums" => maximum(abs.(E .- (sums[1:NEV] .+ c0))),
        "cluster_split_at_cutoff" => gaps[NEV] <= 1e-7, "complete_prefix" => nc,
        "lost_weight_complete_prefix" => lost,
        "degenerate_pairs_oracle" => count(<=(1e-7), diff(sums[1:NEV])),
        "degenerate_pairs_computed" => count(<=(1e-7), diff(E)),
        "cold_parent_seconds" => tc, "cold_parent_n_matvec" => rc.n_matvec, "cold_parent_iterations" => iterations(rc),
        "cold_parent_validation" => check(Hp0, Ec, Vc),
        "cold_parent_max_energy_error_vs_pairwise_sums" => maximum(abs.(Ec .- (sums[1:NEV] .+ c0))))
    f = stages["f3"]
    @printf("F3 g=0 warm materialize %.2f s (H columns %s); |E − pairwise sums| %.1e; degenerate pairs %d/%d; lost weight %.1e; cold parent %.2f s, %d H columns (warm %d) %s\n",
            t, join((f["work"][k]["n_matvec"] for k in ("leaf1", "leaf2", "root")), "/"),
            f["max_energy_error_vs_pairwise_sums"], f["degenerate_pairs_computed"], f["degenerate_pairs_oracle"], lost, tc,
            rc.n_matvec, f["work"]["root"]["n_matvec"],
            ok(f["validation"]) && ok(f["cold_parent_validation"]) ? "ok" : "FAILED")
end

"f9" in STAGES && attempt("f9") do
    wide = build_tree(; grid=(-9π, 9π, 3 * (NPTS - 1) ÷ 2 + 1))   # same spacing, 1.5× the interval
    Tn = build_tree()
    t_n, sn = timed(() -> materialize_leaf(Tn.leaf1, θ))
    t_w, sw = timed(() -> materialize_leaf(wide.leaf1, θ))
    En, Ew = energies(sn, Tn.leaf1), energies(sw, wide.leaf1)
    edge(s, node, grid) = _edge_support(eigenvectors(s, node), materialized(s, node).dims, 2, grid.order ÷ 2)
    en = edge(sn, Tn.leaf1, Tn.leaf1.children[2].second)
    ew = edge(sw, wide.leaf1, wide.leaf1.children[2].second)
    stages["f9"] = Dict{String,Any}(
        "points" => [NPTS, 3 * (NPTS - 1) ÷ 2 + 1], "seconds" => [t_n, t_w], "energies" => [En, Ew],
        "work" => [work(Tn.counters["leaf1"]), work(wide.counters["leaf1"])],
        "validation" => [check(sn, Tn.leaf1), check(sw, wide.leaf1)],
        "max_energy_change" => Dict(string(k) => maximum(abs.(En[1:k] .- Ew[1:k])) for k in unique((20, 100, KEEP)) if k <= KEEP),
        "max_edge_support" => [maximum(en), maximum(ew)],
        "states_with_edge_support_above_1e-6" => [count(>(1e-6), en), count(>(1e-6), ew)],
        "first_state_with_edge_support_above_1e-6" => [something(findfirst(>(1e-6), en), 0), something(findfirst(>(1e-6), ew), 0)])
    f = stages["f9"]
    @printf("F9 leaf grid %d → %d points: %.2f / %.2f s; ground state %.6f → %.6f; max |ΔE| %s; edge support max %.1e → %.1e, states above 1e-6 %d → %d %s\n",
            f["points"]..., t_n, t_w, En[1], Ew[1], join(("lowest $k: $(@sprintf("%.1e", v))" for (k, v) in sort(collect(f["max_energy_change"]); by=x -> parse(Int, x[1]))), ", "),
            f["max_edge_support"]..., f["states_with_edge_support_above_1e-6"]..., all(ok, f["validation"]) ? "ok" : "FAILED")
end

"f8" in STAGES && sys !== nothing && attempt("f8", T.counters) do
    GS = 0.5
    NF8 = SMOKE ? 6 : 20
    KS = SMOKE ? (5, 10, 20, 30) : (10, 25, 50, 100, 200, 300)
    θs = two_zero_pi_params(; g=GS)
    foreach(reset!, values(T.counters))
    t, syss = timed(() -> materialize(T.root, spec, θs; previous=sys))
    rootmn = materialized(syss, T.root)
    routed = route(T.root, syss.terms)
    cS = constant_term(spec)(θs)
    truncated_child(mn::MaterializedNode{T}, k) where {T} =
        MaterializedNode{T}(mn.node, mn.dims, mn.energies[1:k], mn.vectors[:, 1:k], mn.kept[1:k],
                         Dict(key => P[1:k, 1:k] for (key, P) in mn.ports),
                         Dict(key => P[1:k, 1:k] for (key, P) in mn.joint_ports), mn.children, mn.peak_component[1:k],
                         mn.peak_amplitudes[1:k, 1:k])
    s20 = LOBPCG(; on=ON)
    cut = Any[]
    Efull = nothing
    for k in sort(collect(KS); rev=true)
        kids = Any[truncated_child(c, k) for c in rootmn.children]
        H = _node_operator(T.root, _resolve_node_solver(T.root, kids), kids, routed[T.root], θs, cS;
                           real=rootmn isa MaterializedNode{Float64})
        tk, (E, V, res) = timed(() -> lobpcg_solve(s20, H, NF8))
        Efull === nothing && (Efull = E)
        row = Dict{String,Any}("k" => k, "seconds" => tk, "n_matvec" => res.n_matvec, "iterations" => iterations(res),
                               "validation" => check(H, E, V), "energies" => E,
                               "max_energy_difference_to_k_max" => maximum(abs.(E .- Efull)))
        if k^2 <= 2500
            Ed, _ = KronFreeCircuits.solve(Dense(), host(H), NF8)
            row["max_energy_difference_to_dense"] = maximum(abs.(E .- Ed))
        end
        push!(cut, row)
    end
    # The large step at the full retained cutoff: warm from the g = 0.05 parent vectors vs cold.
    Hs = operator(syss)
    tw, (Ew, Vw, rw) = timed(() -> lobpcg_solve(s20, Hs, NF8; X0=eigenvectors(sys)[:, 1:NF8]))
    tcold, (Ec, Vc, rc) = timed(() -> lobpcg_solve(s20, Hs, NF8))
    stages["f8"] = Dict{String,Any}(
        "g" => GS, "nev" => NF8, "warm_materialize_seconds" => t, "work" => Dict(k => work(c) for (k, c) in T.counters),
        "validation" => Dict(k => check(syss, n) for (k, n) in nodes(T)), "cutoff_study" => cut,
        "large_step_warm" => Dict("seconds" => tw, "n_matvec" => rw.n_matvec, "iterations" => iterations(rw), "validation" => check(Hs, Ew, Vw)),
        "large_step_cold" => Dict("seconds" => tcold, "n_matvec" => rc.n_matvec, "iterations" => iterations(rc), "validation" => check(Hs, Ec, Vc)),
        "warm_vs_cold_max_energy_difference" => maximum(abs.(Ew .- Ec)))
    f = stages["f8"]
    @printf("F8 g=%.2f warm materialize from g=%.2f: %.2f s (H columns %s) %s\n", GS, G, t,
            join((f["work"][k]["n_matvec"] for k in ("leaf1", "leaf2", "root")), "/"), ok(f["validation"]) ? "ok" : "FAILED")
    for r in cut
        @printf("F8 cutoff k=%3d: %.2f s, %4d H columns, max |E − E(k=%d)| %.2e%s %s\n", r["k"], r["seconds"], r["n_matvec"],
                maximum(KS), r["max_energy_difference_to_k_max"],
                haskey(r, "max_energy_difference_to_dense") ? @sprintf(", vs Dense %.1e", r["max_energy_difference_to_dense"]) : "",
                r["validation"]["ok"] ? "ok" : "FAILED")
    end
    @printf("F8 large step, %d states: warm %.2f s / %d H columns, cold %.2f s / %d H columns, |ΔE| %.1e %s\n", NF8, tw,
            rw.n_matvec, tcold, rc.n_matvec, f["warm_vs_cold_max_energy_difference"],
            f["large_step_warm"]["validation"]["ok"] && f["large_step_cold"]["validation"]["ok"] ? "ok" : "FAILED")
end

write_json("end_to_end-$DEVICE$(SMOKE ? "-smoke" : "")-$(join(STAGES, "+")).json", record)

# Failed stages and samples that did not validate; the script then exits nonzero.
bad = String[]
for (k, v) in stages
    if v isa AbstractVector
        n = count(x -> x isa Dict && !ok(x), v)
        n > 0 && push!(bad, "$k ($n of $(length(v)) samples)")
    elseif v isa Dict
        (get(v, "failed", false) || !ok(v)) && push!(bad, k)
    end
end
println("summary: ", isempty(bad) ? "no failed stages or samples" : "failed: " * join(bad, ", "))
isempty(bad) || exit(1)
