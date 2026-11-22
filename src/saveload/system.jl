# The tree: one group per node, children as subgroups (`_write_node` / `_read_node`), and the file-level
# `save_system` / `load_system`.

const _FORMAT = "KronFreeCircuits.System"
const _FORMAT_VERSION = 1
const _NODE_SETTINGS = (:compute, :phase, :arithmetic, :retain)

# A list of `(mode, port)` pairs as three columns.
function _write_portlist(g, name, list)
    g["$name/modes"] = String[String(m) for (m, _) in list]
    g["$name/kinds"] = String[String(p.kind) for (_, p) in list]
    g["$name/ks"] = Float64[p.k for (_, p) in list]
    return nothing
end
_read_portlist(g, name) = Tuple{Symbol,Port}[(Symbol(m), Port(Symbol(kind), k))
                                              for (m, kind, k) in zip(g["$name/modes"], g["$name/kinds"], g["$name/ks"])]

function _write_node(g, mn::MaterializedNode{T}, retained::Bool) where {T}
    n = mn.node
    g["eltype"] = string(T)
    for f in (:dims, :energies, :vectors, :kept, :peak_component, :peak_amplitudes)
        g[String(f)] = getfield(mn, f)
    end
    for (name, d) in (("ports", [[k] => M for (k, M) in mn.ports]), ("joint_ports", collect(mn.joint_ports)))
        g["$name/count"] = length(d)
        for (i, (key, M)) in enumerate(d)
            _write_portlist(g, "$name/$i", key)
            g["$name/$i/matrix"] = M
        end
    end
    r = mn.retained
    drop = !retained && r.vectors !== nothing && size(r.vectors, 2) > 0
    g["retained/energies"] = r.energies
    g["retained/unkept"] = r.unkept
    g["retained/complete"] = r.complete && !drop      # no vectors stored: the complement is unknown
    g["retained/followed"] = r.followed
    r.vectors === nothing || drop || (g["retained/vectors"] = r.vectors)
    if n isa GivenSystem          # its children are `Any[n]`, rebuilt from the settings
        g["kind"] = "given"
        g["settings/mode"] = String(n.mode)
        g["settings/given_energies"] = n.energies
        g["settings/operator_names"] = String.(collect(keys(n.operators)))   # the operators group is absent when empty
        for (kind, M) in n.operators
            g["settings/operators/$kind"] = M
        end
        return nothing
    end
    g["kind"] = "node"
    _write_struct(g, "settings/solver", n.solver)
    _write_struct(g, "settings/keep", n.keep)
    foreach(f -> _write_value(g, "settings/$f", getfield(n, f)), _NODE_SETTINGS)
    _write_portlist(g, "settings/ports", n.ports)
    g["children/count"] = length(mn.children)
    for (i, ch) in enumerate(mn.children)
        if ch isa MaterializedNode
            _write_node(JLD2.Group(g, "children/$i"), ch, retained)
        else
            g["children/$i/kind"] = "leaf"
            g["children/$i/mode"] = String(n.children[i].first)
            _write_struct(g, "children/$i/basis", ch)
        end
    end
    return nothing
end

_read_ports(g) = Dict{Tuple{Symbol,Port},NodeMatrix}(only(_read_portlist(g, "ports/$i")) => g["ports/$i/matrix"]
                                                     for i in 1:g["ports/count"])

# The node (`Node` or `GivenSystem`) of group `g` and its `MaterializedNode`, children first, so that every child's
# `MaterializedNode.node` is its parent's `Node.children[i]`. With `node` given (the tree's own node at this position),
# no node or basis is built: the results are read onto it and its descendants.
function _read_node(g; on=Array, node=nothing)
    T = g["eltype"] == "ComplexF64" ? ComplexF64 : Float64
    if node !== nothing && g["kind"] == "given"
        children = Any[node]
    elseif g["kind"] == "given"
        names = g["settings/operator_names"]
        node = GivenSystem(Symbol(g["settings/mode"]); energies=g["settings/given_energies"],
                           operators=Dict(Symbol(k) => g["settings/operators/$k"] for k in names))
        children = Any[node]
    else
        function read_child(i)
            c = g["children/$i"]
            old = node === nothing ? nothing : node.children[i]
            c["kind"] == "leaf" || return _read_node(c; on, node=old)
            old === nothing || return (old, old.second)
            b = _read_struct(c, "basis")
            return (Symbol(c["mode"]) => b, b)
        end
        pairs = [read_child(i) for i in 1:g["children/count"]]
        if node === nothing
            settings = NamedTuple(f => _read_value(g, "settings/$f") for f in _NODE_SETTINGS)
            node = Node(Any[first(p) for p in pairs]; solver=_read_struct(g, "settings/solver"; on),
                        keep=_read_struct(g, "settings/keep"), ports=_read_portlist(g, "settings/ports"), settings...)
        end
        children = Any[last(p) for p in pairs]
    end
    ports = _read_ports(g)
    joint = Dict{_FactorKey,NodeMatrix}(_read_portlist(g, "joint_ports/$i") => g["joint_ports/$i/matrix"]
                                        for i in 1:g["joint_ports/count"])
    vectors = haskey(g, "retained/vectors") ? g["retained/vectors"] : nothing
    retained = Retained{T}(g["retained/energies"], g["retained/unkept"], vectors, g["retained/complete"],
                           g["retained/followed"])
    mn = MaterializedNode{T}(node, g["dims"], g["energies"], g["vectors"], g["kept"], ports, joint, children,
                             g["peak_component"], g["peak_amplitudes"], retained)
    return node, mn
end

# Child indices from `root` down to the node `target`, or `nothing`.
function _node_path(root::Node, target::Node)
    for (i, c) in enumerate(root.children)
        c === target && return [i]
        c isa Node && (p = _node_path(c, target)) !== nothing && return pushfirst!(p, i)
    end
    return nothing
end

# Each augmented child: its place in the tree, its solved node, and its basis, terms, θ and union report. The tree
# position itself is saved as an ordinary node, so the solved node is stored twice.
function _write_augmented(f, sys::System, retained::Bool)
    f["augmented/count"] = length(sys.augmented)
    for (i, a) in enumerate(sys.augmented)
        g = JLD2.Group(f, "augmented/$i")
        path = _node_path(sys.root.node, a.node)
        path === nothing && throw(ArgumentError("augmented child $i is not a node of the system's tree"))
        g["node_path"] = path
        _write_node(JLD2.Group(g, "solved"), a.solved, retained)
        g["basis"] = a.basis
        _write_terms(g, "terms", a.terms)
        _write_theta(g, "theta", a.θ)
        g["report/labels"] = a.report.labels
        g["report/taken"] = a.report.taken
        g["report/dropped"] = a.report.dropped
    end
    return nothing
end

function _read_augmented(f, root::MaterializedNode; on)
    out = Any[]
    for i in 1:f["augmented/count"]
        g = f["augmented/$i"]
        place = root
        for j in g["node_path"]
            place = place.children[j]
        end
        _, solved = _read_node(g["solved"]; on, node=place.node)
        push!(out, AugmentedChild(place.node, solved, g["basis"], _read_terms(g, "terms"), _read_theta(g, "theta"),
                                  UnionReport(g["report/labels"], g["report/taken"], g["report/dropped"])))
    end
    return out
end

"""
    save_system(path, sys::System; retained=true, metadata=(;)) -> path

Write the solved `sys` to the JLD2 file `path` (format version 1: plain arrays, numbers and strings, one group per tree
node), to be read back by [`load_system`](@ref).
Solver callbacks are not saved. The file is uncompressed. A coefficient that would not reload identically makes
`save_system` throw an `ArgumentError` rather than write it. `retained = false` leaves out every node's retained
(unkept) eigenvectors, and such a node loads with `retained.complete = false`; derivatives on it then take the slower paths for a node without them and
agree with the original's to solver tolerance, not bit for bit.
`metadata` (a `NamedTuple` or `AbstractDict`, `Symbol` or `String` keys) holds your own notes: numbers, `Bool`, `String`
or arrays of them, checked before anything is written and read back by [`load_metadata`](@ref).
The file is written to `path * ".partial"` and moved onto `path` when complete; on any error the partial file is
removed and an existing file at `path` is left as it was.
"""
function save_system(path::AbstractString, sys::System; retained::Bool=true, metadata=(;))
    meta = _check_metadata(metadata)
    partial = path * ".partial"
    try
        JLD2.jldopen(partial, "w") do f
            f["format"] = _FORMAT
            f["format_version"] = _FORMAT_VERSION
            f["package_version"] = string(pkgversion(@__MODULE__))
            retained || (f["saved_without_retained"] = true)
            for (k, v) in meta
                f["metadata/$k"] = v
            end
            _write_theta(f, "theta", sys.θ)
            _write_system_terms(f, sys)
            _write_node(JLD2.Group(f, "root"), sys.root, retained)
            _write_augmented(f, sys, retained)
        end
        mv(partial, path; force=true)
    catch
        rm(partial; force=true)
        rethrow()
    end
    return path
end

"""
    load_system(path; on=Array) -> System

Read a `System` written by [`save_system`](@ref). Every loaded solver that has an `on` field gets `on` (the device
is chosen here; the device saved in the file is only recorded, see [`load_node`](@ref)), and every `callback` is
`nothing`. The layout is described in `docs/save-format.md`.
"""
function load_system(path::AbstractString; on=Array)
    return _with_system_file(f -> _load_v1(f; on), path)
end

# Run `read(f)` on the open file at `path` after checking it is a system file of a readable version.
function _with_system_file(read, path)
    notsystem = ArgumentError("$path is not a KronFreeCircuits system file")
    f = try
        JLD2.jldopen(path, "r")
    catch e
        isfile(path) || rethrow()
        throw(ArgumentError("$path is not a KronFreeCircuits system file ($(sprint(showerror, e)))"))
    end
    try
        haskey(f, "format") && f["format"] == _FORMAT && haskey(f, "format_version") || throw(notsystem)
        v = f["format_version"]
        v == _FORMAT_VERSION && return read(f)
        throw(ArgumentError("$path has format version $v; this package reads format version $_FORMAT_VERSION"))
    finally
        close(f)
    end
end

function _load_v1(f; on)
    spec, terms, constant = _read_system_terms(f)
    _, root = _read_node(f["root"]; on)
    return System(root, spec, terms, _read_theta(f, "theta"), constant, _read_augmented(f, root; on))
end

const _METADATA_SCALAR = Union{Base.BitInteger,Bool,Float16,Float32,Float64,ComplexF16,ComplexF32,ComplexF64,String}

# `metadata` as a `String`-keyed dict of plain values (arrays collected), or an ArgumentError naming the bad key.
function _check_metadata(metadata)
    out = Dict{String,Any}()
    for (k, v) in pairs(metadata)
        ok = v isa _METADATA_SCALAR || (v isa AbstractArray && eltype(v) <: _METADATA_SCALAR)
        ok &= !isempty(string(k)) && !('/' in string(k))
        ok || throw(ArgumentError("metadata \"$k\": the key must be non-empty without '/', the value a number, " *
                                  "Bool, String or array of them (got a $(typeof(v)))"))
        out[String(k)] = v isa AbstractArray ? collect(v) : v
    end
    return out
end

"""
    load_metadata(path) -> Dict{String,Any}

The `metadata` given to [`save_system`](@ref) (empty if none), without reading the solved system.
"""
function load_metadata(path::AbstractString)
    return _with_system_file(path) do f
        haskey(f, "metadata") || return Dict{String,Any}()
        g = f["metadata"]
        return Dict{String,Any}(k => g[k] for k in keys(g))
    end
end

# Positions of the node and given groups below group `g` (itself at `pos`), depth first.
function _positions!(out, g, pos)
    push!(out, pos)
    g["kind"] == "node" || return out
    for i in 1:g["children/count"]
        c = g["children/$i"]
        c["kind"] == "leaf" || _positions!(out, c, [pos; i])
    end
    return out
end

"""
    system_nodes(path) -> Vector{Vector{Int}}

Positions of the solved nodes in the file (root `Int[]` first, then each child position depth first; basis leaves are
not nodes), for [`load_node`](@ref). Reads no results.
"""
system_nodes(path::AbstractString) = _with_system_file(f -> _positions!(Vector{Int}[], f["root"], Int[]), path)

"""
    load_node(path, position=Int[]) -> (; energies, kept, vectors, dims, ports, on)

One solved node of the file at `position` (child indices from the root, see [`system_nodes`](@ref)), without reading the
rest. `ports` maps `(mode, port)` to its matrix; `on` is the saved device name, or `nothing` when the node has none.
"""
function load_node(path::AbstractString, position::AbstractVector{<:Integer}=Int[])
    return _with_system_file(path) do f
        g = f["root"]
        for i in position
            g["kind"] == "node" && 1 <= i <= g["children/count"] && g["children/$i/kind"] != "leaf" ||
                throw(ArgumentError("no node at position $(Vector(position)) in $path; " *
                                    "valid positions: $(_positions!(Vector{Int}[], f["root"], Int[]))"))
            g = g["children/$i"]
        end
        ports = _read_ports(g)
        has_on = g["kind"] == "node" && haskey(g, "settings/solver/fields/on")
        on = has_on ? g["settings/solver/fields/on/value"] : nothing
        return (; energies=g["energies"], kept=g["kept"], vectors=g["vectors"], dims=g["dims"], ports, on)
    end
end
