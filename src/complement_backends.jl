# Experimental library back ends for `_complement_solve` (2026-11-17): the same projected, shifted, Jacobi-preconditioned
# CG solve as the built-in block solver, done one column at a time by Krylov.jl's `cg` or by KrylovKit's `linsolve`.
# Chosen per call by the environment variable KRONFREE_LINSOLVE = builtin (default) | krylov | krylovkit, so a test
# or benchmark run can switch without code changes. Same contract and return value as `_complement_solve`.
using Krylov: Krylov

_complement_backend() = Symbol(get(ENV, "KRONFREE_LINSOLVE", "builtin"))

# Debugging: with `_CAPTURE[]` set to a vector, every `_complement_solve` call pushes its exact inputs to it, so one solve
# can be replayed in isolation (bench/autodiff_c66/isolate_solve.jl). `nothing` (the default) captures nothing.
const _CAPTURE = Ref{Union{Nothing,Vector{Any}}}(nothing)
_capture!(H, D, basis, B, shifts, tol, maxiter, eta, scale) = _CAPTURE[] === nothing ||
    push!(_CAPTURE[], (; H, D, basis=copy(basis), B=copy(B), shifts=copy(shifts), tol, maxiter, eta,
                       scale=scale === nothing ? nothing : copy(scale)))
# Per-column reports of the library back ends (iterations, status, seconds), for debugging.
_verbose() = get(ENV, "KRONFREE_LINSOLVE_VERBOSE", "0") == "1"

# y = Q (H − s) Q x on one column, Q = 1 − Φ Φ'; `t` and `c` are scratch on the operator's storage.
struct _ShiftedComplement{TH,TB,TV,TC}
    H::TH
    basis::TB
    s::Float64
    t::TV
    c::TC
end
Base.size(A::_ShiftedComplement) = (size(A.basis, 1), size(A.basis, 1))
Base.size(A::_ShiftedComplement, i::Int) = size(A)[i]
Base.eltype(A::_ShiftedComplement) = eltype(A.t)
function _project_vec!(y, basis, c)
    mul!(c, basis', y)
    mul!(y, basis, c, -1, 1)
    return y
end
function LinearAlgebra.mul!(y::AbstractVector, A::_ShiftedComplement, x::AbstractVector)
    copyto!(A.t, x)
    _project_vec!(A.t, A.basis, A.c)
    mul!(y, A.H, A.t)
    y .-= A.s .* A.t
    return _project_vec!(y, A.basis, A.c)
end

# The projected Jacobi preconditioner y = Q (m .* Q x), symmetric positive semidefinite on the complement.
struct _ProjectedJacobi{TB,TV,TC}
    basis::TB
    m::TV
    c::TC
end
Base.size(P::_ProjectedJacobi) = (length(P.m), length(P.m))
Base.eltype(P::_ProjectedJacobi) = eltype(P.m)
function LinearAlgebra.mul!(y::AbstractVector, P::_ProjectedJacobi, x::AbstractVector)
    copyto!(y, x)
    _project_vec!(y, P.basis, P.c)
    y .*= P.m
    return _project_vec!(y, P.basis, P.c)
end

function _complement_solve_lib(backend::Symbol, H::KronFreeOperator, D::AbstractVector, basis::AbstractMatrix,
                               B::AbstractMatrix, shifts::Vector{Float64}; tol::Real, maxiter::Int, eta::Real=0.01,
                               scale::Union{Nothing,AbstractVector{<:Real}}=nothing)
    backend in (:krylov, :krylovkit, :krylov_stacked) ||
        throw(ArgumentError("KRONFREE_LINSOLVE: builtin, krylov, krylov_stacked or krylovkit, got $backend"))
    n = size(B, 2)
    length(shifts) == n || throw(DimensionMismatch("$n columns and $(length(shifts)) shifts"))
    scale = scale === nothing ? vec(Array(_colnorms(B))) : collect(Float64, scale)
    length(scale) == n || throw(DimensionMismatch("$n columns and $(length(scale)) scales"))
    C = similar(B, size(basis, 2), n)
    project!(Y) = (mul!(C, basis', Y); mul!(Y, basis, C, -1, 1); Y)
    cap = _FLOOR_CAP * eps(Float64) .* scale
    QB = project!(copy(B))
    b = vec(Array(_colnorms(QB)))
    goal = tol .* b
    from_start = findall(b .<= cap)
    X = fill!(similar(B), 0)
    met = falses(n)
    atfloor = falses(n)
    atfloor[from_start] .= true
    iterations = 0
    t = similar(B, size(B, 1)); c = similar(B, size(basis, 2)); tr_buf = similar(B, size(B, 1))
    if backend === :krylov_stacked
        iterations = _stacked_cg!(X, H, D, basis, QB, shifts, project!, goal, cap, met, atfloor, eta, maxiter)
    end
    for j in 1:n
        (backend === :krylov_stacked || atfloor[j]) && continue
        A = _ShiftedComplement(H, basis, shifts[j], t, c)
        m = 1 ./ hypot.(D .- shifts[j], eta)
        bj = QB[:, j]
        truer(x) = (mul!(tr_buf, A, x); tr_buf .-= bj; norm(tr_buf))
        # the round-off floor rule of the built-in solver, on the true residual once the recurrence is under the cap
        anchor, stalled, floored = Inf, 0, false
        function stop(r, x)
            norm(r) <= goal[j] && return true
            if norm(r) <= cap[j]
                tr = truer(x)
                tr < anchor / 2 ? ((anchor, stalled) = (tr, 0)) : (stalled += 1)
                stalled >= _FLOOR_PATIENCE && tr <= cap[j] && (floored = true; return true)
            end
            return false
        end
        if backend === :krylov
            P = _ProjectedJacobi(basis, m, similar(c))
            ws = Krylov.CgWorkspace(A, bj)
            Krylov.cg!(ws, A, bj; M=P, ldiv=false, linesearch=true, atol=0.0, rtol=0.0, itmax=maxiter,
                       callback=w -> stop(w.r, w.x))
            st = ws.stats
            iterations = max(iterations, st.niter)
            _verbose() && @info "Krylov.jl cg, column $j" st.niter st.status st.indefinite st.timer residual = truer(Krylov.solution(ws)) / b[j] goal = tol
            if st.indefinite && truer(Krylov.solution(ws)) > cap[j]
                error("the linear solve on the complement is not positive on the complement at column $j " *
                      "(nonpositive curvature after $(st.niter) iterations): a state below or degenerate with the kept " *
                      "state was not computed (or was not retained); raise `compute` (or retain the node's computed states)")
            end
            X[:, j] .= Krylov.solution(ws)
        else
            # split preconditioning: CG on S' (H − s) S with S = Q diag(√m), x = S y; the same preconditioned CG
            sq = sqrt.(m)
            Sy(y) = _project_vec!(sq .* y, basis, c)
            f(y) = (z = similar(y); mul!(z, A, Sy(y)); sq .* z)
            bt = sq .* bj
            # KrylovKit stops on the transformed residual; tighten it until the true residual meets the target
            rt = tol; x = zero(bj)
            for _ in 1:4
                y, info = KrylovKit.linsolve(f, bt, zero(bt); isposdef=true, ishermitian=true, rtol=rt, atol=0.0,
                                             maxiter=maxiter, krylovdim=min(maxiter, 100), verbosity=0)
                iterations = max(iterations, info.numops)
                x = Sy(y)
                tr = truer(x)
                tr <= goal[j] && break
                tr <= cap[j] && (floored = true; break)
                rt /= 10
            end
            X[:, j] .= x
        end
        floored && (atfloor[j] = true)
    end
    # the built-in solver's final check, on fresh true residuals
    T = similar(B)
    mul!(T, H, X); T .-= X .* _to_storage(B, reshape(shifts, 1, :)); project!(T); T .-= QB
    tr = vec(Array(_colnorms(T)))
    for j in 1:n
        j in from_start && continue
        if tr[j] <= 10 * goal[j] && !atfloor[j]
            met[j] = true
        elseif tr[j] <= cap[j]
            atfloor[j] = true
        else
            error("the linear solve on the complement did not converge at column $j: true residual " *
                  "$(tr[j] / b[j]) relative to its complement part, $(tr[j] / scale[j]) relative to the right-hand " *
                  "side ($backend, tolerance $tol, at most $maxiter iterations)")
        end
    end
    residuals = [met[j] ? tr[j] / b[j] : 0.0 for j in 1:n]
    at_floor = [j for j in 1:n if atfloor[j] && !(j in from_start)]
    floor_residuals = [j in at_floor ? tr[j] / scale[j] : 0.0 for j in 1:n]
    return X, iterations, residuals, from_start, at_floor, floor_residuals
end

# The k columns stacked into one vector of length n k, and one Krylov.jl `cg` on the direct sum
# diag(Q (H − s_1) Q, …, Q (H − s_k) Q): each operator application is one block apply of H to the n × k block, as in the
# built-in solver, but CG's step sizes are shared by all columns. Stops when every column has met its target or reached
# the round-off floor (the built-in rule, per column, in the callback). Columns converged from the start have a zero
# right-hand side and stay zero. Returns the iteration count; fills X, met and atfloor.
struct _StackedComplement{TH,TB,TM,TS,TP}
    H::TH
    basis::TB
    T::TM
    S::TS
    project!::TP
end
Base.size(A::_StackedComplement) = (length(A.T), length(A.T))
Base.size(A::_StackedComplement, i::Int) = size(A)[i]
Base.eltype(A::_StackedComplement) = eltype(A.T)
function LinearAlgebra.mul!(y::AbstractVector, A::_StackedComplement, x::AbstractVector)
    copyto!(A.T, reshape(x, size(A.T)))
    A.project!(A.T)
    Y = reshape(y, size(A.T))
    mul!(Y, A.H, A.T)
    Y .-= A.T .* A.S
    A.project!(Y)
    return y
end
struct _StackedJacobi{TM,TP}
    M::TM
    project!::TP
end
Base.size(P::_StackedJacobi) = (length(P.M), length(P.M))
Base.eltype(P::_StackedJacobi) = eltype(P.M)
function LinearAlgebra.mul!(y::AbstractVector, P::_StackedJacobi, x::AbstractVector)
    Y = reshape(y, size(P.M))
    copyto!(Y, reshape(x, size(P.M)))
    P.project!(Y)
    Y .*= P.M
    P.project!(Y)
    return y
end

function _stacked_cg!(X, H, D, basis, QB, shifts, project!, goal, cap, met, atfloor, eta, maxiter)
    n, k = size(QB)
    S = _to_storage(QB, reshape(shifts, 1, :))
    A = _StackedComplement(H, basis, similar(QB), S, project!)
    P = _StackedJacobi(1 ./ hypot.(D .- S, eta), project!)
    QB0 = copy(QB); QB0 .*= _to_storage(QB, reshape(Float64.(.!atfloor), 1, :))     # from-start columns: zero
    b = vec(QB0)
    TR = similar(QB)
    anchor = fill(Inf, k); stalled = zeros(Int, k)
    function stop(w)
        rn = vec(Array(_colnorms(reshape(w.r, n, k))))
        met .|= .!atfloor .& (rn .<= goal)
        near = .!(met .| atfloor) .& (rn .<= cap)
        if any(near)
            mul!(vec(TR), A, w.x); TR .-= QB0
            tr = vec(Array(_colnorms(TR)))
            for j in findall(near)
                tr[j] < anchor[j] / 2 ? ((anchor[j], stalled[j]) = (tr[j], 0)) : (stalled[j] += 1)
                stalled[j] >= _FLOOR_PATIENCE && tr[j] <= cap[j] && (atfloor[j] = true)
            end
        end
        return all(met .| atfloor)
    end
    all(atfloor) && return 0
    ws = Krylov.CgWorkspace(A, b)
    Krylov.cg!(ws, A, b; M=P, ldiv=false, linesearch=true, atol=0.0, rtol=0.0, itmax=maxiter, callback=stop)
    st = ws.stats
    _verbose() && @info "Krylov.jl cg, stacked" st.niter st.status st.indefinite st.timer
    if st.indefinite && !all(met .| atfloor)
        error("the linear solve on the complement is not positive on the complement (nonpositive curvature after " *
              "$(st.niter) iterations): a state below or degenerate with the kept state was not computed (or was not " *
              "retained); raise `compute` (or retain the node's computed states)")
    end
    X .= reshape(Krylov.solution(ws), n, k)
    met .= false                       # decided again by the final check on fresh true residuals
    return st.niter
end
