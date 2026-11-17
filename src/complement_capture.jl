# Debugging: with `_CAPTURE[]` set to a vector, every `_complement_solve` call pushes its exact inputs to it, so one solve
# can be replayed in isolation (bench/autodiff_c66/isolate_solve.jl, bench/coupled_zero_pi/isolate_solve.jl).
# `nothing` (the default) captures nothing. The library back ends compared against the built-in solver on 2026-11-17
# (Krylov.jl, KrylovKit) are in commit fcc0812; results in bench/coupled_zero_pi/README.md.
const _CAPTURE = Ref{Union{Nothing,Vector{Any}}}(nothing)
_capture!(H, D, basis, B, shifts, tol, maxiter, eta, scale) = _CAPTURE[] === nothing ||
    push!(_CAPTURE[], (; H, D, basis=copy(basis), B=copy(B), shifts=copy(shifts), tol, maxiter, eta,
                       scale=scale === nothing ? nothing : copy(scale)))
