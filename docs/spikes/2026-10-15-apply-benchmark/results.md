# KronFree apply-benchmark spike results (2026-10-15)

Throwaway benchmark of ways to apply a sum-of-products Hamiltonian to a state without building the Kronecker product. See bench.jl. Summary and conclusions are in Section 17 of docs/superpowers/specs/2026-10-15-kronfree-architecture-design.md.

## Mac M2, 8 threads, CPU (run_mac.log)

```
Julia 1.13.2, threads=8, BLAS threads=8, array=Array

=== A: 3 modes cosine-coupled  dims=(20, 20, 20)  n=8000  nterms=6
  sparse   nnz=2.17e+06       1.499 ms
  blas     err=1.6e-14       0.398 ms
  ka       err=1.6e-14       0.668 ms
  itensor  err=1.6e-14       0.453 ms
  -- KrylovKit eigsolve, lowest 10, tol 1e-8
     blas        1.13 s  converged=10  matvecs=237  E0=0.462848
     ka          0.40 s  converged=10  matvecs=237  E0=0.462848
     sparse      0.61 s  converged=10  matvecs=237  E0=0.462848

=== A: 3 modes cosine-coupled  dims=(40, 40, 40)  n=64000  nterms=6
  sparse   skipped (est nnz 3.1e+08)
  blas     err=0.0e+00       3.433 ms
  ka       err=5.2e-16      14.464 ms
  itensor  err=5.6e-15       6.112 ms

=== A: 3 modes cosine-coupled  dims=(60, 60, 60)  n=216000  nterms=6
  sparse   skipped (est nnz 2.4e+09)
  blas     err=0.0e+00      27.919 ms
  ka       err=6.8e-16      62.353 ms
  itensor  err=8.5e-15      32.238 ms

=== A: 3 modes cosine-coupled  dims=(80, 80, 80)  n=512000  nterms=6
  sparse   skipped (est nnz 1.0e+10)
  blas     err=0.0e+00      60.495 ms
  ka       err=7.6e-16     149.422 ms
  itensor  err=1.1e-14      63.923 ms

=== B: two leaves, dense 100x100 ports  dims=(100, 100)  n=10000  nterms=3
  sparse   skipped (est nnz 1.0e+08)
  blas     err=0.0e+00       0.565 ms
  ka       err=2.0e-16       1.389 ms
  itensor  err=8.7e-17       0.652 ms
  -- KrylovKit eigsolve, lowest 10, tol 1e-8
     blas        0.67 s  converged=10  matvecs=331  E0=0.106902
     ka          0.76 s  converged=10  matvecs=331  E0=0.106902

=== B: two leaves, dense 300x300 ports  dims=(300, 300)  n=90000  nterms=3
  sparse   skipped (est nnz 8.2e+09)
  blas     err=0.0e+00       9.495 ms
  ka       err=2.3e-16      41.346 ms
  itensor  err=1.0e-16      11.955 ms

DONE
exit=0
```

## ewdesktop: 16 threads CPU and RTX A4500 CUDA (desktop_logs.txt)

```
==INSTALL
    Manifest No packages added to or removed from `~/kronfree_spike/Manifest.toml`
install exit=0
==CPU
Julia 1.13.2, threads=16, BLAS threads=16, array=Array

=== A: 3 modes cosine-coupled  dims=(20, 20, 20)  n=8000  nterms=6
  sparse   nnz=2.17e+06       2.286 ms
  blas     err=1.6e-14       0.271 ms
  ka       err=1.6e-14       0.256 ms
  itensor  err=1.6e-14       0.255 ms
  -- KrylovKit eigsolve, lowest 10, tol 1e-8
     blas        1.29 s  converged=10  matvecs=237  E0=0.462848
     ka          0.22 s  converged=10  matvecs=237  E0=0.462848
     sparse      0.81 s  converged=10  matvecs=237  E0=0.462848

=== A: 3 modes cosine-coupled  dims=(40, 40, 40)  n=64000  nterms=6
  sparse   skipped (est nnz 3.1e+08)
  blas     err=0.0e+00       2.288 ms
  ka       err=4.7e-16      13.524 ms
  itensor  err=5.6e-15       3.440 ms

=== A: 3 modes cosine-coupled  dims=(60, 60, 60)  n=216000  nterms=6
  sparse   skipped (est nnz 2.4e+09)
  blas     err=0.0e+00       5.505 ms
  ka       err=6.6e-16      70.172 ms
  itensor  err=8.5e-15      20.215 ms

=== A: 3 modes cosine-coupled  dims=(80, 80, 80)  n=512000  nterms=6
  sparse   skipped (est nnz 1.0e+10)
  blas     err=0.0e+00      24.821 ms
  ka       err=7.9e-16     138.785 ms
  itensor  err=1.1e-14      72.522 ms

=== B: two leaves, dense 100x100 ports  dims=(100, 100)  n=10000  nterms=3
  sparse   skipped (est nnz 1.0e+08)
  blas     err=0.0e+00       0.140 ms
  ka       err=1.9e-16       0.596 ms
  itensor  err=8.9e-17       0.209 ms
  -- KrylovKit eigsolve, lowest 10, tol 1e-8
     blas        0.20 s  converged=10  matvecs=331  E0=0.106902
     ka          0.34 s  converged=10  matvecs=331  E0=0.106902

=== B: two leaves, dense 300x300 ports  dims=(300, 300)  n=90000  nterms=3
  sparse   skipped (est nnz 8.2e+09)
  blas     err=0.0e+00       2.431 ms
  ka       err=2.2e-16      45.855 ms
  itensor  err=9.0e-17       3.953 ms

DONE
exit=0
==CUDA
Julia 1.13.2, threads=16, BLAS threads=16, array=CuArray
NVIDIA RTX A4500

=== A: 3 modes cosine-coupled  dims=(20, 20, 20)  n=8000  nterms=6
  sparse   skipped (est nnz 1.0e+07)
  blas     err=0.0e+00       2.062 ms
  ka       err=3.1e-16       0.203 ms
  itensor  err=2.7e-15       0.586 ms
  -- KrylovKit eigsolve, lowest 10, tol 1e-8
     blas        3.35 s  converged=10  matvecs=237  E0=0.462848
     ka          0.61 s  converged=10  matvecs=237  E0=0.462848

=== A: 3 modes cosine-coupled  dims=(40, 40, 40)  n=64000  nterms=6
  sparse   skipped (est nnz 3.1e+08)
  blas     err=0.0e+00       4.954 ms
  ka       err=4.7e-16       1.352 ms
  itensor  err=5.6e-15       3.369 ms

=== A: 3 modes cosine-coupled  dims=(60, 60, 60)  n=216000  nterms=6
  sparse   skipped (est nnz 2.4e+09)
  blas     err=0.0e+00      29.047 ms
  ka       err=6.6e-16       5.177 ms
  itensor  err=8.5e-15       4.027 ms

=== A: 3 modes cosine-coupled  dims=(80, 80, 80)  n=512000  nterms=6
  sparse   skipped (est nnz 1.0e+10)
  blas     err=0.0e+00      19.163 ms
  ka       err=7.9e-16      12.908 ms
  itensor  err=1.1e-14       9.415 ms

=== A: 3 modes cosine-coupled  dims=(100, 100, 100)  n=1000000  nterms=6
  sparse   skipped (est nnz 3.0e+10)
  blas     err=0.0e+00      40.096 ms
  ka       err=9.0e-16      30.167 ms
  itensor  err=1.4e-14      27.336 ms

=== B: two leaves, dense 100x100 ports  dims=(100, 100)  n=10000  nterms=3
  sparse   skipped (est nnz 1.0e+08)
  blas     err=0.0e+00       0.348 ms
  ka       err=1.9e-16       0.241 ms
  itensor  err=1.1e-16       0.415 ms
  -- KrylovKit eigsolve, lowest 10, tol 1e-8
     blas        0.82 s  converged=10  matvecs=331  E0=0.106902
     ka          0.88 s  converged=10  matvecs=331  E0=0.106902

=== B: two leaves, dense 300x300 ports  dims=(300, 300)  n=90000  nterms=3
  sparse   skipped (est nnz 8.2e+09)
  blas     err=0.0e+00       3.217 ms
  ka       err=2.7e-16       4.175 ms
  itensor  err=1.2e-16       3.234 ms

=== B: two leaves, dense 500x500 ports  dims=(500, 500)  n=250000  nterms=3
  sparse   skipped (est nnz 6.3e+10)
  blas     err=0.0e+00      10.440 ms
  ka       err=2.6e-16      16.766 ms
  itensor  err=1.1e-16      10.615 ms

DONE
exit=0
```

## ewdesktop, extended variants permute and batched (desktop_logs2.txt)

```
==CUDA
Julia 1.13.2, threads=16, BLAS threads=16, array=CuArray
NVIDIA RTX A4500

=== A: 3 modes cosine-coupled  dims=(40, 40, 40)  n=64000  nterms=6
  sparse   skipped (est nnz 3.1e+08)
  blas     err=0.0e+00       5.054 ms
  ka       err=4.7e-16       0.950 ms
  permute  err=0.0e+00       1.389 ms
  batched  err=0.0e+00       1.563 ms
  itensor  err=5.6e-15       1.507 ms

=== A: 3 modes cosine-coupled  dims=(80, 80, 80)  n=512000  nterms=6
  sparse   skipped (est nnz 1.0e+10)
  blas     err=0.0e+00      18.979 ms
  ka       err=7.9e-16      12.287 ms
  permute  err=0.0e+00       9.404 ms
  batched  err=0.0e+00       9.286 ms
  itensor  err=1.1e-14       9.333 ms

=== A: 3 modes cosine-coupled  dims=(100, 100, 100)  n=1000000  nterms=6
  sparse   skipped (est nnz 3.0e+10)
  blas     err=0.0e+00      40.099 ms
  ka       err=9.0e-16      30.118 ms
  permute  err=0.0e+00      27.460 ms
  batched  err=0.0e+00      28.212 ms
  itensor  err=1.4e-14      27.293 ms

=== B: two leaves, dense 300x300 ports  dims=(300, 300)  n=90000  nterms=3
  sparse   skipped (est nnz 8.2e+09)
  blas     err=0.0e+00       3.140 ms
  ka       err=2.7e-16       3.949 ms
  permute  err=1.0e-16       3.220 ms
  batched  err=0.0e+00       3.206 ms
  itensor  err=1.2e-16       3.178 ms

=== B: two leaves, dense 500x500 ports  dims=(500, 500)  n=250000  nterms=3
  sparse   skipped (est nnz 6.3e+10)
  blas     err=0.0e+00      10.386 ms
  ka       err=2.6e-16      16.724 ms
  permute  err=8.5e-17      10.682 ms
  batched  err=0.0e+00      10.513 ms
  itensor  err=1.1e-16      10.560 ms

DONE
==CPU
Julia 1.13.2, threads=16, BLAS threads=16, array=Array

=== A: 3 modes cosine-coupled  dims=(80, 80, 80)  n=512000  nterms=6
  sparse   skipped (est nnz 1.0e+10)
  blas     err=0.0e+00      22.453 ms
  ka       err=7.9e-16     122.627 ms
  permute  err=0.0e+00      66.952 ms
  batched  err=0.0e+00      19.756 ms
  itensor  err=1.1e-14      78.968 ms

=== B: two leaves, dense 300x300 ports  dims=(300, 300)  n=90000  nterms=3
  sparse   skipped (est nnz 8.2e+09)
  blas     err=0.0e+00       2.573 ms
  ka       err=2.2e-16      30.995 ms
  permute  err=4.7e-17       3.750 ms
  batched  err=0.0e+00       2.434 ms
  itensor  err=9.0e-17       4.066 ms

DONE
```
