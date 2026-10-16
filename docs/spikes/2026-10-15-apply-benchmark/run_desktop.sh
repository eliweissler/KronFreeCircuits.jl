#!/bin/bash
export PATH=$HOME/.juliaup/bin:$PATH
cd ~/kronfree_spike
julia --project=. -e 'using Pkg; Pkg.add(["KernelAbstractions","KrylovKit","QuantumToolbox","ITensors","CUDA"]); Pkg.precompile()' > install.log 2>&1
echo "install exit=$?" >> install.log
BENCH_ARRAY=Array julia -t auto --project=. bench.jl > run_cpu.log 2>&1
echo "exit=$?" >> run_cpu.log
BENCH_ARRAY=CuArray BENCH_DS=20,40,60,80,100 BENCH_KS=100,300,500 julia -t auto --project=. bench.jl > run_cuda.log 2>&1
echo "exit=$?" >> run_cuda.log
echo ALLDONE > done.flag
