#!/usr/bin/env bash
# One benchmark case on one GPU: bash case.sh <case> <smoke|h200>   (from a slurm script; see README.md)
set -uo pipefail
root="${SLURM_SUBMIT_DIR:?Submit through Slurm from the bundle directory}"
cd "$root"
source /projects/elwe6267/kronfree-julia-env.sh
REPO=/projects/elwe6267/src/KronFreeCircuits.jl
PY=/projects/elwe6267/software/anaconda/envs/scqubits-bench/bin/python
CASE="${1:?case}"; MODE="${2:?smoke or h200}"
OUT="$root/results/${CASE}-${MODE}-${SLURM_JOB_ID:?}"
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
export JULIA_NUM_PRECOMPILE_TASKS=4 BENCH_DEVICE=gpu
J=("$JULIA_BIN" +1.13.2 --startup-file=no --project="$REPO/bench" -t 8 --gcthreads=1)
SMOKE=0; [ "$MODE" = smoke ] && SMOKE=1
status=0
step() {   # step <name> <command...>: run, log its time and exit code, keep going on failure
    local name=$1; shift
    echo "==== $name  $(date -Is)"
    local t0=$SECONDS
    "$@" > "$OUT/$name.log" 2>&1; local rc=$?
    echo "==== $name  exit $rc  $((SECONDS - t0)) s"; tail -4 "$OUT/$name.log"
    [ $rc -ne 0 ] && status=1
    return 0
}
echo "case $CASE mode $MODE job $SLURM_JOB_ID host $(hostname) $(date -Is)"
echo "repo $(git -C "$REPO" rev-parse --short HEAD) $(git -C "$REPO" status --porcelain | wc -l) changed files"
nvidia-smi -L; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
lscpu | grep -E "Model name|^CPU\(s\)"
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm-job.txt"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > "$OUT/gpu-monitor.csv" 2>&1 &
mon=$!; trap 'kill $mon 2>/dev/null' EXIT

case "$CASE" in
  setup)       # instantiate and precompile the bench environment (CUDA artifacts included), check the GPU
    step instantiate "${J[@]}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA, cuTENSOR, KronFreeCircuits; CUDA.versioninfo()'
    step python "$PY" -c 'import torch, scqubits, scipy, numpy; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), scqubits.__version__, scipy.__version__, numpy.__version__)' ;;
  hensel2)     # KronFree: the ladder, the converged reference, the paper's flat space
    stages="ladder reference paper"; [ $SMOKE = 1 ] && stages="ladder"
    step hensel2 env BENCH_SMOKE=$SMOKE BENCH_OUT="$OUT/hensel2" "${J[@]}" "$REPO/bench/hensel_two_fluxonia/run.jl" $stages
    # the default fluxonium grid (±3π, 181) matches the desktop headline but has edge warnings on high leaf states;
    # the same ladder on ±7π, 421 points (same spacing; no kept state at the edge on any rung, A4500 2026-11-14;
    # ±5π still warned on the k1 60 rungs)
    [ $SMOKE = 0 ] && step hensel2_grid7 env HENSEL_GRID=7,421 BENCH_OUT="$OUT/hensel2_grid7" "${J[@]}" "$REPO/bench/hensel_two_fluxonia/run.jl" ladder ;;
  hensel2_grid7) # the ±7π ladder alone
    step hensel2_grid7 env BENCH_SMOKE=$SMOKE HENSEL_GRID=7,421 BENCH_OUT="$OUT/hensel2_grid7" "${J[@]}" "$REPO/bench/hensel_two_fluxonia/run.jl" ladder ;;
  hensel2_scq) # scqubits on the same tree, torch.lobpcg on the GPU at M and the root
    cd "$OUT"
    if [ $SMOKE = 1 ]; then
      step scq_kM120 env MATCHED_FLUXES=0.3 "$PY" "$REPO/bench/hensel_two_fluxonia/scqubits_ref/matched.py" 120 torch-cuda shear
    else
      step scq_kM120 "$PY" "$REPO/bench/hensel_two_fluxonia/scqubits_ref/matched.py" 120 torch-cuda shear
      step scq_kM240 "$PY" "$REPO/bench/hensel_two_fluxonia/scqubits_ref/matched.py" 240 torch-cuda shear
    fi ;;
  hensel3)     # KronFree: the four-level tree's ladder in the rule frame
    step hensel3 env BENCH_SMOKE=$SMOKE BENCH_OUT="$OUT/hensel3" "${J[@]}" "$REPO/bench/hensel_three_fluxonia/run.jl" ;;
  zeropi)      # KronFree: 1000 coupled levels, LOBPCG root; leaves on the CPU, then on the GPU
    reps=5; [ $SMOKE = 1 ] && reps=1
    step zp_gpu_root env BENCH_SMOKE=$SMOKE BENCH_REPEATS=$reps ZP_ROOT=lobpcg BENCH_OUT="$OUT/gpu_root" "${J[@]}" "$REPO/bench/coupled_zero_pi/run.jl"
    step zp_gpu_all env BENCH_SMOKE=$SMOKE BENCH_REPEATS=$reps ZP_ROOT=lobpcg ZP_LEAF=device BENCH_OUT="$OUT/gpu_all" "${J[@]}" "$REPO/bench/coupled_zero_pi/run.jl" ;;
  autodiff)    # KronFree derivatives on the GPU (both circuits), scqubits torch.lobpcg on the GPU (both circuits; the complex c66 root through its real embedding)
    n=10; [ $SMOKE = 1 ] && n=2
    step kf_fluxonium env KF_DEVICE=gpu KF_OUT="$OUT/kronfree_fluxonium_gpu.json" "${J[@]}" "$REPO/bench/autodiff_fluxonium/kronfree.jl"
    step kf_c66 env KF_DEVICE=gpu KF_OUT="$OUT/kronfree_c66_gpu.json" "${J[@]}" "$REPO/bench/autodiff_c66/kronfree.jl"
    step scq_fluxonium env SCQ_SOLVER=torch-cuda SCQ_OUT="$OUT/scqubits_fluxonium_torch-cuda.json" "$PY" "$REPO/bench/scqubits_solve/scq_solve.py" fluxonium $n
    step scq_c66 env SCQ_SOLVER=torch-cuda SCQ_OUT="$OUT/scqubits_c66_torch-cuda.json" "$PY" "$REPO/bench/scqubits_solve/scq_solve.py" c66 $n ;;
  dmrg_setup)  # instantiate and precompile the array-fluxonium DMRG environments (ITensors; CPU, and GPU with CUDA)
    AF="$REPO/bench/array_fluxonium"
    step dmrg_instantiate_cpu "$JULIA_BIN" +1.13.2 --startup-file=no --project="$AF/dmrg" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using ITensorMPS, KronFreeCircuits'
    step dmrg_instantiate_gpu "$JULIA_BIN" +1.13.2 --startup-file=no --project="$AF/dmrg_gpu" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using ITensorMPS, CUDA, Adapt, KronFreeCircuits; CUDA.versioninfo()' ;;
  dmrg)        # array fluxonium N = 40, z = 0.095, n_g = 0 at the production settings, 13 sweeps fixed (desktop 2026-11-16:
               # A4500 423 s, i7 8 BLAS threads 1648 s, 1 thread 2184 s); GPU, then this node's CPU with 8 BLAS threads
    AF="$REPO/bench/array_fluxonium"
    sweeps=13; [ $SMOKE = 1 ] && sweeps=1
    [ -f "$AF/specs/array_N40_z0.095.json" ] || { echo "missing $AF/specs/array_N40_z0.095.json: pull bench-harness"; exit 2; }
    common=(SPEC="$AF/specs/array_N40_z0.095.json" NG=0 D=14 NCUT=12 ESHIFT=auto MAXDIM=150 CUTOFF=1e-12 ETOL=0
            KRYLOVDIM=20 EIGMAXITER=4 SWEEPS=$sweeps)
    step dmrg_gpu env "${common[@]}" DEVICE=gpu "$JULIA_BIN" +1.13.2 --startup-file=no --project="$AF/dmrg_gpu" -t 1 "$AF/dmrg/dmrg_spec.jl"
    step dmrg_cpu8 env "${common[@]}" BLASTHREADS=8 "$JULIA_BIN" +1.13.2 --startup-file=no --project="$AF/dmrg" -t 1 "$AF/dmrg/dmrg_spec.jl"
    grep -h "DMRG eigen" "$OUT"/dmrg_*.log ;;
  *) echo "unknown case $CASE"; exit 2 ;;
esac
echo "edge warnings (PhaseGrid range): $(grep -l 'PhaseGrid on mode' "$OUT"/*.log 2>/dev/null | wc -l) logs; $(grep -h 'PhaseGrid on mode' "$OUT"/*.log 2>/dev/null | wc -l) lines"
echo "DONE $CASE $MODE status $status $(date -Is)"
exit $status
