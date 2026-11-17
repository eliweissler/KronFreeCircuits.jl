#!/usr/bin/env bash
#SBATCH --job-name=kfb-smoke
#SBATCH --account=ucb795_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=kfb-smoke-%j.out
#SBATCH --error=kfb-smoke-%j.err
# Every case at smoke size on a 20 GB A100 slice, after instantiating the environments. Usage: sbatch smoke.slurm
# [cases ...] (default: all). Each case's results land in results/<case>-smoke-<jobid>/.
set -uo pipefail
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(setup autodiff zeropi hensel2 hensel2_scq hensel3)
rc=0
for c in "${cases[@]}"; do bash "${SLURM_SUBMIT_DIR:?}/case.sh" "$c" smoke || rc=1; done
exit $rc
