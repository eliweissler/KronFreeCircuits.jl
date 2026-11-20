#!/usr/bin/env bash
#SBATCH --job-name=kfb-h200
#SBATCH --account=ucb795_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=kfb-h200-%j.out
#SBATCH --error=kfb-h200-%j.err
# One case on one full H200: sbatch [--time=...] h200.slurm <case>   (results/<case>-h200-<jobid>/)
set -uo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/case.sh" "${1:?case: hensel2 hensel2_scq hensel3 zeropi autodiff dmrg}" h200
