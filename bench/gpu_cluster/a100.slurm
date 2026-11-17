#!/usr/bin/env bash
#SBATCH --job-name=kfb-a100
#SBATCH --account=ucb795_asc1
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=78G
#SBATCH --time=04:00:00
#SBATCH --output=kfb-a100-%j.out
#SBATCH --error=kfb-a100-%j.err
# One case on one full 80 GB A100: sbatch [--time=...] a100.slurm <case>   (results/<case>-a100-<jobid>/)
set -uo pipefail
exec bash "${SLURM_SUBMIT_DIR:?}/case.sh" "${1:?case: hensel2 hensel2_scq hensel3 zeropi autodiff}" a100
