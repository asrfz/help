#!/bin/bash
#SBATCH --job-name=boltz_no_cli
#SBATCH --account=def-abdinosa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/no_cli_%j.out
#SBATCH --error=logs/no_cli_%j.err

set -euo pipefail

cd $SCRATCH/boltz_project
mkdir -p logs


module load StdEnv/2023
module load python/3.11 cuda cudnn rdkit/2024.03.4

# venv that can see rdkit module
source $SCRATCH/boltz_project/boltz_env311_sys/bin/activate

# ensure the small missing deps exist in the venv
python -m pip install --no-index typing_extensions mpmath

# pick an obvious output folder
export SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}
export NO_CLI_OUT="$SCRATCH/no_cli_output_${SLURM_JOB_ID}"
mkdir -p "$NO_CLI_OUT"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "PWD: $(pwd)"
echo "NO_CLI_OUT: $NO_CLI_OUT"

# If your python script uses WORKDIR from code, set it via env var (optional):
# export WORKDIR="$NO_CLI_OUT"

python boltz_no_cli_affinity.py

echo "End: $(date)"
