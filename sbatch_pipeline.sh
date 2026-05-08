#!/bin/bash
#SBATCH --job-name=tellu_pipeline
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=etienne.artigau@umontreal.ca

# Usage: sbatch sbatch_pipeline.sh [run_pipeline.py arguments]
# Example: sbatch sbatch_pipeline.sh --skip-sync --skip-hotstar

mkdir -p logs

export LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64:$LD_LIBRARY_PATH"

source ~/.bashrc
conda activate tellu_env

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "Args: $@"

cd "$SLURM_SUBMIT_DIR"

python run_pipeline.py "$@"

echo "Job $SLURM_JOB_ID finished at $(date)"
