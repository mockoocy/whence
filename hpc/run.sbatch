#!/bin/bash -l
# usage: DATA_DIR=... VENV_DIR=... [JOBS=n] sbatch run.sbatch <experiment-dir>...
# e.g.   DATA_DIR=$PROJECT_DIR/data VENV_DIR=$PROJECT_DIR/.venv-gh200 \
#          sbatch run.sbatch experiments/zscore/*
# JOBS: arms run at once, each in its own process (default 1)
# Slurm does not create the log directory: mkdir -p logs first.
#SBATCH --job-name=esa-adb-cad-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the directory holding ESA_Mission1.parquet}"
: "${VENV_DIR:?set VENV_DIR to the virtualenv with whence installed}"

module load GCCcore/15.2.0
module load Python/3.14.2
source "$VENV_DIR/bin/activate"

JOBS="${JOBS:-1}"
# split the cores between the arms, or each starts one thread per core
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / JOBS > 0 ? SLURM_CPUS_PER_TASK / JOBS : 1 ))
fi

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
srun whence run --jobs "$JOBS" --data-root "$DATA_DIR" "$@"
