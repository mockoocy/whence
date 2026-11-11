#!/bin/bash -l
# usage: DATA_DIR=... VENV_DIR=... [UV=...] [JOBS=n] sbatch run.sbatch <experiment-dir>...
# e.g.   DATA_DIR=$PROJECT_DIR/data VENV_DIR=$PROJECT_DIR/.venv-gh200 \
#          sbatch run.sbatch experiments/zscore/*
# JOBS: arms run at once, each in its own process (default 1)
# Slurm does not create the log directory: mkdir -p logs first.
#SBATCH --job-name=esa-adb-cad-baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=memfs
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the directory holding ESA_Mission1.parquet}"
: "${VENV_DIR:?set VENV_DIR to a uv virtualenv, whose Python is copied to the node}"
UV="${UV:-$HOME/.local/bin/uv}"

cd "$SLURM_SUBMIT_DIR"
LOCAL=$MEMFS

PY=$(readlink -f "$VENV_DIR/bin/python")
cp -r "$(dirname "$(dirname "$PY")")" "$LOCAL/python"
export UV_PROJECT_ENVIRONMENT=$LOCAL/venv UV_LINK_MODE=copy UV_OFFLINE=1
time "$UV" sync --frozen --python "$LOCAL/python/bin/$(basename "$PY")"
source "$LOCAL/venv/bin/activate"

JOBS="${JOBS:-1}"
# split the cores between the arms, or each starts one thread per core
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / JOBS > 0 ? SLURM_CPUS_PER_TASK / JOBS : 1 ))
fi

export PYTHONUNBUFFERED=1
srun whence run --jobs "$JOBS" --data-root "$DATA_DIR" "$@"
