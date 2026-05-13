#!/bin/bash
#SBATCH -J mil_uq_train_mdsn
#SBATCH --nodes=1
#SBATCH --gres=gpu:A5000:1
#SBATCH --partition=A5000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# MD-SN training wrapper for train.py.
#
# Examples:
#   sbatch scripts/train_slurm.sh
#   sbatch scripts/train_slurm.sh dummy_data_es output/mdsn_smoke_es bert-base-multilingual-cased 1 2
#   sbatch scripts/train_slurm.sh dummy_data_es output/mdsn_bioehr PlanTL-GOB-ES/bsc-bio-ehr-es 12 4 384 42 8 ce --mdsn_fit oas
#
# Notes:
#   - train.py in this public repo always instantiates the MIL model with use_mdsn=True.
#   - This wrapper launches a single Python process, so it should request a single GPU.
#     If you want multi-GPU DDP, switch the launch mode as well (srun ntasks / accelerate launch).

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SUBMIT_DIR=$(readlink -f "$SLURM_SUBMIT_DIR")
    if [ "$(basename "$SUBMIT_DIR")" = "scripts" ]; then
        PACKAGE_ROOT=$(readlink -f "$SUBMIT_DIR/..")
        DEFAULT_PROJECT_ROOT=$(readlink -f "$PACKAGE_ROOT/..")
    elif [ "$(basename "$SUBMIT_DIR")" = "mil_uq_public" ]; then
        PACKAGE_ROOT="$SUBMIT_DIR"
        DEFAULT_PROJECT_ROOT=$(readlink -f "$PACKAGE_ROOT/..")
    else
        DEFAULT_PROJECT_ROOT="$SUBMIT_DIR"
        PACKAGE_ROOT=$(readlink -f "$DEFAULT_PROJECT_ROOT/mil_uq_public")
    fi
else
    SCRIPT_DIR=$(readlink -f "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)")
    PACKAGE_ROOT=$(readlink -f "$SCRIPT_DIR/..")
    DEFAULT_PROJECT_ROOT=$(readlink -f "$PACKAGE_ROOT/..")
fi

PROJECT_ROOT=$(readlink -f "${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}")
LOG_DIR=$(readlink -f "${LOG_DIR:-$PACKAGE_ROOT/logs}")

DATASET_DIR=${1:-dummy_data_es}
OUTPUT_DIR=${2:-"$PACKAGE_ROOT/output/mdsn_train_${SLURM_JOB_ID:-manual}"}
MODEL_CHECKPOINT=${3:-PlanTL-GOB-ES/bsc-bio-ehr-es}
EPOCHS=${4:-12}
BATCH_SIZE=${5:-4}
MAX_LENGTH=${6:-384}
SEED=${7:-42}
GRAD_ACCUM=${8:-8}
LOSS_TYPE=${9:-ce}
EXTRA_ARGS=("${@:10}")

TRAIN_CSV=${TRAIN_CSV:-"$PACKAGE_ROOT/$DATASET_DIR/train.csv"}
VAL_CSV=${VAL_CSV:-"$PACKAGE_ROOT/$DATASET_DIR/val.csv"}
NUM_WORKERS=${NUM_WORKERS:-0}
MAX_CHUNKS=${MAX_CHUNKS:-64}
CHUNK_OVERLAP=${CHUNK_OVERLAP:-64}
LR=${LR:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
CLASS_WEIGHT=${CLASS_WEIGHT:-none}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
DETERMINISTIC=${DETERMINISTIC:-0}
MDSN_COV_RIDGE=${MDSN_COV_RIDGE:-1e-6}
MDSN_N_POWER_ITERATIONS=${MDSN_N_POWER_ITERATIONS:-1}
MDSN_FIT=${MDSN_FIT:-oas}

# VENV_DIR=${VENV_DIR:-"$PACKAGE_ROOT/.venv"}
PYTHON_BIN_OVERRIDE=${PYTHON_BIN_OVERRIDE:-${PYTHON_BIN:-}}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [ ! -f "$TRAIN_CSV" ]; then
    echo "[train.mdsn] Missing train CSV: $TRAIN_CSV"
    exit 1
fi
if [ ! -f "$VAL_CSV" ]; then
    echo "[train.mdsn] Missing val CSV: $VAL_CSV"
    exit 1
fi

if [ -n "$PYTHON_BIN_OVERRIDE" ]; then
    if [ ! -x "$PYTHON_BIN_OVERRIDE" ]; then
        echo "[train.mdsn] Python override is not executable: $PYTHON_BIN_OVERRIDE"
        exit 1
    fi
    PYTHON_BIN="$PYTHON_BIN_OVERRIDE"
# elif [ -x "$VENV_DIR/bin/python" ]; then
#     PYTHON_BIN="$VENV_DIR/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
else
    echo "[train.mdsn] Could not find a Python interpreter."
    exit 1
fi

mkdir -p "$PACKAGE_ROOT/output"

echo "[train.mdsn] Job ID: ${SLURM_JOB_ID:-local}"
echo "[train.mdsn] Package root: $PACKAGE_ROOT"
echo "[train.mdsn] Python: $PYTHON_BIN"
echo "[train.mdsn] Dataset dir: $DATASET_DIR"
echo "[train.mdsn] Train CSV: $TRAIN_CSV"
echo "[train.mdsn] Val CSV: $VAL_CSV"
echo "[train.mdsn] Output dir: $OUTPUT_DIR"
echo "[train.mdsn] Model checkpoint: $MODEL_CHECKPOINT"
echo "[train.mdsn] Epochs: $EPOCHS"
echo "[train.mdsn] Batch size: $BATCH_SIZE"
echo "[train.mdsn] Max length: $MAX_LENGTH"
echo "[train.mdsn] Seed: $SEED"
echo "[train.mdsn] mdsn_fit: $MDSN_FIT"
echo "[train.mdsn] mdsn_cov_ridge: $MDSN_COV_RIDGE"
echo "[train.mdsn] mdsn_n_power_iterations: $MDSN_N_POWER_ITERATIONS"
echo ""

cd "$PACKAGE_ROOT"

CMD=(
    "$PYTHON_BIN" train.py
    --model_checkpoint "$MODEL_CHECKPOINT"
    --train_csv "$TRAIN_CSV"
    --val_csv "$VAL_CSV"
    --output_dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --grad_accum "$GRAD_ACCUM"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_ratio "$WARMUP_RATIO"
    --max_grad_norm "$MAX_GRAD_NORM"
    --max_length "$MAX_LENGTH"
    --max_chunks "$MAX_CHUNKS"
    --chunk_overlap "$CHUNK_OVERLAP"
    --num_workers "$NUM_WORKERS"
    --seed "$SEED"
    --loss_type "$LOSS_TYPE"
    --class_weight "$CLASS_WEIGHT"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --early_stop_min_delta "$EARLY_STOP_MIN_DELTA"
    --mdsn_cov_ridge "$MDSN_COV_RIDGE"
    --mdsn_n_power_iterations "$MDSN_N_POWER_ITERATIONS"
    --mdsn_fit "$MDSN_FIT"
)

if [ "$DETERMINISTIC" = "1" ]; then
    CMD+=(--deterministic)
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[train.mdsn] Launch command: ${CMD[*]}"
echo ""

if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    srun --unbuffered "${CMD[@]}"
else
    "${CMD[@]}"
fi

echo ""
echo "[train.mdsn] Training completed successfully."