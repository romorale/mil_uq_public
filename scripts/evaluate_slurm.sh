#!/bin/bash
#SBATCH -J mil_uq_eval_mdsn
#SBATCH --nodes=1
#SBATCH --gres=gpu:A5000:1
#SBATCH --partition=A5000
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# MD-SN evaluation wrapper for evaluate.py.
#
# Examples:
#   sbatch scripts/evaluate_slurm.sh
#   sbatch scripts/evaluate_slurm.sh dummy_data_es output/mdsn_train_123 output/mdsn_eval_123
#   sbatch scripts/evaluate_slurm.sh dummy_data_es output/mdsn_train_123 output/mdsn_eval_123 8 42 1000 --dist_quantile 0.995
#
# Notes:
#   - This wrapper fixes --arch mil and --uq_method mdsn.
#   - For the canonical MD-SN path, keep DIST_MODEL=mahalanobis (default).

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
MODEL_PATH=${2:-"$PACKAGE_ROOT/output/mdsn_train"}
OUTPUT_DIR=${3:-"$PACKAGE_ROOT/output/mdsn_train/mdsn_eval_${SLURM_JOB_ID:-manual}"}
BATCH_SIZE=${4:-8}
SEED=${5:-42}
TEST_BOOTSTRAP_N=${6:-0}
EXTRA_ARGS=("${@:7}")

VAL_CSV=${VAL_CSV:-"$PACKAGE_ROOT/$DATASET_DIR/val.csv"}
TEST_CSV=${TEST_CSV:-"$PACKAGE_ROOT/$DATASET_DIR/test.csv"}
ALPHA=${ALPHA:-0.01}
UNC_METRIC=${UNC_METRIC:-mi}
UNC_PERCENTILE=${UNC_PERCENTILE:-90.0}
DIST_MODEL=${DIST_MODEL:-mahalanobis}
DIST_FIT_SOURCE=${DIST_FIT_SOURCE:-val}
DIST_QUANTILE=${DIST_QUANTILE:-0.99}
DIST_TRANSFORM=${DIST_TRANSFORM:-raw}
DIST_K0=${DIST_K0:-1}
DIST_K1=${DIST_K1:-4}
PROB_BORDER_EPS=${PROB_BORDER_EPS:-0.0}
TEST_BOOTSTRAP_SEED=${TEST_BOOTSTRAP_SEED:-$SEED}
SAVE_CALIBRATION_FILE=${SAVE_CALIBRATION_FILE:-"$OUTPUT_DIR/calibration_bundle.npz"}
SKIP_SAVE_CALIBRATION=${SKIP_SAVE_CALIBRATION:-0}
MAX_LENGTH=${MAX_LENGTH:-4096}

PYTHON_BIN_OVERRIDE=${PYTHON_BIN_OVERRIDE:-${PYTHON_BIN:-}}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [ ! -d "$MODEL_PATH" ]; then
    echo "[eval.mdsn] Missing model directory: $MODEL_PATH"
    exit 1
fi
if [ ! -f "$VAL_CSV" ]; then
    echo "[eval.mdsn] Missing val CSV: $VAL_CSV"
    exit 1
fi
if [ ! -f "$TEST_CSV" ]; then
    echo "[eval.mdsn] Missing test CSV: $TEST_CSV"
    exit 1
fi

if [ -n "$PYTHON_BIN_OVERRIDE" ]; then
    if [ ! -x "$PYTHON_BIN_OVERRIDE" ]; then
        echo "[eval.mdsn] Python override is not executable: $PYTHON_BIN_OVERRIDE"
        exit 1
    fi
    PYTHON_BIN="$PYTHON_BIN_OVERRIDE"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
else
    echo "[eval.mdsn] Could not find a Python interpreter."
    exit 1
fi

mkdir -p "$PACKAGE_ROOT/output"
mkdir -p "$LOG_DIR"

echo "[eval.mdsn] Job ID: ${SLURM_JOB_ID:-local}"
echo "[eval.mdsn] Package root: $PACKAGE_ROOT"
echo "[eval.mdsn] Project root: $PROJECT_ROOT"
echo "[eval.mdsn] Python: $PYTHON_BIN"
echo "[eval.mdsn] Dataset dir: $DATASET_DIR"
echo "[eval.mdsn] Val CSV: $VAL_CSV"
echo "[eval.mdsn] Test CSV: $TEST_CSV"
echo "[eval.mdsn] Model path: $MODEL_PATH"
echo "[eval.mdsn] Output dir: $OUTPUT_DIR"
echo "[eval.mdsn] Batch size: $BATCH_SIZE"
echo "[eval.mdsn] Seed: $SEED"
echo "[eval.mdsn] dist_model: $DIST_MODEL"
echo "[eval.mdsn] dist_fit_source: $DIST_FIT_SOURCE"
echo "[eval.mdsn] dist_quantile: $DIST_QUANTILE"
echo "[eval.mdsn] dist_transform: $DIST_TRANSFORM"
echo ""

cd "$PACKAGE_ROOT"

CMD=(
    "$PYTHON_BIN" evaluate.py
    --model_path "$MODEL_PATH"
    --val_csv "$VAL_CSV"
    --test_csv "$TEST_CSV"
    --output_dir "$OUTPUT_DIR"
    --arch mil
    --uq_method mdsn
    --max_length "$MAX_LENGTH"
    --unc_metric "$UNC_METRIC"
    --unc_percentile "$UNC_PERCENTILE"
    --alpha "$ALPHA"
    --dist_model "$DIST_MODEL"
    --dist_fit_source "$DIST_FIT_SOURCE"
    --dist_k0 "$DIST_K0"
    --dist_k1 "$DIST_K1"
    --dist_quantile "$DIST_QUANTILE"
    --dist_transform "$DIST_TRANSFORM"
    --prob_border_eps "$PROB_BORDER_EPS"
    --batch_size "$BATCH_SIZE"
    --seed "$SEED"
    --test_bootstrap_n "$TEST_BOOTSTRAP_N"
    --test_bootstrap_seed "$TEST_BOOTSTRAP_SEED"
)

if [ "$SKIP_SAVE_CALIBRATION" = "1" ]; then
    CMD+=(--skip_save_calibration)
else
    CMD+=(--save_calibration_file "$SAVE_CALIBRATION_FILE")
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[eval.mdsn] Launch command: ${CMD[*]}"
echo ""

if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    srun --unbuffered "${CMD[@]}"
else
    "${CMD[@]}"
fi

echo ""
echo "[eval.mdsn] Evaluation completed successfully."