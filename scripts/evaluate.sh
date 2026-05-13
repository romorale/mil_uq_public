#!/bin/bash
# Example: Evaluate a trained MIL model with MC-Dropout UQ.
# Adapt paths to your setup.

set -euo pipefail

MODEL_PATH="output/mil_model"
VAL_CSV="dummy_data/val.csv"
TEST_CSV="dummy_data/test.csv"
OUTPUT_DIR="output/eval_mc"

python evaluate.py \
  --model_path "$MODEL_PATH" \
  --val_csv "$VAL_CSV" \
  --test_csv "$TEST_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --arch mil \
  --uq_method mc \
  --mc_iters 20 \
  --unc_metric mi \
  --alpha 0.01 \
  --dist_model mixture \
  --dist_k0 1 \
  --dist_k1 4 \
  --dist_quantile 0.99 \
  --batch_size 8 \
  --seed 42 \
  --test_bootstrap_n 1000
