#!/bin/bash
# Example: Train a MIL model with cross-entropy loss.
# Adapt paths and hyperparameters to your setup.

set -euo pipefail

MODEL_CHECKPOINT="bert-base-uncased"
TRAIN_CSV="dummy_data/train.csv"
VAL_CSV="dummy_data/val.csv"
OUTPUT_DIR="output/mil_model"

python train.py \
  --model_checkpoint "$MODEL_CHECKPOINT" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --arch mil \
  --loss ce \
  --epochs 10 \
  --lr 2e-5 \
  --batch_size 4 \
  --max_length 4096 \
  --seed 42
