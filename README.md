# MIL Uncertainty Quantification

A framework for **uncertainty quantification** in text classification using
**Multiple Instance Learning (MIL)** and standard HuggingFace architectures.

## Features

| Method | Key | Description |
|---|---|---|
| Deterministic | `none` | Standard softmax baseline |
| MC-Dropout | `mc` | Monte-Carlo Dropout for epistemic uncertainty |
| MD-SN | `mdsn` | Mahalanobis Distance with Spectral Normalization |
| CV Ensemble | `cv_ensemble` | Cross-validation ensemble with OOF calibration |

**Decision pipeline:**

1. **Temperature scaling** — LBFGS-based post-hoc calibration.
2. **Mondrian Conformal Prediction** — set-valued predictions with coverage guarantees.
3. **Mahalanobis distance veto** — OOD/epistemic rejection via embedding-space distance
   (single-centroid or mixture-of-centroids with OAS shrinkage covariance).
4. **Triage output** — each sample is classified as *Clear* (positive/negative) or *Deferred*
   (ambiguous, null, epistemic veto, high uncertainty).

## Repository Structure

```
mil_uq_public/
├── train.py                 # Training script (MIL + HF)
├── evaluate.py              # Full evaluation pipeline
├── requirements.txt
├── dummy_data/              # Small synthetic CSV files for testing
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── dummy_data_es/           # Spanish version of the synthetic CSV files
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   └── attention_mil.py     # Gated attention MIL classifier (+ MDSN head)
├── uq_eval/
│   ├── core/
│   │   └── uncertainty.py   # Entropy, mutual information
│   ├── data/
│   │   ├── hf_data.py       # HF TextDataset + collate
│   │   ├── mil_data.py      # MIL dataset + collator
│   │   └── text.py          # Text preprocessing
│   ├── methods/
│   │   ├── cv_ensemble.py   # CV ensemble inference + OOF collection
│   │   ├── inference.py     # MC-Dropout inference (MIL + HF)
│   │   ├── mcp.py           # Mondrian Conformal Prediction
│   │   └── temperature.py   # Temperature scaling
│   ├── reporting/
│   │   ├── metrics.py       # AURC, E-AURC, ECE, risk@coverage, etc.
│   │   └── plots.py         # Calibration, rejection, diagnostics plots
│   └── utils/
│       ├── repro.py         # Seed + Accelerate env sanitization
│       └── serialize.py     # JSON serialization helpers
└── scripts/
    ├── train.sh             # Example training command
    └── evaluate.sh          # Example evaluation command
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a model

```bash
bash scripts/train.sh
```

Or directly:

```bash
python train.py \
  --model_checkpoint bert-base-uncased \
  --train_csv dummy_data/train.csv \
  --val_csv dummy_data/val.csv \
  --output_dir output/my_model \
  --epochs 5 \
  --arch mil \
  --loss ce
```

### 3. Evaluate with uncertainty quantification

```bash
bash scripts/evaluate.sh
```

Or directly:

```bash
python evaluate.py \
  --model_path output/my_model \
  --val_csv dummy_data/val.csv \
  --test_csv dummy_data/test.csv \
  --output_dir output/eval_mc \
  --arch mil \
  --uq_method mc \
  --mc_iters 20
```

## Data Format

CSV files with two required columns:

| Column | Type | Description |
|---|---|---|
| `text` | str | Input text (single document or JSON list for MIL) |
| `label` | int | Binary label (0 or 1) |

For MIL models, `text` can be a JSON-encoded list of strings (one per chunk/instance).

Spanish synthetic examples are available under `dummy_data_es/` with the same split sizes and labels as `dummy_data/`.

## UQ Methods

### MC-Dropout (`--uq_method mc`)
Runs multiple forward passes with dropout enabled and aggregates predictions.
Uncertainty is measured via mutual information, predictive entropy, or std of P(positive).

### MD-SN (`--uq_method mdsn`)
Uses a spectrally-normalized encoder with Mahalanobis distance-based uncertainty.
Requires a model trained with `--use_mdsn`.

### CV Ensemble (`--uq_method cv_ensemble`)
Combines predictions from K cross-validation fold models.
Supports two calibration modes:
- `fold_val`: calibrates on the current fold's validation set
- `oof`: calibrates on out-of-fold predictions (non-leaky)

### Distance Veto (`--dist_model`)
Post-hoc epistemic rejection using embedding-space Mahalanobis distance:
- `mahalanobis`: single centroid per class with OAS shrinkage
- `mixture`: k-means mixture of centroids with shared OAS precision

## Key Arguments

### Training
- `--arch {mil,standard}` — Model architecture
- `--loss {ce,balanced_softmax,logit_adjustment,ldam}` — Loss function
- `--use_mdsn` — Enable MD-SN head (MIL only)
- `--rdrop_alpha` — R-Drop regularization weight

### Evaluation
- `--uq_method {none,mc,mdsn,cv_ensemble}` — UQ method
- `--alpha` — Mondrian CP miscoverage level (default: 0.01)
- `--dist_model {mahalanobis,mixture}` — Distance veto model
- `--dist_quantile` — Distance threshold quantile (default: 0.99)
- `--objective {none,max_cov_at_recall,max_rec_at_cov}` — Threshold optimization
- `--test_bootstrap_n` — Bootstrap resamples for CI estimation

## Output Files

| File | Description |
|---|---|
| `metrics.json` | All evaluation metrics |
| `predictions.csv` | Per-sample predictions + triage categories |
| `review_complex_cases.csv` | Epistemic-vetoed cases for human review |
| `calibration_bundle.npz` | Fitted thresholds, centroids, precision matrices |
| `calibration_curve.pdf` | Reliability diagram |
| `rejection_curves.pdf` | Risk-coverage curves |
| `diagnostics.pdf` | Uncertainty distribution analysis |

## HTML Explanations

The repository also includes `explain.py`, a MIL-only explainer that turns
saved evaluation artifacts into interactive HTML reports.

Recommended workflow:

```bash
python evaluate.py \
  --model_path output/my_model \
  --val_csv dummy_data/val.csv \
  --test_csv dummy_data/test.csv \
  --output_dir output/eval_mc \
  --arch mil \
  --uq_method mc \
  --mc_iters 20

python explain.py \
  --model_path output/my_model \
  --test_csv dummy_data/test.csv \
  --eval_output_dir output/eval_mc \
  --output_dir output/eval_mc/html \
  --seed 42
```

Notes:

- `explain_html.py` expects the same `test_csv` and `seed` used in `evaluate.py`
  so that row ordering matches `predictions.csv` exactly.
- The triage summary shown in the HTML comes from `predictions.csv` and
  `metrics.json`, so it stays aligned with the saved evaluation outputs.
- The public explainer intentionally excludes LLM-specific logic, UMLS concept
  mapping, EuroTEST normalization, and private patient-level dataset helpers.

## License

MIT
