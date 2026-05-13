from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

def compute_rcc_auc(y_true, y_prob_pos, uncertainty) -> float:
    """
    RCC-AUC: area under the risk-coverage curve (risk integrated over coverage),
    where coverage decreases by rejecting high-uncertainty samples.

    NOTE: This matches the common AURC definition (your compute_aurc()).
    Kept as a separate name for paper-compatible reporting.
    """
    return float(compute_aurc(y_true, y_prob_pos, uncertainty))


def compute_rpp_error(y_true, y_prob_pos, uncertainty) -> float:
    """
    RPP (Reversed Pair Proportion) for misclassification detection.

    Define y_err = 1 if prediction is wrong else 0.
    Consider all pairs (i,j) where y_err differs. A "reversed pair" is when an error
    has *lower or equal* uncertainty than a correct prediction (ties count as 0.5).

    Efficient computation:
      AUROC(err) = P(u_err > u_corr) + 0.5*P(u_err == u_corr)
      RPP = 1 - AUROC(err)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    uncertainty = np.asarray(uncertainty).astype(float)

    m = np.isfinite(y_true) & np.isfinite(y_prob_pos) & np.isfinite(uncertainty)
    if m.sum() < 2:
        return 0.0

    y_true = y_true[m]
    y_prob_pos = y_prob_pos[m]
    uncertainty = uncertainty[m]

    y_pred = (y_prob_pos >= 0.5).astype(int)
    y_err = (y_pred != y_true).astype(int)

    if len(np.unique(y_err)) < 2:
        return 0.0

    auc = float(roc_auc_score(y_err, uncertainty))
    return float(1.0 - auc)


def compute_calibration_metrics_robust(y_true, y_prob, n_bins: int = 10):
    """ECE/MCE/OE compatible with the original eval script."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bin_edges, right=True) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    oe = 0.0

    total = len(y_true)

    for i in range(n_bins):
        mask = binids == i
        if not np.any(mask):
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        w = float(mask.sum()) / float(max(1, total))

        gap = abs(acc - conf)
        ece += w * gap
        mce = max(mce, gap)
        if conf > acc:
            oe += w * (conf - acc)

    return float(ece), float(mce), float(oe)


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = binids == i
        if not np.any(mask):
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / max(1, n)) * abs(acc - conf)

    return float(ece)


def compute_custom_kappa(y_true, y_pred_trinary):
    """Custom weighted kappa, same mapping/costs as eval_GPT.py."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_trinary = np.asarray(y_pred_trinary).astype(int)

    y_true_mapped = np.where(y_true == 1, 2, 0)

    W = np.zeros((3, 3))
    W[2, 0] = 1.0
    W[0, 2] = 0.5
    W[0, 1] = 0.25
    W[2, 1] = 0.5

    cm = confusion_matrix(y_true_mapped, y_pred_trinary, labels=[0, 1, 2])
    n = cm.sum()
    observed_loss = (cm * W).sum() / max(1, n)

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected_cm = np.outer(row_sums, col_sums) / max(1, n)
    expected_loss = (expected_cm * W).sum() / max(1, n)

    kappa = 1 - (observed_loss / expected_loss) if expected_loss > 0 else 0.0
    return float(kappa), cm


def compute_aurc(y_true, y_prob_pos, uncertainty) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    uncertainty = np.asarray(uncertainty).astype(float)

    y_pred = (y_prob_pos >= 0.5).astype(int)
    errors = (y_pred != y_true).astype(float)

    order = np.argsort(uncertainty)  # low-unc first
    errors_sorted = errors[order]

    cum_errors = np.cumsum(errors_sorted)
    ks = np.arange(1, len(errors_sorted) + 1)
    risks = cum_errors / ks
    coverages = ks / len(errors_sorted)

    return float(np.trapezoid(risks, coverages))


def compute_eaurc(y_true, y_prob_pos, uncertainty) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    base_err = float((((y_prob_pos >= 0.5).astype(int)) != y_true).mean())
    aurc = compute_aurc(y_true, y_prob_pos, uncertainty)
    return float(aurc - base_err)


def compute_unc_auroc_error(y_true, y_prob_pos, uncertainty) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    uncertainty = np.asarray(uncertainty).astype(float)

    m = np.isfinite(y_true) & np.isfinite(y_prob_pos) & np.isfinite(uncertainty)
    if m.sum() < 2:
        return 0.0

    y_true = y_true[m]
    y_prob_pos = y_prob_pos[m]
    uncertainty = uncertainty[m]

    y_pred = (y_prob_pos >= 0.5).astype(int)
    y_err = (y_pred != y_true).astype(int)

    if len(np.unique(y_err)) < 2:
        return 0.0

    return float(roc_auc_score(y_err, uncertainty))


def compute_risk_at_coverages(y_true, y_prob_pos, uncertainty, coverages=(0.80, 0.90, 0.95)) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    uncertainty = np.asarray(uncertainty).astype(float)

    y_pred = (y_prob_pos >= 0.5).astype(int)
    errors = (y_pred != y_true).astype(float)

    order = np.argsort(uncertainty)
    errors_sorted = errors[order]
    n = len(errors_sorted)

    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        out[f"risk@{int(round(c*100))}"] = float(errors_sorted[:k].mean())
    return out


def compute_spearman_uncertainty_margin(y_prob_pos, uncertainty) -> float:
    y_prob_pos = np.asarray(y_prob_pos).astype(float)
    uncertainty = np.asarray(uncertainty).astype(float)
    margin = np.abs(y_prob_pos - 0.5)

    if np.allclose(uncertainty, uncertainty[0]) or np.allclose(margin, margin[0]):
        return 0.0

    rho, _ = spearmanr(uncertainty, margin)
    return float(0.0 if np.isnan(rho) else rho)


def compute_comprehensive_metrics(y_true, y_pred, y_prob, uncertainties=None, entropy=None, piw=None, prefix: str = ""):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    y_prob = np.asarray(y_prob)
    y_prob_pos = y_prob[:, 1].astype(float) if y_prob.ndim == 2 else y_prob.astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        f"{prefix}tp": int(tp),
        f"{prefix}fp": int(fp),
        f"{prefix}tn": int(tn),
        f"{prefix}fn": int(fn),
    }

    p = tp + fn
    n = tn + fp

    metrics[f"{prefix}accuracy"] = float(accuracy_score(y_true, y_pred))
    recall = float(tp / p) if p > 0 else 0.0
    metrics[f"{prefix}recall"] = recall
    metrics[f"{prefix}specificity"] = float(tn / n) if n > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    metrics[f"{prefix}precision"] = precision
    metrics[f"{prefix}npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    metrics[f"{prefix}f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics[f"{prefix}f2"] = float((5 * precision * recall) / (4 * precision + recall)) if (precision + recall) > 0 else 0.0

    if len(np.unique(y_true)) > 1:
        metrics[f"{prefix}auroc"] = float(roc_auc_score(y_true, y_prob_pos))
        metrics[f"{prefix}auprc"] = float(average_precision_score(y_true, y_prob_pos))
        metrics[f"{prefix}brier"] = float(brier_score_loss(y_true, y_prob_pos))
    else:
        metrics[f"{prefix}auroc"] = 0.5
        metrics[f"{prefix}auprc"] = 0.0
        metrics[f"{prefix}brier"] = 0.0

    ece, mce, oe = compute_calibration_metrics_robust(y_true, y_prob_pos, n_bins=10)
    metrics[f"{prefix}ece"] = float(ece)
    metrics[f"{prefix}mce"] = float(mce)
    metrics[f"{prefix}oe"] = float(oe)

    # Negative log-likelihood (binary): -log p(y_true)
    # Uses the probabilistic output (after temperature / UQ method).
    p_pos = np.clip(y_prob_pos, 1e-12, 1.0 - 1e-12)
    p_true = np.where(y_true == 1, p_pos, 1.0 - p_pos)
    metrics[f"{prefix}nll"] = float(np.mean(-np.log(p_true)))

    if uncertainties is not None:
        metrics[f"{prefix}unc_auroc_error"] = float(compute_unc_auroc_error(y_true, y_prob_pos, uncertainties))
        metrics[f"{prefix}rcc_auc"] = float(compute_rcc_auc(y_true, y_prob_pos, uncertainties))
        metrics[f"{prefix}rpp"] = float(compute_rpp_error(y_true, y_prob_pos, uncertainties))

    if entropy is not None:
        metrics[f"{prefix}avg_entropy"] = float(np.mean(entropy))
    if piw is not None:
        metrics[f"{prefix}avg_piw"] = float(np.mean(piw))

    return metrics
