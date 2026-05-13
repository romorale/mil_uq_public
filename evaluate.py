#!/usr/bin/env python3
"""Evaluation pipeline for uncertainty quantification with MIL and standard models.

Supports four UQ methods:
  none         Deterministic baseline.
  mc           MC-Dropout.
  mdsn         Mahalanobis Distance with Spectral Normalization.
  cv_ensemble  Cross-validation ensemble.

Pipeline stages:
  1. Temperature scaling for probability calibration.
  2. Mondrian Conformal Prediction (MCP) for set-valued predictions.
  3. Optional Mahalanobis distance veto for OOD detection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from sklearn.cluster import KMeans
from sklearn.covariance import OAS
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from uq_eval.core.uncertainty import entropy_np
from uq_eval.data.hf_data import TextDataset, make_hf_collate_fn, texts_labels_from_df
from uq_eval.data.mil_data import MILCollator, MILDataset
from uq_eval.methods.cv_ensemble import (
    compute_unc_from_probs_np,
    cv_ensemble_mean_logits,
    cv_ensemble_mean_logits_hf,
    cv_ensemble_probs_unc2,
    cv_ensemble_probs_unc2_hf,
    cv_ensemble_probs_unc2_parallel_models,
    cv_oof_collect_logits_embeddings_labels_fold_ids,
    cv_oof_collect_logits_embeddings_labels_hf_fold_ids,
    resolve_cv_ensemble_paths,
)
from uq_eval.methods.inference import run_inference_hf, run_inference_mil
from uq_eval.methods.mcp import MondrianCP
from uq_eval.methods.temperature import (
    fit_temperature_from_hf_model,
    fit_temperature_from_logits,
    fit_temperature_from_model,
)
from uq_eval.reporting.metrics import (
    compute_aurc,
    compute_comprehensive_metrics,
    compute_custom_kappa,
    compute_eaurc,
    compute_ece,
    compute_risk_at_coverages,
    compute_rcc_auc,
    compute_rpp_error,
    compute_spearman_uncertainty_margin,
    compute_unc_auroc_error,
)
from uq_eval.reporting.plots import (
    plot_calibration_curve,
    plot_combined_safety_analysis_test,
    plot_complexity_analysis_mcp,
    plot_rejection_curves,
    plot_uncertainty_diagnostics,
)
from uq_eval.utils.repro import sanitize_accelerate_env, set_seed
from uq_eval.utils.serialize import json_sanitize


# =========================================================================
# Distance-based uncertainty models
# =========================================================================

class DistanceUncertainty:
    """Single-centroid Mahalanobis distance per class with OAS shrinkage covariance."""

    def __init__(self, eps: float = 1e-12):
        self.means: dict[int, np.ndarray] = {}
        self.precs: dict[int, np.ndarray] = {}
        self.eps = float(eps)

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, self.eps)

    def fit(self, embeddings, labels, single=False):
        """Fit mean + (shrinkage) precision per true class.

        Args:
            single: If True, fit one shared covariance on all centered data
                    (helpful when some classes have very few samples).
        """
        print("Fitting Distance-Based Uncertainty (Mahalanobis)...")
        X = self._l2_normalize(np.asarray(embeddings))
        y = np.asarray(labels, dtype=np.int64)
        X_centered = np.empty_like(X)

        for c in np.unique(y):
            c = int(c)
            Xc = X[y == c]
            if Xc.shape[0] < 2:
                raise ValueError(f"Not enough samples to fit class {c}: N={Xc.shape[0]}")
            self.means[c] = Xc.mean(axis=0)

            if single:
                mask = (y == c)
                X_centered[mask] = X[mask] - self.means[c]
            else:
                cov = OAS(assume_centered=False).fit(Xc)
                self.precs[c] = cov.precision_.astype(np.float64)
                print(f"   Class {c} Centroid fitted. (N={Xc.shape[0]})")

        if single:
            cov = OAS(assume_centered=True).fit(X_centered)
            shared_prec = cov.precision_.astype(np.float64)
            self.precs = {}
            for c in np.unique(y):
                self.precs[int(c)] = shared_prec
            print(f"   Fitted Shared Covariance using N={len(X)} samples.")

    def predict_distance(self, embeddings, predictions):
        """Mahalanobis distance to centroid of the predicted class."""
        X = self._l2_normalize(np.asarray(embeddings))
        preds = np.asarray(predictions, dtype=np.int64)
        dists = np.empty((X.shape[0],), dtype=np.float64)
        for i in range(X.shape[0]):
            c = int(preds[i])
            diff = X[i] - self.means[c]
            q = float(diff @ self.precs[c] @ diff.T)
            dists[i] = np.sqrt(max(q, 0.0))
        return dists.astype(np.float32)


class MixtureDistanceUncertainty:
    """Mixture-of-centroids Mahalanobis distance for OOD/epistemic veto.

    Addresses the single-Gaussian-per-class limitation by using multiple
    centroids (via k-means) per class with a shared precision matrix (OAS).

    Predict returns the *minimum* Mahalanobis distance to any centroid
    of the *predicted* class.
    """

    def __init__(
        self,
        *,
        k_by_class: dict[int, int] | None = None,
        k_max_by_class: dict[int, int] | None = None,
        min_cluster_size: int = 50,
        kmeans_n_init: int = 10,
        kmeans_max_iter: int = 300,
        random_state: int = 0,
        auto_k_elbow_gain_th: float = 0.05,
        eps: float = 1e-12,
    ):
        self.k_by_class = {int(k): int(v) for k, v in (k_by_class or {}).items()}
        self.k_max_by_class = {int(k): int(v) for k, v in (k_max_by_class or {}).items()}
        self.min_cluster_size = int(min_cluster_size)
        self.kmeans_n_init = int(kmeans_n_init)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.random_state = int(random_state)
        self.auto_k_elbow_gain_th = float(auto_k_elbow_gain_th)
        self.eps = float(eps)
        self.means: dict[int, np.ndarray] = {}
        self.precs: dict[int, np.ndarray] = {}
        self.effective_k_by_class: dict[int, int] = {}

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, self.eps)

    def _k_cap(self, n: int) -> int:
        if int(n) < 2:
            return 1
        return max(1, int(n) // max(self.min_cluster_size, 1))

    def _effective_k(self, n: int, requested_k: int, k_max: int | None = None) -> int:
        n, requested_k = int(n), int(requested_k)
        if n < 2 or requested_k <= 1:
            return 1
        k_cap = self._k_cap(n)
        if k_max is None:
            return max(1, min(requested_k, k_cap, n))
        return max(1, min(requested_k, int(k_max), k_cap, n))

    def _select_k_auto_elbow(self, Xc: np.ndarray, *, k_max: int) -> tuple[int, list[float]]:
        """Pick K using inertia elbow: smallest K where relative gain < threshold."""
        Xc = np.asarray(Xc, dtype=np.float64)
        n_c = int(Xc.shape[0])
        k_max = int(min(max(1, k_max), n_c))
        if k_max <= 1 or n_c < 2:
            return 1, []

        inertias: list[float] = []
        mu1 = Xc.mean(axis=0, keepdims=True)
        inertias.append(float(np.square(Xc - mu1).sum()))

        for k in range(2, k_max + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state,
                        n_init=self.kmeans_n_init, max_iter=self.kmeans_max_iter)
            km.fit(Xc)
            inertias.append(float(km.inertia_))

        chosen = k_max
        for k in range(2, k_max + 1):
            prev, cur = inertias[k - 2], inertias[k - 1]
            gain = (prev - cur) / max(prev, self.eps)
            if gain < self.auto_k_elbow_gain_th:
                chosen = k - 1
                break
        return int(max(1, chosen)), inertias

    def fit(self, embeddings, labels, single: bool = False):
        print("Fitting Mixture Distance-Based Uncertainty (Mixture Mahalanobis)...")
        X = self._l2_normalize(np.asarray(embeddings))
        y = np.asarray(labels, dtype=np.int64)
        classes = np.unique(y)
        if classes.size == 0:
            raise ValueError("MixtureDistanceUncertainty.fit: empty labels")

        residuals = []
        for c in classes:
            c = int(c)
            Xc = X[y == c]
            n_c = int(Xc.shape[0])
            if n_c < 2:
                raise ValueError(f"Not enough samples to fit class {c}: N={n_c}")

            req_k = int(self.k_by_class.get(c, 1))
            k_cap = self._k_cap(n_c)
            k_max_user = int(self.k_max_by_class.get(c, max(1, req_k)))
            k_max_eff = int(max(1, min(k_max_user, k_cap, n_c)))

            if req_k <= 0:
                k_eff, inertias = self._select_k_auto_elbow(Xc, k_max=k_max_eff)
                k_eff = self._effective_k(n_c, k_eff, k_max=k_max_eff)
                print(f"   Class {c}: K=auto -> {k_eff} (k_max={k_max_eff}, N={n_c})")
            else:
                k_eff = self._effective_k(n_c, req_k, k_max=k_max_eff)
                print(f"   Class {c}: K={k_eff} (requested={req_k}, k_max={k_max_eff}, N={n_c})")

            self.effective_k_by_class[c] = int(k_eff)

            if k_eff == 1:
                mu = Xc.mean(axis=0, keepdims=True)
                self.means[c] = mu
                residuals.append(Xc - mu)
                continue

            km = KMeans(n_clusters=k_eff, random_state=self.random_state,
                        n_init=self.kmeans_n_init, max_iter=self.kmeans_max_iter)
            assign = km.fit_predict(Xc)
            mu = np.asarray(km.cluster_centers_, dtype=np.float64)
            self.means[c] = mu
            residuals.append(Xc - mu[assign])

        R = np.concatenate(residuals, axis=0)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        cov = OAS(assume_centered=True).fit(R)
        shared_prec = cov.precision_.astype(np.float64)
        if not np.isfinite(shared_prec).all():
            raise ValueError("OAS produced non-finite precision")
        self.precs = {int(c): shared_prec for c in classes.tolist()}
        print(f"   Fitted shared precision on residuals N={int(R.shape[0])}")

    def predict_distance(self, embeddings, predictions):
        X = self._l2_normalize(np.asarray(embeddings))
        preds = np.asarray(predictions, dtype=np.int64)
        dists = np.full((X.shape[0],), 1e12, dtype=np.float64)
        for c in np.unique(preds):
            c = int(c)
            mask = preds == c
            if not np.any(mask) or c not in self.means or c not in self.precs:
                continue
            mu = np.asarray(self.means[c], dtype=np.float64)
            P = np.asarray(self.precs[c], dtype=np.float64)
            Xc = X[mask]
            delta = Xc[:, None, :] - mu[None, :, :]
            q = np.einsum("nkd,dd,nkd->nk", delta, P, delta, optimize=True)
            q = np.maximum(q, 0.0)
            dists[mask] = np.sqrt(np.min(q, axis=1))
        return np.nan_to_num(dists, nan=1e12, posinf=1e12, neginf=1e12).astype(np.float32)


# =========================================================================
# Embedding extraction
# =========================================================================

def get_embeddings(model, loader: DataLoader, *, collate_fn=None) -> np.ndarray:
    """Extract doc-level embeddings (N, D) for MIL or HF models.

    For MIL models, uses return_features=True to get doc_embeddings.
    For HF models, extracts the CLS token from the last hidden state.
    """
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise ValueError("get_embeddings: loader has no .dataset")
    if collate_fn is None:
        collate_fn = getattr(loader, "collate_fn", None)
    if collate_fn is None:
        raise ValueError("get_embeddings: collate_fn is None. Pass it explicitly.")

    batch_size = int(getattr(loader, "batch_size", 8) or 8)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model.eval()
    device = next(model.parameters()).device
    embs = []
    with torch.no_grad():
        for batch in full_loader:
            if not isinstance(batch, Mapping):
                raise ValueError(f"Expected batch to be Mapping, got {type(batch)}")
            if "num_chunks_per_doc" in batch:
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            batch["num_chunks_per_doc"], return_features=True)
                if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                    raise ValueError("MIL model(..., return_features=True) must return (logits, attn, doc_emb)")
                embs.append(out[2].detach().cpu())
            else:
                hf_inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask") if k in batch}
                out = model(**hf_inputs, output_hidden_states=True, return_dict=True)
                hs = getattr(out, "hidden_states", None)
                if hs is not None and len(hs) > 0 and torch.is_tensor(hs[-1]):
                    last = hs[-1]
                else:
                    last = getattr(out, "last_hidden_state", None)
                    if last is None:
                        raise ValueError("HF output missing hidden_states")
                embs.append(last[:, 0, :].detach().cpu())
    return torch.cat(embs, dim=0).numpy().astype(np.float32)


# =========================================================================
# MCP threshold calibration
# =========================================================================

def analyze_threshold_stability(probs, labels, target_class, alpha_start=0.001,
                                alpha_end=0.10, step=0.001, plot=False,
                                output_dir=None, filename="stability_analysis.pdf"):
    """Scan alphas to find a stable (plateau) conformal score threshold."""
    mask = (labels == target_class)
    if np.sum(mask) == 0:
        return 0.5
    scores = 1.0 - probs[mask, target_class]
    alphas = np.arange(alpha_start, alpha_end, step)
    thresholds = []
    for a in alphas:
        thresholds.append(np.quantile(scores, 1.0 - a, method="higher"))
    thresholds = np.asarray(thresholds, dtype=np.float64)

    best_len, best_th = -1, None
    run_start = 0
    for i in range(1, len(thresholds) + 1):
        if i == len(thresholds) or thresholds[i] != thresholds[i - 1]:
            run_len = i - run_start
            th_val = float(thresholds[i - 1])
            if (run_len > best_len) or (run_len == best_len and (best_th is None or th_val > best_th)):
                best_len, best_th = run_len, th_val
            run_start = i
    print(f"   [Stability] Class {target_class}: plateau threshold={best_th:.4f} (run_len={best_len})")

    if plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(alphas, thresholds, drawstyle="steps-post")
        plt.axhline(best_th, color="r", linestyle="--", label=f"Stable: {best_th:.4f}")
        plt.title(f"Stability Analysis (Class {target_class})")
        plt.xlabel("Alpha"); plt.ylabel("Score Threshold")
        plt.legend()
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close()
    return float(best_th)


def _null_rate_from_engine(mcp_engine: MondrianCP, probs: np.ndarray) -> float:
    sets, _ = mcp_engine.predict(probs)
    if sets is None:
        return 0.0
    return float(np.mean([len(s) == 0 for s in sets]))


def _fit_mcp_pattern_a_thresholds(*, probs, labels, unc, unc_percentile, output_dir=".", alpha=0.01):
    """Calibrate Mondrian CP and fit uncertainty threshold.

    Returns (p_low, p_high, unc_th, mcp_engine).
    """
    print("\nConfiguring Mondrian Conformal Prediction...")
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    alpha = float(alpha)

    def _rates(engine, p):
        sets, _ = engine.predict(p)
        if sets is None: return 0.0, 0.0
        return float(np.mean([len(s) == 0 for s in sets])), float(np.mean([len(s) > 1 for s in sets]))

    def _band(th0, th1):
        p_end_neg, p_start_pos = float(th0), float(1.0 - th1)
        return p_start_pos, p_end_neg, max(0.0, p_start_pos - p_end_neg), max(0.0, p_end_neg - p_start_pos)

    # Standard calibration
    mcp_std = MondrianCP(alpha=alpha)
    mcp_std.calibrate(probs, labels)
    th0_std = float(mcp_std.thresholds.get(0, 1.0))
    th1_std = float(mcp_std.thresholds.get(1, 1.0))

    # Stability thresholds
    a0, a1 = max(1e-4, alpha * 0.5), min(0.20, alpha * 5.0)
    th0_stable = analyze_threshold_stability(probs, labels, 0, a0, a1, max(1e-4, alpha/10), False)
    th1_stable = analyze_threshold_stability(probs, labels, 1, a0, a1, max(1e-4, alpha/10), True, output_dir)
    print(f"   Stable (score-space): th0={th0_stable:.4f}, th1={th1_stable:.4f}")

    mcp_stable = MondrianCP(alpha=alpha)
    mcp_stable.thresholds[0] = float(th0_stable)
    mcp_stable.thresholds[1] = float(th1_stable)

    null_std, amb_std = _rates(mcp_std, probs)
    null_st, amb_st = _rates(mcp_stable, probs)
    use_stable = (null_st <= null_std + 1e-6) and (amb_st <= amb_std + 0.02)

    th0 = th0_stable if use_stable else th0_std
    th1 = th1_stable if use_stable else th1_std

    # No-null clamp
    p_start_pos, p_end_neg, gap, overlap = _band(th0, th1)
    if gap > 0:
        delta_th0, delta_th1 = p_start_pos - p_end_neg, (1.0 - th0) - th1
        if delta_th0 <= delta_th1:
            th0 = float(p_start_pos)
        else:
            th1 = float(1.0 - p_end_neg)

    mcp_engine = MondrianCP(alpha=alpha)
    mcp_engine.thresholds[0] = float(th0)
    mcp_engine.thresholds[1] = float(th1)

    p_start_pos, p_end_neg, gap, overlap = _band(th0, th1)
    null_fin, amb_fin = _rates(mcp_engine, probs)
    print(f"   MCP alpha={alpha:.4f} mode={'stable' if use_stable else 'std'} | "
          f"th0={th0:.4f} th1={th1:.4f} | gap={gap:.4f} overlap={overlap:.4f} | "
          f"val_null={null_fin:.3f} val_amb={amb_fin:.3f}")

    # Uncertainty threshold
    unc = np.asarray(unc, dtype=np.float64)
    unc = np.nan_to_num(unc, nan=np.nanmedian(unc) if np.isfinite(unc).any() else 0.0, posinf=1e12, neginf=0.0)
    unc_th = float(np.percentile(unc, float(unc_percentile)))

    band_a, band_b = float(1.0 - th1), float(th0)
    p_low, p_high = float(min(band_a, band_b)), float(max(band_a, band_b))
    return p_low, p_high, unc_th, mcp_engine


# =========================================================================
# Robust-Z distance calibration
# =========================================================================

def _robust_z_calibrate(val, test, eps=1e-12):
    """Global robust z-calibration: fit median/MAD on val, apply to both."""
    val, test = np.asarray(val, dtype=np.float64), np.asarray(test, dtype=np.float64)
    v = val[np.isfinite(val)]
    if v.size == 0:
        return val, test
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = max(mad * 1.4826, eps)
    return (val - med) / scale, (test - med) / scale


def _robust_z_calibrate_by_pred_class(*, val_dists, val_preds, test_dists, test_preds,
                                       cal_mask=None, eps=1e-12):
    """Per-predicted-class robust z-calibration."""
    val_dists = np.asarray(val_dists, dtype=np.float64)
    test_dists = np.asarray(test_dists, dtype=np.float64)
    val_preds = np.asarray(val_preds, dtype=np.int64)
    test_preds = np.asarray(test_preds, dtype=np.int64)
    if cal_mask is None:
        cal_mask = np.ones((len(val_preds),), dtype=bool)
    cal_mask = np.asarray(cal_mask, dtype=bool)
    out_val, out_test = val_dists.copy(), test_dists.copy()
    for c in np.unique(val_preds):
        c = int(c)
        fit_mask = (val_preds == c) & cal_mask
        v = val_dists[fit_mask]
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        scale = max(mad * 1.4826, eps)
        m_val, m_test = (val_preds == c), (test_preds == c)
        out_val[m_val] = (val_dists[m_val] - med) / scale
        out_test[m_test] = (test_dists[m_test] - med) / scale
    return out_val, out_test


def _fit_robust_z_by_pred_stats(*, val_dists, val_preds, cal_mask=None, eps=1e-12):
    """Fit per-predicted-class robust z parameters on validation distances."""
    val_dists = np.asarray(val_dists, dtype=np.float64)
    val_preds = np.asarray(val_preds, dtype=np.int64)
    if cal_mask is None:
        cal_mask = np.ones((len(val_preds),), dtype=bool)
    cal_mask = np.asarray(cal_mask, dtype=bool)
    stats: dict[str, dict[str, float]] = {}
    for c in np.unique(val_preds):
        c = int(c)
        fit_mask = (val_preds == c) & cal_mask
        v = val_dists[fit_mask]
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        scale = float(max(mad * 1.4826, eps))
        stats[str(c)] = {"median": med, "mad": mad, "scale": scale, "n_fit": float(v.size)}
    return stats


def _apply_robust_z_by_pred_stats(*, dists, preds, stats):
    """Apply pre-fitted per-predicted-class robust z parameters."""
    dists = np.asarray(dists, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.int64)
    out = dists.copy()
    for c_str, st in (stats or {}).items():
        c = int(c_str)
        med = float(st.get("median", 0.0))
        scale = float(max(st.get("scale", 1.0), 1e-12))
        mask = (preds == c)
        if np.any(mask):
            out[mask] = (dists[mask] - med) / scale
    return out


# =========================================================================
# Combined decision functions (MCP + distance veto)
# =========================================================================

def predict_combined(mcp_model, dist_model, test_probs, test_unc, test_embeddings,
                     dist_threshold_by_pred):
    """MCP + distance veto using a fitted distance model."""
    sets, _ = mcp_model.predict(test_probs)
    test_preds = np.argmax(test_probs, axis=1)
    dists = dist_model.predict_distance(test_embeddings, test_preds)
    preds_trinary = np.ones(len(sets), dtype=int)
    categories = []
    for i in range(len(sets)):
        s = sets[i]
        pred_c = int(test_preds[i])
        d = dists[i]
        th = float(dist_threshold_by_pred.get(pred_c, np.inf))
        if d > th:
            categories.append("Complex: Epistemic (Geometric Veto)")
        elif len(s) > 1:
            categories.append("Defer: Ambiguous (Probabilistic)")
        elif len(s) == 0:
            categories.append("Defer: Null (Conflict)")
        else:
            preds_trinary[i] = 0 if s[0] == 0 else 2
            categories.append("Clear")
    return preds_trinary, np.asarray(categories), sets, dists


def predict_combined_with_dists(mcp_model, test_probs, dists, dist_threshold_by_pred,
                                *, unc=None, unc_th=None):
    """Same as predict_combined but with precomputed distances and optional unc veto."""
    sets, _ = mcp_model.predict(test_probs)
    test_preds = np.argmax(test_probs, axis=1)
    dists = np.asarray(dists, dtype=np.float64)
    if unc is not None:
        unc = np.nan_to_num(np.asarray(unc, dtype=np.float64), nan=1e12, posinf=1e12, neginf=0.0)
    preds_trinary = np.ones(len(sets), dtype=int)
    categories = []
    for i in range(len(sets)):
        s = sets[i]
        pred_c = int(test_preds[i])
        d = float(dists[i])
        th = float(dist_threshold_by_pred.get(pred_c, np.inf))
        if d > th:
            preds_trinary[i] = 1; categories.append("Complex: Epistemic (Geometric Veto)"); continue
        if unc is not None and unc_th is not None and np.isfinite(unc_th):
            if len(s) == 1 and float(unc[i]) > float(unc_th):
                preds_trinary[i] = 1; categories.append("Defer: High Uncertainty (UQ Threshold)"); continue
        if len(s) > 1:
            categories.append("Defer: Ambiguous (Probabilistic)")
        elif len(s) == 0:
            categories.append("Defer: Null (Conflict)")
        else:
            preds_trinary[i] = 0 if s[0] == 0 else 2; categories.append("Clear")
    return preds_trinary, np.asarray(categories), sets, dists.astype(np.float32)


def predict_combined_with_dists_test(mcp_model, test_probs, dists, dist_threshold_by_pred,
                                     *, unc=None, unc_th=None, prob_border_eps=0.0):
    """Full decision function for TEST: MCP priority, then distance veto, then unc veto."""
    sets, _ = mcp_model.predict(test_probs)
    test_preds = np.argmax(test_probs, axis=1)
    dists = np.asarray(dists, dtype=np.float64)
    if unc is not None:
        unc = np.nan_to_num(np.asarray(unc, dtype=np.float64), nan=1e12, posinf=1e12, neginf=0.0)

    th0 = float(getattr(mcp_model, "thresholds", {}).get(0, 0.5))
    th1 = float(getattr(mcp_model, "thresholds", {}).get(1, 0.5))
    p_end_neg, p_start_pos = th0, 1.0 - th1

    preds_trinary = np.ones(len(sets), dtype=int)
    categories = []
    for i in range(len(sets)):
        s = sets[i]
        pred_c = int(test_preds[i])
        d = float(dists[i])

        if len(s) > 1:
            categories.append("Defer: Ambiguous (Probabilistic)"); continue
        if len(s) == 0:
            categories.append("Defer: Null (Conflict)"); continue

        if float(prob_border_eps) > 0:
            p_pos = float(test_probs[i, 1])
            if abs(p_pos - p_end_neg) <= float(prob_border_eps) or abs(p_pos - p_start_pos) <= float(prob_border_eps):
                categories.append("Defer: Ambiguous (Probabilistic)"); continue

        th = float(dist_threshold_by_pred.get(pred_c, np.inf))
        if d > th:
            categories.append("Complex: Epistemic (Geometric Veto)"); continue

        if unc is not None and unc_th is not None and np.isfinite(unc_th):
            if float(unc[i]) > float(unc_th):
                categories.append("Defer: High Uncertainty (UQ Threshold)"); continue

        preds_trinary[i] = 0 if s[0] == 0 else 2
        categories.append("Clear")
    return preds_trinary, np.asarray(categories), sets, dists.astype(np.float32)


def _triage_objective_metrics(y_true, preds_trinary):
    """Return (coverage, recall_pos_clear) for objective search."""
    y_true = np.asarray(y_true, dtype=np.int64)
    preds_trinary = np.asarray(preds_trinary, dtype=np.int64)
    coverage = float((preds_trinary != 1).mean())
    pos = (y_true == 1)
    recall_pos_clear = float(((preds_trinary == 2) & pos).sum() / max(1, int(pos.sum())))
    return coverage, recall_pos_clear


# =========================================================================
# Helpers
# =========================================================================

def _load_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return AutoTokenizer.from_pretrained(cfg["model_checkpoint"])


def _maybe_flash_attn(accelerator) -> str:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            if accelerator.is_local_main_process:
                print("Flash Attention 2 Enabled!")
            return "flash_attention_2"
    return "sdpa"


def _to_csv_text(x):
    if isinstance(x, list):
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)


def _preview_text(x, max_chars=200):
    if isinstance(x, list):
        s = " | ".join([t for t in x if isinstance(t, str) and t.strip()][:3])
    else:
        s = "" if x is None else str(x)
    return s.replace("\n", " ").strip()[:max_chars]


def _json_ready_dict(d):
    out = {}
    for k, v in (d or {}).items():
        out[str(k)] = float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v
    return out


def _parse_int_csv(s):
    if s is None: return []
    s = str(s).strip()
    if not s: return []
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _save_calibration_bundle(*, output_path, meta, distance_means=None, distance_precs=None):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np_payload = {"meta_json": np.asarray([json.dumps(json_sanitize(meta), ensure_ascii=False)], dtype=object)}
    if distance_means:
        for c, arr in distance_means.items():
            np_payload[f"distance_mean_{int(c)}"] = np.asarray(arr, dtype=np.float64)
    if distance_precs:
        for c, arr in distance_precs.items():
            np_payload[f"distance_prec_{int(c)}"] = np.asarray(arr, dtype=np.float64)
    np.savez_compressed(output_path, **np_payload)
    return output_path



# =========================================================================
# main()
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UQ methods (none | mc | mdsn | cv_ensemble) on MIL or standard architectures."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cv_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="eval_output")

    parser.add_argument("--arch", type=str, default="mil", choices=["mil", "standard"])
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--uq_method", type=str, default="mc",
                        choices=["none", "mc", "mdsn", "cv_ensemble"])
    parser.add_argument("--mc_iters", type=int, default=20)
    parser.add_argument("--mc_batch_size", type=int, default=1,
                        help="MC-dropout replicas per forward pass.")
    parser.add_argument("--assymmetric", action="store_true", default=False)

    # Data (CSV-based)
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to validation CSV (columns: text, label).")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to test CSV (columns: text, label).")

    # CV ensemble
    parser.add_argument("--cv_ensemble_paths", type=str, default="")
    parser.add_argument("--cv_ensemble_root", type=str, default="")
    parser.add_argument("--cv_ensemble_n_models", type=int, default=10)
    parser.add_argument("--cv_ensemble_subdir", type=str, default="")
    parser.add_argument("--cv_ensemble_calib", type=str, default="fold_val",
                        choices=["fold_val", "oof"])
    parser.add_argument("--cv_ensemble_oof_unc_metric", type=str, default="entropy",
                        choices=["entropy", "std_pos"])
    parser.add_argument("--cv_oof_dir", type=str, default="",
                        help="Directory with per-fold train/val JSON splits for OOF collection.")

    # Temperature
    parser.add_argument("--temp_min", type=float, default=0.02)
    parser.add_argument("--temp_max", type=float, default=10.0)

    # Thresholding
    parser.add_argument("--target_recall", type=float, default=0.99)
    parser.add_argument("--target_specificity", type=float, default=None)
    parser.add_argument("--unc_percentile", type=float, default=90.0)

    # Distance veto
    parser.add_argument("--dist_model", type=str, default="mahalanobis",
                        choices=["mahalanobis", "mixture"])
    parser.add_argument("--dist_fit_source", type=str, default="val",
                        choices=["val", "train", "oof"])
    parser.add_argument("--dist_k0", type=int, default=1)
    parser.add_argument("--dist_k1", type=int, default=4)
    parser.add_argument("--dist_k_search", type=str, default="none",
                        choices=["none", "grid"])
    parser.add_argument("--dist_k0_grid", type=str, default="")
    parser.add_argument("--dist_k1_grid", type=str, default="")
    parser.add_argument("--dist_k_search_max_combos", type=int, default=25)
    parser.add_argument("--dist_k0_max", type=int, default=8)
    parser.add_argument("--dist_k1_max", type=int, default=8)
    parser.add_argument("--dist_min_cluster_size", type=int, default=50)
    parser.add_argument("--dist_kmeans_n_init", type=int, default=10)
    parser.add_argument("--dist_k_auto_gain_th", type=float, default=0.05)
    parser.add_argument("--dist_quantile", type=float, default=0.99)
    parser.add_argument("--objective", type=str, default="none",
                        choices=["none", "max_cov_at_recall", "max_rec_at_cov"])
    parser.add_argument("--target_coverage", type=float, default=0.80)
    parser.add_argument("--dist_q_min", type=float, default=0.95)
    parser.add_argument("--dist_q_max", type=float, default=0.999)
    parser.add_argument("--dist_q_steps", type=int, default=50)
    parser.add_argument("--dist_transform", type=str, default="raw",
                        choices=["raw", "robust_z_global", "robust_z_by_pred"])

    parser.add_argument("--unc_metric", type=str, default="mi",
                        choices=["std_pos", "entropy", "mi"])

    parser.add_argument("--prob_border_eps", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_calibration_file", type=str, default="")
    parser.add_argument("--skip_save_calibration", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Miscoverage level for Mondrian CP.")

    # Reporting
    parser.add_argument("--report_oof_fold_metrics", action="store_true", default=False)
    parser.add_argument("--oof_fold_metrics_out", type=str, default="")
    parser.add_argument("--test_bootstrap_n", type=int, default=0)
    parser.add_argument("--test_bootstrap_seed", type=int, default=0)
    parser.add_argument("--test_bootstrap_out", type=str, default="")

    args = parser.parse_args()

    if args.cv_ensemble_paths == "":
        args.cv_ensemble_paths = None
    if args.cv_ensemble_root == "":
        args.cv_ensemble_root = None
    if args.save_calibration_file == "":
        args.save_calibration_file = None

    set_seed(args.seed)
    sanitize_accelerate_env()

    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        print(f"[eval] script={os.path.abspath(__file__)}")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Running on {accelerator.num_processes} devices")

    model_path = os.path.abspath(args.model_path)

    # ---- Import MIL model if needed ----
    AttentionMILClassifier = None
    if args.arch == "mil":
        from models.attention_mil import AttentionMILClassifier  # noqa: F811

    tokenizer = _load_tokenizer(model_path)

    # ---- Load data from CSV ----
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    val_df = val_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    collate = None

    if args.arch == "mil":
        val_ds = MILDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)
        test_ds = MILDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)
        collate = MILCollator(tokenizer)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate, shuffle=False)
    else:
        val_texts, val_labels_list = texts_labels_from_df(val_df)
        test_texts, test_labels_list = texts_labels_from_df(test_df)

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model_config = model.config
        collate_fn = make_hf_collate_fn(tokenizer, model_config, max_length=int(args.max_length))
        collate = collate_fn
        val_loader = DataLoader(TextDataset(val_texts, val_labels_list), batch_size=args.batch_size,
                                collate_fn=collate_fn, shuffle=False)
        test_loader = DataLoader(TextDataset(test_texts, test_labels_list), batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False)

    attn_impl = _maybe_flash_attn(accelerator) if args.arch == "mil" else "sdpa"

    if args.arch == "mil":
        assert AttentionMILClassifier is not None
        model = AttentionMILClassifier.from_pretrained(
            model_path, device=accelerator.device, attn_implementation=attn_impl,
        )
        if accelerator.is_local_main_process and args.uq_method == "mdsn":
            u = accelerator.unwrap_model(model)
            fitted = int(getattr(u, "mdsn_fitted", torch.tensor(0)).item()) if hasattr(u, "mdsn_fitted") else 0
            prec_ok = bool(torch.isfinite(getattr(u, "mdsn_precision", torch.tensor(0.0))).all().item()) if hasattr(u, "mdsn_precision") else False
            cent_ok = bool(torch.isfinite(getattr(u, "mdsn_centroids", torch.tensor(0.0))).all().item()) if hasattr(u, "mdsn_centroids") else False
            print(f"[eval] mdsn_fitted={fitted} prec_finite={prec_ok} centroids_finite={cent_ok}")

    model, val_loader, test_loader = accelerator.prepare(model, val_loader, test_loader)

    def model_loader(path: str):
        if args.arch == "mil":
            return AttentionMILClassifier.from_pretrained(
                os.path.abspath(path), device=accelerator.device, attn_implementation=attn_impl,
            )
        return AutoModelForSequenceClassification.from_pretrained(os.path.abspath(path))

    # =========================================================================
    # 1) Temperature scaling
    # =========================================================================
    optimal_T = 1.0
    oof_logits = None
    oof_labels = None
    oof_probs = None
    oof_embeddings = None
    oof_fold_ids = None
    oof_unc = None

    if args.uq_method == "none":
        optimal_T = 1.0
        if accelerator.is_local_main_process:
            print("Using deterministic baseline (uq_method=none, T=1.0)")

    elif args.uq_method == "cv_ensemble":
        model_paths = resolve_cv_ensemble_paths(args)

        if args.cv_ensemble_calib == "fold_val":
            if accelerator.is_local_main_process:
                print("Fitting Temperature (CV ensemble mean logits, fold-val)...")
            if args.arch == "mil":
                val_mean_logits, val_labels_for_T = cv_ensemble_mean_logits(
                    val_loader, accelerator, model_paths, model_loader)
            else:
                val_mean_logits, val_labels_for_T = cv_ensemble_mean_logits_hf(
                    val_loader, accelerator, model_paths, model_loader)

            if accelerator.is_main_process:
                logits_np = val_mean_logits.astype(np.float32)
                labels_np = val_labels_for_T.astype(np.int64)
            else:
                logits_np = np.zeros((0, 2), dtype=np.float32)
                labels_np = np.zeros((0,), dtype=np.int64)

            optimal_T = fit_temperature_from_logits(
                accelerator, logits_cpu=logits_np, labels_cpu=labels_np,
                temp_min=args.temp_min, temp_max=args.temp_max,
            )

        elif args.cv_ensemble_calib == "oof":
            if accelerator.is_local_main_process:
                print("Fitting Temperature (CV OOF across folds; non-leaky)...")

            if args.arch == "mil":
                assert collate is not None
                oof_logits, oof_embeddings, oof_labels, oof_fold_ids = \
                    cv_oof_collect_logits_embeddings_labels_fold_ids(
                        args=args, accelerator=accelerator, tokenizer=tokenizer,
                        collate=collate, model_paths=model_paths,
                        build_patient_level_from_split_df=None,
                        model_loader=model_loader,
                    )
            else:
                oof_logits, oof_embeddings, oof_labels, oof_fold_ids = \
                    cv_oof_collect_logits_embeddings_labels_hf_fold_ids(
                        args=args, accelerator=accelerator, tokenizer=tokenizer,
                        model_paths=model_paths, batch_size=int(args.batch_size),
                        max_length=int(args.max_length), model_loader=model_loader,
                    )

            if accelerator.is_main_process:
                assert oof_logits is not None and oof_labels is not None
                optimal_T = fit_temperature_from_logits(
                    accelerator, logits_cpu=oof_logits, labels_cpu=oof_labels,
                    temp_min=args.temp_min, temp_max=args.temp_max,
                )
            else:
                optimal_T = fit_temperature_from_logits(
                    accelerator,
                    logits_cpu=np.zeros((0, 2), dtype=np.float32),
                    labels_cpu=np.zeros((0,), dtype=np.int64),
                    temp_min=args.temp_min, temp_max=args.temp_max,
                )
        else:
            raise ValueError(f"Unknown cv_ensemble_calib={args.cv_ensemble_calib}")

        if accelerator.is_local_main_process:
            print(f"Using Temperature T={optimal_T:.4f}")

    else:
        # mc or mdsn: fit temperature from single model
        if args.arch == "mil":
            optimal_T = fit_temperature_from_model(
                accelerator, model, val_loader,
                temp_min=args.temp_min, temp_max=args.temp_max,
            )
        else:
            optimal_T = fit_temperature_from_hf_model(
                accelerator, model, val_loader,
                temp_min=args.temp_min, temp_max=args.temp_max,
            )
        if accelerator.is_local_main_process:
            print(f"Using Temperature T={optimal_T:.4f}")

    # =========================================================================
    # 2) VAL inference + threshold calibration
    # =========================================================================
    optimal_T = float(optimal_T)
    thresholds_tensor = torch.zeros(3, device=accelerator.device, dtype=torch.float32)
    p_low = p_high = unc_th = 0.0
    mcp_engine = None

    val_embeddings = None
    test_embeddings = None
    distance_means_for_export = None
    distance_precs_for_export = None
    robust_z_by_pred_stats_for_export = None

    if args.uq_method == "none":
        if accelerator.is_local_main_process:
            print("Val Inference (deterministic baseline)... skipped")

    elif args.uq_method == "mc":
        if accelerator.is_local_main_process:
            print("Val Inference (MC)...")
        if args.arch == "mil":
            val_probs, val_unc, val_labels, _, _, val_embeddings = run_inference_mil(
                model, val_loader, accelerator, mc_dropout=True, n_iters=args.mc_iters,
                mc_batch_size=args.mc_batch_size, temperature=optimal_T,
                unc_metric=args.unc_metric, return_embeddings=True,
            )
        else:
            val_probs, val_unc, val_labels, _, _, val_embeddings = run_inference_hf(
                model, val_loader, accelerator, mc_dropout=True, n_iters=args.mc_iters,
                mc_batch_size=args.mc_batch_size, temperature=optimal_T,
                unc_metric=args.unc_metric, return_embeddings=True,
            )

        if accelerator.is_main_process:
            p_low, p_high, unc_th, mcp_engine = _fit_mcp_pattern_a_thresholds(
                probs=val_probs, labels=val_labels, unc=val_unc,
                unc_percentile=float(args.unc_percentile),
                output_dir=args.output_dir, alpha=float(args.alpha),
            )
            print(f"Fitted: Band=[{p_low:.3f}, {p_high:.3f}], Unc_Th={unc_th:.3f} (unc={args.unc_metric})")
            thresholds_tensor = torch.tensor([p_low, p_high, unc_th],
                                             device=accelerator.device, dtype=torch.float32)

    elif args.uq_method == "mdsn":
        if args.arch != "mil":
            raise ValueError("uq_method=mdsn is only supported for arch=mil")
        unwrapped = accelerator.unwrap_model(model)
        if not bool(getattr(unwrapped, "use_mdsn", False)):
            raise ValueError("uq_method=mdsn requires model.use_mdsn=True")

        if accelerator.is_local_main_process:
            print("Val Inference (MDSN)...")
        val_probs, val_unc, val_labels, _, _, val_embeddings = run_inference_mil(
            model, val_loader, accelerator, mc_dropout=False, n_iters=1,
            temperature=optimal_T, unc_metric=args.unc_metric,
            return_embeddings=True, return_model_uncertainty=True,
        )
        if accelerator.is_main_process:
            p_low, p_high, unc_th, mcp_engine = _fit_mcp_pattern_a_thresholds(
                probs=val_probs, labels=val_labels, unc=val_unc,
                unc_percentile=float(args.unc_percentile), output_dir=args.output_dir,
            )
            print(f"Fitted: Band=[{p_low:.3f}, {p_high:.3f}], Unc_Th={unc_th:.3f} (uq=mdsn)")
            thresholds_tensor = torch.tensor([p_low, p_high, unc_th],
                                             device=accelerator.device, dtype=torch.float32)

    elif args.uq_method == "cv_ensemble":
        model_paths = resolve_cv_ensemble_paths(args)

        if args.cv_ensemble_calib == "fold_val":
            if accelerator.is_local_main_process:
                print("Val Inference (CV ensemble, fold_val)...")
            if args.arch == "mil":
                fn = cv_ensemble_probs_unc2_parallel_models if accelerator.num_processes > 1 else cv_ensemble_probs_unc2
                val_probs, val_unc, val_labels, _, _, val_embeddings = fn(
                    val_loader, accelerator, model_paths, model_loader,
                    temperature=optimal_T, unc_metric=args.unc_metric,
                    collate_fn=collate, return_embeddings=True,
                )
            else:
                val_probs, val_unc, val_labels, _, _, val_embeddings = cv_ensemble_probs_unc2_hf(
                    val_loader, accelerator, model_paths, model_loader,
                    temperature=optimal_T, unc_metric=args.unc_metric, return_embeddings=True,
                )
            if accelerator.is_main_process:
                p_low, p_high, unc_th, mcp_engine = _fit_mcp_pattern_a_thresholds(
                    probs=val_probs, labels=val_labels, unc=val_unc,
                    unc_percentile=float(args.unc_percentile), output_dir=args.output_dir,
                )
                print(f"Fitted: Band=[{p_low:.3f}, {p_high:.3f}], Unc_Th={unc_th:.3f} (uq=cv_ensemble)")
                thresholds_tensor = torch.tensor([p_low, p_high, unc_th],
                                                 device=accelerator.device, dtype=torch.float32)

        elif args.cv_ensemble_calib == "oof":
            if accelerator.is_local_main_process:
                print("Fitting thresholds on OOF calibration set...")
            if accelerator.is_main_process:
                assert oof_logits is not None and oof_labels is not None
                oof_probs = torch.softmax(
                    torch.from_numpy(oof_logits) / float(max(optimal_T, 1e-6)), dim=1
                ).numpy().astype(np.float32)
                oof_unc = compute_unc_from_probs_np(oof_probs, args.cv_ensemble_oof_unc_metric)
                assert oof_embeddings is not None

                p_low, p_high, unc_th, mcp_engine = _fit_mcp_pattern_a_thresholds(
                    probs=oof_probs, labels=oof_labels, unc=oof_unc,
                    unc_percentile=float(args.unc_percentile),
                    output_dir=args.output_dir, alpha=float(args.alpha),
                )
                print(f"Fitted (OOF): Band=[{p_low:.3f}, {p_high:.3f}], Unc_Th={unc_th:.3f}")
                thresholds_tensor = torch.tensor([p_low, p_high, unc_th],
                                                 device=accelerator.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown cv_ensemble_calib={args.cv_ensemble_calib}")

    else:
        raise ValueError(f"Unknown uq_method={args.uq_method}")

    if accelerator.num_processes > 1:
        torch.distributed.broadcast(thresholds_tensor, src=0)
    p_low, p_high, unc_th = [float(x) for x in thresholds_tensor.tolist()]
    accelerator.wait_for_everyone()

    # =========================================================================
    # 3) TEST inference
    # =========================================================================
    test_probs = None
    test_unc = None
    test_labels = None
    test_entropy = None
    test_piw = None

    if args.uq_method == "none":
        if accelerator.is_local_main_process:
            print("Test Inference (deterministic)...")
        if args.arch == "mil":
            test_probs, test_unc, test_labels, test_entropy, test_piw, test_embeddings = run_inference_mil(
                model, test_loader, accelerator, mc_dropout=False, n_iters=1,
                temperature=optimal_T, unc_metric=args.unc_metric, return_embeddings=True,
            )
        else:
            test_probs, test_unc, test_labels, test_entropy, test_piw, test_embeddings = run_inference_hf(
                model, test_loader, accelerator, mc_dropout=False, n_iters=1,
                temperature=optimal_T, unc_metric=args.unc_metric, return_embeddings=True,
            )

    elif args.uq_method == "mc":
        if accelerator.is_local_main_process:
            print("Test Inference (MC)...")
        if args.arch == "mil":
            test_probs, test_unc, test_labels, test_entropy, test_piw, test_embeddings = run_inference_mil(
                model, test_loader, accelerator, mc_dropout=True, n_iters=args.mc_iters,
                mc_batch_size=args.mc_batch_size, temperature=optimal_T,
                unc_metric=args.unc_metric, return_embeddings=True,
            )
        else:
            test_probs, test_unc, test_labels, test_entropy, test_piw, test_embeddings = run_inference_hf(
                model, test_loader, accelerator, mc_dropout=True, n_iters=args.mc_iters,
                mc_batch_size=args.mc_batch_size, temperature=optimal_T,
                unc_metric=args.unc_metric, return_embeddings=True,
            )

    elif args.uq_method == "mdsn":
        if args.arch != "mil":
            raise ValueError("uq_method=mdsn is only supported for arch=mil")
        if accelerator.is_local_main_process:
            print("Test Inference (MDSN)...")
        test_probs, test_unc, test_labels, test_entropy, test_piw, test_embeddings = run_inference_mil(
            model, test_loader, accelerator, mc_dropout=False, n_iters=1,
            temperature=optimal_T, unc_metric=args.unc_metric,
            return_embeddings=True, return_model_uncertainty=True,
        )

    elif args.uq_method == "cv_ensemble":
        if accelerator.is_local_main_process:
            print("Test Inference (Ensemble)...")
        model_paths = resolve_cv_ensemble_paths(args)
        if args.arch == "mil":
            fn = cv_ensemble_probs_unc2_parallel_models if accelerator.num_processes > 1 else cv_ensemble_probs_unc2
            test_probs, test_unc, test_labels, test_entropy, test_piw, test_embeddings = fn(
                test_loader, accelerator, model_paths, model_loader,
                temperature=optimal_T, unc_metric=args.unc_metric,
                collate_fn=collate, return_embeddings=True,
            )
        else:
            test_probs, test_unc, test_labels, test_entropy, test_piw, test_embeddings = cv_ensemble_probs_unc2_hf(
                test_loader, accelerator, model_paths, model_loader,
                temperature=optimal_T, unc_metric=args.unc_metric, return_embeddings=True,
            )

    accelerator.wait_for_everyone()

    # =========================================================================
    # 3.5) Hybrid decision (MCP + distance veto)
    # =========================================================================
    preds_trinary = None
    categories = None
    sets = None
    test_dists_pred = None
    dist_threshold_by_pred = None
    dist_effective_k_by_class = None
    dist_k_selected_by_class = None

    if accelerator.is_main_process:
        if args.uq_method != "none":
            assert mcp_engine is not None
        assert test_probs is not None and test_unc is not None

        if args.uq_method == "none":
            # Deterministic: no MCP, no distance veto.
            preds_binary_det = (test_probs[:, 1] >= 0.5).astype(np.int64)
            preds_trinary = np.where(preds_binary_det == 1, 2, 0).astype(np.int64)
            categories = np.asarray(["Clear"] * len(preds_trinary), dtype=object)
            sets = [{int(p)} for p in preds_binary_det.tolist()]
            dist_threshold_by_pred = {0: float("inf"), 1: float("inf")}
            test_dists_pred = np.asarray(test_unc, dtype=np.float32)

        elif args.uq_method == "mc":
            # MC-Dropout: use uncertainty threshold instead of distance veto.
            assert mcp_engine is not None
            dist_threshold_by_pred = {0: float("inf"), 1: float("inf")}
            test_dists_pred = np.asarray(test_unc, dtype=np.float32)

            preds_trinary, categories, sets, _ = predict_combined_with_dists_test(
                mcp_engine, test_probs, test_dists_pred, dist_threshold_by_pred,
                unc=np.asarray(test_unc, dtype=np.float64), unc_th=float(unc_th),
                prob_border_eps=float(args.prob_border_eps),
            )

        else:
            # MDSN / cv_ensemble: full distance veto pipeline.
            use_model_dists = (args.uq_method == "mdsn") and (str(args.dist_model) == "mahalanobis")
            if (not use_model_dists) and test_embeddings is None:
                raise ValueError("test_embeddings is None; embeddings must be computed during test inference")

            # Choose calibration set for distance thresholds
            if args.uq_method == "cv_ensemble" and args.cv_ensemble_calib == "oof":
                assert oof_probs is not None and oof_labels is not None and oof_embeddings is not None
                val_probs_dist = oof_probs
                val_labels_dist = oof_labels
                val_emb_dist = oof_embeddings
            else:
                if val_embeddings is None:
                    raise ValueError("val_embeddings is None; need embeddings during val inference")
                val_probs_dist = val_probs
                val_labels_dist = val_labels
                val_emb_dist = val_embeddings

            val_preds = np.argmax(val_probs_dist, axis=1).astype(np.int64)
            test_preds = np.argmax(test_probs, axis=1).astype(np.int64)

            # ID-like calibration mask: singleton MCP & correct prediction
            assert mcp_engine is not None
            val_sets_for_mask, _ = mcp_engine.predict(val_probs_dist)
            val_mcp_singleton = np.asarray([len(s) == 1 for s in val_sets_for_mask], dtype=bool)
            val_correct = (val_preds == np.asarray(val_labels_dist, dtype=np.int64))
            val_cal_mask = val_mcp_singleton & val_correct

            print(
                f"[dist] calibration subset: singleton={float(val_mcp_singleton.mean()):.3f} "
                f"correct={float(val_correct.mean()):.3f} "
                f"singleton&correct={float(val_cal_mask.mean()):.3f} "
                f"(N={int(val_cal_mask.sum())}/{int(len(val_cal_mask))})"
            )

            if use_model_dists:
                val_dists_pred = np.asarray(val_unc, dtype=np.float64)
                test_dists = np.asarray(test_unc, dtype=np.float64)
            else:
                # Choose split to FIT the distance model
                dist_fit_source = str(getattr(args, "dist_fit_source", "val"))
                if dist_fit_source == "train":
                    train_csv = getattr(args, "val_csv", "").replace("val", "train")
                    if not os.path.exists(train_csv):
                        raise FileNotFoundError(f"[dist] train CSV not found: {train_csv}")
                    train_df_dist = pd.read_csv(train_csv)
                    if args.arch == "mil":
                        assert collate is not None
                        train_ds_dist = MILDataset(train_df_dist["text"].tolist(),
                                                   train_df_dist["label"].tolist(), tokenizer)
                        train_loader_dist = DataLoader(train_ds_dist, batch_size=int(args.batch_size),
                                                       collate_fn=collate, shuffle=False)
                    else:
                        t_texts, t_labels = texts_labels_from_df(train_df_dist)
                        assert collate is not None
                        train_ds_dist = TextDataset(t_texts, t_labels)
                        train_loader_dist = DataLoader(train_ds_dist, batch_size=int(args.batch_size),
                                                       collate_fn=collate, shuffle=False)
                    fit_emb_dist = get_embeddings(model, train_loader_dist, collate_fn=collate)
                    fit_labels_dist = np.asarray(train_df_dist["label"].tolist(), dtype=np.int64)
                    print(f"[dist] fitting distance model on TRAIN: N={len(fit_labels_dist)}")

                elif dist_fit_source == "oof":
                    if not (args.uq_method == "cv_ensemble" and args.cv_ensemble_calib == "oof"):
                        raise ValueError("--dist_fit_source=oof requires cv_ensemble + oof calib")
                    if oof_embeddings is None or oof_labels is None:
                        raise ValueError("[dist] OOF embeddings/labels are None")
                    fit_emb_dist = oof_embeddings
                    fit_labels_dist = oof_labels
                    print(f"[dist] fitting distance model on OOF: N={len(fit_labels_dist)}")
                else:
                    fit_emb_dist = val_emb_dist
                    fit_labels_dist = val_labels_dist
                    print(f"[dist] fitting distance model on VAL: N={len(fit_labels_dist)}")

                if str(args.dist_model) == "mixture":
                    selected_k0 = int(args.dist_k0)
                    selected_k1 = int(args.dist_k1)
                    dist_k_selected_for_meta = {0: selected_k0, 1: selected_k1}

                    k_search_mode = str(getattr(args, "dist_k_search", "none"))
                    if k_search_mode == "grid":
                        k0_grid = _parse_int_csv(str(getattr(args, "dist_k0_grid", "")))
                        k1_grid = _parse_int_csv(str(getattr(args, "dist_k1_grid", "")))
                        if not k0_grid:
                            k0_grid = [int(args.dist_k0)]
                        if not k1_grid:
                            k1_grid = [int(args.dist_k1)]
                        k0_grid = list(dict.fromkeys(k0_grid).keys())
                        k1_grid = list(dict.fromkeys(k1_grid).keys())
                        combos = [(k0, k1) for k0 in k0_grid for k1 in k1_grid]

                        max_combos = int(getattr(args, "dist_k_search_max_combos", 25))
                        if len(combos) > max_combos:
                            raise ValueError(f"Too many k combos: {len(combos)} > {max_combos}")

                        assert mcp_engine is not None

                        def _apply_dist_transform_val(d_val_raw, *, preds_val, cal_mask):
                            if str(args.dist_transform) == "robust_z_global":
                                d_val_t, _ = _robust_z_calibrate(d_val_raw, d_val_raw)
                                return np.asarray(d_val_t, dtype=np.float64)
                            if str(args.dist_transform) == "robust_z_by_pred":
                                stats = _fit_robust_z_by_pred_stats(
                                    val_dists=d_val_raw, val_preds=preds_val, cal_mask=cal_mask)
                                return np.asarray(_apply_robust_z_by_pred_stats(
                                    dists=d_val_raw, preds=preds_val, stats=stats), dtype=np.float64)
                            return np.asarray(d_val_raw, dtype=np.float64)

                        def _build_thresholds_from_val(d_val, *, preds_val, q, cal_mask):
                            out = {}
                            for c in np.unique(preds_val):
                                c = int(c)
                                mask = (preds_val == c) & np.asarray(cal_mask, dtype=bool)
                                if int(mask.sum()) < 2:
                                    out[c] = float("inf"); continue
                                arr = d_val[mask][np.isfinite(d_val[mask])]
                                out[c] = float(np.quantile(arr, q, method="higher")) if arr.size else float("inf")
                            return out

                        best_combo = None
                        for k0, k1 in combos:
                            muq = MixtureDistanceUncertainty(
                                k_by_class={0: k0, 1: k1},
                                k_max_by_class={0: int(args.dist_k0_max), 1: int(args.dist_k1_max)},
                                min_cluster_size=int(args.dist_min_cluster_size),
                                kmeans_n_init=int(args.dist_kmeans_n_init),
                                random_state=int(args.seed),
                                auto_k_elbow_gain_th=float(args.dist_k_auto_gain_th),
                            )
                            muq.fit(fit_emb_dist, fit_labels_dist)

                            d_val_raw = np.nan_to_num(muq.predict_distance(val_emb_dist, val_preds),
                                                       nan=1e12, posinf=1e12, neginf=1e12)
                            d_val = np.nan_to_num(
                                _apply_dist_transform_val(d_val_raw, preds_val=val_preds, cal_mask=val_cal_mask),
                                nan=1e12, posinf=1e12, neginf=1e12)

                            if str(args.objective) != "none":
                                qs = np.linspace(float(args.dist_q_min), float(args.dist_q_max), int(args.dist_q_steps))
                                best_q = best_key = best_cov = best_rec = best_th = None
                                for q in qs:
                                    th = _build_thresholds_from_val(d_val, preds_val=val_preds, q=float(q), cal_mask=val_cal_mask)
                                    pv, _, _, _ = predict_combined_with_dists_test(
                                        mcp_engine, val_probs_dist, d_val, th,
                                        prob_border_eps=float(args.prob_border_eps))
                                    cov, rec = _triage_objective_metrics(val_labels_dist, pv)
                                    if args.objective == "max_cov_at_recall":
                                        ok = rec >= float(args.target_recall)
                                        key = (1 if ok else 0, cov, rec)
                                    else:
                                        ok = cov >= float(args.target_coverage)
                                        key = (1 if ok else 0, rec, cov)
                                    if best_key is None or key > best_key:
                                        best_key, best_q, best_cov, best_rec, best_th = key, float(q), cov, rec, th
                                combo_payload = (best_key, k0, k1, best_q, best_cov, best_rec, best_th)
                            else:
                                q0 = float(args.dist_quantile)
                                th = _build_thresholds_from_val(d_val, preds_val=val_preds, q=q0, cal_mask=val_cal_mask)
                                pv, _, _, _ = predict_combined_with_dists_test(
                                    mcp_engine, val_probs_dist, d_val, th,
                                    prob_border_eps=float(args.prob_border_eps))
                                cov, rec = _triage_objective_metrics(val_labels_dist, pv)
                                combo_payload = ((cov, rec), k0, k1, q0, cov, rec, th)

                            if best_combo is None or combo_payload[0] > best_combo[0]:
                                best_combo = combo_payload
                            print(f"[dist-k-search] k0={k0} k1={k1} -> q={combo_payload[3]:.4f} "
                                  f"cov={combo_payload[4]:.3f} rec={combo_payload[5]:.3f}")

                        assert best_combo is not None
                        _, k0_best, k1_best, _, _, _, _ = best_combo
                        selected_k0, selected_k1 = int(k0_best), int(k1_best)
                        dist_k_selected_for_meta = {0: selected_k0, 1: selected_k1}
                        print(f"[dist-k-search] selected k0={selected_k0} k1={selected_k1}")

                    distance_uq = MixtureDistanceUncertainty(
                        k_by_class={0: selected_k0, 1: selected_k1},
                        k_max_by_class={0: int(args.dist_k0_max), 1: int(args.dist_k1_max)},
                        min_cluster_size=int(args.dist_min_cluster_size),
                        kmeans_n_init=int(args.dist_kmeans_n_init),
                        random_state=int(args.seed),
                        auto_k_elbow_gain_th=float(args.dist_k_auto_gain_th),
                    )
                    distance_uq.fit(fit_emb_dist, fit_labels_dist)
                else:
                    distance_uq = DistanceUncertainty()
                    distance_uq.fit(
                        fit_emb_dist, fit_labels_dist,
                        single=True if args.cv_ensemble_calib == "fold_val" else False,
                    )
                    dist_k_selected_for_meta = None

                distance_means_for_export = {
                    int(c): np.asarray(mu, dtype=np.float64) for c, mu in distance_uq.means.items()
                }
                distance_precs_for_export = {
                    int(c): np.asarray(pr, dtype=np.float64) for c, pr in distance_uq.precs.items()
                }
                dist_effective_k_by_class = getattr(distance_uq, "effective_k_by_class", None)
                dist_k_selected_by_class = locals().get("dist_k_selected_for_meta", None)
                val_dists_pred = distance_uq.predict_distance(val_emb_dist, val_preds)
                test_preds = np.argmax(test_probs, axis=1)
                test_dists = distance_uq.predict_distance(test_embeddings, test_preds)

            # Force finiteness
            val_dists_pred = np.nan_to_num(val_dists_pred, nan=1e12, posinf=1e12, neginf=1e12)
            test_dists = np.nan_to_num(test_dists, nan=1e12, posinf=1e12, neginf=1e12)
            val_dists_raw = val_dists_pred.copy()
            test_dists_raw = test_dists.copy()

            # Optional transform
            if str(args.dist_transform) == "robust_z_global":
                val_dists_pred, test_dists = _robust_z_calibrate(val_dists_pred, test_dists)
                print("[md-dist] using robust_z_global transform")
            elif str(args.dist_transform) == "robust_z_by_pred":
                robust_z_by_pred_stats_for_export = _fit_robust_z_by_pred_stats(
                    val_dists=val_dists_pred, val_preds=val_preds, cal_mask=val_cal_mask)
                val_dists_pred = _apply_robust_z_by_pred_stats(
                    dists=val_dists_pred, preds=val_preds, stats=robust_z_by_pred_stats_for_export)
                test_dists = _apply_robust_z_by_pred_stats(
                    dists=test_dists, preds=test_preds, stats=robust_z_by_pred_stats_for_export)
                print("[md-dist] using robust_z_by_pred transform")
            else:
                print("[md-dist] using RAW distances")

            def build_thresholds(q, *, cal_mask=None):
                q = float(q)
                if cal_mask is None:
                    cal_mask = np.ones((len(val_preds),), dtype=bool)
                out = {}
                for c in np.unique(val_preds):
                    c = int(c)
                    mask = (val_preds == c) & np.asarray(cal_mask, dtype=bool)
                    if int(mask.sum()) < 2:
                        out[c] = float("inf"); continue
                    arr = val_dists_pred[mask][np.isfinite(val_dists_pred[mask])]
                    out[c] = float(np.quantile(arr, q, method="higher")) if arr.size else float("inf")
                return out

            # Conformal distance calibration
            q_dist = float(args.dist_quantile)
            md_alpha = float(np.clip(1.0 - q_dist, 1e-6, 0.49))
            dist_quantile_final = float(q_dist)

            dist_threshold_by_pred = build_thresholds(q_dist, cal_mask=val_cal_mask)
            print(f"[conformal-md] q={q_dist:.4f} thresholds={dist_threshold_by_pred}")

            # Objective search
            if str(args.objective) != "none":
                qs = np.linspace(float(args.dist_q_min), float(args.dist_q_max), int(args.dist_q_steps))
                best = None
                for q in qs:
                    th = build_thresholds(float(q), cal_mask=val_cal_mask)
                    pv, _, _, _ = predict_combined_with_dists_test(
                        mcp_engine, val_probs_dist, val_dists_pred, th,
                        prob_border_eps=float(args.prob_border_eps))
                    cov, rec = _triage_objective_metrics(val_labels_dist, pv)
                    if args.objective == "max_cov_at_recall":
                        ok = rec >= float(args.target_recall)
                        key = (1 if ok else 0, cov, rec)
                    else:
                        ok = cov >= float(args.target_coverage)
                        key = (1 if ok else 0, rec, cov)
                    if best is None or key > best[0]:
                        best = (key, float(q), cov, rec, th)

                assert best is not None
                _, q_best, cov_best, rec_best, th_best = best
                dist_threshold_by_pred = th_best
                dist_quantile_final = float(q_best)
                print(f"[objective] selected q={dist_quantile_final:.4f} "
                      f"val_coverage={cov_best:.3f} val_recall_pos={rec_best:.3f}")

            preds_trinary, categories, sets, test_dists_pred = predict_combined_with_dists_test(
                mcp_engine, test_probs, test_dists, dist_threshold_by_pred,
                prob_border_eps=float(args.prob_border_eps),
            )

            # Optional OOF fold metrics
            if bool(getattr(args, "report_oof_fold_metrics", False)) and (
                args.uq_method == "cv_ensemble" and args.cv_ensemble_calib == "oof"
            ):
                _report_oof_fold_metrics(
                    args=args, mcp_engine=mcp_engine,
                    oof_probs=oof_probs, oof_labels=oof_labels,
                    oof_unc=oof_unc, oof_fold_ids=oof_fold_ids,
                    val_dists_pred=val_dists_pred,
                    dist_threshold_by_pred=dist_threshold_by_pred,
                    preds_oof_trinary=predict_combined_with_dists_test(
                        mcp_engine, oof_probs, val_dists_pred, dist_threshold_by_pred,
                        prob_border_eps=float(args.prob_border_eps))[0],
                )

    # =========================================================================
    # 4) Metrics + outputs
    # =========================================================================
    if accelerator.is_main_process:
        if args.uq_method != "none":
            assert mcp_engine is not None
        assert test_probs is not None and test_unc is not None

        test_labels_arr = np.asarray(test_labels, dtype=np.int64)
        assert preds_trinary is not None and categories is not None

        mask_clear = preds_trinary != 1
        if mask_clear.sum() > 0:
            metrics_clear = compute_comprehensive_metrics(
                test_labels_arr[mask_clear], preds_trinary[mask_clear] // 2,
                test_probs[mask_clear][:, 1], prefix="clear_",
            )
        else:
            metrics_clear = {}
            print("[warn] No clear cases (all deferred).")

        mask_neutral = preds_trinary == 1
        if mask_neutral.sum() > 0:
            preds_neutral_forced = (test_probs[mask_neutral][:, 1] >= 0.5).astype(int)
            metrics_neutral = compute_comprehensive_metrics(
                test_labels_arr[mask_neutral], preds_neutral_forced,
                test_probs[mask_neutral][:, 1], prefix="neutral_",
            )
        else:
            metrics_neutral = {}

        preds_binary = (test_probs[:, 1] >= 0.5).astype(int)

        if test_entropy is None:
            test_entropy = entropy_np(test_probs).astype(np.float32)
        if test_piw is None:
            test_piw = np.zeros_like(test_entropy, dtype=np.float32)

        metrics_binary = compute_comprehensive_metrics(
            test_labels_arr, preds_binary, test_probs[:, 1],
            uncertainties=test_unc, entropy=test_entropy, piw=test_piw,
            prefix="binary_",
        )

        custom_kappa, cm_3cat = compute_custom_kappa(test_labels_arr, preds_trinary)
        y_true_mapped = np.where(test_labels_arr == 1, 2, 0)
        kappa_linear = cohen_kappa_score(y_true_mapped, preds_trinary, weights="linear")
        kappa_quad = cohen_kappa_score(y_true_mapped, preds_trinary, weights="quadratic")

        ece_score = compute_ece(test_labels_arr, test_probs[:, 1])
        coverage = float((preds_trinary != 1).mean())

        pos_mask = (test_labels_arr == 1)
        n_pos = int(pos_mask.sum())
        n_pos_deferred = int(((preds_trinary == 1) & pos_mask).sum())
        positive_deferral_rate = float(n_pos_deferred / max(1, n_pos))

        y_prob_pos = test_probs[:, 1]
        aurc = compute_aurc(test_labels_arr, y_prob_pos, test_unc)
        eaurc = compute_eaurc(test_labels_arr, y_prob_pos, test_unc)
        unc_auroc_err = compute_unc_auroc_error(test_labels_arr, y_prob_pos, test_unc)
        rcc_auc = compute_rcc_auc(test_labels_arr, y_prob_pos, test_unc)
        rpp = compute_rpp_error(test_labels_arr, y_prob_pos, test_unc)
        risk_points = compute_risk_at_coverages(test_labels_arr, y_prob_pos, test_unc,
                                                coverages=(0.80, 0.90, 0.95))
        unc_margin_spearman = compute_spearman_uncertainty_margin(y_prob_pos, test_unc)

        ece_clear = float(compute_ece(test_labels_arr[mask_clear], test_probs[mask_clear][:, 1])) if mask_clear.sum() > 0 else 0.0
        ece_neutral = float(compute_ece(test_labels_arr[mask_neutral], test_probs[mask_neutral][:, 1])) if mask_neutral.sum() > 0 else 0.0

        print("\n" + "=" * 40)
        print("FINAL RESULTS")
        print("=" * 40)
        print(f"Coverage:        {coverage:.1%}")
        print(f"Pos deferral:    {positive_deferral_rate:.1%}")
        print(f"ECE:             {ece_score:.4f}")
        print(f"ECE (clear):     {ece_clear:.4f}")
        print(f"ECE (neutral):   {ece_neutral:.4f}")
        print(f"Linear Kappa:    {kappa_linear:.4f}")
        print(f"Quadratic Kappa: {kappa_quad:.4f}")
        print(f"Custom Kappa:    {custom_kappa:.4f}")
        print(f"AURC:            {aurc:.4f}")
        print(f"EAURC:           {eaurc:.4f}")
        print(f"RCC-AUC:         {rcc_auc:.4f}")
        print(f"RPP:             {rpp:.4f}")
        print(f"Unc AUROC(err):  {unc_auroc_err:.4f}")
        print(f"Risk@80/90/95:   {risk_points['risk@80']:.4f} / {risk_points['risk@90']:.4f} / {risk_points['risk@95']:.4f}")
        print(f"Spearman(unc,|p-0.5|): {unc_margin_spearman:.4f}")

        print("-" * 20 + "\nBINARY METRICS:")
        for k, v in metrics_binary.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("-" * 20 + "\nCLEAR CASES METRICS:")
        for k, v in metrics_clear.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        json_metrics = {
            "cv_id": int(args.cv_id),
            "p_low": float(p_low), "p_high": float(p_high), "unc_th": float(unc_th),
            "coverage": float(coverage),
            "positive_deferral_rate": float(positive_deferral_rate),
            "ece": float(ece_score), "ece_clear": float(ece_clear), "ece_neutral": float(ece_neutral),
            "kappa_custom": float(custom_kappa),
            "kappa_linear": float(kappa_linear), "kappa_quadratic": float(kappa_quad),
            "aurc": float(aurc), "eaurc": float(eaurc),
            "rcc_auc": float(rcc_auc), "rpp": float(rpp),
            "unc_auroc_error": float(unc_auroc_err),
            "risk@80": float(risk_points["risk@80"]),
            "risk@90": float(risk_points["risk@90"]),
            "risk@95": float(risk_points["risk@95"]),
            "spearman_unc_margin": float(unc_margin_spearman),
            **metrics_binary, **metrics_clear, **metrics_neutral,
        }
        with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(json_sanitize(json_metrics), f, indent=4, ensure_ascii=False)

        # ---- Bootstrap variability ----
        if int(getattr(args, "test_bootstrap_n", 0) or 0) > 0:
            _run_test_bootstrap(args, test_labels_arr, test_probs, test_unc, preds_trinary)

        # ---- CSV output ----
        n = len(test_probs)
        test_final = test_df
        text_col = "text"
        df_out = pd.DataFrame({
            "id": list(range(n)),
            "label": test_final["label"].tolist(),
            "text_preview": [_preview_text(x) for x in test_final[text_col].tolist()],
            "prob_pos": test_probs[:, 1],
            "uncertainty": test_unc,
            "distance": test_dists_pred,
            "pred_trinary": preds_trinary,
            "category": categories,
        })
        df_out.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

        # ---- Plots ----
        print("\nGenerating Analysis Plots...")
        try:
            plot_calibration_curve(test_labels_arr, test_probs[:, 1], args.output_dir)
        except Exception as e:
            print(f"[plot] calibration failed: {e}")
        try:
            plot_rejection_curves(test_labels_arr, test_probs[:, 1], test_unc, args.output_dir)
        except Exception as e:
            print(f"[plot] rejection_curves failed: {e}")
        try:
            plot_uncertainty_diagnostics(test_labels_arr, test_probs[:, 1], test_unc, args.output_dir)
        except Exception as e:
            print(f"[plot] diagnostics failed: {e}")

        if args.uq_method not in {"mc", "none"}:
            try:
                plot_combined_safety_analysis_test(
                    probs=test_probs[:, 1], dists=test_dists_pred,
                    final_labels=categories, mcp_model=mcp_engine,
                    dist_threshold=dist_threshold_by_pred, output_dir=args.output_dir,
                )
            except Exception as e:
                print(f"[plot] combined_safety failed: {e}")

        # ---- Calibration bundle ----
        if not bool(args.skip_save_calibration):
            calibration_path = (
                os.path.abspath(args.save_calibration_file)
                if args.save_calibration_file is not None
                else os.path.join(args.output_dir, "calibration_bundle.npz")
            )
            mcp_thresholds = {}
            if mcp_engine is not None and hasattr(mcp_engine, "thresholds"):
                mcp_thresholds = {int(k): float(v) for k, v in mcp_engine.thresholds.items()}

            calibration_meta = {
                "format_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": os.path.abspath(__file__),
                "model_path": os.path.abspath(args.model_path),
                "arch": str(args.arch),
                "uq_method": str(args.uq_method),
                "temperature": float(optimal_T),
                "mcp": {
                    "alpha": float(getattr(mcp_engine, "alpha", 0.01) if mcp_engine else 0.01),
                    "thresholds": _json_ready_dict(mcp_thresholds),
                    "p_low": float(p_low), "p_high": float(p_high),
                },
                "uncertainty": {
                    "unc_th": float(unc_th),
                    "unc_metric": str(args.unc_metric),
                    "unc_percentile": float(args.unc_percentile),
                },
                "decision": {
                    "prob_border_eps": float(args.prob_border_eps),
                    "dist_model": str(args.dist_model),
                    "dist_fit_source": str(args.dist_fit_source),
                    "dist_k0": int(args.dist_k0), "dist_k1": int(args.dist_k1),
                    "dist_k_search": str(args.dist_k_search),
                    "dist_k_selected_by_class": _json_ready_dict(dist_k_selected_by_class or {}),
                    "dist_k_effective_by_class": _json_ready_dict(dist_effective_k_by_class or {}),
                    "dist_transform": str(args.dist_transform),
                    "dist_quantile_arg": float(args.dist_quantile),
                    "dist_quantile_final": float(locals().get("dist_quantile_final", float(args.dist_quantile))),
                    "dist_threshold_by_pred": _json_ready_dict(dist_threshold_by_pred or {}),
                    "robust_z_by_pred_stats": robust_z_by_pred_stats_for_export or {},
                },
                "cv_ensemble": {
                    "calib": str(args.cv_ensemble_calib),
                    "oof_unc_metric": str(args.cv_ensemble_oof_unc_metric),
                },
                "runtime_args": json_sanitize(vars(args)),
            }
            saved_path = _save_calibration_bundle(
                output_path=calibration_path, meta=calibration_meta,
                distance_means=distance_means_for_export,
                distance_precs=distance_precs_for_export,
            )
            print(f"Calibration bundle saved to {saved_path}")

        print(f"Results saved to {args.output_dir}")


# =========================================================================
# Bootstrap helper
# =========================================================================

def _run_test_bootstrap(args, test_labels_arr, test_probs, test_unc, preds_trinary):
    """Run bootstrap resampling on test set and write summary JSON."""
    b = int(args.test_bootstrap_n)
    seed = int(args.test_bootstrap_seed or args.seed)
    rng = np.random.default_rng(seed)
    n_total = int(test_labels_arr.shape[0])

    metric_names = [
        "coverage", "positive_deferral_rate", "kappa_custom",
        "aurc", "binary_f2", "binary_recall", "binary_specificity",
        "binary_npv", "clear_f2", "clear_npv", "binary_ece", "clear_ece",
    ]
    samples = {k: [] for k in metric_names}

    for _ in range(b):
        idx = rng.integers(0, n_total, size=n_total, endpoint=False)
        y_b = test_labels_arr[idx]
        probs_b = test_probs[idx]
        unc_b = np.asarray(test_unc, dtype=np.float64)[idx]
        preds_tri_b = np.asarray(preds_trinary, dtype=np.int64)[idx]

        cov_b = float((preds_tri_b != 1).mean())
        pos_b = (y_b == 1)
        pos_def_b = float(((preds_tri_b == 1) & pos_b).sum() / max(1, int(pos_b.sum())))

        kappa_b, _ = compute_custom_kappa(y_b, preds_tri_b)
        aurc_b = float(compute_aurc(y_b, probs_b[:, 1], unc_b))

        pred_bin_b = (probs_b[:, 1] >= 0.5).astype(np.int64)
        tp = int(((y_b == 1) & (pred_bin_b == 1)).sum())
        fp = int(((y_b == 0) & (pred_bin_b == 1)).sum())
        tn = int(((y_b == 0) & (pred_bin_b == 0)).sum())
        fn = int(((y_b == 1) & (pred_bin_b == 0)).sum())

        def _f2(tp, fp, fn):
            if tp <= 0: return 0.0
            p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
            return 5.0 * p * r / max(1e-12, 4.0 * p + r)

        samples["coverage"].append(cov_b)
        samples["positive_deferral_rate"].append(pos_def_b)
        samples["kappa_custom"].append(float(kappa_b))
        samples["aurc"].append(aurc_b)
        samples["binary_f2"].append(_f2(tp, fp, fn))
        samples["binary_recall"].append(tp / max(1, tp + fn))
        samples["binary_specificity"].append(tn / max(1, tn + fp))
        samples["binary_npv"].append(tn / max(1, tn + fn))
        samples["binary_ece"].append(float(compute_ece(y_b, probs_b[:, 1])))

        mask_c = preds_tri_b != 1
        if int(mask_c.sum()) > 0:
            yc, pc = y_b[mask_c], (preds_tri_b[mask_c] // 2).astype(np.int64)
            tp_c = int(((yc == 1) & (pc == 1)).sum())
            fp_c = int(((yc == 0) & (pc == 1)).sum())
            tn_c = int(((yc == 0) & (pc == 0)).sum())
            fn_c = int(((yc == 1) & (pc == 0)).sum())
            samples["clear_f2"].append(_f2(tp_c, fp_c, fn_c))
            samples["clear_npv"].append(tn_c / max(1, tn_c + fn_c))
            samples["clear_ece"].append(float(compute_ece(yc, probs_b[mask_c][:, 1])))
        else:
            samples["clear_f2"].append(float("nan"))
            samples["clear_npv"].append(float("nan"))
            samples["clear_ece"].append(float("nan"))

    def _summ(vals):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0: return {"mean": None, "std": None, "n": 0, "ci95": [None, None]}
        lo, hi = np.quantile(arr, [0.025, 0.975]).tolist()
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
                "n": int(arr.size), "ci95": [float(lo), float(hi)]}

    payload = {"split": "test", "n_bootstrap": b, "sample_size": n_total, "seed": seed,
               "metrics": {k: _summ(v) for k, v in samples.items()}}

    out_path = str(args.test_bootstrap_out or "").strip()
    if not out_path:
        out_path = os.path.join(args.output_dir, "test_bootstrap_metrics.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, ensure_ascii=False)
    print(f"[test-bootstrap] B={b} seed={seed} wrote: {out_path}")


# =========================================================================
# OOF fold metrics helper
# =========================================================================

def _report_oof_fold_metrics(*, args, mcp_engine, oof_probs, oof_labels, oof_unc,
                              oof_fold_ids, val_dists_pred, dist_threshold_by_pred,
                              preds_oof_trinary):
    """Compute and write per-fold metrics on OOF calibration data."""
    fold_ids = np.asarray(oof_fold_ids, dtype=np.int64)
    uniq_folds = sorted(int(x) for x in np.unique(fold_ids))

    def _block(y_true, probs, unc, preds_tri):
        y_true = np.asarray(y_true, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        unc = np.asarray(unc, dtype=np.float64)
        preds_tri = np.asarray(preds_tri, dtype=np.int64)

        mask_clear = preds_tri != 1
        if int(mask_clear.sum()) > 0:
            mc = compute_comprehensive_metrics(y_true[mask_clear], preds_tri[mask_clear] // 2,
                                               probs[mask_clear][:, 1], prefix="clear_")
        else:
            mc = {}

        preds_binary = (probs[:, 1] >= 0.5).astype(int)
        ent = entropy_np(probs).astype(np.float32)
        mb = compute_comprehensive_metrics(y_true, preds_binary, probs[:, 1],
                                           uncertainties=unc, entropy=ent,
                                           piw=np.zeros_like(ent), prefix="binary_")

        kappa_c, _ = compute_custom_kappa(y_true, preds_tri)
        cov = float((preds_tri != 1).mean())
        pos = (y_true == 1)
        pos_def = float(((preds_tri == 1) & pos).sum() / max(1, int(pos.sum())))

        return {
            "n": int(y_true.shape[0]), "coverage": cov,
            "positive_deferral_rate": pos_def,
            "ece": float(compute_ece(y_true, probs[:, 1])),
            "kappa_custom": float(kappa_c),
            "aurc": float(compute_aurc(y_true, probs[:, 1], unc)),
            **mb, **mc,
        }

    overall = _block(oof_labels, oof_probs, oof_unc, preds_oof_trinary)
    per_fold = {}
    fold_dicts = []
    for fi in uniq_folds:
        m = fold_ids == fi
        if int(m.sum()) == 0: continue
        d = _block(np.asarray(oof_labels)[m], np.asarray(oof_probs)[m],
                    np.asarray(oof_unc)[m], np.asarray(preds_oof_trinary)[m])
        per_fold[str(fi)] = d
        fold_dicts.append(d)

    # Mean/std across folds
    summary = {}
    if fold_dicts:
        keys = set()
        for dd in fold_dicts:
            keys.update(dd.keys())
        for k in sorted(keys):
            vals = [float(dd[k]) for dd in fold_dicts
                    if k in dd and isinstance(dd[k], (int, float, np.integer, np.floating))
                    and np.isfinite(float(dd[k]))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                summary[k] = {"mean": float(arr.mean()),
                               "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
                               "n": float(arr.size)}

    payload = {
        "cv_ensemble_calib": str(args.cv_ensemble_calib),
        "n_folds": len(uniq_folds),
        "overall_oof": overall,
        "per_fold": per_fold,
        "summary_mean_std": summary,
    }

    out_path = str(getattr(args, "oof_fold_metrics_out", "") or "").strip()
    if not out_path:
        out_path = os.path.join(args.output_dir, "oof_fold_metrics.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, ensure_ascii=False)
    print(f"[oof-fold-metrics] wrote: {out_path}")


if __name__ == "__main__":
    main()
