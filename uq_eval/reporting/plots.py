from __future__ import annotations

import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score


def _maybe_import_seaborn():
    try:
        return importlib.import_module("seaborn")
    except Exception:
        return None


from sklearn.metrics import fbeta_score
def compute_f2_rejection_curve(y_true, y_prob, uncertainty, thresholds=100):
    """
    Computes the F2 score at different coverage levels by rejecting the most uncertain predictions.
    
    Args:
        y_true: Array of true labels (0 or 1).
        y_prob: Array of predicted probabilities for the positive class.
        uncertainty: Array of uncertainty scores (higher means more uncertain).
        thresholds: Number of points to evaluate along the coverage curve.
        
    Returns:
        coverages: List of coverage percentages (0.0 to 1.0).
        f2_scores: List of F2 scores corresponding to those coverages.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    uncertainty = np.asarray(uncertainty)
    
    # Sort indices by uncertainty in descending order (most uncertain first)
    sorted_indices = np.argsort(uncertainty)[::-1]
    
    coverages = []
    f2_scores = []
    
    n_samples = len(y_true)
    
    # Iterate through different fractions of rejection
    for drop_fraction in np.linspace(0.0, 0.95, thresholds):
        drop_count = int(drop_fraction * n_samples)
        keep_indices = sorted_indices[drop_count:]
        
        if len(keep_indices) == 0:
            continue
            
        y_true_kept = y_true[keep_indices]
        y_prob_kept = y_prob[keep_indices]
        
        # Convert probabilities to binary predictions
        y_pred_kept = (y_prob_kept >= 0.5).astype(int)
        
        # Calculate F2 Score (beta=2 weights recall higher than precision)
        # Avoid zero division if no positive predictions exist
        if np.sum(y_pred_kept) == 0 and np.sum(y_true_kept) == 0:
            f2 = 1.0
        elif np.sum(y_pred_kept) == 0 or np.sum(y_true_kept) == 0:
            f2 = 0.0
        else:
            f2 = fbeta_score(y_true_kept, y_pred_kept, beta=2.0)
            
        coverage = len(keep_indices) / n_samples
        
        coverages.append(coverage)
        f2_scores.append(f2)
        
    return coverages, f2_scores


def plot_f2_vs_coverage_comparison(models_dict, output_path="f2_vs_coverage.pdf"):
    """
    Plots the F2 vs Coverage curves for multiple models.
    
    Args:
        models_dict: Dictionary where keys are model names (e.g., "MD-SN", "MC Dropout")
                     and values are tuples of (y_true, y_prob, uncertainty).
    """
    plt.figure(figsize=(8, 6))
    
    # Use distinct markers and linestyles for academic publication clarity
    styles = [('-', 'o'), ('--', 's'), ('-.', '^'), (':', 'D')]
    
    for idx, (model_name, (y_true, y_prob, unc)) in enumerate(models_dict.items()):
        coverages, f2_scores = compute_f2_rejection_curve(y_true, y_prob, unc)
        
        line_style, marker = styles[idx % len(styles)]
        
        plt.plot(
            coverages, 
            f2_scores, 
            label=model_name, 
            linestyle=line_style, 
            linewidth=2,
            markevery=10, # Add markers to distinguish lines in black-and-white print
            marker=marker,
            markersize=6
        )

    plt.title("Selective Screening Safety: F2-Score vs. Coverage", fontsize=14, fontweight='bold')
    plt.xlabel("Coverage (Fraction of Automated Cases)", fontsize=12)
    plt.ylabel("Discriminative Safety (F2-Score)", fontsize=12)
    
    # Invert X-axis because we usually read from 100% coverage down to stricter regimes
    plt.xlim(1.0, 0.05) 
    plt.ylim(0.75, 1.0) # Adjust based on your actual data ranges
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot successfully saved to {output_path}")


def plot_combined_safety_analysis_test(
    probs,
    dists,
    final_labels,
    mcp_model,
    dist_threshold,
    output_dir: str,
    filename: str = "combined_safety_analysis.pdf",
):
    """
    Plots the Hybrid Safety System:
    - Vertical Lines: Mondrian Probability Thresholds (Ambiguity)
    - Horizontal Line(s): Mahalanobis Distance Threshold(s) (Geometric Veto)

    dist_threshold can be:
      - float/int: a single global threshold
      - dict: per-class thresholds, e.g. {0: thr0, 1: thr1}
    """
    if probs is None or dists is None:
        return

    # Prepare Data
    prob_pos = probs[:, 1] if np.asarray(probs).ndim == 2 else np.asarray(probs, dtype=float)
    dists = np.asarray(dists, dtype=float)
    final_labels = np.asarray(final_labels)

    # Ensure finite dists for plotting limits (keep points inside)
    finite_d = dists[np.isfinite(dists)]
    if finite_d.size == 0:
        return
    y_max = float(np.max(finite_d))
    if y_max <= 0:
        y_max = 1.0
    y_lim_top = y_max * 1.05

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- 1. SETUP THRESHOLDS ---
    th0 = float(mcp_model.thresholds.get(0, 0.5))
    th1 = float(mcp_model.thresholds.get(1, 0.5))

    # Keep same semantics used elsewhere:
    # include class0 if p_pos <= th0; include class1 if p_pos >= 1 - th1
    p_end_neg = th0
    p_start_pos = 1.0 - th1

    dist_thr_by_class = None
    if isinstance(dist_threshold, dict):
        dist_thr_by_class = {int(k): float(v) for k, v in dist_threshold.items()}
        y_lines = [v for v in dist_thr_by_class.values() if np.isfinite(v)]
        y_shade_line = float(min(y_lines)) if len(y_lines) else None
    else:
        y_shade_line = float(dist_threshold) if dist_threshold is not None else None

    # --- 2. DRAW ZONES ---
    rect_amb = patches.Rectangle(
        (min(p_end_neg, p_start_pos), 0),
        abs(p_start_pos - p_end_neg),
        y_lim_top,
        linewidth=0,
        facecolor="gold",
        alpha=0.15,
        label="Zone: Probabilistic Ambiguity",
    )
    ax.add_patch(rect_amb)

    # Draw threshold lines (NO LEGEND entries)
    vline_style = dict(color="gold", linestyle="--", linewidth=1.8)
    ax.axvline(p_end_neg, label="_nolegend_", **vline_style)
    ax.axvline(p_start_pos, label="_nolegend_", **vline_style)

    if y_shade_line is not None and np.isfinite(y_shade_line):
        ax.fill_between(
            x=[0, 1],
            y1=float(y_shade_line),
            y2=y_lim_top,
            color="red",
            alpha=0.07,
            label="Zone: Epistemic Veto (OOD)",
        )

    # Collect horizontal line annotations: (thr_value, color, text)
    hline_ann = []

    if dist_thr_by_class is not None:
        styles = {
            0: dict(color="#C0392B", linestyle="-", linewidth=2.0),   # dark red
            1: dict(color="#E74C3C", linestyle="--", linewidth=2.0),  # red dashed
        }
        for cls, thr in sorted(dist_thr_by_class.items(), key=lambda kv: kv[0]):
            if not np.isfinite(thr):
                continue
            st = styles.get(int(cls), dict(color="red", linestyle=":", linewidth=2.0))
            ax.axhline(thr, label="_nolegend_", **st)
            hline_ann.append((thr, st.get("color", "red"), f"Dist thr (class {cls}) = {thr:.2f}"))
    else:
        if y_shade_line is not None and np.isfinite(y_shade_line):
            ax.axhline(
                float(y_shade_line),
                color="red",
                linestyle="-",
                linewidth=2,
                label="_nolegend_",
            )
            hline_ann.append((float(y_shade_line), "red", f"Dist thr = {float(y_shade_line):.2f}"))

    # --- 3. SCATTER PLOT ---
    mask_clear = np.char.find(final_labels.astype(str), "Clear") >= 0
    ax.scatter(prob_pos[mask_clear], dists[mask_clear], c="tab:blue", alpha=0.3, s=15, label="Automated (Clear)")

    mask_amb = np.char.find(final_labels.astype(str), "Ambiguous") >= 0
    ax.scatter(prob_pos[mask_amb], dists[mask_amb], c="orange", marker="s", alpha=0.6, s=25, label="Defer: Ambiguous")

    mask_epi = np.char.find(final_labels.astype(str), "Epistemic") >= 0
    ax.scatter(prob_pos[mask_epi], dists[mask_epi], c="red", marker="^", alpha=0.8, s=35, label="Defer: OOD/Complex")

    mask_null = np.char.find(final_labels.astype(str), "Null") >= 0
    if np.any(mask_null):
        ax.scatter(prob_pos[mask_null], dists[mask_null], c="black", marker="x", alpha=0.5, s=30, label="Defer: Conflict")

    # --- 4. FORMATTING ---
    ax.set_xlabel("Predicted Probability P(HIV+)")
    ax.set_ylabel("Mahalanobis Distance (Feature Normality)")
    ax.set_title("Hybrid Safety: Mondrian (Ambiguity) + Distance (OOD)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_lim_top)

    # --- 5. ANNOTATE THRESHOLDS ON THE BORDER (not on the lines) ---
    top_text_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    # Small x-offset so text is not exactly on the line
    dx = 0.006

    ax.text(
        float(np.clip(p_end_neg - dx, 0.0, 1.0)),
        1.01,
        f"p_high={p_end_neg:.3f}",
        transform=top_text_transform,
        rotation=0,
        color="goldenrod",
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.5),
        clip_on=False,
    )
    ax.text(
        float(np.clip(p_start_pos + dx, 0.0, 1.0)),
        1.01,
        f"p_low={p_start_pos:.3f}",
        transform=top_text_transform,
        rotation=0,
        color="goldenrod",
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.5),
        clip_on=False,
    )

    # Horizontal line labels on the right side (outside axes) - keep
    right_text_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for thr, color, txt in hline_ann:
        ax.text(
            1.01,
            float(thr),
            txt,
            transform=right_text_transform,
            color=color,
            ha="left",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
            clip_on=False,
        )

    ax.grid(True, alpha=0.2)

    # --- 6. LEGEND AT THE BOTTOM ---
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.9,
        borderaxespad=0.0,
    )

    # Make room for outside legend/labels
    fig.subplots_adjust(right=0.8)

    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()

def plot_combined_safety_analysis(
    probs,
    dists,
    final_labels,
    mcp_model,
    dist_threshold,
    output_dir: str,
    filename: str = "combined_safety_analysis.pdf",
):
    """
    Plots the Hybrid Safety System:
    - Vertical Lines: Mondrian Probability Thresholds (Ambiguity)
    - Horizontal Line(s): Mahalanobis Distance Threshold(s) (Geometric Veto)

    dist_threshold can be:
      - float/int: a single global threshold
      - dict: per-class thresholds, e.g. {0: thr0, 1: thr1}
    """
    if probs is None or dists is None:
        return

    # Prepare Data
    prob_pos = probs[:, 1] if np.asarray(probs).ndim == 2 else np.asarray(probs, dtype=float)
    dists = np.asarray(dists, dtype=float)
    final_labels = np.asarray(final_labels)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- 1. SETUP THRESHOLDS ---
    # Mondrian Thresholds (Vertical)
    th0 = mcp_model.thresholds.get(0, 0.5)
    th1 = mcp_model.thresholds.get(1, 0.5)

    p_end_neg = th0
    p_start_pos = 1.0 - th1

    # Distance Threshold(s) (Horizontal)
    # Allow scalar or dict by class
    dist_thr_by_class = None
    if isinstance(dist_threshold, dict):
        # normalize keys/values
        dist_thr_by_class = {int(k): float(v) for k, v in dist_threshold.items()}
        y_lines = [v for v in dist_thr_by_class.values() if np.isfinite(v)]
        y_shade_line = float(min(y_lines)) if len(y_lines) else None
    else:
        y_shade_line = float(dist_threshold) if dist_threshold is not None else None

    # --- 2. DRAW ZONES ---

    # Zone A: The "Ambiguity" Vertical Strip (Mondrian)
    # if p_start_pos > p_end_neg:
    rect_amb = patches.Rectangle(
        (p_end_neg, 0),
        p_start_pos - p_end_neg,
        np.max(dists) * 1.1,
        linewidth=0,
        facecolor="gold",
        alpha=0.15,
        label="Zone: Probabilistic Ambiguity",
    )
    ax.add_patch(rect_amb)

    ax.axvline(p_end_neg, color="gold", linestyle="--", linewidth=1.8, label=f"MCP th0: p_pos ≤ {p_end_neg:.3f}")
    ax.axvline(p_start_pos, color="gold", linestyle="--", linewidth=1.8, label=f"MCP th1: p_pos ≥ {p_start_pos:.3f}")

    # Zone B: Epistemic area (optional shading)
    if y_shade_line is not None and np.isfinite(y_shade_line):
        ax.fill_between(
            x=[0, 1],
            y1=y_shade_line,
            y2=np.max(dists) * 1.2,
            color="red",
            alpha=0.07,
            label="Zone: Epistemic Veto (OOD) (above min dist-threshold)",
        )

    # Draw horizontal threshold line(s)
    if dist_thr_by_class is not None:
        # One line per class threshold
        styles = {
            0: dict(color="#C0392B", linestyle="-", linewidth=2.0),  # dark red
            1: dict(color="#E74C3C", linestyle="--", linewidth=2.0),  # red dashed
        }
        for cls, thr in sorted(dist_thr_by_class.items(), key=lambda kv: kv[0]):
            if not np.isfinite(thr):
                continue
            st = styles.get(int(cls), dict(color="red", linestyle=":", linewidth=2.0))
            ax.axhline(
                thr,
                **st,
                label=f"Dist Thresh (class {cls}): {thr:.2f}",
            )
    else:
        if y_shade_line is not None and np.isfinite(y_shade_line):
            ax.axhline(
                y_shade_line,
                color="red",
                linestyle="-",
                linewidth=2,
                label=f"Dist Thresh: {y_shade_line:.2f}",
            )

    # --- 3. SCATTER PLOT ---

    mask_clear = np.char.find(final_labels.astype(str), "Clear") >= 0
    ax.scatter(
        prob_pos[mask_clear],
        dists[mask_clear],
        c="tab:blue",
        alpha=0.3,
        s=15,
        label="Automated (Clear)",
    )

    mask_amb = np.char.find(final_labels.astype(str), "Ambiguous") >= 0
    ax.scatter(
        prob_pos[mask_amb],
        dists[mask_amb],
        c="orange",
        marker="s",
        alpha=0.6,
        s=25,
        label="Defer: Ambiguous",
    )

    mask_epi = np.char.find(final_labels.astype(str), "Epistemic") >= 0
    ax.scatter(
        prob_pos[mask_epi],
        dists[mask_epi],
        c="red",
        marker="^",
        alpha=0.8,
        s=35,
        label="Defer: OOD/Complex",
    )

    mask_null = np.char.find(final_labels.astype(str), "Null") >= 0
    if np.any(mask_null):
        ax.scatter(
            prob_pos[mask_null],
            dists[mask_null],
            c="black",
            marker="x",
            alpha=0.5,
            s=30,
            label="Defer: Conflict",
        )

    # --- 4. FORMATTING ---
    ax.set_xlabel("Predicted Probability P(HIV+)")
    ax.set_ylabel("Mahalanobis Distance (Feature Normality)")
    ax.set_title("Hybrid Safety: Mondrian (Ambiguity) + Distance (OOD)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.percentile(dists, 99.5))

    # if y_shade_line is not None and np.isfinite(y_shade_line):
    #     ax.text(
    #         0.5,
    #         y_shade_line + (np.max(dists) * 0.02),
    #         # "GEOMETRIC VETO ZONE",
    #         color="red",
    #         ha="center",
    #         fontsize=9,
    #         fontweight="bold",
    #     )

    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Graph saved to {os.path.join(output_dir, filename)}")
    

def plot_complexity_analysis_mcp(
    probs,
    unc,
    categories,
    y_true,
    p_low: float,
    p_high: float,
    unc_th: float,
    output_dir: str,
    filename: str = "complexity_analysis_mcp.pdf",
):
    if probs is None or unc is None or categories is None:
        return

    prob_pos = probs[:, 1] if np.asarray(probs).ndim == 2 else np.asarray(probs)
    unc = np.asarray(unc, dtype=float)
    categories = np.asarray(categories)

    if len(prob_pos) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))

    high = unc > float(unc_th)
    low = ~high

    m = low & (categories == "Clear")
    plt.scatter(prob_pos[m], unc[m], c="gray", alpha=0.10, s=12, label="Clear (low unc)")

    m = low & (categories == "Defer: Ambiguous (Probabilistic)")
    plt.scatter(prob_pos[m], unc[m], c="blue", alpha=0.50, s=22, label="Gray Area (low unc)")

    m = high
    plt.scatter(prob_pos[m], unc[m], c="red", alpha=0.70, s=30, label="Above uncertainty threshold")

    plt.axhline(float(unc_th), color="red", linestyle="--", linewidth=1.5, label=f"Unc Thresh: {float(unc_th):.3f}")

    plt.axvline(p_high, color="tab:orange", linestyle="--", linewidth=1.8, label=f"MCP th0: p_pos ≤ {p_high:.3f}")
    plt.axvline(p_low, color="tab:purple", linestyle="--", linewidth=1.8, label=f"MCP th1: p_pos ≥ {p_low:.3f}")

    plt.xlabel("Probability P(positive)")
    plt.ylabel("Uncertainty")
    plt.title("Mondrian CP + Model Uncertainty (red = unc > threshold)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_calibration_curve(
    y_true,
    probs_calib,
    output_dir: str,
    probs_uncalib=None,
    filename: str = "calibration.pdf",
    n_bins: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))

    y_true = np.asarray(y_true)
    probs_calib = np.asarray(probs_calib, dtype=float)
    prob_true, prob_pred = calibration_curve(y_true, probs_calib, n_bins=int(n_bins))
    plt.plot(prob_pred, prob_true, marker="s", label="Calibrated", color="blue")

    if probs_uncalib is not None:
        probs_uncalib = np.asarray(probs_uncalib, dtype=float)
        prob_true_u, prob_pred_u = calibration_curve(y_true, probs_uncalib, n_bins=int(n_bins))
        plt.plot(prob_pred_u, prob_true_u, marker="o", label="Non-calibrated", color="tab:orange")

    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.title("Reliability Diagram")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Fraction Positive")
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_rejection_curves(y_true, y_probs, uncertainties, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    y_preds = (np.asarray(y_probs) > 0.5).astype(int)
    sorted_indices = np.argsort(uncertainties)[::-1]
    y_true_sorted = np.asarray(y_true)[sorted_indices]
    y_preds_sorted = y_preds[sorted_indices]

    n_samples = len(y_true)
    rates = np.linspace(0, 0.95, 50)
    accs = []
    sys_accs = []

    for rate in rates:
        n_drop = int(n_samples * rate)
        if n_drop >= n_samples:
            break

        y_true_keep = y_true_sorted[n_drop:]
        y_preds_keep = y_preds_sorted[n_drop:]
        model_acc = accuracy_score(y_true_keep, y_preds_keep)
        accs.append(model_acc)

        sys_acc = (model_acc * (1 - rate)) + (1.0 * rate)
        sys_accs.append(sys_acc)

    rates_acc = rates[: len(accs)]
    rates_sys = rates[: len(sys_accs)]
    auc_acc = float(np.trapz(accs, rates_acc)) if len(accs) > 1 else 0.0
    auc_sys = float(np.trapz(sys_accs, rates_sys)) if len(sys_accs) > 1 else 0.0

    plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    axes[0].plot(rates_acc, accs, linewidth=2.5, color="#2E86C1", label=f"Model (AUC: {auc_acc:.3f})")
    axes[0].set_title("Accuracy-Rejection Curve")
    axes[0].set_xlabel("Rejection Rate")
    axes[0].set_ylabel("Model Accuracy (Retained)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rates_sys, sys_accs, linewidth=2.5, color="#28B463", label=f"Hybrid System (AUC: {auc_sys:.3f})")
    base_acc = accs[0] if len(accs) else 0.0
    random_sys = base_acc * (1 - rates) + 1.0 * rates
    axes[1].plot(rates, random_sys, "k--", alpha=0.5, label="Random Baseline")
    axes[1].set_title("Hybrid System Performance")
    axes[1].set_xlabel("Deferral Rate")
    axes[1].set_ylabel("Total Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "rejection_curves.pdf"))
    plt.close()


def plot_uncertainty_diagnostics(y_true, y_probs, uncertainties, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    y_probs = np.asarray(y_probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    uncertainties = np.asarray(uncertainties, dtype=float)

    y_preds = (y_probs > 0.5).astype(int)

    fig = plt.figure(figsize=(18, 6), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.plot(prob_pred, prob_true, "s-", color="#2E86C1", linewidth=2)
    ax1.hist(y_probs, bins=20, range=(0, 1), alpha=0.2, color="gray", weights=np.ones_like(y_probs) / len(y_probs))
    ax1.set_title("Calibration & Sharpness")
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("True Positive Fraction")

    ax2 = fig.add_subplot(gs[0, 1])
    is_correct = (y_preds == y_true)
    sns = _maybe_import_seaborn()
    if sns is not None:
        sns.kdeplot(uncertainties[is_correct], fill=True, color="#28B463", label="Correct", ax=ax2)
        sns.kdeplot(uncertainties[~is_correct], fill=True, color="#E74C3C", label="Error", ax=ax2)
    else:
        ax2.hist(uncertainties[is_correct], bins=30, alpha=0.5, color="#28B463", label="Correct", density=True)
        ax2.hist(uncertainties[~is_correct], bins=30, alpha=0.5, color="#E74C3C", label="Error", density=True)
    ax2.set_title("Uncertainty Separation")
    ax2.set_xlabel("Uncertainty Score")
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    sorted_idx = np.argsort(y_probs)
    step = max(1, len(y_probs) // 100)
    idx_plot = sorted_idx[::step]
    ax3.plot(np.arange(len(idx_plot)), y_probs[idx_plot], color="#884EA0", linewidth=2)
    ax3.axhline(0.5, color="k", linestyle="--")
    ax3.set_title("Sorted Predictions")
    ax3.set_xlabel("Patient Rank")
    ax3.set_ylabel("Probability")

    plt.savefig(os.path.join(output_dir, "diagnostics.pdf"))
    plt.close()
