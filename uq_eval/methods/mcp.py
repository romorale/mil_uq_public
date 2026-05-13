from __future__ import annotations

import numpy as np


class MondrianCP:
    """Mondrian Conformal Prediction for class-conditional coverage."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.thresholds: dict[int, float] = {}

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray) -> None:
        val_probs = np.asarray(val_probs, dtype=float)
        val_labels = np.asarray(val_labels, dtype=int)

        for c in np.unique(val_labels):
            idx = np.where(val_labels == c)[0]
            if len(idx) == 0:
                continue
            scores = 1.0 - val_probs[idx, c]
            q = np.quantile(scores, 1.0 - self.alpha)
            self.thresholds[int(c)] = float(q)

    def predict(self, test_probs: np.ndarray):
        test_probs = np.asarray(test_probs, dtype=float)
        sets = []
        categories = []

        for p in test_probs:
            s = []
            for c, th in self.thresholds.items():
                if (1.0 - p[c]) <= th:
                    s.append(int(c))

            if len(s) == 1:
                categories.append("Clear")
            elif len(s) > 1:
                categories.append("Gray Area (Uncertain)")
            else:
                categories.append("Null (Empty)")

            sets.append(s)

        return sets, np.asarray(categories)


def apply_mcp(
    mcp_engine: MondrianCP,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    unc_th: float,
):
    """Pattern A (same as eval_GPT.py): auto-predict if |S(x)|==1, else defer; and tag high-unc as Complex."""
    sets, _ = mcp_engine.predict(probs)

    uncertainty = np.asarray(uncertainty, dtype=float)
    preds_trinary = np.ones(len(sets), dtype=int)
    categories = []

    for i, s in enumerate(sets):
        u = float(uncertainty[i])

        if len(s) == 1:
            preds_trinary[i] = 0 if s[0] == 0 else 2
            categories.append("Clear")
        elif len(s) > 1:
            if u >= float(unc_th):
                preds_trinary[i] = 1
                categories.append("Complex (High Unc)")
            else:
                preds_trinary[i] = 1
                categories.append("Gray Area (Uncertain)")
        else:
            preds_trinary[i] = 1
            categories.append("Null (Empty)")

    return preds_trinary, np.asarray(categories), sets

