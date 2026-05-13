from __future__ import annotations

import math
import numpy as np
import torch


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)


def mc_stats_from_stacked_probs(stacked_probs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute mean probs + uncertainty stats from (T,B,C) tensor."""
    mean_probs = stacked_probs.mean(dim=0)

    if stacked_probs.size(-1) >= 2:
        prob_pos = stacked_probs[:, :, 1]
    else:
        prob_pos = stacked_probs[:, :, 0]

    std_pos = prob_pos.std(dim=0, correction=1) if stacked_probs.size(0) > 1 else torch.zeros(prob_pos.size(1), device=prob_pos.device)

    ent = entropy_from_probs(mean_probs)
    exp_ent = entropy_from_probs(stacked_probs).mean(dim=0)
    mi = ent - exp_ent

    lower = torch.quantile(prob_pos.float(), 0.05, dim=0)
    upper = torch.quantile(prob_pos.float(), 0.95, dim=0)
    piw = upper - lower

    return {
        "mean_probs": mean_probs,
        "std_pos": std_pos,
        "entropy": ent,
        "mi": mi,
        "piw": piw,
    }


def entropy_np(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return (-(p * np.log(p)).sum(axis=-1)).astype(np.float32)


def binary_entropy_from_p(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))).astype(np.float32)


def lg_predictive_mean_prob(mu: torch.Tensor, varz: torch.Tensor) -> torch.Tensor:
    """Logistic-Gaussian approx used by the original Laplace code."""
    kappa = torch.sqrt(1.0 + (math.pi / 8.0) * varz)
    return torch.sigmoid(mu / kappa)
