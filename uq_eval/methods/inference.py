"""Inference functions for MIL and standard HF models with optional MC-Dropout."""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from ..core.uncertainty import mc_stats_from_stacked_probs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_numpy_float32(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to float32 NumPy array on CPU."""
    return x.float().cpu().numpy().astype(np.float32)


def _cpu_numpy_int64(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to int64 NumPy array on CPU."""
    return x.long().cpu().numpy().astype(np.int64)


def _mc_group_size(iters: int, mc_batch_size: int) -> int:
    """Compute the number of MC samples to run in a single forward pass."""
    iters = int(max(1, iters))
    mc_batch_size = int(max(1, mc_batch_size))
    return int(min(iters, mc_batch_size))


def _repeat_mil_batch(batch: dict[str, torch.Tensor | list[int]], repeats: int):
    """Replicate a MIL batch along the batch dimension for grouped MC sampling."""
    if repeats <= 1:
        return batch["input_ids"], batch["attention_mask"], batch["num_chunks_per_doc"]

    input_ids = batch["input_ids"].repeat((int(repeats), 1))
    attention_mask = batch["attention_mask"].repeat((int(repeats), 1))
    num_chunks_per_doc = list(batch["num_chunks_per_doc"]) * int(repeats)
    return input_ids, attention_mask, num_chunks_per_doc


def _repeat_hf_inputs(inputs: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    """Replicate HF model inputs along the batch dimension for grouped MC sampling."""
    if repeats <= 1:
        return inputs
    return {k: v.repeat((int(repeats),) + (1,) * (v.ndim - 1)) for k, v in inputs.items()}


def _split_mc_tensor(x: torch.Tensor, repeats: int, batch_size: int) -> torch.Tensor:
    """Reshape a flat (repeats*B, ...) tensor into (repeats, B, ...)."""
    if repeats <= 1:
        return x.unsqueeze(0)
    new_shape = (int(repeats), int(batch_size)) + tuple(x.shape[1:])
    return x.reshape(new_shape)


# ---------------------------------------------------------------------------
# MIL inference with optional MC-Dropout
# ---------------------------------------------------------------------------

def run_inference_mil(
    model,
    dataloader,
    accelerator,
    mc_dropout: bool = False,
    n_iters: int = 1,
    mc_batch_size: int = 1,
    temperature: float = 1.0,
    unc_metric: str = "mi",
    *,
    return_embeddings: bool = False,
    return_model_uncertainty: bool = False,
):
    """Run inference with a MIL (Multiple Instance Learning) model.

    Supports MC-Dropout uncertainty estimation by running multiple stochastic
    forward passes with dropout enabled and aggregating the predictions.

    Args:
        model: MIL classifier (e.g. AttentionMILClassifier).
        dataloader: DataLoader yielding batches with input_ids, attention_mask,
            num_chunks_per_doc, and labels.
        accelerator: HuggingFace Accelerator for distributed gathering.
        mc_dropout: If True, enable dropout during inference for MC sampling.
        n_iters: Number of stochastic forward passes (1 = deterministic).
        mc_batch_size: How many MC samples to group in a single forward pass.
        temperature: Softmax temperature for calibration.
        unc_metric: Uncertainty metric — 'mi' (mutual information), 'entropy',
            or 'std_pos' (std of positive-class probability).
        return_embeddings: If True, also return document embeddings (N, D).
        return_model_uncertainty: If True, use model-internal uncertainty
            (e.g. from a learned variance head) instead of MC-based metrics.

    Returns:
        Tuple of (probs, uncertainty, labels, entropy, piw[, embeddings]).
        All arrays are NumPy float32/int64 on CPU.
    """
    model.eval()
    if mc_dropout:
        # Enable dropout layers for MC sampling while keeping the rest in eval mode
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    all_probs, all_unc, all_labels = [], [], []
    all_entropy, all_piw = [], []
    all_embs = []  # (N,D) if requested

    progress = tqdm(
        total=len(dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Infer (MC={mc_dropout}, iters={n_iters}, unc={unc_metric})",
    )

    T = float(max(temperature, 1e-6))
    iters = int(max(1, n_iters))
    mc_group = _mc_group_size(iters, mc_batch_size if mc_dropout else 1)

    for batch in dataloader:
        batch_probs_iters = []
        batch_unc_iters = [] if return_model_uncertainty else None
        batch_emb_iters = [] if return_embeddings else None
        batch_size = int(batch["labels"].shape[0])

        with torch.no_grad():
            for start in range(0, iters, mc_group):
                group = int(min(mc_group, iters - start))
                input_ids, attention_mask, num_chunks_per_doc = _repeat_mil_batch(batch, group)
                out = model(
                    input_ids,
                    attention_mask,
                    num_chunks_per_doc,
                    return_features=bool(return_embeddings),
                    return_uncertainty=bool(return_model_uncertainty),
                )

                if not isinstance(out, (tuple, list)) or len(out) < 1:
                    raise ValueError("MIL model must return a tuple/list like (logits, ...).")

                logits = out[0]
                logits = _split_mc_tensor(logits, group, batch_size)

                if return_model_uncertainty:
                    # AttentionMILClassifier returns either:
                    #   - (logits, attn_weights, unc)
                    #   - (logits, attn_weights, features, unc) if return_features=True
                    if len(out) < 3:
                        raise ValueError(
                            "return_model_uncertainty=True, but model did not return uncertainty. "
                            "Expected (logits, attn_weights, unc) or (logits, attn_weights, features, unc)."
                        )
                    unc_t = out[-1]
                    if (not torch.is_tensor(unc_t)) or unc_t.ndim != 1:
                        raise ValueError(
                            f"Expected uncertainty tensor with shape (B,), got {type(unc_t)} shape={getattr(unc_t, 'shape', None)}"
                        )
                    batch_unc_iters.append(_split_mc_tensor(unc_t, group, batch_size))

                if return_embeddings:
                    # AttentionMILClassifier returns: (logits, batch_attention_weights, doc_embeddings)
                    if len(out) < 3:
                        raise ValueError(
                            "return_embeddings=True, but model did not return doc_embeddings. "
                            "Expected (logits, attn_weights, doc_embeddings) from AttentionMILClassifier."
                        )
                    doc_emb = out[2]
                    if (not torch.is_tensor(doc_emb)) or doc_emb.ndim != 2:
                        raise ValueError(f"Expected doc_embeddings tensor with shape (B,D), got {type(doc_emb)} shape={getattr(doc_emb, 'shape', None)}")
                    batch_emb_iters.append(_split_mc_tensor(doc_emb, group, batch_size))

                probs = torch.softmax(logits / T, dim=2)
                batch_probs_iters.append(probs)

        # Aggregate across MC samples: (S, B, C) -> stats
        stacked_probs = torch.cat(batch_probs_iters, dim=0)  # (S,B,C)
        stats = mc_stats_from_stacked_probs(stacked_probs)
        mean_probs = stats["mean_probs"]

        if return_model_uncertainty:
            stacked_unc = torch.cat(batch_unc_iters, dim=0)  # (S,B)
            unc = stacked_unc.mean(dim=0)

            # Entropy from mean probs (works for both S=1 and S>1)
            p = mean_probs.clamp(min=1e-12)
            ent = -(p * p.log()).sum(dim=1)
            piw = torch.zeros_like(ent)
        else:
            if iters <= 1:
                unc = torch.zeros(mean_probs.size(0), device=mean_probs.device, dtype=mean_probs.dtype)
                ent = torch.zeros_like(unc)
                piw = torch.zeros_like(unc)
            else:
                ent = stats["entropy"]
                piw = stats["piw"]
                if unc_metric == "std_pos":
                    unc = stats["std_pos"]
                elif unc_metric == "entropy":
                    unc = ent
                elif unc_metric == "mi":
                    unc = stats["mi"]
                else:
                    raise ValueError(f"Unknown unc_metric={unc_metric}")

        all_probs.append(accelerator.gather_for_metrics(mean_probs).cpu())
        all_unc.append(accelerator.gather_for_metrics(unc).cpu())
        all_entropy.append(accelerator.gather_for_metrics(ent).cpu())
        all_piw.append(accelerator.gather_for_metrics(piw).cpu())
        all_labels.append(accelerator.gather_for_metrics(batch["labels"]).cpu())

        if return_embeddings:
            stacked_emb = torch.cat(batch_emb_iters, dim=0)  # (S,B,D)
            mean_emb = stacked_emb.mean(dim=0)               # (B,D)
            all_embs.append(accelerator.gather_for_metrics(mean_emb).cpu())

        progress.update(1)

    progress.close()

    probs_np = _cpu_numpy_float32(torch.cat(all_probs, dim=0))
    unc_np = _cpu_numpy_float32(torch.cat(all_unc, dim=0))
    labels_np = _cpu_numpy_int64(torch.cat(all_labels, dim=0))
    ent_np = _cpu_numpy_float32(torch.cat(all_entropy, dim=0))
    piw_np = _cpu_numpy_float32(torch.cat(all_piw, dim=0))

    if return_embeddings:
        emb_np = _cpu_numpy_float32(torch.cat(all_embs, dim=0))
        return probs_np, unc_np, labels_np, ent_np, piw_np, emb_np

    return probs_np, unc_np, labels_np, ent_np, piw_np


# ---------------------------------------------------------------------------
# Standard HuggingFace inference with optional MC-Dropout
# ---------------------------------------------------------------------------

def run_inference_hf(
    model,
    dataloader,
    accelerator,
    mc_dropout: bool = False,
    n_iters: int = 1,
    mc_batch_size: int = 1,
    temperature: float = 1.0,
    unc_metric: str = "mi",
    *,
    return_embeddings: bool = False,
):
    """Run inference with a HuggingFace sequence classification model.

    Supports MC-Dropout uncertainty estimation by running multiple stochastic
    forward passes with dropout enabled and aggregating the predictions.

    Args:
        model: HuggingFace model with a classification head (e.g.
            AutoModelForSequenceClassification).
        dataloader: DataLoader yielding batches with input_ids, attention_mask,
            and optionally labels.
        accelerator: HuggingFace Accelerator for distributed gathering.
        mc_dropout: If True, enable dropout during inference for MC sampling.
        n_iters: Number of stochastic forward passes (1 = deterministic).
        mc_batch_size: How many MC samples to group in a single forward pass.
        temperature: Softmax temperature for calibration.
        unc_metric: Uncertainty metric — 'mi', 'entropy', or 'std_pos'.
        return_embeddings: If True, also return CLS embeddings (N, D).

    Returns:
        Tuple of (probs, uncertainty, labels, entropy, piw[, embeddings]).
        All arrays are NumPy float32/int64 on CPU.
    """
    model.eval()
    if mc_dropout:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    all_probs, all_unc, all_labels = [], [], []
    all_entropy, all_piw = [], []
    all_embs = []

    progress = tqdm(
        total=len(dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Infer HF (MC={mc_dropout}, iters={n_iters}, unc={unc_metric})",
    )

    T = float(max(temperature, 1e-6))
    iters = int(max(1, n_iters))
    mc_group = _mc_group_size(iters, mc_batch_size if mc_dropout else 1)

    for batch in dataloader:
        batch_probs_iters = []
        batch_emb_iters = [] if return_embeddings else None
        batch_size = int(batch["input_ids"].shape[0])

        labels = batch.get("labels")
        inputs = {k: v for k, v in batch.items() if k != "labels"}

        with torch.no_grad():
            for start in range(0, iters, mc_group):
                group = int(min(mc_group, iters - start))
                repeated_inputs = _repeat_hf_inputs(inputs, group)
                out = model(**repeated_inputs, output_hidden_states=bool(return_embeddings), return_dict=True)
                logits = _split_mc_tensor(out.logits, group, batch_size)
                probs = torch.softmax(logits / T, dim=2)
                batch_probs_iters.append(probs)

                if return_embeddings:
                    hs = getattr(out, "hidden_states", None)
                    if hs is not None and len(hs) > 0 and torch.is_tensor(hs[-1]):
                        last = hs[-1]
                    else:
                        last = getattr(out, "last_hidden_state", None)
                        if last is None:
                            raise ValueError(
                                "return_embeddings=True but model output has neither hidden_states nor last_hidden_state"
                            )
                        cls = last[:, 0, :]
                        batch_emb_iters.append(_split_mc_tensor(cls, group, batch_size))

        # Aggregate across MC samples
        stacked_probs = torch.cat(batch_probs_iters, dim=0)
        stats = mc_stats_from_stacked_probs(stacked_probs)
        mean_probs = stats["mean_probs"]

        if iters <= 1:
            unc = torch.zeros(mean_probs.size(0), device=mean_probs.device, dtype=mean_probs.dtype)
            ent = torch.zeros_like(unc)
            piw = torch.zeros_like(unc)
        else:
            ent = stats["entropy"]
            piw = stats["piw"]
            if unc_metric == "std_pos":
                unc = stats["std_pos"]
            elif unc_metric == "entropy":
                unc = ent
            elif unc_metric == "mi":
                unc = stats["mi"]
            else:
                raise ValueError(f"Unknown unc_metric={unc_metric}")

        all_probs.append(accelerator.gather_for_metrics(mean_probs).cpu())
        all_unc.append(accelerator.gather_for_metrics(unc).cpu())
        all_entropy.append(accelerator.gather_for_metrics(ent).cpu())
        all_piw.append(accelerator.gather_for_metrics(piw).cpu())
        if labels is not None:
            all_labels.append(accelerator.gather_for_metrics(labels).cpu())

        if return_embeddings:
            stacked_emb = torch.cat(batch_emb_iters, dim=0)  # (S,B,D)
            mean_emb = stacked_emb.mean(dim=0)               # (B,D)
            all_embs.append(accelerator.gather_for_metrics(mean_emb).cpu())

        progress.update(1)

    progress.close()

    probs_np = _cpu_numpy_float32(torch.cat(all_probs, dim=0))
    unc_np = _cpu_numpy_float32(torch.cat(all_unc, dim=0))
    labels_np = _cpu_numpy_int64(torch.cat(all_labels, dim=0)) if all_labels else np.zeros((0,), dtype=np.int64)
    ent_np = _cpu_numpy_float32(torch.cat(all_entropy, dim=0))
    piw_np = _cpu_numpy_float32(torch.cat(all_piw, dim=0))

    if return_embeddings:
        emb_np = _cpu_numpy_float32(torch.cat(all_embs, dim=0))
        return probs_np, unc_np, labels_np, ent_np, piw_np, emb_np

    return probs_np, unc_np, labels_np, ent_np, piw_np
