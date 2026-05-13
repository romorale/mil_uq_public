"""HuggingFace (standard transformer) dataset and collate utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .text import clean_text


def _label_to_int_series(labels: pd.Series) -> np.ndarray:
    """Convert a label column to int64.  Assumes labels are already integers."""
    if labels.empty:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(labels.tolist(), dtype=np.int64)


def infer_text_column(df: pd.DataFrame) -> str:
    """Return the name of the first text column found in *df*."""
    for c in ("text", "notes"):
        if c in df.columns:
            return c
    raise KeyError("No text column found (expected 'text' or 'notes').")


def texts_labels_from_df(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Extract (texts, labels) from a DataFrame with 'text'/'notes' and 'label' columns."""
    if "label" not in df.columns:
        raise KeyError("Missing required column: 'label'")

    text_col = infer_text_column(df)

    texts: list[str] = []
    for x in df[text_col].tolist():
        if isinstance(x, list):
            texts.append("\n".join([str(t) for t in x if str(t).strip()]))
        else:
            texts.append("" if x is None else str(x))

    labels = _label_to_int_series(df["label"])
    return texts, labels


def hf_safe_max_length(tokenizer, model_config, requested_max_length: int) -> int:
    """Determine a safe max_length respecting tokenizer and model limits."""
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max is None or int(tok_max) > 1_000_000:
        tok_max = None

    cfg_max = getattr(model_config, "max_position_embeddings", None)

    candidates = [int(requested_max_length)]
    if tok_max is not None:
        candidates.append(int(tok_max))
    if cfg_max is not None:
        candidates.append(int(cfg_max))

    safe = int(min(candidates))
    return max(8, safe)


def hf_add_longformer_gam(enc: dict[str, torch.Tensor], model_config) -> dict[str, torch.Tensor]:
    """Add global_attention_mask for Longformer models."""
    model_type = str(getattr(model_config, "model_type", ""))
    if model_type != "longformer":
        return enc

    input_ids = enc.get("input_ids")
    if input_ids is None:
        return enc

    gam = torch.zeros_like(input_ids)
    gam[:, 0] = 1
    enc["global_attention_mask"] = gam
    return enc


class TextDataset(Dataset):
    """Simple text classification dataset."""

    def __init__(self, texts: list[str], labels: np.ndarray):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i: int):
        return clean_text(self.texts[i]), int(self.labels[i])


def make_hf_collate_fn(tokenizer, model_config, max_length: int):
    """Create a collate function that tokenizes text batches for HF models."""
    safe_max_len = hf_safe_max_length(tokenizer, model_config, int(max_length))

    def collate_fn(batch: list[tuple[str, int]]):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=int(safe_max_len),
            return_tensors="pt",
        )
        enc = hf_add_longformer_gam(enc, model_config)
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate_fn
