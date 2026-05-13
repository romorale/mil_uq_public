"""Cross-validation ensemble and out-of-fold (OOF) collection utilities."""
from __future__ import annotations

import os
import re
import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from ..core.uncertainty import entropy_np


def _dist_info(accelerator) -> tuple[int, int, bool]:
    """Return (rank, world_size, dist_initialized)."""
    world_size = int(getattr(accelerator, "num_processes", 1) or 1)
    rank = int(getattr(accelerator, "process_index", 0) or 0)
    ok = bool(world_size > 1 and dist.is_available() and dist.is_initialized())
    return rank, world_size, ok


def _all_gather_object(accelerator, obj):
    """all_gather_object wrapper that works when dist is not initialized."""
    rank, world_size, ok = _dist_info(accelerator)
    if not ok:
        return [obj]
    out = [None for _ in range(world_size)]
    dist.all_gather_object(out, obj)
    return out


def _rebuild_full_loader(dataloader, *, collate_fn=None, batch_size: int | None = None):
    """Rebuild a non-sharded DataLoader so every process sees the full dataset in the same order."""
    from torch.utils.data import DataLoader

    ds = getattr(dataloader, "dataset", None)
    if ds is None:
        raise ValueError("dataloader has no dataset; cannot rebuild full loader")

    if collate_fn is None:
        collate_fn = getattr(dataloader, "collate_fn", None)
    if collate_fn is None:
        raise ValueError("collate_fn not provided and not found on dataloader")

    bs = int(batch_size or getattr(dataloader, "batch_size", 8) or 8)
    num_workers = int(getattr(dataloader, "num_workers", 0) or 0)
    pin_memory = bool(getattr(dataloader, "pin_memory", False))

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def resolve_cv_ensemble_paths(args) -> list[str]:
    """Resolve list of model directories for CV ensemble.

    Supports explicit ``--cv_ensemble_paths`` (comma-separated) or automatic
    discovery via ``--cv_ensemble_root`` / ``--cv_ensemble_n_models``.
    """
    if getattr(args, "cv_ensemble_paths", None):
        paths = [p.strip() for p in str(args.cv_ensemble_paths).split(",") if p.strip()]
        if len(paths) == 0:
            raise ValueError("--cv_ensemble_paths provided but empty")
        return paths

    root = getattr(args, "cv_ensemble_root", None)
    n = int(getattr(args, "cv_ensemble_n_models", 10))

    if root is None:
        mp = os.path.abspath(args.model_path)
        m = re.search(r"(.*)/MODEL_\d+/", mp)
        if m:
            root = m.group(1)
        else:
            root = os.path.dirname(mp)

    paths = [os.path.join(root, f"MODEL_{i}") for i in range(n)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Some ensemble model paths do not exist. "
            "Provide --cv_ensemble_paths or fix --cv_ensemble_root.\nMissing:\n" + "\n".join(missing[:10])
        )
    return paths


def cv_ensemble_mean_logits(val_loader, accelerator, model_paths: list[str], model_loader) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean logits across ensemble members over *val_loader*."""
    sum_logits = None
    labels_all = None

    for mi, mp in enumerate(model_paths):
        if accelerator.is_local_main_process:
            print(f"  [cv_ens] loading {mi+1}/{len(model_paths)}: {mp}")

        model = model_loader(mp)
        model.to(accelerator.device)
        model.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, disable=not accelerator.is_local_main_process, desc=f"CV logits {mi+1}/{len(model_paths)}"):
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["num_chunks_per_doc"])
                all_logits.append(accelerator.gather_for_metrics(logits).cpu())
                if labels_all is None:
                    all_labels.append(accelerator.gather_for_metrics(batch["labels"]).cpu())

        logits_np = torch.cat(all_logits, dim=0).numpy()
        sum_logits = logits_np if sum_logits is None else (sum_logits + logits_np)

        if labels_all is None:
            labels_all = torch.cat(all_labels, dim=0).numpy().astype(np.int64)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_logits = (sum_logits / float(len(model_paths))).astype(np.float32)
    return mean_logits, labels_all


def cv_ensemble_mean_embeddings(
    dataloader,
    accelerator,
    model_paths: list[str],
    model_loader,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean document embeddings across ensemble members (MIL models).

    Returns:
      mean_embs: (N, D) float32
      labels:    (N,) int64
    """
    sum_embs = None
    labels_all = None

    for mi, mp in enumerate(model_paths):
        if accelerator.is_local_main_process:
            print(f"  [cv_ens] emb member {mi+1}/{len(model_paths)}: {os.path.basename(mp)}")

        model = model_loader(mp)
        model.to(accelerator.device)
        model.eval()

        all_embs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                disable=not accelerator.is_local_main_process,
                desc=f"CV embs {mi+1}/{len(model_paths)}",
            ):
                out = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["num_chunks_per_doc"],
                    return_features=True,
                )
                if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                    raise ValueError(
                        "Expected MIL model(..., return_features=True) -> (logits, attn_weights, doc_embeddings)"
                    )
                doc_emb = out[2]
                all_embs.append(accelerator.gather_for_metrics(doc_emb).cpu())
                if labels_all is None:
                    all_labels.append(accelerator.gather_for_metrics(batch["labels"]).cpu())

        embs_np = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
        sum_embs = embs_np if sum_embs is None else (sum_embs + embs_np)

        if labels_all is None:
            labels_all = torch.cat(all_labels, dim=0).numpy().astype(np.int64)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_embs = (sum_embs / float(len(model_paths))).astype(np.float32)
    return mean_embs, labels_all


def cv_ensemble_probs_unc2(
    dataloader,
    accelerator,
    model_paths: list[str],
    model_loader,
    temperature: float,
    unc_metric: str,
    *,
    return_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble mean probabilities, uncertainty, entropy, and prediction-interval width.

    If *return_embeddings* is True, also returns mean doc embeddings as the last element.
    """
    all_members_probs: list[np.ndarray] = []
    all_members_embs: list[np.ndarray] = []
    labels_all = None

    T = float(max(temperature, 1e-6))
    unc_metric = str(unc_metric)

    for mi, mp in enumerate(model_paths):
        if accelerator.is_local_main_process:
            print(f"  [cv_ens] member {mi+1}/{len(model_paths)}: {os.path.basename(mp)}")

        model = model_loader(mp)
        model.to(accelerator.device)
        model.eval()

        member_probs = []
        member_embs = []
        member_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"Member {mi+1}"):
                out = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["num_chunks_per_doc"],
                    return_features=bool(return_embeddings),
                )
                if not isinstance(out, (tuple, list)) or len(out) < 1:
                    raise ValueError("MIL model must return a tuple/list like (logits, ...).")
                logits = out[0]
                probs = torch.softmax(logits / T, dim=1)

                member_probs.append(accelerator.gather_for_metrics(probs).cpu())
                if return_embeddings:
                    if len(out) < 3:
                        raise ValueError(
                            "return_embeddings=True requires MIL model to return (logits, attn_weights, doc_embeddings)"
                        )
                    doc_emb = out[2]
                    member_embs.append(accelerator.gather_for_metrics(doc_emb).cpu())
                if labels_all is None:
                    member_labels.append(accelerator.gather_for_metrics(batch["labels"]).cpu())

        all_members_probs.append(torch.cat(member_probs, dim=0).numpy().astype(np.float32))
        if return_embeddings:
            all_members_embs.append(torch.cat(member_embs, dim=0).numpy().astype(np.float32))
        if labels_all is None:
            labels_all = torch.cat(member_labels, dim=0).numpy().astype(np.int64)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stacked_probs = np.stack(all_members_probs, axis=1)  # (N, M, C)
    mean_probs = np.mean(stacked_probs, axis=1).astype(np.float32)  # (N, C)

    eps = 1e-8
    p = np.clip(mean_probs, eps, 1.0)
    entropy = (-(p * np.log(p)).sum(axis=-1)).astype(np.float32)

    p_pos_members = stacked_probs[:, :, 1]
    lower = np.percentile(p_pos_members, 5, axis=1).astype(np.float32)
    upper = np.percentile(p_pos_members, 95, axis=1).astype(np.float32)
    piw = (upper - lower).astype(np.float32)

    if unc_metric == "std_pos":
        unc = np.std(p_pos_members, axis=1).astype(np.float32)
    elif unc_metric == "entropy":
        unc = entropy
    elif unc_metric == "mi":
        mean_H = np.mean(-(stacked_probs * np.log(stacked_probs + eps)).sum(axis=-1), axis=1)
        unc = (entropy - mean_H).astype(np.float32)
    else:
        raise ValueError(f"Unknown unc_metric={unc_metric}")

    if return_embeddings:
        stacked_embs = np.stack(all_members_embs, axis=1)  # (N,M,D)
        mean_embs = np.mean(stacked_embs, axis=1).astype(np.float32)  # (N,D)
        return mean_probs, unc, labels_all, entropy, piw, mean_embs

    return mean_probs, unc, labels_all, entropy, piw


def compute_unc_from_probs_np(probs: np.ndarray, unc_metric: str) -> np.ndarray:
    """Compute OOF uncertainty from a single predictive distribution per example."""
    unc_metric = str(unc_metric)
    if unc_metric == "entropy":
        return entropy_np(probs).astype(np.float32)
    if unc_metric == "std_pos":
        p = probs[:, 1].astype(np.float32)
        return np.sqrt(np.clip(p * (1.0 - p), 0.0, None)).astype(np.float32)
    raise ValueError(f"OOF supports unc_metric in {{entropy,std_pos}}. Got: {unc_metric}")


# ---------------------------------------------------------------------------
# OOF collection – MIL models
# ---------------------------------------------------------------------------

def cv_oof_collect_logits_labels(
    args,
    accelerator,
    tokenizer,
    collate,
    model_paths: list[str],
    load_split_df,
    model_loader,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-leaky OOF logit collection: fold *i* uses MODEL_i on fold_i_val.json (main process only)."""
    assert accelerator.is_main_process, "cv_oof_collect_logits_labels must run on main process only"

    from torch.utils.data import DataLoader
    from ..data.mil_data import MILDataset

    all_logits = []
    all_labels = []

    device = accelerator.device

    def load_ds(df):
        labels = df["label"].astype(int).tolist()
        texts = df["text"].tolist() if "text" in df.columns else df["notes"].tolist()
        return MILDataset(texts, labels, tokenizer)

    for i, mp in enumerate(model_paths):
        val_json = os.path.join(args.input_dir, f"fold_{i}_val.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Missing OOF val split: {val_json}")

        if accelerator.is_local_main_process:
            print(f"  [oof] fold {i}: model={mp}  val={val_json}")

        import pandas as pd

        val_split_df = pd.read_json(val_json, lines=True)
        val_final = load_split_df(val_split_df)

        ds = load_ds(val_final)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, shuffle=False)

        model_i = model_loader(mp)
        model_i.to(device)
        model_i.eval()

        with torch.no_grad():
            for batch in tqdm(loader, disable=not accelerator.is_local_main_process, desc=f"OOF logits fold {i}"):
                logits, _ = model_i(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["num_chunks_per_doc"],
                )
                all_logits.append(logits.detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_logits = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    oof_labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return oof_logits, oof_labels


def cv_oof_collect_logits_embeddings_labels(
    args,
    accelerator,
    tokenizer,
    collate,
    model_paths: list[str],
    load_split_df,
    model_loader,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Distributed OOF collection (logits + embeddings) in one pass.

    Each process handles a subset of folds, then results are gathered on main and
    concatenated in fold order to keep a stable ordering.

    Returns ``(oof_logits, oof_embeddings, oof_labels)`` on main process,
    otherwise ``(None, None, None)``.
    """
    from torch.utils.data import DataLoader
    import pandas as pd
    from ..data.mil_data import MILDataset

    rank, world_size, _ = _dist_info(accelerator)
    device = accelerator.device

    def load_ds(df):
        labels = df["label"].astype(int).tolist()
        texts = df["text"].tolist() if "text" in df.columns else df["notes"].tolist()
        return MILDataset(texts, labels, tokenizer)

    local: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i, mp in enumerate(model_paths):
        if (i % world_size) != rank:
            continue

        val_json = os.path.join(args.input_dir, f"fold_{i}_val.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Missing OOF val split: {val_json}")

        if accelerator.is_local_main_process:
            print(f"  [oof] fold {i}: model={mp}  val={val_json}")

        val_split_df = pd.read_json(val_json, lines=True)
        val_final = load_split_df(val_split_df)

        ds = load_ds(val_final)
        loader = DataLoader(ds, batch_size=int(args.batch_size), collate_fn=collate, shuffle=False)

        model_i = model_loader(mp)
        model_i.to(device)
        model_i.eval()

        fold_logits = []
        fold_embs = []
        fold_labels = []
        with torch.no_grad():
            for batch in tqdm(
                loader,
                disable=not accelerator.is_local_main_process,
                desc=f"OOF logits+emb fold {i}",
            ):
                out = model_i(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["num_chunks_per_doc"],
                    return_features=True,
                )
                if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                    raise ValueError(
                        "Expected MIL model(..., return_features=True) -> (logits, attn_weights, doc_embeddings)"
                    )
                logits = out[0]
                doc_emb = out[2]
                fold_logits.append(logits.detach().cpu())
                fold_embs.append(doc_emb.detach().cpu())
                fold_labels.append(batch["labels"].detach().cpu())

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local.append(
            (
                int(i),
                torch.cat(fold_logits, dim=0),
                torch.cat(fold_embs, dim=0),
                torch.cat(fold_labels, dim=0),
            )
        )

    gathered = _all_gather_object(accelerator, local)
    if not accelerator.is_main_process:
        return None, None, None

    merged: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for part in gathered:
        if not part:
            continue
        merged.extend(part)
    merged.sort(key=lambda t: t[0])

    all_logits = [t[1] for t in merged]
    all_embs = [t[2] for t in merged]
    all_labels = [t[3] for t in merged]

    oof_logits = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    oof_embs = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
    oof_labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return oof_logits, oof_embs, oof_labels


def cv_oof_collect_logits_embeddings_labels_fold_ids(
    args,
    accelerator,
    tokenizer,
    collate,
    model_paths: list[str],
    load_split_df,
    model_loader,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Distributed OOF collection (logits + embeddings) with per-sample fold IDs.

    Returns ``(oof_logits, oof_embeddings, oof_labels, oof_fold_ids)`` on main process,
    otherwise ``(None, None, None, None)``.
    """
    from torch.utils.data import DataLoader
    import pandas as pd
    from ..data.mil_data import MILDataset

    rank, world_size, _ = _dist_info(accelerator)
    device = accelerator.device

    def load_ds(df):
        labels = df["label"].astype(int).tolist()
        texts = df["text"].tolist() if "text" in df.columns else df["notes"].tolist()
        return MILDataset(texts, labels, tokenizer)

    local: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i, mp in enumerate(model_paths):
        if (i % world_size) != rank:
            continue

        val_json = os.path.join(args.input_dir, f"fold_{i}_val.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Missing OOF val split: {val_json}")

        if accelerator.is_local_main_process:
            print(f"  [oof] fold {i}: model={mp}  val={val_json}")

        val_split_df = pd.read_json(val_json, lines=True)
        val_final = load_split_df(val_split_df)

        ds = load_ds(val_final)
        loader = DataLoader(ds, batch_size=int(args.batch_size), collate_fn=collate, shuffle=False)

        model_i = model_loader(mp)
        model_i.to(device)
        model_i.eval()

        fold_logits = []
        fold_embs = []
        fold_labels = []
        with torch.no_grad():
            for batch in tqdm(
                loader,
                disable=not accelerator.is_local_main_process,
                desc=f"OOF logits+emb fold {i}",
            ):
                out = model_i(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["num_chunks_per_doc"],
                    return_features=True,
                )
                if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                    raise ValueError(
                        "Expected MIL model(..., return_features=True) -> (logits, attn_weights, doc_embeddings)"
                    )
                logits = out[0]
                doc_emb = out[2]
                fold_logits.append(logits.detach().cpu())
                fold_embs.append(doc_emb.detach().cpu())
                fold_labels.append(batch["labels"].detach().cpu())

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_logits_t = torch.cat(fold_logits, dim=0)
        fold_embs_t = torch.cat(fold_embs, dim=0)
        fold_labels_t = torch.cat(fold_labels, dim=0)
        fold_ids_t = torch.full((int(fold_labels_t.shape[0]),), int(i), dtype=torch.int64)

        local.append((int(i), fold_logits_t, fold_embs_t, fold_labels_t, fold_ids_t))

    gathered = _all_gather_object(accelerator, local)
    if not accelerator.is_main_process:
        return None, None, None, None

    merged: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for part in gathered:
        if not part:
            continue
        merged.extend(part)
    merged.sort(key=lambda t: t[0])

    all_logits = [t[1] for t in merged]
    all_embs = [t[2] for t in merged]
    all_labels = [t[3] for t in merged]
    all_fold_ids = [t[4] for t in merged]

    oof_logits = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    oof_embs = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
    oof_labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    oof_fold_ids = torch.cat(all_fold_ids, dim=0).numpy().astype(np.int64)
    return oof_logits, oof_embs, oof_labels, oof_fold_ids


def cv_oof_collect_embeddings_labels(
    args,
    accelerator,
    tokenizer,
    collate,
    model_paths: list[str],
    load_split_df,
    model_loader,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-leaky OOF embedding collection (main process only).

    Returns:
      oof_embeddings: (N, D) float32
      oof_labels:     (N,) int64
    """
    assert accelerator.is_main_process, "cv_oof_collect_embeddings_labels must run on main process only"

    from torch.utils.data import DataLoader
    import pandas as pd
    from ..data.mil_data import MILDataset

    all_embs = []
    all_labels = []

    device = accelerator.device

    def load_ds(df):
        labels = df["label"].astype(int).tolist()
        texts = df["text"].tolist() if "text" in df.columns else df["notes"].tolist()
        return MILDataset(texts, labels, tokenizer)

    for i, mp in enumerate(model_paths):
        val_json = os.path.join(args.input_dir, f"fold_{i}_val.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Missing OOF val split: {val_json}")

        if accelerator.is_local_main_process:
            print(f"  [oof_emb] fold {i}: model={mp}  val={val_json}")

        val_split_df = pd.read_json(val_json, lines=True)
        val_final = load_split_df(val_split_df)

        ds = load_ds(val_final)
        loader = DataLoader(ds, batch_size=int(args.batch_size), collate_fn=collate, shuffle=False)

        model_i = model_loader(mp)
        model_i.to(device)
        model_i.eval()

        with torch.no_grad():
            for batch in tqdm(loader, disable=not accelerator.is_local_main_process, desc=f"OOF emb fold {i}"):
                out = model_i(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["num_chunks_per_doc"],
                    return_features=True,
                )
                if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                    raise ValueError(
                        "Expected MIL model(..., return_features=True) -> (logits, attn_weights, doc_embeddings)"
                    )
                doc_emb = out[2]
                all_embs.append(doc_emb.detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_embs = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
    oof_labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return oof_embs, oof_labels


# ---------------------------------------------------------------------------
# HF model variants
# ---------------------------------------------------------------------------

def cv_ensemble_mean_logits_hf(val_loader, accelerator, model_paths: list[str], model_loader) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean logits across ensemble members over a HF-style *val_loader*."""
    sum_logits = None
    labels_all = None

    for mi, mp in enumerate(model_paths):
        if accelerator.is_local_main_process:
            print(f"  [cv_ens_hf] loading {mi+1}/{len(model_paths)}: {mp}")

        model = model_loader(mp)
        model.to(accelerator.device)
        model.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, disable=not accelerator.is_local_main_process, desc=f"CV HF logits {mi+1}/{len(model_paths)}"):
                labels = batch.get("labels")
                inputs = {k: v for k, v in batch.items() if k != "labels"}
                out = model(**inputs)
                logits = out.logits
                all_logits.append(accelerator.gather_for_metrics(logits).cpu())
                if labels_all is None:
                    if labels is None:
                        raise ValueError("Batch missing 'labels'")
                    all_labels.append(accelerator.gather_for_metrics(labels).cpu())

        logits_np = torch.cat(all_logits, dim=0).numpy()
        sum_logits = logits_np if sum_logits is None else (sum_logits + logits_np)

        if labels_all is None:
            labels_all = torch.cat(all_labels, dim=0).numpy().astype(np.int64)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_logits = (sum_logits / float(len(model_paths))).astype(np.float32)
    return mean_logits, labels_all


def cv_ensemble_mean_embeddings_hf(
    dataloader,
    accelerator,
    model_paths: list[str],
    model_loader,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean CLS embeddings across ensemble members (HF models).

    Returns:
      mean_embs: (N, D) float32
      labels:    (N,) int64
    """
    sum_embs = None
    labels_all = None

    for mi, mp in enumerate(model_paths):
        if accelerator.is_local_main_process:
            print(f"  [cv_ens_hf] emb member {mi+1}/{len(model_paths)}: {os.path.basename(mp)}")

        model = model_loader(mp)
        model.to(accelerator.device)
        model.eval()

        all_embs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                disable=not accelerator.is_local_main_process,
                desc=f"CV HF embs {mi+1}/{len(model_paths)}",
            ):
                labels = batch.get("labels")
                inputs = {k: v for k, v in batch.items() if k != "labels"}
                out = model(**inputs, output_hidden_states=True, return_dict=True)

                hs = getattr(out, "hidden_states", None)
                if hs is not None and len(hs) > 0 and torch.is_tensor(hs[-1]):
                    last = hs[-1]
                else:
                    last = getattr(out, "last_hidden_state", None)
                    if last is None:
                        raise ValueError("HF output missing hidden_states and last_hidden_state")

                cls = last[:, 0, :]
                all_embs.append(accelerator.gather_for_metrics(cls).cpu())

                if labels_all is None:
                    if labels is None:
                        raise ValueError("Batch missing 'labels'")
                    all_labels.append(accelerator.gather_for_metrics(labels).cpu())

        embs_np = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
        sum_embs = embs_np if sum_embs is None else (sum_embs + embs_np)

        if labels_all is None:
            labels_all = torch.cat(all_labels, dim=0).numpy().astype(np.int64)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_embs = (sum_embs / float(len(model_paths))).astype(np.float32)
    return mean_embs, labels_all


def cv_ensemble_probs_unc2_hf(
    dataloader,
    accelerator,
    model_paths: list[str],
    model_loader,
    temperature: float,
    unc_metric: str,
    *,
    return_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble mean probs + uncertainty for HF models.

    If *return_embeddings* is True, also returns mean CLS embeddings as the last element.
    """
    all_members_probs: list[np.ndarray] = []
    all_members_embs: list[np.ndarray] = []
    labels_all = None

    T = float(max(temperature, 1e-6))
    unc_metric = str(unc_metric)

    for mi, mp in enumerate(model_paths):
        if accelerator.is_local_main_process:
            print(f"  [cv_ens_hf] member {mi+1}/{len(model_paths)}: {os.path.basename(mp)}")

        model = model_loader(mp)
        model.to(accelerator.device)
        model.eval()

        member_probs = []
        member_embs = []
        member_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"HF member {mi+1}"):
                labels = batch.get("labels")
                inputs = {k: v for k, v in batch.items() if k != "labels"}
                out = model(**inputs, output_hidden_states=bool(return_embeddings), return_dict=True)
                logits = out.logits
                probs = torch.softmax(logits / T, dim=1)

                member_probs.append(accelerator.gather_for_metrics(probs).cpu())
                if return_embeddings:
                    hs = getattr(out, "hidden_states", None)
                    if hs is not None and len(hs) > 0 and torch.is_tensor(hs[-1]):
                        last = hs[-1]
                    else:
                        last = getattr(out, "last_hidden_state", None)
                        if last is None:
                            raise ValueError("HF output missing hidden_states and last_hidden_state")
                    cls = last[:, 0, :]
                    member_embs.append(accelerator.gather_for_metrics(cls).cpu())
                if labels_all is None:
                    if labels is None:
                        raise ValueError("Batch missing 'labels'")
                    member_labels.append(accelerator.gather_for_metrics(labels).cpu())

        all_members_probs.append(torch.cat(member_probs, dim=0).numpy().astype(np.float32))
        if return_embeddings:
            all_members_embs.append(torch.cat(member_embs, dim=0).numpy().astype(np.float32))
        if labels_all is None:
            labels_all = torch.cat(member_labels, dim=0).numpy().astype(np.int64)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stacked_probs = np.stack(all_members_probs, axis=1)  # (N, M, C)
    mean_probs = np.mean(stacked_probs, axis=1).astype(np.float32)

    eps = 1e-8
    p = np.clip(mean_probs, eps, 1.0)
    entropy = (-(p * np.log(p)).sum(axis=-1)).astype(np.float32)

    p_pos_members = stacked_probs[:, :, 1]
    lower = np.percentile(p_pos_members, 5, axis=1).astype(np.float32)
    upper = np.percentile(p_pos_members, 95, axis=1).astype(np.float32)
    piw = (upper - lower).astype(np.float32)

    if unc_metric == "std_pos":
        unc = np.std(p_pos_members, axis=1).astype(np.float32)
    elif unc_metric == "entropy":
        unc = entropy
    elif unc_metric == "mi":
        mean_H = np.mean(-(stacked_probs * np.log(stacked_probs + eps)).sum(axis=-1), axis=1)
        unc = (entropy - mean_H).astype(np.float32)
    else:
        raise ValueError(f"Unknown unc_metric={unc_metric}")

    if return_embeddings:
        stacked_embs = np.stack(all_members_embs, axis=1)  # (N,M,D)
        mean_embs = np.mean(stacked_embs, axis=1).astype(np.float32)
        return mean_probs, unc, labels_all, entropy, piw, mean_embs

    return mean_probs, unc, labels_all, entropy, piw


# ---------------------------------------------------------------------------
# OOF collection – HF models
# ---------------------------------------------------------------------------

def cv_oof_collect_logits_labels_hf(
    args,
    accelerator,
    tokenizer,
    model_paths: list[str],
    batch_size: int,
    max_length: int,
    model_loader,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-leaky OOF logit collection for HF models (main process only)."""
    assert accelerator.is_main_process, "cv_oof_collect_logits_labels_hf must run on main process only"

    from torch.utils.data import DataLoader
    import pandas as pd
    from ..data.hf_data import TextDataset, make_hf_collate_fn, texts_labels_from_df

    all_logits = []
    all_labels = []

    device = accelerator.device

    for i, mp in enumerate(model_paths):
        val_json = os.path.join(args.input_dir, f"fold_{i}_val.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Missing OOF val split: {val_json}")

        if accelerator.is_local_main_process:
            print(f"  [oof_hf] fold {i}: model={mp}  val={val_json}")

        df = pd.read_json(val_json, lines=True)
        df = df.sample(frac=1, random_state=int(args.seed)).reset_index(drop=True)
        texts, labels = texts_labels_from_df(df)

        model_i = model_loader(mp)
        model_i.to(device)
        model_i.eval()

        collate_fn = make_hf_collate_fn(tokenizer, model_i.config, max_length=int(max_length))
        loader = DataLoader(TextDataset(texts, labels), batch_size=int(batch_size), collate_fn=collate_fn, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(loader, disable=not accelerator.is_local_main_process, desc=f"OOF HF logits fold {i}"):
                labels_t = batch.get("labels")
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                out = model_i(**inputs)
                logits = out.logits

                all_logits.append(logits.detach().cpu())
                if labels_t is None:
                    raise ValueError("Batch missing 'labels'")
                all_labels.append(labels_t.detach().cpu())

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_logits = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    oof_labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return oof_logits, oof_labels


def cv_oof_collect_logits_embeddings_labels_hf(
    args,
    accelerator,
    tokenizer,
    model_paths: list[str],
    batch_size: int,
    max_length: int,
    model_loader,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Distributed OOF collection for HF models (logits + CLS embeddings) in one pass."""
    from torch.utils.data import DataLoader
    import pandas as pd
    from ..data.hf_data import TextDataset, make_hf_collate_fn, texts_labels_from_df

    rank, world_size, _ = _dist_info(accelerator)
    device = accelerator.device

    local: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i, mp in enumerate(model_paths):
        if (i % world_size) != rank:
            continue

        val_json = os.path.join(args.input_dir, f"fold_{i}_val.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Missing OOF val split: {val_json}")

        if accelerator.is_local_main_process:
            print(f"  [oof_hf] fold {i}: model={mp}  val={val_json}")

        df = pd.read_json(val_json, lines=True)
        df = df.sample(frac=1, random_state=int(args.seed)).reset_index(drop=True)
        texts, labels = texts_labels_from_df(df)

        model_i = model_loader(mp)
        model_i.to(device)
        model_i.eval()

        collate_fn = make_hf_collate_fn(tokenizer, model_i.config, max_length=int(max_length))
        loader = DataLoader(
            TextDataset(texts, labels),
            batch_size=int(batch_size),
            collate_fn=collate_fn,
            shuffle=False,
        )

        fold_logits = []
        fold_embs = []
        fold_labels = []
        with torch.no_grad():
            for batch in tqdm(
                loader,
                disable=not accelerator.is_local_main_process,
                desc=f"OOF HF logits+emb fold {i}",
            ):
                labels_t = batch.get("labels")
                if labels_t is None:
                    raise ValueError("Batch missing 'labels'")
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                out = model_i(**inputs, output_hidden_states=True, return_dict=True)
                logits = out.logits

                hs = getattr(out, "hidden_states", None)
                if hs is not None and len(hs) > 0 and torch.is_tensor(hs[-1]):
                    last = hs[-1]
                else:
                    last = getattr(out, "last_hidden_state", None)
                    if last is None:
                        raise ValueError("HF output missing hidden_states and last_hidden_state")
                cls = last[:, 0, :]

                fold_logits.append(logits.detach().cpu())
                fold_embs.append(cls.detach().cpu())
                fold_labels.append(labels_t.detach().cpu())

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local.append(
            (
                int(i),
                torch.cat(fold_logits, dim=0),
                torch.cat(fold_embs, dim=0),
                torch.cat(fold_labels, dim=0),
            )
        )

    gathered = _all_gather_object(accelerator, local)
    if not accelerator.is_main_process:
        return None, None, None

    merged: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for part in gathered:
        if not part:
            continue
        merged.extend(part)
    merged.sort(key=lambda t: t[0])

    all_logits = [t[1] for t in merged]
    all_embs = [t[2] for t in merged]
    all_labels = [t[3] for t in merged]

    oof_logits = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    oof_embs = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
    oof_labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return oof_logits, oof_embs, oof_labels


def cv_oof_collect_logits_embeddings_labels_hf_fold_ids(
    args,
    accelerator,
    tokenizer,
    model_paths: list[str],
    batch_size: int,
    max_length: int,
    model_loader,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Distributed OOF collection for HF models (logits + CLS embeddings) with fold IDs."""
    from torch.utils.data import DataLoader
    import pandas as pd
    from ..data.hf_data import TextDataset, make_hf_collate_fn, texts_labels_from_df

    rank, world_size, _ = _dist_info(accelerator)
    device = accelerator.device

    local: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for i, mp in enumerate(model_paths):
        if (i % world_size) != rank:
            continue

        val_json = os.path.join(args.input_dir, f"fold_{i}_val.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Missing OOF val split: {val_json}")

        if accelerator.is_local_main_process:
            print(f"  [oof_hf] fold {i}: model={mp}  val={val_json}")

        df = pd.read_json(val_json, lines=True)
        df = df.sample(frac=1, random_state=int(args.seed)).reset_index(drop=True)
        texts, labels = texts_labels_from_df(df)

        model_i = model_loader(mp)
        model_i.to(device)
        model_i.eval()

        collate_fn = make_hf_collate_fn(tokenizer, model_i.config, max_length=int(max_length))
        loader = DataLoader(
            TextDataset(texts, labels),
            batch_size=int(batch_size),
            collate_fn=collate_fn,
            shuffle=False,
        )

        fold_logits = []
        fold_embs = []
        fold_labels = []
        with torch.no_grad():
            for batch in tqdm(
                loader,
                disable=not accelerator.is_local_main_process,
                desc=f"OOF HF logits+emb fold {i}",
            ):
                labels_t = batch.get("labels")
                if labels_t is None:
                    raise ValueError("Batch missing 'labels'")
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                out = model_i(**inputs, output_hidden_states=True, return_dict=True)
                logits = out.logits

                hs = getattr(out, "hidden_states", None)
                if hs is not None and len(hs) > 0 and torch.is_tensor(hs[-1]):
                    last = hs[-1]
                else:
                    last = getattr(out, "last_hidden_state", None)
                    if last is None:
                        raise ValueError("HF output missing hidden_states and last_hidden_state")
                cls = last[:, 0, :]

                fold_logits.append(logits.detach().cpu())
                fold_embs.append(cls.detach().cpu())
                fold_labels.append(labels_t.detach().cpu())

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_logits_t = torch.cat(fold_logits, dim=0)
        fold_embs_t = torch.cat(fold_embs, dim=0)
        fold_labels_t = torch.cat(fold_labels, dim=0)
        fold_ids_t = torch.full((int(fold_labels_t.shape[0]),), int(i), dtype=torch.int64)

        local.append((int(i), fold_logits_t, fold_embs_t, fold_labels_t, fold_ids_t))

    gathered = _all_gather_object(accelerator, local)
    if not accelerator.is_main_process:
        return None, None, None, None

    merged: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for part in gathered:
        if not part:
            continue
        merged.extend(part)
    merged.sort(key=lambda t: t[0])

    all_logits = [t[1] for t in merged]
    all_embs = [t[2] for t in merged]
    all_labels = [t[3] for t in merged]
    all_fold_ids = [t[4] for t in merged]

    oof_logits = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    oof_embs = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
    oof_labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    oof_fold_ids = torch.cat(all_fold_ids, dim=0).numpy().astype(np.int64)
    return oof_logits, oof_embs, oof_labels, oof_fold_ids


def cv_oof_collect_embeddings_labels_hf(
    args,
    accelerator,
    tokenizer,
    model_paths: list[str],
    batch_size: int,
    max_length: int,
    model_loader,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-leaky OOF embedding collection for HF models (main process only).

    Returns:
      oof_embeddings: (N, D) float32 (CLS from last hidden state)
      oof_labels:     (N,) int64
    """
    assert accelerator.is_main_process, "cv_oof_collect_embeddings_labels_hf must run on main process only"

    from torch.utils.data import DataLoader
    import pandas as pd
    from ..data.hf_data import TextDataset, make_hf_collate_fn, texts_labels_from_df

    all_embs = []
    all_labels = []

    device = accelerator.device

    for i, mp in enumerate(model_paths):
        val_json = os.path.join(args.input_dir, f"fold_{i}_val.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(f"Missing OOF val split: {val_json}")

        if accelerator.is_local_main_process:
            print(f"  [oof_emb_hf] fold {i}: model={mp}  val={val_json}")

        df = pd.read_json(val_json, lines=True)
        df = df.sample(frac=1, random_state=int(args.seed)).reset_index(drop=True)
        texts, labels = texts_labels_from_df(df)

        model_i = model_loader(mp)
        model_i.to(device)
        model_i.eval()

        collate_fn = make_hf_collate_fn(tokenizer, model_i.config, max_length=int(max_length))
        loader = DataLoader(TextDataset(texts, labels), batch_size=int(batch_size), collate_fn=collate_fn, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(loader, disable=not accelerator.is_local_main_process, desc=f"OOF HF emb fold {i}"):
                labels_t = batch.get("labels")
                if labels_t is None:
                    raise ValueError("Batch missing 'labels'")
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

                out = model_i(**inputs, output_hidden_states=True, return_dict=True)

                hs = getattr(out, "hidden_states", None)
                if hs is not None and len(hs) > 0 and torch.is_tensor(hs[-1]):
                    last = hs[-1]  # (B,L,D)
                else:
                    last = getattr(out, "last_hidden_state", None)
                    if last is None:
                        raise ValueError("HF output missing hidden_states and last_hidden_state")

                cls = last[:, 0, :]  # (B,D)
                all_embs.append(cls.detach().cpu())
                all_labels.append(labels_t.detach().cpu())

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_embs = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
    oof_labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    return oof_embs, oof_labels


# ---------------------------------------------------------------------------
# Parallel-model ensemble (distributes models across processes)
# ---------------------------------------------------------------------------

def cv_ensemble_probs_unc2_parallel_models(
    dataloader,
    accelerator,
    model_paths: list[str],
    model_loader,
    temperature: float,
    unc_metric: str,
    *,
    collate_fn=None,
    return_embeddings: bool = False,
):
    """Like :func:`cv_ensemble_probs_unc2`, but parallelises across ensemble members.

    Each process evaluates only a subset of models over the *full* dataset (non-sharded
    loader), then per-member probabilities are gathered on main to compute uncertainty.
    """
    rank, world_size, dist_ok = _dist_info(accelerator)
    if world_size <= 1:
        return cv_ensemble_probs_unc2(
            dataloader,
            accelerator,
            model_paths,
            model_loader,
            temperature=float(temperature),
            unc_metric=str(unc_metric),
            return_embeddings=bool(return_embeddings),
        )

    full_loader = _rebuild_full_loader(dataloader, collate_fn=collate_fn)
    device = accelerator.device
    T = float(max(float(temperature), 1e-6))

    # Collect labels once (same on all processes)
    labels_list = []
    for batch in full_loader:
        labels_list.append(batch["labels"].detach().cpu())
    labels_all = torch.cat(labels_list, dim=0).numpy().astype(np.int64)

    local_member_probs: list[tuple[int, np.ndarray]] = []
    local_sum_embs = None
    local_n_models = 0

    for mi, mp in enumerate(model_paths):
        if (mi % world_size) != rank:
            continue

        if accelerator.is_local_main_process:
            print(f"  [cv_ens|par] member {mi+1}/{len(model_paths)}: {os.path.basename(mp)}")

        model = model_loader(mp)
        model.to(device)
        model.eval()

        probs_all = []
        emb_all = []

        with torch.no_grad():
            for batch in tqdm(
                full_loader,
                disable=not accelerator.is_local_main_process,
                desc=f"Member(par) {mi+1}",
            ):
                out = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["num_chunks_per_doc"],
                    return_features=bool(return_embeddings),
                )
                if not isinstance(out, (tuple, list)) or len(out) < 1:
                    raise ValueError("MIL model must return a tuple/list like (logits, ...).")
                logits = out[0]
                probs = torch.softmax(logits / T, dim=1).detach().cpu()
                probs_all.append(probs)

                if return_embeddings:
                    if len(out) < 3:
                        raise ValueError(
                            "return_embeddings=True requires MIL model to return (logits, attn_weights, doc_embeddings)"
                        )
                    doc_emb = out[2].detach().to(device=device, dtype=torch.float32)
                    emb_all.append(doc_emb)

        probs_np = torch.cat(probs_all, dim=0).numpy().astype(np.float32)
        local_member_probs.append((int(mi), probs_np))

        if return_embeddings:
            if len(emb_all) == 0:
                raise ValueError("return_embeddings=True but no embeddings were collected")
            member_embs = torch.cat(emb_all, dim=0)  # (N,D) on device
            local_sum_embs = member_embs if local_sum_embs is None else (local_sum_embs + member_embs)
        local_n_models += 1

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gathered = _all_gather_object(accelerator, local_member_probs)

    # Mean embeddings via all_reduce on CPU tensors (same shape on all procs)
    mean_embs = None
    if return_embeddings:
        if local_sum_embs is None:
            raise ValueError("return_embeddings=True but no embeddings were accumulated")
        sum_embs_t = local_sum_embs.to(device=device, dtype=torch.float32)
        n_models_t = torch.tensor([float(local_n_models)], dtype=torch.float32, device=device)
        if dist_ok:
            dist.all_reduce(sum_embs_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_models_t, op=dist.ReduceOp.SUM)
        denom = float(max(n_models_t.item(), 1.0))
        mean_embs = (sum_embs_t / denom).detach().cpu().numpy().astype(np.float32)

    if not accelerator.is_main_process:
        if return_embeddings:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
            )
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    # Rebuild member list in correct order
    probs_by_idx: dict[int, np.ndarray] = {}
    for part in gathered:
        for mi, probs_np in (part or []):
            probs_by_idx[int(mi)] = probs_np
    if len(probs_by_idx) != len(model_paths):
        missing = sorted(set(range(len(model_paths))) - set(probs_by_idx.keys()))
        raise ValueError(f"Missing member probs for indices: {missing[:10]}")

    stacked_probs = np.stack([probs_by_idx[i] for i in range(len(model_paths))], axis=1).astype(np.float32)  # (N,M,C)
    mean_probs = np.mean(stacked_probs, axis=1).astype(np.float32)

    eps = 1e-8
    p = np.clip(mean_probs, eps, 1.0)
    entropy = (-(p * np.log(p)).sum(axis=-1)).astype(np.float32)

    if mean_probs.shape[1] != 2:
        raise ValueError("cv_ensemble_probs_unc2_parallel_models currently assumes binary classification")

    p_pos_members = stacked_probs[:, :, 1]
    lower = np.percentile(p_pos_members, 5, axis=1).astype(np.float32)
    upper = np.percentile(p_pos_members, 95, axis=1).astype(np.float32)
    piw = (upper - lower).astype(np.float32)

    unc_metric = str(unc_metric)
    if unc_metric == "std_pos":
        unc = np.std(p_pos_members, axis=1).astype(np.float32)
    elif unc_metric == "entropy":
        unc = entropy
    elif unc_metric == "mi":
        mean_H = np.mean(-(stacked_probs * np.log(stacked_probs + eps)).sum(axis=-1), axis=1)
        unc = (entropy - mean_H).astype(np.float32)
    else:
        raise ValueError(f"Unknown unc_metric={unc_metric}")

    if return_embeddings:
        return mean_probs, unc, labels_all, entropy, piw, mean_embs
    return mean_probs, unc, labels_all, entropy, piw
