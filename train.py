"""Train a Gated Attention MIL classifier with optional MD-SN and class-imbalance losses."""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from sklearn.covariance import OAS
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.attention_mil import AttentionMILClassifier
from uq_eval.data.mil_data import MILCollator, MILDataset
from uq_eval.reporting.metrics import compute_comprehensive_metrics
from uq_eval.utils.repro import sanitize_accelerate_env, set_seed

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_determinism(seed: int):
    """Lock down CUDA/cuDNN for fully reproducible runs (slower)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    _seed_everything(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # TF32 can change results across runs/hardware; disable for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Enforce deterministic algorithms when available
    # warn_only=True avoids hard crashes if an op has no deterministic implementation
    torch.use_deterministic_algorithms(True, warn_only=True)


def _maybe_flash_attn(accelerator: Accelerator, *, deterministic: bool = False) -> str:
    """Pick the fastest attention implementation available on the current GPU."""
    if deterministic:
        if accelerator.is_local_main_process:
            print("[train] deterministic=1 -> disabling flash attention kernels")
        return "sdpa"

    use_flash_attn = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            use_flash_attn = True
            if accelerator.is_local_main_process:
                print("Flash Attention 2 Enabled!")
    return "flash_attention_2" if use_flash_attn else "sdpa"


# ---------------------------------------------------------------------------
# MD-SN: Mahalanobis-distance spectral-norm feature extraction & fitting
# ---------------------------------------------------------------------------


@torch.no_grad()
def _extract_mdsn_features_labels(
    model: AttentionMILClassifier,
    loader: DataLoader,
    accelerator: Accelerator,
):
    """Forward-pass the train set and collect penultimate-layer features + labels."""
    model.eval()

    feats_all = []
    labs_all = []

    progress = tqdm(
        total=len(loader),
        disable=not accelerator.is_local_main_process,
        desc="Extract MD-SN features",
    )

    for batch in loader:
        out = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["num_chunks_per_doc"],
            return_features=True,
        )
        if not (isinstance(out, (tuple, list)) and len(out) >= 3):
            raise ValueError("Expected (logits, attn, features) from MIL model")

        feats = out[2]
        labs = batch["labels"]

        feats_g = accelerator.gather_for_metrics(feats.float()).detach().cpu()
        labs_g = accelerator.gather_for_metrics(labs).detach().cpu()
        feats_all.append(feats_g)
        labs_all.append(labs_g)

        progress.update(1)

    progress.close()

    X = torch.cat(feats_all, dim=0).numpy().astype(np.float32)
    y = torch.cat(labs_all, dim=0).numpy().astype(np.int64)
    return X, y


def _fit_shared_precision_oas(X: np.ndarray, y: np.ndarray, n_classes: int = 2):
    """Fit a shared OAS-shrinkage precision matrix around per-class centroids."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    centroids = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        Xc = X[y == c]
        if Xc.shape[0] == 0:
            continue
        centroids[c] = Xc.mean(axis=0)

    # Build centered residuals around the centroid of the TRUE class
    residuals = []
    for c in range(n_classes):
        Xc = X[y == c]
        if Xc.shape[0] == 0:
            continue
        residuals.append(Xc - centroids[c][None, :])

    R = np.concatenate(residuals, axis=0)
    # Residuals are already centered -> assume_centered=True
    oas = OAS(assume_centered=True).fit(R)
    precision = oas.precision_.astype(np.float64)

    centroids32 = centroids.astype(np.float32)
    precision32 = precision.astype(np.float32)

    if (not np.isfinite(centroids32).all()) or (not np.isfinite(precision32).all()):
        raise ValueError("OAS produced non-finite centroids/precision")

    return centroids32, precision32


# ---------------------------------------------------------------------------
# Loss functions for class-imbalanced training
# ---------------------------------------------------------------------------


@dataclass
class BestState:
    best_f1: float = -1.0


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax (Ren et al.): CE(logits + log(class_count)).
    Good first option for minority recall while keeping calibration relatively stable.
    """
    def __init__(self, class_counts: np.ndarray, eps: float = 1e-12):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        self.register_buffer("log_counts", torch.log(torch.clamp(counts, min=eps)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj = logits + self.log_counts.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(adj, targets)


class LogitAdjustmentLoss(nn.Module):
    """
    Logit adjustment (Menon et al.): CE(logits + tau * log(prior)).
    tau controls strength; tau in [0.5, 1.0] is a good starting range.
    """
    def __init__(self, class_counts: np.ndarray, tau: float = 1.0, eps: float = 1e-12):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        priors = counts / torch.clamp(counts.sum(), min=eps)
        self.tau = float(tau)
        self.register_buffer("log_priors", torch.log(torch.clamp(priors, min=eps)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adj = logits + (self.tau * self.log_priors.to(device=logits.device, dtype=logits.dtype))
        return F.cross_entropy(adj, targets)


class LDAMLoss(nn.Module):
    """
    LDAM (Cao et al.): subtract class-dependent margin from the true-class logit.
    Common default: s=30, max_m=0.5. Often paired with CE weights.
    """
    def __init__(
        self,
        class_counts: np.ndarray,
        *,
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        margins = float(max_m) / torch.pow(torch.clamp(counts, min=eps), 0.25)
        self.register_buffer("margins", margins)
        self.s = float(s)
        self.weight = weight  # optional CE reweighting tensor length C

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape
        margins = self.margins.to(device=logits.device, dtype=logits.dtype)

        idx = torch.arange(B, device=logits.device)
        m_y = margins.gather(0, targets)  # (B,)

        logits_m = logits.clone()
        logits_m[idx, targets] = logits_m[idx, targets] - m_y

        scaled = self.s * logits_m
        if self.weight is not None:
            w = self.weight.to(device=logits.device, dtype=logits.dtype)
            return F.cross_entropy(scaled, targets, weight=w)
        return F.cross_entropy(scaled, targets)


def _build_loss_fn(
    *,
    loss_type: str,
    class_counts: np.ndarray,
    ce_weight: torch.Tensor | None,
    logit_adj_tau: float,
    ldam_s: float,
    ldam_max_m: float,
    accelerator: Accelerator,
):
    """Instantiate the chosen loss function for training."""
    loss_type = str(loss_type).lower().strip()
    if loss_type in {"ce", "cross_entropy"}:
        def _ce(logits, targets):
            return F.cross_entropy(logits, targets, weight=ce_weight)
        return _ce

    if loss_type in {"balanced_softmax", "bs"}:
        if ce_weight is not None and accelerator.is_local_main_process:
            print("[train] WARNING: ignoring --class_weight/--class_weights for balanced_softmax (avoid double correction).")
        crit = BalancedSoftmaxLoss(class_counts=class_counts)
        return lambda logits, targets: crit(logits, targets)

    if loss_type in {"logit_adjust", "logit_adjustment", "la"}:
        if ce_weight is not None and accelerator.is_local_main_process:
            print("[train] WARNING: ignoring --class_weight/--class_weights for logit_adjust (avoid double correction).")
        crit = LogitAdjustmentLoss(class_counts=class_counts, tau=float(logit_adj_tau))
        return lambda logits, targets: crit(logits, targets)

    if loss_type in {"ldam"}:
        crit = LDAMLoss(
            class_counts=class_counts,
            max_m=float(ldam_max_m),
            s=float(ldam_s),
            weight=ce_weight,  # LDAM is commonly paired with reweighting
        )
        return lambda logits, targets: crit(logits, targets)

    raise ValueError(f"Unknown --loss_type={loss_type!r}. Choose from: ce, balanced_softmax, logit_adjust, ldam.")


# ---------------------------------------------------------------------------
# R-Drop regularization
# ---------------------------------------------------------------------------


def _rdrop_sym_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """
    Symmetric KL between predictive distributions.
    R-Drop: 0.5*(KL(p||q)+KL(q||p)) where p,q are softmax outputs.
    """
    logp = F.log_softmax(logits_a, dim=-1)
    logq = F.log_softmax(logits_b, dim=-1)
    p = logp.exp()
    q = logq.exp()
    kl_pq = F.kl_div(logp, q, reduction="batchmean")
    kl_qp = F.kl_div(logq, p, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train a Gated Attention MIL classifier with optional MD-SN.",
    )

    # ---- Data (CSV-based) ----
    parser.add_argument(
        "--train_csv", type=str, required=True,
        help="Path to training CSV with 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--val_csv", type=str, required=True,
        help="Path to validation CSV with 'text' and 'label' columns.",
    )

    # ---- Model ----
    parser.add_argument("--model_checkpoint", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, required=True)

    # ---- Reproducibility ----
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training (slower).")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # ---- Data / MIL chunking ----
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--chunk_overlap", type=int, default=64)

    # ---- Training hyperparameters ----
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument(
        "--warmup_steps", type=int, default=0,
        help="If >0, overrides --warmup_ratio with an absolute number of warmup steps.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ---- Class-imbalance weighting ----
    parser.add_argument(
        "--class_weight", type=str, default="none",
        choices=["none", "balanced", "balanced_sqrt"],
        help=(
            "Class weighting for cross-entropy. "
            "'balanced' uses inverse-frequency weights from the TRAIN split; "
            "'balanced_sqrt' uses sqrt(inverse-frequency) for a gentler effect."
        ),
    )
    parser.add_argument(
        "--class_weights", type=str, default="",
        help="Optional manual CE weights as 'w0,w1' (overrides --class_weight). Example: '1.0,3.0'.",
    )

    # ---- Early stopping ----
    parser.add_argument(
        "--early_stop_patience", type=int, default=3,
        help="Stop if val F1 doesn't improve for this many epochs. 0 disables.",
    )
    parser.add_argument(
        "--early_stop_min_delta", type=float, default=0.01,
        help="Minimum absolute F1 improvement to reset patience.",
    )

    # ---- MD-SN configuration ----
    parser.add_argument("--mdsn_cov_ridge", type=float, default=1e-6)
    parser.add_argument("--mdsn_n_power_iterations", type=int, default=1)
    parser.add_argument("--mdsn_fit", type=str, default="oas", choices=["oas", "ridge"])

    # ---- Loss strategy (imbalance-friendly) ----
    parser.add_argument(
        "--loss_type", type=str, default="ce",
        choices=["ce", "balanced_softmax", "logit_adjust", "ldam"],
        help=(
            "Loss for class imbalance. "
            "balanced_softmax/logit_adjust are usually more calibration-friendly; "
            "ldam can push recall more strongly (often with reweighting)."
        ),
    )
    parser.add_argument("--logit_adj_tau", type=float, default=1.0)
    parser.add_argument("--ldam_s", type=float, default=30.0)
    parser.add_argument("--ldam_max_m", type=float, default=0.5)

    # ---- R-Drop (optional, costs ~2x forward when active) ----
    parser.add_argument("--r_drop", action="store_true", help="Enable R-Drop regularization (2 forward passes).")
    parser.add_argument("--r_drop_alpha", type=float, default=0.5, help="Weight for symmetric KL term.")
    parser.add_argument(
        "--r_drop_start_epoch", type=int, default=0,
        help="Start applying R-Drop from this epoch (0 = from start).",
    )
    parser.add_argument(
        "--r_drop_every", type=int, default=1,
        help="Apply R-Drop every k training steps (1 = every step).",
    )

    args = parser.parse_args()

    # ---- Environment & seed ----
    sanitize_accelerate_env()
    set_seed(int(args.seed))

    if args.deterministic:
        _enable_determinism(int(args.seed))

    accelerator = Accelerator(gradient_accumulation_steps=int(args.grad_accum))

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load train / val data from CSV
    # ------------------------------------------------------------------
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    for name, df in [("train_csv", train_df), ("val_csv", val_df)]:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"--{name} must have 'text' and 'label' columns; found {list(df.columns)}")

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].astype(int).to_numpy(dtype=np.int64)
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].astype(int).to_numpy(dtype=np.int64)

    # ------------------------------------------------------------------
    # Optional class weights (computed from TRAIN split only)
    # ------------------------------------------------------------------
    class_weights_np = None
    if args.class_weights.strip():
        parts = [p.strip() for p in args.class_weights.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("--class_weights must be two comma-separated floats: 'w0,w1'")
        class_weights_np = np.asarray([float(parts[0]), float(parts[1])], dtype=np.float32)
    elif args.class_weight in {"balanced", "balanced_sqrt"}:
        counts = np.bincount(train_labels, minlength=2).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (2.0 * counts)  # inverse frequency
        if args.class_weight == "balanced_sqrt":
            w = np.sqrt(w)
        w = w / np.mean(w)  # keep average weight ~ 1.0 for stability
        class_weights_np = w.astype(np.float32)

    # Class counts (needed by balanced_softmax / logit_adjust / LDAM)
    class_counts_np = np.bincount(train_labels, minlength=2).astype(np.float64)
    class_counts_np = np.maximum(class_counts_np, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Tokenizer & MIL datasets
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    train_ds = MILDataset(
        train_texts, train_labels, tokenizer,
        chunk_max_length=int(args.max_length),
        chunk_overlap=int(args.chunk_overlap),
    )
    val_ds = MILDataset(
        val_texts, val_labels, tokenizer,
        chunk_max_length=int(args.max_length),
        chunk_overlap=int(args.chunk_overlap),
    )

    collate = MILCollator(tokenizer, max_length=int(args.max_length), max_chunks=int(args.max_chunks))

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    attn_impl = _maybe_flash_attn(accelerator, deterministic=args.deterministic)
    model = AttentionMILClassifier(
        args.model_checkpoint,
        num_labels=2,
        dropout=0.1,
        attention_dim=128,
        attn_implementation=attn_impl,
        use_mdsn=True,
        mdsn_spectral_norm=True,
        mdsn_n_power_iterations=int(args.mdsn_n_power_iterations),
        mdsn_cov_ridge=float(args.mdsn_cov_ridge),
    )

    # ------------------------------------------------------------------
    # Optimizer & LR scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    num_update_steps_per_epoch = int(np.ceil(len(train_loader) / max(int(args.grad_accum), 1)))
    total_steps = int(args.epochs) * num_update_steps_per_epoch
    warmup_steps = int(args.warmup_steps) if int(args.warmup_steps) > 0 else int(float(args.warmup_ratio) * total_steps)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Prepare everything with Accelerate
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    best = BestState()

    # ------------------------------------------------------------------
    # Build loss function
    # ------------------------------------------------------------------
    ce_weight = None
    if class_weights_np is not None:
        ce_weight = torch.tensor(class_weights_np, dtype=torch.float32, device=accelerator.device)
        if accelerator.is_local_main_process:
            n0 = int((train_labels == 0).sum())
            n1 = int((train_labels == 1).sum())
            print(f"[train] class counts: n0={n0} n1={n1}")
            print(f"[train] CE class weights: w0={float(class_weights_np[0]):.3f} w1={float(class_weights_np[1]):.3f}")

    if accelerator.is_local_main_process:
        print(f"[train] loss_type={args.loss_type}")
        if args.loss_type == "logit_adjust":
            print(f"[train] logit_adj_tau={float(args.logit_adj_tau):.3f}")
        if args.loss_type == "ldam":
            print(f"[train] ldam_s={float(args.ldam_s):.1f} ldam_max_m={float(args.ldam_max_m):.3f}")

    loss_fn = _build_loss_fn(
        loss_type=str(args.loss_type),
        class_counts=class_counts_np,
        ce_weight=ce_weight,
        logit_adj_tau=float(args.logit_adj_tau),
        ldam_s=float(args.ldam_s),
        ldam_max_m=float(args.ldam_max_m),
        accelerator=accelerator,
    )

    # ------------------------------------------------------------------
    # Helper: all-reduce for distributed loss averaging
    # ------------------------------------------------------------------
    def _allreduce_sum_count(sum_t: torch.Tensor, count_t: torch.Tensor):
        if accelerator.num_processes > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count_t, op=torch.distributed.ReduceOp.SUM)

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------
    def eval_one_epoch(epoch_idx: int) -> float:
        model.eval()
        all_probs = []
        all_y = []

        loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        count = torch.zeros((), device=accelerator.device, dtype=torch.float32)

        progress = tqdm(
            total=len(val_loader),
            disable=not accelerator.is_local_main_process,
            desc=f"Val epoch {epoch_idx}",
        )

        with torch.no_grad():
            for batch in val_loader:
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["num_chunks_per_doc"])
                loss = loss_fn(logits, batch["labels"])
                bsz = float(batch["labels"].shape[0])
                loss_sum += loss.detach().float() * bsz
                count += bsz
                probs = torch.softmax(logits, dim=1)

                probs_g = accelerator.gather_for_metrics(probs).detach().cpu()
                y_g = accelerator.gather_for_metrics(batch["labels"]).detach().cpu()

                all_probs.append(probs_g)
                all_y.append(y_g)
                progress.update(1)

        progress.close()

        _allreduce_sum_count(loss_sum, count)

        if not accelerator.is_main_process:
            return 0.0

        probs_np = torch.cat(all_probs, dim=0).numpy()
        y_np = torch.cat(all_y, dim=0).numpy().astype(np.int64)
        y_pred = probs_np.argmax(axis=1)

        metrics = compute_comprehensive_metrics(
            y_true=y_np,
            y_pred=y_pred,
            y_prob=probs_np,
            uncertainties=None,
            entropy=None,
            piw=None,
            prefix="val_",
        )
        val_loss = float((loss_sum / torch.clamp(count, min=1.0)).item())

        print(
            f"[val] epoch={epoch_idx} loss={val_loss:.4f} "
            f"acc={metrics['val_accuracy']:.4f} prec={metrics['val_precision']:.4f} rec={metrics['val_recall']:.4f} "
            f"f1={metrics['val_f1']:.4f} auroc={metrics['val_auroc']:.4f} ece={metrics['val_ece']:.4f} nll={metrics['val_nll']:.4f}"
        )
        return float(metrics["val_f1"])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    epochs_no_improve = 0

    for epoch in range(int(args.epochs)):
        model.train()

        train_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
        train_count = torch.zeros((), device=accelerator.device, dtype=torch.float32)

        progress = tqdm(
            total=len(train_loader),
            disable=not accelerator.is_local_main_process,
            desc=f"Train epoch {epoch}",
        )

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].clone()
                attention_mask = batch["attention_mask"].clone()
                num_chunks_per_doc = (
                    batch["num_chunks_per_doc"].clone()
                    if torch.is_tensor(batch["num_chunks_per_doc"])
                    else batch["num_chunks_per_doc"]
                )

                logits1, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_chunks_per_doc=num_chunks_per_doc,
                )
                loss1 = loss_fn(logits1, batch["labels"])

                # R-Drop: optional second forward pass for KL consistency
                do_rdrop = (
                    bool(args.r_drop)
                    and (epoch >= int(args.r_drop_start_epoch))
                    and (int(args.r_drop_every) > 0)
                    and ((global_step % int(args.r_drop_every)) == 0)
                )

                if do_rdrop:
                    logits2, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_chunks_per_doc=num_chunks_per_doc,
                    )
                    loss2 = loss_fn(logits2, batch["labels"])

                    kl = _rdrop_sym_kl(logits1, logits2)
                    loss = 0.5 * (loss1 + loss2) + float(args.r_drop_alpha) * kl
                else:
                    loss = loss1

                bsz = float(batch["labels"].shape[0])
                train_loss_sum += loss.detach().float() * bsz
                train_count += bsz

                accelerator.backward(loss)

                if accelerator.sync_gradients and float(args.max_grad_norm) > 0:
                    accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

        progress.close()

        # Log training loss
        _allreduce_sum_count(train_loss_sum, train_count)
        if accelerator.is_main_process:
            train_loss = float((train_loss_sum / torch.clamp(train_count, min=1.0)).item())
            lr_now = float(lr_scheduler.get_last_lr()[0]) if hasattr(lr_scheduler, "get_last_lr") else float(args.lr)
            print(f"[train] epoch={epoch} loss={train_loss:.4f} lr={lr_now:.6g}")

        # Evaluate
        f1 = eval_one_epoch(epoch)
        accelerator.wait_for_everyone()

        # Save best checkpoint on main process
        if accelerator.is_main_process and f1 > best.best_f1:
            best.best_f1 = f1
            epochs_no_improve = 0
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(args.output_dir, safe_serialization=False)
            print(f"Saved best checkpoint to {args.output_dir} (f1={f1:.4f})")
        elif accelerator.is_main_process and int(args.early_stop_patience) > 0:
            if f1 <= (best.best_f1 + float(args.early_stop_min_delta)):
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0

        accelerator.wait_for_everyone()

        # Early stopping coordination across processes
        stop_tensor = torch.zeros((), device=accelerator.device, dtype=torch.uint8)
        if accelerator.is_main_process and int(args.early_stop_patience) > 0:
            if epochs_no_improve >= int(args.early_stop_patience):
                stop_tensor.fill_(1)

        if accelerator.num_processes > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_tensor, src=0)

        if int(stop_tensor.item()) == 1:
            if accelerator.is_local_main_process:
                print(
                    f"Early stop at epoch={epoch} (patience={int(args.early_stop_patience)}, "
                    f"best_f1={best.best_f1:.4f})"
                )
            break

    # ------------------------------------------------------------------
    # Post-training: fit MD-SN statistics on train features
    # ------------------------------------------------------------------
    train_loader_fit = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate)
    train_loader_fit = accelerator.prepare(train_loader_fit)

    X_train, y_train = _extract_mdsn_features_labels(accelerator.unwrap_model(model), train_loader_fit, accelerator)

    if accelerator.is_main_process:
        u = accelerator.unwrap_model(model)
        if args.mdsn_fit == "oas":
            try:
                centroids, precision = _fit_shared_precision_oas(X_train, y_train, n_classes=2)
                u.mdsn_centroids.copy_(torch.from_numpy(centroids).to(u.mdsn_centroids.device, dtype=u.mdsn_centroids.dtype))
                u.mdsn_precision.copy_(torch.from_numpy(precision).to(u.mdsn_precision.device, dtype=u.mdsn_precision.dtype))
                u.mdsn_fitted.fill_(1)
                print("Fitted MD-SN stats with OAS shrinkage")
            except Exception as e:
                u.fit_mdsn(torch.from_numpy(X_train), torch.from_numpy(y_train))
                print(f"OAS fit failed ({e}); fitted MD-SN stats with ridge covariance")
        else:
            u.fit_mdsn(torch.from_numpy(X_train), torch.from_numpy(y_train))
            print("Fitted MD-SN stats with ridge covariance")

        u.save_pretrained(args.output_dir, safe_serialization=False)
        print(f"Saved final model (with MD stats) to {args.output_dir}")


if __name__ == "__main__":
    main()
