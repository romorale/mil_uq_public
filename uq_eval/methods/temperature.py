from __future__ import annotations

import numpy as np
import torch
from accelerate import Accelerator


DEFAULT_TEMP_MIN = 0.5
DEFAULT_TEMP_MAX = 10.0


def _normalize_temperature_bounds(temp_min: float, temp_max: float) -> tuple[float, float, str]:
    Tmin = float(temp_min)
    Tmax = float(temp_max)
    notes = []

    if not np.isfinite(Tmin) or Tmin <= 0.0:
        notes.append(f"temp_min={Tmin!r} -> {DEFAULT_TEMP_MIN}")
        Tmin = DEFAULT_TEMP_MIN
    if not np.isfinite(Tmax) or Tmax <= 0.0:
        notes.append(f"temp_max={Tmax!r} -> {DEFAULT_TEMP_MAX}")
        Tmax = DEFAULT_TEMP_MAX
    if Tmax < Tmin:
        notes.append(f"swapped bounds to keep temp_min <= temp_max ({Tmin} > {Tmax})")
        Tmin, Tmax = Tmax, Tmin

    return Tmin, Tmax, "; ".join(notes)


def fit_temperature_from_logits(
    accelerator: Accelerator,
    logits_cpu: np.ndarray,
    labels_cpu: np.ndarray,
    temp_min: float = DEFAULT_TEMP_MIN,
    temp_max: float = DEFAULT_TEMP_MAX,
) -> float:
    """Stable temperature fit (LBFGS on log(T) + grid fallback)."""
    optimal_T = 1.0
    Tmin, Tmax, bounds_note = _normalize_temperature_bounds(temp_min, temp_max)

    if accelerator.is_main_process:
        logits = torch.from_numpy(logits_cpu).to(accelerator.device).float()
        labels = torch.from_numpy(labels_cpu).to(accelerator.device).long()

        nll = torch.nn.CrossEntropyLoss()

        def loss_at_T(T: torch.Tensor) -> torch.Tensor:
            return nll(logits / T, labels)

        with torch.no_grad():
            base_loss = float(loss_at_T(torch.tensor([1.0], device=accelerator.device)).item())

        logT = torch.nn.Parameter(torch.log(torch.tensor([1.0], device=accelerator.device)))
        opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=120, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            T = torch.exp(logT).clamp(min=Tmin, max=Tmax)
            loss = loss_at_T(T)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            with torch.no_grad():
                T_lbfgs = float(torch.exp(logT).clamp(min=Tmin, max=Tmax).item())
                loss_lbfgs = float(loss_at_T(torch.tensor([T_lbfgs], device=accelerator.device)).item())
        except Exception:
            T_lbfgs, loss_lbfgs = 1.0, base_loss

        grid = np.exp(np.linspace(np.log(Tmin), np.log(Tmax), 31)).astype(np.float32)
        best_T, best_loss = T_lbfgs, loss_lbfgs
        if (not np.isfinite(best_loss)) or (best_loss > base_loss - 1e-6):
            best_T, best_loss = 1.0, base_loss
            for t in grid:
                with torch.no_grad():
                    l = float(loss_at_T(torch.tensor([float(t)], device=accelerator.device)).item())
                if l < best_loss:
                    best_loss, best_T = l, float(t)

        optimal_T = float(best_T)
        print(
            f"✓ Temperature T={optimal_T:.4f} | NLL@1.0={base_loss:.4f} -> NLL@T={best_loss:.4f} | "
            f"bounds=[{Tmin:.3g},{Tmax:.3g}]"
        )
        if bounds_note:
            print(f"[temperature] adjusted invalid bounds: {bounds_note}")

    # Broadcast
    T_tensor = torch.tensor([float(optimal_T)], device=accelerator.device)
    if accelerator.num_processes > 1:
        torch.distributed.broadcast(T_tensor, src=0)
    return float(T_tensor.item())


def fit_temperature_from_model(
    accelerator: Accelerator,
    model,
    val_loader,
    temp_min: float = DEFAULT_TEMP_MIN,
    temp_max: float = DEFAULT_TEMP_MAX,
) -> float:
    """Fit temperature for a (single) MIL model by collecting logits on val."""
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["num_chunks_per_doc"])
            all_logits.append(accelerator.gather_for_metrics(logits).detach().cpu())
            all_labels.append(accelerator.gather_for_metrics(batch["labels"]).detach().cpu())

    if accelerator.is_main_process:
        logits_np = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
        labels_np = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    else:
        logits_np = np.zeros((0, 2), dtype=np.float32)
        labels_np = np.zeros((0,), dtype=np.int64)

    return fit_temperature_from_logits(
        accelerator,
        logits_cpu=logits_np,
        labels_cpu=labels_np,
        temp_min=temp_min,
        temp_max=temp_max,
    )


def fit_temperature_from_hf_model(
    accelerator: Accelerator,
    model,
    val_loader,
    temp_min: float = DEFAULT_TEMP_MIN,
    temp_max: float = DEFAULT_TEMP_MAX,
) -> float:
    """Fit temperature for a HuggingFace sequence classifier by collecting logits on val."""
    model.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch.get("labels")
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            out = model(**inputs)
            logits = out.logits
            all_logits.append(accelerator.gather_for_metrics(logits).detach().cpu())
            if labels is None:
                raise ValueError("Batch missing 'labels' key for temperature fitting")
            all_labels.append(accelerator.gather_for_metrics(labels).detach().cpu())

    if accelerator.is_main_process:
        logits_np = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
        labels_np = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    else:
        logits_np = np.zeros((0, 2), dtype=np.float32)
        labels_np = np.zeros((0,), dtype=np.int64)

    return fit_temperature_from_logits(
        accelerator,
        logits_cpu=logits_np,
        labels_cpu=labels_np,
        temp_min=temp_min,
        temp_max=temp_max,
    )
