"""Gated Attention MIL classifier with optional Mahalanobis Distance Spectral Normalization (MD-SN)."""

import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from transformers import AutoModel, AutoConfig
import os

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


class AttentionMILClassifier(nn.Module):
    """
    Attention-based Multiple Instance Learning for document classification.
    Uses attention mechanism to weight chunks by importance.

    Two classification head modes:
      - Default: standard ``nn.Linear`` head.
      - MD-SN (``use_mdsn=True``): spectrally-normalised feature layer with
        a fitted Mahalanobis distance for post-hoc uncertainty estimation.
    """
    def __init__(
        self,
        checkpoint,
        num_labels: int = 2,
        dropout: float = 0.2,
        attention_dim: int = 128,
        load_pretrained: bool = True,
        attn_implementation: str | None = None,
        # Mahalanobis Distance + Spectral Normalization (MD-SN)
        use_mdsn: bool = False,
        mdsn_spectral_norm: bool = True,
        mdsn_n_power_iterations: int = 1,
        mdsn_cov_ridge: float = 1e-6,
        mdsn_unc_mode: str = "pred",
        **hf_model_kwargs,
    ):
        super().__init__()
        self.attn_implementation = attn_implementation

        # ----- Encoder initialisation -----
        if load_pretrained:
            try:
                kwargs = {"add_pooling_layer": False, **hf_model_kwargs}
                if attn_implementation is not None:
                    kwargs["attn_implementation"] = attn_implementation
                self.encoder = AutoModel.from_pretrained(checkpoint, **kwargs)
            except TypeError:
                config = AutoConfig.from_pretrained(checkpoint)
                if hasattr(config, "add_pooling_layer"):
                    config.add_pooling_layer = False
                if attn_implementation is not None and hasattr(config, "attn_implementation"):
                    config.attn_implementation = attn_implementation
                self.encoder = AutoModel.from_pretrained(checkpoint, config=config)
            except Exception:
                config = AutoConfig.from_pretrained(checkpoint)
                if hasattr(config, "add_pooling_layer"):
                    config.add_pooling_layer = False
                if attn_implementation is not None and hasattr(config, "attn_implementation"):
                    config.attn_implementation = attn_implementation
                self.encoder = AutoModel.from_pretrained(checkpoint, config=config)

            if hasattr(self.encoder, "pooler"):
                self.encoder.pooler = None
        else:
            try:
                config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
            except Exception:
                config = AutoConfig.from_pretrained(checkpoint)

            if hasattr(config, "add_pooling_layer"):
                config.add_pooling_layer = False
            if attn_implementation is not None and hasattr(config, "attn_implementation"):
                config.attn_implementation = attn_implementation

            self.encoder = AutoModel.from_config(config)
            if hasattr(self.encoder, "pooler"):
                self.encoder.pooler = None

        hidden_size = self.encoder.config.hidden_size

        # ----- Gated attention pooling (Ilse et al., 2018) -----
        self.attention_V = nn.Sequential(nn.Linear(hidden_size, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_size, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)

        self.dropout = nn.Dropout(dropout)

        # Store minimal config for save_pretrained
        self.model_checkpoint = checkpoint
        self.num_labels = int(num_labels)
        self.dropout_p = float(dropout)
        self.attention_dim = int(attention_dim)

        self.use_mdsn = bool(use_mdsn)

        # ----- Classification head -----
        if self.use_mdsn:
            # MD-SN head: spectrally-normalised feature layer + linear classifier.
            # Mahalanobis distance is fitted post-training via fit_mdsn().
            if str(mdsn_unc_mode) not in {"pred", "min", "margin"}:
                raise ValueError("mdsn_unc_mode must be one of ['pred','min','margin']")
            self.mdsn_unc_mode = str(mdsn_unc_mode)

            feat_linear = nn.Linear(hidden_size, hidden_size)
            if mdsn_spectral_norm:
                feat_linear = spectral_norm(feat_linear, n_power_iterations=int(mdsn_n_power_iterations))
            self.mdsn_feature = nn.Sequential(feat_linear, nn.GELU())
            self.mdsn_classifier = nn.Linear(hidden_size, num_labels)
            self.classifier = None

            # Fitted MD statistics (buffers so they are saved in state_dict)
            self.register_buffer("mdsn_centroids", torch.zeros(int(num_labels), hidden_size))
            self.register_buffer("mdsn_precision", torch.eye(hidden_size))
            self.register_buffer("mdsn_fitted", torch.zeros((), dtype=torch.uint8))
            self.mdsn_cov_ridge = float(mdsn_cov_ridge)
        else:
            # Default linear head (no uncertainty estimation)
            self.classifier = nn.Linear(hidden_size, num_labels)
            self.mdsn_feature = None
            self.mdsn_classifier = None

        self.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_pretrained(self, output_dir: str, *, safe_serialization: bool = False):
        os.makedirs(output_dir, exist_ok=True)

        cfg = {
            "model_checkpoint": self.model_checkpoint,
            "num_labels": self.num_labels,
            "dropout": self.dropout_p,
            "attention_dim": self.attention_dim,
            "attn_implementation": self.attn_implementation,
            "use_mdsn": bool(self.use_mdsn),
            "mdsn_cov_ridge": float(getattr(self, "mdsn_cov_ridge", 0.0)) if self.use_mdsn else None,
            "mdsn_unc_mode": str(getattr(self, "mdsn_unc_mode", "pred")) if self.use_mdsn else None,
        }
        # Clean None values (keep file compact and backward-friendly)
        cfg = {k: v for k, v in cfg.items() if v is not None}

        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        if safe_serialization:
            try:
                from safetensors.torch import save_file as safe_save_file
            except Exception as e:
                raise RuntimeError("safe_serialization=True requires safetensors") from e
            safe_save_file(self.state_dict(), os.path.join(output_dir, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # ------------------------------------------------------------------
    # MD-SN: fit and distance helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_mdsn(self, features: torch.Tensor, labels: torch.Tensor, *, cov_ridge: float | None = None):
        """Fit centroids + shared precision for MD-SN.

        `features`: (N,D) in the same space returned by forward(..., return_features=True) under use_mdsn.
        `labels`:   (N,) int64
        """
        if not self.use_mdsn:
            raise RuntimeError("fit_mdsn called but use_mdsn=False")

        X = features.detach().float()
        y = labels.detach().long()
        if X.ndim != 2:
            raise ValueError(f"features must be 2D (N,D), got shape={tuple(X.shape)}")
        if y.ndim != 1:
            y = y.view(-1)

        C = int(self.num_labels)
        D = int(X.shape[1])

        # Compute per-class centroids
        centroids = []
        for c in range(C):
            mask = (y == c)
            if int(mask.sum().item()) == 0:
                centroids.append(torch.zeros(D, device=X.device, dtype=X.dtype))
            else:
                centroids.append(X[mask].mean(dim=0))
        centroids = torch.stack(centroids, dim=0)  # (C,D)

        # Shared within-class covariance (pooled)
        cov = torch.zeros((D, D), device=X.device, dtype=torch.float32)
        n_total = max(int(X.shape[0]), 1)
        for c in range(C):
            mask = (y == c)
            if int(mask.sum().item()) == 0:
                continue
            diff = (X[mask] - centroids[c]).float()  # (Nc,D)
            cov = cov + diff.t().matmul(diff)
        cov = cov / float(n_total)

        # Ridge regularisation for numerical stability
        ridge = float(self.mdsn_cov_ridge if cov_ridge is None else cov_ridge)
        if ridge > 0:
            cov = cov + ridge * torch.eye(D, device=cov.device, dtype=cov.dtype)

        try:
            precision = torch.linalg.inv(cov)
        except Exception:
            precision = torch.linalg.pinv(cov)

        self.mdsn_centroids.copy_(centroids.to(self.mdsn_centroids.device, dtype=self.mdsn_centroids.dtype))
        self.mdsn_precision.copy_(precision.to(self.mdsn_precision.device, dtype=self.mdsn_precision.dtype))
        self.mdsn_fitted.fill_(1)

    def _mdsn_mahalanobis_sq_pred(self, z: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """Squared Mahalanobis distance to centroid of predicted class (fp32, autocast disabled)."""
        if preds.ndim != 1:
            preds = preds.view(-1)
        preds = preds.long()

        with torch.cuda.amp.autocast(enabled=False):
            z32 = z.float()
            mu32 = self.mdsn_centroids.to(device=z.device, dtype=torch.float32)  # (C,D)
            P32 = self.mdsn_precision.to(device=z.device, dtype=torch.float32)   # (D,D)

            # Symmetrize precision for stability
            P32 = 0.5 * (P32 + P32.transpose(-1, -2))

            preds = preds.clamp(min=0, max=mu32.shape[0] - 1)
            mu_y = mu32.index_select(0, preds)  # (B,D)
            delta = z32 - mu_y                  # (B,D)

            d2 = (delta.matmul(P32) * delta).sum(dim=-1)  # (B,)

            # sanitize
            d2 = torch.clamp(d2, min=0.0, max=1e12)
            d2 = torch.nan_to_num(d2, nan=1e12, posinf=1e12, neginf=1e12)

        return d2

    def _mdsn_mahalanobis_sq_min(self, z: torch.Tensor) -> torch.Tensor:
        """Squared Mahalanobis distance to closest centroid (min over classes), fp32, autocast disabled."""
        with torch.cuda.amp.autocast(enabled=False):
            z32 = z.float()
            mu32 = self.mdsn_centroids.to(device=z.device, dtype=torch.float32)  # (C,D)
            P32 = self.mdsn_precision.to(device=z.device, dtype=torch.float32)   # (D,D)
            P32 = 0.5 * (P32 + P32.transpose(-1, -2))

            # (B,C,D)
            delta = z32.unsqueeze(1) - mu32.unsqueeze(0)
            # (B,C) quadratic form
            d2 = torch.einsum("bcd,dd,bcd->bc", delta, P32, delta)

            d2 = torch.clamp(d2, min=0.0, max=1e12)
            d2 = torch.nan_to_num(d2, nan=1e12, posinf=1e12, neginf=1e12)

            d2_min = d2.min(dim=1).values  # (B,)

        return d2_min

    def _mdsn_distance(self, z: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Return a *finite* distance-like uncertainty score (float32)."""
        if (not hasattr(self, "mdsn_fitted")) or int(self.mdsn_fitted.item()) == 0:
            return torch.zeros(z.shape[0], device=z.device, dtype=torch.float32)

        mode = str(getattr(self, "mdsn_unc_mode", "pred"))
        if mode == "pred":
            preds = torch.argmax(logits, dim=-1)
            d2 = self._mdsn_mahalanobis_sq_pred(z, preds)
        else:
            d2 = self._mdsn_mahalanobis_sq_min(z)

        # Use sqrt so scale matches typical "distance" semantics
        d = torch.sqrt(d2 + 1e-12)

        # final sanitize
        d = torch.nan_to_num(d, nan=1e6, posinf=1e6, neginf=1e6)
        return d.to(dtype=torch.float32)

    def _mdsn_all_d2(self, z: torch.Tensor) -> torch.Tensor:
        """Return squared Mahalanobis distances to all centroids: (B, C), fp32, autocast disabled."""
        with torch.cuda.amp.autocast(enabled=False):
            z32 = z.float()
            mu32 = self.mdsn_centroids.to(device=z.device, dtype=torch.float32)  # (C,D)
            P32 = self.mdsn_precision.to(device=z.device, dtype=torch.float32)   # (D,D)
            P32 = 0.5 * (P32 + P32.transpose(-1, -2))

            delta = z32.unsqueeze(1) - mu32.unsqueeze(0)  # (B,C,D)
            d2 = torch.einsum("bcd,dd,bcd->bc", delta, P32, delta)  # (B,C)

            d2 = torch.clamp(d2, min=0.0, max=1e12)
            d2 = torch.nan_to_num(d2, nan=1e12, posinf=1e12, neginf=1e12)
        return d2

    def _mdsn_uncertainty(self, z: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Return a finite float32 uncertainty vector (B,). Modes: pred|min|margin."""
        if (not hasattr(self, "mdsn_fitted")) or int(self.mdsn_fitted.item()) == 0:
            return torch.zeros(z.size(0), device=z.device, dtype=torch.float32)

        mode = str(getattr(self, "mdsn_unc_mode", "pred"))

        d2_all = self._mdsn_all_d2(z)               # (B,C)
        d_all = torch.sqrt(d2_all + 1e-12)          # (B,C)

        if mode == "min":
            u = d_all.min(dim=1).values             # (B,)
        else:
            preds = torch.argmax(logits, dim=-1)    # (B,)
            d_pred = d_all.gather(1, preds.view(-1, 1)).squeeze(1)

            if mode == "margin":
                # distance margin: d_pred - best_other
                mask = torch.ones_like(d_all, dtype=torch.bool)
                mask.scatter_(1, preds.view(-1, 1), False)
                d_other = d_all.masked_fill(~mask, float("inf")).min(dim=1).values
                u = d_pred - d_other
            else:
                # default: predicted-class distance
                u = d_pred

        u = torch.nan_to_num(u, nan=1e6, posinf=1e6, neginf=-1e6)
        return u.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Encoding & Aggregation
    # ------------------------------------------------------------------

    def _encode_chunks(self, input_ids, attention_mask, clear_cuda_cache: bool = False) -> torch.Tensor:
        """Encode each chunk independently through the transformer; return CLS embeddings."""
        B, L = input_ids.shape

        # HF configs vary; cap at 512 as the effective maximum.
        cfg_max_pos = int(getattr(self.encoder.config, "max_position_embeddings", 512))
        max_len = min(cfg_max_pos, 512)

        if L > max_len:
            input_ids = input_ids[:, :max_len].contiguous()
            attention_mask = attention_mask[:, :max_len].contiguous()
            B, L = input_ids.shape

        # Build position_ids for the (possibly truncated) length
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        # Pass dedicated clones to the encoder to avoid "modified by inplace op"
        # errors during backward if something else touches the original tensors.
        ids_enc = input_ids.clone()
        mask_enc = attention_mask.clone()
        pos_enc = position_ids.clone()

        try:
            outputs = self.encoder(
                input_ids=ids_enc,
                attention_mask=mask_enc,
                position_ids=pos_enc,
            )
        except TypeError:
            # some models don't accept position_ids
            outputs = self.encoder(input_ids=ids_enc, attention_mask=mask_enc)

        chunk_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        del outputs
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return chunk_embeddings  # [total_chunks, hidden]

    def _aggregate_doc(self, doc_chunks: torch.Tensor):
        """Gated attention aggregation: weight chunks and produce a single document embedding."""
        # doc_chunks: [num_chunks, hidden]
        A_V = self.attention_V(doc_chunks)        # [num_chunks, att_dim]
        A_U = self.attention_U(doc_chunks)        # [num_chunks, att_dim]
        A = self.attention_w(A_V * A_U)           # [num_chunks, 1]
        attention_weights = F.softmax(A, dim=0)   # [num_chunks, 1]
        doc_embedding = torch.sum(attention_weights * doc_chunks, dim=0)  # [hidden]
        return doc_embedding, attention_weights.squeeze(-1)               # [hidden], [num_chunks]

    # ------------------------------------------------------------------
    # Gradient checkpointing helpers
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        if hasattr(self.encoder, "config"):
            self.encoder.config.use_cache = False

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
            self.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path, device="cpu", attn_implementation: str | None = None):
        cfg_path = os.path.join(model_path, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.json in {model_path}")

        with open(cfg_path, "r") as f:
            config = json.load(f)

        if attn_implementation is None:
            attn_implementation = config.get("attn_implementation", None)

        model = cls(
            config["model_checkpoint"],
            num_labels=config["num_labels"],
            dropout=config.get("dropout", 0.2),
            attention_dim=config.get("attention_dim", 128),
            load_pretrained=False,
            attn_implementation=attn_implementation,
            use_mdsn=bool(config.get("use_mdsn", False)),
            mdsn_cov_ridge=float(config.get("mdsn_cov_ridge", 1e-6)),
            mdsn_unc_mode=str(config.get("mdsn_unc_mode", "margin")),
        )

        candidates = [
            os.path.join(model_path, "model.safetensors"),
            os.path.join(model_path, "pytorch_model.safetensors"),
            os.path.join(model_path, "pytorch_model.bin"),
        ]
        weights_path = next((p for p in candidates if os.path.exists(p)), None)
        if weights_path is None:
            raise FileNotFoundError(f"No weights found in {model_path}. Tried:\n" + "\n".join(candidates))

        if weights_path.endswith(".safetensors"):
            if safe_load_file is None:
                raise RuntimeError("safetensors is not available but a .safetensors file was found.")
            state_dict = safe_load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")

        model.load_state_dict(state_dict, strict=False)

        # Guardrail: if MD-SN buffers are non-finite, distances will be NaN/Inf.
        # Reset to safe defaults and mark as not fitted.
        if bool(getattr(model, "use_mdsn", False)) and hasattr(model, "mdsn_precision") and hasattr(model, "mdsn_centroids"):
            with torch.no_grad():
                prec_ok = torch.isfinite(model.mdsn_precision).all().item()
                cent_ok = torch.isfinite(model.mdsn_centroids).all().item()
                if not (prec_ok and cent_ok):
                    model.mdsn_precision.copy_(torch.eye(model.mdsn_precision.shape[0], device=model.mdsn_precision.device, dtype=model.mdsn_precision.dtype))
                    model.mdsn_centroids.zero_()
                    if hasattr(model, "mdsn_fitted"):
                        model.mdsn_fitted.zero_()

        model.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids,
        attention_mask,
        num_chunks_per_doc=None,
        return_features: bool = False,
        return_uncertainty: bool = False,
        clear_cuda_cache: bool = False,
    ):
        # IMPORTANT: never mutate caller-provided tensors in-place.
        # Also avoid any in-place ops on tensors used by embedding backward.
        input_ids = input_ids.contiguous()
        attention_mask = attention_mask.contiguous()

        # If you need to sanitize padding, do it OUT-OF-PLACE like this:
        pad_id = getattr(getattr(self, "encoder", None), "config", None)
        pad_id = getattr(pad_id, "pad_token_id", 0)

        # Ensure padded positions are pad tokens (out-of-place)
        input_ids = torch.where(attention_mask.bool(), input_ids, torch.full_like(input_ids, pad_id))

        if num_chunks_per_doc is None:
            raise ValueError("num_chunks_per_doc must be provided for MIL models.")

        if isinstance(num_chunks_per_doc, torch.Tensor):
            num_chunks_per_doc = num_chunks_per_doc.tolist()

        total_chunks = int(sum(num_chunks_per_doc))
        if input_ids.shape[0] != total_chunks:
            raise ValueError(
                f"Chunk/doc mismatch: input_ids has {input_ids.shape[0]} chunks "
                f"but sum(num_chunks_per_doc)={total_chunks}."
            )

        # 1) Encode all chunks through the transformer
        chunk_embeddings = self._encode_chunks(
            input_ids=input_ids,
            attention_mask=attention_mask,
            clear_cuda_cache=clear_cuda_cache,
        )

        # 2) Aggregate per document via gated attention (variable chunks per doc)
        batch_attention_weights = []
        batch_doc_embeddings = []

        start_idx = 0
        for n in num_chunks_per_doc:
            end_idx = start_idx + int(n)
            doc_chunks = chunk_embeddings[start_idx:end_idx]  # [n, hidden]
            doc_embedding, attn = self._aggregate_doc(doc_chunks)
            batch_doc_embeddings.append(doc_embedding)
            batch_attention_weights.append(attn)
            start_idx = end_idx

        doc_embeddings = torch.stack(batch_doc_embeddings, dim=0)  # [B, hidden]
        doc_embeddings_d = self.dropout(doc_embeddings)            # [B, hidden]

        # 3) Classification head
        if self.use_mdsn:
            z = self.mdsn_feature(doc_embeddings_d)   # (B,D) feature space used for MD stats
            logits = self.mdsn_classifier(z)          # (B,C)

            if return_uncertainty:
                u = self._mdsn_uncertainty(z, logits)    # (B,) finite float32
                if return_features:
                    return logits, batch_attention_weights, z, u
                return logits, batch_attention_weights, u

            if return_features:
                return logits, batch_attention_weights, z
            return logits, batch_attention_weights

        else:
            # Default linear head
            logits = self.classifier(doc_embeddings_d)
            if return_features:
                return logits, batch_attention_weights, doc_embeddings
            return logits, batch_attention_weights


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for calibrating classifier logits (Guo et al., 2017)."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, max_iter=50, lr=0.01):
        device = self.temperature.device
        logits = torch.tensor(logits, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        return self.temperature.item()
