#!/usr/bin/env python3
"""Generate interactive HTML explanations for MIL-UQ predictions.

- It expects a MIL checkpoint from this repository.
- It consumes the authoritative outputs from evaluate.py (predictions.csv and
  metrics.json, optionally calibration_bundle.npz) so the triage decision shown
  in the HTML matches the saved evaluation artifacts.
- It does not include LLM explainers, UMLS normalization, EuroTEST mapping, or
  private patient-level TSV helpers.

Recommended workflow:

1. Run evaluate.py on the target split.
2. Run this script with the same test CSV and seed used in evaluation.
3. Review the generated HTML files in --output_dir.
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import os
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from models.attention_mil import AttentionMILClassifier
from uq_eval.data.text import chunk_text, clean_text


_ESP_STOPWORDS_MIN = {
    "a", "al", "algo", "algunos", "ante", "antes", "con", "como", "contra",
    "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el",
    "ella", "ellas", "ellos", "en", "entre", "era", "es", "esa", "ese", "eso",
    "esta", "este", "esto", "fue", "ha", "han", "hay", "la", "las", "le",
    "les", "lo", "los", "mas", "me", "mi", "mis", "muy", "nos", "o", "para",
    "pero", "por", "porque", "que", "se", "si", "sobre", "su", "sus",
    "tambien", "te", "tiene", "todo", "tu", "tus", "un", "una", "uno", "y",
    "ya", "anonimo",
}
_CLINICAL_STOPWORDS = {
    "num", "paciente", "refiere", "antecedentes", "historia", "acude",
    "presenta", "realiza", "observacion", "analitica", "jc", "ap", "map",
    "tratamiento", "exploracion", "fisica", "diagnostico", "plan",
    "evolucion", "motivo", "consulta", "enfermedad", "actual", "previos",
    "solicito", "estudio", "clinica", "planta", "ta", "fc", "fr", "sat",
    "o2", "temp", "temperatura", "t", "lpm", "rpm", "mmhg", "mg", "dl",
    "ef", "nd", "nr", "cv", "ac", "mvc", "rha", "eess", "eeii", "rx",
    "tx", "dx", "sospecha", "orientacion", "aparato", "aparatos", "sistema",
    "general", "estado", "regular", "bien", "mal", "sin", "con", "familiar",
    "solicitamos", "pautamos", "indicamos", "recomienda", "recomiendo", "sugiere",
    "sugiero", "aceptable", "ingreso", "mgmi", "ingresa", "medicina", "interna",
    "tac", "mc", "analitica", "analiticas", "pcr", "nota", "resultado",
    "resultados", "juicio", "clinico",
}
STOPWORDS = _ESP_STOPWORDS_MIN | _CLINICAL_STOPWORDS


def _normalize_id(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        ivalue = int(value)
        return str(ivalue) if float(ivalue) == float(value) else str(value)
    return str(value).strip()


def _row_doc_key(row: Any, fallback_index: int) -> tuple[str, str]:
    doc_id = ""
    for key in ("paciente_id", "id", "patient_id", "doc_id"):
        try:
            value = row.get(key, None)
        except Exception:
            value = None
        if value is None:
            continue
        normalized = _normalize_id(value)
        if normalized:
            doc_id = normalized
            break
    if not doc_id:
        doc_id = str(int(fallback_index))
    try:
        group = row.get("group", "")
    except Exception:
        group = ""
    return doc_id, str(group).strip()


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return cleaned.strip("._") or "sample"


def _parse_notes_cell(value: Any) -> list[Any] | str:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""
        if raw[0] in "[(":
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(raw)
                except Exception:
                    continue
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
        return value
    return str(value)


def clean_token(token: str) -> str:
    token = token.replace("▁", "").replace("##", "").replace("Ġ", "").replace("Ċ", "").strip()
    token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
    return token.lower()


def is_valid_token(token: str, min_length: int = 2) -> bool:
    t = token.lower()
    if (
        len(t) < min_length
        or t in STOPWORDS
        or re.match(r"^[\d\W]+$", t)
        or t in {"[cls]", "[sep]", "[pad]", "[unk]"}
    ):
        return False
    return True


def _ngrams(seq: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(seq) < n:
        return []
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def _feature_key(text: Any) -> str:
    parts = [clean_token(tok) for tok in str(text or "").split()]
    return " ".join(tok for tok in parts if tok)


def _collect_top_features(
    unigrams: list[dict[str, Any]] | None,
    ngrams: list[dict[str, Any]] | None,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for items, kind in ((ngrams or [], "N-gram"), (unigrams or [], "Unigram")):
        for item in items:
            token = re.sub(r"\s+", " ", str(item.get("token", "")).strip())
            if not token:
                continue
            try:
                importance = float(item.get("importance", 0.0))
            except Exception:
                importance = 0.0
            ranked.append({"token": token, "importance": importance, "kind": kind})

    ranked.sort(key=lambda item: abs(float(item["importance"])), reverse=True)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    covered_unigrams: set[str] = set()
    for item in ranked:
        key = _feature_key(item["token"])
        if not key or key in seen:
            continue
        if item["kind"] == "Unigram" and key in covered_unigrams:
            continue
        out.append(item)
        seen.add(key)
        if item["kind"] == "N-gram":
            covered_unigrams.update(tok for tok in key.split() if tok)
        if len(out) >= int(limit):
            break
    return out


def _load_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        return AutoTokenizer.from_pretrained(cfg["model_checkpoint"])


def _resolve_file_path(raw_path: str, *, directory_files: tuple[str, ...] = (), candidates: tuple[str, ...] = ()) -> str:
    search_space: list[str] = []

    def _add(path: str) -> None:
        value = str(path or "").strip()
        if value:
            search_space.append(os.path.abspath(value))

    if raw_path:
        abs_path = os.path.abspath(raw_path)
        if os.path.isdir(abs_path):
            for filename in directory_files:
                _add(os.path.join(abs_path, filename))
        else:
            _add(abs_path)

    for candidate in candidates:
        _add(candidate)

    seen: set[str] = set()
    for candidate in search_space:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate
    return ""


def _resolve_predictions_csv_path(predictions_csv: str, eval_output_dir: str) -> str:
    eval_dir = os.path.abspath(eval_output_dir) if eval_output_dir else ""
    return _resolve_file_path(
        predictions_csv,
        directory_files=("predictions.csv", os.path.join("predictions", "predictions.csv")),
        candidates=(
            os.path.join(eval_dir, "predictions.csv") if eval_dir else "",
            os.path.join(eval_dir, "predictions", "predictions.csv") if eval_dir else "",
        ),
    )


def _resolve_metrics_json_path(metrics_json: str, eval_output_dir: str, predictions_csv: str) -> str:
    eval_dir = os.path.abspath(eval_output_dir) if eval_output_dir else ""
    pred_dir = os.path.dirname(os.path.abspath(predictions_csv)) if predictions_csv else ""
    return _resolve_file_path(
        metrics_json,
        directory_files=("metrics.json", "metric.json"),
        candidates=(
            os.path.join(pred_dir, "metrics.json") if pred_dir else "",
            os.path.join(pred_dir, "metric.json") if pred_dir else "",
            os.path.join(eval_dir, "metrics.json") if eval_dir else "",
            os.path.join(eval_dir, "metric.json") if eval_dir else "",
        ),
    )


def _resolve_calibration_bundle_path(calibration_bundle: str, eval_output_dir: str, predictions_csv: str) -> str:
    eval_dir = os.path.abspath(eval_output_dir) if eval_output_dir else ""
    pred_dir = os.path.dirname(os.path.abspath(predictions_csv)) if predictions_csv else ""
    return _resolve_file_path(
        calibration_bundle,
        directory_files=("calibration_bundle.npz",),
        candidates=(
            os.path.join(pred_dir, "calibration_bundle.npz") if pred_dir else "",
            os.path.join(eval_dir, "calibration_bundle.npz") if eval_dir else "",
        ),
    )


def _load_metrics_thresholds(path: str) -> tuple[float, float] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        p_low = float(payload["p_low"])
        p_high = float(payload["p_high"])
    except Exception:
        return None
    if not (np.isfinite(p_low) and np.isfinite(p_high)):
        return None
    return p_low, p_high


def _load_bundle_meta(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    payload = np.load(path, allow_pickle=True)
    if "meta_json" not in payload:
        return {}
    meta_raw = payload["meta_json"].tolist()
    if isinstance(meta_raw, list) and meta_raw:
        meta_raw = meta_raw[0]
    try:
        return json.loads(str(meta_raw))
    except Exception:
        return {}


def _as_int_key_dict(d: dict[str, Any]) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in (d or {}).items():
        try:
            out[int(key)] = float(value)
        except Exception:
            continue
    return out


def _predictions_df_is_row_aligned(df: pd.DataFrame, predictions_df: pd.DataFrame | None) -> bool:
    if predictions_df is None or len(predictions_df) != len(df):
        return False
    for index in range(len(df)):
        if _row_doc_key(df.iloc[int(index)], int(index)) != _row_doc_key(predictions_df.iloc[int(index)], int(index)):
            return False
    return True


def _load_predictions_lookup(predictions_csv: str) -> tuple[pd.DataFrame | None, dict[tuple[str, str], dict[str, Any]]]:
    if not predictions_csv or not os.path.isfile(predictions_csv):
        return None, {}
    predictions_df = pd.read_csv(predictions_csv)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in predictions_df.iterrows():
        key = _row_doc_key(row, -1)
        if key[0]:
            lookup[key] = row.to_dict()
    return predictions_df, lookup


def _all_indices(df: pd.DataFrame) -> list[int]:
    return list(range(len(df)))


def _random_sample_indices(indices: list[int], n_examples: int, seed: int) -> list[int]:
    if len(indices) == 0 or int(n_examples) <= 0:
        return []
    rng = random.Random(int(seed))
    if len(indices) <= int(n_examples):
        return [int(i) for i in indices]
    return [int(i) for i in rng.sample(indices, int(n_examples))]


def _sort_indices_by_prob_pos(
    *,
    indices: list[int],
    df: pd.DataFrame,
    predictions_lookup: dict[tuple[str, str], dict[str, Any]] | None,
    descending: bool = True,
) -> list[int]:
    if not indices:
        return []
    if not predictions_lookup:
        return [int(i) for i in indices]

    scored: list[tuple[int, float | None]] = []
    for index in indices:
        row = df.iloc[int(index)]
        match = predictions_lookup.get(_row_doc_key(row, int(index)))
        prob: float | None = None
        if match is not None:
            value = match.get("prob_pos")
            if value is not None and not pd.isna(value):
                try:
                    prob = float(value)
                except Exception:
                    prob = None
        scored.append((int(index), prob))

    with_prob = [(idx, prob) for idx, prob in scored if prob is not None]
    without_prob = [idx for idx, prob in scored if prob is None]
    with_prob.sort(key=lambda item: float(item[1]), reverse=bool(descending))
    return [idx for idx, _ in with_prob] + without_prob


def _select_explain_indices_from_mode(
    *,
    df: pd.DataFrame,
    n_examples: int,
    mode: str,
    seed: int,
    predictions_lookup: dict[tuple[str, str], dict[str, Any]] | None,
) -> list[int]:
    idx_all = _all_indices(df)
    if len(idx_all) == 0 or int(n_examples) <= 0:
        return []

    mode = str(mode)
    if mode == "deferred_first":
        deferred: list[int] = []
        non_deferred: list[int] = []
        if predictions_lookup:
            for index in idx_all:
                row = df.iloc[int(index)]
                match = predictions_lookup.get(_row_doc_key(row, int(index)))
                if match is None:
                    non_deferred.append(int(index))
                    continue
                is_deferred = False
                try:
                    tri = match.get("pred_trinary")
                    if tri is not None and not pd.isna(tri) and int(float(tri)) == 1:
                        deferred.append(int(index))
                        is_deferred = True
                except Exception:
                    pass
                if not is_deferred:
                    category = str(match.get("category", "")).lower()
                    if "defer" in category or "complex" in category:
                        deferred.append(int(index))
                        is_deferred = True
                if not is_deferred:
                    non_deferred.append(int(index))
        if not deferred:
            return _random_sample_indices(idx_all, int(n_examples), int(seed))

        ordered = _sort_indices_by_prob_pos(
            indices=deferred,
            df=df,
            predictions_lookup=predictions_lookup,
            descending=True,
        ) + _sort_indices_by_prob_pos(
            indices=non_deferred,
            df=df,
            predictions_lookup=predictions_lookup,
            descending=True,
        )
        return [int(index) for index in ordered[: int(n_examples)]]

    if mode == "top_uncertainty" and predictions_lookup:
        scored: list[tuple[int, float]] = []
        for index in idx_all:
            row = df.iloc[int(index)]
            match = predictions_lookup.get(_row_doc_key(row, int(index)))
            if match is None:
                continue
            value = match.get("uncertainty")
            if value is None or pd.isna(value):
                continue
            try:
                scored.append((int(index), float(value)))
            except Exception:
                continue
        if scored:
            scored.sort(key=lambda item: item[1], reverse=True)
            return [int(index) for index, _ in scored[: int(n_examples)]]

    if mode == "clear_errors" and predictions_lookup:
        selected: list[int] = []

        def _to_label_int(value: Any) -> int | None:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            if isinstance(value, (int, np.integer)):
                return int(value)
            text = str(value).strip().upper()
            if text == "NO_SOSPECHA":
                return 0
            if text == "SOSPECHA":
                return 1
            try:
                return int(float(value))
            except Exception:
                return None

        for index in idx_all:
            row = df.iloc[int(index)]
            match = predictions_lookup.get(_row_doc_key(row, int(index)))
            if match is None:
                continue
            if str(match.get("category", "")).strip().lower() != "clear":
                continue
            label_int = _to_label_int(match.get("label"))
            try:
                pred_tri = int(float(match.get("pred_trinary")))
            except Exception:
                pred_tri = None
            if label_int is None or pred_tri is None:
                continue
            if (label_int == 0 and pred_tri == 2) or (label_int == 1 and pred_tri == 0):
                selected.append(int(index))

        if selected:
            ordered = _sort_indices_by_prob_pos(
                indices=selected,
                df=df,
                predictions_lookup=predictions_lookup,
                descending=True,
            )
            return [int(index) for index in ordered[: int(n_examples)]]

    return _random_sample_indices(idx_all, int(n_examples), int(seed))


def _chunks_with_metadata(notes: Any, tokenizer, *, max_chunks: int = 64) -> tuple[list[str], list[dict[str, str]]]:
    all_chunks: list[str] = []
    all_meta: list[dict[str, str]] = []

    note_list = notes if isinstance(notes, list) else [notes]
    for note in note_list:
        if isinstance(note, dict):
            source_text = str(note.get("text", note.get("notes", "")))
            raw_text = str(note.get("raw_text", source_text))
            meta = {
                "date": str(note.get("date", note.get("fecha", "Unknown date"))),
                "type": str(note.get("type", note.get("tipo", "Clinical Note"))),
                "raw_text": raw_text,
            }
        else:
            source_text = "" if note is None else str(note)
            raw_text = source_text
            meta = {
                "date": "Unknown date",
                "type": "Clinical Note",
                "raw_text": raw_text,
            }

        cleaned = clean_text(source_text)
        if not cleaned:
            continue

        sub_chunks = chunk_text(cleaned, tokenizer)
        display_chunks = list(sub_chunks)
        if raw_text.strip():
            raw_display_chunks = chunk_text(raw_text, tokenizer)
            if len(raw_display_chunks) == len(sub_chunks):
                display_chunks = raw_display_chunks

        all_chunks.extend(sub_chunks)
        for chunk_text_clean, chunk_text_display in zip(sub_chunks, display_chunks):
            chunk_meta = dict(meta)
            chunk_meta["chunk_text"] = chunk_text_clean
            chunk_meta["display_chunk_text"] = chunk_text_display
            all_meta.append(chunk_meta)

    if not all_chunks:
        all_chunks = [""]
        all_meta = [{"date": "", "type": "", "raw_text": "", "chunk_text": "", "display_chunk_text": ""}]

    if len(all_chunks) > int(max_chunks):
        n_head = int(max_chunks * 0.25)
        n_tail = int(max_chunks) - n_head
        all_chunks = all_chunks[:n_head] + all_chunks[-n_tail:]
        all_meta = all_meta[:n_head] + all_meta[-n_tail:]

    return all_chunks, all_meta


@dataclass
class ChunkUncertainty:
    mean_prob: float
    uncertainty: float


def _chunk_stats_from_attention(mil_weights: np.ndarray, doc_prob: float) -> list[ChunkUncertainty]:
    weights = np.asarray(mil_weights, dtype=np.float64)
    mean_weight = float(np.mean(weights))
    if mean_weight < 1e-12:
        return [ChunkUncertainty(mean_prob=float(doc_prob), uncertainty=0.0) for _ in weights]

    doc_prob = float(np.clip(doc_prob, 1e-6, 1.0 - 1e-6))
    doc_logit = float(np.log(doc_prob / (1.0 - doc_prob)))
    weight_ratio = weights / mean_weight
    chunk_logits = doc_logit * weight_ratio
    chunk_probs = 1.0 / (1.0 + np.exp(-chunk_logits))
    return [ChunkUncertainty(mean_prob=float(prob), uncertainty=0.0) for prob in chunk_probs]


class AttentionMILExplainer:
    def __init__(self, model, tokenizer, device: torch.device, *, temperature: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = float(max(float(temperature), 1e-6))
        self.model.eval()

    def _disable_dropout(self) -> None:
        self.model.eval()

    def get_chunk_data(self, chunks: list[str]) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        inputs = self.tokenizer(
            chunks,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        self._disable_dropout()
        with torch.no_grad():
            outputs = self.model(
                inputs["input_ids"],
                inputs["attention_mask"],
                num_chunks_per_doc=[len(chunks)],
                return_features=False,
                return_uncertainty=False,
            )
            logits = outputs[0].cpu().view(-1)
            probs = torch.softmax(logits / self.temperature, dim=-1).numpy()
            weights = outputs[1][0].detach().cpu().numpy().flatten()
        return probs, weights, logits

    def get_token_attributions(self, chunks: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            scores = np.zeros(len(tokens), dtype=np.float32)
            encoder_cfg = None
            prev_attn_impl = None
            try:
                encoder_cfg = getattr(getattr(self.model, "encoder", None), "config", None)
                prev_attn_impl = getattr(encoder_cfg, "_attn_implementation", None)
                if encoder_cfg is not None:
                    encoder_cfg._attn_implementation = "eager"
                with torch.no_grad():
                    attentions = getattr(
                        self.model.encoder(**inputs, output_attentions=True),
                        "attentions",
                        None,
                    )
                if attentions:
                    imp = (
                        0.5 * attentions[-1][0, :, 0, :].mean(0).cpu().numpy()
                        + 0.5 * attentions[-1][0].mean((0, 1)).cpu().numpy()
                    )
                    if np.sum(imp) > 0:
                        imp = imp / np.sum(imp)
                    if len(imp) == len(tokens):
                        scores = imp.astype(np.float32)
            except Exception:
                pass
            finally:
                if encoder_cfg is not None:
                    encoder_cfg._attn_implementation = prev_attn_impl
            out.append({"text": chunk, "tokens": tokens, "scores": scores.tolist()})
        return out

    def _word_importance_from_offsets(self, text: str, tokens: list[str], scores: list[float]) -> dict[str, float]:
        enc = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            max_length=512,
        )
        offsets = enc["offset_mapping"][0].tolist()
        if len(offsets) != len(tokens):
            return {}

        word_scores: dict[str, float] = defaultdict(float)
        word_counts: dict[str, int] = defaultdict(int)
        current_start: int | None = None
        current_end: int | None = None
        current_score = 0.0

        def flush() -> None:
            nonlocal current_start, current_end, current_score
            if current_start is not None and current_end is not None and current_end > current_start:
                word = clean_token(text[current_start:current_end])
                if is_valid_token(word):
                    word_scores[word] += float(current_score)
                    word_counts[word] += 1
            current_start, current_end, current_score = None, None, 0.0

        for index, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                flush()
                continue
            if current_start is None:
                current_start, current_end, current_score = int(start), int(end), float(scores[index])
                continue
            gap = text[int(current_end):int(start)]
            if gap == "":
                current_end = int(end)
                current_score += float(scores[index])
            else:
                flush()
                current_start, current_end, current_score = int(start), int(end), float(scores[index])
        flush()

        for word, count in list(word_counts.items()):
            if count > 1:
                word_scores[word] /= count
        return dict(word_scores)

    def _compute_word_impacts(
        self,
        chunks: list[str],
        top_words: list[str],
        base_prob: float,
        *,
        base_logits: torch.Tensor,
    ) -> dict[str, float]:
        impacts: dict[str, float] = {}
        valid_words: list[str] = []
        masked_chunks_list: list[list[str]] = []
        base_margin = float(base_logits[1] - base_logits[0]) if len(base_logits) >= 2 else None

        for word in top_words:
            masked_chunks = [re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE).sub("", chunk) for chunk in chunks]
            if masked_chunks != chunks:
                valid_words.append(word)
                masked_chunks_list.append(masked_chunks)
            else:
                impacts[word] = 0.0

        self._disable_dropout()
        for word, masked_chunks in zip(valid_words, masked_chunks_list):
            if not masked_chunks:
                impacts[word] = 0.0
                continue
            masked_inputs = self.tokenizer(
                masked_chunks,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(
                    masked_inputs["input_ids"],
                    masked_inputs["attention_mask"],
                    num_chunks_per_doc=[len(masked_chunks)],
                )[0].cpu().view(-1)
            if base_margin is not None:
                delta = float(base_margin) - float(logits[1] - logits[0])
            else:
                delta = float(base_prob) - float(torch.softmax(logits, dim=-1)[1])
            impacts[word] = delta
        return impacts

    def aggregate_attributions(
        self,
        chunks: list[str],
        token_attr: list[dict[str, Any]],
        mil_weights: np.ndarray,
        base_doc_prob: float,
        *,
        base_logits: torch.Tensor,
    ) -> dict[str, Any]:
        weights = (
            mil_weights / np.sum(mil_weights)
            if np.sum(mil_weights) > 0
            else np.ones_like(mil_weights) / max(len(mil_weights), 1)
        )
        raw_importance: dict[str, float] = defaultdict(float)
        for chunk, attr, weight in zip(chunks, token_attr, weights):
            for word, score in self._word_importance_from_offsets(chunk, attr["tokens"], attr["scores"]).items():
                raw_importance[word] += float(score) * float(weight)

        candidates = sorted(raw_importance.items(), key=lambda item: item[1], reverse=True)[:50]
        deltas = self._compute_word_impacts(
            chunks,
            [word for word, _ in candidates],
            base_doc_prob,
            base_logits=base_logits,
        )

        positives: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        for word, attention_score in candidates:
            delta = deltas.get(word, 0.0)
            if abs(delta) < 1e-4:
                continue
            if delta > 0:
                positives.append({"token": word, "importance": attention_score})
            else:
                negatives.append({"token": word, "importance": attention_score})

        raw_ngram_importance: dict[str, float] = defaultdict(float)
        for chunk, attr, weight in zip(chunks, token_attr, weights):
            word_scores = self._word_importance_from_offsets(chunk, attr["tokens"], attr["scores"])
            chunk_words = [clean_token(tok) for tok in re.split(r"[\s\W]+", chunk) if tok]
            chunk_words = [tok for tok in chunk_words if is_valid_token(tok)]
            for n_value in (2, 3):
                for gram in _ngrams(chunk_words, n_value):
                    gram_str = " ".join(gram)
                    gram_score = sum(word_scores.get(token, 0.0) for token in gram) / n_value
                    raw_ngram_importance[gram_str] += gram_score * float(weight)

        positive_ngrams: list[dict[str, Any]] = []
        negative_ngrams: list[dict[str, Any]] = []
        for gram_str, attention_score in sorted(raw_ngram_importance.items(), key=lambda item: item[1], reverse=True)[:40]:
            constituent_deltas = [deltas.get(token, 0.0) for token in gram_str.split()]
            if not constituent_deltas:
                continue
            dominant_delta = max(constituent_deltas, key=abs)
            if abs(dominant_delta) < 1e-4:
                continue
            if dominant_delta > 0:
                positive_ngrams.append({"token": gram_str, "importance": attention_score})
            else:
                negative_ngrams.append({"token": gram_str, "importance": attention_score})

        return {
            "top_positive_tokens": positives,
            "top_negative_tokens": negatives,
            "top_positive_ngrams": positive_ngrams,
            "top_negative_ngrams": negative_ngrams,
        }

    def _beautify_text(self, text: str) -> str:
        if not text:
            return text
        acronyms = ["vih", "sida", "ets", "vhc", "vhb", "pcr", "its", "urgencias", "uci"]
        words = text.split()
        for index, word in enumerate(words):
            if word.lower() in acronyms:
                words[index] = word.upper()
        if words:
            words[0] = words[0].capitalize()
        chunks: list[str] = []
        chunk_size = 25
        for index in range(0, len(words), chunk_size):
            segment = " ".join(words[index:index + chunk_size])
            if index > 0 and segment:
                segment = segment[0].capitalize() + segment[1:]
            chunks.append(segment)
        return "<br><br>".join(chunks)

    def generate_html_report(
        self,
        *,
        chunks: list[str],
        chunk_metas: list[dict[str, str]],
        chunk_stats: list[ChunkUncertainty],
        mil_weights: np.ndarray,
        output_path: Path,
        doc_probs: np.ndarray,
        aggregation: dict[str, Any] | None,
        p_low: float,
        p_high: float,
        doc_id: str,
        group: str,
        true_label_str: str,
        distance_info: dict[str, Any] | None,
        category_text: str,
    ) -> None:
        doc_prob = float(doc_probs[1])
        prob_pct = doc_prob * 100.0
        pred_class = "HIV SUSPICION" if doc_prob >= p_high else "ROUTINE (NO HIV SUSPICION)"
        pred_is_positive = bool(doc_prob >= p_high)

        def _display_label(label: str) -> str:
            normalized = str(label).strip().upper()
            if normalized == "SOSPECHA":
                return "HIV SUSPICION"
            if normalized == "NO_SOSPECHA":
                return "NO HIV SUSPICION"
            return normalized.replace("_", " ")

        display_true_label = _display_label(true_label_str)

        epistemic_defer = False
        aleatoric_defer = False
        lowered_category = str(category_text or "").lower()
        if "epistemic" in lowered_category or "complex" in lowered_category:
            epistemic_defer = True
        elif "ambiguous" in lowered_category or "null" in lowered_category or "uncertainty" in lowered_category or "defer" in lowered_category:
            aleatoric_defer = True
        else:
            aleatoric_defer = bool(p_low <= doc_prob <= p_high)
            epistemic_defer = bool(distance_info and distance_info.get("distance_outside", False))

        if aleatoric_defer and epistemic_defer:
            triage_status = "⚠️ MANUAL REVIEW REQUIRED"
            status_desc = "The model detected conflicting information and an atypical clinical profile that requires expert review."
            status_color = "#e67e22"
        elif aleatoric_defer:
            triage_status = "⚠️ MANUAL REVIEW REQUIRED"
            status_desc = "The information in the text is ambiguous or insufficient for a safe automatic decision."
            status_color = "#f1c40f"
        elif epistemic_defer:
            triage_status = "⚠️ MANUAL REVIEW REQUIRED"
            status_desc = "The patient's clinical profile is unusual compared with the training cases."
            status_color = "#e74c3c"
        else:
            triage_status = "✅ SAFE TO AUTOMATE"
            status_desc = f"The system has high confidence that this is a {pred_class} case."
            status_color = "#2ecc71"

        pos_unigrams = aggregation.get("top_positive_tokens", []) if aggregation else []
        pos_ngrams = aggregation.get("top_positive_ngrams", []) if aggregation else []
        neg_unigrams = aggregation.get("top_negative_tokens", []) if aggregation else []
        neg_ngrams = aggregation.get("top_negative_ngrams", []) if aggregation else []
        pos_features = _collect_top_features(pos_unigrams, pos_ngrams, limit=12)
        neg_features = _collect_top_features(neg_unigrams, neg_ngrams, limit=12)

        def build_feature_html(features: list[dict[str, Any]], polarity: str) -> str:
            if not features:
                return (
                    "<div style='color: #95a5a6; font-style: italic; padding: 10px 0;'>"
                    "No influential features were detected in this category."
                    "</div>"
                )
            rows: list[str] = []
            for rank, item in enumerate(features, start=1):
                rows.append(
                    f"<div class='feature-item feature-item-{polarity}'>"
                    f"<span class='feature-rank'>#{rank}</span>"
                    f"<span class='feature-name'>{html.escape(str(item['token']))}</span>"
                    f"<span class='feature-kind'>{html.escape(str(item['kind']))}</span>"
                    f"<span class='feature-score'>Impact {abs(float(item['importance'])):.4f}</span>"
                    "</div>"
                )
            return f"<div class='feature-list'>{''.join(rows)}</div>"

        pos_html_block = build_feature_html(pos_features, "risk")
        neg_html_block = build_feature_html(neg_features, "safe")
        is_deferred = aleatoric_defer or epistemic_defer

        if is_deferred:
            concept_html = f"""
            <div class='deferral-warning'><strong>Clinical note:</strong> The system recommends manual review because it found conflicting or atypical evidence. The main conflicting factors are shown below:</div>
            <div class="concept-split">
                <div class="concept-col risk-col">
                    <h4 style="color: #c0392b; border-bottom: 2px solid #f2d7d5; padding-bottom: 5px; margin-top: 0;">🔴 Detected Risk Factors</h4>
                    {pos_html_block}
                </div>
                <div class="concept-col safe-col">
                    <h4 style="color: #27ae60; border-bottom: 2px solid #d4efdf; padding-bottom: 5px; margin-top: 0;">🟢 Detected Routine Factors</h4>
                    {neg_html_block}
                </div>
            </div>
            """
        elif pred_is_positive:
            concept_html = f"""
            <div class="concept-split">
                <div class="concept-col risk-col" style="width: 100%;">
                    <h4 style="color: #c0392b; border-bottom: 2px solid #f2d7d5; padding-bottom: 5px; margin-top: 0;">🔴 Main Evidence for Suspicion</h4>
                    {pos_html_block}
                </div>
            </div>
            """
        else:
            concept_html = f"""
            <div class="concept-split">
                <div class="concept-col safe-col" style="width: 100%;">
                    <h4 style="color: #27ae60; border-bottom: 2px solid #d4efdf; padding-bottom: 5px; margin-top: 0;">🟢 Main Evidence for Routine</h4>
                    {neg_html_block}
                </div>
            </div>
            """

        highlight_feature_items: list[dict[str, Any]] = []
        max_abs_importance = 0.001
        include_positive = is_deferred or pred_is_positive
        include_negative = is_deferred or (not pred_is_positive)
        if include_positive:
            for item in pos_features:
                value = abs(float(item["importance"]))
                highlight_feature_items.append({"token": str(item["token"]), "importance": value, "polarity": "positive"})
                max_abs_importance = max(max_abs_importance, value)
        if include_negative:
            for item in neg_features:
                value = abs(float(item["importance"]))
                highlight_feature_items.append({"token": str(item["token"]), "importance": -value, "polarity": "negative"})
                max_abs_importance = max(max_abs_importance, value)

        def _strip_accents(text: str) -> str:
            return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")

        def _normalize_highlight_token(token: str) -> str:
            cleaned = clean_token(token)
            if not cleaned:
                return ""
            if any(ch.isdigit() for ch in cleaned):
                return "num"
            return _strip_accents(cleaned)

        def _feature_tokens_for_highlight(text: str) -> list[str]:
            tokens: list[str] = []
            for word in re.findall(r"\w+", str(text or ""), flags=re.UNICODE):
                normalized = _normalize_highlight_token(word)
                if normalized:
                    tokens.append(normalized)
            return tokens

        hide_negatives = (not is_deferred) and pred_is_positive
        highlight_specs: list[dict[str, Any]] = []
        seen_highlight_specs: set[tuple[tuple[str, ...], str]] = set()
        for item in highlight_feature_items:
            polarity = str(item["polarity"])
            if polarity == "negative" and hide_negatives:
                continue
            norm_tokens = _feature_tokens_for_highlight(str(item["token"]))
            if not norm_tokens:
                continue
            key = (tuple(norm_tokens), polarity)
            if key in seen_highlight_specs:
                continue
            seen_highlight_specs.add(key)
            signed_importance = float(item["importance"])
            rel = abs(signed_importance) / max_abs_importance if max_abs_importance > 0 else 0.0
            if polarity == "positive":
                css_class = "highlight-pos-strong" if rel > 0.40 else "highlight-pos-light"
                title = f"Suspicion indicator: {item['token']}"
            else:
                css_class = "highlight-neg-strong" if rel > 0.40 else "highlight-neg-light"
                title = f"Routine indicator: {item['token']}"
            highlight_specs.append({
                "tokens": norm_tokens,
                "css_class": css_class,
                "title": title,
                "abs_importance": abs(signed_importance),
            })
        highlight_specs.sort(key=lambda item: (-len(item["tokens"]), -float(item["abs_importance"])))

        def _highlight_raw(raw_text: str) -> str:
            if not raw_text:
                return "Text unavailable."
            if not highlight_specs:
                return html.escape(raw_text).replace("\n", "<br>")

            segments: list[dict[str, Any]] = []
            word_entries: list[dict[str, Any]] = []
            for match in re.finditer(r"\w+|\W+", raw_text, flags=re.UNICODE):
                segment_text = match.group(0)
                is_word = bool(re.fullmatch(r"\w+", segment_text, flags=re.UNICODE))
                seg_index = len(segments)
                segments.append({"text": segment_text, "is_word": is_word})
                if is_word:
                    word_entries.append({"seg_index": seg_index, "norm": _normalize_highlight_token(segment_text)})

            if not word_entries:
                return html.escape(raw_text).replace("\n", "<br>")

            occupied = [False] * len(word_entries)
            matches: list[tuple[int, int, dict[str, Any]]] = []
            for spec in highlight_specs:
                tokens = list(spec["tokens"])
                n_tokens = len(tokens)
                if n_tokens == 0 or n_tokens > len(word_entries):
                    continue
                cursor = 0
                while cursor <= len(word_entries) - n_tokens:
                    if any(occupied[cursor:cursor + n_tokens]):
                        cursor += 1
                        continue
                    candidate = [str(word_entries[j]["norm"]) for j in range(cursor, cursor + n_tokens)]
                    if candidate != tokens:
                        cursor += 1
                        continue
                    start_seg = int(word_entries[cursor]["seg_index"])
                    end_seg = int(word_entries[cursor + n_tokens - 1]["seg_index"])
                    matches.append((start_seg, end_seg, spec))
                    for j in range(cursor, cursor + n_tokens):
                        occupied[j] = True
                    cursor += n_tokens

            if not matches:
                return html.escape(raw_text).replace("\n", "<br>")

            start_map = {start: spec for start, _, spec in matches}
            end_map = {end: spec for _, end, spec in matches}
            parts: list[str] = []
            for index, segment in enumerate(segments):
                if index in start_map:
                    spec = start_map[index]
                    parts.append(f'<span class="{spec["css_class"]}" title="{html.escape(str(spec["title"]))}">')
                parts.append(html.escape(str(segment["text"])).replace("\n", "<br>"))
                if index in end_map:
                    parts.append("</span>")
            return "".join(parts)

        chunks_html = ""
        max_att = float(np.max(mil_weights)) if len(mil_weights) > 0 else 1.0
        visible_chunks_count = 0
        for index, (text, stat, att) in enumerate(zip(chunks, chunk_stats, mil_weights)):
            att_pct = float(att) / max(max_att, 1e-12)
            if att_pct < 0.15:
                continue
            visible_chunks_count += 1
            relevance = "High Relevance" if att_pct > 0.6 else "Moderate Relevance"
            orientation = "SUSPICION" if stat.mean_prob >= 0.5 else "ROUTINE"
            confidence = stat.mean_prob * 100.0
            if stat.mean_prob >= 0.5:
                border_color = f"rgba(231, 76, 60, {max(0.4, att_pct):.3f})"
            else:
                border_color = f"rgba(39, 174, 96, {max(0.4, att_pct):.3f})"

            meta = chunk_metas[index] if chunk_metas and index < len(chunk_metas) else {}
            note_date = str(meta.get("date", "Unknown date"))[:10]
            note_type = str(meta.get("type", "Clinical Note")).upper()
            chunk_display_text = str(meta.get("display_chunk_text", meta.get("chunk_text", text)))
            chunks_html += (
                f'<div class="chunk-card" style="border-left-color: {border_color};">\n'
                f'    <div class="chunk-header">\n'
                f'        <span class="chunk-title">{note_type} | {note_date} <span style="font-weight:normal; color:#95a5a6; font-size:0.9em;">(Chunk #{index + 1})</span></span>\n'
                f'        <span class="chunk-meta"><strong>{relevance}</strong> | Orientation: {orientation} ({confidence:.1f}%)</span>\n'
                f'    </div>\n'
                f'    <div class="chunk-body raw-text-format">{_highlight_raw(chunk_display_text)}</div>\n'
                f'</div>\n'
            )
        if visible_chunks_count == 0:
            chunks_html = "<p style='color: #7f8c8d;'>No text chunks with enough clinical relevance were detected for display.</p>"

        means = [chunk.mean_prob * 100.0 for chunk in chunk_stats]
        indices = list(range(1, len(means) + 1))
        bar_colors = [
            "#27ae60" if (mean / 100.0) < p_low else "#e74c3c" if (mean / 100.0) > p_high else "#f1c40f"
            for mean in means
        ]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=indices,
                y=means,
                name="Suspicion Prob. (%)",
                marker_color=bar_colors,
                hovertemplate="Chunk %{x}<br>Probability: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=list(mil_weights),
                mode="lines+markers",
                name="Model Attention",
                line=dict(color="#3498db", width=2),
                marker=dict(size=8, color="#2980b9"),
                hovertemplate="Relevance: %{y:.4f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_hrect(y0=p_low * 100.0, y1=p_high * 100.0, fillcolor="#f1c40f", opacity=0.15, line_width=0)
        fig.update_layout(
            title="Suspicion Timeline Across the Clinical History",
            xaxis_title="Chronological Note Order (Chunks)",
            yaxis_title="Suspicion Probability (%)",
            yaxis_range=[0, 100],
            yaxis2_title="Assigned Relevance",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(showticklabels=False, ticks="")
        plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        meta_bits = [f"Patient ID: {doc_id}"]
        if group:
            meta_bits.append(f"Group: {group}")
        meta_bits.append(f"Ground Truth: {display_true_label}")

        html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Clinical Report - Patient {doc_id}</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        :root {{
            --danger: #c0392b; --danger-light: #f9ebea;
            --success: #27ae60; --success-light: #eaeded;
            --info: #2980b9; --info-light: #ebf5fb;
            --text-main: #2c3e50; --text-muted: #7f8c8d;
            --bg-page: #f4f6f8; --bg-card: #ffffff;
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0 auto; max-width: 1100px; padding: 20px; color: var(--text-main); background-color: var(--bg-page); line-height: 1.6; }}
        .dashboard-card {{ background: var(--bg-card); border-radius: 10px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); border: 1px solid #eaecee; }}
        .header-card {{ border-top: 6px solid {status_color}; }}
        .section-title {{ font-size: 1.15em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px; }}
        h1 {{ font-size: 1.8em; margin: 0 0 10px 0; color: {status_color}; }}
        h2 {{ display: flex; justify-content: space-between; align-items: center; font-size: 1.4em; border-bottom: 1px solid #ecf0f1; padding-bottom: 12px; margin-top: 0; }}
        .meta-tag {{ font-size: 0.65em; font-weight: 500; color: #95a5a6; background: #f8f9fa; padding: 4px 10px; border-radius: 12px; border: 1px solid #e5e8e8; }}
        .triage-decision {{ font-size: 1.15em; font-weight: 600; padding: 12px 15px; background: #f8f9fa; border-radius: 6px; margin: 15px 0; border-left: 4px solid {status_color}; }}
        .concept-split {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 15px; }}
        .concept-col {{ flex: 1; min-width: 300px; background: #fafbfc; padding: 15px; border-radius: 8px; border: 1px solid #ecf0f1; }}
        .feature-list {{ display: grid; gap: 10px; }}
        .feature-item {{ display: grid; grid-template-columns: auto 1fr auto auto; gap: 10px; align-items: center; background: #fff; border: 1px solid #ecf0f1; border-left: 4px solid #bdc3c7; border-radius: 8px; padding: 10px 12px; }}
        .feature-item-risk {{ border-left-color: var(--danger); }}
        .feature-item-safe {{ border-left-color: var(--success); }}
        .feature-rank {{ font-weight: 700; color: var(--text-muted); min-width: 30px; }}
        .feature-name {{ font-weight: 600; color: var(--text-main); word-break: break-word; }}
        .feature-kind {{ font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.3px; color: var(--text-muted); background: #f8f9fa; padding: 4px 8px; border-radius: 999px; }}
        .feature-score {{ font-variant-numeric: tabular-nums; font-size: 0.9em; color: var(--text-muted); }}
        .deferral-warning {{ background: #fff3cd; color: #856404; padding: 10px 15px; border-radius: 6px; border-left: 4px solid #ffeeba; margin-bottom: 15px; font-size: 0.95em; }}
        .chunk-card {{ background: var(--bg-card); border-radius: 8px; padding: 20px; margin-bottom: 15px; border-left: 5px solid #bdc3c7; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
        .chunk-header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #f0f3f4; padding-bottom: 10px; margin-bottom: 12px; font-size: 0.9em; }}
        .chunk-title {{ font-weight: 600; color: var(--text-main); }}
        .chunk-meta {{ color: var(--text-muted); }}
        .chunk-body {{ font-size: 1.05em; line-height: 1.7; }}
        .highlight-pos-strong {{ background-color: rgba(231, 76, 60, 0.15); border-bottom: 2px solid rgba(231, 76, 60, 0.7); font-weight: 500; padding: 0 2px; border-radius: 2px; cursor: help; }}
        .highlight-pos-light {{ background-color: rgba(231, 76, 60, 0.05); border-bottom: 1px dashed rgba(231, 76, 60, 0.5); padding: 0 2px; cursor: help; }}
        .highlight-neg-strong {{ background-color: rgba(39, 174, 96, 0.15); border-bottom: 2px solid rgba(39, 174, 96, 0.7); font-weight: 500; padding: 0 2px; border-radius: 2px; cursor: help; }}
        .highlight-neg-light {{ background-color: rgba(39, 174, 96, 0.05); border-bottom: 1px dashed rgba(39, 174, 96, 0.5); padding: 0 2px; cursor: help; }}
        .raw-text-format {{ white-space: pre-wrap; font-family: inherit; }}
        @media (max-width: 760px) {{
            .feature-item {{ grid-template-columns: auto 1fr; }}
            .feature-kind, .feature-score {{ grid-column: 2; justify-self: start; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-card header-card">
        <h2>
            Clinical Triage Summary
            <span class="meta-tag">{' | '.join(meta_bits)}</span>
        </h2>
        <h1>{triage_status}</h1>
        <div class="triage-decision">
            System Conclusion: <span style="color: {status_color};">{pred_class}</span> (Confidence: {prob_pct:.1f}%)<br>
            <span style="font-weight: normal; color: var(--text-muted); font-size: 0.9em; display: inline-block; margin-top: 5px;">Reason: {status_desc}</span>
        </div>
        <div class="section-title">Most Important Model Features</div>
        {concept_html}
    </div>

    <div class="dashboard-card">
        <div class="section-title" style="margin-top: 0;">Clinical History Timeline</div>
        <p style="color: var(--text-muted); font-size: 0.95em; margin-bottom: 0;">Visualization of the analyzed text chunks. Peaks in the blue line mark the moments where the model detected the strongest clinical evidence.</p>
        {plot_html}
    </div>

    <div class="section-title">Evidence in Clinical Notes ({'Full Document' if visible_chunks_count == 1 else f'Top {visible_chunks_count} Chunks'})</div>
    <p style="color: var(--text-muted); font-size: 0.95em; margin-bottom: 20px;">The text associated with each analyzed chunk is shown below. Hover over the underlined text to identify whether it supports suspicion (red) or routine (green).</p>
    {chunks_html}
</body>
</html>"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html_output)


def _build_distance_info(
    *,
    pred_match: dict[str, Any] | None,
    p_high: float,
    dist_threshold_by_pred: dict[int, float],
    dist_transform: str,
) -> dict[str, Any] | None:
    if pred_match is None:
        return None
    value = pred_match.get("distance")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        distance = float(value)
    except Exception:
        return None
    try:
        prob_pos = float(pred_match.get("prob_pos"))
    except Exception:
        return {
            "distance": distance,
            "distance_outside": False,
            "distance_transform": dist_transform,
        }
    pred_idx = int(prob_pos >= p_high)
    threshold = float(dist_threshold_by_pred.get(pred_idx, float("inf")))
    return {
        "distance": distance,
        "distance_z": distance,
        "distance_threshold_z": threshold,
        "distance_outside": bool(np.isfinite(threshold) and distance > threshold),
        "distance_pred_class": pred_idx,
        "distance_transform": dist_transform,
        "distance_source": "predictions_csv",
    }


def _explain_rows(
    *,
    df: pd.DataFrame,
    predictions_df: pd.DataFrame | None,
    predictions_lookup: dict[tuple[str, str], dict[str, Any]],
    explainer: AttentionMILExplainer,
    output_dir: Path,
    p_low: float,
    p_high: float,
    n_examples: int,
    max_chunks: int,
    all_instances: bool,
    explain_select: str,
    seed: int,
    dist_threshold_by_pred: dict[int, float],
    dist_transform: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = (
        _all_indices(df)
        if bool(all_instances)
        else _select_explain_indices_from_mode(
            df=df,
            n_examples=int(n_examples),
            mode=str(explain_select),
            seed=int(seed),
            predictions_lookup=predictions_lookup,
        )
    )
    print(f"[explain] selected {len(selected_indices)} rows")
    use_aligned_predictions = _predictions_df_is_row_aligned(df, predictions_df)

    for index in tqdm(selected_indices, desc="explaining"):
        row = df.iloc[int(index)]
        key = _row_doc_key(row, int(index))
        pred_match = predictions_lookup.get(key)
        if pred_match is None and use_aligned_predictions and predictions_df is not None:
            aligned = predictions_df.iloc[int(index)].to_dict()
            if _row_doc_key(aligned, int(index)) == key:
                pred_match = aligned

        text_value = row.get("text", "")
        notes_source = _parse_notes_cell(text_value)
        chunks, chunk_metas = _chunks_with_metadata(notes_source, explainer.tokenizer, max_chunks=int(max_chunks))
        model_probs, mil_weights, base_logits = explainer.get_chunk_data(chunks)

        if pred_match is not None:
            try:
                prob_pos = float(pred_match.get("prob_pos"))
            except Exception:
                prob_pos = float(model_probs[1])
        else:
            prob_pos = float(model_probs[1])

        chunk_stats = _chunk_stats_from_attention(mil_weights, prob_pos)
        token_attrs = explainer.get_token_attributions(chunks)
        aggregation = explainer.aggregate_attributions(
            chunks,
            token_attrs,
            mil_weights,
            prob_pos,
            base_logits=base_logits,
        )

        label_value = row.get("label", 0)
        if pred_match is not None and pred_match.get("label") is not None and not pd.isna(pred_match.get("label")):
            label_value = pred_match.get("label")
        if isinstance(label_value, str):
            true_label_str = str(label_value).strip().upper()
            if true_label_str not in {"SOSPECHA", "NO_SOSPECHA"}:
                true_label_str = "SOSPECHA" if str(label_value).strip() in {"1", "1.0"} else "NO_SOSPECHA"
        else:
            true_label_str = "SOSPECHA" if int(label_value) == 1 else "NO_SOSPECHA"

        doc_id, group = key
        category_text = "" if pred_match is None else str(pred_match.get("category", ""))
        distance_info = _build_distance_info(
            pred_match=pred_match,
            p_high=float(p_high),
            dist_threshold_by_pred=dist_threshold_by_pred,
            dist_transform=dist_transform,
        )

        suffix_parts = [f"doc_{_sanitize_filename(doc_id)}"]
        if group:
            suffix_parts.append(f"group_{_sanitize_filename(group)}")
        suffix_parts.append(f"true_{true_label_str}")
        output_name = "_".join(suffix_parts) + ".html"

        explainer.generate_html_report(
            chunks=chunks,
            chunk_metas=chunk_metas,
            chunk_stats=chunk_stats,
            mil_weights=mil_weights,
            output_path=output_dir / output_name,
            doc_probs=np.array([1.0 - float(prob_pos), float(prob_pos)], dtype=np.float32),
            aggregation=aggregation,
            p_low=float(p_low),
            p_high=float(p_high),
            doc_id=str(doc_id),
            group=str(group),
            true_label_str=true_label_str,
            distance_info=distance_info,
            category_text=category_text,
        )

    print(f"[explain] wrote HTML reports to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIL-only HTML explanations using evaluate.py artifacts.",
    )
    parser.add_argument("--model_path", type=str, required=True, help="MIL checkpoint directory.")
    parser.add_argument("--test_csv", type=str, required=True, help="Test CSV used in evaluate.py.")
    parser.add_argument("--output_dir", type=str, default="explain_output", help="Directory where HTML files will be written.")
    parser.add_argument("--eval_output_dir", type=str, default="", help="Directory produced by evaluate.py; used to auto-discover predictions.csv and metrics.json.")
    parser.add_argument("--predictions_csv", type=str, default="", help="Path to predictions.csv or a directory containing it.")
    parser.add_argument("--metrics_json", type=str, default="", help="Path to metrics.json/metric.json or a directory containing it.")
    parser.add_argument("--calibration_bundle", type=str, default="", help="Optional calibration bundle for distance thresholds and saved temperature.")
    parser.add_argument("--n_examples", type=int, default=25)
    parser.add_argument("--explain_select", type=str, default="deferred_first", choices=["deferred_first", "top_uncertainty", "clear_errors", "random"])
    parser.add_argument("--all_instances", action="store_true", default=False)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed used in evaluate.py for exact row alignment.")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    test_csv = os.path.abspath(args.test_csv)
    output_dir = Path(os.path.abspath(args.output_dir))

    predictions_csv = _resolve_predictions_csv_path(str(args.predictions_csv).strip(), str(args.eval_output_dir).strip())
    if not predictions_csv:
        raise FileNotFoundError(
            "Could not resolve predictions.csv. Run evaluate.py first, or pass --predictions_csv/--eval_output_dir."
        )

    metrics_json = _resolve_metrics_json_path(str(args.metrics_json).strip(), str(args.eval_output_dir).strip(), predictions_csv)
    calibration_bundle = _resolve_calibration_bundle_path(str(args.calibration_bundle).strip(), str(args.eval_output_dir).strip(), predictions_csv)

    bundle_meta = _load_bundle_meta(calibration_bundle)
    thresholds = _load_metrics_thresholds(metrics_json)
    if thresholds is None:
        mcp_meta = bundle_meta.get("mcp", {})
        try:
            thresholds = (float(mcp_meta["p_low"]), float(mcp_meta["p_high"]))
        except Exception:
            thresholds = None
    if thresholds is None:
        raise FileNotFoundError(
            "Could not resolve p_low/p_high from metrics.json or calibration_bundle.npz."
        )
    p_low, p_high = thresholds

    temperature = 1.0
    try:
        temperature = float(bundle_meta.get("temperature", 1.0))
    except Exception:
        temperature = 1.0

    decision_meta = bundle_meta.get("decision", {}) if isinstance(bundle_meta, dict) else {}
    dist_threshold_by_pred = _as_int_key_dict(decision_meta.get("dist_threshold_by_pred", {}))
    dist_transform = str(decision_meta.get("dist_transform", "raw"))

    print(f"[explain] loading test CSV: {test_csv}")
    df = pd.read_csv(test_csv)
    df = df.sample(frac=1, random_state=int(args.seed)).reset_index(drop=True)

    predictions_df, predictions_lookup = _load_predictions_lookup(predictions_csv)
    print(f"[explain] loaded predictions: {predictions_csv}")
    if metrics_json:
        print(f"[explain] loaded metrics: {metrics_json}")
    if calibration_bundle:
        print(f"[explain] loaded bundle: {calibration_bundle}")

    tokenizer = _load_tokenizer(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMILClassifier.from_pretrained(model_path, device=device, attn_implementation="sdpa")
    explainer = AttentionMILExplainer(model, tokenizer, device, temperature=temperature)

    _explain_rows(
        df=df,
        predictions_df=predictions_df,
        predictions_lookup=predictions_lookup,
        explainer=explainer,
        output_dir=output_dir,
        p_low=float(p_low),
        p_high=float(p_high),
        n_examples=int(args.n_examples),
        max_chunks=int(args.max_chunks),
        all_instances=bool(args.all_instances),
        explain_select=str(args.explain_select),
        seed=int(args.seed),
        dist_threshold_by_pred=dist_threshold_by_pred,
        dist_transform=dist_transform,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()