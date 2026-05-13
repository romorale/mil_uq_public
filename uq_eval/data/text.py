from __future__ import annotations

import re
import unicodedata


_RE_PUNCT = re.compile(r"[,;.:ºª¡!¿?@#$%&[\](){}<>~=+\-*/|\\_^`\"'\"\"]")
_RE_NUM_1 = re.compile(r"\s(?:-(?:[1-9](?:\d{0,2}(?:,\d{3})+|\d*))|(?:0|(?:[1-9](?:\d{0,2}(?:,\d{3})+|\d*))))(?:.\d+|)")
_RE_NUM_2 = re.compile(r"^(?:-(?:[1-9](?:\d{0,2}(?:,\d{3})+|\d*))|(?:0|(?:[1-9](?:\d{0,2}(?:,\d{3})+|\d*))))(?:.\d+|)")
_RE_DIGIT = re.compile(r"\d")
_RE_MULTI_NUM = re.compile(r"(\s+num)+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = text.lower()
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_NUM_1.sub(" num", text)
    text = _RE_NUM_2.sub(" num", text)
    text = _RE_DIGIT.sub(" ", text)
    text = _RE_MULTI_NUM.sub(" num ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, tokenizer, max_length: int = 384, overlap: int = 64) -> list[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    effective_max = max_length - 2
    if len(tokens) <= effective_max:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + effective_max, len(tokens))
        chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start += (effective_max - overlap)

    return chunks
