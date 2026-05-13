from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .text import clean_text, chunk_text


class MILDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        *,
        chunk_max_length: int = 384,
        chunk_overlap: int = 64,
    ):
        self.texts = texts  # list[str] or list[list[str]]
        self.labels = labels
        self.tokenizer = tokenizer
        self.chunk_max_length = int(chunk_max_length)
        self.chunk_overlap = int(chunk_overlap)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.texts[idx]
        label = self.labels[idx]

        source_texts = item if isinstance(item, list) else [item]

        all_chunks: list[str] = []
        for note_text in source_texts:
            t = clean_text(note_text)
            if t:
                all_chunks.extend(
                    chunk_text(
                        t,
                        self.tokenizer,
                        max_length=self.chunk_max_length,
                        overlap=self.chunk_overlap,
                    )
                )

        if len(all_chunks) == 0:
            all_chunks = [""]

        return {"chunks": all_chunks, "label": int(label), "id": int(idx)}


class MILCollator:
    def __init__(self, tokenizer, max_length: int = 512, max_chunks: int = 64):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_chunks = int(max_chunks)

    def __call__(self, batch):
        all_chunks: list[str] = []
        labels: list[int] = []
        num_chunks_per_doc: list[int] = []

        for item in batch:
            original_chunks = item["chunks"]
            total_chunks = len(original_chunks)

            # Head+Tail strategy (matches original eval_GPT.py)
            if total_chunks > self.max_chunks:
                n_head = int(self.max_chunks * 0.25)
                n_tail = self.max_chunks - n_head
                selected_chunks = original_chunks[:n_head] + original_chunks[-n_tail:]
            else:
                selected_chunks = original_chunks

            all_chunks.extend(selected_chunks)
            labels.append(int(item["label"]))
            num_chunks_per_doc.append(len(selected_chunks))

        enc = self.tokenizer(
            all_chunks,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "num_chunks_per_doc": num_chunks_per_doc,
        }
