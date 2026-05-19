"""Language modeling data pipeline with token packing.

Wraps HuggingFace ``datasets`` in a ``LightningDataModule`` that implements
efficient token packing. This ensures that every training batch is fully
utilized (no padding tokens), maximizing compute throughput.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from datasets import load_dataset
from transformers import AutoTokenizer
from omegaconf import DictConfig


class PackedTokenDataset(Dataset):
    """Fixed-length token chunks packed from a flat stream.

    Each sample consists of a sequence of token IDs of length ``seq_len``.
    This prevents computation waste on padding tokens in standard LM training.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        self.seq_len = seq_len
        # Drop trailing tokens to ensure uniform chunk sizes.
        n_chunks = len(token_ids) // seq_len
        self.data = token_ids[: n_chunks * seq_len].view(n_chunks, seq_len)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class LanguageModelDataModule(LightningDataModule):
    """LightningDataModule for large-scale causal language modeling.

    Handles dataset downloading, tokenization, and Disk caching.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._train: Optional[PackedTokenDataset] = None
        self._val: Optional[PackedTokenDataset] = None
        self._test: Optional[PackedTokenDataset] = None

    def _tokenise_split(self, split: str) -> torch.Tensor:
        """Downloads and tokenizes a specific split, returning a flat tensor."""
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        raw = load_dataset(
            self.cfg.dataset_name,
            self.cfg.dataset_config,
            split=split,
            trust_remote_code=True,
        )

        def _encode(batch: dict) -> dict:
            return tokenizer(
                batch["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )

        tokenised = raw.map(
            _encode,
            batched=True,
            remove_columns=raw.column_names,
            desc=f"Tokenising {split}",
        )

        # Flatten nested token lists into a single continuous tensor.
        all_ids: list[int] = []
        for sample in tokenised:
            all_ids.extend(sample["input_ids"])

        return torch.tensor(all_ids, dtype=torch.long)

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepares splits for the requested training stage."""
        seq_len = self.cfg.max_seq_len

        if stage in (None, "fit", "validate"):
            if self._train is None:
                ids = self._tokenise_split(self.cfg.train_split)
                self._train = PackedTokenDataset(ids, seq_len)
            if self._val is None:
                ids = self._tokenise_split(self.cfg.val_split)
                self._val = PackedTokenDataset(ids, seq_len)

        if stage in (None, "test"):
            if self._test is None:
                ids = self._tokenise_split(self.cfg.test_split)
                self._test = PackedTokenDataset(ids, seq_len)

    def _make_loader(self, dataset: PackedTokenDataset, shuffle: bool) -> DataLoader:
        """Standardizes DataLoader construction."""
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test, shuffle=False)

