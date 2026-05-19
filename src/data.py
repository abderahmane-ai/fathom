"""Language-modelling data pipeline.

Wraps HuggingFace ``datasets`` in a ``LightningDataModule`` that:

* Downloads and tokenises any text dataset via ``datasets.load_dataset``.
* Packs tokenised tokens into fixed-length chunks of ``max_seq_len`` tokens
  (no padding, no truncation waste — standard LM packing).
* Returns ``(input_ids, labels)`` where ``labels = input_ids`` shifted by the
  DataLoader consumer (cross-entropy loss on next-token prediction).

Usage::

    dm = LanguageModelDataModule(cfg.data)
    dm.setup()
    trainer.fit(model, dm)
"""
from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from datasets import load_dataset          # type: ignore[import]
from transformers import AutoTokenizer     # type: ignore[import]
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class PackedTokenDataset(Dataset):
    """Fixed-length token chunks packed from a flat token stream.

    Packs all tokens contiguously — no padding, no wasted compute.
    Each sample is ``(input_ids,)`` of length ``seq_len``.

    Args:
        token_ids: 1-D integer tensor of all tokens in the split.
        seq_len: Chunk size (equals ``max_seq_len`` from config).
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        self.seq_len = seq_len
        # Drop remainder tokens so every chunk has exactly seq_len tokens.
        n_chunks = len(token_ids) // seq_len
        self.data = token_ids[: n_chunks * seq_len].view(n_chunks, seq_len)

    def __len__(self) -> int:  # noqa: D105
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return token chunk at ``idx``.

        Args:
            idx: Sample index.

        Returns:
            Integer tensor of shape ``(seq_len,)``.
        """
        return self.data[idx]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class LanguageModelDataModule(LightningDataModule):
    """LightningDataModule for causal language modelling.

    Supports any ``datasets`` text corpus.  Tokenises once, caches on disk,
    and packs into fixed-length chunks.

    Args:
        cfg: Data configuration (``conf/data/*.yaml``).  Expected keys:
            ``dataset_name``, ``dataset_config``, ``tokenizer_name``,
            ``max_seq_len``, ``batch_size``, ``num_workers``,
            ``train_split``, ``val_split``, ``test_split``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._train: Optional[PackedTokenDataset] = None
        self._val: Optional[PackedTokenDataset] = None
        self._test: Optional[PackedTokenDataset] = None

    # ------------------------------------------------------------------

    def _tokenise_split(self, split: str) -> torch.Tensor:
        """Download, tokenise, and pack a single dataset split.

        Args:
            split: HuggingFace split name (e.g. ``"train"``, ``"validation"``).

        Returns:
            1-D integer tensor of all token IDs in the split.
        """
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

        # Flatten all token lists into a single 1-D tensor.
        all_ids: list[int] = []
        for sample in tokenised:
            all_ids.extend(sample["input_ids"])

        return torch.tensor(all_ids, dtype=torch.long)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Tokenise and pack datasets for the requested stage.

        Args:
            stage: ``"fit"``, ``"validate"``, ``"test"``, or ``None`` (all).
        """
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
        """Build a DataLoader with standard options.

        Args:
            dataset: The packed dataset to wrap.
            shuffle: Whether to shuffle samples.

        Returns:
            Configured ``DataLoader``.
        """
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
        """Return the training DataLoader."""
        assert self._train is not None, "Call setup() first."
        return self._make_loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        assert self._val is not None, "Call setup() first."
        return self._make_loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        assert self._test is not None, "Call setup() first."
        return self._make_loader(self._test, shuffle=False)
