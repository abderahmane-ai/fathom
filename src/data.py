"""Language modeling data pipeline with token packing.

Wraps HuggingFace ``datasets`` in a ``LightningDataModule`` that implements
efficient token packing. This ensures that every training batch is fully
utilized (no padding tokens), maximizing compute throughput.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import pyarrow.compute as pc
import torch
from datasets import load_dataset
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer


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
        self._train: PackedTokenDataset | None = None
        self._val: PackedTokenDataset | None = None
        self._test: PackedTokenDataset | None = None

    def _tokenise_split(self, split: str) -> torch.Tensor:
        """Download, tokenize, and flatten one split.

        Args:
            split: Hugging Face dataset split expression.

        Returns:
            Flat token ID tensor.
        """
        cache_path = self._token_cache_path(split)
        if cache_path is not None and cache_path.is_file():
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name, use_fast=True)
        assert tokenizer is not None
        tokenizer.model_max_length = 1_000_000  # suppress HF warning; we pack+chunk to actual seq_len
        raw = load_dataset(
            self.cfg.dataset_name,
            self.cfg.dataset_config,
            split=split,
            cache_dir=getattr(self.cfg, "cache_dir", None),
        )

        def _encode(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
            """Tokenize a text batch without padding."""
            return tokenizer(
                batch["text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )

        num_proc = max(1, int(getattr(self.cfg, "tokenization_num_proc", self.cfg.num_workers)))
        tokenised = raw.map(
            _encode,
            batched=True,
            num_proc=num_proc,
            remove_columns=raw.column_names,
            desc=f"Tokenising {split}",
        )

        chunks = []
        for chunk in tokenised.data.column("input_ids").chunks:
            flattened = pc.list_flatten(chunk)
            chunks.append(torch.as_tensor(flattened.to_numpy(zero_copy_only=False).copy(), dtype=torch.long))
        if not chunks:
            return torch.empty(0, dtype=torch.long)

        token_ids = torch.cat(chunks)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(token_ids, cache_path)
        return token_ids

    def _token_cache_path(self, split: str) -> Path | None:
        """Return a local cache path for flattened token IDs.

        Args:
            split: Hugging Face dataset split expression.

        Returns:
            Cache path, or ``None`` when caching is disabled.
        """
        cache_dir = getattr(self.cfg, "packed_cache_dir", None)
        if cache_dir is None:
            cache_dir = os.environ.get("RR_PACKED_CACHE_DIR")
        if cache_dir is None:
            return None
        key = "|".join(
            [
                str(self.cfg.dataset_name),
                str(self.cfg.dataset_config),
                str(self.cfg.tokenizer_name),
                split,
                str(self.cfg.max_seq_len),
            ]
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return Path(cache_dir).expanduser() / f"{digest}.pt"

    def setup(self, stage: str | None = None) -> None:
        """Prepares splits for the requested training stage."""
        seq_len = self.cfg.max_seq_len

        if stage in (None, "fit") and self._train is None:
            ids = self._tokenise_split(self.cfg.train_split)
            self._train = PackedTokenDataset(ids, seq_len)

        if stage in (None, "fit", "validate") and self._val is None:
            ids = self._tokenise_split(self.cfg.val_split)
            self._val = PackedTokenDataset(ids, seq_len)

        if stage in (None, "test") and self._test is None:
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
        assert self._train is not None, "setup('fit') must run before train_dataloader()."
        return self._make_loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None, "setup('validate') must run before val_dataloader()."
        return self._make_loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None, "setup('test') must run before test_dataloader()."
        return self._make_loader(self._test, shuffle=False)


class DeepNeedleDataset(IterableDataset):
    """Infinite or fixed stream of Deep Needle sequences.

    Args:
        seq_len: Number of tokens per sequence.
        vocab_size: Number of token IDs.
        start_token: Token placed at position 0.
        blank_token: Token used for filler positions.
        output_token: Marker token placed at the final position.
        n_samples: Fixed sample count for evaluation, or ``None`` for training.
        seed: Base random seed. DataLoader workers add their worker id.
    """

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        start_token: int,
        blank_token: int,
        output_token: int,
        n_samples: int | None,
        seed: int,
    ) -> None:
        super().__init__()
        if seq_len < 3:
            raise ValueError("seq_len must be at least 3.")
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4.")
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.blank_token = blank_token
        self.output_token = output_token
        self.n_samples = n_samples
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield token/target pairs.

        Returns:
            Iterator over ``(tokens, targets)`` tensors.
        """
        worker_info = torch.utils.data.get_worker_info()
        rng = torch.Generator()
        rng.manual_seed(self.seed + (worker_info.id if worker_info else 0))

        count = 0
        while self.n_samples is None or count < self.n_samples:
            # Safe payload range: [2, vocab_size-2] excludes all reserved markers.
            payload = torch.randint(2, self.vocab_size - 1, (1,), generator=rng).item()

            tokens = torch.full((self.seq_len,), self.blank_token, dtype=torch.long)
            tokens[0] = self.start_token
            tokens[1] = payload
            tokens[self.seq_len - 1] = self.output_token

            target = torch.full((self.seq_len,), -100, dtype=torch.long)
            target[self.seq_len - 1] = payload

            yield tokens, target
            count += 1

    def __len__(self) -> int:
        """Return fixed dataset length.

        Returns:
            Number of samples for finite evaluation datasets.

        Preconditions:
            ``n_samples`` is not ``None``.
        """
        if self.n_samples is None:
            raise TypeError("Infinite dataset has no length.")
        return self.n_samples


class DeepNeedleDataModule(LightningDataModule):
    """LightningDataModule for Deep Needle.

    Args:
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        start_token: Start marker token.
        blank_token: Filler token.
        output_token: Output marker token.
        batch_size: Batch size for training and evaluation.
        n_eval: Number of fixed-seed evaluation samples.
        num_workers: DataLoader workers.
    """

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        start_token: int,
        blank_token: int,
        output_token: int,
        batch_size: int,
        n_eval: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.blank_token = blank_token
        self.output_token = output_token
        self.batch_size = batch_size
        self.n_eval = n_eval
        self.num_workers = num_workers

    def _dataset(self, n_samples: int | None, seed: int) -> DeepNeedleDataset:
        """Create a configured Deep Needle dataset.

        Args:
            n_samples: Fixed sample count, or ``None`` for infinite training.
            seed: Dataset seed.

        Returns:
            Configured dataset.
        """
        return DeepNeedleDataset(
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            start_token=self.start_token,
            blank_token=self.blank_token,
            output_token=self.output_token,
            n_samples=n_samples,
            seed=seed,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the infinite training dataloader.

        Returns:
            Training dataloader.
        """
        return DataLoader(
            self._dataset(n_samples=None, seed=0),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the fixed validation dataloader.

        Returns:
            Validation dataloader.
        """
        return DataLoader(
            self._dataset(n_samples=self.n_eval, seed=99_999),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
