"""Deep Needle synthetic dataset for long-range information preservation benchmarks.

Protocol:
    Sequence layout (length = 512):
        pos 0   : START token (0)
        pos 1   : PAYLOAD token (random 1..VOCAB_SIZE-2)
        pos 2..510 : BLANK token (0)
        pos 511 : OUTPUT token (VOCAB_SIZE-1)

    Target: Only the OUTPUT position is supervised.
            The model must predict the PAYLOAD token at position 511,
            having seen it only at position 1 — 510 positions earlier.

Why this works:
    - Standard residuals dilute the payload signal across 510 BLANK tokens.
    - AttnRes (block_size=2) can only attend to 2 adjacent layer outputs,
      not back to position 1 directly.
    - RR's global EMA memory m can hold the payload with a nearly-closed write
      gate (alpha≈0) for all BLANK positions, then output it at position 511.
"""
from __future__ import annotations

from collections.abc import Iterator

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset

VOCAB_SIZE = 64
SEQ_LEN = 512
START_TOKEN = 0
BLANK_TOKEN = 0
OUTPUT_TOKEN = VOCAB_SIZE - 1  # token 63


class DeepNeedleDataset(IterableDataset):
    """Infinite stream of Deep Needle sequences.

    Args:
        n_samples: For non-infinite (eval) datasets, the exact count.
                   Pass None for infinite training stream.
        seed: Base random seed. Each worker offsets by its worker_id.
    """

    def __init__(self, n_samples: int | None = None, seed: int = 42) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        rng = torch.Generator()
        rng.manual_seed(self.seed + (worker_info.id if worker_info else 0))

        count = 0
        while self.n_samples is None or count < self.n_samples:
            # Random payload in [1, VOCAB_SIZE-2] (avoid START and OUTPUT tokens)
            payload = torch.randint(1, VOCAB_SIZE - 1, (1,), generator=rng).item()

            tokens = torch.zeros(SEQ_LEN, dtype=torch.long)
            tokens[0] = START_TOKEN
            tokens[1] = payload
            # positions 2..510 stay BLANK (0)
            tokens[SEQ_LEN - 1] = OUTPUT_TOKEN

            # Target: -100 everywhere except the OUTPUT position
            target = torch.full((SEQ_LEN,), -100, dtype=torch.long)
            target[SEQ_LEN - 1] = payload

            yield tokens, target
            count += 1

    def __len__(self) -> int:
        if self.n_samples is None:
            raise TypeError("Infinite dataset has no length")
        return self.n_samples


class DeepNeedleDataModule(L.LightningDataModule):
    """LightningDataModule for the Deep Needle diagnostic task.

    Args:
        batch_size: Batch size for training and evaluation.
        n_eval: Number of fixed-seed evaluation samples.
        num_workers: DataLoader workers.
    """

    def __init__(
        self,
        batch_size: int = 64,
        n_eval: int = 1000,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.n_eval = n_eval
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            DeepNeedleDataset(n_samples=None, seed=0),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            DeepNeedleDataset(n_samples=self.n_eval, seed=99999),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
