"""Shared pytest fixtures for all test modules."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture(autouse=True)
def seed() -> None:
    """Fix RNG seed for reproducibility in every test."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device() -> torch.device:
    """Return CPU device (tests run on CPU; GPU is optional)."""
    return torch.device("cpu")


@pytest.fixture
def B() -> int:
    """Batch size."""
    return 2


@pytest.fixture
def S() -> int:
    """Sequence length."""
    return 8


@pytest.fixture
def d_model() -> int:
    """Hidden dimension."""
    return 64


@pytest.fixture
def n_heads() -> int:
    """Number of attention heads."""
    return 4


@pytest.fixture
def ff_dim(d_model: int) -> int:
    """FFN intermediate dimension (4×d_model)."""
    return d_model * 4


@pytest.fixture
def num_layers() -> int:
    """Number of transformer layers."""
    return 6


@pytest.fixture
def standard_cfg(d_model, n_heads, ff_dim, num_layers):
    """Minimal OmegaConf config for standard residual mode."""
    return OmegaConf.create({
        "d_model": d_model,
        "n_heads": n_heads,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "max_seq_len": 32,
        "vocab_size": 256,
        "dropout": 0.0,
        "residual_mode": "standard",
    })


@pytest.fixture
def attnres_cfg(d_model, n_heads, ff_dim, num_layers):
    """Minimal OmegaConf config for attnres_block mode (block_size=4 sublayers)."""
    return OmegaConf.create({
        "d_model": d_model,
        "n_heads": n_heads,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "max_seq_len": 32,
        "vocab_size": 256,
        "dropout": 0.0,
        "residual_mode": "attnres_block",
        "attnres_block": {"block_size": 4},  # 4 sublayers = 2 layers per block
    })


@pytest.fixture
def rr_cfg(d_model, n_heads, ff_dim, num_layers):
    """Minimal OmegaConf config for recurrent_residual mode."""
    return OmegaConf.create({
        "d_model": d_model,
        "n_heads": n_heads,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "max_seq_len": 32,
        "vocab_size": 256,
        "dropout": 0.0,
        "residual_mode": "recurrent_residual",
    })
