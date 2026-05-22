"""Configuration helpers for benchmark entrypoints."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from benchmarks.common.artifacts import repo_root


def benchmark_config_path(benchmark_name: str) -> Path:
    """Return the config path for a benchmark.

    Args:
        benchmark_name: Folder under ``benchmarks``.

    Returns:
        Absolute path to the benchmark config.
    """
    return repo_root() / "benchmarks" / benchmark_name / "config.yaml"


def load_benchmark_config(benchmark_name: str) -> DictConfig:
    """Load one benchmark YAML config.

    Args:
        benchmark_name: Folder under ``benchmarks``.

    Returns:
        OmegaConf config object.

    Preconditions:
        The benchmark folder contains ``config.yaml``.
    """
    path = benchmark_config_path(benchmark_name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark config: {path}")
    return OmegaConf.load(path)


def make_run_id(prefix: str | None = None) -> str:
    """Create a stable timestamp run id.

    Args:
        prefix: Optional prefix.

    Returns:
        Filesystem-safe run id.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}" if prefix else stamp


def benchmark_modes(cfg: DictConfig) -> list[str]:
    """Return residual modes configured for a benchmark.

    Args:
        cfg: Benchmark config with a ``modes`` list.

    Returns:
        Residual mode names.
    """
    return list(cfg.get("modes", ["standard", "recurrent_residual", "block_attnres"]))


def model_sweep(cfg: DictConfig) -> Iterable[DictConfig]:
    """Yield model configs for a benchmark sweep.

    Args:
        cfg: Benchmark config with either ``model`` or ``sweep``.

    Returns:
        Iterable of model config objects.
    """
    sweep = cfg.get("sweep")
    if not sweep:
        yield cfg.model
        return
    for item in sweep:
        model_cfg = OmegaConf.merge(cfg.model, item)
        yield model_cfg


def config_for_mode(
    cfg: DictConfig,
    residual_mode: str,
    model_cfg: DictConfig | None = None,
) -> DictConfig:
    """Create a benchmark config for one residual mode.

    Args:
        cfg: Base benchmark config.
        residual_mode: Residual mode to set.
        model_cfg: Optional model override from a sweep.

    Returns:
        Deep-copied config with the selected residual mode.
    """
    selected = OmegaConf.create(deepcopy(OmegaConf.to_container(cfg, resolve=True)))
    selected.model = model_cfg if model_cfg is not None else selected.model
    selected.model.residual_mode = residual_mode
    if residual_mode == "block_attnres" and "attnres_block" not in selected.model:
        raise ValueError("block_attnres requires model.attnres_block config.")
    return selected
