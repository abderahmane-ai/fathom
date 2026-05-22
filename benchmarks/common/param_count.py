"""Parameter counting and benchmark size guards."""
from __future__ import annotations

from omegaconf import DictConfig
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable and frozen parameters.

    Args:
        model: PyTorch module.

    Returns:
        Total parameter count.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def assert_model_under_cap(model_cfg: DictConfig, max_params: int) -> int:
    """Instantiate a model config and enforce a parameter cap.

    Args:
        model_cfg: Transformer model config.
        max_params: Maximum allowed parameter count.

    Returns:
        Actual parameter count.

    Raises:
        ValueError: If the model exceeds ``max_params``.
    """
    from src.modules import TransformerDecoder

    model = TransformerDecoder(model_cfg)
    params = count_parameters(model)
    if params > max_params:
        raise ValueError(
            f"Model has {params:,} parameters, exceeding cap of {max_params:,}."
        )
    return params

