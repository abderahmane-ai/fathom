"""Public API for the ``src.modules`` package.

Import any module component from here::

    from src.modules import TransformerDecoder, BlockAttnRes
"""
from src.modules.attention import Attention
from src.modules.attnres_block import BlockAttnRes
from src.modules.ffn import FeedForward
from src.modules.recurrent_residual import RecurrentResidualCell
from src.modules.transformer import TransformerDecoder
from src.modules.transformer_layer import TransformerLayer

__all__ = [
    "Attention",
    "BlockAttnRes",
    "FeedForward",
    "RecurrentResidualCell",
    "TransformerDecoder",
    "TransformerLayer",
]