"""Quantized weight generation strategy.

Generates weights restricted to N evenly-spaced quantization levels
between -0.5 and 0.5. The number of levels is determined by the
n_bits parameter (default 8, giving 256 levels).

Capabilities:
    Supports Safetensors format.
    Supports gradient computation (training).
    Storage reduction: (1/n_bits) of original float precision.

Example:
    >>> strategy = QuantizedStrategy(n_bits=4)
    >>> tensor = strategy.generate_tensor((1024, 1024), torch.float32, "weight")
"""

import torch
from typing import Dict
from .base import WeightGenerationStrategy, StrategyCapabilities
from safetensors.torch import save_file


class QuantizedStrategy(WeightGenerationStrategy):
    """Weights restricted to N levels."""

    def __init__(self, device: str = "cpu", n_bits: int = 8, **kwargs):
        super().__init__(device)
        self.n_bits = n_bits

    @property
    def capabilities(self) -> StrategyCapabilities:
        return StrategyCapabilities(
            supports_safetensors=True,
            supports_training=True,
            requires_contiguous=False,
            max_compression_ratio=0.5,
            description=f"Quantized weights with {self.n_bits}-bit precision"
        )

    def generate_tensor(self, shape: tuple, dtype: torch.dtype, name: str, **kwargs) -> torch.Tensor:
        if dtype == torch.float32:
            dtype = torch.bfloat16

        n_levels = 2 ** self.n_bits
        levels = torch.linspace(-0.5, 0.5, n_levels, dtype=dtype, device=self.device)
        indices = torch.randint(0, n_levels, shape, device=self.device)
        return levels[indices]

    def save_shard(self, shard_data: Dict[str, torch.Tensor], path: str) -> None:
        save_file(shard_data, path, metadata={"format": "pt"})

    @property
    def storage_format(self) -> str:
        return "safetensors"
