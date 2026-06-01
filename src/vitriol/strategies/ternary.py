"""Ternary weight generation strategy.

Restricts generated weights to three values: {-α, 0, +α}.
Similar to Binary but with a sparsity-inducing zero level,
achieving higher compression through zero-value compression.

Capabilities:
    Supports Safetensors format.
    Supports gradient computation (training).
    ~66% theoretical storage reduction with zero compression.

Example:
    >>> strategy = TernaryStrategy(alpha=0.1)
    >>> tensor = strategy.generate_tensor((1024, 1024), torch.float32, "weight")
    >>> set(tensor.unique().tolist())
    {-0.1, 0.0, 0.1}
"""

from typing import Dict

import torch
from safetensors.torch import save_file

from .base import StrategyCapabilities, WeightGenerationStrategy

_TERNARY_DEFAULT_ALPHA: float = 0.1


class TernaryStrategy(WeightGenerationStrategy):
    """Weights restricted to {-α, 0, α}."""

    def __init__(self, device: str = "cpu", alpha: float = _TERNARY_DEFAULT_ALPHA, **kwargs):
        super().__init__(device, **kwargs)
        self.alpha = alpha

    @property
    def capabilities(self) -> StrategyCapabilities:
        return StrategyCapabilities(
            supports_safetensors=True,
            supports_training=True,
            requires_contiguous=False,
            max_compression_ratio=0.5,
            description="Ternary weights {-α, 0, α} for quantization"
        )

    def generate_tensor(self, shape: tuple, dtype: torch.dtype, name: str, **kwargs) -> torch.Tensor:
        if dtype == torch.float32:
            dtype = torch.bfloat16
        values = torch.tensor([-self.alpha, 0.0, self.alpha], dtype=dtype, device=self.device)
        indices = torch.randint(0, 3, shape, device=self.device)
        return values[indices]

    def save_shard(self, shard_data: Dict[str, torch.Tensor], path: str) -> None:
        save_file(shard_data, path, metadata={"format": "pt"})

    @property
    def storage_format(self) -> str:
        return "safetensors"
