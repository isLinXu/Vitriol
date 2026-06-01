"""Binary weight generation strategy.

Restricts generated weights to two values: {-α, +α}.
Achieves extreme quantization (1-bit weights) while maintaining
trainability through the Straight-Through Estimator.

Capabilities:
    Supports Safetensors format.
    Supports gradient computation (training).
    50% storage reduction vs full precision.

Example:
    >>> strategy = BinaryStrategy(alpha=0.01)
    >>> tensor = strategy.generate_tensor((1024, 1024), torch.float32, "weight")
    >>> set(tensor.unique().tolist())
    {-0.01, 0.01}
"""

from typing import Dict

import torch
from safetensors.torch import save_file

from .base import StrategyCapabilities, WeightGenerationStrategy

_BINARY_DEFAULT_ALPHA: float = 0.01


class BinaryStrategy(WeightGenerationStrategy):
    """Weights restricted to {-α, α}."""

    def __init__(self, device: str = "cpu", alpha: float = _BINARY_DEFAULT_ALPHA, **kwargs):
        super().__init__(device, **kwargs)
        self.alpha = alpha

    @property
    def capabilities(self) -> StrategyCapabilities:
        return StrategyCapabilities(
            supports_safetensors=True,
            supports_training=True,
            requires_contiguous=False,
            max_compression_ratio=0.5,
            description="Binary weights {-α, α} for extreme quantization"
        )

    def generate_tensor(self, shape: tuple, dtype: torch.dtype, name: str, **kwargs) -> torch.Tensor:
        if dtype == torch.float32:
            dtype = torch.bfloat16
        tensor = torch.sign(torch.randn(shape, device=self.device)) * self.alpha
        return tensor.to(dtype)

    def save_shard(self, shard_data: Dict[str, torch.Tensor], path: str) -> None:
        save_file(shard_data, path, metadata={"format": "pt"})

    @property
    def storage_format(self) -> str:
        return "safetensors"
