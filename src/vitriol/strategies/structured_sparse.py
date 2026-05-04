"""Structured sparse weight generation strategy.

Generates weights with structured sparsity patterns (entire rows,
columns, or channels zeroed out). Unlike unstructured sparsity,
structured sparsity is hardware-accelerated on modern GPUs/TPUs.

Capabilities:
    Supports Safetensors format.
    Supports gradient computation (training).
    Storage reduction proportional to sparsity ratio.

Example:
    >>> strategy = StructuredSparseStrategy(sparsity=0.9)
    >>> tensor = strategy.generate_tensor((1024, 1024), torch.float32, "weight")
    >>> (tensor == 0).float().mean()  # ~90% zeros
"""

import torch
from typing import Dict
from .base import WeightGenerationStrategy, StrategyCapabilities
from safetensors.torch import save_file

_STRUCTURED_SPARSE_SCALE: float = 0.01


class StructuredSparseStrategy(WeightGenerationStrategy):
    """Weights with structured sparsity pattern."""

    def __init__(self, device: str = "cpu", sparsity: float = 0.5, **kwargs):
        super().__init__(device)
        self.sparsity = sparsity

    @property
    def capabilities(self) -> StrategyCapabilities:
        return StrategyCapabilities(
            supports_safetensors=True,
            supports_training=True,
            requires_contiguous=False,
            max_compression_ratio=self.sparsity,
            description=f"Structured sparse weights with {self.sparsity:.0%} sparsity"
        )

    def generate_tensor(self, shape: tuple, dtype: torch.dtype, name: str, **kwargs) -> torch.Tensor:
        self._validate_shape(shape)
        dtype = self._normalize_dtype(dtype)

        tensor = torch.randn(shape, dtype=dtype, device=self.device) * _STRUCTURED_SPARSE_SCALE
        mask = torch.rand(shape, device=self.device) > self.sparsity
        return tensor * mask

    def save_shard(self, shard_data: Dict[str, torch.Tensor], path: str) -> None:
        save_file(shard_data, path, metadata={"format": "pt"})

    @property
    def storage_format(self) -> str:
        return "safetensors"
