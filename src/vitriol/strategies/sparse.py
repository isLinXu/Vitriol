"""Sparse weight generation strategy.

Generates weights as SparseSpec descriptors that record the expected
shape and dtype without allocating actual tensor data. The save format
uses the Safetensors header layout with sparse data holes.

Capabilities:
    Supports Safetensors format (with sparse holes).
    Supports gradient computation (training).
    Near-zero storage for very sparse models.

Note:
    Unlike other strategies, generate_tensor returns a SparseSpec
    object rather than a torch.Tensor.
"""

import json
import struct
from typing import Dict

import torch

from .base import StrategyCapabilities, WeightGenerationStrategy


class SparseSpec:
    """Descriptor for a sparse tensor (no actual data allocated)."""

    def __init__(self, name, shape, dtype_str, size):
        self.name = name
        self.shape = shape
        self.dtype_str = dtype_str
        self.size = size


class SparseStrategy(WeightGenerationStrategy):
    """Generates sparse weights (safetensors with holes)."""

    @property
    def capabilities(self) -> StrategyCapabilities:
        return StrategyCapabilities(
            supports_safetensors=True,
            supports_training=True,
            requires_contiguous=False,
            max_compression_ratio=0.5,
            description="Sparse weight generation with configurable sparsity"
        )

    def _get_dtype_str(self, dtype: torch.dtype) -> str:
        mapping = {
            torch.float32: "F32",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.int64: "I64",
            torch.int32: "I32",
            torch.int16: "I16",
            torch.int8: "I8",
            torch.uint8: "U8",
            torch.bool: "BOOL"
        }
        return mapping.get(dtype, "F32")

    def _get_dtype_size(self, dtype: torch.dtype) -> int:
        mapping = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int64: 8,
            torch.int32: 4,
            torch.int16: 2,
            torch.int8: 1,
            torch.uint8: 1,
            torch.bool: 1
        }
        return mapping.get(dtype, 4)

    def generate_tensor(self, shape: tuple, dtype: torch.dtype, name: str, **kwargs) -> SparseSpec:
        if dtype == torch.float32:
            dtype = torch.bfloat16

        numel = 1
        for dim in shape:
            numel *= dim
        size = numel * self._get_dtype_size(dtype)

        return SparseSpec(name, list(shape), self._get_dtype_str(dtype), size)

    def save_shard(self, shard_data: Dict[str, 'SparseSpec'], path: str) -> None:
        header = {"__metadata__": {"format": "pt"}}
        data_offset = 0

        specs = list(shard_data.values())

        for spec in specs:
            header[spec.name] = {
                "dtype": spec.dtype_str,
                "shape": spec.shape,
                "data_offsets": [data_offset, data_offset + spec.size]
            }
            data_offset += spec.size

        header_json = json.dumps(header).encode('utf-8')
        header_size = len(header_json)

        with open(path, "wb") as f:
            f.write(struct.pack('<Q', header_size))
            f.write(header_json)
            if data_offset > 0:
                f.seek(data_offset - 1, 1)
                f.write(b'\0')

    @property
    def storage_format(self) -> str:
        return "safetensors"
