"""
Compact strategy using zero-filled tensors.

Generates minimal weight files using zeros with optional caching
for fast generation.
"""

from typing import Dict

import torch

from .base import StrategyCapabilities, WeightGenerationStrategy


class CompactStrategy(WeightGenerationStrategy):
    """
    Compact strategy using zero-filled tensors.

    Generates weights filled with zeros, achieving good compression
    while maintaining compatibility with all formats.

    Capabilities:
        ✅ Supports Safetensors format
        ⚠️ Limited training support (zeros may cause gradient issues)
        ✅ Good compression (compresses well with zip/gzip)
        ✅ Fast generation

    Example:
        >>> strategy = CompactStrategy()
        >>> tensor = strategy.generate_tensor((1024, 1024), torch.float32, "weight")
        >>> tensor.shape
        torch.Size([1024, 1024])
        >>> tensor.sum()
        tensor(0.)
    """

    def __init__(
        self,
        device: str = "cpu",
        save_dummy_config: bool = False,
        cache_size: int = 100,
        **kwargs
    ):
        """
        Initialize Compact strategy.

        Args:
            device: Device to generate tensors on
            save_dummy_config: Whether to save dummy config files
            cache_size: Number of tensors to cache for reuse
            **kwargs: Additional parameters
        """
        super().__init__(device, save_dummy_config=save_dummy_config, **kwargs)
        self.cache_size = cache_size
        self._cache: Dict[tuple, torch.Tensor] = {}

    @property
    def capabilities(self) -> StrategyCapabilities:
        """Return Compact strategy capabilities."""
        return StrategyCapabilities(
            supports_safetensors=True,
            supports_training=True,  # Technically supports, but may have issues
            requires_contiguous=False,
            max_compression_ratio=0.01,  # Zeros compress very well
            description=(
                "Zero-filled tensors. Good compression and fast generation. "
                "May cause training issues with gradient flow."
            )
        )

    def generate_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype,
        name: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a zero-filled tensor (with caching).

        Args:
            shape: Tensor shape
            dtype: Data type
            name: Parameter name
            **kwargs: Additional parameters (ignored)

        Returns:
            Zero-filled tensor
        """
        dtype = self._normalize_dtype(dtype)
        self._validate_shape(shape)

        # Check cache
        cache_key = (shape, dtype)
        if cache_key in self._cache:
            # Return a clone to avoid shared storage issues
            return self._cache[cache_key].clone()

        # Generate new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=self.device)

        # Add to cache if space available
        if len(self._cache) < self.cache_size:
            self._cache[cache_key] = tensor.clone()

        return tensor

    def save_shard(self, shard_data: Dict[str, torch.Tensor], path: str) -> None:
        """
        Save shard to disk.

        Args:
            shard_data: Dict mapping parameter names to tensors
            path: Output file path

        Raises:
            OSError: If file cannot be written
        """
        if not shard_data:
            return

        if self.storage_format == "safetensors":
            try:
                from safetensors.torch import save_file
                # Ensure tensors are contiguous (safetensors requirement)
                contiguous_data = {}
                for k, v in shard_data.items():
                    if not v.is_contiguous():
                        v = v.contiguous()
                    contiguous_data[k] = v
                save_file(contiguous_data, path)
                return
            except ImportError:
                import logging
                logging.getLogger(__name__).warning(
                    "safetensors not installed, falling back to PyTorch format."
                )
            except (OSError, RuntimeError) as e:
                import logging
                logging.getLogger(__name__).warning(
                    "safetensors save failed for %s: %s. Trying PyTorch fallback.",
                    path, e,
                )
            # Fallback to PyTorch
            fallback_path = path.replace(".safetensors", ".bin") if path.endswith(".safetensors") else path
            torch.save(shard_data, fallback_path)
        else:
            torch.save(shard_data, path)

    def clear_cache(self) -> None:
        """Clear the tensor cache to free memory."""
        self._cache.clear()
