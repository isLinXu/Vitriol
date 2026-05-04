"""
Random initialization strategy.

Generates weights using standard random initialization, suitable for
training and most use cases.
"""

import torch
from typing import Dict

from .base import WeightGenerationStrategy, StrategyCapabilities


class RandomStrategy(WeightGenerationStrategy):
    """
    Random initialization strategy.
    
    Generates weights using standard normal distribution (mean=0, std=1).
    This is the closest to real model initialization and supports
    all operations including training.
    
    Capabilities:
        ✅ Supports Safetensors format
        ✅ Supports gradient computation (training)
        ✅ Most realistic initialization
        ❌ No compression (full size)
    
    Example:
        >>> strategy = RandomStrategy()
        >>> tensor = strategy.generate_tensor((1024, 1024), torch.float32, "weight")
        >>> tensor.shape
        torch.Size([1024, 1024])
        >>> tensor.mean()  # Should be close to 0
        tensor(0.0012)
    """
    
    @property
    def capabilities(self) -> StrategyCapabilities:
        """Return Random strategy capabilities."""
        return StrategyCapabilities(
            supports_safetensors=True,
            supports_training=True,
            requires_contiguous=False,
            max_compression_ratio=1.0,  # No compression
            description=(
                "Random initialization using standard normal distribution. "
                "Most realistic but no compression."
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
        Generate a random tensor.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            name: Parameter name
            **kwargs: Additional parameters (ignored)
        
        Returns:
            Random tensor
        """
        dtype = self._normalize_dtype(dtype)
        self._validate_shape(shape)
        
        return torch.randn(shape, dtype=dtype, device=self.device)
    
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
                save_file(shard_data, path)
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
