"""
Ultra compression strategy using strided tensors.

This strategy achieves extreme compression by using stride=0 tensors,
allowing a single float value to represent an arbitrarily large tensor shape.
"""

import logging
from typing import Dict

import torch

from ..utils.exceptions import IncompatibleStrategyError
from .base import StrategyCapabilities, WeightGenerationStrategy

logger = logging.getLogger(__name__)


class UltraStrategy(WeightGenerationStrategy):
    """
    Ultra compression strategy using strided tensors (stride=0).

    This strategy creates tensors with stride=0, allowing a single
    float value to represent an arbitrarily large tensor shape.

    Capabilities:
        ✅ Extreme compression (up to 99.99% size reduction)
        ✅ Zero memory allocation for weights
        ❌ Does NOT support Safetensors format
        ❌ Does NOT support gradient computation (training)

    Limitations:
        - Incompatible with Safetensors format (requires contiguous tensors)
        - Only supports PyTorch .bin format
        - May not work with all training frameworks (stride=0 gradients)
        - Not suitable for actual inference (outputs will be meaningless)

    Example:
        >>> strategy = UltraStrategy()
        >>> tensor = strategy.generate_tensor((4096, 4096), torch.bfloat16, "weight")
        >>> tensor.shape
        torch.Size([4096, 4096])
        >>> tensor.storage().size()
        1  # Only 1 element in storage!

    Technical Details:
        The trick uses PyTorch's `as_strided` to create a view with stride=0:

        ```python
        storage = torch.zeros(1, dtype=dtype)  # 1 element
        tensor = torch.as_strided(storage, shape=(4096, 4096), strides=(0, 0))
        # All elements of tensor point to the same memory location
        ```

        This allows us to "simulate" a large tensor without allocating
        the actual memory.
    """

    def __init__(
        self,
        device: str = "cpu",
        save_dummy_config: bool = False,
        **kwargs
    ):
        """
        Initialize Ultra strategy.

        Args:
            device: Device to generate tensors on
            save_dummy_config: Whether to save dummy config files
            **kwargs: Additional parameters (ignored)
        """
        super().__init__(device, save_dummy_config=save_dummy_config, **kwargs)
        self.first_tensor_logged = False
        self._storage_format = "pytorch"  # Always PyTorch for Ultra

    @property
    def capabilities(self) -> StrategyCapabilities:
        """Return Ultra strategy capabilities."""
        return StrategyCapabilities(
            supports_safetensors=False,  # ❌ NOT supported
            supports_training=False,     # ❌ May not work with gradient computation
            requires_contiguous=False,
            max_compression_ratio=0.0001,  # ~99.99% compression
            description=(
                "Ultra compression using stride=0 tensors. "
                "Extreme size reduction but incompatible with Safetensors and training."
            )
        )

    @property
    def storage_format(self) -> str:
        """Always return 'pytorch' for Ultra strategy."""
        return "pytorch"

    def set_storage_format(self, fmt: str) -> None:
        """
        Ultra strategy only supports PyTorch format.

        Args:
            fmt: Desired format

        Raises:
            IncompatibleStrategyError: If fmt is 'safetensors'
        """
        if fmt == "safetensors":
            raise IncompatibleStrategyError(
                strategy="ultra",
                format="safetensors",
                reason="Ultra strategy uses stride=0 tensors which are incompatible with Safetensors"
            )
        # Ignore other format requests

    def generate_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype,
        name: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a strided tensor with stride=0.

        Args:
            shape: Desired shape (will be simulated, not actually allocated)
            dtype: Data type (float32 -> bfloat16 conversion)
            name: Parameter name (for logging)
            **kwargs: Additional parameters (ignored)

        Returns:
            Strided tensor with stride=0

        Raises:
            ValueError: If shape has non-positive dimensions
        """
        # [Hardening] Validate shape — prevent invalid shapes
        if not shape or any(d <= 0 for d in shape):
            raise ValueError(
                f"Ultra: invalid shape {shape} for parameter '{name}'. "
                f"All dimensions must be positive."
            )

        # Convert float32 to bfloat16 for even smaller size
        if dtype == torch.float32:
            dtype = torch.bfloat16

        # Create minimal storage (1 element)
        storage = torch.zeros(1, dtype=dtype, device=self.device)

        # Create strided view with stride=0
        # All elements point to the same memory location
        tensor = torch.as_strided(storage, shape, [0] * len(shape))

        # Log first tensor for debugging
        if not self.first_tensor_logged:
            logger.info(
                f"Ultra strategy: Generated strided tensor '{name}' "
                f"with shape {shape}, stride {tensor.stride()}, "
                f"actual storage: {storage.numel()} elements"
            )
            self.first_tensor_logged = True

        return tensor

    def save_shard(self, shard_data: Dict[str, torch.Tensor], path: str) -> None:
        """
        Save shard using PyTorch format.

        Note: We always use torch.save for Ultra strategy, even if
        the filename suggests .safetensors, because stride=0 tensors
        cannot be saved in Safetensors format.

        Args:
            shard_data: Dict mapping parameter names to tensors
            path: Output file path

        Raises:
            OSError: If file cannot be written
        """
        if not shard_data:
            logger.warning("Ultra: save_shard called with empty data for %s", path)
            return

        # Always use PyTorch format for Ultra
        if path.endswith(".safetensors"):
            # Change extension to .bin
            path = path.rsplit(".", 1)[0] + ".bin"
            logger.warning(
                "Ultra strategy: Changed output format from .safetensors to .bin "
                "(stride=0 tensors incompatible with Safetensors)"
            )

        logger.info(f"Saving Ultra shard to {path}")
        try:
            torch.save(shard_data, path)
        except (OSError, RuntimeError) as e:
            from ..utils.exceptions import ShardSaveError
            raise ShardSaveError(path, str(e)) from e
