"""
Base classes for weight generation strategies.

This module provides the abstract base class and capabilities system
for implementing different weight generation strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class StrategyCapabilities:
    """
    Declare what a strategy can and cannot do.
    
    This helps users understand strategy limitations and enables
    automatic format negotiation.
    
    Attributes:
        supports_safetensors: Whether the strategy can save in Safetensors format
        supports_training: Whether generated weights support gradient computation
        requires_contiguous: Whether the strategy requires contiguous tensors
        max_compression_ratio: Maximum compression ratio (1.0 = no compression)
        description: Human-readable description of the strategy
    """
    supports_safetensors: bool = True
    supports_training: bool = True
    requires_contiguous: bool = False
    max_compression_ratio: float = 1.0
    description: str = ""


class WeightGenerationStrategy(ABC):
    """
    Abstract base class for weight generation strategies.
    
    Each strategy defines how to generate weights for a model, with
    different trade-offs between size, training support, and compatibility.
    
    Example:
        >>> class RandomStrategy(WeightGenerationStrategy):
        ...     @property
        ...     def capabilities(self):
        ...         return StrategyCapabilities(
        ...             supports_safetensors=True,
        ...             supports_training=True,
        ...             description="Random initialization"
        ...         )
        ...
        ...     def generate_tensor(self, shape, dtype, name):
        ...         return torch.randn(shape, dtype=dtype)
    """
    
    def __init__(self, device: str = "cpu", save_dummy_config: bool = False, **kwargs):
        """
        Initialize the strategy.
        
        Args:
            device: Device to generate tensors on ("cpu", "cuda", "mps")
            save_dummy_config: Whether to save dummy configuration files
            **kwargs: Additional strategy-specific parameters
        """
        self.device = device
        self.save_dummy_config = save_dummy_config
    
    @property
    @abstractmethod
    def capabilities(self) -> StrategyCapabilities:
        """
        Return strategy capabilities.
        
        Returns:
            StrategyCapabilities object describing what this strategy can do
        """
        pass
    
    @abstractmethod
    def generate_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype,
        name: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a single tensor with the given shape and dtype.
        
        Args:
            shape: Shape of the tensor to generate
            dtype: Data type of the tensor
            name: Name of the parameter (for logging/debugging)
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            Generated tensor
        """
        pass
    
    @abstractmethod
    def save_shard(self, shard_data: Dict[str, torch.Tensor], path: str) -> None:
        """
        Save a shard of tensors to disk.
        
        Args:
            shard_data: Dict mapping parameter names to tensors
            path: Output file path
        """
        pass
    
    @property
    def storage_format(self) -> str:
        """
        Return storage format: 'safetensors' or 'pytorch'.
        
        Returns:
            Storage format string
        """
        return "safetensors" if self.capabilities.supports_safetensors else "pytorch"
    
    @property
    def file_extension(self) -> str:
        """
        Return file extension for saved shards.
        
        Returns:
            File extension (e.g., "safetensors" or "bin")
        """
        return "safetensors" if self.storage_format == "safetensors" else "bin"
    
    def get_shard_prefix(self) -> str:
        """
        Return shard filename prefix.
        
        Returns:
            Filename prefix (e.g., "model" or "pytorch_model")
        """
        return "model" if self.storage_format == "safetensors" else "pytorch_model"
    
    def set_storage_format(self, fmt: str):
        """
        Attempt to set storage format.
        
        Args:
            fmt: Desired format ("safetensors" or "pytorch")
        
        Raises:
            ValueError: If format is not supported by this strategy
        """
        if fmt == "safetensors" and not self.capabilities.supports_safetensors:
            from ..utils.exceptions import IncompatibleStrategyError
            raise IncompatibleStrategyError(
                strategy=self.__class__.__name__,
                format=fmt,
                reason="This strategy does not support Safetensors format"
            )
        
        # Subclasses can override this to actually change format
        # Default implementation just validates
        pass
    
    def _normalize_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """
        Standardize dtype: float32/float64 -> bfloat16 for storage efficiency.

        Args:
            dtype: Input dtype

        Returns:
            Normalized dtype
        """
        if dtype in (torch.float32, torch.float64):
            return torch.bfloat16
        return dtype
    
    def _validate_shape(self, shape: tuple) -> None:
        """
        Ensure shape dimensions are positive.
        
        Args:
            shape: Shape to validate
        
        Raises:
            ValueError: If shape has non-positive dimensions
        """
        if any(d <= 0 for d in shape):
            raise ValueError(f"Invalid shape: {shape}")
    
    def __repr__(self) -> str:
        """Return string representation."""
        caps = self.capabilities
        return (
            f"{self.__class__.__name__}("
            f"safetensors={caps.supports_safetensors}, "
            f"training={caps.supports_training}, "
            f"compression={caps.max_compression_ratio:.4f})"
        )

    def get_recipe(self) -> Dict[str, str]:
        """
        Return a human-readable recipe describing the generation strategy.

        Subclasses should override this to include strategy-specific parameters.

        Returns:
            Dict describing the strategy configuration
        """
        caps = self.capabilities
        return {
            "strategy": self.__class__.__name__,
            "safetensors": str(caps.supports_safetensors),
            "training": str(caps.supports_training),
            "compression_ratio": f"{caps.max_compression_ratio:.4f}",
            "device": self.device,
        }

    def validate_config(self) -> bool:
        """
        Validate the current strategy configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If configuration is inconsistent.
        """
        return True
