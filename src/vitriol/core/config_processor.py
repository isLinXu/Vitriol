"""
Configuration processor for model loading and patching.

This module handles loading model configurations from HuggingFace Hub
and applying necessary patches for compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig

from ..patches.model_family_patches import PatchRegistry
from ..adapters.registry import AdapterRegistry
from ..utils.hf_loading import load_config as hf_load_config

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Raised when configuration loading fails."""
    pass


class ConfigProcessor:
    """
    Process and validate model configurations.
    
    This class handles:
    - Loading configs from HuggingFace Hub or local paths
    - Applying model-family specific patches
    - Applying adapter-specific patches
    - Validation and normalization
    
    Example:
        >>> processor = ConfigProcessor("Qwen/Qwen2.5-7B")
        >>> config = processor.load()
        >>> config = processor.process(config)
    """
    
    def __init__(self, model_id: str, cache_dir: Optional[str] = None):
        """
        Initialize the config processor.
        
        Args:
            model_id: Model identifier (HuggingFace Hub ID or local path)
            cache_dir: Optional cache directory for downloaded configs
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.raw_config: Optional[PretrainedConfig] = None
        self.processed_config: Optional[PretrainedConfig] = None
    
    def load(self, **kwargs: Any) -> PretrainedConfig:
        """
        Load configuration from HuggingFace Hub or local path.
        
        Args:
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained
        
        Returns:
            Loaded configuration
        
        Raises:
            ConfigLoadError: If loading fails
        """
        try:
            logger.info(f"Loading config for {self.model_id}")
            
            # Check if local path
            if Path(self.model_id).exists():
                # Local paths typically do not require remote code; still allow explicit trust_remote_code for local custom impls.
                trust_remote_code = bool(kwargs.pop("trust_remote_code", False))
                allow_network = bool(kwargs.pop("allow_network", True))
                local_files_only = bool(kwargs.pop("local_files_only", True))

                config = hf_load_config(
                    self.model_id,
                    security={
                        "trust_remote_code": trust_remote_code,
                        "allow_network": allow_network,
                        "local_files_only": local_files_only,
                    },
                    cache_dir=self.cache_dir,
                    **kwargs,
                )
            else:
                trust_remote_code = bool(kwargs.pop("trust_remote_code", True))
                allow_network = bool(kwargs.pop("allow_network", True))
                local_files_only = bool(kwargs.pop("local_files_only", False))
                config = hf_load_config(
                    self.model_id,
                    security={
                        "trust_remote_code": trust_remote_code,
                        "allow_network": allow_network,
                        "local_files_only": local_files_only,
                    },
                    cache_dir=self.cache_dir,
                    **kwargs,
                )
            
            self.raw_config = config
            logger.info(f"Config loaded successfully: {type(config).__name__}")
            return config
            
        except Exception as e:
            raise ConfigLoadError(
                f"Failed to load config for {self.model_id}. "
                f"Please check the model ID and your network connection.\n"
                f"Original error: {e}"
            ) from e
    
    def process(self, config: Optional[PretrainedConfig] = None) -> PretrainedConfig:
        """
        Apply all necessary patches to configuration.
        
        Args:
            config: Configuration to process (uses raw_config if None)
        
        Returns:
            Processed configuration
        """
        if config is None:
            config = self.raw_config
        
        if config is None:
            raise ConfigLoadError("No config loaded. Call load() first.")
        
        logger.debug(f"Processing config for {self.model_id}")
        
        # Apply family-specific patches
        PatchRegistry.apply(config, self.model_id)
        
        # Apply adapter-specific patches
        adapter = AdapterRegistry.get_adapter(self.model_id, config)
        if adapter:
            logger.debug(f"Applying adapter: {adapter.__class__.__name__}")
            config = adapter.patch_config(config)
        
        # Register custom classes if needed
        if adapter and hasattr(adapter, 'register_classes'):
            adapter.register_classes()
        
        self.processed_config = config
        return config
    
    def load_and_process(self, **kwargs: Any) -> PretrainedConfig:
        """
        Convenience method to load and process in one step.
        
        Args:
            **kwargs: Additional arguments passed to load()
        
        Returns:
            Processed configuration
        """
        config = self.load(**kwargs)
        return self.process(config)
    
    def validate(self, config: Optional[PretrainedConfig] = None) -> bool:
        """
        Validate configuration for common issues.
        
        Args:
            config: Configuration to validate (uses processed_config if None)
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If configuration has issues
        """
        if config is None:
            config = self.processed_config or self.raw_config
        
        if config is None:
            raise ValueError("No config to validate")
        
        # Check for reasonable dimensions
        hidden_size = getattr(config, "hidden_size", 0)
        if hidden_size > 100000:
            logger.warning(f"Suspiciously large hidden_size: {hidden_size}")
        
        num_layers = getattr(config, "num_hidden_layers", 0)
        if num_layers > 1000:
            logger.warning(f"Suspiciously large layer count: {num_layers}")
        
        # Check for required attributes
        required_attrs = ["hidden_size", "num_hidden_layers"]
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise ValueError(f"Missing required attribute: {attr}")
        
        return True
