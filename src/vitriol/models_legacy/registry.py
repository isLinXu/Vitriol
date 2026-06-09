import logging
from abc import ABC, abstractmethod
from typing import Optional, Type

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

class ModelAdapter(ABC):
    """
    Base class for model-specific adapters.
    Allows injecting custom logic for config patching, model instantiation, and weight generation.
    """

    @classmethod
    @abstractmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        """Check if this adapter applies to the given model."""
        pass

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        """Modify config before model instantiation (e.g. fix attributes)."""
        return config

    def register_classes(self) -> None:  # noqa: B027  (optional hook for subclasses)
        """Register custom classes with AutoConfig/AutoModel if needed."""
        pass

    def get_model_class(self, config: PretrainedConfig) -> Optional[Type]:
        """Return a specific model class to use, or None for AutoModel."""
        return None

class DefaultAdapter(ModelAdapter):
    """Default adapter that handles standard transformers models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        return True

class ModelRegistry:
    """Registry for legacy model configurations."""
    _adapters: list[Type[ModelAdapter]] = []

    @classmethod
    def register(cls, adapter_cls: Type[ModelAdapter]) -> None:
        cls._adapters.insert(0, adapter_cls) # LIFO priority

    @classmethod
    def get_adapter(cls, model_id: str, config: PretrainedConfig) -> ModelAdapter:
        for adapter_cls in cls._adapters:
            if adapter_cls.match(model_id, config):
                logger.info(f"Using adapter: {adapter_cls.__name__}")
                return adapter_cls()
        return DefaultAdapter()
