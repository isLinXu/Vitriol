import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Type

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Base class for model-specific adapters.
    Allows injecting custom logic for config patching, model instantiation, and weight generation.

    Subclasses must implement:
        - match(cls, model_id, config): Whether this adapter applies

    Optional overrides:
        - patch_config(config): Modify config before model instantiation
        - register_classes(): Register custom classes with AutoConfig/AutoModel
        - get_model_class(config): Return a specific model class to use
    """

    @classmethod
    @abstractmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        """Check if this adapter applies to the given model.

        Args:
            model_id: HuggingFace model ID or local path
            config: PretrainedConfig from AutoConfig

        Returns:
            True if this adapter should handle the model
        """
        pass

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        """Modify config before model instantiation (e.g. fix attributes).

        Args:
            config: The config to patch

        Returns:
            The patched config (may be the same object, modified in-place)
        """
        return config

    def register_classes(self) -> None:  # noqa: B027  (optional hook for subclasses)
        """Register custom classes with AutoConfig/AutoModel if needed.

        This is called once when the adapter is selected. Use it to
        register custom model types that aren't in the transformers library.
        """
        pass

    def get_model_class(self, config: PretrainedConfig) -> Optional[Type]:
        """Return a specific model class to use, or None for AutoModel.

        Args:
            config: The model config

        Returns:
            A model class (e.g. LlamaForCausalLM) or None
        """
        return None

    def validate_config(self, config: PretrainedConfig) -> bool:
        """Validate that the config is compatible with this adapter.

        Args:
            config: The config to validate

        Returns:
            True if the config is valid

        Raises:
            ValueError: If the config has incompatible settings
        """
        return True


class DefaultAdapter(ModelAdapter):
    """Default adapter that handles standard transformers models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        return True
