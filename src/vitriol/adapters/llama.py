
import logging

from transformers import PretrainedConfig

from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)

class LlamaAdapter(ModelAdapter):
    """Adapter for Llama models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "")
        if "llama" in model_type.lower():
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        """Fix Llama-specific config quirks."""
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False

        # Ensure rope_scaling is compatible
        # Some Llama 3.1 models have complex scaling factors
        return config

# Register adapter
AdapterRegistry.register(LlamaAdapter)
