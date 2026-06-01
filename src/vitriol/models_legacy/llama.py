import logging

from transformers import PretrainedConfig

from .registry import ModelAdapter, ModelRegistry

logger = logging.getLogger(__name__)

class LlamaAdapter(ModelAdapter):
    """Adapter for Llama models (Llama 2, Llama 3, Llama 3.1)."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "")
        if "llama" in model_type.lower():
            return True

        architectures = getattr(config, "architectures", [])
        if architectures and any("Llama" in arch for arch in architectures):
            return True

        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        # Common patches for Llama models
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False

        # Llama 3.1 405B might have rope_scaling issues if transformers version is old
        # But for weight generation we usually don't care about rope implementation details
        # unless they affect param shapes (which they don't usually)

        # Remove generation config to avoid warnings/errors during empty init
        if hasattr(config, "generation_config"):
            delattr(config, "generation_config")

        return config

# Register adapter
ModelRegistry.register(LlamaAdapter)
