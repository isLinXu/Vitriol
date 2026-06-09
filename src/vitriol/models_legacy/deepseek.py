import logging

from transformers import PretrainedConfig

from .registry import ModelAdapter, ModelRegistry

logger = logging.getLogger(__name__)

class DeepSeekAdapter(ModelAdapter):
    """Adapter for DeepSeek models (V2, V3, MoE)."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "")
        # DeepSeek often uses 'deepseek_v2' or 'deepseek_v3' (though v3 might use v2 config class)
        # Also check for 'deepseek' in architectures or config name
        if "deepseek" in model_type.lower():
            return True

        architectures = getattr(config, "architectures", [])
        if architectures and any("Deepseek" in arch for arch in architectures):
            return True

        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        # Common patches for DeepSeek models
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False

        # Remove generation config to avoid warnings/errors during empty init
        if hasattr(config, "generation_config"):
            delattr(config, "generation_config")

        # Ensure trust_remote_code is implied (though core handles it)
        # DeepSeek often requires specific attention implementation details
        # We might need to ensure 'flash_attn' is not strictly required if running on CPU?
        # But for meta device init, it shouldn't matter as we don't run forward pass.

        return config

    def register_classes(self) -> None:
        # DeepSeek models rely heavily on remote code, so standard registration might not be needed
        # unless we want to patch specific classes.
        # However, for V2/V3, sometimes local code is safer if remote code fails.
        # For now, we rely on trust_remote_code being enabled in core.
        pass

# Register adapter
ModelRegistry.register(DeepSeekAdapter)
