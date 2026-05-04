"""Adapter for Cohere family models (Command-R, Command-R+, Aya, etc.).

Cohere models use a custom ``model_type`` (``cohere`` / ``cohere2``) that
may not be in older transformers CONFIG_MAPPING.  This adapter registers
aliases and patches config for compatibility.
"""

import logging
from transformers import PretrainedConfig, AutoConfig
from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


class CohereAdapter(ModelAdapter):
    """Adapter for Cohere family models (Command-R, Command-R+, Aya, etc.)."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "").lower()
        if model_type.startswith("cohere"):
            return True
        architectures = getattr(config, "architectures", [])
        if architectures and any("Cohere" in arch for arch in architectures):
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        # Ensure rope_theta has a default
        if not getattr(config, "rope_theta", None):
            try:
                config.rope_theta = 8000000.0  # Cohere default
            except Exception:
                pass
        # head_dim derivation
        hidden_size = getattr(config, "hidden_size", 0) or 0
        num_heads = getattr(config, "num_attention_heads", 0) or 0
        if hidden_size > 0 and num_heads > 0 and not getattr(config, "head_dim", None):
            try:
                config.head_dim = hidden_size // num_heads
            except Exception:
                pass
        return config

    def register_classes(self):
        try:
            from transformers import CONFIG_MAPPING
            # cohere2 → cohere alias if cohere2 not available
            if "cohere2" not in CONFIG_MAPPING and "cohere" in CONFIG_MAPPING:
                try:
                    cohere_cls = CONFIG_MAPPING["cohere"]
                    AutoConfig.register("cohere2", cohere_cls)
                    logger.info("Registered cohere2 → cohere alias.")
                except (ValueError, AttributeError) as e:
                    logger.debug("cohere2 alias registration failed: %s", e)
        except Exception as e:
            logger.debug("Cohere adapter register_classes failed: %s", e)


# Register adapter
AdapterRegistry.register(CohereAdapter)
