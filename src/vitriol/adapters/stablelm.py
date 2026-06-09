"""Adapter for StableLM / Stability AI models (StableLM, StableDiffusion, etc.)."""

import logging

from transformers import AutoConfig, PretrainedConfig

from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


class StableLMAdapter(ModelAdapter):
    """Adapter for StableLM family models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "").lower()
        if model_type in ("stablelm", "stablelm_epoch", "stableplankton"):
            return True
        architectures = getattr(config, "architectures", [])
        if architectures and any("StableLm" in arch for arch in architectures):
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        # Ensure num_key_value_heads has a default
        if not getattr(config, "num_key_value_heads", None):
            try:
                config.num_key_value_heads = getattr(config, "num_attention_heads", 1)
            except Exception:
                logger.debug("Failed to set num_key_value_heads in StableLM config")
        return config

    def register_classes(self) -> None:
        try:
            from transformers import CONFIG_MAPPING
            if "stablelm_epoch" not in CONFIG_MAPPING and "stablelm" in CONFIG_MAPPING:
                try:
                    AutoConfig.register("stablelm_epoch", CONFIG_MAPPING["stablelm"])
                    logger.info("Registered stablelm_epoch → stablelm alias.")
                except (ValueError, AttributeError):
                    pass
        except Exception:
            logger.debug("Failed to register StableLM config mapping")


# Register adapter
AdapterRegistry.register(StableLMAdapter)
