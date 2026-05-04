"""Adapter for Mistral family models (Mistral, Mixtral, Codestral, etc.)."""

import logging
from transformers import PretrainedConfig
from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


class MistralAdapter(ModelAdapter):
    """Adapter for Mistral family models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "").lower()
        if model_type in ("mistral", "mixtral"):
            return True
        architectures = getattr(config, "architectures", [])
        if architectures and any("Mistral" in arch for arch in architectures):
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        if hasattr(config, "sliding_window") and config.sliding_window is not None:
            # Disable sliding window for shrink mode compatibility
            try:
                config.sliding_window = None
            except Exception:
                pass
        return config


# Register adapter
AdapterRegistry.register(MistralAdapter)
