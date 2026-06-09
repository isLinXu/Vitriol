"""Adapter for Microsoft Phi family models (Phi-1, Phi-1.5, Phi-2, Phi-3, Phi-4).

Phi models have evolved significantly across versions.  Phi-3+ uses
``model_type=phi3`` which maps to ``Phi3ForCausalLM``.  This adapter
handles the version transitions and config patching.
"""

import logging
from typing import Optional, Type

from transformers import AutoConfig, PretrainedConfig

from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


class PhiAdapter(ModelAdapter):
    """Adapter for Microsoft Phi family models (Phi-1/1.5/2/3/4)."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "").lower()
        if model_type.startswith("phi"):
            return True
        architectures = getattr(config, "architectures", [])
        if architectures and any("Phi" in arch for arch in architectures):
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        # Phi models sometimes have None for these
        if not getattr(config, "num_key_value_heads", None):
            try:
                config.num_key_value_heads = getattr(config, "num_attention_heads", 1)
            except (AttributeError, TypeError):
                pass
        return config

    def register_classes(self) -> None:
        try:
            from transformers import CONFIG_MAPPING
            # phi4 → phi3 alias if phi4 not available
            if "phi4" not in CONFIG_MAPPING and "phi3" in CONFIG_MAPPING:
                try:
                    phi3_cls = CONFIG_MAPPING["phi3"]
                    AutoConfig.register("phi4", phi3_cls)
                    logger.info("Registered phi4 → phi3 alias.")
                except (ValueError, AttributeError) as e:
                    logger.debug("phi4 alias registration failed: %s", e)
        except Exception as e:
            logger.debug("Phi adapter register_classes failed: %s", e)

    def get_model_class(self, config: PretrainedConfig) -> Optional[Type]:
        model_type = getattr(config, "model_type", "")
        # Try version-specific model classes
        if model_type in ("phi3", "phi4"):
            try:
                from transformers import Phi3ForCausalLM
                return Phi3ForCausalLM
            except ImportError:
                pass
        if model_type in ("phi", "phi1", "phi2"):
            try:
                from transformers import PhiForCausalLM
                return PhiForCausalLM
            except ImportError:
                pass
        return None


# Register adapter — this replaces the simpler phi.py version
AdapterRegistry.register(PhiAdapter)
