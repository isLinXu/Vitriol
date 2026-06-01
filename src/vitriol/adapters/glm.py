"""Adapter for GLM family models (GLM-4, GLM-5, ChatGLM, etc.).

GLM models use MLA (Multi-head Latent Attention) and have custom
rope/qk dimensions that need careful handling during shrink.
"""

import logging
from typing import Optional, Type

from transformers import PretrainedConfig

from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


class GLMAdapter(ModelAdapter):
    """Adapter for GLM family models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "").lower()
        if model_type in ("glm4", "glm5", "glm_moe_dsa", "chatglm", "glm"):
            return True
        if "glm" in model_type:
            return True
        architectures = getattr(config, "architectures", [])
        if architectures and any("Glm" in arch or "ChatGLM" in arch for arch in architectures):
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        # Ensure add_bias_linear and add_qkv_bias are set
        if not hasattr(config, "add_bias_linear"):
            try:
                config.add_bias_linear = False
            except Exception:
                logger.debug("Failed to set add_bias_linear in GLM config")
        if not hasattr(config, "add_qkv_bias"):
            try:
                config.add_qkv_bias = True
            except Exception:
                logger.debug("Failed to set add_qkv_bias in GLM config")
        return config

    def get_model_class(self, config: PretrainedConfig) -> Optional[Type]:
        # Let AutoModel handle GLM models since they may use custom modeling code
        return None


# Register adapter
AdapterRegistry.register(GLMAdapter)
