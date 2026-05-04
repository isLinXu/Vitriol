
import logging
from transformers import PretrainedConfig
from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)

class DeepSeekAdapter(ModelAdapter):
    """Adapter for DeepSeek models (V2, V3, MoE)."""
    
    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "")
        if "deepseek" in model_type.lower():
            return True
            
        architectures = getattr(config, "architectures", [])
        if architectures and any("Deepseek" in arch for arch in architectures):
            return True
            
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        if hasattr(config, "generation_config"):
            delattr(config, "generation_config")
        return config

# Register adapter
AdapterRegistry.register(DeepSeekAdapter)
