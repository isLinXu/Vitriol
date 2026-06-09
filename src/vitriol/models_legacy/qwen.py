import logging
from typing import Optional, Type

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig

from .registry import ModelAdapter, ModelRegistry

logger = logging.getLogger(__name__)

class QwenMoeAdapter(ModelAdapter):
    """Adapter for Qwen1.5-MoE and Qwen2-MoE models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        # Check by architecture or model_type
        # Qwen models often use 'qwen2_moe' as type
        if getattr(config, "model_type", "") == "qwen2_moe":
            return True
        if getattr(config, "architectures", []) and any("Qwen2Moe" in arch for arch in config.architectures):
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        # Qwen-MoE specific patches
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        if hasattr(config, "generation_config"):
            # Remove generation config to avoid warnings/errors during empty init
            if hasattr(config, "generation_config"):
                delattr(config, "generation_config")
        return config

    def register_classes(self) -> None:
        # Ensure Qwen2Moe classes are registered if transformers version is old
        try:
            from transformers import CONFIG_MAPPING
            from transformers.models.qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM

            if "qwen2_moe" not in CONFIG_MAPPING:
                try:
                    AutoConfig.register("qwen2_moe", Qwen2MoeConfig)
                    AutoModelForCausalLM.register(Qwen2MoeConfig, Qwen2MoeForCausalLM)
                    AutoModel.register(Qwen2MoeConfig, Qwen2MoeForCausalLM)
                    logger.info("Registered Qwen2Moe compatibility patch.")
                except AttributeError:
                    pass
        except ImportError:
            logger.warning("Could not import transformers.models.qwen2_moe. Qwen-MoE support may be limited.")

class Qwen35MoeAdapter(ModelAdapter):
    """Adapter for Qwen3.5-MoE models (often misidentified as qwen3_5_moe type)."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        # Check specifically for Qwen3.5 identifier or failed qwen3_5_moe type
        # Or if config loading failed previously due to unknown type (handled in core logic via try-except)
        # Here we assume config is loaded but might be generic PretrainedConfig
        # Or if we manually check raw config dict before instantiation.

        # If config is already a Qwen2MoeConfig instance but architectures implies Qwen3.5
        if getattr(config, "architectures", []) and "Qwen3_5MoeForConditionalGeneration" in config.architectures:
            return True

        # If model_type is explicitly qwen3_5_moe (which transformers might not know)
        if getattr(config, "model_type", "") == "qwen3_5_moe":
            return True

        return False

    def register_classes(self) -> None:
        try:
            from transformers import CONFIG_MAPPING
            from transformers.models.qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM

            # Create a subclass to handle model_type mismatch if needed
            # But usually registering Qwen2Moe classes for "qwen3_5_moe" key is enough
            if "qwen3_5_moe" not in CONFIG_MAPPING:
                class Qwen3_5MoeConfig(Qwen2MoeConfig):
                    """Configuration for Qwen 3.5 MoE models."""
                    model_type = "qwen3_5_moe"

                try:
                    AutoConfig.register("qwen3_5_moe", Qwen3_5MoeConfig)
                    AutoModelForCausalLM.register(Qwen3_5MoeConfig, Qwen2MoeForCausalLM)
                    AutoModel.register(Qwen3_5MoeConfig, Qwen2MoeForCausalLM)
                    logger.info("Registered Qwen3.5-MoE compatibility patch.")
                except AttributeError:
                    pass
        except ImportError:
            logger.warning("Could not import transformers.models.qwen2_moe for Qwen3.5 patching.")

    def get_model_class(self, config: PretrainedConfig) -> Optional[Type]:
        try:
            from transformers.models.qwen2_moe import Qwen2MoeForCausalLM
            return Qwen2MoeForCausalLM
        except ImportError:
            return None

# Register adapters
ModelRegistry.register(QwenMoeAdapter)
ModelRegistry.register(Qwen35MoeAdapter)
