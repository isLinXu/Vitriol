
import logging
from typing import Optional, Type

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig

from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


class QwenMoeAdapter(ModelAdapter):
    """Adapter for Qwen1.5-MoE and Qwen2-MoE models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        if getattr(config, "model_type", "") == "qwen2_moe":
            return True
        if getattr(config, "architectures", []) and any("Qwen2Moe" in arch for arch in config.architectures):
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        if hasattr(config, "generation_config"):
            delattr(config, "generation_config")
        return config

    def register_classes(self):
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
    """Adapter for Qwen3.5-MoE models.

    Qwen3.5-MoE uses ``model_type: qwen3_5_moe`` which is not in standard
    ``CONFIG_MAPPING`` of older transformers releases.  The original HF config
    stores the *real* architecture parameters inside ``text_config`` and
    ``vision_config`` sub-dicts while leaving the top-level values as ``None``.

    This adapter:
    * Promotes text_config scalar fields up to the top-level config so that
      ``Qwen2MoeConfig`` / the model can read them directly.
    * Strips ``text_config`` / ``vision_config`` dicts so they don't get
      serialised as raw dicts in ``config.json`` (which would crash
      ``PretrainedConfig`` deserialisation).
    * Registers ``qwen3_5_moe`` → ``Qwen2MoeConfig`` for ``AutoConfig`` /
      ``AutoModelForCausalLM`` if not already present.
    """

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        if getattr(config, "architectures", []) and "Qwen3_5MoeForConditionalGeneration" in config.architectures:
            return True
        if getattr(config, "model_type", "") == "qwen3_5_moe":
            return True
        return False

    # ── Config patching ──────────────────────────────────────────────────

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        """Flatten sub-configs into the top-level config for compatibility.

        Qwen3.5 stores the real architecture inside ``text_config``.  The
        top-level ``hidden_size``, ``num_hidden_layers``, etc. are ``None``.
        We promote those scalar values up so that ``Qwen2MoeConfig`` can
        construct the model, then *remove* the sub-config dicts to prevent
        them from being serialised as raw ``dict`` objects in ``config.json``.
        """
        # 0. If config was loaded via a type-alias (e.g. deepseek_v3), correct
        #    model_type back to qwen3_5_moe so that save_pretrained produces the
        #    correct type tag.
        if getattr(config, "model_type", "") != "qwen3_5_moe":
            archs = getattr(config, "architectures", [])
            if "Qwen3_5MoeForConditionalGeneration" in archs:
                try:
                    config.model_type = "qwen3_5_moe"
                    logger.info(
                        "Corrected model_type '%s' → 'qwen3_5_moe' "
                        "(detected via architecture).",
                        getattr(config, "model_type", "N/A"),
                    )
                except Exception as e:
                    logger.debug("Could not correct model_type: %s", e)

        # 1. Promote text_config scalar fields to top-level
        self._promote_sub_config(config, "text_config")
        self._promote_sub_config(config, "vision_config")

        # 2. Remove sub-config dict attributes so save_pretrained()
        #    does NOT serialise them as raw dicts (which breaks
        #    PretrainedConfig deserialisation later).
        for sub in ("text_config", "vision_config",
                     "encoder_config", "decoder_config"):
            sub_val = getattr(config, sub, None)
            if isinstance(sub_val, dict):
                try:
                    delattr(config, sub)
                    logger.info(
                        "Removed raw dict sub-config '%s' from %s "
                        "to prevent serialisation issues.",
                        sub, type(config).__name__,
                    )
                except AttributeError:
                    pass

        # 3. Ensure required scalar fields are not None
        for attr in ("vocab_size", "hidden_size", "num_hidden_layers",
                      "num_attention_heads", "intermediate_size",
                      "num_key_value_heads"):
            val = getattr(config, attr, None)
            if val is None:
                # Try to infer sensible defaults
                defaults = {
                    "vocab_size": 32000,
                    "hidden_size": getattr(config, "hidden_size", 2048) or 2048,
                    "num_hidden_layers": getattr(config, "num_hidden_layers", 2) or 2,
                    "num_attention_heads": getattr(config, "num_attention_heads", 8) or 8,
                    "intermediate_size": getattr(config, "intermediate_size", 512) or 512,
                    "num_key_value_heads": getattr(config, "num_attention_heads", 8) or 8,
                }
                try:
                    setattr(config, attr, defaults[attr])
                except Exception as e:
                    logger.debug("Could not set default for %s: %s", attr, e)

        return config

    @staticmethod
    def _promote_sub_config(config: PretrainedConfig, sub_attr: str) -> None:
        """Promote scalar attrs from a sub-config dict/object to top-level."""
        sub = getattr(config, sub_attr, None)
        if sub is None:
            return

        # Sub-config could be a PretrainedConfig object or a raw dict
        src = (sub.to_dict() if hasattr(sub, "to_dict")
               else (sub if isinstance(sub, dict) else {}))

        promoted = 0
        for k, v in src.items():
            # Only promote scalar, non-None values that are missing or None at top-level
            if not isinstance(v, (int, float, str, bool)):
                continue
            current = getattr(config, k, None)
            if current is None:
                try:
                    setattr(config, k, v)
                    promoted += 1
                except Exception:
                    pass

        if promoted:
            logger.info(
                "Promoted %d scalar fields from %s to top-level %s.",
                promoted, sub_attr, type(config).__name__,
            )

    # ── Class registration ───────────────────────────────────────────────

    def register_classes(self):
        try:
            from transformers import CONFIG_MAPPING
            from transformers.models.qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM

            if "qwen3_5_moe" not in CONFIG_MAPPING:
                # Create a dynamic Config subclass with model_type = "qwen3_5_moe"
                Qwen3_5MoeConfig = type(
                    "Qwen3_5MoeConfig", (Qwen2MoeConfig,),
                    {"model_type": "qwen3_5_moe"},
                )

                # Register the config type for AutoConfig
                try:
                    AutoConfig.register("qwen3_5_moe", Qwen3_5MoeConfig)
                except (AttributeError, ValueError) as e:
                    logger.debug("AutoConfig.register failed: %s", e)

                # Create a Model wrapper whose config_class matches the dynamic config
                # so AutoModelForCausalLM.register does not complain about mismatch.
                _OrigModel = Qwen2MoeForCausalLM
                Qwen3_5MoeForCausalLM = type(
                    "Qwen3_5MoeForCausalLM", (_OrigModel,),
                    {"config_class": Qwen3_5MoeConfig},
                )

                try:
                    AutoModelForCausalLM.register(
                        Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM,
                    )
                except (AttributeError, ValueError) as e:
                    logger.debug("AutoModelForCausalLM.register failed: %s", e)

                try:
                    AutoModel.register(Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM)
                except (AttributeError, ValueError) as e:
                    logger.debug("AutoModel.register failed: %s", e)

                # Verify registration succeeded
                from transformers.models.auto.configuration_auto import CONFIG_MAPPING as _CM
                if "qwen3_5_moe" in _CM:
                    logger.info("Registered Qwen3.5-MoE compatibility patch.")
                else:
                    logger.warning(
                        "Qwen3.5-MoE config registration did not persist in "
                        "CONFIG_MAPPING.  Alias fallback will be used."
                    )
        except ImportError:
            logger.warning("Could not import transformers.models.qwen2_moe for Qwen3.5 patching.")

    def get_model_class(self, config: PretrainedConfig) -> Optional[Type]:
        try:
            from transformers.models.qwen2_moe import Qwen2MoeForCausalLM
            return Qwen2MoeForCausalLM
        except ImportError:
            return None


# Register adapters
AdapterRegistry.register(QwenMoeAdapter)
AdapterRegistry.register(Qwen35MoeAdapter)
