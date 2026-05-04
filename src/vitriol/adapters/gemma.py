"""Adapter for Gemma family models (Gemma 2, Gemma 3, Gemma 4, etc.).

Gemma-4 multimodal models have a vision tower and text_config that require
special handling during shrink mode.  This adapter:

* Strips ``quantization_config`` when triton is unavailable (CPU/Mac).
* Registers ``gemma4`` / ``gemma4_text`` type aliases.
* Patches vision_config compatibility for VLM shrink mode — rebuilds the
  vision_config dict from top-level promoted fields so that
  ``from_pretrained()`` can correctly instantiate the vision tower even
  after the generator strips raw-dict sub-configs from config.json.
* Rebuilds ``text_config`` from top-level language-model fields so that
  ``Gemma4ForConditionalGeneration`` creates the correct number of layers
  (shrink: 2) instead of falling back to the default (e.g. 30).
"""

import logging
from typing import Optional, Type
from transformers import PretrainedConfig, AutoConfig
from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


# ── Fields that the Gemma-4 vision tower needs ──────────────────────────
_VISION_FIELDS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "intermediate_size", "head_dim",
    "image_size", "patch_size", "num_channels", "num_positions",
    "vision_soft_tokens_per_image",
)

# ── Fields that Gemma-4 text_config needs ───────────────────────────────
_TEXT_FIELDS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "intermediate_size", "head_dim",
    "vocab_size", "max_position_embeddings", "rms_norm_eps",
    "rope_theta", "sliding_window", "num_experts",
    "expert_intermediate_size", "moe_intermediate_size",
    "num_kv_shared_layers", "num_global_key_value_heads",
    "global_head_dim", "hidden_activation",
    "final_logit_softcapping", "use_bidirectional_attention",
    "use_double_wide_mlp", "use_clipped_linears",
    "use_deterministic_attn", "use_flash_attention_2",
    "use_flash_attn", "attention_bias", "attention_dropout",
    "attention_k_eq_v", "enable_moe_block",
    "hidden_size_per_layer_input", "vocab_size_per_layer_input",
    "default_output_length",
)


class GemmaAdapter(ModelAdapter):
    """Adapter for Gemma family models (Gemma 2/3/4, PaliGemma, etc.)."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "").lower()
        if model_type.startswith("gemma"):
            return True
        architectures = getattr(config, "architectures", [])
        if architectures and any("Gemma" in arch for arch in architectures):
            return True
        return False

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_multimodal(config: PretrainedConfig) -> bool:
        """Heuristic: does this config represent a multimodal (VLM) model?"""
        architectures = getattr(config, "architectures", [])
        if any("ConditionalGeneration" in a for a in architectures):
            return True
        if any("Vision" in a for a in architectures):
            return True
        # Check for vision-specific top-level fields
        if getattr(config, "vision_soft_tokens_per_image", None):
            return True
        if getattr(config, "image_token_id", None) is not None:
            return True
        # A vision_config that was promoted (dict → PretrainedConfig) still
        # counts as multimodal.
        vc = getattr(config, "vision_config", None)
        if vc is not None:
            return True
        return False

    @staticmethod
    def _rebuild_vision_config(config: PretrainedConfig) -> None:
        """Reconstruct vision_config from promoted top-level fields.

        The generator's ``_save_configs`` strips raw-dict sub-configs from
        config.json.  For multimodal Gemma-4 models this means the
        vision_config disappears and ``from_pretrained()`` can't create the
        vision tower.

        This method gathers known vision-related fields from the top-level
        config and builds a new ``PretrainedConfig`` under the
        ``vision_config`` attribute so the model class can find it.
        """
        vc = getattr(config, "vision_config", None)
        if vc is not None:
            # Already present — nothing to do
            return

        # Build a dict from vision-specific top-level fields
        vision_dict: dict = {}
        for field in _VISION_FIELDS:
            val = getattr(config, field, None)
            if val is not None:
                vision_dict[field] = val

        # We also need model_type for the vision config
        # Gemma-4 uses Siglip-like vision encoder
        vision_dict.setdefault("model_type", "siglip_vision_model")
        vision_dict.setdefault("hidden_size", 1152)
        vision_dict.setdefault("num_hidden_layers", 2)
        vision_dict.setdefault("num_attention_heads", 16)
        vision_dict.setdefault("num_key_value_heads", 16)
        vision_dict.setdefault("intermediate_size", 4304)
        vision_dict.setdefault("image_size", 14)
        vision_dict.setdefault("patch_size", 2)
        vision_dict.setdefault("num_channels", 3)

        try:
            config.vision_config = PretrainedConfig.from_dict(vision_dict)
            logger.info("Rebuilt vision_config for Gemma multimodal model.")
        except Exception as e:
            logger.debug("Could not rebuild vision_config: %s", e)

    @staticmethod
    def _rebuild_text_config(config: PretrainedConfig) -> None:
        """Reconstruct text_config from promoted top-level fields.

        ``Gemma4ForConditionalGeneration`` internally calls
        ``AutoModel.from_config(config.text_config)`` to create the language
        model.  When the generator's ``_save_configs`` strips the nested
        ``text_config`` dict from config.json, this attribute disappears and
        transformers falls back to the architecture's default (e.g. 30 layers
        for Gemma-4-31B), creating a mismatch with the actual 2-layer shrink
        weights.

        This method gathers language-model fields from the top-level config
        and builds a ``PretrainedConfig`` under ``text_config`` so the model
        class instantiates the correct (shrink) number of layers.
        """
        tc = getattr(config, "text_config", None)
        if tc is not None:
            # Already present — nothing to do
            return

        text_dict: dict = {}
        for field in _TEXT_FIELDS:
            val = getattr(config, field, None)
            if val is not None:
                text_dict[field] = val

        # Required fields with sensible shrink defaults
        text_dict.setdefault("model_type", "gemma3_text")
        text_dict.setdefault("hidden_size", 256)
        text_dict.setdefault("num_hidden_layers", 2)
        text_dict.setdefault("num_attention_heads", 2)
        text_dict.setdefault("num_key_value_heads", 2)
        text_dict.setdefault("intermediate_size", 512)
        text_dict.setdefault("vocab_size", 262144)
        text_dict.setdefault("head_dim", 128)

        try:
            config.text_config = PretrainedConfig.from_dict(text_dict)
            logger.info("Rebuilt text_config for Gemma multimodal model.")
        except Exception as e:
            logger.debug("Could not rebuild text_config: %s", e)

    # ── main hooks ──────────────────────────────────────────────────────

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        # Strip FP8 quantization config when triton is unavailable
        try:
            import importlib.util
            if importlib.util.find_spec("triton") is None:
                if hasattr(config, "quantization_config"):
                    try:
                        delattr(config, "quantization_config")
                        logger.info("Removed quantization_config (triton unavailable).")
                    except AttributeError:
                        pass
                # Also check if quantization_config is stored differently
                if hasattr(config, "_quantization_config"):
                    try:
                        delattr(config, "_quantization_config")
                    except AttributeError:
                        pass
        except Exception:
            pass

        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False

        # Ensure head_dim is derived correctly
        hidden_size = getattr(config, "hidden_size", 0) or 0
        num_heads = getattr(config, "num_attention_heads", 0) or 0
        if hidden_size > 0 and num_heads > 0 and not getattr(config, "head_dim", None):
            try:
                config.head_dim = hidden_size // num_heads
            except Exception:
                pass

        # ── Vision config handling for multimodal Gemma models ─────────
        if self._is_multimodal(config):
            self._rebuild_vision_config(config)
            self._rebuild_text_config(config)

        return config

    def register_classes(self):
        """Register Gemma-4 type aliases if needed."""
        try:
            from transformers import CONFIG_MAPPING

            # gemma4 → gemma3 alias
            if "gemma4" not in CONFIG_MAPPING and "gemma3" in CONFIG_MAPPING:
                try:
                    gemma3_cls = CONFIG_MAPPING["gemma3"]
                    AutoConfig.register("gemma4", gemma3_cls)
                    logger.info("Registered gemma4 → gemma3 alias.")
                except (ValueError, AttributeError) as e:
                    logger.debug("gemma4 alias registration failed: %s", e)

            # gemma4_text → gemma3_text alias
            if "gemma4_text" not in CONFIG_MAPPING and "gemma3_text" in CONFIG_MAPPING:
                try:
                    gemma3_text_cls = CONFIG_MAPPING["gemma3_text"]
                    AutoConfig.register("gemma4_text", gemma3_text_cls)
                    logger.info("Registered gemma4_text → gemma3_text alias.")
                except (ValueError, AttributeError) as e:
                    logger.debug("gemma4_text alias registration failed: %s", e)
        except Exception as e:
            logger.debug("Gemma adapter register_classes failed: %s", e)

    def get_model_class(self, config: PretrainedConfig) -> Optional[Type]:
        model_type = getattr(config, "model_type", "")
        # For text-only Gemma models, use GemmaForCausalLM
        if model_type in ("gemma3_text", "gemma4_text", "gemma2", "gemma"):
            try:
                from transformers import GemmaForCausalLM
                return GemmaForCausalLM
            except ImportError:
                pass
        # For multimodal Gemma, let AutoModel handle it
        return None


# Register adapter
AdapterRegistry.register(GemmaAdapter)
