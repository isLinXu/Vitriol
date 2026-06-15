"""Config-shrinking logic for minimal-weight generation.

Extracted from :mod:`generator.py` to reduce file size and allow
independent testing of the shrinking behaviour.
"""
from __future__ import annotations

import logging
from typing import Any

from ._generator_utils import (
    _SHRINK_D_CONV,
    _SHRINK_D_STATE,
    _SHRINK_HIDDEN_LAYERS,
    _SHRINK_HIDDEN_SIZE,
    _SHRINK_INTERMEDIATE_SIZE,
    _SHRINK_MOE_INTERMEDIATE_SIZE,
    _SHRINK_NUM_ATTENTION_HEADS,
    _SHRINK_NUM_EXPERTS,
    _SHRINK_NUM_EXPERTS_PER_TOK,
    _SHRINK_NUM_KEY_VALUE_HEADS,
    _SHRINK_SHARED_EXPERT_INTERMEDIATE_SIZE,
)

logger = logging.getLogger(__name__)


class ConfigShrinker:
    """Shrink a HuggingFace config to tiny dimensions for ultra-compact models."""

    # VLM vision tower uses larger min dims to avoid dimension mismatch
    _VISION_HIDDEN_SIZE: int = 64
    _VISION_NUM_ATTENTION_HEADS: int = 4
    _VISION_INTERMEDIATE_SIZE: int = 128
    _VISION_NUM_HIDDEN_LAYERS: int = 2
    _VISION_IMAGE_SIZE: int = 14
    _VISION_PATCH_SIZE: int = 2

    def shrink(self, config: Any, *, is_vision_sub: bool = False) -> None:
        """Shrink *config* to minimal dimensions.

        Args:
            config: A HuggingFace ``PretrainedConfig`` instance (or sub-config).
            is_vision_sub: When ``True``, apply VLM-compatible minimum dims for
                the vision tower so ``from_pretrained()`` does not crash on
                shape-mismatch errors.
        """

        def _set(obj: Any, attr: str, val: Any) -> None:
            if hasattr(obj, attr):
                setattr(obj, attr, val)

        def _set_force(obj: Any, attr: str, val: Any) -> None:
            """Set even if attribute doesn't exist — for important overrides."""
            try:
                setattr(obj, attr, val)
            except Exception as e:
                logger.debug("Could not _set %s on %s: %s", attr, type(obj).__name__, e)

        if is_vision_sub:
            _hidden = self._VISION_HIDDEN_SIZE
            _n_heads = self._VISION_NUM_ATTENTION_HEADS
            _n_kv_heads = self._VISION_NUM_ATTENTION_HEADS
            _inter = self._VISION_INTERMEDIATE_SIZE
            _n_layers = self._VISION_NUM_HIDDEN_LAYERS
        else:
            _hidden = _SHRINK_HIDDEN_SIZE
            _n_heads = _SHRINK_NUM_ATTENTION_HEADS
            _n_kv_heads = _SHRINK_NUM_KEY_VALUE_HEADS
            _inter = _SHRINK_INTERMEDIATE_SIZE
            _n_layers = _SHRINK_HIDDEN_LAYERS

        _set(config, "num_hidden_layers", _n_layers)
        _set(config, "num_layers", _n_layers)
        _set(config, "hidden_size", _hidden)
        _set(config, "num_attention_heads", _n_heads)
        _set(config, "num_key_value_heads", _n_kv_heads)
        _set(config, "intermediate_size", _inter)
        _set_force(config, "intermediate_size", _inter)

        if not is_vision_sub:
            _set(config, "num_experts", _SHRINK_NUM_EXPERTS)
            _set(config, "num_experts_per_tok", _SHRINK_NUM_EXPERTS_PER_TOK)
            _set(config, "n_routed_experts", _SHRINK_NUM_EXPERTS)
            _set(config, "moe_intermediate_size", _SHRINK_MOE_INTERMEDIATE_SIZE)
            _set(config, "shared_expert_intermediate_size", _SHRINK_SHARED_EXPERT_INTERMEDIATE_SIZE)
            _set(config, "expert_intermediate_size", _SHRINK_MOE_INTERMEDIATE_SIZE)
            _set(config, "d_state", _SHRINK_D_STATE)   # Mamba
            _set(config, "d_conv", _SHRINK_D_CONV)

        # Derive head dims from hidden_size / num_attention_heads
        n_heads = getattr(config, "num_attention_heads", 0) or 0
        h_size = getattr(config, "hidden_size", 0) or 0
        if n_heads > 0 and h_size > 0:
            derived_head_dim = h_size // n_heads
            _set(config, "head_dim", derived_head_dim)
            _set(config, "global_head_dim", derived_head_dim)
            if getattr(config, "model_type", None) == "glm_moe_dsa":
                original_qk_nope = getattr(config, "qk_nope_head_dim", 0) or 0
                original_qk_rope = getattr(config, "qk_rope_head_dim", 0) or 0
                original_qk_total = original_qk_nope + original_qk_rope
                rope_ratio = (
                    float(original_qk_rope) / float(original_qk_total)
                    if original_qk_total > 0 else 0.25
                )
                rope_dim = int(round(derived_head_dim * rope_ratio))
                rope_dim = max(1, rope_dim)
                if rope_dim % 2:
                    rope_dim = max(2, rope_dim - 1)
                rope_dim = min(rope_dim, max(derived_head_dim - 1, 1))
                nope_dim = max(derived_head_dim - rope_dim, 1)
                _set(config, "qk_rope_head_dim", rope_dim)
                _set(config, "qk_nope_head_dim", nope_dim)
                try:
                    qk_total = int(getattr(config, "qk_nope_head_dim", 0) or 0) + int(getattr(config, "qk_rope_head_dim", 0) or 0)
                except (ValueError, TypeError) as exc:
                    logger.debug("GLM qk_total calculation failed, using derived_head_dim: %s", exc)
                    qk_total = int(derived_head_dim)
                _set_force(config, "qk_head_dim", qk_total)
                _set(config, "v_head_dim", derived_head_dim)
            else:
                _set(config, "qk_nope_head_dim", derived_head_dim)
                _set(config, "qk_rope_head_dim", max(derived_head_dim // 4, 4))
                _set(config, "qk_head_dim", derived_head_dim)
                _set(config, "v_head_dim", derived_head_dim)
            for lr_attr in ("kv_lora_rank", "q_lora_rank"):
                if hasattr(config, lr_attr):
                    old = getattr(config, lr_attr, h_size)
                    _set(config, lr_attr, min(old or h_size, h_size))
            _set(config, "linear_key_head_dim", derived_head_dim)
            _set(config, "linear_value_head_dim", derived_head_dim)
            _set(config, "linear_num_key_heads", _n_heads)
            _set(config, "linear_num_value_heads", _n_heads)

        _set(config, "num_global_key_value_heads", _n_kv_heads)
        _set(config, "num_kv_shared_layers", 0)
        _set(config, "vision_soft_tokens_per_image", 1)
        _set(config, "depth", _n_layers)

        if is_vision_sub:
            _set(config, "image_size", self._VISION_IMAGE_SIZE)
            _set(config, "patch_size", self._VISION_PATCH_SIZE)
            _set(config, "num_channels", 3)
            _set(config, "num_positions", (self._VISION_IMAGE_SIZE // self._VISION_PATCH_SIZE) ** 2 + 1)

        _set(config, "max_position_embeddings", 4096)
        sw = getattr(config, "sliding_window", None)
        if sw is not None and isinstance(sw, int):
            _set(config, "sliding_window", 4096)

        # Recurse into sub-configs
        for sub in ("text_config", "encoder", "decoder",
                     "audio_config", "ngram_config", "decoder_config"):
            sub_cfg = getattr(config, sub, None)
            if sub_cfg is not None:
                self.shrink(sub_cfg, is_vision_sub=False)

        for sub in ("vision_config",):
            sub_cfg = getattr(config, sub, None)
            if sub_cfg is not None:
                self.shrink(sub_cfg, is_vision_sub=True)

        lt = getattr(config, "layer_types", None)
        if isinstance(lt, list):
            config.layer_types = lt[:_n_layers]

        for attr in ("hybrid_layer_pattern", "moe_layer_freq"):
            seq = getattr(config, attr, None)
            if not isinstance(seq, list):
                continue
            if len(seq) >= _n_layers:
                setattr(config, attr, seq[:_n_layers])
            else:
                setattr(config, attr, seq + [0] * (_n_layers - len(seq)))
