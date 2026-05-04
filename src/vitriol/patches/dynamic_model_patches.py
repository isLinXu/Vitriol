"""
Runtime patches for dynamically loaded remote model modules.

These patches target classes loaded through transformers dynamic modules
and are kept separate from the generator so model bootstrap logic stays
focused on orchestration instead of monkey-patch implementation details.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _set_missing(obj: Any, **kv: Any) -> None:
    for k, v in kv.items():
        if not hasattr(obj, k):
            try:
                setattr(obj, k, v)
            except Exception as e:
                logger.debug("Could not set attribute %s on remote module: %s", k, e)


def patch_remote_module(module: Any) -> None:
    """Apply all known remote-module compatibility patches."""
    patch_moon_vit(module)
    patch_tie_weights_loop(module)
    patch_deepseek_lm(module)
    patch_mimo_v2_rope(module)


def patch_moon_vit(module: Any) -> None:
    cls = getattr(module, "MoonViT3dEncoder", None)
    if cls is None or getattr(cls, "_vitriol_patched", False):
        return

    original_init = cls.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        self.use_deterministic_attn = False
        for arg in args:
            if hasattr(arg, "__dict__") or hasattr(arg, "to_dict"):
                _set_missing(arg, use_deterministic_attn=False)
        for value in kwargs.values():
            if hasattr(value, "__dict__") or hasattr(value, "to_dict"):
                _set_missing(value, use_deterministic_attn=False)
        original_init(self, *args, **kwargs)

    cls.__init__ = patched_init
    cls._vitriol_patched = True
    logger.info("Patched MoonViT3dEncoder.__init__")


def patch_tie_weights_loop(module: Any) -> None:
    """Strip unsupported tie_weights kwargs from dynamic model classes."""

    def make_patch(original: Any):
        def patched(self, *args: Any, **kwargs: Any) -> Any:
            kwargs.pop("recompute_mapping", None)
            return original(self, *args, **kwargs)

        return patched

    for name, cls in module.__dict__.items():
        if not isinstance(cls, type):
            continue
        if not (name.endswith("ForConditionalGeneration") or name.endswith("ForCausalLM")):
            continue
        if not hasattr(cls, "tie_weights"):
            continue
        if getattr(cls, "_vitriol_tie_patched", False):
            continue
        cls.tie_weights = make_patch(cls.tie_weights)
        cls._vitriol_tie_patched = True
        logger.info("Patched %s.tie_weights (closure-safe)", name)


def patch_deepseek_lm(module: Any) -> None:
    cls = getattr(module, "DeepseekV3ForCausalLM", None)
    if cls is None or getattr(cls, "_vitriol_lm_patched", False):
        return

    original_init = cls.__init__

    class DummyLM:
        def tie_weights(self, *args: Any, **kwargs: Any) -> None:
            return None

        def get_input_embeddings(self) -> None:
            return None

        def set_input_embeddings(self, value: Any) -> None:
            return None

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            return None

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, "language_model", DummyLM())
        original_init(self, *args, **kwargs)
        if isinstance(self.language_model, DummyLM) and hasattr(self, "model"):
            self.language_model = self.model

    cls.__init__ = patched_init
    cls._vitriol_lm_patched = True
    logger.info("Patched DeepseekV3ForCausalLM.language_model alias")


def patch_mimo_v2_rope(module: Any) -> None:
    cfg_cls = getattr(module, "MiMoV2Config", None)
    if cfg_cls is None or getattr(cfg_cls, "_vitriol_rope_patched", False):
        return
    if hasattr(cfg_cls, "standardize_rope_params"):
        cfg_cls._vitriol_rope_patched = True
        return

    def standardize_rope_params(self) -> None:
        rope_parameters = getattr(self, "rope_parameters", None)
        if rope_parameters is None or not isinstance(rope_parameters, dict):
            rope_parameters = {}

        rope_scaling = getattr(self, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type is not None and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_type
            if rope_scaling.get("rope_theta") is not None and "rope_theta" not in rope_parameters:
                rope_parameters["rope_theta"] = rope_scaling["rope_theta"]
            if rope_scaling.get("partial_rotary_factor") is not None and "partial_rotary_factor" not in rope_parameters:
                rope_parameters["partial_rotary_factor"] = rope_scaling["partial_rotary_factor"]

        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = getattr(self, "rope_theta", 10000.0)
        if "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = getattr(self, "partial_rotary_factor", 1.0)

        self.rope_parameters = rope_parameters

    cfg_cls.standardize_rope_params = standardize_rope_params
    cfg_cls._vitriol_rope_patched = True
    logger.info("Patched MiMoV2Config.standardize_rope_params")
