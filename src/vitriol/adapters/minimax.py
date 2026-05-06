"""Adapter for MiniMax family models (MiniMax-M1, MiniMax-M2, etc.).

MiniMax models use custom rope_type='linear' and may have FP8 quantization
configs that require triton. This adapter:
* Strips quantization_config when triton is unavailable.
* Ensures rope_parameters defaults are set.
"""

import logging
from transformers import PretrainedConfig
from .base import ModelAdapter
from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


class MiniMaxAdapter(ModelAdapter):
    """Adapter for MiniMax family models."""

    @classmethod
    def match(cls, model_id: str, config: PretrainedConfig) -> bool:
        model_type = getattr(config, "model_type", "").lower()
        if "minimax" in model_type:
            return True
        # Check model_id as well
        if isinstance(model_id, str) and "minimax" in model_id.lower():
            return True
        return False

    def patch_config(self, config: PretrainedConfig) -> PretrainedConfig:
        # Strip FP8 quantization config when triton is unavailable
        try:
            import importlib.util
            if importlib.util.find_spec("triton") is None:
                if hasattr(config, "quantization_config"):
                    try:
                        delattr(config, "quantization_config")
                        logger.info("MiniMax: Removed quantization_config (triton unavailable).")
                    except AttributeError:
                        pass
        except Exception:
            logger.debug("Failed to process triton quantization config in MiniMax")

        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False

        # MiniMax checkpoints generated against newer transformers builds may
        # carry rope_parameters={"rope_type": "default"}, but current local
        # builds expect a registered rope type such as "linear".
        try:
            rope_theta = float(getattr(config, "rope_theta", 10000.0) or 10000.0)
            rope_parameters = getattr(config, "rope_parameters", None)
            if not isinstance(rope_parameters, dict):
                rope_parameters = {}

            rope_type = rope_parameters.get("rope_type") or rope_parameters.get("type")
            if rope_type in (None, "default"):
                rope_type = "linear"

            normalized = {
                "rope_type": rope_type,
                "rope_theta": rope_theta,
            }
            factor = rope_parameters.get("factor")
            if factor is not None:
                normalized["factor"] = factor

            config.rope_parameters = normalized
        except Exception:
            logger.debug("Failed to normalize rope parameters in MiniMax config")

        return config


# Register adapter
AdapterRegistry.register(MiniMaxAdapter)
