"""
Vitriol Configuration Constants
================================

Centralised defaults for generation, shrink, and runtime parameters.

These values are used as fallbacks throughout the codebase.  They can be
overridden via :class:`~vitriol.config.settings.ConfigManager` or
environment variables (``VITRIOL_SHRINK_*``).

All constants are grouped by subsystem for clarity.
"""

from __future__ import annotations

import os
from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Generation defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STRATEGY: str = "compact"
DEFAULT_DTYPE: str = "bfloat16"
DEFAULT_MAX_SHARD_SIZE: str = "5GB"
DEFAULT_PARALLEL_WORKERS: int = 4

# ─────────────────────────────────────────────────────────────────────────────
# Shrink config defaults (used by MinimalWeightGenerator._shrink_config)
#
# These control the dimensions of the ultra-compact "test" model that is
# built when the full-sized config would exhaust memory or fail to instantiate.
# ─────────────────────────────────────────────────────────────────────────────

SHRINK_HIDDEN_LAYERS: int = int(os.getenv("VITRIOL_SHRINK_HIDDEN_LAYERS", "2"))
SHRINK_HIDDEN_SIZE: int = int(os.getenv("VITRIOL_SHRINK_HIDDEN_SIZE", "256"))
SHRINK_NUM_ATTENTION_HEADS: int = int(os.getenv("VITRIOL_SHRINK_NUM_ATTENTION_HEADS", "2"))
SHRINK_NUM_KEY_VALUE_HEADS: int = int(os.getenv("VITRIOL_SHRINK_NUM_KEY_VALUE_HEADS", "2"))
SHRINK_INTERMEDIATE_SIZE: int = int(os.getenv("VITRIOL_SHRINK_INTERMEDIATE_SIZE", "512"))
SHRINK_NUM_EXPERTS: int = int(os.getenv("VITRIOL_SHRINK_NUM_EXPERTS", "8"))
SHRINK_NUM_EXPERTS_PER_TOK: int = int(os.getenv("VITRIOL_SHRINK_NUM_EXPERTS_PER_TOK", "2"))
SHRINK_MOE_INTERMEDIATE_SIZE: int = int(os.getenv("VITRIOL_SHRINK_MOE_INTERMEDIATE_SIZE", "64"))
SHRINK_SHARED_EXPERT_INTERMEDIATE_SIZE: int = int(
    os.getenv("VITRIOL_SHRINK_SHARED_EXPERT_INTERMEDIATE_SIZE", "64")
)
SHRINK_D_STATE: int = int(os.getenv("VITRIOL_SHRINK_D_STATE", "4"))   # Mamba
SHRINK_D_CONV: int = int(os.getenv("VITRIOL_SHRINK_D_CONV", "2"))

# ─────────────────────────────────────────────────────────────────────────────
# RoPE defaults
# ─────────────────────────────────────────────────────────────────────────────

ROPE_DEFAULTS: Dict[str, Any] = {
    "rope_type": "default",
    "rope_theta": 10000.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Custom code / asset handling
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CODE_PREFIXES: tuple = (
    "configuration_",
    "modeling_",
    "tokenization_",
    "processing_",
    "image_processing_",
    "feature_extraction_",
)

CUSTOM_ASSET_EXTENSIONS: tuple = (
    ".json",
    ".txt",
    ".model",
    ".spm",
    ".tiktoken",
    ".tokens",
    ".vocab",
    ".merges",
    ".yaml",
    ".yml",
)

BLOCKED_CUSTOM_ASSET_EXTENSIONS: tuple = (
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".msgpack",
    ".h5",
    ".pkl",
    ".pickle",
)

# ─────────────────────────────────────────────────────────────────────────────
# Fallback config chain (used when AutoConfig fails)
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_CONFIG_CHAIN: list = [
    "transformers.LlamaConfig",
    "transformers.Qwen2Config",
    "transformers.MistralConfig",
    "transformers.PhiConfig",
    "transformers.GemmaConfig",
]

# ─────────────────────────────────────────────────────────────────────────────
# Validation thresholds
# ─────────────────────────────────────────────────────────────────────────────

LOW_MEMORY_THRESHOLD_GB: float = 8.0
MEMORY_FRACTION_FOR_MODEL: float = 0.6

__all__ = [
    "DEFAULT_STRATEGY",
    "DEFAULT_DTYPE",
    "DEFAULT_MAX_SHARD_SIZE",
    "DEFAULT_PARALLEL_WORKERS",
    "SHRINK_HIDDEN_LAYERS",
    "SHRINK_HIDDEN_SIZE",
    "SHRINK_NUM_ATTENTION_HEADS",
    "SHRINK_NUM_KEY_VALUE_HEADS",
    "SHRINK_INTERMEDIATE_SIZE",
    "SHRINK_NUM_EXPERTS",
    "SHRINK_NUM_EXPERTS_PER_TOK",
    "SHRINK_MOE_INTERMEDIATE_SIZE",
    "SHRINK_SHARED_EXPERT_INTERMEDIATE_SIZE",
    "SHRINK_D_STATE",
    "SHRINK_D_CONV",
    "ROPE_DEFAULTS",
    "CUSTOM_CODE_PREFIXES",
    "CUSTOM_ASSET_EXTENSIONS",
    "BLOCKED_CUSTOM_ASSET_EXTENSIONS",
    "FALLBACK_CONFIG_CHAIN",
    "LOW_MEMORY_THRESHOLD_GB",
    "MEMORY_FRACTION_FOR_MODEL",
]
