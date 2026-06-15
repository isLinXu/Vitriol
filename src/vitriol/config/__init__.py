"""Vitriol Configuration Package.

Provides:
    - :class:`GenerationConfig` — weight-generation parameter dataclass
    - :func:`build_generation_config` — unified config builder
    - :mod:`constants` — centralised default constants (shrink dims, RoPE, etc.)
    - :class:`ConfigManager` / :func:`get_config` — hierarchical settings manager
"""

from .constants import (
    BLOCKED_CUSTOM_ASSET_EXTENSIONS,
    CUSTOM_ASSET_EXTENSIONS,
    CUSTOM_CODE_PREFIXES,
    DEFAULT_DTYPE,
    DEFAULT_MAX_SHARD_SIZE,
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_STRATEGY,
    FALLBACK_CONFIG_CHAIN,
    LOW_MEMORY_THRESHOLD_GB,
    MEMORY_FRACTION_FOR_MODEL,
    ROPE_DEFAULTS,
    SHRINK_D_CONV,
    SHRINK_D_STATE,
    SHRINK_HIDDEN_LAYERS,
    SHRINK_HIDDEN_SIZE,
    SHRINK_INTERMEDIATE_SIZE,
    SHRINK_MOE_INTERMEDIATE_SIZE,
    SHRINK_NUM_ATTENTION_HEADS,
    SHRINK_NUM_EXPERTS,
    SHRINK_NUM_EXPERTS_PER_TOK,
    SHRINK_NUM_KEY_VALUE_HEADS,
    SHRINK_SHARED_EXPERT_INTERMEDIATE_SIZE,
)
from .manager import GenerationConfig as GenerationConfig
from .manager import build_generation_config as build_generation_config
from .manager import generation_config_schema as generation_config_schema
from .manager import validate_generation_dict as validate_generation_dict
from .settings import ConfigManager as ConfigManager
from .settings import get_config as get_config
from .settings import init_config as init_config

__all__ = [
    # Generation config
    "GenerationConfig",
    "build_generation_config",
    "generation_config_schema",
    "validate_generation_dict",
    # Settings manager
    "ConfigManager",
    "get_config",
    "init_config",
    # Constants
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
