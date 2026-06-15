"""Low-level helpers and constants used by the weight generator.

Keeping these in a dedicated module avoids bloating :mod:`generator.py`
with module-level plumbing and allows lightweight import of
type-safe constants/dataclasses without pulling torch/transformers.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — shrink_config defaults for the ultra (compact) test model.
# These can be overridden via the ConfigManager (VITRIOL_SHRINK_* env vars)
# or by passing explicit values to shrink_config().
# ---------------------------------------------------------------------------
_SHRINK_HIDDEN_LAYERS: int = 2
_SHRINK_HIDDEN_SIZE: int = 256
_SHRINK_NUM_ATTENTION_HEADS: int = 2
_SHRINK_NUM_KEY_VALUE_HEADS: int = 2
_SHRINK_INTERMEDIATE_SIZE: int = 512
_SHRINK_NUM_EXPERTS: int = 8
_SHRINK_NUM_EXPERTS_PER_TOK: int = 2
_SHRINK_MOE_INTERMEDIATE_SIZE: int = 64
_SHRINK_SHARED_EXPERT_INTERMEDIATE_SIZE: int = 64
_SHRINK_D_STATE: int = 4   # Mamba
_SHRINK_D_CONV: int = 2

_ROPE_DEFAULTS: Dict[str, Any] = {
    "rope_type": "default",
    "rope_theta": 10000.0,
}

_CUSTOM_CODE_PREFIXES = (
    "configuration_",
    "modeling_",
    "tokenization_",
    "processing_",
    "image_processing_",
    "feature_extraction_",
)
_CUSTOM_ASSET_EXTENSIONS = (
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
_BLOCKED_CUSTOM_ASSET_EXTENSIONS = (
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".msgpack",
    ".h5",
    ".pkl",
    ".pickle",
    ".so",
    ".dylib",
    ".dll",
    ".sh",
    ".bash",
    ".zsh",
)
_CUSTOM_CODE_MAX_FILES_ENV = "VITRIOL_CUSTOM_CODE_MAX_FILES"
_CUSTOM_CODE_MAX_PY_BYTES_ENV = "VITRIOL_CUSTOM_CODE_MAX_PY_BYTES"
_CUSTOM_CODE_MAX_ASSET_BYTES_ENV = "VITRIOL_CUSTOM_CODE_MAX_ASSET_BYTES"
_CUSTOM_CODE_DEFAULT_MAX_FILES = 32
_CUSTOM_CODE_DEFAULT_MAX_PY_BYTES = 1 * 1024 * 1024
_CUSTOM_CODE_DEFAULT_MAX_ASSET_BYTES = 50 * 1024 * 1024

# [N2 fix] regex compiled once at module level
_SHARD_ID_RE = re.compile(r"[-_](\d{5})(?:-of-|\.)")   # matches 5-digit shard index
_SHARD_NUM_RE = re.compile(r"[-_](\d+)(?:-of-|\.)")     # fallback for non-padded ids

_SAFE_KEYS = [
    "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "intermediate_size",
    "num_key_value_heads", "max_position_embeddings",
    "num_experts", "num_experts_per_tok", "n_routed_experts", "n_shared_experts",
    "moe_intermediate_size", "shared_expert_intermediate_size",
    "first_k_dense_replace", "qk_nope_head_dim", "qk_rope_head_dim",
    "qk_head_dim", "v_head_dim", "head_dim"
]

# Ordered fallback chain used when Auto* loaders fail
_FALLBACK_CHAIN: List[Tuple[str, str]] = [
    ("LlamaConfig",   "LlamaForCausalLM"),
    ("Qwen2Config",   "Qwen2ForCausalLM"),
    ("MistralConfig", "MistralForCausalLM"),
    ("PhiConfig",     "PhiForCausalLM"),
    ("GemmaConfig",   "GemmaForCausalLM"),
]

_MOE_FALLBACK_CHAIN: List[Tuple[str, str]] = [
    ("DeepseekV3Config", "DeepseekV3ForCausalLM"),
    ("DeepseekV2Config", "DeepseekV2ForCausalLM"),
    ("Qwen2MoeConfig",   "Qwen2MoeForCausalLM"),
    ("MixtralConfig",    "MixtralForCausalLM"),
]


# ---------------------------------------------------------------------------
# Type-safe helpers
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Result container for a weight generation run."""
    output_dir: str
    manifest_path: Optional[str]
    index_path: Optional[str]
    total_size: int
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "index_path": self.index_path,
            "total_size": self.total_size,
            "generated_at": self.generated_at,
        }


def positive_int_env(name: str, default: int) -> int:
    """Read a positive integer from an environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    return value


def custom_repo_file_size_limit(file_name: str) -> int:
    """Return the max allowed size for a custom repo file."""
    if file_name.lower().endswith(".py"):
        return positive_int_env(_CUSTOM_CODE_MAX_PY_BYTES_ENV, _CUSTOM_CODE_DEFAULT_MAX_PY_BYTES)
    return positive_int_env(_CUSTOM_CODE_MAX_ASSET_BYTES_ENV, _CUSTOM_CODE_DEFAULT_MAX_ASSET_BYTES)


def set_missing(obj, **kv) -> None:
    """Set attributes on *obj* only if they are absent."""
    for k, v in kv.items():
        if not hasattr(obj, k):
            try:
                setattr(obj, k, v)
            except (AttributeError, TypeError) as e:
                logger.debug("Could not set missing attribute %s on %s: %s", k, type(obj).__name__, e)


def inject_recursive(obj, attr: str, val: Any,
                     _seen: Optional[Set[int]] = None) -> None:
    """[B4 fix] Set attr=val on obj if absent; recurse into sub-configs with cycle guard."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return
    _seen.add(oid)

    if not hasattr(obj, attr):
        try:
            setattr(obj, attr, val)
        except (AttributeError, TypeError) as e:
            logger.debug("Could not inject attribute %s on %s: %s", attr, type(obj).__name__, e)

    if not hasattr(obj, "__dict__"):
        return
    for v in obj.__dict__.values():
        if isinstance(v, (str, int, float, bool, type(None))):
            continue
        if isinstance(v, (list, tuple)):
            continue
        if id(v) in _seen:
            continue
        if hasattr(v, "to_dict") or isinstance(v, dict):
            inject_recursive(v, attr, val, _seen)


def copy_safe_attrs(src, dst) -> None:
    """Copy scalar architecture attrs from src → dst, sanitising list values."""
    for key in _SAFE_KEYS:
        val = getattr(src, key, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        if val is not None:
            try:
                setattr(dst, key, val)
            except (AttributeError, TypeError) as e:
                logger.debug("Could not copy safe attr %s to %s: %s", key, type(dst).__name__, e)


def build_fallback_config(cfg_name: str, cls_name: str, hf_config):
    """[A2] Generic fallback: fresh config, copy safe attrs, return model instance."""
    import transformers as _tf
    cfg_cls = getattr(_tf, cfg_name)
    mdl_cls = getattr(_tf, cls_name)
    fb = cfg_cls()
    copy_safe_attrs(hf_config, fb)
    fb.rope_theta = 10000.0
    fb.rope_scaling = None
    if not getattr(fb, "rope_parameters", None):
        fb.rope_parameters = dict(_ROPE_DEFAULTS)
    logger.info("Fallback: initialising %s", cls_name)
    return mdl_cls(fb)


def extract_shard_id(filename: str, fallback_idx: int) -> int:
    """[N1 fix] Robustly extract the numeric shard index from a weight filename."""
    m = _SHARD_ID_RE.search(filename)
    if m:
        return int(m.group(1))
    m = _SHARD_NUM_RE.search(filename)
    if m:
        return int(m.group(1))
    return fallback_idx


# ---------------------------------------------------------------------------
# Type alias map for unknown model types
# ---------------------------------------------------------------------------

_TYPE_ALIAS_MAP: Dict[str, str] = {
    "gemma4": "gemma3",
    "gemma4_text": "gemma3_text",
    "deepseek_v4": "deepseek_v3",
    "deepseek_v3": "deepseek_v2",
    "hy3": "llama",
    "hy_v3": "llama",
    "hunyuan": "llama",
}


def find_best_alias(original_type: str) -> List[str]:
    """Find the best alias(es) for an unknown model_type.

    Strategy:
    1. Check explicit _TYPE_ALIAS_MAP first.
    2. Strip trailing version digits and try prefix match in CONFIG_MAPPING
       (e.g. "llama4" → "llama3" → "llama").
    3. Fall back to generic base types.
    """
    aliases: List[str] = []

    if original_type in _TYPE_ALIAS_MAP:
        aliases.append(_TYPE_ALIAS_MAP[original_type])

    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        base = re.sub(r'[\d_.]+$', '', original_type)
        if base and base != original_type:
            nums = re.findall(r'\d+', original_type)
            if nums:
                max_ver = int(nums[-1])
                for v in range(max_ver - 1, 0, -1):
                    candidate = re.sub(r'\d+(?=[^0-9]*$)', str(v), original_type, count=1)
                    if candidate in CONFIG_MAPPING and candidate not in aliases:
                        aliases.append(candidate)
            if base in CONFIG_MAPPING and base not in aliases:
                aliases.append(base)
    except Exception as e:
        logger.debug("Alias discovery failed: %s", e)

    return aliases


# ---------------------------------------------------------------------------
# Custom-code validation helpers
# ---------------------------------------------------------------------------

def extract_auto_map_modules(auto_map: Any) -> Set[str]:
    """Extract importable module names from HuggingFace ``auto_map`` metadata."""
    modules: Set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            ref = value.strip()
            if "." not in ref:
                return
            module_name = ref.rsplit(".", 1)[0].replace("\\", ".").replace("/", ".").strip(".")
            if module_name and all(part and part != ".." for part in module_name.split(".")):
                modules.add(module_name)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    visit(auto_map)
    return modules


def custom_code_file_matches_auto_map(file_name: str, auto_map_modules: Set[str]) -> bool:
    """Return whether *file_name* matches any module in *auto_map_modules*."""
    normalized = file_name.replace("\\", "/")
    if not normalized.lower().endswith(".py"):
        return False
    module_path = normalized[:-3].replace("/", ".").strip(".")
    module_basenames = {module.rsplit(".", 1)[-1] for module in auto_map_modules}
    return module_path in auto_map_modules or module_path.rsplit(".", 1)[-1] in module_basenames


def is_allowed_custom_repo_file(file_name: str) -> bool:
    """Return whether a repo file is safe enough to mirror into output_dir."""
    normalized = file_name.replace("\\", "/")
    if os.path.isabs(file_name):
        return False
    parts = normalized.split("/")
    if not parts or any(part in {"", ".."} for part in parts):
        return False

    base_name = parts[-1]
    lower_name = base_name.lower()
    if lower_name.endswith(".py"):
        return base_name.startswith(_CUSTOM_CODE_PREFIXES)

    if "/" not in normalized:
        return False
    if lower_name.endswith(_BLOCKED_CUSTOM_ASSET_EXTENSIONS):
        return False
    return lower_name.endswith(_CUSTOM_ASSET_EXTENSIONS)
