from __future__ import annotations

import datetime
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
import shutil
import traceback
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants — shrink_config defaults for the ultra (compact) test model
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

import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
)
from transformers.utils import cached_file  # noqa: E402

from ..patches import PatchRegistry, apply_all_patches, patch_remote_module  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Logger — must exist before any module-level patch references it
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


_ROPE_DEFAULTS: Dict[str, Any] = {
    "rope_type": "default",
    "rope_theta": 10000.0,
}


@dataclass
class GenerationResult:
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


def _set_missing(obj, **kv) -> None:
    for k, v in kv.items():
        if not hasattr(obj, k):
            try:
                setattr(obj, k, v)
            except Exception as e:
                logger.debug("Could not set missing attribute %s on %s: %s", k, type(obj).__name__, e)


apply_all_patches()


# ═════════════════════════════════════════════════════════════════════════════
# § 2  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

# [N2 fix] regex compiled once at module level
_SHARD_ID_RE  = re.compile(r"[-_](\d{5})(?:-of-|\.)")   # matches 5-digit shard index
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


def _inject_recursive(obj, attr: str, val: Any,
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
        except Exception as e:
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
            _inject_recursive(v, attr, val, _seen)


def _copy_safe_attrs(src, dst) -> None:
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
            except Exception as e:
                logger.debug("Could not copy safe attr %s to %s: %s", key, type(dst).__name__, e)


def _build_fallback_config(cfg_name: str, cls_name: str, hf_config):
    """[A2] Generic fallback: fresh config, copy safe attrs, return model instance."""
    import transformers as _tf
    cfg_cls = getattr(_tf, cfg_name)
    mdl_cls = getattr(_tf, cls_name)
    fb = cfg_cls()
    _copy_safe_attrs(hf_config, fb)
    fb.rope_theta   = 10000.0
    fb.rope_scaling = None
    if not getattr(fb, "rope_parameters", None):
        setattr(fb, "rope_parameters", dict(_ROPE_DEFAULTS))
    logger.info("Fallback: initialising %s", cls_name)
    return mdl_cls(fb)


def _extract_shard_id(filename: str, fallback_idx: int) -> int:
    """[N1 fix] Robustly extract the numeric shard index from a weight filename."""
    # Try anchored 5-digit pattern first
    m = _SHARD_ID_RE.search(filename)
    if m:
        return int(m.group(1))
    # Try any digit run after a separator
    m = _SHARD_NUM_RE.search(filename)
    if m:
        return int(m.group(1))
    return fallback_idx


# ═════════════════════════════════════════════════════════════════════════════
# § 3  LATE IMPORTS (project-internal; after patches)
# ═════════════════════════════════════════════════════════════════════════════

from ..config.manager import GenerationConfig            # noqa: E402
from ..adapters.registry import AdapterRegistry          # noqa: E402
from ..strategies import get_strategy                    # noqa: E402
from ..utils.exceptions import GenerationError, ModelNotSupportedError  # noqa: E402
from .incremental import IncrementalGenerator            # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# § 4  MinimalWeightGenerator
# ═════════════════════════════════════════════════════════════════════════════

class MinimalWeightGenerator:
    """
    Generates minimal (dummy) weight shards for any HuggingFace LLM/VLM.
    Preserves the original sharding structure for architectural inspection.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str,
        config: Optional["GenerationConfig"] = None,
        save_dummy_config: bool = False,
        shrink_config: Optional[bool] = None,
        **kwargs,
    ):
        self.model_id    = model_id
        self.output_dir  = output_dir
        self.config      = config if config else GenerationConfig(**kwargs)

        if shrink_config is None:
            self.shrink_config = self.config.strategy in ("ultra", "hybrid_ultra")
            if self.shrink_config:
                logger.info("Auto-enabling shrink_config for '%s' strategy.", self.config.strategy)
        else:
            self.shrink_config = shrink_config

        self.strategy = get_strategy(
            self.config.strategy,
            n_bits=self.config.n_bits,
            rank=self.config.rank,
            sparsity=self.config.sparsity,
            save_dummy_config=save_dummy_config,
        )
        self.max_shard_size: int = self._parse_size(self.config.max_shard_size)
        self.shard_map:  Dict[str, str] = {}
        self.total_size: int = 0
        self.incremental = IncrementalGenerator(output_dir)
        # _shard_counter kept for compatibility; real assignment uses md5 hash
        self._shard_counter = 0

    # ──────────────────────────────────────────────────────────────────────
    # Static helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_size(size_str: str) -> int:
        """Parse human-readable size string to bytes.

        Supports: GB, MB, KB suffixes. Plain numbers are treated as bytes.

        Args:
            size_str: Size string like "5GB", "512MB", "1024KB", or "1073741824"

        Returns:
            Size in bytes as integer

        Raises:
            ValueError: If size_str cannot be parsed
        """
        if not size_str or not isinstance(size_str, str):
            raise ValueError(f"Invalid size string: {size_str!r}")
        upper = size_str.upper().strip()
        for unit, mult in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
            if upper.endswith(unit):
                try:
                    val = float(size_str[:-2])
                except ValueError:
                    raise ValueError(
                        f"Cannot parse numeric value from size string: {size_str!r}"
                    )
                if val < 0:
                    raise ValueError(f"Negative size not allowed: {size_str!r}")
                return int(val * mult)
        try:
            result = int(size_str)
        except ValueError:
            raise ValueError(f"Cannot parse size string: {size_str!r}")
        if result < 0:
            raise ValueError(f"Negative size not allowed: {size_str!r}")
        return result

    @staticmethod
    def _get_dtype_size(dtype: torch.dtype) -> int:
        return {
            torch.float64: 8,
            torch.float32: 4,   # [B1 fix] was accidentally hard-coded as 2 in generate()
            torch.bfloat16: 2,
            torch.float16: 2,
            torch.int64: 8,
            torch.int32: 4,
            torch.int16: 2,
            torch.int8: 1,
            torch.uint8: 1,
            torch.bool: 1,
        }.get(dtype, 4)

    @staticmethod
    def _estimate_tensor_nbytes(
        tensor: torch.Tensor,
        *,
        fallback_numel: int,
        fallback_dtype: torch.dtype,
    ) -> int:
        """
        Estimate how many bytes a tensor occupies (used for total_size and streaming flush).

        Design motivation:
        - The Ultra strategy can create stride=0 as_strided tensors: the logical size
          (numel * element_size) is huge, but the underlying storage contains only a few
          elements. In this case we must prefer the real storage bytes; otherwise total_size
          will be severely overestimated.
        """
        try:
            return int(tensor.untyped_storage().nbytes())
        except Exception:
            pass
        try:
            return int(tensor.element_size() * tensor.nelement())
        except Exception:
            return int(fallback_numel) * int(MinimalWeightGenerator._get_dtype_size(fallback_dtype))

    # ──────────────────────────────────────────────────────────────────────
    # Config shrinking
    # ──────────────────────────────────────────────────────────────────────

    # VLM vision tower uses larger min dims to avoid dimension mismatch
    # when the model class expects vision weights with specific shapes.
    _SHRINK_VISION_HIDDEN_SIZE: int = 64
    _SHRINK_VISION_NUM_ATTENTION_HEADS: int = 4
    _SHRINK_VISION_INTERMEDIATE_SIZE: int = 128
    _SHRINK_VISION_NUM_HIDDEN_LAYERS: int = 2
    _SHRINK_VISION_IMAGE_SIZE: int = 14
    _SHRINK_VISION_PATCH_SIZE: int = 2

    def _shrink_config(self, config, _is_vision_sub: bool = False) -> None:
        """Shrink config to minimal dimensions for ultra-compact dummy models.

        This aggressively reduces all size-related attributes to tiny values.
        Uses _set which only modifies attributes that already exist on the
        config object, so model-specific attributes that aren't present are
        safely skipped.

        When ``_is_vision_sub`` is True, we apply VLM-compatible minimum
        dimensions for the vision tower to prevent dimension-mismatch errors
        when ``from_pretrained()`` loads the model (Gemma-4, Qwen2-VL, etc.).
        """
        def _set(obj, attr, val):
            if hasattr(obj, attr):
                setattr(obj, attr, val)

        def _set_force(obj, attr, val):
            """Set even if attribute doesn't exist — for important overrides."""
            try:
                setattr(obj, attr, val)
            except Exception as e:
                logger.debug("Could not _set %s on %s: %s", attr, type(obj).__name__, e)

        # Choose dims based on whether this is a vision sub-config
        if _is_vision_sub:
            _hidden = self._SHRINK_VISION_HIDDEN_SIZE
            _n_heads = self._SHRINK_VISION_NUM_ATTENTION_HEADS
            _n_kv_heads = self._SHRINK_VISION_NUM_ATTENTION_HEADS
            _inter = self._SHRINK_VISION_INTERMEDIATE_SIZE
            _n_layers = self._SHRINK_VISION_NUM_HIDDEN_LAYERS
        else:
            _hidden = _SHRINK_HIDDEN_SIZE
            _n_heads = _SHRINK_NUM_ATTENTION_HEADS
            _n_kv_heads = _SHRINK_NUM_KEY_VALUE_HEADS
            _inter = _SHRINK_INTERMEDIATE_SIZE
            _n_layers = _SHRINK_HIDDEN_LAYERS

        _set(config, "num_hidden_layers", _n_layers)
        _set(config, "num_layers",        _n_layers)
        _set(config, "hidden_size",       _hidden)
        _set(config, "num_attention_heads", _n_heads)
        _set(config, "num_key_value_heads", _n_kv_heads)
        _set(config, "intermediate_size",  _inter)
        # Force-set intermediate_size even if None/missing — many models
        # compute it dynamically but LlamaForCausalLM fallback needs it
        _set_force(config, "intermediate_size", _inter)

        if not _is_vision_sub:
            _set(config, "num_experts",        _SHRINK_NUM_EXPERTS)
            _set(config, "num_experts_per_tok", _SHRINK_NUM_EXPERTS_PER_TOK)
            _set(config, "n_routed_experts",   _SHRINK_NUM_EXPERTS)
            _set(config, "moe_intermediate_size", _SHRINK_MOE_INTERMEDIATE_SIZE)
            _set(config, "shared_expert_intermediate_size", _SHRINK_SHARED_EXPERT_INTERMEDIATE_SIZE)
            _set(config, "expert_intermediate_size", _SHRINK_MOE_INTERMEDIATE_SIZE)
            _set(config, "d_state", _SHRINK_D_STATE)   # Mamba
            _set(config, "d_conv",  _SHRINK_D_CONV)

        # Derive head dims from hidden_size / num_attention_heads so that
        # model construction does not fail on constraint checks.
        n_heads = getattr(config, "num_attention_heads", 0) or 0
        h_size = getattr(config, "hidden_size", 0) or 0
        if n_heads > 0 and h_size > 0:
            derived_head_dim = h_size // n_heads
            _set(config, "head_dim", derived_head_dim)
            _set(config, "global_head_dim", derived_head_dim)
            # GQA / MLA specific head dims
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
                # GLM constraint: qk_head_dim == qk_nope_head_dim + qk_rope_head_dim
                # Some configs omit qk_head_dim initially, so we force-populate it.
                try:
                    qk_total = int(getattr(config, "qk_nope_head_dim", 0) or 0) + int(getattr(config, "qk_rope_head_dim", 0) or 0)
                except Exception:
                    qk_total = int(derived_head_dim)
                _set_force(config, "qk_head_dim", qk_total)
                _set(config, "v_head_dim", derived_head_dim)
            else:
                _set(config, "qk_nope_head_dim", derived_head_dim)
                _set(config, "qk_rope_head_dim", max(derived_head_dim // 4, 4))
                _set(config, "qk_head_dim", derived_head_dim)
                _set(config, "v_head_dim", derived_head_dim)
            # Lora ranks — keep ≤ hidden_size
            for lr_attr in ("kv_lora_rank", "q_lora_rank"):
                if hasattr(config, lr_attr):
                    old = getattr(config, lr_attr, h_size)
                    _set(config, lr_attr, min(old or h_size, h_size))
            # Linear attention dims
            _set(config, "linear_key_head_dim", derived_head_dim)
            _set(config, "linear_value_head_dim", derived_head_dim)
            _set(config, "linear_num_key_heads", _n_heads)
            _set(config, "linear_num_value_heads", _n_heads)

        # GQA heads
        _set(config, "num_global_key_value_heads", _n_kv_heads)
        _set(config, "num_kv_shared_layers", 0)

        # VLM / multimodal dims
        _set(config, "vision_soft_tokens_per_image", 1)
        _set(config, "depth", _n_layers)

        # Vision-specific fields (only relevant for vision sub-configs)
        if _is_vision_sub:
            _set(config, "image_size", self._SHRINK_VISION_IMAGE_SIZE)
            _set(config, "patch_size", self._SHRINK_VISION_PATCH_SIZE)
            _set(config, "num_channels", 3)
            _set(config, "num_positions", (self._SHRINK_VISION_IMAGE_SIZE // self._SHRINK_VISION_PATCH_SIZE) ** 2 + 1)

        # Sliding window / positional
        _set(config, "max_position_embeddings", 4096)
        # sliding_window: Some configs (Gemma-4, etc.) validate this as int and
        # reject None.  Only set if the attribute already exists and is an int.
        sw = getattr(config, "sliding_window", None)
        if sw is not None and isinstance(sw, int):
            _set(config, "sliding_window", 4096)
        elif hasattr(config, "sliding_window"):
            # Attribute exists but is None or non-int — leave it alone
            pass

        # Recurse into sub-configs — use _is_vision_sub=True for vision tower
        for sub in ("text_config", "encoder", "decoder",
                     "audio_config", "ngram_config", "decoder_config"):
            sub_cfg = getattr(config, sub, None)
            if sub_cfg is not None:
                self._shrink_config(sub_cfg, _is_vision_sub=False)

        # Vision sub-config gets special treatment
        for sub in ("vision_config",):
            sub_cfg = getattr(config, sub, None)
            if sub_cfg is not None:
                self._shrink_config(sub_cfg, _is_vision_sub=True)

        # Truncate layer_types list to match num_hidden_layers
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

    # ──────────────────────────────────────────────────────────────────────
    # Shard map utilities
    # ──────────────────────────────────────────────────────────────────────

    def _get_original_shard_map(self) -> Dict[str, str]:
        """
        Fetch weight_map from HF Hub index, normalise to standard -NNNNN-of-MMMMM format.
        Falls back to listing repo files if index is unavailable.
        [N1][N2][N3][N4] all fixed here.
        """

        local_files_only = bool(
            getattr(self.config.security, "local_files_only", False)
            or not getattr(self.config.security, "allow_network", True)
        )

        def _fetch_index() -> Dict[str, str]:
            if os.path.isdir(self.model_id):
                for fname in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
                    p = os.path.join(self.model_id, fname)
                    if os.path.exists(p):
                        try:
                            with open(p) as f:
                                return json.load(f).get("weight_map", {})
                        except Exception as e:
                            logger.debug("Skip weight_map file %s: %s", p, e)
                            continue
            for fname in ("model.safetensors.index.json",
                          "pytorch_model.bin.index.json"):
                try:
                    resolved = cached_file(
                        self.model_id, fname,
                        _raise_exceptions_for_missing_entries=False,
                        local_files_only=local_files_only,
                    )
                    if resolved and os.path.exists(resolved):
                        logger.info("Found shard index: %s", fname)
                        with open(resolved) as f:
                            return json.load(f).get("weight_map", {})
                except Exception as e:
                    logger.debug("Skip %s: %s", fname, e)
            return {}

        original_map = _fetch_index()

        # ── Fallback: list repo files ────────────────────────────────────
        if not original_map:
            try:
                if os.path.isdir(self.model_id) or local_files_only:
                    return {}
                from huggingface_hub import list_repo_files
                logger.info("Index unavailable — listing repo files…")
                all_files = list(list_repo_files(self.model_id))

                # [N3 fix] Separate sharded from single-file models
                sharded = [
                    f for f in all_files
                    if (f.endswith(".safetensors") or f.endswith(".bin"))
                    and "-of-" in f
                    and "model" in f
                ]
                single = [
                    f for f in all_files
                    if (f.endswith(".safetensors") or f.endswith(".bin"))
                    and "-of-" not in f
                    and f in ("model.safetensors", "pytorch_model.bin",
                              "model.bin", "pytorch_model.safetensors")
                ]

                if sharded:
                    logger.info("Recovered %d shards from repo listing.", len(sharded))
                    original_map = {f"_dummy_{i}": f for i, f in enumerate(sorted(sharded))}
                elif single:
                    logger.info("Single-file model detected: %s", single[0])
                    original_map = {"_dummy_0": single[0]}
                else:
                    logger.warning("No weight files found in repo listing.")
            except Exception as e:
                logger.warning("Failed to list repo files: %s", e)

        if not original_map:
            return {}

        # ── Normalise filenames to -NNNNN-of-MMMMM format ────────────────
        unique_shards  = sorted(set(original_map.values()))
        total_shards   = len(unique_shards)

        # [N4 fix] Use sequential enumerate index, not the parsed digit,
        # so the normalised index is always in [1..total_shards]
        norm_table: Dict[str, str] = {}
        for seq_idx, filename in enumerate(unique_shards, start=1):
            ext    = "bin" if filename.endswith(".bin") else "safetensors"
            prefix = "pytorch_model" if "pytorch_model" in filename else "model"
            norm_table[filename] = (
                f"{prefix}-{seq_idx:05d}-of-{total_shards:05d}.{ext}"
            )

        normalised = {p: norm_table.get(f, f) for p, f in original_map.items()}
        logger.info("Normalised %d unique shard names.", total_shards)
        return normalised

    def _resolve_target_shard(
        self,
        name: str,
        original_shard_map: Dict[str, str],
        current_target: Optional[str],
        param_seq_idx: int = 0,          # [F-3] sequential index for even distribution
    ) -> Optional[str]:
        if not original_shard_map:
            return None

        available = sorted(set(original_shard_map.values()))
        n_shards  = len(available)

        # 1. Exact match
        if name in original_shard_map:
            return original_shard_map[name]

        # 2. With/without "model." prefix
        for candidate in (f"model.{name}",
                          name[6:] if name.startswith("model.") else None):
            if candidate and candidate in original_shard_map:
                return original_shard_map[candidate]

        # 3. [F-3] Name mismatch fallback: distribute evenly across ALL shards
        #    using param_seq_idx % n_shards — guarantees every shard gets params
        #    when num_params >= n_shards.  For shrink mode (few params, many shards)
        #    placeholder filling (Step 5b) will cover remaining empty shards.
        return available[param_seq_idx % n_shards]

    # ──────────────────────────────────────────────────────────────────────
    # Config patching
    # ──────────────────────────────────────────────────────────────────────

    def _patch_model_config(self, hf_config) -> None:
        """Apply compatibility and model-family patches via the shared patches package."""

        # 1. Merge sub-config scalar attrs up to top-level — ONLY if the top-level
        #    value is None/missing.  This prevents overwriting valid top-level values
        #    (e.g. after adapter.patch_config() has already promoted them) and avoids
        #    leaking sub-config-specific fields like "model_type" into the top-level.
        #    We also explicitly skip "model_type" to never let a sub-config's type
        #    overwrite the top-level type.
        _PROMOTE_SKIP_KEYS = {"model_type", "architectures", "torch_dtype"}
        for sub_attr in ("text_config", "vision_config",
                         "encoder_config", "decoder_config"):
            sub_cfg = getattr(hf_config, sub_attr, None)
            if sub_cfg is None:
                continue
            src = (sub_cfg.to_dict() if hasattr(sub_cfg, "to_dict")
                   else (sub_cfg if isinstance(sub_cfg, dict) else {}))
            for k, v in src.items():
                if k in _PROMOTE_SKIP_KEYS:
                    continue
                if not isinstance(v, (int, float, str, bool)):
                    continue
                # Only promote when top-level value is missing or None
                current = getattr(hf_config, k, None)
                if current is None:
                    try:
                        setattr(hf_config, k, v)
                    except Exception as e:
                        logger.debug("Could not merge sub-config attr %s: %s", k, e)

        # 2. Disable Flash Attention if package absent
        if importlib.util.find_spec("flash_attn") is None:
            logger.info("flash_attn absent → forcing eager attention.")
            for attr in ("_attn_implementation", "attn_implementation"):
                try:
                    setattr(hf_config, attr, "eager")
                except Exception as e:
                    logger.debug("Could not set %s='eager': %s", attr, e)
            _inject_recursive(hf_config, "use_flash_attention_2", False)
            _inject_recursive(hf_config, "use_flash_attn",        False)
            _inject_recursive(hf_config, "use_deterministic_attn", False)

        # 3. Family-specific patches
        PatchRegistry.apply(hf_config, self.model_id)

        logger.debug(
            "Config patched: _attn_impl=%s  fa2=%s",
            getattr(hf_config, "_attn_implementation", "N/A"),
            getattr(hf_config, "use_flash_attention_2", "N/A"),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Remote-class patching (Kimi, DeepSeek, MoonViT …)
    # ──────────────────────────────────────────────────────────────────────

    def _patch_remote_classes(self, hf_config) -> None:
        if not hasattr(hf_config, "auto_map"):
            return
        for key in ("AutoModelForCausalLM", "AutoModel"):
            if key not in hf_config.auto_map:
                continue
            try:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                cls = get_class_from_dynamic_module(hf_config.auto_map[key], self.model_id)
                mod = sys.modules[cls.__module__]
            except Exception as e:
                logger.warning("Cannot load dynamic module: %s", e)
                continue
            patch_remote_module(mod)
            break

    # ──────────────────────────────────────────────────────────────────────
    # Model initialisation  [A4: split into sub-methods]
    # ──────────────────────────────────────────────────────────────────────

    def _initialize_model(self, hf_config, adapter=None):
        logger.info("Initialising empty model structure…")
        self._patch_model_config(hf_config)
        self._patch_remote_classes(hf_config)

        with init_empty_weights():
            return (
                self._try_load_adapter(hf_config, adapter=adapter)
                or self._try_load_auto(hf_config)
                or self._try_load_fallback_chain(hf_config)
            )

    def _try_load_adapter(self, hf_config, adapter=None):
        try:
            adapter = adapter or AdapterRegistry.get_adapter(self.model_id, hf_config)
            model_cls = adapter.get_model_class(hf_config)
            if model_cls:
                logger.info("Adapter class: %s", model_cls.__name__)
                return model_cls(hf_config)
        except Exception as e:
            logger.debug("Adapter load failed: %s", e)
        return None

    def _try_load_auto(self, hf_config):
        _trc = self.config.security.trust_remote_code
        for loader, label in (
            (
                lambda: __import__("vitriol.utils.hf_loading", fromlist=["load_causallm_from_config"])
                .load_causallm_from_config(
                    hf_config,
                    security={"trust_remote_code": _trc, "allow_network": True, "local_files_only": False},
                ),
                "AutoModelForCausalLM",
            ),
            (
                lambda: __import__("vitriol.utils.hf_loading", fromlist=["load_model_from_config"])
                .load_model_from_config(
                    hf_config,
                    security={"trust_remote_code": _trc, "allow_network": True, "local_files_only": False},
                ),
                "AutoModel",
            ),
        ):
            try:
                return loader()
            except Exception as e:
                logger.warning("%s failed: %s", label, e)
        return None

    def _try_load_fallback_chain(self, hf_config):
        """[A3] Walk Llama→Qwen2→Mistral→Phi→Gemma until one succeeds."""
        model_type = str(getattr(hf_config, "model_type", "") or "").lower()
        if "glm" in model_type:
            # GLM configs often expose MoE-like metadata but are not wire-compatible
            # with DeepSeek's MLA implementation. Prefer a plain causal fallback so
            # generated minimal weights stay reloadable and runnable offline.
            chain = _FALLBACK_CHAIN
        else:
            num_experts = getattr(hf_config, "num_experts", getattr(hf_config, "n_routed_experts", 0)) or 0
            chain = _MOE_FALLBACK_CHAIN + _FALLBACK_CHAIN if num_experts > 0 else _FALLBACK_CHAIN
        for cfg_name, cls_name in chain:
            try:
                return _build_fallback_config(cfg_name, cls_name, hf_config)
            except Exception as e:
                logger.debug("%s fallback: %s", cls_name, e)

        raise ModelNotSupportedError(
            self.model_id,
            "All loaders failed (adapter / AutoCausalLM / AutoModel / "
            "MoE / Llama / Qwen2 / Mistral / Phi / Gemma).\n" + traceback.format_exc(),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Config loading with type-alias fallback
    # ──────────────────────────────────────────────────────────────────────

    # Known type-alias map for model types that are NOT yet in
    # transformers CONFIG_MAPPING but whose config is compatible
    # with an existing registered type.
    _TYPE_ALIAS_MAP: Dict[str, str] = {
        "gemma4": "gemma3",
        "gemma4_text": "gemma3_text",
        "deepseek_v4": "deepseek_v3",
        "deepseek_v3": "deepseek_v2",
        "hy3": "llama",
        "hy_v3": "llama",
        "hunyuan": "llama",
    }

    @classmethod
    def _find_best_alias(cls, original_type: str) -> List[str]:
        """Find the best alias(es) for an unknown model_type.

        Strategy:
        1. Check explicit _TYPE_ALIAS_MAP first.
        2. Strip trailing version digits and try prefix match in CONFIG_MAPPING
           (e.g. "llama4" → "llama3" → "llama").
        3. Fall back to generic base types.
        """
        aliases: List[str] = []

        # 1. Explicit map
        if original_type in cls._TYPE_ALIAS_MAP:
            aliases.append(cls._TYPE_ALIAS_MAP[original_type])

        # 2. Auto-detect by stripping version suffixes
        #    e.g. "gemma4" → try "gemma3", "gemma2", "gemma"
        #         "qwen3_5_moe" → try "qwen3_moe", "qwen2_moe"
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING
            import re
            # Try progressively shorter version suffixes
            base = re.sub(r'[\d_.]+$', '', original_type)  # "gemma4" → "gemma"
            if base and base != original_type:
                # Try numbered variants descending
                nums = re.findall(r'\d+', original_type)
                if nums:
                    max_ver = int(nums[-1])
                    for v in range(max_ver - 1, 0, -1):
                        candidate = re.sub(r'\d+(?=[^0-9]*$)', str(v), original_type, count=1)
                        if candidate in CONFIG_MAPPING and candidate not in aliases:
                            aliases.append(candidate)
                # Try base without any numbers
                if base in CONFIG_MAPPING and base not in aliases:
                    aliases.append(base)
        except Exception as e:
            logger.debug("Alias discovery failed: %s", e)

        return aliases

    def _load_hf_config(self):
        local_files_only = bool(
            getattr(self.config.security, "local_files_only", False)
            or not getattr(self.config.security, "allow_network", True)
        )
        # Pre-register all adapters so custom model types are available
        try:
            AdapterRegistry._load_builtin_adapters()
            for adapter_cls in AdapterRegistry._adapters:
                try:
                    adapter_cls().register_classes()
                except Exception as e:
                    logger.debug("Adapter %s register_classes failed: %s", adapter_cls, e)
        except Exception as e:
            logger.debug("Adapter pre-registration failed: %s", e)

        # Strategy 1: Standard AutoConfig (handles trust_remote_code models)
        try:
            from ..utils.hf_loading import load_config as hf_load_config

            return hf_load_config(
                self.model_id,
                security={
                    "trust_remote_code": self.config.security.trust_remote_code,
                    "allow_network": not local_files_only,
                    "local_files_only": local_files_only,
                },
            )
        except Exception as e:
            err_str = str(e).lower()
            is_unknown = "model type" in err_str or "not recognize" in err_str
            if not is_unknown:
                raise GenerationError(f"Config load error: {e}") from e

        # Strategy 2: Load raw config.json and try adapter-registered types + aliases
        logger.warning("Unknown model_type or config load error. Trying adapter/alias fallback…")
        try:
            config_path = cached_file(
                self.model_id,
                "config.json",
                _raise_exceptions_for_missing_entries=False,
                local_files_only=local_files_only,
            )
            if not config_path or not os.path.exists(config_path):
                raise FileNotFoundError("config.json not found")

            with open(config_path) as f:
                config_dict = json.load(f)
            original_type = config_dict.get("model_type", "unknown")
            logger.info("Original model_type: %s", original_type)

            # Remove model_type to avoid conflicts with for_model()
            config_dict_no_type = {k: v for k, v in config_dict.items() if k != "model_type"}

            # 2a. Try the original model_type (may work if adapter registered it)
            try:
                cfg = AutoConfig.for_model(original_type, **config_dict_no_type)
                logger.info("Loaded config as original type '%s'", original_type)
                return cfg
            except Exception as e:
                logger.debug("for_model construction failed for '%s': %s", original_type, e)

            # 2b. Try CONFIG_MAPPING direct construction
            try:
                from transformers.models.auto.configuration_auto import CONFIG_MAPPING
                if original_type in CONFIG_MAPPING:
                    cfg_cls = CONFIG_MAPPING[original_type]
                    logger.info("Direct CONFIG_MAPPING construction: %s", cfg_cls.__name__)
                    return cfg_cls(**config_dict_no_type)
            except Exception as e:
                logger.debug("CONFIG_MAPPING construction failed for '%s': %s", original_type, e)

            # 2c. Auto-discover best alias
            aliases = self._find_best_alias(original_type)
            for alias in aliases:
                try:
                    logger.info("Retrying as alias '%s'…", alias)
                    cfg = AutoConfig.for_model(alias, **config_dict_no_type)
                    logger.info("Loaded config via alias '%s' → %s", alias, type(cfg).__name__)
                    return cfg
                except Exception as fe:
                    logger.debug("Alias '%s' failed: %s", alias, fe)

            # 2d. Last resort: raw PretrainedConfig (preserves all fields)
            logger.warning("All typed loaders failed — falling back to raw PretrainedConfig.")
            from transformers import PretrainedConfig
            return PretrainedConfig.from_dict(config_dict)

        except GenerationError:
            raise
        except Exception as e:
            raise GenerationError(f"Failed to load raw config.json: {e}") from e

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def _build_generation_result(self) -> GenerationResult:
        prefix = self.strategy.get_shard_prefix()
        index_name = (
            f"{prefix}.bin.index.json"
            if self.strategy.storage_format == "pytorch"
            else "model.safetensors.index.json"
        )
        index_path = os.path.join(self.output_dir, index_name)
        manifest_path = os.path.join(self.output_dir, "vitriol-manifest.json")
        return GenerationResult(
            output_dir=self.output_dir,
            manifest_path=manifest_path if os.path.exists(manifest_path) else None,
            index_path=index_path if os.path.exists(index_path) else None,
            total_size=self.total_size,
            generated_at=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def generate(self) -> GenerationResult:
        """Generate minimal weights.

        This method is the public entry point. Internally it delegates to an
        experimental pipeline wrapper to improve testability and maintainability.
        In the initial stage, behavior is preserved by running the legacy
        implementation unchanged.
        """
        # Local imports to keep module import lightweight and avoid circular deps
        from .pipeline.context import GenerationContext
        from .pipeline.pipeline import GenerationPipeline
        from .pipeline.steps import BootstrapStep, LegacyGenerateStep

        ctx = GenerationContext(model_id=self.model_id, output_dir=self.output_dir, config=self.config, generator=self)
        GenerationPipeline([BootstrapStep(), LegacyGenerateStep()]).run(ctx)
        return self._build_generation_result()

    def _generate_legacy_impl(self) -> GenerationResult:
        os.makedirs(self.output_dir, exist_ok=True)
        # [Cleanup] Remove existing weight files to prevent confusion from previous runs
        for f in os.listdir(self.output_dir):
            if f.endswith(".bin") or f.endswith(".safetensors") or f.endswith(".index.json"):
                try:
                    os.remove(os.path.join(self.output_dir, f))
                except OSError as e:
                    logger.warning("Failed to clean up %s: %s", f, e)

        if os.listdir(self.output_dir):
            logger.warning("Output directory %s contains residual files.", self.output_dir)

        # ── 1. Config ───────────────────────────────────────────────────
        logger.info("Loading config for %s…", self.model_id)
        hf_config = self._load_hf_config()

        adapter = AdapterRegistry.get_adapter(self.model_id, hf_config)
        hf_config = adapter.patch_config(hf_config)

        if self.shrink_config:
            logger.info("Shrinking config for compact mode…")
            self._shrink_config(hf_config)

        # ── 2. Model ────────────────────────────────────────────────────
        model = self._initialize_model(hf_config, adapter=adapter)
        try:
            model.tie_weights()
        except Exception as e:
            logger.warning("tie_weights skipped (%s): %s", type(e).__name__, e)

        # ── 2b. Choose an "active" config to save ────────────────────────
        # Some model_ids / model_type values are not recognized by the local
        # Transformers version, so _load_hf_config() may fall back to a raw
        # PretrainedConfig. In that case, _initialize_model() may have selected
        # a fallback typed config/model (e.g. LlamaForCausalLM). We should save
        # the typed config so that AutoConfig/AutoModel can reload the generated
        # dummy weights offline, while still keeping the full original config in
        # meta-config.json (handled in _save_configs()).
        #
        # We also copy over extra scalar attributes (e.g. GLM qk_* dims) from
        # the original config so invariants/tests remain satisfied.
        active_config = getattr(model, "config", hf_config)
        if active_config is not hf_config:
            skip = {"model_type", "architectures", "torch_dtype"}
            for k, v in getattr(hf_config, "__dict__", {}).items():
                if k in skip:
                    continue
                if hasattr(active_config, k):
                    continue
                if isinstance(v, (int, float, str, bool, type(None))):
                    try:
                        setattr(active_config, k, v)
                    except Exception:
                        pass
            for k in ("qk_nope_head_dim", "qk_rope_head_dim", "qk_head_dim", "v_head_dim"):
                if hasattr(hf_config, k) and not hasattr(active_config, k):
                    try:
                        setattr(active_config, k, getattr(hf_config, k))
                    except Exception:
                        pass

        # ── 3. Prepare ──────────────────────────────────────────────────
        logger.info("Strategy: %s", self.config.strategy)
        original_shard_map = self._get_original_shard_map()

        checkpoint = self.incremental.load_checkpoint()
        if checkpoint:
            logger.info("Resuming from checkpoint…")
            # [B3 fix] checkpoint key renamed to generated_param_names
            generated_names: Set[str] = set(
                checkpoint.get("generated_param_names",
                               checkpoint.get("generated_param_ids", []))
            )
            shard_count     = checkpoint.get("shard_count", 0)
            self.total_size = checkpoint.get("total_size", 0)
            self.shard_map  = checkpoint.get("shard_map", {})
        else:
            generated_names = set()
            shard_count     = 0
            self.total_size = 0
            self.shard_map  = {}

        # [F-4] Determine canonical shard count ONCE — used everywhere below
        expected_shards: List[str] = (
            sorted(set(original_shard_map.values())) if original_shard_map else []
        )
        n_expected = len(expected_shards)   # e.g. 94 for a 94-shard model

        shard_buffers: Dict[str, Dict[str, Any]] = {f: {} for f in expected_shards}

        non_persistent_buffers: Set[str] = set()
        for module_prefix, module in model.named_modules():
            local_names: Set[str] = getattr(module, "_non_persistent_buffers_set", set())
            for local_name in local_names:
                qualified = f"{module_prefix}.{local_name}" if module_prefix else local_name
                non_persistent_buffers.add(qualified)

        def _iter_export_tensors():
            for item in model.named_parameters():
                yield item
            for name, buf in model.named_buffers():
                if name in non_persistent_buffers:
                    logger.debug("Skipping non-persistent buffer during export: %s", name)
                    continue
                yield name, buf

        # Snapshot names for progress bar (without materialising tensors)
        all_names = [n for n, _ in _iter_export_tensors()]
        total_params = len(all_names)

        current_target: Optional[str] = None
        if all_names and original_shard_map:
            current_target = self._resolve_target_shard(
                all_names[0], original_shard_map, None, param_seq_idx=0)

        pbar = tqdm(total=total_params, desc="Generating tensors")
        pbar.update(len(generated_names))

        try:
            # Per-shard byte tracker for streaming flush
            buf_bytes: Dict[str, int] = {}
            # [F-1] Track shards written by streaming flush (distinct from "still empty")
            flushed_shards: Set[str] = set()
            param_seq_idx = 0   # monotonically increments; used for even fallback distribution

            # ── 4. Generation loop ──────────────────────────────────────────
            for name, param in _iter_export_tensors():  # [A7] iterator, no full list
                if name in generated_names:  # [B3 fix] stable name key
                    param_seq_idx += 1
                    pbar.update(1)
                    continue

                # [F-3] Pass param_seq_idx for even fallback distribution
                target = self._resolve_target_shard(
                    name, original_shard_map, current_target,
                    param_seq_idx=param_seq_idx,
                )
                if original_shard_map and not target:
                    # Still no target? Use round-robin on expected_shards
                    target = expected_shards[param_seq_idx % n_expected] if n_expected else None
                if not target:
                    target = (
                        f"{self.strategy.get_shard_prefix()}-00001-of-XXXXX"
                        f".{self.strategy.file_extension}"
                    )

                try:
                    tensor = self.strategy.generate_tensor(param.shape, param.dtype, name)
                except Exception as e:
                    logger.error("Tensor generation failed for '%s': %s", name, e)
                    raise

                if target not in shard_buffers:
                    shard_buffers[target] = {}
                shard_buffers[target][name] = tensor

                # Compute byte size: use actual storage size for stride tricks,
                # fall back to logical size (numel × dtype_bytes) otherwise.
                nbytes = self._estimate_tensor_nbytes(
                    tensor,
                    fallback_numel=int(param.numel()),
                    fallback_dtype=param.dtype,
                )
                self.total_size += nbytes
                buf_bytes[target] = buf_bytes.get(target, 0) + nbytes

                # [A5] Streaming flush when buffer reaches max_shard_size
                if (buf_bytes.get(target, 0) >= self.max_shard_size
                        and "XXXXX" not in target):
                    self._save_shard(shard_buffers[target], target, shard_count)
                    shard_buffers[target] = {}
                    buf_bytes[target]     = 0
                    flushed_shards.add(target)   # [F-1] mark as written
                    shard_count += 1

                generated_names.add(name)
                param_seq_idx += 1
                pbar.update(1)
                current_target = target
        finally:
            pbar.close()

        # ── 5a. Fill shards that are still empty with a placeholder tensor ──
        # [F-2] Ensures every expected shard gets a file (e.g. 94 of 94).
        # Safetensors / bin formats require ≥ 1 tensor per file.
        # Placeholders use a reserved prefix "__vitriol_pad" so callers can
        # identify / filter them.  They are tiny (shape=(1,), float16 → 2 bytes)
        # and do NOT count toward total_size so the index metadata stays accurate.
        if expected_shards:
            for shard_name in expected_shards:
                if shard_name in flushed_shards:
                    continue   # already written
                if not shard_buffers.get(shard_name):
                    ph_key = f"__vitriol_pad__{shard_name}"
                    try:
                        shard_buffers[shard_name][ph_key] = (
                            self.strategy.generate_tensor((1,), torch.float16, ph_key)
                        )
                        logger.debug("Placeholder added to empty shard: %s", shard_name)
                    except Exception as e:
                        logger.warning("Could not add placeholder to %s: %s", shard_name, e)

        # ── 5b. Final flush ─────────────────────────────────────────────
        # Use expected_shards order so filenames stay deterministic.
        # If no expected_shards (no original index), fall back to shard_buffers keys.
        ordered = expected_shards if expected_shards else sorted(shard_buffers.keys())
        n_ordered = len(ordered)
        logger.info("Flushing %d shard(s) (%d expected)…", n_ordered, n_expected)

        for i, filename in enumerate(ordered):
            if filename in flushed_shards:
                continue   # [F-1] already written via streaming flush
            buf = shard_buffers.get(filename, {})
            real = (
                f"{self.strategy.get_shard_prefix()}"
                f"-{i + 1:05d}-of-{n_ordered:05d}"
                f".{self.strategy.file_extension}"
                if "XXXXX" in filename else filename
            )
            self._save_shard(buf, real, i)

        # ── 6. Metadata ─────────────────────────────────────────────────
        # [F-4] Always use the canonical expected shard count, never len(shard_buffers)
        final_shard_count = n_expected if n_expected else n_ordered
        self._save_index(final_shard_count, original_shard_map)
        self.incremental.clear_checkpoint()
        self._save_configs(active_config)
        self._copy_custom_code_files() # Ensure custom Python code is copied
        self._save_tokenizer()
        self._add_readme_metadata()
        self._write_manifest()
        logger.info("✓ Done — weights saved to %s", self.output_dir)
        return self._build_generation_result()

    # ──────────────────────────────────────────────────────────────────────
    # Custom Code Sync
    # ──────────────────────────────────────────────────────────────────────

    def _copy_custom_code_files(self) -> None:
        """Downloads/Copies any custom python files (e.g. modeling_*.py) required when trust_remote_code is enabled."""
        try:
            if not getattr(self.config.security, "trust_remote_code", False):
                return
            if os.path.isdir(self.model_id):
                return
            if bool(
                getattr(self.config.security, "local_files_only", False)
                or not getattr(self.config.security, "allow_network", True)
            ):
                return
            from huggingface_hub import list_repo_files, hf_hub_download
            repo_id = self.model_id
            # List all files in the repo
            files = list_repo_files(repo_id)
            # Filter for python files (usually modeling, configuration, tokenization)
            # Also include any files in custom subdirectories (like encoding/, inference/) except heavy weights
            target_files = []
            for f in files:
                if f.endswith(".py"):
                    target_files.append(f)
                elif "/" in f and not any(f.endswith(ext) for ext in (".bin", ".safetensors", ".pt", ".pth", ".msgpack", ".h5")):
                    target_files.append(f)
            
            if not target_files:
                return
                
            logger.info("Downloading %d custom code/asset files for trust_remote_code...", len(target_files))
            for file_name in target_files:
                try:
                    file_path = hf_hub_download(repo_id=repo_id, filename=file_name)
                    dest_path = os.path.join(self.output_dir, file_name)
                    dest_dir = os.path.dirname(dest_path)
                    if dest_dir:
                        os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                    logger.debug("Copied custom code file: %s", file_name)
                except Exception as e:
                    logger.warning("Failed to copy %s: %s", file_name, e)
        except Exception as e:
            logger.warning("Could not sync custom code files: %s", e)

    # ──────────────────────────────────────────────────────────────────────
    # Shard I/O
    # ──────────────────────────────────────────────────────────────────────

    def _save_shard(
        self,
        shard_data: Dict[str, Any],
        filename: str,
        shard_id: int = 0,
    ) -> None:
        prefix = self.strategy.get_shard_prefix()
        ext    = self.strategy.file_extension

        if not filename:
            filename = f"{prefix}-{shard_id + 1:05d}.{ext}"
        elif filename.endswith(".safetensors") and ext == "bin":
            filename = filename.replace(".safetensors", ".bin")
            filename = filename.replace("model-", "pytorch_model-")
        elif filename.endswith(".bin") and ext == "safetensors":
            filename = filename.replace(".bin", ".safetensors")
            filename = filename.replace("pytorch_model-", "model-")

        path = os.path.join(self.output_dir, filename)
        logger.info("Saving shard → %s  (%d tensors)", filename, len(shard_data))
        self.strategy.save_shard(shard_data, path)
        for key in shard_data:
            self.shard_map[key] = filename

    # ──────────────────────────────────────────────────────────────────────
    # Index save
    # ──────────────────────────────────────────────────────────────────────

    def _save_index(
        self,
        total_shards: int,
        original_shard_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        [F-4] Build weight_map index.
        `total_shards` must equal len(set(original_shard_map.values()));
        it is passed from generate() which knows this value authoritatively.
        Placeholder entries (__vitriol_pad__*) are excluded from the map
        so they don't pollute the visible weight list.
        """
        ext    = self.strategy.file_extension
        prefix = self.strategy.get_shard_prefix()
        final:  Dict[str, str] = {}

        for key, filename in self.shard_map.items():
            # Skip internal padding tensors
            if key.startswith("__vitriol_pad__"):
                continue

            if "-of-" in filename:
                final[key] = filename
            else:
                try:
                    shard_id  = int(filename.rsplit(".", 1)[0].split("-")[-1])
                    new_name  = f"{prefix}-{shard_id:05d}-of-{total_shards:05d}.{ext}"
                    final[key] = new_name
                    old = os.path.join(self.output_dir, filename)
                    new = os.path.join(self.output_dir, new_name)
                    if os.path.exists(old) and not os.path.exists(new):
                        os.rename(old, new)
                except (ValueError, IndexError):
                    final[key] = filename

        # Validate: warn if actual file count != declared total_shards
        written_files = set(
            f for f in os.listdir(self.output_dir)
            if f.endswith(f".{ext}") and "-of-" in f
        )
        if len(written_files) != total_shards:
            logger.warning(
                "Shard count mismatch: declared=%d, files on disk=%d",
                total_shards, len(written_files),
            )
        else:
            logger.info("Shard count verified: %d/%d ✓", len(written_files), total_shards)

        idx_name = (
            f"{prefix}.bin.index.json"
            if self.strategy.storage_format == "pytorch"
            else "model.safetensors.index.json"
        )
        with open(os.path.join(self.output_dir, idx_name), "w") as f:
            json.dump({"metadata": {"total_size": self.total_size}, "weight_map": final},
                      f, indent=2)
        logger.info("Saved index → %s  (%d weight entries, %d shards)",
                    idx_name, len(final), total_shards)

    def _write_manifest(self) -> None:
        try:
            import platform
            import vitriol
            import transformers
            from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

            def _hash_file(p: str) -> Optional[str]:
                if not p or not os.path.exists(p):
                    return None
                h = sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                return h.hexdigest()

            config_path = os.path.join(self.output_dir, "config.json")
            meta_path = os.path.join(self.output_dir, "meta-config.json")
            reconcile_path = os.path.join(self.output_dir, "vitriol-reconcile.json")

            prefix = self.strategy.get_shard_prefix()
            idx_name = (
                f"{prefix}.bin.index.json"
                if self.strategy.storage_format == "pytorch"
                else "model.safetensors.index.json"
            )
            idx_path = os.path.join(self.output_dir, idx_name)

            idx_data = None
            if os.path.exists(idx_path):
                try:
                    with open(idx_path) as f:
                        idx_data = json.load(f)
                except Exception as e:
                    logger.debug("Failed to read index file %s: %s", idx_path, e)
                    idx_data = None

            weight_map = (idx_data or {}).get("weight_map", {})
            unique_shards = sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []

            active_dims = None
            try:
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        active_cfg = json.load(f)
                    active_dims = {
                        k: active_cfg.get(k)
                        for k in (
                            "vocab_size",
                            "hidden_size",
                            "intermediate_size",
                            "num_hidden_layers",
                            "num_attention_heads",
                            "num_key_value_heads",
                            "head_dim",
                            "num_experts",
                            "num_experts_per_tok",
                        )
                    }
            except Exception as e:
                logger.debug("Active dims extraction failed: %s", e)
                active_dims = None

            reconcile_info = None
            try:
                if os.path.exists(reconcile_path):
                    with open(reconcile_path) as f:
                        r = json.load(f)
                    if isinstance(r, dict):
                        _diff_raw = r.get("diff")
                        diff_obj = _diff_raw if isinstance(_diff_raw, dict) else {}
                        reconcile_info = {
                            "patched": bool(r.get("patched")),
                            "diff_keys": sorted(diff_obj.keys()),
                            "diff_count": len(diff_obj),
                        }
            except Exception as e:
                logger.debug("Reconcile info extraction failed: %s", e)
                reconcile_info = None

            loadability: Dict[str, Any] = {
                "checked": False,
                "ok": None,
                "loader": None,
                "error": None,
            }
            try:
                loadability["checked"] = True
                trust_rc = bool(getattr(self.config.security, "trust_remote_code", True))
                local_only = True
                last_err = None
                # Common kwargs for manifest loadability check
                _load_kwargs: Dict[str, Any] = dict(
                    local_files_only=local_only,
                    trust_remote_code=trust_rc,
                    low_cpu_mem_usage=True,
                    torch_dtype="auto",
                    device_map="cpu",
                )
                for loader, name in (
                    (AutoModelForCausalLM, "AutoModelForCausalLM"),
                    (AutoModelForSeq2SeqLM, "AutoModelForSeq2SeqLM"),
                    (AutoModel, "AutoModel"),
                ):
                    try:
                        loader.from_pretrained(self.output_dir, **_load_kwargs)
                        loadability["ok"] = True
                        loadability["loader"] = name
                        break
                    except Exception as e:
                        last_err = e
                        err_str = str(e).lower()
                        # Retry with ignore_mismatched_sizes for size mismatches
                        if "size mismatch" in err_str or "mismatch" in err_str:
                            try:
                                loader.from_pretrained(
                                    self.output_dir,
                                    **_load_kwargs,
                                    ignore_mismatched_sizes=True,
                                )
                                loadability["ok"] = True
                                loadability["loader"] = f"{name} (ignore_mismatched)"
                                break
                            except Exception as e2:
                                last_err = e2
                if loadability["ok"] is not True:
                    loadability["ok"] = False
                    loadability["error"] = str(last_err)[:500] if last_err else "unknown"
            except Exception as e:
                loadability["checked"] = True
                loadability["ok"] = False
                loadability["error"] = str(e)[:500]

            meta_equals_source = None
            meta_source_path = None
            try:
                if os.path.isdir(self.model_id):
                    p = os.path.join(self.model_id, "config.json")
                    if os.path.exists(p):
                        meta_source_path = p
                else:
                    meta_source_path = cached_file(
                        self.model_id,
                        "config.json",
                        _raise_exceptions_for_missing_entries=False,
                        local_files_only=bool(
                            getattr(self.config.security, "local_files_only", False)
                            or not getattr(self.config.security, "allow_network", True)
                        ),
                    )
                if meta_source_path and os.path.exists(meta_source_path) and os.path.exists(meta_path):
                    meta_equals_source = _hash_file(meta_source_path) == _hash_file(meta_path)
            except Exception as e:
                logger.debug("meta-config comparison failed: %s", e)
                meta_equals_source = None

            from .manifest import build_manifest

            security_context = getattr(self.config, "security_context", None)

            manifest = build_manifest(
                schema_version=2,
                generated_at=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                source={
                    "model_id": self.model_id,
                    "source_config_sha256": _hash_file(meta_path),
                    "meta_config_equals_source_config": meta_equals_source,
                },
                environment={
                    "vitriol_version": getattr(vitriol, "__version__", None),
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "torch": getattr(torch, "__version__", None),
                    "transformers": getattr(transformers, "__version__", None),
                },
                security={
                    "trust_remote_code": bool(getattr(self.config.security, "trust_remote_code", True)),
                    "allow_network": bool(getattr(self.config.security, "allow_network", True)),
                    "local_files_only": bool(getattr(self.config.security, "local_files_only", False)),
                },
                security_context=security_context,
                generation={
                    "strategy": self.config.strategy,
                    "shrink_config": bool(self.shrink_config),
                    "dtype": self.config.dtype,
                    "max_shard_size": self.config.max_shard_size,
                    "storage_format": self.strategy.storage_format,
                    "file_extension": self.strategy.file_extension,
                    "active_dims": active_dims,
                },
                artifacts={
                    "config_json": {"sha256": _hash_file(config_path)},
                    "meta_config_json": {"sha256": _hash_file(meta_path)},
                    "reconcile": {
                        "path": "vitriol-reconcile.json" if os.path.exists(reconcile_path) else None,
                        "sha256": _hash_file(reconcile_path),
                        "info": reconcile_info,
                    },
                    "index": {
                        "name": idx_name if os.path.exists(idx_path) else None,
                        "sha256": _hash_file(idx_path),
                        "total_size": (idx_data or {}).get("metadata", {}).get("total_size"),
                        "weight_entries": len(weight_map) if isinstance(weight_map, dict) else None,
                        "unique_shards": len(unique_shards),
                    },
                    "viz": {
                        "architecture_html": os.path.exists(os.path.join(self.output_dir, "architecture.html")),
                        "architecture_png": os.path.exists(os.path.join(self.output_dir, "architecture.png")),
                        "architecture_detail_png": os.path.exists(os.path.join(self.output_dir, "architecture_detail.png")),
                    },
                    "tokenizer": {
                        "tokenizer_json": os.path.exists(os.path.join(self.output_dir, "tokenizer.json")),
                        "tokenizer_config_json": os.path.exists(os.path.join(self.output_dir, "tokenizer_config.json")),
                    },
                },
                loadability=loadability,
            )

            with open(os.path.join(self.output_dir, "vitriol-manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Manifest write failed: %s", e)

    # ──────────────────────────────────────────────────────────────────────
    # Config / tokenizer save
    # ──────────────────────────────────────────────────────────────────────

    def _save_configs(self, hf_config) -> None:
        logger.info("Saving config…")

        # ── Always save original HF config as meta-config.json ──────────
        # This preserves the full, unmodified model configuration from HuggingFace
        # so that visualization tools can render the real architecture even when
        # the active config.json has been shrunk or modified.
        try:
            meta_src = None
            if os.path.isdir(self.model_id):
                local_cfg = os.path.join(self.model_id, "config.json")
                if os.path.exists(local_cfg):
                    meta_src = local_cfg
            if meta_src is None:
                meta_src = cached_file(
                    self.model_id,
                    "config.json",
                    _raise_exceptions_for_missing_entries=False,
                    local_files_only=(
                        getattr(self.config.security, "local_files_only", False)
                        or not getattr(self.config.security, "allow_network", True)
                    ),
                )
            if meta_src and os.path.exists(meta_src):
                shutil.copy2(meta_src, os.path.join(self.output_dir, "meta-config.json"))
                logger.info("Saved raw config.json → meta-config.json")
            else:
                logger.warning("meta-config.json save skipped (config.json not found).")
        except Exception as e:
            logger.warning("meta-config.json save failed: %s", e)

        # ── Legacy: also save config_meta.json if save_dummy_config is set ──
        if getattr(self.strategy, "save_dummy_config", False):
            meta_cfg_path = os.path.join(self.output_dir, "meta-config.json")
            legacy_path = os.path.join(self.output_dir, "config_meta.json")
            if os.path.exists(meta_cfg_path) and not os.path.exists(legacy_path):
                shutil.copy2(meta_cfg_path, legacy_path)
                logger.info("Copied meta-config.json → config_meta.json (legacy)")

        # ── Save the (shrink) active config ──────────────────────────────
        hf_config.save_pretrained(self.output_dir)

        # ── Post-process: strip raw dict sub-configs from the saved config ──
        # Some adapters (e.g. Qwen35MoeAdapter) strip sub-config dicts in-memory
        # but save_pretrained() may re-serialise them.  Also, _patch_model_config
        # may have already removed them.  As a safety net, re-read the saved JSON
        # and strip any remaining raw dict sub-configs.
        #
        # IMPORTANT: For multimodal models (Gemma-4, etc.), we preserve
        # vision_config because the model class needs it to instantiate the
        # vision tower.  We only strip if it's a raw dict that would crash
        # PretrainedConfig deserialisation.  If vision_config has been
        # converted to a PretrainedConfig object by the adapter, it's safe
        # to keep.
        saved_cfg_path = os.path.join(self.output_dir, "config.json")
        try:
            with open(saved_cfg_path) as f:
                saved_cfg = json.load(f)
            _stripped = False

            # Detect if this is a multimodal model
            _is_vlm = bool(
                saved_cfg.get("vision_soft_tokens_per_image")
                or saved_cfg.get("image_token_id") is not None
                or any("ConditionalGeneration" in a for a in saved_cfg.get("architectures", []))
            )

            def _is_transformers_known_model_type(model_type: Any) -> bool:
                if not isinstance(model_type, str) or not model_type:
                    return False
                try:
                    AutoConfig.for_model(model_type)
                    return True
                except Exception:
                    return False

            # --- Align model_type and architectures with original meta-config ---
            meta_cfg_path = os.path.join(self.output_dir, "meta-config.json")
            if os.path.exists(meta_cfg_path):
                try:
                    with open(meta_cfg_path) as mf:
                        meta_cfg = json.load(mf)
                    should_align_model_type = _is_transformers_known_model_type(meta_cfg.get("model_type"))
                    if (
                        "model_type" in meta_cfg
                        and saved_cfg.get("model_type") != meta_cfg["model_type"]
                        and should_align_model_type
                    ):
                        saved_cfg["model_type"] = meta_cfg["model_type"]
                        _stripped = True
                        logger.info("Aligned model_type to original: %s", meta_cfg["model_type"])
                    elif "model_type" in meta_cfg and saved_cfg.get("model_type") != meta_cfg["model_type"]:
                        logger.info(
                            "Kept reloadable fallback model_type '%s'; original unknown model_type is preserved in meta-config.json: %s",
                            saved_cfg.get("model_type"),
                            meta_cfg["model_type"],
                        )
                    if (
                        "architectures" in meta_cfg
                        and saved_cfg.get("architectures") != meta_cfg["architectures"]
                        and should_align_model_type
                    ):
                        saved_cfg["architectures"] = meta_cfg["architectures"]
                        _stripped = True
                        logger.info("Aligned architectures to original: %s", meta_cfg["architectures"])
                    if (
                        getattr(self.config.security, "trust_remote_code", False)
                        and "auto_map" in meta_cfg
                        and saved_cfg.get("auto_map") != meta_cfg["auto_map"]
                    ):
                        saved_cfg["auto_map"] = meta_cfg["auto_map"]
                        _stripped = True
                        logger.info("Aligned auto_map to original (trust_remote_code enabled).")
                except Exception as e:
                    logger.warning("Failed to align model_type/architectures: %s", e)

            if getattr(self, "shrink_config", False) and isinstance(saved_cfg.get("quantization_config"), dict):
                del saved_cfg["quantization_config"]
                _stripped = True

            for sub_key in ("text_config", "vision_config",
                            "encoder_config", "decoder_config",
                            "audio_config"):
                if not isinstance(saved_cfg.get(sub_key), dict):
                    continue
                # For VLMs, keep vision_config if it looks like a valid
                # PretrainedConfig dict (has model_type or hidden_size).
                if sub_key == "vision_config" and _is_vlm:
                    vc = saved_cfg[sub_key]
                    if isinstance(vc, dict) and (
                        vc.get("model_type") or vc.get("hidden_size")
                    ):
                        # Looks like a proper config dict — keep it
                        logger.info(
                            "Preserving vision_config dict in config.json for VLM model."
                        )
                        continue
                del saved_cfg[sub_key]
                _stripped = True
                logger.info("Stripped raw dict sub-config '%s' from saved config.json", sub_key)
            if _stripped:
                with open(saved_cfg_path, "w") as f:
                    json.dump(saved_cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Post-process config strip failed: %s", e)

        if self.shrink_config:
            # ── Reconcile config.json with actual generated weight shapes ──
            # The shrink process + model initialisation (which may fall back to
            # LlamaForCausalLM) can produce weights whose shapes do NOT match
            # the config's dimension fields (e.g. intermediate_size is 512 in
            # config but the actual gate_proj has shape [11008, 256]).
            # We scan the actual saved weights and patch config.json so that
            # `from_pretrained()` does not hit size-mismatch errors.
            self._reconcile_config_with_weights(saved_cfg_path)

    def _reconcile_config_with_weights(self, cfg_path: str) -> None:
        """Patch config.json so that its dimension fields match actual weight shapes.

        After shrink + model initialisation (which may fallback to Llama, etc.),
        the config may declare intermediate_size=512 while the generated weights
        actually have gate_proj shape [11008, 256].  This mismatch causes
        ``from_pretrained()`` to fail with RuntimeError (size mismatch).

        This method:
        1. Reads the first .bin shard to discover actual weight shapes.
        2. Infers vocab_size, hidden_size, intermediate_size, num_hidden_layers
           from the weight shapes.
        3. Patches config.json with the inferred values.
        """
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception as e:
            logger.debug("Failed to load config for reconciliation: %s", e)
            return

        # Find weight shapes from the first few shards
        shapes: Dict[str, tuple] = {}
        idx_path = os.path.join(self.output_dir, "pytorch_model.bin.index.json")
        if not os.path.exists(idx_path):
            return
        try:
            with open(idx_path) as f:
                idx_data = json.load(f)
            weight_map = idx_data.get("weight_map", {})
        except Exception as e:
            logger.debug("Failed to read index for reconciliation: %s", e)
            return

        # Load a representative sample of shards to get key shapes
        shards_to_read = set()
        for key in weight_map:
            if any(p in key for p in ("embed_tokens", "gate_proj", "q_proj", "lm_head")):
                shards_to_read.add(weight_map[key])
            if len(shards_to_read) >= 4:
                break

        for shard_name in shards_to_read:
            shard_path = os.path.join(self.output_dir, shard_name)
            if os.path.exists(shard_path):
                try:
                    # [Security hardening] Prefer weights_only=True for torch.load
                    # to prevent arbitrary code execution via pickle deserialization.
                    # Fall back to weights_only=False only if the shard contains
                    # non-tensor objects (unlikely for weight shards).
                    try:
                        d = torch.load(shard_path, map_location="cpu", weights_only=True)
                    except Exception:
                        d = torch.load(shard_path, map_location="cpu", weights_only=False)
                    for k, v in d.items():
                        shapes[k] = tuple(v.shape)
                except Exception as e:
                    logger.debug("Failed to load shard %s for shape extraction: %s", shard_path, e)

        if not shapes:
            logger.debug("No weight shapes found — skipping config reconciliation.")
            return

        # Infer dimensions from weight shapes
        patched = False
        diff: Dict[str, Dict[str, Any]] = {}

        def _set(k: str, v: Any) -> None:
            nonlocal patched
            if cfg.get(k) != v:
                diff[k] = {"before": cfg.get(k), "after": v}
                cfg[k] = v
                patched = True

        # vocab_size from embed_tokens.weight [vocab, hidden]
        for k, s in shapes.items():
            if "embed_tokens" in k and len(s) == 2:
                _set("vocab_size", s[0])
                _set("hidden_size", s[1])
                break

        # intermediate_size from gate_proj.weight [inter, hidden]
        for k, s in shapes.items():
            if "gate_proj" in k and len(s) == 2:
                _set("intermediate_size", s[0])
                break

        # num_hidden_layers from highest layer index
        max_layer = -1
        for k in weight_map:
            import re as _re
            m = _re.search(r"layers\.(\d+)\.", k)
            if m:
                max_layer = max(max_layer, int(m.group(1)))
        if max_layer >= 0:
            inferred_layers = max_layer + 1
            _set("num_hidden_layers", inferred_layers)

        # num_attention_heads / num_key_value_heads from q_proj/k_proj shapes
        cfg.get("hidden_size", 256)
        for k, s in shapes.items():
            if "self_attn.q_proj" in k and len(s) == 2:
                # q_proj: [num_heads * head_dim, hidden]
                head_dim = cfg.get("head_dim")
                if head_dim and head_dim > 0:
                    n_heads = s[0] // head_dim
                    if n_heads > 0:
                        _set("num_attention_heads", n_heads)
                break

        if patched:
            try:
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                logger.info(
                    "Reconciled config.json with actual weight shapes: "
                    "vocab=%s, hidden=%s, inter=%s, layers=%s",
                    cfg.get("vocab_size"), cfg.get("hidden_size"),
                    cfg.get("intermediate_size"), cfg.get("num_hidden_layers"),
                )
            except Exception as e:
                logger.warning("Config reconciliation write failed: %s", e)
        else:
            logger.debug("Config already consistent with weight shapes — no patch needed.")

        diff_path = os.path.join(self.output_dir, "vitriol-reconcile.json")
        try:
            with open(diff_path, "w") as f:
                json.dump({"patched": bool(patched), "diff": diff}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to write reconcile diff: %s", e)

    def _save_tokenizer(self) -> None:
        logger.info("Saving tokenizer…")
        try:
            from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

            local_files_only = bool(
                getattr(self.config.security, "local_files_only", False)
                or not getattr(self.config.security, "allow_network", True)
            )
            tok = hf_load_tokenizer(
                self.model_id,
                security={
                    "trust_remote_code": self.config.security.trust_remote_code,
                    "allow_network": not local_files_only,
                    "local_files_only": local_files_only,
                },
            )
            tok.save_pretrained(self.output_dir)
        except Exception as e:
            logger.warning("Tokenizer save failed: %s", e)
            self._copy_tokenizer_assets()

    def _copy_tokenizer_assets(self) -> None:
        local_files_only = bool(
            getattr(self.config.security, "local_files_only", False)
            or not getattr(self.config.security, "allow_network", True)
        )
        if os.path.isdir(self.model_id):
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",
                "vocab.json",
                "merges.txt",
                "sentencepiece.bpe.model",
            ):
                src = os.path.join(self.model_id, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(self.output_dir, name))
            return
        if local_files_only:
            return
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            repo_files = set(list_repo_files(self.model_id))
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",
                "vocab.json",
                "merges.txt",
                "sentencepiece.bpe.model",
            ):
                if name not in repo_files:
                    continue
                src = hf_hub_download(
                    repo_id=self.model_id,
                    filename=name,
                    local_files_only=local_files_only,
                )
                shutil.copy2(src, os.path.join(self.output_dir, name))
        except Exception as copy_exc:
            logger.warning("Tokenizer asset fallback failed: %s", copy_exc)

    # ──────────────────────────────────────────────────────────────────────
    # README
    # ──────────────────────────────────────────────────────────────────────

    def _add_readme_metadata(self) -> None:
        has_viz   = self._generate_visualization(
            os.path.join(self.output_dir, "architecture.png"))
        viz_block = (
            "## 🔍 Architecture Visualization\n\n"
            "![Block Diagram](architecture.png)\n"
            "![Detailed Architecture](architecture_detail.png)\n\n"
            "*Interactive: [architecture.html](architecture.html)*\n"
        ) if has_viz else "## 🔍 Architecture Visualization\n*(Unavailable)*\n"

        # [B6 fix] datetime instead of subprocess call
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        content = f"""---
library_name: transformers
tags:
- vitriol
- minimal-weights
- architectural-test
---
# Vitriol-Compressed: {self.model_id}

⚠️ **IMPORTANT WARNING**: This model contains **minimal/dummy weights** generated by
[Vitriol](https://github.com/isLinXu/Vitriol). It is **NOT** intended for inference.

## 📊 Model Details
| Field | Value |
|-------|-------|
| Original | `{self.model_id}` |
| Strategy | `{self.config.strategy}` |
| Shrunk | `{self.shrink_config}` |
| Generated | {ts} |

## 🛠️ How to Use
```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

path = "{self.output_dir}"
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
trust_remote_code = True  # set False to disable remote code execution
cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code)
for k in ("text_config", "vision_config", "encoder_config", "decoder_config"):
    v = getattr(cfg, k, None)
    if isinstance(v, dict):
        setattr(cfg, k, PretrainedConfig.from_dict(v))
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=trust_remote_code, config=cfg)
out = model.generate(**tok("hello", return_tensors="pt"), max_new_tokens=8)
logger.info(tok.decode(out[0]))
```

{viz_block}

## 🔗 About Vitriol
- **GitHub**: https://github.com/isLinXu/Vitriol

---
*Generated with ❤️ by Vitriol*
"""
        readme = os.path.join(self.output_dir, "README.md")
        with open(readme, "w") as f:
            f.write(content)
        logger.info("README.md written.")

    # ──────────────────────────────────────────────────────────────────────
    # Architecture visualisation
    # ──────────────────────────────────────────────────────────────────────

    def _generate_visualization(self, viz_path: str) -> bool:
        try:
            from ..arch_viz.visualizer import ArchitectureVisualizer
        except ImportError:
            logger.debug("arch_viz unavailable; skipping visualization.")
            return False

        cfg_json    = os.path.join(self.output_dir, "config.json")
        cfg_meta    = os.path.join(self.output_dir, "meta-config.json")
        cfg_meta_legacy = os.path.join(self.output_dir, "config_meta.json")
        cfg_shrunk  = os.path.join(self.output_dir, "config_shrunk.json")
        temp_swap   = False

        # Resolve meta config path (prefer meta-config.json, fallback config_meta.json)
        meta_path = cfg_meta if os.path.exists(cfg_meta) else (
            cfg_meta_legacy if os.path.exists(cfg_meta_legacy) else None
        )

        try:
            # Strategy A: meta config exists → swap to expose original for visualization
            if meta_path:
                os.rename(cfg_json, cfg_shrunk)
                shutil.copy2(meta_path, cfg_json)
                temp_swap = True

            # Strategy B: shrink mode, no meta → fetch from Hub
            elif self.shrink_config:
                try:
                    orig = self._load_hf_config()
                    self._patch_model_config(orig)
                    with tempfile.TemporaryDirectory() as tmp:
                        orig.save_pretrained(tmp)
                        os.rename(cfg_json, cfg_shrunk)
                        shutil.move(os.path.join(tmp, "config.json"), cfg_json)
                    temp_swap = True
                    logger.info("Fetched original config for visualization.")
                except Exception as e:
                    logger.warning("Config fetch for viz failed: %s. Using shrunk.", e)

            viz = ArchitectureVisualizer(
                self.output_dir,
                trust_remote_code=bool(getattr(self.config.security, "trust_remote_code", True)),
                local_files_only=True,
            )
            viz.generate_block_diagram(os.path.join(self.output_dir, "architecture.png"))
            viz.generate_detailed_diagram(os.path.join(self.output_dir, "architecture_detail.png"))
            viz.generate_interactive_html(os.path.join(self.output_dir, "architecture.html"))
            return True

        except Exception as e:
            logger.warning("Visualization failed: %s", e)
            return False

        finally:
            # Restore: shrunk → config.json; meta-config.json stays as-is
            if temp_swap and os.path.exists(cfg_shrunk):
                try:
                    if os.path.exists(cfg_json):
                        os.remove(cfg_json)
                    # Restore shrunk config as the active config.json
                    os.rename(cfg_shrunk, cfg_json)
                except Exception as e:
                    logger.error("Config restore in finally block failed: %s", e)
