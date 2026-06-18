from __future__ import annotations

import datetime
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.utils import cached_file  # noqa: E402

from ..patches import PatchRegistry, apply_all_patches, patch_remote_module  # noqa: E402

from ._generator_utils import (  # noqa: E402
    GenerationResult,
    _FALLBACK_CHAIN,
    _MOE_FALLBACK_CHAIN,
    _ROPE_DEFAULTS,
    _SAFE_KEYS,
    _BLOCKED_CUSTOM_ASSET_EXTENSIONS,
    _CUSTOM_ASSET_EXTENSIONS,
    _CUSTOM_CODE_PREFIXES,
    _CUSTOM_CODE_MAX_FILES_ENV,
    _CUSTOM_CODE_MAX_PY_BYTES_ENV,
    _CUSTOM_CODE_MAX_ASSET_BYTES_ENV,
    _CUSTOM_CODE_DEFAULT_MAX_FILES,
    _CUSTOM_CODE_DEFAULT_MAX_PY_BYTES,
    _CUSTOM_CODE_DEFAULT_MAX_ASSET_BYTES,
    positive_int_env,
    custom_repo_file_size_limit,
    set_missing,
    inject_recursive,
    copy_safe_attrs,
    build_fallback_config,
    extract_shard_id,
    find_best_alias,
    extract_auto_map_modules,
    custom_code_file_matches_auto_map,
    is_allowed_custom_repo_file,
)
from .shrinker import ConfigShrinker  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Logger — must exist before any module-level patch references it
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

apply_all_patches()

# ═════════════════════════════════════════════════════════════════════════════
# § 3  LATE IMPORTS (project-internal; after patches)
# ═════════════════════════════════════════════════════════════════════════════

from ..adapters.registry import AdapterRegistry  # noqa: E402
from ..config.manager import GenerationConfig  # noqa: E402
from ..strategies import get_strategy  # noqa: E402
from ..utils.exceptions import GenerationError, ModelNotSupportedError  # noqa: E402
from ..utils.size import parse_size_to_bytes  # noqa: E402
from .incremental import IncrementalGenerator  # noqa: E402

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
        config: Optional[GenerationConfig] = None,
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
        return parse_size_to_bytes(size_str)

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
            logger.debug("tensor.untyped_storage().nbytes() failed, falling back to element_size * nelement")
        try:
            return int(tensor.element_size() * tensor.nelement())
        except Exception:
            return int(fallback_numel) * int(MinimalWeightGenerator._get_dtype_size(fallback_dtype))

    # ──────────────────────────────────────────────────────────────────────
    # Config shrinking
    # ──────────────────────────────────────────────────────────────────────

    # VLM vision tower uses larger min dims to avoid dimension mismatch
    # when the model class expects vision weights with specific shapes.
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
        available_shards: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not original_shard_map:
            return None

        available = available_shards if available_shards is not None else sorted(set(original_shard_map.values()))
        n_shards  = len(available)
        if n_shards == 0:
            return None

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
            inject_recursive(hf_config, "use_flash_attention_2", False)
            inject_recursive(hf_config, "use_flash_attn",        False)
            inject_recursive(hf_config, "use_deterministic_attn", False)

        # 3. Family-specific patches
        PatchRegistry.apply(hf_config, self.model_id)

        logger.debug(
            "Config patched: _attn_impl=%s  fa2=%s",
            getattr(hf_config, "_attn_implementation", "N/A"),
            getattr(hf_config, "use_flash_attention_2", "N/A"),
        )

    @staticmethod
    def _snapshot_export_tensors(model) -> List[Tuple[str, Any]]:
        """Collect exportable parameters and persistent buffers in a reusable snapshot."""
        non_persistent_buffers: Set[str] = set()
        for module_prefix, module in model.named_modules():
            local_names: Set[str] = getattr(module, "_non_persistent_buffers_set", set())
            for local_name in local_names:
                qualified = f"{module_prefix}.{local_name}" if module_prefix else local_name
                non_persistent_buffers.add(qualified)

        export_items: List[Tuple[str, Any]] = list(model.named_parameters())
        for name, buf in model.named_buffers():
            if name in non_persistent_buffers:
                logger.debug("Skipping non-persistent buffer during export: %s", name)
                continue
            export_items.append((name, buf))
        return export_items

    # ──────────────────────────────────────────────────────────────────────
    # Remote-class patching (Kimi, DeepSeek, MoonViT …)
    # ──────────────────────────────────────────────────────────────────────

    def _patch_remote_classes(self, hf_config) -> None:
        if not hasattr(hf_config, "auto_map"):
            return
        if not getattr(self.config.security, "trust_remote_code", False):
            logger.debug("Skipping dynamic module patching because trust_remote_code is disabled.")
            return
        local_files_only = bool(
            getattr(self.config.security, "local_files_only", False)
            or not getattr(self.config.security, "allow_network", True)
        )
        for key in ("AutoModelForCausalLM", "AutoModel"):
            if key not in hf_config.auto_map:
                continue
            try:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                cls = get_class_from_dynamic_module(
                    hf_config.auto_map[key],
                    self.model_id,
                    local_files_only=local_files_only,
                )
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
                return build_fallback_config(cfg_name, cls_name, hf_config)
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

    def _load_hf_config(self):
        from .config_loader import load_hf_config

        return load_hf_config(self)

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
            generated_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
            ConfigShrinker().shrink(hf_config)

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
                        logger.debug("Failed to set config attribute %s", k)
            for k in ("qk_nope_head_dim", "qk_rope_head_dim", "qk_head_dim", "v_head_dim"):
                if hasattr(hf_config, k) and not hasattr(active_config, k):
                    try:
                        setattr(active_config, k, getattr(hf_config, k))
                    except Exception:
                        logger.debug("Failed to copy config attribute %s from hf_config", k)

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

        export_items = self._snapshot_export_tensors(model)

        # Snapshot names for progress bar (without materialising tensors)
        all_names = [n for n, _ in export_items]
        total_params = len(all_names)

        current_target: Optional[str] = None
        if all_names and original_shard_map:
            current_target = self._resolve_target_shard(
                all_names[0], original_shard_map, None, param_seq_idx=0, available_shards=expected_shards)

        pbar = tqdm(total=total_params, desc="Generating tensors")
        pbar.update(len(generated_names))

        try:
            # Per-shard byte tracker for streaming flush
            buf_bytes: Dict[str, int] = {}
            # [F-1] Track shards written by streaming flush (distinct from "still empty")
            flushed_shards: Set[str] = set()
            param_seq_idx = 0   # monotonically increments; used for even fallback distribution

            # ── 4. Generation loop ──────────────────────────────────────────
            for name, param in export_items:  # [A7] reusable snapshot, no repeated traversal
                if name in generated_names:  # [B3 fix] stable name key
                    param_seq_idx += 1
                    pbar.update(1)
                    continue

                # [F-3] Pass param_seq_idx for even fallback distribution
                target = self._resolve_target_shard(
                    name, original_shard_map, current_target,
                    param_seq_idx=param_seq_idx,
                    available_shards=expected_shards,
                )
                if original_shard_map and not target:
                    # Still no target? Use round-robin on expected_shards
                    target = expected_shards[param_seq_idx % n_expected] if n_expected else None
                if not target:
                    target = (
                        f"{self.strategy.get_shard_prefix()}-00001-of-{{total:05d}}"
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
                        and "{total:05d}" not in target):
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
                if "{total:05d}" in filename else filename
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
        from .custom_code_sync import copy_custom_code_files

        copy_custom_code_files(self)

    def _custom_code_modules_from_saved_config(self) -> Set[str] | None:
        from .custom_code_sync import custom_code_modules_from_saved_config

        return custom_code_modules_from_saved_config(self)

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
        """Write ``vitriol-manifest.json``.

        Delegated to :mod:`vitriol.core.manifest_writer` to keep this module
        focused on generation orchestration. The wrapper preserves the legacy
        method name for any tests or subclasses that may still call it.
        """
        from .manifest_writer import write_manifest

        write_manifest(self)

    # ──────────────────────────────────────────────────────────────────────
    # Config / tokenizer save
    # ──────────────────────────────────────────────────────────────────────

    def _save_configs(self, hf_config) -> bool:
        from .generator_persistence import save_configs

        save_configs(self, hf_config)
        return True

    def _reconcile_config_with_weights(self, cfg_path: str) -> None:
        from .generator_persistence import reconcile_config_with_weights

        reconcile_config_with_weights(self, cfg_path)

    def _save_tokenizer(self) -> None:
        from .generator_persistence import save_tokenizer

        save_tokenizer(self)

    def _copy_tokenizer_assets(self) -> None:
        from .generator_persistence import copy_tokenizer_assets

        copy_tokenizer_assets(self)

    def _add_readme_metadata(self) -> None:
        from .generator_persistence import add_readme_metadata

        add_readme_metadata(self)

    def _generate_visualization(self, viz_path: str) -> bool:
        from .generator_persistence import generate_visualization

        return generate_visualization(self, viz_path)
