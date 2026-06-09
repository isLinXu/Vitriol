"""Manifest writer — generates the ``vitriol-manifest.json`` audit trail.

Extracted from ``core/generator.py`` to keep that file's surface area
manageable. The implementation reads only from a generator instance
(treated as a stable internal contract) and writes a single JSON file;
behavior is intentionally identical to the previous in-class method.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import sys
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.utils import cached_file

import vitriol

from .manifest import build_manifest

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from .generator import MinimalWeightGenerator

logger = logging.getLogger(__name__)


def _hash_file(path: str) -> Optional[str]:
    """Stream-hash a file with SHA-256; return None when missing."""
    if not path or not os.path.exists(path):
        return None
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_index_path(generator: MinimalWeightGenerator) -> tuple[str, str, str]:
    """Return ``(idx_name, idx_path, prefix)`` for the active storage format."""
    prefix = generator.strategy.get_shard_prefix()
    idx_name = (
        f"{prefix}.bin.index.json"
        if generator.strategy.storage_format == "pytorch"
        else "model.safetensors.index.json"
    )
    idx_path = os.path.join(generator.output_dir, idx_name)
    return idx_name, idx_path, prefix


def _load_index_data(idx_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(idx_path):
        return None
    try:
        with open(idx_path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to read index file %s: %s", idx_path, e)
        return None


def _extract_active_dims(config_path: str) -> Optional[Dict[str, Any]]:
    """Read selected dimension fields from the saved active config.json."""
    try:
        if not os.path.exists(config_path):
            return None
        with open(config_path) as f:
            active_cfg = json.load(f)
        return {
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
        return None


def _extract_reconcile_info(reconcile_path: str) -> Optional[Dict[str, Any]]:
    """Summarise vitriol-reconcile.json into a compact dict for the manifest."""
    if not os.path.exists(reconcile_path):
        return None
    try:
        with open(reconcile_path) as f:
            r = json.load(f)
        if not isinstance(r, dict):
            return None
        diff_raw = r.get("diff")
        diff_obj = diff_raw if isinstance(diff_raw, dict) else {}
        return {
            "patched": bool(r.get("patched")),
            "diff_keys": sorted(diff_obj.keys()),
            "diff_count": len(diff_obj),
        }
    except Exception as e:
        logger.debug("Reconcile info extraction failed: %s", e)
        return None


def _check_loadability(
    generator: MinimalWeightGenerator,
) -> Dict[str, Any]:
    """Probe whether the generated artifacts can be loaded by AutoModel*.

    The probe runs in offline mode (``local_files_only=True``) and tries
    ``CausalLM → Seq2Seq → AutoModel`` in order, with a retry that enables
    ``ignore_mismatched_sizes`` for size-mismatch errors.
    """
    loadability: Dict[str, Any] = {
        "checked": False,
        "ok": None,
        "loader": None,
        "error": None,
    }
    try:
        loadability["checked"] = True
        trust_rc = bool(getattr(generator.config.security, "trust_remote_code", True))
        last_err: Optional[BaseException] = None
        load_kwargs: Dict[str, Any] = dict(
            local_files_only=True,
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
                loader.from_pretrained(generator.output_dir, **load_kwargs)
                loadability["ok"] = True
                loadability["loader"] = name
                break
            except Exception as e:
                last_err = e
                logger.debug("Loadability check with %s failed: %s", name, e)
                err_str = str(e).lower()
                # Retry with ignore_mismatched_sizes for shape mismatches.
                if "size mismatch" in err_str or "mismatch" in err_str:
                    try:
                        loader.from_pretrained(
                            generator.output_dir,
                            **load_kwargs,
                            ignore_mismatched_sizes=True,
                        )
                        loadability["ok"] = True
                        loadability["loader"] = f"{name} (ignore_mismatched)"
                        break
                    except Exception as e2:
                        last_err = e2
                        logger.debug("Loadability retry with %s (ignore_mismatched) also failed: %s", name, e2)
        if loadability["ok"] is not True:
            loadability["ok"] = False
            loadability["error"] = str(last_err)[:500] if last_err else "unknown"
    except Exception as e:
        loadability["checked"] = True
        loadability["ok"] = False
        loadability["error"] = str(e)[:500]
    return loadability


def _resolve_source_config_match(
    generator: MinimalWeightGenerator,
    meta_path: str,
) -> Optional[bool]:
    """Compare on-disk meta-config.json against the upstream HF source."""
    try:
        if os.path.isdir(generator.model_id):
            candidate = os.path.join(generator.model_id, "config.json")
            meta_source_path = candidate if os.path.exists(candidate) else None
        else:
            meta_source_path = cached_file(
                generator.model_id,
                "config.json",
                _raise_exceptions_for_missing_entries=False,
                local_files_only=bool(
                    getattr(generator.config.security, "local_files_only", False)
                    or not getattr(generator.config.security, "allow_network", True)
                ),
            )
        if (
            meta_source_path
            and os.path.exists(meta_source_path)
            and os.path.exists(meta_path)
        ):
            return _hash_file(meta_source_path) == _hash_file(meta_path)
    except Exception as e:
        logger.debug("meta-config comparison failed: %s", e)
    return None


def write_manifest(generator: MinimalWeightGenerator) -> None:
    """Write ``vitriol-manifest.json`` summarising this generation run.

    Failures are logged but never raised — manifest emission is a best-effort
    audit trail; a missing manifest must not break weight generation.
    """
    try:
        config_path = os.path.join(generator.output_dir, "config.json")
        meta_path = os.path.join(generator.output_dir, "meta-config.json")
        reconcile_path = os.path.join(generator.output_dir, "vitriol-reconcile.json")

        idx_name, idx_path, _prefix = _resolve_index_path(generator)
        idx_data = _load_index_data(idx_path)

        weight_map = (idx_data or {}).get("weight_map", {})
        unique_shards = sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []

        active_dims = _extract_active_dims(config_path)
        reconcile_info = _extract_reconcile_info(reconcile_path)
        loadability = _check_loadability(generator)
        meta_equals_source = _resolve_source_config_match(generator, meta_path)
        security_context = getattr(generator.config, "security_context", None)

        manifest = build_manifest(
            schema_version=2,
            generated_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            source={
                "model_id": generator.model_id,
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
                "trust_remote_code": bool(getattr(generator.config.security, "trust_remote_code", True)),
                "allow_network": bool(getattr(generator.config.security, "allow_network", True)),
                "local_files_only": bool(getattr(generator.config.security, "local_files_only", False)),
            },
            security_context=security_context,
            generation={
                "strategy": generator.config.strategy,
                "shrink_config": bool(generator.shrink_config),
                "dtype": generator.config.dtype,
                "max_shard_size": generator.config.max_shard_size,
                "storage_format": generator.strategy.storage_format,
                "file_extension": generator.strategy.file_extension,
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
                    "architecture_html": os.path.exists(os.path.join(generator.output_dir, "architecture.html")),
                    "architecture_png": os.path.exists(os.path.join(generator.output_dir, "architecture.png")),
                    "architecture_detail_png": os.path.exists(
                        os.path.join(generator.output_dir, "architecture_detail.png")
                    ),
                },
                "tokenizer": {
                    "tokenizer_json": os.path.exists(os.path.join(generator.output_dir, "tokenizer.json")),
                    "tokenizer_config_json": os.path.exists(
                        os.path.join(generator.output_dir, "tokenizer_config.json")
                    ),
                },
            },
            loadability=loadability,
        )

        with open(os.path.join(generator.output_dir, "vitriol-manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Manifest write failed: %s", e)
