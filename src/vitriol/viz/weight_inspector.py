"""
Weight inspector.

Reads real weight shards exported by Ultra/HybridUltra and extracts per-layer
statistics for 3D visualization.

Capabilities:
  - Read .safetensors / .bin weight files
  - Extract tensor shape / dtype / parameter count
  - Compute summary stats: mean, std, min, max, sparsity, L2 norm
  - Produce JSON-friendly data for the 3D frontend
  - Prefer meta-config.json to preserve the original HF architecture config
"""

import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Optional dependencies
# ──────────────────────────────────────────────────────────────────────
try:
    from safetensors.torch import load_file as _safetensors_load
except ImportError:
    _safetensors_load = None

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _TensorStub:
    """Lightweight safetensors header stub used when torch is unavailable."""

    shape: Tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return int(n)

    def element_size(self) -> int:
        # Common safetensors dtypes include F32/F16/BF16/I32/U8, etc.
        dtype = self.dtype.upper()
        return {
            "F64": 8,
            "F32": 4,
            "F16": 2,
            "BF16": 2,
            "I64": 8,
            "I32": 4,
            "I16": 2,
            "I8": 1,
            "U8": 1,
            "BOOL": 1,
        }.get(dtype, 0)


def _read_safetensors_header(file_path: Path) -> Dict[str, Any]:
    """Read a safetensors header without torch.

    Safetensors format: first 8 bytes (little-endian) = header length,
    followed by a JSON header blob.
    """
    with file_path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            return {}
        header_len = int.from_bytes(raw_len, "little")
        header_raw = f.read(header_len)
    try:
        return json.loads(header_raw.decode("utf-8"))
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────
# Effective config loading (prefer meta-config.json)
# ──────────────────────────────────────────────────────────────────────

def load_effective_config(model_dir: str) -> Dict[str, Any]:
    """Load an effective config, preferring meta-config.json (original HF config).

    Priority: meta-config.json > config_meta.json > config.json

    Args:
        model_dir: Model directory path.

    Returns:
        Parsed config dictionary.
    """
    model_path = Path(model_dir)

    for meta_name in ("meta-config.json", "config_meta.json"):
        meta_path = model_path / meta_name
        if meta_path.exists():
            try:
                cfg = json.loads(meta_path.read_text(encoding="utf-8"))
                logger.info("Loaded effective config from %s (original HF config)", meta_name)
                return cfg
            except Exception as e:
                logger.warning("Failed to parse %s: %s", meta_name, e)

    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            logger.info("Loaded effective config from config.json")
            return cfg
        except Exception as e:
            logger.warning("Failed to parse config.json: %s", e)

    return {}


# ──────────────────────────────────────────────────────────────────────
# Weight loading
# ──────────────────────────────────────────────────────────────────────

def _list_weight_files(model_dir: Path) -> Tuple[List[Path], bool]:
    """List weight shard files, preferring safetensors.

    Returns:
        (file list, is_safetensors)
    """
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    bin_files = sorted(model_dir.glob("*.bin"))

    if safetensor_files:
        return safetensor_files, True
    elif bin_files:
        return bin_files, False
    return [], False


_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


def _numel_from_shape(shape: Any) -> int:
    total = 1
    if not isinstance(shape, (list, tuple)):
        return 0
    for dim in shape:
        try:
            total *= int(dim)
        except (TypeError, ValueError):
            return 0
    return int(total)


def _read_safetensors_metadata(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read safetensors tensor metadata without importing torch or loading tensor data."""
    try:
        with file_path.open("rb") as fh:
            header_len_raw = fh.read(8)
            if len(header_len_raw) != 8:
                return {}
            header_len = int.from_bytes(header_len_raw, "little")
            header = json.loads(fh.read(header_len).decode("utf-8"))
    except Exception as exc:
        logger.debug("Failed to read safetensors metadata from %s: %s", file_path, exc)
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = [int(x) for x in meta.get("shape", [])]
        dtype = str(meta.get("dtype", "UNKNOWN"))
        offsets = meta.get("data_offsets") or [0, 0]
        try:
            storage_bytes = max(int(offsets[1]) - int(offsets[0]), 0)
        except Exception:
            storage_bytes = 0
        if storage_bytes <= 0:
            storage_bytes = _numel_from_shape(shape) * _DTYPE_BYTES.get(dtype.upper(), 0)
        out[name] = {
            "shape": shape,
            "dtype": dtype,
            "numel": _numel_from_shape(shape),
            "storage_bytes": storage_bytes,
        }
    return out


def _metadata_stats(name: str, meta: Dict[str, Any], shard_file: str) -> Dict[str, Any]:
    shape = list(meta.get("shape") or [])
    dtype = str(meta.get("dtype") or "UNKNOWN")
    numel = int(meta.get("numel", _numel_from_shape(shape)) or 0)
    elem_size = _DTYPE_BYTES.get(dtype.upper(), 0)
    params_bytes = numel * elem_size
    storage_bytes = int(meta.get("storage_bytes", params_bytes) or params_bytes)
    return {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "numel": numel,
        "params_bytes": params_bytes,
        "storage_bytes": storage_bytes,
        "is_strided": False,
        "compression_ratio": params_bytes / max(storage_bytes, 1),
        "metadata_only": True,
        "layer_type": _classify_layer(name),
        "shard_file": shard_file,
    }


def _scan_safetensors_metadata(weight_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for wf in weight_files:
        for name, meta in _read_safetensors_metadata(wf).items():
            meta = dict(meta)
            meta["shard_file"] = wf.name
            metadata[name] = meta
    return metadata


def _load_shard(file_path: Path, is_safetensors: bool) -> Dict[str, Any]:
    """Load a single weight shard.

    Args:
        file_path: Shard file path.
        is_safetensors: Whether this shard is a safetensors file.

    Returns:
        Mapping from tensor name to tensor (or a metadata stub in header-only mode).
    """
    if is_safetensors:
        if _safetensors_load is not None:
            return _safetensors_load(str(file_path))

        # If torch/safetensors.torch is unavailable, fall back to header-only mode.
        header = _read_safetensors_header(file_path)
        if not header:
            logger.error("safetensors header unavailable — cannot load %s", file_path)
            return {}

        out: Dict[str, Any] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                continue
            shape = tuple(int(x) for x in meta.get("shape", []) if isinstance(x, (int, float, str)))
            dtype = str(meta.get("dtype", "UNKNOWN"))
            out[name] = _TensorStub(shape=shape, dtype=dtype)
        return out
    else:
        if torch is None:
            logger.error("PyTorch not installed — cannot load %s", file_path)
            return {}
        try:
            return torch.load(str(file_path), map_location="cpu", weights_only=True)
        except Exception:
            # fallback for legacy format
            return torch.load(str(file_path), map_location="cpu", weights_only=False)


# ──────────────────────────────────────────────────────────────────────
# Stats extraction
# ──────────────────────────────────────────────────────────────────────

def _compute_tensor_stats(
    tensor: Any,
    *,
    seed: int = 42,
    sample_size: int = 1_000_000,
) -> Dict[str, Any]:
    """Compute statistics for a single tensor.

    For Ultra-style stride=0 tensors, this keeps the logical shape while computing
    stats from the underlying storage.

    Args:
        tensor: A PyTorch tensor.

    Returns:
        A stats dictionary.
    """
    # When torch is unavailable, return best-effort metadata (shape/dtype/numel/bytes)
    # to avoid the visualization treating everything as missing.
    if torch is None or not hasattr(tensor, "flatten"):
        shape = list(getattr(tensor, "shape", []))
        dtype_str = str(getattr(tensor, "dtype", "unknown"))
        numel = int(getattr(tensor, "numel", 0) or 0)
        if callable(getattr(tensor, "numel", None)):
            try:
                numel = int(tensor.numel)  # type: ignore[attr-defined]
            except Exception:
                numel = 0
        if hasattr(tensor, "element_size") and callable(getattr(tensor, "element_size", None)):
            try:
                element_size = int(tensor.element_size())  # type: ignore[call-arg]
            except Exception:
                element_size = 0
        else:
            element_size = 0
        return {
            "shape": shape,
            "dtype": dtype_str,
            "numel": numel,
            "params_bytes": int(numel * element_size),
            "storage_bytes": int(numel * element_size),
            "is_strided": False,
            "compression_ratio": 1.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sparsity": 0.0,
            "l2_norm": 0.0,
        }

    try:
        shape = list(tensor.shape)
        numel = tensor.numel()
        dtype_str = str(tensor.dtype)

        # For stride=0 tensors, all logical elements share the same underlying value.
        is_strided = any(s == 0 for s in tensor.stride())

        if numel == 0:
            return {
                "shape": shape,
                "dtype": dtype_str,
                "numel": 0,
                "params_bytes": 0,
                "is_strided": is_strided,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sparsity": 1.0,
                "l2_norm": 0.0,
            }

        # Sample to avoid OOM: if numel > sample_size, sample with a fixed seed for reproducibility.
        flat = tensor.flatten().float()
        if sample_size <= 0:
            sample_size = 1_000_000

        if numel > sample_size:
            # Note: randperm(numel) can be expensive for very large tensors, so we use randint sampling (with replacement).
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            indices = torch.randint(0, numel, (int(sample_size),), generator=gen)
            sample = flat[indices].cpu().numpy()
        else:
            sample = flat.cpu().numpy()

        import numpy as np
        sample_np = np.asarray(sample)

        mean_val = float(np.mean(sample_np))
        std_val = float(np.std(sample_np))
        min_val = float(np.min(sample_np))
        max_val = float(np.max(sample_np))
        l2_norm = float(np.linalg.norm(sample_np))
        sparsity = float(1.0 - np.count_nonzero(sample_np) / len(sample_np))

        # Compute the true storage footprint.
        try:
            storage_bytes = int(tensor.untyped_storage().nbytes())
        except Exception:
            storage_bytes = int(numel * tensor.element_size())

        return {
            "shape": shape,
            "dtype": dtype_str,
            "numel": numel,
            "params_bytes": numel * tensor.element_size(),
            "storage_bytes": storage_bytes,
            "is_strided": is_strided,
            "compression_ratio": numel * tensor.element_size() / max(storage_bytes, 1),
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "sparsity": sparsity,
            "l2_norm": l2_norm,
        }
    except Exception as e:
        logger.debug("Stats computation failed: %s", e)
        return {
            "shape": list(tensor.shape) if hasattr(tensor, "shape") else [],
            "error": str(e),
        }


def inspect_weights(
    model_dir: str,
    pattern: Optional[str] = None,
    limit: Optional[int] = None,
    max_per_layer_sample: int = 1_000_000,
) -> Dict[str, Any]:
    """Inspect model weights and extract detailed statistics.

    Reads .safetensors/.bin shards produced by Ultra/HybridUltra and combines them
    with meta-config.json to produce visualization-friendly stats.

    Args:
        model_dir: Model output directory (weights + meta-config.json).
        pattern: Optional regex filter; only keep matching parameter names.
        limit: Max number of tensors to load (None = no limit).
        max_per_layer_sample: Max sampled elements per tensor to avoid OOM.

    Returns:
        {
            "model_dir": str,
            "config": dict,       # meta-config.json contents
            "total_shards": int,
            "total_tensors": int,
            "layers": [
                {
                    "name": str,
                    "shape": [int, ...],
                    "dtype": str,
                    "numel": int,
                    "params_bytes": int,
                    "storage_bytes": int,
                    "is_strided": bool,
                    "compression_ratio": float,
                    "mean": float,
                    "std": float,
                    "min": float,
                    "max": float,
                    "sparsity": float,
                    "l2_norm": float,
                    "layer_type": str,   # embedding/attention/ffn/norm/output
                },
                ...
            ],
            "summary": {
                "total_params": int,
                "total_storage_bytes": int,
                "total_logical_bytes": int,
                "overall_compression_ratio": float,
                "overall_sparsity": float,
                "strategy_hint": str,  # "ultra" / "hybrid_ultra" / "unknown"
            }
        }
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error("Model directory not found: %s", model_dir)
        return {"model_dir": model_dir, "error": "Directory not found", "layers": []}

    # 1) Load meta-config
    effective_config = load_effective_config(model_dir)

    # 2) List weight files
    weight_files, is_safetensors = _list_weight_files(model_path)
    if not weight_files:
        logger.warning("No weight files found in %s", model_dir)
        return {
            "model_dir": model_dir,
            "config": effective_config,
            "total_shards": 0,
            "total_tensors": 0,
            "layers": [],
            "summary": {},
        }

    # 3) Load and compute stats
    regex = re.compile(pattern) if pattern else None
    all_stats: List[Dict[str, Any]] = []
    loaded_count = 0
    has_strided = False
    metadata_fallback = _scan_safetensors_metadata(weight_files) if is_safetensors else {}

    for wf in weight_files:
        if is_safetensors and (_safetensors_load is None or torch is None):
            for name, meta in _read_safetensors_metadata(wf).items():
                if name.startswith("__vitriol_pad__"):
                    continue
                if regex and not regex.search(name):
                    continue
                all_stats.append(_metadata_stats(name, meta, wf.name))
                loaded_count += 1
                if limit and loaded_count >= limit:
                    break
            if limit and loaded_count >= limit:
                break
            continue

        try:
            shard = _load_shard(wf, is_safetensors)
        except Exception as e:
            logger.warning("Failed to load %s: %s", wf.name, e)
            continue

        for name, tensor in shard.items():
            # Skip internal padding tensors.
            if name.startswith("__vitriol_pad__"):
                continue

            if regex and not regex.search(name):
                continue

            stats = _compute_tensor_stats(tensor)
            stats["name"] = name
            stats["layer_type"] = _classify_layer(name)
            stats["shard_file"] = wf.name

            if stats.get("is_strided"):
                has_strided = True

            all_stats.append(stats)
            loaded_count += 1

            if limit and loaded_count >= limit:
                break

        if not shard and is_safetensors:
            for name, meta in metadata_fallback.items():
                if meta.get("shard_file") != wf.name:
                    continue
                if name.startswith("__vitriol_pad__"):
                    continue
                if regex and not regex.search(name):
                    continue
                all_stats.append(_metadata_stats(name, meta, wf.name))
                loaded_count += 1
                if limit and loaded_count >= limit:
                    break

        if limit and loaded_count >= limit:
            break

    # 4) Aggregate
    total_params = sum(s.get("numel", 0) for s in all_stats)
    total_logical_bytes = sum(s.get("params_bytes", 0) for s in all_stats)
    total_storage_bytes = sum(s.get("storage_bytes", 0) for s in all_stats)

    overall_sparsity = 0.0
    if total_params > 0:
        zero_params = sum(
            int(s.get("sparsity", 0) * s.get("numel", 0))
            for s in all_stats
        )
        overall_sparsity = zero_params / total_params

    strategy_hint = "ultra" if has_strided else (
        "hybrid_ultra" if overall_sparsity > 0.8 and not has_strided else "unknown"
    )

    return {
        "model_dir": str(model_path),
        "config": effective_config,
        "total_shards": len(weight_files),
        "total_tensors": len(all_stats),
        "layers": all_stats,
        "summary": {
            "total_params": total_params,
            "total_storage_bytes": total_storage_bytes,
            "total_logical_bytes": total_logical_bytes,
            "overall_compression_ratio": (
                total_logical_bytes / max(total_storage_bytes, 1)
            ),
            "overall_sparsity": overall_sparsity,
            "strategy_hint": strategy_hint,
            "is_safetensors_format": is_safetensors,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Layer classification
# ──────────────────────────────────────────────────────────────────────

def _classify_layer(name: str) -> str:
    """Infer a layer/component type from a parameter name.

    Args:
        name: Parameter name (e.g. "model.layers.0.self_attn.q_proj.weight").

    Returns:
        Layer type string: embedding, attention_q, attention_k, attention_v,
        attention_o, ffn_gate, ffn_up, ffn_down, norm, output, other
    """
    name_lower = name.lower()

    # Embedding
    if any(k in name_lower for k in ("embed_tokens", "wte", "wpe", "word_embeddings", "embeddings.word", "tok_embeddings")):
        return "embedding"

    # LM Head
    if any(k in name_lower for k in ("lm_head", "embed_out", "output_projection", "output.dense")):
        return "output"

    # Norm
    if any(k in name_lower for k in ("norm.weight", "norm.bias", "layernorm", "rmsnorm", "ln.")):
        return "norm"

    # Attention
    if any(k in name_lower for k in ("q_proj", "query.weight", "query_key_value", "qkv", "c_attn", "w_pack")):
        return "attention_q"
    if any(k in name_lower for k in ("k_proj", "key.weight")):
        return "attention_k"
    if any(k in name_lower for k in ("v_proj", "value.weight")):
        return "attention_v"
    if any(k in name_lower for k in ("o_proj", "out_proj", "dense.weight", "c_proj")) and any(k in name_lower for k in ("attn", "attention", "self_attn")):
        return "attention_o"

    # MLA / DeepSeek
    if any(k in name_lower for k in ("kv_b_proj", "kv_a_layernorm", "k_b_proj", "v_b_proj")):
        return "attention_kv"

    # FFN / MoE
    if any(k in name_lower for k in ("gate_proj", "w1.weight")):
        return "ffn_gate"
    if any(k in name_lower for k in ("up_proj", "w3.weight", "c_fc.weight", "fc1.weight")):
        return "ffn_up"
    if any(k in name_lower for k in ("down_proj", "w2.weight", "c_proj.weight", "fc2.weight")) and any(k in name_lower for k in ("mlp", "ffn", "feed_forward")):
        return "ffn_down"

    if any(k in name_lower for k in ("mixer", "ssm", "in_proj", "x_proj", "dt_proj", "time_mix")):
        return "sequence_mixer"

    if any(k in name_lower for k in ("vision", "visual", "patch_embed", "conv1")):
        return "vision"

    # MoE gate
    if "gate" in name_lower and "proj" not in name_lower:
        return "moe_gate"

    # MoE expert
    if "experts" in name_lower or "expert" in name_lower:
        return "moe_expert"

    # Shared expert
    if "shared_expert" in name_lower:
        return "moe_shared"

    return "other"


_LAYER_INDEX_PATTERNS = (
    re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:transformer\.)?h\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)encoder\.layer\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:blocks|block)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:gpt_neox\.)?layers\.(\d+)(?:\.|$)"),
)


def _layer_index_from_name(name: str) -> Optional[int]:
    for pattern in _LAYER_INDEX_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _find_first_tensor(tensor_shapes: Dict[str, Tuple[int, ...]], layer_type: str) -> Optional[str]:
    for name in tensor_shapes:
        if _classify_layer(name) == layer_type:
            return name
    return None


def _params_from_shape(shape: Tuple[int, ...]) -> int:
    return _numel_from_shape(shape)


def _selected_layer_indices(layer_map: Dict[int, List[str]], num_layers: int, max_layers: int) -> List[int]:
    if layer_map:
        indices = sorted(layer_map)
        if len(indices) <= max_layers:
            return indices
        step = max(1, len(indices) // max_layers)
        selected = indices[::step][:max_layers]
        if indices[-1] not in selected:
            selected[-1] = indices[-1]
        return selected
    return list(range(min(num_layers, max_layers)))


def _select_representative_tensor_names(tensor_shapes: Dict[str, Tuple[int, ...]], num_layers: int, max_layers: int) -> set[str]:
    if not tensor_shapes:
        return set()
    layer_map: Dict[int, List[str]] = {}
    selected: set[str] = set()
    for name in tensor_shapes:
        layer_type = _classify_layer(name)
        if layer_type in {"embedding", "output"}:
            selected.add(name)
        idx = _layer_index_from_name(name)
        if idx is not None:
            layer_map.setdefault(idx, []).append(name)

    for idx in _selected_layer_indices(layer_map, num_layers, max_layers):
        ranked = sorted(
            layer_map.get(idx, []),
            key=lambda name: (
                {
                    "norm": 0,
                    "attention_q": 1,
                    "attention_k": 2,
                    "attention_v": 3,
                    "attention_o": 4,
                    "attention_kv": 5,
                    "sequence_mixer": 6,
                    "ffn_gate": 7,
                    "ffn_up": 8,
                    "ffn_down": 9,
                    "moe_gate": 10,
                    "moe_expert": 11,
                    "moe_shared": 12,
                }.get(_classify_layer(name), 99),
                name,
            ),
        )
        selected.update(ranked[:12])
    return selected


# ──────────────────────────────────────────────────────────────────────
# Visualization payload generation
# ──────────────────────────────────────────────────────────────────────

def generate_viz_data(
    model_dir: str,
    max_layers: int = 64,
    *,
    seed: int = 42,
    sample_size: int = 1_000_000,
) -> Dict[str, Any]:
    """Generate a compact payload for the 3D visualizer.

    Compared to inspect_weights(), this output is frontend-oriented:
    - Aggregated per Transformer block
    - Uses meta-config/analyzer for more accurate model-level params when available
    - Bounded output size via max_layers

    Args:
        model_dir: Model output directory.
        max_layers: Max Transformer blocks to include.

    Returns:
        Visualization-friendly JSON payload.
    """
    # 1) Load config (prefer meta-config)
    effective_config = load_effective_config(model_dir)

    # Parse text_config (nested for multimodal models)
    text_config = effective_config.get("text_config", effective_config)

    hidden_size = text_config.get("hidden_size", 0)
    num_layers = text_config.get("num_hidden_layers", 0)
    vocab_size = text_config.get("vocab_size", 0)
    intermediate_size = text_config.get("intermediate_size", 0)
    num_attention_heads = text_config.get("num_attention_heads", 0)
    num_key_value_heads = text_config.get("num_key_value_heads", num_attention_heads)
    num_experts = text_config.get("num_experts", text_config.get("n_routed_experts", 0))
    moe_intermediate_size = text_config.get("moe_intermediate_size", intermediate_size)

    model_total_params = 0
    params_source = "config_derived"

    try:
        from vitriol.arch_viz.analyzer import ArchitectureAnalyzer
        analyzer = ArchitectureAnalyzer()
        class MockConfig:
            def __init__(self, d):
                self.__dict__.update(d)
                self.model_type = d.get('model_type', 'unknown')
        arch = analyzer.analyze(MockConfig(effective_config))
        num_layers = arch.total_layers
        hidden_size = arch.parameters.get('hidden_size', hidden_size)
        num_experts = arch.parameters.get('num_experts', num_experts)
        try:
            model_total_params = int(getattr(arch, "total_params", 0) or 0)
        except Exception:
            model_total_params = 0
        if model_total_params > 0:
            params_source = "analyzer"
    except Exception as e:
        logger.debug(f"ArchitectureAnalyzer failed for weight inspector: {e}")

    # 2) Check weight files
    model_path = Path(model_dir)
    weight_files, is_safetensors = _list_weight_files(model_path)

    # 3) Scan weights for shape + stats
    # Use index files to build a param-name -> shard mapping
    weight_map: Dict[str, str] = {}
    for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx_path = model_path / idx_name
        if idx_path.exists():
            try:
                idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
                weight_map = idx_data.get("weight_map", {})
                break
            except Exception:
                pass

    # If no index file is present, scan weight files directly
    tensor_shapes: Dict[str, Tuple[int, ...]] = {}
    tensor_stats: Dict[str, Dict[str, Any]] = {}
    tensor_metadata: Dict[str, Dict[str, Any]] = _scan_safetensors_metadata(weight_files) if is_safetensors else {}
    for name, meta in tensor_metadata.items():
        tensor_shapes[name] = tuple(int(x) for x in (meta.get("shape") or []))

    # Load only the required shards to compute stats for representative tensors.
    # If names are unknown, sample from observed weight names first.
    target_names = _select_representative_tensor_names(tensor_shapes, num_layers, max_layers)
    if not target_names:
        target_names = set()
        target_names.add("model.embed_tokens.weight")
        target_names.add("lm_head.weight")

        # Sample layer indices
        layer_step = max(1, num_layers // min(num_layers, 12)) if num_layers else 1
        for i in range(0, min(num_layers, max_layers), layer_step):
            prefix = f"model.layers.{i}"
            for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                           "mlp.gate_proj.weight", "mlp.down_proj.weight",
                           "input_layernorm.weight", "post_attention_layernorm.weight"):
                target_names.add(f"{prefix}.{suffix}")

    # Resolve which shards to load
    shards_to_load: Dict[str, Path] = {}
    if weight_map:
        for name in target_names:
            shard_name = weight_map.get(name)
            if shard_name:
                shard_path = model_path / shard_name
                if shard_path.exists() and shard_name not in shards_to_load:
                    shards_to_load[shard_name] = shard_path
    else:
        # Without an index file, load only the first two shards (best-effort).
        for wf in weight_files[:2]:
            shards_to_load[wf.name] = wf

    # Load shards and compute stats
    for shard_name, shard_path in shards_to_load.items():
        try:
            shard_data = _load_shard(shard_path, is_safetensors)
            for name, tensor in shard_data.items():
                if name.startswith("__vitriol_pad__"):
                    continue
                tensor_shapes[name] = tuple(tensor.shape)
                # Only compute numeric stats when real tensor values are available (torch tensor).
                # Otherwise, keep stats unavailable.
                if name in target_names and torch is not None and hasattr(tensor, "flatten"):
                    stats = _compute_tensor_stats(tensor, seed=seed, sample_size=sample_size)
                    tensor_stats[name] = stats
        except Exception as e:
            logger.warning("Failed to load shard %s: %s", shard_name, e)

    if not tensor_stats and tensor_metadata:
        for name in target_names:
            meta = tensor_metadata.get(name)
            if meta:
                tensor_stats[name] = _metadata_stats(name, meta, str(meta.get("shard_file", "")))

    # If we couldn't infer any shapes from index/weight files,
    # derive them from meta-config instead.
    if not tensor_shapes:
        logger.info("No weight shapes found — deriving from meta-config.json")
        tensor_shapes = _derive_shapes_from_config(text_config, max_layers)

    # 4) Build visualization payload
    head_dim = hidden_size // max(num_attention_heads, 1) if num_attention_heads and hidden_size else 0

    layers_data = []

    # Embedding
    emb_name = _find_first_tensor(tensor_shapes, "embedding") or "model.embed_tokens.weight"
    emb_shape = tensor_shapes.get(emb_name, (vocab_size, hidden_size))
    emb_stats = tensor_stats.get(emb_name, {})
    layers_data.append({
        "name": emb_name,
        "type": "Embedding",
        "shape": list(emb_shape),
        "params": _params_from_shape(emb_shape),
        "stats": emb_stats,
    })

    # Transformer Blocks
    layer_map: Dict[int, List[str]] = {}
    for name in tensor_shapes:
        idx = _layer_index_from_name(name)
        if idx is not None:
            layer_map.setdefault(idx, []).append(name)

    selected_indices = _selected_layer_indices(layer_map, num_layers, max_layers)
    display_layers = len(selected_indices) if selected_indices else min(num_layers, max_layers)
    for i in selected_indices:
        block_data = {
            "block_index": i,
            "sub_layers": [],
        }

        actual_names = sorted(
            layer_map.get(i, []),
            key=lambda name: (
                {
                    "norm": 0,
                    "attention_q": 1,
                    "attention_k": 2,
                    "attention_v": 3,
                    "attention_o": 4,
                    "attention_kv": 5,
                    "sequence_mixer": 6,
                    "ffn_gate": 7,
                    "ffn_up": 8,
                    "ffn_down": 9,
                    "moe_gate": 10,
                    "moe_expert": 11,
                    "moe_shared": 12,
                }.get(_classify_layer(name), 99),
                name,
            ),
        )

        if actual_names:
            for name in actual_names[:16]:
                shape = tensor_shapes.get(name, ())
                block_data["sub_layers"].append({
                    "name": name,
                    "type": _classify_layer(name),
                    "shape": list(shape),
                    "params": _params_from_shape(shape),
                    "stats": tensor_stats.get(name, {}),
                })
        else:
            # Attention fallback for LLaMA-like configs when no weight metadata is available.
            for proj, ltype in [("q_proj", "Linear"), ("k_proj", "Linear"),
                                ("v_proj", "Linear"), ("o_proj", "Linear")]:
                key = f"model.layers.{i}.self_attn.{proj}.weight"
                shape = tensor_shapes.get(key, (hidden_size, hidden_size))
                stats = tensor_stats.get(key, {})
                block_data["sub_layers"].append({
                    "name": f"layers.{i}.self_attn.{proj}",
                    "type": ltype,
                    "shape": list(shape),
                    "params": _params_from_shape(shape),
                    "stats": stats,
                })

            # FFN fallback
            for proj, ltype in [("gate_proj", "Linear"), ("up_proj", "Linear"),
                                ("down_proj", "Linear")]:
                key = f"model.layers.{i}.mlp.{proj}.weight"
                shape = tensor_shapes.get(key, (intermediate_size, hidden_size))
                stats = tensor_stats.get(key, {})
                block_data["sub_layers"].append({
                    "name": f"layers.{i}.mlp.{proj}",
                    "type": ltype,
                    "shape": list(shape),
                    "params": _params_from_shape(shape),
                    "stats": stats,
                })

        layers_data.append(block_data)

    # LM Head
    lm_name = _find_first_tensor(tensor_shapes, "output") or "lm_head.weight"
    lm_shape = tensor_shapes.get(lm_name, (vocab_size, hidden_size))
    lm_stats = tensor_stats.get(lm_name, {})
    layers_data.append({
        "name": lm_name,
        "type": "Linear",
        "shape": list(lm_shape),
        "params": _params_from_shape(lm_shape),
        "stats": lm_stats,
    })

    # 5) Compute total parameter counts
    display_params_estimate = sum(
        layer.get("params", 0) if "sub_layers" not in layer
        else sum(sl.get("params", 0) for sl in layer.get("sub_layers", []))
        for layer in layers_data
    )

    total_params_compat = model_total_params or display_params_estimate

    return {
        "model_name": effective_config.get("model_type", Path(model_dir).name),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "display_layers": display_layers,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "num_experts": num_experts,
        "moe_intermediate_size": moe_intermediate_size,
        # Compatibility: prefer the model-level total params when available,
        # otherwise fall back to the displayed-layer estimate.
        "total_params": total_params_compat,
        # P0: explicitly distinguish model_total_params vs display_params_estimate.
        "model_total_params": model_total_params,
        "display_params_estimate": display_params_estimate,
        # P0: provenance metadata
        "params_source": params_source,
        "sampling": {
            "enabled": True,
            "method": "uniform_random",
            "sample_size": int(sample_size),
            "seed": int(seed),
        },
        "config_source": "meta-config.json" if (model_path / "meta-config.json").exists() else "config.json",
        "weight_stats_available": bool(tensor_stats),
        "layers": layers_data,
        "raw_config": effective_config,
    }


def _derive_shapes_from_config(text_config: Dict[str, Any], max_layers: int) -> Dict[str, Tuple[int, ...]]:
    """Derive tensor shapes from config when weight files are unavailable.

    Args:
        text_config: text_config dict.
        max_layers: Max number of layers.

    Returns:
        Mapping from parameter name -> shape.
    """
    hidden = text_config.get("hidden_size", 0)
    vocab = text_config.get("vocab_size", 0)
    inter = text_config.get("intermediate_size", 0)
    num_heads = text_config.get("num_attention_heads", 0)
    num_kv_heads = text_config.get("num_key_value_heads", num_heads)
    head_dim = hidden // max(num_heads, 1) if num_heads and hidden else 0
    n_layers = min(text_config.get("num_hidden_layers", 0), max_layers)

    shapes: Dict[str, Tuple[int, ...]] = {}
    shapes["model.embed_tokens.weight"] = (vocab, hidden)

    for i in range(n_layers):
        shapes[f"model.layers.{i}.self_attn.q_proj.weight"] = (num_heads * head_dim, hidden)
        shapes[f"model.layers.{i}.self_attn.k_proj.weight"] = (num_kv_heads * head_dim, hidden)
        shapes[f"model.layers.{i}.self_attn.v_proj.weight"] = (num_kv_heads * head_dim, hidden)
        shapes[f"model.layers.{i}.self_attn.o_proj.weight"] = (hidden, num_heads * head_dim)
        shapes[f"model.layers.{i}.mlp.gate_proj.weight"] = (inter, hidden)
        shapes[f"model.layers.{i}.mlp.up_proj.weight"] = (inter, hidden)
        shapes[f"model.layers.{i}.mlp.down_proj.weight"] = (hidden, inter)

    shapes["lm_head.weight"] = (vocab, hidden)
    return shapes
