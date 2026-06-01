"""Policy preset selection, plan building/diffing and KV-config helpers.

Leaf module: depends only on kv.* primitives, never on the runner engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..kv.cache_store import KVCacheStoreConfig
from ..kv.policy import (
    KVLayerType,
    KVPolicyPreset,
    Turbo3ExactKApproxVPolicy,
    build_policy,
    resolve_layer_strategy,
)
from ..utils.hf_loading import load_config as hf_load_config


def _plan_from_suite_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract plan metadata from a suite result for diffing."""
    return {
        "model_id": result.get("model_id"),
        "preset": result.get("preset"),
        "chosen_v_quantize_only_first_n": result.get("chosen_v_quantize_only_first_n"),
        "policy_insights": result.get("policy_insights") or {},
    }


def _select_preset(name: str, preset_params: Dict[str, Any]) -> KVPolicyPreset:
    """Select a KV policy preset by name, applying optional parameter overrides.

    Args:
        name: Preset name (safe, balanced, fast-balanced, etc.).
        preset_params: Dict of parameter overrides.
    """
    normalized = str(name).replace("_", "-")
    if normalized == "safe":
        return KVPolicyPreset.safe_default()
    if normalized == "fast-balanced":
        base = KVPolicyPreset.fast_balanced_default()
    elif normalized == "ultra-long":
        base = KVPolicyPreset.ultra_long_default()
    elif normalized == "deepseek-v4":
        base = KVPolicyPreset.deepseek_v4_default()
    elif normalized == "hy3":
        base = KVPolicyPreset.hy3_default()
    elif normalized == "aggressive":
        base = KVPolicyPreset.aggressive_default()
    else:
        base = KVPolicyPreset.balanced_default()

    if not preset_params:
        return base

    merged = dict(base.params)
    merged.update({k: v for k, v in preset_params.items() if v is not None})
    return KVPolicyPreset(name=base.name, policy_type=base.policy_type, params=merged)


def _search_max_passing_n(max_n: int, is_ok) -> int:
    """Binary search for the maximum n where is_ok(n) is True.

    Used to find the optimal number of full-attention layers to keep
    before switching to V-only quantization.
    """
    max_n = int(max_n)
    if max_n <= 0:
        return 0

    if not is_ok(1):
        return 0

    lo = 1
    hi = 2
    while hi <= max_n and is_ok(hi):
        lo = hi
        hi *= 2

    if hi > max_n and is_ok(max_n):
        return max_n

    left = lo + 1
    right = min(hi - 1, max_n)
    best = lo
    while left <= right:
        mid = (left + right) // 2
        if is_ok(mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1

    if not is_ok(best):
        best = 0

    if best < max_n and is_ok(best + 1):
        best = 0
        for n in range(1, max_n + 1):
            if is_ok(n):
                best = n
            else:
                break

    return int(best)


def _runtime_flags_for_preset(preset_name: str, chosen_n: int | None) -> tuple[bool, bool]:
    """Determine runtime flags (exact_match, use_tuned) from preset and chosen_n."""
    name = str(preset_name)
    n = int(chosen_n or 0)
    if name == "safe":
        return True, False
    if n <= 0:
        return True, False
    return False, True


def _policy_with_chosen_n(policy: Any, chosen_n: int) -> Any:
    """Create a modified policy with the chosen V-only quantization layer count."""
    if not isinstance(policy, Turbo3ExactKApproxVPolicy):
        return policy
    return Turbo3ExactKApproxVPolicy(
        v_quantize_only_first_n_full_attention_layers=int(chosen_n),
        v_protect_last_n_full_attention_layers=int(policy.v_protect_last_n_full_attention_layers),
        turbo_k_format=str(policy.turbo_k_format),
        turbo_v_format=str(policy.turbo_v_format),
        turbo_block_size=int(policy.turbo_block_size),
        quantized_kv_start=int(policy.quantized_kv_start),
        enable_turbo_residual_qjl=bool(policy.enable_turbo_residual_qjl),
        turbo_residual_strength=float(policy.turbo_residual_strength),
        enable_sparse_v=bool(policy.enable_sparse_v),
        sparse_v_only_first_n_full_attention_layers=int(policy.sparse_v_only_first_n_full_attention_layers),
        enable_compute_skip=bool(policy.enable_compute_skip),
        compute_skip_only_first_n_full_attention_layers=int(policy.compute_skip_only_first_n_full_attention_layers),
    )


def _cfg_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get attribute or dict key from an object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg_int(obj: Any, key: str, default: int = 0) -> int:
    """Safely get integer attribute or dict key from an object."""
    try:
        return int(_cfg_attr(obj, key, default) or 0)
    except (TypeError, ValueError):
        return int(default)


def _cfg_list(obj: Any, key: str) -> list[Any]:
    """Safely get list attribute or dict key from an object."""
    value = _cfg_attr(obj, key, None)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _config_model_tokens(config: Any) -> set[str]:
    """Extract model type tokens from a HuggingFace config for architecture inference."""
    text_cfg = _cfg_attr(config, "text_config", None)
    tokens = {
        str(_cfg_attr(config, "model_type", "") or "").lower(),
        str(_cfg_attr(text_cfg, "model_type", "") or "").lower(),
    }
    for source in (config, text_cfg):
        for arch in _cfg_list(source, "architectures"):
            tokens.add(str(arch or "").lower())
    return {token for token in tokens if token}


def _explicit_layer_types(config: Any) -> list[str]:
    """Read explicit layer types from config if available."""
    text_cfg = _cfg_attr(config, "text_config", None)
    keys = (
        "layer_types",
        "layers_block_type",
        "layer_block_types",
        "block_types",
        "attention_types",
        "attn_types",
    )
    for source in (config, text_cfg):
        layer_types = []
        for key in keys:
            layer_types = _cfg_list(source, key)
            if layer_types:
                break
        if layer_types:
            return [_normalize_kv_layer_type(item) for item in layer_types]
    return []


def _normalize_kv_layer_type(name: Any) -> str:
    """Normalize a layer type name to a standard KVLayerType string."""
    normalized = str(name or "").lower().replace("-", "_")
    if not normalized:
        return "full_attention"
    if any(token in normalized for token in ("mamba", "ssm", "state_space", "rwkv", "retnet", "hyena")):
        return "linear_attention"
    if "sliding" in normalized or "local_attention" in normalized or normalized in {"local", "window_attention"}:
        return "sliding_window"
    if "mla" in normalized or "latent" in normalized:
        return "mla"
    if "hash" in normalized:
        return "hash_attention"
    if "compressed" in normalized or "csa" in normalized or "hca" in normalized:
        return "compressed_attention"
    if "linear" in normalized:
        return "linear_attention"
    if normalized in {"attention", "self_attention", "full", "global", "global_attention", "mha", "gqa", "mqa"}:
        return "full_attention"
    if "full" in normalized or "global" in normalized:
        return "full_attention"
    return "other"


def _infer_kv_layer_types(config: Any) -> list[str]:
    """Infer KV layer types from model config when not explicitly specified.

    Supports DeepSeek-V4, Mamba, Mistral, and other architectures.
    """
    text_cfg = _cfg_attr(config, "text_config", None)
    num_layers = _cfg_int(config, "num_hidden_layers", _cfg_int(text_cfg, "num_hidden_layers", 0))
    layer_types = _explicit_layer_types(config)
    if layer_types:
        if num_layers > len(layer_types):
            return layer_types + ["full_attention"] * (num_layers - len(layer_types))
        return layer_types[:num_layers] if num_layers > 0 else layer_types

    if num_layers <= 0:
        return []

    tokens = _config_model_tokens(config)
    no_kv_tokens = ("mamba", "rwkv", "retnet", "hyena", "state_space", "ssm")
    if any(any(token in model_token for token in no_kv_tokens) for model_token in tokens):
        return ["linear_attention"] * num_layers

    is_deepseek_v4 = "deepseek_v4" in tokens or any("deepseekv4" in token for token in tokens)
    if is_deepseek_v4:
        compress_ratios = []
        for item in _cfg_list(config, "compress_ratios") or _cfg_list(text_cfg, "compress_ratios"):
            try:
                compress_ratios.append(int(item or 0))
            except (TypeError, ValueError):
                compress_ratios.append(0)
        num_hash_layers = _cfg_int(config, "num_hash_layers", _cfg_int(text_cfg, "num_hash_layers", 0))
        sliding_window = _cfg_int(config, "sliding_window", _cfg_int(text_cfg, "sliding_window", 0))
        inferred = []
        for idx in range(num_layers):
            ratio = compress_ratios[idx] if idx < len(compress_ratios) else 0
            if idx < num_hash_layers:
                inferred.append("hash_attention")
            elif ratio > 0:
                inferred.append("compressed_attention")
            elif sliding_window:
                inferred.append("sliding_window")
            else:
                inferred.append("full_attention")
        return inferred

    sliding_window = _cfg_int(config, "sliding_window", _cfg_int(text_cfg, "sliding_window", 0))
    use_sliding_window = bool(_cfg_attr(config, "use_sliding_window", _cfg_attr(text_cfg, "use_sliding_window", False)))
    if sliding_window and (use_sliding_window or any(token in " ".join(tokens) for token in ("mistral", "mixtral", "gemma", "cohere", "longformer"))):
        return ["sliding_window"] * num_layers

    return ["full_attention"] * num_layers


def _collect_policy_insights(config: Any, policy: Any, chosen_n: int) -> Dict[str, Any]:
    """Collect per-layer policy statistics for reporting.

    Returns counts of each layer type and optimization flag.
    """
    effective_policy = _policy_with_chosen_n(policy, chosen_n)
    layer_types = _infer_kv_layer_types(config)
    text_cfg = _cfg_attr(config, "text_config", None)
    num_layers = _cfg_int(config, "num_hidden_layers", _cfg_int(text_cfg, "num_hidden_layers", 0))
    if not layer_types and num_layers > 0:
        layer_types = ["full_attention"] * num_layers
    elif num_layers > len(layer_types):
        layer_types = list(layer_types) + ["full_attention"] * (num_layers - len(layer_types))

    class _Handle:
        """Minimal handle for resolve_layer_strategy."""
        def __init__(self, types: List[str]) -> None:
            self.layer_types = types

        def __len__(self) -> int:
            return len(self.layer_types)

    handle = _Handle(list(layer_types))
    layers = []
    counts = {
        "full_attention": 0,
        "sliding_window": 0,
        "mla": 0,
        "compressed_attention": 0,
        "hash_attention": 0,
        "linear_attention": 0,
        "other": 0,
        "turbo_k": 0,
        "turbo_v": 0,
        "sparse_v": 0,
        "compute_skip": 0,
    }

    for idx in range(len(handle)):
        strategy = resolve_layer_strategy(effective_policy, handle, idx)
        layer_type = str(strategy.layer_type.value if isinstance(strategy.layer_type, KVLayerType) else strategy.layer_type)
        counts[layer_type] = counts.get(layer_type, 0) + 1
        if strategy.turbo_quantize_k:
            counts["turbo_k"] += 1
        if strategy.turbo_quantize_v:
            counts["turbo_v"] += 1
        if strategy.enable_sparse_v:
            counts["sparse_v"] += 1
        if strategy.enable_compute_skip:
            counts["compute_skip"] += 1
        layers.append(
            {
                "layer_idx": idx,
                "layer_type": layer_type,
                "turbo_quantize_k": bool(strategy.turbo_quantize_k),
                "turbo_quantize_v": bool(strategy.turbo_quantize_v),
                "enable_sparse_v": bool(strategy.enable_sparse_v),
                "enable_compute_skip": bool(strategy.enable_compute_skip),
            }
        )

    return {
        "quantized_kv_start": int(getattr(effective_policy, "quantized_kv_start", 0)),
        "layers": layers,
        "counts": counts,
    }


def build_policy_plan(
    model_id: str,
    preset: str = "balanced",
    preset_params: Dict[str, Any] | None = None,
    *,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """Build a KV policy plan for a model without running inference.

    Loads the model config and computes layer-wise policy decisions.
    """
    config = hf_load_config(
        model_id,
        security={
            "trust_remote_code": trust_remote_code,
            "allow_network": True,
            "local_files_only": False,
        },
    )
    preset_obj = _select_preset(str(preset), dict(preset_params or {}))
    _, chosen_n, policy = _preset_to_kv_cfg(preset_obj)
    return {
        "model_id": model_id,
        "preset": preset_obj.to_dict(),
        "chosen_v_quantize_only_first_n": int(chosen_n),
        "policy_insights": _collect_policy_insights(config, policy, int(chosen_n)),
    }


def diff_policy_plans(base: Dict[str, Any], compare: Dict[str, Any]) -> Dict[str, Any]:
    """Diff two KV policy plans and return changed layers.

    Compares layer_type, turbo_quantize_k, turbo_quantize_v,
    enable_sparse_v, and enable_compute_skip fields.
    """
    base_layers = {
        int(layer["layer_idx"]): layer
        for layer in (base.get("policy_insights", {}).get("layers") or [])
    }
    compare_layers = {
        int(layer["layer_idx"]): layer
        for layer in (compare.get("policy_insights", {}).get("layers") or [])
    }
    changed = []
    all_indices = sorted(set(base_layers) | set(compare_layers))
    for idx in all_indices:
        lhs = base_layers.get(idx, {})
        rhs = compare_layers.get(idx, {})
        fields = ["layer_type", "turbo_quantize_k", "turbo_quantize_v", "enable_sparse_v", "enable_compute_skip"]
        diffs = {
            field: {"base": lhs.get(field), "compare": rhs.get(field)}
            for field in fields
            if lhs.get(field) != rhs.get(field)
        }
        if diffs:
            changed.append({"layer_idx": idx, "changes": diffs})

    return {
        "model_id": base.get("model_id") or compare.get("model_id"),
        "base": base,
        "compare": compare,
        "changed_layers": changed,
    }


def _preset_to_kv_cfg(preset: KVPolicyPreset) -> Tuple[KVCacheStoreConfig, int, Any]:
    policy = build_policy(preset)
    if policy.mode.value == "exact":
        return KVCacheStoreConfig(enable_turbo_quant=False), 0, policy

    if not isinstance(policy, Turbo3ExactKApproxVPolicy):
        raise ValueError(f"Unsupported policy_type={preset.policy_type}")

    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        turbo_quantize_k=True,
        turbo_quantize_v=True,
        turbo_k_format=policy.turbo_k_format,
        turbo_v_format=policy.turbo_v_format,
        turbo_block_size=int(policy.turbo_block_size),
        quantized_kv_start=int(policy.quantized_kv_start),
        enable_turbo_residual_qjl=bool(policy.enable_turbo_residual_qjl),
        turbo_residual_strength=float(policy.turbo_residual_strength),
        enable_sparse_v=bool(policy.enable_sparse_v),
        enable_compute_skip=bool(policy.enable_compute_skip),
    )
    return cfg, int(policy.v_quantize_only_first_n_full_attention_layers), policy


class _LayerTypeHandle:
    def __init__(self, layer_types: List[str]) -> None:
        self.layer_types = list(layer_types)

    def __len__(self) -> int:
        return len(self.layer_types)
