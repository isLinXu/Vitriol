from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def cfg_attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def cfg_int(obj: Any, key: str, default: int = 0) -> int:
    try:
        value = cfg_attr(obj, key, default)
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def cfg_list(obj: Any, key: str) -> list[Any]:
    value = cfg_attr(obj, key, None)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def config_model_tokens(config: Any) -> set[str]:
    text_cfg = cfg_attr(config, "text_config", None)
    tokens = {
        str(cfg_attr(config, "model_type", "") or "").lower(),
        str(cfg_attr(text_cfg, "model_type", "") or "").lower(),
    }
    for source in (config, text_cfg):
        for arch in cfg_list(source, "architectures"):
            tokens.add(str(arch or "").lower())
    return {token for token in tokens if token}


def normalize_kv_layer_type(name: Any) -> str:
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


def explicit_layer_types(config: Any) -> list[str]:
    text_cfg = cfg_attr(config, "text_config", None)
    keys = (
        "layer_types",
        "layers_block_type",
        "layer_block_types",
        "block_types",
        "attention_types",
        "attn_types",
    )
    for source in (config, text_cfg):
        for key in keys:
            layer_types = cfg_list(source, key)
            if layer_types:
                return [normalize_kv_layer_type(item) for item in layer_types]
    return []


def infer_num_layers(config: Any) -> int:
    text_cfg = cfg_attr(config, "text_config", None)
    for source in (config, text_cfg):
        for key in ("num_hidden_layers", "n_layer", "n_layers", "num_layers", "decoder_layers"):
            value = cfg_int(source, key, 0)
            if value > 0:
                return value
    return 0


def infer_kv_layer_types(config: Any) -> list[str]:
    text_cfg = cfg_attr(config, "text_config", None)
    num_layers = infer_num_layers(config)
    layer_types = explicit_layer_types(config)
    if layer_types:
        if num_layers > len(layer_types):
            return layer_types + ["full_attention"] * (num_layers - len(layer_types))
        return layer_types[:num_layers] if num_layers > 0 else layer_types

    if num_layers <= 0:
        return []

    tokens = config_model_tokens(config)
    no_kv_tokens = ("mamba", "rwkv", "retnet", "hyena", "state_space", "ssm")
    if any(any(token in model_token for token in no_kv_tokens) for model_token in tokens):
        return ["linear_attention"] * num_layers

    is_deepseek_v4 = "deepseek_v4" in tokens or any("deepseekv4" in token for token in tokens)
    if is_deepseek_v4:
        compress_ratios = []
        for item in cfg_list(config, "compress_ratios") or cfg_list(text_cfg, "compress_ratios"):
            try:
                compress_ratios.append(int(item or 0))
            except (TypeError, ValueError):
                compress_ratios.append(0)
        num_hash_layers = cfg_int(config, "num_hash_layers", cfg_int(text_cfg, "num_hash_layers", 0))
        sliding_window = cfg_int(config, "sliding_window", cfg_int(text_cfg, "sliding_window", 0))
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

    sliding_window = cfg_int(config, "sliding_window", cfg_int(text_cfg, "sliding_window", 0))
    use_sliding_window = bool(cfg_attr(config, "use_sliding_window", cfg_attr(text_cfg, "use_sliding_window", False)))
    token_blob = " ".join(tokens)
    if sliding_window and (use_sliding_window or any(token in token_blob for token in ("mistral", "mixtral", "gemma", "cohere", "longformer"))):
        return ["sliding_window"] * num_layers

    return ["full_attention"] * num_layers


@dataclass(frozen=True)
class ModelCapabilities:
    """Capability descriptor for supported model features."""
    model_type: str
    architecture_kind: str
    supports_kv_cache: bool
    layer_types: list[str]
    reason: str


def infer_model_capabilities(config: Any) -> ModelCapabilities:
    config_model_tokens(config)
    model_type = str(cfg_attr(config, "model_type", "") or "unknown")
    layer_types = infer_kv_layer_types(config)
    no_kv = bool(layer_types) and all(item == "linear_attention" for item in layer_types)
    if no_kv:
        return ModelCapabilities(
            model_type=model_type,
            architecture_kind="sequence_mixer",
            supports_kv_cache=False,
            layer_types=layer_types,
            reason="state-space/recurrent families do not expose standard attention KV cache",
        )
    if any(item in {"compressed_attention", "hash_attention", "mla", "sliding_window"} for item in layer_types):
        return ModelCapabilities(
            model_type=model_type,
            architecture_kind="hybrid_attention",
            supports_kv_cache=True,
            layer_types=layer_types,
            reason="config exposes non-standard attention layer types",
        )
    return ModelCapabilities(
        model_type=model_type,
        architecture_kind="transformer_attention" if layer_types else "unknown",
        supports_kv_cache=bool(layer_types),
        layer_types=layer_types,
        reason="default full-attention compatibility path",
    )
