"""KV cache quantization analysis — extracted from runner.py.

Provides :func:`analyze_kv_quantization` which computes per-layer MSE,
cosine similarity, and residual gain for KV cache compression presets.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from ..kv.policy import KVLayerType, resolve_layer_strategy
from ..patches.turboquant import turbo_quantize

logger = logging.getLogger(__name__)


def _tensor_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean squared error between two tensors."""
    return float(torch.mean((a - b) ** 2).item())


def _tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    af = a.reshape(-1).to(torch.float32)
    bf = b.reshape(-1).to(torch.float32)
    denom = torch.linalg.norm(af) * torch.linalg.norm(bf)
    if float(denom.item()) <= 0.0:
        return 1.0
    return float(torch.dot(af, bf).item() / denom.item())


def _proxy_attention_metrics(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_q: torch.Tensor, value_q: torch.Tensor) -> dict[str, float]:
    """Compute attention metrics (MSE, cosine) between full-precision and quantized KV.

    Uses a single query vector (last token) as a proxy for the full attention computation.
    """
    scale = 1.0 / (query.shape[-1] ** 0.5)
    logits = torch.matmul(query, key.transpose(-2, -1)) * scale
    logits_q = torch.matmul(query, key_q.transpose(-2, -1)) * scale
    attn = torch.softmax(logits, dim=-1)
    attn_q = torch.softmax(logits_q, dim=-1)
    out = torch.matmul(attn, value)
    out_q = torch.matmul(attn_q, value_q)
    return {
        "logits_mse": _tensor_mse(logits, logits_q),
        "logits_cosine": _tensor_cosine(logits, logits_q),
        "output_mse": _tensor_mse(out, out_q),
        "output_cosine": _tensor_cosine(out, out_q),
    }


def _residual_gain(original: torch.Tensor, with_residual: torch.Tensor, without_residual: torch.Tensor) -> float:
    """Compute residual gain: how much the residual correction improves MSE.

    Returns (MSE_without - MSE_with) / MSE_without.
    """
    mse_without = _tensor_mse(original, without_residual)
    mse_with = _tensor_mse(original, with_residual)
    if mse_without <= 0:
        return 0.0
    return float((mse_without - mse_with) / mse_without)


def analyze_kv_quantization(
    model_id: str,
    prompt_tokens: int,
    preset: str = "balanced",
    compare_preset: str | None = None,
    preset_params: Dict[str, Any] | None = None,
    compare_preset_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Analyze per-layer KV quantization quality for a model.

    Loads the model, prefills a prompt, and computes MSE/cosine/residual gain
    for each layer under the specified preset(s).
    """
    # Local imports to avoid circular deps and keep this module lightweight
    from ._planning import (
        _collect_policy_insights,
        _infer_kv_layer_types,
        _preset_to_kv_cfg,
        _select_preset,
        _LayerTypeHandle,
    )
    from .runner import (
        build_long_prompt,
        select_device,
        _extract_dynamic_cache_kv,
        _prefill_cache,
    )
    from ..utils.hf_loading import load_causallm as hf_load_causallm
    from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

    device = select_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    tokenizer = hf_load_tokenizer(
        model_id,
        security={"trust_remote_code": False, "allow_network": True, "local_files_only": False},
    )
    model = hf_load_causallm(
        model_id,
        security={"trust_remote_code": False, "allow_network": True, "local_files_only": False},
        torch_dtype=dtype,
        device=device,
    )
    inferred_layer_types = _infer_kv_layer_types(model.config)

    prompt = build_long_prompt(tokenizer, min_tokens=int(prompt_tokens))
    past = _prefill_cache(model, tokenizer, prompt, device)
    kv_layers = _extract_dynamic_cache_kv(past)

    def build_result(preset_name: str, params: Dict[str, Any] | None) -> Dict[str, Any]:
        preset_obj = _select_preset(str(preset_name), dict(params or {}))
        _, chosen_n, policy = _preset_to_kv_cfg(preset_obj)
        insights = _collect_policy_insights(model.config, policy, int(chosen_n))
        policy_handle = past
        if inferred_layer_types and getattr(past, "layer_types", None) is None:
            policy_handle = _LayerTypeHandle(inferred_layer_types)
        layer_rows = []
        for item in kv_layers:
            layer_idx = int(item["layer_idx"])
            strategy = resolve_layer_strategy(policy, policy_handle, layer_idx)
            key = item["key"].to(torch.float32)
            value = item["value"].to(torch.float32)
            key_q = key
            value_q = value
            k_gain = 0.0
            v_gain = 0.0
            if strategy.turbo_quantize_k:
                key_q = turbo_quantize(
                    key,
                    format_type=getattr(policy, "turbo_k_format", "turbo3"),
                    block_size=int(getattr(policy, "turbo_block_size", 32)),
                    use_residual_qjl=bool(getattr(policy, "enable_turbo_residual_qjl", True)),
                    residual_strength=float(getattr(policy, "turbo_residual_strength", 0.5)),
                )
                key_q_no_residual = turbo_quantize(
                    key,
                    format_type=getattr(policy, "turbo_k_format", "turbo3"),
                    block_size=int(getattr(policy, "turbo_block_size", 32)),
                    use_residual_qjl=False,
                    residual_strength=float(getattr(policy, "turbo_residual_strength", 0.5)),
                )
                k_gain = _residual_gain(key, key_q, key_q_no_residual)
            if strategy.turbo_quantize_v:
                value_q = turbo_quantize(
                    value,
                    format_type=getattr(policy, "turbo_v_format", "turbo3"),
                    block_size=int(getattr(policy, "turbo_block_size", 32)),
                    use_residual_qjl=bool(getattr(policy, "enable_turbo_residual_qjl", True)),
                    residual_strength=float(getattr(policy, "turbo_residual_strength", 0.5)),
                )
                value_q_no_residual = turbo_quantize(
                    value,
                    format_type=getattr(policy, "turbo_v_format", "turbo3"),
                    block_size=int(getattr(policy, "turbo_block_size", 32)),
                    use_residual_qjl=False,
                    residual_strength=float(getattr(policy, "turbo_residual_strength", 0.5)),
                )
                v_gain = _residual_gain(value, value_q, value_q_no_residual)

            query = key[..., -1:, :]
            proxy = _proxy_attention_metrics(query, key, value, key_q, value_q)
            layer_rows.append(
                {
                    "layer_idx": layer_idx,
                    "layer_type": strategy.layer_type.value if isinstance(strategy.layer_type, KVLayerType) else str(strategy.layer_type),
                    "turbo_quantize_k": bool(strategy.turbo_quantize_k),
                    "turbo_quantize_v": bool(strategy.turbo_quantize_v),
                    "key_mse": _tensor_mse(key, key_q),
                    "value_mse": _tensor_mse(value, value_q),
                    "key_cosine": _tensor_cosine(key, key_q),
                    "value_cosine": _tensor_cosine(value, value_q),
                    "residual_gain_k": k_gain,
                    "residual_gain_v": v_gain,
                    **proxy,
                }
            )

        quantized_layers = [row for row in layer_rows if row["turbo_quantize_k"] or row["turbo_quantize_v"]]
        def avg(field: str) -> float:
            rows = quantized_layers or layer_rows
            if not rows:
                return 0.0
            return float(sum(float(row.get(field, 0.0)) for row in rows) / len(rows))

        return {
            "preset": preset_obj.to_dict(),
            "chosen_v_quantize_only_first_n": int(chosen_n),
            "policy_insights": insights,
            "summary": {
                "quantized_layers": len(quantized_layers),
                "avg_key_mse": avg("key_mse"),
                "avg_value_mse": avg("value_mse"),
                "avg_key_cosine": avg("key_cosine"),
                "avg_value_cosine": avg("value_cosine"),
                "avg_logits_mse": avg("logits_mse"),
                "avg_logits_cosine": avg("logits_cosine"),
                "avg_output_mse": avg("output_mse"),
                "avg_output_cosine": avg("output_cosine"),
                "avg_residual_gain_k": avg("residual_gain_k"),
                "avg_residual_gain_v": avg("residual_gain_v"),
            },
            "layers": layer_rows,
        }

    base = build_result(preset, preset_params)
    result = {
        "model_id": model_id,
        "device": device.type,
        "dtype": str(dtype),
        "prompt_tokens": int(prompt_tokens),
        "base": base,
    }
    if compare_preset is not None:
        compare = build_result(compare_preset, compare_preset_params)
        result["compare"] = compare
    return result
