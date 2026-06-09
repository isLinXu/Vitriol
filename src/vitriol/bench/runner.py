"""
Benchmark runner for Vitriol KV cache optimization system.

Provides core benchmarking primitives:
  - Smoke tests (short prompts, quick correctness check)
  - Long-context tests (32K+ tokens)
  - Suite tests (multiple prompt lengths)
  - Policy plan building and diffing
  - KV quantization quality analysis

All functions return structured dicts suitable for JSON serialization
and CLI formatting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple

import torch
from transformers.cache_utils import DynamicCache

from ..kv.backend import KVStoreBackend
from ..kv.cache_store import KVCacheStoreConfig
from ..kv.policy import (
    KVLayerType,
    apply_policy_to_store_cfg,
    build_policy,
    classify_kv_layer,
    resolve_layer_strategy,
)
from ..patches.cache_hooks import CacheHookConfig, CacheHookPatcher, UniversalAttentionPatcher, get_cache_hook_stats
from ..patches.turboquant import get_turboquant_stats, reset_turboquant_stats, turbo_quantize
from ..telemetry.run_context import new_run_id
from ..utils.hf_loading import load_causallm as hf_load_causallm
from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

try:
    from ..kv.turboquantum import (
        TurboQuantumConfig,
        compute_attention_entropy,
        create_turboquantum_codec,
        get_turboquantum_presets,
        turboquantum_compress,
    )
    _HAS_TURBOQUANTUM = True
except ImportError:
    _HAS_TURBOQUANTUM = False
    TurboQuantumConfig = None
    turboquantum_compress = None
    compute_attention_entropy = None
    get_turboquantum_presets = None
    create_turboquantum_codec = None
# Re-export the policy-planning leaf cluster so the historical
# `vitriol.bench.runner.<name>` import paths keep working. The `name as name`
# form marks these as intentional re-exports (ruff will not flag them as unused).
from ._planning import _cfg_attr as _cfg_attr
from ._planning import _cfg_int as _cfg_int
from ._planning import _cfg_list as _cfg_list
from ._planning import _collect_policy_insights as _collect_policy_insights
from ._planning import _config_model_tokens as _config_model_tokens
from ._planning import _explicit_layer_types as _explicit_layer_types
from ._planning import _infer_kv_layer_types as _infer_kv_layer_types
from ._planning import _LayerTypeHandle as _LayerTypeHandle
from ._planning import _normalize_kv_layer_type as _normalize_kv_layer_type
from ._planning import _plan_from_suite_result as _plan_from_suite_result
from ._planning import _policy_with_chosen_n as _policy_with_chosen_n
from ._planning import _preset_to_kv_cfg as _preset_to_kv_cfg
from ._planning import _runtime_flags_for_preset as _runtime_flags_for_preset
from ._planning import _search_max_passing_n as _search_max_passing_n
from ._planning import _select_preset as _select_preset
from ._planning import build_policy_plan as build_policy_plan
from ._planning import diff_policy_plans as diff_policy_plans
from .autokv import default_prompt_suite, prefix_match_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a benchmark run.

    Attributes:
        model_id: HuggingFace model identifier.
        prompt_tokens: List of prompt lengths to benchmark.
        max_new_tokens: Number of new tokens to generate.
        calib_new_tokens: Tokens used for calibration/search.
        preset: KV policy preset name.
        search_max_n: Max layers to search for V-only quantization.
        preset_params: Optional overrides for preset parameters.
    """
    model_id: str
    prompt_tokens: List[int]
    max_new_tokens: int
    calib_new_tokens: int
    preset: str
    search_max_n: int
    preset_params: Dict[str, Any] = field(default_factory=dict)


def select_device() -> torch.device:
    """Select the best available torch device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Synchronize the torch device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()


def build_long_prompt(tokenizer, min_tokens: int) -> str:
    """Build a long prompt of at least min_tokens by repeating a base phrase."""
    base = "Vitriol is an advanced AI system. " * 64
    text = ""
    while True:
        text += base
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if int(ids.numel()) >= int(min_tokens):
            return tokenizer.decode(ids[:min_tokens], skip_special_tokens=True)


def _prefill_cache(model, tokenizer, prompt: str, device: torch.device):
    """Prefill the KV cache with a prompt and return past_key_values."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return getattr(out, "past_key_values", None)


def _extract_dynamic_cache_kv(past_key_values: Any) -> list[dict[str, Any]]:
    """Extract per-layer K/V tensors from a DynamicCache object."""
    layers = getattr(past_key_values, "layers", None) or []
    out: list[dict[str, Any]] = []
    for idx, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None or keys.numel() == 0 or values.numel() == 0:
            continue
        out.append({"layer_idx": int(idx), "key": keys, "value": values})
    return out


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


def prefill_decode(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    eos = tokenizer.eos_token_id
    eos_set = set()
    if eos is not None:
        eos_set.add(int(eos))
    cfg_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(cfg_eos, int):
        eos_set.add(int(cfg_eos))
    if isinstance(cfg_eos, list):
        eos_set.update(int(x) for x in cfg_eos)

    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    sync(device)
    t1 = time.perf_counter()

    past = getattr(out, "past_key_values", None)
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated = []

    sync(device)
    t2 = time.perf_counter()
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            out = model(input_ids=next_token, use_cache=True, past_key_values=past)
            past = getattr(out, "past_key_values", None)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            if eos_set and int(next_token.item()) in eos_set:
                break
    sync(device)
    t3 = time.perf_counter()

    if generated:
        gen_ids = torch.cat(generated, dim=1)
        full = torch.cat([input_ids, gen_ids], dim=1)
    else:
        gen_ids = None
        full = input_ids

    decode_tokens = int(full.size(1) - input_ids.size(1))
    decode_s = float(t3 - t2)
    return {
        "prefill_s": float(t1 - t0),
        "decode_s": decode_s,
        "decode_tokens": decode_tokens,
        "decode_toks_per_s": (float(decode_tokens) / decode_s) if decode_s > 0 else 0.0,
        "prompt_tokens": int(input_ids.size(1)),
        "gen_token_ids": gen_ids[0].tolist() if gen_ids is not None else [],
        "_final_past_key_values": past,
    }


def _peak_device_bytes(device: torch.device) -> int | None:
    try:
        if device.type == "cuda":
            return int(torch.cuda.max_memory_allocated())
        if device.type == "mps":
            current = getattr(torch.mps, "current_allocated_memory", None)
            driver = getattr(torch.mps, "driver_allocated_memory", None)
            vals = [fn() for fn in (current, driver) if callable(fn)]
            if vals:
                return int(max(vals))
    except Exception as exc:
        logger.debug("GPU memory query failed: %s", exc)
        return None


def _benchmark_memory_stats(run_out: Dict[str, Any], backend: KVStoreBackend | None, device: torch.device) -> Dict[str, Any]:
    handle = run_out.get("_final_past_key_values")
    kv_stats = backend.stats(handle) if backend is not None and handle is not None else {}
    estimated_kv_bytes = int(kv_stats.get("estimated_kv_bytes", 0) or 0)
    peak_device_bytes = _peak_device_bytes(device)
    return {
        "estimated_kv_bytes": estimated_kv_bytes,
        "estimated_kv_megabytes": float(estimated_kv_bytes) / (1024.0 ** 2),
        "peak_device_bytes": peak_device_bytes,
        "peak_device_megabytes": (float(peak_device_bytes) / (1024.0 ** 2)) if peak_device_bytes is not None else None,
        "layer_stats": kv_stats.get("layer_stats", {}),
    }


def _benchmark_turboquant_stats() -> Dict[str, Any]:
    return dict(get_turboquant_stats())


def _strip_internal_benchmark_fields(run_out: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in run_out.items() if not str(key).startswith("_")}


def _make_backend(
    store_cfg: KVCacheStoreConfig,
    v_quantize_only_first_n_layers: int,
    policy: Any | None = None,
    layer_types: List[str] | None = None,
) -> KVStoreBackend:
    n = int(v_quantize_only_first_n_layers)
    if not store_cfg.enable_turbo_quant or not store_cfg.turbo_quantize_v:
        return KVStoreBackend(store_cfg)
    policy_handle = _LayerTypeHandle(layer_types) if layer_types else None

    def store_cfg_factory(handle: Any, layer_idx: int) -> KVCacheStoreConfig:
        if policy is not None:
            effective_handle = handle
            if policy_handle is not None and getattr(handle, "layer_types", None) is None:
                effective_handle = policy_handle
            return apply_policy_to_store_cfg(store_cfg, policy, effective_handle, layer_idx)
        layer_types = getattr(handle, "layer_types", None)
        if layer_types is not None:
            idx = int(layer_idx)
            if idx < 0 or idx >= len(layer_types) or classify_kv_layer(handle, idx) is not KVLayerType.FULL_ATTENTION:
                return replace(store_cfg, turbo_quantize_v=False)
            full = [i for i, _ in enumerate(layer_types) if classify_kv_layer(handle, i) is KVLayerType.FULL_ATTENTION]
            pos = {li: p for p, li in enumerate(full)}.get(idx, None)
            if pos is None:
                return replace(store_cfg, turbo_quantize_v=False)
            if n <= 0 or pos >= n:
                return replace(store_cfg, turbo_quantize_v=False)
            return store_cfg

        if n <= 0:
            return replace(store_cfg, turbo_quantize_v=False)
        if int(layer_idx) >= n:
            return replace(store_cfg, turbo_quantize_v=False)
        return store_cfg

    return KVStoreBackend(store_cfg, store_cfg_factory=store_cfg_factory)


def _apply_vitriol_universal(
    store_cfg: KVCacheStoreConfig,
    v_quantize_only_first_n_layers: int,
    policy: Any | None = None,
    layer_types: List[str] | None = None,
    passthrough_update: bool = False,
    enable_attention_patch: bool = True,
):
    backend = _make_backend(store_cfg, v_quantize_only_first_n_layers, policy=policy, layer_types=layer_types)
    cache_patcher = CacheHookPatcher(
        cfg=CacheHookConfig(enabled=True, passthrough_update=bool(passthrough_update), auto_enable_mode=True),
        backend=backend,
    )
    cache_patcher.apply_to_class(DynamicCache)
    attn_patcher = UniversalAttentionPatcher(backend=backend) if enable_attention_patch else None
    if attn_patcher is not None:
        attn_patcher.apply()
    return backend, cache_patcher, attn_patcher


def run_short_suite(cfg: RunConfig) -> Dict[str, Any]:
    device = select_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    tokenizer = hf_load_tokenizer(
        cfg.model_id,
        security={"trust_remote_code": False, "allow_network": True, "local_files_only": False},
    )
    model = hf_load_causallm(
        cfg.model_id,
        security={"trust_remote_code": False, "allow_network": True, "local_files_only": False},
        torch_dtype=dtype,
        device=device,
    )

    preset = _select_preset(str(cfg.preset), dict(cfg.preset_params))

    tuned_cfg, preset_first_n, policy = _preset_to_kv_cfg(preset)
    inferred_layer_types = _infer_kv_layer_types(model.config)

    cases: List[Tuple[str, str]] = []
    for pt in cfg.prompt_tokens:
        filler = build_long_prompt(tokenizer, min_tokens=int(pt))
        for tag, task in default_prompt_suite():
            cases.append((f"pt{pt}:{tag}", filler + "\n\n" + task))

    baselines: Dict[str, Dict[str, Any]] = {}
    for name, prompt in cases:
        _ = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=4)
        baselines[name] = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(cfg.calib_new_tokens))

    chosen_n = int(preset_first_n)
    if preset.name in {"balanced", "fast-balanced"} and build_policy(preset).mode.value == "approx":
        passthrough_update, enable_attention_patch = _runtime_flags_for_preset(preset.name, 1)

        def is_ok(n: int) -> bool:
            for name, prompt in cases:
                backend, cache_patcher, attn_patcher = _apply_vitriol_universal(
                    tuned_cfg,
                    v_quantize_only_first_n_layers=int(n),
                    policy=policy,
                    layer_types=inferred_layer_types,
                    passthrough_update=passthrough_update,
                    enable_attention_patch=enable_attention_patch,
                )
                _ = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=4)
                out = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(cfg.calib_new_tokens))
                cache_patcher.restore()
                if attn_patcher is not None:
                    attn_patcher.restore()
                if out["gen_token_ids"] != baselines[name]["gen_token_ids"]:
                    return False
            return True

        chosen_n = _search_max_passing_n(int(cfg.search_max_n), is_ok)

    passthrough_update, enable_attention_patch = _runtime_flags_for_preset(preset.name, chosen_n)

    results = []
    ok_all = True
    for name, prompt in cases:
        base = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(cfg.max_new_tokens))

        backend, cache_patcher, attn_patcher = _apply_vitriol_universal(
            tuned_cfg,
            v_quantize_only_first_n_layers=int(chosen_n),
            policy=policy,
            layer_types=inferred_layer_types,
            passthrough_update=passthrough_update,
            enable_attention_patch=enable_attention_patch,
        )
        reset_turboquant_stats()
        out = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(cfg.max_new_tokens))
        tuned_memory = _benchmark_memory_stats(out, backend, device)
        tuned_turboquant = _benchmark_turboquant_stats()
        out_public = _strip_internal_benchmark_fields(out)
        cache_patcher.restore()
        if attn_patcher is not None:
            attn_patcher.restore()

        tuned_ok = out_public["gen_token_ids"] == base["gen_token_ids"]
        ok_all = ok_all and tuned_ok
        pm = prefix_match_tokens(base["gen_token_ids"], out_public["gen_token_ids"])
        rate = (pm / max(1, len(base["gen_token_ids"]))) * 100.0
        sp = (out_public["decode_toks_per_s"] / base["decode_toks_per_s"]) if base["decode_toks_per_s"] > 0 else 0.0
        results.append(
            {
                "name": name,
                "base_toks_per_s": base["decode_toks_per_s"],
                "tuned_toks_per_s": out_public["decode_toks_per_s"],
                "speedup": sp,
                "exact": tuned_ok,
                "prefix_match": (pm, len(base["gen_token_ids"]), rate),
                "tuned_memory": tuned_memory,
                "tuned_turboquant": tuned_turboquant,
            }
        )

    return {
        "model_id": cfg.model_id,
        "device": device.type,
        "dtype": str(dtype),
        "preset": preset.to_dict(),
        "chosen_v_quantize_only_first_n": int(chosen_n),
        "policy_insights": _collect_policy_insights(model.config, policy, int(chosen_n)),
        "all_cases_exact_match": bool(ok_all),
        "results": results,
    }


def compare_short_suite(base_cfg: RunConfig, compare_preset: str, compare_preset_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = run_short_suite(base_cfg)
    compare_cfg = replace(
        base_cfg,
        preset=str(compare_preset),
        preset_params=dict(compare_preset_params or {}),
    )
    compare = run_short_suite(compare_cfg)

    base_rows = {str(row.get("name")): row for row in (base.get("results") or [])}
    compare_rows = {str(row.get("name")): row for row in (compare.get("results") or [])}
    case_names = sorted(set(base_rows) | set(compare_rows))

    case_diffs: List[Dict[str, Any]] = []
    for name in case_names:
        lhs = base_rows.get(name, {})
        rhs = compare_rows.get(name, {})
        base_speedup = float(lhs.get("speedup", 0.0) or 0.0)
        compare_speedup = float(rhs.get("speedup", 0.0) or 0.0)
        case_diffs.append(
            {
                "name": name,
                "base_speedup": base_speedup,
                "compare_speedup": compare_speedup,
                "delta_speedup": compare_speedup - base_speedup,
                "base_exact": lhs.get("exact"),
                "compare_exact": rhs.get("exact"),
                "base_prefix_match": lhs.get("prefix_match"),
                "compare_prefix_match": rhs.get("prefix_match"),
                "base_toks_per_s": lhs.get("tuned_toks_per_s"),
                "compare_toks_per_s": rhs.get("tuned_toks_per_s"),
            }
        )

    return {
        "model_id": base_cfg.model_id,
        "base": base,
        "compare": compare,
        "case_diffs": case_diffs,
        "policy_diff": diff_policy_plans(_plan_from_suite_result(base), _plan_from_suite_result(compare)),
    }


def run_long_context_preset(
    model_id: str,
    prompt_tokens: int,
    max_new_tokens: int,
    preset: str,
    calib_new_tokens: int,
    search_max_n: int,
    preset_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
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

    p = _select_preset(str(preset), dict(preset_params or {}))

    tuned_cfg, preset_first_n, policy = _preset_to_kv_cfg(p)
    inferred_layer_types = _infer_kv_layer_types(model.config)

    prompt = build_long_prompt(tokenizer, min_tokens=int(prompt_tokens))

    base_calib = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(calib_new_tokens))
    chosen_n = int(preset_first_n)

    passthrough_update, enable_attention_patch = _runtime_flags_for_preset(p.name, 1)

    if p.name in {"balanced", "fast-balanced"} and build_policy(p).mode.value == "approx":
        def is_ok(n: int) -> bool:
            backend, cache_patcher, attn_patcher = _apply_vitriol_universal(
                tuned_cfg,
                v_quantize_only_first_n_layers=int(n),
                policy=policy,
                layer_types=inferred_layer_types,
                passthrough_update=passthrough_update,
                enable_attention_patch=enable_attention_patch,
            )
            out = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(calib_new_tokens))
            cache_patcher.restore()
            if attn_patcher is not None:
                attn_patcher.restore()
            return out["gen_token_ids"] == base_calib["gen_token_ids"]

        chosen_n = _search_max_passing_n(int(search_max_n), is_ok)

    passthrough_update, enable_attention_patch = _runtime_flags_for_preset(p.name, chosen_n)

    base = _strip_internal_benchmark_fields(prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(max_new_tokens)))

    backend, cache_patcher, attn_patcher = _apply_vitriol_universal(
        tuned_cfg,
        v_quantize_only_first_n_layers=int(chosen_n),
        policy=policy,
        layer_types=inferred_layer_types,
        passthrough_update=passthrough_update,
        enable_attention_patch=enable_attention_patch,
    )
    reset_turboquant_stats()
    tuned = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(max_new_tokens))
    tuned_memory = _benchmark_memory_stats(tuned, backend, device)
    tuned_turboquant = _benchmark_turboquant_stats()
    tuned_public = _strip_internal_benchmark_fields(tuned)
    cache_patcher.restore()
    if attn_patcher is not None:
        attn_patcher.restore()

    pm = prefix_match_tokens(base["gen_token_ids"], tuned_public["gen_token_ids"])
    rate = (pm / max(1, len(base["gen_token_ids"]))) * 100.0
    sp = (tuned_public["decode_toks_per_s"] / base["decode_toks_per_s"]) if base["decode_toks_per_s"] > 0 else 0.0

    return {
        "model_id": model_id,
        "device": device.type,
        "dtype": str(dtype),
        "prompt_tokens": int(prompt_tokens),
        "max_new_tokens": int(max_new_tokens),
        "calib_new_tokens": int(calib_new_tokens),
        "preset": p.to_dict(),
        "chosen_v_quantize_only_first_n": int(chosen_n),
        "policy_insights": _collect_policy_insights(model.config, policy, int(chosen_n)),
        "tuned_memory": tuned_memory,
        "tuned_turboquant": tuned_turboquant,
        "baseline": base,
        "tuned": tuned_public,
        "tuned_exact": tuned_public["gen_token_ids"] == base["gen_token_ids"],
        "tuned_speedup": float(sp),
        "tuned_prefix_match": (int(pm), int(len(base["gen_token_ids"])), float(rate)),
    }


def compare_long_context_preset(
    model_id: str,
    prompt_tokens: int,
    max_new_tokens: int,
    preset: str,
    compare_preset: str,
    calib_new_tokens: int,
    search_max_n: int,
    preset_params: Dict[str, Any] | None = None,
    compare_preset_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    base = run_long_context_preset(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        preset=preset,
        calib_new_tokens=calib_new_tokens,
        search_max_n=search_max_n,
        preset_params=preset_params,
    )
    compare = run_long_context_preset(
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        preset=compare_preset,
        calib_new_tokens=calib_new_tokens,
        search_max_n=search_max_n,
        preset_params=compare_preset_params,
    )
    base_speedup = float(base.get("tuned_speedup", 0.0) or 0.0)
    compare_speedup = float(compare.get("tuned_speedup", 0.0) or 0.0)
    return {
        "model_id": model_id,
        "prompt_tokens": int(prompt_tokens),
        "max_new_tokens": int(max_new_tokens),
        "calib_new_tokens": int(calib_new_tokens),
        "base": base,
        "compare": compare,
        "delta_speedup": compare_speedup - base_speedup,
        "policy_diff": diff_policy_plans(_plan_from_suite_result(base), _plan_from_suite_result(compare)),
    }


def run_long_context(model_id: str, prompt_tokens: int, max_new_tokens: int) -> Dict[str, Any]:
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

    prompt = build_long_prompt(tokenizer, min_tokens=int(prompt_tokens))
    base = _strip_internal_benchmark_fields(prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(max_new_tokens)))

    _, cache_patcher, attn_patcher = _apply_vitriol_universal(
        KVCacheStoreConfig(enable_turbo_quant=False),
        v_quantize_only_first_n_layers=0,
    )
    exact = _strip_internal_benchmark_fields(prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(max_new_tokens)))
    cache_patcher.restore()
    if attn_patcher is not None:
        attn_patcher.restore()

    _, cache_patcher, attn_patcher = _apply_vitriol_universal(
        KVCacheStoreConfig(enable_turbo_quant=True, turbo_quantize_k=False, turbo_quantize_v=True),
        v_quantize_only_first_n_layers=0,
    )
    approx = _strip_internal_benchmark_fields(prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(max_new_tokens)))
    cache_patcher.restore()
    if attn_patcher is not None:
        attn_patcher.restore()

    return {
        "model_id": model_id,
        "device": device.type,
        "dtype": str(dtype),
        "prompt_tokens": int(prompt_tokens),
        "max_new_tokens": int(max_new_tokens),
        "baseline": base,
        "vitriol_exact": exact,
        "vitriol_turbo3": approx,
    }


def run_smoke(
    model_id: str,
    preset: str = "balanced",
    prompt_tokens: int = 64,
    max_new_tokens: int = 8,
    calib_new_tokens: int = 8,
    search_max_n: int = 2,
    preset_params: Dict[str, Any] | None = None,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    run_id = new_run_id()
    device = select_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    try:
        tokenizer = hf_load_tokenizer(
            model_id,
            security={"trust_remote_code": trust_remote_code, "allow_network": True, "local_files_only": False},
        )
        model = hf_load_causallm(
            model_id,
            security={"trust_remote_code": trust_remote_code, "allow_network": True, "local_files_only": False},
            torch_dtype=dtype,
            device=device,
        )
    except Exception as e:
        logger.warning("Model load failed for %s: %s", model_id, e)
        return {
            "run_id": run_id,
            "model_id": model_id,
            "ok": False,
            "error": repr(e),
        }

    p = _select_preset(str(preset), dict(preset_params or {}))
    tuned_cfg, preset_first_n, policy = _preset_to_kv_cfg(p)
    inferred_layer_types = _infer_kv_layer_types(model.config)
    prompt = build_long_prompt(tokenizer, min_tokens=int(prompt_tokens)) + "\n\nReturn exactly the word: OK"

    base_calib = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(calib_new_tokens))
    chosen_n = int(preset_first_n)

    if p.name in {"balanced", "fast-balanced"} and build_policy(p).mode.value == "approx":
        passthrough_update, enable_attention_patch = _runtime_flags_for_preset(p.name, 1)

        def is_ok(n: int) -> bool:
            backend, cache_patcher, attn_patcher = _apply_vitriol_universal(
                tuned_cfg,
                v_quantize_only_first_n_layers=int(n),
                policy=policy,
                layer_types=inferred_layer_types,
                passthrough_update=passthrough_update,
                enable_attention_patch=enable_attention_patch,
            )
            out = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(calib_new_tokens))
            cache_patcher.restore()
            if attn_patcher is not None:
                attn_patcher.restore()
            return out["gen_token_ids"] == base_calib["gen_token_ids"]

        chosen_n = _search_max_passing_n(int(search_max_n), is_ok)

    passthrough_update, enable_attention_patch = _runtime_flags_for_preset(p.name, chosen_n)

    base = _strip_internal_benchmark_fields(prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(max_new_tokens)))

    backend, cache_patcher, attn_patcher = _apply_vitriol_universal(
        tuned_cfg,
        v_quantize_only_first_n_layers=int(chosen_n),
        policy=policy,
        layer_types=inferred_layer_types,
        passthrough_update=passthrough_update,
        enable_attention_patch=enable_attention_patch,
    )
    reset_turboquant_stats()
    tuned = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(max_new_tokens))
    tuned_memory = _benchmark_memory_stats(tuned, backend, device)
    tuned_turboquant = _benchmark_turboquant_stats()
    tuned_public = _strip_internal_benchmark_fields(tuned)
    cache_patcher.restore()
    if attn_patcher is not None:
        attn_patcher.restore()

    exact = tuned_public["gen_token_ids"] == base["gen_token_ids"]
    pm = prefix_match_tokens(base["gen_token_ids"], tuned_public["gen_token_ids"])
    rate = (pm / max(1, len(base["gen_token_ids"]))) * 100.0
    sp = (tuned_public["decode_toks_per_s"] / base["decode_toks_per_s"]) if base["decode_toks_per_s"] > 0 else 0.0

    return {
        "run_id": run_id,
        "model_id": model_id,
        "device": device.type,
        "dtype": str(dtype),
        "preset": p.to_dict(),
        "chosen_v_quantize_only_first_n": int(chosen_n),
        "policy_insights": _collect_policy_insights(model.config, policy, int(chosen_n)),
        "kv": {
            "compute_path": "store_hook",
            "storage_path": "packed" if bool(tuned_cfg.enable_turbo_quant) else "raw",
            "estimated_kv_bytes": int((tuned_memory or {}).get("estimated_kv_bytes", 0) or 0),
            "layer_stats": (tuned_memory or {}).get("layer_stats", {}) or {},
        },
        "stats": {
            "cache_hooks": get_cache_hook_stats(),
            "turboquant": tuned_turboquant,
            "kv_store": tuned_memory,
        },
        "tuned_memory": tuned_memory,
        "tuned_turboquant": tuned_turboquant,
        "passthrough_update": bool(passthrough_update),
        "enable_attention_patch": bool(enable_attention_patch),
        "ok": True,
        "tuned_exact": bool(exact),
        "tuned_speedup": float(sp),
        "tuned_prefix_match": (int(pm), int(len(base["gen_token_ids"])), float(rate)),
    }


def run_generate_preset(
    model_id: str,
    prompt: str,
    preset: str = "balanced",
    max_new_tokens: int = 64,
    calib_new_tokens: int = 8,
    search_max_n: int = 2,
    preset_params: Dict[str, Any] | None = None,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    run_id = new_run_id()
    device = select_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    try:
        tokenizer = hf_load_tokenizer(
            model_id,
            security={"trust_remote_code": trust_remote_code, "allow_network": True, "local_files_only": False},
        )
        model = hf_load_causallm(
            model_id,
            security={"trust_remote_code": trust_remote_code, "allow_network": True, "local_files_only": False},
            torch_dtype=dtype,
            device=device,
        )
    except Exception as e:
        logger.warning("Model/tokenizer load failed for %s: %s", model_id, e)
        return {
            "run_id": run_id,
            "model_id": model_id,
            "ok": False,
            "error": f"Failed to load model or tokenizer. This architecture may not be supported by your current version of transformers. Error: {e}",
        }

    p = _select_preset(str(preset), dict(preset_params or {}))
    tuned_cfg, preset_first_n, policy = _preset_to_kv_cfg(p)
    inferred_layer_types = _infer_kv_layer_types(model.config)

    chosen_n = int(preset_first_n)
    if p.name in {"balanced", "fast-balanced"} and build_policy(p).mode.value == "approx":
        base_calib = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(calib_new_tokens))
        passthrough_update, enable_attention_patch = _runtime_flags_for_preset(p.name, 1)

        def is_ok(n: int) -> bool:
            backend, cache_patcher, attn_patcher = _apply_vitriol_universal(
                tuned_cfg,
                v_quantize_only_first_n_layers=int(n),
                policy=policy,
                layer_types=inferred_layer_types,
                passthrough_update=passthrough_update,
                enable_attention_patch=enable_attention_patch,
            )
            out = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(calib_new_tokens))
            cache_patcher.restore()
            if attn_patcher is not None:
                attn_patcher.restore()
            return out["gen_token_ids"] == base_calib["gen_token_ids"]

        chosen_n = _search_max_passing_n(int(search_max_n), is_ok)

    passthrough_update, enable_attention_patch = _runtime_flags_for_preset(p.name, chosen_n)
    backend, cache_patcher, attn_patcher = _apply_vitriol_universal(
        tuned_cfg,
        v_quantize_only_first_n_layers=int(chosen_n),
        policy=policy,
        layer_types=inferred_layer_types,
        passthrough_update=passthrough_update,
        enable_attention_patch=enable_attention_patch,
    )
    reset_turboquant_stats()
    tuned = prefill_decode(model, tokenizer, prompt, device, max_new_tokens=int(max_new_tokens))
    tuned_memory = _benchmark_memory_stats(tuned, backend, device)
    tuned_turboquant = _benchmark_turboquant_stats()
    tuned_public = _strip_internal_benchmark_fields(tuned)
    cache_patcher.restore()
    if attn_patcher is not None:
        attn_patcher.restore()

    gen_token_ids = list(tuned_public.get("gen_token_ids") or [])
    generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

    return {
        "run_id": run_id,
        "model_id": model_id,
        "device": device.type,
        "dtype": str(dtype),
        "preset": p.to_dict(),
        "prompt": prompt,
        "prompt_tokens": int(tuned_public.get("prompt_tokens", 0) or 0),
        "max_new_tokens": int(max_new_tokens),
        "calib_new_tokens": int(calib_new_tokens),
        "chosen_v_quantize_only_first_n": int(chosen_n),
        "policy_insights": _collect_policy_insights(model.config, policy, int(chosen_n)),
        "kv": {
            "compute_path": "store_hook",
            "storage_path": "packed" if bool(tuned_cfg.enable_turbo_quant) else "raw",
            "estimated_kv_bytes": int((tuned_memory or {}).get("estimated_kv_bytes", 0) or 0),
            "layer_stats": (tuned_memory or {}).get("layer_stats", {}) or {},
        },
        "stats": {
            "cache_hooks": get_cache_hook_stats(),
            "turboquant": tuned_turboquant,
            "kv_store": tuned_memory,
        },
        "tuned_memory": tuned_memory,
        "tuned_turboquant": tuned_turboquant,
        "prefill_s": float(tuned_public.get("prefill_s", 0.0) or 0.0),
        "decode_s": float(tuned_public.get("decode_s", 0.0) or 0.0),
        "decode_tokens": int(tuned_public.get("decode_tokens", 0) or 0),
        "decode_toks_per_s": float(tuned_public.get("decode_toks_per_s", 0.0) or 0.0),
        "gen_token_ids": gen_token_ids,
        "generated_text": generated_text,
    }


def compare_smoke(
    model_id: str,
    preset: str = "balanced",
    compare_preset: str = "ultra-long",
    prompt_tokens: int = 64,
    max_new_tokens: int = 8,
    calib_new_tokens: int = 8,
    search_max_n: int = 2,
    preset_params: Dict[str, Any] | None = None,
    compare_preset_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    base = run_smoke(
        model_id=model_id,
        preset=preset,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        calib_new_tokens=calib_new_tokens,
        search_max_n=search_max_n,
        preset_params=preset_params,
    )
    compare = run_smoke(
        model_id=model_id,
        preset=compare_preset,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        calib_new_tokens=calib_new_tokens,
        search_max_n=search_max_n,
        preset_params=compare_preset_params,
    )
    base_speedup = float(base.get("speedup", base.get("tuned_speedup", 0.0)) or 0.0)
    compare_speedup = float(compare.get("speedup", compare.get("tuned_speedup", 0.0)) or 0.0)
    return {
        "model_id": model_id,
        "prompt_tokens": int(prompt_tokens),
        "max_new_tokens": int(max_new_tokens),
        "calib_new_tokens": int(calib_new_tokens),
        "base": base,
        "compare": compare,
        "delta_speedup": compare_speedup - base_speedup,
        "policy_diff": diff_policy_plans(_plan_from_suite_result(base), _plan_from_suite_result(compare)),
    }


# ============================================================================
# TurboQuantum Benchmark Functions
# ============================================================================

def run_turboquantum_synthetic(
    batch_size: int = 1,
    num_heads: int = 8,
    seq_len: int = 256,
    head_dim: int = 128,
    mode: str = "balanced",
    target_avg_bits: float = 3.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a synthetic TurboQuantum benchmark without loading any real model.

    Useful for:
    - Quick validation that TurboQuantum is working
    - Comparing different modes (conservative/balanced/aggressive)
    - CI/CD regression testing
    """
    if not _HAS_TURBOQUANTUM:
        return {"ok": False, "error": "TurboQuantum not available."}

    device = select_device()
    torch.manual_seed(seed)

    b, h, s, d = batch_size, num_heads, seq_len, head_dim
    q = torch.randn(b, h, s, d, dtype=torch.float32, device=device)
    k = torch.randn(b, h, s, d, dtype=torch.float32, device=device)
    v = torch.randn(b, h, s, d, dtype=torch.float32, device=device)

    config = TurboQuantumConfig(mode=mode, target_avg_bits=target_avg_bits)
    t0 = time.perf_counter()
    result = turboquantum_compress(q, k, v, config)
    t1 = time.perf_counter()

    entropy_val, entropy_report = compute_attention_entropy(q, k)

    k_mse = float(torch.mean((result.compressed_key - k) ** 2).item())
    v_mse = float(torch.mean((result.compressed_value - v) ** 2).item())
    k_cosine = float(torch.nn.functional.cosine_similarity(
        result.compressed_key.flatten(), k.flatten(), dim=0
    ).item())
    v_cosine = float(torch.nn.functional.cosine_similarity(
        result.compressed_value.flatten(), v.flatten(), dim=0
    ).item())

    original_bytes = b * h * s * d * 2
    comp_bpv = result.report.get("effective_bpv", 3.0)
    compressed_kb = (b * h * s * d * comp_bpv / 8) * 2 / 1024.0

    return {
        "ok": True,
        "device": device.type,
        "shape": {"batch": b, "heads": h, "seq_len": s, "head_dim": d},
        "mode": mode,
        "target_avg_bits": target_avg_bits,
        "compression": {
            "effective_bpv": comp_bpv,
            "compression_ratio_vs_fp16": result.report.get("compression_vs_fp16", 0.0),
            "original_kb": original_bytes / 1024.0,
            "compressed_kb": compressed_kb,
            "savings_percent": result.report.get("compression_vs_fp16", 0.0) * 100.0,
        },
        "quality": {
            "k_mse": k_mse, "v_mse": v_mse,
            "k_cosine": k_cosine, "v_cosine": v_cosine,
        },
        "entropy_analysis": entropy_report,
        "timing_ms": round((t1 - t0) * 1000.0, 2),
        "tunneling_stats": {
            "protected_tokens_pct": result.report.get("protected_token_fraction", 0.0) * 100.0,
            "tunneling_enabled": config.enable_tunneling,
        },
    }


def compare_turboquantum_modes(
    batch_size: int = 1,
    num_heads: int = 16,
    seq_len: int = 512,
    head_dim: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare all TurboQuantum modes side-by-side on synthetic data."""
    if not _HAS_TURBOQUANTUM:
        return {"ok": False, "error": "TurboQuantum not available."}
    results = {}
    for mode in ["conservative", "balanced", "aggressive", "ultra-long"]:
        try:
            r = run_turboquantum_synthetic(batch_size=batch_size, num_heads=num_heads,
                seq_len=seq_len, head_dim=head_dim, mode=mode, seed=seed)
            results[mode] = r
        except Exception as e:
            logger.warning("TurboQuantum mode '%s' failed: %s", mode, e)
            results[mode] = {"ok": False, "error": str(e)}

    comparison = []
    for mode, r in results.items():
        if r.get("ok"):
            comparison.append({
                "mode": mode,
                "bpv": r["compression"]["effective_bpv"],
                "k_cosine": r["quality"]["k_cosine"],
                "v_cosine": r["quality"]["v_cosine"],
                "k_mse": r["quality"]["k_mse"],
                "v_mse": r["quality"]["v_mse"],
                "savings_pct": r["compression"]["savings_percent"],
                "time_ms": r["timing_ms"],
            })

    return {
        "ok": True,
        "shape": {"batch": batch_size, "heads": num_heads, "seq_len": seq_len, "head_dim": head_dim},
        "results": results,
        "comparison_table": comparison,
        "best_quality_mode": max(comparison, key=lambda x: x["k_cosine"] + x["v_cosine"])["mode"] if comparison else None,
        "best_compression_mode": min(comparison, key=lambda x: x["bpv"])["mode"] if comparison else None,
    }


def run_turboquantum_on_model_kv(
    model_id: str,
    prompt_tokens: int = 128,
    mode: str = "balanced",
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """Run TurboQuantum compression on a real model's KV cache."""
    if not _HAS_TURBOQUANTUM:
        return {"ok": False, "error": "TurboQuantum not available."}
    device = select_device()
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    try:
        tokenizer = hf_load_tokenizer(
            model_id,
            security={"trust_remote_code": trust_remote_code, "allow_network": True, "local_files_only": False},
        )
        model = hf_load_causallm(
            model_id,
            security={"trust_remote_code": trust_remote_code, "allow_network": True, "local_files_only": False},
            torch_dtype=dtype,
            device=device,
        )
    except Exception as e:
        logger.warning("Model load failed for %s: %s", model_id, e)
        return {"ok": False, "error": f"Model load failed: {e}", "model_id": model_id}
    prompt = build_long_prompt(tokenizer, min_tokens=prompt_tokens)
    past = _prefill_cache(model, tokenizer, prompt, device)
    kv_layers = _extract_dynamic_cache_kv(past)
    if not kv_layers:
        return {"ok": False, "error": "No KV cache layers extracted.", "model_id": model_id}

    layer_results = []
    total_original_bytes = 0
    total_compressed_bytes = 0
    config = TurboQuantumConfig(mode=mode)

    for item in kv_layers:
        layer_idx = item["layer_idx"]
        key = item["key"].to(torch.float32)
        value = item["value"].to(torch.float32)
        orig_shape = key.shape

        # Ensure 4D format for turboquantum_compress: [batch, heads, seq_len, head_dim]
        if len(orig_shape) == 4:
            b, nh, s, d = orig_shape
            key_4d = key
            value_4d = value
        elif len(orig_shape) == 3:
            nh, s, d = orig_shape
            # Add batch dimension to make it 4D
            key_4d = key.unsqueeze(0)  # [1, nh, s, d]
            value_4d = value.unsqueeze(0)  # [1, nh, s, d]
        else:
            continue

        # Create dummy query from key (last position, broadcast to full seq)
        q_dummy = key_4d[..., -1:, :].expand(-1, -1, s, -1)  # [b, nh, s, d]

        t0 = time.perf_counter()
        result = turboquantum_compress(q_dummy, key_4d, value_4d, config)
        t1 = time.perf_counter()

        # Compute MSE against original (use 4D for comparison)
        k_mse = float(torch.mean((result.compressed_key - key_4d) ** 2).item())
        v_mse = float(torch.mean((result.compressed_value - value_4d) ** 2).item())
        orig_bytes = key.numel() * 2 + value.numel() * 2
        comp_bpv = result.report.get("effective_bpv", 3.0)
        comp_bytes = (key.numel() + value.numel()) * comp_bpv / 8
        total_original_bytes += orig_bytes
        total_compressed_bytes += comp_bytes
        layer_results.append({
            "layer_idx": layer_idx,
            "shape": list(orig_shape),
            "bpv": comp_bpv,
            "k_mse": k_mse, "v_mse": v_mse,
            "k_cosine": result.report.get("k_cosine", 0.0),
            "v_cosine": result.report.get("v_cosine", 0.0),
            "time_ms": round((t1 - t0) * 1000.0, 3),
            "orig_kb": orig_bytes / 1024.0,
            "comp_kb": comp_bytes / 1024.0,
        })

    n = len(layer_results)
    return {
        "ok": True,
        "model_id": model_id,
        "device": device.type,
        "dtype": str(dtype),
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "total_layers": len(kv_layers),
        "total_original_mb": total_original_bytes / (1024.0 ** 2),
        "total_compressed_mb": total_compressed_bytes / (1024.0 ** 2),
        "overall_savings_pct": (1.0 - total_compressed_bytes / max(total_original_bytes, 1)) * 100.0,
        "averages": {
            "k_mse": sum(r["k_mse"] for r in layer_results) / max(n, 1),
            "v_mse": sum(r["v_mse"] for r in layer_results) / max(n, 1),
        },
        "layer_details": layer_results,
    }
