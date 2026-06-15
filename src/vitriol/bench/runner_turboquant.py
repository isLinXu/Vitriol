"""TurboQuantum benchmark functions — extracted from runner.py.

Provides synthetic and model-based TurboQuantum compression benchmarks.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

import torch

from ..kv.turboquantum import (
    TurboQuantumConfig,
    compute_attention_entropy,
    turboquantum_compress,
)
from ..utils.hf_loading import load_causallm as hf_load_causallm
from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

logger = logging.getLogger(__name__)


def run_turboquantum_synthetic(
    batch_size: int = 1,
    num_heads: int = 8,
    seq_len: int = 256,
    head_dim: int = 128,
    mode: str = "balanced",
    target_avg_bits: float = 3.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a synthetic TurboQuantum benchmark without loading any real model.

    Useful for:
    - Quick validation that TurboQuantum is working
    - Comparing different modes (conservative/balanced/aggressive)
    - CI/CD regression testing
    """
    # Local import to avoid pulling runner.py eagerly
    from .runner import select_device

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
    # Local imports to avoid pulling runner.py eagerly
    from .runner import (
        build_long_prompt,
        select_device,
        _extract_dynamic_cache_kv,
        _prefill_cache,
    )

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

        if len(orig_shape) == 4:
            b, nh, s, d = orig_shape
            key_4d = key
            value_4d = value
        elif len(orig_shape) == 3:
            nh, s, d = orig_shape
            key_4d = key.unsqueeze(0)
            value_4d = value.unsqueeze(0)
        else:
            continue

        q_dummy = key_4d[..., -1:, :].expand(-1, -1, s, -1)

        t0 = time.perf_counter()
        result = turboquantum_compress(q_dummy, key_4d, value_4d, config)
        t1 = time.perf_counter()

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
