from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from ..kv.codec import AdaptiveKVCodec, ComputeSkipConfig, compute_skip_attention
from .turboquant import resolve_turbo_kv_formats, sparse_v_attention, turbo_quantize


@dataclass(frozen=True)
class KVRuntimePatchConfig:
    decode_only: bool = True
    decode_query_len: int = 1
    enable_sparse_v: bool = False
    sparse_v_threshold: float = 0.01
    enable_compute_skip: bool = False
    compute_skip: ComputeSkipConfig = ComputeSkipConfig()
    enable_adaptive_bits: bool = False
    adaptive_bits: AdaptiveKVCodec = AdaptiveKVCodec()
    enable_turbo_quant: bool = False
    turbo_bits: Optional[float] = None
    turbo_k_bits: Optional[float] = None
    turbo_v_bits: Optional[float] = None
    turbo_format: str = "turbo3"
    turbo_k_format: str = "turbo3"
    turbo_v_format: str = "turbo3"
    turbo_block_size: int = 32
    turbo_quantize_k: bool = True
    turbo_quantize_v: bool = True
    quantized_kv_start: int = 0
    enable_turbo_residual_qjl: bool = True
    turbo_residual_strength: float = 0.5

    # Preprocess cache (reuse K/V after turbo/adaptive transforms).
    # Note: keys include the tensor id by default, so hit rate depends on upstream object reuse.
    # To avoid unbounded growth in long runs, enforce a capacity limit; <=0 disables the cache.
    preprocess_cache_max_entries: int = 128

    def __post_init__(self) -> None:
        block_size = int(self.turbo_block_size)
        if block_size <= 0:
            raise ValueError(f"turbo_block_size must be > 0, got {self.turbo_block_size!r}")
        object.__setattr__(self, "turbo_block_size", block_size)

        quantized_start = int(self.quantized_kv_start)
        if quantized_start < 0:
            raise ValueError(f"quantized_kv_start must be >= 0, got {self.quantized_kv_start!r}")
        object.__setattr__(self, "quantized_kv_start", quantized_start)

        residual_strength = float(self.turbo_residual_strength)
        if not math.isfinite(residual_strength) or residual_strength < 0:
            raise ValueError(
                f"turbo_residual_strength must be finite and non-negative, got {self.turbo_residual_strength!r}"
            )
        object.__setattr__(self, "turbo_residual_strength", residual_strength)

        sparse_threshold = float(self.sparse_v_threshold)
        if not math.isfinite(sparse_threshold) or sparse_threshold < 0:
            raise ValueError(f"sparse_v_threshold must be finite and non-negative, got {self.sparse_v_threshold!r}")
        object.__setattr__(self, "sparse_v_threshold", sparse_threshold)

        cache_max = int(self.preprocess_cache_max_entries)
        if cache_max < 0:
            raise ValueError(f"preprocess_cache_max_entries must be >= 0, got {self.preprocess_cache_max_entries!r}")
        object.__setattr__(self, "preprocess_cache_max_entries", cache_max)

        turbo_k_format, turbo_v_format = resolve_turbo_kv_formats(
            turbo_format=self.turbo_format,
            turbo_k_format=self.turbo_k_format,
            turbo_v_format=self.turbo_v_format,
            turbo_bits=self.turbo_bits,
            turbo_k_bits=self.turbo_k_bits,
            turbo_v_bits=self.turbo_v_bits,
        )
        object.__setattr__(self, "turbo_k_format", turbo_k_format)
        object.__setattr__(self, "turbo_v_format", turbo_v_format)


class KVRuntimePatcher:
    def __init__(self, cfg: KVRuntimePatchConfig) -> None:
        self.cfg = cfg
        self._original: Optional[Callable[..., Any]] = None
        self._preprocess_cache: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._calls_total: int = 0
        self._calls_bypassed: int = 0
        self._calls_patched: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_evictions: int = 0

    def apply(self) -> None:
        if getattr(F.scaled_dot_product_attention, "_vitriol_kv_patched", False):
            return
        self._original = F.scaled_dot_product_attention

        def patched(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any):
            self._calls_total += 1
            attn_mask = kwargs.get("attn_mask", None)
            dropout_p = float(kwargs.get("dropout_p", 0.0))
            is_causal = bool(kwargs.get("is_causal", False))

            if self.cfg.decode_only and int(query.size(-2)) != int(self.cfg.decode_query_len):
                self._calls_bypassed += 1
                return self._original(query, key, value, *args, **kwargs)

            self._calls_patched += 1
            q = query
            k = key
            v = value

            cache_key = (
                id(q),
                id(k),
                id(v),
                bool(self.cfg.enable_turbo_quant),
                str(self.cfg.turbo_k_format),
                str(self.cfg.turbo_v_format),
                int(self.cfg.turbo_block_size),
                bool(self.cfg.turbo_quantize_k),
                bool(self.cfg.turbo_quantize_v),
                int(self.cfg.quantized_kv_start),
                bool(self.cfg.enable_turbo_residual_qjl),
                float(self.cfg.turbo_residual_strength),
                bool(self.cfg.enable_adaptive_bits),
                float(self.cfg.adaptive_bits.target_avg_bits),
                int(self.cfg.adaptive_bits.block_size),
                float(self.cfg.adaptive_bits.min_bits),
                float(self.cfg.adaptive_bits.max_bits),
                float(self.cfg.adaptive_bits.k_share),
                float(self.cfg.adaptive_bits.rotate_kurtosis_threshold),
            )
            cached = self._preprocess_cache.get(cache_key)
            max_entries = int(getattr(self.cfg, "preprocess_cache_max_entries", 0) or 0)
            cache_enabled = max_entries > 0 and (self.cfg.enable_turbo_quant or self.cfg.enable_adaptive_bits)

            if cache_enabled and cached is not None:
                self._cache_hits += 1
                self._preprocess_cache.move_to_end(cache_key)
                k, v = cached
            else:
                if cache_enabled:
                    self._cache_misses += 1

                if self.cfg.enable_turbo_quant and int(k.size(-2)) >= int(self.cfg.quantized_kv_start):
                    if self.cfg.turbo_quantize_k:
                        k = turbo_quantize(
                            k,
                            format_type=self.cfg.turbo_k_format,
                            block_size=self.cfg.turbo_block_size,
                            use_residual_qjl=bool(self.cfg.enable_turbo_residual_qjl),
                            residual_strength=float(self.cfg.turbo_residual_strength),
                        )
                    if self.cfg.turbo_quantize_v:
                        v = turbo_quantize(
                            v,
                            format_type=self.cfg.turbo_v_format,
                            block_size=self.cfg.turbo_block_size,
                            use_residual_qjl=bool(self.cfg.enable_turbo_residual_qjl),
                            residual_strength=float(self.cfg.turbo_residual_strength),
                        )

                if self.cfg.enable_adaptive_bits:
                    k, v, _ = self.cfg.adaptive_bits.quantize_kv(query=q, key=k, value=v)

                if cache_enabled:
                    self._preprocess_cache[cache_key] = (k, v)
                    self._preprocess_cache.move_to_end(cache_key)
                    while len(self._preprocess_cache) > max_entries:
                        self._preprocess_cache.popitem(last=False)
                        self._cache_evictions += 1

            if self.cfg.enable_compute_skip:
                res = compute_skip_attention(
                    q,
                    k,
                    v,
                    cfg=self.cfg.compute_skip,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
                return res.output

            if self.cfg.enable_sparse_v:
                return sparse_v_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    threshold=float(self.cfg.sparse_v_threshold),
                )

            return self._original(q, k, v, *args, **kwargs)

        patched._vitriol_kv_patched = True
        patched._vitriol_kv_original = self._original
        F.scaled_dot_product_attention = patched

    def restore(self) -> None:
        current = F.scaled_dot_product_attention
        original = getattr(current, "_vitriol_kv_original", None)
        if getattr(current, "_vitriol_kv_patched", False) and original is not None:
            F.scaled_dot_product_attention = original

    def stats(self) -> dict:
        hits = int(self._cache_hits)
        misses = int(self._cache_misses)
        evictions = int(self._cache_evictions)
        denom = hits + misses
        return {
            "calls_total": int(self._calls_total),
            "calls_bypassed": int(self._calls_bypassed),
            "calls_patched": int(self._calls_patched),
            "preprocess_cache_entries": int(len(self._preprocess_cache)),
            "preprocess_cache_hits": hits,
            "preprocess_cache_misses": misses,
            "preprocess_cache_evictions": evictions,
            "preprocess_cache_hit_rate": (float(hits) / float(denom)) if denom > 0 else 0.0,
        }


def patch_kv_runtime(cfg: KVRuntimePatchConfig) -> KVRuntimePatcher:
    patcher = KVRuntimePatcher(cfg)
    patcher.apply()
    return patcher
