from __future__ import annotations

import math
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from ..kv.codec import AdaptiveKVCodec, ComputeSkipConfig, compute_skip_attention
from .turboquant import resolve_turbo_kv_formats, sparse_v_attention, turbo_quantize

_PATCH_LOCK = threading.RLock()


@dataclass(frozen=True)
class KVRuntimePatchConfig:
    """Configuration for KV runtime patching."""
    decode_only: bool = True
    decode_query_len: int = 1
    enable_sparse_v: bool = False
    sparse_v_threshold: float = 0.01
    enable_compute_skip: bool = False
    compute_skip: ComputeSkipConfig = ComputeSkipConfig()
    enable_adaptive_bits: bool = False
    adaptive_bits: AdaptiveKVCodec = field(default_factory=AdaptiveKVCodec)
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
    # Reuse requires identical live tensors with unchanged version/layout.
    # Inference tensors without version counters and autograd/autocast work bypass reuse.
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


@dataclass
class _PreprocessEntry:
    inputs: tuple[weakref.ReferenceType, ...]
    signature: tuple[Any, ...]
    key: torch.Tensor
    value: torch.Tensor


def _input_signature(tensors: tuple[torch.Tensor, ...]) -> Optional[tuple[Any, ...]]:
    """Return tracked input state, or disable reuse where state cannot be verified."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return None
    try:
        autocast = torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda")
    except TypeError:  # PyTorch versions before the device-type argument.
        autocast = torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
    if autocast:
        return None
    try:
        return tuple((t._version, tuple(t.shape), t.stride(), t.storage_offset(),
                      t.data_ptr(), t.dtype, t.device) for t in tensors)
    except RuntimeError:
        # torch.inference_mode() tensors have no version counter. Identity alone
        # cannot establish whether they have changed since the previous call.
        return None


class KVRuntimePatcher:
    """Patcher that modifies KV cache behavior at runtime."""
    def __init__(self, cfg: KVRuntimePatchConfig) -> None:
        self.cfg = cfg
        self._original: Optional[Callable[..., Any]] = None
        self._installed: Optional[Callable[..., Any]] = None
        self._deactivate: Optional[Callable[[], None]] = None
        self._cache_lock = threading.RLock()
        self._context_depth = 0
        self._preprocess_cache: OrderedDict[tuple[Any, ...], _PreprocessEntry] = OrderedDict()
        self._calls_total: int = 0
        self._calls_bypassed: int = 0
        self._calls_patched: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_evictions: int = 0

    def apply(self) -> Any:
        with _PATCH_LOCK:
            self._apply()

    def _apply(self) -> None:
        if self._installed is not None:
            return
        if getattr(F.scaled_dot_product_attention, "_vitriol_kv_patched", False) and getattr(
            F.scaled_dot_product_attention, "_vitriol_kv_is_active", lambda: True,
        )():
            raise RuntimeError("Another KV runtime patch is already active")
        original = F.scaled_dot_product_attention
        self._original = original
        active = True

        def patched(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None, dropout_p: float = 0.0,
            is_causal: bool = False, *, scale: Optional[float] = None,
            enable_gqa: bool = False,
        ) -> Any:
            kwargs = {"attn_mask": attn_mask, "dropout_p": dropout_p, "is_causal": is_causal}
            if scale is not None:
                kwargs["scale"] = scale
            if enable_gqa:
                kwargs["enable_gqa"] = True
            if not active:
                return original(query, key, value, **kwargs)

            self._calls_total += 1
            if self.cfg.decode_only and int(query.size(-2)) != int(self.cfg.decode_query_len):
                self._calls_bypassed += 1
                return original(query, key, value, **kwargs)

            self._calls_patched += 1
            q = query
            k = key
            v = value
            inputs = (q, k, v)

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
            max_entries = int(getattr(self.cfg, "preprocess_cache_max_entries", 0) or 0)
            cache_enabled = max_entries > 0 and (self.cfg.enable_turbo_quant or self.cfg.enable_adaptive_bits)
            signature = _input_signature(inputs) if cache_enabled else None
            cache_enabled = cache_enabled and signature is not None
            with self._cache_lock:
                cached = self._preprocess_cache.get(cache_key) if cache_enabled and active else None
                if cached is not None and cached.signature == signature and all(
                    ref() is tensor for ref, tensor in zip(cached.inputs, inputs)
                ):
                    self._cache_hits += 1
                    self._preprocess_cache.move_to_end(cache_key)
                else:
                    cached = None
            if cached is not None:
                k, v = cached.key, cached.value
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
                    with self._cache_lock:
                        if active:
                            self._preprocess_cache[cache_key] = _PreprocessEntry(
                                tuple(weakref.ref(t) for t in inputs), signature, k, v,
                            )
                            self._preprocess_cache.move_to_end(cache_key)
                            while len(self._preprocess_cache) > max_entries:
                                self._preprocess_cache.popitem(last=False)
                                self._cache_evictions += 1

            if self.cfg.enable_compute_skip or self.cfg.enable_sparse_v:
                if enable_gqa and k.size(-3) != v.size(-3):
                    raise ValueError("GQA requires matching key and value head counts")
                q_heads = q.size(-3)
                expanded = []
                for tensor in (k, v):
                    heads = tensor.size(-3)
                    if heads != q_heads:
                        if heads <= 0 or q_heads % heads or (not enable_gqa and heads != 1):
                            raise ValueError("Query heads must match KV heads or enable valid GQA")
                        tensor = tensor.repeat_interleave(q_heads // heads, dim=-3)
                    expanded.append(tensor)
                k, v = expanded

            if self.cfg.enable_compute_skip:
                res = compute_skip_attention(
                    q,
                    k,
                    v,
                    cfg=self.cfg.compute_skip,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scaling=scale,
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
                    scaling=scale,
                )

            return original(q, k, v, **kwargs)

        def deactivate() -> None:
            nonlocal active
            active = False

        patched._vitriol_kv_patched = True
        patched._vitriol_kv_original = original
        patched._vitriol_kv_is_active = lambda: active
        self._installed = patched
        self._deactivate = deactivate
        F.scaled_dot_product_attention = patched

    def restore(self) -> None:
        with _PATCH_LOCK, self._cache_lock:
            self._restore()

    def _restore(self) -> None:
        if self._deactivate is not None:
            self._deactivate()
        if self._installed is not None and F.scaled_dot_product_attention is self._installed:
            F.scaled_dot_product_attention = self._original
        self._preprocess_cache.clear()
        self._installed = None
        self._original = None
        self._deactivate = None
        self._context_depth = 0

    def __enter__(self) -> KVRuntimePatcher:
        with _PATCH_LOCK:
            self.apply()
            self._context_depth += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        with _PATCH_LOCK:
            self._context_depth = max(0, self._context_depth - 1)
            if self._context_depth == 0:
                self.restore()

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
