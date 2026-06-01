from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from ..kv.codec import AdaptiveKVCodec, ComputeSkipConfig, compute_skip_attention
from .turboquant import resolve_turbo_kv_formats, sparse_v_attention, turbo_quantize


@dataclass(frozen=True)
class Qwen35AttentionPatchConfig:
    decode_only: bool = True
    decode_query_len: int = 1
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
    enable_adaptive_bits: bool = False
    adaptive_bits: AdaptiveKVCodec = AdaptiveKVCodec()
    enable_sparse_v: bool = False
    sparse_v_threshold: float = 0.01
    enable_compute_skip: bool = False
    compute_skip: ComputeSkipConfig = ComputeSkipConfig()

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


class Qwen35AttentionPatcher:
    def __init__(self, cfg: Qwen35AttentionPatchConfig) -> None:
        self.cfg = cfg
        self._orig_sdpa: Optional[Callable[..., Any]] = None
        self._orig_sdpa_registry: Optional[Callable[..., Any]] = None
        self._orig_paged_sdpa_registry: Optional[Callable[..., Any]] = None
        self._orig_eager: Optional[Callable[..., Any]] = None
        self._calls_total: int = 0
        self._calls_bypassed: int = 0
        self._calls_patched: int = 0

    def apply(self) -> None:
        import transformers.modeling_utils as mu
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        if getattr(m.eager_attention_forward, "_vitriol_qwen35_patched", False):
            return

        self._orig_sdpa = mu.sdpa_attention_forward
        self._orig_sdpa_registry = mu.ALL_ATTENTION_FUNCTIONS.get("sdpa")
        self._orig_paged_sdpa_registry = mu.ALL_ATTENTION_FUNCTIONS.get("paged|sdpa")
        self._orig_eager = m.eager_attention_forward

        def wrapped(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float,
            dropout: float = 0.0,
            **kwargs: Any,
        ):
            self._calls_total += 1

            if module.__class__.__name__ != "Qwen3_5Attention":
                return self._orig_sdpa(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

            if self.cfg.decode_only and int(query.size(-2)) != int(self.cfg.decode_query_len):
                self._calls_bypassed += 1
                return self._orig_sdpa(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

            self._calls_patched += 1

            key_states = m.repeat_kv(key, module.num_key_value_groups)
            value_states = m.repeat_kv(value, module.num_key_value_groups)

            if self.cfg.enable_turbo_quant and int(key_states.size(-2)) >= int(self.cfg.quantized_kv_start):
                if self.cfg.turbo_quantize_k:
                    key_states = turbo_quantize(
                        key_states,
                        format_type=self.cfg.turbo_k_format,
                        block_size=self.cfg.turbo_block_size,
                        use_residual_qjl=bool(self.cfg.enable_turbo_residual_qjl),
                        residual_strength=float(self.cfg.turbo_residual_strength),
                    )
                if self.cfg.turbo_quantize_v:
                    value_states = turbo_quantize(
                        value_states,
                        format_type=self.cfg.turbo_v_format,
                        block_size=self.cfg.turbo_block_size,
                        use_residual_qjl=bool(self.cfg.enable_turbo_residual_qjl),
                        residual_strength=float(self.cfg.turbo_residual_strength),
                    )

            if self.cfg.enable_adaptive_bits:
                key_states, value_states, _ = self.cfg.adaptive_bits.quantize_kv(query=query, key=key_states, value=value_states)

            if self.cfg.enable_compute_skip:
                out = compute_skip_attention(
                    query,
                    key_states,
                    value_states,
                    cfg=self.cfg.compute_skip,
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    is_causal=True,
                ).output
                return out, None

            if self.cfg.enable_sparse_v:
                out = sparse_v_attention(
                    query,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    is_causal=True,
                    threshold=float(self.cfg.sparse_v_threshold),
                )
                return out, None

            out, attn = m.eager_attention_forward(
                module,
                query=query,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                **kwargs,
            )
            return out, attn

        wrapped._vitriol_qwen35_patched = True
        wrapped._vitriol_qwen35_original = self._orig_sdpa
        mu.sdpa_attention_forward = wrapped
        mu.ALL_ATTENTION_FUNCTIONS["sdpa"] = wrapped
        if "paged|sdpa" in mu.ALL_ATTENTION_FUNCTIONS:
            mu.ALL_ATTENTION_FUNCTIONS["paged|sdpa"] = wrapped

        def eager_wrapped(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float,
            dropout: float = 0.0,
            **kwargs: Any,
        ):
            self._calls_total += 1

            if module.__class__.__name__ != "Qwen3_5Attention":
                return self._orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

            if self.cfg.decode_only and int(query.size(-2)) != int(self.cfg.decode_query_len):
                self._calls_bypassed += 1
                return self._orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

            self._calls_patched += 1

            key_states = m.repeat_kv(key, module.num_key_value_groups)
            value_states = m.repeat_kv(value, module.num_key_value_groups)

            if self.cfg.enable_turbo_quant and int(key_states.size(-2)) >= int(self.cfg.quantized_kv_start):
                if self.cfg.turbo_quantize_k:
                    key_states = turbo_quantize(
                        key_states,
                        format_type=self.cfg.turbo_k_format,
                        block_size=self.cfg.turbo_block_size,
                        use_residual_qjl=bool(self.cfg.enable_turbo_residual_qjl),
                        residual_strength=float(self.cfg.turbo_residual_strength),
                    )
                if self.cfg.turbo_quantize_v:
                    value_states = turbo_quantize(
                        value_states,
                        format_type=self.cfg.turbo_v_format,
                        block_size=self.cfg.turbo_block_size,
                        use_residual_qjl=bool(self.cfg.enable_turbo_residual_qjl),
                        residual_strength=float(self.cfg.turbo_residual_strength),
                    )

            if self.cfg.enable_adaptive_bits:
                key_states, value_states, _ = self.cfg.adaptive_bits.quantize_kv(query=query, key=key_states, value=value_states)

            if self.cfg.enable_compute_skip:
                out = compute_skip_attention(
                    query,
                    key_states,
                    value_states,
                    cfg=self.cfg.compute_skip,
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    is_causal=True,
                ).output
                return out, None

            if self.cfg.enable_sparse_v:
                out = sparse_v_attention(
                    query,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    is_causal=True,
                    threshold=float(self.cfg.sparse_v_threshold),
                )
                return out, None

            return self._orig_eager(
                module,
                query,
                key_states,
                value_states,
                attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )

        eager_wrapped._vitriol_qwen35_patched = True
        eager_wrapped._vitriol_qwen35_original = self._orig_eager
        m.eager_attention_forward = eager_wrapped

    def restore(self) -> None:
        import transformers.modeling_utils as mu
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        current = mu.sdpa_attention_forward
        original = getattr(current, "_vitriol_qwen35_original", None)
        if getattr(current, "_vitriol_qwen35_patched", False) and original is not None:
            mu.sdpa_attention_forward = original
        if self._orig_sdpa_registry is not None:
            mu.ALL_ATTENTION_FUNCTIONS["sdpa"] = self._orig_sdpa_registry
        if self._orig_paged_sdpa_registry is not None and "paged|sdpa" in mu.ALL_ATTENTION_FUNCTIONS:
            mu.ALL_ATTENTION_FUNCTIONS["paged|sdpa"] = self._orig_paged_sdpa_registry
        if self._orig_eager is not None and getattr(m.eager_attention_forward, "_vitriol_qwen35_patched", False):
            m.eager_attention_forward = self._orig_eager

    def stats(self) -> dict:
        return {
            "calls_total": int(self._calls_total),
            "calls_bypassed": int(self._calls_bypassed),
            "calls_patched": int(self._calls_patched),
        }


def patch_qwen35_attention(cfg: Qwen35AttentionPatchConfig) -> Qwen35AttentionPatcher:
    patcher = Qwen35AttentionPatcher(cfg)
    patcher.apply()
    return patcher
