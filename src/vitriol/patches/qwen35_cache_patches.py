from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from ..kv.codec import blockwise_minmax_quantize_dequantize
from ..kv.policy import KVLayerType, classify_kv_layer


def _levels_for_kv_format(fmt: str) -> int:
    if fmt == "turbo2":
        return 4
    if fmt == "turbo3":
        return 8
    if fmt == "turbo4":
        return 16
    if fmt == "q8_0":
        return 256
    if fmt == "q4_0":
        return 16
    return 8


@dataclass(frozen=True)
class Qwen35CachePatchConfig:
    enable_kv_quant: bool = True
    kv_format: str = "turbo3"
    block_size: int = 32


class Qwen35CachePatcher:
    def __init__(self, cfg: Qwen35CachePatchConfig) -> None:
        self.cfg = cfg
        self._orig_update: Optional[Callable[..., Any]] = None
        self._calls_total: int = 0
        self._calls_quantized: int = 0

    def apply(self) -> None:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        if getattr(m.Qwen3_5DynamicCache.update, "_vitriol_qwen35_cache_patched", False):
            return

        self._orig_update = m.Qwen3_5DynamicCache.update
        levels = _levels_for_kv_format(self.cfg.kv_format)
        bs = int(self.cfg.block_size)

        def wrapped(self_cache, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
            self._calls_total += 1

            if not self.cfg.enable_kv_quant:
                return self._orig_update(self_cache, key_states, value_states, layer_idx, cache_kwargs)

            if getattr(self_cache, "layer_types", None) is not None:
                if classify_kv_layer(self_cache, layer_idx) is not KVLayerType.FULL_ATTENTION:
                    return self._orig_update(self_cache, key_states, value_states, layer_idx, cache_kwargs)

            self._calls_quantized += 1
            key_q = blockwise_minmax_quantize_dequantize(key_states, levels=levels, block_size=bs)
            value_q = blockwise_minmax_quantize_dequantize(value_states, levels=levels, block_size=bs)
            return self._orig_update(self_cache, key_q, value_q, layer_idx, cache_kwargs)

        wrapped._vitriol_qwen35_cache_patched = True
        wrapped._vitriol_qwen35_cache_original = self._orig_update
        m.Qwen3_5DynamicCache.update = wrapped

    def restore(self) -> None:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        current = m.Qwen3_5DynamicCache.update
        original = getattr(current, "_vitriol_qwen35_cache_original", None)
        if getattr(current, "_vitriol_qwen35_cache_patched", False) and original is not None:
            m.Qwen3_5DynamicCache.update = original

    def stats(self) -> dict:
        return {
            "update_calls_total": int(self._calls_total),
            "update_calls_quantized": int(self._calls_quantized),
        }


def patch_qwen35_cache(cfg: Qwen35CachePatchConfig) -> Qwen35CachePatcher:
    patcher = Qwen35CachePatcher(cfg)
    patcher.apply()
    return patcher
