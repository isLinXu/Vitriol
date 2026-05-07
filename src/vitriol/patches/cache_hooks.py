from __future__ import annotations

from dataclasses import dataclass
import threading
import logging
from typing import Any, Callable, Optional, Type

import torch

from ..kv.backend import KVStoreBackend


logger = logging.getLogger(__name__)

_thread_local = threading.local()

_STATS_LOCK = threading.Lock()
_CACHE_HOOK_STATS: dict[str, int] = {}


def _bump_cache_hook_stat(name: str, delta: int = 1) -> None:
    """线程安全地累加 KV hook 统计计数（仅用于可观测性，禁止改变逻辑分支）。"""
    try:
        d = int(delta)
    except Exception:
        d = 1
    if d == 0:
        return
    with _STATS_LOCK:
        _CACHE_HOOK_STATS[name] = int(_CACHE_HOOK_STATS.get(name, 0)) + d


def get_cache_hook_stats(*, reset: bool = False) -> dict[str, int]:
    """获取 KV hook 统计快照；如 reset=True 则清空计数器。"""
    with _STATS_LOCK:
        snap = dict(_CACHE_HOOK_STATS)
        if reset:
            _CACHE_HOOK_STATS.clear()
    return snap


def reset_cache_hook_stats() -> None:
    """清空 KV hook 统计计数器。"""
    with _STATS_LOCK:
        _CACHE_HOOK_STATS.clear()


def _cache_position_query_len(cache_position: Any) -> int:
    if isinstance(cache_position, torch.Tensor):
        if cache_position.ndim == 0:
            return 1
        return int(cache_position.shape[0])
    if isinstance(cache_position, (int, float)):
        return 1
    return 1


@dataclass(frozen=True)
class CacheHookConfig:
    enabled: bool = True
    passthrough_update: bool = False
    auto_enable_mode: bool = False


class CacheHookPatcher:
    def __init__(self, cfg: CacheHookConfig, backend: KVStoreBackend) -> None:
        self.cfg = cfg
        self.backend = backend
        self._orig_update: Optional[Callable[..., Any]] = None
        self._orig_get_seq: Optional[Callable[..., Any]] = None
        self._orig_get_mask_sizes: Optional[Callable[..., Any]] = None
        self._target_cls: Optional[Type[Any]] = None

    def apply_to_class(self, cache_cls: Type[Any]) -> None:
        if not self.cfg.enabled:
            return
        if getattr(cache_cls.update, "_vitriol_cache_hook_patched", False):
            return

        self._target_cls = cache_cls
        self._orig_update = cache_cls.update
        self._orig_get_seq = cache_cls.get_seq_length
        self._orig_get_mask_sizes = getattr(cache_cls, "get_mask_sizes", None)

        def update_wrapped(
            self_cache,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
        ):
            _bump_cache_hook_stat("cache_update_calls")
            if self.cfg.auto_enable_mode and not getattr(self_cache, "_vitriol_kv_store_mode", False):
                setattr(self_cache, "_vitriol_kv_store_mode", True)
            mode = getattr(self_cache, "_vitriol_kv_store_mode", False)
            if not mode:
                return self._orig_update(self_cache, key_states, value_states, layer_idx, cache_kwargs)

            _thread_local.current_cache = self_cache

            info = dict(cache_kwargs or {})
            q_len = int(key_states.size(-2))
            info["q_len"] = q_len
            _bump_cache_hook_stat("write_kv_calls")
            self.backend.write_kv(self_cache, int(layer_idx), key_states, value_states, info)

            seq_lens = getattr(self_cache, "_vitriol_seq_lens", None)
            if seq_lens is None:
                size = len(getattr(self_cache, "layer_types", [])) or (int(layer_idx) + 1)
                seq_lens = [0 for _ in range(size)]
                setattr(self_cache, "_vitriol_seq_lens", seq_lens)

            if int(layer_idx) >= len(seq_lens):
                seq_lens.extend([0] * (int(layer_idx) - len(seq_lens) + 1))

            seq_lens[int(layer_idx)] += q_len
            if q_len > 1:
                _bump_cache_hook_stat("update_passthrough_prefill")
                return self._orig_update(self_cache, key_states, value_states, layer_idx, cache_kwargs)
            if self.cfg.passthrough_update:
                _bump_cache_hook_stat("update_passthrough_decode")
                return self._orig_update(self_cache, key_states, value_states, layer_idx, cache_kwargs)
            _bump_cache_hook_stat("update_short_circuit_decode")
            return key_states.contiguous(), value_states.contiguous()

        def get_seq_length_wrapped(self_cache, layer_idx: int | None = 0) -> int:
            mode = getattr(self_cache, "_vitriol_kv_store_mode", False)
            if not mode:
                return self._orig_get_seq(self_cache, layer_idx)

            seq_lens = getattr(self_cache, "_vitriol_seq_lens", None)
            if not seq_lens:
                return 0

            if layer_idx is None:
                for x in seq_lens:
                    if x:
                        return int(x)
                return 0

            idx = int(layer_idx)
            if idx < 0 or idx >= len(seq_lens):
                return 0
            return int(seq_lens[idx])

        def get_mask_sizes_wrapped(self_cache, cache_position: Any, layer_idx: int) -> tuple[int, int]:
            mode = getattr(self_cache, "_vitriol_kv_store_mode", False)
            if not mode or self._orig_get_mask_sizes is None:
                return self._orig_get_mask_sizes(self_cache, cache_position, layer_idx)

            seq_lens = getattr(self_cache, "_vitriol_seq_lens", None)
            past = 0
            if seq_lens:
                idx = int(layer_idx)
                if 0 <= idx < len(seq_lens):
                    past = int(seq_lens[idx])
            query_len = _cache_position_query_len(cache_position)
            return past + query_len, 0

        update_wrapped._vitriol_cache_hook_patched = True
        update_wrapped._vitriol_cache_hook_original = self._orig_update
        get_seq_length_wrapped._vitriol_cache_hook_patched = True
        get_seq_length_wrapped._vitriol_cache_hook_original = self._orig_get_seq
        if self._orig_get_mask_sizes is not None:
            get_mask_sizes_wrapped._vitriol_cache_hook_patched = True
            get_mask_sizes_wrapped._vitriol_cache_hook_original = self._orig_get_mask_sizes

        cache_cls.update = update_wrapped
        cache_cls.get_seq_length = get_seq_length_wrapped
        if self._orig_get_mask_sizes is not None:
            cache_cls.get_mask_sizes = get_mask_sizes_wrapped

    def restore(self) -> None:
        if self._target_cls is None:
            return
        cache_cls = self._target_cls
        if self._orig_update is not None and getattr(cache_cls.update, "_vitriol_cache_hook_patched", False):
            cache_cls.update = self._orig_update
        if self._orig_get_seq is not None and getattr(cache_cls.get_seq_length, "_vitriol_cache_hook_patched", False):
            cache_cls.get_seq_length = self._orig_get_seq
        if self._orig_get_mask_sizes is not None and getattr(cache_cls.get_mask_sizes, "_vitriol_cache_hook_patched", False):
            cache_cls.get_mask_sizes = self._orig_get_mask_sizes


class UniversalAttentionPatcher:
    def __init__(self, backend: KVStoreBackend) -> None:
        self.backend = backend
        import transformers.modeling_utils as mu
        registry = getattr(mu, "ALL_ATTENTION_FUNCTIONS", None)
        self._supported = registry is not None and hasattr(registry, "get_interface")
        self.orig_get_interface = registry.get_interface if self._supported else None
        self._patched = False
        self._warned_unsupported = False

    def apply(self) -> None:
        if not self._supported:
            _bump_cache_hook_stat("attention_hook_unsupported")
            if not self._warned_unsupported:
                self._warned_unsupported = True
                logger.warning(
                    "UniversalAttentionPatcher unsupported: transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS.get_interface not found"
                )
            return
        if self._patched:
            return

        import transformers.modeling_utils as mu

        def custom_get_interface(config_attn_implementation, eager_attention_forward):
            orig_interface = self.orig_get_interface(config_attn_implementation, eager_attention_forward)

            def custom_attention_forward(module, query_states, key_states, value_states, attention_mask, **kwargs):
                cache = getattr(_thread_local, "current_cache", None)
                q_len = query_states.size(-2)
                if cache is not None and q_len == 1 and getattr(cache, "_vitriol_kv_store_mode", False):
                    layer_idx = getattr(module, "layer_idx", None)
                    if layer_idx is not None:
                        _bump_cache_hook_stat("read_attention_attempt")
                        dropout = kwargs.get("dropout", 0.0)
                        scaling = kwargs.get("scaling", None)
                        is_causal = kwargs.get("is_causal", False)
                        if is_causal is None:
                            is_causal = getattr(module, "is_causal", True)
                        is_causal = bool(is_causal) and (query_states.shape[2] > 1) and (attention_mask is None)
                        info = {"dropout_p": dropout}
                        if "sliding_window" in kwargs:
                            info["sliding_window"] = kwargs.get("sliding_window")
                        try:
                            attn_output = self.backend.read_attention(
                                handle=cache,
                                layer_idx=layer_idx,
                                query=query_states,
                                attn_mask=attention_mask,
                                is_causal=is_causal,
                                scale=scaling,
                                info=info
                            )
                            _bump_cache_hook_stat("read_attention_hit")
                            attn_output = attn_output.transpose(1, 2).contiguous()
                            return attn_output, None
                        except Exception:
                            _bump_cache_hook_stat("read_attention_fallback")
                            logger.debug("Failed to call attention interface for cache hooks")
                return orig_interface(module, query_states, key_states, value_states, attention_mask, **kwargs)

            return custom_attention_forward

        mu.ALL_ATTENTION_FUNCTIONS.get_interface = custom_get_interface
        self._patched = True
        _bump_cache_hook_stat("attention_hook_enabled")

    def restore(self) -> None:
        if not self._supported:
            return
        if not self._patched:
            return
        import transformers.modeling_utils as mu
        mu.ALL_ATTENTION_FUNCTIONS.get_interface = self.orig_get_interface
        self._patched = False
