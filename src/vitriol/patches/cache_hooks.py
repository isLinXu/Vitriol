from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import MutableMapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Type

import torch

from ..kv.backend import KVStoreBackend

logger = logging.getLogger(__name__)

_thread_local = threading.local()

_HOOK_LOCK = threading.RLock()
_ATTENTION_OWNERS: dict[int, Any] = {}

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
    """Configuration for cache hook patching."""
    enabled: bool = True
    # Retained for compatibility; generic hooks always preserve native history.
    passthrough_update: bool = True
    auto_enable_mode: bool = False


@dataclass
class _AttentionBinding:
    cache: Any
    key: Any
    value: Any
    backend: Any
    layer_idx: int
    epoch: int
    state: Any
    consumed: bool = False


def _invalidate_cache(handle: Any) -> None:
    handle._vitriol_kv_stores = {}
    handle._vitriol_disabled_layers = set()
    handle._vitriol_cache_epoch = getattr(handle, "_vitriol_cache_epoch", 0) + 1


class CacheHookPatcher:
    """Mirror native KV into a backend while retaining complete fallback history.

    The legacy passthrough_update=False option no longer suppresses native
    updates: without model-specific routing that can silently drop decode KV.
    """

    def __init__(self, cfg: CacheHookConfig, backend: KVStoreBackend) -> None:
        self.cfg = cfg
        self.backend = backend
        self._target_cls: Optional[Type[Any]] = None
        self._state = None
        self._installed: dict[str, Any] = {}
        self._descriptors: dict[str, tuple[bool, Any]] = {}

    def apply_to_class(self, cache_cls: Type[Any]) -> None:
        with _HOOK_LOCK:
            self._apply_to_class(cache_cls)

    def _apply_to_class(self, cache_cls: Type[Any]) -> None:
        if not self.cfg.enabled:
            return
        if self._target_cls is not None:
            if self._target_cls is cache_cls:
                return
            raise RuntimeError("Cache hook is already installed on another class")
        current = cache_cls.update
        if getattr(current, "_vitriol_cache_hook_patched", False) and getattr(
            current, "_vitriol_cache_hook_is_active", lambda: True,
        )():
            raise RuntimeError("Another cache hook is already active on this class")
        if not self.cfg.passthrough_update:
            logger.warning("Generic cache hooks retain native KV history even when passthrough_update=False")

        state = SimpleNamespace(active=True)
        original_update = cache_cls.update
        backend, cfg = self.backend, self.cfg

        def update_wrapped(handle, key_states, value_states, layer_idx, cache_kwargs=None):
            if not state.active:
                return original_update(handle, key_states, value_states, layer_idx, cache_kwargs)
            _bump_cache_hook_stat("cache_update_calls")
            # Compatibility placeholder only; routing never uses a thread-global handle.
            _thread_local.current_cache = None
            key, value = original_update(handle, key_states, value_states, layer_idx, cache_kwargs)
            if cfg.auto_enable_mode:
                handle._vitriol_kv_store_mode = True
            if not getattr(handle, "_vitriol_kv_store_mode", False):
                _invalidate_cache(handle)
                return key, value

            if getattr(handle, "_vitriol_cache_backend", None) is not backend:
                _invalidate_cache(handle)
                handle._vitriol_cache_backend = backend
            handle._vitriol_native_cache_retained = True
            handle._vitriol_cache_epoch = getattr(handle, "_vitriol_cache_epoch", 0) + 1
            disabled = getattr(handle, "_vitriol_disabled_layers", set())
            idx = int(layer_idx)
            if idx in disabled:
                return key, value
            q_len = int(key_states.size(-2))
            info = dict(cache_kwargs or {})
            info.update(q_len=q_len, native_key=key, native_value=value)
            _bump_cache_hook_stat("write_kv_calls")
            try:
                backend.write_kv(handle, idx, key_states, value_states, info)
            except Exception:
                disabled.add(idx)
                handle._vitriol_disabled_layers = disabled
                getattr(handle, "_vitriol_kv_stores", {}).pop(idx, None)
                _bump_cache_hook_stat("write_kv_fallback")
                logger.warning("KV backend write failed at layer %s; retaining native attention", idx, exc_info=True)
                return key, value

            _bump_cache_hook_stat("update_passthrough_prefill" if q_len > 1 else "update_passthrough_decode")
            try:
                key._vitriol_attention_binding = _AttentionBinding(
                    weakref.ref(handle), weakref.ref(key), weakref.ref(value), backend,
                    idx, handle._vitriol_cache_epoch, state,
                )
            except (TypeError, AttributeError):
                _bump_cache_hook_stat("attention_binding_unsupported")
            return key, value

        methods = {"update": update_wrapped}
        original_mask = getattr(cache_cls, "get_mask_sizes", None)
        if original_mask is not None:
            def get_mask_sizes_wrapped(handle, cache_position, layer_idx):
                if state.active:
                    if isinstance(cache_position, (int, float)):
                        cache_position = torch.tensor([int(cache_position)], dtype=torch.long)
                    elif isinstance(cache_position, torch.Tensor) and cache_position.ndim == 0:
                        cache_position = cache_position.unsqueeze(0)
                # Preserve native sliding/static cache offsets and lengths.
                return original_mask(handle, cache_position, layer_idx)
            methods["get_mask_sizes"] = get_mask_sizes_wrapped

        def mutation_wrapper(original):
            def wrapped(handle, *args, **kwargs):
                try:
                    return original(handle, *args, **kwargs)
                finally:
                    # A native method may mutate tensors before raising.
                    if state.active:
                        _invalidate_cache(handle)
            return wrapped

        for name in ("reset", "crop", "reorder_cache", "batch_select_indices", "batch_repeat_interleave"):
            original = getattr(cache_cls, name, None)
            if original is not None:
                methods[name] = mutation_wrapper(original)
        self._target_cls, self._state = cache_cls, state
        try:
            for name, wrapper in methods.items():
                self._descriptors[name] = (name in cache_cls.__dict__, cache_cls.__dict__.get(name))
                wrapper._vitriol_cache_hook_patched = True
                wrapper._vitriol_cache_hook_is_active = lambda: state.active
                wrapper._vitriol_cache_hook_original = getattr(cache_cls, name)
                setattr(cache_cls, name, wrapper)
                self._installed[name] = wrapper
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        with _HOOK_LOCK:
            if self._state is not None:
                self._state.active = False
            cls = self._target_cls
            if cls is not None:
                for name, wrapper in self._installed.items():
                    if getattr(cls, name) is wrapper:
                        was_local, original = self._descriptors[name]
                        if was_local:
                            setattr(cls, name, original)
                        else:
                            delattr(cls, name)
            self._installed.clear()
            self._descriptors.clear()
            self._target_cls = None
            self._state = None
            _thread_local.current_cache = None


class UniversalAttentionPatcher:
    """Route single-use K/V bindings through a backend, with native fallback."""

    def __init__(self, backend: KVStoreBackend) -> None:
        self.backend = backend
        self._patched = False
        self._registry = None
        self._state = None
        self._installed: dict[str, Any] = {}
        self._originals: dict[str, tuple[bool, Any]] = {}
        self.orig_get_interface = None

    def _wrap(self, original, state):
        backend = self.backend

        def attention(module, query_states, key_states, value_states, attention_mask=None, **kwargs):
            binding = getattr(key_states, "_vitriol_attention_binding", None)
            cache = binding.cache() if isinstance(binding, _AttentionBinding) else None
            matches = (
                state.active and isinstance(binding, _AttentionBinding) and binding.state.active
                and not binding.consumed and cache is not None and binding.backend is backend
                and binding.key() is key_states and binding.value() is value_states
                and binding.layer_idx == getattr(module, "layer_idx", None)
                and binding.epoch == getattr(cache, "_vitriol_cache_epoch", None)
                and getattr(cache, "_vitriol_kv_store_mode", False)
            )
            if matches:
                binding.consumed = True
                if query_states.size(-2) == 1 and binding.layer_idx not in getattr(cache, "_vitriol_disabled_layers", set()):
                    _bump_cache_hook_stat("read_attention_attempt")
                    info = {"dropout_p": kwargs.get("dropout", 0.0)}
                    if "sliding_window" in kwargs:
                        info["sliding_window"] = kwargs["sliding_window"]
                    try:
                        output = backend.read_attention(
                            handle=cache, layer_idx=binding.layer_idx, query=query_states,
                            attn_mask=attention_mask, is_causal=False,
                            scale=kwargs.get("scaling"), info=info,
                        )
                        _bump_cache_hook_stat("read_attention_hit")
                        return output.transpose(1, 2).contiguous(), None
                    except Exception:
                        disabled = getattr(cache, "_vitriol_disabled_layers", set())
                        disabled.add(binding.layer_idx)
                        cache._vitriol_disabled_layers = disabled
                        _bump_cache_hook_stat("read_attention_fallback")
                        logger.warning("KV backend attention failed; using complete native KV", exc_info=True)
            return original(module, query_states, key_states, value_states, attention_mask, **kwargs)

        return attention

    def apply(self) -> None:
        with _HOOK_LOCK:
            if self._patched:
                return
            import transformers.modeling_utils as mu
            registry = getattr(mu, "ALL_ATTENTION_FUNCTIONS", None)
            mapping = isinstance(registry, MutableMapping)
            legacy = registry is not None and callable(getattr(registry, "get_interface", None))
            if not mapping and not legacy:
                _bump_cache_hook_stat("attention_hook_unsupported")
                raise RuntimeError("Unsupported Transformers attention registry")
            if id(registry) in _ATTENTION_OWNERS:
                raise RuntimeError("Another attention hook is already active on this registry")
            self._registry = registry
            state = self._state = SimpleNamespace(active=True)
            _ATTENTION_OWNERS[id(registry)] = self
            self._patched = True
            try:
                if mapping:
                    local = getattr(registry, "_local_mapping", registry)
                    for name, original in list(registry.items()):
                        if not callable(original):
                            continue
                        wrapper = self._wrap(original, state)
                        self._originals[name] = (name in local, original)
                        registry[name] = wrapper
                        self._installed[name] = wrapper
                else:
                    original_get = registry.get_interface
                    self.orig_get_interface = original_get

                    def get_interface(*args, **kwargs):
                        original = original_get(*args, **kwargs)
                        return self._wrap(original, state) if state.active else original

                    self._originals["get_interface"] = ("get_interface" in vars(registry), original_get)
                    registry.get_interface = get_interface
                    self._installed["get_interface"] = get_interface
            except Exception:
                self.restore()
                raise
            _bump_cache_hook_stat("attention_hook_enabled")

    def restore(self) -> None:
        with _HOOK_LOCK:
            if not self._patched:
                return
            self._state.active = False
            registry = self._registry
            for name, wrapper in self._installed.items():
                was_local, original = self._originals[name]
                if isinstance(registry, MutableMapping):
                    if registry.get(name) is wrapper:
                        if was_local:
                            registry[name] = original
                        else:
                            del registry[name]
                elif getattr(registry, name) is wrapper:
                    if was_local:
                        setattr(registry, name, original)
                    else:
                        delattr(registry, name)
            if _ATTENTION_OWNERS.get(id(registry)) is self:
                del _ATTENTION_OWNERS[id(registry)]
            self._installed.clear()
            self._originals.clear()
            self._registry = None
            self._patched = False
