"""Cache hooks must associate actual tensors and preserve native fallback history."""

import gc
import weakref
from functools import wraps
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import transformers.modeling_utils as mu

from vitriol.kv.backend import KVStoreBackend
from vitriol.kv.cache_store import KVCacheStoreConfig
from vitriol.patches.cache_hooks import CacheHookConfig, CacheHookPatcher, UniversalAttentionPatcher


class NativeCache:
    def __init__(self):
        self.layers = {}
        self._vitriol_kv_store_mode = True

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.layers:
            old_key, old_value = self.layers[layer_idx]
            key, value = torch.cat((old_key, key), -2), torch.cat((old_value, value), -2)
        self.layers[layer_idx] = key, value
        return key, value

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx][0].shape[-2] if layer_idx in self.layers else 0

    def get_mask_sizes(self, cache_position, layer_idx):
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    def reset(self):
        self.layers.clear()

    def crop(self, length):
        self.layers = {i: (k[..., :length, :], v[..., :length, :]) for i, (k, v) in self.layers.items()}

    def reorder_cache(self, indices):
        self.layers = {i: (k.index_select(0, indices), v.index_select(0, indices)) for i, (k, v) in self.layers.items()}


def native_attention(module, query, key, value, mask=None, **kwargs):
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask).transpose(1, 2), None


class LegacyRegistry:
    def get_interface(self, name, eager):
        return native_attention


@pytest.fixture(params=["legacy", "mapping"])
def hooks(request, monkeypatch):
    registry = LegacyRegistry() if request.param == "legacy" else {"sdpa": native_attention}
    monkeypatch.setattr(mu, "ALL_ATTENTION_FUNCTIONS", registry)
    backend = KVStoreBackend(KVCacheStoreConfig())
    attention = UniversalAttentionPatcher(backend)
    cache = CacheHookPatcher(CacheHookConfig(), backend)
    try:
        attention.apply()
        cache.apply_to_class(NativeCache)
        interface = registry.get_interface("sdpa", native_attention) if request.param == "legacy" else registry["sdpa"]
        yield backend, cache, attention, interface
    finally:
        cache.restore()
        attention.restore()


def tensors(batch=1, length=1):
    g = torch.Generator().manual_seed(12)
    return [torch.randn(batch, 2, length, 8, generator=g) for _ in range(3)]


def test_attention_reads_backend_for_registered_mapping_and_legacy_interface(hooks, monkeypatch):
    backend, _, _, interface = hooks
    cache = NativeCache()
    query, key, value = tensors()
    key, value = cache.update(key, value, 0)
    sentinel = torch.full_like(query, 9)
    monkeypatch.setattr(backend, "read_attention", lambda **kwargs: sentinel)
    output, _ = interface(SimpleNamespace(layer_idx=0), query, key, value, None)
    torch.testing.assert_close(output, sentinel.transpose(1, 2))


def test_decode_and_backend_failure_keep_complete_native_history(hooks, monkeypatch):
    backend, _, _, interface = hooks
    cache = NativeCache()
    _, key, value = tensors(length=3)
    cache.update(key, value, 0)
    query, new_key, new_value = tensors()
    expected_key, expected_value = torch.cat((key, new_key), -2), torch.cat((value, new_value), -2)
    key, value = cache.update(new_key, new_value, 0)

    def fail(**kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(backend, "read_attention", fail)
    output, _ = interface(SimpleNamespace(layer_idx=0), query, key, value, None)
    expected = native_attention(None, query, expected_key, expected_value)[0]
    torch.testing.assert_close(output, expected)
    assert cache.get_seq_length() == 4
    assert cache.layers[0][0].shape[-2] == 4


def test_interleaved_requests_use_their_own_returned_kv(hooks, monkeypatch):
    backend, _, _, interface = hooks
    first, second = NativeCache(), NativeCache()
    query, key, value = tensors()
    first_kv = first.update(key, value, 0)
    second_kv = second.update(key + 1, value + 1, 0)
    seen = []

    def read(**kwargs):
        seen.append(kwargs["handle"])
        return torch.zeros_like(query)

    monkeypatch.setattr(backend, "read_attention", read)
    interface(SimpleNamespace(layer_idx=0), query, *first_kv, None)
    interface(SimpleNamespace(layer_idx=0), query, *second_kv, None)
    assert seen == [first, second]


def test_attention_binding_is_single_use_and_never_routes_unrelated_tensors(hooks, monkeypatch):
    backend, _, _, interface = hooks
    query, key, value = tensors()
    cache = NativeCache()
    returned = cache.update(key, value, 0)
    calls = []

    def read(**kwargs):
        calls.append(kwargs["handle"])
        return torch.zeros_like(query)

    monkeypatch.setattr(backend, "read_attention", read)
    interface(SimpleNamespace(layer_idx=0), query, key.clone(), value.clone(), None)
    assert not calls
    interface(SimpleNamespace(layer_idx=0), query, *returned, None)
    interface(SimpleNamespace(layer_idx=0), query, *returned, None)
    assert calls == [cache]


def test_write_failure_preserves_native_result_and_disables_broken_layer(hooks, monkeypatch):
    backend, _, _, interface = hooks
    cache = NativeCache()
    query, key, value = tensors()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("write failed")

    monkeypatch.setattr(backend, "write_kv", fail)
    for _ in range(2):
        out_key, out_value = cache.update(key, value, 0)
        output, _ = interface(SimpleNamespace(layer_idx=0), query, out_key, out_value, None)
        torch.testing.assert_close(output, native_attention(None, query, out_key, out_value)[0])
    assert calls == [1]
    assert cache.get_seq_length() == 2


@pytest.mark.parametrize("mutation", ["reset", "crop", "reorder"])
def test_native_cache_mutation_resynchronizes_backend(hooks, mutation):
    backend, _, _, interface = hooks
    cache = NativeCache()
    _, key, value = tensors(batch=2, length=3)
    cache.update(key, value, 0)
    if mutation == "reset":
        cache.reset()
    elif mutation == "crop":
        cache.crop(1)
    else:
        cache.reorder_cache(torch.tensor([1, 0]))
    query, new_key, new_value = tensors(batch=2)
    key, value = cache.update(new_key, new_value, 0)
    output, _ = interface(SimpleNamespace(layer_idx=0), query, key, value, None)
    torch.testing.assert_close(output, native_attention(None, query, key, value)[0])
    assert backend.stats(cache)["seq_lens"][0] == cache.get_seq_length()


def test_restoration_leaves_native_cache_usable(hooks):
    _, cache_patcher, attention_patcher, _ = hooks
    cache = NativeCache()
    query, key, value = tensors(length=3)
    cache.update(key, value, 0)
    cache.update(key[..., :1, :], value[..., :1, :], 0)
    cache_patcher.restore()
    attention_patcher.restore()
    assert cache.get_seq_length() == 4
    returned = cache.update(key[..., :1, :], value[..., :1, :], 0)
    assert returned[0].shape[-2] == 5


def test_binding_does_not_keep_finished_cache_alive(hooks):
    _, _, _, _ = hooks
    cache = NativeCache()
    query, key, value = tensors()
    cache.update(key, value, 0)
    ref = weakref.ref(cache)
    del cache
    gc.collect()
    assert ref() is None


def test_conflicting_cache_patch_fails_explicitly(hooks):
    backend, owner, _, _ = hooks
    other = CacheHookPatcher(CacheHookConfig(), backend)
    current = NativeCache.update
    try:
        owner.apply_to_class(NativeCache)
        with pytest.raises(RuntimeError, match="already"):
            other.apply_to_class(NativeCache)
    finally:
        other.restore()
    assert NativeCache.update is current


def test_wrong_layer_and_changed_value_do_not_consume_binding(hooks, monkeypatch):
    backend, _, _, interface = hooks
    cache = NativeCache()
    query, key, value = tensors()
    key, value = cache.update(key, value, 0)
    seen = []

    def read(**kwargs):
        seen.append(kwargs["layer_idx"])
        return torch.zeros_like(query)

    monkeypatch.setattr(backend, "read_attention", read)
    interface(SimpleNamespace(layer_idx=1), query, key, value, None)
    interface(SimpleNamespace(layer_idx=0), query, key, value.clone(), None)
    assert not seen
    interface(SimpleNamespace(layer_idx=0), query, key, value, None)
    assert seen == [0]


def test_attaching_to_warm_native_cache_seeds_complete_history(hooks):
    backend, _, _, interface = hooks
    cache = NativeCache()
    _, key, value = tensors(length=3)
    cache._vitriol_kv_store_mode = False
    cache.update(key, value, 0)
    cache._vitriol_kv_store_mode = True
    query, new_key, new_value = tensors()
    key, value = cache.update(new_key, new_value, 0)
    output, _ = interface(SimpleNamespace(layer_idx=0), query, key, value, None)
    torch.testing.assert_close(output, native_attention(None, query, key, value)[0])
    assert backend.stats(cache)["seq_lens"] == {0: 4}


def test_attention_owner_restore_cannot_remove_someone_elses_registry_hook(hooks):
    backend, _, owner, _ = hooks
    other = UniversalAttentionPatcher(backend)
    registry = mu.ALL_ATTENTION_FUNCTIONS
    current = registry["sdpa"] if isinstance(registry, dict) else registry.get_interface
    try:
        owner.apply()
        with pytest.raises(RuntimeError, match="already"):
            other.apply()
    finally:
        other.restore()
    assert (registry["sdpa"] if isinstance(registry, dict) else registry.get_interface) is current


def test_combined_install_rolls_back_attention_if_cache_class_is_owned(monkeypatch):
    from vitriol.bench import runner

    registry = {"sdpa": native_attention}
    monkeypatch.setattr(mu, "ALL_ATTENTION_FUNCTIONS", registry)
    monkeypatch.setattr(runner, "DynamicCache", NativeCache)
    owner = CacheHookPatcher(CacheHookConfig(), KVStoreBackend(KVCacheStoreConfig()))
    owner.apply_to_class(NativeCache)
    installed = NativeCache.update
    try:
        with pytest.raises(RuntimeError, match="already"):
            runner._apply_vitriol_universal(KVCacheStoreConfig(), 0)
        assert NativeCache.update is installed
        assert registry["sdpa"] is native_attention
        # A failed install must also release registry ownership.
        attention = UniversalAttentionPatcher(owner.backend)
        attention.apply()
        attention.restore()
    finally:
        owner.restore()


def test_unsupported_registry_fails_before_mutating_cache_class(monkeypatch):
    from vitriol.bench import runner

    monkeypatch.setattr(mu, "ALL_ATTENTION_FUNCTIONS", SimpleNamespace())
    monkeypatch.setattr(runner, "DynamicCache", NativeCache)
    original = NativeCache.update
    with pytest.raises(RuntimeError, match="Unsupported"):
        runner._apply_vitriol_universal(KVCacheStoreConfig(), 0)
    assert NativeCache.update is original


def test_failed_native_mutation_invalidates_backend_state(monkeypatch):
    def reset_then_fail(self):
        self.layers.clear()
        raise RuntimeError("native operation failed")

    monkeypatch.setattr(NativeCache, "reset", reset_then_fail)
    backend = KVStoreBackend(KVCacheStoreConfig())
    patcher = CacheHookPatcher(CacheHookConfig(), backend)
    try:
        patcher.apply_to_class(NativeCache)
        cache = NativeCache()
        _, key, value = tensors(length=3)
        cache.update(key, value, 0)
        with pytest.raises(RuntimeError, match="native operation failed"):
            cache.reset()
        assert backend.stats(cache)["layers"] == 0
    finally:
        patcher.restore()


def test_external_registry_wrapper_survives_restore_and_reinstallation(hooks, monkeypatch):
    backend, _, owner, _ = hooks
    registry = mu.ALL_ATTENTION_FUNCTIONS
    installed = registry["sdpa"] if isinstance(registry, dict) else registry.get_interface

    @wraps(installed)
    def foreign(*args, **kwargs):
        return installed(*args, **kwargs)

    if isinstance(registry, dict):
        registry["sdpa"] = foreign
    else:
        registry.get_interface = foreign
    owner.restore()
    assert (registry["sdpa"] if isinstance(registry, dict) else registry.get_interface) is foreign
    owner.apply()
    interface = registry["sdpa"] if isinstance(registry, dict) else registry.get_interface("sdpa", native_attention)
    cache = NativeCache()
    query, key, value = tensors()
    key, value = cache.update(key, value, 0)
    monkeypatch.setattr(backend, "read_attention", lambda **kwargs: torch.full_like(query, 17))
    torch.testing.assert_close(interface(SimpleNamespace(layer_idx=0), query, key, value, None)[0],
                               torch.full_like(query.transpose(1, 2), 17))


def test_external_cache_wrapper_survives_restore_and_reinstallation(hooks):
    _, owner, _, _ = hooks
    installed = NativeCache.update

    @wraps(installed)
    def foreign(*args, **kwargs):
        return installed(*args, **kwargs)

    NativeCache.update = foreign
    try:
        owner.restore()
        assert NativeCache.update is foreign
        owner.apply_to_class(NativeCache)
        cache = NativeCache()
        _, key, value = tensors(length=3)
        cache.update(key, value, 0)
        assert cache.get_seq_length() == 3
        owner.restore()
        assert NativeCache.update is foreign
    finally:
        # Remove the test's own wrapper, restoring the original native descriptor.
        NativeCache.update = installed._vitriol_cache_hook_original


def test_benchmark_reports_native_retention_and_payload_scope(hooks):
    from vitriol.bench.runner import _benchmark_memory_stats

    backend, _, _, _ = hooks
    cache = NativeCache()
    _, key, value = tensors(length=3)
    cache.update(key, value, 0)
    stats = _benchmark_memory_stats({"_final_past_key_values": cache}, backend, torch.device("cpu"))
    assert stats["native_cache_retained"] is True
    assert stats["estimated_kv_bytes_scope"] == "vitriol_store_payload_excludes_native_cache"
    assert stats["fallback_layers"] == []


def test_reading_unknown_layer_does_not_allocate_an_empty_store():
    backend = KVStoreBackend(KVCacheStoreConfig())
    cache = NativeCache()
    with pytest.raises(RuntimeError, match="has not been written"):
        backend.read_attention(cache, 0, tensors()[0], None, False, None, {})
    assert not hasattr(cache, "_vitriol_kv_stores")


def test_native_window_mask_offset_is_not_replaced_by_logical_sequence_length():
    class WindowCache(NativeCache):
        def get_mask_sizes(self, cache_position, layer_idx):
            assert cache_position.shape == (1,)
            return 4, 17

    patcher = CacheHookPatcher(CacheHookConfig(), KVStoreBackend(KVCacheStoreConfig()))
    try:
        patcher.apply_to_class(WindowCache)
        cache = WindowCache()
        cache._vitriol_seq_lens = [20]
        assert cache.get_mask_sizes(20, 0) == (4, 17)
    finally:
        patcher.restore()
