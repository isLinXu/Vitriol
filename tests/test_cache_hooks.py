import torch

from vitriol.kv.backend import KVStoreBackend
from vitriol.kv.cache_store import KVCacheStoreConfig
from vitriol.patches.cache_hooks import CacheHookConfig, CacheHookPatcher, _cache_position_query_len


def test_cache_position_query_len_handles_tensor_and_int() -> None:
    assert _cache_position_query_len(torch.tensor([4, 5])) == 2
    assert _cache_position_query_len(torch.tensor(3)) == 1
    assert _cache_position_query_len(7) == 1
    assert _cache_position_query_len(None) == 1


def test_cache_hook_get_mask_sizes_accepts_int_cache_position() -> None:
    class DummyCache:
        def __init__(self) -> None:
            self._vitriol_kv_store_mode = True
            self._vitriol_seq_lens = [5]
            self.layer_types = ["full_attention"]

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            return key_states, value_states

        def get_seq_length(self, layer_idx=0):
            return 0

        def get_mask_sizes(self, cache_position, layer_idx):
            return 0, 0

    backend = KVStoreBackend(KVCacheStoreConfig(enable_turbo_quant=False))
    patcher = CacheHookPatcher(CacheHookConfig(enabled=True), backend)
    patcher.apply_to_class(DummyCache)
    cache = DummyCache()

    assert cache.get_mask_sizes(3, 0) == (6, 0)
    assert cache.get_mask_sizes(torch.tensor([7]), 0) == (6, 0)

    patcher.restore()


def test_backend_stats_include_estimated_kv_bytes() -> None:
    class DummyHandle:
        pass

    handle = DummyHandle()
    backend = KVStoreBackend(KVCacheStoreConfig(enable_turbo_quant=True, turbo_k_format="turbo3", turbo_v_format="turbo3"))
    key = torch.zeros(1, 2, 4, 8, dtype=torch.float16)
    value = torch.zeros(1, 2, 4, 8, dtype=torch.float16)
    backend.write_kv(handle, 0, key, value, {"q_len": 4})

    stats = backend.stats(handle)
    assert stats["layers"] == 1
    assert stats["estimated_kv_bytes"] > 0
    assert stats["layer_stats"][0]["seq_len"] == 4
    assert stats["layer_stats"][0]["estimated_kv_bytes"] == stats["estimated_kv_bytes"]


def test_backend_write_kv_can_append_with_residual_qjl_enabled() -> None:
    class DummyHandle:
        pass

    handle = DummyHandle()
    backend = KVStoreBackend(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_k_format="turbo3",
            turbo_v_format="turbo3",
            enable_turbo_residual_qjl=True,
        )
    )
    key_prefill = torch.randn(1, 2, 4, 32, dtype=torch.float16)
    value_prefill = torch.randn(1, 2, 4, 32, dtype=torch.float16)
    key_decode = torch.randn(1, 2, 1, 32, dtype=torch.float16)
    value_decode = torch.randn(1, 2, 1, 32, dtype=torch.float16)

    backend.write_kv(handle, 0, key_prefill, value_prefill, {"q_len": 4})
    backend.write_kv(handle, 0, key_decode, value_decode, {"q_len": 1})

    stats = backend.stats(handle)
    assert stats["layers"] == 1
    assert stats["layer_stats"][0]["seq_len"] == 5
    assert stats["estimated_kv_bytes"] > 0
