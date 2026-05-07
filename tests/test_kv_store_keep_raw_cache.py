import torch

from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig


def test_keep_raw_cache_false_drops_raw_and_preserves_seq_len_and_bytes() -> None:
    torch.manual_seed(0)
    cfg = KVCacheStoreConfig(enable_turbo_quant=True, turbo_bits=3.5, quantized_kv_start=0, keep_raw_cache=False)
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    store.set_prefill(key, value)

    assert store._k_raw is None
    assert store._v_raw is None
    assert store._k_enc is not None
    assert store._v_enc is not None

    assert store.estimated_kv_bytes() > 0
    assert store.seq_len == 8


def test_keep_raw_cache_false_append_increments_seq_len() -> None:
    torch.manual_seed(1)
    cfg = KVCacheStoreConfig(enable_turbo_quant=True, turbo_bits=3.5, quantized_kv_start=0, keep_raw_cache=False)
    store = KVCacheStore(cfg)

    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)
    store.set_prefill(key, value)

    bytes_before = store.estimated_kv_bytes()

    key_new = torch.randn(1, 2, 4, 32)
    value_new = torch.randn(1, 2, 4, 32)
    store.append(key_new, value_new)

    assert store._k_raw is None
    assert store._v_raw is None
    assert store.seq_len == 12
    assert store.estimated_kv_bytes() >= bytes_before

