import pytest


def test_cache_hook_stats_api_exists_and_resets():
    from vitriol.patches import cache_hooks

    assert hasattr(cache_hooks, "get_cache_hook_stats")
    assert hasattr(cache_hooks, "reset_cache_hook_stats")

    cache_hooks.reset_cache_hook_stats()
    stats0 = cache_hooks.get_cache_hook_stats()
    assert isinstance(stats0, dict)

    # Manually bump counters through helper (should exist)
    cache_hooks._bump_cache_hook_stat("read_attention_hit", 1)
    stats1 = cache_hooks.get_cache_hook_stats()
    assert stats1["read_attention_hit"] == stats0.get("read_attention_hit", 0) + 1

    cache_hooks.reset_cache_hook_stats()
    stats2 = cache_hooks.get_cache_hook_stats()
    assert stats2.get("read_attention_hit", 0) == 0


def test_clear_vitriol_kv_removes_handle_fields():
    from vitriol.kv.utils import clear_vitriol_kv

    class Handle:
        pass

    h = Handle()
    h._vitriol_kv_stores = {0: object()}
    h._vitriol_seq_lens = [123]
    h._vitriol_kv_store_mode = True

    clear_vitriol_kv(h)
    assert not hasattr(h, "_vitriol_kv_stores")
    assert not hasattr(h, "_vitriol_seq_lens")
    assert not hasattr(h, "_vitriol_kv_store_mode")

