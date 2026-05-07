from vitriol.telemetry.kv_stats import merge_kv_stats


def test_merge_kv_stats_merges_dicts() -> None:
    out = merge_kv_stats({"a": 1}, {"b": 2})
    assert out["a"] == 1
    assert out["b"] == 2


def test_merge_kv_stats_last_write_wins_and_ignores_none() -> None:
    out = merge_kv_stats({"a": 1, "x": 1}, None, {"x": 2})
    assert out["a"] == 1
    assert out["x"] == 2

