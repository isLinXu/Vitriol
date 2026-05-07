from vitriol.bench.runner import _benchmark_memory_stats


def test_benchmark_memory_stats_shape_is_stable() -> None:
    # 这是纯结构测试：不依赖真实模型推理，避免 CI 环境波动。
    # 只验证我们在 runner 里承诺输出的关键字段形状稳定。
    dummy = {"_final_past_key_values": None}
    out = _benchmark_memory_stats(dummy, backend=None, device=type("D", (), {"type": "cpu"})())
    assert "estimated_kv_bytes" in out
    assert "layer_stats" in out

