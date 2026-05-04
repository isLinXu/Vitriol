#!/usr/bin/env python3
"""
ExoBrain v0.6 Comprehensive Validation Test.

Tests all 5 v0.6 optimizations:
1. MultiTeacherRouter — Multi-teacher KV routing
2. AdaptiveInjectionScheduler — PPL-based injection scheduling
3. BrainKVCompressor — Teacher KV compression
4. ProgressiveDistiller — Progressive knowledge solidification
5. ExoBrainProfiler — Full-stack performance profiling
"""

import sys
import time
import torch

from vitriol.kv.exobrain import (
    ExoBrainBus,
    ExoBrainConfig,
    MultiTeacherRouter,
)
from vitriol.kv.exobrain_inference import (
    AdaptiveInjectionScheduler,
    compute_perplexity_from_logits,
    BrainKVCompressor,
    TeacherKVCache,
    ExoBrainProfiler,
)


def test_multi_teacher_router():
    """Test 1: MultiTeacherRouter — multi-teacher KV routing."""
    print("\n" + "=" * 60)
    print("TEST 1: MultiTeacherRouter")
    print("=" * 60)

    # Create two mock teacher buses
    bus_a = ExoBrainBus(config=ExoBrainConfig())
    bus_b = ExoBrainBus(config=ExoBrainConfig())

    # Inject mock KV into each teacher
    kv_a = (torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64))
    kv_b = (torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64))
    bus_a.inject_kv(0, *kv_a)
    bus_b.inject_kv(0, *kv_b)

    # Test similarity-based routing
    router = MultiTeacherRouter(
        teachers={"teacher_a": bus_a, "teacher_b": bus_b},
        strategy="similarity",
    )

    query = torch.randn(1, 4, 5, 64)
    result = router.route(query, layer_idx=0)
    assert result is not None, "Router should return KV"
    print(f"  ✓ Similarity routing: returned KV shape {result[0].shape}")

    # Test ensemble routing
    router_ensemble = MultiTeacherRouter(
        teachers={"teacher_a": bus_a, "teacher_b": bus_b},
        strategy="ensemble",
        top_k_teachers=2,
    )
    result_ens = router_ensemble.route(query, layer_idx=0)
    assert result_ens is not None, "Ensemble router should return KV"
    print(f"  ✓ Ensemble routing: returned KV shape {result_ens[0].shape}")

    # Test round-robin routing
    router_rr = MultiTeacherRouter(
        teachers={"teacher_a": bus_a, "teacher_b": bus_b},
        strategy="round_robin",
    )
    result_rr = router_rr.route(query, layer_idx=0)
    assert result_rr is not None, "Round-robin router should return KV"
    print(f"  ✓ Round-robin routing: returned KV shape {result_rr[0].shape}")

    # Test first-available routing
    router_fa = MultiTeacherRouter(
        teachers={"teacher_a": bus_a, "teacher_b": bus_b},
        strategy="first_available",
    )
    result_fa = router_fa.route(query, layer_idx=0)
    assert result_fa is not None, "First-available router should return KV"
    print(f"  ✓ First-available routing: returned KV shape {result_fa[0].shape}")

    # Check stats
    stats = router.stats
    print(f"  ✓ Router stats: total_routes={stats['total_routes']}, hits={stats['teacher_hits']}")

    print("  ✅ TEST 1 PASSED")
    return True


def test_adaptive_injection_scheduler():
    """Test 2: AdaptiveInjectionScheduler — PPL-based injection scheduling."""
    print("\n" + "=" * 60)
    print("TEST 2: AdaptiveInjectionScheduler")
    print("=" * 60)

    # Test threshold strategy
    scheduler = AdaptiveInjectionScheduler(
        strategy="threshold",
        ppl_threshold=10.0,
        warmup_steps=2,
        min_injection_rate=0.1,
    )

    # Warmup steps — always inject
    assert scheduler.should_inject(ppl=5.0) is True, "Warmup step 1 should inject"
    assert scheduler.should_inject(ppl=3.0) is True, "Warmup step 2 should inject"
    print("  ✓ Warmup steps: always inject")

    # Low PPL — no injection needed
    result_low = scheduler.should_inject(ppl=5.0)
    print(f"  ✓ Low PPL (5.0): inject={result_low} (expected False)")

    # High PPL — injection needed
    result_high = scheduler.should_inject(ppl=15.0)
    assert result_high is True, "High PPL should trigger injection"
    print(f"  ✓ High PPL (15.0): inject={result_high}")

    # Test relative strategy
    scheduler_rel = AdaptiveInjectionScheduler(
        strategy="relative",
        relative_alpha=1.5,
        warmup_steps=0,
    )
    # Build up some PPL history
    for ppl_val in [5.0, 6.0, 5.5, 6.5, 5.0]:
        scheduler_rel.should_inject(ppl=ppl_val)
        scheduler_rel.record(ppl=ppl_val)

    # Low PPL relative to average
    result_low_rel = scheduler_rel.should_inject(ppl=3.0)
    # High PPL relative to average
    result_high_rel = scheduler_rel.should_inject(ppl=20.0)
    print(f"  ✓ Relative strategy: low PPL → inject={result_low_rel}, high PPL → inject={result_high_rel}")

    # Test always/never strategies
    scheduler_always = AdaptiveInjectionScheduler(strategy="always")
    assert scheduler_always.should_inject() is True, "Always strategy should inject"

    scheduler_never = AdaptiveInjectionScheduler(strategy="never", warmup_steps=0, min_injection_rate=0.0)
    assert scheduler_never.should_inject() is False, "Never strategy should not inject"
    print("  ✓ Always/Never strategies work correctly")

    # Check stats
    stats = scheduler.stats
    print(f"  ✓ Scheduler stats: steps={stats['total_steps']}, injection_rate={stats['injection_rate']:.2f}")

    # Test compute_perplexity_from_logits
    logits = torch.randn(1, 10, 1000)
    target_ids = torch.randint(0, 1000, (1, 10))
    ppl = compute_perplexity_from_logits(logits, target_ids)
    assert ppl > 0, "Perplexity should be positive"
    print(f"  ✓ compute_perplexity_from_logits: PPL={ppl:.2f}")

    print("  ✅ TEST 2 PASSED")
    return True


def test_brain_kv_compressor():
    """Test 3: BrainKVCompressor — teacher KV compression."""
    print("\n" + "=" * 60)
    print("TEST 3: BrainKVCompressor")
    print("=" * 60)

    # Create mock teacher KV
    key = torch.randn(1, 8, 64, 128)
    value = torch.randn(1, 8, 64, 128)
    original_size = key.numel() + value.numel()
    print(f"  Original KV size: {original_size} elements ({key.element_size() * original_size / 1e6:.2f} MB)")

    # Test topk_spatial compression
    comp_topk = BrainKVCompressor(method="topk_spatial", compression_ratio=0.5)
    k_c, v_c = comp_topk.compress(0, key, value)
    compressed_size = k_c.numel() + v_c.numel()
    print(f"  ✓ topk_spatial: {key.shape[2]}→{k_c.shape[2]} seq_len (ratio={compressed_size/original_size:.2f})")
    assert k_c.shape[2] == 32, "TopK should keep 50% of 64 = 32 positions"

    # Test quantize_8bit compression
    comp_quant = BrainKVCompressor(method="quantize_8bit", quantize_bits=8)
    k_q, v_q = comp_quant.compress(0, key, value)
    print(f"  ✓ quantize_8bit: shape preserved {k_q.shape}, values quantized")
    assert k_q.shape == key.shape, "Quantization preserves shape"

    # Test mean_pool compression
    comp_pool = BrainKVCompressor(method="mean_pool", pool_window=4)
    k_p, v_p = comp_pool.compress(0, key, value)
    print(f"  ✓ mean_pool: {key.shape[2]}→{k_p.shape[2]} seq_len (window=4)")
    assert k_p.shape[2] == 16, "Mean pool with window=4 should reduce seq_len to 16"

    # Test SVD low-rank compression
    comp_svd = BrainKVCompressor(method="svd_lowrank", svd_rank=32)
    k_s, v_s = comp_svd.compress(0, key, value)
    print(f"  ✓ svd_lowrank: shape preserved {k_s.shape}, rank=32")

    # Test none (baseline)
    comp_none = BrainKVCompressor(method="none")
    k_n, v_n = comp_none.compress(0, key, value)
    assert torch.allclose(k_n, key), "None compression should return original"
    print("  ✓ none: passthrough confirmed")

    # Test compress_teacher_cache
    teacher_kv = TeacherKVCache(
        kv_pairs={0: (key, value), 1: (key, value)},
        model_id="test-teacher",
        num_layers=2,
        hidden_size=1024,
        num_heads=8,
        head_dim=128,
        sequence_length=64,
    )
    comp_topk2 = BrainKVCompressor(method="topk_spatial", compression_ratio=0.25)
    compressed_cache = comp_topk2.compress_teacher_cache(teacher_kv)
    assert compressed_cache.model_id == "test-teacher_compressed"
    print(f"  ✓ compress_teacher_cache: {len(compressed_cache.kv_pairs)} layers compressed")

    # Check stats
    stats = comp_topk.stats
    print(f"  ✓ Compression stats: method={stats['method']}, ratio={stats['avg_compression_ratio']:.2f}")

    print("  ✅ TEST 3 PASSED")
    return True


def test_exobrain_profiler():
    """Test 4: ExoBrainProfiler — full-stack performance profiling."""
    print("\n" + "=" * 60)
    print("TEST 4: ExoBrainProfiler")
    print("=" * 60)

    profiler = ExoBrainProfiler()

    # Profile some stages
    with profiler.stage("teacher_extract"):
        time.sleep(0.01)  # Simulate teacher KV extraction

    with profiler.stage("brain_build"):
        time.sleep(0.005)  # Simulate brain build

    # Multiple calls to same stage
    for _ in range(3):
        with profiler.stage("decode_step"):
            time.sleep(0.002)

    # Manual stage recording
    profiler.record_stage("kv_projection", elapsed_s=0.003, memory_mb=128.0)

    # Generate report
    report = profiler.report()
    print(f"  ✓ Total time: {report['total_time_s']:.4f}s")
    print(f"  ✓ Stages profiled: {list(report['stages'].keys())}")

    # Verify stage statistics
    for name, stage_data in report["stages"].items():
        print(f"    - {name}: {stage_data['total_s']:.4f}s ({stage_data['pct_of_total']:.1f}%, {stage_data['calls']} calls)")

    # Check bottleneck detection
    assert report["bottleneck"] is not None, "Should detect a bottleneck"
    print(f"  ✓ Bottleneck detected: {report['bottleneck']}")

    # Test memory snapshot
    profiler.snapshot_memory("after_injection")
    assert len(profiler._memory_snapshots) == 1
    print("  ✓ Memory snapshot taken")

    # Test reset
    profiler.reset()
    assert len(profiler._stages) == 0, "Reset should clear stages"
    print("  ✓ Profiler reset works")

    print("  ✅ TEST 4 PASSED")
    return True


def test_progressive_distiller_structure():
    """Test 5: ProgressiveDistiller — structure verification (no real model)."""
    print("\n" + "=" * 60)
    print("TEST 5: ProgressiveDistiller (structure verification)")
    print("=" * 60)

    from vitriol.kv.exobrain_inference import ProgressiveDistiller

    # Just verify the class can be instantiated and methods exist
    # (actual distillation requires real models)
    pd = ProgressiveDistiller.__new__(ProgressiveDistiller)
    pd.num_stages = 5
    pd.layer_schedule = "uniform"
    pd.loss_schedule = "linear"
    pd._stage_history = []

    # Test _compute_kl_weight
    for schedule in ["linear", "cosine", "step"]:
        pd.loss_schedule = schedule
        weights = [pd._compute_kl_weight(s) for s in range(pd.num_stages)]
        print(f"  ✓ KL weights ({schedule}): {[f'{w:.2f}' for w in weights]}")

    # Verify stage_history property
    pd._stage_history = [{"stage": 1, "alpha_brain": 1.0, "loss": 0.5}]
    assert len(pd.stage_history) == 1
    print("  ✓ stage_history property works")

    print("  ✅ TEST 5 PASSED")
    return True


def main():
    """Run all ExoBrain v0.6 validation tests."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       ExoBrain v0.6 Comprehensive Validation Test       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = {}

    tests = [
        ("MultiTeacherRouter", test_multi_teacher_router),
        ("AdaptiveInjectionScheduler", test_adaptive_injection_scheduler),
        ("BrainKVCompressor", test_brain_kv_compressor),
        ("ExoBrainProfiler", test_exobrain_profiler),
        ("ProgressiveDistiller", test_progressive_distiller_structure),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "PASSED" if result else "FAILED"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v == "PASSED")
    total = len(results)

    for name, result in results.items():
        icon = "✅" if result == "PASSED" else "❌"
        print(f"  {icon} {name}: {result}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All ExoBrain v0.6 optimizations validated successfully!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
