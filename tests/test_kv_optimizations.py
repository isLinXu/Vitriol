#!/usr/bin/env python3
"""
KV cache optimization integration tests.

Covers three modules:
- layer_adaptive
- temporal_pooling
- hybrid_pipeline
"""

import sys
import time
import traceback

import torch


# ─────────────────────────────────────────────
# 1) Layer-aware adaptive bit allocation
# ─────────────────────────────────────────────
from vitriol.kv.layer_adaptive import (
    LayerAdaptiveBitAllocator,
    LayerAdaptiveConfig,
)


def test_layer_adaptive_basic():
    """Basic: allocator returns per-head bitwidth assignments."""
    config = LayerAdaptiveConfig(
        target_avg_bits=3.0,
        min_bits=1.5,
        max_bits=5.0,
        depth_profile="u_shape",
    )
    allocator = LayerAdaptiveBitAllocator(config)

    # Mock 4D tensors: [batch, heads, seq, dim]
    batch, heads, seq, dim = 2, 8, 32, 16
    query = torch.randn(batch, heads, seq, dim)
    key = torch.randn(batch, heads, seq, dim)
    value = torch.randn(batch, heads, seq, dim)

    # Test a mid-layer index
    layer_idx = 12
    total_layers = 32
    k_bits, v_bits, report = allocator.allocate(
        query=query, key=key, value=value,
        layer_idx=layer_idx, total_layers=total_layers,
    )

    assert k_bits.shape == (batch, heads), f"Expected ({batch},{heads}), got {k_bits.shape}"
    assert v_bits.shape == (batch, heads), f"Expected ({batch},{heads}), got {v_bits.shape}"
    assert k_bits.min() >= config.min_bits, f"K min bits {k_bits.min():.2f} < {config.min_bits}"
    assert k_bits.max() <= config.max_bits, f"K max bits {k_bits.max():.2f} > {config.max_bits}"

    # Report should include key diagnostics
    assert report["mode"] == "layer_adaptive"
    assert "depth_weight" in report
    assert "avg_k_bits" in report
    assert "actual_avg_bits" in report

    print(f"  ✅ Layer-Adaptive: k_avg={report['avg_k_bits']:.2f}, v_avg={report['avg_v_bits']:.2f}, "
          f"actual_avg={report['actual_avg_bits']:.2f}, depth_w={report['depth_weight']:.3f}")


def test_layer_adaptive_depth_profile():
    """Effect of different depth profiles on bit allocation."""
    allocator = LayerAdaptiveBitAllocator()
    batch, heads, seq, dim = 1, 4, 16, 8
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    total_layers = 16
    results = {}
    for profile in ["u_shape", "decay", "inv_decay", "uniform"]:
        cfg = LayerAdaptiveConfig(depth_profile=profile)
        alloc = LayerAdaptiveBitAllocator(cfg)
        # Compare shallow vs deep layers
        shallow_k, _, _ = alloc.allocate(q, k, v, layer_idx=0, total_layers=total_layers)
        deep_k, _, _ = alloc.allocate(q, k, v, layer_idx=total_layers - 1, total_layers=total_layers)
        results[profile] = (shallow_k.mean().item(), deep_k.mean().item())

    for profile, (shallow, deep) in results.items():
        print(f"    {profile}: shallow={shallow:.2f}, deep={deep:.2f}")

    print("  ✅ Depth profiles: all 4 tested")


def test_layer_adaptive_flat_input():
    """3D flat input (TurboQuant flow): should use depth_only_flat mode."""
    allocator = LayerAdaptiveBitAllocator()
    # 3D input: [batch*heads, seq, dim]
    q_flat = torch.randn(8, 16, 32)
    k_flat = torch.randn(8, 16, 32)
    v_flat = torch.randn(8, 16, 32)

    k_bits, v_bits, report = allocator.allocate(q_flat, k_flat, v_flat, layer_idx=5, total_layers=12)
    assert report["mode"] == "depth_only_flat", f"Expected depth_only_flat, got {report['mode']}"
    assert k_bits.shape == (8,), f"Expected (8,), got {k_bits.shape}"
    print(f"  ✅ Flat input mode: k_avg={report['avg_k_bits']:.2f}, v_avg={report['avg_v_bits']:.2f}")


def test_layer_adaptive_all_layers():
    """allocate_all_layers: allocate bitwidths for all layers in one call."""
    allocator = LayerAdaptiveBitAllocator(LayerAdaptiveConfig(target_avg_bits=3.0))
    batch, heads, seq, dim = 1, 4, 16, 8
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    total_layers = 8
    results = allocator.allocate_all_layers(q, k, v, total_layers)

    assert len(results) == total_layers
    for idx, (k_bits, v_bits, report) in enumerate(results):
        assert report["layer_idx"] == idx
    print(f"  ✅ All layers: {total_layers} layers allocated, "
          f"avg_bits range: [{min(r[2]['actual_avg_bits'] for r in results):.2f}, "
          f"{max(r[2]['actual_avg_bits'] for r in results):.2f}]")


# ─────────────────────────────────────────────
# 2) Temporal importance pooling
# ─────────────────────────────────────────────
from vitriol.kv.temporal_pooling import (
    TemporalPoolingConfig,
    temporal_importance_attention,
    create_temporal_pooling_config_from_preset,
)


def test_temporal_pooling_basic():
    """Basic: soft-gating as an alternative to hard threshold truncation."""
    batch, heads, seq_len, head_dim = 2, 8, 64, 32
    Q = torch.randn(batch, heads, 1, head_dim)       # decode: 1 query token
    K = torch.randn(batch, heads, seq_len, head_dim)
    V = torch.randn(batch, heads, seq_len, head_dim)

    config = TemporalPoolingConfig(
        temporal_decay=0.5,
        temperature=0.1,
        min_attention_mass=0.95,
        adaptive_threshold=True,
        enable_temporal_decay=True,
    )

    output, report = temporal_importance_attention(Q, K, V, config)

    assert output.shape == (batch, heads, 1, head_dim), \
        f"Expected ({batch},{heads},1,{head_dim}), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert "sparsity" in report
    assert "avg_soft_mask" in report

    print(f"  ✅ Temporal Pooling: output shape={output.shape}, "
          f"sparsity={report['sparsity']:.2%}, "
          f"avg_soft_mask={report['avg_soft_mask']:.4f}")


def test_temporal_pooling_vs_hard_threshold():
    """Compare: soft-gate vs hard threshold; soft-gate should preserve more signal."""
    torch.manual_seed(42)
    batch, heads, seq_len, head_dim = 1, 4, 128, 16
    Q = torch.randn(batch, heads, 1, head_dim)
    K = torch.randn(batch, heads, seq_len, head_dim)
    V = torch.randn(batch, heads, seq_len, head_dim)

    # Baseline attention
    scale = 1.0 / (head_dim ** 0.5)
    weights = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)

    # Hard threshold (Sparse-V approximation)
    hard_mask = (weights > 0.01).float()
    hard_output = (weights * hard_mask) @ V
    hard_sparsity = (hard_mask == 0).float().mean().item()

    # Soft gate (TIP)
    config = TemporalPoolingConfig(temperature=0.05, adaptive_threshold=True)
    soft_output, report = temporal_importance_attention(Q, K, V, config)

    # Soft gate should not introduce abrupt zeroing artifacts
    print(f"  ✅ Soft vs Hard: hard_sparsity={hard_sparsity:.2%}, "
          f"tip_sparsity={report['sparsity']:.2%}, "
          f"hard_norm={hard_output.norm():.4f}, soft_norm={soft_output.norm():.4f}")


def test_temporal_pooling_presets():
    """Preset config construction."""
    presets = {}
    for preset in ["conservative", "balanced", "aggressive", "ultra_long"]:
        config = create_temporal_pooling_config_from_preset(preset)
        presets[preset] = config
        assert config is not None

    print(f"  ✅ Presets: "
          f"conservative(decay={presets['conservative'].temporal_decay:.1f}), "
          f"balanced(decay={presets['balanced'].temporal_decay:.1f}), "
          f"aggressive(decay={presets['aggressive'].temporal_decay:.1f}), "
          f"ultra_long(pooling={presets['ultra_long'].enable_pooling})")


def test_temporal_pooling_no_decay():
    """Disable temporal decay: behavior should be close to baseline attention."""
    batch, heads, seq_len, head_dim = 1, 4, 32, 8
    Q = torch.randn(batch, heads, 1, head_dim)
    K = torch.randn(batch, heads, seq_len, head_dim)
    V = torch.randn(batch, heads, seq_len, head_dim)

    config = TemporalPoolingConfig(enable_temporal_decay=False)
    output, report = temporal_importance_attention(Q, K, V, config)

    assert not torch.isnan(output).any()
    # When enable_temporal_decay=False, report should not mark temporal_decay_applied
    assert "temporal_decay_applied" not in report or report["temporal_decay_applied"] is not True
    print(f"  ✅ No-decay mode: sparsity={report['sparsity']:.2%}")


# ─────────────────────────────────────────────
# 3) Hybrid pipeline + sliding window + zero-copy decode
# ─────────────────────────────────────────────
from vitriol.kv.hybrid_pipeline import (
    HybridKVCacheStore,
    HybridPipelineConfig,
    SlidingWindowConfig,
    SlidingWindowEvictor,
    ZeroCopyDecodeCache,
)


def test_sliding_window_evictor():
    """Sliding-window evictor."""
    config = SlidingWindowConfig(
        max_seq_len=64,
        attention_based=False,
        min_recent_tokens=16,
        eviction_chunk_size=16,
    )
    evictor = SlidingWindowEvictor(config)

    # 128 tokens -> should evict down to 64
    seq_len = 128
    should = evictor.should_evict(seq_len, "full_attention")
    assert should, "Should need eviction at 128 tokens"

    keep_indices = evictor.compute_eviction_indices(seq_len)
    assert len(keep_indices) <= config.max_seq_len, \
        f"Kept {len(keep_indices)} > max {config.max_seq_len}"

    # Most recent tokens must be preserved
    recent_indices = set(range(seq_len - config.min_recent_tokens, seq_len))
    kept_set = set(keep_indices.tolist())
    assert recent_indices.issubset(kept_set), "Recent tokens not preserved!"

    # Apply eviction
    K = torch.randn(1, 4, seq_len, 16)
    V = torch.randn(1, 4, seq_len, 16)
    K_evicted, V_evicted = evictor.evict_kv(K, V, keep_indices)
    assert K_evicted.size(-2) == len(keep_indices)

    stats = evictor.stats
    assert stats["eviction_count"] == 1
    print(f"  ✅ Sliding Window: {seq_len}→{len(keep_indices)} tokens, "
          f"evicted={stats['total_evicted_tokens']}")


def test_sliding_window_attention_based():
    """Attention-based eviction."""
    config = SlidingWindowConfig(
        max_seq_len=32,
        attention_based=True,
        min_recent_tokens=8,
    )
    evictor = SlidingWindowEvictor(config)
    seq_len = 64

    # Mock attention weights
    attn_weights = torch.rand(1, 4, 1, seq_len)
    attn_weights = torch.softmax(attn_weights, dim=-1)

    keep_indices = evictor.compute_eviction_indices(seq_len, attn_weights)
    assert len(keep_indices) <= config.max_seq_len
    print(f"  ✅ Attention-based eviction: {seq_len}→{len(keep_indices)} tokens")


def test_zero_copy_decode_cache():
    """Zero-copy decode cache."""
    cache = ZeroCopyDecodeCache()

    # Mock decode function
    def decode_fn(encoded):
        return encoded.float()

    # Store and retrieve
    K = torch.randn(1, 8, 64, 32)
    V = torch.randn(1, 8, 64, 32)

    # First lookup: cache miss
    k_dec, v_dec = cache.get_decoded_kv(K, V, decode_fn)
    assert k_dec.shape == K.shape

    # Second lookup: cache hit (same seq_len)
    k_dec2, v_dec2 = cache.get_decoded_kv(K, V, decode_fn)
    assert torch.equal(k_dec, k_dec2), "Cached result should be identical"

    stats = cache.stats
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert abs(stats["hit_rate"] - 0.5) < 0.01

    # After invalidation, should miss again
    cache.invalidate()
    k_dec3, v_dec3 = cache.get_decoded_kv(K, V, decode_fn)
    stats2 = cache.stats
    assert stats2["cache_misses"] == 2

    print(f"  ✅ Zero-Copy Decode: hits={stats2['cache_hits']}, misses={stats2['cache_misses']}, "
          f"hit_rate={stats2['hit_rate']:.2%}")


def test_hybrid_kv_cache_store():
    """End-to-end flow for the hybrid KV cache store."""
    config = HybridPipelineConfig(
        use_turbo_quant=False,  # Simplify: do not use TurboQuant
        use_packed_kv=False,     # Simplify: do not use PackedKV
        sliding_window=SlidingWindowConfig(
            max_seq_len=64,
            min_recent_tokens=8,
        ),
        enable_zero_copy_decode=True,
    )

    store = HybridKVCacheStore(config)

    # Prefill phase
    batch, heads, seq, dim = 1, 4, 48, 16
    K = torch.randn(batch, heads, seq, dim)
    V = torch.randn(batch, heads, seq, dim)
    store.set_prefill(K, V)
    assert store.seq_len == seq

    # Decode phase: incremental append
    for step in range(8):
        K_new = torch.randn(batch, heads, 1, dim)
        V_new = torch.randn(batch, heads, 1, dim)
        store.append(K_new, V_new)

    # Attention computation
    Q = torch.randn(batch, heads, 1, dim)
    output = store.attention(Q, is_causal=True)
    assert output.shape == (batch, heads, 1, dim)
    assert not torch.isnan(output).any()

    # Stats
    stats = store.stats
    print(f"  ✅ Hybrid Store: seq_len={stats['seq_len']}, "
          f"kv_bytes={stats['estimated_kv_bytes']}, "
          f"decode_cache={stats.get('decode_cache', {})}")


def test_hybrid_store_with_eviction():
    """Hybrid store + automatic eviction when over capacity."""
    config = HybridPipelineConfig(
        sliding_window=SlidingWindowConfig(
            max_seq_len=32,
            min_recent_tokens=8,
        ),
        enable_zero_copy_decode=True,
    )

    store = HybridKVCacheStore(config)

    # Prefill exceeds window size
    batch, heads, dim = 1, 4, 16
    K = torch.randn(batch, heads, 48, dim)  # 48 > 32
    V = torch.randn(batch, heads, 48, dim)
    store.set_prefill(K, V)

    # Should be evicted down to max_seq_len
    assert store.seq_len <= 32, f"Expected seq_len <= 32, got {store.seq_len}"

    print(f"  ✅ Auto-eviction: 48→{store.seq_len} tokens")


# ─────────────────────────────────────────────
# 4) End-to-end integration test
# ─────────────────────────────────────────────


def test_end_to_end_pipeline():
    """End-to-end: Layer-Adaptive bits -> Temporal Pooling attention -> Hybrid Store."""
    print("\n🔬 End-to-end Integration Test")
    print("-" * 50)

    # 1) Layer-adaptive bit allocation
    allocator = LayerAdaptiveBitAllocator(LayerAdaptiveConfig(target_avg_bits=3.0))
    batch, heads, seq, dim = 1, 4, 32, 16
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    k_bits, v_bits, report = allocator.allocate(q, k, v, layer_idx=5, total_layers=8)
    print(f"  Layer-Adaptive: k_avg={report['avg_k_bits']:.2f}, v_avg={report['avg_v_bits']:.2f}")

    # 2) Hybrid store
    store_config = HybridPipelineConfig(
        enable_zero_copy_decode=True,
        sliding_window=SlidingWindowConfig(max_seq_len=64, min_recent_tokens=8),
    )
    store = HybridKVCacheStore(store_config)
    store.set_prefill(k, v)

    # 3) Decode steps (using store attention)
    tconfig = TemporalPoolingConfig(temporal_decay=0.5, temperature=0.1)

    for step in range(4):
        K_new = torch.randn(batch, heads, 1, dim)
        V_new = torch.randn(batch, heads, 1, dim)
        store.append(K_new, V_new)

        # Use the store's baseline attention
        Q_decode = torch.randn(batch, heads, 1, dim)
        output = store.attention(Q_decode, is_causal=True)
        assert not torch.isnan(output).any()

    # 4) Compute temporal pooling attention independently
    Q_final = torch.randn(batch, heads, 1, dim)
    tip_output, tip_report = temporal_importance_attention(Q_final, k, v, tconfig)
    assert not torch.isnan(tip_output).any()

    print(f"  Temporal Pooling: sparsity={tip_report['sparsity']:.2%}")
    print(f"  Hybrid Store: seq_len={store.seq_len}")
    print("  End-to-end pipeline completed ✅")


# ─────────────────────────────────────────────
# 5) Micro-benchmarks
# ─────────────────────────────────────────────


def bench_layer_adaptive():
    """Layer-Adaptive allocation micro-benchmark."""
    config = LayerAdaptiveConfig(target_avg_bits=3.5)
    allocator = LayerAdaptiveBitAllocator(config)
    batch, heads, seq, dim = 1, 32, 64, 128
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    # Warmup
    for _ in range(3):
        allocator.allocate(q, k, v, layer_idx=10, total_layers=60)

    N = 20
    start = time.perf_counter()
    for _ in range(N):
        allocator.allocate(q, k, v, layer_idx=10, total_layers=60)
    elapsed = (time.perf_counter() - start) / N * 1000

    print(f"  ⚡ Layer-Adaptive: {elapsed:.2f} ms/alloc (1×32×64×128)")


def bench_temporal_pooling():
    """Temporal pooling attention micro-benchmark."""
    batch, heads, seq_len, head_dim = 1, 32, 512, 128
    Q = torch.randn(batch, heads, 1, head_dim)
    K = torch.randn(batch, heads, seq_len, head_dim)
    V = torch.randn(batch, heads, seq_len, head_dim)
    config = TemporalPoolingConfig()

    # Warmup
    for _ in range(3):
        temporal_importance_attention(Q, K, V, config)

    N = 20
    start = time.perf_counter()
    for _ in range(N):
        temporal_importance_attention(Q, K, V, config)
    elapsed = (time.perf_counter() - start) / N * 1000

    print(f"  ⚡ Temporal Pooling: {elapsed:.2f} ms/attn (1×32×512×128)")


def bench_hybrid_store():
    """Hybrid store micro-benchmark."""
    config = HybridPipelineConfig(
        enable_zero_copy_decode=True,
        sliding_window=SlidingWindowConfig(max_seq_len=256, min_recent_tokens=16),
    )
    store = HybridKVCacheStore(config)

    batch, heads, dim = 1, 32, 128
    # Prefill 512 tokens
    K = torch.randn(batch, heads, 512, dim)
    V = torch.randn(batch, heads, 512, dim)
    store.set_prefill(K, V)

    # Decode 32 steps
    start = time.perf_counter()
    for step in range(32):
        K_new = torch.randn(batch, heads, 1, dim)
        V_new = torch.randn(batch, heads, 1, dim)
        store.append(K_new, V_new)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"  ⚡ Hybrid Store: {elapsed:.1f} ms for 32 decode steps")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_all():
    """Run all tests."""
    print("=" * 60)
    print("  KV Cache Optimization Comprehensive Test")
    print("=" * 60)

    tests = [
        # Layer-Adaptive
        ("Layer-Adaptive Basic", test_layer_adaptive_basic),
        ("Layer-Adaptive Depth Profile", test_layer_adaptive_depth_profile),
        ("Layer-Adaptive Flat Input", test_layer_adaptive_flat_input),
        ("Layer-Adaptive All Layers", test_layer_adaptive_all_layers),
        # Temporal Pooling
        ("Temporal Pooling Basic", test_temporal_pooling_basic),
        ("Temporal Pooling vs Hard Threshold", test_temporal_pooling_vs_hard_threshold),
        ("Temporal Pooling Presets", test_temporal_pooling_presets),
        ("Temporal Pooling No Decay", test_temporal_pooling_no_decay),
        # Hybrid Pipeline
        ("Sliding Window Evictor (FIFO)", test_sliding_window_evictor),
        ("Sliding Window Evictor (Attention)", test_sliding_window_attention_based),
        ("Zero-Copy Decode Cache", test_zero_copy_decode_cache),
        ("Hybrid KV Cache Store", test_hybrid_kv_cache_store),
        ("Hybrid Store Auto-Eviction", test_hybrid_store_with_eviction),
        # E2E
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n🧪 {name}")
        print("-" * 40)
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}")
            traceback.print_exc()

    # Micro-benchmarks
    print("\n" + "=" * 60)
    print("  Performance Benchmarks")
    print("=" * 60)
    for bench_name, bench_fn in [
        ("Layer-Adaptive", bench_layer_adaptive),
        ("Temporal Pooling", bench_temporal_pooling),
        ("Hybrid Store", bench_hybrid_store),
    ]:
        try:
            bench_fn()
        except Exception as e:
            print(f"  ⚠️ {bench_name} bench failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Results: ✅ {passed} passed, ❌ {failed} failed")
    print("=" * 60)

    return failed == 0


# ─────────────────────────────────────────────
# 6) KVCacheStore integration tests (new options wired into the existing system)
# ─────────────────────────────────────────────

def _run_store_integration_tests():
    """Validate that new optimization options are wired via KVCacheStoreConfig into KVCacheStore."""
    from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig

    print("\n" + "=" * 60)
    print("  KVCacheStore Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    # ── Test 1: Temporal Pooling through KVCacheStore ──
    print("\n🧪 KVCacheStore + Temporal Pooling")
    print("-" * 40)
    try:
        cfg = KVCacheStoreConfig(
            enable_temporal_pooling=True,
            temporal_pooling_decay=0.5,
            temporal_pooling_temperature=0.1,
        )
        store = KVCacheStore(cfg)
        assert store._tip_config is not None

        batch, heads, seq, dim = 1, 4, 32, 16
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)
        store.set_prefill(K, V)

        Q = torch.randn(batch, heads, 1, dim)
        output = store.attention(Q, is_causal=True)
        assert output.shape == (batch, heads, 1, dim)
        assert not torch.isnan(output).any()
        print(f"  ✅ Temporal Pooling through KVCacheStore: output shape={output.shape}")
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Test 2: Zero-Copy Decode through KVCacheStore ──
    print("\n🧪 KVCacheStore + Zero-Copy Decode")
    print("-" * 40)
    try:
        cfg = KVCacheStoreConfig(
            enable_zero_copy_decode=True,
        )
        store = KVCacheStore(cfg)
        assert store._decode_cache is not None

        batch, heads, seq, dim = 1, 4, 32, 16
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)
        store.set_prefill(K, V)

        # Multiple decode steps (q_len=1) should hit the cache
        for _ in range(3):
            Q = torch.randn(batch, heads, 1, dim)
            output = store.attention(Q, is_causal=True)
            assert not torch.isnan(output).any()

        stats = store._decode_cache.stats
        print(f"  ✅ Zero-Copy Decode: hits={stats['cache_hits']}, misses={stats['cache_misses']}, "
              f"hit_rate={stats['hit_rate']:.2%}")
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Test 3: Sliding Window Eviction through KVCacheStore ──
    print("\n🧪 KVCacheStore + Sliding Window Eviction")
    print("-" * 40)
    try:
        cfg = KVCacheStoreConfig(
            enable_sliding_window_eviction=True,
            eviction_max_seq_len=64,
            eviction_min_recent_tokens=8,
        )
        store = KVCacheStore(cfg)
        assert store._evictor is not None

        batch, heads, dim = 1, 4, 16
        # Prefill exceeds the sliding window
        K = torch.randn(batch, heads, 96, dim)
        V = torch.randn(batch, heads, 96, dim)
        store.set_prefill(K, V)
        assert store.seq_len <= 64, f"Expected <= 64, got {store.seq_len}"

        # Further appends should trigger eviction
        for _ in range(10):
            K_new = torch.randn(batch, heads, 1, dim)
            V_new = torch.randn(batch, heads, 1, dim)
            store.append(K_new, V_new)

        assert store.seq_len <= 64, f"After append: expected <= 64, got {store.seq_len}"
        print(f"  ✅ Sliding Window: seq_len stays <= 64 (actual: {store.seq_len})")
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Test 4: All optimizations combined ──
    print("\n🧪 KVCacheStore + All Optimizations Combined")
    print("-" * 40)
    try:
        cfg = KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_k_format="turbo3",
            turbo_v_format="turbo3",
            enable_turbo_residual_qjl=True,
            enable_temporal_pooling=True,
            temporal_pooling_decay=0.5,
            enable_zero_copy_decode=True,
            enable_sliding_window_eviction=True,
            eviction_max_seq_len=128,
            eviction_min_recent_tokens=16,
            enable_layer_adaptive=True,
            layer_adaptive_target_avg_bits=3.0,
        )
        store = KVCacheStore(cfg)

        batch, heads, seq, dim = 1, 4, 64, 16
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)
        store.set_prefill(K, V)

        # Decode steps
        for step in range(8):
            K_new = torch.randn(batch, heads, 1, dim)
            V_new = torch.randn(batch, heads, 1, dim)
            store.append(K_new, V_new)

            Q = torch.randn(batch, heads, 1, dim)
            output = store.attention(Q, is_causal=True)
            assert not torch.isnan(output).any(), f"NaN at step {step}"

        print(f"  ✅ All combined: seq_len={store.seq_len}, no NaN")
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    # ── Test 5: Smart preset through policy ──
    print("\n🧪 Smart Policy Preset")
    print("-" * 40)
    try:
        from vitriol.kv.policy import list_policy_presets
        presets = list_policy_presets()
        smart = next((p for p in presets if p.name == "smart"), None)
        assert smart is not None, "smart preset not found"
        assert smart.params.get("enable_temporal_pooling") is True
        assert smart.params.get("enable_zero_copy_decode") is True
        assert smart.params.get("enable_sliding_window_eviction") is True
        print(f"  ✅ Smart preset: {smart.params}")
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  Integration tests: ✅ {passed} passed, ❌ {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    integration_success = _run_store_integration_tests()
    sys.exit(0 if (success and integration_success) else 1)
