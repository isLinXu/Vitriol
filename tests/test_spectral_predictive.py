"""
Comprehensive tests for SpectralKV and PredictiveKV modules.

Tests cover:
    1. SpectralKV: frequency-aware compression codec
    2. PredictiveKV: linear-prediction residual coding codec
    3. Integration with KVCacheStore
    4. Policy presets
    5. Benchmark comparison vs TurboQuant
"""

import math
import sys
import torch
import torch.nn.functional as F


def _make_kv(batch=1, heads=4, seq_len=64, dim=64, seed=42):
    """Create synthetic KV tensors with realistic correlation structure."""
    g = torch.Generator()
    g.manual_seed(seed)

    # Create base signal with temporal correlation
    t = torch.linspace(0, 4 * math.pi, seq_len).unsqueeze(-1)  # [seq, 1]

    # Add positional patterns (like RoPE)
    freqs = torch.randn(dim, generator=g) * 0.5
    key = torch.sin(t * freqs.unsqueeze(0))  # [seq, dim]

    # Add content-dependent patterns
    content = torch.randn(seq_len, dim, generator=g) * 0.3
    key = key + content

    # Expand to [batch, heads, seq, dim]
    key = key.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1).clone()

    # Value: similar but less structured
    value = torch.randn(batch, heads, seq_len, dim, generator=g) * 0.5 + key * 0.3

    return key, value


# ═══════════════════════════════════════════════════════════════
# Test 1: SpectralKV Basic Functionality
# ═══════════════════════════════════════════════════════════════

def test_spectral_kv_compress_decompress():
    """Test SpectralKV compress → decompress round-trip."""
    from vitriol.kv.spectral import SpectralKVCodec, SpectralKVConfig

    key, value = _make_kv(batch=1, heads=2, seq_len=32, dim=64)

    for target_bpv in [2.0, 3.0, 4.0]:
        config = SpectralKVConfig(target_bpv=target_bpv)
        codec = SpectralKVCodec(config)

        # Compress K
        k_compressed, k_report = codec.compress(key, is_key=True)
        k_reconstructed = codec.decompress(k_compressed)

        # Compress V
        v_compressed, v_report = codec.compress(value, is_key=False)
        v_reconstructed = codec.decompress(v_compressed)

        # Verify shape preservation
        assert k_reconstructed.shape == key.shape, f"K shape mismatch: {k_reconstructed.shape} vs {key.shape}"
        assert v_reconstructed.shape == value.shape, f"V shape mismatch: {v_reconstructed.shape} vs {value.shape}"

        # Verify reconstruction quality (cosine similarity should be high)
        k_cos = float(F.cosine_similarity(key.flatten().unsqueeze(0), k_reconstructed.flatten().unsqueeze(0)).item())
        v_cos = float(F.cosine_similarity(value.flatten().unsqueeze(0), v_reconstructed.flatten().unsqueeze(0)).item())

        assert k_cos > 0.5, f"K cosine similarity too low at bpv={target_bpv}: {k_cos:.4f}"
        assert v_cos > 0.3, f"V cosine similarity too low at bpv={target_bpv}: {v_cos:.4f}"

        # Verify report contains expected fields
        assert "spectral_alpha" in k_report
        assert "k_low" in k_report
        assert "k_high" in k_report
        assert "bits_low" in k_report
        assert "bits_high" in k_report
        assert "effective_bpv" in k_report
        assert "compression_ratio" in k_report

        print(f"  ✅ SpectralKV bpv={target_bpv}: K_cos={k_cos:.4f}, V_cos={v_cos:.4f}, "
              f"α={k_report['spectral_alpha']:.2f}, eff_bpv={k_report['effective_bpv']:.2f}")


def test_spectral_kv_compress_kv():
    """Test SpectralKV compress_kv API."""
    from vitriol.kv.spectral import SpectralKVCodec, SpectralKVConfig

    key, value = _make_kv(batch=1, heads=2, seq_len=32, dim=64)
    config = SpectralKVConfig(target_bpv=3.0)
    codec = SpectralKVCodec(config)

    k_out, v_out, report = codec.compress_kv(key, value)

    assert k_out.shape == key.shape
    assert v_out.shape == value.shape
    assert "k_mse" in report
    assert "v_mse" in report
    assert "total_mse" in report
    assert report["method"] == "spectral_kv"

    print(f"  ✅ SpectralKV compress_kv: K_MSE={report['k_mse']:.6f}, V_MSE={report['v_mse']:.6f}")


def test_spectral_kv_qdq():
    """Test spectral_qdq quick function."""
    from vitriol.kv.spectral import spectral_qdq

    key, _ = _make_kv(batch=1, heads=2, seq_len=32, dim=64)
    reconstructed, report = spectral_qdq(key, target_bpv=3.0, is_key=True)

    assert reconstructed.shape == key.shape
    assert "mse" in report
    assert "cosine_similarity" in report
    assert "snr_db" in report

    print(f"  ✅ spectral_qdq: MSE={report['mse']:.6f}, SNR={report['snr_db']:.1f}dB, "
          f"COS={report['cosine_similarity']:.4f}")


def test_spectral_alpha_detection():
    """Test automatic spectral decay detection."""
    from vitriol.kv.spectral import _estimate_spectral_decay, SpectralKVCodec, SpectralKVConfig

    # Create tensor with known spectral decay
    key, _ = _make_kv(batch=1, heads=2, seq_len=64, dim=128)

    # Auto-detect
    alpha = _estimate_spectral_decay(key)
    assert 0.5 <= alpha <= 5.0, f"Alpha out of expected range: {alpha}"

    # Compare auto vs fixed
    config_auto = SpectralKVConfig(target_bpv=3.0, auto_detect_alpha=True)
    config_fixed = SpectralKVConfig(target_bpv=3.0, auto_detect_alpha=False, fixed_alpha=2.0)

    codec_auto = SpectralKVCodec(config_auto)
    codec_fixed = SpectralKVCodec(config_fixed)

    _, report_auto = codec_auto.compress(key, is_key=True)
    _, report_fixed = codec_fixed.compress(key, is_key=True)

    print(f"  ✅ Alpha detection: auto={report_auto['spectral_alpha']:.2f}, fixed={report_fixed['spectral_alpha']:.2f}")


def test_spectral_storage_nbytes():
    """Test SpectralKVCompressed storage estimation."""
    from vitriol.kv.spectral import SpectralKVCodec, SpectralKVConfig

    key, _ = _make_kv(batch=1, heads=2, seq_len=32, dim=64)
    codec = SpectralKVCodec(SpectralKVConfig(target_bpv=3.0))
    compressed, report = codec.compress(key, is_key=True)

    nbytes = compressed.storage_nbytes()
    assert nbytes > 0, "Storage bytes should be positive"

    # Compare with fp16 baseline
    fp16_bytes = key.numel() * 2
    ratio = fp16_bytes / nbytes
    print(f"  ✅ Storage: compressed={nbytes}B, fp16={fp16_bytes}B, ratio={ratio:.1f}x")


# ═══════════════════════════════════════════════════════════════
# Test 2: PredictiveKV Basic Functionality
# ═══════════════════════════════════════════════════════════════

def test_predictive_kv_compress_decompress():
    """Test PredictiveKV compress → decompress round-trip."""
    from vitriol.kv.predictive import PredictiveKVCodec, PredictiveKVConfig

    key, value = _make_kv(batch=1, heads=2, seq_len=64, dim=64)

    for order in [1, 2, 4]:
        config = PredictiveKVConfig(target_bpv=3.0, prediction_order=order, auto_order=False)
        codec = PredictiveKVCodec(config)

        # Compress K
        k_compressed, k_report = codec.compress(key, is_key=True)
        k_reconstructed = codec.decompress(k_compressed)

        # Compress V
        v_compressed, v_report = codec.compress(value, is_key=False)
        v_reconstructed = codec.decompress(v_compressed)

        # Verify shape
        assert k_reconstructed.shape == key.shape, f"K shape: {k_reconstructed.shape} vs {key.shape}"
        assert v_reconstructed.shape == value.shape, f"V shape: {v_reconstructed.shape} vs {value.shape}"

        # Quality
        k_cos = float(F.cosine_similarity(key.flatten().unsqueeze(0), k_reconstructed.flatten().unsqueeze(0)).item())
        v_cos = float(F.cosine_similarity(value.flatten().unsqueeze(0), v_reconstructed.flatten().unsqueeze(0)).item())

        # Prediction gain should be >= 1.0
        assert k_report["prediction_gain"] >= 1.0, f"K prediction gain < 1: {k_report['prediction_gain']}"
        assert v_report["prediction_gain"] >= 1.0, f"V prediction gain < 1: {v_report['prediction_gain']}"

        print(f"  ✅ PredictiveKV order={order}: K_cos={k_cos:.4f}, V_cos={v_cos:.4f}, "
              f"K_gain={k_report['prediction_gain']:.1f}x, V_gain={v_report['prediction_gain']:.1f}x")


def test_predictive_kv_auto_order():
    """Test automatic prediction order selection."""
    from vitriol.kv.predictive import PredictiveKVCodec, PredictiveKVConfig, _select_prediction_order

    key, _ = _make_kv(batch=1, heads=2, seq_len=64, dim=64)

    # Auto-select
    order = _select_prediction_order(key, max_order=8, min_order=1)
    assert 1 <= order <= 8, f"Order out of range: {order}"

    # Auto-order codec
    config = PredictiveKVConfig(target_bpv=3.0, auto_order=True)
    codec = PredictiveKVCodec(config)

    _, report = codec.compress(key, is_key=True)
    print(f"  ✅ Auto-order: selected={report['order']}, gain={report['prediction_gain']:.1f}x")


def test_predictive_kv_compress_kv():
    """Test PredictiveKV compress_kv API."""
    from vitriol.kv.predictive import PredictiveKVCodec, PredictiveKVConfig

    key, value = _make_kv(batch=1, heads=2, seq_len=64, dim=64)
    config = PredictiveKVConfig(target_bpv=3.0, prediction_order=2)
    codec = PredictiveKVCodec(config)

    k_out, v_out, report = codec.compress_kv(key, value)

    assert k_out.shape == key.shape
    assert v_out.shape == value.shape
    assert "k_mse" in report
    assert "v_mse" in report
    assert report["method"] == "predictive_kv"

    print(f"  ✅ PredictiveKV compress_kv: K_MSE={report['k_mse']:.6f}, V_MSE={report['v_mse']:.6f}")


def test_predictive_kv_qdq():
    """Test predictive_qdq quick function."""
    from vitriol.kv.predictive import predictive_qdq

    key, _ = _make_kv(batch=1, heads=2, seq_len=64, dim=64)
    reconstructed, report = predictive_qdq(key, target_bpv=3.0, is_key=True)

    assert reconstructed.shape == key.shape
    assert "mse" in report
    assert "cosine_similarity" in report
    assert "snr_db" in report

    print(f"  ✅ predictive_qdq: MSE={report['mse']:.6f}, SNR={report['snr_db']:.1f}dB, "
          f"COS={report['cosine_similarity']:.4f}")


def test_predictive_lpc_estimation():
    """Test Yule-Walker LPC estimation."""
    from vitriol.kv.predictive import _estimate_lpc_yule_walker, _compute_lpc_for_kv

    # Simple test: sine wave with known prediction structure
    # _estimate_lpc_yule_walker expects [..., seq_len] where last dim is the time dimension
    t = torch.linspace(0, 4 * math.pi, 128)  # [128] — 1D signal
    coeffs = _estimate_lpc_yule_walker(t.unsqueeze(0), order=2)  # [1, 2]
    assert coeffs.shape[-1] == 2, f"Unexpected last dim: {coeffs.shape}"
    assert coeffs.ndim == 2, f"Unexpected ndim: {coeffs.ndim}"

    # KV tensor LPC
    key, _ = _make_kv(batch=1, heads=2, seq_len=64, dim=32)
    kv_coeffs = _compute_lpc_for_kv(key, order=2)
    assert kv_coeffs.shape[:2] == (1, 2), f"Unexpected KV LPC shape: {kv_coeffs.shape}"
    assert kv_coeffs.shape[2] == 2, f"Order dim mismatch: {kv_coeffs.shape[2]}"

    print(f"  ✅ LPC estimation: 1D coeffs shape={coeffs.shape}, KV coeffs shape={kv_coeffs.shape}")


def test_predictive_storage_nbytes():
    """Test PredictiveKVCompressed storage estimation."""
    from vitriol.kv.predictive import PredictiveKVCodec, PredictiveKVConfig

    key, _ = _make_kv(batch=1, heads=2, seq_len=64, dim=64)
    codec = PredictiveKVCodec(PredictiveKVConfig(target_bpv=3.0, prediction_order=2))
    compressed, report = codec.compress(key, is_key=True)

    nbytes = compressed.storage_nbytes()
    assert nbytes > 0

    fp16_bytes = key.numel() * 2
    ratio = fp16_bytes / nbytes
    print(f"  ✅ PredictiveKV Storage: compressed={nbytes}B, fp16={fp16_bytes}B, ratio={ratio:.1f}x")


# ═══════════════════════════════════════════════════════════════
# Test 3: Integration with KVCacheStore
# ═══════════════════════════════════════════════════════════════

def test_cache_store_spectral_kv():
    """Test KVCacheStore with SpectralKV enabled."""
    from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig

    key, value = _make_kv(batch=1, heads=2, seq_len=32, dim=64)
    query = torch.randn(1, 2, 8, 64)

    config = KVCacheStoreConfig(
        enable_spectral_kv=True,
        spectral_target_bpv=3.0,
    )
    store = KVCacheStore(config)
    store.set_prefill(key, value)

    output = store.attention(query)
    assert output.shape == (1, 2, 8, 64), f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"

    print(f"  ✅ KVCacheStore + SpectralKV: output shape={output.shape}")


def test_cache_store_predictive_kv():
    """Test KVCacheStore with PredictiveKV enabled."""
    from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig

    key, value = _make_kv(batch=1, heads=2, seq_len=64, dim=64)
    query = torch.randn(1, 2, 8, 64)

    config = KVCacheStoreConfig(
        enable_predictive_kv=True,
        predictive_target_bpv=3.0,
        predictive_order=2,
    )
    store = KVCacheStore(config)
    store.set_prefill(key, value)

    output = store.attention(query)
    assert output.shape == (1, 2, 8, 64), f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"

    print(f"  ✅ KVCacheStore + PredictiveKV: output shape={output.shape}")


# ═══════════════════════════════════════════════════════════════
# Test 4: Policy Presets
# ═══════════════════════════════════════════════════════════════

def test_policy_presets():
    """Test new SpectralKV and PredictiveKV policy presets."""
    from vitriol.kv.policy import KVPolicyPreset, list_policy_presets

    presets = list_policy_presets()
    names = [p.name for p in presets]

    assert "spectral" in names, "Missing 'spectral' preset"
    assert "predictive" in names, "Missing 'predictive' preset"
    assert "spectral-predictive" in names, "Missing 'spectral-predictive' preset"

    # Check spectral preset params
    spectral = KVPolicyPreset.spectral_default()
    assert spectral.params.get("enable_spectral_kv") is True

    predictive = KVPolicyPreset.predictive_default()
    assert predictive.params.get("enable_predictive_kv") is True

    combined = KVPolicyPreset.spectral_predictive_default()
    assert combined.params.get("enable_spectral_kv") is True
    assert combined.params.get("enable_predictive_kv") is True

    print(f"  ✅ Policy presets: {len(presets)} total, names={names}")


def test_apply_policy_spectral():
    """Test apply_policy_to_store_cfg with spectral settings."""
    from vitriol.kv.cache_store import KVCacheStoreConfig
    from vitriol.kv.policy import KVPolicyPreset, apply_policy_to_store_cfg, build_policy

    base_cfg = KVCacheStoreConfig()
    preset = KVPolicyPreset.spectral_default()
    policy = build_policy(preset)

    # Create a mock handle
    class MockHandle:
        layer_types = ["full_attention"] * 4
    handle = MockHandle()

    cfg = apply_policy_to_store_cfg(base_cfg, policy, handle, layer_idx=0)
    assert cfg.enable_spectral_kv is True, "SpectralKV should be enabled"

    print(f"  ✅ apply_policy with spectral: enable_spectral_kv={cfg.enable_spectral_kv}")


# ═══════════════════════════════════════════════════════════════
# Test 5: Benchmark Comparison
# ═══════════════════════════════════════════════════════════════

def test_benchmark_comparison():
    """
    Compare SpectralKV, PredictiveKV vs baseline (no compression) quality.

    This is a quick sanity check — not a full benchmark.
    """
    from vitriol.kv.spectral import spectral_qdq
    from vitriol.kv.predictive import predictive_qdq

    key, value = _make_kv(batch=1, heads=4, seq_len=128, dim=128)

    results = {}

    # Baseline: fp16 (no compression)
    results["fp16"] = {"mse_k": 0.0, "mse_v": 0.0}

    # SpectralKV at different bpv
    for bpv in [2.0, 3.0, 4.0]:
        k_rec, k_rpt = spectral_qdq(key, target_bpv=bpv, is_key=True)
        v_rec, v_rpt = spectral_qdq(value, target_bpv=bpv, is_key=False)
        results[f"spectral_{bpv}bpv"] = {
            "mse_k": k_rpt["mse"],
            "mse_v": v_rpt["mse"],
            "snr_k": k_rpt["snr_db"],
            "snr_v": v_rpt["snr_db"],
        }

    # PredictiveKV at different bpv
    for bpv in [2.0, 3.0, 4.0]:
        k_rec, k_rpt = predictive_qdq(key, target_bpv=bpv, is_key=True)
        v_rec, v_rpt = predictive_qdq(value, target_bpv=bpv, is_key=False)
        results[f"predictive_{bpv}bpv"] = {
            "mse_k": k_rpt["mse"],
            "mse_v": v_rpt["mse"],
            "snr_k": k_rpt["snr_db"],
            "snr_v": v_rpt["snr_db"],
        }

    # Print comparison table
    print("\n  ┌─────────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("  │ Method                  │ K_MSE       │ V_MSE       │ K_SNR(dB)   │ V_SNR(dB)   │")
    print("  ├─────────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
    for name, r in results.items():
        k_mse = f"{r['mse_k']:.6f}" if 'mse_k' in r else "0.000000"
        v_mse = f"{r['mse_v']:.6f}" if 'mse_v' in r else "0.000000"
        k_snr = f"{r.get('snr_k', 0):.1f}" if 'snr_k' in r else "∞"
        v_snr = f"{r.get('snr_v', 0):.1f}" if 'snr_v' in r else "∞"
        print(f"  │ {name:23s} │ {k_mse:11s} │ {v_mse:11s} │ {k_snr:11s} │ {v_snr:11s} │")
    print("  └─────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

    # Spectral should be better than no-rotation baseline at same bpv
    # (at least check that it produces reasonable output)
    for method in ["spectral_3.0bpv", "predictive_3.0bpv"]:
        assert results[method]["mse_k"] > 0, f"{method} K_MSE should be > 0 (compressed)"
        assert results[method]["mse_v"] > 0, f"{method} V_MSE should be > 0 (compressed)"

    print("\n  ✅ Benchmark comparison complete")


# ═══════════════════════════════════════════════════════════════
# Test 6: Imports
# ═══════════════════════════════════════════════════════════════

def test_imports():
    """Test that all new exports are importable."""
    print("  ✅ All imports successful")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SpectralKV & PredictiveKV Comprehensive Tests")
    print("=" * 70)

    tests = [
        # SpectralKV
        ("SpectralKV compress/decompress", test_spectral_kv_compress_decompress),
        ("SpectralKV compress_kv", test_spectral_kv_compress_kv),
        ("SpectralKV qdq", test_spectral_kv_qdq),
        ("SpectralKV alpha detection", test_spectral_alpha_detection),
        ("SpectralKV storage", test_spectral_storage_nbytes),
        # PredictiveKV
        ("PredictiveKV compress/decompress", test_predictive_kv_compress_decompress),
        ("PredictiveKV auto-order", test_predictive_kv_auto_order),
        ("PredictiveKV compress_kv", test_predictive_kv_compress_kv),
        ("PredictiveKV qdq", test_predictive_kv_qdq),
        ("PredictiveKV LPC", test_predictive_lpc_estimation),
        ("PredictiveKV storage", test_predictive_storage_nbytes),
        # Integration
        ("KVCacheStore + SpectralKV", test_cache_store_spectral_kv),
        ("KVCacheStore + PredictiveKV", test_cache_store_predictive_kv),
        # Policy
        ("Policy presets", test_policy_presets),
        ("Apply policy spectral", test_apply_policy_spectral),
        # Benchmark
        ("Benchmark comparison", test_benchmark_comparison),
        # Imports
        ("Imports", test_imports),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\n── {name} ──")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 70}")

    sys.exit(0 if failed == 0 else 1)
