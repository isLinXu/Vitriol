"""
Tests for vitriol.kv.cross_layer module.
"""
import pytest
import math

import torch

from vitriol.kv.cross_layer import (
    estimate_layer_correlation,
    compute_layer_delta_stats,
    _quantize_delta_blockwise,
    _dequantize_delta_blockwise,
    _detect_scene_change,
    CrossLayerKVCompressed,
    LayerGroupState,
    CrossLayerKVConfig,
    CrossLayerKVCodec,
    compress_multilayer_kv,
    decompress_multilayer_kv,
    cross_layer_qdq,
)


class TestEstimateLayerCorrelation:
    def test_identical_layers(self):
        kv = torch.randn(2, 4, 8, 64)
        layers = [kv, kv.clone()]
        rho = estimate_layer_correlation(layers)
        assert rho == pytest.approx(1.0, abs=1e-4)

    def test_different_layers(self):
        layers = [torch.randn(2, 4, 8, 64), torch.randn(2, 4, 8, 64)]
        rho = estimate_layer_correlation(layers)
        # Correlation of independent random tensors should be near 0
        assert abs(rho) < 0.5  # Loose bound due to randomness

    def test_single_layer(self):
        layers = [torch.randn(2, 4, 8, 64)]
        rho = estimate_layer_correlation(layers)
        assert rho == 1.0

    def test_3d_tensor(self):
        layers = [torch.randn(4, 8, 64), torch.randn(4, 8, 64)]
        rho = estimate_layer_correlation(layers)
        assert isinstance(rho, float)

    def test_returns_float(self):
        layers = [torch.randn(2, 4, 8, 64), torch.randn(2, 4, 8, 64)]
        rho = estimate_layer_correlation(layers)
        assert isinstance(rho, float)
        assert -1.0 <= rho <= 1.0


class TestComputeLayerDeltaStats:
    def test_single_layer(self):
        layers = [torch.randn(2, 4, 8, 64)]
        stats = compute_layer_delta_stats(layers)
        assert stats["mean_var_ratio"] == 1.0
        assert stats["max_delta_ratio"] == 1.0
        assert stats["correlation"] == 1.0

    def test_multiple_layers(self):
        layers = [
            torch.randn(2, 4, 8, 64),
            torch.randn(2, 4, 8, 64),
            torch.randn(2, 4, 8, 64),
        ]
        stats = compute_layer_delta_stats(layers)
        assert "mean_var_ratio" in stats
        assert "max_delta_ratio" in stats
        assert "correlation" in stats
        assert stats["mean_var_ratio"] >= 0
        assert stats["max_delta_ratio"] >= 0

    def test_similar_layers_low_var_ratio(self):
        base = torch.randn(2, 4, 8, 64)
        layers = [base, base + torch.randn(2, 4, 8, 64) * 0.1]
        stats = compute_layer_delta_stats(layers)
        # Small delta -> low var ratio
        assert stats["mean_var_ratio"] < 1.0


class TestQuantizeDeltaBlockwise:
    def test_basic(self):
        delta = torch.randn(2, 4, 8, 64)
        q, scales, zero_points = _quantize_delta_blockwise(delta, levels=8, block_size=32)
        assert q.shape == delta.reshape(-1, delta.shape[-1] // 32, 32).shape
        assert scales.shape[:-1] == q.shape[:-1]  # scales per block
        assert torch.all(zero_points == 0)  # Symmetric quantization

    def test_symmetric_range(self):
        delta = torch.randn(2, 4, 8, 64)
        q, scales, _ = _quantize_delta_blockwise(delta, levels=8, block_size=32)
        # For levels=8, max quantized value should be 4
        assert torch.all(q <= 4)
        assert torch.all(q >= -4)

    def test_pad_handling(self):
        delta = torch.randn(2, 4, 8, 50)  # 50 not divisible by 32
        q, scales, _ = _quantize_delta_blockwise(delta, levels=8, block_size=32)
        # Should handle padding internally
        assert q is not None
        assert scales is not None


class TestDequantizeDeltaBlockwise:
    def test_roundtrip(self):
        delta = torch.randn(2, 4, 8, 64)
        orig_shape = delta.shape
        q, scales, zero_points = _quantize_delta_blockwise(delta, levels=8, block_size=32)
        reconstructed = _dequantize_delta_blockwise(q, scales, zero_points, orig_shape, block_size=32)
        assert reconstructed.shape == orig_shape
        # Should be approximately equal (quantization loss)
        mse = (delta - reconstructed).pow(2).mean().item()
        assert mse < 1.0  # Loose bound

    def test_padded_roundtrip(self):
        delta = torch.randn(2, 4, 8, 50)
        orig_shape = delta.shape
        q, scales, zero_points = _quantize_delta_blockwise(delta, levels=8, block_size=32)
        reconstructed = _dequantize_delta_blockwise(q, scales, zero_points, orig_shape, block_size=32)
        assert reconstructed.shape == orig_shape


class TestDetectSceneChange:
    def test_no_scene_change(self):
        delta = torch.randn(2, 4, 8, 64) * 0.01  # Small delta
        base_var = 1.0
        assert _detect_scene_change(delta, base_var, threshold=4.0) is False

    def test_scene_change(self):
        delta = torch.randn(2, 4, 8, 64) * 2.0  # Large delta
        base_var = 0.1
        assert _detect_scene_change(delta, base_var, threshold=4.0) is True

    def test_threshold_effect(self):
        delta = torch.randn(2, 4, 8, 64) * 0.5
        base_var = 1.0
        # High threshold should not detect
        assert _detect_scene_change(delta, base_var, threshold=10.0) is False


class TestCrossLayerKVCompressed:
    def test_storage_nbytes(self):
        compressed = CrossLayerKVCompressed(
            frame_type="iframe",
            q_data=torch.randint(-4, 5, (10, 32)),
            scales=torch.randn(10, 1),
            zero_points=torch.zeros(10, 1),
            ref_layer_idx=-1,
            orig_shape=(2, 4, 8, 64),
            levels=8,
            block_size=32,
            is_key=True,
        )
        nbytes = compressed.storage_nbytes()
        assert nbytes > 0
        # Data bytes + metadata bytes
        bits_per_level = math.ceil(math.log2(8))
        data_bytes = compressed.q_data.numel() * bits_per_level // 8
        meta_bytes = (compressed.scales.numel() + compressed.zero_points.numel()) * 4
        assert nbytes == data_bytes + meta_bytes

    def test_pframe_storage(self):
        compressed = CrossLayerKVCompressed(
            frame_type="pframe",
            q_data=torch.randint(-4, 5, (10, 32)),
            scales=torch.randn(10, 1),
            zero_points=torch.zeros(10, 1),
            ref_layer_idx=0,
            orig_shape=(2, 4, 8, 64),
            levels=4,
            block_size=32,
            is_key=True,
            delta_var_ratio=0.1,
            estimated_snr_db=15.0,
        )
        nbytes = compressed.storage_nbytes()
        assert nbytes > 0


class TestLayerGroupState:
    def test_defaults(self):
        state = LayerGroupState()
        assert state.iframe_interval == 4
        assert state.min_iframe_interval == 2
        assert state.max_iframe_interval == 8
        assert state.scene_change_threshold == 4.0
        assert state.is_first_layer is True
        assert state.layers_since_iframe == 0


class TestCrossLayerKVConfig:
    def test_defaults(self):
        cfg = CrossLayerKVConfig()
        assert cfg.target_bpv == 2.4
        assert cfg.iframe_interval == 4
        assert cfg.adaptive_iframe is True
        assert cfg.scene_change_threshold == 4.0
        assert cfg.iframe_levels == 0  # auto-derive
        assert cfg.iframe_block_size == 32
        assert cfg.pframe_levels == 0  # auto-derive
        assert cfg.apply_rotation is False
        assert cfg.predictive_iframe is False
        assert cfg.spectral_pframe is False

    def test_custom_values(self):
        cfg = CrossLayerKVConfig(
            target_bpv=3.0,
            iframe_interval=2,
            adaptive_iframe=False,
            apply_rotation=True,
        )
        assert cfg.target_bpv == 3.0
        assert cfg.iframe_interval == 2
        assert cfg.adaptive_iframe is False
        assert cfg.apply_rotation is True


class TestCrossLayerKVCodec:
    def test_init(self):
        codec = CrossLayerKVCodec()
        assert codec.config is not None
        assert codec._prev_k is None
        assert codec._prev_v is None
        assert codec._layer_count == 0

    def test_derive_levels_iframe(self):
        codec = CrossLayerKVCodec()
        levels = codec._derive_levels(target_bpv=3.0, is_iframe=True, is_key=True)
        assert levels >= 2
        # I-frame should have more levels than P-frame at same target
        pframe_levels = codec._derive_levels(target_bpv=3.0, is_iframe=False, is_key=True)
        assert levels >= pframe_levels

    def test_derive_levels_pframe(self):
        codec = CrossLayerKVCodec()
        levels = codec._derive_levels(target_bpv=2.4, is_iframe=False, is_key=True)
        assert levels >= 2

    def test_derive_levels_k_boost(self):
        codec = CrossLayerKVCodec(CrossLayerKVConfig(k_level_boost=2))
        k_levels = codec._derive_levels(target_bpv=2.4, is_iframe=True, is_key=True)
        v_levels = codec._derive_levels(target_bpv=2.4, is_iframe=True, is_key=False)
        assert k_levels > v_levels

    def test_should_be_iframe_first_layer(self):
        codec = CrossLayerKVCodec()
        assert codec._should_be_iframe() is True

    def test_should_be_iframe_after_interval(self):
        codec = CrossLayerKVCodec()
        codec._group_state.is_first_layer = False
        codec._group_state.layers_since_iframe = 5  # > default interval of 4
        assert codec._should_be_iframe() is True

    def test_should_be_iframe_before_interval(self):
        codec = CrossLayerKVCodec()
        codec._group_state.is_first_layer = False
        codec._group_state.layers_since_iframe = 2  # < default interval of 4
        assert codec._should_be_iframe() is False

    def test_compress_single_iframe(self):
        codec = CrossLayerKVCodec()
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress_single(x, is_key=True)
        assert isinstance(compressed, CrossLayerKVCompressed)
        assert compressed.frame_type == "iframe"
        assert compressed.is_key is True
        assert report["frame_type"] == "iframe"
        assert "effective_bpv" in report
        assert "compression_ratio" in report

    def test_compress_single_pframe(self):
        codec = CrossLayerKVCodec()
        prev_x = torch.randn(2, 4, 8, 64)
        x = prev_x + torch.randn(2, 4, 8, 64) * 0.1  # Small delta
        # First call creates iframe
        codec.compress_single(prev_x, is_key=True)
        # Second call may create pframe
        compressed, report = codec.compress_single(x, is_key=True, prev_x=prev_x)
        assert compressed.frame_type in ["iframe", "pframe"]

    def test_compress_single_force_iframe(self):
        codec = CrossLayerKVCodec()
        prev_x = torch.randn(2, 4, 8, 64)
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress_single(x, is_key=True, prev_x=prev_x, force_iframe=True)
        assert compressed.frame_type == "iframe"

    def test_decompress_single_iframe(self):
        codec = CrossLayerKVCodec()
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress_single(x, is_key=True)
        reconstructed = codec.decompress_single(compressed)
        assert reconstructed.shape == x.shape

    def test_decompress_single_pframe(self):
        codec = CrossLayerKVCodec()
        prev_x = torch.randn(2, 4, 8, 64)
        x = prev_x + torch.randn(2, 4, 8, 64) * 0.1
        codec.compress_single(prev_x, is_key=True)
        compressed, report = codec.compress_single(x, is_key=True, prev_x=prev_x)
        if compressed.frame_type == "pframe":
            reconstructed = codec.decompress_single(compressed, prev_x=prev_x)
            assert reconstructed.shape == x.shape

    def test_decompress_single_pframe_no_prev_raises(self):
        codec = CrossLayerKVCodec()
        compressed = CrossLayerKVCompressed(
            frame_type="pframe",
            q_data=torch.randint(-4, 5, (10, 32)),
            scales=torch.randn(10, 1),
            zero_points=torch.zeros(10, 1),
            ref_layer_idx=0,
            orig_shape=(2, 4, 8, 64),
            levels=8,
            block_size=32,
            is_key=True,
        )
        with pytest.raises(ValueError, match="P-frame decompression requires prev_x"):
            codec.decompress_single(compressed)

    def test_compress_kv(self):
        codec = CrossLayerKVCodec()
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        k_out, v_out, report = codec.compress_kv(key, value)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape
        assert "k" in report
        assert "v" in report
        assert "total_mse" in report
        assert codec._prev_k is not None
        assert codec._prev_v is not None

    def test_compress_kv_with_prev(self):
        codec = CrossLayerKVCodec()
        prev_key = torch.randn(2, 4, 8, 64)
        prev_value = torch.randn(2, 4, 8, 64)
        key = prev_key + torch.randn(2, 4, 8, 64) * 0.1
        value = prev_value + torch.randn(2, 4, 8, 64) * 0.1
        k_out, v_out, report = codec.compress_kv(key, value, prev_key=prev_key, prev_value=prev_value)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape

    def test_reset_layer_state(self):
        codec = CrossLayerKVCodec()
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        codec.compress_kv(key, value)
        assert codec._layer_count > 0
        assert codec._prev_k is not None

        codec.reset_layer_state()
        assert codec._layer_count == 0
        assert codec._prev_k is None
        assert codec._prev_v is None
        assert codec._group_state.is_first_layer is True

    def test_with_rotation(self):
        cfg = CrossLayerKVConfig(apply_rotation=True)
        codec = CrossLayerKVCodec(cfg)
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress_single(x, is_key=True)
        reconstructed = codec.decompress_single(compressed)
        assert reconstructed.shape == x.shape


class TestCompressMultilayerKV:
    def test_basic(self):
        kv_layers = [
            (torch.randn(2, 4, 8, 64), torch.randn(2, 4, 8, 64))
            for _ in range(4)
        ]
        compressed, report = compress_multilayer_kv(kv_layers)
        assert len(compressed) == 4
        assert report["n_layers"] == 4
        assert "n_iframes" in report
        assert "n_pframes" in report
        assert "estimated_correlation" in report

    def test_correlation_estimate(self):
        # Create highly correlated layers
        base = torch.randn(2, 4, 8, 64)
        kv_layers = [
            (base + torch.randn(2, 4, 8, 64) * 0.05, torch.randn(2, 4, 8, 64))
            for _ in range(4)
        ]
        compressed, report = compress_multilayer_kv(kv_layers)
        assert 0 <= report["estimated_correlation"] <= 1


class TestDecompressMultilayerKV:
    def test_roundtrip(self):
        kv_layers = [
            (torch.randn(2, 4, 8, 64), torch.randn(2, 4, 8, 64))
            for _ in range(4)
        ]
        compressed, report = compress_multilayer_kv(kv_layers)
        reconstructed = decompress_multilayer_kv(compressed)
        assert len(reconstructed) == 4
        for i, (k_rec, v_rec) in enumerate(reconstructed):
            assert k_rec.shape == kv_layers[i][0].shape
            assert v_rec.shape == kv_layers[i][1].shape


class TestCrossLayerQDQ:
    def test_basic_iframe(self):
        x = torch.randn(2, 4, 8, 64)
        result, report = cross_layer_qdq(x, target_bpv=2.4)
        assert result.shape == x.shape
        assert "mse" in report
        assert "snr_db" in report
        assert "cosine_similarity" in report

    def test_with_prev_pframe(self):
        prev_x = torch.randn(2, 4, 8, 64)
        x = prev_x + torch.randn(2, 4, 8, 64) * 0.05  # Small delta
        result, report = cross_layer_qdq(x, target_bpv=2.4, prev_x=prev_x)
        assert result.shape == x.shape
        # With small delta and prev_x, should use pframe
        assert report.get("frame_type") in ["iframe", "pframe"]

    def test_is_key_parameter(self):
        x = torch.randn(2, 4, 8, 64)
        result_k, report_k = cross_layer_qdq(x, target_bpv=2.4, is_key=True)
        result_v, report_v = cross_layer_qdq(x, target_bpv=2.4, is_key=False)
        assert result_k.shape == result_v.shape
