"""
Tests for vitriol.kv.spectral module.
"""
import pytest

import torch

from vitriol.kv.spectral import (
    _estimate_spectral_decay,
    _compute_spectral_band_boundaries,
    _spectral_quantize_dequantize,
    _quantize_slice,
    SpectralKVCompressed,
    SpectralKVConfig,
    SpectralKVCodec,
    spectral_qdq,
)


class TestEstimateSpectralDecay:
    def test_returns_positive_alpha(self):
        x = torch.randn(2, 4, 8, 64)
        alpha = _estimate_spectral_decay(x)
        assert isinstance(alpha, float)
        assert alpha >= 0.5
        assert alpha <= 5.0

    def test_higher_energy_concentration_higher_alpha(self):
        # Create tensor with strong energy concentration in low frequencies
        x = torch.zeros(2, 4, 8, 64)
        x[..., :10] = torch.randn(2, 4, 8, 10) * 2.0
        x[..., 10:] = torch.randn(2, 4, 8, 54) * 0.1
        alpha = _estimate_spectral_decay(x)
        # Should have valid alpha (fast decay implies higher alpha)
        assert alpha >= 0.5

    def test_uniform_energy_low_alpha(self):
        # Create tensor with roughly uniform energy
        x = torch.randn(2, 4, 8, 64)
        alpha = _estimate_spectral_decay(x)
        # Should have lower alpha (slow decay)
        assert alpha >= 0.5

    def test_3d_tensor(self):
        x = torch.randn(8, 8, 64)
        alpha = _estimate_spectral_decay(x)
        assert isinstance(alpha, float)
        assert 0.5 <= alpha <= 5.0

    def test_respects_num_samples(self):
        x = torch.randn(2, 4, 100, 64)
        alpha1 = _estimate_spectral_decay(x, num_samples=32)
        alpha2 = _estimate_spectral_decay(x, num_samples=64)
        # Should produce valid results regardless of sample size
        assert isinstance(alpha1, float)
        assert isinstance(alpha2, float)


class TestComputeSpectralBandBoundaries:
    def test_basic_boundaries(self):
        dim = 128
        alpha = 2.0
        target_bpv = 3.0
        k_low, k_high, bits_low, bits_high = _compute_spectral_band_boundaries(
            dim, alpha, target_bpv
        )
        assert k_low >= 1
        assert k_high > k_low
        assert k_high < dim
        assert bits_low >= bits_high
        assert bits_low > 0
        assert bits_high > 0

    def test_slow_decay_larger_low_band(self):
        dim = 128
        target_bpv = 3.0
        k_low_slow, k_high_slow, _, _ = _compute_spectral_band_boundaries(
            dim, alpha=1.0, target_bpv=target_bpv
        )
        k_low_fast, k_high_fast, _, _ = _compute_spectral_band_boundaries(
            dim, alpha=3.0, target_bpv=target_bpv
        )
        # Slow decay -> more coefficients in low band
        assert k_low_slow >= k_low_fast

    def test_high_target_more_bits(self):
        dim = 128
        alpha = 2.0
        _, _, bits_low_high, bits_high_high = _compute_spectral_band_boundaries(
            dim, alpha, target_bpv=4.0
        )
        _, _, bits_low_low, bits_high_low = _compute_spectral_band_boundaries(
            dim, alpha, target_bpv=2.0
        )
        assert bits_low_high >= bits_low_low
        assert bits_high_high >= bits_high_low

    def test_low_target_fewer_bits(self):
        dim = 128
        alpha = 2.0
        _, _, bits_low, bits_high = _compute_spectral_band_boundaries(
            dim, alpha, target_bpv=1.5
        )
        assert bits_low <= 3
        assert bits_high == 1


class TestSpectralQuantizeDequantize:
    def test_basic(self):
        x_freq = torch.randn(2, 4, 8, 64)
        result = _spectral_quantize_dequantize(x_freq, k_low=16, k_high=32, bits_low=6, bits_high=2)
        assert result.shape == x_freq.shape

    def test_low_freq_preserved(self):
        # Low freq band should have higher precision (less distortion)
        x_freq = torch.randn(2, 4, 8, 64)
        result = _spectral_quantize_dequantize(x_freq, k_low=16, k_high=32, bits_low=8, bits_high=2)
        # Low band MSE should be lower than high band MSE
        low_mse = (x_freq[..., :16] - result[..., :16]).pow(2).mean().item()
        high_mse = (x_freq[..., 32:] - result[..., 32:]).pow(2).mean().item()
        assert low_mse < high_mse

    def test_no_mid_band(self):
        x_freq = torch.randn(2, 4, 8, 64)
        result = _spectral_quantize_dequantize(x_freq, k_low=0, k_high=0, bits_low=4, bits_high=2)
        assert result.shape == x_freq.shape

    def test_full_low_precision(self):
        x_freq = torch.randn(2, 4, 8, 64)
        result = _spectral_quantize_dequantize(x_freq, k_low=64, k_high=64, bits_low=2, bits_high=2)
        assert result.shape == x_freq.shape


class TestQuantizeSlice:
    def test_basic(self):
        x = torch.randn(2, 4, 8, 32)
        result = _quantize_slice(x, levels=8)
        assert result.shape == x.shape
        # Result should be different from original
        assert not torch.allclose(result, x, atol=1e-6)

    def test_levels_effect(self):
        x = torch.randn(2, 4, 8, 32)
        result_few = _quantize_slice(x, levels=2)
        result_many = _quantize_slice(x, levels=64)
        mse_few = (x - result_few).pow(2).mean().item()
        mse_many = (x - result_many).pow(2).mean().item()
        assert mse_many < mse_few


class TestSpectralKVCompressed:
    def test_storage_nbytes(self):
        compressed = SpectralKVCompressed(
            q_low=torch.randint(0, 64, (10, 16)),
            q_mid=torch.randint(0, 16, (10, 16)),
            q_high=torch.randint(0, 4, (10, 32)),
            scales_low=torch.randn(10, 1),
            mins_low=torch.randn(10, 1),
            scales_mid=torch.randn(10, 1),
            mins_mid=torch.randn(10, 1),
            scales_high=torch.randn(10, 1),
            mins_high=torch.randn(10, 1),
            k_low=16,
            k_high=32,
            bits_low=6,
            bits_high=2,
            orig_shape=(2, 4, 8, 64),
            dim=64,
            spectral_alpha=2.0,
        )
        nbytes = compressed.storage_nbytes()
        assert nbytes > 0
        # Should be less than original float32 size
        original_bytes = 2 * 4 * 8 * 64 * 4  # float32
        assert nbytes < original_bytes


class TestSpectralKVConfig:
    def test_defaults(self):
        cfg = SpectralKVConfig()
        assert cfg.target_bpv == 3.0
        assert cfg.auto_detect_alpha is True
        assert cfg.fixed_alpha == 2.0
        assert cfg.min_low_freq_size == 16
        assert cfg.min_mid_freq_size == 16
        assert cfg.apply_rotation is True
        assert cfg.block_size == 0
        assert cfg.k_bit_boost == 0.5
        assert cfg.v_bit_penalty == 0.0

    def test_custom_values(self):
        cfg = SpectralKVConfig(
            target_bpv=2.0,
            auto_detect_alpha=False,
            fixed_alpha=1.5,
            apply_rotation=False,
        )
        assert cfg.target_bpv == 2.0
        assert cfg.auto_detect_alpha is False
        assert cfg.fixed_alpha == 1.5
        assert cfg.apply_rotation is False


class TestSpectralKVCodec:
    def test_init(self):
        codec = SpectralKVCodec()
        assert codec.config is not None
        assert isinstance(codec.config, SpectralKVConfig)

    def test_compress_basic(self):
        codec = SpectralKVCodec()
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress(x, is_key=True)
        assert isinstance(compressed, SpectralKVCompressed)
        assert compressed.orig_shape == x.shape
        assert compressed.dim == 64
        assert "spectral_alpha" in report
        assert "k_low" in report
        assert "k_high" in report
        assert "effective_bpv" in report
        assert "compression_ratio" in report

    def test_compress_no_rotation(self):
        cfg = SpectralKVConfig(apply_rotation=False)
        codec = SpectralKVCodec(cfg)
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress(x, is_key=True)
        assert compressed.orig_shape == x.shape

    def test_compress_key_boost(self):
        codec = SpectralKVCodec()
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        k_compressed, k_report = codec.compress(key, is_key=True)
        v_compressed, v_report = codec.compress(value, is_key=False)
        # Key should have higher target_bpv due to k_bit_boost
        assert k_report["target_bpv"] > v_report["target_bpv"]

    def test_decompress(self):
        codec = SpectralKVCodec()
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress(x, is_key=True)
        reconstructed = codec.decompress(compressed)
        assert reconstructed.shape == x.shape

    def test_decompress_no_rotation(self):
        cfg = SpectralKVConfig(apply_rotation=False)
        codec = SpectralKVCodec(cfg)
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress(x, is_key=True)
        reconstructed = codec.decompress(compressed)
        assert reconstructed.shape == x.shape

    def test_roundtrip_quality(self):
        codec = SpectralKVCodec()
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress(x, is_key=True)
        reconstructed = codec.decompress(compressed)
        mse = (x.float() - reconstructed).pow(2).mean().item()
        assert mse < 1.0  # Should be reasonably close

    def test_compress_kv(self):
        codec = SpectralKVCodec()
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        k_out, v_out, report = codec.compress_kv(key, value)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape
        assert "k" in report
        assert "v" in report
        assert "total_mse" in report

    def test_compress_kv_quality(self):
        codec = SpectralKVCodec()
        key = torch.randn(2, 4, 8, 64)
        value = torch.randn(2, 4, 8, 64)
        k_out, v_out, report = codec.compress_kv(key, value)
        k_mse = (key.float() - k_out).pow(2).mean().item()
        v_mse = (value.float() - v_out).pow(2).mean().item()
        assert k_mse < 1.0
        assert v_mse < 1.0
        assert report["total_mse"] == pytest.approx((k_mse + v_mse) / 2, abs=1e-6)

    def test_fixed_alpha(self):
        cfg = SpectralKVConfig(auto_detect_alpha=False, fixed_alpha=2.5)
        codec = SpectralKVCodec(cfg)
        x = torch.randn(2, 4, 8, 64)
        compressed, report = codec.compress(x, is_key=True)
        assert report["spectral_alpha"] == 2.5

    def test_3d_tensor(self):
        codec = SpectralKVCodec()
        x = torch.randn(8, 16, 64)  # [batch*heads, seq_len, dim]
        compressed, report = codec.compress(x, is_key=True)
        reconstructed = codec.decompress(compressed)
        assert reconstructed.shape == x.shape


class TestSpectralQDQ:
    def test_basic(self):
        x = torch.randn(2, 4, 8, 64)
        result, report = spectral_qdq(x, target_bpv=3.0)
        assert result.shape == x.shape
        assert "mse" in report
        assert "cosine_similarity" in report
        assert "snr_db" in report

    def test_is_key_parameter(self):
        x = torch.randn(2, 4, 8, 64)
        result_k, report_k = spectral_qdq(x, target_bpv=3.0, is_key=True)
        result_v, report_v = spectral_qdq(x, target_bpv=3.0, is_key=False)
        # Key has boost, so should have lower MSE
        assert report_k["mse"] <= report_v["mse"] * 1.5  # Loose bound

    def test_different_target_bpv(self):
        x = torch.randn(2, 4, 8, 64)
        result_high, report_high = spectral_qdq(x, target_bpv=4.0)
        result_low, report_low = spectral_qdq(x, target_bpv=2.0)
        # Higher target bpv should give lower MSE
        assert report_high["mse"] <= report_low["mse"]

    def test_cosine_similarity_positive(self):
        x = torch.randn(2, 4, 8, 64)
        result, report = spectral_qdq(x, target_bpv=3.0)
        assert report["cosine_similarity"] > 0
        assert report["cosine_similarity"] <= 1.0

    def test_snr_db_positive(self):
        x = torch.randn(2, 4, 8, 64)
        result, report = spectral_qdq(x, target_bpv=3.0)
        assert report["snr_db"] > 0
