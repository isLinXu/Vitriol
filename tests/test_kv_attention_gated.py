"""
Tests for vitriol.kv.attention_gated module.
"""
import math

import torch
import torch.nn.functional as F

from vitriol.kv.attention_gated import (
    compute_attention_importance,
    compute_importance_tiers,
    _quantize_tier_blockwise,
    _quantize_tiered_kv,
    attention_gated_sdpa,
    AttentionGatedKVCompressed,
    AttentionGatedKVConfig,
    AttentionGatedKVCodec,
    _auto_tune_tier_fractions,
    attention_gated_qdq,
)


class TestComputeAttentionImportance:
    def test_output_shape(self):
        batch, heads, q_len, dim = 2, 4, 8, 64
        query = torch.randn(batch, heads, q_len, dim)
        key = torch.randn(batch, heads, 16, dim)
        importance = compute_attention_importance(query, key)
        assert importance.shape == (batch, heads, 16)

    def test_values_in_range(self):
        query = torch.randn(2, 4, 8, 64)
        key = torch.randn(2, 4, 16, 64)
        importance = compute_attention_importance(query, key)
        assert torch.all(importance >= 0)
        assert torch.all(importance <= 1)
        # Should sum to approximately 1
        sums = importance.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_with_custom_scale(self):
        query = torch.randn(2, 4, 8, 64)
        key = torch.randn(2, 4, 16, 64)
        importance1 = compute_attention_importance(query, key, scale=1.0)
        importance2 = compute_attention_importance(query, key, scale=0.1)
        # Different scales should produce different distributions
        assert not torch.allclose(importance1, importance2)

    def test_sharpening_effect(self):
        query = torch.randn(2, 4, 8, 64)
        key = torch.randn(2, 4, 16, 64)
        importance = compute_attention_importance(query, key)
        # Due to pow(1.5) sharpening, max should be higher than uniform
        max_val = importance.max()
        uniform_max = 1.0 / 16  # uniform distribution max
        assert max_val > uniform_max

    def test_topk_fraction(self):
        query = torch.randn(2, 4, 8, 64)
        key = torch.randn(2, 4, 16, 64)
        importance = compute_attention_importance(query, key, topk_fraction=0.5)
        assert importance.shape == (2, 4, 16)


class TestComputeImportanceTiers:
    def test_three_tiers(self):
        importance = torch.tensor([[[0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.15, 0.25, 0.35, 0.45]]])
        high, medium, low = compute_importance_tiers(importance)
        assert high.shape == importance.shape
        assert medium.shape == importance.shape
        assert low.shape == importance.shape
        # Each position should be in exactly one tier
        for i in range(importance.shape[-1]):
            tier_count = int(high[0, 0, i]) + int(medium[0, 0, i]) + int(low[0, 0, i])
            assert tier_count == 1

    def test_tier_fractions(self):
        importance = torch.rand(2, 4, 100)
        high, medium, low = compute_importance_tiers(
            importance, tier_fractions=(0.2, 0.3, 0.5)
        )
        n_high = high.sum().item()
        n_medium = medium.sum().item()
        n_low = low.sum().item()
        total = importance.numel()
        assert n_high == int(100 * 0.2) * 2 * 4
        assert n_medium == int(100 * 0.3) * 2 * 4
        assert n_low == total - n_high - n_medium

    def test_high_tier_has_highest_importance(self):
        importance = torch.tensor([[[0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6]]])
        high, medium, low = compute_importance_tiers(
            importance, tier_fractions=(0.25, 0.25, 0.5)
        )
        # High tier should include positions with highest importance
        high_positions = torch.where(high[0, 0])[0]
        for pos in high_positions:
            imp = importance[0, 0, pos].item()
            # All high-tier importance should be >= all low-tier
            low_positions = torch.where(low[0, 0])[0]
            for low_pos in low_positions:
                assert imp >= importance[0, 0, low_pos].item()


class TestQuantizeTierBlockwise:
    def test_basic_quantization(self):
        x = torch.randn(2, 4, 8, 64)
        mask = torch.ones(2, 4, 8, dtype=torch.bool)
        result, bpv = _quantize_tier_blockwise(x, mask, levels=8, block_size=32)
        assert result.shape == x.shape
        assert bpv == math.log2(8)

    def test_empty_mask_returns_clone(self):
        x = torch.randn(2, 4, 8, 64)
        mask = torch.zeros(2, 4, 8, dtype=torch.bool)
        result, bpv = _quantize_tier_blockwise(x, mask, levels=8, block_size=32)
        assert torch.allclose(result, x)
        assert bpv == 0.0

    def test_quantization_reduces_precision(self):
        x = torch.randn(2, 4, 8, 64)
        mask = torch.ones(2, 4, 8, dtype=torch.bool)
        result, _ = _quantize_tier_blockwise(x, mask, levels=4, block_size=32)
        # Result should be different from original (quantized)
        assert not torch.allclose(result, x, atol=1e-6)

    def test_padded_block_size(self):
        x = torch.randn(2, 4, 8, 50)  # 50 is not divisible by 32
        mask = torch.ones(2, 4, 8, dtype=torch.bool)
        result, bpv = _quantize_tier_blockwise(x, mask, levels=8, block_size=32)
        assert result.shape == x.shape

    def test_only_affects_masked_positions(self):
        x = torch.randn(2, 4, 8, 64)
        mask = torch.zeros(2, 4, 8, dtype=torch.bool)
        mask[0, 0, :4] = True  # Only first 4 positions
        result, _ = _quantize_tier_blockwise(x, mask, levels=8, block_size=32)
        # Unmasked positions should be unchanged
        assert torch.allclose(result[0, 0, 4:], x[0, 0, 4:])


class TestQuantizeTieredKV:
    def test_basic(self):
        x = torch.randn(2, 4, 16, 64)
        importance = torch.softmax(torch.randn(2, 4, 16), dim=-1)
        result, report = _quantize_tiered_kv(x, importance)
        assert result.shape == x.shape
        assert "method" in report
        assert report["method"] == "attention_gated_kv"
        assert "effective_bpv" in report
        assert "compression_ratio" in report
        assert "mse" in report

    def test_report_fields(self):
        x = torch.randn(2, 4, 16, 64)
        importance = torch.softmax(torch.randn(2, 4, 16), dim=-1)
        result, report = _quantize_tiered_kv(x, importance)
        assert "n_high" in report
        assert "n_medium" in report
        assert "n_low" in report
        assert "n_skip" in report
        assert "high_bpv" in report
        assert "medium_bpv" in report
        assert "low_bpv" in report
        assert report["compression_ratio"] > 0

    def test_skip_threshold(self):
        x = torch.randn(2, 4, 16, 64)
        importance = torch.ones(2, 4, 16) / 16
        # Set some very low importance
        importance[0, 0, 0] = 0.00001
        result, report = _quantize_tiered_kv(x, importance, skip_threshold=0.001)
        assert report["n_skip"] >= 0

    def test_compression_ratio_calculation(self):
        x = torch.randn(2, 4, 16, 64)
        importance = torch.softmax(torch.randn(2, 4, 16), dim=-1)
        result, report = _quantize_tiered_kv(x, importance)
        # Compression ratio should be 16 / effective_bpv (vs fp16)
        expected_ratio = 16.0 / max(report["effective_bpv"], 0.01)
        assert abs(report["compression_ratio"] - expected_ratio) < 1e-6


class TestAttentionGatedSDPA:
    def test_output_shape(self):
        batch, heads, q_len, dim = 2, 4, 8, 64
        query = torch.randn(batch, heads, q_len, dim)
        key = torch.randn(batch, heads, 16, dim)
        value = torch.randn(batch, heads, 16, dim)
        importance = torch.softmax(torch.randn(batch, heads, 16), dim=-1)
        output, report = attention_gated_sdpa(query, key, value, importance)
        assert output.shape == query.shape
        assert "method" in report

    def test_with_causal_mask(self):
        batch, heads, seq, dim = 2, 4, 8, 64
        query = torch.randn(batch, heads, seq, dim)
        key = torch.randn(batch, heads, seq, dim)
        value = torch.randn(batch, heads, seq, dim)
        importance = torch.softmax(torch.randn(batch, heads, seq), dim=-1)
        output, report = attention_gated_sdpa(
            query, key, value, importance, is_causal=True
        )
        assert output.shape == query.shape

    def test_different_from_standard_sdpa(self):
        batch, heads, q_len, dim = 2, 4, 8, 64
        query = torch.randn(batch, heads, q_len, dim)
        key = torch.randn(batch, heads, 16, dim)
        value = torch.randn(batch, heads, 16, dim)
        importance = torch.softmax(torch.randn(batch, heads, 16), dim=-1)
        gated_output, _ = attention_gated_sdpa(query, key, value, importance)
        standard_output = F.scaled_dot_product_attention(query, key, value)
        # Should be different due to quantization
        assert not torch.allclose(gated_output, standard_output, atol=1e-3)


class TestAttentionGatedKVCompressed:
    def test_storage_nbytes(self):
        compressed = AttentionGatedKVCompressed(
            q_high=torch.randint(0, 64, (10, 32)),
            q_medium=torch.randint(0, 8, (15, 32)),
            q_low=torch.randint(0, 2, (25, 32)),
            high_mask=torch.ones(2, 4, 50, dtype=torch.bool),
            medium_mask=torch.zeros(2, 4, 50, dtype=torch.bool),
            low_mask=torch.zeros(2, 4, 50, dtype=torch.bool),
            skip_mask=torch.zeros(2, 4, 50, dtype=torch.bool),
            scales_high=torch.randn(10, 1),
            mins_high=torch.randn(10, 1),
            scales_medium=torch.randn(15, 1),
            mins_medium=torch.randn(15, 1),
            scales_low=torch.randn(25, 1),
            mins_low=torch.randn(25, 1),
            tier_levels=(64, 8, 2),
            tier_fractions=(0.2, 0.3, 0.5),
            orig_shape=(2, 4, 50, 32),
            is_key=True,
        )
        nbytes = compressed.storage_nbytes()
        assert nbytes > 0


class TestAttentionGatedKVConfig:
    def test_default_values(self):
        cfg = AttentionGatedKVConfig()
        assert cfg.target_bpv == 2.4
        assert cfg.tier_levels == (128, 8, 4)
        assert cfg.tier_fractions == (0.15, 0.35, 0.50)
        assert cfg.skip_threshold == 0.001
        assert cfg.importance_gamma == 1.5
        assert cfg.apply_rotation is False
        assert cfg.weighted_attention is True
        assert cfg.auto_tune_tiers is True
        assert cfg.block_size == 32

    def test_custom_values(self):
        cfg = AttentionGatedKVConfig(
            target_bpv=3.0,
            tier_levels=(64, 16, 4),
            skip_threshold=0.01,
            apply_rotation=True,
        )
        assert cfg.target_bpv == 3.0
        assert cfg.tier_levels == (64, 16, 4)
        assert cfg.skip_threshold == 0.01
        assert cfg.apply_rotation is True


class TestAutoTuneTierFractions:
    def test_very_sparse(self):
        # Very sparse: top 5% carry 80% of mass
        importance = torch.zeros(1, 1, 100)
        importance[0, 0, :5] = 0.16  # 5 positions * 0.16 = 0.8
        fracs = _auto_tune_tier_fractions(importance, target_bpv=2.4, tier_levels=(64, 8, 2))
        assert fracs[0] == 0.10  # high
        assert fracs[1] == 0.20  # medium
        assert fracs[2] == 0.70  # low

    def test_diffuse_attention(self):
        # Diffuse: many positions carry mass
        importance = torch.ones(1, 1, 100) / 100
        fracs = _auto_tune_tier_fractions(importance, target_bpv=2.4, tier_levels=(64, 8, 2))
        assert fracs[0] == 0.30  # high
        assert fracs[1] == 0.35  # medium
        assert fracs[2] == 0.35  # low


class TestAttentionGatedKVCodec:
    def test_init_default_config(self):
        codec = AttentionGatedKVCodec()
        assert codec.config is not None
        assert isinstance(codec.config, AttentionGatedKVConfig)
        assert codec._cached_importance is None

    def test_init_custom_config(self):
        cfg = AttentionGatedKVConfig(target_bpv=3.0)
        codec = AttentionGatedKVCodec(cfg)
        assert codec.config.target_bpv == 3.0

    def test_compress_with_importance(self):
        codec = AttentionGatedKVCodec()
        x = torch.randn(2, 4, 16, 64)
        importance = torch.softmax(torch.randn(2, 4, 16), dim=-1)
        result, report = codec.compress(x, is_key=True, importance=importance)
        assert result.shape == x.shape
        assert "effective_bpv" in report
        assert "snr_db" in report
        assert codec._cached_importance is not None

    def test_compress_without_importance(self):
        codec = AttentionGatedKVCodec()
        x = torch.randn(2, 4, 16, 64)
        result, report = codec.compress(x, is_key=True)
        assert result.shape == x.shape
        # Should use uniform importance
        assert "effective_bpv" in report

    def test_compress_with_rotation(self):
        cfg = AttentionGatedKVConfig(apply_rotation=True)
        codec = AttentionGatedKVCodec(cfg)
        x = torch.randn(2, 4, 16, 64)
        result, report = codec.compress(x, is_key=True)
        assert result.shape == x.shape

    def test_compress_kv(self):
        codec = AttentionGatedKVCodec()
        key = torch.randn(2, 4, 16, 64)
        value = torch.randn(2, 4, 16, 64)
        query = torch.randn(2, 4, 8, 64)
        k_out, v_out, report = codec.compress_kv(key, value, query=query)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape
        assert "k" in report
        assert "v" in report
        assert "total_mse" in report

    def test_compress_kv_without_query(self):
        codec = AttentionGatedKVCodec()
        key = torch.randn(2, 4, 16, 64)
        value = torch.randn(2, 4, 16, 64)
        k_out, v_out, report = codec.compress_kv(key, value)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape

    def test_k_level_boost(self):
        cfg = AttentionGatedKVConfig(k_level_boost=2)
        codec = AttentionGatedKVCodec(cfg)
        x = torch.randn(2, 4, 16, 64)
        importance = torch.softmax(torch.randn(2, 4, 16), dim=-1)
        _, report_k = codec.compress(x, is_key=True, importance=importance)
        _, report_v = codec.compress(x, is_key=False, importance=importance)
        # K should have higher tier levels than V
        assert report_k["tier_levels"][0] > report_v["tier_levels"][0]

    def test_v_level_penalty(self):
        cfg = AttentionGatedKVConfig(v_level_penalty=1)
        codec = AttentionGatedKVCodec(cfg)
        x = torch.randn(2, 4, 16, 64)
        importance = torch.softmax(torch.randn(2, 4, 16), dim=-1)
        _, report_v = codec.compress(x, is_key=False, importance=importance)
        # V low tier should be reduced
        assert report_v["tier_levels"][2] >= 2


class TestAttentionGatedQDQ:
    def test_basic(self):
        x = torch.randn(2, 4, 16, 64)
        result, report = attention_gated_qdq(x, target_bpv=2.4)
        assert result.shape == x.shape
        assert "cosine_similarity" in report
        assert "mse" in report

    def test_with_importance(self):
        x = torch.randn(2, 4, 16, 64)
        importance = torch.softmax(torch.randn(2, 4, 16), dim=-1)
        result, report = attention_gated_qdq(x, target_bpv=2.4, importance=importance)
        assert result.shape == x.shape
        assert report["cosine_similarity"] > 0

    def test_is_key_parameter(self):
        x = torch.randn(2, 4, 16, 64)
        result_k, report_k = attention_gated_qdq(x, target_bpv=2.4, is_key=True)
        result_v, report_v = attention_gated_qdq(x, target_bpv=2.4, is_key=False)
        # Same input, but different reports due to is_key flag
        assert result_k.shape == result_v.shape
