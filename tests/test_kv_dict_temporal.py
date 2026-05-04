"""
Tests for vitriol.kv.dict_kv and vitriol.kv.temporal_pooling modules.
"""
import pytest
import torch

from vitriol.kv.dict_kv import (
    orthogonal_matching_pursuit,
    learn_dictionary_ksvd,
    learn_dictionary_online,
    DictKVCompressed,
    DictKVConfig,
    DictKVCodec,
    dict_kv_qdq,
)
from vitriol.kv.temporal_pooling import (
    TemporalPoolingConfig,
    _temporal_decay_mask,
    temporal_importance_attention,
    create_temporal_pooling_config_from_preset,
)


# ─────────────────────────────────────────────────────────────
# Orthogonal Matching Pursuit
# ─────────────────────────────────────────────────────────────

class TestOrthogonalMatchingPursuit:
    def test_omp_basic(self):
        # Simple dictionary: 4 atoms in 8-dim space
        dictionary = torch.eye(8)[:4]  # [4, 8]
        x = dictionary[0].unsqueeze(0) + 0.5 * dictionary[1].unsqueeze(0)  # [1, 8]

        coeffs, indices = orthogonal_matching_pursuit(x, dictionary, sparsity=2)

        assert coeffs.shape == (1, 4)
        assert indices.shape == (1, 2)
        # Should select atoms 0 and 1
        assert 0 in indices[0].tolist()

    def test_omp_batch(self):
        dictionary = torch.randn(16, 32)
        dictionary = dictionary / (dictionary.norm(dim=-1, keepdim=True) + 1e-12)
        x = torch.randn(4, 3, 10, 32)  # [batch, heads, seq, dim]

        coeffs, indices = orthogonal_matching_pursuit(x, dictionary, sparsity=4)

        assert coeffs.shape == (4, 3, 10, 16)
        assert indices.shape == (4, 3, 10, 4)

    def test_omp_sparsity_one(self):
        dictionary = torch.randn(8, 16)
        dictionary = dictionary / (dictionary.norm(dim=-1, keepdim=True) + 1e-12)
        x = torch.randn(2, 16)

        coeffs, indices = orthogonal_matching_pursuit(x, dictionary, sparsity=1)

        assert coeffs.shape == (2, 8)
        assert indices.shape == (2, 1)

    def test_omp_reconstruction_quality(self):
        torch.manual_seed(42)
        dictionary = torch.randn(64, 128)
        dictionary = dictionary / (dictionary.norm(dim=-1, keepdim=True) + 1e-12)
        x = torch.randn(10, 128)

        coeffs, indices = orthogonal_matching_pursuit(x, dictionary, sparsity=4)
        recon = coeffs.reshape(-1, 64) @ dictionary
        mse = (x - recon).pow(2).mean().item()
        # With enough atoms, reconstruction should be reasonable
        assert mse < x.pow(2).mean().item()


# ─────────────────────────────────────────────────────────────
# Dictionary Learning
# ─────────────────────────────────────────────────────────────

class TestDictionaryLearning:
    def test_ksvd_basic(self):
        torch.manual_seed(42)
        data = torch.randn(100, 16)
        dictionary = learn_dictionary_ksvd(data, n_atoms=8, n_iterations=3, sparsity=2)

        assert dictionary.shape == (8, 16)
        # Atoms should be normalized
        norms = dictionary.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_ksvd_fewer_samples_than_atoms(self):
        torch.manual_seed(42)
        data = torch.randn(5, 8)
        dictionary = learn_dictionary_ksvd(data, n_atoms=10, n_iterations=2, sparsity=2)

        assert dictionary.shape == (10, 8)

    def test_online_basic(self):
        torch.manual_seed(42)
        data = torch.randn(200, 16)
        dictionary = learn_dictionary_online(data, n_atoms=8, n_iterations=5, sparsity=2)

        assert dictionary.shape == (8, 16)
        norms = dictionary.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_online_fewer_samples_than_atoms(self):
        torch.manual_seed(42)
        data = torch.randn(3, 8)
        dictionary = learn_dictionary_online(data, n_atoms=10, n_iterations=2, sparsity=1)

        assert dictionary.shape == (10, 8)


# ─────────────────────────────────────────────────────────────
# DictKVCompressed
# ─────────────────────────────────────────────────────────────

class TestDictKVCompressed:
    def test_storage_nbytes(self):
        compressed = DictKVCompressed(
            coefficients=torch.randn(2, 4, 8, 16),
            indices=torch.randint(0, 16, (2, 4, 8, 4)),
            values=torch.randn(2, 4, 8, 4),
            orig_shape=(2, 4, 8, 32),
            n_atoms=16,
            sparsity=4,
            is_key=True,
        )
        nbytes = compressed.storage_nbytes()
        assert nbytes > 0

    def test_storage_nbytes_with_residual(self):
        compressed = DictKVCompressed(
            coefficients=torch.randn(2, 4, 8, 16),
            indices=torch.randint(0, 16, (2, 4, 8, 4)),
            values=torch.randn(2, 4, 8, 4),
            q_residual=torch.randint(0, 4, (2, 4, 8, 4)),
            residual_scales=torch.randn(2, 4, 8, 4, 1),
            residual_mins=torch.randn(2, 4, 8, 4, 1),
            orig_shape=(2, 4, 8, 32),
            n_atoms=16,
            sparsity=4,
            is_key=True,
        )
        nbytes_with = compressed.storage_nbytes()
        assert nbytes_with > 0


# ─────────────────────────────────────────────────────────────
# DictKVConfig
# ─────────────────────────────────────────────────────────────

class TestDictKVConfig:
    def test_defaults(self):
        cfg = DictKVConfig()
        assert cfg.n_atoms == 1024
        assert cfg.sparsity == 4
        assert cfg.learning_method == "online"
        assert cfg.quantize_residual is True

    def test_custom_values(self):
        cfg = DictKVConfig(n_atoms=256, sparsity=2, learning_method="ksvd")
        assert cfg.n_atoms == 256
        assert cfg.sparsity == 2
        assert cfg.learning_method == "ksvd"


# ─────────────────────────────────────────────────────────────
# DictKVCodec
# ─────────────────────────────────────────────────────────────

class TestDictKVCodec:
    def test_init_defaults(self):
        codec = DictKVCodec()
        assert codec.config.n_atoms == 1024
        assert codec.dictionary is None
        assert codec._dictionary_learned is False

    def test_init_custom_config(self):
        cfg = DictKVConfig(n_atoms=64, sparsity=2)
        codec = DictKVCodec(cfg)
        assert codec.config.n_atoms == 64

    def test_ensure_dictionary_creates_random(self):
        codec = DictKVCodec(DictKVConfig(n_atoms=8))
        dictionary = codec._ensure_dictionary(16, torch.device("cpu"), is_key=True)

        assert dictionary.shape == (8, 16)
        norms = dictionary.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_learn_dictionary(self):
        torch.manual_seed(42)
        codec = DictKVCodec(DictKVConfig(n_atoms=8, sparsity=2, learning_iterations=3))
        kv = torch.randn(2, 4, 10, 16)

        codec.learn_dictionary([kv], is_key=True)

        assert codec._dictionary_k is not None
        assert codec._dictionary_learned is True
        assert codec._dictionary_k.shape == (8, 16)

    def test_compress_decompress_roundtrip(self):
        torch.manual_seed(42)
        cfg = DictKVConfig(n_atoms=16, sparsity=2, quantize_residual=False)
        codec = DictKVCodec(cfg)
        x = torch.randn(1, 2, 4, 8)

        compressed, report = codec.compress(x, is_key=True)
        reconstructed = codec.decompress(compressed)

        assert reconstructed.shape == x.shape
        assert report["method"] == "dict_kv"
        assert report["is_key"] is True
        assert "compression_ratio" in report

    def test_compress_key_vs_value_sparsity(self):
        torch.manual_seed(42)
        cfg = DictKVConfig(n_atoms=16, sparsity=2, k_sparsity_boost=1, v_sparsity_penalty=0)
        codec = DictKVCodec(cfg)
        x = torch.randn(1, 2, 4, 8)

        k_compressed, k_report = codec.compress(x, is_key=True)
        v_compressed, v_report = codec.compress(x, is_key=False)

        assert k_report["sparsity"] == 3  # 2 + 1 boost
        assert v_report["sparsity"] == 2

    def test_compress_kv_api(self):
        torch.manual_seed(42)
        cfg = DictKVConfig(n_atoms=16, sparsity=2, quantize_residual=False)
        codec = DictKVCodec(cfg)
        key = torch.randn(1, 2, 4, 8)
        value = torch.randn(1, 2, 4, 8)

        k_out, v_out, report = codec.compress_kv(key, value)

        assert k_out.shape == key.shape
        assert v_out.shape == value.shape
        assert "k_mse" in report
        assert "v_mse" in report
        assert "total_mse" in report

    def test_dictionary_property(self):
        codec = DictKVCodec()
        assert codec.dictionary is None

        codec._dictionary_k = torch.randn(8, 16)
        assert codec.dictionary is codec._dictionary_k

    def test_compress_with_residual(self):
        torch.manual_seed(42)
        cfg = DictKVConfig(
            n_atoms=16,
            sparsity=2,
            quantize_residual=True,
            residual_levels=4,
            residual_block_size=4,
        )
        codec = DictKVCodec(cfg)
        x = torch.randn(1, 2, 4, 8)

        compressed, _ = codec.compress(x, is_key=True)
        assert compressed.q_residual is not None
        assert compressed.residual_scales is not None

        reconstructed = codec.decompress(compressed)
        assert reconstructed.shape == x.shape


# ─────────────────────────────────────────────────────────────
# dict_kv_qdq
# ─────────────────────────────────────────────────────────────

class TestDictKVQDQ:
    def test_qdq_basic(self):
        torch.manual_seed(42)
        x = torch.randn(1, 2, 4, 8)

        reconstructed, report = dict_kv_qdq(x, n_atoms=16, sparsity=2, learn_from_data=True)

        assert reconstructed.shape == x.shape
        assert "mse" in report
        assert "cosine_similarity" in report
        assert "snr_db" in report
        assert report["method"] == "dict_kv"

    def test_qdq_without_learning(self):
        torch.manual_seed(42)
        x = torch.randn(1, 2, 4, 8)

        reconstructed, report = dict_kv_qdq(x, n_atoms=16, sparsity=2, learn_from_data=False)

        assert reconstructed.shape == x.shape
        assert "mse" in report


# ─────────────────────────────────────────────────────────────
# Temporal Decay Mask
# ─────────────────────────────────────────────────────────────

class TestTemporalDecayMask:
    def test_decay_shape(self):
        decay = _temporal_decay_mask(10, torch.device("cpu"), decay_rate=0.5)
        assert decay.shape == (10,)

    def test_newest_is_one(self):
        decay = _temporal_decay_mask(10, torch.device("cpu"), decay_rate=1.0)
        assert decay[-1].item() == pytest.approx(1.0, abs=1e-5)

    def test_oldest_is_smallest(self):
        decay = _temporal_decay_mask(10, torch.device("cpu"), decay_rate=1.0)
        assert decay[0].item() < decay[-1].item()

    def test_zero_decay_rate_returns_ones(self):
        decay = _temporal_decay_mask(10, torch.device("cpu"), decay_rate=0.0)
        assert torch.allclose(decay, torch.ones(10))

    def test_single_element(self):
        decay = _temporal_decay_mask(1, torch.device("cpu"), decay_rate=1.0)
        assert decay.shape == (1,)
        assert decay[0].item() == pytest.approx(1.0, abs=1e-5)


# ─────────────────────────────────────────────────────────────
# Temporal Importance Attention
# ─────────────────────────────────────────────────────────────

class TestTemporalImportanceAttention:
    def test_basic_attention(self):
        torch.manual_seed(42)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        config = TemporalPoolingConfig(temporal_decay=0.5, temperature=0.1)

        output, report = temporal_importance_attention(q, k, v, config=config)

        assert output.shape == (1, 2, 4, 8)
        assert "sparsity" in report
        assert "effective_mass" in report
        assert report["config_temporal_decay"] == 0.5

    def test_causal_attention(self):
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        config = TemporalPoolingConfig()

        output, report = temporal_importance_attention(q, k, v, config=config, is_causal=True)

        assert output.shape == (1, 1, 4, 8)

    def test_no_temporal_decay(self):
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        config = TemporalPoolingConfig(enable_temporal_decay=False)

        output, report = temporal_importance_attention(q, k, v, config=config)

        assert output.shape == (1, 1, 4, 8)

    def test_fixed_threshold(self):
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        config = TemporalPoolingConfig(adaptive_threshold=False, fixed_threshold=0.01)

        output, report = temporal_importance_attention(q, k, v, config=config)

        assert output.shape == (1, 1, 4, 8)

    def test_attention_mask_bool(self):
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        mask = torch.tensor([[True, True, False, False]])
        config = TemporalPoolingConfig()

        output, report = temporal_importance_attention(q, k, v, config=config, attn_mask=mask)

        assert output.shape == (1, 1, 4, 8)

    def test_attention_mask_additive(self):
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        mask = torch.zeros(1, 1, 4, 4)
        mask[..., :, 2:] = float("-inf")
        config = TemporalPoolingConfig()

        output, report = temporal_importance_attention(q, k, v, config=config, attn_mask=mask)

        assert output.shape == (1, 1, 4, 8)

    def test_min_attention_mass_protection(self):
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        config = TemporalPoolingConfig(min_attention_mass=0.99)

        output, report = temporal_importance_attention(q, k, v, config=config)

        assert output.shape == (1, 1, 4, 8)
        assert report["effective_mass"] > 0.9

    def test_custom_scale(self):
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        config = TemporalPoolingConfig()

        output1, _ = temporal_importance_attention(q, k, v, config=config, scale=0.5)
        output2, _ = temporal_importance_attention(q, k, v, config=config, scale=2.0)

        assert output1.shape == output2.shape

    def test_dropout(self):
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        config = TemporalPoolingConfig()

        output, report = temporal_importance_attention(q, k, v, config=config, dropout_p=0.5)
        assert output.shape == (1, 1, 4, 8)


# ─────────────────────────────────────────────────────────────
# TemporalPoolingConfig presets
# ─────────────────────────────────────────────────────────────

class TestTemporalPoolingPresets:
    def test_balanced_preset(self):
        config = create_temporal_pooling_config_from_preset("balanced")
        assert config.temporal_decay == 0.5
        assert config.min_attention_mass == 0.95
        assert config.enable_temporal_decay is True

    def test_conservative_preset(self):
        config = create_temporal_pooling_config_from_preset("conservative")
        assert config.temporal_decay == 0.2
        assert config.min_attention_mass == 0.98

    def test_aggressive_preset(self):
        config = create_temporal_pooling_config_from_preset("aggressive")
        assert config.temporal_decay == 1.5
        assert config.min_attention_mass == 0.90

    def test_ultra_long_preset(self):
        config = create_temporal_pooling_config_from_preset("ultra_long")
        assert config.temporal_decay == 0.8
        assert config.enable_pooling is True
        assert config.pool_group_size == 4

    def test_unknown_preset_defaults_to_balanced(self):
        config = create_temporal_pooling_config_from_preset("unknown")
        assert config.temporal_decay == 0.5
        assert config.min_attention_mass == 0.95
