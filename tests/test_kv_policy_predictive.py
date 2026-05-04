"""Tests for kv/policy.py and kv/predictive.py."""

from unittest.mock import MagicMock

import pytest
import torch

from vitriol.kv.policy import (
    ApproxMode,
    KVLayerType,
    KVPolicy,
    KVPolicyPreset,
    SafeExactPolicy,
    Turbo3ExactKApproxVPolicy,
    _classify_kv_layer_name,
    _full_attention_layers,
    _full_attention_pos,
    apply_policy_to_store_cfg,
    build_policy,
    classify_kv_layer,
    list_policy_presets,
    resolve_layer_strategy,
)
from vitriol.kv.predictive import (
    PredictiveKVCodec,
    PredictiveKVCompressed,
    PredictiveKVConfig,
    _compute_lpc_for_kv,
    _dequantize_residual_blockwise,
    _estimate_lpc_yule_walker,
    _predict_and_residual,
    _quantize_residual_blockwise,
    _select_prediction_order,
    predictive_qdq,
)


class TestApproxMode:
    """Tests for ApproxMode enum."""

    def test_values(self):
        assert ApproxMode.EXACT == "exact"
        assert ApproxMode.APPROX == "approx"


class TestKVPolicy:
    """Tests for KVPolicy."""

    def test_creation(self):
        policy = KVPolicy(mode=ApproxMode.EXACT)
        assert policy.mode == ApproxMode.EXACT


class TestSafeExactPolicy:
    """Tests for SafeExactPolicy."""

    def test_creation(self):
        policy = SafeExactPolicy()
        assert policy.mode == ApproxMode.EXACT


class TestTurbo3ExactKApproxVPolicy:
    """Tests for Turbo3ExactKApproxVPolicy."""

    def test_defaults(self):
        policy = Turbo3ExactKApproxVPolicy()
        assert policy.mode == ApproxMode.APPROX
        assert policy.turbo_k_format == "turbo3"
        assert policy.turbo_block_size == 32
        assert policy.enable_turbo_residual_qjl is True

    def test_custom_values(self):
        policy = Turbo3ExactKApproxVPolicy(
            v_quantize_only_first_n_full_attention_layers=3,
            turbo_block_size=64,
            enable_sparse_v=True,
        )
        assert policy.v_quantize_only_first_n_full_attention_layers == 3
        assert policy.turbo_block_size == 64
        assert policy.enable_sparse_v is True


class TestClassifyKVLayerName:
    """Tests for _classify_kv_layer_name."""

    def test_full_attention(self):
        assert _classify_kv_layer_name("full_attention") is KVLayerType.FULL_ATTENTION
        assert _classify_kv_layer_name("mha") is KVLayerType.FULL_ATTENTION
        assert _classify_kv_layer_name("gqa") is KVLayerType.FULL_ATTENTION
        assert _classify_kv_layer_name("") is KVLayerType.FULL_ATTENTION

    def test_sliding_window(self):
        assert _classify_kv_layer_name("sliding_window") is KVLayerType.SLIDING_WINDOW
        assert _classify_kv_layer_name("local_attention") is KVLayerType.SLIDING_WINDOW

    def test_mla(self):
        assert _classify_kv_layer_name("mla") is KVLayerType.MLA
        assert _classify_kv_layer_name("latent") is KVLayerType.MLA

    def test_hash(self):
        assert _classify_kv_layer_name("hash_attention") is KVLayerType.HASH_ATTENTION

    def test_linear(self):
        assert _classify_kv_layer_name("linear_attention") is KVLayerType.LINEAR
        assert _classify_kv_layer_name("mamba") is KVLayerType.LINEAR
        assert _classify_kv_layer_name("ssm") is KVLayerType.LINEAR

    def test_other(self):
        assert _classify_kv_layer_name("unknown_type") is KVLayerType.OTHER


class TestClassifyKVLayer:
    """Tests for classify_kv_layer."""

    def test_no_layer_types(self):
        handle = MagicMock()
        handle.layer_types = None
        assert classify_kv_layer(handle, 0) is KVLayerType.FULL_ATTENTION

    def test_with_layer_types(self):
        handle = MagicMock()
        handle.layer_types = ["full_attention", "sliding_window", "mla"]
        assert classify_kv_layer(handle, 0) is KVLayerType.FULL_ATTENTION
        assert classify_kv_layer(handle, 1) is KVLayerType.SLIDING_WINDOW
        assert classify_kv_layer(handle, 2) is KVLayerType.MLA

    def test_out_of_bounds(self):
        handle = MagicMock()
        handle.layer_types = ["full"]
        assert classify_kv_layer(handle, 99) is KVLayerType.FULL_ATTENTION
        assert classify_kv_layer(handle, -1) is KVLayerType.FULL_ATTENTION


class TestResolveLayerStrategy:
    """Tests for resolve_layer_strategy."""

    def test_exact_policy(self):
        policy = SafeExactPolicy()
        handle = MagicMock()
        handle.layer_types = ["full_attention"]
        strategy = resolve_layer_strategy(policy, handle, 0)
        assert strategy.layer_type is KVLayerType.FULL_ATTENTION
        assert strategy.turbo_quantize_k is False
        assert strategy.turbo_quantize_v is False

    def test_turbo3_full_attention(self):
        policy = Turbo3ExactKApproxVPolicy()
        handle = MagicMock()
        handle.layer_types = ["full_attention"]
        strategy = resolve_layer_strategy(policy, handle, 0)
        assert strategy.layer_type is KVLayerType.FULL_ATTENTION
        assert strategy.turbo_quantize_k is True
        assert strategy.turbo_quantize_v is True

    def test_turbo3_sliding_window(self):
        policy = Turbo3ExactKApproxVPolicy()
        handle = MagicMock()
        handle.layer_types = ["sliding_window"]
        strategy = resolve_layer_strategy(policy, handle, 0)
        assert strategy.turbo_quantize_k is False


class TestFullAttentionLayers:
    """Tests for _full_attention_layers and _full_attention_pos."""

    def test_full_attention_layers_with_types(self):
        handle = MagicMock()
        handle.layer_types = ["full", "sliding", "full"]
        result = _full_attention_layers(handle)
        assert result == [0, 2]

    def test_full_attention_layers_without_types(self):
        handle = [1, 2, 3]  # has __len__
        result = _full_attention_layers(handle)
        assert result == [0, 1, 2]

    def test_full_attention_pos(self):
        handle = MagicMock()
        handle.layer_types = ["full", "sliding", "full"]
        assert _full_attention_pos(handle, 0) == 0
        assert _full_attention_pos(handle, 2) == 1
        assert _full_attention_pos(handle, 1) is None


class TestApplyPolicyToStoreCfg:
    """Tests for apply_policy_to_store_cfg."""

    def test_exact_policy(self):
        from vitriol.kv.cache_store import KVCacheStoreConfig
        policy = SafeExactPolicy()
        handle = MagicMock()
        handle.layer_types = ["full_attention"]
        base_cfg = KVCacheStoreConfig()
        result = apply_policy_to_store_cfg(base_cfg, policy, handle, 0)
        assert result.turbo_quantize_k is False

    def test_turbo3_policy(self):
        from vitriol.kv.cache_store import KVCacheStoreConfig
        policy = Turbo3ExactKApproxVPolicy()
        handle = MagicMock()
        handle.layer_types = ["full_attention"]
        base_cfg = KVCacheStoreConfig()
        result = apply_policy_to_store_cfg(base_cfg, policy, handle, 0)
        assert result.turbo_quantize_k is True


class TestKVPolicyPreset:
    """Tests for KVPolicyPreset."""

    def test_to_dict(self):
        preset = KVPolicyPreset.safe_default()
        d = preset.to_dict()
        assert d["name"] == "safe"
        assert d["policy_type"] == "SafeExactPolicy"

    def test_from_dict(self):
        d = {"name": "test", "policy_type": "SafeExactPolicy", "params": {}}
        preset = KVPolicyPreset.from_dict(d)
        assert preset.name == "test"

    def test_safe_default(self):
        preset = KVPolicyPreset.safe_default()
        assert preset.name == "safe"

    def test_balanced_default(self):
        preset = KVPolicyPreset.balanced_default()
        assert preset.name == "balanced"
        assert preset.policy_type == "Turbo3ExactKApproxVPolicy"

    def test_aggressive_default(self):
        preset = KVPolicyPreset.aggressive_default()
        assert preset.name == "aggressive"

    def test_ultra_long_default(self):
        preset = KVPolicyPreset.ultra_long_default()
        assert preset.name == "ultra-long"

    def test_smart_default(self):
        preset = KVPolicyPreset.smart_default()
        assert preset.name == "smart"
        assert "enable_temporal_pooling" in preset.params

    def test_spectral_default(self):
        preset = KVPolicyPreset.spectral_default()
        assert preset.name == "spectral"
        assert preset.params["enable_spectral_kv"] is True

    def test_predictive_default(self):
        preset = KVPolicyPreset.predictive_default()
        assert preset.name == "predictive"
        assert preset.params["enable_predictive_kv"] is True

    def test_ultimate_default(self):
        preset = KVPolicyPreset.ultimate_default()
        assert preset.name == "ultimate"

    def test_attention_gated_default(self):
        preset = KVPolicyPreset.attention_gated_default()
        assert preset.name == "attention-gated"


class TestListPolicyPresets:
    """Tests for list_policy_presets."""

    def test_list_returns_many(self):
        presets = list_policy_presets()
        names = [p.name for p in presets]
        assert "safe" in names
        assert "balanced" in names
        assert "aggressive" in names
        assert len(presets) >= 10


class TestBuildPolicy:
    """Tests for build_policy."""

    def test_safe_exact(self):
        preset = KVPolicyPreset.safe_default()
        policy = build_policy(preset)
        assert isinstance(policy, SafeExactPolicy)

    def test_turbo3(self):
        preset = KVPolicyPreset.balanced_default()
        policy = build_policy(preset)
        assert isinstance(policy, Turbo3ExactKApproxVPolicy)

    def test_unknown_policy_type(self):
        preset = KVPolicyPreset(name="bad", policy_type="UnknownPolicy", params={})
        with pytest.raises(ValueError, match="Unknown policy_type"):
            build_policy(preset)


class TestEstimateLPCYuleWalker:
    """Tests for _estimate_lpc_yule_walker."""

    def test_basic(self):
        x = torch.randn(10, 50)
        coeffs = _estimate_lpc_yule_walker(x, order=2)
        assert coeffs.shape == (10, 2)

    def test_short_sequence(self):
        x = torch.randn(5, 3)
        coeffs = _estimate_lpc_yule_walker(x, order=5)
        assert coeffs.shape == (5, 5)
        assert torch.allclose(coeffs, torch.zeros_like(coeffs))

    def test_single_stream(self):
        x = torch.randn(50)
        coeffs = _estimate_lpc_yule_walker(x, order=2)
        assert coeffs.shape == (2,)


class TestComputeLPCForKV:
    """Tests for _compute_lpc_for_kv."""

    def test_4d_tensor(self):
        x = torch.randn(2, 4, 20, 64)
        coeffs = _compute_lpc_for_kv(x, order=2)
        assert coeffs.shape == (2, 4, 2, 64)

    def test_3d_tensor(self):
        x = torch.randn(3, 20, 64)
        coeffs = _compute_lpc_for_kv(x, order=2, per_head=True)
        assert coeffs.shape == (3, 2, 64)

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            _compute_lpc_for_kv(torch.randn(2, 64), order=2)


class TestPredictAndResidual:
    """Tests for _predict_and_residual."""

    def test_basic(self):
        x = torch.randn(2, 4, 20, 64)
        coeffs = torch.randn(2, 4, 2, 64) * 0.1
        predicted, residual = _predict_and_residual(x, coeffs)
        assert predicted.shape == x.shape
        assert residual.shape == x.shape
        assert not torch.allclose(residual, torch.zeros_like(residual))


class TestQuantizeDequantizeResidual:
    """Tests for residual quantization/dequantization."""

    def test_roundtrip(self):
        residual = torch.randn(2, 4, 20, 64)
        q, scales, mins = _quantize_residual_blockwise(residual, levels=8, block_size=32)
        dq = _dequantize_residual_blockwise(q, scales, mins, residual.shape, block_size=32)
        assert dq.shape == residual.shape

    def test_pad_handling(self):
        residual = torch.randn(2, 4, 20, 50)  # not divisible by 32
        q, scales, mins = _quantize_residual_blockwise(residual, levels=8, block_size=32)
        assert q is not None


class TestSelectPredictionOrder:
    """Tests for _select_prediction_order."""

    def test_short_sequence(self):
        x = torch.randn(2, 4, 5, 64)
        order = _select_prediction_order(x, max_order=8, min_order=1)
        assert 1 <= order <= 8

    def test_long_sequence(self):
        x = torch.randn(2, 4, 100, 64)
        order = _select_prediction_order(x, max_order=4, min_order=1)
        assert 1 <= order <= 4


class TestPredictiveKVConfig:
    """Tests for PredictiveKVConfig."""

    def test_defaults(self):
        cfg = PredictiveKVConfig()
        assert cfg.target_bpv == 3.0
        assert cfg.prediction_order == 2
        assert cfg.auto_order is True
        assert cfg.block_size == 32

    def test_custom(self):
        cfg = PredictiveKVConfig(target_bpv=2.0, prediction_order=4, auto_order=False)
        assert cfg.target_bpv == 2.0
        assert cfg.prediction_order == 4
        assert cfg.auto_order is False


class TestPredictiveKVCodec:
    """Tests for PredictiveKVCodec."""

    def test_creation(self):
        codec = PredictiveKVCodec()
        assert codec.config.target_bpv == 3.0

    def test_compress_decompress_4d(self):
        codec = PredictiveKVCodec(PredictiveKVConfig(target_bpv=3.0, auto_order=False, prediction_order=2))
        x = torch.randn(2, 4, 50, 64)
        compressed, report = codec.compress(x, is_key=True)
        assert isinstance(compressed, PredictiveKVCompressed)
        assert "order" in report
        assert "prediction_gain" in report

        reconstructed = codec.decompress(compressed)
        assert reconstructed.shape == x.shape

    def test_compress_decompress_3d(self):
        codec = PredictiveKVCodec(PredictiveKVConfig(target_bpv=3.0, auto_order=False, prediction_order=2))
        x = torch.randn(3, 50, 64)
        compressed, report = codec.compress(x, is_key=False)
        reconstructed = codec.decompress(compressed)
        assert reconstructed.shape == x.shape

    def test_compress_kv(self):
        codec = PredictiveKVCodec(PredictiveKVConfig(target_bpv=3.0, auto_order=False, prediction_order=2))
        key = torch.randn(2, 4, 50, 64)
        value = torch.randn(2, 4, 50, 64)
        k_out, v_out, report = codec.compress_kv(key, value)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape
        assert "k_mse" in report
        assert "v_mse" in report
        assert "total_mse" in report

    def test_derive_levels(self):
        codec = PredictiveKVCodec()
        assert codec._derive_levels(4.0, is_key=True) >= 16
        assert codec._derive_levels(3.0, is_key=True) >= 8
        assert codec._derive_levels(2.0, is_key=True) >= 4
        assert codec._derive_levels(1.0, is_key=True) >= 2

    def test_storage_nbytes(self):
        codec = PredictiveKVCodec(PredictiveKVConfig(target_bpv=3.0, auto_order=False, prediction_order=2))
        x = torch.randn(2, 4, 50, 64)
        compressed, _ = codec.compress(x, is_key=True)
        nbytes = compressed.storage_nbytes()
        assert nbytes > 0

    def test_short_sequence(self):
        codec = PredictiveKVCodec(PredictiveKVConfig(target_bpv=3.0, auto_order=False, prediction_order=2))
        x = torch.randn(2, 4, 3, 64)
        compressed, _ = codec.compress(x, is_key=True)
        assert compressed.order <= 1


class TestPredictiveQDQ:
    """Tests for predictive_qdq."""

    def test_basic(self):
        x = torch.randn(2, 4, 50, 64)
        reconstructed, report = predictive_qdq(x, target_bpv=3.0, is_key=True, order=2)
        assert reconstructed.shape == x.shape
        assert "mse" in report
        assert "cosine_similarity" in report
        assert "snr_db" in report

    def test_auto_order(self):
        x = torch.randn(2, 4, 50, 64)
        reconstructed, report = predictive_qdq(x, target_bpv=3.0, is_key=True, order=0)
        assert reconstructed.shape == x.shape
        assert "order" in report
