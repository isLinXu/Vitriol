"""Tests for KV optimization modules."""

import torch
from unittest.mock import MagicMock

from vitriol.kv.layer_adaptive import (
    _layer_depth_weight, _compute_head_entropy, LayerAdaptiveConfig, LayerAdaptiveBitAllocator
)
from vitriol.kv.temporal_pooling import (
    TemporalPoolingConfig, temporal_importance_attention
)
from vitriol.kv.backend import KVMeta, KVStoreBackend
from vitriol.kv.codec import (
    PackedKVTensor,
    ResidualQJLPackedTensor,
    _rademacher_projection,
    clear_projection_cache,
    kv_bytes_per_value,
)
from vitriol.kv.cache_store import KVCacheStoreConfig


# ─────────────────────────────────────────────────────────────────────────────
# layer_adaptive tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerDepthWeight:
    def test_u_shape(self):
        # Early layer should have high weight
        w0 = _layer_depth_weight(0, 32, "u_shape")
        w_mid = _layer_depth_weight(16, 32, "u_shape")
        w_last = _layer_depth_weight(31, 32, "u_shape")
        assert w0 > w_mid
        assert w_last > w_mid
        assert 0 < w_mid <= 1.0

    def test_decay(self):
        w0 = _layer_depth_weight(0, 32, "decay")
        w_last = _layer_depth_weight(31, 32, "decay")
        assert w0 > w_last
        assert w0 <= 1.0

    def test_inv_decay(self):
        w0 = _layer_depth_weight(0, 32, "inv_decay")
        w_last = _layer_depth_weight(31, 32, "inv_decay")
        assert w0 < w_last
        assert w_last <= 1.0

    def test_uniform(self):
        w0 = _layer_depth_weight(0, 32, "uniform")
        w_mid = _layer_depth_weight(16, 32, "uniform")
        assert w0 == w_mid == 1.0

    def test_single_layer(self):
        assert _layer_depth_weight(0, 1, "u_shape") == 1.0


class TestLayerAdaptiveConfig:
    def test_defaults(self):
        cfg = LayerAdaptiveConfig()
        assert cfg.target_avg_bits > 0

    def test_custom(self):
        cfg = LayerAdaptiveConfig(target_avg_bits=2.5)
        assert cfg.target_avg_bits == 2.5


class TestLayerAdaptiveBitAllocator:
    def test_init(self):
        allocator = LayerAdaptiveBitAllocator(LayerAdaptiveConfig())
        assert allocator is not None

    def test_allocate_shape(self):
        allocator = LayerAdaptiveBitAllocator(LayerAdaptiveConfig(target_avg_bits=3.0))
        b, h, s, d = 2, 8, 64, 128
        q = torch.randn(b, h, s, d)
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)

        k_bits, v_bits, report = allocator.allocate(
            query=q, key=k, value=v,
            layer_idx=5, total_layers=32,
            layer_type="full_attention",
        )
        assert k_bits.shape == (b, h)
        assert v_bits.shape == (b, h)
        assert all(k_bits.flatten() > 0)
        assert all(v_bits.flatten() > 0)

    def test_entropy_subsampling_avoids_full_randperm(self, monkeypatch):
        query = torch.randn(1, 2, 512, 32)
        key = torch.randn(1, 2, 512, 32)

        def _boom(*_args, **_kwargs):
            raise AssertionError("randperm should not be used for entropy subsampling")

        monkeypatch.setattr(torch, "randperm", _boom)

        entropy = _compute_head_entropy(query, key, num_sample_positions=64)
        assert entropy.shape == (1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# temporal_pooling tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalPoolingConfig:
    def test_defaults(self):
        cfg = TemporalPoolingConfig()
        assert cfg.temporal_decay == 0.5
        assert cfg.temperature == 0.1
        assert cfg.min_attention_mass == 0.95
        assert cfg.adaptive_threshold is True

    def test_custom(self):
        cfg = TemporalPoolingConfig(temporal_decay=1.0, temperature=0.5)
        assert cfg.temporal_decay == 1.0
        assert cfg.temperature == 0.5


class TestTemporalImportanceAttention:
    def test_basic(self):
        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)
        result, report = temporal_importance_attention(q, k, v)
        assert result.shape == q.shape
        assert isinstance(report, dict)

    def test_causal(self):
        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)
        result, report = temporal_importance_attention(q, k, v, is_causal=True)
        assert result.shape == q.shape

    def test_no_decay(self):
        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)
        cfg = TemporalPoolingConfig(enable_temporal_decay=False)
        result, report = temporal_importance_attention(q, k, v, config=cfg)
        assert result.shape == q.shape


# ─────────────────────────────────────────────────────────────────────────────
# backend tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKVMeta:
    def test_creation(self):
        meta = KVMeta(model_id="test", device="cpu", dtype="bfloat16")
        assert meta.model_id == "test"
        assert meta.device == "cpu"
        assert meta.dtype == "bfloat16"


class TestKVStoreBackend:
    def test_init(self):
        cfg = KVCacheStoreConfig()
        backend = KVStoreBackend(store_cfg=cfg)
        assert backend.store_cfg == cfg

    def test_ensure_store(self):
        cfg = KVCacheStoreConfig()
        backend = KVStoreBackend(store_cfg=cfg)
        handle = MagicMock()
        store = backend._ensure_store(handle, 0)
        assert store is not None
        # Second call should return same store
        store2 = backend._ensure_store(handle, 0)
        assert store is store2

    def test_write_kv(self):
        cfg = KVCacheStoreConfig()
        backend = KVStoreBackend(store_cfg=cfg)
        handle = MagicMock()
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)
        backend.write_kv(handle, 0, k, v, {"q_len": 10})
        # Should have created store
        assert hasattr(handle, "_vitriol_kv_stores")


# ─────────────────────────────────────────────────────────────────────────────
# codec tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPackedKVTensor:
    def test_storage_nbytes(self):
        q_data = torch.randint(0, 255, (10, 10), dtype=torch.uint8)
        scales = torch.randn(10)
        mins = torch.randn(10)
        packed = PackedKVTensor(
            q_data=q_data, scales=scales, mins=mins,
            orig_shape=(10, 10), padded_last_dim=10,
            block_size=32, levels=8, bit_width=3,
        )
        nbytes = packed.storage_nbytes()
        assert nbytes > 0
        assert nbytes == q_data.numel() + scales.numel() * 4 + mins.numel() * 4


class TestResidualQJLPackedTensor:
    def test_storage_nbytes(self):
        base = PackedKVTensor(
            q_data=torch.randint(0, 255, (10, 10), dtype=torch.uint8),
            scales=torch.randn(10), mins=torch.randn(10),
            orig_shape=(10, 10), padded_last_dim=10,
            block_size=32, levels=8, bit_width=3,
        )
        residual = ResidualQJLPackedTensor(
            base=base,
            projection=torch.randn(10, 8),
            residual_sign_bits=torch.randint(0, 255, (10,), dtype=torch.uint8),
            residual_scale=torch.randn(10),
            residual_norms=torch.randn(10),
            sketch_dim=8,
            seed=42,
        )
        nbytes = residual.storage_nbytes()
        assert nbytes > base.storage_nbytes()

    def test_rademacher_projection_cache_reuses_projection_matrix(self):
        clear_projection_cache()
        p1 = _rademacher_projection(32, 8, seed=7, device=torch.device("cpu"), use_cache=True, dtype=torch.float32)
        p2 = _rademacher_projection(32, 8, seed=7, device=torch.device("cpu"), use_cache=True, dtype=torch.float32)

        assert p1 is p2
        assert clear_projection_cache() >= 1


class TestKVBytesPerValue:
    def test_none(self):
        assert kv_bytes_per_value(None) == 2.0

    def test_turbo2(self):
        assert kv_bytes_per_value("turbo2") == 2.5 / 8.0

    def test_turbo3(self):
        assert kv_bytes_per_value("turbo3") == 3.5 / 8.0

    def test_turbo4(self):
        assert kv_bytes_per_value("turbo4") == 4.25 / 8.0

    def test_q8_0(self):
        assert kv_bytes_per_value("q8_0") == 8.5 / 8.0

    def test_q4_0(self):
        assert kv_bytes_per_value("q4_0") == 4.5 / 8.0

    def test_bf16(self):
        assert kv_bytes_per_value("bf16") == 2.0

    def test_dict_adaptive(self):
        d = {"name": "adaptive_bits", "target_avg_bits": 3.5}
        assert kv_bytes_per_value(d) == 3.5 / 8.0

    def test_dict_turbo(self):
        d = {"name": "turbo", "format": "turbo3"}
        assert kv_bytes_per_value(d) == 3.5 / 8.0

    def test_dict_unknown(self):
        d = {"format": "turbo3"}  # missing "name" key
        assert kv_bytes_per_value(d) == 2.0


# ─────────────────────────────────────────────────────────────────────────────
# cache_store config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKVCacheStoreConfig:
    def test_defaults(self):
        cfg = KVCacheStoreConfig()
        assert cfg.enable_turbo_quant is False
        assert cfg.turbo_k_format == "turbo3"
        assert cfg.turbo_v_format == "turbo3"
        assert cfg.turbo_block_size == 32

    def test_custom(self):
        cfg = KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_format="turbo4",
            turbo_bits=4.0,
        )
        assert cfg.enable_turbo_quant is True
        assert cfg.turbo_bits == 4.0
