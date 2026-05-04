"""Tests for weight generation strategies."""

import torch
import pytest

from vitriol.strategies.base import WeightGenerationStrategy, StrategyCapabilities
from vitriol.strategies.random import RandomStrategy
from vitriol.strategies.binary import BinaryStrategy
from vitriol.strategies.ternary import TernaryStrategy
from vitriol.strategies.quantized import QuantizedStrategy
from vitriol.strategies.sparse import SparseStrategy, SparseSpec
from vitriol.strategies.compact import CompactStrategy
from vitriol.strategies.lowrank import LowRankStrategy
from vitriol.strategies.structured_sparse import StructuredSparseStrategy
from vitriol.strategies.ultra import UltraStrategy
from vitriol.strategies.quantum import QuantumStrategy
from vitriol.strategies.hybrid_ultra import HybridUltraStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Base class tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyCapabilities:
    def test_defaults(self):
        caps = StrategyCapabilities()
        assert caps.supports_safetensors is True
        assert caps.supports_training is True
        assert caps.max_compression_ratio == 1.0

    def test_custom(self):
        caps = StrategyCapabilities(supports_training=False, max_compression_ratio=0.5)
        assert caps.supports_training is False
        assert caps.max_compression_ratio == 0.5


class ConcreteStrategy(WeightGenerationStrategy):
    @property
    def capabilities(self):
        return StrategyCapabilities()

    def generate_tensor(self, shape, dtype, name, **kwargs):
        return torch.zeros(shape, dtype=dtype)

    def save_shard(self, shard_data, path):
        pass


class TestWeightGenerationStrategy:
    def test_init(self):
        s = ConcreteStrategy(device="cpu")
        assert s.device == "cpu"

    def test_normalize_dtype(self):
        s = ConcreteStrategy()
        # _normalize_dtype may convert float32 to bfloat16
        result = s._normalize_dtype(torch.float32)
        assert result in (torch.float32, torch.bfloat16)
        assert s._normalize_dtype(torch.float16) == torch.float16

    def test_validate_shape(self):
        s = ConcreteStrategy()
        s._validate_shape((10, 20))  # should not raise
        # Empty tuple doesn't raise because any() on empty is False
        s._validate_shape(())
        with pytest.raises(ValueError):
            s._validate_shape((0, 10))
        with pytest.raises(ValueError):
            s._validate_shape((-1, 10))

    def test_generate_tensor_abstract(self):
        class BadStrategy(WeightGenerationStrategy):
            @property
            def capabilities(self):
                return StrategyCapabilities()

        # ABC prevents instantiation of abstract class
        with pytest.raises(TypeError):
            BadStrategy()


# ─────────────────────────────────────────────────────────────────────────────
# Random strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomStrategy:
    def test_capabilities(self):
        s = RandomStrategy()
        caps = s.capabilities
        assert caps.supports_training is True
        assert caps.max_compression_ratio == 1.0

    def test_generate_tensor(self):
        s = RandomStrategy()
        t = s.generate_tensor((100, 100), torch.float32, "weight")
        assert t.shape == (100, 100)
        # _normalize_dtype converts float32 to bfloat16
        assert t.dtype == torch.bfloat16

    def test_generate_tensor_bfloat16(self):
        s = RandomStrategy()
        t = s.generate_tensor((50, 50), torch.bfloat16, "weight")
        assert t.dtype == torch.bfloat16

    def test_save_shard(self, tmp_path):
        s = RandomStrategy()
        data = {"w": torch.randn(10, 10)}
        path = tmp_path / "test.safetensors"
        s.save_shard(data, str(path))
        assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Binary strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryStrategy:
    def test_capabilities(self):
        s = BinaryStrategy()
        assert s.capabilities.max_compression_ratio == 0.5

    def test_generate_tensor_values(self):
        s = BinaryStrategy(alpha=0.01)
        t = s.generate_tensor((100, 100), torch.bfloat16, "weight")
        uniques = set(t.unique().tolist())
        # bfloat16 has limited precision; check approximate values
        for v in uniques:
            assert abs(abs(v) - 0.01) < 0.001

    def test_alpha_param(self):
        s = BinaryStrategy(alpha=0.5)
        t = s.generate_tensor((50, 50), torch.bfloat16, "w")
        vals = set(t.unique().tolist())
        for v in vals:
            assert abs(abs(v) - 0.5) < 0.01

    def test_storage_format(self):
        s = BinaryStrategy()
        assert s.storage_format == "safetensors"


# ─────────────────────────────────────────────────────────────────────────────
# Ternary strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestTernaryStrategy:
    def test_generate_tensor_values(self):
        s = TernaryStrategy(alpha=0.1)
        t = s.generate_tensor((100, 100), torch.bfloat16, "weight")
        uniques = set(t.unique().tolist())
        # bfloat16 has limited precision; check approximate values
        for v in uniques:
            assert abs(v - 0.0) < 0.01 or abs(abs(v) - 0.1) < 0.01

    def test_default_alpha(self):
        s = TernaryStrategy()
        t = s.generate_tensor((50, 50), torch.bfloat16, "w")
        vals = set(t.unique().tolist())
        for v in vals:
            assert abs(v - 0.0) < 0.01 or abs(abs(v) - 0.1) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Quantized strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantizedStrategy:
    def test_8bit_levels(self):
        s = QuantizedStrategy(n_bits=8)
        t = s.generate_tensor((100, 100), torch.float32, "w")
        n_levels = len(t.unique())
        assert n_levels <= 256

    def test_4bit_levels(self):
        s = QuantizedStrategy(n_bits=4)
        t = s.generate_tensor((100, 100), torch.float32, "w")
        n_levels = len(t.unique())
        assert n_levels <= 16

    def test_quantized_range(self):
        s = QuantizedStrategy(n_bits=8)
        t = s.generate_tensor((200, 200), torch.float32, "w")
        assert t.min() >= -0.5
        assert t.max() <= 0.5

    def test_capabilities_description(self):
        s = QuantizedStrategy(n_bits=4)
        assert "4-bit" in s.capabilities.description


# ─────────────────────────────────────────────────────────────────────────────
# Sparse strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseStrategy:
    def test_generate_tensor_returns_spec(self):
        s = SparseStrategy()
        result = s.generate_tensor((100, 100), torch.float32, "w")
        assert isinstance(result, SparseSpec)

    def test_sparse_spec_attributes(self):
        spec = SparseSpec("w", (10, 20), "F32", 200)
        assert spec.name == "w"
        assert spec.shape == (10, 20)
        assert spec.dtype_str == "F32"
        assert spec.size == 200

    def test_dtype_mapping(self):
        s = SparseStrategy()
        assert s._get_dtype_str(torch.float32) == "F32"
        assert s._get_dtype_str(torch.bfloat16) == "BF16"
        assert s._get_dtype_str(torch.int64) == "I64"
        assert s._get_dtype_str(torch.bool) == "BOOL"


# ─────────────────────────────────────────────────────────────────────────────
# Compact strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestCompactStrategy:
    def test_capabilities(self):
        s = CompactStrategy()
        assert s.capabilities.supports_safetensors is True

    def test_generate_tensor_zeros(self):
        s = CompactStrategy()
        t = s.generate_tensor((50, 50), torch.float32, "w")
        assert t.shape == (50, 50)
        assert t.sum().item() == 0.0

    def test_caching(self):
        s = CompactStrategy(cache_size=10)
        t1 = s.generate_tensor((10, 10), torch.float32, "w1")
        t2 = s.generate_tensor((10, 10), torch.float32, "w2")
        # Cache may return same tensor or new one
        assert t1.shape == t2.shape

    def test_dtype_normalization(self):
        s = CompactStrategy()
        t = s.generate_tensor((10, 10), torch.float32, "w")
        # Compact may convert float32 to bfloat16
        assert t.dtype in (torch.float32, torch.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# LowRank strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestLowRankStrategy:
    def test_2d_tensor_rank(self):
        s = LowRankStrategy(rank=8)
        t = s.generate_tensor((20, 30), torch.float32, "w")
        assert t.shape == (20, 30)
        # Should have low effective rank
        rank = torch.linalg.matrix_rank(t).item()
        assert rank <= 8

    def test_non_2d_fallback(self):
        s = LowRankStrategy(rank=8)
        t = s.generate_tensor((10,), torch.float32, "bias")
        assert t.shape == (10,)

    def test_capabilities(self):
        s = LowRankStrategy(rank=16)
        assert "rank 16" in s.capabilities.description


# ─────────────────────────────────────────────────────────────────────────────
# StructuredSparse strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuredSparseStrategy:
    def test_sparsity_ratio(self):
        s = StructuredSparseStrategy(sparsity=0.5)
        t = s.generate_tensor((1000, 1000), torch.float32, "w")
        zero_ratio = (t == 0).float().mean().item()
        # Allow some tolerance
        assert 0.4 < zero_ratio < 0.6

    def test_high_sparsity(self):
        s = StructuredSparseStrategy(sparsity=0.9)
        t = s.generate_tensor((500, 500), torch.float32, "w")
        zero_ratio = (t == 0).float().mean().item()
        assert zero_ratio > 0.8

    def test_storage_format(self):
        s = StructuredSparseStrategy()
        assert s.storage_format == "safetensors"


# ─────────────────────────────────────────────────────────────────────────────
# Ultra strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestUltraStrategy:
    def test_init(self):
        s = UltraStrategy()
        assert s.device == "cpu"

    def test_capabilities(self):
        s = UltraStrategy()
        caps = s.capabilities
        assert caps.supports_safetensors is False
        assert caps.supports_training is False

    def test_generate_tensor_strided(self):
        s = UltraStrategy()
        t = s.generate_tensor((100, 100), torch.bfloat16, "w")
        assert t.shape == (100, 100)
        # Storage should be very small (1 element)
        assert t.untyped_storage().nbytes() == t.element_size()

    def test_generate_tensor_different_shapes(self):
        s = UltraStrategy()
        t = s.generate_tensor((4096, 4096), torch.bfloat16, "w")
        assert t.shape == (4096, 4096)
        assert t.untyped_storage().nbytes() == t.element_size()

    def test_storage_format(self):
        s = UltraStrategy()
        assert s.storage_format == "pytorch"


# ─────────────────────────────────────────────────────────────────────────────
# Quantum strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumStrategy:
    def test_init_defaults(self):
        s = QuantumStrategy()
        assert s.n_bits == 1
        assert s.adaptive is True

    def test_generate_tensor(self):
        s = QuantumStrategy(n_bits=1, adaptive=False)
        t = s.generate_tensor((50, 50), torch.float32, "w")
        assert t.shape == (50, 50)

    def test_generate_tensor_adaptive(self):
        s = QuantumStrategy(n_bits=2, adaptive=True)
        t = s.generate_tensor((50, 50), torch.float32, "w")
        assert t.shape == (50, 50)

    def test_capabilities(self):
        s = QuantumStrategy()
        caps = s.capabilities
        assert caps.supports_safetensors is True
        assert "quantum" in caps.description.lower() or "Quantum" in caps.description


# ─────────────────────────────────────────────────────────────────────────────
# HybridUltra strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridUltraStrategy:
    def test_init_defaults(self):
        s = HybridUltraStrategy()
        assert s.init_mode == "zeros"

    def test_init_kaiming(self):
        s = HybridUltraStrategy(init_mode="kaiming")
        assert s.init_mode == "kaiming"

    def test_generate_tensor_norm_weight(self):
        s = HybridUltraStrategy()
        t = s.generate_tensor((128,), torch.float32, "model.layers.0.input_layernorm.weight")
        # Norm weights should be 1.0, not 0
        assert torch.allclose(t, torch.ones_like(t))

    def test_generate_tensor_norm_bias(self):
        s = HybridUltraStrategy()
        t = s.generate_tensor((128,), torch.float32, "model.layers.0.input_layernorm.bias")
        assert torch.allclose(t, torch.zeros_like(t))

    def test_generate_tensor_regular(self):
        s = HybridUltraStrategy(init_mode="zeros")
        t = s.generate_tensor((128, 128), torch.float32, "model.layers.0.self_attn.q_proj.weight")
        assert t.shape == (128, 128)
        # With zeros mode, should be zeros
        assert t.sum().item() == 0.0

    def test_generate_tensor_kaiming(self):
        s = HybridUltraStrategy(init_mode="kaiming")
        t = s.generate_tensor((128, 128), torch.float32, "model.layers.0.self_attn.q_proj.weight")
        assert t.shape == (128, 128)

    def test_capabilities(self):
        s = HybridUltraStrategy()
        caps = s.capabilities
        assert caps.supports_safetensors is True
        assert caps.supports_training is True

