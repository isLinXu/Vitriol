"""
Vitriol Framework Hardening Test Suite (v2)

Validates all security hardening, boundary condition handling,
error recovery, and robustness improvements applied to the framework.

Covers:
  - HybridUltra strategy: shape validation, dtype mapping, save_shard fallback,
    optimize_loaded_model edge cases, config validation
  - Base strategy: get_recipe, validate_config, _normalize_dtype float64
  - Compact/Random: safetensors ImportError fallback
  - Ultra: shape validation, save_shard error wrapping
  - Generator: _parse_size validation, torch.load security
  - Adapter: validate_config default implementation
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════
# § 1  HybridUltra Strategy Hardening Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHybridUltraShapeValidation:
    """Validate that HybridUltra rejects invalid shapes."""

    def test_empty_shape_raises(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        with pytest.raises(ValueError, match="invalid shape"):
            s.generate_tensor((), torch.float32, "test.empty")

    def test_zero_dim_raises(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        with pytest.raises(ValueError, match="invalid shape"):
            s.generate_tensor((0, 128), torch.float32, "test.zero_dim")

    def test_negative_dim_raises(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        with pytest.raises(ValueError, match="invalid shape"):
            s.generate_tensor((-1, 64), torch.float32, "test.neg_dim")

    def test_valid_shape_succeeds(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        t = s.generate_tensor((4, 8), torch.float32, "model.layers.0.self_attn.q_proj.weight")
        assert t.shape == (4, 8)

    def test_single_dim_succeeds(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        t = s.generate_tensor((128,), torch.float32, "model.layers.0.self_attn.q_proj.bias")
        assert t.shape == (128,)


class TestHybridUltraDtypeMapping:
    """Validate _resolve_dtype handles all dtypes correctly."""

    def test_float64_override_to_bfloat16(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy(dtype_override="bfloat16")
        assert s._resolve_dtype(torch.float64) == torch.bfloat16

    def test_float64_override_to_float64(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy(dtype_override="float64")
        assert s._resolve_dtype(torch.float32) == torch.float64

    def test_int_dtype_preserved(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy(dtype_override="bfloat16")
        assert s._resolve_dtype(torch.int64) == torch.int64

    def test_bool_dtype_preserved(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy(dtype_override="bfloat16")
        assert s._resolve_dtype(torch.bool) == torch.bool

    def test_none_override_preserves_original(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy(dtype_override=None)
        assert s._resolve_dtype(torch.float32) == torch.float32
        assert s._resolve_dtype(torch.float64) == torch.float64

    def test_unknown_override_preserves_and_warns(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy(dtype_override="invalid_type")
        with patch("vitriol.strategies.hybrid_ultra.logger") as mock_logger:
            result = s._resolve_dtype(torch.float32)
            assert result == torch.float32  # preserved
            mock_logger.warning.assert_called()


class TestHybridUltraSaveShard:
    """Validate save_shard fallback and error handling."""

    def test_save_empty_shard_is_noop(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        # Should not raise, just log warning
        s.save_shard({}, "/tmp/test_empty.safetensors")

    @patch("safetensors.torch.save_file", side_effect=ImportError)
    def test_safetensors_importerror_fallback(self, mock_save):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.bin")
            data = {"w": torch.zeros(2, 2)}
            # Should fall back to torch.save
            s.save_shard(data, path)
            assert os.path.exists(path)

    def test_safetensors_runtimeerror_fallback(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.bin")
            data = {"w": torch.zeros(2, 2)}
            with patch("safetensors.torch.save_file", side_effect=RuntimeError("test error")):
                # Should fall back to torch.save
                s.save_shard(data, path)
                # The fallback saves as .bin
                assert os.path.exists(path) or os.path.exists(
                    path.replace(".safetensors", ".bin")
                )


class TestHybridUltraOptimizeModel:
    """Validate optimize_loaded_model edge cases."""

    def test_empty_parameter_skipped(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        model = MagicMock()
        # Create a parameter with 0 elements
        empty_param = torch.nn.Parameter(torch.empty(0))
        model.named_parameters.return_value = [("empty.weight", empty_param)]
        stats = HybridUltraStrategy.optimize_loaded_model(model)
        assert stats["total_params"] == 1
        assert stats["zero_params"] == 0

    def test_nonzero_parameter_kept(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        model = MagicMock()
        nonzero_param = torch.nn.Parameter(torch.ones(4))
        model.named_parameters.return_value = [("norm.weight", nonzero_param)]
        stats = HybridUltraStrategy.optimize_loaded_model(model)
        assert stats["zero_params"] == 0

    def test_all_zeros_optimized(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        model = MagicMock()
        zero_param = torch.nn.Parameter(torch.zeros(16))
        model.named_parameters.return_value = [("linear.weight", zero_param)]
        stats = HybridUltraStrategy.optimize_loaded_model(model)
        assert stats["zero_params"] == 1
        assert stats["saved_mb"] > 0


class TestHybridUltraConfigValidation:
    """Validate validate_config method."""

    def test_valid_defaults(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy()
        assert s.validate_config() is True

    def test_invalid_init_mode_raises(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        with pytest.raises(ValueError, match="Invalid init_mode"):
            HybridUltraStrategy(init_mode="invalid_mode")

    def test_invalid_embed_init_raises(self):
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        with pytest.raises(ValueError, match="Invalid embed_init"):
            HybridUltraStrategy(embed_init="invalid_init")

    def test_invalid_dtype_override_raises_after_construction(self):
        """Test that validate_config catches invalid dtype_override."""
        from vitriol.strategies.hybrid_ultra import HybridUltraStrategy
        s = HybridUltraStrategy.__new__(HybridUltraStrategy)
        s.init_mode = "zeros"
        s.norm_init = True
        s.embed_init = "zeros"
        s.dtype_override = "invalid_dtype"
        s.device = "cpu"
        s._first_tensor_logged = False
        with pytest.raises(ValueError, match="invalid dtype_override"):
            s.validate_config()


class TestHybridUltraNormEmbedCoverage:
    """Validate expanded norm/embedding suffix coverage."""

    def test_glm_norm_detected(self):
        from vitriol.strategies.hybrid_ultra import _is_norm_weight
        assert _is_norm_weight("model.layers.0.input_layernorm.weight")

    def test_mamba_ln_f_detected(self):
        from vitriol.strategies.hybrid_ultra import _is_norm_weight
        assert _is_norm_weight("backbone.layers.0.ln_f.weight")

    def test_deepseek_attention_norm_detected(self):
        from vitriol.strategies.hybrid_ultra import _is_norm_weight
        assert _is_norm_weight("model.layers.0.attention_norm.weight")

    def test_glm_embedding_detected(self):
        from vitriol.strategies.hybrid_ultra import _is_embedding
        assert _is_embedding("transformer.embedding.word_embeddings.weight")

    def test_vision_embed_detected(self):
        from vitriol.strategies.hybrid_ultra import _is_embedding
        assert _is_embedding("model.vision_embed_tokens.weight")


# ═══════════════════════════════════════════════════════════════════════
# § 2  Base Strategy Hardening Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBaseStrategyMethods:
    """Validate default implementations in WeightGenerationStrategy."""

    def test_get_recipe_default(self):
        # Use a concrete subclass
        from vitriol.strategies.random import RandomStrategy
        s = RandomStrategy()
        recipe = s.get_recipe()
        assert "strategy" in recipe
        assert recipe["strategy"] == "RandomStrategy"

    def test_validate_config_default(self):
        from vitriol.strategies.random import RandomStrategy
        s = RandomStrategy()
        assert s.validate_config() is True

    def test_normalize_dtype_float64(self):
        from vitriol.strategies.compact import CompactStrategy
        s = CompactStrategy()
        assert s._normalize_dtype(torch.float64) == torch.bfloat16

    def test_normalize_dtype_float32(self):
        from vitriol.strategies.compact import CompactStrategy
        s = CompactStrategy()
        assert s._normalize_dtype(torch.float32) == torch.bfloat16

    def test_normalize_dtype_bfloat16_unchanged(self):
        from vitriol.strategies.compact import CompactStrategy
        s = CompactStrategy()
        assert s._normalize_dtype(torch.bfloat16) == torch.bfloat16

    def test_normalize_dtype_int_preserved(self):
        from vitriol.strategies.compact import CompactStrategy
        s = CompactStrategy()
        assert s._normalize_dtype(torch.int32) == torch.int32


# ═══════════════════════════════════════════════════════════════════════
# § 3  Ultra Strategy Hardening Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUltraShapeValidation:
    """Validate that Ultra rejects invalid shapes."""

    def test_empty_shape_raises(self):
        from vitriol.strategies.ultra import UltraStrategy
        s = UltraStrategy()
        with pytest.raises(ValueError, match="invalid shape"):
            s.generate_tensor((), torch.bfloat16, "test.empty")

    def test_zero_dim_raises(self):
        from vitriol.strategies.ultra import UltraStrategy
        s = UltraStrategy()
        with pytest.raises(ValueError, match="invalid shape"):
            s.generate_tensor((0, 128), torch.bfloat16, "test.zero")

    def test_valid_shape_succeeds(self):
        from vitriol.strategies.ultra import UltraStrategy
        s = UltraStrategy()
        t = s.generate_tensor((4, 8), torch.bfloat16, "model.layers.0.weight")
        assert t.shape == (4, 8)


class TestUltraSaveShard:
    """Validate Ultra save_shard error handling."""

    def test_empty_data_is_noop(self):
        from vitriol.strategies.ultra import UltraStrategy
        s = UltraStrategy()
        # Should not raise
        s.save_shard({}, "/tmp/test_empty.bin")

    def test_save_shard_wraps_oserror(self):
        from vitriol.strategies.ultra import UltraStrategy
        from vitriol.utils.exceptions import ShardSaveError
        s = UltraStrategy()
        with patch("torch.save", side_effect=OSError("disk full")):
            with pytest.raises(ShardSaveError):
                s.save_shard({"w": torch.zeros(2, 2)}, "/tmp/test.bin")


# ═══════════════════════════════════════════════════════════════════════
# § 4  Compact/Random Strategy Hardening Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCompactSaveShardFallback:
    """Validate Compact save_shard fallback behavior."""

    def test_empty_data_is_noop(self):
        from vitriol.strategies.compact import CompactStrategy
        s = CompactStrategy()
        s.save_shard({}, "/tmp/test_empty.safetensors")

    def test_safetensors_importerror_fallback(self):
        from vitriol.strategies.compact import CompactStrategy
        s = CompactStrategy()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.bin")
            data = {"w": torch.zeros(2, 2)}
            with patch.dict("sys.modules", {"safetensors": None, "safetensors.torch": None}):
                s.save_shard(data, path)


class TestRandomSaveShardFallback:
    """Validate Random save_shard fallback behavior."""

    def test_empty_data_is_noop(self):
        from vitriol.strategies.random import RandomStrategy
        s = RandomStrategy()
        s.save_shard({}, "/tmp/test_empty.safetensors")


# ═══════════════════════════════════════════════════════════════════════
# § 5  Generator Hardening Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGeneratorParseSize:
    """Validate _parse_size input validation."""

    def test_valid_gb(self):
        from vitriol.core.generator import MinimalWeightGenerator
        assert MinimalWeightGenerator._parse_size("5GB") == 5 * (1 << 30)

    def test_valid_mb(self):
        from vitriol.core.generator import MinimalWeightGenerator
        assert MinimalWeightGenerator._parse_size("512MB") == 512 * (1 << 20)

    def test_valid_kb(self):
        from vitriol.core.generator import MinimalWeightGenerator
        assert MinimalWeightGenerator._parse_size("1024KB") == 1024 * (1 << 10)

    def test_plain_bytes(self):
        from vitriol.core.generator import MinimalWeightGenerator
        assert MinimalWeightGenerator._parse_size("1073741824") == 1073741824

    def test_empty_string_raises(self):
        from vitriol.core.generator import MinimalWeightGenerator
        with pytest.raises(ValueError, match="Invalid size"):
            MinimalWeightGenerator._parse_size("")

    def test_negative_raises(self):
        from vitriol.core.generator import MinimalWeightGenerator
        with pytest.raises(ValueError, match="Negative size"):
            MinimalWeightGenerator._parse_size("-5GB")

    def test_garbage_raises(self):
        from vitriol.core.generator import MinimalWeightGenerator
        with pytest.raises(ValueError, match="Cannot parse"):
            MinimalWeightGenerator._parse_size("abc")


# ═══════════════════════════════════════════════════════════════════════
# § 6  Adapter Hardening Tests
# ═══════════════════════════════════════════════════════════════════════

class TestAdapterBaseValidation:
    """Validate adapter base class improvements."""

    def test_default_adapter_validate_config(self):
        from vitriol.adapters.base import DefaultAdapter
        adapter = DefaultAdapter()
        assert adapter.validate_config(MagicMock()) is True

    def test_default_adapter_always_matches(self):
        from vitriol.adapters.base import DefaultAdapter
        assert DefaultAdapter.match("any-model", MagicMock()) is True


# ═══════════════════════════════════════════════════════════════════════
# § 7  Strategy Registry Hardening Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStrategyRegistryValidation:
    """Validate get_strategy validates configs."""

    def test_valid_strategy_passes(self):
        from vitriol.strategies import get_strategy
        s = get_strategy("hybrid_ultra", init_mode="zeros")
        assert s is not None

    def test_unknown_strategy_raises(self):
        from vitriol.strategies import get_strategy
        from vitriol.utils.exceptions import StrategyNotFoundError
        with pytest.raises(StrategyNotFoundError):
            get_strategy("nonexistent_strategy")

    def test_all_strategies_have_validate_config(self):
        """Every registered strategy must have validate_config method."""
        from vitriol.strategies import STRATEGY_REGISTRY
        for name, cls in STRATEGY_REGISTRY.items():
            s = cls()
            assert hasattr(s, "validate_config"), (
                f"Strategy '{name}' missing validate_config()"
            )
            assert s.validate_config() is True

    def test_all_strategies_have_get_recipe(self):
        """Every registered strategy must have get_recipe method."""
        from vitriol.strategies import STRATEGY_REGISTRY
        for name, cls in STRATEGY_REGISTRY.items():
            s = cls()
            assert hasattr(s, "get_recipe"), (
                f"Strategy '{name}' missing get_recipe()"
            )
            recipe = s.get_recipe()
            assert isinstance(recipe, dict)
            assert "strategy" in recipe
