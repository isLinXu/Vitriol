"""
Tests for vitriol.utils.exceptions and vitriol.utils.strategy_discovery modules.
"""
import pytest
from unittest.mock import MagicMock

from vitriol.utils.exceptions import (
    VitriolError,
    ConfigError,
    ConfigLoadError,
    ConfigValidationError,
    ModelBuildError,
    WeightGenerationError,
    GenerationError,
    ShardSaveError,
    IncompatibleStrategyError,
    StrategyNotFoundError,
    AdapterNotFoundError,
    ModelNotSupportedError,
    DatasetLoadError,
    CheckpointCorruptedError,
    CheckpointSaveError,
    ValidationError,
)
from vitriol.utils.strategy_discovery import discover_strategy_names


# ─────────────────────────────────────────────────────────────
# VitriolError base
# ─────────────────────────────────────────────────────────────

class TestVitriolError:
    def test_base_error(self):
        err = VitriolError("something went wrong")
        assert err.message == "something went wrong"
        assert err.recoverable is False
        assert str(err) == "something went wrong"

    def test_recoverable_error(self):
        err = VitriolError("retryable", recoverable=True)
        assert err.recoverable is True

    def test_is_exception_subclass(self):
        with pytest.raises(VitriolError):
            raise VitriolError("test")


# ─────────────────────────────────────────────────────────────
# Config errors
# ─────────────────────────────────────────────────────────────

class TestConfigErrors:
    def test_config_error_inherits(self):
        err = ConfigError("bad config")
        assert isinstance(err, VitriolError)

    def test_config_load_error(self):
        err = ConfigLoadError("meta-llama/Llama-2-7b", reason="timeout")
        assert "meta-llama/Llama-2-7b" in err.message
        assert "timeout" in err.message
        assert err.recoverable is True
        assert "Suggestions" in err.message

    def test_config_load_error_without_reason(self):
        err = ConfigLoadError("model-x")
        assert "model-x" in err.message
        assert err.recoverable is True

    def test_config_validation_error(self):
        err = ConfigValidationError("hidden_size", "must be positive")
        assert "hidden_size" in err.message
        assert "must be positive" in err.message
        assert err.recoverable is False


# ─────────────────────────────────────────────────────────────
# Model errors
# ─────────────────────────────────────────────────────────────

class TestModelErrors:
    def test_model_build_error(self):
        err = ModelBuildError("model-x", reason="missing arch")
        assert "model-x" in err.message
        assert "missing arch" in err.message
        assert err.recoverable is False

    def test_weight_generation_error(self):
        err = WeightGenerationError("layer1.weight", "shape mismatch")
        assert "layer1.weight" in err.message
        assert "shape mismatch" in err.message
        assert err.recoverable is False

    def test_generation_error_alias(self):
        assert GenerationError is WeightGenerationError


# ─────────────────────────────────────────────────────────────
# Shard & Strategy errors
# ─────────────────────────────────────────────────────────────

class TestShardAndStrategyErrors:
    def test_shard_save_error(self):
        err = ShardSaveError("/path/shard.bin", reason="disk full")
        assert "/path/shard.bin" in err.message
        assert "disk full" in err.message
        assert err.recoverable is True

    def test_incompatible_strategy_with_format(self):
        err = IncompatibleStrategyError("ultra", format="safetensors")
        assert "ultra" in err.message
        assert "safetensors" in err.message
        assert err.recoverable is True

    def test_incompatible_strategy_safetensors_ultra(self):
        err = IncompatibleStrategyError("ultra", format="safetensors")
        assert "pytorch" in err.message or "compact" in err.message

    def test_incompatible_strategy_generic(self):
        err = IncompatibleStrategyError("random", format="onnx")
        assert "random" in err.message
        assert "different strategy" in err.message or "documentation" in err.message

    def test_strategy_not_found_error(self):
        err = StrategyNotFoundError("magic", ["random", "compact"])
        assert "magic" in err.message
        assert "random" in err.message
        assert isinstance(err, KeyError)


# ─────────────────────────────────────────────────────────────
# Adapter errors
# ─────────────────────────────────────────────────────────────

class TestAdapterErrors:
    def test_adapter_not_found_error(self):
        err = AdapterNotFoundError("custom-model")
        assert "custom-model" in err.message
        assert err.recoverable is True

    def test_model_not_supported_error(self):
        err = ModelNotSupportedError("old-model", reason="deprecated")
        assert "old-model" in err.message
        assert "deprecated" in err.message
        assert err.recoverable is True


# ─────────────────────────────────────────────────────────────
# NAS & Checkpoint errors
# ─────────────────────────────────────────────────────────────

class TestNASAndCheckpointErrors:
    def test_dataset_load_error(self):
        err = DatasetLoadError("c4", reason="not found")
        assert "c4" in err.message
        assert err.recoverable is True

    def test_checkpoint_corrupted_error(self):
        err = CheckpointCorruptedError("/path/ckpt", reason="bad json")
        assert "/path/ckpt" in err.message
        assert err.recoverable is True

    def test_checkpoint_save_error(self):
        err = CheckpointSaveError("/path/ckpt", reason="permission denied")
        assert "/path/ckpt" in err.message
        assert err.recoverable is True


# ─────────────────────────────────────────────────────────────
# ValidationError
# ─────────────────────────────────────────────────────────────

class TestValidationError:
    def test_validation_error(self):
        err = ValidationError("/model/path", reason="missing weights")
        assert "/model/path" in err.message
        assert "missing weights" in err.message
        assert err.recoverable is False


# ─────────────────────────────────────────────────────────────
# strategy_discovery
# ─────────────────────────────────────────────────────────────

class TestStrategyDiscovery:
    def test_discovers_strategies(self):
        names = discover_strategy_names()
        assert isinstance(names, list)
        # Should discover at least some common strategies
        assert len(names) > 0

    def test_returns_unique_names(self):
        names = discover_strategy_names()
        assert len(names) == len(set(names))

    def test_strategies_are_strings(self):
        names = discover_strategy_names()
        for name in names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_discovers_from_mock_registry(self, tmp_path):
        from vitriol.utils import strategy_discovery as sd
        # Create a fake strategies __init__.py
        fake_strategies = tmp_path / "strategies"
        fake_strategies.mkdir()
        init_file = fake_strategies / "__init__.py"
        init_file.write_text('''
STRATEGY_REGISTRY = {
    "random": "RandomStrategy",
    "compact": "CompactStrategy",
}
STRATEGY_REGISTRY["sparse"] = "SparseStrategy"
''')

        # Temporarily override the path
        orig_path = sd.Path
        fake_path = MagicMock()
        fake_path.resolve.return_value.parents = [tmp_path, tmp_path, tmp_path]
        sd.Path = lambda p: fake_path if p == __file__ else orig_path(p)

        try:
            names = discover_strategy_names()
            assert "random" in names
            assert "compact" in names
            assert "sparse" in names
        finally:
            sd.Path = orig_path

    def test_discovers_known_strategies_in_actual_registry(self):
        names = discover_strategy_names()
        # The actual strategies/__init__.py should have these
        common = {"random", "compact", "sparse", "quantized"}
        found = common & set(names)
        # At least some common strategies should be found
        assert len(found) >= 1
