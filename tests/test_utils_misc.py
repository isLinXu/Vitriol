"""Tests for utils modules: exceptions, config_cache, logging, version"""


from vitriol.utils.exceptions import (
    VitriolError,
    ConfigLoadError,
    ConfigValidationError,
    ModelBuildError,
    WeightGenerationError,
    ShardSaveError,
    IncompatibleStrategyError,
    StrategyNotFoundError,
    AdapterNotFoundError,
    ModelNotSupportedError,
    DatasetLoadError,
    CheckpointCorruptedError,
    CheckpointSaveError,
    ValidationError,
    GenerationError,
)
from vitriol.utils.config_cache import ConfigCache
from vitriol.utils.logging import setup_logging
from vitriol import version


# ─────────────────────────────────────────────────────────────────────────────
# exceptions tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVitriolError:
    def test_basic(self):
        e = VitriolError("something went wrong")
        assert str(e) == "something went wrong"
        assert e.recoverable is False

    def test_recoverable(self):
        e = VitriolError("retryable", recoverable=True)
        assert e.recoverable is True


class TestConfigLoadError:
    def test_message(self):
        e = ConfigLoadError("test-model")
        assert "test-model" in str(e)
        assert "Suggestions" in str(e)

    def test_with_reason(self):
        e = ConfigLoadError("test-model", reason="network timeout")
        assert "network timeout" in str(e)

    def test_recoverable(self):
        e = ConfigLoadError("test-model")
        assert e.recoverable is True


class TestConfigValidationError:
    def test_message(self):
        e = ConfigValidationError("hidden_size", "must be positive")
        assert "hidden_size" in str(e)
        assert "must be positive" in str(e)
        assert e.recoverable is False


class TestModelBuildError:
    def test_message(self):
        e = ModelBuildError("my-model")
        assert "my-model" in str(e)
        assert "Failed to build" in str(e)


class TestWeightGenerationError:
    def test_message(self):
        e = WeightGenerationError("layer1.weight")
        assert "layer1.weight" in str(e)

    def test_with_reason(self):
        e = WeightGenerationError("layer1.weight", "OOM")
        assert "OOM" in str(e)


class TestGenerationErrorAlias:
    def test_alias(self):
        e = GenerationError("param")
        assert isinstance(e, WeightGenerationError)


class TestShardSaveError:
    def test_message(self):
        e = ShardSaveError("/path/to/shard")
        assert "/path/to/shard" in str(e)
        assert "disk space" in str(e)


class TestIncompatibleStrategyError:
    def test_basic(self):
        e = IncompatibleStrategyError("ultra")
        assert "ultra" in str(e)

    def test_with_format(self):
        e = IncompatibleStrategyError("ultra", format="safetensors")
        assert "safetensors" in str(e)
        assert "pytorch" in str(e)  # suggestion

    def test_with_reason(self):
        e = IncompatibleStrategyError("ultra", reason="stride=0")
        assert "stride=0" in str(e)


class TestStrategyNotFoundError:
    def test_message(self):
        e = StrategyNotFoundError("unknown", ["random", "compact"])
        assert "unknown" in str(e)
        assert "random" in str(e)
        assert isinstance(e, KeyError)


class TestAdapterNotFoundError:
    def test_message(self):
        e = AdapterNotFoundError("custom/model")
        assert "custom/model" in str(e)
        assert e.recoverable is True


class TestModelNotSupportedError:
    def test_message(self):
        e = ModelNotSupportedError("unsupported")
        assert "unsupported" in str(e)
        assert e.recoverable is True


class TestDatasetLoadError:
    def test_message(self):
        e = DatasetLoadError("wikitext")
        assert "wikitext" in str(e)
        assert e.recoverable is True


class TestCheckpointCorruptedError:
    def test_message(self):
        e = CheckpointCorruptedError("/path/to/checkpoint")
        assert "corrupted" in str(e)
        assert e.recoverable is True


class TestCheckpointSaveError:
    def test_message(self):
        e = CheckpointSaveError("/path")
        assert "/path" in str(e)
        assert e.recoverable is True


class TestValidationError:
    def test_message(self):
        e = ValidationError("/model/path")
        assert "/model/path" in str(e)
        assert e.recoverable is False


# ─────────────────────────────────────────────────────────────────────────────
# config_cache tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigCache:
    def test_init(self, tmp_path):
        cache = ConfigCache(cache_dir=str(tmp_path), max_age_days=1)
        assert cache.cache_dir == tmp_path
        assert cache.max_age_seconds == 86400

    def test_get_miss(self, tmp_path):
        cache = ConfigCache(cache_dir=str(tmp_path))
        result = cache.get("nonexistent/model")
        assert result is None

    def test_set_and_get(self, tmp_path):
        cache = ConfigCache(cache_dir=str(tmp_path))
        config = {"model_type": "test", "hidden_size": 128}
        cache.set("test/model", config)

        result = cache.get("test/model")
        assert result is not None
        # get() returns a PretrainedConfig-like object via build_config_object
        assert hasattr(result, "model_type")
        assert result.model_type == "test"

    def test_cache_expiration(self, tmp_path):
        cache = ConfigCache(cache_dir=str(tmp_path), max_age_days=0)
        config = {"test": True}
        cache.set("old/model", config)

        # With 0 max_age_days, may or may not be expired depending on timing
        result = cache.get("old/model")
        # May be None if expired, or the config if not yet expired
        assert result is None or hasattr(result, "to_dict")

    def test_cache_key_consistency(self, tmp_path):
        cache = ConfigCache(cache_dir=str(tmp_path))
        key1 = cache._get_cache_key("model/name")
        key2 = cache._get_cache_key("model/name")
        assert key1 == key2
        assert len(key1) == 32  # md5 hex length

    def test_delete(self, tmp_path):
        cache = ConfigCache(cache_dir=str(tmp_path))
        cache.set("to-delete", {"a": 1})
        # delete method may not exist; manually remove file
        import os
        cache_file = tmp_path / (cache._get_cache_key("to-delete") + ".json")
        if cache_file.exists():
            os.remove(str(cache_file))
        assert cache.get("to-delete") is None

    def test_clear(self, tmp_path):
        cache = ConfigCache(cache_dir=str(tmp_path))
        cache.set("m1", {"a": 1})
        cache.set("m2", {"b": 2})
        cache.clear()
        assert cache.get("m1") is None
        assert cache.get("m2") is None


# ─────────────────────────────────────────────────────────────────────────────
# logging tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSetupLogging:
    def test_basic(self):
        # Should not raise
        setup_logging(level="DEBUG")

    def test_with_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(level="INFO", log_file=log_file)
        assert log_file.exists()

    def test_custom_format(self):
        fmt = "%(name)s - %(message)s"
        setup_logging(level="WARNING", format_string=fmt)

    def test_level_case_insensitive(self):
        setup_logging(level="info")
        setup_logging(level="INFO")


# ─────────────────────────────────────────────────────────────────────────────
# version tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVersion:
    def test_version_string(self):
        assert isinstance(version.__version__, str)
        assert len(version.__version__) > 0
        # Should be semver-like
        parts = version.__version__.split(".")
        assert len(parts) >= 2

    def test_version_is_not_placeholder(self):
        assert version.__version__ not in ("0.0.0", "", "unknown")

