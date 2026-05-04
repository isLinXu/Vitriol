"""Tests for registry/model_store module."""

import tempfile
from pathlib import Path

import pytest

from vitriol.registry.model_store import (
    ModelRegistry,
    ModelStore,
    ModelVersion,
    get_registry,
    get_store,
)


# ─────────────────────────────────────────────────────────────────────────────
# ModelVersion Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_creation(self):
        version = ModelVersion(
            version="1.0.0",
            created_at="2024-01-01T00:00:00",
            description="Test version",
            tags=["test"],
            metadata={"key": "value"},
            files=["model.bin"],
            size_bytes=1024,
            checksum="abc123",
        )
        assert version.version == "1.0.0"
        assert version.size_bytes == 1024


# ─────────────────────────────────────────────────────────────────────────────
# ModelRegistry Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            entry = registry.register_model("model1", "Test Model", description="A test model")
            assert entry.id == "model1"
            assert entry.name == "Test Model"
            assert entry.description == "A test model"
            assert "model1" in registry.models

    def test_register_model_default_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            entry = registry.register_model("m1", "Model")
            assert entry.tags == []
            assert entry.author == ""
            assert entry.stats == {"downloads": 0, "views": 0}

    def test_get_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model")
            entry = registry.get_model("m1")
            assert entry is not None
            assert entry.id == "m1"

    def test_get_model_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            assert registry.get_model("nonexistent") is None

    def test_publish_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model")

            # Create a dummy file
            dummy_file = Path(tmpdir) / "dummy.bin"
            dummy_file.write_text("dummy content")

            version = registry.publish_version("m1", "1.0.0", [str(dummy_file)])
            assert version is not None
            assert version.version == "1.0.0"
            assert version.size_bytes > 0

    def test_publish_version_model_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            version = registry.publish_version("nonexistent", "1.0.0", [])
            assert version is None

    def test_get_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model")

            dummy_file = Path(tmpdir) / "dummy.bin"
            dummy_file.write_text("content")
            registry.publish_version("m1", "1.0.0", [str(dummy_file)])

            version = registry.get_version("m1", "1.0.0")
            assert version is not None
            assert version.version == "1.0.0"

    def test_list_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model 1", tags=["llm"])
            registry.register_model("m2", "Model 2", tags=["vision"])
            registry.register_model("m3", "Model 3", tags=["llm", "multimodal"])

            all_models = registry.list_models()
            assert len(all_models) == 3

            llm_models = registry.list_models(tags=["llm"])
            assert len(llm_models) == 2

    def test_list_models_by_author(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model 1", author="Alice")
            registry.register_model("m2", "Model 2", author="Bob")

            alice_models = registry.list_models(author="Alice")
            assert len(alice_models) == 1
            assert alice_models[0].id == "m1"

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "GPT-4 Clone", description="A GPT-like model")
            registry.register_model("m2", "Vision Transformer", tags=["vision"])

            results = registry.search("gpt")
            assert len(results) == 1
            assert results[0].id == "m1"

            results = registry.search("vision")
            assert len(results) == 1
            assert results[0].id == "m2"

    def test_search_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model 1")
            assert registry.search("nonexistent") == []

    def test_increment_stat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model")
            registry.increment_stat("m1", "downloads")
            assert registry.models["m1"].stats["downloads"] == 1
            registry.increment_stat("m1", "downloads")
            assert registry.models["m1"].stats["downloads"] == 2

    def test_increment_stat_unknown_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.increment_stat("nonexistent", "downloads")  # should not raise

    def test_delete_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model")
            registry.delete_model("m1")
            assert "m1" not in registry.models

    def test_delete_model_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.delete_model("nonexistent")  # should not raise

    def test_get_storage_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            stats = registry.get_storage_stats()
            assert stats["total_models"] == 0
            assert stats["total_versions"] == 0
            assert stats["total_size_gb"] == 0.0

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg1 = ModelRegistry(storage_path=str(tmpdir))
            reg1.register_model("m1", "Model 1", tags=["test"])

            reg2 = ModelRegistry(storage_path=str(tmpdir))
            assert "m1" in reg2.models
            assert reg2.models["m1"].name == "Model 1"
            assert reg2.models["m1"].tags == ["test"]

    def test_checksum_computation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model")

            dummy_file = Path(tmpdir) / "dummy.bin"
            dummy_file.write_text("checksum test content")
            version = registry.publish_version("m1", "1.0.0", [str(dummy_file)])

            assert version is not None
            assert len(version.checksum) == 16
            assert version.checksum != ""

    def test_multiple_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            registry.register_model("m1", "Model")

            dummy_file = Path(tmpdir) / "dummy.bin"
            dummy_file.write_text("v1")
            v1 = registry.publish_version("m1", "1.0.0", [str(dummy_file)])

            dummy_file.write_text("v2")
            v2 = registry.publish_version("m1", "2.0.0", [str(dummy_file)])

            assert len(registry.models["m1"].versions) == 2
            assert registry.get_version("m1", "1.0.0") is not None
            assert registry.get_version("m1", "2.0.0") is not None


# ─────────────────────────────────────────────────────────────────────────────
# ModelStore Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelStore:
    """Tests for ModelStore."""

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            store = ModelStore(registry)

            dummy_file = Path(tmpdir) / "model.bin"
            dummy_file.write_text("model weights")

            success = store.save_model("m1", "1.0.0", [str(dummy_file)], metadata={"format": "bin"})
            assert success is True

            loaded = store.load_model("m1", "1.0.0")
            assert loaded is not None
            assert loaded["model_id"] == "m1"
            assert loaded["version"] == "1.0.0"
            assert loaded["metadata"]["format"] == "bin"

    def test_load_latest_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            store = ModelStore(registry)

            dummy_file = Path(tmpdir) / "model.bin"
            dummy_file.write_text("v1")
            store.save_model("m1", "1.0.0", [str(dummy_file)])

            dummy_file.write_text("v2")
            store.save_model("m1", "2.0.0", [str(dummy_file)])

            loaded = store.load_model("m1")  # latest
            assert loaded is not None
            assert loaded["version"] == "2.0.0"

    def test_load_nonexistent_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            store = ModelStore(registry)
            assert store.load_model("nonexistent") is None

    def test_load_nonexistent_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            store = ModelStore(registry)

            dummy_file = Path(tmpdir) / "model.bin"
            dummy_file.write_text("content")
            store.save_model("m1", "1.0.0", [str(dummy_file)])

            assert store.load_model("m1", "9.9.9") is None

    def test_save_model_auto_register(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            store = ModelStore(registry)

            dummy_file = Path(tmpdir) / "model.bin"
            dummy_file.write_text("content")
            store.save_model("new_model", "1.0.0", [str(dummy_file)])

            assert "new_model" in registry.models

    def test_load_increments_downloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(storage_path=str(tmpdir))
            store = ModelStore(registry)

            dummy_file = Path(tmpdir) / "model.bin"
            dummy_file.write_text("content")
            store.save_model("m1", "1.0.0", [str(dummy_file)])

            store.load_model("m1")
            assert registry.models["m1"].stats.get("downloads", 0) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Global Instance Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalInstances:
    """Tests for global registry and store instances."""

    def test_get_registry_singleton(self):
        # Use temporary directory to avoid polluting real registry
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.MonkeyPatch.context() as mp:
                # Global instances persist, so just test they return correct types
                reg = get_registry()
                assert isinstance(reg, ModelRegistry)

    def test_get_store_singleton(self):
        store = get_store()
        assert isinstance(store, ModelStore)
        assert isinstance(store.registry, ModelRegistry)
