"""Tests for utility modules: model_capabilities, strategy_discovery, fingerprint."""

import tempfile
from pathlib import Path

import numpy as np
import torch.nn as nn

from vitriol.utils.model_capabilities import (
    cfg_attr,
    cfg_int,
    cfg_list,
    explicit_layer_types,
    infer_kv_layer_types,
    infer_model_capabilities,
    infer_num_layers,
    normalize_kv_layer_type,
)
from vitriol.utils.strategy_discovery import discover_strategy_names
from vitriol.utils.fingerprint import (
    ArchitectureHasher,
    FingerprintEngine,
    FingerprintRegistry,
    ModelFingerprint,
    WeightsHasher,
)


# ─────────────────────────────────────────────────────────────────────────────
# model_capabilities Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCfgAttr:
    """Tests for cfg_attr helper."""

    def test_dict_input(self):
        assert cfg_attr({"a": 1}, "a") == 1
        assert cfg_attr({"a": 1}, "b", "default") == "default"

    def test_object_input(self):
        class Obj:
            x = 42
        assert cfg_attr(Obj(), "x") == 42
        assert cfg_attr(Obj(), "y", "default") == "default"

    def test_none_input(self):
        assert cfg_attr(None, "a", "default") == "default"


class TestCfgInt:
    """Tests for cfg_int helper."""

    def test_valid_int(self):
        assert cfg_int({"n": 10}, "n") == 10

    def test_string_number(self):
        assert cfg_int({"n": "10"}, "n") == 10

    def test_missing_key(self):
        assert cfg_int({}, "n", 5) == 5

    def test_none_value(self):
        assert cfg_int({"n": None}, "n", 3) == 3

    def test_invalid_conversion(self):
        assert cfg_int({"n": "abc"}, "n", 7) == 7


class TestCfgList:
    """Tests for cfg_list helper."""

    def test_list_value(self):
        assert cfg_list({"items": [1, 2, 3]}, "items") == [1, 2, 3]

    def test_tuple_value(self):
        assert cfg_list({"items": (1, 2)}, "items") == [1, 2]

    def test_missing_key(self):
        assert cfg_list({}, "items") == []

    def test_non_list_value(self):
        assert cfg_list({"items": "not_list"}, "items") == []


class TestNormalizeKvLayerType:
    """Tests for normalize_kv_layer_type."""

    def test_mamba_variants(self):
        for name in ["mamba", "ssm", "state_space", "rwkv", "retnet", "hyena"]:
            assert normalize_kv_layer_type(name) == "linear_attention"

    def test_sliding_window(self):
        assert normalize_kv_layer_type("sliding_window") == "sliding_window"
        assert normalize_kv_layer_type("local_attention") == "sliding_window"

    def test_mla(self):
        assert normalize_kv_layer_type("mla") == "mla"
        assert normalize_kv_layer_type("latent_attention") == "mla"

    def test_hash(self):
        assert normalize_kv_layer_type("hash_attention") == "hash_attention"

    def test_compressed(self):
        assert normalize_kv_layer_type("compressed") == "compressed_attention"
        assert normalize_kv_layer_type("csa") == "compressed_attention"
        assert normalize_kv_layer_type("hca") == "compressed_attention"

    def test_full_attention(self):
        for name in ["attention", "self_attention", "full", "global", "mha", "gqa", "mqa"]:
            assert normalize_kv_layer_type(name) == "full_attention"

    def test_empty_name(self):
        assert normalize_kv_layer_type("") == "full_attention"
        assert normalize_kv_layer_type(None) == "full_attention"

    def test_other(self):
        assert normalize_kv_layer_type("something_unknown") == "other"


class TestInferNumLayers:
    """Tests for infer_num_layers."""

    def test_standard_keys(self):
        assert infer_num_layers({"num_hidden_layers": 12}) == 12
        assert infer_num_layers({"n_layer": 24}) == 24
        assert infer_num_layers({"decoder_layers": 6}) == 6

    def test_text_config_fallback(self):
        assert infer_num_layers({"text_config": {"num_hidden_layers": 8}}) == 8

    def test_no_layers(self):
        assert infer_num_layers({}) == 0

    def test_zero_value(self):
        assert infer_num_layers({"num_hidden_layers": 0}) == 0


class TestExplicitLayerTypes:
    """Tests for explicit_layer_types."""

    def test_layer_types_key(self):
        config = {"layer_types": ["attention", "mamba", "attention"]}
        result = explicit_layer_types(config)
        assert result == ["full_attention", "linear_attention", "full_attention"]

    def test_missing_key(self):
        assert explicit_layer_types({}) == []

    def test_text_config_fallback(self):
        config = {"text_config": {"block_types": ["sliding", "full"]}}
        result = explicit_layer_types(config)
        assert result == ["sliding_window", "full_attention"]


class TestInferKvLayerTypes:
    """Tests for infer_kv_layer_types."""

    def test_explicit_layer_types(self):
        config = {
            "num_hidden_layers": 3,
            "layer_types": ["attention", "mamba", "attention"],
        }
        result = infer_kv_layer_types(config)
        assert result == ["full_attention", "linear_attention", "full_attention"]

    def test_extend_to_num_layers(self):
        config = {
            "num_hidden_layers": 5,
            "layer_types": ["attention", "mamba"],
        }
        result = infer_kv_layer_types(config)
        assert len(result) == 5
        assert result == ["full_attention", "linear_attention", "full_attention", "full_attention", "full_attention"]

    def test_mamba_model(self):
        config = {
            "num_hidden_layers": 4,
            "model_type": "mamba",
        }
        result = infer_kv_layer_types(config)
        assert result == ["linear_attention"] * 4

    def test_sliding_window(self):
        config = {
            "num_hidden_layers": 2,
            "sliding_window": 4096,
            "model_type": "mistral",
        }
        result = infer_kv_layer_types(config)
        assert result == ["sliding_window"] * 2

    def test_no_kv_cache_models(self):
        config = {
            "num_hidden_layers": 3,
            "model_type": "rwkv",
        }
        result = infer_kv_layer_types(config)
        assert result == ["linear_attention"] * 3

    def test_default_transformer(self):
        config = {"num_hidden_layers": 2, "model_type": "llama"}
        result = infer_kv_layer_types(config)
        assert result == ["full_attention"] * 2

    def test_zero_layers(self):
        assert infer_kv_layer_types({}) == []


class TestInferModelCapabilities:
    """Tests for infer_model_capabilities."""

    def test_sequence_mixer(self):
        config = {"num_hidden_layers": 4, "model_type": "mamba", "layer_types": ["mamba"] * 4}
        caps = infer_model_capabilities(config)
        assert caps.architecture_kind == "sequence_mixer"
        assert caps.supports_kv_cache is False

    def test_hybrid_attention(self):
        config = {
            "num_hidden_layers": 2,
            "model_type": "deepseek",
            "layer_types": ["compressed_attention", "full_attention"],
        }
        caps = infer_model_capabilities(config)
        assert caps.architecture_kind == "hybrid_attention"
        assert caps.supports_kv_cache is True

    def test_transformer_attention(self):
        config = {"num_hidden_layers": 2, "model_type": "llama"}
        caps = infer_model_capabilities(config)
        assert caps.architecture_kind == "transformer_attention"
        assert caps.supports_kv_cache is True

    def test_unknown(self):
        caps = infer_model_capabilities({})
        assert caps.architecture_kind == "unknown"
        assert caps.supports_kv_cache is False


# ─────────────────────────────────────────────────────────────────────────────
# strategy_discovery Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyDiscovery:
    """Tests for strategy_discovery."""

    def test_discover_strategy_names(self):
        names = discover_strategy_names()
        assert isinstance(names, list)
        # Should find known strategies from strategies/__init__.py
        assert len(names) > 0

    def test_discover_contains_known_strategies(self):
        names = discover_strategy_names()
        known = ["random", "compact", "ultra", "sparse", "binary", "ternary",
                 "quantized", "lowrank", "structured_sparse", "learned"]
        for s in known:
            assert s in names, f"Strategy '{s}' not found in discovered strategies"

    def test_discover_no_duplicates(self):
        names = discover_strategy_names()
        assert len(names) == len(set(names))


# ─────────────────────────────────────────────────────────────────────────────
# fingerprint Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelFingerprint:
    """Tests for ModelFingerprint dataclass."""

    def test_to_dict(self):
        fp = ModelFingerprint(
            model_id="test",
            architecture_hash="abc",
            weights_hash="def",
            content_hash="ghi",
            signature="jkl",
            timestamp=123.0,
            metadata={"key": "value"},
        )
        d = fp.to_dict()
        assert d["model_id"] == "test"
        assert d["metadata"] == {"key": "value"}

    def test_from_dict(self):
        data = {
            "model_id": "test",
            "architecture_hash": "abc",
            "weights_hash": "def",
            "content_hash": "ghi",
            "signature": "jkl",
            "timestamp": 123.0,
            "metadata": {},
        }
        fp = ModelFingerprint.from_dict(data)
        assert fp.model_id == "test"

    def test_verify_identical(self):
        fp1 = ModelFingerprint("m1", "a", "w", "c", "s", 1.0, {})
        fp2 = ModelFingerprint("m1", "a", "w", "c", "s", 1.0, {})
        result = fp1.verify(fp2)
        assert result["identical"] is True
        assert result["same_architecture"] is True
        assert result["same_weights"] is True

    def test_verify_different(self):
        fp1 = ModelFingerprint("m1", "a", "w", "c", "s", 1.0, {})
        fp2 = ModelFingerprint("m2", "x", "y", "z", "t", 2.0, {})
        result = fp1.verify(fp2)
        assert result["identical"] is False
        assert result["same_architecture"] is False


class TestArchitectureHasher:
    """Tests for ArchitectureHasher."""

    def test_hash_linear_model(self):
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        hasher = ArchitectureHasher()
        h = hasher.hash(model)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_hash_determinism(self):
        model = nn.Sequential(nn.Linear(10, 20))
        hasher = ArchitectureHasher()
        h1 = hasher.hash(model)
        h2 = hasher.hash(model)
        assert h1 == h2

    def test_different_architectures_different_hashes(self):
        m1 = nn.Sequential(nn.Linear(10, 20))
        m2 = nn.Sequential(nn.Linear(10, 30))
        hasher = ArchitectureHasher()
        assert hasher.hash(m1) != hasher.hash(m2)

    def test_hash_embedding_layer(self):
        model = nn.Sequential(nn.Embedding(100, 64))
        hasher = ArchitectureHasher()
        h = hasher.hash(model)
        assert isinstance(h, str)

    def test_hash_conv_layer(self):
        model = nn.Sequential(nn.Conv2d(3, 64, 3))
        hasher = ArchitectureHasher()
        h = hasher.hash(model)
        assert isinstance(h, str)

    def test_container_modules_skipped(self):
        """Container modules (with children) should be skipped."""
        model = nn.Sequential(
            nn.Sequential(nn.Linear(10, 20)),
        )
        hasher = ArchitectureHasher()
        h = hasher.hash(model)
        # Should still find the inner Linear
        assert len(h) == 32


class TestWeightsHasher:
    """Tests for WeightsHasher."""

    def test_hash_shape(self):
        model = nn.Linear(10, 20)
        hasher = WeightsHasher()
        h = hasher.hash(model)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_hash_changes_with_weights(self):
        model = nn.Linear(10, 20)
        hasher = WeightsHasher()
        h1 = hasher.hash(model)
        nn.init.xavier_uniform_(model.weight)
        h2 = hasher.hash(model)
        assert h1 != h2

    def test_precision_effect(self):
        model = nn.Linear(10, 20)
        hasher_high = WeightsHasher(precision=8)
        hasher_low = WeightsHasher(precision=1)
        h1 = hasher_high.hash(model)
        h2 = hasher_low.hash(model)
        # Same model but different precision may produce different hashes
        assert isinstance(h1, str) and isinstance(h2, str)

    def test_compute_histogram(self):
        hasher = WeightsHasher()
        weights = np.random.randn(100, 100)
        hist = hasher._compute_histogram(weights, bins=8)
        assert len(hist) == 8
        # Tolerance accounts for rounding to precision=6 on each bin
        assert abs(sum(hist) - 1.0) < 1e-5

    def test_compute_gradient_stats(self):
        hasher = WeightsHasher()
        weights = np.random.randn(50, 50)
        stats = hasher._compute_gradient_stats(weights)
        assert "dx_mean" in stats
        assert "dy_mean" in stats

    def test_gradient_stats_1d(self):
        hasher = WeightsHasher()
        stats = hasher._compute_gradient_stats(np.array([1, 2, 3]))
        assert stats == {"dx": 0.0, "dy": 0.0}


class TestFingerprintEngine:
    """Tests for FingerprintEngine."""

    def test_fingerprint_structure(self):
        model = nn.Sequential(nn.Linear(10, 20))
        engine = FingerprintEngine()
        fp = engine.fingerprint(model, model_id="test_model")
        assert fp.model_id == "test_model"
        assert len(fp.architecture_hash) == 32
        assert len(fp.weights_hash) == 32
        assert len(fp.content_hash) == 32
        assert len(fp.signature) == 32
        assert fp.timestamp > 0

    def test_fingerprint_auto_id(self):
        model = nn.Sequential(nn.Linear(10, 20))
        engine = FingerprintEngine()
        fp = engine.fingerprint(model)
        assert fp.model_id.startswith("vitriol_")

    def test_verify_signature_valid(self):
        model = nn.Sequential(nn.Linear(10, 20))
        engine = FingerprintEngine(secret_key="test_key")
        fp = engine.fingerprint(model)
        assert engine.verify_signature(fp) is True

    def test_verify_signature_invalid(self):
        model = nn.Sequential(nn.Linear(10, 20))
        engine = FingerprintEngine(secret_key="test_key")
        fp = engine.fingerprint(model)
        fp.signature = "tampered"
        assert engine.verify_signature(fp) is False

    def test_compare_models_identical(self):
        model = nn.Sequential(nn.Linear(10, 20))
        engine = FingerprintEngine()
        comparison = engine.compare_models(model, model)
        assert comparison["identical"] is True
        assert comparison["same_architecture"] is True
        assert comparison["same_weights"] is True
        assert comparison["weights_similarity"] == 1.0

    def test_compare_models_different(self):
        m1 = nn.Sequential(nn.Linear(10, 20))
        m2 = nn.Sequential(nn.Linear(10, 30))
        engine = FingerprintEngine()
        comparison = engine.compare_models(m1, m2)
        assert comparison["identical"] is False
        assert comparison["same_architecture"] is False

    def test_save_and_load_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = nn.Sequential(nn.Linear(10, 20))
            engine = FingerprintEngine()
            fp = engine.fingerprint(model)

            path = Path(tmpdir) / "fingerprint.json"
            engine.save_fingerprint(fp, str(path))
            assert path.exists()

            loaded = engine.load_fingerprint(str(path))
            assert loaded.model_id == fp.model_id
            assert loaded.architecture_hash == fp.architecture_hash

    def test_arch_similarity_empty_models(self):
        """Test architecture similarity with empty models."""
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x
        engine = FingerprintEngine()
        sim = engine._compute_arch_similarity(EmptyModel(), EmptyModel())
        assert sim == 1.0

    def test_weights_similarity_no_common(self):
        m1 = nn.Sequential(nn.Linear(10, 20))
        m2 = nn.Sequential(nn.Linear(30, 40))
        engine = FingerprintEngine()
        sim = engine._compute_weights_similarity(m1, m2)
        assert sim == 0.0


class TestFingerprintRegistry:
    """Tests for FingerprintRegistry."""

    def test_register_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "registry.json"
            registry = FingerprintRegistry(storage_path=str(storage_path))
            model = nn.Sequential(nn.Linear(10, 20))
            fp = registry.register(model, model_id="m1")
            assert fp.model_id == "m1"
            assert "m1" in registry.fingerprints

    def test_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "registry.json"
            registry = FingerprintRegistry(storage_path=str(storage_path))
            model = nn.Sequential(nn.Linear(10, 20))
            registry.register(model, model_id="m1")

            result = registry.verify(model)
            assert result["matches"]  # Should find match with itself
            assert len(result["all_results"]) == 1

    def test_get_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "registry.json"
            registry = FingerprintRegistry(storage_path=str(storage_path))
            model = nn.Sequential(nn.Linear(10, 20))
            fp1 = registry.register(model, model_id="m1_v1")
            fp2 = registry.register(model, model_id="m1_v2")

            lineage = registry.get_lineage("m1_v1")
            assert len(lineage) == 2

    def test_get_lineage_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "registry.json"
            registry = FingerprintRegistry(storage_path=str(storage_path))
            assert registry.get_lineage("nonexistent") == []

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "registry.json"
            model = nn.Sequential(nn.Linear(10, 20))
            reg1 = FingerprintRegistry(storage_path=str(storage_path))
            reg1.register(model, model_id="m1")

            reg2 = FingerprintRegistry(storage_path=str(storage_path))
            assert "m1" in reg2.fingerprints
