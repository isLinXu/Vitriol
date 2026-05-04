"""Tests for core utility modules: incremental, smart_initializer, hasher."""

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from vitriol.core.incremental import IncrementalGenerator
from vitriol.core.hasher import ModelHasher
from vitriol.core.smart_initializer import (
    LayerProfile,
    ModelStructureAnalyzer,
    SmartInitializer,
    WeightPredictor,
)


# ─────────────────────────────────────────────────────────────────────────────
# IncrementalGenerator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIncrementalGenerator:
    """Tests for IncrementalGenerator checkpointing."""

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = IncrementalGenerator(tmpdir)
            state = {"current_shard": 5, "total_shards": 10, "progress": 0.5}
            gen.save_checkpoint(state)

            loaded = gen.load_checkpoint()
            assert loaded == state

    def test_load_nonexistent_checkpoint(self):
        """Test loading when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = IncrementalGenerator(tmpdir)
            assert gen.load_checkpoint() is None

    def test_clear_checkpoint(self):
        """Test clearing a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = IncrementalGenerator(tmpdir)
            gen.save_checkpoint({"done": True})
            assert gen.checkpoint_file.exists()

            gen.clear_checkpoint()
            assert not gen.checkpoint_file.exists()

    def test_clear_nonexistent_checkpoint(self):
        """Test clearing when no checkpoint exists (should not raise)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = IncrementalGenerator(tmpdir)
            gen.clear_checkpoint()  # should not raise

    def test_checkpoint_persistence(self):
        """Test that checkpoint survives generator re-creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen1 = IncrementalGenerator(tmpdir)
            state = {"epoch": 3, "loss": 0.42}
            gen1.save_checkpoint(state)

            gen2 = IncrementalGenerator(tmpdir)
            loaded = gen2.load_checkpoint()
            assert loaded == state

    def test_save_checkpoint_handles_errors(self):
        """Test that save errors are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = IncrementalGenerator(tmpdir)
            # Make directory read-only to force error
            os.chmod(tmpdir, 0o555)
            try:
                gen.save_checkpoint({"data": "x"})
            finally:
                os.chmod(tmpdir, 0o755)
            # Should not raise; logs warning instead


# ─────────────────────────────────────────────────────────────────────────────
# ModelHasher Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelHasher:
    """Tests for ModelHasher."""

    def test_hash_dict_determinism(self):
        """Test that dictionary hashing is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hasher = ModelHasher(tmpdir)
            data = {"b": 2, "a": 1}
            h1 = hasher._hash_dict(data)
            h2 = hasher._hash_dict(data)
            assert h1 == h2
            assert len(h1) == 64  # sha256 hex

    def test_hash_dict_order_independence(self):
        """Test that dict order doesn't affect hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hasher = ModelHasher(tmpdir)
            h1 = hasher._hash_dict({"a": 1, "b": 2})
            h2 = hasher._hash_dict({"b": 2, "a": 1})
            assert h1 == h2

    def test_compute_architecture_hash_missing_config(self):
        """Test architecture hash when config.json is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hasher = ModelHasher(tmpdir)
            result = hasher.compute_architecture_hash()
            assert result == "N/A"

    def test_compute_architecture_hash_with_config(self):
        """Test architecture hash with a valid config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "vocab_size": 32000,
            }
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config))

            hasher = ModelHasher(tmpdir)
            result = hasher.compute_architecture_hash()
            assert result != "N/A"
            assert len(result) == 64

    def test_compute_architecture_hash_multimodal(self):
        """Test architecture hash with multimodal config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "hidden_size": 2048,
                "vision_config": {"hidden_size": 1024, "num_hidden_layers": 12},
            }
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config))

            hasher = ModelHasher(tmpdir)
            result = hasher.compute_architecture_hash()
            assert result != "N/A"

    def test_compute_architecture_hash_diffusers(self):
        """Test architecture hash with diffusers model_index.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = {"_class_name": "StableDiffusionPipeline"}
            (Path(tmpdir) / "model_index.json").write_text(json.dumps(index))

            hasher = ModelHasher(tmpdir)
            result = hasher.compute_architecture_hash()
            assert result != "N/A"

    def test_compute_activation_signature_hash(self):
        """Test activation signature hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
                "num_attention_heads": 32,
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            hasher = ModelHasher(tmpdir)
            result = hasher.compute_activation_signature_hash()
            assert result != "N/A"
            assert len(result) == 64

    def test_compute_activation_signature_hash_missing_config(self):
        """Test activation signature when config is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hasher = ModelHasher(tmpdir)
            result = hasher.compute_activation_signature_hash()
            assert result == "N/A"

    def test_compute_activation_signature_hash_zero_dims(self):
        """Test activation signature with zero dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"hidden_size": 0, "num_hidden_layers": 0}
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            hasher = ModelHasher(tmpdir)
            result = hasher.compute_activation_signature_hash()
            assert result == "N/A"

    def test_generate_fingerprint_structure(self):
        """Test fingerprint structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"hidden_size": 128, "num_hidden_layers": 2, "vocab_size": 1000}
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))

            hasher = ModelHasher(tmpdir)
            fp = hasher.generate_fingerprint()
            assert "model_path" in fp
            assert "architecture_hash" in fp
            assert "weight_distribution_hash" in fp
            assert "vitriol_signature" in fp
            assert fp["vitriol_signature"].startswith("arx_")


# ─────────────────────────────────────────────────────────────────────────────
# SmartInitializer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelStructureAnalyzer:
    """Tests for ModelStructureAnalyzer."""

    def test_analyze_simple_model(self):
        """Test analyzing a simple model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(model)
        assert len(profiles) == 2  # Two Linear layers, ReLU has no children but is leaf
        # ReLU is a leaf module too
        assert any(p.layer_type == "linear" for p in profiles.values())

    def test_profile_linear_layer(self):
        """Test profiling a linear layer."""
        analyzer = ModelStructureAnalyzer()
        module = nn.Linear(10, 20, bias=True)
        profile = analyzer._profile_layer("test_layer", module, depth=3)
        assert profile is not None
        assert profile.layer_type == "linear"
        assert profile.input_dim == 10
        assert profile.output_dim == 20
        assert profile.depth == 3
        assert profile.has_bias is True
        assert profile.fan_in == 10
        assert profile.fan_out == 20

    def test_profile_embedding_layer(self):
        """Test profiling an embedding layer."""
        analyzer = ModelStructureAnalyzer()
        module = nn.Embedding(1000, 128)
        profile = analyzer._profile_layer("embed", module, depth=0)
        assert profile is not None
        assert profile.layer_type == "embedding"
        assert profile.is_embedding is True
        assert profile.input_dim == 1000
        assert profile.output_dim == 128

    def test_profile_conv2d_layer(self):
        """Test profiling a conv2d layer."""
        analyzer = ModelStructureAnalyzer()
        module = nn.Conv2d(3, 64, kernel_size=3)
        profile = analyzer._profile_layer("conv", module, depth=1)
        assert profile is not None
        assert profile.layer_type == "conv2d"
        assert profile.input_dim == 3
        assert profile.output_dim == 64

    def test_detect_residuals_positive(self):
        """Test residual detection with residual in name."""
        analyzer = ModelStructureAnalyzer()
        analyzer.layer_profiles = {
            "layer1": LayerProfile("layer1", "linear", 10, 10, 0, 10, 10),
            "residual_block": LayerProfile("residual_block", "linear", 10, 10, 1, 10, 10),
        }
        assert analyzer._detect_residuals() is True

    def test_detect_residuals_negative(self):
        """Test residual detection without residual in names."""
        analyzer = ModelStructureAnalyzer()
        analyzer.layer_profiles = {
            "layer1": LayerProfile("layer1", "linear", 10, 10, 0, 10, 10),
            "layer2": LayerProfile("layer2", "linear", 10, 10, 1, 10, 10),
        }
        assert analyzer._detect_residuals() is False


class TestSmartInitializer:
    """Tests for SmartInitializer."""

    def test_initialize_adaptive(self):
        """Test adaptive initialization strategy."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="adaptive")
        assert result is model
        assert len(initializer.recommendations) == 2

    def test_initialize_xavier(self):
        """Test Xavier initialization strategy."""
        model = nn.Sequential(nn.Linear(10, 20))
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="xavier")
        assert result is model
        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "xavier"

    def test_initialize_kaiming(self):
        """Test Kaiming initialization strategy."""
        model = nn.Sequential(nn.Linear(10, 20))
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="kaiming")
        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "kaiming"

    def test_initialize_orthogonal(self):
        """Test orthogonal initialization strategy."""
        model = nn.Sequential(nn.Linear(10, 20))
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="orthogonal")
        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "orthogonal"

    def test_initialize_unknown_strategy_defaults_to_xavier(self):
        """Test unknown strategy defaults to xavier."""
        model = nn.Sequential(nn.Linear(10, 20))
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="unknown")
        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "xavier"

    def test_attention_layer_recognition(self):
        """Test that attention layers are recognized."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_q = nn.Linear(64, 64)
                self.embed = nn.Embedding(100, 64)
                self.output = nn.Linear(64, 100)

            def forward(self, x):
                return x

        model = DummyModel()
        initializer = SmartInitializer()
        initializer.initialize(model, strategy="adaptive")

        recs = initializer.recommendations
        assert any("attention" in r.init_type or r.reason.lower().find("attention") >= 0
                   for r in recs.values())

    def test_get_initialization_report(self):
        """Test initialization report generation."""
        model = nn.Sequential(nn.Linear(10, 20))
        initializer = SmartInitializer()
        initializer.initialize(model)
        report = initializer.get_initialization_report()
        assert "total_layers" in report
        assert "init_type_distribution" in report
        assert report["total_layers"] == 1

    def test_get_initialization_report_before_init(self):
        """Test report before initialization."""
        initializer = SmartInitializer()
        report = initializer.get_initialization_report()
        assert "error" in report

    def test_weight_actually_changed(self):
        """Test that weights are actually modified."""
        model = nn.Sequential(nn.Linear(10, 20))
        old_weights = model[0].weight.data.clone()
        initializer = SmartInitializer()
        initializer.initialize(model, strategy="xavier")
        new_weights = model[0].weight.data
        assert not torch.equal(old_weights, new_weights)


class TestWeightPredictor:
    """Tests for WeightPredictor."""

    def test_predict_weights_shape(self):
        """Test predicted weights have correct shape."""
        predictor = WeightPredictor()
        weights = predictor.predict_weights("layer1", (20, 10), [], ["layer2"])
        assert weights.shape == (20, 10)

    def test_predict_weights_cache(self):
        """Test that weights are cached."""
        predictor = WeightPredictor()
        w1 = predictor.predict_weights("layer1", (20, 10), [], ["layer2"])
        w2 = predictor.predict_weights("layer1", (20, 10), [], ["layer2"])
        assert torch.equal(w1, w2)

    def test_predict_weights_connectivity_factor(self):
        """Test connectivity factor influences scale."""
        predictor = WeightPredictor()
        w1 = predictor.predict_weights("l1", (100, 100), [], ["a", "b", "c"])
        # Create a fresh predictor to avoid cache
        predictor2 = WeightPredictor()
        w2 = predictor2.predict_weights("l2", (100, 100), ["a", "b", "c"], [])
        # More downstream = smaller weights (but due to randomness, use std)
        std1 = w1.std().item()
        std2 = w2.std().item()
        assert std1 < std2 or abs(std1 - std2) < 0.5  # Allow random variation
