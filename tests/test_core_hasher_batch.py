"""Tests for core/hasher.py and core/batch.py."""

import json
import os
import tempfile
from pathlib import Path


from vitriol.core.hasher import ModelHasher


class TestModelHasher:
    """Tests for ModelHasher."""

    def test_init(self):
        hasher = ModelHasher("/tmp/test_model")
        assert hasher.model_path == Path("/tmp/test_model")

    def test_hash_dict_deterministic(self):
        hasher = ModelHasher("/tmp/test")
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert hasher._hash_dict(d1) == hasher._hash_dict(d2)

    def test_hash_dict_different_data(self):
        hasher = ModelHasher("/tmp/test")
        h1 = hasher._hash_dict({"a": 1})
        h2 = hasher._hash_dict({"a": 2})
        assert h1 != h2

    def test_compute_architecture_hash_no_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            hasher = ModelHasher(tmp)
            result = hasher.compute_architecture_hash()
            assert result == "N/A"

    def test_compute_architecture_hash_with_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "vocab_size": 32000,
            }
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            hasher = ModelHasher(tmp)
            result = hasher.compute_architecture_hash()
            assert result != "N/A"
            assert len(result) == 64  # SHA-256 hex

    def test_compute_architecture_hash_with_multimodal(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vision_config": {"hidden_size": 256, "num_hidden_layers": 4},
            }
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            hasher = ModelHasher(tmp)
            result = hasher.compute_architecture_hash()
            assert result != "N/A"

    def test_compute_architecture_hash_diffusers(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = {"_class_name": "StableDiffusionPipeline"}
            (Path(tmp) / "model_index.json").write_text(json.dumps(index))
            hasher = ModelHasher(tmp)
            result = hasher.compute_architecture_hash()
            assert result != "N/A"

    def test_compute_weight_distribution_hash_no_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            hasher = ModelHasher(tmp)
            result = hasher.compute_weight_distribution_hash()
            assert result == "N/A"

    def test_compute_activation_signature_hash_no_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            hasher = ModelHasher(tmp)
            result = hasher.compute_activation_signature_hash()
            assert result == "N/A"

    def test_compute_activation_signature_hash_with_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 32000}
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            hasher = ModelHasher(tmp)
            result = hasher.compute_activation_signature_hash()
            assert result != "N/A"
            assert len(result) == 64

    def test_generate_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"hidden_size": 768, "num_hidden_layers": 12, "vocab_size": 32000}
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            hasher = ModelHasher(tmp)
            fp = hasher.generate_fingerprint()
            assert "model_path" in fp
            assert "architecture_hash" in fp
            assert "weight_distribution_hash" in fp
            assert "vitriol_signature" in fp
            assert fp["vitriol_signature"].startswith("arx_")


class TestBatchGeneratorImports:
    """Tests for batch generator module structure."""

    def test_batch_generator_imports(self):
        from vitriol.core.batch import BatchGenerator
        assert BatchGenerator is not None

    def test_batch_generator_init_with_yaml(self):
        from vitriol.core.batch import BatchGenerator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("models:\n")
            f.write("  - id: test-model\n")
            f.write("    output: /tmp/out\n")
            path = f.name
        try:
            gen = BatchGenerator(path)
            assert gen.config is not None
            assert "models" in gen.config
        finally:
            os.unlink(path)

    def test_batch_generator_init_empty(self):
        from vitriol.core.batch import BatchGenerator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("models: []\n")
            path = f.name
        try:
            gen = BatchGenerator(path)
            assert gen.config.get("models") == []
        finally:
            os.unlink(path)
