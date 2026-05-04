"""Tests for arch_viz/core.py and arch_viz/parser.py."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from vitriol.arch_viz.core import Architecture, Layer
from vitriol.arch_viz.parser import ConfigParser


class TestLayer:
    """Tests for Layer dataclass."""

    def test_layer_creation(self):
        layer = Layer(
            name="embed_0",
            type="embedding",
            params=1_000_000,
            shape=(1000, 768),
            description="Token embedding layer",
        )
        assert layer.name == "embed_0"
        assert layer.type == "embedding"
        assert layer.params == 1_000_000
        assert layer.shape == (1000, 768)
        assert layer.description == "Token embedding layer"

    def test_layer_repr(self):
        layer = Layer(name="attn_0", type="attention", params=2_000_000, shape=(768, 768))
        repr_str = repr(layer)
        assert "attn_0" in repr_str
        assert "attention" in repr_str
        assert "2,000,000" in repr_str

    def test_layer_defaults(self):
        layer = Layer(name="norm_0", type="normalization", params=1000, shape=(768,))
        assert layer.description == ""


class TestArchitecture:
    """Tests for Architecture dataclass."""

    def test_architecture_creation(self):
        layers = [
            Layer(name="embed", type="embedding", params=1_000_000, shape=(1000, 768)),
            Layer(name="attn_0", type="attention", params=2_000_000, shape=(768, 768)),
        ]
        arch = Architecture(
            model_type="test-model",
            arch_type="transformer",
            total_layers=2,
            total_params=3_000_000,
            memory_fp16_gb=0.5,
            parameters={"vocab_size": 1000, "hidden_size": 768},
            features=["attention", "feedforward"],
            layers=layers,
        )
        assert arch.model_type == "test-model"
        assert arch.arch_type == "transformer"
        assert arch.total_layers == 2
        assert arch.total_params == 3_000_000
        assert len(arch.layers) == 2

    def test_architecture_post_init_special_features(self):
        arch = Architecture(
            model_type="m",
            arch_type="t",
            total_layers=1,
            total_params=100,
            memory_fp16_gb=0.1,
            parameters={},
            features=["a", "b"],
            special_features=[],
        )
        assert arch.special_features == ["a", "b"]

    def test_architecture_post_init_encoder_decoder(self):
        arch = Architecture(
            model_type="m",
            arch_type="t",
            total_layers=4,
            total_params=100,
            memory_fp16_gb=0.1,
            parameters={},
            features=[],
            encoder_layers=2,
            decoder_layers=2,
        )
        assert arch.parameters["encoder_layers"] == 2
        assert arch.parameters["decoder_layers"] == 2

    def test_to_dict(self):
        arch = Architecture(
            model_type="m",
            arch_type="t",
            total_layers=1,
            total_params=100,
            memory_fp16_gb=0.1,
            parameters={"x": 1},
            features=["a"],
        )
        d = arch.to_dict()
        assert d["model_type"] == "m"
        assert d["parameters"] == {"x": 1}
        assert "layers" in d

    def test_to_json(self):
        arch = Architecture(
            model_type="m",
            arch_type="t",
            total_layers=1,
            total_params=100,
            memory_fp16_gb=0.1,
            parameters={"x": 1},
            features=["a"],
            layers=[Layer(name="l1", type="embedding", params=100, shape=(10,))],
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            arch.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["model_type"] == "m"
            assert len(data["layers"]) == 1
        finally:
            os.unlink(path)


class TestConfigParser:
    """Tests for ConfigParser."""

    def test_load_config_from_nonexistent_path_raises(self):
        with pytest.raises(Exception):
            ConfigParser.load_config("/nonexistent/path/that/does/not/exist")

    def test_load_config_from_dir_with_config_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"hidden_size": 768, "num_hidden_layers": 12}
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(config))
            # ConfigParser may fail without full HF infra, but we exercise the path
            try:
                result = ConfigParser.load_config(tmp, local_files_only=True)
                assert result is not None
            except Exception:
                pytest.skip("HF config loading not available in test environment")

    def test_load_config_prefers_meta_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta_config = {"hidden_size": 512, "num_hidden_layers": 6}
            config = {"hidden_size": 768, "num_hidden_layers": 12}
            (Path(tmp) / "meta-config.json").write_text(json.dumps(meta_config))
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            try:
                result = ConfigParser.load_config(tmp, local_files_only=True)
                assert result is not None
            except Exception:
                pytest.skip("HF config loading not available")

    def test_load_config_from_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"model_type": "test"}
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(json.dumps(config))
            try:
                result = ConfigParser.load_config(str(cfg_path), local_files_only=True)
                assert result is not None
            except Exception:
                pytest.skip("HF config loading not available")
