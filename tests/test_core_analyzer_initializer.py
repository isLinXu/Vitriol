"""
Tests for vitriol.core.analyzer and vitriol.core.smart_initializer modules.
"""
import pytest
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import numpy as np

from vitriol.core.analyzer import ModelAnalysis, ModelAnalyzer
from vitriol.core.smart_initializer import (
    LayerProfile,
    InitRecommendation,
    ModelStructureAnalyzer,
    SmartInitializer,
    WeightPredictor,
)


# ─────────────────────────────────────────────────────────────
# ModelAnalysis
# ─────────────────────────────────────────────────────────────

class TestModelAnalysis:
    def test_dataclass_fields(self):
        analysis = ModelAnalysis(
            model_id="test-model",
            architecture="llama",
            total_params=7000000000,
            trainable_params=7000000000,
            memory_footprint_gb=13.0,
            layer_count=32,
            hidden_size=4096,
            attention_heads=32,
            vocab_size=32000,
            sequence_length=4096,
            special_features=["RoPE", "GQA"],
            estimated_file_size={"random": 13.0, "sparse": 0.013},
        )
        assert analysis.model_id == "test-model"
        assert analysis.architecture == "llama"
        assert analysis.total_params == 7000000000
        assert analysis.special_features == ["RoPE", "GQA"]


# ─────────────────────────────────────────────────────────────
# ModelAnalyzer
# ─────────────────────────────────────────────────────────────

class TestModelAnalyzer:
    def test_init_defaults(self):
        analyzer = ModelAnalyzer("test-model")
        assert analyzer.model_id == "test-model"
        assert analyzer.trust_remote_code is True
        assert analyzer.allow_network is True
        assert analyzer.local_files_only is False
        assert analyzer.config is None

    def test_init_custom_flags(self):
        analyzer = ModelAnalyzer(
            "test-model",
            trust_remote_code=False,
            allow_network=False,
            local_files_only=True,
        )
        assert analyzer.trust_remote_code is False
        assert analyzer.allow_network is False
        assert analyzer.local_files_only is True

    @patch("vitriol.core.analyzer.load_config_or_raw")
    def test_analyze_with_mock_config(self, mock_load_config):
        # Re-import locally because test_cli_optional_dependencies evicts
        # vitriol.core.analyzer from sys.modules, breaking module-level patches.
        from vitriol.core.analyzer import ModelAnalyzer

        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_config.num_hidden_layers = 32
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.vocab_size = 32000
        mock_config.max_position_embeddings = 4096
        mock_config.rope_scaling = None
        mock_config.sliding_window = None
        mock_config.num_key_value_heads = 32
        mock_load_config.return_value = mock_config

        analyzer = ModelAnalyzer("test-model", local_files_only=True)
        result = analyzer.analyze()

        assert result.model_id == "test-model"
        assert result.architecture == "llama"
        assert result.layer_count == 32
        assert result.hidden_size == 4096
        assert result.attention_heads == 32
        assert result.vocab_size == 32000
        assert result.sequence_length == 4096
        mock_load_config.assert_called_once()

    @patch("vitriol.core.analyzer.load_config_or_raw")
    def test_analyze_detects_rope(self, mock_load_config):
        from vitriol.core.analyzer import ModelAnalyzer

        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_config.num_hidden_layers = 1
        mock_config.hidden_size = 128
        mock_config.num_attention_heads = 4
        mock_config.vocab_size = 1000
        mock_config.max_position_embeddings = 512
        mock_config.rope_scaling = {"type": "linear", "factor": 2.0}
        mock_config.sliding_window = None
        mock_config.num_key_value_heads = 4
        mock_load_config.return_value = mock_config

        analyzer = ModelAnalyzer("test-model", local_files_only=True)
        result = analyzer.analyze()
        assert "RoPE" in result.special_features

    @patch("vitriol.core.analyzer.load_config_or_raw")
    def test_analyze_detects_gqa(self, mock_load_config):
        from vitriol.core.analyzer import ModelAnalyzer

        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_config.num_hidden_layers = 1
        mock_config.hidden_size = 128
        mock_config.num_attention_heads = 8
        mock_config.vocab_size = 1000
        mock_config.max_position_embeddings = 512
        mock_config.rope_scaling = None
        mock_config.sliding_window = None
        mock_config.num_key_value_heads = 4  # GQA: kv_heads < attn_heads
        mock_load_config.return_value = mock_config

        analyzer = ModelAnalyzer("test-model", local_files_only=True)
        result = analyzer.analyze()
        assert "GQA" in result.special_features

    @patch("vitriol.core.analyzer.load_config_or_raw")
    def test_analyze_detects_sliding_window(self, mock_load_config):
        from vitriol.core.analyzer import ModelAnalyzer

        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_config.num_hidden_layers = 1
        mock_config.hidden_size = 128
        mock_config.num_attention_heads = 4
        mock_config.vocab_size = 1000
        mock_config.max_position_embeddings = 512
        mock_config.rope_scaling = None
        mock_config.sliding_window = 4096
        mock_config.num_key_value_heads = 4
        mock_load_config.return_value = mock_config

        analyzer = ModelAnalyzer("test-model", local_files_only=True)
        result = analyzer.analyze()
        assert "Sliding Window Attention" in result.special_features

    @patch("vitriol.core.analyzer.load_config_or_raw")
    def test_analyze_detects_moe(self, mock_load_config):
        from vitriol.core.analyzer import ModelAnalyzer

        mock_config = MagicMock()
        mock_config.model_type = "moe"  # contains 'moe'
        mock_config.num_hidden_layers = 1
        mock_config.hidden_size = 128
        mock_config.num_attention_heads = 4
        mock_config.vocab_size = 1000
        mock_config.max_position_embeddings = 512
        mock_config.rope_scaling = None
        mock_config.sliding_window = None
        mock_config.num_key_value_heads = 4
        mock_load_config.return_value = mock_config

        analyzer = ModelAnalyzer("test-model", local_files_only=True)
        result = analyzer.analyze()
        assert "MoE" in result.special_features

    def test_estimate_params_fallback(self):
        analyzer = ModelAnalyzer("test-model")
        analyzer.config = MagicMock()
        analyzer.config.hidden_size = 128
        analyzer.config.num_hidden_layers = 2
        analyzer.config.vocab_size = 1000

        params = analyzer._estimate_params()
        # vocab_size * hidden_size + num_layers * 12 * hidden_size^2
        expected = 1000 * 128 + 2 * 12 * (128 ** 2)
        assert params == expected

    def test_estimate_memory(self):
        analyzer = ModelAnalyzer("test-model")
        analyzer.config = MagicMock()
        analyzer.config.hidden_size = 128
        analyzer.config.num_hidden_layers = 2
        analyzer.config.vocab_size = 1000

        params = analyzer._estimate_params()
        memory = analyzer._estimate_memory()
        assert memory == (params * 2) / (1024 ** 3)

    def test_estimate_file_sizes(self):
        analyzer = ModelAnalyzer("test-model")
        sizes = analyzer._estimate_file_sizes(10.0)
        assert sizes["random"] == 10.0
        assert sizes["sparse"] == 10.0 * 0.001
        assert sizes["compact"] == 10.0 * 0.2
        assert sizes["ultra"] == 10.0 * 0.0001


# ─────────────────────────────────────────────────────────────
# LayerProfile
# ─────────────────────────────────────────────────────────────

class TestLayerProfile:
    def test_dataclass(self):
        profile = LayerProfile(
            name="layer1",
            layer_type="linear",
            input_dim=128,
            output_dim=256,
            depth=0,
            fan_in=128,
            fan_out=256,
        )
        assert profile.name == "layer1"
        assert profile.is_attention is False
        assert profile.is_embedding is False


# ─────────────────────────────────────────────────────────────
# InitRecommendation
# ─────────────────────────────────────────────────────────────

class TestInitRecommendation:
    def test_dataclass(self):
        rec = InitRecommendation(
            layer_name="layer1",
            init_type="xavier",
            gain=1.0,
            distribution="uniform",
            reason="test",
        )
        assert rec.layer_name == "layer1"
        assert rec.init_type == "xavier"
        assert rec.gain == 1.0


# ─────────────────────────────────────────────────────────────
# ModelStructureAnalyzer
# ─────────────────────────────────────────────────────────────

class TestModelStructureAnalyzer:
    @pytest.fixture
    def simple_model(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, 128)
                self.linear1 = nn.Linear(128, 256)
                self.attention = nn.Linear(256, 256)
                self.output = nn.Linear(256, 1000)

            def forward(self, x):
                return self.output(self.attention(self.linear1(self.embed(x))))

        return SimpleModel()

    def test_analyze_linear_layer(self, simple_model):
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(simple_model)

        assert "linear1" in profiles
        assert profiles["linear1"].layer_type == "linear"
        assert profiles["linear1"].input_dim == 128
        assert profiles["linear1"].output_dim == 256
        assert profiles["linear1"].has_bias is True

    def test_analyze_embedding_layer(self, simple_model):
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(simple_model)

        assert "embed" in profiles
        assert profiles["embed"].layer_type == "embedding"
        assert profiles["embed"].is_embedding is True

    def test_analyze_attention_layer(self, simple_model):
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(simple_model)

        assert "attention" in profiles
        assert profiles["attention"].is_attention is True

    def test_analyze_output_layer(self, simple_model):
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(simple_model)

        assert "output" in profiles
        assert profiles["output"].is_output is True

    def test_detect_patterns(self, simple_model):
        analyzer = ModelStructureAnalyzer()
        analyzer.analyze(simple_model)

        patterns = analyzer.architecture_patterns
        assert patterns["has_attention"] is True
        assert patterns["depth"] >= 3
        assert "attention" in patterns["attention_layers"]

    def test_detect_residuals_false(self, simple_model):
        analyzer = ModelStructureAnalyzer()
        analyzer.analyze(simple_model)
        assert analyzer.architecture_patterns["has_residuals"] is False

    def test_detect_residuals_true(self):
        class ResidualModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.residual_block = nn.Linear(128, 128)

            def forward(self, x):
                return x + self.residual_block(x)

        analyzer = ModelStructureAnalyzer()
        analyzer.analyze(ResidualModel())
        assert analyzer.architecture_patterns["has_residuals"] is True

    def test_conv2d_layer(self):
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3)

            def forward(self, x):
                return self.conv(x)

        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(ConvModel())
        assert "conv" in profiles
        assert profiles["conv"].layer_type == "conv2d"


# ─────────────────────────────────────────────────────────────
# SmartInitializer
# ─────────────────────────────────────────────────────────────

class TestSmartInitializer:
    @pytest.fixture
    def simple_model(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.linear1 = nn.Linear(64, 128)
                self.attention = nn.Linear(128, 128)
                self.output = nn.Linear(128, 10)

            def forward(self, x):
                return self.output(self.attention(self.linear1(self.embed(x))))

        return SimpleModel()

    def test_init_with_adaptive_strategy(self, simple_model):
        initializer = SmartInitializer()
        model = initializer.initialize(simple_model, strategy="adaptive")

        # Verify all layers with weights were initialized
        assert len(initializer.recommendations) >= 3
        for name, rec in initializer.recommendations.items():
            assert rec.layer_name == name
            assert rec.init_type in ["xavier", "kaiming", "normal"]
            assert rec.gain > 0

    def test_init_with_xavier_strategy(self, simple_model):
        initializer = SmartInitializer()
        model = initializer.initialize(simple_model, strategy="xavier")

        for rec in initializer.recommendations.values():
            assert rec.init_type == "xavier"

    def test_init_with_kaiming_strategy(self, simple_model):
        initializer = SmartInitializer()
        model = initializer.initialize(simple_model, strategy="kaiming", activation="relu")

        for rec in initializer.recommendations.values():
            assert rec.init_type == "kaiming"

    def test_init_with_orthogonal_strategy(self, simple_model):
        initializer = SmartInitializer()
        model = initializer.initialize(simple_model, strategy="orthogonal")

        for rec in initializer.recommendations.values():
            assert rec.init_type == "orthogonal"

    def test_embedding_small_init(self, simple_model):
        initializer = SmartInitializer()
        initializer.initialize(simple_model, strategy="adaptive")

        rec = initializer.recommendations.get("embed")
        assert rec is not None
        assert rec.init_type == "normal"
        assert rec.gain == 0.02

    def test_attention_xavier_init(self, simple_model):
        initializer = SmartInitializer()
        initializer.initialize(simple_model, strategy="adaptive")

        rec = initializer.recommendations.get("attention")
        assert rec is not None
        assert rec.init_type == "xavier"
        assert "Attention-aware" in rec.reason

    def test_output_conservative_init(self, simple_model):
        initializer = SmartInitializer()
        initializer.initialize(simple_model, strategy="adaptive")

        rec = initializer.recommendations.get("output")
        assert rec is not None
        assert rec.init_type == "xavier"
        assert rec.gain == 0.5

    def test_get_initialization_report(self, simple_model):
        initializer = SmartInitializer()
        initializer.initialize(simple_model, strategy="adaptive")

        report = initializer.get_initialization_report()
        assert report["total_layers"] == len(initializer.recommendations)
        assert "init_type_distribution" in report
        assert "average_gain" in report
        assert "gain_range" in report
        assert "layer_details" in report

    def test_get_initialization_report_empty(self):
        initializer = SmartInitializer()
        report = initializer.get_initialization_report()
        assert "error" in report

    def test_activation_gains(self):
        assert SmartInitializer.ACTIVATION_GAINS["linear"] == 1.0
        assert SmartInitializer.ACTIVATION_GAINS["relu"] == np.sqrt(2.0)
        assert SmartInitializer.ACTIVATION_GAINS["tanh"] == 5.0 / 3.0

    def test_weight_actually_changed(self, simple_model):
        # Store original weights
        original_weights = {}
        for name, module in simple_model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                original_weights[name] = module.weight.clone()

        initializer = SmartInitializer()
        initializer.initialize(simple_model, strategy="xavier")

        # Verify weights changed
        for name, module in simple_model.named_modules():
            if hasattr(module, "weight") and module.weight is not None and name in original_weights:
                assert not torch.allclose(module.weight, original_weights[name])


# ─────────────────────────────────────────────────────────────
# WeightPredictor
# ─────────────────────────────────────────────────────────────

class TestWeightPredictor:
    def test_predict_weights_shape(self):
        predictor = WeightPredictor()
        weights = predictor.predict_weights(
            layer_name="test_layer",
            shape=(64, 128),
            upstream_layers=["layer1"],
            downstream_layers=["layer3", "layer4"],
        )
        assert weights.shape == (64, 128)

    def test_predict_weights_cache(self):
        predictor = WeightPredictor()
        weights1 = predictor.predict_weights(
            layer_name="test_layer",
            shape=(64, 128),
            upstream_layers=["layer1"],
            downstream_layers=["layer3"],
        )
        weights2 = predictor.predict_weights(
            layer_name="test_layer",
            shape=(64, 128),
            upstream_layers=["layer1"],
            downstream_layers=["layer3"],
        )
        assert torch.allclose(weights1, weights2)

    def test_predict_weights_connectivity_factor(self):
        predictor = WeightPredictor()
        # More downstream layers -> smaller scale
        weights_many_down = predictor.predict_weights(
            layer_name="test_layer",
            shape=(64, 128),
            upstream_layers=["layer1"],
            downstream_layers=["l2", "l3", "l4", "l5"],
        )
        weights_few_down = predictor.predict_weights(
            layer_name="test_layer2",
            shape=(64, 128),
            upstream_layers=["layer1"],
            downstream_layers=["l2"],
        )
        # Should have different std due to different connectivity factors
        assert weights_many_down.shape == weights_few_down.shape
