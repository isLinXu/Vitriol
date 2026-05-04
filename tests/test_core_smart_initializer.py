"""Tests for vitriol.core.smart_initializer module."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from vitriol.core.smart_initializer import (
    LayerProfile,
    InitRecommendation,
    ModelStructureAnalyzer,
    SmartInitializer,
    WeightPredictor,
)


class TestLayerProfile:
    """Tests for LayerProfile dataclass."""

    def test_creation(self):
        """Test LayerProfile creation."""
        profile = LayerProfile(
            name="test_layer",
            layer_type="linear",
            input_dim=100,
            output_dim=200,
            depth=1,
            fan_in=100,
            fan_out=200
        )
        assert profile.name == "test_layer"
        assert profile.layer_type == "linear"
        assert profile.input_dim == 100
        assert profile.output_dim == 200
        assert profile.depth == 1
        assert profile.activation is None
        assert profile.has_bias is True
        assert profile.is_attention is False
        assert profile.is_embedding is False
        assert profile.is_output is False


class TestInitRecommendation:
    """Tests for InitRecommendation dataclass."""

    def test_creation(self):
        """Test InitRecommendation creation."""
        rec = InitRecommendation(
            layer_name="test",
            init_type="xavier",
            gain=1.0,
            distribution="uniform",
            reason="Test reason"
        )
        assert rec.layer_name == "test"
        assert rec.init_type == "xavier"
        assert rec.gain == 1.0
        assert rec.distribution == "uniform"
        assert rec.reason == "Test reason"


class TestModelStructureAnalyzer:
    """Tests for ModelStructureAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = ModelStructureAnalyzer()
        assert analyzer.layer_profiles == {}
        assert analyzer.architecture_patterns == {}

    def test_analyze_linear_layers(self):
        """Test analyzing model with linear layers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(model)

        assert len(profiles) == 2
        assert any(p.layer_type == "linear" for p in profiles.values())

    def test_analyze_conv2d(self):
        """Test analyzing model with Conv2d layers."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Conv2d(16, 32, 3)
        )
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(model)

        assert len(profiles) == 2
        assert any(p.layer_type == "conv2d" for p in profiles.values())

    def test_analyze_embedding(self):
        """Test analyzing model with Embedding layer."""
        model = nn.Sequential(
            nn.Embedding(1000, 128),
            nn.Linear(128, 10)
        )
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(model)

        assert len(profiles) == 2
        assert any(p.layer_type == "embedding" for p in profiles.values())
        assert any(p.is_embedding for p in profiles.values())

    def test_analyze_attention_detection(self):
        """Test attention layer detection."""
        model = nn.ModuleDict({
            "attention": nn.Linear(64, 64),
            "attn_proj": nn.Linear(64, 64)
        })
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(model)

        assert any(p.is_attention for p in profiles.values())

    def test_analyze_output_detection(self):
        """Test output layer detection."""
        model = nn.ModuleDict({
            "output": nn.Linear(64, 10),
            "lm_head": nn.Linear(64, 1000)
        })
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(model)

        assert any(p.is_output for p in profiles.values())

    def test_detect_residuals(self):
        """Test residual connection detection."""
        analyzer = ModelStructureAnalyzer()
        analyzer.layer_profiles = {
            "layer1": LayerProfile("layer1", "linear", 10, 10, 0, 10, 10),
            "residual_proj": LayerProfile("residual_proj", "linear", 10, 10, 1, 10, 10)
        }
        assert analyzer._detect_residuals() is True

    def test_detect_no_residuals(self):
        """Test when no residuals present."""
        analyzer = ModelStructureAnalyzer()
        analyzer.layer_profiles = {
            "layer1": LayerProfile("layer1", "linear", 10, 10, 0, 10, 10)
        }
        assert analyzer._detect_residuals() is False

    def test_empty_model(self):
        """Test analyzing empty model."""
        model = nn.Module()
        analyzer = ModelStructureAnalyzer()
        profiles = analyzer.analyze(model)
        assert len(profiles) == 0


class TestSmartInitializer:
    """Tests for SmartInitializer class."""

    def test_init(self):
        """Test initialization."""
        initializer = SmartInitializer()
        assert initializer.analyzer is not None
        assert initializer.recommendations == {}

    def test_activation_gains(self):
        """Test activation gain constants."""
        assert SmartInitializer.ACTIVATION_GAINS["linear"] == 1.0
        assert SmartInitializer.ACTIVATION_GAINS["relu"] == np.sqrt(2.0)
        assert SmartInitializer.ACTIVATION_GAINS["tanh"] == 5.0 / 3.0

    def test_initialize_adaptive(self):
        """Test adaptive initialization."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="adaptive")

        assert result is model
        assert len(initializer.recommendations) == 2

    def test_initialize_xavier(self):
        """Test xavier initialization strategy."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="xavier")

        assert result is model
        assert len(initializer.recommendations) == 2
        for rec in initializer.recommendations.values():
            assert rec.init_type == "xavier"

    def test_initialize_kaiming(self):
        """Test kaiming initialization strategy."""
        model = nn.Sequential(
            nn.Linear(10, 20)
        )
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="kaiming", activation="relu")

        assert result is model
        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "kaiming"

    def test_initialize_orthogonal(self):
        """Test orthogonal initialization strategy."""
        model = nn.Sequential(
            nn.Linear(10, 20)
        )
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="orthogonal")

        assert result is model
        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "orthogonal"

    def test_initialize_embedding_layer(self):
        """Test initialization of embedding layer."""
        model = nn.Sequential(
            nn.Embedding(100, 64)
        )
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="adaptive")

        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "normal"
        assert rec.gain == 0.02

    def test_initialize_attention_layer(self):
        """Test initialization of attention layer."""
        model = nn.ModuleDict({
            "attention": nn.Linear(64, 64)
        })
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="adaptive")

        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "xavier"
        assert "Attention-aware" in rec.reason

    def test_initialize_output_layer(self):
        """Test initialization of output layer."""
        model = nn.ModuleDict({
            "output": nn.Linear(64, 10)
        })
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="adaptive")

        rec = list(initializer.recommendations.values())[0]
        assert rec.init_type == "xavier"
        assert rec.gain == 0.5

    def test_apply_recommendation_xavier_uniform(self):
        """Test applying xavier uniform recommendation."""
        layer = nn.Linear(10, 10)
        rec = InitRecommendation("test", "xavier", 1.0, "uniform")
        initializer = SmartInitializer()
        initializer._apply_recommendation(layer, rec)

        # Weights should be modified (not all zeros)
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_apply_recommendation_kaiming_normal(self):
        """Test applying kaiming normal recommendation."""
        layer = nn.Linear(10, 10)
        rec = InitRecommendation("test", "kaiming", np.sqrt(2.0), "normal")
        initializer = SmartInitializer()
        initializer._apply_recommendation(layer, rec)

        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_apply_recommendation_orthogonal(self):
        """Test applying orthogonal recommendation."""
        layer = nn.Linear(10, 10)
        rec = InitRecommendation("test", "orthogonal", 1.0, "normal")
        initializer = SmartInitializer()
        initializer._apply_recommendation(layer, rec)

        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_apply_recommendation_normal(self):
        """Test applying normal recommendation."""
        layer = nn.Linear(10, 10)
        rec = InitRecommendation("test", "normal", 0.02, "normal")
        initializer = SmartInitializer()
        initializer._apply_recommendation(layer, rec)

        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_apply_recommendation_uniform(self):
        """Test applying uniform recommendation."""
        layer = nn.Linear(10, 10)
        rec = InitRecommendation("test", "uniform", 0.1, "uniform")
        initializer = SmartInitializer()
        initializer._apply_recommendation(layer, rec)

        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

    def test_apply_recommendation_no_weight(self):
        """Test applying recommendation to layer without weight."""
        layer = nn.ReLU()
        rec = InitRecommendation("test", "xavier", 1.0, "uniform")
        initializer = SmartInitializer()
        # Should not raise
        initializer._apply_recommendation(layer, rec)

    def test_get_initialization_report_empty(self):
        """Test report before initialization."""
        initializer = SmartInitializer()
        report = initializer.get_initialization_report()
        assert "error" in report

    def test_get_initialization_report(self):
        """Test report after initialization."""
        model = nn.Sequential(nn.Linear(10, 10))
        initializer = SmartInitializer()
        initializer.initialize(model)

        report = initializer.get_initialization_report()
        assert report["total_layers"] == 1
        assert "init_type_distribution" in report
        assert "average_gain" in report
        assert "gain_range" in report
        assert "layer_details" in report

    def test_generate_adaptive_recommendations_empty(self):
        """Test adaptive recommendations with empty profiles."""
        initializer = SmartInitializer()
        recs = initializer._generate_adaptive_recommendations({}, "gelu")
        assert recs == {}

    def test_generate_uniform_recommendations(self):
        """Test uniform recommendations generation."""
        profiles = {
            "layer1": LayerProfile("layer1", "linear", 10, 10, 0, 10, 10)
        }
        initializer = SmartInitializer()
        recs = initializer._generate_uniform_recommendations(profiles, "xavier", "gelu")

        assert len(recs) == 1
        assert recs["layer1"].init_type == "xavier"


class TestWeightPredictor:
    """Tests for WeightPredictor class."""

    def test_init(self):
        """Test initialization."""
        predictor = WeightPredictor()
        assert predictor.initialization_cache == {}

    def test_predict_weights_shape(self):
        """Test predicted weights shape."""
        predictor = WeightPredictor()
        weights = predictor.predict_weights("layer1", (10, 20), ["prev"], ["next"])
        assert weights.shape == (10, 20)

    def test_predict_weights_caching(self):
        """Test that predictions are cached."""
        predictor = WeightPredictor()
        weights1 = predictor.predict_weights("layer1", (5, 5), [], ["next"])
        weights2 = predictor.predict_weights("layer1", (5, 5), [], ["next"])

        assert torch.allclose(weights1, weights2)
        assert len(predictor.initialization_cache) == 1

    def test_predict_weights_connectivity_factor(self):
        """Test connectivity factor influences scale."""
        predictor = WeightPredictor()
        weights_many_downstream = predictor.predict_weights("l1", (10, 10), ["a"], ["b", "c", "d"])
        weights_few_downstream = predictor.predict_weights("l2", (10, 10), ["a"], ["b"])

        # More downstream layers should have smaller weights on average
        scale_many = weights_many_downstream.abs().mean().item()
        scale_few = weights_few_downstream.abs().mean().item()
        assert scale_many < scale_few
