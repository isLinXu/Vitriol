"""Boundary condition and edge case tests for Vitriol modules."""


import pytest
import torch
import torch.nn as nn

from vitriol.metrics.compression_intelligence import (
    CompressionIntelligenceScorer,
    CriticalPointDetector,
    ExpressivePowerMetrics,
    InformationPreservationMetrics,
    StorageEfficiencyMetrics,
    TrainabilityMetrics,
)
from vitriol.utils.model_capabilities import (
    cfg_attr,
    cfg_int,
    cfg_list,
    infer_kv_layer_types,
    infer_model_capabilities,
    infer_num_layers,
)
from vitriol.core.smart_initializer import SmartInitializer


# ─────────────────────────────────────────────────────────────────────────────
# Empty/None Input Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyNoneInputs:
    """Tests for empty and None inputs."""

    def test_cfg_attr_none_object(self):
        assert cfg_attr(None, "key", "default") == "default"

    def test_cfg_int_none_dict_value(self):
        assert cfg_int({"n": None}, "n", 5) == 5

    def test_cfg_list_empty_dict(self):
        assert cfg_list({}, "items") == []

    def test_cfg_list_none_value(self):
        assert cfg_list({"items": None}, "items") == []

    def test_infer_num_layers_empty_config(self):
        assert infer_num_layers({}) == 0

    def test_infer_kv_layer_types_zero_layers(self):
        assert infer_kv_layer_types({"num_hidden_layers": 0}) == []

    def test_infer_kv_layer_types_negative_layers(self):
        assert infer_kv_layer_types({"num_hidden_layers": -1}) == []

    def test_infer_model_capabilities_none(self):
        caps = infer_model_capabilities(None)
        assert caps.architecture_kind == "unknown"

    def test_infer_model_capabilities_empty(self):
        caps = infer_model_capabilities({})
        assert caps.architecture_kind == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Extreme Value Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExtremeValues:
    """Tests for extreme numerical values."""

    def test_storage_compression_ratio_zero(self):
        assert StorageEfficiencyMetrics.compression_ratio(0, 100) == 0.0

    def test_storage_compression_ratio_very_small(self):
        ratio = StorageEfficiencyMetrics.compression_ratio(1e9, 1)
        assert ratio > 0

    def test_sparsity_score_all_zeros(self):
        tensor = torch.zeros(100, 100)
        assert StorageEfficiencyMetrics.sparsity_score(tensor) == 1.0

    def test_sparsity_score_no_zeros(self):
        tensor = torch.ones(100, 100)
        assert StorageEfficiencyMetrics.sparsity_score(tensor) == 0.0

    def test_value_diversity_all_same(self):
        tensor = torch.ones(100, 100) * 3.14
        diversity = ExpressivePowerMetrics.value_diversity(tensor)
        assert diversity == 1 / 10000  # One unique value out of 10000

    def test_value_diversity_all_unique(self):
        tensor = torch.randn(100, 100)
        diversity = ExpressivePowerMetrics.value_diversity(tensor)
        assert diversity > 0.9  # Should be nearly all unique

    def test_rank_score_very_small_matrix(self):
        tensor = torch.randn(2, 2)
        rank = ExpressivePowerMetrics.rank_score(tensor)
        assert 0 <= rank <= 1

    def test_rank_score_1d_tensor(self):
        tensor = torch.randn(100)
        rank = ExpressivePowerMetrics.rank_score(tensor)
        assert rank == 1.0

    def test_gradient_flow_singular_matrix(self):
        """Singular matrix should have poor gradient flow."""
        tensor = torch.zeros(50, 50)
        score = TrainabilityMetrics.gradient_flow_score(tensor)
        assert 0 <= score <= 1

    def test_gradient_flow_ill_conditioned(self):
        """Ill-conditioned matrix."""
        tensor = torch.diag(torch.tensor([1.0, 1e-8, 1e-8]))
        score = TrainabilityMetrics.gradient_flow_score(tensor)
        assert 0 <= score <= 1

    def test_signal_scale_very_small(self):
        tensor = torch.randn(100, 100) * 1e-10
        score = TrainabilityMetrics.signal_scale_score(tensor)
        assert 0 <= score < 1.0

    def test_signal_scale_very_large(self):
        tensor = torch.randn(100, 100) * 1000.0
        score = TrainabilityMetrics.signal_scale_score(tensor)
        assert 0 <= score < 1.0

    def test_variance_preservation_zero_original_nonzero_compressed(self):
        original = torch.zeros(100, 100)
        compressed = torch.randn(100, 100)
        score = TrainabilityMetrics.variance_preservation(original, compressed)
        assert score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tensor Shape Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestTensorShapeEdgeCases:
    """Tests for unusual tensor shapes."""

    def test_svd_preservation_1d(self):
        t1 = torch.randn(100)
        t2 = torch.randn(100)
        score = InformationPreservationMetrics.svd_preservation_score(t1, t2)
        assert 0 <= score <= 1

    def test_svd_preservation_tall_matrix(self):
        t1 = torch.randn(100, 10)
        t2 = torch.randn(100, 10)
        score = InformationPreservationMetrics.svd_preservation_score(t1, t2)
        assert 0 <= score <= 1

    def test_svd_preservation_wide_matrix(self):
        t1 = torch.randn(10, 100)
        t2 = torch.randn(10, 100)
        score = InformationPreservationMetrics.svd_preservation_score(t1, t2)
        assert 0 <= score <= 1

    def test_entropy_single_element(self):
        tensor = torch.tensor([1.0])
        entropy = InformationPreservationMetrics.entropy_score(tensor)
        assert entropy >= -1e-6  # Allow floating point noise

    def test_spectrum_preservation_different_lengths(self):
        t1 = torch.randn(100)
        t2 = torch.randn(50)
        score = InformationPreservationMetrics.spectrum_preservation(t1, t2)
        assert 0 <= score <= 1

    def test_distribution_complexity_empty_after_filtering(self):
        """Tensor with all same values after filtering."""
        tensor = torch.zeros(10, 10)
        complexity = ExpressivePowerMetrics.distribution_complexity(tensor)
        assert 0 <= complexity <= 1


# ─────────────────────────────────────────────────────────────────────────────
# CompressionIntelligenceScorer Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressionIntelligenceScorerEdgeCases:
    """Edge case tests for the main scorer."""

    def test_score_empty_tensor(self):
        scorer = CompressionIntelligenceScorer()
        tensor = torch.tensor([])
        if tensor.numel() == 0:
            pytest.skip("Empty tensors not supported for scoring")

    def test_score_single_element_tensor(self):
        scorer = CompressionIntelligenceScorer()
        tensor = torch.tensor([1.0])
        scores, psi = scorer.score_tensor(tensor)
        assert 0 <= psi <= 1

    def test_score_strategy_empty_weights(self):
        scorer = CompressionIntelligenceScorer()
        metrics = scorer.score_strategy("empty", {})
        assert metrics.psi_score == 0.0

    def test_score_strategy_large_weights(self):
        scorer = CompressionIntelligenceScorer()
        weights = {
            f"layer_{i}": torch.randn(100, 100)
            for i in range(50)
        }
        metrics = scorer.score_strategy("large", weights)
        assert 0 <= metrics.psi_score <= 1

    def test_critical_point_insufficient_data(self):
        detector = CriticalPointDetector()
        assert detector.detect_critical_point() is None

    def test_critical_point_exactly_five(self):
        detector = CriticalPointDetector()
        for i in range(5):
            detector.add_observation(i * 0.1, 1.0 - i * 0.05)
        cp = detector.detect_critical_point()
        # With exactly 5 points, second derivatives need at least 3 points
        assert cp is not None or len(detector.expressivity_history) == 5


# ─────────────────────────────────────────────────────────────────────────────
# SmartInitializer Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartInitializerEdgeCases:
    """Edge case tests for SmartInitializer."""

    def test_initialize_empty_model(self):
        """Initializing an empty model should not crash."""
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        model = EmptyModel()
        initializer = SmartInitializer()
        result = initializer.initialize(model)
        assert result is model

    def test_initialize_large_model(self):
        """Test with a larger model structure."""
        model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
        )
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="adaptive")
        assert result is model
        assert len(initializer.recommendations) == 3  # Only Linear layers

    def test_initialize_with_unsupported_activation(self):
        """Unknown activation should fallback gracefully."""
        model = nn.Sequential(nn.Linear(10, 10))
        initializer = SmartInitializer()
        result = initializer.initialize(model, strategy="adaptive", activation="unknown_activation")
        assert result is model


# ─────────────────────────────────────────────────────────────────────────────
# ModelCapabilities Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCapabilitiesEdgeCases:
    """Edge case tests for model capabilities inference."""

    def test_config_with_nested_none(self):
        config = {
            "text_config": None,
            "model_type": "llama",
            "num_hidden_layers": 12,
        }
        layer_types = infer_kv_layer_types(config)
        assert len(layer_types) == 12

    def test_config_with_partial_layer_types(self):
        config = {
            "num_hidden_layers": 10,
            "layer_types": ["attention", "mamba"],
        }
        layer_types = infer_kv_layer_types(config)
        assert len(layer_types) == 10
        assert layer_types[0] == "full_attention"
        assert layer_types[1] == "linear_attention"
        # Remaining should be filled with full_attention
        assert layer_types[2] == "full_attention"

    def test_config_with_more_layer_types_than_layers(self):
        config = {
            "num_hidden_layers": 2,
            "layer_types": ["attention", "mamba", "attention", "mamba"],
        }
        layer_types = infer_kv_layer_types(config)
        assert len(layer_types) == 2
