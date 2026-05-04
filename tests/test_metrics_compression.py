"""Tests for metrics/compression_intelligence module."""


import pytest
import torch

from vitriol.metrics.compression_intelligence import (
    CompressionIntelligenceScorer,
    CompressionScores,
    CriticalPointDetector,
    ExpressivePowerMetrics,
    InformationPreservationMetrics,
    StorageEfficiencyMetrics,
    TrainabilityMetrics,
    compute_theoretical_psi,
    generate_score_comparison_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# CompressionScores Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressionScores:
    """Tests for CompressionScores dataclass."""

    def test_default_values(self):
        scores = CompressionScores()
        assert scores.info_preservation == 0.0
        assert scores.storage_efficiency == 0.0
        assert scores.expressive_power == 0.0
        assert scores.trainability == 0.0
        assert scores.phase_transition is False

    def test_to_dict(self):
        scores = CompressionScores(
            info_preservation=0.8,
            storage_efficiency=0.9,
            expressive_power=0.7,
            trainability=0.6,
            phase_transition=True,
        )
        d = scores.to_dict()
        assert d["info_preservation"] == 0.8
        assert d["storage_efficiency"] == 0.9
        assert d["phase_transition"] == 1.0  # bool converted to float


# ─────────────────────────────────────────────────────────────────────────────
# InformationPreservationMetrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInformationPreservationMetrics:
    """Tests for InformationPreservationMetrics."""

    def test_svd_preservation_identical(self):
        """SVD preservation of identical tensors should be ~1.0."""
        tensor = torch.randn(50, 50)
        score = InformationPreservationMetrics.svd_preservation_score(tensor, tensor)
        assert 0.99 <= score <= 1.0

    def test_svd_preservation_range(self):
        """Score should be in [0, 1]."""
        t1 = torch.randn(30, 30)
        t2 = torch.randn(30, 30)
        score = InformationPreservationMetrics.svd_preservation_score(t1, t2)
        assert 0.0 <= score <= 1.0

    def test_svd_preservation_different_shapes(self):
        """SVD with different shapes should still work."""
        t1 = torch.randn(50, 30)
        t2 = torch.randn(40, 30)
        score = InformationPreservationMetrics.svd_preservation_score(t1, t2)
        assert 0.0 <= score <= 1.0

    def test_entropy_score(self):
        """Entropy score should be positive for random data."""
        tensor = torch.randn(100, 100)
        entropy = InformationPreservationMetrics.entropy_score(tensor)
        assert entropy > 0

    def test_entropy_score_uniform(self):
        """Uniform data should have lower entropy."""
        uniform = torch.ones(100, 100)
        entropy = InformationPreservationMetrics.entropy_score(uniform)
        assert entropy >= -1e-6  # Allow floating point noise

    def test_spectrum_preservation_identical(self):
        """Spectrum preservation of identical tensors."""
        tensor = torch.randn(64, 64)
        score = InformationPreservationMetrics.spectrum_preservation(tensor, tensor)
        assert score > 0.9

    def test_spectrum_preservation_range(self):
        """Spectrum score should be in [0, 1]."""
        t1 = torch.randn(64, 64)
        t2 = torch.randn(64, 64)
        score = InformationPreservationMetrics.spectrum_preservation(t1, t2)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# StorageEfficiencyMetrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStorageEfficiencyMetrics:
    """Tests for StorageEfficiencyMetrics."""

    def test_compression_ratio_basic(self):
        ratio = StorageEfficiencyMetrics.compression_ratio(1000, 100)
        assert ratio == 0.1

    def test_compression_ratio_zero_original(self):
        ratio = StorageEfficiencyMetrics.compression_ratio(0, 100)
        assert ratio == 0.0

    def test_storage_score_high_compression(self):
        score = StorageEfficiencyMetrics.storage_score_from_ratio(0.01)
        assert score > 0.5  # 99% compression should score well

    def test_storage_score_no_compression(self):
        score = StorageEfficiencyMetrics.storage_score_from_ratio(1.0)
        assert score == 0.0

    def test_storage_score_zero(self):
        score = StorageEfficiencyMetrics.storage_score_from_ratio(0.0)
        assert score == 0.0

    def test_sparsity_score_empty(self):
        tensor = torch.zeros(10, 10)
        score = StorageEfficiencyMetrics.sparsity_score(tensor)
        assert score == 1.0

    def test_sparsity_score_dense(self):
        tensor = torch.ones(10, 10)
        score = StorageEfficiencyMetrics.sparsity_score(tensor)
        assert score == 0.0

    def test_metadata_overhead(self):
        score = StorageEfficiencyMetrics.metadata_overhead(1000, 100)
        assert score == 0.9

    def test_metadata_overhead_zero_compressed(self):
        score = StorageEfficiencyMetrics.metadata_overhead(0, 100)
        assert score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ExpressivePowerMetrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExpressivePowerMetrics:
    """Tests for ExpressivePowerMetrics."""

    def test_value_diversity_high(self):
        tensor = torch.randn(100, 100)
        diversity = ExpressivePowerMetrics.value_diversity(tensor)
        assert diversity > 0.5  # Random values should be diverse

    def test_value_diversity_low(self):
        tensor = torch.ones(100, 100) * 3.14
        diversity = ExpressivePowerMetrics.value_diversity(tensor)
        assert diversity < 0.1

    def test_distribution_complexity_random(self):
        tensor = torch.randn(100, 100)
        complexity = ExpressivePowerMetrics.distribution_complexity(tensor)
        assert 0.0 <= complexity <= 1.0

    def test_distribution_complexity_uniform(self):
        tensor = torch.ones(100, 100)
        complexity = ExpressivePowerMetrics.distribution_complexity(tensor)
        assert 0.0 <= complexity <= 1.0

    def test_rank_score_full_rank(self):
        tensor = torch.randn(50, 50)
        rank = ExpressivePowerMetrics.rank_score(tensor)
        assert 0.0 <= rank <= 1.0
        # Full-rank random matrix should have high rank
        assert rank > 0.8

    def test_rank_score_low_rank(self):
        """Low-rank matrix should score low."""
        u = torch.randn(50, 2)
        v = torch.randn(2, 50)
        tensor = u @ v
        rank = ExpressivePowerMetrics.rank_score(tensor)
        assert rank < 0.5

    def test_rank_score_1d(self):
        tensor = torch.randn(100)
        rank = ExpressivePowerMetrics.rank_score(tensor)
        assert rank == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TrainabilityMetrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainabilityMetrics:
    """Tests for TrainabilityMetrics."""

    def test_gradient_flow_score_well_conditioned(self):
        tensor = torch.eye(50)
        score = TrainabilityMetrics.gradient_flow_score(tensor)
        assert 0.0 <= score <= 1.0
        # Identity matrix has condition number 1 -> good score
        assert score > 0.5

    def test_gradient_flow_score_1d(self):
        tensor = torch.randn(100)
        score = TrainabilityMetrics.gradient_flow_score(tensor)
        assert score == 1.0

    def test_signal_scale_score_optimal(self):
        tensor = torch.randn(100, 100) * 0.05
        score = TrainabilityMetrics.signal_scale_score(tensor)
        assert score == 1.0  # In good range

    def test_signal_scale_score_too_small(self):
        tensor = torch.randn(100, 100) * 1e-8
        score = TrainabilityMetrics.signal_scale_score(tensor)
        assert score < 1.0

    def test_signal_scale_score_too_large(self):
        tensor = torch.randn(100, 100) * 10.0
        score = TrainabilityMetrics.signal_scale_score(tensor)
        assert score < 1.0

    def test_variance_preservation_identical(self):
        tensor = torch.randn(100, 100)
        score = TrainabilityMetrics.variance_preservation(tensor, tensor)
        assert score > 0.9  # Identical tensors -> perfect preservation

    def test_variance_preservation_zero_original(self):
        original = torch.zeros(100, 100)
        compressed = torch.zeros(100, 100)
        score = TrainabilityMetrics.variance_preservation(original, compressed)
        assert score == 1.0

    def test_variance_preservation_different_variance(self):
        original = torch.randn(100, 100)
        compressed = original * 0.5
        score = TrainabilityMetrics.variance_preservation(original, compressed)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CriticalPointDetector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCriticalPointDetector:
    """Tests for CriticalPointDetector."""

    def test_detect_critical_point_not_enough_data(self):
        detector = CriticalPointDetector()
        assert detector.detect_critical_point() is None

    def test_detect_critical_point_with_data(self):
        detector = CriticalPointDetector()
        for i in range(10):
            detector.add_observation(compression_ratio=i * 0.1, expressivity_score=1.0 - i * 0.05)
        cp = detector.detect_critical_point()
        assert cp is not None

    def test_is_above_critical_point_default(self):
        detector = CriticalPointDetector()
        assert detector.is_above_critical_point(0.95) is True  # Default critical point ~0.9
        assert detector.is_above_critical_point(0.5) is False

    def test_is_above_critical_point_with_data(self):
        detector = CriticalPointDetector()
        for i in range(10):
            detector.add_observation(compression_ratio=i * 0.1, expressivity_score=1.0)
        result = detector.is_above_critical_point(0.5)
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# CompressionIntelligenceScorer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressionIntelligenceScorer:
    """Tests for CompressionIntelligenceScorer."""

    def test_init_valid_weights(self):
        scorer = CompressionIntelligenceScorer(alpha=0.3, beta=0.3, gamma=0.25, delta=0.15)
        assert scorer.alpha == 0.3

    def test_init_invalid_weights(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            CompressionIntelligenceScorer(alpha=0.5, beta=0.5, gamma=0.5, delta=0.5)

    def test_score_tensor_with_original(self):
        scorer = CompressionIntelligenceScorer()
        original = torch.randn(50, 50)
        compressed = original + torch.randn(50, 50) * 0.1
        scores, psi = scorer.score_tensor(compressed, original=original)
        assert 0.0 <= psi <= 1.0
        assert 0.0 <= scores.info_preservation <= 1.0
        assert 0.0 <= scores.storage_efficiency <= 1.0
        assert 0.0 <= scores.expressive_power <= 1.0
        assert 0.0 <= scores.trainability <= 1.0

    def test_score_tensor_without_original(self):
        scorer = CompressionIntelligenceScorer()
        tensor = torch.randn(50, 50)
        scores, psi = scorer.score_tensor(tensor)
        assert 0.0 <= psi <= 1.0

    def test_score_strategy(self):
        scorer = CompressionIntelligenceScorer()
        weights = {
            "layer1": torch.randn(100, 100),
            "layer2": torch.randn(100, 50),
        }
        metrics = scorer.score_strategy("test_strategy", weights)
        assert metrics.strategy_name == "test_strategy"
        assert 0.0 <= metrics.psi_score <= 1.0
        assert len(metrics.layer_metrics) == 2
        assert len(metrics.radar_vector) == 4

    def test_score_strategy_empty(self):
        scorer = CompressionIntelligenceScorer()
        metrics = scorer.score_strategy("empty", {})
        assert metrics.psi_score == 0.0
        assert metrics.layer_metrics == {}

    def test_compare_strategies(self):
        scorer = CompressionIntelligenceScorer()
        m1 = scorer.score_strategy("s1", {"l1": torch.randn(10, 10)})
        m2 = scorer.score_strategy("s2", {"l1": torch.randn(10, 10)})
        comparison = CompressionIntelligenceScorer.compare_strategies([m1, m2])
        assert len(comparison) == 2
        assert comparison[0][2] == 1  # First is rank 1

    def test_generate_report(self):
        scorer = CompressionIntelligenceScorer()
        weights = {"l1": torch.randn(10, 10)}
        metrics = scorer.score_strategy("test", weights)
        report = CompressionIntelligenceScorer.generate_report(metrics)
        assert "Compression Intelligence Report" in report
        assert "test" in report
        assert "PSI Score" in report


# ─────────────────────────────────────────────────────────────────────────────
# Module-level Function Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTheoreticalPsi:
    """Tests for compute_theoretical_psi."""

    def test_known_strategies(self):
        for name in ["random", "compact", "ultra", "sparse", "binary", "ternary",
                     "quantized", "lowrank", "structured_sparse", "learned"]:
            psi = compute_theoretical_psi(name)
            assert 0.0 <= psi <= 1.0

    def test_unknown_strategy(self):
        psi = compute_theoretical_psi("nonexistent_strategy")
        assert psi == 0.0

    def test_custom_weights(self):
        psi1 = compute_theoretical_psi("compact", alpha=0.3, beta=0.3, gamma=0.25, delta=0.15)
        psi2 = compute_theoretical_psi("compact", alpha=0.0, beta=1.0, gamma=0.0, delta=0.0)
        assert psi2 > psi1  # More weight on storage should favor compact


class TestGenerateScoreComparisonTable:
    """Tests for generate_score_comparison_table."""

    def test_table_format(self):
        table = generate_score_comparison_table()
        assert "Strategy" in table
        assert "PSI Score" in table
        assert "|" in table  # Markdown table format

    def test_all_strategies_present(self):
        table = generate_score_comparison_table()
        for name in ["random", "compact", "ultra", "learned"]:
            assert name in table

    def test_sorted_by_psi(self):
        table = generate_score_comparison_table()
        lines = [l for l in table.split("\n") if l.startswith("|") and "Strategy" not in l and "---" not in l]
        # Check descending order by PSI (last column)
        psi_values = []
        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            # Filter out empty strings
            parts = [p for p in parts if p]
            if len(parts) >= 5:
                try:
                    psi = float(parts[-1])
                    psi_values.append(psi)
                except ValueError:
                    pass
        for i in range(len(psi_values) - 1):
            assert psi_values[i] >= psi_values[i + 1]
