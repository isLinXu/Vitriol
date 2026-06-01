"""
Metrics module for Vitriol compression evaluation.

Provides:
- Compression Intelligence Score (CIS) evaluation framework
- Strategy comparison and ranking
- Phase transition detection
"""

from .compression_intelligence import (
    STRATEGY_SCORE_MATRIX,
    CompressionIntelligenceScorer,
    CompressionScores,
    CriticalPointDetector,
    ExpressivePowerMetrics,
    InformationPreservationMetrics,
    StorageEfficiencyMetrics,
    StrategyMetrics,
    TrainabilityMetrics,
    compute_theoretical_psi,
    generate_score_comparison_table,
)

__all__ = [
    "CompressionScores",
    "StrategyMetrics",
    "CompressionIntelligenceScorer",
    "CriticalPointDetector",
    "STRATEGY_SCORE_MATRIX",
    "compute_theoretical_psi",
    "generate_score_comparison_table",
    "InformationPreservationMetrics",
    "StorageEfficiencyMetrics",
    "ExpressivePowerMetrics",
    "TrainabilityMetrics",
]
