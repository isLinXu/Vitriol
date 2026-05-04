"""
Metrics module for Vitriol compression evaluation.

Provides:
- Compression Intelligence Score (CIS) evaluation framework
- Strategy comparison and ranking
- Phase transition detection
"""

from .compression_intelligence import (
    CompressionScores,
    StrategyMetrics,
    CompressionIntelligenceScorer,
    CriticalPointDetector,
    STRATEGY_SCORE_MATRIX,
    compute_theoretical_psi,
    generate_score_comparison_table,
    InformationPreservationMetrics,
    StorageEfficiencyMetrics,
    ExpressivePowerMetrics,
    TrainabilityMetrics,
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
