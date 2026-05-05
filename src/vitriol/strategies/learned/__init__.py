"""
Learned Weight Strategy - Sub-package.

This package provides learned weight generation strategies,
transforming compression from engineering to learning.

Primary innovation: compression as a learning problem.
"""

from ..layer_config.learned import LearnedWeightStrategy, HybridLearnedStrategy

__all__ = [
    "LearnedWeightStrategy",
    "HybridLearnedStrategy",
]
