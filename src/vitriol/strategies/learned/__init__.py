# Learned Weight Strategy Package
"""
This package provides learned weight generation strategies,
transforming compression from engineering to learning.

Primary innovation: compression as a learning problem.
"""

from .layer_config import LayerConfig
from .generator_network import WeightGeneratorNetwork
from .spectral_matching import SpectralMatchingLoss
from .training_utils import TrainingProgress

__all__ = [
    "LayerConfig",
    "WeightGeneratorNetwork",
    "SpectralMatchingLoss",
    "TrainingProgress",
]
