"""
Learned Weight Strategy - Sub-package.

This package provides learned weight generation strategies,
transforming compression from engineering to learning.

Primary innovation: compression as a learning problem.
"""

import sys
import os

# Add src to path for absolute imports
_src_dir = os.path.join(os.path.dirname(__file__), '..', '..')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

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
