"""
Learned Weight Strategy - Sub-package.

This package provides learned weight generation strategies,
transforming compression from engineering to learning.

Primary innovation: compression as a learning problem.
"""

import sys
import os

# Add the src directory to sys.path so we can import from it
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from .layer_config import LayerConfig
from .generator_network import WeightGeneratorNetwork
from .spectral_matching import SpectralMatchingLoss
from .training_utils import TrainingProgress