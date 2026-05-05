#!/usr/bin/env python3
"""Test script to verify the split module imports work correctly."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now try relative imports since absolute imports fail
try:
    from vitriol.strategies.learned.layer_config import LayerConfig
    from vitriol.strategies.learned.generator_network import WeightGeneratorNetwork
    from vitriol.strategies.learned.spectral_matching import SpectralMatchingLoss
    from vitriol.strategies.learned.training_utils import TrainingProgress
    
    print("✅ All split module imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
