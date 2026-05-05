"""Training utilities and progress tracking."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class TrainingProgress:
    """Training progress tracking."""
    epoch: int
    total_epochs: int
    total_loss: float
    svd_loss: float
    stats_loss: float
    norm_loss: float
    activation_loss: float
    avg_gen_time_ms: float
    learning_rate: float
    layers_trained: int
