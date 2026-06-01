"""Layer configuration data class."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class LayerConfig:
    """Configuration encoding a layer's properties."""
    name: str                          # Parameter name
    shape: Tuple[int, ...]             # Tensor shape
    layer_type: str                    # "linear", "embedding", "conv2d", etc.
    depth: int = 0                     # Layer depth in network
    num_params: int = 0               # Total parameters
    is_attention: bool = False         # Is attention layer
    is_embedding: bool = False         # Is embedding layer
    is_output: bool = False            # Is output layer
    fan_in: int = 0                    # Fan-in
    fan_out: int = 0                  # Fan-out

    def to_vector(self) -> torch.Tensor:
        """Convert to fixed-size feature vector for neural network."""
        features = [
            np.log1p(self.shape[0] if len(self.shape) > 0 else 1),  # Input dim (log scale)
            np.log1p(self.shape[-1] if len(self.shape) > 0 else 1), # Output dim (log scale)
            float(self.layer_type == "linear"),
            float(self.layer_type == "embedding"),
            float(self.is_output),
            np.log1p(self.depth),
            np.log1p(self.num_params),
        ]
        return torch.tensor(features, dtype=torch.float32)
