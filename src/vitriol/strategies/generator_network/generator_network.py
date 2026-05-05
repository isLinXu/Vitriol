"""Weight Generator Network."""

import torch
import torch.nn as nn
import numpy as np


@dataclass
class LayerConfig:
    """Configuration encoding a layer's properties."""
    name: str
    shape: Tuple[int, ...]
    layer_type: str
    depth: int = 0
    num_params: int = 0
    is_attention: bool = False
    is_embedding: bool = False
    is_output: bool = False

    def to_vector(self) -> torch.Tensor:
        features = [
            np.log1p(self.shape[0] if len(self.shape) > 0 else 1),
            np.log1p(self.shape[-1] if len(self.shape) > 0 else 1),
            float(self.layer_type == "linear"),
            float(self.layer_type == "embedding"),
            float(self.is_output),
            np.log1p(self.depth),
            np.log1p(self.num_params),
        ]
        return torch.tensor(features, dtype=torch.float32)


class WeightGeneratorNetwork(nn.Module):
    """Neural network that generates weights from latent noise + layer config.

    Architecture (v2 - Shape-Aware):
        z (latent) ─┬─► MLP ─► concat(layer_config) ─► Combined MLP ─► [scale, bias, gate]

        layer_config ──────────────────┘

        Output: base_noise * scale + bias, gated by sigmoid(gate)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        config_dim: int = 10,
        hidden_dims: list[int] = [256, 512, 256],
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.output_scale = output_scale
        self.latent_dim = latent_dim

        # Latent noise processing
        self.latent_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
        )

        # Layer config processing
        self.config_net = nn.Sequential(
            nn.Linear(config_dim, hidden_dims[0] // 2),
            nn.ReLU(),
        )

        # Combined processing
        combined_dim = hidden_dims[0] + hidden_dims[0] // 2
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
        )

        # Output heads
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )
        self.bias_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, layer_config: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        z_features = self.latent_net(z)
        config_features = self.config_net(layer_config)

        combined = torch.cat([z_features, config_features], dim=-1)
        features = self.combined_net(combined)

        # Generate output parameters
        scale = self.scale_head(features)
        bias = self.bias_head(features)
        gate = self.gate_head(features)

        # Compose output
        output = gate * (base * scale * self.output_scale) + bias * 0.01

        return output
