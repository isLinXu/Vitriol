"""Spectral Matching Loss."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class SpectralMatchingLoss(nn.Module):
    """Multi-component loss for spectral distribution matching."""

    def __init__(
        self,
        svd_rank: int = 32,
        num_probe_vectors: int = 8,
        svd_weight: float = 0.4,
        stats_weight: float = 0.25,
        norm_weight: float = 0.15,
        activation_weight: float = 0.2,
    ):
        super().__init__()
        self.svd_rank = svd_rank
        self.num_probe_vectors = num_probe_vectors
        self.svd_weight = svd_weight
        self.stats_weight = stats_weight
        self.norm_weight = norm_weight
        self.activation_weight = activation_weight

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute spectral matching loss.

        Args:
            generated: Generated weights [batch, ...shape]
            target: Reference (real) weights [batch, ...shape] or [...shape]

        Returns:
            (total_loss, loss_components_dict)
        """
        # Ensure both are 2D for SVD-based losses
        gen_flat = generated.flatten(1)   # [batch, d]
        tar_flat = target.flatten(1).to(generated.dtype).to(generated.device)

        # 1. SVD singular value matching (spectral fingerprint)
        svd_loss = self._svd_matching_loss(gen_flat, tar_flat)

        # 2. Per-dimension statistics matching
        stats_loss = self._stats_matching_loss(generated, target)

        # 3. Frobenius norm matching
        norm_loss = self._norm_matching_loss(generated, target)

        # 4. Activation response matching
        act_loss = self._activation_response_loss(generated, target)

        components = {}

        # Weighted combination
        total = (
            self.svd_weight * svd_loss +
            self.stats_weight * stats_loss +
            self.norm_weight * norm_loss +
            self.activation_weight * act_loss
        )
        components["total"] = total

        return total, components
