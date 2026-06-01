"""ExoBrain configuration and adaptive layer selection."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ExoBrainConfig:
    """
    Configuration for the ExoBrain system (v0.4+ heterogenous reasoning).

    IMPORTANT: The shell model MUST have real, trainable weights.
    Zero-weight shells cannot generate meaningful queries for KV retrieval.
    """

    # Fusion mode: "replace", "residual", "gated"
    fusion_mode: str = "replace"

    # Alpha for residual fusion: ŷ = α·shell + (1-α)·brain
    residual_alpha: float = 0.1

    # Gate temperature for gated fusion
    gate_temperature: float = 1.0

    # v0.5: Gate computation mode for gated fusion:
    # - "max_similarity": Default, fast
    # - "mean_similarity": Smoother
    # - "per_head_entropy": Per-head attention entropy (v0.5)
    # - "learned": Learned projection (requires external module)
    gate_mode: str = "max_similarity"

    # Number of top-K external KV pairs to retrieve
    retrieval_top_k: int = 5

    # Whether to use cross-attention for KV injection
    use_cross_attention: bool = True

    # Dimension projection: if external KV has different dim than shell
    auto_project: bool = True

    # Key layers for KV injection (CognitiveAlignmentStrategy).
    # Only these layers receive external brain KV injection.
    # Middle semantic layers (3-8) and high-level reasoning (9-14) are typical.
    # Empty list = all layers (backward compatible).
    key_layers: List[int] = field(default_factory=list)

    # Alias for key_layers (backward compatibility)
    @property
    def active_layers(self) -> List[int]:
        """Alias for key_layers (deprecated, use key_layers)."""
        return self.key_layers

    @active_layers.setter
    def active_layers(self, value: List[int]) -> None:
        self.key_layers = value

    # Number of KV pairs to inject per key layer
    kv_injection_top_k: int = 5

    # Injection strength for residual/gated modes (0.0-1.0)
    injection_strength: float = 1.0

    # Whether to fall back to standard attention on brain failure
    fallback_on_error: bool = True

    # Confidence threshold: skip brain if query norm is below this
    min_query_norm: float = 1e-6

    # ── v0.5: Adaptive Layer Selection ──────────────────────────────
    # Strategy for selecting key layers:
    # - "manual": Use key_layers list (backward compatible, default)
    # - "entropy_top_k": Select top-K layers by attention entropy
    # - "entropy_threshold": Select layers above entropy threshold
    # - "middle_heavy": Prioritize middle layers (heuristic)
    # - "all": Inject all layers
    layer_selection_strategy: str = "manual"

    # Ratio of layers to select (for entropy_top_k strategy)
    layer_selection_top_k_ratio: float = 0.5

    # Entropy threshold (for entropy_threshold strategy)
    layer_selection_entropy_threshold: float = 0.7

    # Minimum number of layers to select
    layer_selection_min_layers: int = 4

    def __post_init__(self) -> None:
        valid_modes = {"replace", "residual", "gated"}
        if self.fusion_mode not in valid_modes:
            raise ValueError(
                f"ExoBrain: invalid fusion_mode '{self.fusion_mode}'. "
                f"Choose from: {valid_modes}"
            )

    def is_key_layer(self, layer_idx: int) -> bool:
        """
        Check if a layer is a key layer for injection.

        Returns True if:
        - key_layers is empty (all layers are key layers, backward compatible)
        - OR layer_idx is explicitly in key_layers
        - OR layer_selection_strategy is not "manual" and the adaptive selector includes it

        Note: For non-"manual" strategies, the caller should use
        AdaptiveLayerSelector.select() to determine key layers.
        """
        if self.layer_selection_strategy != "manual":
            # Adaptive mode: delegate to caller with selector
            # Fall back to all layers if no selector is used
            if not self.key_layers:
                return True  # All layers until selector is configured
        if not self.key_layers:
            return True  # All layers
        return layer_idx in self.key_layers


# ─────────────────────────────────────────────────────────────
# Adaptive Layer Selector (v0.5)
# ─────────────────────────────────────────────────────────────

class AdaptiveLayerSelector:
    """
    Selects key layers for ExoBrain KV injection based on attention entropy.

    Insight: Not all layers benefit equally from external brain injection.
    Layers with high attention entropy (diffuse, uncertain attention) benefit
    more from external guidance than layers with low entropy (confident, focused).

    Strategy:
    - Compute per-layer attention entropy from the shell model's forward pass
    - Rank layers by entropy (descending)
    - Select top-K layers or layers above a threshold
    - Cache the selection for reuse across prompts (stable selection)

    This replaces the old manual key_layers config with data-driven selection.

    Usage:
        selector = AdaptiveLayerSelector(
            total_layers=32,
            strategy="entropy_top_k",
            top_k_ratio=0.5,  # select top 50% layers
        )
        # After observing attention patterns:
        selector.observe(entropy_per_layer)
        key_layers = selector.select()
    """

    def __init__(
        self,
        total_layers: int = 0,
        strategy: str = "entropy_top_k",
        top_k_ratio: float = 0.5,
        entropy_threshold: float = 0.7,
        min_layers: int = 4,
        max_layers: Optional[int] = None,
        stability_window: int = 3,
    ) -> None:
        """
        Args:
            total_layers: Total number of transformer layers
            strategy: Selection strategy:
                - "entropy_top_k": Select top-K layers by entropy
                - "entropy_threshold": Select layers above entropy threshold
                - "middle_heavy": Prioritize middle layers (default heuristic)
                - "all": Select all layers (backward compatible)
            top_k_ratio: Fraction of layers to select (for entropy_top_k)
            entropy_threshold: Entropy threshold (for entropy_threshold strategy)
            min_layers: Minimum number of layers to select
            max_layers: Maximum number of layers to select (None = no limit)
            stability_window: Number of observations before selection stabilizes
        """
        self.total_layers = total_layers
        self.strategy = strategy
        self.top_k_ratio = top_k_ratio
        self.entropy_threshold = entropy_threshold
        self.min_layers = min_layers
        self.max_layers = max_layers or total_layers
        self.stability_window = stability_window

        # Entropy history: List[Dict[int, float]] — per-observation entropy
        self._entropy_history: List[Dict[int, float]] = []
        # Cached selection
        self._cached_selection: Optional[List[int]] = None
        # Per-layer statistics
        self._layer_stats: Dict[int, Dict[str, float]] = {}

    def observe(self, entropy_per_layer: Dict[int, float]) -> None:
        """
        Record attention entropy observation for each layer.

        Args:
            entropy_per_layer: {layer_idx: entropy_value}
        """
        self._entropy_history.append(dict(entropy_per_layer))
        self._cached_selection = None  # Invalidate cache

        # Update per-layer running stats
        for idx, ent in entropy_per_layer.items():
            if idx not in self._layer_stats:
                self._layer_stats[idx] = {"sum": 0.0, "count": 0, "max": 0.0}
            self._layer_stats[idx]["sum"] += ent
            self._layer_stats[idx]["count"] += 1
            self._layer_stats[idx]["max"] = max(self._layer_stats[idx]["max"], ent)

    def select(self) -> List[int]:
        """
        Select key layers for ExoBrain KV injection.

        Returns:
            List of layer indices (sorted ascending)
        """
        # Return cached selection if available
        if self._cached_selection is not None:
            return self._cached_selection

        if self.strategy == "all":
            return list(range(self.total_layers))

        if self.strategy == "middle_heavy":
            selection = self._select_middle_heavy()
        elif self.strategy == "entropy_threshold":
            selection = self._select_by_threshold()
        elif self.strategy == "entropy_top_k":
            selection = self._select_by_top_k()
        else:
            # Default: middle_heavy
            selection = self._select_middle_heavy()

        # Enforce min/max constraints
        if len(selection) < self.min_layers and self.total_layers > 0:
            # Add more layers (prefer middle layers)
            middle = list(range(self.total_layers // 4, 3 * self.total_layers // 4))
            for idx in middle:
                if idx not in selection:
                    selection.append(idx)
                if len(selection) >= self.min_layers:
                    break
        if len(selection) > self.max_layers:
            # Keep the top layers by average entropy
            avg_entropy = self._get_average_entropy()
            selection.sort(key=lambda idx: avg_entropy.get(idx, 0.0), reverse=True)
            selection = selection[:self.max_layers]

        selection = sorted(set(selection))
        self._cached_selection = selection
        return selection

    def _get_average_entropy(self) -> Dict[int, float]:
        """Compute average entropy per layer from observation history."""
        if not self._layer_stats:
            return {}
        return {
            idx: stats["sum"] / max(stats["count"], 1)
            for idx, stats in self._layer_stats.items()
        }

    def _select_by_top_k(self) -> List[int]:
        """Select top-K layers by average entropy."""
        avg_entropy = self._get_average_entropy()

        if not avg_entropy:
            return self._select_middle_heavy()

        k = max(self.min_layers, int(self.total_layers * self.top_k_ratio))
        k = min(k, self.max_layers)

        # Sort by entropy descending, take top-K
        sorted_layers = sorted(avg_entropy.keys(), key=lambda idx: avg_entropy[idx], reverse=True)
        return sorted(sorted_layers[:k])

    def _select_by_threshold(self) -> List[int]:
        """Select layers with average entropy above threshold."""
        avg_entropy = self._get_average_entropy()

        if not avg_entropy:
            return self._select_middle_heavy()

        selected = [idx for idx, ent in avg_entropy.items() if ent >= self.entropy_threshold]
        return sorted(selected)

    def _select_middle_heavy(self) -> List[int]:
        """
        Heuristic: prioritize middle layers.

        Based on the insight that:
        - Early layers (0-25%): lexical/syntax — shell's own
        - Middle layers (25-75%): semantic/concept — KEY LAYERS
        - Late layers (75-100%): output mapping — shell's own
        """
        if self.total_layers == 0:
            return []

        start = self.total_layers // 4
        end = 3 * self.total_layers // 4
        return list(range(start, end))

    def is_stable(self) -> bool:
        """Check if enough observations have been made for stable selection."""
        return len(self._entropy_history) >= self.stability_window

    @property
    def stats(self) -> Dict[str, Any]:
        """Return selection statistics."""
        avg_entropy = self._get_average_entropy()
        return {
            "strategy": self.strategy,
            "total_layers": self.total_layers,
            "observations": len(self._entropy_history),
            "selected_layers": self.select(),
            "num_selected": len(self.select()),
            "is_stable": self.is_stable(),
            "avg_entropy": {str(k): round(v, 4) for k, v in sorted(avg_entropy.items())},
        }


def compute_attention_entropy(
    attention_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute attention entropy per head per layer.

    Entropy = -Σ p·log(p), where p is the attention weight distribution.
    High entropy → diffuse/uncertain attention → benefits from external KV.
    Low entropy → focused/confident attention → shell's own is sufficient.

    Args:
        attention_weights: [batch, heads, q_len, kv_len] — softmaxed attention weights
        eps: Small value to avoid log(0)

    Returns:
        entropy: [batch, heads, q_len] — per-query-position entropy
    """
    log_weights = torch.log(attention_weights + eps)
    entropy = -torch.sum(attention_weights * log_weights, dim=-1)  # [B, H, Q]
    return entropy


# ─────────────────────────────────────────────────────────────
# Multi-Teacher Ensemble Router (v0.6)
# ─────────────────────────────────────────────────────────────
