"""
Compression Intelligence Score (CIS) - A Multi-dimensional Evaluation Framework.

This module implements the "Compression is Intelligence" theory as a quantitative
evaluation framework for weight generation strategies.

The Core Thesis:
    "Compression is Intelligence" - The ability to compress information while
    preserving essential patterns is fundamental to intelligence.

Mathematical Foundation:
    Ψ(S) = α·η_info + β·η_storage + γ·η_express + δ·T_train

Where:
    - η_info: Information Preservation Score (0-1)
    - η_storage: Storage Efficiency Score (0-1)
    - η_express: Expressive Power Score (0-1)
    - T_train: Trainability Score (0-1)
    - α, β, γ, δ: Weights summing to 1

Academic Value:
    - Proposes a new evaluation metric for model compression
    - Unifies multiple desiderata into a single score
    - Enables principled strategy comparison
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — tunable hyper-parameters for CIS scoring
# ---------------------------------------------------------------------------
_SVD_RANK_THRESHOLD_FACTOR: float = 1e-5  # relative to largest singular value
_CIS_DEFAULT_ALPHA: float = 0.3           # information retention weight


# ─────────────────────────────────────────────────────────────────────────────
# Score Components
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompressionScores:
    """Individual score components for compression evaluation."""
    # Information Preservation (0-1): How much information is retained
    info_preservation: float = 0.0

    # Storage Efficiency (0-1): Compression ratio achieved
    storage_efficiency: float = 0.0

    # Expressive Power (0-1): Diversity of generated values
    expressive_power: float = 0.0

    # Trainability (0-1): Fitness for gradient-based training
    trainability: float = 0.0

    # Phase Transition Indicator: Whether critical point is crossed
    phase_transition: bool = False

    def to_dict(self) -> Dict[str, float]:
        return {
            "info_preservation": self.info_preservation,
            "storage_efficiency": self.storage_efficiency,
            "expressive_power": self.expressive_power,
            "trainability": self.trainability,
            "phase_transition": float(self.phase_transition),
        }


@dataclass
class StrategyMetrics:
    """Complete metrics for a strategy."""
    strategy_name: str
    compression_ratio: float           # Actual compression ratio
    scores: CompressionScores
    psi_score: float                 # Weighted composite score
    layer_metrics: Dict[str, Dict[str, float]]  # Per-layer metrics
    radar_vector: List[float]        # For radar chart visualization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "compression_ratio": self.compression_ratio,
            "psi_score": self.psi_score,
            "scores": self.scores.to_dict(),
            "layer_metrics": self.layer_metrics,
            "radar_vector": self.radar_vector,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Information Preservation Metrics
# ─────────────────────────────────────────────────────────────────────────────

class InformationPreservationMetrics:
    """
    Measures how much information is preserved after compression.

    Methods:
    - Singular Value Decomposition (SVD) based
    - Mutual Information estimation
    - Spectral analysis
    """

    @staticmethod
    def svd_preservation_score(
        original_weights: torch.Tensor,
        compressed_weights: torch.Tensor,
    ) -> float:
        """
        Compute preservation score using SVD.

        Compares the singular value distributions to measure
        information retention.
        """
        try:
            # Compute SVD of both tensors
            orig_svd = torch.linalg.svd(original_weights, full_matrices=False)
            comp_svd = torch.linalg.svd(compressed_weights, full_matrices=False)

            orig_s = orig_svd.S
            comp_s = comp_svd.S

            # Normalize
            orig_s = orig_s / orig_s.sum()
            comp_s = comp_s[:len(orig_s)] / comp_s[:len(orig_s)].sum()

            # Pad if necessary
            if len(comp_s) < len(orig_s):
                comp_s = torch.cat([
                    comp_s,
                    torch.zeros(len(orig_s) - len(comp_s), device=comp_s.device)
                ])

            # Compute KL divergence (lower is better)
            kl_div = torch.nn.functional.kl_div(
                comp_s.log(),
                orig_s,
                reduction='batchmean'
            )

            # Convert to preservation score (0-1)
            score = torch.exp(-kl_div).item()
            return min(1.0, max(0.0, score))

        except Exception as e:
            logger.warning("SVD preservation score failed: %s", e)
            return 0.0

    @staticmethod
    def entropy_score(tensor: torch.Tensor) -> float:
        """Compute entropy of weight distribution."""
        try:
            # Flatten and compute histogram
            flat = tensor.flatten().cpu().numpy()
            hist, _ = np.histogram(flat, bins=50)

            # Normalize to probability distribution
            total = hist.sum()
            if total == 0:
                return 0.0
            probs = hist / total
            probs = probs[probs > 0]

            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return float(entropy)
        except Exception as e:
            logger.warning("Failed to compute entropy score: %s", e)
            return 0.0

    @staticmethod
    def spectrum_preservation(
        original: torch.Tensor,
        compressed: torch.Tensor,
    ) -> float:
        """Compare frequency spectrums."""
        try:
            # FFT of flattened weights
            orig_fft = torch.fft.fft(original.flatten())
            comp_fft = torch.fft.fft(compressed.flatten())

            # Power spectra
            orig_power = torch.abs(orig_fft) ** 2
            comp_power = torch.abs(comp_fft) ** 2

            # Normalize
            orig_power = orig_power / orig_power.sum()
            comp_power = comp_power / comp_power.sum()

            # Correlation
            correlation = torch.corrcoef(
                torch.stack([orig_power[:100], comp_power[:100]])
            )[0, 1].item()

            return max(0.0, correlation)
        except Exception as e:
            logger.warning("Failed to compute spectrum preservation: %s", e)
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Storage Efficiency Metrics
# ─────────────────────────────────────────────────────────────────────────────

class StorageEfficiencyMetrics:
    """
    Measures storage efficiency of compression strategies.

    This captures the core insight of "Compression is Intelligence":
    doing more with less.
    """

    @staticmethod
    def compression_ratio(
        original_size_bytes: int,
        compressed_size_bytes: int,
    ) -> float:
        """Calculate compression ratio (higher is better)."""
        if original_size_bytes == 0:
            return 0.0
        ratio = compressed_size_bytes / original_size_bytes
        return min(1.0, ratio)

    @staticmethod
    def storage_score_from_ratio(compression_ratio: float) -> float:
        """
        Convert compression ratio to 0-1 score.

        A compression ratio of 0.01 (99% compression) should score highly
        if information is preserved.
        """
        # Log scale: 0.01 -> 1.0, 0.1 -> 0.5, 1.0 -> 0.0
        if compression_ratio <= 0:
            return 0.0
        score = -np.log10(compression_ratio + 1e-10) / 2.0  # 0.01 -> 1.0
        return min(1.0, max(0.0, score))

    @staticmethod
    def sparsity_score(tensor: torch.Tensor) -> float:
        """Measure sparsity of tensor."""
        try:
            total = tensor.numel()
            zero = (tensor == 0).sum().item()
            return zero / total
        except Exception as e:
            logger.warning("Failed to compute sparsity score: %s", e)
            return 0.0

    @staticmethod
    def metadata_overhead(
        compressed_size: int,
        metadata_size: int,
    ) -> float:
        """
        Measure metadata overhead.

        High overhead reduces effective compression.
        """
        if compressed_size == 0:
            return 0.0
        overhead = metadata_size / compressed_size
        return max(0.0, 1.0 - overhead)


# ─────────────────────────────────────────────────────────────────────────────
# Expressive Power Metrics
# ─────────────────────────────────────────────────────────────────────────────

class ExpressivePowerMetrics:
    """
    Measures the expressive power of compressed weights.

    Key insight: Random/uniform weights have low expressiveness,
    while well-structured weights can capture complex patterns.
    """

    @staticmethod
    def value_diversity(tensor: torch.Tensor) -> float:
        """
        Measure diversity of unique values.

        Too few unique values = low expressiveness.
        """
        try:
            unique = torch.unique(tensor).numel()
            total = tensor.numel()
            return min(1.0, unique / total)
        except Exception as e:
            logger.warning("Failed to compute value diversity: %s", e)
            return 0.0

    @staticmethod
    def distribution_complexity(tensor: torch.Tensor) -> float:
        """
        Measure complexity of weight distribution.

        Uses multiple statistics to capture distribution shape.
        """
        try:
            flat = tensor.flatten()

            # Statistics
            mean = flat.mean().item()
            std = flat.std().item()
            skew = ((flat - mean) ** 3).mean().item() / (std ** 3 + 1e-10)
            kurtosis = ((flat - mean) ** 4).mean().item() / (std ** 4 + 1e-10) - 3

            # Non-zero statistics
            nonzero = flat[flat != 0]
            if len(nonzero) > 0:
                nonzero_ratio = len(nonzero) / len(flat)
                nonzero_std = nonzero.std().item()
            else:
                nonzero_ratio = 0.0
                nonzero_std = 0.0

            # Combine into complexity score
            complexity = (
                0.2 * nonzero_ratio +
                0.3 * min(1.0, abs(skew) / 3) +
                0.3 * min(1.0, abs(kurtosis) / 3) +
                0.2 * min(1.0, nonzero_std)
            )
            return complexity
        except Exception as e:
            logger.warning("Failed to compute distribution complexity: %s", e)
            return 0.0

    @staticmethod
    def rank_score(tensor: torch.Tensor) -> float:
        """
        Measure numerical rank relative to full rank.

        Higher rank = more expressive.
        """
        try:
            if tensor.dim() < 2:
                return 1.0

            # Approximate rank using SVD threshold
            s = torch.linalg.svd(tensor, full_matrices=False).S
            threshold = s[0] * _SVD_RANK_THRESHOLD_FACTOR
            rank = (s > threshold).sum().item()
            max_rank = min(tensor.shape)

            return rank / max_rank
        except Exception as e:
            logger.warning("Failed to compute rank score: %s", e)
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Trainability Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TrainabilityMetrics:
    """
    Measures fitness of compressed weights for gradient-based training.

    Key insight: Some compressions destroy gradient flow,
    making fine-tuning impossible.
    """

    @staticmethod
    def gradient_flow_score(
        tensor: torch.Tensor,
        signal_scale: float = 1e-3,
    ) -> float:
        """
        Estimate gradient flow potential.

        Uses singular value analysis of input-output Jacobian.
        """
        try:
            if tensor.dim() < 2:
                return 1.0

            # Approximate gradient flow via SVD
            s = torch.linalg.svd(tensor, full_matrices=False).S

            # Condition number
            if s[-1] > 0:
                cond = s[0] / s[-1]
            else:
                cond = float('inf')

            # Good condition number (< 100) allows gradient flow
            score = 1.0 / (1.0 + np.log10(cond + 1))
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning("Failed to compute gradient flow score: %s", e)
            return 0.5

    @staticmethod
    def signal_scale_score(tensor: torch.Tensor) -> float:
        """
        Measure if weight scale is appropriate for training.

        Too small -> vanishing signals
        Too large -> exploding signals
        """
        try:
            mean_abs = tensor.abs().mean().item()

            # Optimal range is around 0.01-0.1 for typical initialization
            if mean_abs < 1e-6:
                return mean_abs / 1e-6  # Very small
            elif mean_abs > 1.0:
                return 1.0 / mean_abs  # Very large
            else:
                # In good range
                return 1.0
        except Exception as e:
            logger.warning("Failed to compute signal scale score: %s", e)
            return 0.5

    @staticmethod
    def variance_preservation(
        original: torch.Tensor,
        compressed: torch.Tensor,
    ) -> float:
        """
        Compare variance to Xavier/He initialization theory.

        Good initialization should preserve variance across layers.
        """
        try:
            orig_var = original.var().item()
            comp_var = compressed.var().item()

            if orig_var == 0:
                return 1.0 if comp_var < 1e-6 else 0.0

            ratio = comp_var / orig_var

            # Ideal ratio is 1.0, penalize deviations
            if 0.5 <= ratio <= 2.0:
                return 1.0 - abs(np.log2(ratio)) / 4
            else:
                return max(0.0, 0.5 - abs(np.log2(ratio)) / 2)
        except Exception as e:
            logger.warning("Failed to compute variance preservation: %s", e)
            return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Critical Point Detection (Phase Transition)
# ─────────────────────────────────────────────────────────────────────────────

class CriticalPointDetector:
    """
    Detects the "Intelligence Critical Point" in compression.

    Core Thesis: At ~90% compression, there is a phase transition where
    the nature of intelligence changes from "pattern storage" to "pattern abstraction".

    This is measured by analyzing the derivative of expressivity with respect
    to compression ratio.
    """

    def __init__(self):
        self.expressivity_history: List[Tuple[float, float]] = []

    def add_observation(
        self,
        compression_ratio: float,
        expressivity_score: float,
    ) -> None:
        """Add observation for critical point detection."""
        self.expressivity_history.append((compression_ratio, expressivity_score))

    def detect_critical_point(self) -> Optional[float]:
        """
        Detect phase transition point.

        Returns:
            Critical compression ratio (e.g., 0.9 for 90%) or None
        """
        if len(self.expressivity_history) < 5:
            return None

        # Sort by compression ratio
        sorted_history = sorted(self.expressivity_history, key=lambda x: x[0])

        compressions = [h[0] for h in sorted_history]
        expressivities = [h[1] for h in sorted_history]

        # Compute second derivative (acceleration of expressivity loss)
        # High acceleration = critical point
        second_derivatives = []
        for i in range(2, len(expressivities)):
            d1 = expressivities[i-1] - expressivities[i-2]
            d2 = expressivities[i] - expressivities[i-1]
            dd = d2 - d1

            # Normalize by compression step
            dc = compressions[i] - compressions[i-1]
            if dc > 0:
                second_derivatives.append((compressions[i-1], dd / dc))

        if not second_derivatives:
            return None

        # Find maximum acceleration (critical point)
        critical_idx = np.argmax([abs(d[1]) for d in second_derivatives])
        critical_compression = second_derivatives[critical_idx][0]

        return critical_compression

    def is_above_critical_point(self, compression_ratio: float) -> bool:
        """Check if compression is above the critical point."""
        critical = self.detect_critical_point()
        if critical is None:
            return compression_ratio > 0.9  # Default: above 90% compression
        return compression_ratio > critical


# ─────────────────────────────────────────────────────────────────────────────
# Main Scoring Class
# ─────────────────────────────────────────────────────────────────────────────

class CompressionIntelligenceScorer:
    """
    Main class for computing Compression Intelligence Score (CIS).

    Usage:
        >>> scorer = CompressionIntelligenceScorer()
        >>> metrics = scorer.score_strategy(
        ...     strategy_name="ultra",
        ...     original_weights=original_tensor,
        ...     compressed_weights=compressed_tensor,
        ...     storage_size=1024,
        ... )
        >>> print(f"PSI Score: {metrics.psi_score:.3f}")
    """

    def __init__(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        alpha: float = 0.3,   # Information weight
        beta: float = 0.3,   # Storage weight
        gamma: float = 0.25, # Expressive power weight
        delta: float = 0.15, # Trainability weight
    ):
        """
        Initialize scorer.

        Args:
            weights: Optional dict of original weights for comparison
            alpha, beta, gamma, delta: Score component weights (must sum to 1)
        """
        self.weights = weights or {}
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Verify weights sum to 1
        total = alpha + beta + gamma + delta
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Score weights must sum to 1, got {total}")

        # Initialize metric calculators
        self.info_metrics = InformationPreservationMetrics()
        self.storage_metrics = StorageEfficiencyMetrics()
        self.expressive_metrics = ExpressivePowerMetrics()
        self.trainability_metrics = TrainabilityMetrics()
        self.critical_detector = CriticalPointDetector()

    def compute_info_preservation(
        self,
        original: Optional[torch.Tensor],
        compressed: torch.Tensor,
    ) -> float:
        """Compute information preservation score."""
        if original is None:
            # No reference, use entropy-based score (normalized to [0, 1])
            entropy = self.info_metrics.entropy_score(compressed)
            # Max entropy for 50 bins is ln(50) ~ 3.91 nats
            max_entropy = np.log(50)
            return min(1.0, max(0.0, entropy / max_entropy))

        # Use SVD preservation
        return self.info_metrics.svd_preservation_score(original, compressed)

    def compute_storage_efficiency(
        self,
        compressed_size_bytes: int,
        original_size_bytes: Optional[int] = None,
    ) -> float:
        """Compute storage efficiency score."""
        if original_size_bytes is None or original_size_bytes == 0:
            # Assume compression based on size
            return self.storage_metrics.storage_score_from_ratio(0.01)

        ratio = compressed_size_bytes / original_size_bytes
        return self.storage_metrics.storage_score_from_ratio(ratio)

    def compute_expressive_power(self, tensor: torch.Tensor) -> float:
        """Compute expressive power score."""
        diversity = self.expressive_metrics.value_diversity(tensor)
        complexity = self.expressive_metrics.distribution_complexity(tensor)
        rank = self.expressive_metrics.rank_score(tensor)

        return 0.4 * diversity + 0.3 * complexity + 0.3 * rank

    def compute_trainability(
        self,
        original: Optional[torch.Tensor],
        compressed: torch.Tensor,
    ) -> float:
        """Compute trainability score."""
        gradient_flow = self.trainability_metrics.gradient_flow_score(compressed)
        signal_scale = self.trainability_metrics.signal_scale_score(compressed)

        # Variance preservation if original available
        if original is not None:
            variance = self.trainability_metrics.variance_preservation(original, compressed)
        else:
            variance = 0.5

        return 0.4 * gradient_flow + 0.3 * signal_scale + 0.3 * variance

    def score_tensor(
        self,
        tensor: torch.Tensor,
        original: Optional[torch.Tensor] = None,
        compressed_size_bytes: Optional[int] = None,
        original_size_bytes: Optional[int] = None,
    ) -> Tuple[CompressionScores, float]:
        """
        Score a single tensor.

        Args:
            tensor: Compressed tensor
            original: Original tensor (optional)
            compressed_size_bytes: Actual storage size
            original_size_bytes: Original size in bytes

        Returns:
            Tuple of (CompressionScores, psi_score)
        """
        # Compute individual scores
        info_score = self.compute_info_preservation(original, tensor)
        storage_score = self.compute_storage_efficiency(
            compressed_size_bytes or tensor.numel() * 2,  # Assume bfloat16
            original_size_bytes or tensor.numel() * tensor.element_size(),
        )
        express_score = self.compute_expressive_power(tensor)
        train_score = self.compute_trainability(original, tensor)

        # Detect phase transition
        self.critical_detector.add_observation(
            compression_ratio=1.0 - storage_score,
            expressivity_score=express_score
        )
        self.critical_detector.detect_critical_point()
        above_critical = self.critical_detector.is_above_critical_point(
            1.0 - storage_score
        )

        # Create scores object
        scores = CompressionScores(
            info_preservation=info_score,
            storage_efficiency=storage_score,
            expressive_power=express_score,
            trainability=train_score,
            phase_transition=above_critical,
        )

        # Compute weighted PSI score
        psi = (
            self.alpha * info_score +
            self.beta * storage_score +
            self.gamma * express_score +
            self.delta * train_score
        )

        return scores, psi

    def score_strategy(
        self,
        strategy_name: str,
        weights: Dict[str, torch.Tensor],
        original_weights: Optional[Dict[str, torch.Tensor]] = None,
        compression_ratio: float = 0.01,
    ) -> StrategyMetrics:
        """
        Score an entire strategy based on its generated weights.

        Args:
            strategy_name: Name of the strategy
            weights: Generated weights (param_name -> tensor)
            original_weights: Original weights for comparison
            compression_ratio: Target compression ratio

        Returns:
            StrategyMetrics with detailed scores
        """
        layer_metrics = {}
        total_psi = 0.0
        total_info = 0.0
        total_storage = 0.0
        total_express = 0.0
        total_train = 0.0
        num_tensors = 0

        for name, tensor in weights.items():
            original = original_weights.get(name) if original_weights else None

            # Estimate compressed size (simplified)
            compressed_size = tensor.numel() * tensor.element_size()

            # Estimate original size
            if original is not None:
                original_size = original.numel() * original.element_size()
            else:
                # Assume based on compression ratio
                original_size = int(compressed_size / compression_ratio)

            scores, psi = self.score_tensor(
                tensor=tensor,
                original=original,
                compressed_size_bytes=compressed_size,
                original_size_bytes=original_size,
            )

            layer_metrics[name] = {
                "psi": psi,
                "info_preservation": scores.info_preservation,
                "storage_efficiency": scores.storage_efficiency,
                "expressive_power": scores.expressive_power,
                "trainability": scores.trainability,
            }

            total_psi += psi
            total_info += scores.info_preservation
            total_storage += scores.storage_efficiency
            total_express += scores.expressive_power
            total_train += scores.trainability
            num_tensors += 1

        # Average scores
        if num_tensors > 0:
            avg_scores = CompressionScores(
                info_preservation=total_info / num_tensors,
                storage_efficiency=total_storage / num_tensors,
                expressive_power=total_express / num_tensors,
                trainability=total_train / num_tensors,
                phase_transition=total_express / num_tensors > 0.5,
            )
            avg_psi = total_psi / num_tensors
        else:
            avg_scores = CompressionScores()
            avg_psi = 0.0

        # Radar vector for visualization [info, storage, express, train]
        radar = [
            avg_scores.info_preservation,
            avg_scores.storage_efficiency,
            avg_scores.expressive_power,
            avg_scores.trainability,
        ]

        return StrategyMetrics(
            strategy_name=strategy_name,
            compression_ratio=compression_ratio,
            scores=avg_scores,
            psi_score=avg_psi,
            layer_metrics=layer_metrics,
            radar_vector=radar,
        )

    def score_all_strategies(self) -> List[Tuple[str, float]]:
        """Return theoretical PSI scores for all strategies in the score matrix."""
        ranked = [
            (name, compute_theoretical_psi(name, self.alpha, self.beta, self.gamma, self.delta))
            for name in STRATEGY_SCORE_MATRIX
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    @staticmethod
    def compare_strategies(
        metrics_list: List[StrategyMetrics],
    ) -> List[Tuple[str, float, int]]:
        """
        Compare multiple strategies.

        Returns:
            List of (strategy_name, psi_score, rank) sorted by PSI
        """
        sorted_metrics = sorted(
            metrics_list,
            key=lambda m: m.psi_score,
            reverse=True
        )

        return [
            (m.strategy_name, m.psi_score, rank + 1)
            for rank, m in enumerate(sorted_metrics)
        ]

    @staticmethod
    def generate_report(metrics: StrategyMetrics) -> str:
        """Generate markdown report for strategy metrics."""
        scores = metrics.scores
        radar = metrics.radar_vector

        report = f"""# Compression Intelligence Report: {metrics.strategy_name}

## Overall Score
**PSI Score: {metrics.psi_score:.4f}**

## Score Breakdown

| Component | Score | Visualization |
|-----------|-------|---------------|
| Information Preservation (η_info) | {scores.info_preservation:.4f} | {'█' * int(scores.info_preservation * 20)}{'░' * (20 - int(scores.info_preservation * 20))} |
| Storage Efficiency (η_storage) | {scores.storage_efficiency:.4f} | {'█' * int(scores.storage_efficiency * 20)}{'░' * (20 - int(scores.storage_efficiency * 20))} |
| Expressive Power (η_express) | {scores.expressive_power:.4f} | {'█' * int(scores.expressive_power * 20)}{'░' * (20 - int(scores.expressive_power * 20))} |
| Trainability (T_train) | {scores.trainability:.4f} | {'█' * int(scores.trainability * 20)}{'░' * (20 - int(scores.trainability * 20))} |

## Compression Ratio: {metrics.compression_ratio:.4f}

## Phase Transition
**Above Critical Point:** {'✓ Yes (Abstraction Mode)' if scores.phase_transition else '✗ No (Storage Mode)'}

## Radar Chart Vector
```
Info: {radar[0]:.2f}, Storage: {radar[1]:.2f}, Express: {radar[2]:.2f}, Train: {radar[3]:.2f}
```

## Layer Metrics ({len(metrics.layer_metrics)} layers)

| Layer | PSI | Info | Storage | Express | Train |
|-------|-----|------|---------|---------|-------|
"""
        for name, m in list(metrics.layer_metrics.items())[:10]:
            report += f"| {name[:30]} | {m['psi']:.3f} | {m['info_preservation']:.3f} | {m['storage_efficiency']:.3f} | {m['expressive_power']:.3f} | {m['trainability']:.3f} |\n"

        if len(metrics.layer_metrics) > 10:
            report += f"\n... and {len(metrics.layer_metrics) - 10} more layers\n"

        return report


# ─────────────────────────────────────────────────────────────────────────────
# Score Matrix for All Strategies
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_SCORE_MATRIX = {
    # Name: (info_preservation, storage_efficiency, expressive_power, trainability)
    # Based on theoretical analysis
    "random":        (0.90, 0.00, 0.95, 0.95),
    "compact":       (0.30, 0.95, 0.05, 0.40),
    "ultra":         (0.10, 1.00, 0.02, 0.10),
    "sparse":        (0.50, 0.80, 0.40, 0.60),
    "binary":        (0.30, 0.97, 0.15, 0.35),
    "ternary":       (0.40, 0.94, 0.25, 0.45),
    "quantized":     (0.70, 0.75, 0.60, 0.70),
    "lowrank":       (0.75, 0.70, 0.65, 0.75),
    "structured_sparse": (0.55, 0.80, 0.45, 0.65),
    "learned":       (0.85, 0.85, 0.80, 0.85),
    "hybrid_ultra":  (0.15, 0.98, 0.10, 0.20),
    "hybrid_learned": (0.82, 0.82, 0.78, 0.82),
    "quantum":       (0.45, 0.85, 0.50, 0.30),
}


def compute_theoretical_psi(
    strategy_name: str,
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.25,
    delta: float = 0.15,
) -> float:
    """
    Compute theoretical PSI score for a strategy.

    Uses pre-computed score matrix for quick estimation.
    """
    if strategy_name not in STRATEGY_SCORE_MATRIX:
        logger.warning("Unknown strategy: %s", strategy_name)
        return 0.0

    scores = STRATEGY_SCORE_MATRIX[strategy_name]

    psi = (
        alpha * scores[0] +
        beta * scores[1] +
        gamma * scores[2] +
        delta * scores[3]
    )

    return psi


def generate_score_comparison_table() -> str:
    """Generate markdown comparison table for all strategies."""
    header = "| Strategy | η_info | η_storage | η_express | T_train | PSI Score |"
    separator = "|----------|--------|-----------|-----------|---------|-----------|"

    rows = [header, separator]

    strategies_psi = []
    for name, scores in STRATEGY_SCORE_MATRIX.items():
        psi = compute_theoretical_psi(name)
        strategies_psi.append((name, scores, psi))

    # Sort by PSI
    strategies_psi.sort(key=lambda x: x[2], reverse=True)

    for name, scores, psi in strategies_psi:
        rows.append(
            f"| {name:10} | {scores[0]:.2f} | {scores[1]:.2f} | {scores[2]:.2f} | {scores[3]:.2f} | {psi:.4f} |"
        )

    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "CompressionScores",
    "StrategyMetrics",
    "CompressionIntelligenceScorer",
    "CriticalPointDetector",
    "STRATEGY_SCORE_MATRIX",
    "compute_theoretical_psi",
    "generate_score_comparison_table",
    "InformationPreservationMetrics",
    "StorageEfficiencyMetrics",
    "ExpressivePowerMetrics",
    "TrainabilityMetrics",
]
