"""
Temporal Importance Pooling for KV Cache Attention.

Problem with current Sparse V (cache_store.py:386):
    sparse = torch.where(weights > threshold, weights, zeros)

    This hard-threshold causes:
    1. Abrupt quality cliff at threshold boundary
    2. Information loss for tokens slightly below threshold
    3. No temporal decay — old tokens treated same as recent
    4. No gradient information for optimization

Solution: Temporal Importance Pooling (TIP)

Instead of hard threshold, apply:
    1. Soft decay: recent tokens keep more weight, old tokens decay smoothly
    2. Importance-weighted pooling: merge low-importance token groups
    3. Smooth sparsification: exponential decay instead of step function

Formula:
    For attention weight w[i] at position i (0=oldest, s-1=newest):

    decay_factor[i] = exp(-λ · (s - 1 - i) / s)  # temporal decay

    importance_score[i] = w[i] · decay_factor[i]    # combined score

    soft_mask[i] = sigmoid((importance_score[i] - μ) / τ)  # smooth gating

    pooled_weight[i] = w[i] · soft_mask[i]

Where:
    λ = temporal decay rate (0 = no decay, ∞ = only last token)
    μ = mean importance (adaptive threshold)
    τ = temperature (0 = hard threshold, ∞ = uniform)

This replaces both:
    - Sparse V's hard threshold in cache_store.py
    - Compute Skip's block-level binary mask
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class TemporalPoolingConfig:
    """Configuration for Temporal Importance Pooling."""

    # Temporal decay rate λ
    # 0.0 = no decay (recent and old tokens equally important)
    # Higher values = stronger preference for recent tokens
    temporal_decay: float = 0.5

    # Smoothing temperature τ for soft gating
    # Lower = closer to hard threshold (more aggressive sparsification)
    # Higher = smoother, more gradual sparsification
    temperature: float = 0.1

    # Minimum attention mass to preserve (safety floor)
    # Ensures at least this fraction of total attention mass is retained
    min_attention_mass: float = 0.95

    # Whether to use adaptive threshold (based on mean importance)
    adaptive_threshold: bool = True

    # Fixed threshold (used when adaptive_threshold=False)
    fixed_threshold: float = 0.01

    # Whether to apply temporal decay
    enable_temporal_decay: bool = True

    # Pooling: merge consecutive low-importance tokens
    enable_pooling: bool = False

    # Pooling group size (only if enable_pooling=True)
    pool_group_size: int = 4


def _temporal_decay_mask(
    seq_len: int,
    device: torch.device,
    decay_rate: float = 0.5,
) -> torch.Tensor:
    """
    Generate temporal decay factors for sequence positions.

    Args:
        seq_len: Sequence length
        device: Torch device
        decay_rate: Temporal decay rate λ

    Returns:
        decay_factors: [seq_len] with values in (0, 1]
                       Position 0 (oldest) has smallest value,
                       Position seq_len-1 (newest) has value 1.0
    """
    if decay_rate <= 0:
        return torch.ones(seq_len, device=device)

    # positions: 0 (oldest) to seq_len-1 (newest)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    # Normalized distance from newest token
    distance = (seq_len - 1 - positions) / max(1, seq_len - 1)  # [0, 1]
    # Exponential decay
    decay = torch.exp(-decay_rate * distance * seq_len / 4.0)  # Scale by sequence length
    return decay


def temporal_importance_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: Optional[TemporalPoolingConfig] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Attention with temporal importance pooling instead of hard-threshold sparse V.

    This is a drop-in replacement for sparse_v_attention() and can also
    replace compute_skip_attention() with smoother behavior.

    Args:
        query: [batch, heads, q_len, d]
        key:   [batch, heads, k_len, d]
        value: [batch, heads, v_len, d]
        config: TemporalPoolingConfig
        attn_mask, dropout_p, is_causal, scale: standard attention args

    Returns:
        output: [batch, heads, q_len, d]
        report: Diagnostic dictionary
    """
    if config is None:
        config = TemporalPoolingConfig()
    d = query.size(-1)
    scale_factor = float(scale) if scale is not None else (1.0 / math.sqrt(d))

    # Standard attention logits
    logits = (query @ key.transpose(-2, -1)) * scale_factor

    # Apply causal mask
    if is_causal:
        s = logits.size(-1)
        causal = torch.ones(
            query.size(-2), s, dtype=torch.bool, device=query.device
        ).tril(diagonal=0)
        logits = logits.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Apply attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            logits = logits.masked_fill(~attn_mask, float("-inf"))
        else:
            logits = logits + attn_mask

    # Softmax to get attention weights
    weights = torch.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        weights = torch.dropout(weights, dropout_p, train=True)

    # ── Temporal Importance Pooling ──

    seq_len = weights.size(-1)
    report: Dict[str, Any] = {}

    if config.enable_temporal_decay:
        # Generate decay factors: [seq_len]
        decay = _temporal_decay_mask(
            seq_len, query.device, config.temporal_decay
        )
        # Apply decay: modulate weights by temporal position
        # decay shape: [seq_len] → broadcast to [1, 1, 1, seq_len]
        temporal_weights = weights * decay.view(1, 1, 1, -1)
        report["temporal_decay_applied"] = True
        report["avg_decay_factor"] = float(decay.mean())
    else:
        temporal_weights = weights

    # ── Soft gating instead of hard threshold ──
    if config.adaptive_threshold:
        # Adaptive: threshold based on mean importance per (batch, head, query)
        importance = temporal_weights
        mu = importance.mean(dim=-1, keepdim=True)  # [b, h, q, 1]
    else:
        mu = config.fixed_threshold

    # Soft mask: sigmoid((importance - μ) / τ)
    # τ controls sharpness: small τ → near-binary, large τ → soft
    soft_mask = torch.sigmoid((temporal_weights - mu) / max(config.temperature, 1e-6))

    # Apply soft mask
    masked_weights = weights * soft_mask

    # ── Ensure minimum attention mass ──
    if config.min_attention_mass < 1.0:
        total_mass = weights.sum(dim=-1, keepdim=True)
        masked_mass = masked_weights.sum(dim=-1, keepdim=True)
        mass_ratio = masked_mass / (total_mass + 1e-12)

        # If mass ratio is below threshold, blend back original weights
        deficit = (config.min_attention_mass - mass_ratio).clamp(min=0.0)
        # Blend: masked + deficit * original
        masked_weights = masked_weights + deficit * weights

    # Renormalize
    row_sums = masked_weights.sum(dim=-1, keepdim=True)
    masked_weights = masked_weights / (row_sums + 1e-12)

    # Compute output
    output = masked_weights @ value

    # ── Diagnostics ──
    sparsity = float((masked_weights < 1e-4).float().mean().item())
    effective_mass = float(masked_weights.sum(dim=-1).mean().item())

    report.update({
        "sparsity": sparsity,
        "effective_mass": effective_mass,
        "avg_soft_mask": float(soft_mask.mean().item()),
        "min_soft_mask": float(soft_mask.min().item()),
        "max_soft_mask": float(soft_mask.max().item()),
        "config_temporal_decay": config.temporal_decay,
        "config_temperature": config.temperature,
        "config_min_mass": config.min_attention_mass,
    })

    return output, report


def temporal_importance_attention_with_residual_proxy(
    query: torch.Tensor,
    packed_key: Any,  # ResidualQJLPackedTensor
    value: torch.Tensor,
    config: Optional[TemporalPoolingConfig] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Temporal importance attention with QJL residual proxy for packed keys.

    This is the TIP replacement for _attention_with_sparse_v_residual_proxy()
    in cache_store.py.

    Key difference: instead of hard-threshold sparse V, we use soft temporal
    importance gating on the attention weights.
    """
    if config is None:
        config = TemporalPoolingConfig()
    from .codec import approx_inner_product_with_qjl_residual

    d = query.size(-1)
    scale_factor = float(scale) if scale is not None else (1.0 / math.sqrt(d))

    # Approximate logits via QJL proxy
    logits = approx_inner_product_with_qjl_residual(query, packed_key) * scale_factor

    if is_causal:
        causal = torch.ones(
            query.size(-2), logits.size(-1), dtype=torch.bool, device=query.device
        ).tril(diagonal=0)
        logits = logits.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            logits = logits.masked_fill(~attn_mask, float("-inf"))
        else:
            logits = logits + attn_mask

    weights = torch.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        weights = torch.dropout(weights, dropout_p, train=True)

    # ── Temporal Importance Pooling ──
    seq_len = weights.size(-1)
    report: Dict[str, Any] = {}

    if config.enable_temporal_decay:
        decay = _temporal_decay_mask(seq_len, query.device, config.temporal_decay)
        temporal_weights = weights * decay.view(1, 1, 1, -1)
    else:
        temporal_weights = weights

    # Soft gating
    if config.adaptive_threshold:
        mu = temporal_weights.mean(dim=-1, keepdim=True)
    else:
        mu = config.fixed_threshold

    soft_mask = torch.sigmoid((temporal_weights - mu) / max(config.temperature, 1e-6))
    masked_weights = weights * soft_mask

    # Ensure minimum mass
    if config.min_attention_mass < 1.0:
        total_mass = weights.sum(dim=-1, keepdim=True)
        masked_mass = masked_weights.sum(dim=-1, keepdim=True)
        deficit = (config.min_attention_mass - masked_mass / (total_mass + 1e-12)).clamp(min=0.0)
        masked_weights = masked_weights + deficit * weights

    # Renormalize
    row_sums = masked_weights.sum(dim=-1, keepdim=True)
    masked_weights = masked_weights / (row_sums + 1e-12)

    output = masked_weights @ value

    sparsity = float((masked_weights < 1e-4).float().mean().item())
    report = {
        "sparsity": sparsity,
        "avg_soft_mask": float(soft_mask.mean().item()),
        "mode": "temporal_importance_with_qjl_proxy",
    }

    return output, report


# ─────────────────────────────────────────────────────────────
# Integration: bridge to KVCacheStore
# ─────────────────────────────────────────────────────────────

def create_temporal_pooling_config_from_preset(
    preset: str = "balanced",
) -> TemporalPoolingConfig:
    """
    Create TIP config from a preset name.

    Presets:
        "conservative": Minimal sparsification, high quality
        "balanced": Moderate sparsification (default)
        "aggressive": Strong sparsification, maximum memory savings
        "ultra_long": Optimized for very long contexts
    """
    if preset == "conservative":
        return TemporalPoolingConfig(
            temporal_decay=0.2,
            temperature=0.2,
            min_attention_mass=0.98,
            enable_temporal_decay=True,
        )
    if preset == "aggressive":
        return TemporalPoolingConfig(
            temporal_decay=1.5,
            temperature=0.05,
            min_attention_mass=0.90,
            enable_temporal_decay=True,
        )
    if preset == "ultra_long":
        return TemporalPoolingConfig(
            temporal_decay=0.8,
            temperature=0.08,
            min_attention_mass=0.92,
            enable_temporal_decay=True,
            enable_pooling=True,
            pool_group_size=4,
        )
    # balanced (default)
    return TemporalPoolingConfig(
        temporal_decay=0.5,
        temperature=0.1,
        min_attention_mass=0.95,
        enable_temporal_decay=True,
    )
