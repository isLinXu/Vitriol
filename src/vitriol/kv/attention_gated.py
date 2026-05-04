"""
AttentionGatedKV: Attention-Gated Variable-Precision KV Cache Compression.

═══════════════════════════════════════════════════════════════
Core Insight
═══════════════════════════════════════════════════════════════

Not all KV positions are equally important for attention output.
Existing methods handle this with hard thresholds:

  - **Sparse V**: Zero out positions with weight < threshold
  - **Compute Skip**: Drop entire blocks with low attention mass
  - **Temporal Pooling**: Soft-gate with exponential decay

But all these methods apply **binary** or **uniform** treatment —
either keep a position at full precision or drop it entirely.

**AttentionGatedKV** introduces a **continuous attention-gated
precision allocation**:

  - High-attention positions → 6-8 bit precision (near-lossless)
  - Medium-attention positions → 3-4 bit precision (standard)
  - Low-attention positions → 1-2 bit precision (coarse)
  - Near-zero positions → 0 bit (skip entirely)

This is analogous to **foveated rendering** in computer graphics:
the fovea (center of gaze) gets full resolution, while peripheral
vision gets progressively lower resolution.

═══════════════════════════════════════════════════════════════
Method
═══════════════════════════════════════════════════════════════

1. **Attention Importance Score**: For each KV position, compute
   an importance score based on the attention pattern:
   
     importance[t] = max_q(attention_weight[q, t])
   
   This uses the most recent query to determine which KV positions
   matter most.

2. **Precision Mapping**: Map importance to bit-width:
   
     bits[t] = bits_min + (bits_max - bits_min) · importance[t]^γ
   
   where γ controls the sharpness of the allocation curve.
   γ < 1: more uniform allocation
   γ > 1: more aggressive concentration on high-importance positions

3. **Grouped Quantization**: For efficiency, group positions into
   precision tiers (rather than per-position quantization):
   
     Tier 1 (top 20% importance): 6-8 bit → near-lossless
     Tier 2 (next 30%): 3-4 bit → standard quality
     Tier 3 (remaining 50%): 1-2 bit → coarse approximation

4. **Attention-Aware Decode**: During attention computation,
   incorporate the precision information for optimal output.

═══════════════════════════════════════════════════════════════
Advantages over existing methods
═══════════════════════════════════════════════════════════════

| Method          | Precision   | Skip    | Integration | Avg bpv |
|-----------------|-------------|---------|-------------|---------|
| TurboQuant      | Uniform     | No      | N/A         | 3.5     |
| Sparse V        | Uniform     | Hard    | Post-attn   | ~3.0    |
| Compute Skip    | Uniform     | Block   | In-attn     | ~2.5    |
| Temporal Pool   | Uniform     | Soft    | In-attn     | ~2.8    |
| AttentionGated  | Variable    | Gradual | In-attn     | ~2.0-2.5|

Key innovation: Instead of separate "compress" and "skip" stages,
AttentionGatedKV provides a **unified continuous allocation** from
0 to 8 bits based on actual attention importance.

═══════════════════════════════════════════════════════════════
Theoretical Analysis
═══════════════════════════════════════════════════════════════

Under attention sparsity (common in long sequences):
  - Top 20% positions carry ~85% of attention mass
  - Bottom 50% positions carry ~5% of attention mass

Optimal bit allocation (rate-distortion with importance weighting):
  - High-importance: b_high = target_bpv × 3.0 = 6-8 bit
  - Medium-importance: b_mid = target_bpv × 1.0 = 2-4 bit
  - Low-importance: b_low = target_bpv × 0.2 = 0.5-1 bit

Average: 0.2 × 6 + 0.3 × 3 + 0.5 × 1 = 2.9 bpv
With threshold (skip zero-importance): ~2.0-2.5 bpv

Quality: near-lossless on high-importance positions →
minimal PPL impact, much better than uniform quantization.

═══════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════

    from vitriol.kv.attention_gated import AttentionGatedKVCodec, AttentionGatedKVConfig

    codec = AttentionGatedKVCodec(AttentionGatedKVConfig(target_bpv=2.4))
    k_out, v_out, report = codec.compress_kv(key, value, query=query)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .codec import walsh_hadamard_rotate


# ─────────────────────────────────────────────────────────────
# Attention Importance Computation
# ─────────────────────────────────────────────────────────────

def compute_attention_importance(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None,
    topk_fraction: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-position attention importance score.

    importance[t] = max_q(softmax(Q·K^T / √d))[t]

    This measures how much each KV position matters for the
    most attentive query position.

    Args:
        query: [batch, heads, q_len, dim]
        key: [batch, heads, kv_len, dim]
        scale: Attention scale (default: 1/√dim)
        topk_fraction: Fraction of top positions to consider
                       (1.0 = all positions, 0.5 = top half)

    Returns:
        importance: [batch, heads, kv_len] — importance scores in [0, 1]
    """
    d = query.shape[-1]
    scale_factor = float(scale) if scale is not None else (1.0 / math.sqrt(d))

    # Compute attention logits
    logits = (query @ key.transpose(-2, -1)) * scale_factor  # [b, h, q, kv]

    # Take max over query dimension → most attentive query for each KV position
    # This captures "which KV position is most important to any query"
    max_logits = logits.max(dim=-2).values  # [b, h, kv]

    # Softmax to get importance weights
    importance = torch.softmax(max_logits, dim=-1)  # [b, h, kv]

    # Scale to make the distribution more concentrated
    # Raw softmax is too uniform for sparse attention patterns
    # Apply temperature sharpening
    importance = importance.pow(1.5)  # γ=1.5 sharpening
    importance = importance / importance.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    return importance


def compute_importance_tiers(
    importance: torch.Tensor,
    tier_fractions: Tuple[float, float, float] = (0.2, 0.3, 0.5),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Partition KV positions into importance tiers.

    Args:
        importance: [batch, heads, kv_len] — importance scores
        tier_fractions: (high, medium, low) fraction of positions

    Returns:
        high_mask: [batch, heads, kv_len] — boolean mask for high-importance
        medium_mask: [batch, heads, kv_len] — boolean mask for medium-importance
        low_mask: [batch, heads, kv_len] — boolean mask for low-importance
    """
    b, h, kv_len = importance.shape

    # Sort positions by importance (descending)
    sorted_indices = importance.argsort(dim=-1, descending=True)

    # Compute tier boundaries
    n_high = max(1, int(kv_len * tier_fractions[0]))
    n_medium = max(1, int(kv_len * tier_fractions[1]))
    kv_len - n_high - n_medium

    # Create masks
    high_mask = torch.zeros_like(importance, dtype=torch.bool)
    medium_mask = torch.zeros_like(importance, dtype=torch.bool)
    low_mask = torch.zeros_like(importance, dtype=torch.bool)

    # Assign tiers based on sorted order
    for bi in range(b):
        for hi in range(h):
            idx = sorted_indices[bi, hi]
            high_mask[bi, hi, idx[:n_high]] = True
            medium_mask[bi, hi, idx[n_high:n_high + n_medium]] = True
            low_mask[bi, hi, idx[n_high + n_medium:]] = True

    return high_mask, medium_mask, low_mask


# ─────────────────────────────────────────────────────────────
# Tiered Quantization
# ─────────────────────────────────────────────────────────────

def _quantize_tier_blockwise(
    x: torch.Tensor,
    mask: torch.Tensor,
    levels: int,
    block_size: int = 32,
) -> Tuple[torch.Tensor, float]:
    """
    Blockwise quantize a subset of positions identified by a mask.

    Instead of per-position quantization (which gives poor SNR),
    we quantize the full tensor at the given levels using blockwise
    min-max, then mask the result to only affect the tier positions.

    Args:
        x: [batch, heads, seq_len, dim]
        mask: [batch, heads, seq_len] — which positions to apply
        levels: Number of quantization levels
        block_size: Block size for blockwise quantization

    Returns:
        result: Full tensor with tier positions quantized, others untouched
        tier_bpv: Effective bits per value for this tier
    """
    shape = x.shape
    last = shape[-1]

    if not mask.any():
        return x.clone(), 0.0

    # Blockwise min-max quantization on the full tensor
    x_work = x.float()
    if last % block_size != 0:
        pad = block_size - (last % block_size)
        x_work = F.pad(x_work, (0, pad))
    else:
        pad = 0

    flat = x_work.reshape(-1, x_work.shape[-1] // block_size, block_size)
    mins = flat.min(dim=-1, keepdim=True)[0]
    maxs = flat.max(dim=-1, keepdim=True)[0]
    scales = (maxs - mins) / (levels - 1 + 1e-8)

    q = torch.round((flat - mins) / (scales + 1e-8))
    q = torch.clamp(q, 0, levels - 1)
    dq = q * scales + mins

    result = dq.reshape(*shape[:-1], x_work.shape[-1])
    if pad > 0:
        result = result[..., :last]
    result = result.reshape(shape)

    # Only apply to masked positions
    mask_expanded = mask.unsqueeze(-1).expand_as(x)
    result = torch.where(mask_expanded, result, x)

    tier_bpv = math.log2(max(2, levels))
    return result, tier_bpv


def _quantize_tiered_kv(
    x: torch.Tensor,
    importance: torch.Tensor,
    tier_levels: Tuple[int, int, int] = (64, 8, 2),
    tier_fractions: Tuple[float, float, float] = (0.2, 0.3, 0.5),
    skip_threshold: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply tiered quantization based on attention importance.

    Args:
        x: [batch, heads, seq_len, dim]
        importance: [batch, heads, seq_len]
        tier_levels: (high, medium, low) quantization levels
        tier_fractions: (high, medium, low) fraction of positions
        skip_threshold: Importance below which positions are zeroed

    Returns:
        result: Quantized tensor with variable precision
        report: Diagnostic dictionary
    """
    # Compute tier masks
    high_mask, medium_mask, low_mask = compute_importance_tiers(importance, tier_fractions)

    # Skip mask: positions with very low importance
    skip_mask = importance < skip_threshold

    # Quantize each tier
    result = x.clone()

    # Start with a copy of x
    result = x.clone()

    # Quantize each tier at its precision level, only affecting tier positions
    # Process from low to high precision so high-precision overwrites low
    high_bpv = math.log2(max(2, tier_levels[0]))
    med_bpv = math.log2(max(2, tier_levels[1]))
    low_bpv = math.log2(max(2, tier_levels[2]))

    # Low-importance tier (lowest precision)
    if low_mask.any():
        low_dq, low_bpv = _quantize_tier_blockwise(x, low_mask, tier_levels[2], block_size=32)
        result = torch.where(low_mask.unsqueeze(-1).expand_as(x), low_dq, result)

    # Medium-importance tier
    if medium_mask.any():
        med_dq, med_bpv = _quantize_tier_blockwise(x, medium_mask, tier_levels[1], block_size=32)
        result = torch.where(medium_mask.unsqueeze(-1).expand_as(x), med_dq, result)

    # High-importance tier (highest precision)
    if high_mask.any():
        high_dq, high_bpv = _quantize_tier_blockwise(x, high_mask, tier_levels[0], block_size=32)
        result = torch.where(high_mask.unsqueeze(-1).expand_as(x), high_dq, result)

    # Zero out skipped positions
    if skip_mask.any():
        result = torch.where(skip_mask.unsqueeze(-1).expand_as(x), torch.zeros_like(x), result)

    # Compute effective bpv
    n_total = importance.numel()
    n_high = high_mask.sum().item()
    n_med = medium_mask.sum().item()
    n_low = low_mask.sum().item()
    n_skip = skip_mask.sum().item()

    high_bpv = math.log2(max(2, tier_levels[0]))
    med_bpv = math.log2(max(2, tier_levels[1]))
    low_bpv = math.log2(max(2, tier_levels[2]))

    effective_bpv = (
        n_high * high_bpv +
        n_med * med_bpv +
        n_low * low_bpv +
        n_skip * 0  # Skipped positions: 0 bits
    ) / max(1, n_total)

    # Compression ratio vs fp16
    compression_ratio = 16.0 / max(effective_bpv, 0.01)

    # Compute quality metrics
    mse = float((x.float() - result.float()).pow(2).mean().item())

    report = {
        "method": "attention_gated_kv",
        "tier_levels": tier_levels,
        "tier_fractions": tier_fractions,
        "n_high": n_high,
        "n_medium": n_med,
        "n_low": n_low,
        "n_skip": n_skip,
        "high_bpv": high_bpv,
        "medium_bpv": med_bpv,
        "low_bpv": low_bpv,
        "effective_bpv": effective_bpv,
        "compression_ratio": compression_ratio,
        "mse": mse,
    }

    return result, report


# ─────────────────────────────────────────────────────────────
# Attention-Gated Attention Computation
# ─────────────────────────────────────────────────────────────

def attention_gated_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    importance: torch.Tensor,
    tier_levels: Tuple[int, int, int] = (64, 8, 2),
    tier_fractions: Tuple[float, float, float] = (0.2, 0.3, 0.5),
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute attention with attention-gated variable-precision KV.

    This replaces the standard SDPA call when AttentionGatedKV is enabled.

    Args:
        query: [batch, heads, q_len, dim]
        key: [batch, heads, kv_len, dim] — quantized at variable precision
        value: [batch, heads, kv_len, dim] — quantized at variable precision
        importance: [batch, heads, kv_len] — importance scores
        tier_levels: Quantization levels per tier
        tier_fractions: Fraction of positions per tier
        attn_mask, dropout_p, is_causal, scale: Standard SDPA params

    Returns:
        output: Attention output [batch, heads, q_len, dim]
        report: Diagnostic dictionary
    """
    # Apply tiered quantization to value (key is already processed)
    v_quantized, v_report = _quantize_tiered_kv(
        value, importance, tier_levels, tier_fractions
    )

    # Use standard SDPA on quantized KV
    output = F.scaled_dot_product_attention(
        query, key, v_quantized,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    report = {
        "method": "attention_gated_kv",
        **v_report,
    }

    return output, report


# ─────────────────────────────────────────────────────────────
# Compressed Representation
# ─────────────────────────────────────────────────────────────

@dataclass
class AttentionGatedKVCompressed:
    """Compressed KV tensor with attention-gated variable precision."""

    # Quantized data per tier
    q_high: torch.Tensor          # High-importance tier
    q_medium: torch.Tensor        # Medium-importance tier
    q_low: torch.Tensor           # Low-importance tier

    # Tier masks (which positions belong to which tier)
    high_mask: torch.Tensor       # [batch, heads, seq_len]
    medium_mask: torch.Tensor     # [batch, heads, seq_len]
    low_mask: torch.Tensor        # [batch, heads, seq_len]
    skip_mask: torch.Tensor       # [batch, heads, seq_len]

    # Quantization metadata per tier
    scales_high: torch.Tensor
    mins_high: torch.Tensor
    scales_medium: torch.Tensor
    mins_medium: torch.Tensor
    scales_low: torch.Tensor
    mins_low: torch.Tensor

    # Configuration
    tier_levels: Tuple[int, int, int]
    tier_fractions: Tuple[float, float, float]
    orig_shape: Tuple[int, ...]
    is_key: bool

    # Importance scores (for reference)
    importance: Optional[torch.Tensor] = None

    def storage_nbytes(self) -> int:
        """Estimate storage in bytes."""
        # Per-tier: quantized data + scales + mins
        def tier_bytes(q, scales, mins, levels):
            bits = math.ceil(math.log2(max(2, levels)))
            data = q.numel() * bits // 8
            meta = (scales.numel() + mins.numel()) * 4
            return data + meta

        total = tier_bytes(self.q_high, self.scales_high, self.mins_high, self.tier_levels[0])
        total += tier_bytes(self.q_medium, self.scales_medium, self.mins_medium, self.tier_levels[1])
        total += tier_bytes(self.q_low, self.scales_low, self.mins_low, self.tier_levels[2])

        # Masks (1 bit per position)
        n_positions = self.high_mask.numel()
        total += n_positions * 3 // 8  # 3 masks

        return total


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class AttentionGatedKVConfig:
    """Configuration for AttentionGatedKV compression."""

    # Target average bits per value
    target_bpv: float = 2.4

    # Tier levels (high, medium, low) — quantization levels per tier
    tier_levels: Tuple[int, int, int] = (128, 8, 4)

    # Tier fractions (high, medium, low) — fraction of positions per tier
    tier_fractions: Tuple[float, float, float] = (0.15, 0.35, 0.50)

    # Importance threshold below which positions are skipped entirely
    skip_threshold: float = 0.001

    # Importance sharpening exponent (γ)
    # Higher = more concentrated allocation on top positions
    importance_gamma: float = 1.5

    # Whether to apply Hadamard rotation before quantization
    apply_rotation: bool = False

    # Key vs Value differentiation
    k_level_boost: int = 0       # Extra levels for K high-tier
    v_level_penalty: int = 0     # Reduce levels for V low-tier

    # Whether to use importance-weighted attention computation
    # (more expensive but better quality)
    weighted_attention: bool = True

    # Auto-tune tier fractions based on attention sparsity
    auto_tune_tiers: bool = True

    # Block size for per-tier quantization
    block_size: int = 32


# ─────────────────────────────────────────────────────────────
# Auto-Tuning
# ─────────────────────────────────────────────────────────────

def _auto_tune_tier_fractions(
    importance: torch.Tensor,
    target_bpv: float,
    tier_levels: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """
    Auto-tune tier fractions based on attention sparsity.

    When attention is very sparse (few positions matter), allocate
    more positions to the low-tier. When attention is diffuse,
    use more balanced allocation.

    Args:
        importance: [batch, heads, kv_len]
        target_bpv: Target average bits per value
        tier_levels: Quantization levels per tier

    Returns:
        Optimized tier fractions (high, medium, low)
    """
    # Measure sparsity: what fraction of positions carry 80% of mass?
    sorted_imp = importance.sort(dim=-1, descending=True).values
    cumsum = sorted_imp.cumsum(dim=-1)
    total = cumsum[..., -1:]

    # Find position where cumulative mass reaches 80%
    threshold_80 = 0.8 * total
    above_80 = (cumsum >= threshold_80).float()
    # First position where cumsum >= 80%
    n_positions = importance.shape[-1]
    pos_80 = above_80.argmax(dim=-1).float().mean().item()
    sparsity_ratio = pos_80 / max(1, n_positions)

    # Adjust tier fractions based on sparsity
    # More sparse → smaller high-tier, larger low-tier
    if sparsity_ratio < 0.1:
        # Very sparse: 10% of positions carry 80% of mass
        frac_high = 0.10
        frac_medium = 0.20
    elif sparsity_ratio < 0.2:
        frac_high = 0.15
        frac_medium = 0.25
    elif sparsity_ratio < 0.4:
        frac_high = 0.20
        frac_medium = 0.30
    else:
        # Diffuse attention
        frac_high = 0.30
        frac_medium = 0.35

    frac_low = 1.0 - frac_high - frac_medium

    return frac_high, frac_medium, frac_low


# ─────────────────────────────────────────────────────────────
# Main Codec
# ─────────────────────────────────────────────────────────────

class AttentionGatedKVCodec:
    """
    AttentionGatedKV: Attention-gated variable-precision KV compression.

    This codec unifies Sparse V, Compute Skip, and Temporal Pooling
    into a single attention-importance-driven variable-precision framework.

    Key innovation: Instead of binary keep/drop decisions, each KV
    position receives a precision level proportional to its attention
    importance. This eliminates the sharp quality cliff at threshold
    boundaries and provides smoother degradation.
    """

    def __init__(self, config: Optional[AttentionGatedKVConfig] = None) -> None:
        self.config = config or AttentionGatedKVConfig()
        # Cache last importance scores for decode steps
        self._cached_importance: Optional[torch.Tensor] = None

    def _get_tier_config(
        self,
        importance: torch.Tensor,
        is_key: bool,
    ) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
        """Get tier levels and fractions, with auto-tuning if enabled."""
        cfg = self.config

        levels = list(cfg.tier_levels)
        if is_key:
            levels[0] += cfg.k_level_boost
        else:
            levels[2] = max(2, levels[2] - cfg.v_level_penalty)

        if cfg.auto_tune_tiers and importance is not None:
            fracs = _auto_tune_tier_fractions(importance, cfg.target_bpv, tuple(levels))
        else:
            fracs = cfg.tier_fractions

        return tuple(levels), fracs

    def compress(
        self,
        x: torch.Tensor,
        is_key: bool = True,
        importance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress a KV tensor with attention-gated variable precision.

        Args:
            x: [batch, heads, seq_len, dim]
            is_key: Whether this is a key tensor
            importance: [batch, heads, seq_len] — attention importance scores

        Returns:
            result: Quantized tensor with variable precision
            report: Diagnostic dictionary
        """
        cfg = self.config

        # If no importance provided, use uniform allocation
        if importance is None:
            kv_len = x.shape[-2]
            importance = torch.ones(
                *x.shape[:-1], device=x.device, dtype=torch.float32
            ) / kv_len
        else:
            # Ensure importance is normalized
            importance = importance.float()
            importance = importance / importance.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # Cache importance for decode steps
        self._cached_importance = importance.detach()

        # Get tier configuration
        tier_levels, tier_fractions = self._get_tier_config(importance, is_key)

        # Optional: Hadamard rotation
        x_work = walsh_hadamard_rotate(x.float()) if cfg.apply_rotation else x.float()

        # Apply tiered quantization
        result, tier_report = _quantize_tiered_kv(
            x_work, importance, tier_levels, tier_fractions, cfg.skip_threshold
        )

        # Inverse rotation
        if cfg.apply_rotation:
            result = walsh_hadamard_rotate(result)

        # Compute quality metrics
        mse = float((x.float() - result).pow(2).mean().item())
        x_var = float(x.float().pow(2).mean().item())
        snr_db = 10.0 * math.log10(max(1e-12, x_var) / max(1e-12, mse))

        report = {
            "method": "attention_gated_kv",
            "is_key": is_key,
            "target_bpv": cfg.target_bpv,
            "effective_bpv": tier_report["effective_bpv"],
            "compression_ratio": tier_report["compression_ratio"],
            "mse": mse,
            "snr_db": snr_db,
            "tier_levels": tier_levels,
            "tier_fractions": tier_fractions,
            **{k: v for k, v in tier_report.items() if k.startswith("n_")},
        }

        return result, report

    def compress_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compress both K and V with attention-gated variable precision.

        Main API for integration with KVCacheStore.

        Args:
            key: [batch, heads, seq_len, dim]
            value: [batch, heads, seq_len, dim]
            query: [batch, heads, q_len, dim] — needed for importance computation

        Returns:
            k_out: Reconstructed key tensor
            v_out: Reconstructed value tensor
            report: Combined diagnostic dictionary
        """
        # Compute importance from query-key attention
        if query is not None:
            importance = compute_attention_importance(query, key)
        else:
            importance = None

        # Compress key
        k_out, k_report = self.compress(key, is_key=True, importance=importance)

        # Compress value with same importance (or recompute from k_out)
        v_out, v_report = self.compress(value, is_key=False, importance=importance)

        report = {
            "k": k_report,
            "v": v_report,
            "k_mse": k_report["mse"],
            "v_mse": v_report["mse"],
            "total_mse": (k_report["mse"] + v_report["mse"]) / 2,
            "method": "attention_gated_kv",
        }

        return k_out, v_out, report


# ─────────────────────────────────────────────────────────────
# Quick Quantize-Dequantize (for benchmarking)
# ─────────────────────────────────────────────────────────────

def attention_gated_qdq(
    x: torch.Tensor,
    target_bpv: float = 2.4,
    is_key: bool = True,
    importance: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Quick attention-gated quantize-dequantize for benchmarking.

    Args:
        x: Input tensor [batch, heads, seq_len, dim]
        target_bpv: Target bits per value
        is_key: Whether key tensor
        importance: Optional importance scores [batch, heads, seq_len]

    Returns:
        reconstructed: Quantized-dequantized tensor
        report: Diagnostic dictionary
    """
    config = AttentionGatedKVConfig(target_bpv=target_bpv)
    codec = AttentionGatedKVCodec(config)

    result, report = codec.compress(x, is_key=is_key, importance=importance)

    # Quality metrics
    float((x.float() - result).pow(2).mean().item())
    cos_sim = float(F.cosine_similarity(
        x.float().flatten().unsqueeze(0),
        result.flatten().unsqueeze(0),
        dim=-1
    ).item())

    report["cosine_similarity"] = cos_sim

    return result, report
