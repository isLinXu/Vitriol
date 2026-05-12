"""
Layer-Aware Adaptive Bit Allocation for KV Cache Compression.

This module addresses a key limitation of the existing policy system:
    - Turbo3ExactKApproxVPolicy only distinguishes "quantize / don't quantize" per layer
    - Within a quantized layer, ALL heads use the SAME bit-width
    - No layer-depth consideration (early vs late layers have different sensitivity)

Key Innovation:
    Layer depth + attention entropy → per-(layer, head) bit allocation

Theory:
    - Early layers: capture syntax/positional info → high sensitivity → need more bits
    - Middle layers: semantic features → moderate sensitivity
    - Late layers: task-specific features → often redundant → can use fewer bits
    - Attention entropy per head: collapsed (low entropy) heads need fewer bits,
      superposition (high entropy) heads need more bits

Usage:
    from vitriol.kv.layer_adaptive import LayerAdaptiveBitAllocator, LayerAdaptiveConfig

    allocator = LayerAdaptiveBitAllocator(LayerAdaptiveConfig(target_avg_bits=3.0))
    k_bits, v_bits, report = allocator.allocate(
        query=q, key=k, value=v,
        layer_idx=12, total_layers=32,
        layer_type="full_attention",
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────
# Layer depth sensitivity profile
# ─────────────────────────────────────────────────────────────

def _layer_depth_weight(
    layer_idx: int,
    total_layers: int,
    profile: str = "u_shape",
) -> float:
    """
    Compute depth-dependent sensitivity weight for a layer.

    Profiles:
        "u_shape":   Early & late layers sensitive, middle less so  (default, matches empirical findings)
        "decay":     Sensitivity decreases monotonically with depth  (early layers most important)
        "inv_decay": Sensitivity increases with depth               (late layers most important)
        "uniform":   All layers equally sensitive                   (baseline)

    Returns:
        Weight in [0, 1] where 1.0 = most sensitive = needs most bits
    """
    if total_layers <= 1:
        return 1.0

    t = layer_idx / max(1, total_layers - 1)  # Normalized depth [0, 1]

    if profile == "u_shape":
        # U-shape: minimum at t=0.5, maximum at t=0 and t=1
        # w = 1.0 - 0.4 * (1 - (2t - 1)^2)  → range [0.6, 1.0]
        return 1.0 - 0.4 * (1.0 - (2.0 * t - 1.0) ** 2)

    if profile == "decay":
        # Exponential decay: early layers matter most
        return math.exp(-2.0 * t)

    if profile == "inv_decay":
        # Inverse: late layers matter most
        return math.exp(-2.0 * (1.0 - t))

    # uniform
    return 1.0


def _layer_depth_weights(
    total_layers: int,
    profile: str = "u_shape",
) -> List[float]:
    """Compute depth weights for all layers."""
    return [_layer_depth_weight(i, total_layers, profile) for i in range(total_layers)]


# ─────────────────────────────────────────────────────────────
# Per-head attention entropy computation (optimized)
# ─────────────────────────────────────────────────────────────

def _compute_head_entropy(
    query: torch.Tensor,
    key: torch.Tensor,
    num_sample_positions: int = 0,
) -> torch.Tensor:
    """
    Compute per-head normalized attention entropy.

    Args:
        query: [batch, heads, q_len, d]
        key:   [batch, heads, k_len, d]
        num_sample_positions: If > 0, subsample query positions for speed

    Returns:
        head_entropy: [batch, heads] ∈ [0, 1]
    """
    d = query.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # Optional subsampling for long sequences
    if num_sample_positions > 0 and query.shape[-2] > num_sample_positions:
        indices = torch.randint(
            0,
            query.shape[-2],
            (num_sample_positions,),
            device=query.device,
        )
        q_sample = query[:, :, indices, :]
    else:
        q_sample = query

    logits = (q_sample @ key.transpose(-2, -1)) * scale  # [b, h, q', k]
    w = torch.softmax(logits, dim=-1)

    # Entropy: H = -Σ p·log(p)
    eps = 1e-12
    w_clamp = w.clamp(min=eps)
    entropy = -(w_clamp * torch.log(w_clamp)).sum(dim=-1)  # [b, h, q']

    # Normalize by max entropy
    max_ent = math.log(max(1, key.shape[-2]))
    normalized = entropy / (max_ent + 1e-12)

    # Average over query positions
    return normalized.mean(dim=-1)  # [b, h]


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class LayerAdaptiveConfig:
    """Configuration for layer-aware adaptive bit allocation."""

    # Target average bits per value (across all layers & heads)
    target_avg_bits: float = 3.0

    # Bit range
    min_bits: float = 1.5
    max_bits: float = 5.0

    # K/V bit budget ratio
    k_share: float = 0.65

    # Depth profile: "u_shape", "decay", "inv_decay", "uniform"
    depth_profile: str = "u_shape"

    # Entropy non-linearity exponent
    # Higher alpha → more aggressive compression on low-entropy heads
    entropy_alpha: float = 1.2

    # Depth weight strength [0, 1]
    # 0 = ignore depth, 1 = full depth modulation
    depth_strength: float = 0.3

    # Entropy weight strength [0, 1]
    # 0 = ignore entropy, 1 = full entropy modulation
    entropy_strength: float = 0.7

    # Subsample query positions for entropy (0 = no subsampling)
    entropy_subsample_positions: int = 64

    # Per-layer-type defaults
    # MLA layers: typically smaller, can use fewer bits
    mla_bit_penalty: float = 0.5
    # Sliding window: limited context, can use fewer bits
    sliding_window_bit_penalty: float = 0.3

    # Value RMS importance weight (0-1)
    v_rms_weight: float = 0.3


# ─────────────────────────────────────────────────────────────
# Main allocator
# ─────────────────────────────────────────────────────────────

class LayerAdaptiveBitAllocator:
    """
    Layer-aware adaptive bit allocation for KV cache.

    Combines three signals to allocate bits per (layer, head):
        1. Layer depth weight: position-dependent sensitivity
        2. Attention entropy: head-dependent uncertainty
        3. Value RMS: magnitude-dependent importance

    This is fundamentally more fine-grained than:
        - TurboQuant's per-head-only allocation
        - AdaptiveKVCodec's per-head-only allocation
        - Turbo3ExactKApproxVPolicy's binary quantize/don't-quantize
    """

    def __init__(self, config: Optional[LayerAdaptiveConfig] = None) -> None:
        self.config = config or LayerAdaptiveConfig()
        self._depth_weight_cache: Dict[int, List[float]] = {}

    def _get_depth_weight(self, layer_idx: int, total_layers: int) -> float:
        """Get cached depth weight for a layer."""
        if total_layers not in self._depth_weight_cache:
            self._depth_weight_cache[total_layers] = _layer_depth_weights(
                total_layers, self.config.depth_profile
            )
        weights = self._depth_weight_cache[total_layers]
        if 0 <= layer_idx < len(weights):
            return weights[layer_idx]
        return 1.0

    def allocate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        total_layers: int,
        layer_type: str = "full_attention",
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Allocate bits per (batch, head) considering layer depth.

        Args:
            query: [batch, heads, q_len, d] or [batch*heads, q_len, d]
            key:   [batch, heads, k_len, d] or [batch*heads, k_len, d]
            value: [batch, heads, v_len, d] or [batch*heads, v_len, d]
            layer_idx: Current layer index
            total_layers: Total number of layers
            layer_type: "full_attention", "mla", "sliding_window", etc.

        Returns:
            k_bits: [batch, heads] per-head K bit allocation
            v_bits: [batch, heads] per-head V bit allocation
            report: Diagnostic dictionary
        """
        cfg = self.config

        # Handle flat input (from TurboQuant flow)
        if query.ndim == 3:
            # Assume [batch*heads, seq, d] — no per-head allocation possible
            # Use uniform allocation with depth modulation
            depth_w = self._get_depth_weight(layer_idx, total_layers)
            base_bits = cfg.target_avg_bits
            depth_modulated = base_bits * (0.7 + 0.3 * depth_w)
            layer_penalty = self._layer_type_penalty(layer_type)
            uniform_bits = max(cfg.min_bits, min(cfg.max_bits, depth_modulated - layer_penalty))

            # Simple K/V split
            k_bits_val = uniform_bits * (2.0 * cfg.k_share)
            v_bits_val = uniform_bits * (2.0 * (1.0 - cfg.k_share))

            report = {
                "mode": "depth_only_flat",
                "layer_idx": layer_idx,
                "total_layers": total_layers,
                "depth_weight": depth_w,
                "layer_type": layer_type,
                "avg_k_bits": k_bits_val,
                "avg_v_bits": v_bits_val,
            }
            # Return scalar-like tensors for compatibility
            b = query.shape[0]
            k_bits = torch.full((b,), k_bits_val, device=query.device, dtype=torch.float32)
            v_bits = torch.full((b,), v_bits_val, device=query.device, dtype=torch.float32)
            return k_bits, v_bits, report

        b, h = key.shape[:2]

        # ── Signal 1: Layer depth weight ──
        depth_w = self._get_depth_weight(layer_idx, total_layers)

        # ── Signal 2: Attention entropy ──
        head_entropy = _compute_head_entropy(
            query, key,
            num_sample_positions=cfg.entropy_subsample_positions,
        )  # [b, h]

        # ── Signal 3: Value RMS importance ──
        v_rms = torch.sqrt((value * value).mean(dim=(-2, -1)))  # [b, h]
        v_rms_norm = v_rms / (v_rms.mean(dim=-1, keepdim=True) + 1e-12)
        v_importance = torch.clamp(v_rms_norm, 0.3, 1.0)

        # ── Combine signals ──
        # Base allocation from entropy (same as adaptive_kv_bits)
        importance = torch.clamp(1.0 - head_entropy, 0.0, 1.0)

        # Depth modulation: high sensitivity → shift bits upward
        depth_boost = cfg.depth_strength * (depth_w - 0.5)  # Center around 0
        entropy_boost = cfg.entropy_strength * (importance - 0.5)

        # Layer type penalty
        layer_penalty = self._layer_type_penalty(layer_type)

        # K bits: entropy-driven with depth modulation
        raw_k_bits = cfg.target_avg_bits + depth_boost + entropy_boost - layer_penalty
        raw_k_bits = raw_k_bits + (cfg.max_bits - cfg.min_bits) * 0.1 * (head_entropy ** cfg.entropy_alpha - 0.5)

        # V bits: value-RMS-driven with depth modulation
        raw_v_bits = cfg.target_avg_bits + depth_boost + cfg.entropy_strength * (v_importance - 0.5) * 0.5 - layer_penalty

        # Apply global scaling to hit target
        current_avg = raw_k_bits.mean() * cfg.k_share + raw_v_bits.mean() * (1.0 - cfg.k_share)
        if current_avg > 0:
            scale = cfg.target_avg_bits / float(current_avg)
            scale = max(0.5, min(2.0, scale))
            k_bits = torch.clamp(raw_k_bits * scale, cfg.min_bits, cfg.max_bits)
            v_bits = torch.clamp(raw_v_bits * scale, cfg.min_bits, cfg.max_bits)
        else:
            k_bits = raw_k_bits.clamp(cfg.min_bits, cfg.max_bits)
            v_bits = raw_v_bits.clamp(cfg.min_bits, cfg.max_bits)

        # ── Report ──
        actual_avg = float(
            (k_bits.mean() * cfg.k_share + v_bits.mean() * (1.0 - cfg.k_share)).item()
        )

        report = {
            "mode": "layer_adaptive",
            "layer_idx": layer_idx,
            "total_layers": total_layers,
            "depth_weight": float(depth_w),
            "layer_type": layer_type,
            "layer_penalty": float(layer_penalty),
            "avg_k_bits": float(k_bits.mean()),
            "avg_v_bits": float(v_bits.mean()),
            "actual_avg_bits": actual_avg,
            "target_avg_bits": cfg.target_avg_bits,
            "k_min": float(k_bits.min()),
            "k_max": float(k_bits.max()),
            "v_min": float(v_bits.min()),
            "v_max": float(v_bits.max()),
            "mean_entropy": float(head_entropy.mean()),
            "collapsed_heads": float((head_entropy < 0.3).float().mean().item()),
            "superposition_heads": float((head_entropy > 0.7).float().mean().item()),
        }

        return k_bits, v_bits, report

    def _layer_type_penalty(self, layer_type: str) -> float:
        """Bit penalty for non-full-attention layers."""
        normalized = str(layer_type or "").lower().replace("-", "_")
        if "mla" in normalized or "latent" in normalized:
            return self.config.mla_bit_penalty
        if "sliding" in normalized or "local_attention" in normalized:
            return self.config.sliding_window_bit_penalty
        if any(token in normalized for token in ("compressed", "hash", "csa", "hca", "linear", "mamba", "ssm", "rwkv", "retnet", "hyena")):
            return max(self.config.mla_bit_penalty, self.config.sliding_window_bit_penalty)
        return 0.0

    def allocate_all_layers(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        total_layers: int,
        layer_types: Optional[List[str]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Convenience: allocate bits for all layers at once.

        Returns list of (k_bits, v_bits, report) per layer.
        Typically called once during prefill to set per-layer bit budgets.
        """
        results = []
        for idx in range(total_layers):
            lt = (layer_types[idx] if layer_types and idx < len(layer_types) else "full_attention")
            k_bits, v_bits, report = self.allocate(
                query=query, key=key, value=value,
                layer_idx=idx, total_layers=total_layers,
                layer_type=lt,
            )
            results.append((k_bits, v_bits, report))
        return results


# ─────────────────────────────────────────────────────────────
# Integration helper: bridge to existing KVCacheStoreConfig
# ─────────────────────────────────────────────────────────────

def apply_layer_adaptive_to_config(
    base_cfg: Any,  # KVCacheStoreConfig
    allocator: LayerAdaptiveBitAllocator,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_idx: int,
    total_layers: int,
    layer_type: str = "full_attention",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a layer-specific KVCacheStoreConfig using adaptive bit allocation.

    Instead of creating a new config class, we reuse the existing
    enable_adaptive_bits + AdaptiveKVCodec mechanism, but pre-compute
    the per-head levels from LayerAdaptiveBitAllocator.

    Returns:
        (modified_config, report)
    """
    from .codec import AdaptiveKVCodec

    k_bits, v_bits, report = allocator.allocate(
        query=query, key=key, value=value,
        layer_idx=layer_idx, total_layers=total_layers,
        layer_type=layer_type,
    )

    # Create AdaptiveKVCodec with the computed levels
    avg_k = float(k_bits.mean())
    avg_v = float(v_bits.mean())
    avg_total = avg_k * allocator.config.k_share + avg_v * (1.0 - allocator.config.k_share)

    codec = AdaptiveKVCodec(
        block_size=base_cfg.adaptive_bits.block_size if hasattr(base_cfg, 'adaptive_bits') and base_cfg.adaptive_bits else 32,
        min_bits=float(k_bits.min()),
        max_bits=float(k_bits.max()),
        target_avg_bits=avg_total,
        k_share=allocator.config.k_share,
    )

    from dataclasses import replace
    new_cfg = replace(
        base_cfg,
        enable_adaptive_bits=True,
        adaptive_bits=codec,
        # Also set turbo format based on average bits
        turbo_k_format=_bits_to_turbo_format(avg_k),
        turbo_v_format=_bits_to_turbo_format(avg_v),
    )

    return new_cfg, report


def _bits_to_turbo_format(avg_bits: float) -> str:
    """Map average bits to closest turbo format."""
    if avg_bits <= 2.75:
        return "turbo2"
    if avg_bits <= 3.75:
        return "turbo3"
    return "turbo4"
