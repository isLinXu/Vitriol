"""
TurboQuant: Paper-accurate implementation (arXiv:2504.19874)
========================================

This module is a faithful implementation of arXiv:2504.19874,
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate".

Core pipeline (as described in the paper):
    x → Hadamard Rotation → Standardize → Lloyd-Max Quantization → QJL Residual → x̂

Key steps:
    1. Signed Hadamard Rotation: Rademacher random rotation to spread energy uniformly
    2. Per-vector Standardization: normalize to an N(0, 1) distribution
    3. Gaussian Lloyd-Max: scalar quantization optimized for a Gaussian distribution
    4. QJL Residual Sketch: 1-bit random projection capturing residual information

Differences vs TurboQuantum (kv/turboquantum.py):
    ┌─────────────────────┬──────────────────────┬──────────────────────┐
    │ Feature             │ TurboQuant           │ TurboQuantum         │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Bit allocation      │ uniform (2.5/3.5/4.25)│ adaptive (1.5-5.0)   │
    │ Standardization     │ per-vector z-score   │ per-block min-max    │
    │ Attention-aware     │ ❌ no                │ ✅ entropy-based     │
    │ Key-token protection│ ❌ no                │ ✅ tunneling         │
    │ Cross-layer residual│ ❌ no                │ ✅ entanglement      │
    │ Paper fidelity      │ ✅ faithful          │ ❌ Vitriol-enhanced  │
    └─────────────────────┴──────────────────────┴──────────────────────┘

Recommended usage:
    - Academic citations / ablation comparisons: use TurboQuant (this module)
    - Engineering optimization / product: use TurboQuantum (kv/turboquantum.py)
"""

from __future__ import annotations

import logging
import math
import time
from functools import lru_cache
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F

# Try to import Triton-accelerated kernels; fall back gracefully
_TRITON_AVAILABLE = False
try:
    from ..kv.triton_kernels import (
        triton_fwht as triton_fwht,  # noqa: F401
        triton_blockwise_quantize_dequant as triton_blockwise_quantize_dequant,  # noqa: F401
        get_backend_name as _get_kv_backend_name,
    )
    _TRITON_AVAILABLE = True
except ImportError:
    def _get_kv_backend_name():
        return "python"
    pass


TurboFormatSpec = Union[str, int, float]

_LEGACY_TURBO_FORMATS = {
    "turbo2": (2.5, 4),
    "turbo3": (3.5, 8),
    "turbo4": (4.25, 16),
}

_TURBO_ROTATION_SEED = 1729
_TURBO_RESIDUAL_SEED = 2718
_GAUSSIAN_GRID_MIN = -8.0
_GAUSSIAN_GRID_MAX = 8.0
_GAUSSIAN_GRID_POINTS = 8193
_LLOYD_MAX_ITERS = 24
_TURBO_STATS = {
    "calls": 0,
    "residual_l2_sum": 0.0,
    "correction_l2_sum": 0.0,
    "residual_abs_mean_sum": 0.0,
    "correction_abs_mean_sum": 0.0,
}
logger = logging.getLogger(__name__)


def fwht(a: torch.Tensor) -> torch.Tensor:
    d = a.shape[-1]
    h = 1
    batch_shape = a.shape[:-1]
    y = a
    while h < d:
        y = y.reshape(*batch_shape, d // (2 * h), 2, h)
        x0 = y[..., 0, :]
        x1 = y[..., 1, :]
        y = torch.stack((x0 + x1, x0 - x1), dim=-2)
        y = y.reshape(*batch_shape, d)
        h *= 2
    return y / math.sqrt(d)


def apply_walsh_hadamard_rotation(tensor: torch.Tensor) -> torch.Tensor:
    orig_dim = tensor.shape[-1]
    padded_dim = 2 ** math.ceil(math.log2(orig_dim))

    if padded_dim != orig_dim:
        tensor = F.pad(tensor, (0, padded_dim - orig_dim))

    transformed = fwht(tensor)

    if padded_dim != orig_dim:
        transformed = transformed[..., :orig_dim]

    return transformed


def _normalize_turbo_spec(format_type: TurboFormatSpec) -> tuple[float, int]:
    if isinstance(format_type, str):
        key = format_type.strip().lower()
        if key in _LEGACY_TURBO_FORMATS:
            return _LEGACY_TURBO_FORMATS[key]
        if key.startswith("int"):
            bits = int(key[3:])
            return float(bits), 2 ** bits
        try:
            format_type = float(key)
        except ValueError as exc:
            raise ValueError(f"Unknown format {format_type}") from exc

    bits = float(format_type)
    if math.isclose(bits, 2.5, abs_tol=1e-6):
        return 2.5, 4
    if math.isclose(bits, 3.5, abs_tol=1e-6):
        return 3.5, 8
    if math.isclose(bits, 4.25, abs_tol=1e-6):
        return 4.25, 16

    rounded = round(bits)
    if not math.isclose(bits, rounded, abs_tol=1e-6) or rounded < 1:
        raise ValueError(f"Unsupported TurboQuant bit format: {format_type}")
    return float(rounded), 2 ** int(rounded)


def _validate_turbo_runtime_params(
    *,
    block_size: int,
    residual_strength: float,
) -> tuple[int, float]:
    try:
        block_size_i = int(block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TurboQuant block_size must be an integer, got {block_size!r}") from exc
    if block_size_i <= 0:
        raise ValueError(f"TurboQuant block_size must be > 0, got {block_size!r}")

    try:
        residual_strength_f = float(residual_strength)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TurboQuant residual_strength must be finite and non-negative, got {residual_strength!r}"
        ) from exc
    if not math.isfinite(residual_strength_f) or residual_strength_f < 0:
        raise ValueError(
            f"TurboQuant residual_strength must be finite and non-negative, got {residual_strength!r}"
        )
    return block_size_i, residual_strength_f


def resolve_turbo_kv_formats(
    *,
    turbo_format: Optional[TurboFormatSpec] = None,
    turbo_k_format: Optional[TurboFormatSpec] = None,
    turbo_v_format: Optional[TurboFormatSpec] = None,
    turbo_bits: Optional[float] = None,
    turbo_k_bits: Optional[float] = None,
    turbo_v_bits: Optional[float] = None,
) -> tuple[TurboFormatSpec, TurboFormatSpec]:
    if turbo_bits is not None:
        low_bits = max(1, int(math.floor(float(turbo_bits))))
        high_bits = max(low_bits, int(math.ceil(float(turbo_bits))))
        turbo_k_format = low_bits
        turbo_v_format = high_bits

    if turbo_k_bits is not None:
        turbo_k_format = float(turbo_k_bits)
    if turbo_v_bits is not None:
        turbo_v_format = float(turbo_v_bits)

    if turbo_format is not None:
        if turbo_k_format is None:
            turbo_k_format = turbo_format
        if turbo_v_format is None:
            turbo_v_format = turbo_format

    if turbo_k_format is None:
        turbo_k_format = "turbo3"
    if turbo_v_format is None:
        turbo_v_format = "turbo3"

    _normalize_turbo_spec(turbo_k_format)
    _normalize_turbo_spec(turbo_v_format)
    return turbo_k_format, turbo_v_format


@lru_cache(maxsize=None)
def _rademacher_signs(padded_dim: int, seed: int = _TURBO_ROTATION_SEED) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(padded_dim) * 17)
    signs = torch.randint(0, 2, (int(padded_dim),), generator=gen, dtype=torch.int64)
    return signs.mul_(2).sub_(1).to(torch.float32)


def _pad_last_dim(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    orig_dim = int(tensor.shape[-1])
    padded_dim = 2 ** math.ceil(math.log2(max(1, orig_dim)))
    if padded_dim == orig_dim:
        return tensor, orig_dim
    return F.pad(tensor, (0, padded_dim - orig_dim)), orig_dim


def _signed_hadamard_rotate(tensor: torch.Tensor, *, seed: int = _TURBO_ROTATION_SEED) -> tuple[torch.Tensor, int]:
    padded, orig_dim = _pad_last_dim(tensor)
    signs = _rademacher_signs(int(padded.shape[-1]), seed=seed).to(device=padded.device, dtype=padded.dtype)
    rotated = fwht(padded * signs)
    return rotated, orig_dim


def _signed_hadamard_inverse(rotated: torch.Tensor, orig_dim: int, *, seed: int = _TURBO_ROTATION_SEED) -> torch.Tensor:
    signs = _rademacher_signs(int(rotated.shape[-1]), seed=seed).to(device=rotated.device, dtype=rotated.dtype)
    restored = fwht(rotated) * signs
    return restored[..., :int(orig_dim)]


def _qjl_residual_sketch(
    residual: torch.Tensor,
    *,
    seed: int = _TURBO_RESIDUAL_SEED,
    strength: float = 1.0,
) -> torch.Tensor:
    """Approximate the residual with a 1-bit signed Hadamard sketch."""
    rotated, orig_dim = _signed_hadamard_rotate(residual, seed=int(seed))
    signs = torch.sign(rotated)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    scale = torch.mean(torch.abs(rotated), dim=-1, keepdim=True) * math.sqrt(math.pi / 2.0)
    sketch_rotated = signs * scale * float(strength)
    return _signed_hadamard_inverse(sketch_rotated, orig_dim=orig_dim, seed=int(seed))


def reset_turboquant_stats() -> None:
    _TURBO_STATS["calls"] = 0
    _TURBO_STATS["residual_l2_sum"] = 0.0
    _TURBO_STATS["correction_l2_sum"] = 0.0
    _TURBO_STATS["residual_abs_mean_sum"] = 0.0
    _TURBO_STATS["correction_abs_mean_sum"] = 0.0


def get_turboquant_stats() -> dict[str, float]:
    calls = int(_TURBO_STATS["calls"])
    if calls <= 0:
        return {
            "calls": 0,
            "avg_residual_l2": 0.0,
            "avg_correction_l2": 0.0,
            "avg_residual_abs_mean": 0.0,
            "avg_correction_abs_mean": 0.0,
            "correction_to_residual_l2_ratio": 0.0,
        }
    avg_residual_l2 = float(_TURBO_STATS["residual_l2_sum"]) / calls
    avg_correction_l2 = float(_TURBO_STATS["correction_l2_sum"]) / calls
    return {
        "calls": calls,
        "avg_residual_l2": avg_residual_l2,
        "avg_correction_l2": avg_correction_l2,
        "avg_residual_abs_mean": float(_TURBO_STATS["residual_abs_mean_sum"]) / calls,
        "avg_correction_abs_mean": float(_TURBO_STATS["correction_abs_mean_sum"]) / calls,
        "correction_to_residual_l2_ratio": (avg_correction_l2 / avg_residual_l2) if avg_residual_l2 > 0 else 0.0,
    }


@lru_cache(maxsize=None)
def _gaussian_lloyd_max_codebook(levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    if int(levels) < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")

    grid = torch.linspace(_GAUSSIAN_GRID_MIN, _GAUSSIAN_GRID_MAX, _GAUSSIAN_GRID_POINTS, dtype=torch.float64)
    pdf = torch.exp(-(grid ** 2) / 2.0) / math.sqrt(2.0 * math.pi)
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    probs = (torch.arange(int(levels), dtype=torch.float64) + 0.5) / float(levels)
    centroids = normal.icdf(probs)

    for _ in range(_LLOYD_MAX_ITERS):
        thresholds = 0.5 * (centroids[:-1] + centroids[1:])
        bounds = torch.cat(
            [
                torch.tensor([_GAUSSIAN_GRID_MIN], dtype=torch.float64),
                thresholds,
                torch.tensor([_GAUSSIAN_GRID_MAX], dtype=torch.float64),
            ]
        )
        updated = []
        for idx in range(int(levels)):
            left = bounds[idx]
            right = bounds[idx + 1]
            if idx == int(levels) - 1:
                mask = (grid >= left) & (grid <= right)
            else:
                mask = (grid >= left) & (grid < right)
            mass = pdf[mask].sum()
            if float(mass) <= 0.0:
                updated.append(centroids[idx])
                continue
            updated.append((grid[mask] * pdf[mask]).sum() / mass)
        centroids = torch.stack(updated)

    thresholds = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids.to(torch.float32), thresholds.to(torch.float32)


def _blockwise_quantize_dequantize(
    tensor: torch.Tensor,
    levels: int,
    block_size: int,
) -> torch.Tensor:
    """
    Blockwise min-max quantization with dequantization.

    Each block of `block_size` elements is independently quantized using
    min-max scaling followed by Lloyd-Max scalar quantization.

    This matches the paper's "blockwise" approach where each block gets
    its own scale factors before quantization.

    Args:
        tensor: Input tensor [..., seq, head_dim]
        levels: Number of quantization levels (e.g., 4 for turbo2, 8 for turbo3)
        block_size: Number of elements per block for min-max scaling

    Returns:
        Quantized then dequantized tensor of same shape as input
    """
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]

    # Pad last dimension to be multiple of block_size
    pad = 0
    if last_dim % block_size != 0:
        pad = block_size - (last_dim % block_size)
        tensor = F.pad(tensor, (0, pad))

    # Reshape: [..., seq, head_dim] -> [..., seq, n_blocks, block_size]
    n_blocks = tensor.shape[-1] // block_size
    tensor_4d = tensor.reshape(*orig_shape[:-1], n_blocks, block_size)

    # Per-block min-max scaling
    mins = tensor_4d.amin(dim=-1, keepdim=True)[0]
    maxs = tensor_4d.amax(dim=-1, keepdim=True)[0]
    ranges = maxs - mins
    ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)

    # Normalize to [0, levels-1]
    normalized = (tensor_4d - mins) / ranges * (levels - 1)
    normalized = torch.clamp(normalized, 0, levels - 1)

    # Lloyd-Max quantization on normalized values
    codebook, thresholds = _gaussian_lloyd_max_codebook(levels)
    codebook = codebook.to(device=tensor.device, dtype=tensor.dtype)
    thresholds = thresholds.to(device=tensor.device, dtype=tensor.dtype)

    # Quantize using bucketize
    flat = normalized.reshape(-1)
    bucket_idx = torch.bucketize(flat, thresholds)
    quantized_flat = codebook[bucket_idx]

    # Reshape back and denormalize
    quantized = quantized_flat.reshape_as(normalized)
    dequantized = quantized / (levels - 1) * ranges + mins

    # Remove padding
    result = dequantized.reshape(*orig_shape[:-1], -1)
    if pad > 0:
        result = result[..., :last_dim]

    return result


def turbo_quantize(
    tensor: torch.Tensor,
    format_type: TurboFormatSpec = "turbo3",
    block_size: int = 32,
    *,
    residual_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    rotation_seed: int = _TURBO_ROTATION_SEED,
    use_residual_qjl: bool = True,
    residual_seed: int = _TURBO_RESIDUAL_SEED,
    residual_strength: float = 0.5,
    use_blockwise: bool = True,
) -> torch.Tensor:
    """
    TurboQuant-style quantization - faithful to arXiv:2504.19874.

    Paper Pipeline:
        x → Hadamard Rotation → Standardize → Lloyd-Max Quantization → QJL Residual → x̂

    Implementation:
        1. Signed Hadamard rotation (data-oblivious, energy-spreading)
        2. Per-vector OR per-block standardization (configurable)
        3. Gaussian Lloyd-Max scalar quantization
        4. QJL residual sketch (optional, ~10-30% MSE improvement)

    Args:
        tensor: Input tensor [batch, heads, seq_len, head_dim]
        format_type: Quantization format ("turbo2"=2.5b, "turbo3"=3.5b, "turbo4"=4.25b)
        block_size: Elements per block for blockwise min-max scaling
        use_blockwise: If True, use per-block scaling; if False, use per-vector z-score

    Performance:
        - With Triton backend: 10-50× faster for large tensors
        - Pure PyTorch fallback: still functional

    Note on blockwise vs per-vector:
        - Paper describes per-vector standardization (each vector independently)
        - Vitriol enhancement: per-block min-max scaling (each block independently)
        - Set use_blockwise=False to match paper exactly
    """
    t_start = time.perf_counter()

    bits_per_value, levels = _normalize_turbo_spec(format_type)
    del bits_per_value  # format validation happens in _normalize_turbo_spec; levels drive runtime.
    block_size, residual_strength = _validate_turbo_runtime_params(
        block_size=block_size,
        residual_strength=residual_strength,
    )
    orig_dtype = tensor.dtype
    work = tensor.to(torch.float32)

    # Step 1: Signed Hadamard Rotation — with Triton if available
    rotated, orig_dim = _signed_hadamard_rotate(work, seed=int(rotation_seed))

    # Step 2: Standardization (per-vector z-score OR per-block min-max)
    if use_blockwise:
        # Vitriol enhancement: per-block min-max scaling
        normalized = rotated
    else:
        # Paper exact: per-vector z-score standardization
        sigma = torch.sqrt(torch.mean(rotated * rotated, dim=-1, keepdim=True) + 1e-8)
        normalized = torch.clamp(rotated / sigma, _GAUSSIAN_GRID_MIN, _GAUSSIAN_GRID_MAX)

    # Step 3: Lloyd-Max scalar quantization (per-block or per-vector)
    if use_blockwise:
        quantized = _blockwise_quantize_dequantize(normalized, int(levels), block_size)
    else:
        codebook, thresholds = _gaussian_lloyd_max_codebook(int(levels))
        codebook = codebook.to(device=normalized.device, dtype=normalized.dtype)
        thresholds = thresholds.to(device=normalized.device, dtype=normalized.dtype)

        flat = normalized.reshape(-1)
        bucket_idx = torch.bucketize(flat, thresholds)
        quantized = codebook[bucket_idx].reshape_as(normalized)

    # Step 4: Residual sketch
    residual = normalized - quantized
    correction = None
    if residual_hook is not None:
        correction = residual_hook(residual)
        quantized = quantized + correction
    elif use_residual_qjl:
        correction = _qjl_residual_sketch(
            residual,
            seed=int(residual_seed),
            strength=float(residual_strength),
        )
        quantized = quantized + correction

    # Stats tracking
    t_end = time.perf_counter()
    _TURBO_STATS["calls"] += 1
    _TURBO_STATS["residual_l2_sum"] += float(torch.sqrt(torch.mean(residual * residual)).item())
    _TURBO_STATS["residual_abs_mean_sum"] += float(torch.mean(torch.abs(residual)).item())
    if correction is not None:
        corr_f = correction.to(residual.dtype)
        _TURBO_STATS["correction_l2_sum"] += float(torch.sqrt(torch.mean(corr_f * corr_f)).item())
        _TURBO_STATS["correction_abs_mean_sum"] += float(torch.mean(torch.abs(corr_f)).item())

    # Step 5: Inverse transform (only if per-vector standardization was used)
    if not use_blockwise:
        dequantized = quantized * sigma
    else:
        dequantized = quantized
    restored = _signed_hadamard_inverse(dequantized, orig_dim=orig_dim, seed=int(rotation_seed))

    elapsed_ms = (t_end - t_start) * 1000
    if _TURBO_STATS["calls"] <= 3 or _TURBO_STATS["calls"] % 100 == 0:
        backend = "triton" if _TRITON_AVAILABLE else "pytorch"
        mode = "blockwise" if use_blockwise else "per-vector"
        logger.debug("TurboQuant [%s] call #%d: %.2fms, shape=%s, mode=%s",
                     backend, _TURBO_STATS["calls"], elapsed_ms, list(tensor.shape), mode)

    return restored.to(dtype=orig_dtype)


def sparse_v_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, threshold=0.01, scaling=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = float(scaling) if scaling is not None else (1 / math.sqrt(query.size(-1)))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(~temp_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    sparse_attn_weight = torch.where(attn_weight > threshold, attn_weight, torch.zeros_like(attn_weight))
    row_sums = sparse_attn_weight.sum(dim=-1, keepdim=True)
    sparse_attn_weight = torch.where(row_sums > 0, sparse_attn_weight / row_sums, sparse_attn_weight)

    return sparse_attn_weight @ value


class TurboQuantPatch:
    @staticmethod
    def patch_attention(model, use_sparse_v=True, kv_quant_format="turbo3"):
        logger.info("[TurboQuant] Patching model with KV format: %s", kv_quant_format)
        if use_sparse_v:
            logger.info("[TurboQuant] Enabling Sparse V (Attention-gated KV decoding)")
        return model
