"""
KV Cache Codec Module.

Provides tensor packing and encoding for KV cache compression:
- PackedKVTensor: Quantized KV tensor with scales and bounds
- ResidualQJLPackedTensor: Packed tensor with residual QJL sketch
- Codec functions: _fwht_last_dim, _pack_tensor, _quantize_to_levels

Supports multiple quantization formats:
- turbo2, turbo3, turbo4: Turbo series formats
- q8_0, q4_0: Standard quantization formats
- adaptive_bits: Variable bits per value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F


@dataclass
class PackedKVTensor:
    q_data: torch.Tensor
    scales: torch.Tensor
    mins: torch.Tensor
    orig_shape: Tuple[int, ...]
    padded_last_dim: int
    block_size: int
    levels: int
    bit_width: int

    def storage_nbytes(self) -> int:
        total = int(self.q_data.numel() * self.q_data.element_size())
        total += int(self.scales.numel() * self.scales.element_size())
        total += int(self.mins.numel() * self.mins.element_size())
        return total


@dataclass
class ResidualQJLPackedTensor:
    base: PackedKVTensor
    projection: torch.Tensor
    residual_sign_bits: torch.Tensor
    residual_scale: torch.Tensor
    residual_norms: torch.Tensor
    sketch_dim: int
    seed: int

    @property
    def residual_signs(self) -> torch.Tensor:
        return _unpack_sign_tensor(self.residual_sign_bits, self.sketch_dim)

    @property
    def residual_magnitudes(self) -> torch.Tensor:
        return self.residual_scale.expand(*self.residual_scale.shape[:-1], self.sketch_dim)

    def storage_nbytes(self) -> int:
        total = self.base.storage_nbytes()
        total += int(self.residual_sign_bits.numel() * self.residual_sign_bits.element_size())
        total += int(self.residual_scale.numel() * self.residual_scale.element_size())
        total += int(self.residual_norms.numel() * self.residual_norms.element_size())
        return total


def kv_bytes_per_value(kv_quant: Optional[Union[str, Dict[str, Any]]]) -> float:
    if kv_quant is None:
        return 2.0
    if isinstance(kv_quant, str):
        mapping = {
            "turbo2": 2.5 / 8.0,
            "turbo3": 3.5 / 8.0,
            "turbo4": 4.25 / 8.0,
            "q8_0": 8.5 / 8.0,
            "q4_0": 4.5 / 8.0,
            "bf16": 2.0,
        }
        return float(mapping.get(kv_quant, 2.0))
    name = kv_quant.get("name")
    if name == "adaptive_bits":
        target_avg_bits = float(kv_quant.get("target_avg_bits", 3.5))
        return target_avg_bits / 8.0
    if name == "turbo":
        fmt = kv_quant.get("format", "turbo3")
        return kv_bytes_per_value(fmt)
    return 2.0


def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    h = 1
    batch_shape = x.shape[:-1]
    y = x
    while h < d:
        y = y.reshape(*batch_shape, d // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        y = y.reshape(*batch_shape, d)
        h *= 2
    return y / math.sqrt(d)


def walsh_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    orig = x.shape[-1]
    padded = 2 ** math.ceil(math.log2(orig))
    if padded != orig:
        x = F.pad(x, (0, padded - orig))
    y = _fwht_last_dim(x)
    if padded != orig:
        y = y[..., :orig]
    return y


def blockwise_minmax_quantize_dequantize(
    x: torch.Tensor,
    levels: int,
    block_size: int,
) -> torch.Tensor:
    shape = x.shape
    last = shape[-1]
    if last % block_size != 0:
        pad = block_size - (last % block_size)
        x = F.pad(x, (0, pad))
    flat = x.reshape(-1, x.shape[-1] // block_size, block_size)
    mins = flat.min(dim=-1, keepdim=True)[0]
    maxs = flat.max(dim=-1, keepdim=True)[0]
    scales = (maxs - mins) / (levels - 1 + 1e-5)
    q = torch.round((flat - mins) / (scales + 1e-5))
    q = torch.clamp(q, 0, levels - 1)
    dq = q * scales + mins
    out = dq.reshape(*x.shape[:-1], x.shape[-1])
    if out.shape[-1] != last:
        out = out[..., :last]
    return out.reshape(shape)


def _vectorized_blockwise_qdq(
    x: torch.Tensor,
    per_batch_levels: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Vectorized blockwise quantize-dequantize with per-batch level configuration.

    This is the key optimization that eliminates the O(b·h) Python loop.

    Args:
        x: Input tensor of shape [N, seq_len, d] where N = batch * heads
        per_batch_levels: Integer levels for each of the N entries [N]
        block_size: Block size for quantization

    Returns:
        Quantized-dequantized tensor same shape as x

    How it works:
        Instead of looping over each (batch, head) pair and calling
        blockwise_minmax_quantize_dequantize individually, we:

        1. Reshape all N entries into one big batch [total_blocks, block_size]
        2. Compute min/max per block in a single vectorized pass
        3. Use broadcasting to apply per-entry levels via gathered scales
        4. Reshape back to original shape

        This reduces N sequential calls to 1 vectorized call.
    """
    orig_shape = x.shape  # [N, s, d]
    last = orig_shape[-1]

    # Pad if needed
    pad = 0
    if last % block_size != 0:
        pad = block_size - (last % block_size)
        x = F.pad(x, (0, pad))

    # Flatten into blocks: [N * (s*d/block_size), block_size]
    N = x.shape[0]
    flat = x.reshape(N, -1, block_size)  # [N, num_blocks_per_entry, block_size]
    total_N, num_blocks, _ = flat.shape

    # Merge batch and block dims for unified processing
    flat_all = flat.reshape(total_N * num_blocks, block_size)  # [total_blocks_all, bs]

    # Vectorized min/max per block
    mins = flat_all.min(dim=-1, keepdim=True)[0]   # [total, 1]
    maxs = flat_all.max(dim=-1, keepdim=True)[0]   # [total, 1]

    # Per-block scales — but we need to use the correct `levels` for each original entry
    # Expand per_batch_levels from [N] to [N, num_blocks] -> flatten to [total]
    expanded_levels = per_batch_levels.unsqueeze(1).expand(-1, num_blocks).reshape(-1)  # [total]
    expanded_levels_f = expanded_levels.float().unsqueeze(-1)  # [total, 1]

    scales = (maxs - mins) / (expanded_levels_f - 1 + 1e-5)

    # Vectorized quantize + dequantize
    q = torch.round((flat_all - mins) / (scales + 1e-5))
    q = torch.clamp(q, 0, (expanded_levels_f - 1).clamp(min=2))
    dq = q * scales + mins

    # Restore shape
    out = dq.reshape(total_N, num_blocks, block_size).reshape(orig_shape[:-1] + (-1,))
    if pad > 0:
        out = out[..., :last]

    return out


def pack_blockwise_tensor(
    x: torch.Tensor,
    *,
    levels: int,
    block_size: int,
    bit_width: int,
) -> PackedKVTensor:
    shape = tuple(int(v) for v in x.shape)
    last = int(shape[-1])
    padded_last = last
    if last % block_size != 0:
        padded_last = last + (block_size - (last % block_size))
        x = F.pad(x, (0, padded_last - last))

    flat = x.reshape(-1, padded_last // block_size, block_size)
    mins = flat.min(dim=-1, keepdim=True)[0]
    maxs = flat.max(dim=-1, keepdim=True)[0]
    scales = (maxs - mins) / (levels - 1 + 1e-5)
    q = torch.round((flat - mins) / (scales + 1e-5))
    q = torch.clamp(q, 0, levels - 1).to(torch.int32)

    values_per_byte = max(1, 8 // bit_width)
    packed_width = (block_size + values_per_byte - 1) // values_per_byte
    padded_block = packed_width * values_per_byte
    if q.shape[-1] < padded_block:
        q = F.pad(q, (0, padded_block - q.shape[-1]))
    q = q.reshape(*q.shape[:-1], packed_width, values_per_byte)
    shifts = torch.arange(values_per_byte, device=q.device, dtype=torch.int32) * int(bit_width)
    packed = torch.sum(q << shifts, dim=-1).to(torch.uint8)

    return PackedKVTensor(
        q_data=packed,
        scales=scales.to(torch.float32),
        mins=mins.to(torch.float32),
        orig_shape=shape,
        padded_last_dim=padded_last,
        block_size=int(block_size),
        levels=int(levels),
        bit_width=int(bit_width),
    )


def _unpack_blockwise_q_values(packed: PackedKVTensor) -> torch.Tensor:
    values_per_byte = max(1, 8 // packed.bit_width)
    mask = (1 << packed.bit_width) - 1
    rows = packed.q_data.shape[0]
    blocks = packed.q_data.shape[1]
    shifts = torch.arange(values_per_byte, device=packed.q_data.device, dtype=torch.int32) * int(packed.bit_width)
    q = ((packed.q_data.to(torch.int32).unsqueeze(-1) >> shifts) & mask).to(torch.float32)
    q = q.reshape(rows, blocks, -1)
    if q.shape[-1] > packed.block_size:
        q = q[..., : packed.block_size]
    return q


def unpack_blockwise_tensor(packed: PackedKVTensor) -> torch.Tensor:
    q = _unpack_blockwise_q_values(packed)
    restored = q * packed.scales + packed.mins
    restored = restored.reshape(*packed.orig_shape[:-1], packed.padded_last_dim)
    if packed.padded_last_dim != packed.orig_shape[-1]:
        restored = restored[..., : packed.orig_shape[-1]]
    return restored.reshape(packed.orig_shape)


# Global cache for projection matrices to avoid recomputation per layer
_PROJECTION_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


def _rademacher_projection(
    dim: int,
    sketch_dim: int,
    *,
    seed: int,
    device: torch.device,
    use_cache: bool = True,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Generate a Rademacher (±1) projection matrix with caching and precision control.

    Optimizations vs original:
        1. Global cache: same (dim, sketch_dim) reused across all layers
        2. Half-precision: float16 cuts memory by 2×
        3. Configurable sketch_dim: default tuned for quality/cost balance

    Memory: dim=4096, sketch_dim=16, fp16 → 128KB/layer (vs 128KB before at fp32/sketch=8)
    For 60 layers: 7.68MB total
    """
    # Use cache if available
    cache_key = (dim, sketch_dim, seed)
    if use_cache and cache_key in _PROJECTION_CACHE:
        cached = _PROJECTION_CACHE[cache_key]
        if cached.device != device or cached.dtype != dtype:
            cached = cached.to(device=device, dtype=dtype)
            _PROJECTION_CACHE[cache_key] = cached
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    proj_int = torch.randint(
        0,
        2,
        (dim, sketch_dim),
        generator=generator,
        dtype=torch.int8,   # Generate as int8 (values are only -1, 0, +1)
    ).to(device=device)

    # Convert ±1 pattern: int8(0) -> -1.0, int8(1) -> +1.0
    proj_f = proj_int.to(torch.float32) * 2.0 - 1.0
    proj = proj_f.to(dtype) / math.sqrt(float(sketch_dim))

    if use_cache:
        _PROJECTION_CACHE[cache_key] = proj

    return proj


def clear_projection_cache():
    """Clear the global projection matrix cache."""
    global _PROJECTION_CACHE
    count = len(_PROJECTION_CACHE)
    _PROJECTION_CACHE.clear()
    return count


def _pack_sign_tensor(signs: torch.Tensor) -> torch.Tensor:
    bits = (signs > 0).to(torch.uint8)
    logical_dim = int(bits.shape[-1])
    packed_width = (logical_dim + 7) // 8
    padded_dim = packed_width * 8
    if bits.shape[-1] < padded_dim:
        bits = F.pad(bits, (0, padded_dim - bits.shape[-1]))
    bits = bits.reshape(*bits.shape[:-1], packed_width, 8).to(torch.int32)
    shifts = torch.arange(8, device=bits.device, dtype=torch.int32)
    return torch.sum(bits << shifts, dim=-1).to(torch.uint8)


def _unpack_sign_tensor(packed: torch.Tensor, logical_dim: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
    bits = ((packed.to(torch.int32).unsqueeze(-1) >> shifts) & 1).reshape(*packed.shape[:-1], -1)
    if bits.shape[-1] > logical_dim:
        bits = bits[..., :logical_dim]
    return bits.to(torch.int8).mul_(2).sub_(1)


def pack_blockwise_tensor_with_qjl_residual(
    x: torch.Tensor,
    *,
    levels: int,
    block_size: int,
    bit_width: int,
    sketch_dim: int = 16,
    seed: int = 0,
    residual_strength: float = 1.0,
) -> ResidualQJLPackedTensor:
    base = pack_blockwise_tensor(
        x,
        levels=levels,
        block_size=block_size,
        bit_width=bit_width,
    )
    restored = unpack_blockwise_tensor(base)
    residual = x - restored

    # Use optimized projection with caching and half-precision
    proj_dtype = torch.float16 if x.device.type == "cuda" else torch.float32
    projection = _rademacher_projection(
        x.shape[-1],
        int(sketch_dim),
        seed=int(seed),
        device=x.device,
        use_cache=True,
        dtype=proj_dtype,
    )

    residual_proj = (residual.float() @ projection.float())  # Compute in float32 for accuracy
    residual_signs = torch.where(residual_proj >= 0, 1, -1).to(torch.int8)
    residual_sign_bits = _pack_sign_tensor(residual_signs)
    residual_norms = torch.linalg.vector_norm(residual, dim=-1, keepdim=True).to(torch.float32)
    residual_scale = (
        residual_proj.abs().mean(dim=-1, keepdim=True) * float(residual_strength)
    ).to(torch.float32)

    return ResidualQJLPackedTensor(
        base=base,
        projection=projection,          # Now fp16 (or cached), much smaller
        residual_sign_bits=residual_sign_bits,
        residual_scale=residual_scale,
        residual_norms=residual_norms,
        sketch_dim=int(sketch_dim),
        seed=int(seed),
    )


def unpack_qjl_residual_tensor(packed: ResidualQJLPackedTensor) -> torch.Tensor:
    base = unpack_blockwise_tensor(packed.base)
    residual_proj = packed.residual_signs.to(torch.float32) * packed.residual_scale
    approx_residual = residual_proj @ packed.projection.transpose(0, 1)
    return base + approx_residual.to(base.dtype)


def approx_inner_product_with_packed_tensor(
    query: torch.Tensor,
    packed: PackedKVTensor,
) -> torch.Tensor:
    prefix = tuple(int(v) for v in packed.orig_shape[:-2])
    seq_len = int(packed.orig_shape[-2])
    num_blocks = packed.padded_last_dim // packed.block_size
    query_last = int(query.shape[-1])
    query_padded = query
    if query_last < packed.padded_last_dim:
        query_padded = F.pad(query, (0, packed.padded_last_dim - query_last))
    elif query_last > packed.padded_last_dim:
        query_padded = query[..., : packed.padded_last_dim]

    q_query = query_padded.to(torch.float32).reshape(*query.shape[:-1], num_blocks, packed.block_size)
    q_values = _unpack_blockwise_q_values(packed).reshape(*prefix, seq_len, num_blocks, packed.block_size)
    restored_blocks = q_values * packed.scales.reshape(*prefix, seq_len, num_blocks, 1) + packed.mins.reshape(
        *prefix, seq_len, num_blocks, 1
    )

    if query.shape == packed.orig_shape:
        return (q_query * restored_blocks).sum(dim=(-2, -1))
    return torch.einsum("...lnd,...snd->...ls", q_query, restored_blocks)


def approx_inner_product_with_qjl_residual(
    query: torch.Tensor,
    packed: ResidualQJLPackedTensor,
) -> torch.Tensor:
    query_proj = query @ packed.projection
    residual_proj = packed.residual_signs.to(torch.float32) * packed.residual_scale
    base_ip = approx_inner_product_with_packed_tensor(query, packed.base)
    sketch_dim = float(packed.sketch_dim)
    if sketch_dim < 32.0:
        correction_gain = math.sqrt(sketch_dim / 32.0)
    else:
        correction_gain = 1.0 + (1.0 - (32.0 / sketch_dim)) * (math.sqrt(math.pi / 2.0) - 1.0)
    if query.shape == packed.base.orig_shape:
        correction = (query_proj * residual_proj).sum(dim=-1) * correction_gain
        return base_ip + correction
    correction = (query_proj @ residual_proj.transpose(-2, -1)) * correction_gain
    return base_ip + correction


def _entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = torch.clamp(p, min=1e-12)
    return -(p * torch.log(p)).sum(dim=dim)


def adaptive_kv_bits(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    min_bits: float = 3.0,
    max_bits: float = 5.0,
    target_avg_bits: float = 3.5,
    k_share: float = 0.65,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    d = query.shape[-1]
    s = key.shape[-2]

    scale = 1.0 / math.sqrt(d)
    logits = (query @ key.transpose(-2, -1)) * scale
    w = torch.softmax(logits, dim=-1)
    ent = _entropy(w, dim=-1)
    ent = ent.mean(dim=-1)
    ent = ent / math.log(s)
    importance = torch.clamp(1.0 - ent, 0.0, 1.0)

    v_rms = torch.sqrt((value * value).mean(dim=(-2, -1)))
    v_rms = v_rms / (v_rms.mean(dim=-1, keepdim=True) + 1e-12)
    v_importance = torch.clamp(v_rms / (v_rms.max(dim=-1, keepdim=True)[0] + 1e-12), 0.0, 1.0)

    k_bits = min_bits + (max_bits - min_bits) * importance
    v_bits = min_bits + (max_bits - min_bits) * v_importance

    avg = (k_bits.mean(dim=-1) * k_share + v_bits.mean(dim=-1) * (1.0 - k_share)).mean()
    if float(avg) > 0:
        scale_bits = target_avg_bits / float(avg)
        k_bits = torch.clamp(k_bits * scale_bits, min_bits, max_bits)
        v_bits = torch.clamp(v_bits * scale_bits, min_bits, max_bits)

    report = {
        "min_bits": float(min_bits),
        "max_bits": float(max_bits),
        "target_avg_bits": float(target_avg_bits),
        "k_share": float(k_share),
        "avg_bits_k": float(k_bits.mean()),
        "avg_bits_v": float(v_bits.mean()),
        "avg_bits_total": float((k_bits.mean() * k_share + v_bits.mean() * (1.0 - k_share)).item()),
    }
    return k_bits, v_bits, report


def _levels_from_bits(bits: torch.Tensor) -> torch.Tensor:
    levels = torch.clamp(torch.round(torch.pow(2.0, bits)), 2.0, 256.0)
    return levels.to(dtype=torch.int64)


class AdaptiveKVCodec:
    def __init__(
        self,
        block_size: int = 32,
        min_bits: float = 3.0,
        max_bits: float = 5.0,
        target_avg_bits: float = 3.5,
        k_share: float = 0.65,
        rotate_kurtosis_threshold: float = 10.0,
    ) -> None:
        self.block_size = int(block_size)
        self.min_bits = float(min_bits)
        self.max_bits = float(max_bits)
        self.target_avg_bits = float(target_avg_bits)
        self.k_share = float(k_share)
        self.rotate_kurtosis_threshold = float(rotate_kurtosis_threshold)

    def _maybe_rotate(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        z = (x - x.mean()) / (x.std() + 1e-12)
        kurt = torch.mean(z.pow(4))
        if float(kurt) >= self.rotate_kurtosis_threshold:
            return walsh_hadamard_rotate(x), True
        return x, False

    def quantize_kv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        key2, rotated_k = self._maybe_rotate(key)
        value2, rotated_v = self._maybe_rotate(value)

        k_bits, v_bits, report = adaptive_kv_bits(
            query=query,
            key=key2,
            value=value2,
            min_bits=self.min_bits,
            max_bits=self.max_bits,
            target_avg_bits=self.target_avg_bits,
            k_share=self.k_share,
        )

        b, h = key2.shape[:2]
        k_levels = _levels_from_bits(k_bits).view(b, h, 1, 1)
        v_levels = _levels_from_bits(v_bits).view(b, h, 1, 1)

        # ── VECTORIZED: eliminate O(b·h) Python loop ──
        # Reshape to [b*h, s, d] for batched processing
        k_flat = key2.reshape(b * h, *key2.shape[2:])
        v_flat = value2.reshape(b * h, *value2.shape[2:])
        k_levels_flat = k_levels.reshape(b * h)      # [b*h]
        v_levels_flat = v_levels.reshape(b * h)       # [b*h]

        # Batched blockwise quantize: process all (batch*heads) in parallel
        k_out = _vectorized_blockwise_qdq(k_flat, k_levels_flat, self.block_size)
        v_out = _vectorized_blockwise_qdq(v_flat, v_levels_flat, self.block_size)

        # Restore shape
        k_out = k_out.reshape(key2.shape)
        v_out = v_out.reshape(value2.shape)

        if rotated_k:
            k_out = walsh_hadamard_rotate(k_out)
        if rotated_v:
            v_out = walsh_hadamard_rotate(v_out)

        report = dict(report)
        report.update(
            {
                "rotated_k": bool(rotated_k),
                "rotated_v": bool(rotated_v),
                "avg_bytes_per_value": float(self.target_avg_bits / 8.0),
                "k_levels": [int(x) for x in k_levels[0, :, 0, 0].tolist()],
                "v_levels": [int(x) for x in v_levels[0, :, 0, 0].tolist()],
            }
        )
        return k_out, v_out, report


@dataclass(frozen=True)
class ComputeSkipConfig:
    block_size: int = 128
    epsilon: float = 0.02


@dataclass(frozen=True)
class ComputeSkipResult:
    output: torch.Tensor
    kept_fraction: float


def compute_skip_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cfg: ComputeSkipConfig = ComputeSkipConfig(),
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scaling: Optional[float] = None,
) -> ComputeSkipResult:
    b, h, seq_len, d = query.shape
    s = key.shape[-2]
    bs = int(cfg.block_size)
    pad = 0
    if s % bs != 0:
        pad = bs - (s % bs)
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        s = s + pad

    scale = float(scaling) if scaling is not None else (1.0 / math.sqrt(d))
    logits = (query @ key.transpose(-2, -1)) * scale

    attn_bias = None
    if is_causal:
        causal = torch.ones(seq_len, s, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias = torch.zeros(seq_len, s, dtype=query.dtype, device=query.device)
        attn_bias.masked_fill_(~causal, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias = torch.zeros_like(logits)
            if pad:
                attn_mask = F.pad(attn_mask, (0, pad), value=False)
            bias.masked_fill_(~attn_mask, float("-inf"))
        else:
            bias = attn_mask
            if pad:
                bias = F.pad(bias, (0, pad))
        logits = logits + bias

    if attn_bias is not None:
        logits = logits + attn_bias

    w = torch.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        w = torch.dropout(w, float(dropout_p), train=True)

    n_blocks = s // bs
    w_blk = w.view(b, h, seq_len, n_blocks, bs)
    v_blk = value.view(b, h, n_blocks, bs, d)

    attn_mass = w_blk.sum(dim=-1)
    v_norm = torch.sqrt((v_blk * v_blk).sum(dim=(-2, -1)) + 1e-12)
    bound = attn_mass * v_norm.unsqueeze(-2)
    total = bound.sum(dim=-1, keepdim=True) + 1e-12
    keep = bound >= (float(cfg.epsilon) * total)

    w_masked = w_blk * keep.to(dtype=w_blk.dtype).unsqueeze(-1)
    mass = w_masked.sum(dim=-1).sum(dim=-1, keepdim=True)
    cond = (mass > 0).unsqueeze(-1)
    w_masked = torch.where(cond, w_masked / (mass.unsqueeze(-1) + 1e-12), w_masked)
    out = torch.einsum("bhlne,bhned->bhld", w_masked, v_blk)
    kept = keep.to(dtype=torch.float32).mean()
    return ComputeSkipResult(output=out, kept_fraction=float(kept.item()))
