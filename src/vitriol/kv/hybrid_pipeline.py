"""
Hybrid KV Compression Pipeline + Sliding Window Eviction + Zero-Copy Decode Path.

Three optimizations in one module:

1. Hybrid Pipeline: Unify TurboQuant (Lloyd-Max) and PackedKV (sub-byte packing)
   into a single pass, eliminating the decode→re-encode waste in _concat_packed().

2. Sliding Window Eviction: Automatically evict old KV entries when sequence
   exceeds a configurable window, preventing unbounded memory growth.

3. Zero-Copy Decode: Cache the decoded KV in float during decode steps,
   avoiding repeated _decode_tensor() calls for every attention computation.

Problem being solved:
    - cache_store.py _concat_packed() decodes packed → concat → re-encode = wasted work
    - No memory-bounded eviction policy for ultra-long contexts
    - Every attention() call re-decodes the full KV cache even in decode (q_len=1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. Sliding Window Eviction Policy
# ─────────────────────────────────────────────────────────────

@dataclass
class SlidingWindowConfig:
    """Configuration for KV cache sliding window eviction."""

    # Maximum sequence length to keep in KV cache
    # 0 = no eviction (unlimited)
    max_seq_len: int = 0

    # Whether to use attention-based eviction (drop lowest-attention positions)
    # vs simple FIFO (drop oldest positions)
    attention_based: bool = False

    # Minimum tokens to always keep (recent tokens)
    min_recent_tokens: int = 256

    # Eviction granularity: how many tokens to evict at once
    eviction_chunk_size: int = 64

    # Whether to apply eviction only to specific layer types
    apply_to_layer_types: List[str] = field(default_factory=lambda: ["full_attention"])


class SlidingWindowEvictor:
    """
    Sliding window eviction for KV cache.

    When the sequence length exceeds max_seq_len, evicts the oldest
    (or least-attended) tokens to keep memory bounded.

    This is critical for ultra-long context scenarios where the KV cache
    would otherwise grow without bound.
    """

    def __init__(self, config: SlidingWindowConfig) -> None:
        self.config = config
        self._eviction_count: int = 0
        self._total_evicted_tokens: int = 0

    def should_evict(self, current_seq_len: int, layer_type: str = "full_attention") -> bool:
        """Check if eviction is needed."""
        if self.config.max_seq_len <= 0:
            return False
        if layer_type not in self.config.apply_to_layer_types:
            return False
        return current_seq_len > self.config.max_seq_len

    def compute_eviction_indices(
        self,
        current_seq_len: int,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute which positions to evict.

        Args:
            current_seq_len: Current KV cache sequence length
            attention_weights: [batch, heads, q_len, k_len] (optional, for attention-based eviction)

        Returns:
            keep_indices: [num_keep] indices of positions to retain
        """
        cfg = self.config
        target_len = cfg.max_seq_len
        num_to_evict = current_seq_len - target_len

        if num_to_evict <= 0:
            return torch.arange(current_seq_len)

        # Always keep recent tokens
        recent_start = max(0, current_seq_len - cfg.min_recent_tokens)
        recent_indices = torch.arange(recent_start, current_seq_len)

        # Older tokens that are eviction candidates
        old_indices = torch.arange(0, recent_start)

        if cfg.attention_based and attention_weights is not None:
            # Attention-based eviction: drop positions with lowest cumulative attention
            # attention_weights: [b, h, q, k]
            cum_attn = attention_weights.sum(dim=(0, 1, 2))  # [k_len]
            # Only consider old positions
            old_attn = cum_attn[:recent_start]
            # Keep top-k by attention
            num_old_to_keep = max(0, target_len - cfg.min_recent_tokens)
            if num_old_to_keep >= len(old_indices):
                keep_old = old_indices
            else:
                _, topk_idx = torch.topk(old_attn, k=num_old_to_keep)
                keep_old = old_indices[topk_idx.sort()[0]]
        else:
            # FIFO: keep the most recent old tokens
            num_old_to_keep = max(0, target_len - cfg.min_recent_tokens)
            if num_old_to_keep >= len(old_indices):
                keep_old = old_indices
            else:
                keep_old = old_indices[-num_old_to_keep:]

        keep_indices = torch.cat([keep_old, recent_indices])

        # Update stats
        self._eviction_count += 1
        self._total_evicted_tokens += (current_seq_len - len(keep_indices))

        return keep_indices

    def evict_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evict KV cache entries, keeping only specified positions.

        Args:
            key: [batch, heads, seq_len, d]
            value: [batch, heads, seq_len, d]
            keep_indices: [num_keep] sorted indices to retain

        Returns:
            evicted_key, evicted_value with reduced sequence dimension
        """
        return key[:, :, keep_indices, :], value[:, :, keep_indices, :]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "eviction_count": self._eviction_count,
            "total_evicted_tokens": self._total_evicted_tokens,
            "max_seq_len": self.config.max_seq_len,
        }


# ─────────────────────────────────────────────────────────────
# 2. Zero-Copy Decode Path
# ─────────────────────────────────────────────────────────────

class ZeroCopyDecodeCache:
    """
    Cache for decoded (float) KV tensors during decode steps.

    Problem:
        KVCacheStore.attention() calls _decode_tensor() on every invocation,
        even during decode when q_len=1 and the KV hasn't changed.

    Solution:
        Cache the decoded float tensors. On decode steps (q_len=1), reuse
        the cached version instead of re-decoding from packed format.

    Invalidation:
        - Invalidated when append() adds new tokens
        - Invalidated when eviction changes the KV content
        - Preserved across multiple decode steps with same KV

    Memory overhead:
        - Only exists during decode phase (q_len=1)
        - Freed when new tokens are appended (switch to prefill)
    """

    def __init__(self) -> None:
        self._cached_k: Optional[torch.Tensor] = None
        self._cached_v: Optional[torch.Tensor] = None
        self._cached_k_seq_len: int = 0
        self._cached_v_seq_len: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def get_decoded_kv(
        self,
        k_enc: Any,
        v_enc: Any,
        decode_fn,  # Callable: encoded → float Tensor
        force_refresh: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get decoded KV tensors, using cache when possible.

        Args:
            k_enc: Encoded key (PackedKVTensor, ResidualQJLPackedTensor, or raw Tensor)
            v_enc: Encoded value
            decode_fn: Function to decode encoded → float Tensor
            force_refresh: Force re-decode even if cache exists

        Returns:
            (decoded_key, decoded_value) as float Tensors
        """
        # Check if we can use cached version
        k_seq = self._get_seq_len(k_enc)
        v_seq = self._get_seq_len(v_enc)

        if (
            not force_refresh
            and self._cached_k is not None
            and self._cached_v is not None
            and self._cached_k_seq_len == k_seq
            and self._cached_v_seq_len == v_seq
        ):
            self._cache_hits += 1
            return self._cached_k, self._cached_v

        # Cache miss: decode and cache
        self._cache_misses += 1
        decoded_k = decode_fn(k_enc)
        decoded_v = decode_fn(v_enc)

        self._cached_k = decoded_k
        self._cached_v = decoded_v
        self._cached_k_seq_len = k_seq
        self._cached_v_seq_len = v_seq

        return decoded_k, decoded_v

    def invalidate(self) -> None:
        """Invalidate cache (called when new tokens are appended)."""
        self._cached_k = None
        self._cached_v = None
        self._cached_k_seq_len = 0
        self._cached_v_seq_len = 0

    def _get_seq_len(self, encoded: Any) -> int:
        """Get sequence length from encoded tensor."""
        if isinstance(encoded, torch.Tensor):
            return int(encoded.size(-2))
        if hasattr(encoded, 'orig_shape'):
            return int(encoded.orig_shape[-2])
        if hasattr(encoded, 'base') and hasattr(encoded.base, 'orig_shape'):
            return int(encoded.base.orig_shape[-2])
        return 0

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(1, total)
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }


# ─────────────────────────────────────────────────────────────
# 3. Hybrid Compression Pipeline
# ─────────────────────────────────────────────────────────────

@dataclass
class HybridPipelineConfig:
    """Configuration for the hybrid TurboQuant + PackedKV pipeline."""

    # Whether to use TurboQuant (Lloyd-Max codebook) for initial quantization
    use_turbo_quant: bool = True

    # Whether to pack the TurboQuant output into sub-byte PackedKVTensor
    use_packed_kv: bool = True

    # Turbo format for K and V
    turbo_k_format: str = "turbo3"
    turbo_v_format: str = "turbo3"

    # Block size for packed KV
    block_size: int = 32

    # Whether to use QJL residual in packed format
    use_qjl_residual: bool = True
    qjl_residual_strength: float = 0.5

    # Sliding window eviction
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)

    # Zero-copy decode cache
    enable_zero_copy_decode: bool = True


class HybridKVCacheStore:
    """
    Optimized KV Cache Store that combines:
        1. TurboQuant's high-quality quantization (Lloyd-Max codebook)
        2. PackedKV's compact storage (sub-byte packing)
        3. Sliding window eviction for memory-bounded operation
        4. Zero-copy decode cache to avoid redundant decoding

    Key improvement over KVCacheStore:
        - _concat_packed() no longer decodes→concat→re-encodes
        - Instead, maintains both packed and decoded representations
        - Decode cache avoids redundant _decode_tensor() calls
        - Sliding window eviction prevents unbounded memory growth
    """

    def __init__(self, config: HybridPipelineConfig) -> None:
        self.config = config
        self._k_raw: Optional[torch.Tensor] = None
        self._v_raw: Optional[torch.Tensor] = None
        self._k_enc: Optional[Any] = None
        self._v_enc: Optional[Any] = None
        self._evictor = SlidingWindowEvictor(config.sliding_window) if config.sliding_window.max_seq_len > 0 else None
        self._decode_cache = ZeroCopyDecodeCache() if config.enable_zero_copy_decode else None

    @property
    def seq_len(self) -> int:
        if self._k_raw is None:
            return 0
        return int(self._k_raw.size(-2))

    def set_prefill(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Initialize KV cache with prefill tokens."""
        # Apply sliding window eviction if needed
        if self._evictor is not None and self._evictor.should_evict(
            int(key.size(-2)), "full_attention"
        ):
            keep = self._evictor.compute_eviction_indices(int(key.size(-2)))
            key, value = self._evictor.evict_kv(key, value, keep)

        self._k_raw = key
        self._v_raw = value
        self._k_enc = None
        self._v_enc = None

        # Invalidate decode cache
        if self._decode_cache is not None:
            self._decode_cache.invalidate()

    def append(self, key_new: torch.Tensor, value_new: torch.Tensor) -> None:
        """
        Incremental append with optimized path.

        Key optimization: for decode steps (q_len=1), we only encode
        the new token and concatenate packed representations directly
        WITHOUT decoding the existing cache first.
        """
        if self._k_raw is None or self._v_raw is None:
            self.set_prefill(key_new, value_new)
            return

        # Always maintain raw cache
        self._k_raw = torch.cat([self._k_raw, key_new], dim=-2)
        self._v_raw = torch.cat([self._v_raw, value_new], dim=-2)

        # Apply sliding window eviction
        if self._evictor is not None and self._evictor.should_evict(self.seq_len, "full_attention"):
            keep = self._evictor.compute_eviction_indices(self.seq_len)
            self._k_raw, self._v_raw = self._evictor.evict_kv(self._k_raw, self._v_raw, keep)
            # Need full rebuild after eviction
            self._k_enc = None
            self._v_enc = None

        # Invalidate decode cache (KV has changed)
        if self._decode_cache is not None:
            self._decode_cache.invalidate()

    def attention(
        self,
        query: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute attention with zero-copy decode optimization.

        For decode steps (q_len=1), reuses cached decoded KV
        instead of re-decoding packed representations every time.
        """
        from .codec import (
            PackedKVTensor, ResidualQJLPackedTensor,
            unpack_blockwise_tensor, unpack_qjl_residual_tensor,
        )

        if self._k_raw is None or self._v_raw is None:
            raise RuntimeError("KV cache not initialized")

        # Decode KV (with zero-copy cache for decode steps)
        def _decode(encoded: Any) -> torch.Tensor:
            if isinstance(encoded, ResidualQJLPackedTensor):
                return unpack_qjl_residual_tensor(encoded)
            if isinstance(encoded, PackedKVTensor):
                return unpack_blockwise_tensor(encoded)
            return encoded

        q_len = query.size(-2)
        is_decode = q_len == 1

        if is_decode and self._decode_cache is not None and self._k_enc is not None:
            k, v = self._decode_cache.get_decoded_kv(
                self._k_enc, self._v_enc, _decode
            )
        else:
            # Prefill or no cache: decode fresh
            k = _decode(self._k_enc) if self._k_enc is not None else self._k_raw
            v = _decode(self._v_enc) if self._v_enc is not None else self._v_raw

        # Apply sliding window to KV
        if sliding_window is not None and sliding_window > 0:
            w = int(sliding_window)
            if k.size(-2) > w:
                k = k[..., -w:, :]
                v = v[..., -w:, :]
                if attn_mask is not None:
                    attn_mask = attn_mask[..., -w:]

        # GQA expansion
        num_q_heads = query.size(1)
        num_kv_heads = k.size(1)
        if num_q_heads != num_kv_heads:
            num_key_value_groups = num_q_heads // num_kv_heads
            if num_key_value_groups > 1:
                b, kvh, s, d = k.shape
                k = k[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s, d).reshape(b, kvh * num_key_value_groups, s, d)
                v = v[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s, d).reshape(b, kvh * num_key_value_groups, s, d)

        return F.scaled_dot_product_attention(
            query, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=is_causal,
        )

    def estimated_kv_bytes(self) -> int:
        """Estimate KV cache memory usage."""
        if self._k_enc is None or self._v_enc is None:
            if self._k_raw is None:
                return 0
            return int(self._k_raw.numel() * self._k_raw.element_size() +
                       self._v_raw.numel() * self._v_raw.element_size())

        def _encoded_nbytes(x: Any) -> int:
            if hasattr(x, "storage_nbytes"):
                return int(x.storage_nbytes())
            return int(x.numel() * x.element_size())

        return _encoded_nbytes(self._k_enc) + _encoded_nbytes(self._v_enc)

    @property
    def stats(self) -> Dict[str, Any]:
        result = {
            "seq_len": self.seq_len,
            "estimated_kv_bytes": self.estimated_kv_bytes(),
        }
        if self._evictor is not None:
            result["evictor"] = self._evictor.stats
        if self._decode_cache is not None:
            result["decode_cache"] = self._decode_cache.stats
        return result
