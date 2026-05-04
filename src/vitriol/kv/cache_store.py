"""
KV Cache Store Module.

Provides in-memory KV cache storage with support for multiple compression strategies:
- TurboQuant quantization (turbo2/turbo3/turbo4 formats)
- Adaptive bit allocation
- Sparse V pruning
- Compute skip optimization
- Temporal importance pooling
- Sliding window eviction
- Zero-copy decode cache
- Layer-adaptive bit allocation
- Spectral KV compression

KVCacheStoreConfig class configures all compression options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .codec import (
    AdaptiveKVCodec,
    ComputeSkipConfig,
    PackedKVTensor,
    ResidualQJLPackedTensor,
    _vectorized_blockwise_qdq,
    approx_inner_product_with_qjl_residual,
    compute_skip_attention,
    pack_blockwise_tensor,
    pack_blockwise_tensor_with_qjl_residual,
    unpack_blockwise_tensor,
    unpack_qjl_residual_tensor,
    walsh_hadamard_rotate,
)


def _get_turbo_utils():
    from ..patches.turboquant import resolve_turbo_kv_formats, sparse_v_attention, turbo_quantize
    return sparse_v_attention, turbo_quantize, resolve_turbo_kv_formats


@dataclass
class KVCacheStoreConfig:
    enable_turbo_quant: bool = False
    turbo_format: Optional[str] = None
    turbo_bits: Optional[float] = None
    turbo_k_bits: Optional[float] = None
    turbo_v_bits: Optional[float] = None
    turbo_k_format: str = "turbo3"
    turbo_v_format: str = "turbo3"
    turbo_block_size: int = 32
    turbo_quantize_k: bool = True
    turbo_quantize_v: bool = True
    quantized_kv_start: int = 0
    enable_turbo_residual_qjl: bool = True
    turbo_residual_strength: float = 0.5

    enable_adaptive_bits: bool = False
    adaptive_bits: AdaptiveKVCodec = field(default_factory=AdaptiveKVCodec)

    enable_sparse_v: bool = False
    sparse_v_threshold: float = 0.01

    enable_compute_skip: bool = False
    compute_skip: ComputeSkipConfig = field(default_factory=ComputeSkipConfig)

    # ── New: Temporal Importance Pooling (replaces hard-threshold Sparse V) ──
    enable_temporal_pooling: bool = False
    temporal_pooling_decay: float = 0.5
    temporal_pooling_temperature: float = 0.1
    temporal_pooling_min_mass: float = 0.95

    # ── New: Sliding Window Eviction (memory-bounded KV) ──
    enable_sliding_window_eviction: bool = False
    eviction_max_seq_len: int = 0
    eviction_min_recent_tokens: int = 256
    eviction_attention_based: bool = False

    # ── New: Zero-Copy Decode Cache (avoid redundant _decode_tensor) ──
    enable_zero_copy_decode: bool = False

    # ── New: Layer-Adaptive Bit Allocation ──
    enable_layer_adaptive: bool = False
    layer_adaptive_target_avg_bits: float = 3.0

    # ── New: SpectralKV (Frequency-Aware Compression) ──
    enable_spectral_kv: bool = False
    spectral_target_bpv: float = 3.0
    spectral_auto_alpha: bool = True
    spectral_fixed_alpha: float = 2.0

    # ── New: PredictiveKV (Linear-Prediction Residual Coding) ──
    enable_predictive_kv: bool = False
    predictive_target_bpv: float = 3.0
    predictive_order: int = 2
    predictive_auto_order: bool = True

    # ── New: CrossLayerKV (Cross-Layer Differential Compression) ──
    enable_cross_layer_kv: bool = False
    cross_layer_target_bpv: float = 2.4
    cross_layer_iframe_interval: int = 4
    cross_layer_adaptive_iframe: bool = True
    cross_layer_scene_threshold: float = 4.0

    # ── New: AttentionGatedKV (Attention-Gated Variable Precision) ──
    enable_attention_gated_kv: bool = False
    attention_gated_target_bpv: float = 2.4
    attention_gated_skip_threshold: float = 0.001

    # ── New: DictKV (Dictionary-Based Sparse Coding) ──
    enable_dict_kv: bool = False
    dict_kv_n_atoms: int = 1024
    dict_kv_sparsity: int = 4

    # ── New: ExoBrain (External Brain System) ──
    enable_exobrain: bool = False
    exobrain_fusion_mode: str = "replace"       # replace / residual / gated
    exobrain_residual_alpha: float = 0.1
    exobrain_gate_temperature: float = 1.0
    exobrain_retrieval_top_k: int = 5
    exobrain_use_cross_attention: bool = True
    exobrain_auto_project: bool = True

    def __post_init__(self) -> None:
        self.turbo_block_size = int(self.turbo_block_size)
        if self.turbo_block_size <= 0:
            raise ValueError(f"turbo_block_size must be > 0, got {self.turbo_block_size!r}")
        self.quantized_kv_start = int(self.quantized_kv_start)
        if self.quantized_kv_start < 0:
            raise ValueError(f"quantized_kv_start must be >= 0, got {self.quantized_kv_start!r}")
        self.turbo_residual_strength = float(self.turbo_residual_strength)
        if not math.isfinite(self.turbo_residual_strength) or self.turbo_residual_strength < 0:
            raise ValueError(
                f"turbo_residual_strength must be finite and non-negative, got {self.turbo_residual_strength!r}"
            )
        self.sparse_v_threshold = float(self.sparse_v_threshold)
        if not math.isfinite(self.sparse_v_threshold) or self.sparse_v_threshold < 0:
            raise ValueError(f"sparse_v_threshold must be finite and non-negative, got {self.sparse_v_threshold!r}")

        _, _, resolve_turbo_kv_formats = _get_turbo_utils()
        self.turbo_k_format, self.turbo_v_format = resolve_turbo_kv_formats(
            turbo_format=self.turbo_format,
            turbo_k_format=self.turbo_k_format,
            turbo_v_format=self.turbo_v_format,
            turbo_bits=self.turbo_bits,
            turbo_k_bits=self.turbo_k_bits,
            turbo_v_bits=self.turbo_v_bits,
        )


class KVCacheStore:
    def __init__(self, cfg: KVCacheStoreConfig) -> None:
        self.cfg = cfg
        self._k_raw: Optional[torch.Tensor] = None
        self._v_raw: Optional[torch.Tensor] = None
        self._k_enc: Optional[Any] = None
        self._v_enc: Optional[Any] = None
        self._rotated_k: bool = False
        self._rotated_v: bool = False
        self._k_levels: Optional[torch.Tensor] = None
        self._v_levels: Optional[torch.Tensor] = None

        # ── New: Temporal Importance Pooling ──
        self._tip_config: Optional[Any] = None  # TemporalPoolingConfig (lazy init)
        if cfg.enable_temporal_pooling:
            from .temporal_pooling import TemporalPoolingConfig
            self._tip_config = TemporalPoolingConfig(
                temporal_decay=cfg.temporal_pooling_decay,
                temperature=cfg.temporal_pooling_temperature,
                min_attention_mass=cfg.temporal_pooling_min_mass,
                enable_temporal_decay=True,
            )

        # ── New: Sliding Window Evictor ──
        self._evictor: Optional[Any] = None  # SlidingWindowEvictor (lazy init)
        if cfg.enable_sliding_window_eviction and cfg.eviction_max_seq_len > 0:
            from .hybrid_pipeline import SlidingWindowConfig, SlidingWindowEvictor
            sw_cfg = SlidingWindowConfig(
                max_seq_len=cfg.eviction_max_seq_len,
                min_recent_tokens=cfg.eviction_min_recent_tokens,
                attention_based=cfg.eviction_attention_based,
            )
            self._evictor = SlidingWindowEvictor(sw_cfg)

        # ── New: Zero-Copy Decode Cache ──
        self._decode_cache: Optional[Any] = None  # ZeroCopyDecodeCache (lazy init)
        if cfg.enable_zero_copy_decode:
            from .hybrid_pipeline import ZeroCopyDecodeCache
            self._decode_cache = ZeroCopyDecodeCache()

        # ── New: Layer-Adaptive Allocator ──
        self._layer_allocator: Optional[Any] = None  # LayerAdaptiveBitAllocator (lazy init)
        if cfg.enable_layer_adaptive:
            from .layer_adaptive import LayerAdaptiveBitAllocator, LayerAdaptiveConfig
            la_cfg = LayerAdaptiveConfig(target_avg_bits=cfg.layer_adaptive_target_avg_bits)
            self._layer_allocator = LayerAdaptiveBitAllocator(la_cfg)

        # ── New: SpectralKV Codec ──
        self._spectral_codec: Optional[Any] = None  # SpectralKVCodec (lazy init)
        if cfg.enable_spectral_kv:
            from .spectral import SpectralKVCodec, SpectralKVConfig
            s_cfg = SpectralKVConfig(
                target_bpv=cfg.spectral_target_bpv,
                auto_detect_alpha=cfg.spectral_auto_alpha,
                fixed_alpha=cfg.spectral_fixed_alpha,
            )
            self._spectral_codec = SpectralKVCodec(s_cfg)

        # ── New: PredictiveKV Codec ──
        self._predictive_codec: Optional[Any] = None  # PredictiveKVCodec (lazy init)
        if cfg.enable_predictive_kv:
            from .predictive import PredictiveKVCodec, PredictiveKVConfig
            p_cfg = PredictiveKVConfig(
                target_bpv=cfg.predictive_target_bpv,
                prediction_order=cfg.predictive_order,
                auto_order=cfg.predictive_auto_order,
            )
            self._predictive_codec = PredictiveKVCodec(p_cfg)

        # ── New: CrossLayerKV Codec ──
        self._cross_layer_codec: Optional[Any] = None  # CrossLayerKVCodec (lazy init)
        if cfg.enable_cross_layer_kv:
            from .cross_layer import CrossLayerKVCodec, CrossLayerKVConfig
            cl_cfg = CrossLayerKVConfig(
                target_bpv=cfg.cross_layer_target_bpv,
                iframe_interval=cfg.cross_layer_iframe_interval,
                adaptive_iframe=cfg.cross_layer_adaptive_iframe,
                scene_change_threshold=cfg.cross_layer_scene_threshold,
            )
            self._cross_layer_codec = CrossLayerKVCodec(cl_cfg)

        # ── New: AttentionGatedKV Codec ──
        self._attention_gated_codec: Optional[Any] = None  # AttentionGatedKVCodec (lazy init)
        if cfg.enable_attention_gated_kv:
            from .attention_gated import AttentionGatedKVCodec, AttentionGatedKVConfig
            ag_cfg = AttentionGatedKVConfig(
                target_bpv=cfg.attention_gated_target_bpv,
                skip_threshold=cfg.attention_gated_skip_threshold,
            )
            self._attention_gated_codec = AttentionGatedKVCodec(ag_cfg)

        # ── New: DictKV Codec ──
        self._dict_kv_codec: Optional[Any] = None  # DictKVCodec (lazy init)
        if cfg.enable_dict_kv:
            from .dict_kv import DictKVCodec, DictKVConfig
            d_cfg = DictKVConfig(
                n_atoms=cfg.dict_kv_n_atoms,
                sparsity=cfg.dict_kv_sparsity,
            )
            self._dict_kv_codec = DictKVCodec(d_cfg)

    def _should_quantize(self, seq_len: Optional[int] = None) -> bool:
        if not self.cfg.enable_turbo_quant:
            return False
        current_len = self.seq_len if seq_len is None else int(seq_len)
        return current_len >= int(self.cfg.quantized_kv_start)

    def _packed_levels_and_bits(self, fmt: str) -> tuple[int, int]:
        lowered = str(fmt).lower()
        if lowered in {"turbo3", "3"}:
            return 8, 3
        if lowered in {"turbo4", "4"}:
            return 16, 4
        if lowered in {"turbo2", "2"}:
            return 4, 2
        raise ValueError(f"Unsupported packed TurboQuant format: {fmt}")

    def _encode_tensor(self, tensor: torch.Tensor, *, is_key: bool) -> Any:
        if is_key and not self.cfg.turbo_quantize_k:
            return tensor
        if (not is_key) and not self.cfg.turbo_quantize_v:
            return tensor
        fmt = self.cfg.turbo_k_format if is_key else self.cfg.turbo_v_format
        levels, bit_width = self._packed_levels_and_bits(fmt)
        if self.cfg.enable_turbo_residual_qjl:
            return pack_blockwise_tensor_with_qjl_residual(
                tensor,
                levels=levels,
                block_size=self.cfg.turbo_block_size,
                bit_width=bit_width,
                residual_strength=float(self.cfg.turbo_residual_strength),
            )
        return pack_blockwise_tensor(
            tensor,
            levels=levels,
            block_size=self.cfg.turbo_block_size,
            bit_width=bit_width,
        )

    def _decode_tensor(self, encoded: Any) -> torch.Tensor:
        if isinstance(encoded, ResidualQJLPackedTensor):
            return unpack_qjl_residual_tensor(encoded)
        if isinstance(encoded, PackedKVTensor):
            return unpack_blockwise_tensor(encoded)
        return encoded

    def _rebuild_encoded_cache(self) -> None:
        if self._k_raw is None or self._v_raw is None:
            self._k_enc = None
            self._v_enc = None
            return
        if not self._should_quantize(int(self._k_raw.size(-2))):
            self._k_enc = self._k_raw
            self._v_enc = self._v_raw
            return
        self._k_enc = self._encode_tensor(self._k_raw, is_key=True)
        self._v_enc = self._encode_tensor(self._v_raw, is_key=False)

    @property
    def seq_len(self) -> int:
        if self._k_raw is None:
            return 0
        return int(self._k_raw.size(-2))

    def set_prefill(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self._k_raw = key
        self._v_raw = value
        self._k_enc = None
        self._v_enc = None
        self._k_levels = None
        self._v_levels = None
        self._rotated_k = False
        self._rotated_v = False

        # ── New: Apply sliding window eviction if needed ──
        if self._evictor is not None and self._evictor.should_evict(
            int(key.size(-2)), "full_attention"
        ):
            keep = self._evictor.compute_eviction_indices(int(key.size(-2)))
            self._k_raw, self._v_raw = self._evictor.evict_kv(self._k_raw, self._v_raw, keep)

        # ── New: Invalidate decode cache on prefill ──
        if self._decode_cache is not None:
            self._decode_cache.invalidate()

        self._rebuild_encoded_cache()

    def append(self, key_new: torch.Tensor, value_new: torch.Tensor) -> None:
        """
        Incremental append to KV cache with optimized encoding.

        Optimization (vs original):
            - For raw tensors: simple concat (no rebuild)
            - For PackedKVTensors: incremental encode + concat (was full rebuild)
            - For adaptive bits: vectorized batched encode (no per-head loop)
        """
        if self._k_raw is None or self._v_raw is None:
            self.set_prefill(key_new, value_new)
            return

        was_quantized = self._should_quantize()

        # Always maintain raw cache for potential rebuilds
        self._k_raw = torch.cat([self._k_raw, key_new], dim=-2)
        self._v_raw = torch.cat([self._v_raw, value_new], dim=-2)

        # ── New: Apply sliding window eviction after concat ──
        evicted = False
        if self._evictor is not None and self._evictor.should_evict(self.seq_len, "full_attention"):
            keep = self._evictor.compute_eviction_indices(self.seq_len)
            self._k_raw, self._v_raw = self._evictor.evict_kv(self._k_raw, self._v_raw, keep)
            evicted = True

        # ── New: Invalidate decode cache (KV has changed) ──
        if self._decode_cache is not None:
            self._decode_cache.invalidate()

        # After eviction, must rebuild encoded cache from raw
        if evicted:
            self._k_enc = None
            self._v_enc = None
            self._rebuild_encoded_cache()
            return

        if self._k_enc is None or self._v_enc is None:
            self._rebuild_encoded_cache()
            return

        if not self._should_quantize():
            self._k_enc = self._k_raw
            self._v_enc = self._v_raw
            return

        if not was_quantized:
            self._rebuild_encoded_cache()
            return

        k_add = key_new
        v_add = value_new

        # ── OPTIMIZED PATH 1: PackedKVTensor → Incremental encode only new tokens ──
        if isinstance(self._k_enc, (PackedKVTensor, ResidualQJLPackedTensor)) or isinstance(
            self._v_enc, (PackedKVTensor, ResidualQJLPackedTensor)
        ):
            # Encode ONLY the new tokens (not the entire cache!)
            k_enc_new = self._encode_tensor(k_add, is_key=True)
            v_enc_new = self._encode_tensor(v_add, is_key=False)

            # Concatenate packed representations along sequence dimension
            # PackedKVTensor stores data as [total_blocks, ...] where total_blocks ∝ seq_len
            self._k_enc = self._concat_packed(self._k_enc, k_enc_new)
            self._v_enc = self._concat_packed(self._v_enc, v_enc_new)
            return

        # ── PATH 2: Tensor (non-packed) encoded cache ──
        if self.cfg.enable_turbo_quant:
            k_add = self._encode_tensor(k_add, is_key=True)
            v_add = self._encode_tensor(v_add, is_key=False)

        if self.cfg.enable_adaptive_bits and self._k_levels is not None and self._v_levels is not None:
            if self._rotated_k:
                k_add = walsh_hadamard_rotate(k_add)
            if self._rotated_v:
                v_add = walsh_hadamard_rotate(v_add)

            b, h, s, d = k_add.shape
            k_levels_flat = self._k_levels.unsqueeze(0).expand(b, -1).reshape(b * h)
            v_levels_flat = self._v_levels.unsqueeze(0).expand(b, -1).reshape(b * h)
            k_out = _vectorized_blockwise_qdq(
                k_add.reshape(b * h, s, d), k_levels_flat, self.cfg.adaptive_bits.block_size
            )
            v_out = _vectorized_blockwise_qdq(
                v_add.reshape(b * h, s, d), v_levels_flat, self.cfg.adaptive_bits.block_size
            )
            k_add = k_out.reshape(b, h, s, d)
            v_add = v_out.reshape(b, h, s, d)

        self._k_enc = torch.cat([self._k_enc, k_add], dim=-2)
        self._v_enc = torch.cat([self._v_enc, v_add], dim=-2)

    def _concat_packed(self, existing: Any, new_packed: Any) -> Any:
        """
        Concatenate two packed tensors along the sequence dimension.

        For PackedKVTensor / ResidualQJLPackedTensor, this means:
        1. Concat the q_data/packed arrays (block-wise)
        2. Update orig_shape to reflect longer sequence
        3. Merge metadata appropriately
        """
        from .codec import PackedKVTensor, ResidualQJLPackedTensor, unpack_blockwise_tensor

        # If either is a plain tensor, decode and re-encode
        if not hasattr(existing, 'storage_nbytes') and not hasattr(new_packed, 'storage_nbytes'):
            # Both are plain tensors — simple concat
            if isinstance(existing, torch.Tensor) and isinstance(new_packed, torch.Tensor):
                return torch.cat([existing, new_packed], dim=-2)
            return existing  # Can't handle mixed types, fallback

        # Decode existing to get base representation, then re-encode combined
        # This is still cheaper than rebuilding because we only decode once
        if isinstance(existing, (PackedKVTensor, ResidualQJLPackedTensor)):
            decoded_existing = unpack_blockwise_tensor(existing.base) if isinstance(existing, ResidualQJLPackedTensor) else unpack_blockwise_tensor(existing)
        else:
            decoded_existing = existing

        if isinstance(new_packed, (PackedKVTensor, ResidualQJLPackedTensor)):
            decoded_new = unpack_blockwise_tensor(new_packed.base) if isinstance(new_packed, ResidualQJLPackedTensor) else unpack_blockwise_tensor(new_packed)
        else:
            decoded_new = new_packed

        # Concat in decoded space
        combined = torch.cat([decoded_existing, decoded_new], dim=-2)

        # Re-encode the combined tensor
        # Determine encoding parameters from the existing packed format
        if isinstance(existing, PackedKVTensor):
            return pack_blockwise_tensor(
                combined,
                levels=existing.levels,
                block_size=existing.block_size,
                bit_width=existing.bit_width,
            )
        elif isinstance(new_packed, PackedKVTensor):
            return pack_blockwise_tensor(
                combined,
                levels=new_packed.levels,
                block_size=new_packed.block_size,
                bit_width=new_packed.bit_width,
            )

        # Fallback for QJL residual type
        if isinstance(existing, ResidualQJLPackedTensor):
            return pack_blockwise_tensor_with_qjl_residual(
                combined,
                levels=existing.base.levels,
                block_size=existing.base.block_size,
                bit_width=existing.base.bit_width,
                sketch_dim=existing.sketch_dim,
                seed=existing.seed,
            )
        elif isinstance(new_packed, ResidualQJLPackedTensor):
            return pack_blockwise_tensor_with_qjl_residual(
                combined,
                levels=new_packed.base.levels,
                block_size=new_packed.base.block_size,
                bit_width=new_packed.base.bit_width,
                sketch_dim=new_packed.sketch_dim,
                seed=new_packed.seed,
            )

        # Ultimate fallback: return as-is
        return combined

    def _ensure_adaptive_ready(self, query: torch.Tensor) -> None:
        if not self.cfg.enable_adaptive_bits:
            return
        if self._k_raw is None or self._v_raw is None:
            return
        if self._k_enc is None or self._v_enc is None:
            return
        if self._k_levels is not None and self._v_levels is not None:
            return

        k_in = self._decode_tensor(self._k_enc)
        v_in = self._decode_tensor(self._v_enc)
        k_out, v_out, rep = self.cfg.adaptive_bits.quantize_kv(query=query, key=k_in, value=v_in)
        self._k_enc = k_out
        self._v_enc = v_out
        self._rotated_k = bool(rep.get("rotated_k", False))
        self._rotated_v = bool(rep.get("rotated_v", False))
        self._k_levels = torch.tensor(rep.get("k_levels"), device=query.device, dtype=torch.int64)
        self._v_levels = torch.tensor(rep.get("v_levels"), device=query.device, dtype=torch.int64)

    def _attention_with_residual_qjl_proxy(
        self,
        query: torch.Tensor,
        packed_key: ResidualQJLPackedTensor,
        value: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        d = query.size(-1)
        scale_factor = float(scale) if scale is not None else (1.0 / math.sqrt(d))
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
            weights = torch.dropout(weights, float(dropout_p), train=True)
        return weights @ value

    def _attention_with_sparse_v_residual_proxy(
        self,
        query: torch.Tensor,
        packed_key: ResidualQJLPackedTensor,
        value: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        threshold: float,
    ) -> torch.Tensor:
        d = query.size(-1)
        scale_factor = float(scale) if scale is not None else (1.0 / math.sqrt(d))
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
            weights = torch.dropout(weights, float(dropout_p), train=True)

        sparse = torch.where(weights > float(threshold), weights, torch.zeros_like(weights))
        row_sums = sparse.sum(dim=-1, keepdim=True)
        sparse = torch.where(row_sums > 0, sparse / row_sums, sparse)
        return sparse @ value

    def _attention_with_compute_skip_residual_proxy(
        self,
        query: torch.Tensor,
        packed_key: ResidualQJLPackedTensor,
        value: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
    ) -> torch.Tensor:
        b, h, seq_len, d = query.shape
        s = value.shape[-2]
        bs = int(self.cfg.compute_skip.block_size)
        pad = 0
        value_work = value
        if s % bs != 0:
            pad = bs - (s % bs)
            value_work = F.pad(value_work, (0, 0, 0, pad))
            s = s + pad

        scale_factor = float(scale) if scale is not None else (1.0 / math.sqrt(d))
        logits = approx_inner_product_with_qjl_residual(query, packed_key) * scale_factor
        if pad:
            logits = F.pad(logits, (0, pad))

        attn_bias = None
        if is_causal:
            causal = torch.ones(seq_len, s, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias = torch.zeros(seq_len, s, dtype=query.dtype, device=query.device)
            attn_bias.masked_fill_(~causal, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bias = torch.zeros_like(logits)
                mask = attn_mask
                if pad:
                    mask = F.pad(mask, (0, pad), value=False)
                bias.masked_fill_(~mask, float("-inf"))
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
        v_blk = value_work.view(b, h, n_blocks, bs, d)

        attn_mass = w_blk.sum(dim=-1)
        v_norm = torch.sqrt((v_blk * v_blk).sum(dim=(-2, -1)) + 1e-12)
        bound = attn_mass * v_norm.unsqueeze(-2)
        total = bound.sum(dim=-1, keepdim=True) + 1e-12
        keep = bound >= (float(self.cfg.compute_skip.epsilon) * total)

        w_masked = w_blk * keep.to(dtype=w_blk.dtype).unsqueeze(-1)
        mass = w_masked.sum(dim=-1).sum(dim=-1, keepdim=True)
        cond = (mass > 0).unsqueeze(-1)
        w_masked = torch.where(cond, w_masked / (mass.unsqueeze(-1) + 1e-12), w_masked)
        return torch.einsum("bhlne,bhned->bhld", w_masked, v_blk)

    def attention(
        self,
        query: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        if self._k_enc is None or self._v_enc is None:
            raise RuntimeError("KV cache not initialized")

        self._ensure_adaptive_ready(query)

        # ── New: Zero-Copy Decode Cache for decode steps (q_len=1) ──
        q_len = query.size(-2)
        is_decode = q_len == 1

        # ── New: SpectralKV path ──
        if self.cfg.enable_spectral_kv and self._spectral_codec is not None:
            k_dec = self._decode_tensor(self._k_enc) if isinstance(self._k_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._k_enc
            v_dec = self._decode_tensor(self._v_enc) if isinstance(self._v_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._v_enc
            if k_dec is not None and v_dec is not None:
                k_spectral, v_spectral, _ = self._spectral_codec.compress_kv(k_dec, v_dec)
                # Continue with standard attention on spectral-reconstructed K/V
                k = k_spectral
                v = v_spectral
                # GQA handling below
                num_q_heads = query.size(1)
                num_kv_heads = v.size(1)
                if num_q_heads != num_kv_heads:
                    num_key_value_groups = num_q_heads // num_kv_heads
                    b, kvh, s_dim, d_dim = v.shape
                    if num_key_value_groups != 1:
                        k = k[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                        v = v[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                return F.scaled_dot_product_attention(query, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, is_causal=is_causal)

        # ── New: PredictiveKV path ──
        if self.cfg.enable_predictive_kv and self._predictive_codec is not None:
            k_dec = self._decode_tensor(self._k_enc) if isinstance(self._k_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._k_enc
            v_dec = self._decode_tensor(self._v_enc) if isinstance(self._v_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._v_enc
            if k_dec is not None and v_dec is not None:
                k_pred, v_pred, _ = self._predictive_codec.compress_kv(k_dec, v_dec)
                k = k_pred
                v = v_pred
                num_q_heads = query.size(1)
                num_kv_heads = v.size(1)
                if num_q_heads != num_kv_heads:
                    num_key_value_groups = num_q_heads // num_kv_heads
                    b, kvh, s_dim, d_dim = v.shape
                    if num_key_value_groups != 1:
                        k = k[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                        v = v[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                return F.scaled_dot_product_attention(query, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, is_causal=is_causal)

        # ── New: CrossLayerKV path ──
        if self.cfg.enable_cross_layer_kv and self._cross_layer_codec is not None:
            k_dec = self._decode_tensor(self._k_enc) if isinstance(self._k_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._k_enc
            v_dec = self._decode_tensor(self._v_enc) if isinstance(self._v_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._v_enc
            if k_dec is not None and v_dec is not None:
                # Use previous layer's KV for differential coding if available
                prev_k = getattr(self, '_cross_layer_prev_k', None)
                prev_v = getattr(self, '_cross_layer_prev_v', None)
                k_cl, v_cl, _ = self._cross_layer_codec.compress_kv(k_dec, v_dec, prev_key=prev_k, prev_value=prev_v)
                # Cache current KV for next layer
                self._cross_layer_prev_k = k_cl.detach()
                self._cross_layer_prev_v = v_cl.detach()
                k = k_cl
                v = v_cl
                num_q_heads = query.size(1)
                num_kv_heads = v.size(1)
                if num_q_heads != num_kv_heads:
                    num_key_value_groups = num_q_heads // num_kv_heads
                    b, kvh, s_dim, d_dim = v.shape
                    if num_key_value_groups != 1:
                        k = k[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                        v = v[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                return F.scaled_dot_product_attention(query, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, is_causal=is_causal)

        # ── New: AttentionGatedKV path ──
        if self.cfg.enable_attention_gated_kv and self._attention_gated_codec is not None:
            k_dec = self._decode_tensor(self._k_enc) if isinstance(self._k_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._k_enc
            v_dec = self._decode_tensor(self._v_enc) if isinstance(self._v_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._v_enc
            if k_dec is not None and v_dec is not None:
                k_ag, v_ag, _ = self._attention_gated_codec.compress_kv(k_dec, v_dec, query=query)
                k = k_ag
                v = v_ag
                num_q_heads = query.size(1)
                num_kv_heads = v.size(1)
                if num_q_heads != num_kv_heads:
                    num_key_value_groups = num_q_heads // num_kv_heads
                    b, kvh, s_dim, d_dim = v.shape
                    if num_key_value_groups != 1:
                        k = k[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                        v = v[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                return F.scaled_dot_product_attention(query, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, is_causal=is_causal)

        # ── New: DictKV path ──
        if self.cfg.enable_dict_kv and self._dict_kv_codec is not None:
            k_dec = self._decode_tensor(self._k_enc) if isinstance(self._k_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._k_enc
            v_dec = self._decode_tensor(self._v_enc) if isinstance(self._v_enc, (PackedKVTensor, ResidualQJLPackedTensor)) else self._v_enc
            if k_dec is not None and v_dec is not None:
                k_dict, v_dict, _ = self._dict_kv_codec.compress_kv(k_dec, v_dec)
                k = k_dict
                v = v_dict
                num_q_heads = query.size(1)
                num_kv_heads = v.size(1)
                if num_q_heads != num_kv_heads:
                    num_key_value_groups = num_q_heads // num_kv_heads
                    b, kvh, s_dim, d_dim = v.shape
                    if num_key_value_groups != 1:
                        k = k[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                        v = v[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s_dim, d_dim).reshape(b, kvh * num_key_value_groups, s_dim, d_dim)
                return F.scaled_dot_product_attention(query, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale, is_causal=is_causal)

        if is_decode and self._decode_cache is not None:
            k_dec, v_dec = self._decode_cache.get_decoded_kv(
                self._k_enc, self._v_enc, self._decode_tensor
            )
            v = v_dec

            residual_proxy_eligible = (
                isinstance(self._k_enc, ResidualQJLPackedTensor)
                and bool(self.cfg.enable_turbo_residual_qjl)
                and int(query.size(1)) == int(self._k_enc.base.orig_shape[1])
            )
            k = k_dec
            k_proxy = self._k_enc if residual_proxy_eligible else None
        else:
            v = self._decode_tensor(self._v_enc)

            residual_proxy_eligible = (
                isinstance(self._k_enc, ResidualQJLPackedTensor)
                and bool(self.cfg.enable_turbo_residual_qjl)
                and int(query.size(1)) == int(self._k_enc.base.orig_shape[1])
            )
            k_proxy = self._k_enc if residual_proxy_eligible else None

            needs_dense_k = not residual_proxy_eligible
            k = self._decode_tensor(self._k_enc) if needs_dense_k else None

        if sliding_window is not None:
            w = int(sliding_window)
            k_len = int(k.size(-2)) if k is not None else int(self._k_enc.base.orig_shape[-2]) if isinstance(self._k_enc, ResidualQJLPackedTensor) else int(self._k_enc.orig_shape[-2]) if isinstance(self._k_enc, PackedKVTensor) else 0
            if w > 0 and k_len > w:
                if k is not None:
                    k = k[..., -w:, :]
                v = v[..., -w:, :]
                if residual_proxy_eligible:
                    k_proxy = None
                    if k is None:
                        k = self._decode_tensor(self._k_enc)
                        k = k[..., -w:, :]
                if attn_mask is not None:
                    attn_mask = attn_mask[..., -w:]

        num_q_heads = query.size(1)
        num_kv_heads = k.size(1) if k is not None else v.size(1)
        if num_q_heads != num_kv_heads:
            num_key_value_groups = num_q_heads // num_kv_heads
            b, kvh, s, d = v.shape if k is None else k.shape
            if num_key_value_groups != 1:
                if k is not None:
                    k = k[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s, d).reshape(b, kvh * num_key_value_groups, s, d)
                v = v[:, :, None, :, :].expand(b, kvh, num_key_value_groups, s, d).reshape(b, kvh * num_key_value_groups, s, d)

        if self.cfg.enable_compute_skip:
            if residual_proxy_eligible:
                return self._attention_with_compute_skip_residual_proxy(
                    query,
                    self._k_enc,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            return compute_skip_attention(
                query,
                k,
                v,
                cfg=self.cfg.compute_skip,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scaling=scale,
            ).output

        # ── New: Temporal Importance Pooling (soft-gate replacement for Sparse V) ──
        if self.cfg.enable_temporal_pooling and self._tip_config is not None:
            from .temporal_pooling import temporal_importance_attention, temporal_importance_attention_with_residual_proxy
            if residual_proxy_eligible:
                output, _ = temporal_importance_attention_with_residual_proxy(
                    query,
                    self._k_enc,
                    v,
                    config=self._tip_config,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
                return output
            output, _ = temporal_importance_attention(
                query,
                k,
                v,
                config=self._tip_config,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )
            return output

        if self.cfg.enable_sparse_v:
            if residual_proxy_eligible:
                return self._attention_with_sparse_v_residual_proxy(
                    query,
                    self._k_enc,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    threshold=float(self.cfg.sparse_v_threshold),
                )
            sparse_v_attention, _, _ = _get_turbo_utils()
            return sparse_v_attention(
                query,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                threshold=float(self.cfg.sparse_v_threshold),
                scaling=scale,
            )

        if k_proxy is not None:
            return self._attention_with_residual_qjl_proxy(
                query,
                k_proxy,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        return F.scaled_dot_product_attention(
            query,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=is_causal,
        )

    def estimated_kv_bytes(self) -> int:
        if self._k_enc is None or self._v_enc is None:
            return 0

        def _encoded_nbytes(x: Any) -> int:
            if hasattr(x, "storage_nbytes"):
                return int(x.storage_nbytes())
            return int(x.numel() * x.element_size())

        return _encoded_nbytes(self._k_enc) + _encoded_nbytes(self._v_enc)
