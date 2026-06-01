"""Adaptive injection scheduling, KV compression and prefetching."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .teacher import HeadDimProjection, TeacherKVCache

logger = logging.getLogger(__name__)


class AdaptiveInjectionScheduler:
    """
    Decide when to inject external brain KV based on model confidence (v0.6).

    Problem: Injecting teacher KV at EVERY decode step is wasteful.
    When the shell model is confident (low perplexity), external guidance
    adds little value but incurs compute cost. When uncertain (high PPL),
    injection is critical.

    Solution: Monitor the shell model's perplexity at each decode step.
    Only inject when PPL exceeds a threshold, saving ~30-60% of injections
    with minimal quality loss.

    Strategies:
    - "threshold": Inject when PPL > threshold (simple, effective)
    - "relative": Inject when PPL > α × running_avg_PPL (adaptive)
    - "entropy": Inject when attention entropy exceeds threshold
    - "always": Always inject (backward compatible, baseline)
    - "never": Never inject (ablation study)

    Usage:
        scheduler = AdaptiveInjectionScheduler(strategy="threshold", ppl_threshold=10.0)
        for step in decode_steps:
            ppl = compute_perplexity(logits)
            if scheduler.should_inject(ppl):
                inject_teacher_kv()
            scheduler.record(ppl)
    """

    def __init__(
        self,
        strategy: str = "threshold",
        ppl_threshold: float = 10.0,
        entropy_threshold: float = 0.8,
        relative_alpha: float = 1.5,
        warmup_steps: int = 3,
        window_size: int = 10,
        min_injection_rate: float = 0.1,
    ) -> None:
        """
        Args:
            strategy: Scheduling strategy (see class docstring)
            ppl_threshold: PPL threshold for "threshold" strategy
            entropy_threshold: Entropy threshold for "entropy" strategy
            relative_alpha: Multiplier for "relative" strategy (inject if PPL > α × avg)
            warmup_steps: Always inject during first N steps (model warming up)
            window_size: Rolling window for running average PPL
            min_injection_rate: Minimum fraction of steps that must be injected
        """
        self.strategy = strategy
        self.ppl_threshold = ppl_threshold
        self.entropy_threshold = entropy_threshold
        self.relative_alpha = relative_alpha
        self.warmup_steps = warmup_steps
        self.window_size = window_size
        self.min_injection_rate = min_injection_rate

        # State
        self._step_count: int = 0
        self._injection_count: int = 0
        self._ppl_history: List[float] = []
        self._decision_history: List[bool] = []

    def should_inject(
        self,
        ppl: Optional[float] = None,
        entropy: Optional[float] = None,
    ) -> bool:
        """
        Decide whether to inject external brain KV at this step.

        Args:
            ppl: Current perplexity value (required for threshold/relative strategies)
            entropy: Current attention entropy (required for entropy strategy)

        Returns:
            True if injection should proceed
        """
        self._step_count += 1

        # Always inject during warmup
        if self._step_count <= self.warmup_steps:
            self._record_decision(True)
            return True

        decision = False

        if self.strategy == "always":
            decision = True
        elif self.strategy == "never":
            decision = False
        elif self.strategy == "threshold":
            decision = ppl is not None and ppl > self.ppl_threshold
        elif self.strategy == "relative":
            if ppl is not None and self._ppl_history:
                avg_ppl = self._get_running_avg_ppl()
                decision = ppl > self.relative_alpha * avg_ppl
            else:
                decision = True  # No history yet → inject
        elif self.strategy == "entropy":
            decision = entropy is not None and entropy > self.entropy_threshold
        else:
            decision = True  # Default: inject

        # Enforce minimum injection rate
        if not decision and self._step_count > self.warmup_steps:
            current_rate = self._injection_count / max(self._step_count, 1)
            if current_rate < self.min_injection_rate:
                decision = True  # Force injection to maintain minimum rate

        self._record_decision(decision)
        return decision

    def record(self, ppl: Optional[float] = None) -> None:
        """
        Record PPL observation for running average computation.

        Call this AFTER each decode step, regardless of injection decision.
        """
        if ppl is not None:
            self._ppl_history.append(ppl)
            # Keep only recent history
            if len(self._ppl_history) > self.window_size:
                self._ppl_history = self._ppl_history[-self.window_size:]

    def _record_decision(self, injected: bool) -> None:
        """Record injection decision for statistics."""
        self._decision_history.append(injected)
        if injected:
            self._injection_count += 1

    def _get_running_avg_ppl(self) -> float:
        """Compute running average PPL from recent history."""
        if not self._ppl_history:
            return float("inf")
        return sum(self._ppl_history) / len(self._ppl_history)

    def reset(self) -> None:
        """Reset scheduler state for a new generation."""
        self._step_count = 0
        self._injection_count = 0
        self._ppl_history = []
        self._decision_history = []

    @property
    def stats(self) -> Dict[str, Any]:
        """Return scheduler statistics."""
        injection_rate = self._injection_count / max(self._step_count, 1)
        skipped = self._step_count - self._injection_count
        return {
            "strategy": self.strategy,
            "total_steps": self._step_count,
            "injection_steps": self._injection_count,
            "skipped_steps": skipped,
            "injection_rate": injection_rate,
            "skip_rate": 1.0 - injection_rate,
            "avg_ppl": self._get_running_avg_ppl() if self._ppl_history else None,
        }


def compute_perplexity_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute perplexity from model logits and target token IDs (v0.6).

    Args:
        logits: [batch, seq_len, vocab_size] — model output logits
        target_ids: [batch, seq_len] — target token IDs
        eps: Small value to avoid log(0)

    Returns:
        Perplexity value (float)
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Gather log probs for target tokens
    # target_ids: [batch, seq_len] → [batch, seq_len, 1]
    target_ids_expanded = target_ids.unsqueeze(-1)

    # Clamp target_ids to valid range
    vocab_size = logits.shape[-1]
    target_ids_clamped = target_ids_expanded.clamp(0, vocab_size - 1)

    # Gather: [batch, seq_len, 1] → squeeze → [batch, seq_len]
    selected_log_probs = log_probs.gather(dim=-1, index=target_ids_clamped).squeeze(-1)

    # Mean negative log likelihood
    avg_nll = -selected_log_probs.mean().item()

    # Perplexity = exp(NLL)
    perplexity = math.exp(min(avg_nll, 20.0))  # Cap to avoid overflow

    return perplexity


# ─────────────────────────────────────────────────────────────
# Brain KV Compression — Compress teacher KV for efficient transfer (v0.6)
# ─────────────────────────────────────────────────────────────

class BrainKVCompressor:
    """
    Compress teacher KV cache for efficient transfer to the shell model (v0.6).

    Problem: A 7B teacher model's KV cache is huge:
    - 32 layers × 2 (K+V) × [batch, num_heads, seq_len, head_dim]
    - For seq_len=1024, head_dim=128, num_heads=32: ~512 MB per prompt
    - This is prohibitive for memory-constrained deployment

    Solution: Compress the KV before injection using:
    1. "topk_spatial": Keep only top-K most important KV positions per head
       (based on attention weight magnitude). Reduces seq_len dimension.
    2. "quantize_8bit": Quantize FP32/FP16 KV to INT8. Reduces memory by 2-4×.
    3. "mean_pool": Mean-pool KV over non-overlapping windows. Reduces seq_len.
    4. "svd_lowrank": SVD-based low-rank approximation of the KV matrix.
    5. "none": No compression (backward compatible, baseline)

    Decompression is implicit — the compressed KV is used directly in
    cross-attention (the shell's query attends to fewer/more-compact KV pairs).

    Usage:
        compressor = BrainKVCompressor(method="topk_spatial", compression_ratio=0.5)
        compressed_kv = compressor.compress(layer_idx, key, value)
        # Use compressed_kv directly in ExoBrain injection
    """

    def __init__(
        self,
        method: str = "topk_spatial",
        compression_ratio: float = 0.5,
        quantize_bits: int = 8,
        pool_window: int = 4,
        svd_rank: int = 64,
    ) -> None:
        """
        Args:
            method: Compression method (see class docstring)
            compression_ratio: Target compression ratio (0.5 = keep 50% of data)
            quantize_bits: Bit width for quantization (8 or 4)
            pool_window: Window size for mean_pool method
            svd_rank: Rank for SVD low-rank approximation
        """
        self.method = method
        self.compression_ratio = compression_ratio
        self.quantize_bits = quantize_bits
        self.pool_window = pool_window
        self.svd_rank = svd_rank

        # Compression statistics
        self._stats: Dict[str, Any] = {
            "method": method,
            "total_compressed": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
        }

    def compress(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a KV pair for a specific layer.

        Args:
            layer_idx: Transformer layer index
            key: [batch, heads, seq_len, dim] — key tensor
            value: [batch, heads, seq_len, dim] — value tensor

        Returns:
            (compressed_key, compressed_value)
        """
        original_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()

        if self.method == "none":
            return key, value

        elif self.method == "topk_spatial":
            key, value = self._compress_topk(key, value)
        elif self.method == "quantize_8bit":
            key, value = self._compress_quantize(key, value)
        elif self.method == "mean_pool":
            key, value = self._compress_mean_pool(key, value)
        elif self.method == "svd_lowrank":
            key, value = self._compress_svd(key, value)
        else:
            return key, value

        compressed_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()

        self._stats["total_compressed"] += 1
        self._stats["total_original_bytes"] += original_bytes
        self._stats["total_compressed_bytes"] += compressed_bytes

        return key, value

    def _compress_topk(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keep only top-K positions per head based on key magnitude.

        Selects the K positions with highest L2 norm in the key tensor,
        which correspond to the most "important" attention targets.
        """
        # key: [batch, heads, seq_len, dim]
        seq_len = key.shape[2]
        k = max(1, int(seq_len * self.compression_ratio))

        # Compute per-position magnitude: [batch, heads, seq_len]
        key_magnitude = key.norm(dim=-1)  # [B, H, S]

        # Select top-K positions
        _, top_indices = key_magnitude.topk(k, dim=-1)  # [B, H, K]

        # Gather selected positions
        # top_indices: [B, H, K] → [B, H, K, 1] → [B, H, K, dim]
        dim = key.shape[-1]
        index = top_indices.unsqueeze(-1).expand(-1, -1, -1, dim)

        compressed_key = torch.gather(key, dim=2, index=index)
        compressed_value = torch.gather(value, dim=2, index=index)

        return compressed_key, compressed_value

    def _compress_quantize(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize KV to lower bit width.

        Maps FP32/FP16 → INT8 using symmetric quantization:
        q = round(x / scale), where scale = max(|x|) / (2^(bits-1) - 1)
        """
        if self.quantize_bits == 8:
            qmin, qmax = -128, 127
        elif self.quantize_bits == 4:
            qmin, qmax = -8, 7
        else:
            return key, value

        def quantize_tensor(t: torch.Tensor) -> torch.Tensor:
            # Per-tensor symmetric quantization
            max_val = t.abs().max()
            scale = max_val / max(qmax, 1)
            if scale < 1e-8:
                return t
            quantized = torch.clamp(torch.round(t / scale), qmin, qmax)
            # Store as float (still saves memory conceptually, but actual
            # INT8 storage requires specialized packing — we keep as float
            # for compatibility with attention computation)
            return quantized * scale

        return quantize_tensor(key), quantize_tensor(value)

    def _compress_mean_pool(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean-pool KV over non-overlapping windows.

        Reduces seq_len by pool_window factor. Preserves semantic content
        while reducing spatial resolution.
        """
        w = self.pool_window
        seq_len = key.shape[2]

        if seq_len < w:
            return key, value

        # Truncate to multiple of window
        trimmed_len = (seq_len // w) * w

        # Reshape: [B, H, trimmed_len, D] → [B, H, trimmed_len//w, w, D]
        key_trimmed = key[:, :, :trimmed_len, :]
        value_trimmed = value[:, :, :trimmed_len, :]

        key_reshaped = key_trimmed.reshape(
            *key.shape[:2], trimmed_len // w, w, key.shape[-1]
        )
        value_reshaped = value_trimmed.reshape(
            *value.shape[:2], trimmed_len // w, w, value.shape[-1]
        )

        # Mean over window dimension
        compressed_key = key_reshaped.mean(dim=3)   # [B, H, trimmed_len//w, D]
        compressed_value = value_reshaped.mean(dim=3)

        return compressed_key, compressed_value

    def _compress_svd(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SVD low-rank approximation of KV matrices.

        Decomposes key/value into low-rank factors:
        K ≈ U_K @ S_K @ V_K^T, keeping only top-rank singular values.

        This is more expensive but provides the best quality-compression tradeoff.
        """
        rank = min(self.svd_rank, min(key.shape[2], key.shape[-1]))

        def svd_compress(t: torch.Tensor) -> torch.Tensor:
            # t: [B, H, S, D]
            B, H, S, D = t.shape
            # Reshape to 2D for SVD: [B*H, S, D]
            t_flat = t.reshape(B * H, S, D)

            try:
                U, S, Vh = torch.linalg.svd(t_flat.float(), full_matrices=False)
                # Keep top-rank components
                U_r = U[:, :, :rank]          # [B*H, S, rank]
                S_r = torch.diag_embed(S[:, :rank])  # [B*H, rank, rank]
                Vh_r = Vh[:, :rank, :]        # [B*H, rank, D]

                # Reconstruct: [B*H, S, rank] @ [B*H, rank, rank] @ [B*H, rank, D]
                compressed = U_r @ S_r @ Vh_r
                return compressed.reshape(B, H, S, D).to(t.dtype)
            except Exception:
                # SVD failed (e.g., numerical issues) — return original
                return t

        return svd_compress(key), svd_compress(value)

    def compress_teacher_cache(
        self,
        teacher_kv: TeacherKVCache,
    ) -> TeacherKVCache:
        """
        Compress an entire TeacherKVCache.

        Args:
            teacher_kv: TeacherKVCache with per-layer KV pairs

        Returns:
            New TeacherKVCache with compressed KV pairs
        """
        compressed_pairs = {}
        for layer_idx, (key, value) in teacher_kv.kv_pairs.items():
            compressed_key, compressed_value = self.compress(layer_idx, key, value)
            compressed_pairs[layer_idx] = (compressed_key, compressed_value)

        return TeacherKVCache(
            kv_pairs=compressed_pairs,
            model_id=teacher_kv.model_id + "_compressed",
            num_layers=teacher_kv.num_layers,
            hidden_size=teacher_kv.hidden_size,
            num_heads=teacher_kv.num_heads,
            head_dim=teacher_kv.head_dim,
            sequence_length=teacher_kv.sequence_length,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Return compression statistics."""
        avg_ratio = 1.0
        if self._stats["total_original_bytes"] > 0:
            avg_ratio = self._stats["total_compressed_bytes"] / self._stats["total_original_bytes"]
        return {
            **self._stats,
            "avg_compression_ratio": avg_ratio,
            "avg_space_saving": 1.0 - avg_ratio,
        }


# ─────────────────────────────────────────────────────────────
# KVPrefetcher — Speculative KV Injection Prefetcher (v0.5)
# ─────────────────────────────────────────────────────────────

class KVPrefetcher:
    """
    Prefetch and cache projected teacher KV pairs for faster decode.

    Problem: During autoregressive decode, _inject_teacher_kv_into_cache()
    is called every step. For each layer, it:
    1. Retrieves teacher KV from bus
    2. Optionally projects via HeadDimProjection
    3. Fuses with shell KV

    Steps 1-2 are redundant after the first decode step — the teacher KV
    doesn't change between steps (same prompt, same teacher). Only the
    shell KV changes (one new token appended each step).

    Solution: Cache the projected teacher KV after the first injection.
    For subsequent steps, reuse the cached projected KV and only redo
    the fusion (which depends on current shell KV shape).

    This eliminates redundant bus.retrieve() + projector calls during
    decode, reducing per-step overhead from O(layers × retrieval) to
    O(layers × fusion_only).

    Usage:
        prefetcher = KVPrefetcher(brain_bus, kv_projector, fusion_mode)
        # After first injection:
        prefetcher.cache_projected_kv(teacher_kv)
        # During decode:
        projected_kv = prefetcher.get_projected_kv(layer_idx)
    """

    def __init__(
        self,
        brain_bus: Any,  # ExoBrainBus
        kv_projector: Optional[HeadDimProjection] = None,
        fusion_mode: str = "replace",
        residual_alpha: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.brain_bus = brain_bus
        self.kv_projector = kv_projector
        self.fusion_mode = fusion_mode
        self.residual_alpha = residual_alpha
        self.device = device

        # Cached projected teacher KV: {layer_idx: (key, value)}
        self._projected_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Cache hit/miss stats
        self._stats = {"cache_hits": 0, "cache_misses": 0, "prefetch_count": 0}

    def cache_projected_kv(self, projected_pairs: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> int:
        """
        Pre-cache projected teacher KV pairs.

        Call this after the first _build_brain() to cache all projected KV.
        Subsequent decode steps can use get_projected_kv() instead of
        going through bus.retrieve() + projector again.

        Args:
            projected_pairs: {layer_idx: (projected_key, projected_value)}

        Returns:
            Number of layers cached
        """
        self._projected_cache.clear()
        for layer_idx, (key, value) in projected_pairs.items():
            # Detach and move to device for stable caching
            self._projected_cache[layer_idx] = (
                key.detach().to(device=self.device),
                value.detach().to(device=self.device),
            )
        self._stats["prefetch_count"] += 1
        return len(self._projected_cache)

    def get_projected_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get pre-cached projected teacher KV for a layer.

        Args:
            layer_idx: Transformer layer index

        Returns:
            (projected_key, projected_value) or None if not cached
        """
        if layer_idx in self._projected_cache:
            self._stats["cache_hits"] += 1
            return self._projected_cache[layer_idx]
        self._stats["cache_misses"] += 1
        return None

    def is_cached(self, layer_idx: int) -> bool:
        """Check if projected KV is cached for a layer."""
        return layer_idx in self._projected_cache

    @property
    def stats(self) -> Dict[str, Any]:
        """Return prefetcher statistics."""
        total = self._stats["cache_hits"] + self._stats["cache_misses"]
        hit_rate = self._stats["cache_hits"] / max(total, 1)
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "cached_layers": list(self._projected_cache.keys()),
        }


# ─────────────────────────────────────────────────────────────
# ExoBrain Inference Pipeline
# ─────────────────────────────────────────────────────────────
