"""Knowledge bus, multi-teacher routing and cross-attention fusion."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import ExoBrainConfig
from .sources import KnowledgeSource

logger = logging.getLogger(__name__)


class MultiTeacherRouter:
    """
    Routes queries to the most suitable teacher model in a multi-teacher
    ExoBrain ensemble (v0.6).

    Problem: Different teacher models have different strengths:
    - A code model is better at programming queries
    - A math model is better at arithmetic
    - A general model is better at commonsense reasoning

    Instead of using all teachers equally (expensive) or picking one
    manually (rigid), the router dynamically selects the best teacher
    for each query based on:
    1. Similarity-based routing: Which teacher's KV most aligns with the query
    2. Perplexity-based routing: Which teacher produces lowest perplexity
    3. Entropy-based routing: Which teacher's attention is most confident
    4. Weighted ensemble: Blend multiple teachers with learned weights

    Usage:
        router = MultiTeacherRouter(
            teachers={"code": code_bus, "math": math_bus, "general": gen_bus},
            strategy="similarity",
        )
        best_kv = router.route(query, layer_idx)
    """

    def __init__(
        self,
        teachers: Optional[Dict[str, ExoBrainBus]] = None,
        strategy: str = "similarity",
        ensemble_weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        top_k_teachers: int = 2,
    ) -> None:
        """
        Args:
            teachers: {teacher_name: ExoBrainBus} — multiple teacher buses
            strategy: Routing strategy:
                - "similarity": Route to teacher with highest query-KV similarity
                - "ensemble": Weighted blend of all teachers' KV
                - "round_robin": Cycle through teachers (baseline)
                - "first_available": Use first teacher with a hit
            ensemble_weights: Optional manual weights for ensemble strategy
            temperature: Temperature for softmax routing (lower = sharper)
            top_k_teachers: Number of top teachers to blend in ensemble mode
        """
        self.teachers: Dict[str, ExoBrainBus] = teachers or {}
        self.strategy = strategy
        self.ensemble_weights = ensemble_weights or {}
        self.temperature = temperature
        self.top_k_teachers = top_k_teachers

        # Routing statistics
        self._stats: Dict[str, Any] = {
            "total_routes": 0,
            "teacher_hits": {name: 0 for name in self.teachers},
            "strategy": strategy,
        }
        # Round-robin counter
        self._rr_counter: int = 0

    def add_teacher(self, name: str, bus: ExoBrainBus) -> None:
        """Add a teacher model bus to the router."""
        self.teachers[name] = bus
        self._stats["teacher_hits"][name] = 0

    def remove_teacher(self, name: str) -> None:
        """Remove a teacher model bus from the router."""
        self.teachers.pop(name, None)
        self._stats["teacher_hits"].pop(name, None)

    def route(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Route a query to the best teacher and retrieve KV pairs.

        Args:
            query: [batch, heads, q_len, dim] — shell model query
            layer_idx: Current transformer layer index

        Returns:
            (key, value) from the best teacher, or None if all miss
        """
        self._stats["total_routes"] += 1

        if not self.teachers:
            return None

        if self.strategy == "similarity":
            return self._route_by_similarity(query, layer_idx)
        elif self.strategy == "ensemble":
            return self._route_ensemble(query, layer_idx)
        elif self.strategy == "round_robin":
            return self._route_round_robin(query, layer_idx)
        elif self.strategy == "first_available":
            return self._route_first_available(query, layer_idx)
        else:
            return self._route_first_available(query, layer_idx)

    def _route_by_similarity(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Route to the teacher whose KV has highest similarity with query."""
        best_sim = float("-inf")
        best_kv = None
        best_teacher = None

        # Use mean query for similarity computation
        query_mean = query.mean(dim=(1, 2)).float()  # [batch, dim]

        for name, bus in self.teachers.items():
            kv = bus.retrieve(query, layer_idx)
            if kv is None:
                continue

            key, value = kv
            # Compute similarity between query and key
            key_mean = key.mean(dim=(1, 2)).float()  # [batch, dim]

            # Handle dimension mismatch
            min_dim = min(query_mean.shape[-1], key_mean.shape[-1])
            sim = F.cosine_similarity(
                query_mean[..., :min_dim],
                key_mean[..., :min_dim],
                dim=-1,
            ).mean().item()

            if sim > best_sim:
                best_sim = sim
                best_kv = kv
                best_teacher = name

        if best_teacher is not None:
            self._stats["teacher_hits"][best_teacher] = self._stats["teacher_hits"].get(best_teacher, 0) + 1

        return best_kv

    def _route_ensemble(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Weighted ensemble of multiple teachers' KV pairs.

        Blends the top-K teachers by similarity using softmax weights.
        This provides richer knowledge than any single teacher.
        """
        teacher_sims: List[Tuple[str, Tuple[torch.Tensor, torch.Tensor], float]] = []

        query_mean = query.mean(dim=(1, 2)).float()  # [batch, dim]

        for name, bus in self.teachers.items():
            kv = bus.retrieve(query, layer_idx)
            if kv is None:
                continue

            key, value = kv
            key_mean = key.mean(dim=(1, 2)).float()
            min_dim = min(query_mean.shape[-1], key_mean.shape[-1])
            sim = F.cosine_similarity(
                query_mean[..., :min_dim],
                key_mean[..., :min_dim],
                dim=-1,
            ).mean().item()

            teacher_sims.append((name, kv, sim))

        if not teacher_sims:
            return None

        if len(teacher_sims) == 1:
            name, kv, _ = teacher_sims[0]
            self._stats["teacher_hits"][name] = self._stats["teacher_hits"].get(name, 0) + 1
            return kv

        # Select top-K teachers
        teacher_sims.sort(key=lambda x: x[2], reverse=True)
        top_teachers = teacher_sims[:self.top_k_teachers]

        # Compute softmax weights from similarities
        sims = torch.tensor([s for _, _, s in top_teachers])
        weights = F.softmax(sims / max(self.temperature, 1e-6), dim=0)

        # Weighted blend of KV pairs
        # Use the first teacher's KV as reference for shape
        ref_key, ref_value = top_teachers[0][1]
        blended_key = torch.zeros_like(ref_key)
        blended_value = torch.zeros_like(ref_value)

        for i, (name, (key, value), _) in enumerate(top_teachers):
            w = weights[i].item()
            # Handle shape mismatches (pad/truncate to reference)
            key_aligned = self._align_kv(key, ref_key.shape)
            value_aligned = self._align_kv(value, ref_value.shape)
            blended_key += w * key_aligned
            blended_value += w * value_aligned
            self._stats["teacher_hits"][name] = self._stats["teacher_hits"].get(name, 0) + 1

        return blended_key, blended_value

    def _route_round_robin(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Cycle through teachers in order."""
        teacher_names = list(self.teachers.keys())
        if not teacher_names:
            return None

        for i in range(len(teacher_names)):
            idx = (self._rr_counter + i) % len(teacher_names)
            name = teacher_names[idx]
            kv = self.teachers[name].retrieve(query, layer_idx)
            if kv is not None:
                self._rr_counter = (self._rr_counter + 1) % len(teacher_names)
                self._stats["teacher_hits"][name] = self._stats["teacher_hits"].get(name, 0) + 1
                return kv

        return None

    def _route_first_available(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Use the first teacher that has a hit."""
        for name, bus in self.teachers.items():
            kv = bus.retrieve(query, layer_idx)
            if kv is not None:
                self._stats["teacher_hits"][name] = self._stats["teacher_hits"].get(name, 0) + 1
                return kv
        return None

    def _align_kv(
        self,
        kv: torch.Tensor,
        ref_shape: torch.Size,
    ) -> torch.Tensor:
        """Align a KV tensor to match a reference shape (pad/truncate)."""
        if kv.shape == ref_shape:
            return kv

        # Handle sequence length mismatch
        result = kv
        if kv.shape[2] > ref_shape[2]:
            result = result[:, :, :ref_shape[2], :]
        elif kv.shape[2] < ref_shape[2]:
            pad_len = ref_shape[2] - kv.shape[2]
            padding = torch.zeros(*kv.shape[:2], pad_len, kv.shape[-1],
                                  dtype=kv.dtype, device=kv.device)
            result = torch.cat([result, padding], dim=2)

        # Handle dimension mismatch
        if result.shape[-1] > ref_shape[-1]:
            result = result[..., :ref_shape[-1]]
        elif result.shape[-1] < ref_shape[-1]:
            pad_dim = ref_shape[-1] - result.shape[-1]
            result = F.pad(result, (0, pad_dim))

        # Handle batch/head mismatch
        if result.shape[0] < ref_shape[0]:
            result = result.expand(ref_shape[0], -1, -1, -1)
        if result.shape[1] < ref_shape[1]:
            result = result.expand(-1, ref_shape[1], -1, -1)

        return result

    @property
    def stats(self) -> Dict[str, Any]:
        """Return router statistics."""
        return dict(self._stats)


class ExoBrainBus:
    """
    Unified knowledge retrieval bus for ExoBrain.

    Aggregates multiple knowledge sources and provides a single
    retrieve() interface. Supports priority ordering, caching,
    and automatic dimension projection.
    """

    def __init__(
        self,
        sources: Optional[List[KnowledgeSource]] = None,
        config: Optional[ExoBrainConfig] = None,
    ) -> None:
        self.sources: List[KnowledgeSource] = sources or []
        self.config = config or ExoBrainConfig()
        # KV cache per layer: {layer_idx: (K, V)}
        self._injected_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Statistics
        self._retrieve_count: int = 0
        self._hit_count: int = 0
        self._miss_count: int = 0

    def add_source(self, source: KnowledgeSource) -> None:
        """Add a knowledge source to the bus."""
        self.sources.append(source)

    def remove_source(self, name: str) -> None:
        """Remove a knowledge source by name."""
        self.sources = [s for s in self.sources if s.name != name]

    def inject_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Directly inject KV pairs for a specific layer.

        This bypasses retrieval and uses pre-computed KV directly.
        Useful for pre-loading teacher model KV.
        """
        self._injected_kv[layer_idx] = (key.detach(), value.detach())

    def retrieve(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve external KV pairs for the given query and layer.

        Tries sources in order, returning the first successful result.
        Also checks directly injected KV.

        Priority:
        0. Directly injected KV (highest priority, bypasses projection)
        1-N. Knowledge sources in registration order (lowest priority)

        Args:
            query: [batch, heads, q_len, dim]
            layer_idx: Current transformer layer index

        Returns:
            (external_key, external_value) or None
        """
        self._retrieve_count += 1

        # Priority 0: Check directly injected KV (already projected by caller)
        if layer_idx in self._injected_kv:
            self._hit_count += 1
            return self._injected_kv[layer_idx]

        # Priority 1-N: Try knowledge sources in order
        top_k = self.config.retrieval_top_k
        for source in self.sources:
            try:
                result = source.retrieve_kv(query, layer_idx, top_k=top_k)
                if result is not None:
                    ext_k, ext_v = result
                    # Auto-project dimensions if needed and dimensions still differ
                    # Note: Directly-injected KV is already projected, so we skip
                    # projection for it. Knowledge sources handle their own projection
                    # internally (e.g. LocalWeightSource), but if they return
                    # mismatched dimensions, we apply a safety pad/truncate here.
                    if self.config.auto_project:
                        d_query = query.shape[-1]
                        d_retrieved = ext_k.shape[-1]
                        if d_retrieved != d_query:
                            ext_k, ext_v = self._maybe_project(ext_k, ext_v, query)
                    self._hit_count += 1
                    return ext_k, ext_v
            except Exception:
                continue

        self._miss_count += 1
        return None

    def _maybe_project(
        self,
        ext_k: torch.Tensor,
        ext_v: torch.Tensor,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project external KV to match query dimensions if needed.

        Uses zero-padding / truncation as a safe fallback.
        For learned projection, use HeadDimProjection in exobrain_inference.py.
        """
        d_query = query.shape[-1]
        d_ext_k = ext_k.shape[-1]
        d_ext_v = ext_v.shape[-1]

        # Fast path: dimensions already match
        if d_ext_k == d_query and d_ext_v == d_query:
            return ext_k, ext_v

        # Project key
        if d_ext_k > d_query:
            ext_k = ext_k[..., :d_query]
        elif d_ext_k < d_query:
            ext_k = F.pad(ext_k, (0, d_query - d_ext_k))

        # Project value
        if d_ext_v > d_query:
            ext_v = ext_v[..., :d_query]
        elif d_ext_v < d_query:
            ext_v = F.pad(ext_v, (0, d_query - d_ext_v))

        return ext_k, ext_v

    def clear_injected(self) -> None:
        """Clear all directly injected KV pairs."""
        self._injected_kv.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / max(1, total)
        return {
            "sources": [s.name for s in self.sources],
            "retrieve_count": self._retrieve_count,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "injected_layers": list(self._injected_kv.keys()),
        }


# ─────────────────────────────────────────────────────────────
# Cross-Attention Fusion
# ─────────────────────────────────────────────────────────────

def cross_attention_fusion(
    query: torch.Tensor,
    external_key: torch.Tensor,
    external_value: torch.Tensor,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    dropout_seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute cross-attention between shell query and external KV.

    This is the core fusion operation: the shell model's query
    attends to externally provided key-value pairs.

    Args:
        query: [batch, heads, q_len, dim] — from shell model
        external_key: [batch, heads, kv_len, dim] — from external brain
        external_value: [batch, heads, kv_len, dim] — from external brain
        scale: Attention scale factor
        attn_mask: Optional attention mask
        dropout_p: Dropout probability for attention weights (default: 0.0)
        training: Whether in training mode (enables dropout)
        dropout_seed: Optional seed for reproducible dropout

    Returns:
        output: [batch, heads, q_len, dim] — fused attention output
    """
    d = query.shape[-1]
    scale_factor = float(scale) if scale is not None else (1.0 / math.sqrt(d))

    # Standard scaled dot-product attention
    logits = (query @ external_key.transpose(-2, -1)) * scale_factor

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            logits = logits.masked_fill(~attn_mask, float("-inf"))
        else:
            logits = logits + attn_mask

    weights = torch.softmax(logits, dim=-1)

    # Apply dropout to attention weights if in training mode
    if dropout_p > 0.0 and training:
        if dropout_seed is not None:
            torch.manual_seed(dropout_seed)
        # Dropout on attention weights (like attention dropout in transformers)
        weights = F.dropout(weights, p=dropout_p, training=training)

    return weights @ external_value


def compute_gate(
    query: torch.Tensor,
    external_key: torch.Tensor,
    temperature: float = 1.0,
    mode: str = "max_similarity",
    learned_proj: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Compute attention gate for gated fusion mode.

    The gate determines how much external brain knowledge to use
    vs. the shell model's own computation.

    v0.5: Supports per-head gating — each attention head independently
    decides how much external knowledge to incorporate. This is crucial
    because different heads attend to different patterns:
    - Some heads may be confident (low gate → trust shell)
    - Some heads may be uncertain (high gate → trust brain)

    Args:
        query: [batch, heads, q_len, dim]
        external_key: [batch, heads, kv_len, dim]
        temperature: Gate temperature (lower = sharper)
        mode: Gate computation mode:
            - "max_similarity": Use max similarity (default, fast)
            - "mean_similarity": Use mean similarity (smoother)
            - "learned": Use a learned projection (requires learned_proj)
            - "per_head_entropy": Use per-head attention entropy (v0.5)
        learned_proj: Optional learned projection module for "learned" mode.
            If provided, should map [batch, heads, q_len, dim] → [batch, heads, q_len, 1]

    Returns:
        gate: [batch, heads, q_len, 1] — values in [0, 1]
    """
    d = query.shape[-1]
    scale = 1.0 / math.sqrt(d)

    if mode == "learned" and learned_proj is not None:
        # Learned gate: pass query through a learned projection
        gate_logits = learned_proj(query)  # [b, h, q, 1]
        gate = torch.sigmoid(gate_logits)
        return gate

    if mode == "per_head_entropy":
        # v0.5: Per-head entropy-based gating
        # Compute attention logits for entropy estimation
        logits = (query @ external_key.transpose(-2, -1)) * scale  # [b, h, q, kv]
        # Convert to probabilities for entropy computation
        attn_weights = torch.softmax(logits, dim=-1)  # [b, h, q, kv]
        # Compute per-head entropy: H = -Σ p·log(p)
        eps = 1e-8
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)  # [b, h, q]
        # Normalize entropy to [0, 1] range (max entropy = log(kv_len))
        max_entropy = math.log(max(external_key.shape[2], 1))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy  # [b, h, q]
        else:
            normalized_entropy = torch.zeros_like(entropy)
        # Gate = normalized_entropy (high entropy → trust brain more)
        gate = torch.sigmoid(normalized_entropy / max(temperature, 1e-6))
        return gate.unsqueeze(-1)  # [b, h, q, 1]

    # Similarity-based gates
    logits = (query @ external_key.transpose(-2, -1)) * scale  # [b, h, q, kv]

    if mode == "mean_similarity":
        # Mean pooling over KV dimension — smoother but slower
        sim = logits.mean(dim=-1)  # [b, h, q]
    else:
        # Max pooling — default, faster and captures strongest match
        sim = logits.max(dim=-1).values  # [b, h, q]

    gate = torch.sigmoid(sim / max(temperature, 1e-6))  # [b, h, q]
    return gate.unsqueeze(-1)  # [b, h, q, 1]


# ─────────────────────────────────────────────────────────────
# P1: ExoBrainBackend — KV-Level Injection
# ─────────────────────────────────────────────────────────────
