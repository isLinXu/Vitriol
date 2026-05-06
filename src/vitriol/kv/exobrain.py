"""
ExoBrain: External Brain System for Heterogeneous Reasoning.

═══════════════════════════════════════════════════════════════
Core Concept — Heterogeneous Cognitive Alignment (v0.4+)
═══════════════════════════════════════════════════════════════

ExoBrain enables a lightweight "shell model" (0.1B params with real
weights) to reason using KV pairs from an external "brain" (7B+ model).

IMPORTANT: This is NOT the old "zero-weight shell" approach.
That approach is mathematically broken — a zero-weight model produces
random noise queries that cannot meaningfully attend to external KV.

The corrected architecture:
  - Shell Model: 0.1B tiny model with REAL, trainable weights
    (must have "question-asking ability" — generate meaningful Query)
  - ShellProjection: Thin learned layer (hidden_dim → brain_hidden_dim)
    (required for cognitive alignment between heterogeneous spaces)
  - External Brain: 7B+ model KV cache (knowledge source)

This is "borrowing a brain to give birth to a child":
cognitive interface alignment, not empty-shell grafting.

Key difference from RAG:
  - RAG injects knowledge at the embedding layer (text → tokens)
  - ExoBrain injects knowledge at the attention layer (KV pairs)
  - ExoBrain provides "memory patterns" not "reading material"

═══════════════════════════════════════════════════════════════
Architecture (v0.4 — Heterogeneous Cognitive Alignment)
═══════════════════════════════════════════════════════════════

    Shell Model (0.1B real weights)     External Brain (7B KV cache)
    ┌─────────────────────────┐         ┌─────────────────────────┐
    │ Layer 0  (real weights) │         │  Layer 0  KV cache      │
    │ Layer 1  (real weights) │──Query──│→ Layer 1  KV cache      │
    │ ...                     │  ↓      │→ ...                    │
    │ Layer N  (real weights) │  │      │→ Layer N  KV cache      │
    ├─────────────────────────┤  │      └─────────────────────────┘
    │ 🔑 ShellProjection      │  │                ↑
    │ (thin learned layer)    │  │                │
    ├─────────────────────────┤  │      ┌─────────┴──────────┐
    │ LM Head (real weights)  │  │      │  Cross-Attention   │
    └─────────────────────────┘  │      │  Fusion (replace /  │
                                 │      │   residual / gated) │
                                 │      └────────────────────┘
    Key question:
    "Can shell Query precisely hit external KV?"
    (This is the core validation target for demos)

═══════════════════════════════════════════════════════════════
Key-Layer Injection Strategy
═══════════════════════════════════════════════════════════════

Not all layers need external brain injection:

    L0-L2:  Surface encoding (lexical, syntax) — shell's own
    L3-L8:  Middle semantic (concepts, entities) — 🔑 KEY LAYERS
    L9-L14: High-level reasoning (logic, commonsense) — 🔑 KEY LAYERS
    L15-L20: Deep abstraction (generalization) — partial injection
    L21-L23: Output mapping (decoding) — shell's own

Use config.key_layers to specify which layers receive KV injection.

═══════════════════════════════════════════════════════════════
Fusion Modes
═══════════════════════════════════════════════════════════════

Shell model has real weights (f_θ(x) ≠ 0):

  Replace:  ŷ = I(K, x)                        — Full external brain
  Residual: ŷ = α·f_θ(x) + (1-α)·I(K, x)     — Shell + Brain blend
  Gated:    ŷ = g(x,K)·I(K,x) + (1-g)·f_θ(x) — Attention-gated fusion

With a trained shell + ShellProjection, all three modes are meaningful.

═══════════════════════════════════════════════════════════════
Integration Points
═══════════════════════════════════════════════════════════════

1. ExoBrainBackend — inherits KVStoreBackend, overrides read_attention()
   (injects external KV at decode step)

2. ExoBrainAttentionPatcher — extends UniversalAttentionPatcher
   (intercepts attention forward, applies fusion modes)

3. ExoBrainBus — unified knowledge retrieval interface
   (connects to Vector DB, API, local weights)

4. ShellProjection — cognitive alignment layer (v0.4+ new)
   (projects shell hidden space → brain hidden space)

═══════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════

    from vitriol.kv.exobrain import (
        ExoBrainBackend, ExoBrainBus, ExoBrainConfig,
        ExoBrainAttentionPatcher, ShellProjection,
        VectorDBSource, APIKnowledgeSource, LocalWeightSource,
    )

    # 1. Create knowledge sources
    vector_db = VectorDBSource(embeddings=..., texts=...)
    api_source = APIKnowledgeSource(endpoint="http://...")

    # 2. Create ShellProjection (thin alignment layer)
    shell_proj = ShellProjection(
        shell_hidden_dim=768,
        brain_hidden_dim=4096,
        mode="linear",  # thin — only ~3M params
    )

    # 3. Create ExoBrain bus
    bus = ExoBrainBus(sources=[vector_db, api_source])

    # 4. Create ExoBrain backend with key-layer config
    config = ExoBrainConfig(
        fusion_mode="replace",
        key_layers=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    )
    backend = ExoBrainBackend(
        store_cfg=kv_store_cfg,
        brain_bus=bus,
        brain_cfg=config,
        shell_projection=shell_proj,
    )

    # 5. Patch attention for full pipeline
    patcher = ExoBrainAttentionPatcher(backend=backend, brain_bus=bus, brain_cfg=config)
    patcher.apply()

    # Core validation question:
    # "Can shell model's Query precisely hit external KV?"
    # Run: python -m vitriol.demos.exobrain_query_hit_demo
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Protocol

import torch
import torch.nn.functional as F

from .backend import KVStoreBackend
from .cache_store import KVCacheStoreConfig

# Optional HTTP client for APIKnowledgeSource
try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ShellProjection — Cognitive Alignment Layer (v0.4+)
# ─────────────────────────────────────────────────────────────

class ShellProjection(torch.nn.Module):
    """
    Thin cognitive alignment layer between shell model and external brain.

    PURPOSE:
    When the shell model (0.1B) and external brain (7B+) have different
    hidden dimensions, ShellProjection provides a lightweight learned mapping
    so that the shell's queries can semantically align with the brain's KV space.

    This is NOT optional — without cognitive alignment, the shell's query
    (e.g., 768-dim) cannot meaningfully attend to brain's KV (e.g., 4096-dim).

    MODES:
    - "linear": Single linear layer (thin, ~3M params for 768→4096)
    - "mlp": Linear → GELU → Linear (slightly more expressive)
    - "linear_ln": Linear → LayerNorm (stable for training)

    DESIGN PRINCIPLE:
    Keep it thin! The projection should be ~1% of shell model size.
    Weights are trained via Feature Alignment Distillation.

    EXAMPLE:
        shell_hidden_dim = 768    # 0.1B model
        brain_hidden_dim = 4096   # 7B model
        projection = ShellProjection(768, 4096, mode="linear")
        # ~3M parameters (768 * 4096 / projection_ratio)
    """

    def __init__(
        self,
        shell_hidden_dim: int,
        brain_hidden_dim: int,
        mode: str = "linear",
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.shell_hidden_dim = shell_hidden_dim
        self.brain_hidden_dim = brain_hidden_dim
        self.mode = mode

        if mode == "linear":
            self.proj = torch.nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias)
        elif mode == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(shell_hidden_dim, shell_hidden_dim, bias=bias),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias),
            )
        elif mode == "linear_ln":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(shell_hidden_dim, brain_hidden_dim, bias=bias),
                torch.nn.LayerNorm(brain_hidden_dim),
                torch.nn.Dropout(p=dropout),
            )
        else:
            raise ValueError(f"ShellProjection: unknown mode '{mode}'. Use: linear, mlp, linear_ln")

        # Initialize with small std (near identity mapping is a good start)
        self._init_near_identity()

    def _init_near_identity(self) -> None:
        """Initialize projection near identity for stable training start."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                # Small std — reduces initial distortion
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project shell hidden states → brain hidden space.

        Args:
            x: Shell hidden states [batch, seq, shell_hidden_dim]
               or [batch, heads, seq, head_dim]

        Returns:
            Projected tensor [batch, seq, brain_hidden_dim]
             or [batch, heads, seq, brain_head_dim]
        """
        original_shape = x.shape
        ndims = len(original_shape)

        # Normalize to [batch*heads, seq, dim] for projection
        if ndims == 4:
            # [B, H, S, D] → [B, H*S, D] → project → [B, H*S, brain_d]
            B, H, S, D = original_shape
            x = x.reshape(B, H * S, D)
            x = self.proj(x)
            # Return [B, H, S, brain_d]
            brain_d = self.brain_hidden_dim
            return x.reshape(B, H, S, brain_d)
        elif ndims == 3:
            # [B, S, D] → project → [B, S, brain_d]
            return self.proj(x)
        else:
            raise ValueError(
                f"ShellProjection: expected 3D [B,S,D] or 4D [B,H,S,D], got {ndims}D"
            )

    def project_query(self, query: torch.Tensor) -> torch.Tensor:
        """Convenience: project query tensor to brain space."""
        return self.forward(query)

    def project_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Convenience: project KV tensor to brain space."""
        return self.forward(kv)

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters in this projection."""
        return sum(p.numel() for p in self.parameters())

    @property
    def parameter_count_str(self) -> str:
        """Human-readable parameter count."""
        n = self.num_parameters
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.2f}K"
        return str(n)


# ─────────────────────────────────────────────────────────────
# Knowledge Sources — Protocols & Implementations
# ─────────────────────────────────────────────────────────────

class KnowledgeSource(Protocol):
    """Protocol for external knowledge sources."""

    def retrieve_kv(
        self,
        query: torch.Tensor,
        layer_idx: int,
        top_k: int = 5,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve external KV pairs relevant to the query.

        Args:
            query: [batch, heads, q_len, dim] — current query tensor
            layer_idx: Current transformer layer index
            top_k: Number of top-K results to retrieve

        Returns:
            (external_key, external_value) or None
            Shape: [batch, heads, top_k, dim]
        """
        ...

    @property
    def name(self) -> str:
        """Source identifier."""
        ...


class VectorDBSource:
    """
    Vector database knowledge source.

    Pre-computed embeddings are stored in a simple in-memory index.
    At retrieve time, we compute query → top-k retrieval and return
    the corresponding KV pairs.

    This is a lightweight implementation suitable for demos.
    For production, replace with FAISS / ChromaDB / Milvus.

    Enhanced with:
    - Dynamic index updates (add/remove documents)
    - Per-document metadata storage
    - Automatic embedding recomputation from KV
    """

    def __init__(
        self,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        name: str = "vector_db",
    ) -> None:
        self._keys = keys       # [n_docs, dim] or [n_docs, n_heads, top_k, dim]
        self._values = values   # [n_docs, dim] or [n_docs, n_heads, top_k, dim]
        self._embeddings = embeddings  # [n_docs, embed_dim]
        self._texts = texts or []
        self._metadata = metadata or []
        self._name = name
        self._stats = {"retrievals": 0, "hits": 0}

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_docs(self) -> int:
        """Return the number of documents in the index."""
        return self._embeddings.shape[0] if self._embeddings is not None else 0

    def add_document(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        embedding: torch.Tensor,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a new document to the index.

        Args:
            key: KV key tensor [heads, kv_len, dim] or [kv_len, dim]
            value: KV value tensor [heads, kv_len, dim] or [kv_len, dim]
            embedding: Document embedding vector [embed_dim]
            text: Optional text content
            metadata: Optional metadata dict

        Returns:
            New document index
        """
        if self._keys is None:
            # First document — initialize
            if key.ndim == 3:
                # [heads, kv_len, dim] → unsqueeze to [1, heads, kv_len, dim]
                self._keys = key.unsqueeze(0)
                self._values = value.unsqueeze(0)
            else:
                # [kv_len, dim] → expand to [1, kv_len, dim]
                self._keys = key.unsqueeze(0).unsqueeze(0)
                self._values = value.unsqueeze(0).unsqueeze(0)
            self._embeddings = embedding.unsqueeze(0)
        else:
            if key.ndim == 3:
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
            else:
                key = key.unsqueeze(0).unsqueeze(0)
                value = value.unsqueeze(0).unsqueeze(0)
            self._keys = torch.cat([self._keys, key], dim=0)
            self._values = torch.cat([self._values, value], dim=0)
            self._embeddings = torch.cat([self._embeddings, embedding.unsqueeze(0)], dim=0)

        self._texts.append(text or "")
        self._metadata.append(metadata or {})
        return len(self._texts) - 1

    def remove_document(self, index: int) -> bool:
        """Remove a document by index. Returns True if successful."""
        if self._keys is None or index < 0 or index >= self.num_docs:
            return False
        self._keys = torch.cat([self._keys[:index], self._keys[index + 1:]], dim=0)
        self._values = torch.cat([self._values[:index], self._values[index + 1:]], dim=0)
        self._embeddings = torch.cat([self._embeddings[:index], self._embeddings[index + 1:]], dim=0)
        self._texts.pop(index)
        self._metadata.pop(index)
        return True

    def recompute_embeddings(self, normalize: bool = True) -> None:
        """
        Recompute embeddings from stored KV keys (mean pooling).

        Uses L2 normalization if normalize=True.
        """
        if self._keys is None:
            return
        if self._keys.ndim == 4:
            # [n_docs, n_heads, seq, dim] → mean over heads, seq
            emb = self._keys.mean(dim=(1, 2))  # [n_docs, dim]
        else:
            # [n_docs, kv_len, dim] → mean over seq
            emb = self._keys.mean(dim=1)  # [n_docs, dim]
        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        self._embeddings = emb

    def retrieve_kv(
        self,
        query: torch.Tensor,
        layer_idx: int,
        top_k: int = 5,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._keys is None or self._embeddings is None:
            return None

        self._stats["retrievals"] += 1

        # query: [batch, heads, q_len, dim]
        # Use mean over q_len and heads as query embedding
        query_emb = query.mean(dim=(1, 2)).float()  # [batch, dim]

        # Compute similarity: query_emb @ embeddings^T
        # embeddings: [n_docs, embed_dim]
        if query_emb.shape[-1] != self._embeddings.shape[-1]:
            # Project to match embedding dimension
            min_dim = min(query_emb.shape[-1], self._embeddings.shape[-1])
            q_proj = query_emb[..., :min_dim]
            e_proj = self._embeddings[..., :min_dim]
        else:
            q_proj = query_emb
            e_proj = self._embeddings

        # [batch, n_docs]
        similarity = q_proj @ e_proj.t()

        # Top-k indices
        k = min(top_k, similarity.shape[-1])
        _, indices = similarity.topk(k, dim=-1)  # [batch, k]

        # Retrieve KV pairs
        batch_size = query.shape[0]
        n_heads = query.shape[1]
        target_dim = query.shape[-1]

        # Gather from stored keys/values
        ext_k_list = []
        ext_v_list = []
        for b in range(batch_size):
            idx = indices[b]  # [k]
            if self._keys.ndim == 4:
                # [n_docs, n_heads, seq, dim]
                k_gathered = self._keys[idx]  # [k, n_heads, seq, dim]
                v_gathered = self._values[idx]
                # Take mean over seq to get [n_heads, k, dim]
                k_gathered = k_gathered.mean(dim=2).transpose(0, 1)  # [n_heads, k, dim]
                v_gathered = v_gathered.mean(dim=2).transpose(0, 1)
            else:
                # [n_docs, kv_len, dim] → mean-pool over kv_len → [n_docs, dim]
                # For each top-k result, broadcast across n_heads
                k_gathered = self._keys[idx]  # [k, kv_len, dim]
                v_gathered = self._values[idx]
                if k_gathered.ndim == 3:
                    # [k, kv_len, dim] → mean over kv_len → [k, dim]
                    k_gathered = k_gathered.mean(dim=1)
                    v_gathered = v_gathered.mean(dim=1)
                else:
                    # [k, dim] → mean over k (same embedding replicated) → [dim]
                    k_gathered = k_gathered.mean(dim=0)
                    v_gathered = v_gathered.mean(dim=0)
                # Unsqueeze + expand: [dim] → [1, dim] → [n_heads, dim]
                k_gathered = k_gathered.unsqueeze(0).expand(n_heads, -1)  # [n_heads, dim]
                v_gathered = v_gathered.unsqueeze(0).expand(n_heads, -1)
                # Tile for k results: [n_heads, dim] → [n_heads, k, dim]
                k_gathered = k_gathered.unsqueeze(1).expand(-1, k, -1)
                v_gathered = v_gathered.unsqueeze(1).expand(-1, k, -1)

            ext_k_list.append(k_gathered)
            ext_v_list.append(v_gathered)

        ext_k = torch.stack(ext_k_list, dim=0)  # [batch, n_heads, k, dim]
        ext_v = torch.stack(ext_v_list, dim=0)  # [batch, n_heads, k, dim]

        # Ensure dimension matches query
        if ext_k.shape[-1] != target_dim:
            ext_k = self._pad_or_truncate(ext_k, target_dim)
            ext_v = self._pad_or_truncate(ext_v, target_dim)

        self._stats["hits"] += 1
        return ext_k, ext_v

    def _pad_or_truncate(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pad or truncate last dimension to target_dim."""
        current_dim = tensor.shape[-1]
        if current_dim > target_dim:
            return tensor[..., :target_dim]
        elif current_dim < target_dim:
            pad = target_dim - current_dim
            return F.pad(tensor, (0, pad))
        return tensor

    def get_document(self, index: int) -> Optional[Dict[str, Any]]:
        """Get document info by index."""
        if index < 0 or index >= self.num_docs:
            return None
        return {
            "text": self._texts[index] if index < len(self._texts) else None,
            "metadata": self._metadata[index] if index < len(self._metadata) else {},
            "embedding_norm": float(self._embeddings[index].norm()) if self._embeddings is not None else 0.0,
        }

    def search_texts(self, query_texts: List[str], top_k: int = 5) -> List[List[int]]:
        """
        Simple text-based search (fallback when embeddings not available).

        Returns list of doc indices for each query text.
        """
        if not self._texts:
            return [[] for _ in query_texts]
        results = []
        for q in query_texts:
            q_lower = q.lower()
            scores = [sum(1 for w in q_lower.split() if w in t.lower()) for t in self._texts]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results.append(top_indices)
        return results

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


class APIKnowledgeSource:
    """
    External API knowledge source.

    Calls an external LLM or knowledge API to generate KV pairs
    for the given query. This is the "API-as-brain" mode.

    Supports both synchronous (requests) and asynchronous (aiohttp) modes.
    For production use with high-throughput, use the async interface.

    Example usage:
        # Sync mode
        source = APIKnowledgeSource(
            endpoint="http://localhost:8000/kv",
            api_key="secret-key",
            model_name="teacher-llm",
        )
        result = source.retrieve_kv(query, layer_idx=0)

        # Async mode
        result = await source.retrieve_kv_async(query, layer_idx=0)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "external-brain",
        name: str = "api_source",
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 8,
        session: Optional[Any] = None,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model_name = model_name
        self._name = name
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._session = session
        # Cache for repeated queries (keyed by query hash)
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Cache TTL: {key: (kv_tuple, expiry_time)}
        self._cache_ttl: Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], float]] = {}
        self._cache_ttl_seconds: float = 3600.0  # 1 hour default
        # Separate storage for manually injected KV (not query-keyed)
        self._injected_kv: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._stats: Dict[str, int] = {"api_calls": 0, "cache_hits": 0, "errors": 0, "injected_hits": 0}

    @property
    def name(self) -> str:
        return self._name

    @property
    def stats(self) -> Dict[str, int]:
        """Return API call statistics."""
        return dict(self._stats)

    def set_cache_ttl(self, seconds: float) -> None:
        """Set TTL for cached KV pairs in seconds."""
        self._cache_ttl_seconds = max(0.0, seconds)

    def _query_to_cache_key(self, query: torch.Tensor) -> str:
        """Generate a cache key from a query tensor."""
        data = query.cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cached entry is still valid."""
        if key not in self._cache_ttl:
            return key in self._cache
        _, expiry = self._cache_ttl[key]
        return time.time() < expiry

    def retrieve_kv(
        self,
        query: torch.Tensor,
        layer_idx: int,
        top_k: int = 5,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve KV pairs from the external API (synchronous mode).

        Returns cached results if available and not expired.
        Falls back to HTTP POST request if endpoint is configured.
        Injected KV takes priority when present (matches by user-specified key).
        """
        cache_key = self._query_to_cache_key(query)

        # Check cache (hash-keyed) first
        if self._is_cache_valid(cache_key):
            self._stats["cache_hits"] += 1
            return self._cache.get(cache_key)

        # Fall back to injected KV (user-specified keys)
        if self._injected_kv:
            # Return first available injected entry (for layer-agnostic retrieval)
            self._stats["injected_hits"] += 1
            return list(self._injected_kv.values())[0]

        # If no endpoint configured, return None
        if not self._endpoint:
            return None

        # Build request payload
        query_np = query.cpu().float().numpy()
        payload = {
            "query": query_np.tolist(),
            "layer_idx": layer_idx,
            "top_k": top_k,
            "model": self._model_name,
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Retry loop
        for attempt in range(self._max_retries):
            try:
                if _HAS_REQUESTS:
                    response = requests.post(
                        self._endpoint,
                        json=payload,
                        headers=headers,
                        timeout=self._timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                else:
                    import urllib.request
                    import urllib.error
                    body = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        self._endpoint,
                        data=body,
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8"))

                self._stats["api_calls"] += 1

                # Parse response: expect {"keys": [...], "values": [...]}
                keys_data = data.get("keys", [])
                values_data = data.get("values", [])

                if not keys_data or not values_data:
                    return None

                keys_tensor = torch.tensor(keys_data, dtype=query.dtype)
                values_tensor = torch.tensor(values_data, dtype=query.dtype)

                # Cache the result
                self._cache[cache_key] = (keys_tensor, values_tensor)
                self._cache_ttl[cache_key] = (
                    (keys_tensor, values_tensor),
                    time.time() + self._cache_ttl_seconds,
                )

                return keys_tensor, values_tensor

            except Exception:
                self._stats["errors"] += 1
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # All retries failed
        return None

    async def retrieve_kv_async(
        self,
        query: torch.Tensor,
        layer_idx: int,
        top_k: int = 5,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve KV pairs from the external API (async mode).

        Requires aiohttp to be installed. Much more efficient for
        high-throughput scenarios as it reuses connections.
        """
        cache_key = self._query_to_cache_key(query)

        # Check cache first
        if self._is_cache_valid(cache_key):
            self._stats["cache_hits"] += 1
            cached = self._cache.get(cache_key)
            return cached if cached else None

        if not self._endpoint:
            return None

        if not _HAS_AIOHTTP:
            # Fall back to sync mode
            return self.retrieve_kv(query, layer_idx, top_k)

        query_np = query.cpu().float().numpy()
        payload = {
            "query": query_np.tolist(),
            "layer_idx": layer_idx,
            "top_k": top_k,
            "model": self._model_name,
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(self._max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self._endpoint,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self._timeout),
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

                self._stats["api_calls"] += 1

                keys_data = data.get("keys", [])
                values_data = data.get("values", [])

                if not keys_data or not values_data:
                    return None

                keys_tensor = torch.tensor(keys_data, dtype=query.dtype)
                values_tensor = torch.tensor(values_data, dtype=query.dtype)

                # Cache the result
                self._cache[cache_key] = (keys_tensor, values_tensor)
                self._cache_ttl[cache_key] = (
                    (keys_tensor, values_tensor),
                    time.time() + self._cache_ttl_seconds,
                )

                return keys_tensor, values_tensor

            except Exception:
                self._stats["errors"] += 1
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        return None

    def inject_kv(
        self,
        cache_key: str,
        key: torch.Tensor,
        value: torch.Tensor,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Manually inject KV pairs into the cache.

        Args:
            cache_key: Unique key for this KV entry
            key: Key tensor [batch, heads, kv_len, dim]
            value: Value tensor [batch, heads, kv_len, dim]
            ttl: Optional TTL in seconds (default: self._cache_ttl_seconds)

        Note: KV is stored both under the user-specified cache_key AND
        under a hash-derived key (via _query_to_cache_key of the key tensor).
        This allows retrieve_kv() to find manually injected KV without
        knowing the original cache_key.
        """
        self._cache[cache_key] = (key, value)
        expiry = time.time() + (ttl if ttl is not None else self._cache_ttl_seconds)
        self._cache_ttl[cache_key] = ((key, value), expiry)
        # Also store in injected dict with user key for direct lookup
        self._injected_kv[cache_key] = (key, value)

    def clear_cache(self) -> None:
        """Clear all cached KV pairs."""
        self._cache.clear()
        self._cache_ttl.clear()

    def clear_expired(self) -> int:
        """Remove expired entries from cache. Returns count of removed entries."""
        now = time.time()
        expired_keys = [
            k for k, (_, expiry) in list(self._cache_ttl.items()) if now >= expiry
        ]
        for k in expired_keys:
            self._cache.pop(k, None)
            self._cache_ttl.pop(k, None)
        return len(expired_keys)


class LocalWeightSource:
    """
    Local weight knowledge source.

    Uses pre-computed weights from a teacher model as external knowledge.
    Implements the "Weight Distill" path — projecting teacher weights
    into the shell model's parameter space.
    """

    def __init__(
        self,
        teacher_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        projection_matrix: Optional[torch.Tensor] = None,
        name: str = "local_weight",
    ) -> None:
        self._teacher_kv = teacher_kv or {}  # {layer_idx: (K, V)}
        self._projection = projection_matrix  # [d_teacher, d_shell]
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def set_teacher_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Set teacher KV for a specific layer."""
        self._teacher_kv[layer_idx] = (key, value)

    def retrieve_kv(
        self,
        query: torch.Tensor,
        layer_idx: int,
        top_k: int = 5,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if layer_idx not in self._teacher_kv:
            return None

        t_key, t_value = self._teacher_kv[layer_idx]

        # If dimensions don't match, project
        if self._projection is not None:
            d_q = query.shape[-1]
            d_t = t_key.shape[-1]

            if d_q != d_t:
                # Project teacher KV to shell dimension
                # projection: [d_teacher, d_shell]
                proj = self._projection.to(t_key.device, t_key.dtype)
                # t_key: [..., d_teacher] → [..., d_shell]
                shape = t_key.shape[:-1]
                t_key = (t_key.reshape(-1, d_t) @ proj).reshape(*shape, d_q)
                t_value = (t_value.reshape(-1, d_t) @ proj).reshape(*shape, d_q)

        # Select top-k by query-key similarity
        # query: [batch, heads, q_len, d]
        # t_key: [batch, heads, seq_len, d] (from teacher)
        scale = 1.0 / math.sqrt(query.shape[-1])
        # Use mean query for retrieval
        q_mean = query.mean(dim=2, keepdim=True)  # [b, h, 1, d]
        logits = (q_mean @ t_key.transpose(-2, -1)) * scale  # [b, h, 1, seq]
        logits = logits.squeeze(2)  # [b, h, seq]

        k = min(top_k, logits.shape[-1])
        _, top_indices = logits.topk(k, dim=-1)  # [b, h, k]

        # Gather
        b, h, _, d = query.shape
        ext_k = torch.gather(
            t_key.expand(b, -1, -1, -1) if t_key.ndim == 3 else t_key,
            dim=2,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, d),
        )
        ext_v = torch.gather(
            t_value.expand(b, -1, -1, -1) if t_value.ndim == 3 else t_value,
            dim=2,
            index=top_indices.unsqueeze(-1).expand(-1, -1, -1, d),
        )

        return ext_k, ext_v


# ─────────────────────────────────────────────────────────────
# ExoBrain Bus — Unified Knowledge Retrieval Interface
# ─────────────────────────────────────────────────────────────

@dataclass
class ExoBrainConfig:
    """
    Configuration for the ExoBrain system (v0.4+ heterogenous reasoning).

    IMPORTANT: The shell model MUST have real, trainable weights.
    Zero-weight shells cannot generate meaningful queries for KV retrieval.
    """

    # Fusion mode: "replace", "residual", "gated"
    fusion_mode: str = "replace"

    # Alpha for residual fusion: ŷ = α·shell + (1-α)·brain
    residual_alpha: float = 0.1

    # Gate temperature for gated fusion
    gate_temperature: float = 1.0

    # v0.5: Gate computation mode for gated fusion:
    # - "max_similarity": Default, fast
    # - "mean_similarity": Smoother
    # - "per_head_entropy": Per-head attention entropy (v0.5)
    # - "learned": Learned projection (requires external module)
    gate_mode: str = "max_similarity"

    # Number of top-K external KV pairs to retrieve
    retrieval_top_k: int = 5

    # Whether to use cross-attention for KV injection
    use_cross_attention: bool = True

    # Dimension projection: if external KV has different dim than shell
    auto_project: bool = True

    # Key layers for KV injection (CognitiveAlignmentStrategy).
    # Only these layers receive external brain KV injection.
    # Middle semantic layers (3-8) and high-level reasoning (9-14) are typical.
    # Empty list = all layers (backward compatible).
    key_layers: List[int] = field(default_factory=list)

    # Alias for key_layers (backward compatibility)
    @property
    def active_layers(self) -> List[int]:
        """Alias for key_layers (deprecated, use key_layers)."""
        return self.key_layers

    @active_layers.setter
    def active_layers(self, value: List[int]) -> None:
        self.key_layers = value

    # Number of KV pairs to inject per key layer
    kv_injection_top_k: int = 5

    # Injection strength for residual/gated modes (0.0-1.0)
    injection_strength: float = 1.0

    # Whether to fall back to standard attention on brain failure
    fallback_on_error: bool = True

    # Confidence threshold: skip brain if query norm is below this
    min_query_norm: float = 1e-6

    # ── v0.5: Adaptive Layer Selection ──────────────────────────────
    # Strategy for selecting key layers:
    # - "manual": Use key_layers list (backward compatible, default)
    # - "entropy_top_k": Select top-K layers by attention entropy
    # - "entropy_threshold": Select layers above entropy threshold
    # - "middle_heavy": Prioritize middle layers (heuristic)
    # - "all": Inject all layers
    layer_selection_strategy: str = "manual"

    # Ratio of layers to select (for entropy_top_k strategy)
    layer_selection_top_k_ratio: float = 0.5

    # Entropy threshold (for entropy_threshold strategy)
    layer_selection_entropy_threshold: float = 0.7

    # Minimum number of layers to select
    layer_selection_min_layers: int = 4

    def __post_init__(self) -> None:
        valid_modes = {"replace", "residual", "gated"}
        if self.fusion_mode not in valid_modes:
            raise ValueError(
                f"ExoBrain: invalid fusion_mode '{self.fusion_mode}'. "
                f"Choose from: {valid_modes}"
            )

    def is_key_layer(self, layer_idx: int) -> bool:
        """
        Check if a layer is a key layer for injection.

        Returns True if:
        - key_layers is empty (all layers are key layers, backward compatible)
        - OR layer_idx is explicitly in key_layers
        - OR layer_selection_strategy is not "manual" and the adaptive selector includes it

        Note: For non-"manual" strategies, the caller should use
        AdaptiveLayerSelector.select() to determine key layers.
        """
        if self.layer_selection_strategy != "manual":
            # Adaptive mode: delegate to caller with selector
            # Fall back to all layers if no selector is used
            if not self.key_layers:
                return True  # All layers until selector is configured
        if not self.key_layers:
            return True  # All layers
        return layer_idx in self.key_layers


# ─────────────────────────────────────────────────────────────
# Adaptive Layer Selector (v0.5)
# ─────────────────────────────────────────────────────────────

class AdaptiveLayerSelector:
    """
    Selects key layers for ExoBrain KV injection based on attention entropy.

    Insight: Not all layers benefit equally from external brain injection.
    Layers with high attention entropy (diffuse, uncertain attention) benefit
    more from external guidance than layers with low entropy (confident, focused).

    Strategy:
    - Compute per-layer attention entropy from the shell model's forward pass
    - Rank layers by entropy (descending)
    - Select top-K layers or layers above a threshold
    - Cache the selection for reuse across prompts (stable selection)

    This replaces the old manual key_layers config with data-driven selection.

    Usage:
        selector = AdaptiveLayerSelector(
            total_layers=32,
            strategy="entropy_top_k",
            top_k_ratio=0.5,  # select top 50% layers
        )
        # After observing attention patterns:
        selector.observe(entropy_per_layer)
        key_layers = selector.select()
    """

    def __init__(
        self,
        total_layers: int = 0,
        strategy: str = "entropy_top_k",
        top_k_ratio: float = 0.5,
        entropy_threshold: float = 0.7,
        min_layers: int = 4,
        max_layers: Optional[int] = None,
        stability_window: int = 3,
    ) -> None:
        """
        Args:
            total_layers: Total number of transformer layers
            strategy: Selection strategy:
                - "entropy_top_k": Select top-K layers by entropy
                - "entropy_threshold": Select layers above entropy threshold
                - "middle_heavy": Prioritize middle layers (default heuristic)
                - "all": Select all layers (backward compatible)
            top_k_ratio: Fraction of layers to select (for entropy_top_k)
            entropy_threshold: Entropy threshold (for entropy_threshold strategy)
            min_layers: Minimum number of layers to select
            max_layers: Maximum number of layers to select (None = no limit)
            stability_window: Number of observations before selection stabilizes
        """
        self.total_layers = total_layers
        self.strategy = strategy
        self.top_k_ratio = top_k_ratio
        self.entropy_threshold = entropy_threshold
        self.min_layers = min_layers
        self.max_layers = max_layers or total_layers
        self.stability_window = stability_window

        # Entropy history: List[Dict[int, float]] — per-observation entropy
        self._entropy_history: List[Dict[int, float]] = []
        # Cached selection
        self._cached_selection: Optional[List[int]] = None
        # Per-layer statistics
        self._layer_stats: Dict[int, Dict[str, float]] = {}

    def observe(self, entropy_per_layer: Dict[int, float]) -> None:
        """
        Record attention entropy observation for each layer.

        Args:
            entropy_per_layer: {layer_idx: entropy_value}
        """
        self._entropy_history.append(dict(entropy_per_layer))
        self._cached_selection = None  # Invalidate cache

        # Update per-layer running stats
        for idx, ent in entropy_per_layer.items():
            if idx not in self._layer_stats:
                self._layer_stats[idx] = {"sum": 0.0, "count": 0, "max": 0.0}
            self._layer_stats[idx]["sum"] += ent
            self._layer_stats[idx]["count"] += 1
            self._layer_stats[idx]["max"] = max(self._layer_stats[idx]["max"], ent)

    def select(self) -> List[int]:
        """
        Select key layers for ExoBrain KV injection.

        Returns:
            List of layer indices (sorted ascending)
        """
        # Return cached selection if available
        if self._cached_selection is not None:
            return self._cached_selection

        if self.strategy == "all":
            return list(range(self.total_layers))

        if self.strategy == "middle_heavy":
            selection = self._select_middle_heavy()
        elif self.strategy == "entropy_threshold":
            selection = self._select_by_threshold()
        elif self.strategy == "entropy_top_k":
            selection = self._select_by_top_k()
        else:
            # Default: middle_heavy
            selection = self._select_middle_heavy()

        # Enforce min/max constraints
        if len(selection) < self.min_layers and self.total_layers > 0:
            # Add more layers (prefer middle layers)
            middle = list(range(self.total_layers // 4, 3 * self.total_layers // 4))
            for idx in middle:
                if idx not in selection:
                    selection.append(idx)
                if len(selection) >= self.min_layers:
                    break
        if len(selection) > self.max_layers:
            # Keep the top layers by average entropy
            avg_entropy = self._get_average_entropy()
            selection.sort(key=lambda idx: avg_entropy.get(idx, 0.0), reverse=True)
            selection = selection[:self.max_layers]

        selection = sorted(set(selection))
        self._cached_selection = selection
        return selection

    def _get_average_entropy(self) -> Dict[int, float]:
        """Compute average entropy per layer from observation history."""
        if not self._layer_stats:
            return {}
        return {
            idx: stats["sum"] / max(stats["count"], 1)
            for idx, stats in self._layer_stats.items()
        }

    def _select_by_top_k(self) -> List[int]:
        """Select top-K layers by average entropy."""
        avg_entropy = self._get_average_entropy()

        if not avg_entropy:
            return self._select_middle_heavy()

        k = max(self.min_layers, int(self.total_layers * self.top_k_ratio))
        k = min(k, self.max_layers)

        # Sort by entropy descending, take top-K
        sorted_layers = sorted(avg_entropy.keys(), key=lambda idx: avg_entropy[idx], reverse=True)
        return sorted(sorted_layers[:k])

    def _select_by_threshold(self) -> List[int]:
        """Select layers with average entropy above threshold."""
        avg_entropy = self._get_average_entropy()

        if not avg_entropy:
            return self._select_middle_heavy()

        selected = [idx for idx, ent in avg_entropy.items() if ent >= self.entropy_threshold]
        return sorted(selected)

    def _select_middle_heavy(self) -> List[int]:
        """
        Heuristic: prioritize middle layers.

        Based on the insight that:
        - Early layers (0-25%): lexical/syntax — shell's own
        - Middle layers (25-75%): semantic/concept — KEY LAYERS
        - Late layers (75-100%): output mapping — shell's own
        """
        if self.total_layers == 0:
            return []

        start = self.total_layers // 4
        end = 3 * self.total_layers // 4
        return list(range(start, end))

    def is_stable(self) -> bool:
        """Check if enough observations have been made for stable selection."""
        return len(self._entropy_history) >= self.stability_window

    @property
    def stats(self) -> Dict[str, Any]:
        """Return selection statistics."""
        avg_entropy = self._get_average_entropy()
        return {
            "strategy": self.strategy,
            "total_layers": self.total_layers,
            "observations": len(self._entropy_history),
            "selected_layers": self.select(),
            "num_selected": len(self.select()),
            "is_stable": self.is_stable(),
            "avg_entropy": {str(k): round(v, 4) for k, v in sorted(avg_entropy.items())},
        }


def compute_attention_entropy(
    attention_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute attention entropy per head per layer.

    Entropy = -Σ p·log(p), where p is the attention weight distribution.
    High entropy → diffuse/uncertain attention → benefits from external KV.
    Low entropy → focused/confident attention → shell's own is sufficient.

    Args:
        attention_weights: [batch, heads, q_len, kv_len] — softmaxed attention weights
        eps: Small value to avoid log(0)

    Returns:
        entropy: [batch, heads, q_len] — per-query-position entropy
    """
    log_weights = torch.log(attention_weights + eps)
    entropy = -torch.sum(attention_weights * log_weights, dim=-1)  # [B, H, Q]
    return entropy


# ─────────────────────────────────────────────────────────────
# Multi-Teacher Ensemble Router (v0.6)
# ─────────────────────────────────────────────────────────────

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

class ExoBrainBackend(KVStoreBackend):
    """
    ExoBrain-enhanced KV Store Backend (v0.4+).

    Inherits from KVStoreBackend and overrides read_attention()
    to inject external KV pairs at decode time.

    KEY IMPROVEMENT v0.4:
    Now supports ShellProjection for cognitive alignment between
    shell model and external brain hidden spaces.

    Usage:
        bus = ExoBrainBus(sources=[...])
        config = ExoBrainConfig(fusion_mode="replace", key_layers=[3,4,5,6,7,8])
        shell_proj = ShellProjection(768, 4096, mode="linear")
        backend = ExoBrainBackend(
            store_cfg=KVCacheStoreConfig(),
            brain_bus=bus,
            brain_cfg=config,
            shell_projection=shell_proj,
        )
    """

    def __init__(
        self,
        store_cfg: KVCacheStoreConfig,
        brain_bus: ExoBrainBus,
        brain_cfg: Optional[ExoBrainConfig] = None,
        store_cfg_factory: Optional[Callable[[Any, int], KVCacheStoreConfig]] = None,
        shell_projection: Optional[ShellProjection] = None,
    ) -> None:
        super().__init__(store_cfg=store_cfg, store_cfg_factory=store_cfg_factory)
        self.brain_bus = brain_bus
        self.brain_cfg = brain_cfg or ExoBrainConfig()
        self.shell_projection = shell_projection  # Optional cognitive alignment
        self._fusion_stats: Dict[str, int] = {
            "replace_count": 0,
            "residual_count": 0,
            "gated_count": 0,
            "fallback_count": 0,
            "error_count": 0,
        }

    def read_attention(
        self,
        handle: Any,
        layer_idx: int,
        query: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        scale: Optional[float],
        info: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Override read_attention() to inject external brain KV.

        Decision tree:
        1. Check if this is a key layer for injection
        2. Optionally project query via ShellProjection (cognitive alignment)
        3. Retrieve external KV from bus
        4. Apply fusion mode (replace / residual / gated)
        5. Fall back to standard KVStoreBackend on failure
        """
        cfg = self.brain_cfg

        # Check if this layer is a key layer for KV injection
        if not cfg.is_key_layer(layer_idx):
            return super().read_attention(
                handle, layer_idx, query, attn_mask, is_causal, scale, info
            )

        # Check query norm — skip brain for near-zero queries
        query_norm = float(query.float().norm().item())
        if query_norm < cfg.min_query_norm:
            return super().read_attention(
                handle, layer_idx, query, attn_mask, is_causal, scale, info
            )

        # Apply ShellProjection for cognitive alignment (if configured)
        # This projects shell's hidden_dim → brain's hidden_dim
        projected_query = query
        if self.shell_projection is not None:
            projected_query = self.shell_projection.project_query(query)

        # Try to retrieve external KV
        try:
            external_kv = self.brain_bus.retrieve(projected_query, layer_idx)
        except Exception:
            external_kv = None

        if external_kv is None:
            # No external brain available — fall back to standard path
            self._fusion_stats["fallback_count"] += 1
            if cfg.fallback_on_error:
                return super().read_attention(
                    handle, layer_idx, query, attn_mask, is_causal, scale, info
                )
            # No fallback — return zeros (shell model behavior)
            return torch.zeros_like(query)

        ext_k, ext_v = external_kv

        # Apply fusion mode (use original query for shell part)
        if cfg.fusion_mode == "replace":
            return self._fuse_replace(query, ext_k, ext_v, scale, attn_mask)
        elif cfg.fusion_mode == "residual":
            return self._fuse_residual(
                handle, layer_idx, query, ext_k, ext_v,
                attn_mask, is_causal, scale, info
            )
        elif cfg.fusion_mode == "gated":
            return self._fuse_gated(
                handle, layer_idx, query, ext_k, ext_v,
                attn_mask, is_causal, scale, info
            )
        else:
            # Unknown mode — fallback
            self._fusion_stats["fallback_count"] += 1
            return super().read_attention(
                handle, layer_idx, query, attn_mask, is_causal, scale, info
            )

    def _fuse_replace(
        self,
        query: torch.Tensor,
        ext_k: torch.Tensor,
        ext_v: torch.Tensor,
        scale: Optional[float],
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Replace mode: ŷ = I(K, x) — full external brain."""
        self._fusion_stats["replace_count"] += 1

        if self.brain_cfg.use_cross_attention:
            return cross_attention_fusion(query, ext_k, ext_v, scale, attn_mask)

        # Fallback: simple weighted sum
        d = query.shape[-1]
        scale_factor = float(scale) if scale is not None else (1.0 / math.sqrt(d))
        logits = (query @ ext_k.transpose(-2, -1)) * scale_factor
        weights = torch.softmax(logits, dim=-1)
        return weights @ ext_v

    def _fuse_residual(
        self,
        handle: Any,
        layer_idx: int,
        query: torch.Tensor,
        ext_k: torch.Tensor,
        ext_v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        scale: Optional[float],
        info: Dict[str, Any],
    ) -> torch.Tensor:
        """Residual mode: ŷ = α·f_θ(x) + (1-α)·I(K, x)."""
        self._fusion_stats["residual_count"] += 1
        alpha = self.brain_cfg.residual_alpha

        # Shell model attention
        try:
            shell_output = super().read_attention(
                handle, layer_idx, query, attn_mask, is_causal, scale, info
            )
        except Exception:
            shell_output = torch.zeros_like(query)

        # External brain attention
        brain_output = cross_attention_fusion(query, ext_k, ext_v, scale, attn_mask)

        return alpha * shell_output + (1.0 - alpha) * brain_output

    def _fuse_gated(
        self,
        handle: Any,
        layer_idx: int,
        query: torch.Tensor,
        ext_k: torch.Tensor,
        ext_v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        scale: Optional[float],
        info: Dict[str, Any],
    ) -> torch.Tensor:
        """Gated mode: ŷ = g·I(K,x) + (1-g)·f_θ(x)."""
        self._fusion_stats["gated_count"] += 1

        # Shell model attention
        try:
            shell_output = super().read_attention(
                handle, layer_idx, query, attn_mask, is_causal, scale, info
            )
        except Exception:
            shell_output = torch.zeros_like(query)

        # External brain attention
        brain_output = cross_attention_fusion(query, ext_k, ext_v, scale, attn_mask)

        # v0.5: Use configurable gate mode
        gate = compute_gate(
            query, ext_k,
            temperature=self.brain_cfg.gate_temperature,
            mode=self.brain_cfg.gate_mode,
        )

        return gate * brain_output + (1.0 - gate) * shell_output

    @property
    def fusion_stats(self) -> Dict[str, Any]:
        """Get fusion mode statistics."""
        return {
            **self._fusion_stats,
            "bus_stats": self.brain_bus.stats,
        }


# ─────────────────────────────────────────────────────────────
# P2: ExoBrainAttentionPatcher — Attention-Level Interception
# ─────────────────────────────────────────────────────────────

class ExoBrainAttentionPatcher:
    """
    Extended attention patcher for ExoBrain.

    Unlike UniversalAttentionPatcher which only intercepts decode
    steps (q_len==1), ExoBrainAttentionPatcher can also intercept
    prefill steps, enabling external brain knowledge injection
    from the very first token.

    This extends the patching mechanism to support:
    1. Replace mode: External KV completely replaces model KV
    2. Residual mode: External KV blended with model output
    3. Gated mode: Attention-gated dynamic blending
    """

    def __init__(
        self,
        backend: ExoBrainBackend,
        brain_bus: ExoBrainBus,
        brain_cfg: Optional[ExoBrainConfig] = None,
    ) -> None:
        self.backend = backend
        self.brain_bus = brain_bus
        self.brain_cfg = brain_cfg or ExoBrainConfig()
        self._patched = False
        self._orig_get_interface = None

        # Try to import transformers for attention patching
        try:
            import transformers.modeling_utils as mu
            registry = getattr(mu, "ALL_ATTENTION_FUNCTIONS", None)
            self._supported = registry is not None and hasattr(registry, "get_interface")
            self._registry = registry
        except ImportError:
            self._supported = False
            self._registry = None

    def apply(self) -> None:
        """Apply the ExoBrain attention patch."""
        if not self._supported:
            return
        if self._patched:
            return

        import transformers.modeling_utils as mu
        from ..patches.cache_hooks import _thread_local

        orig_get_interface = self._registry.get_interface
        self._orig_get_interface = orig_get_interface
        backend = self.backend

        def exobrain_get_interface(config_attn_implementation, eager_attention_forward):
            orig_interface = orig_get_interface(config_attn_implementation, eager_attention_forward)

            def exobrain_attention_forward(module, query_states, key_states, value_states, attention_mask, **kwargs):
                cache = getattr(_thread_local, "current_cache", None)
                query_states.size(-2)

                if cache is not None and getattr(cache, "_vitriol_kv_store_mode", False):
                    layer_idx = getattr(module, "layer_idx", None)
                    if layer_idx is not None:
                        # Try ExoBrain injection for both prefill and decode
                        try:
                            attn_output = backend.read_attention(
                                handle=cache,
                                layer_idx=layer_idx,
                                query=query_states,
                                attn_mask=attention_mask,
                                is_causal=bool(kwargs.get("is_causal", False)),
                                scale=kwargs.get("scaling", None),
                                info={"dropout_p": kwargs.get("dropout", 0.0)},
                            )
                            attn_output = attn_output.transpose(1, 2).contiguous()
                            return attn_output, None
                        except Exception:
                            logger.debug("Failed to call attention interface for external brain KV injection")

                return orig_interface(module, query_states, key_states, value_states, attention_mask, **kwargs)

            return exobrain_attention_forward

        mu.ALL_ATTENTION_FUNCTIONS.get_interface = exobrain_get_interface
        self._patched = True

    def restore(self) -> None:
        """Restore original attention function."""
        if not self._supported or not self._patched:
            return

        import transformers.modeling_utils as mu
        mu.ALL_ATTENTION_FUNCTIONS.get_interface = self._orig_get_interface
        self._patched = False
