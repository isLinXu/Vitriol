"""Knowledge sources for ExoBrain (Vector DB / API / local weights)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn.functional as F

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
                    import urllib.error
                    import urllib.request
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
