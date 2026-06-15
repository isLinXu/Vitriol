from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

import torch

from .cache_store import KVCacheStore, KVCacheStoreConfig


@dataclass(frozen=True)
class KVMeta:
    """Metadata descriptor for a KV cache entry."""
    model_id: str
    device: str
    dtype: str


class KVBackend(Protocol):
    """Protocol defining the KV cache backend interface."""
    def write_kv(self, handle: Any, layer_idx: int, key_new: torch.Tensor, value_new: torch.Tensor, info: Dict[str, Any]) -> None: ...

    def read_attention(
        self,
        handle: Any,
        layer_idx: int,
        query: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        scale: Optional[float],
        info: Dict[str, Any],
    ) -> torch.Tensor: ...

    def stats(self, handle: Any) -> Dict[str, Any]: ...


@dataclass
class KVStoreBackend:
    """Concrete KV store backend with persistence support."""
    store_cfg: KVCacheStoreConfig
    store_cfg_factory: Optional[Callable[[Any, int], KVCacheStoreConfig]] = None

    def _ensure_store(self, handle: Any, layer_idx: int) -> KVCacheStore:
        stores = getattr(handle, "_vitriol_kv_stores", None)
        if stores is None:
            stores = {}
            handle._vitriol_kv_stores = stores
        store = stores.get(int(layer_idx))
        if store is None:
            cfg = self.store_cfg_factory(handle, int(layer_idx)) if self.store_cfg_factory is not None else self.store_cfg
            store = KVCacheStore(cfg)
            stores[int(layer_idx)] = store
        return store

    def write_kv(self, handle: Any, layer_idx: int, key_new: torch.Tensor, value_new: torch.Tensor, info: Dict[str, Any]) -> None:
        store = self._ensure_store(handle, int(layer_idx))
        if store.seq_len == 0 and int(info.get("q_len", 0)) > 1:
            store.set_prefill(key_new, value_new)
        elif store.seq_len == 0:
            store.set_prefill(key_new, value_new)
        else:
            store.append(key_new, value_new)

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
        store = self._ensure_store(handle, int(layer_idx))
        return store.attention(
            query,
            attn_mask=attn_mask,
            dropout_p=float(info.get("dropout_p", 0.0)),
            is_causal=is_causal,
            scale=scale,
            sliding_window=info.get("sliding_window"),
        )

    def stats(self, handle: Any) -> Dict[str, Any]:
        stores = getattr(handle, "_vitriol_kv_stores", None) or {}
        layer_stats = {
            int(k): {
                "seq_len": int(v.seq_len),
                "estimated_kv_bytes": int(v.estimated_kv_bytes()),
            }
            for k, v in stores.items()
        }
        return {
            "layers": len(stores),
            "seq_lens": {k: item["seq_len"] for k, item in layer_stats.items()},
            "estimated_kv_bytes": int(sum(item["estimated_kv_bytes"] for item in layer_stats.values())),
            "layer_stats": layer_stats,
        }
