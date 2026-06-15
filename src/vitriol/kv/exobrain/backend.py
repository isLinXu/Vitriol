"""ExoBrain KV store backend and attention patcher."""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Optional

import torch

from ..backend import KVStoreBackend
from ..cache_store import KVCacheStoreConfig
from .config import ExoBrainConfig
from .fusion import ExoBrainBus, compute_gate, cross_attention_fusion
from .projection import ShellProjection

logger = logging.getLogger(__name__)


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

    def apply(self) -> Any:
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

        def exobrain_get_interface(config_attn_implementation, eager_attention_forward) -> Any:
            orig_interface = orig_get_interface(config_attn_implementation, eager_attention_forward)

            def exobrain_attention_forward(module, query_states, key_states, value_states, attention_mask, **kwargs) -> Any:
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
