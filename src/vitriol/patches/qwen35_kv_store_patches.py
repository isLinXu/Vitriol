from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import torch

from ..kv.cache_store import KVCacheStore, KVCacheStoreConfig
from ..kv.policy import KVLayerType, KVPolicy, apply_policy_to_store_cfg, classify_kv_layer


def _full_attention_layers(past_key_values: Any) -> Optional[list[int]]:
    layer_types = getattr(past_key_values, "layer_types", None)
    if layer_types is None:
        return None
    return [
        i
        for i, _ in enumerate(layer_types)
        if classify_kv_layer(past_key_values, i) is KVLayerType.FULL_ATTENTION
    ]


def _is_full_attention_layer(past_key_values: Any, layer_idx: int) -> bool:
    layer_types = getattr(past_key_values, "layer_types", None)
    if layer_types is None:
        return True
    idx = int(layer_idx)
    if idx < 0 or idx >= len(layer_types):
        return False
    return classify_kv_layer(past_key_values, idx) is KVLayerType.FULL_ATTENTION


@dataclass(frozen=True)
class Qwen35KVStorePatchConfig:
    decode_only: bool = True
    decode_query_len: int = 1
    store_cfg: KVCacheStoreConfig = field(default_factory=KVCacheStoreConfig)
    policy: Optional[KVPolicy] = None
    v_protect_last_n_full_attention_layers: int = 0
    v_quantize_only_first_n_full_attention_layers: int = 0


class Qwen35KVStorePatcher:
    def __init__(self, cfg: Qwen35KVStorePatchConfig) -> None:
        self.cfg = cfg
        self._orig_attn_forward: Optional[Callable[..., Any]] = None
        self._calls_total: int = 0
        self._calls_bypassed: int = 0
        self._calls_store: int = 0

    def apply(self) -> None:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        if getattr(m.Qwen3_5Attention.forward, "_vitriol_qwen35_kv_store_patched", False):
            return

        self._orig_attn_forward = m.Qwen3_5Attention.forward
        orig_cache_update = m.Qwen3_5DynamicCache.update
        orig_cache_get_seq_length = m.Qwen3_5DynamicCache.get_seq_length

        def cache_update_wrapped(self_cache, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[dict[str, Any]] = None):
            mode = getattr(self_cache, "_vitriol_kv_store_mode", False)
            if not mode:
                return orig_cache_update(self_cache, key_states, value_states, layer_idx, cache_kwargs)

            seq_lens = getattr(self_cache, "_vitriol_seq_lens", None)
            if seq_lens is None:
                seq_lens = [0 for _ in range(len(getattr(self_cache, "layer_types", [])) or len(self_cache))]
                self_cache._vitriol_seq_lens = seq_lens

            added = int(key_states.size(-2))
            if seq_lens[layer_idx] == 0:
                seq_lens[layer_idx] = added
                return key_states, value_states

            seq_lens[layer_idx] += added
            return key_states, value_states

        def cache_get_seq_length_wrapped(self_cache, layer_idx: int | None = 0) -> int:
            mode = getattr(self_cache, "_vitriol_kv_store_mode", False)
            if not mode:
                return orig_cache_get_seq_length(self_cache, layer_idx)
            seq_lens = getattr(self_cache, "_vitriol_seq_lens", None)
            if seq_lens is None:
                return 0
            if layer_idx is None:
                for x in seq_lens:
                    if x:
                        return int(x)
                return 0
            idx = int(layer_idx)
            full_layers = _full_attention_layers(self_cache)
            if full_layers and not _is_full_attention_layer(self_cache, idx):
                idx = full_layers[0]
            if idx < 0 or idx >= len(seq_lens):
                return 0
            return int(seq_lens[idx])

        m.Qwen3_5DynamicCache.update = cache_update_wrapped
        m.Qwen3_5DynamicCache.get_seq_length = cache_get_seq_length_wrapped

        def patched_forward(
            module: torch.nn.Module,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Any = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Any,
        ):
            self._calls_total += 1

            input_shape = hidden_states.shape[:-1]
            head_dim = getattr(module.config, "head_dim", module.config.hidden_size // module.config.num_attention_heads)
            hidden_shape = (*input_shape, -1, head_dim)

            q_proj_out = module.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2)
            query_states, gate = torch.chunk(q_proj_out, 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)

            query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = m.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_new = key_states
            value_new = value_states

            if past_key_values is not None:
                past_key_values._vitriol_kv_store_mode = True
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

            q_len = int(query_states.size(-2))
            use_store = (past_key_values is not None) and (not self.cfg.decode_only or q_len == int(self.cfg.decode_query_len))

            stores = None
            store = None
            if past_key_values is not None:
                stores = getattr(past_key_values, "_vitriol_kv_stores", None)
                if stores is None:
                    stores = {}
                    past_key_values._vitriol_kv_stores = stores
                store = stores.get(module.layer_idx)
                if store is None:
                    store_cfg = replace(self.cfg.store_cfg)
                    if self.cfg.policy is not None:
                        store_cfg = apply_policy_to_store_cfg(store_cfg, self.cfg.policy, past_key_values, int(module.layer_idx))
                    else:
                        protect_n = int(self.cfg.v_protect_last_n_full_attention_layers)
                        first_n = int(self.cfg.v_quantize_only_first_n_full_attention_layers)
                        full_layers = _full_attention_layers(past_key_values)
                        if full_layers:
                            layer_idx_int = int(module.layer_idx)
                            if first_n == 0:
                                if layer_idx_int in full_layers:
                                    store_cfg = replace(store_cfg, turbo_quantize_v=False)
                            else:
                                if protect_n > 0:
                                    protected = set(full_layers[-protect_n:])
                                    if layer_idx_int in protected:
                                        store_cfg = replace(store_cfg, turbo_quantize_v=False)
                                pos = {idx: p for p, idx in enumerate(full_layers)}.get(layer_idx_int, None)
                                if pos is not None and pos >= first_n:
                                    store_cfg = replace(store_cfg, turbo_quantize_v=False)
                    store = KVCacheStore(store_cfg)
                    stores[module.layer_idx] = store

                if store.seq_len == 0 and q_len > 1:
                    key_full = m.repeat_kv(key_states, module.num_key_value_groups)
                    value_full = m.repeat_kv(value_states, module.num_key_value_groups)
                    store.set_prefill(key_full, value_full)
                elif store.seq_len > 0:
                    key_rep = m.repeat_kv(key_new, module.num_key_value_groups)
                    value_rep = m.repeat_kv(value_new, module.num_key_value_groups)
                    store.append(key_rep, value_rep)
            if not use_store:
                if self.cfg.decode_only:
                    self._calls_bypassed += 1
                attention_interface: Callable = m.ALL_ATTENTION_FUNCTIONS.get_interface(module.config._attn_implementation, m.eager_attention_forward)
                attn_output, attn_weights = attention_interface(
                    module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling,
                    **kwargs,
                )
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate)
                attn_output = module.o_proj(attn_output)
                return attn_output, attn_weights

            self._calls_store += 1
            attn_output = store.attention(
                query_states,
                attn_mask=attention_mask,
                dropout_p=0.0 if not module.training else module.attention_dropout,
                is_causal=False,
                scale=float(module.scaling),
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = module.o_proj(attn_output)
            return attn_output, None

        patched_forward._vitriol_qwen35_kv_store_patched = True
        patched_forward._vitriol_qwen35_kv_store_original = self._orig_attn_forward
        m.Qwen3_5Attention.forward = patched_forward
        m.Qwen3_5Attention.forward._vitriol_qwen35_kv_store_cache_update_original = orig_cache_update
        m.Qwen3_5Attention.forward._vitriol_qwen35_kv_store_cache_get_seq_length_original = orig_cache_get_seq_length

    def restore(self) -> None:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        current = m.Qwen3_5Attention.forward
        original = getattr(current, "_vitriol_qwen35_kv_store_original", None)
        if getattr(current, "_vitriol_qwen35_kv_store_patched", False) and original is not None:
            m.Qwen3_5Attention.forward = original
        orig_update = getattr(current, "_vitriol_qwen35_kv_store_cache_update_original", None)
        orig_get_seq = getattr(current, "_vitriol_qwen35_kv_store_cache_get_seq_length_original", None)
        if orig_update is not None:
            m.Qwen3_5DynamicCache.update = orig_update
        if orig_get_seq is not None:
            m.Qwen3_5DynamicCache.get_seq_length = orig_get_seq

    def stats(self) -> dict:
        return {
            "calls_total": int(self._calls_total),
            "calls_bypassed": int(self._calls_bypassed),
            "calls_store": int(self._calls_store),
        }


def patch_qwen35_kv_store(cfg: Qwen35KVStorePatchConfig) -> Qwen35KVStorePatcher:
    patcher = Qwen35KVStorePatcher(cfg)
    patcher.apply()
    return patcher
