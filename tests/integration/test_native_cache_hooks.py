"""Actual DynamicCache/attention-registry routing and full-history fallback."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from vitriol.bench.runner import _apply_vitriol_universal
from vitriol.kv.cache_store import KVCacheStoreConfig
from vitriol.patches.cache_hooks import get_cache_hook_stats

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("failure", [None, "read", "write"])
@pytest.mark.parametrize("passthrough", [True, False])
def test_local_llama_hooks_and_backend_failure_preserve_full_context(monkeypatch, failure, passthrough):
    torch.manual_seed(121)
    config = LlamaConfig(
        vocab_size=32, hidden_size=160, intermediate_size=192, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    original_update = DynamicCache.update
    get_cache_hook_stats(reset=True)
    backend, cache_patcher, attention_patcher = _apply_vitriol_universal(
        KVCacheStoreConfig(keep_raw_cache=False), 0, passthrough_update=passthrough,
    )
    if failure == "read":
        def read_failed(**kwargs):
            raise RuntimeError("injected read failure")
        monkeypatch.setattr(backend, "read_attention", read_failed)
    if failure == "write":
        write = backend.write_kv

        def write_failed(handle, layer_idx, key_new, value_new, info):
            if key_new.shape[-2] == 1:
                raise RuntimeError("injected write failure")
            return write(handle, layer_idx, key_new, value_new, info)
        monkeypatch.setattr(backend, "write_kv", write_failed)

    last_cache = None
    try:
        with torch.inference_mode():
            for prompt in ([1, 4, 7], [2, 6, 11]):
                tokens = torch.tensor([prompt])
                output = model(tokens, use_cache=True)
                for token in (9, 12, 5):
                    step = torch.tensor([[token]])
                    tokens = torch.cat((tokens, step), -1)
                    output = model(step, past_key_values=output.past_key_values, use_cache=True)
                    reference = model(tokens, use_cache=False)
                    torch.testing.assert_close(output.logits[:, -1], reference.logits[:, -1], atol=2e-6, rtol=1e-5)
                    assert output.past_key_values.get_seq_length() == tokens.shape[-1]
                last_cache = output.past_key_values
                stats = backend.stats(last_cache)
                assert stats["native_cache_retained"] is True
                assert stats["fallback_layers"] == ([] if failure is None else [0, 1])
        stats = get_cache_hook_stats()
        if failure is None:
            assert stats["read_attention_hit"] == 12
        else:
            counter = "read_attention_fallback" if failure == "read" else "write_kv_fallback"
            assert stats[counter] == 4
    finally:
        cache_patcher.restore()
        attention_patcher.restore()
    assert DynamicCache.update is original_update
    # The same handle must remain usable after removing both hooks.
    with torch.inference_mode():
        step = torch.tensor([[13]])
        tokens = torch.cat((tokens, step), -1)
        output = model(step, past_key_values=last_cache, use_cache=True)
        reference = model(tokens, use_cache=False)
        torch.testing.assert_close(output.logits[:, -1], reference.logits[:, -1], atol=2e-6, rtol=1e-5)
