"""Offline transformer validation of adaptive KV prefill and cached decode."""

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from vitriol.kv.codec import AdaptiveKVCodec
from vitriol.patches import apply_all_patches
from vitriol.patches.kv_runtime_patches import KVRuntimePatchConfig, KVRuntimePatcher

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("cache_entries", [0, 2])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_local_llama_non_power_of_two_gqa_cached_decode_matches_full_prefix(cache_entries, inference_mode):
    apply_all_patches()
    torch.manual_seed(2026)
    config = LlamaConfig(
        vocab_size=32, hidden_size=160, intermediate_size=192,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).eval()
    assert model.config._attn_implementation == "sdpa"
    # Fixed precision isolates cache/rotation correctness from changing
    # query-dependent precision decisions. Variable precision has unit coverage.
    patcher = KVRuntimePatcher(KVRuntimePatchConfig(
        decode_only=False, enable_adaptive_bits=True, preprocess_cache_max_entries=cache_entries,
        adaptive_bits=AdaptiveKVCodec(
            min_bits=8, max_bits=8, target_avg_bits=8, rotate_kurtosis_threshold=-1,
        ),
    ))
    original_attention = F.scaled_dot_product_attention
    try:
        patcher.apply()
        with torch.inference_mode() if inference_mode else torch.no_grad():
            # Distinct consecutive requests must not reuse the first request's KV.
            for prompt in ([1, 4, 7], [2, 6, 11]):
                tokens = torch.tensor([prompt])
                output = model(tokens, use_cache=True)
                assert torch.isfinite(output.logits).all()
                for token in (9, 12, 5):
                    new_token = torch.tensor([[token]])
                    tokens = torch.cat((tokens, new_token), dim=-1)
                    output = model(new_token, past_key_values=output.past_key_values, use_cache=True)
                    reference = model(tokens, use_cache=False)
                    torch.testing.assert_close(
                        output.logits[:, -1], reference.logits[:, -1], atol=2e-6, rtol=1e-5,
                    )
                    assert torch.isfinite(output.logits).all()
                    assert output.past_key_values.get_seq_length() == tokens.shape[-1]
        stats = patcher.stats()
        assert stats["calls_patched"] == 28
        assert stats["preprocess_cache_entries"] <= cache_entries
        if cache_entries and not inference_mode:
            assert stats["preprocess_cache_misses"] == 28
            assert stats["preprocess_cache_evictions"] > 0
        if inference_mode:
            assert stats["preprocess_cache_entries"] == 0
    finally:
        patcher.restore()
    assert F.scaled_dot_product_attention is original_attention
    assert patcher.stats()["preprocess_cache_entries"] == 0
