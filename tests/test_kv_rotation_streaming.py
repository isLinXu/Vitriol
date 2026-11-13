"""Rotation invariants and adaptive prefill/decode numerical contracts."""

import math

import pytest
import torch
import torch.nn.functional as F

from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig
from vitriol.kv.codec import (
    AdaptiveKVCodec,
    adaptive_kv_bits,
    blockwise_minmax_quantize_dequantize,
    walsh_hadamard_rotate,
)
from vitriol.kv.hybrid_pipeline import SlidingWindowConfig, SlidingWindowEvictor
from vitriol.patches.turboquant import apply_walsh_hadamard_rotation


@pytest.mark.parametrize("dim", [1, 3, 5, 64, 80, 96, 127, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_rotation_preserves_vectors_norms_and_inner_products(dim, dtype):
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(2, dim, 3, generator=generator).transpose(-1, -2).to(dtype)
    y = torch.randn(x.shape, generator=generator).to(dtype)
    before = x.clone()
    rotated = walsh_hadamard_rotate(x)
    tolerance = {torch.float16: 0.006, torch.bfloat16: 0.04}.get(dtype, 2e-6)
    assert rotated.shape == x.shape and rotated.dtype == dtype
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(walsh_hadamard_rotate(rotated), x, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(rotated.float().norm(dim=-1), x.float().norm(dim=-1),
                               rtol=tolerance, atol=tolerance)
    torch.testing.assert_close((rotated.float() * walsh_hadamard_rotate(y).float()).sum(-1),
                               (x.float() * y.float()).sum(-1), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(apply_walsh_hadamard_rotation(x), rotated)


def test_power_of_two_rotation_matches_dense_sylvester_matrix():
    matrix = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(6):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    x = torch.randn(3, 64, dtype=torch.float64)
    torch.testing.assert_close(walsh_hadamard_rotate(x), x @ matrix / 8)


@pytest.mark.parametrize("x", [torch.tensor(1.0), torch.empty(2, 0)])
def test_rotation_rejects_missing_or_empty_feature_dimension(x):
    with pytest.raises(ValueError, match="non-empty last dimension"):
        walsh_hadamard_rotate(x)


def test_rotation_autograd_preserves_squared_norm_gradient():
    x = torch.randn(2, 80, dtype=torch.float64, requires_grad=True)
    walsh_hadamard_rotate(x).square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x)


def test_half_rotation_avoids_overflow_in_intermediate_sums():
    x = torch.full((2, 128), 1000.0, dtype=torch.float16)
    actual = walsh_hadamard_rotate(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, walsh_hadamard_rotate(x.float()).half())


def test_adaptive_allocation_does_not_depend_on_rotation_coordinates():
    torch.manual_seed(23)
    key = torch.randn(2, 3, 7, 80)
    query = key * 4
    value = torch.randn(2, 3, 7, 48)
    rotated = AdaptiveKVCodec(rotate_kurtosis_threshold=-1).quantize_kv(query, key, value)[2]
    plain = AdaptiveKVCodec(rotate_kurtosis_threshold=math.inf).quantize_kv(query, key, value)[2]
    assert rotated["rotated_k"] and rotated["rotated_v"]
    assert rotated["k_levels"] == plain["k_levels"]
    assert rotated["avg_bits_total"] == pytest.approx(plain["avg_bits_total"])


def _reference_quantize(x, levels, rotate, block_size):
    result = torch.empty_like(x)
    for batch in range(x.shape[0]):
        for head in range(x.shape[1]):
            work = x[batch, head].float()
            if rotate:
                work = walsh_hadamard_rotate(work)
            work = blockwise_minmax_quantize_dequantize(work, int(levels[batch, head]), block_size)
            result[batch, head] = walsh_hadamard_rotate(work) if rotate else work
    return result


@pytest.mark.parametrize("keep_raw", [True, False])
@pytest.mark.parametrize("turbo", [True, False])
@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_adaptive_streaming_preserves_prefix_and_quantizes_each_batch(keep_raw, turbo, rotate, dtype):
    torch.manual_seed(81)
    codec = AdaptiveKVCodec(block_size=16, rotate_kurtosis_threshold=-1 if rotate else math.inf)
    store = KVCacheStore(KVCacheStoreConfig(
        keep_raw_cache=keep_raw, enable_turbo_quant=turbo, enable_turbo_residual_qjl=False,
        enable_adaptive_bits=True, adaptive_bits=codec, enable_zero_copy_decode=True,
    ))
    key = torch.randn(2, 2, 5, 80).to(dtype)
    value = torch.randn(2, 2, 5, 48).to(dtype)
    query = key[:, :, :1].clone()
    query[0] *= 0.01
    query[1] *= 10
    store.set_prefill(key, value)
    key_in, value_in = store._decode_kv_if_packed()
    k_bits, v_bits, _ = adaptive_kv_bits(query, key_in, value_in)
    k_levels = (2 ** k_bits).round().clamp(2, 256).long()
    v_levels = (2 ** v_bits).round().clamp(2, 256).long()
    assert not torch.equal(k_levels[0], k_levels[1])
    expected_k = _reference_quantize(key_in, k_levels, rotate, 16)
    expected_v = _reference_quantize(value_in, v_levels, rotate, 16)
    torch.testing.assert_close(store.attention(query), F.scaled_dot_product_attention(query, expected_k.to(dtype), expected_v.to(dtype)))
    for _ in range(3):
        new_key = torch.randn(2, 2, 1, 80).to(dtype)
        new_value = torch.randn(2, 2, 1, 48).to(dtype)
        key_input = store._decode_tensor(store._encode_tensor(new_key, is_key=True)) if turbo else new_key
        value_input = store._decode_tensor(store._encode_tensor(new_value, is_key=False)) if turbo else new_value
        expected_k = torch.cat((expected_k, _reference_quantize(key_input, k_levels, rotate, 16)), -2)
        expected_v = torch.cat((expected_v, _reference_quantize(value_input, v_levels, rotate, 16)), -2)
        store.append(new_key, new_value)
        torch.testing.assert_close(store._k_enc, expected_k)
        torch.testing.assert_close(store._v_enc, expected_v)
        torch.testing.assert_close(store.attention(query), F.scaled_dot_product_attention(query, expected_k.to(dtype), expected_v.to(dtype)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_adaptive_gqa_averages_query_group_importance_and_supports_value_dimension(dtype):
    torch.manual_seed(7)
    query = torch.randn(2, 6, 3, 80).to(dtype)
    key = torch.randn(2, 2, 5, 80).to(dtype)
    value = torch.randn(2, 2, 5, 48).to(dtype)
    codec = AdaptiveKVCodec(rotate_kurtosis_threshold=-1)
    k_bits, v_bits, _ = adaptive_kv_bits(query, key, value)
    weights = torch.softmax(query.float() @ key.float().repeat_interleave(3, dim=1).transpose(-2, -1) / math.sqrt(80), -1)
    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1).mean(-1) / math.log(5)
    importance = (1 - entropy).clamp(0, 1).reshape(2, 2, 3).mean(-1)
    expected_k = 3 + 2 * importance
    rms = value.float().square().mean((-2, -1)).sqrt()
    rms = rms / (rms.mean(-1, keepdim=True) + 1e-12)
    expected_v = 3 + 2 * (rms / (rms.max(-1, keepdim=True).values + 1e-12)).clamp(0, 1)
    adjustment = 3.5 / (expected_k.mean(-1) * 0.65 + expected_v.mean(-1) * 0.35).mean()
    torch.testing.assert_close(k_bits, (expected_k * adjustment).clamp(3, 5))
    torch.testing.assert_close(v_bits, (expected_v * adjustment).clamp(3, 5))
    restored_k, restored_v, _ = codec.quantize_kv(query, key, value)
    assert restored_k.dtype == dtype and restored_v.dtype == dtype
    store = KVCacheStore(KVCacheStoreConfig(enable_adaptive_bits=True, adaptive_bits=codec))
    store.set_prefill(key, value)
    expected = F.scaled_dot_product_attention(query, restored_k.repeat_interleave(3, 1), restored_v.repeat_interleave(3, 1))
    torch.testing.assert_close(store.attention(query), expected)


@pytest.mark.parametrize("keep_raw", [True, False])
@pytest.mark.parametrize("event", ["threshold", "eviction"])
def test_adaptive_rebuild_reallocates_after_threshold_or_eviction(keep_raw, event):
    torch.manual_seed(101)
    cfg = KVCacheStoreConfig(
        keep_raw_cache=keep_raw, enable_turbo_quant=True, enable_turbo_residual_qjl=False,
        quantized_kv_start=4 if event == "threshold" else 0,
        enable_adaptive_bits=True, adaptive_bits=AdaptiveKVCodec(rotate_kurtosis_threshold=-1),
        enable_sliding_window_eviction=event == "eviction", eviction_max_seq_len=3,
        eviction_min_recent_tokens=2, enable_zero_copy_decode=True,
    )
    store = KVCacheStore(cfg)
    key, value = torch.randn(1, 2, 3, 80), torch.randn(1, 2, 3, 80)
    query = torch.randn(1, 2, 1, 80)
    store.set_prefill(key, value)
    store.attention(query)
    old_key = store._k_raw if keep_raw else store._k_enc
    old_value = store._v_raw if keep_raw else store._v_enc
    new_key, new_value = torch.randn_like(query), torch.randn_like(query)
    reference = KVCacheStore(cfg)
    reference.set_prefill(torch.cat((old_key, new_key), -2), torch.cat((old_value, new_value), -2))
    store.append(new_key, new_value)
    torch.testing.assert_close(store.attention(query), reference.attention(query))
    assert store.seq_len == reference.seq_len


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_adaptive_zero_values_and_sharp_logits_produce_finite_levels(dtype):
    query = torch.full((1, 2, 1, 80), 1000.0, dtype=dtype)
    key = torch.cat((query, -query), dim=-2)
    value = torch.zeros(1, 2, 2, 48, dtype=dtype)
    k_bits, v_bits, _ = adaptive_kv_bits(query, key, value)
    assert torch.isfinite(k_bits).all() and torch.isfinite(v_bits).all()


@pytest.mark.parametrize("recent", [0, 2, 4, 256])
@pytest.mark.parametrize("attention_based", [False, True])
def test_sliding_window_never_exceeds_budget_when_recent_reserve_fills_window(recent, attention_based):
    evictor = SlidingWindowEvictor(SlidingWindowConfig(
        max_seq_len=4, min_recent_tokens=recent, attention_based=attention_based,
    ))
    key = torch.arange(9.0).reshape(1, 1, 9, 1)
    keep = evictor.compute_eviction_indices(9, attention_weights=key.transpose(-1, -2))
    restored_k, restored_v = evictor.evict_kv(key, -key, keep)
    torch.testing.assert_close(keep, torch.arange(5, 9))
    torch.testing.assert_close(restored_k.flatten(), torch.arange(5.0, 9.0))
    torch.testing.assert_close(restored_v, -restored_k)
