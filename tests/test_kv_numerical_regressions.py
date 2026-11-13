"""Numerical contracts for packed and block-skipping attention."""

import pytest
import torch
import torch.nn.functional as F

from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig
from vitriol.kv.codec import (
    AdaptiveKVCodec,
    ComputeSkipConfig,
    adaptive_kv_bits,
    approx_inner_product_with_qjl_residual,
    compute_skip_attention,
    pack_blockwise_tensor_with_qjl_residual,
    unpack_blockwise_tensor,
    unpack_qjl_residual_tensor,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("residual", [False, True])
def test_packed_attention_prefill_and_decode_match_dense_reference(dtype, residual):
    torch.manual_seed(42)
    key = torch.randn(1, 2, 3, 16).to(dtype)
    value = torch.randn_like(key)
    store = KVCacheStore(KVCacheStoreConfig(
        enable_turbo_quant=True, enable_turbo_residual_qjl=residual,
    ))
    store.set_prefill(key, value)
    for query_len in [3, 1]:
        query = torch.randn(1, 2, query_len, 16).to(dtype)
        if residual:
            # sketch_dim=16 uses gain sqrt(16/32), as specified by the proxy.
            packed = store._k_enc
            decoded_key = unpack_blockwise_tensor(packed.base)
            correction = unpack_qjl_residual_tensor(packed) - decoded_key
            decoded_key = decoded_key + correction * (0.5 ** 0.5)
        else:
            decoded_key = unpack_blockwise_tensor(store._k_enc)
        decoded_value = store._decode_tensor(store._v_enc)
        expected = F.scaled_dot_product_attention(
            query.float(), decoded_key.float(), decoded_value.float(),
        )
        actual = store.attention(query)
        assert actual.shape == query.shape
        assert actual.dtype == dtype
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


def test_qjl_unpack_accepts_half_precision_projection():
    packed = pack_blockwise_tensor_with_qjl_residual(
        torch.randn(1, 2, 3, 16), levels=8, block_size=16, bit_width=3,
    )
    # CUDA selects fp16 projections; reproduce the same dtype layout on CPU.
    packed.projection = packed.projection.half()
    actual = unpack_qjl_residual_tensor(packed)
    expected = unpack_blockwise_tensor(packed.base) + (
        packed.residual_signs.float() * packed.residual_scale
    ) @ packed.projection.float().T
    torch.testing.assert_close(actual, expected)
    result = approx_inner_product_with_qjl_residual(
        torch.randn(1, 2, 1, 16, dtype=torch.bfloat16), packed,
    )
    assert result.shape == (1, 2, 1, 3)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("mode", ["plain", "skip", "sparse", "temporal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_residual_prefill_preserves_shape_and_zeroes_fully_masked_rows(mode, dtype):
    store = KVCacheStore(KVCacheStoreConfig(
        enable_turbo_quant=True,
        enable_compute_skip=mode == "skip",
        enable_sparse_v=mode == "sparse",
        enable_temporal_pooling=mode == "temporal",
    ))
    key = torch.randn(1, 2, 3, 16).to(dtype)
    store.set_prefill(key, torch.randn_like(key))
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[0] = False
    output = store.attention(torch.randn_like(key), attn_mask=mask)
    assert output.shape == key.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[..., 0, :]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["skip", "temporal"])
def test_unpacked_half_attention_zeroes_fully_masked_rows(dtype, mode):
    store = KVCacheStore(KVCacheStoreConfig(
        enable_compute_skip=mode == "skip",
        enable_temporal_pooling=mode == "temporal",
    ))
    key = torch.randn(1, 2, 3, 16).to(dtype)
    store.set_prefill(key, torch.randn_like(key))
    mask = torch.zeros(3, 3, dtype=torch.bool)
    output = store.attention(torch.randn_like(key), attn_mask=mask)
    assert output.shape == key.shape
    assert output.dtype == dtype
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("mode", ["plain", "skip", "sparse", "temporal"])
def test_residual_attention_window_excludes_evicted_tokens(mode):
    torch.manual_seed(7)
    cfg = KVCacheStoreConfig(
        enable_turbo_quant=True,
        enable_compute_skip=mode == "skip",
        compute_skip=ComputeSkipConfig(block_size=2, epsilon=0.0),
        enable_sparse_v=mode == "sparse", sparse_v_threshold=0.0,
        enable_temporal_pooling=mode == "temporal",
    )
    store = KVCacheStore(cfg)
    key = torch.randn(1, 2, 7, 16)
    value = torch.randn_like(key)
    store.set_prefill(key, value)
    query = torch.randn(1, 2, 1, 16)
    actual = store.attention(query, sliding_window=3)
    reference = KVCacheStore(KVCacheStoreConfig(
        enable_compute_skip=cfg.enable_compute_skip, compute_skip=cfg.compute_skip,
        enable_sparse_v=cfg.enable_sparse_v, sparse_v_threshold=0.0,
        enable_temporal_pooling=cfg.enable_temporal_pooling,
    ))
    reference.set_prefill(
        store._decode_tensor(store._k_enc)[..., -3:, :],
        store._decode_tensor(store._v_enc)[..., -3:, :],
    )
    torch.testing.assert_close(actual, reference.attention(query))


@pytest.mark.parametrize("mask_kind", ["none", "additive", "boolean", "fully_masked"])
@pytest.mark.parametrize("residual", [False, True])
def test_compute_skip_zero_epsilon_matches_attention_with_partial_block(mask_kind, residual):
    torch.manual_seed(10)
    key = torch.randn(1, 2, 5, 16)
    value = torch.randn_like(key)
    query = torch.randn(1, 2, 2, 16)
    mask = None
    if mask_kind == "additive":
        mask = torch.zeros(2, 5)
        mask[:, 0] = float("-inf")
    elif mask_kind in {"boolean", "fully_masked"}:
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, 0] = False
        if mask_kind == "fully_masked":
            mask[0] = False
    if residual:
        store = KVCacheStore(KVCacheStoreConfig(
            enable_turbo_quant=True, enable_compute_skip=True,
            compute_skip=ComputeSkipConfig(block_size=4, epsilon=0.0),
        ))
        store.set_prefill(key, value)
        actual = store.attention(query, attn_mask=mask)
        store.cfg.enable_compute_skip = False
        expected = store.attention(query, attn_mask=mask)
    else:
        actual = compute_skip_attention(
            query, key, value, ComputeSkipConfig(block_size=4, epsilon=0.0), attn_mask=mask,
        ).output
        expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_compute_skip_retains_dropout_scaling_when_no_blocks_are_skipped():
    torch.manual_seed(12)
    query = torch.randn(1, 2, 2, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn_like(key)
    weights = torch.softmax(query @ key.transpose(-2, -1) / 4.0, dim=-1)
    torch.manual_seed(4)
    expected = torch.dropout(weights, 0.5, train=True) @ value
    torch.manual_seed(4)
    actual = compute_skip_attention(
        query, key, value, ComputeSkipConfig(block_size=4, epsilon=0.0), dropout_p=0.5,
    ).output
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_adaptive_single_token_has_finite_bit_allocation():
    tensor = torch.randn(1, 2, 1, 16)
    key, value, report = AdaptiveKVCodec().quantize_kv(tensor, tensor, tensor)
    assert torch.isfinite(key).all() and torch.isfinite(value).all()
    assert all(2 <= n <= 256 for n in report["k_levels"])


def test_adaptive_empty_key_has_explicit_error():
    query = torch.randn(1, 2, 1, 16)
    empty = torch.empty(1, 2, 0, 16)
    with pytest.raises(ValueError, match="at least one key token"):
        adaptive_kv_bits(query, empty, empty)
