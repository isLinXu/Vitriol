import torch
import torch.nn.functional as F

from vitriol.kv.cache_store import KVCacheStore, KVCacheStoreConfig
from vitriol.kv.codec import approx_inner_product_with_qjl_residual


def _build_attention_triplet(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    turbo_bits: float = 3.5,
    turbo_quantize_v: bool = False,
):
    base_store = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=turbo_bits,
            quantized_kv_start=0,
            enable_turbo_residual_qjl=False,
            turbo_quantize_v=turbo_quantize_v,
        )
    )
    proxy_store = KVCacheStore(
        KVCacheStoreConfig(
            enable_turbo_quant=True,
            turbo_bits=turbo_bits,
            quantized_kv_start=0,
            enable_turbo_residual_qjl=True,
            turbo_quantize_v=turbo_quantize_v,
        )
    )
    base_store.set_prefill(key, value)
    proxy_store.set_prefill(key, value)
    return base_store, proxy_store


def _base_scores(store: KVCacheStore, query: torch.Tensor) -> torch.Tensor:
    key = store._decode_tensor(store._k_enc)
    return query @ key.transpose(-2, -1)


def _proxy_scores(store: KVCacheStore, query: torch.Tensor) -> torch.Tensor:
    return approx_inner_product_with_qjl_residual(query, store._k_enc)


def test_long_context_residual_proxy_attention_is_not_worse_than_base_at_512_tokens():
    torch.manual_seed(101)
    seq_len = 512
    query = torch.randn(1, 1, 4, 32)
    key = torch.randn(1, 1, seq_len, 32) * 0.01
    value = torch.randn(1, 1, seq_len, 32) * 0.01
    for offset, scale in zip(
        [seq_len // 4, seq_len // 2, seq_len - 7],
        [8.0, 7.0, 6.0],
    ):
        key[..., offset, :] = query[..., 0, :] * scale
        value[..., offset, :] = query[..., 0, :] * (scale / 2.0)
    ref = F.scaled_dot_product_attention(query, key, value)
    base_store, proxy_store = _build_attention_triplet(key=key, value=value)
    base_out = base_store.attention(query)
    proxy_out = proxy_store.attention(query)
    base_err = torch.mean((base_out - ref) ** 2).item()
    proxy_err = torch.mean((proxy_out - ref) ** 2).item()

    assert proxy_err <= base_err


def test_long_context_single_needle_stays_top_ranked_with_proxy():
    torch.manual_seed(102)
    query = torch.randn(1, 1, 1, 32)
    key = torch.randn(1, 1, 512, 32) * 0.01
    value = torch.randn(1, 1, 512, 32) * 0.01
    needle_index = 417
    key[..., needle_index, :] = query[..., 0, :] * 8.0
    value[..., needle_index, :] = query[..., 0, :] * 4.0

    base_store, proxy_store = _build_attention_triplet(key=key, value=value)
    reference_scores = (query @ key.transpose(-2, -1)).squeeze()
    base_scores = _base_scores(base_store, query).squeeze()
    proxy_scores = _proxy_scores(proxy_store, query).squeeze()

    assert int(torch.argmax(reference_scores).item()) == needle_index
    assert int(torch.argmax(base_scores).item()) == needle_index
    assert int(torch.argmax(proxy_scores).item()) == needle_index


def test_long_context_multi_needle_recall_is_not_worse_with_proxy():
    torch.manual_seed(103)
    query = torch.randn(1, 1, 1, 32)
    key = torch.randn(1, 1, 512, 32) * 0.01
    value = torch.randn(1, 1, 512, 32) * 0.01
    needle_positions = [129, 301, 487]
    for idx, scale in zip(needle_positions, [8.0, 7.0, 6.0]):
        key[..., idx, :] = query[..., 0, :] * scale
        value[..., idx, :] = query[..., 0, :] * (scale / 2.0)

    base_store, proxy_store = _build_attention_triplet(key=key, value=value)
    base_topk = set(torch.topk(_base_scores(base_store, query).squeeze(), k=5).indices.tolist())
    proxy_topk = set(torch.topk(_proxy_scores(proxy_store, query).squeeze(), k=5).indices.tolist())

    base_hits = len(set(needle_positions) & base_topk)
    proxy_hits = len(set(needle_positions) & proxy_topk)
    assert proxy_hits >= base_hits


def test_long_context_residual_proxy_keeps_capacity_advantage_over_bf16():
    torch.manual_seed(104)
    seq_len = 512
    key = torch.randn(1, 2, seq_len, 32)
    value = torch.randn(1, 2, seq_len, 32)
    _, proxy_store = _build_attention_triplet(
        key=key,
        value=value,
        turbo_quantize_v=True,
    )

    bf16_bytes = int(key.numel() * 2 + value.numel() * 2)
    packed_bytes = proxy_store.estimated_kv_bytes()
    capacity_gain = bf16_bytes / float(packed_bytes)

    assert packed_bytes < bf16_bytes
    assert capacity_gain > 1.0
