import torch

from vitriol.kv.codec import (
    approx_inner_product_with_qjl_residual,
    pack_blockwise_tensor,
    pack_blockwise_tensor_with_qjl_residual,
    unpack_blockwise_tensor,
)


def _topk_recall(reference_scores: torch.Tensor, approx_scores: torch.Tensor, k: int) -> float:
    ref_idx = torch.topk(reference_scores, k=k, dim=-1).indices
    approx_idx = torch.topk(approx_scores, k=k, dim=-1).indices
    total = 0.0
    count = 0
    for ref_row, approx_row in zip(ref_idx, approx_idx):
        total += len(set(ref_row.tolist()) & set(approx_row.tolist())) / float(k)
        count += 1
    return total / float(count)


def test_topk_retrieval_overlap_improves_with_residual_qjl():
    torch.manual_seed(21)
    queries = torch.randn(16, 32)
    keys = torch.randn(128, 32)

    base = pack_blockwise_tensor(keys, levels=8, block_size=32, bit_width=3)
    restored = unpack_blockwise_tensor(base)
    qjl = pack_blockwise_tensor_with_qjl_residual(
        keys,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=64,
        seed=31,
    )

    original_scores = queries @ keys.transpose(0, 1)
    base_scores = queries @ restored.transpose(0, 1)
    qjl_scores = approx_inner_product_with_qjl_residual(
        queries.unsqueeze(1),
        qjl,
    ).squeeze(1)

    assert _topk_recall(original_scores, qjl_scores, k=8) >= _topk_recall(original_scores, base_scores, k=8)


def test_single_needle_remains_top_ranked_under_quantized_paths():
    torch.manual_seed(22)
    query = torch.randn(1, 32)
    keys = torch.randn(64, 32) * 0.05
    needle = query * 5.0
    keys[7] = needle

    base = unpack_blockwise_tensor(pack_blockwise_tensor(keys, levels=8, block_size=32, bit_width=3))
    qjl = pack_blockwise_tensor_with_qjl_residual(
        keys,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=64,
        seed=41,
    )

    original_scores = (query @ keys.transpose(0, 1)).squeeze(0)
    base_scores = (query @ base.transpose(0, 1)).squeeze(0)
    qjl_scores = approx_inner_product_with_qjl_residual(query.unsqueeze(1), qjl).squeeze()

    assert int(torch.argmax(original_scores).item()) == 7
    assert int(torch.argmax(base_scores).item()) == 7
    assert int(torch.argmax(qjl_scores).item()) == 7


def test_multi_needle_recall_is_not_worse_with_residual_qjl():
    torch.manual_seed(23)
    query = torch.randn(1, 32)
    keys = torch.randn(128, 32) * 0.05
    needle_indices = [5, 17, 63]
    for i, scale in zip(needle_indices, [5.0, 4.5, 4.0]):
        keys[i] = query.squeeze(0) * scale

    base = unpack_blockwise_tensor(pack_blockwise_tensor(keys, levels=8, block_size=32, bit_width=3))
    qjl = pack_blockwise_tensor_with_qjl_residual(
        keys,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=64,
        seed=53,
    )

    original_scores = (query @ keys.transpose(0, 1)).squeeze(0)
    base_scores = (query @ base.transpose(0, 1)).squeeze(0)
    qjl_scores = approx_inner_product_with_qjl_residual(query.unsqueeze(1), qjl).squeeze()

    base_topk = set(torch.topk(base_scores, k=5).indices.tolist())
    qjl_topk = set(torch.topk(qjl_scores, k=5).indices.tolist())

    base_needle_hits = len(set(needle_indices) & base_topk)
    qjl_needle_hits = len(set(needle_indices) & qjl_topk)

    assert qjl_needle_hits >= base_needle_hits
