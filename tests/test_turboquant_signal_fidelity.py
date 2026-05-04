import torch

from vitriol.kv.codec import (
    approx_inner_product_with_qjl_residual,
    pack_blockwise_tensor,
    pack_blockwise_tensor_with_qjl_residual,
    unpack_blockwise_tensor,
)


def _mse(x: torch.Tensor, y: torch.Tensor) -> float:
    diff = x - y
    return float(torch.mean(diff * diff).item())


def _mae(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(x - y)).item())


def _topk_overlap_ratio(scores_a: torch.Tensor, scores_b: torch.Tensor, k: int) -> float:
    topk_a = torch.topk(scores_a, k=k, dim=-1).indices
    topk_b = torch.topk(scores_b, k=k, dim=-1).indices
    overlap = 0.0
    total = 0
    for row_a, row_b in zip(topk_a, topk_b):
        overlap += len(set(row_a.tolist()) & set(row_b.tolist())) / float(k)
        total += 1
    return overlap / float(total)


def test_distortion_mse_and_mae_improve_with_higher_bit_width():
    torch.manual_seed(11)
    x = torch.randn(1, 2, 64, 32, dtype=torch.float32)

    restored2 = unpack_blockwise_tensor(pack_blockwise_tensor(x, levels=4, block_size=32, bit_width=2))
    restored3 = unpack_blockwise_tensor(pack_blockwise_tensor(x, levels=8, block_size=32, bit_width=3))
    restored4 = unpack_blockwise_tensor(pack_blockwise_tensor(x, levels=16, block_size=32, bit_width=4))

    mse2, mse3, mse4 = _mse(x, restored2), _mse(x, restored3), _mse(x, restored4)
    mae2, mae3, mae4 = _mae(x, restored2), _mae(x, restored3), _mae(x, restored4)

    assert mse4 <= mse3 <= mse2
    assert mae4 <= mae3 <= mae2


def test_residual_qjl_reduces_inner_product_mse_vs_base_quantization():
    torch.manual_seed(12)
    query = torch.randn(1, 2, 32, 32)
    key = torch.randn(1, 2, 32, 32)

    base = pack_blockwise_tensor(key, levels=8, block_size=32, bit_width=3)
    restored = unpack_blockwise_tensor(base)
    qjl = pack_blockwise_tensor_with_qjl_residual(
        key,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=128,
        seed=19,
    )

    original_ip = (query * key).sum(dim=-1)
    base_ip = (query * restored).sum(dim=-1)
    qjl_ip = approx_inner_product_with_qjl_residual(query, qjl)

    assert _mse(qjl_ip, original_ip) <= _mse(base_ip, original_ip)
    assert _mae(qjl_ip, original_ip) <= _mae(base_ip, original_ip)


def test_residual_qjl_preserves_topk_overlap_at_least_as_well_as_base():
    torch.manual_seed(13)
    queries = torch.randn(16, 32)
    keys = torch.randn(64, 32)

    base = pack_blockwise_tensor(keys, levels=8, block_size=32, bit_width=3)
    restored = unpack_blockwise_tensor(base)
    qjl = pack_blockwise_tensor_with_qjl_residual(
        keys,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=32,
        seed=23,
    )

    original_scores = queries @ keys.transpose(0, 1)
    base_scores = queries @ restored.transpose(0, 1)
    qjl_scores = approx_inner_product_with_qjl_residual(
        queries.unsqueeze(1),
        qjl,
    ).squeeze(1)

    base_overlap = _topk_overlap_ratio(original_scores, base_scores, k=8)
    qjl_overlap = _topk_overlap_ratio(original_scores, qjl_scores, k=8)

    assert qjl_overlap >= base_overlap
