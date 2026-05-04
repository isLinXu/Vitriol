import torch

from vitriol.kv.codec import (
    approx_inner_product_with_qjl_residual,
    pack_blockwise_tensor,
    pack_blockwise_tensor_with_qjl_residual,
    unpack_blockwise_tensor,
)


def _relative_error_stats(reference: torch.Tensor, approx: torch.Tensor) -> tuple[float, float]:
    denom = torch.clamp(reference.abs(), min=1e-6)
    rel = ((approx - reference).abs() / denom).reshape(-1)
    mean_rel = float(rel.mean().item())
    p95_rel = float(torch.quantile(rel, 0.95).item())
    return mean_rel, p95_rel


def _rank_agreement(scores_a: torch.Tensor, scores_b: torch.Tensor) -> float:
    order_a = torch.argsort(scores_a, dim=-1, descending=True)
    order_b = torch.argsort(scores_b, dim=-1, descending=True)
    total = 0.0
    count = 0
    for row_a, row_b in zip(order_a, order_b):
        total += float((row_a == row_b).to(torch.float32).mean().item())
        count += 1
    return total / float(count)


def test_residual_qjl_reduces_mean_and_p95_relative_inner_product_error():
    torch.manual_seed(31)
    query = torch.randn(1, 2, 64, 32)
    key = torch.randn(1, 2, 64, 32)

    base = pack_blockwise_tensor(key, levels=8, block_size=32, bit_width=3)
    base_restored = unpack_blockwise_tensor(base)
    qjl = pack_blockwise_tensor_with_qjl_residual(
        key,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=128,
        seed=67,
    )

    reference = (query * key).sum(dim=-1)
    base_ip = (query * base_restored).sum(dim=-1)
    qjl_ip = approx_inner_product_with_qjl_residual(query, qjl)

    base_mean, base_p95 = _relative_error_stats(reference, base_ip)
    qjl_mean, qjl_p95 = _relative_error_stats(reference, qjl_ip)

    assert qjl_mean <= base_mean
    assert qjl_p95 <= base_p95


def test_residual_qjl_improves_rank_agreement_over_base_quantization():
    torch.manual_seed(32)
    queries = torch.randn(24, 32)
    keys = torch.randn(96, 32)

    base = pack_blockwise_tensor(keys, levels=8, block_size=32, bit_width=3)
    base_restored = unpack_blockwise_tensor(base)
    qjl = pack_blockwise_tensor_with_qjl_residual(
        keys,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=128,
        seed=71,
    )

    reference_scores = queries @ keys.transpose(0, 1)
    base_scores = queries @ base_restored.transpose(0, 1)
    qjl_scores = approx_inner_product_with_qjl_residual(queries.unsqueeze(1), qjl).squeeze(1)

    assert _rank_agreement(reference_scores, qjl_scores) >= _rank_agreement(reference_scores, base_scores)


def test_normalized_and_max_error_improve_with_higher_bit_width():
    torch.manual_seed(33)
    x = torch.randn(1, 2, 64, 32)

    restored2 = unpack_blockwise_tensor(pack_blockwise_tensor(x, levels=4, block_size=32, bit_width=2))
    restored3 = unpack_blockwise_tensor(pack_blockwise_tensor(x, levels=8, block_size=32, bit_width=3))
    restored4 = unpack_blockwise_tensor(pack_blockwise_tensor(x, levels=16, block_size=32, bit_width=4))

    def _norm_err(y: torch.Tensor) -> float:
        return float(torch.norm(x - y).item() / (torch.norm(x).item() + 1e-12))

    err2 = _norm_err(restored2)
    err3 = _norm_err(restored3)
    err4 = _norm_err(restored4)
    max2 = float((x - restored2).abs().max().item())
    max3 = float((x - restored3).abs().max().item())
    max4 = float((x - restored4).abs().max().item())

    assert err4 <= err3 <= err2
    assert max4 <= max3 <= max2
