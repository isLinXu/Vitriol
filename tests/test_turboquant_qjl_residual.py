import torch

from vitriol.kv.codec import (
    approx_inner_product_with_qjl_residual,
    pack_blockwise_tensor,
    pack_blockwise_tensor_with_qjl_residual,
    unpack_blockwise_tensor,
)


def test_qjl_residual_skeleton_builds_expected_structure():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 16, 32)

    packed = pack_blockwise_tensor_with_qjl_residual(
        x,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=8,
        seed=123,
    )

    assert packed.base.orig_shape == tuple(x.shape)
    assert packed.sketch_dim == 8
    assert packed.residual_signs.dtype == torch.int8
    assert packed.residual_magnitudes.shape[-1] == 8


def test_qjl_residual_uses_shared_scale_1bit_structure():
    torch.manual_seed(2)
    x = torch.randn(1, 2, 16, 32)

    packed = pack_blockwise_tensor_with_qjl_residual(
        x,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=8,
        seed=123,
    )

    assert packed.residual_signs.dtype == torch.int8
    assert packed.residual_scale.shape[-1] == 1
    assert packed.residual_norms.shape[-1] == 1
    expanded = packed.residual_magnitudes
    assert expanded.shape[-1] == packed.sketch_dim
    assert torch.allclose(expanded, packed.residual_scale.expand_as(expanded))


def test_qjl_residual_path_reduces_inner_product_bias_vs_base_quantization():
    torch.manual_seed(1)
    query = torch.randn(1, 2, 32, 32)
    key = torch.randn(1, 2, 32, 32)

    base = pack_blockwise_tensor(key, levels=8, block_size=32, bit_width=3)
    base_restored = unpack_blockwise_tensor(base)
    qjl = pack_blockwise_tensor_with_qjl_residual(
        key,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=32,
        seed=7,
    )

    original_ip = (query * key).sum(dim=-1)
    base_ip = (query * base_restored).sum(dim=-1)
    qjl_ip = approx_inner_product_with_qjl_residual(query, qjl)

    base_bias = (base_ip - original_ip).abs().mean().item()
    qjl_bias = (qjl_ip - original_ip).abs().mean().item()

    assert qjl_bias <= base_bias


def test_qjl_residual_correction_does_not_systematically_shrink_true_residual_ip():
    torch.manual_seed(5)
    key = torch.randn(1, 2, 32, 32)
    qjl = pack_blockwise_tensor_with_qjl_residual(
        key,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=256,
        seed=37,
    )

    restored = unpack_blockwise_tensor(qjl.base)
    residual = key - restored
    query = residual.clone()

    true_correction = (query * residual).sum(dim=-1)
    base_ip = (query * restored).sum(dim=-1)
    qjl_ip = approx_inner_product_with_qjl_residual(query, qjl)
    approx_correction = qjl_ip - base_ip

    ratio = float((approx_correction.mean() / true_correction.mean()).item())

    assert ratio >= 0.78
    assert ratio <= 1.2


def test_qjl_residual_packs_signs_into_uint8_buffer():
    torch.manual_seed(3)
    x = torch.randn(1, 2, 16, 32)

    packed = pack_blockwise_tensor_with_qjl_residual(
        x,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=10,
        seed=17,
    )

    assert packed.residual_sign_bits.dtype == torch.uint8
    assert packed.residual_sign_bits.shape[-1] == 2
    assert packed.residual_signs.dtype == torch.int8
    assert set(torch.unique(packed.residual_signs).tolist()) <= {-1, 1}


def test_qjl_residual_bitpacked_storage_is_smaller_than_naive_sign_storage():
    torch.manual_seed(4)
    x = torch.randn(1, 2, 32, 32)

    packed = pack_blockwise_tensor_with_qjl_residual(
        x,
        levels=8,
        block_size=32,
        bit_width=3,
        sketch_dim=17,
        seed=29,
    )

    naive_sign_bytes = x.shape[0] * x.shape[1] * x.shape[2] * 17
    assert packed.storage_nbytes() > packed.base.storage_nbytes()
    assert packed.residual_sign_bits.numel() < naive_sign_bytes
