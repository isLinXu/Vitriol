import torch


from vitriol.kv.codec import (
    _pack_sign_tensor,
    _unpack_sign_tensor,
    pack_blockwise_tensor,
    unpack_blockwise_tensor,
)


def test_pack_unpack_blockwise_tensor_preserves_shape_and_reduces_storage():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 32, dtype=torch.float32)

    packed = pack_blockwise_tensor(x, levels=8, block_size=32, bit_width=3)
    restored = unpack_blockwise_tensor(packed)

    assert restored.shape == x.shape
    assert packed.q_data.dtype == torch.uint8
    assert packed.storage_nbytes() < x.numel() * x.element_size()


def test_pack_unpack_blockwise_tensor_stays_within_quantization_error_budget():
    torch.manual_seed(1)
    x = torch.randn(1, 1, 8, 32, dtype=torch.float32)

    packed = pack_blockwise_tensor(x, levels=16, block_size=32, bit_width=4)
    restored = unpack_blockwise_tensor(packed)

    max_abs_err = (x - restored).abs().max().item()
    assert max_abs_err < 0.5


def test_pack_unpack_blockwise_tensor_supports_partial_last_byte_groups():
    torch.manual_seed(2)
    x = torch.randn(1, 1, 3, 5, dtype=torch.float32)

    packed = pack_blockwise_tensor(x, levels=4, block_size=5, bit_width=2)
    restored = unpack_blockwise_tensor(packed)

    assert restored.shape == x.shape
    assert packed.q_data.shape[-1] == 2


def test_sign_bit_packing_roundtrip_preserves_minus_one_plus_one_pattern():
    signs = torch.tensor([[[1, -1, 1, 1, -1, -1, 1, -1, 1, -1]]], dtype=torch.int8)

    packed = _pack_sign_tensor(signs)
    restored = _unpack_sign_tensor(packed, logical_dim=signs.shape[-1])

    assert packed.dtype == torch.uint8
    assert torch.equal(restored, signs)
