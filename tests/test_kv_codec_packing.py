import torch


from vitriol.kv.codec import pack_blockwise_tensor, unpack_blockwise_tensor


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
